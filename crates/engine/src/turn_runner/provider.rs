//! Provider stage.
//!
//! Mirrors the Python `engine/turn_runner/provider_and_tools_stage.py` (the
//! provider half). It is the core stage that calls the LLM. It:
//!
//! * invokes the [`crate::agent::TurnGenerator`] with the accumulated
//!   messages,
//! * applies rate limiting, retry-with-backoff, and optional failover to an
//!   ordered list of fallback generators,
//! * computes token usage with a deterministic estimate,
//! * appends the response messages to the context,
//! * when streaming is enabled, emits `ContentBlockStart` / `MessageStop`
//!   events through the context's stream sender.
//!
//! This stage is the one the runtime's agent loop drives repeatedly: after it
//! runs, the runtime inspects the newly appended messages for tool calls and
//! loops back here with the tool results appended.

use crate::agent::TurnGenerator;
use crate::stages::{Stage, StageContext, StageError, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::{Error, Result};
use opensquilla_core::events::StreamEvent;
use opensquilla_core::types::{Message, MessageRole, Usage};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

/// Retry policy for provider calls.
#[derive(Debug, Clone)]
pub struct ProviderRetryPolicy {
    /// Maximum number of retries after the initial attempt.
    pub max_retries: u32,
    /// Base backoff delay in milliseconds (doubled per attempt).
    pub backoff_ms: u64,
    /// Whether retries are limited to errors that look transient
    /// (rate-limit, timeout, 5xx). When `false`, every error is retried.
    pub retry_transient_only: bool,
}

impl Default for ProviderRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_ms: 250,
            retry_transient_only: true,
        }
    }
}

/// A token-bucket rate limiter for provider calls.
///
/// The bucket refills at `refill_per_sec` tokens per second up to `capacity`.
/// [`RateLimiter::acquire`] waits until the requested number of tokens is
/// available and consumes them.
#[derive(Debug)]
pub struct RateLimiter {
    capacity: f64,
    refill_per_sec: f64,
    state: tokio::sync::Mutex<RateLimiterState>,
}

#[derive(Debug)]
struct RateLimiterState {
    tokens: f64,
    last: Instant,
}

impl RateLimiter {
    /// Create a limiter with the given capacity and refill rate (tokens/sec).
    pub fn new(capacity: u64, refill_per_sec: f64) -> Self {
        Self {
            capacity: capacity.max(1) as f64,
            refill_per_sec: refill_per_sec.max(0.0001),
            state: tokio::sync::Mutex::new(RateLimiterState {
                tokens: capacity.max(1) as f64,
                last: Instant::now(),
            }),
        }
    }

    /// A disabled limiter: never throttles.
    pub fn unlimited() -> Self {
        Self::new(u64::MAX, f64::MAX)
    }

    /// Refill the bucket up to capacity, returning the current token count.
    async fn refill(&self, state: &mut RateLimiterState) -> f64 {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last).as_secs_f64();
        state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        state.last = now;
        state.tokens
    }

    /// Acquire `cost` tokens, waiting as needed.
    pub async fn acquire(&self, cost: u64) {
        let cost_f = cost as f64;
        loop {
            let mut state = self.state.lock().await;
            let tokens = self.refill(&mut state).await;
            if tokens >= cost_f {
                state.tokens -= cost_f;
                return;
            }
            let deficit = cost_f - tokens;
            let wait = (deficit / self.refill_per_sec).min(60.0);
            drop(state);
            tokio::time::sleep(Duration::from_secs_f64(wait)).await;
        }
    }

    /// Try to acquire `cost` tokens without waiting.
    pub fn try_acquire(&self, cost: u64) -> bool {
        let mut state = self.state.blocking_lock();
        let tokens = {
            let now = Instant::now();
            let elapsed = now.duration_since(state.last).as_secs_f64();
            state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
            state.last = now;
            state.tokens
        };
        let cost_f = cost as f64;
        if tokens >= cost_f {
            state.tokens -= cost_f;
            true
        } else {
            false
        }
    }
}

/// A report of the most recent provider call.
#[derive(Debug, Clone, Default)]
pub struct ProviderCallReport {
    /// Whether the call succeeded.
    pub success: bool,
    /// Number of attempts made (including retries).
    pub attempts: u32,
    /// The generator name that ultimately served the call.
    pub served_by: Option<String>,
    /// The delay waited before the final attempt, if any.
    pub last_backoff_ms: u64,
    /// The final error message, if the call failed.
    pub error_message: Option<String>,
}

/// The failover ordering strategy for the provider stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailoverOrder {
    /// Try the primary first, then fallbacks in registration order.
    PrimaryFirst,
    /// Try the most recently successful provider first.
    LastSuccessfulFirst,
    /// Round-robin across all providers.
    RoundRobin,
}

impl FailoverOrder {
    /// The canonical name of the strategy.
    pub fn as_str(&self) -> &'static str {
        match self {
            FailoverOrder::PrimaryFirst => "primary_first",
            FailoverOrder::LastSuccessfulFirst => "last_successful_first",
            FailoverOrder::RoundRobin => "round_robin",
        }
    }
}

/// A provider failover policy describing how fallback generators are ordered
/// and selected.
#[derive(Debug, Clone)]
pub struct ProviderFailoverPolicy {
    /// The ordering strategy.
    pub order: FailoverOrder,
    /// Whether a failed fallback is skipped for the rest of the turn.
    pub skip_failed_fallbacks: bool,
    /// The maximum number of fallbacks tried before surfacing the error.
    pub max_fallbacks: u32,
}

impl Default for ProviderFailoverPolicy {
    fn default() -> Self {
        Self {
            order: FailoverOrder::PrimaryFirst,
            skip_failed_fallbacks: true,
            max_fallbacks: 2,
        }
    }
}

impl ProviderFailoverPolicy {
    /// Create a new failover policy.
    pub fn new(order: FailoverOrder) -> Self {
        Self {
            order,
            ..Default::default()
        }
    }

    /// Set whether failed fallbacks are skipped for the turn.
    pub fn with_skip_failed(mut self, skip: bool) -> Self {
        self.skip_failed_fallbacks = skip;
        self
    }

    /// Set the maximum number of fallbacks tried.
    pub fn with_max_fallbacks(mut self, max: u32) -> Self {
        self.max_fallbacks = max;
        self
    }
}

/// A per-turn tracker of provider outcomes, used by the failover policy.
#[derive(Debug, Clone, Default)]
pub struct ProviderOutcomeTracker {
    /// The provider names that succeeded this turn, in order.
    pub successes: Vec<String>,
    /// The provider names that failed this turn.
    pub failures: Vec<String>,
}

impl ProviderOutcomeTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a success for a provider.
    pub fn record_success(&mut self, provider: &str) {
        if !self.successes.iter().any(|p| p == provider) {
            self.successes.push(provider.to_string());
        }
    }

    /// Record a failure for a provider.
    pub fn record_failure(&mut self, provider: &str) {
        if !self.failures.iter().any(|p| p == provider) {
            self.failures.push(provider.to_string());
        }
    }

    /// Whether a provider failed earlier in the turn.
    pub fn has_failed(&self, provider: &str) -> bool {
        self.failures.iter().any(|p| p == provider)
    }

    /// The most recently successful provider, if any.
    pub fn last_successful(&self) -> Option<&str> {
        self.successes.last().map(|s| s.as_str())
    }

    /// Reorder fallback generators according to the policy.
    ///
    /// `primary` is the primary generator; `fallbacks` are the registered
    /// fallbacks. Returns the ordered candidate list (primary first unless
    /// the policy says otherwise).
    pub fn order_candidates<'a>(
        &self,
        primary: &'a dyn TurnGenerator,
        fallbacks: &'a [Arc<dyn TurnGenerator>],
        policy: &ProviderFailoverPolicy,
    ) -> Vec<&'a dyn TurnGenerator> {
        let mut candidates: Vec<&dyn TurnGenerator> = Vec::new();
        let mut used: Vec<String> = Vec::new();
        let is_used = |name: &str, used: &Vec<String>| used.iter().any(|u| u == name);

        // Primary is always first for PrimaryFirst.
        let order = policy.order;
        match order {
            FailoverOrder::PrimaryFirst => {
                if !is_used(primary.provider_name(), &used) {
                    candidates.push(primary);
                    used.push(primary.provider_name().to_string());
                }
                for fb in fallbacks {
                    if policy.skip_failed_fallbacks && self.has_failed(fb.provider_name()) {
                        continue;
                    }
                    if !is_used(fb.provider_name(), &used) {
                        candidates.push(fb.as_ref());
                        used.push(fb.provider_name().to_string());
                    }
                }
            }
            FailoverOrder::LastSuccessfulFirst => {
                if let Some(last) = self.last_successful() {
                    if let Some(fb) = fallbacks
                        .iter()
                        .find(|f| f.provider_name() == last && !self.has_failed(last))
                    {
                        candidates.push(fb.as_ref());
                        used.push(last.to_string());
                    }
                }
                if !is_used(primary.provider_name(), &used) {
                    candidates.push(primary);
                    used.push(primary.provider_name().to_string());
                }
                for fb in fallbacks {
                    if policy.skip_failed_fallbacks && self.has_failed(fb.provider_name()) {
                        continue;
                    }
                    if !is_used(fb.provider_name(), &used) {
                        candidates.push(fb.as_ref());
                        used.push(fb.provider_name().to_string());
                    }
                }
            }
            FailoverOrder::RoundRobin => {
                // Start after the most recent success.
                let mut all: Vec<&dyn TurnGenerator> = Vec::new();
                for fb in fallbacks {
                    all.push(fb.as_ref());
                }
                all.push(primary);
                let start = self
                    .last_successful()
                    .and_then(|last| all.iter().position(|c| c.provider_name() == last))
                    .map(|i| (i + 1) % all.len())
                    .unwrap_or(0);
                for i in 0..all.len() {
                    let idx = (start + i) % all.len();
                    let candidate = all[idx];
                    if policy.skip_failed_fallbacks && self.has_failed(candidate.provider_name()) {
                        continue;
                    }
                    if !is_used(candidate.provider_name(), &used) {
                        candidates.push(candidate);
                        used.push(candidate.provider_name().to_string());
                    }
                }
            }
        }

        // Cap the number of candidates tried.
        let max = policy.max_fallbacks.max(1) as usize;
        candidates.truncate(max + 1);
        candidates
    }
}

/// The provider stage in the turn pipeline.
#[derive(Debug)]
pub struct ProviderStage {
    /// Fallback model used when the generator does not name one.
    default_model: String,
    /// Fallback provider used when the generator does not name one.
    default_provider: String,
    /// Whether to emit stream events for the response.
    streaming_enabled: bool,
    /// Retry policy applied around the generator call.
    retry_policy: ProviderRetryPolicy,
    /// Optional rate limiter gating provider calls.
    rate_limiter: Option<Arc<RateLimiter>>,
    /// Ordered fallback generators used when the primary fails.
    fallback_generators: Vec<Arc<dyn TurnGenerator>>,
    /// The cost (in rate-limiter tokens) charged per call.
    request_cost: u64,
    /// Report from the most recent call.
    last_report: std::sync::Mutex<Option<ProviderCallReport>>,
}

impl ProviderStage {
    /// Create a new provider stage.
    pub fn new(default_model: String, default_provider: String, streaming_enabled: bool) -> Self {
        Self {
            default_model,
            default_provider,
            streaming_enabled,
            retry_policy: ProviderRetryPolicy::default(),
            rate_limiter: None,
            fallback_generators: Vec::new(),
            request_cost: 1,
            last_report: std::sync::Mutex::new(None),
        }
    }

    /// Set the retry policy.
    pub fn with_retry_policy(mut self, policy: ProviderRetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Attach a rate limiter shared across calls.
    pub fn with_rate_limiter(mut self, limiter: Arc<RateLimiter>) -> Self {
        self.rate_limiter = Some(limiter);
        self
    }

    /// Set the rate-limiter token cost charged per call.
    pub fn with_request_cost(mut self, cost: u64) -> Self {
        self.request_cost = cost.max(1);
        self
    }

    /// Register ordered fallback generators tried when the primary fails.
    pub fn with_fallback_generators(mut self, generators: Vec<Arc<dyn TurnGenerator>>) -> Self {
        self.fallback_generators = generators;
        self
    }

    /// The report from the most recent call, if any.
    pub fn last_report(&self) -> Option<ProviderCallReport> {
        self.last_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The retry policy.
    pub fn retry_policy(&self) -> &ProviderRetryPolicy {
        &self.retry_policy
    }

    /// Compute a deterministic token estimate for a message list.
    ///
    /// Mirrors the existing `crate::stages::ProviderStage`: roughly one token
    /// per four content characters.
    pub fn estimate_tokens(messages: &[Message]) -> u64 {
        messages
            .iter()
            .map(|m| m.text_content().chars().count() as u64 / 4)
            .sum()
    }

    /// Classify whether an error looks transient (worth a retry).
    fn is_transient(&self, error: &Error) -> bool {
        let text = error.to_string().to_ascii_lowercase();
        text.contains("rate")
            || text.contains("429")
            || text.contains("timeout")
            || text.contains("timed out")
            || text.contains("network")
            || text.contains("503")
            || text.contains("502")
            || text.contains("temporarily")
            || text.contains("overloaded")
    }

    /// Generate a response from a single generator, applying the rate limiter.
    async fn generate_from(
        &self,
        generator: &dyn TurnGenerator,
        messages: &[Message],
    ) -> Result<Vec<Message>> {
        if let Some(limiter) = &self.rate_limiter {
            limiter.acquire(self.request_cost).await;
        }
        generator.generate(messages).await
    }

    /// Generate a response with the retry policy applied to one generator.
    async fn generate_with_retries(
        &self,
        generator: &dyn TurnGenerator,
        messages: &[Message],
        report: &mut ProviderCallReport,
    ) -> Result<Vec<Message>> {
        let mut attempt = 0u32;
        let max_attempts = self.retry_policy.max_retries + 1;
        loop {
            attempt += 1;
            report.attempts = attempt;
            match self.generate_from(generator, messages).await {
                Ok(response) => {
                    report.success = true;
                    return Ok(response);
                }
                Err(e) => {
                    let transient = self.is_transient(&e);
                    if attempt >= max_attempts
                        || (self.retry_policy.retry_transient_only && !transient)
                    {
                        report.error_message = Some(e.to_string());
                        return Err(e);
                    }
                    let backoff = self
                        .retry_policy
                        .backoff_ms
                        .saturating_mul(1u64 << (attempt - 1));
                    report.last_backoff_ms = backoff;
                    warn!(
                        attempt = attempt,
                        max_attempts = max_attempts,
                        backoff_ms = backoff,
                        error = %e,
                        "provider call failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(backoff.min(10_000))).await;
                }
            }
        }
    }

    /// Generate a response, trying the primary generator then each fallback.
    async fn generate_with_failover(
        &self,
        ctx: &StageContext,
        primary: &dyn TurnGenerator,
    ) -> Result<Vec<Message>> {
        let mut report = ProviderCallReport::default();
        report.served_by = Some(primary.provider_name().to_string());

        match self
            .generate_with_retries(primary, &ctx.messages, &mut report)
            .await
        {
            Ok(response) => {
                *self.last_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(report);
                return Ok(response);
            }
            Err(primary_err) => {
                for fallback in &self.fallback_generators {
                    report.served_by = Some(fallback.provider_name().to_string());
                    debug!(
                        turn_id = %ctx.turn_id,
                        fallback = fallback.provider_name(),
                        "trying fallback generator"
                    );
                    match self
                        .generate_with_retries(fallback.as_ref(), &ctx.messages, &mut report)
                        .await
                    {
                        Ok(response) => {
                            report.success = true;
                            *self.last_report.lock().unwrap_or_else(|e| e.into_inner()) =
                                Some(report);
                            return Ok(response);
                        }
                        Err(_) => continue,
                    }
                }
                *self.last_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(report);
                Err(primary_err)
            }
        }
    }

    /// Emit streaming events for a response through the context channel.
    pub async fn emit_stream_events(
        tx: &tokio::sync::mpsc::Sender<StreamEvent>,
        assistant_messages: &[Message],
        usage: Usage,
    ) {
        let mut index = 0usize;
        for message in assistant_messages {
            for block in &message.content {
                let _ = tx
                    .send(StreamEvent::ContentBlockStart {
                        index,
                        block: block.clone(),
                    })
                    .await;
                let _ = tx.send(StreamEvent::ContentBlockStop { index }).await;
                index += 1;
            }
        }
        let _ = tx
            .send(StreamEvent::MessageStop {
                content: assistant_messages
                    .iter()
                    .flat_map(|m| m.content.clone())
                    .collect(),
                usage: Some(usage),
            })
            .await;
    }

    /// Build a provider [`ChatConfig`] from the turn context.
    ///
    /// Only available with the `provider` feature.
    #[cfg(feature = "provider")]
    pub fn build_chat_config(
        &self,
        ctx: &StageContext,
        stream: bool,
    ) -> opensquilla_provider::ChatConfig {
        let model = if ctx.current_model.is_empty() {
            self.default_model.clone()
        } else {
            ctx.current_model.clone()
        };
        opensquilla_provider::ChatConfig {
            model,
            stream,
            ..Default::default()
        }
    }

    /// Stream a provider response through the context channel.
    ///
    /// Consumes the provider's [`StreamEvent`] stream, translates each event
    /// into a core [`opensquilla_core::events::StreamEvent`], and forwards it.
    /// Returns the fully assembled assistant messages and the final usage.
    ///
    /// Only available with the `provider` feature.
    #[cfg(feature = "provider")]
    pub async fn stream_from_provider(
        &self,
        provider: &dyn opensquilla_provider::Provider,
        config: &opensquilla_provider::ChatConfig,
        messages: &[Message],
        tools: &[opensquilla_core::types::ToolDefinition],
        tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<(Vec<Message>, Usage)> {
        use futures::StreamExt;
        if let Some(limiter) = &self.rate_limiter {
            limiter.acquire(self.request_cost).await;
        }
        let mut stream = provider
            .stream_chat(config, messages, tools)
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<opensquilla_core::types::ToolCall> = Vec::new();
        let mut index = 0usize;
        let mut usage = Usage::default();
        let mut stop_reason: Option<String> = None;

        while let Some(event) = stream.next().await {
            match event.map_err(|e| Error::Provider(e.to_string()))? {
                opensquilla_provider::types::StreamEvent::Text { text: delta } => {
                    text.push_str(&delta);
                    let _ = tx
                        .send(StreamEvent::ContentBlockDelta {
                            index,
                            delta: opensquilla_core::events::ContentBlockDelta::TextDelta {
                                text: delta,
                            },
                        })
                        .await;
                }
                opensquilla_provider::types::StreamEvent::Reasoning { reasoning: delta } => {
                    reasoning.push_str(&delta);
                    let _ = tx
                        .send(StreamEvent::ContentBlockDelta {
                            index,
                            delta: opensquilla_core::events::ContentBlockDelta::ReasoningDelta {
                                reasoning: delta,
                            },
                        })
                        .await;
                }
                opensquilla_provider::types::StreamEvent::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    // Accumulate a per-tool-call JSON argument string. In this
                    // translation layer we treat each delta as a full tool call
                    // fragment and merge by id.
                    merge_tool_call(&mut tool_calls, &id, &name, &arguments);
                }
                opensquilla_provider::types::StreamEvent::Done {
                    usage: final_usage,
                    stop_reason: reason,
                } => {
                    if let Some(u) = final_usage {
                        usage = u;
                    }
                    stop_reason = reason;
                    break;
                }
                opensquilla_provider::types::StreamEvent::Error { message } => {
                    return Err(Error::Provider(message));
                }
            }
            index += 1;
        }

        let mut blocks: Vec<opensquilla_core::types::ContentBlock> = Vec::new();
        if !reasoning.is_empty() {
            blocks.push(opensquilla_core::types::ContentBlock::Reasoning(reasoning));
        }
        if !text.is_empty() {
            blocks.push(opensquilla_core::types::ContentBlock::Text(text));
        }
        for call in &tool_calls {
            blocks.push(opensquilla_core::types::ContentBlock::ToolUse(call.clone()));
        }
        let message = Message {
            role: MessageRole::Assistant,
            content: blocks,
            name: None,
            tool_call_id: None,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_result: None,
        };
        let _ = stop_reason;
        Ok((vec![message], usage))
    }
}

/// Merge a streaming tool-call delta into an accumulated tool-call list.
///
/// When the id already exists, the argument JSON is concatenated; otherwise a
/// new tool call is started.
#[cfg(feature = "provider")]
fn merge_tool_call(
    calls: &mut Vec<opensquilla_core::types::ToolCall>,
    id: &str,
    name: &str,
    arguments: &str,
) {
    if let Some(existing) = calls.iter_mut().find(|c| c.id == id) {
        if let Some(s) = existing.input.as_str() {
            existing.input = serde_json::Value::String(format!("{s}{arguments}"));
        } else {
            existing.input = serde_json::Value::String(arguments.to_string());
        }
    } else {
        calls.push(opensquilla_core::types::ToolCall::new(
            id,
            name,
            serde_json::Value::String(arguments.to_string()),
        ));
    }
}

#[async_trait]
impl Stage for ProviderStage {
    #[instrument(skip(self, ctx, generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("provider: sending messages to model");

        // Ensure a model/provider are named for observability.
        if ctx.current_model.is_empty() {
            ctx.current_model = self.default_model.clone();
        }
        if ctx.current_provider.is_empty() {
            ctx.current_provider = self.default_provider.clone();
        }

        let started = Instant::now();
        let response = match self.generate_with_failover(ctx, generator).await {
            Ok(response) => response,
            Err(e) => {
                let report = self.last_report();
                // Record the failed physical execution leg on the turn's route
                // plan telemetry (mirrors `route_plan.record_execution_leg`).
                crate::route_plan::record_execution_leg(
                    &mut ctx.metadata,
                    &ctx.current_provider,
                    &ctx.current_model,
                    "chat",
                    None,
                    None,
                    "provider_error",
                );
                info!(
                    turn_id = %ctx.turn_id,
                    attempts = report.as_ref().map(|r| r.attempts).unwrap_or(0),
                    duration_ms = started.elapsed().as_millis() as u64,
                    error = %e,
                    "provider call failed after retries"
                );
                return Ok(StageOutput::Error(StageError {
                    message: e.to_string(),
                    code: Some("PROVIDER_CALL_FAILED".to_string()),
                    stage: self.name().to_string(),
                }));
            }
        };
        let duration_ms = started.elapsed().as_millis() as u64;

        // Record the successful physical execution leg on the turn's route plan
        // telemetry (mirrors `route_plan.record_execution_leg`).
        crate::route_plan::record_execution_leg(
            &mut ctx.metadata,
            &ctx.current_provider,
            &ctx.current_model,
            "chat",
            None,
            None,
            "",
        );

        let input_tokens = Self::estimate_tokens(&ctx.messages);
        let output_tokens = Self::estimate_tokens(&response);
        ctx.usage = Usage::new(input_tokens, output_tokens);

        let assistant_messages: Vec<Message> = response
            .into_iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .collect();

        // Emit stream events when streaming is enabled and a channel exists.
        if self.streaming_enabled {
            if let Some(tx) = &ctx.streaming_tx {
                Self::emit_stream_events(tx, &assistant_messages, ctx.usage).await;
            }
        }

        ctx.messages.extend(assistant_messages);

        info!(
            turn_id = %ctx.turn_id,
            model = %ctx.current_model,
            provider = %ctx.current_provider,
            input_tokens = ctx.usage.input_tokens,
            output_tokens = ctx.usage.output_tokens,
            duration_ms = duration_ms,
            "provider stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "provider"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_rate_limiter_unlimited() {
        let limiter = RateLimiter::unlimited();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            limiter.acquire(1).await;
            limiter.acquire(1000).await;
        });
    }

    #[test]
    fn test_rate_limiter_throttles() {
        let limiter = RateLimiter::new(1, 10.0);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            limiter.acquire(1).await; // consumes the only token
            let start = Instant::now();
            limiter.acquire(1).await; // must wait for refill
            assert!(start.elapsed() >= Duration::from_millis(80));
        });
    }

    #[test]
    fn test_is_transient_classification() {
        let stage = ProviderStage::new(String::new(), String::new(), false);
        assert!(stage.is_transient(&Error::RateLimited(1)));
        assert!(stage.is_transient(&Error::Provider("timed out".into())));
        assert!(stage.is_transient(&Error::Provider("rate limited".into())));
        assert!(!stage.is_transient(&Error::Provider("invalid api key".into())));
        assert!(!stage.is_transient(&Error::InvalidInput("bad".into())));
    }

    #[tokio::test]
    async fn test_generate_retries_then_succeeds() {
        let failing = FlakyGenerator {
            failures_before_success: 2,
            attempts: Arc::new(AtomicU32::new(0)),
        };
        let stage = ProviderStage::new(String::new(), String::new(), false).with_retry_policy(
            ProviderRetryPolicy {
                max_retries: 3,
                backoff_ms: 1,
                retry_transient_only: false,
            },
        );
        let mut report = ProviderCallReport::default();
        let result = stage
            .generate_with_retries(&failing, &[Message::user("hi")], &mut report)
            .await;
        assert!(result.is_ok());
        assert_eq!(report.attempts, 3);
        assert!(report.success);
    }

    #[tokio::test]
    async fn test_generate_retries_exhausted() {
        let failing = FlakyGenerator {
            failures_before_success: 99,
            attempts: Arc::new(AtomicU32::new(0)),
        };
        let stage = ProviderStage::new(String::new(), String::new(), false).with_retry_policy(
            ProviderRetryPolicy {
                max_retries: 2,
                backoff_ms: 1,
                retry_transient_only: false,
            },
        );
        let mut report = ProviderCallReport::default();
        let result = stage
            .generate_with_retries(&failing, &[Message::user("hi")], &mut report)
            .await;
        assert!(result.is_err());
        assert_eq!(report.attempts, 3);
        assert!(!report.success);
    }

    #[tokio::test]
    async fn test_transient_only_skips_permanent() {
        let failing = PermanentFailGenerator;
        let stage = ProviderStage::new(String::new(), String::new(), false).with_retry_policy(
            ProviderRetryPolicy {
                max_retries: 3,
                backoff_ms: 1,
                retry_transient_only: true,
            },
        );
        let mut report = ProviderCallReport::default();
        let result = stage
            .generate_with_retries(&failing, &[Message::user("hi")], &mut report)
            .await;
        assert!(result.is_err());
        assert_eq!(report.attempts, 1); // permanent errors are not retried
    }

    #[tokio::test]
    async fn test_estimate_tokens() {
        let msgs = vec![Message::user("a fairly long user message")];
        let tokens = ProviderStage::estimate_tokens(&msgs);
        assert!(tokens > 0);
    }

    #[derive(Debug)]
    struct FlakyGenerator {
        failures_before_success: u32,
        attempts: Arc<AtomicU32>,
    }

    #[async_trait]
    impl TurnGenerator for FlakyGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.failures_before_success {
                Err(Error::Provider("temporary overload".into()))
            } else {
                Ok(vec![Message::assistant("recovered")])
            }
        }
        fn model_name(&self) -> &str {
            "flaky"
        }
        fn provider_name(&self) -> &str {
            "flaky"
        }
    }

    #[derive(Debug)]
    struct PermanentFailGenerator;

    #[async_trait]
    impl TurnGenerator for PermanentFailGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
            Err(Error::Provider("invalid api key".into()))
        }
        fn model_name(&self) -> &str {
            "perm"
        }
        fn provider_name(&self) -> &str {
            "perm"
        }
    }
}
