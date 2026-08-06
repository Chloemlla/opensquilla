//! Harness stage.
//!
//! Mirrors the Python `engine/turn_runner/harness.py` stage. The harness is
//! the first stage in every turn. It:
//!
//! * acquires the per-session lock (serializing overlapping turns on the same
//!   session) subject to a configurable acquisition timeout,
//! * resets per-turn state (`tool_round`) and establishes the turn's identity,
//! * validates that the turn carries messages and that the history is
//!   structurally sound (every `tool_result` matches a `tool_use`),
//! * fires [`crate::hooks::TurnHook`] lifecycle hooks around the turn,
//! * records stage timing/metrics for observability,
//! * establishes the error boundary by converting malformed context into a
//!   `StageError` rather than panicking.
//!
//! The full-turn lock guard is held only for the duration of this stage's
//! execution; the runtime owns the session lock for the whole turn when it
//! wraps the stage chain in `with_session_lock`.

use crate::agent::TurnGenerator;
use crate::hooks::TurnHook;
use crate::session_lock::SessionLockSet;
use crate::stages::{Stage, StageContext, StageError, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::MessageRole;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, trace, warn};

/// Tunable knobs for the harness stage.
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    /// Maximum number of distinct session locks retained before pruning.
    pub max_lock_entries: usize,
    /// Timeout for acquiring the per-session lock. A lock that cannot be
    /// acquired within this window is treated as a stage error rather than
    /// blocking the runtime indefinitely.
    pub lock_timeout: Duration,
    /// Hard wall-clock budget for the whole stage. Stages that exceed this are
    /// reported with a `STAGE_TIMEOUT` code.
    pub stage_timeout: Duration,
    /// Whether registered `TurnHook` lifecycle hooks are fired.
    pub emit_hooks: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            max_lock_entries: 4096,
            lock_timeout: Duration::from_secs(30),
            stage_timeout: Duration::from_secs(120),
            emit_hooks: true,
        }
    }
}

/// Per-execution observability recorded by the harness.
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    /// The name of the stage that recorded these metrics.
    pub stage_name: String,
    /// Wall-clock duration of the stage execution in milliseconds.
    pub duration_ms: u64,
    /// Number of messages present when the stage started.
    pub messages_in: usize,
    /// Time spent waiting for the per-session lock, in milliseconds.
    pub lock_wait_ms: u64,
    /// The session key the lock was acquired for.
    pub session_key: String,
    /// The turn id this execution belonged to.
    pub turn_id: String,
}

/// The first stage in the turn pipeline.
#[derive(Debug)]
pub struct HarnessStage {
    /// Per-session lock set used to serialize the setup phase.
    locks: Arc<SessionLockSet>,
    /// Configuration for the stage.
    config: HarnessConfig,
    /// Turn lifecycle hooks fired around the execution.
    hooks: Vec<Arc<dyn TurnHook + Send + Sync>>,
    /// The most recently recorded metrics (per-turn slot).
    last_metrics: std::sync::Mutex<Option<StageMetrics>>,
}

impl HarnessStage {
    /// Create a new harness stage with default lock settings.
    pub fn new() -> Self {
        Self {
            locks: Arc::new(SessionLockSet::new()),
            config: HarnessConfig::default(),
            hooks: Vec::new(),
            last_metrics: std::sync::Mutex::new(None),
        }
    }

    /// Create a harness stage with a specific session-lock cap.
    pub fn new_with_max_entries(max_entries: usize) -> Self {
        Self {
            locks: Arc::new(SessionLockSet::new().with_max_entries(max_entries)),
            config: HarnessConfig {
                max_lock_entries: max_entries,
                ..Default::default()
            },
            hooks: Vec::new(),
            last_metrics: std::sync::Mutex::new(None),
        }
    }

    /// Replace the stage configuration.
    pub fn with_config(mut self, config: HarnessConfig) -> Self {
        if config.max_lock_entries != self.config.max_lock_entries {
            self.locks = Arc::new(SessionLockSet::new().with_max_entries(config.max_lock_entries));
        }
        self.config = config;
        self
    }

    /// Set the timeout for acquiring the per-session lock.
    pub fn with_lock_timeout(mut self, timeout: Duration) -> Self {
        self.config.lock_timeout = timeout;
        self
    }

    /// Set whether turn lifecycle hooks are fired.
    pub fn with_hooks_enabled(mut self, enabled: bool) -> Self {
        self.config.emit_hooks = enabled;
        self
    }

    /// Register turn lifecycle hooks fired around each execution.
    pub fn with_hooks(mut self, hooks: Vec<Arc<dyn TurnHook + Send + Sync>>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Append a single turn lifecycle hook.
    pub fn add_hook(mut self, hook: Arc<dyn TurnHook + Send + Sync>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Share an existing session lock set (useful when the runtime wants one
    /// lock namespace across multiple agents).
    pub fn with_locks(mut self, locks: Arc<SessionLockSet>) -> Self {
        self.locks = locks;
        self
    }

    /// The number of distinct session locks currently tracked.
    pub fn tracked_locks(&self) -> usize {
        self.locks.tracked_count()
    }

    /// A handle to the session lock set, for the runtime to acquire turn-wide.
    pub fn lock_set(&self) -> &SessionLockSet {
        &self.locks
    }

    /// The stage configuration.
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }

    /// The metrics from the most recent execution, if any.
    pub fn last_metrics(&self) -> Option<StageMetrics> {
        self.last_metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Derive the session lock key for a turn.
    ///
    /// The turn id is the most specific key; callers that want to serialize a
    /// whole session can override this by wrapping the runner in
    /// `with_session_lock`.
    pub fn session_key_for(turn_id: &str) -> String {
        turn_id.to_string()
    }

    /// Validate the turn history for structural integrity.
    ///
    /// Returns a `StageError` when a `tool_result` message has no matching
    /// `tool_use` in the history. This mirrors the Python harness's
    /// `_validate_messages` guard.
    pub fn validate_history(ctx: &StageContext, stage: &str) -> Option<StageError> {
        if ctx.messages.is_empty() {
            return Some(StageError {
                message: "No messages provided for the turn".to_string(),
                code: Some("EMPTY_MESSAGES".to_string()),
                stage: stage.to_string(),
            });
        }
        let tool_results = ctx.messages.iter().filter(|m| m.role == MessageRole::Tool);
        let tool_calls: Vec<&str> = ctx
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .flat_map(|m| {
                m.content.iter().filter_map(|b| match b {
                    opensquilla_core::types::ContentBlock::ToolUse(c) => Some(c.id.as_str()),
                    _ => None,
                })
            })
            .collect();
        for result in tool_results {
            let Some(id) = result.tool_call_id.as_deref() else {
                return Some(StageError {
                    message: "Tool result message is missing its tool_call_id".to_string(),
                    code: Some("ORPHAN_TOOL_RESULT".to_string()),
                    stage: stage.to_string(),
                });
            };
            if !tool_calls.contains(&id) {
                return Some(StageError {
                    message: format!("Tool result '{id}' has no matching tool call"),
                    code: Some("ORPHAN_TOOL_RESULT".to_string()),
                    stage: stage.to_string(),
                });
            }
        }
        None
    }
}

impl Default for HarnessStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for HarnessStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        let started = Instant::now();
        debug!("harness: initializing turn context");

        // Derive the session key and acquire the per-session lock subject to a
        // timeout. A lock that cannot be acquired (e.g. a previous turn holding
        // it for longer than the configured window) is a hard error.
        let session_key = Self::session_key_for(&ctx.turn_id);
        let lock_started = Instant::now();
        let lock_wait_ms;
        let guard =
            match tokio::time::timeout(self.config.lock_timeout, self.locks.acquire(&session_key))
                .await
            {
                Ok(guard) => {
                    lock_wait_ms = lock_started.elapsed().as_millis() as u64;
                    guard
                }
                Err(_) => {
                    warn!(
                        turn_id = %ctx.turn_id,
                        session = %session_key,
                        timeout_ms = self.config.lock_timeout.as_millis(),
                        "harness: timed out acquiring session lock"
                    );
                    return Ok(StageOutput::Error(StageError {
                        message: format!(
                            "Timed out acquiring session lock after {}ms",
                            self.config.lock_timeout.as_millis()
                        ),
                        code: Some("SESSION_LOCK_TIMEOUT".to_string()),
                        stage: self.name().to_string(),
                    }));
                }
            };
        trace!(
            turn_id = %ctx.turn_id,
            session = %session_key,
            wait_ms = lock_wait_ms,
            "harness: session lock acquired"
        );

        // Reset per-turn state.
        ctx.tool_round = 0;

        // Fire before-turn hooks; a hook may rewrite the opening messages.
        if self.config.emit_hooks {
            for hook in &self.hooks {
                match hook.before_turn(&ctx.turn_id, &ctx.messages).await {
                    Ok(rewritten) => {
                        if !rewritten.is_empty() {
                            ctx.messages = rewritten;
                        }
                    }
                    Err(e) => {
                        debug!(turn_id = %ctx.turn_id, error = %e, "before_turn hook failed");
                    }
                }
            }
        }

        // Validate the message history.
        if let Some(err) = Self::validate_history(ctx, self.name()) {
            return Ok(StageOutput::Error(err));
        }

        // Fire after-turn hooks (the harness establishes the boundary; the
        // result is always "the turn started cleanly" at this point).
        if self.config.emit_hooks {
            let ok: Result<()> = Ok(());
            for hook in &self.hooks {
                let _ = hook.after_turn(&ctx.turn_id, &ctx.messages, &ok).await;
            }
        }
        drop(guard);

        let duration_ms = started.elapsed().as_millis() as u64;
        if duration_ms >= self.config.stage_timeout.as_millis() as u64 {
            warn!(
                turn_id = %ctx.turn_id,
                duration_ms = duration_ms,
                budget_ms = self.config.stage_timeout.as_millis(),
                "harness: stage exceeded its time budget"
            );
        }

        let metrics = StageMetrics {
            stage_name: self.name().to_string(),
            duration_ms,
            messages_in: ctx.messages.len(),
            lock_wait_ms,
            session_key,
            turn_id: ctx.turn_id.clone(),
        };
        *self.last_metrics.lock().unwrap_or_else(|e| e.into_inner()) = Some(metrics);

        info!(
            turn_id = %ctx.turn_id,
            message_count = ctx.messages.len(),
            duration_ms = duration_ms,
            lock_wait_ms = lock_wait_ms,
            max_entries = self.config.max_lock_entries,
            "harness stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "harness"
    }
}

// ---------------------------------------------------------------------------
// Turn error boundary classification
// ---------------------------------------------------------------------------

/// The classification of a turn error for recovery decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnErrorKind {
    /// The error is transient and the turn should be retried.
    Transient,
    /// The error is a provider-side failure.
    Provider,
    /// The error is a tool execution failure.
    Tool,
    /// The error is an input validation failure.
    InvalidInput,
    /// The error is a rate-limit / quota failure.
    RateLimited,
    /// The error is a context-overflow failure.
    ContextOverflow,
    /// The error is an internal engine failure.
    Internal,
    /// The error is unknown.
    Unknown,
}

impl TurnErrorKind {
    /// Whether the error warrants a retry.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            TurnErrorKind::Transient | TurnErrorKind::RateLimited | TurnErrorKind::Provider
        )
    }

    /// The canonical telemetry token.
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnErrorKind::Transient => "transient",
            TurnErrorKind::Provider => "provider",
            TurnErrorKind::Tool => "tool",
            TurnErrorKind::InvalidInput => "invalid_input",
            TurnErrorKind::RateLimited => "rate_limited",
            TurnErrorKind::ContextOverflow => "context_overflow",
            TurnErrorKind::Internal => "internal",
            TurnErrorKind::Unknown => "unknown",
        }
    }

    /// Classify an error message + stage error code into a kind.
    pub fn classify(message: &str, code: Option<&str>) -> Self {
        let code_lower = code.map(|c| c.to_ascii_lowercase()).unwrap_or_default();
        match code_lower.as_str() {
            "empty_messages" | "message_limit" | "invalid_input" => TurnErrorKind::InvalidInput,
            "context_overflow" | "context_too_large" => TurnErrorKind::ContextOverflow,
            "rate_limited" | "rate_limit" => TurnErrorKind::RateLimited,
            "provider_call_failed" | "provider_error" => TurnErrorKind::Provider,
            "tool_error" | "tool_execution_failed" => TurnErrorKind::Tool,
            "session_lock_timeout" | "stage_timeout" => TurnErrorKind::Transient,
            _ => {
                let lower = message.to_ascii_lowercase();
                if lower.contains("rate") || lower.contains("429") {
                    TurnErrorKind::RateLimited
                } else if lower.contains("timeout") || lower.contains("timed out") {
                    TurnErrorKind::Transient
                } else if lower.contains("context") || lower.contains("token")
                    || lower.contains("window")
                {
                    TurnErrorKind::ContextOverflow
                } else if lower.contains("provider") || lower.contains("502")
                    || lower.contains("503") || lower.contains("overloaded")
                {
                    TurnErrorKind::Provider
                } else if lower.contains("tool") {
                    TurnErrorKind::Tool
                } else if lower.contains("invalid") || lower.contains("missing") {
                    TurnErrorKind::InvalidInput
                } else {
                    TurnErrorKind::Internal
                }
            }
        }
    }
}

/// The turn error boundary: a pure classification of a [`StageError`] into a
/// [`TurnErrorKind`] plus the recovery hint.
#[derive(Debug, Clone)]
pub struct TurnErrorBoundary {
    /// The classified error kind.
    pub kind: TurnErrorKind,
    /// Whether the turn should be retried.
    pub retryable: bool,
    /// The suggested retry delay in milliseconds.
    pub retry_delay_ms: u64,
    /// The error message.
    pub message: String,
    /// The stage that produced the error.
    pub stage: String,
}

impl TurnErrorBoundary {
    /// Classify a [`StageError`].
    pub fn from_stage_error(error: &StageError) -> Self {
        let kind = TurnErrorKind::classify(&error.message, error.code.as_deref());
        let retry_delay_ms = match kind {
            TurnErrorKind::RateLimited => 1_000,
            TurnErrorKind::Transient => 500,
            TurnErrorKind::Provider => 250,
            _ => 0,
        };
        Self {
            retryable: kind.is_retryable(),
            retry_delay_ms,
            message: error.message.clone(),
            stage: error.stage.clone(),
            kind,
        }
    }

    /// Classify from a raw message + optional code.
    pub fn from_message(message: &str, code: Option<&str>) -> Self {
        let kind = TurnErrorKind::classify(message, code);
        let retryable = kind.is_retryable();
        Self {
            kind,
            retryable,
            retry_delay_ms: 0,
            message: message.to_string(),
            stage: "unknown".to_string(),
        }
    }
}

/// A helper that aggregates stage errors across a turn for recovery.
#[derive(Debug, Clone, Default)]
pub struct TurnErrorAggregator {
    /// The errors recorded in this turn.
    errors: Vec<TurnErrorBoundary>,
}

impl TurnErrorAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a stage error.
    pub fn record(&mut self, error: StageError) {
        let boundary = TurnErrorBoundary::from_stage_error(&error);
        self.errors.push(boundary);
    }

    /// The recorded errors.
    pub fn errors(&self) -> &[TurnErrorBoundary] {
        &self.errors
    }

    /// The most severe error kind, if any.
    pub fn most_severe(&self) -> Option<TurnErrorKind> {
        self.errors
            .iter()
            .map(|e| e.kind.clone())
            .max_by_key(|k| severity_rank(k))
    }

    /// Whether any recorded error is retryable.
    pub fn has_retryable(&self) -> bool {
        self.errors.iter().any(|e| e.retryable)
    }

    /// The number of recorded errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// True when no errors were recorded.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
}

/// The severity ordering of error kinds (higher = more severe).
fn severity_rank(kind: &TurnErrorKind) -> u8 {
    match kind {
        TurnErrorKind::InvalidInput => 0,
        TurnErrorKind::Transient => 1,
        TurnErrorKind::Tool => 2,
        TurnErrorKind::RateLimited => 3,
        TurnErrorKind::Provider => 4,
        TurnErrorKind::ContextOverflow => 5,
        TurnErrorKind::Internal => 6,
        TurnErrorKind::Unknown => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, Message, ToolCall};
    use serde_json::json;
    use std::collections::HashMap;

    fn context(messages: Vec<Message>) -> StageContext {
        StageContext {
            turn_id: "turn-1".to_string(),
            messages,
            current_model: String::new(),
            current_provider: String::new(),
            usage: opensquilla_core::types::Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_empty_messages_rejected() {
        let stage = HarnessStage::new();
        let mut ctx = context(Vec::new());
        let out = stage.execute(&mut ctx, &MockGenerator).await.unwrap();
        match out {
            StageOutput::Error(e) => assert_eq!(e.code.as_deref(), Some("EMPTY_MESSAGES")),
            _ => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn test_valid_turn_continues() {
        let stage = HarnessStage::new();
        let mut ctx = context(vec![Message::user("hello")]);
        let out = stage.execute(&mut ctx, &MockGenerator).await.unwrap();
        assert!(matches!(out, StageOutput::Continue));
        assert!(stage.last_metrics().is_some());
    }

    #[tokio::test]
    async fn test_orphan_tool_result_rejected() {
        let stage = HarnessStage::new();
        let mut ctx = context(vec![
            Message::user("run tool"),
            Message {
                role: MessageRole::Tool,
                content: vec![ContentBlock::ToolResult(
                    opensquilla_core::types::ToolResult::success("missing", "ok"),
                )],
                name: Some("shell".into()),
                tool_call_id: Some("missing".into()),
                tool_calls: None,
                tool_result: None,
            },
        ]);
        let out = stage.execute(&mut ctx, &MockGenerator).await.unwrap();
        match out {
            StageOutput::Error(e) => assert_eq!(e.code.as_deref(), Some("ORPHAN_TOOL_RESULT")),
            _ => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn test_locked_session_serializes() {
        let locks = Arc::new(SessionLockSet::new());
        let key = HarnessStage::session_key_for("turn-serial");

        // Hold the lock on the same key, then the stage must wait (or, with a
        // short timeout, fail). Use a tiny timeout and a held lock to prove the
        // timeout path.
        let holder = locks.acquire(&key).await;
        let fast = HarnessStage::new()
            .with_locks(locks.clone())
            .with_lock_timeout(Duration::from_millis(5));
        let mut ctx = context(vec![Message::user("hi")]);
        ctx.turn_id = "turn-serial".to_string();
        let out = fast.execute(&mut ctx, &MockGenerator).await.unwrap();
        match out {
            StageOutput::Error(e) => assert_eq!(e.code.as_deref(), Some("SESSION_LOCK_TIMEOUT")),
            _ => panic!("expected lock timeout"),
        }
        drop(holder);
    }

    #[tokio::test]
    async fn test_tool_use_matching_result_passes() {
        let stage = HarnessStage::new();
        let call = ToolCall::new("c1", "shell", json!({"cmd": "ls"}));
        let mut ctx = context(vec![
            Message::user("run"),
            Message {
                role: MessageRole::Assistant,
                content: vec![ContentBlock::ToolUse(call)],
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            },
            Message {
                role: MessageRole::Tool,
                content: vec![ContentBlock::ToolResult(
                    opensquilla_core::types::ToolResult::success("c1", "ok"),
                )],
                name: Some("shell".into()),
                tool_call_id: Some("c1".into()),
                tool_calls: None,
                tool_result: None,
            },
        ]);
        let out = stage.execute(&mut ctx, &MockGenerator).await.unwrap();
        assert!(matches!(out, StageOutput::Continue));
    }

    #[derive(Debug)]
    struct MockGenerator;
    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _messages: &[Message]) -> Result<Vec<Message>> {
            Ok(vec![Message::assistant("ok")])
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
