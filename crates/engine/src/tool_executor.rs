//! Tool execution lifecycle: dispatch, timeout, result capture, error
//! mapping, and streaming output.
//!
//! This module provides a structured tool execution layer that wraps the
//! raw [`crate::runtime::ToolExecutor`] trait with:
//!
//! * configurable timeouts per tool call,
//! * structured error classification and mapping,
//! * result capture and caching,
//! * streaming output for long-running tools,
//! * retry logic for transient failures,
//! * concurrency limits and parallel execution,
//! * execution context propagation.
//!
//! It mirrors the Python backend's `engine/tool_executor.py` and the
//! tool execution lifecycle embedded in `engine/agent.py`.

use crate::agent::RecoveryAction;
use crate::turn_control;
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall, ToolResult};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, instrument, warn};

pub use crate::runtime::{NoopToolExecutor, ToolExecutor};

/// The built-in TokenJuice rules, parsed once and reused across every tool
/// call (mirrors Python's `load_rules()` caching). Only compiled with the
/// `plugins` feature; without it the engine falls back to byte-truncation.
#[cfg(feature = "plugins")]
static TOKENJUICE_RULES: LazyLock<Vec<opensquilla_plugins::Rule>> =
    LazyLock::new(opensquilla_plugins::default_rules);

/// Configuration for a single tool execution.
#[derive(Debug, Clone)]
pub struct ToolExecutionConfig {
    /// The timeout for this tool call.
    pub timeout: Duration,
    /// Whether retries are enabled for transient failures.
    pub retry_enabled: bool,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Base backoff delay between retries.
    pub backoff: Duration,
    /// Whether to capture the result in the result cache.
    pub cache_result: bool,
    /// Whether to stream the tool's output.
    pub stream_output: bool,
    /// Maximum output size in bytes (0 = unbounded).
    pub max_output_bytes: usize,
}

impl Default for ToolExecutionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            retry_enabled: true,
            max_retries: 2,
            backoff: Duration::from_millis(250),
            cache_result: true,
            stream_output: false,
            max_output_bytes: 0,
        }
    }
}

impl ToolExecutionConfig {
    /// Create a new config with the given timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Default::default()
        }
    }

    /// Set the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable or disable retries.
    pub fn with_retries(mut self, enabled: bool) -> Self {
        self.retry_enabled = enabled;
        self
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set the backoff delay.
    pub fn with_backoff(mut self, backoff: Duration) -> Self {
        self.backoff = backoff;
        self
    }

    /// Enable or disable result caching.
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_result = enabled;
        self
    }

    /// Enable or disable streaming output.
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.stream_output = enabled;
        self
    }

    /// Set the maximum output size.
    pub fn with_max_output(mut self, max: usize) -> Self {
        self.max_output_bytes = max;
        self
    }
}

/// The outcome of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolExecutionOutcome {
    /// The tool call that was executed.
    pub call: ToolCall,
    /// The result of the execution, if successful.
    pub result: Option<ToolResult>,
    /// The error message, if the execution failed.
    pub error: Option<String>,
    /// The classified error kind, if any.
    pub error_kind: Option<ToolErrorKind>,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
    /// Number of attempts made (including retries).
    pub attempts: u32,
    /// Whether the result was served from the cache.
    pub from_cache: bool,
    /// Whether the output was truncated.
    pub truncated: bool,
}

impl ToolExecutionOutcome {
    /// Whether the execution succeeded.
    pub fn is_success(&self) -> bool {
        self.error.is_none() && self.result.is_some()
    }

    /// Whether the execution failed.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the result, or a synthetic error result.
    pub fn into_result(self) -> ToolResult {
        match self.result {
            Some(result) => result,
            None => {
                let message = self.error.unwrap_or_else(|| "unknown error".to_string());
                ToolResult::error(&self.call.id, message)
            }
        }
    }

    /// Convert the outcome into a tool-role message.
    pub fn into_message(self) -> Message {
        let name = self.call.name.clone();
        let result = self.into_result();
        Message {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(result.clone())],
            name: Some(name),
            tool_call_id: Some(result.tool_use_id.clone()),
            tool_calls: None,
            tool_result: Some(result),
        }
    }
}

/// Classified tool execution error kinds.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolErrorKind {
    /// The tool call timed out.
    #[error("tool timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },

    /// The tool is unknown or not registered.
    #[error("unknown tool: {0}")]
    UnknownTool(String),

    /// The tool call input was invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The tool execution failed with a runtime error.
    #[error("execution error: {0}")]
    ExecutionError(String),

    /// The tool was not permitted (permissions denied).
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// The tool output exceeded the maximum size.
    #[error("output exceeded maximum size of {max_bytes} bytes")]
    OutputTooLarge { max_bytes: usize },

    /// A transient error that may succeed on retry.
    #[error("transient error: {0}")]
    Transient(String),

    /// The tool executor is not configured.
    #[error("no tool executor configured")]
    NotConfigured,
}

impl ToolErrorKind {
    /// Whether this error is transient and worth retrying.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            ToolErrorKind::Timeout { .. } | ToolErrorKind::Transient(_)
        )
    }

    /// Classify a raw error string into a tool error kind.
    pub fn classify(error: &str) -> Self {
        let lower = error.to_ascii_lowercase();
        if lower.contains("timeout") || lower.contains("timed out") {
            return ToolErrorKind::Timeout { timeout_ms: 60_000 };
        }
        if lower.contains("not found") || lower.contains("unknown tool") {
            return ToolErrorKind::UnknownTool(error.to_string());
        }
        if lower.contains("permission") || lower.contains("denied") {
            return ToolErrorKind::PermissionDenied(error.to_string());
        }
        if lower.contains("rate") || lower.contains("429") || lower.contains("overloaded") {
            return ToolErrorKind::Transient(error.to_string());
        }
        ToolErrorKind::ExecutionError(error.to_string())
    }
}

/// A registry of tool executors indexed by tool name.
///
/// When a tool call arrives, the registry looks up the executor for the
/// tool name. If no executor is registered for the name, a fallback
/// executor is used.
#[derive(Debug)]
pub struct ToolExecutorRegistry {
    /// Named executors for specific tools.
    executors: HashMap<String, Arc<dyn ToolExecutor>>,
    /// The fallback executor used when no named executor matches.
    fallback: Option<Arc<dyn ToolExecutor>>,
}

impl ToolExecutorRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
            fallback: None,
        }
    }

    /// Register an executor for a specific tool name.
    pub fn register(&mut self, tool_name: impl Into<String>, executor: Arc<dyn ToolExecutor>) {
        self.executors.insert(tool_name.into(), executor);
    }

    /// Set the fallback executor.
    pub fn with_fallback(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.fallback = Some(executor);
        self
    }

    /// Resolve the executor for a tool call.
    pub fn resolve(&self, tool_name: &str) -> Option<&Arc<dyn ToolExecutor>> {
        self.executors.get(tool_name).or(self.fallback.as_ref())
    }

    /// The number of registered executors.
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// True when no executors are registered and no fallback is set.
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty() && self.fallback.is_none()
    }
}

impl Default for ToolExecutorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A cache for tool execution results, keyed by a hash of the tool call.
#[derive(Debug, Clone)]
pub struct ToolResultCache {
    /// The cached results, keyed by a canonical hash of the tool call.
    cache: Arc<Mutex<HashMap<String, ToolResult>>>,
    /// Maximum number of entries.
    max_entries: usize,
}

impl ToolResultCache {
    /// Create a new result cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_entries: max_entries.max(1),
        }
    }

    /// Compute a cache key for a tool call.
    pub fn key_for(call: &ToolCall) -> String {
        let input_str = call.input.to_string();
        format!("{}:{}", call.name, input_str)
    }

    /// Look up a cached result.
    pub async fn get(&self, call: &ToolCall) -> Option<ToolResult> {
        let key = Self::key_for(call);
        self.cache.lock().await.get(&key).cloned()
    }

    /// Insert a result into the cache.
    pub async fn insert(&self, call: &ToolCall, result: &ToolResult) {
        let key = Self::key_for(call);
        let mut cache = self.cache.lock().await;
        if cache.len() >= self.max_entries {
            // Evict the oldest entry (first inserted).
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        cache.insert(key, result.clone());
    }

    /// Clear the cache.
    pub async fn clear(&self) {
        self.cache.lock().await.clear();
    }

    /// The number of cached entries.
    pub async fn len(&self) -> usize {
        self.cache.lock().await.len()
    }

    /// True when the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.cache.lock().await.is_empty()
    }
}

impl Default for ToolResultCache {
    fn default() -> Self {
        Self::new(256)
    }
}

/// A concurrency limiter for tool executions.
#[derive(Debug)]
pub struct ToolConcurrencyLimiter {
    /// The semaphore that limits concurrent executions.
    semaphore: Arc<Semaphore>,
    /// The maximum number of concurrent executions.
    max_concurrent: usize,
}

impl ToolConcurrencyLimiter {
    /// Create a new limiter with the given concurrency cap.
    pub fn new(max_concurrent: usize) -> Self {
        let max = max_concurrent.max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(max)),
            max_concurrent: max,
        }
    }

    /// The maximum number of concurrent executions.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Acquire a permit for execution.
    pub async fn acquire(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("semaphore closed")
    }

    /// Try to acquire a permit without waiting.
    pub fn try_acquire(&self) -> Option<tokio::sync::OwnedSemaphorePermit> {
        self.semaphore.clone().try_acquire_owned().ok()
    }

    /// The number of currently available permits.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

impl Default for ToolConcurrencyLimiter {
    fn default() -> Self {
        Self::new(8)
    }
}

/// A streaming output channel for long-running tools.
#[derive(Debug, Clone)]
pub struct ToolOutputStream {
    /// The sender for streaming output chunks.
    tx: tokio::sync::mpsc::Sender<String>,
}

impl ToolOutputStream {
    /// Create a new output stream with the given channel.
    pub fn new(tx: tokio::sync::mpsc::Sender<String>) -> Self {
        Self { tx }
    }

    /// Send a chunk of output.
    pub async fn send(&self, chunk: impl Into<String>) -> std::result::Result<(), String> {
        self.tx
            .send(chunk.into())
            .await
            .map_err(|_| "output stream closed".to_string())
    }

    /// The sender handle.
    pub fn sender(&self) -> &tokio::sync::mpsc::Sender<String> {
        &self.tx
    }
}

/// The structured tool execution engine.
///
/// Wraps a [`ToolExecutor`] with timeout, retry, caching, and concurrency
/// controls. This is the main entry point for the agent loop's tool
/// execution.
#[derive(Debug)]
pub struct ToolExecutionEngine {
    /// The underlying tool executor.
    executor: Arc<dyn ToolExecutor>,
    /// The result cache.
    cache: ToolResultCache,
    /// The concurrency limiter.
    limiter: ToolConcurrencyLimiter,
    /// The default execution config.
    default_config: ToolExecutionConfig,
    /// The permission policy gate (when attached).
    policy: Option<ToolPolicy>,
}

impl ToolExecutionEngine {
    /// Create a new execution engine around the given executor.
    pub fn new(executor: Arc<dyn ToolExecutor>) -> Self {
        Self {
            executor,
            cache: ToolResultCache::default(),
            limiter: ToolConcurrencyLimiter::default(),
            default_config: ToolExecutionConfig::default(),
            policy: None,
        }
    }

    /// Set the result cache.
    pub fn with_cache(mut self, cache: ToolResultCache) -> Self {
        self.cache = cache;
        self
    }

    /// Attach a permission policy that gates every execution.
    pub fn with_policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Set the concurrency limiter.
    pub fn with_limiter(mut self, limiter: ToolConcurrencyLimiter) -> Self {
        self.limiter = limiter;
        self
    }

    /// Set the default execution config.
    pub fn with_default_config(mut self, config: ToolExecutionConfig) -> Self {
        self.default_config = config;
        self
    }

    /// The underlying executor.
    pub fn executor(&self) -> &dyn ToolExecutor {
        self.executor.as_ref()
    }

    /// The result cache.
    pub fn cache(&self) -> &ToolResultCache {
        &self.cache
    }

    /// The concurrency limiter.
    pub fn limiter(&self) -> &ToolConcurrencyLimiter {
        &self.limiter
    }

    /// Execute a single tool call with the full lifecycle.
    ///
    /// This applies:
    /// 1. Cache lookup (if caching is enabled).
    /// 2. Concurrency limiting.
    /// 3. Timeout enforcement.
    /// 4. Retry with backoff (for transient errors).
    /// 5. Result caching.
    /// 6. Output truncation.
    #[instrument(skip(self), fields(tool = %call.name, call_id = %call.id))]
    pub async fn execute(&self, call: &ToolCall) -> ToolExecutionOutcome {
        self.execute_with_config(call, &self.default_config).await
    }

    /// Execute a tool call with a custom config.
    pub async fn execute_with_config(
        &self,
        call: &ToolCall,
        config: &ToolExecutionConfig,
    ) -> ToolExecutionOutcome {
        let start = Instant::now();

        // 0. Permission policy gate.
        if let Some(policy) = &self.policy {
            let permission = policy.evaluate(call);
            if !permission.is_allowed() {
                let reason = match permission {
                    ToolPermission::Deny { reason } => reason,
                    ToolPermission::RequiresConfirmation { reason } => {
                        format!("confirmation required: {reason}")
                    }
                    ToolPermission::Allow => unreachable!(),
                };
                debug!(
                    tool = %call.name,
                    call_id = %call.id,
                    reason = %reason,
                    "tool call blocked by policy"
                );
                return ToolExecutionOutcome {
                    call: call.clone(),
                    result: Some(ToolResult::error(&call.id, reason.clone())),
                    error: None,
                    error_kind: Some(ToolErrorKind::PermissionDenied(reason)),
                    duration: start.elapsed(),
                    attempts: 1,
                    from_cache: false,
                    truncated: false,
                };
            }
        }

        // 1. Cache lookup.
        if config.cache_result {
            if let Some(cached) = self.cache.get(call).await {
                debug!(tool = %call.name, call_id = %call.id, "served from cache");
                return ToolExecutionOutcome {
                    call: call.clone(),
                    result: Some(cached),
                    error: None,
                    error_kind: None,
                    duration: start.elapsed(),
                    attempts: 0,
                    from_cache: true,
                    truncated: false,
                };
            }
        }

        // 2. Concurrency limit.
        let _permit = self.limiter.acquire().await;

        // 3. Retry loop.
        let max_attempts = if config.retry_enabled {
            config.max_retries + 1
        } else {
            1
        };
        let mut attempts = 0u32;
        let mut last_error: Option<String>;
        let mut last_kind: Option<ToolErrorKind>;

        loop {
            attempts += 1;
            debug!(
                tool = %call.name,
                call_id = %call.id,
                attempt = attempts,
                max_attempts = max_attempts,
                "executing tool call"
            );

            let result = tokio::time::timeout(config.timeout, self.executor.execute(call)).await;

            match result {
                Ok(Ok(tool_result)) => {
                    // Before byte-truncating, try a structural reduction via
                    // TokenJuice (mirrors Python's `reduce_tool_result_with_tokenjuice`
                    // call site in `engine/agent.py`). When a rule matches, the
                    // reducer returns a compact head/tail + counter summary that is
                    // shorter than the original; only when no rule matches (or the
                    // `plugins` feature is off) do we fall back to byte-truncation.
                    // The reducer's own no-op guard already discards reductions that
                    // aren't shorter than the input.
                    #[cfg(feature = "plugins")]
                    let tokenjuice_reduced: Option<String> = {
                        let arguments = Some(&call.input);
                        let command = call.input.get("command").and_then(|v| v.as_str());
                        let max_inline = if config.max_output_bytes > 0 {
                            Some(config.max_output_bytes)
                        } else {
                            None
                        };
                        match opensquilla_plugins::reduce_tool_result_with_limit(
                            &call.name,
                            &tool_result.content,
                            tool_result.is_error,
                            arguments,
                            command,
                            &TOKENJUICE_RULES,
                            max_inline,
                        ) {
                            Some(reduction) => {
                                debug!(
                                    tool = %call.name,
                                    call_id = %call.id,
                                    reducer = ?reduction.reducer,
                                    raw_chars = reduction.raw_chars,
                                    reduced_chars = reduction.reduced_chars,
                                    ratio = %format!("{:.2}", reduction.ratio),
                                    "tokenjuice reduced tool result"
                                );
                                Some(reduction.inline_text)
                            }
                            None => None,
                        }
                    };

                    #[cfg(not(feature = "plugins"))]
                    let tokenjuice_reduced: Option<String> = None;

                    // Truncate output if needed: skip when TokenJuice already
                    // produced a (shorter) inline summary.
                    let (final_result, truncated) = if let Some(reduced) = tokenjuice_reduced {
                        (
                            ToolResult {
                                content: reduced,
                                ..tool_result
                            },
                            false,
                        )
                    } else if config.max_output_bytes > 0
                        && tool_result.content.len() > config.max_output_bytes
                    {
                        let truncated_content: String = tool_result
                            .content
                            .chars()
                            .take(config.max_output_bytes)
                            .collect();
                        (
                            ToolResult {
                                content: format!("{truncated_content}\n[...truncated]"),
                                ..tool_result
                            },
                            true,
                        )
                    } else {
                        (tool_result, false)
                    };

                    // TODO(safety): the Python backend XML-escapes untrusted tool
                    // output via `opensquilla_safety::injection::wrap_untrusted_with_source`
                    // (signature `wrap_untrusted_with_source(content: &str, source: &str)
                    // -> String`) before it re-enters the LLM context. Python only
                    // wraps workspace-context and ensemble-candidate text today, NOT
                    // the tool-result projection path, so the gating condition for
                    // tool results is unresolved. When that policy is decided, wrap
                    // `final_result.content` here when the result is untrusted
                    // (e.g. shell/file output) and leave trusted/user-requested
                    // output unwrapped.

                    // Cache the result.
                    if config.cache_result {
                        self.cache.insert(call, &final_result).await;
                    }

                    return ToolExecutionOutcome {
                        call: call.clone(),
                        result: Some(final_result),
                        error: None,
                        error_kind: None,
                        duration: start.elapsed(),
                        attempts,
                        from_cache: false,
                        truncated,
                    };
                }
                Ok(Err(e)) => {
                    let error_str = e.to_string();
                    let kind = ToolErrorKind::classify(&error_str);
                    last_error = Some(error_str);
                    last_kind = Some(kind.clone());

                    if !config.retry_enabled || !kind.is_transient() || attempts >= max_attempts {
                        break;
                    }

                    let backoff_ms = config.backoff.as_millis() as u64 * (1 << (attempts - 1));
                    warn!(
                        tool = %call.name,
                        call_id = %call.id,
                        attempt = attempts,
                        error = %last_error.as_ref().unwrap(),
                        backoff_ms = backoff_ms,
                        "tool execution failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms.min(10_000))).await;
                }
                Err(_) => {
                    let kind = ToolErrorKind::Timeout {
                        timeout_ms: config.timeout.as_millis() as u64,
                    };
                    last_error = Some(kind.to_string());
                    last_kind = Some(kind.clone());

                    if !config.retry_enabled || !kind.is_transient() || attempts >= max_attempts {
                        break;
                    }

                    let backoff_ms = config.backoff.as_millis() as u64 * (1 << (attempts - 1));
                    warn!(
                        tool = %call.name,
                        call_id = %call.id,
                        attempt = attempts,
                        timeout_ms = config.timeout.as_millis(),
                        backoff_ms = backoff_ms,
                        "tool execution timed out, retrying"
                    );
                    tokio::time::sleep(Duration::from_millis(backoff_ms.min(10_000))).await;
                }
            }
        }

        // All retries exhausted; return the error.
        ToolExecutionOutcome {
            call: call.clone(),
            result: None,
            error: last_error,
            error_kind: last_kind,
            duration: start.elapsed(),
            attempts,
            from_cache: false,
            truncated: false,
        }
    }

    /// Execute a batch of tool calls in parallel, respecting the concurrency
    /// limit.
    pub async fn execute_batch(
        &self,
        calls: &[ToolCall],
        config: &ToolExecutionConfig,
    ) -> Vec<ToolExecutionOutcome> {
        let futures: Vec<_> = calls
            .iter()
            .map(|call| self.execute_with_config(call, config))
            .collect();
        futures::future::join_all(futures).await
    }

    /// Execute a batch of tool calls and collect their results as a tool-role
    /// message.
    pub async fn execute_batch_into_message(
        &self,
        calls: &[ToolCall],
        config: &ToolExecutionConfig,
    ) -> Message {
        let outcomes = self.execute_batch(calls, config).await;
        let results: Vec<ToolResult> = outcomes.into_iter().map(|o| o.into_result()).collect();
        let content = results.into_iter().map(ContentBlock::ToolResult).collect();
        Message {
            role: MessageRole::Tool,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    /// Classify a tool execution outcome into a recovery action.
    pub fn classify_outcome(outcome: &ToolExecutionOutcome) -> RecoveryAction {
        if outcome.is_success() {
            return RecoveryAction::Continue;
        }
        match &outcome.error_kind {
            Some(ToolErrorKind::Timeout { .. }) | Some(ToolErrorKind::Transient(_)) => {
                RecoveryAction::Retry { delay_ms: 500 }
            }
            Some(ToolErrorKind::PermissionDenied(_)) => RecoveryAction::Stop {
                message: "Permission denied".to_string(),
            },
            Some(ToolErrorKind::UnknownTool(_)) => RecoveryAction::Continue,
            Some(ToolErrorKind::NotConfigured) => RecoveryAction::Continue,
            _ => RecoveryAction::Continue,
        }
    }

    /// Clear the result cache.
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
    }
}

/// A hook that observes tool execution lifecycle events.
#[async_trait]
pub trait ToolExecutionHook: Send + Sync + fmt::Debug {
    /// Called before a tool call is executed.
    async fn before_execution(&self, call: &ToolCall) -> Result<()>;

    /// Called after a tool call completes.
    async fn after_execution(&self, call: &ToolCall, outcome: &ToolExecutionOutcome) -> Result<()>;

    /// Called when a tool call fails.
    async fn on_error(&self, call: &ToolCall, kind: &ToolErrorKind) -> Result<()>;
}

/// The permission decision for a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolPermission {
    /// The tool call is allowed.
    Allow,
    /// The tool call is denied.
    Deny {
        /// The reason for the denial.
        reason: String,
    },
    /// The tool call requires confirmation.
    RequiresConfirmation {
        /// The reason confirmation is required.
        reason: String,
    },
}

impl ToolPermission {
    /// Whether the tool call is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, ToolPermission::Allow)
    }

    /// Whether the tool call is denied.
    pub fn is_denied(&self) -> bool {
        matches!(self, ToolPermission::Deny { .. })
    }
}

/// A rule that decides whether a tool call is permitted.
///
/// Rules are evaluated in order; the first non-`Allow` result wins. When no
/// rule fires, the policy's default permission applies.
#[derive(Debug, Clone)]
pub struct ToolPermissionRule {
    /// The tool name pattern this rule applies to (exact match, or `*` for
    /// all tools).
    pub tool_pattern: String,
    /// The permission this rule grants.
    pub permission: ToolPermission,
}

impl ToolPermissionRule {
    /// Create a new permission rule.
    pub fn new(tool_pattern: impl Into<String>, permission: ToolPermission) -> Self {
        Self {
            tool_pattern: tool_pattern.into(),
            permission,
        }
    }

    /// Whether this rule applies to the given tool name.
    pub fn applies_to(&self, tool_name: &str) -> bool {
        self.tool_pattern == "*" || self.tool_pattern == tool_name
    }
}

/// A policy for gating tool execution.
///
/// The policy evaluates registered rules and applies the default permission
/// when no rule matches. It supports a separate default for subprocess tools
/// (shell, git, code execution) so dangerous tools are gated by default.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// The ordered permission rules.
    rules: Vec<ToolPermissionRule>,
    /// The default permission for matched tools.
    default_permission: ToolPermission,
    /// The default permission for subprocess tools (when stricter).
    subprocess_permission: ToolPermission,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default_permission: ToolPermission::Allow,
            subprocess_permission: ToolPermission::Deny {
                reason: "subprocess execution not permitted by policy".to_string(),
            },
        }
    }
}

impl ToolPolicy {
    /// Create a new policy with the given default permission.
    pub fn new(default_permission: ToolPermission) -> Self {
        Self {
            default_permission,
            ..Default::default()
        }
    }

    /// A permissive policy that allows everything.
    pub fn permissive() -> Self {
        Self {
            default_permission: ToolPermission::Allow,
            subprocess_permission: ToolPermission::Allow,
            rules: Vec::new(),
        }
    }

    /// A restrictive policy that denies everything except explicitly allowed
    /// tools.
    pub fn restrictive() -> Self {
        Self {
            default_permission: ToolPermission::Deny {
                reason: "tool not permitted by policy".to_string(),
            },
            subprocess_permission: ToolPermission::Deny {
                reason: "subprocess execution not permitted by policy".to_string(),
            },
            rules: Vec::new(),
        }
    }

    /// Add a permission rule.
    pub fn add_rule(mut self, rule: ToolPermissionRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Set the default permission for subprocess tools.
    pub fn with_subprocess_permission(mut self, permission: ToolPermission) -> Self {
        self.subprocess_permission = permission;
        self
    }

    /// Evaluate the policy for a tool call.
    pub fn evaluate(&self, call: &ToolCall) -> ToolPermission {
        // Rules first (in order).
        for rule in &self.rules {
            if rule.applies_to(&call.name) {
                return rule.permission.clone();
            }
        }

        // Subprocess tools use the stricter default.
        if crate::agent::is_subprocess_tool(&call.name) {
            return self.subprocess_permission.clone();
        }

        self.default_permission.clone()
    }

    /// Evaluate a batch of tool calls, returning the denied calls.
    pub fn evaluate_batch(&self, calls: &[ToolCall]) -> Vec<(ToolCall, ToolPermission)> {
        calls
            .iter()
            .filter(|c| !self.evaluate(c).is_allowed())
            .map(|c| (c.clone(), self.evaluate(c)))
            .collect()
    }

    /// Whether a tool call is allowed.
    pub fn is_allowed(&self, call: &ToolCall) -> bool {
        self.evaluate(call).is_allowed()
    }
}

/// A logging tool hook that records tool execution events.
#[derive(Debug)]
pub struct LoggingToolHook;

#[async_trait]
impl ToolExecutionHook for LoggingToolHook {
    async fn before_execution(&self, call: &ToolCall) -> Result<()> {
        info!(tool = %call.name, call_id = %call.id, "tool execution starting");
        Ok(())
    }

    async fn after_execution(&self, call: &ToolCall, outcome: &ToolExecutionOutcome) -> Result<()> {
        info!(
            tool = %call.name,
            call_id = %call.id,
            success = outcome.is_success(),
            duration_ms = outcome.duration.as_millis(),
            attempts = outcome.attempts,
            from_cache = outcome.from_cache,
            "tool execution complete"
        );
        Ok(())
    }

    async fn on_error(&self, call: &ToolCall, kind: &ToolErrorKind) -> Result<()> {
        warn!(
            tool = %call.name,
            call_id = %call.id,
            error = %kind,
            "tool execution error"
        );
        Ok(())
    }
}

/// A builder for constructing a tool execution engine.
#[derive(Debug)]
pub struct ToolExecutionEngineBuilder {
    executor: Option<Arc<dyn ToolExecutor>>,
    cache: Option<ToolResultCache>,
    limiter: Option<ToolConcurrencyLimiter>,
    config: ToolExecutionConfig,
    policy: Option<ToolPolicy>,
}

impl ToolExecutionEngineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            executor: None,
            cache: None,
            limiter: None,
            config: ToolExecutionConfig::default(),
            policy: None,
        }
    }

    /// Set the underlying executor.
    pub fn executor(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Set the result cache.
    pub fn cache(mut self, cache: ToolResultCache) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set the concurrency limiter.
    pub fn limiter(mut self, limiter: ToolConcurrencyLimiter) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// Attach a permission policy.
    pub fn policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Set the default execution config.
    pub fn config(mut self, config: ToolExecutionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the maximum number of retries.
    pub fn max_retries(mut self, max: u32) -> Self {
        self.config.max_retries = max;
        self
    }

    /// Build the execution engine.
    pub fn build(self) -> ToolExecutionEngine {
        let executor = self.executor.expect("executor is required");
        let mut engine = ToolExecutionEngine::new(executor).with_default_config(self.config);
        if let Some(cache) = self.cache {
            engine = engine.with_cache(cache);
        }
        if let Some(limiter) = self.limiter {
            engine = engine.with_limiter(limiter);
        }
        if let Some(policy) = self.policy {
            engine = engine.with_policy(policy);
        }
        engine
    }
}

impl Default for ToolExecutionEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a batch of tool calls before execution.
///
/// Returns a list of valid calls and a list of invalid calls with their
/// error messages.
pub fn validate_tool_calls(calls: &[ToolCall]) -> (Vec<ToolCall>, Vec<(ToolCall, ToolErrorKind)>) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();
    for call in calls {
        match turn_control::validate_tool_calls(std::slice::from_ref(call)) {
            Ok(()) => valid.push(call.clone()),
            Err(e) => invalid.push((call.clone(), ToolErrorKind::InvalidInput(format!("{e:?}")))),
        }
    }
    (valid, invalid)
}

/// Build a tool-result message from a call and its result (mirrors
/// `crate::runtime::tool_result_message` but is publicly accessible).
pub fn build_tool_result_message(call: ToolCall, result: ToolResult) -> Message {
    Message {
        role: MessageRole::Tool,
        content: vec![ContentBlock::ToolResult(result.clone())],
        name: Some(call.name),
        tool_call_id: Some(result.tool_use_id.clone()),
        tool_calls: None,
        tool_result: Some(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A mock executor that returns a configurable result.
    #[derive(Debug, Clone)]
    struct MockExecutor {
        result_text: String,
        is_error: bool,
        delay_ms: u64,
    }

    #[async_trait]
    impl ToolExecutor for MockExecutor {
        async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            if self.is_error {
                Ok(ToolResult::error(&call.id, &self.result_text))
            } else {
                Ok(ToolResult::success(&call.id, &self.result_text))
            }
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall::new("c1", name, json!({}))
    }

    #[test]
    fn test_tool_error_kind_classify() {
        assert!(matches!(
            ToolErrorKind::classify("timed out after 30s"),
            ToolErrorKind::Timeout { .. }
        ));
        assert!(matches!(
            ToolErrorKind::classify("unknown tool: foo"),
            ToolErrorKind::UnknownTool(_)
        ));
        assert!(matches!(
            ToolErrorKind::classify("permission denied"),
            ToolErrorKind::PermissionDenied(_)
        ));
        assert!(matches!(
            ToolErrorKind::classify("rate limited"),
            ToolErrorKind::Transient(_)
        ));
        assert!(matches!(
            ToolErrorKind::classify("random error"),
            ToolErrorKind::ExecutionError(_)
        ));
    }

    #[test]
    fn test_tool_error_kind_is_transient() {
        assert!(ToolErrorKind::Timeout { timeout_ms: 1000 }.is_transient());
        assert!(ToolErrorKind::Transient("rate".to_string()).is_transient());
        assert!(!ToolErrorKind::UnknownTool("x".to_string()).is_transient());
        assert!(!ToolErrorKind::ExecutionError("x".to_string()).is_transient());
    }

    #[tokio::test]
    async fn test_execute_success() {
        let executor = Arc::new(MockExecutor {
            result_text: "ok".to_string(),
            is_error: false,
            delay_ms: 0,
        });
        let engine = ToolExecutionEngine::new(executor);
        let outcome = engine.execute(&call("read_file")).await;
        assert!(outcome.is_success());
        assert_eq!(outcome.attempts, 1);
        assert!(!outcome.from_cache);
    }

    #[tokio::test]
    async fn test_execute_cached() {
        let executor = Arc::new(MockExecutor {
            result_text: "cached".to_string(),
            is_error: false,
            delay_ms: 0,
        });
        let engine = ToolExecutionEngine::new(executor);
        let call = call("read_file");
        let _ = engine.execute(&call).await;
        let outcome = engine.execute(&call).await;
        assert!(outcome.is_success());
        assert!(outcome.from_cache);
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let executor = Arc::new(MockExecutor {
            result_text: "slow".to_string(),
            is_error: false,
            delay_ms: 100,
        });
        let config = ToolExecutionConfig::new(Duration::from_millis(10)).with_retries(false);
        let engine = ToolExecutionEngine::new(executor).with_default_config(config);
        let outcome = engine.execute(&call("slow_tool")).await;
        assert!(outcome.is_error());
        assert!(matches!(
            outcome.error_kind,
            Some(ToolErrorKind::Timeout { .. })
        ));
    }

    #[tokio::test]
    async fn test_execute_batch() {
        let executor = Arc::new(MockExecutor {
            result_text: "ok".to_string(),
            is_error: false,
            delay_ms: 0,
        });
        let engine = ToolExecutionEngine::new(executor);
        let calls = vec![call("a"), call("b"), call("c")];
        let outcomes = engine
            .execute_batch(&calls, &ToolExecutionConfig::default())
            .await;
        assert_eq!(outcomes.len(), 3);
        for outcome in &outcomes {
            assert!(outcome.is_success());
        }
    }

    #[tokio::test]
    async fn test_execute_batch_into_message() {
        let executor = Arc::new(MockExecutor {
            result_text: "ok".to_string(),
            is_error: false,
            delay_ms: 0,
        });
        let engine = ToolExecutionEngine::new(executor);
        let calls = vec![call("a"), call("b")];
        let message = engine
            .execute_batch_into_message(&calls, &ToolExecutionConfig::default())
            .await;
        assert_eq!(message.role, MessageRole::Tool);
        assert_eq!(message.content.len(), 2);
    }

    #[test]
    fn test_tool_executor_registry() {
        let mut registry = ToolExecutorRegistry::new();
        let executor = Arc::new(NoopToolExecutor);
        registry.register("read_file", executor);
        assert!(registry.resolve("read_file").is_some());
        assert!(registry.resolve("unknown").is_none());
    }

    #[test]
    fn test_tool_result_cache_key() {
        let call1 = ToolCall::new("c1", "read_file", json!({"path": "/a"}));
        let call2 = ToolCall::new("c2", "read_file", json!({"path": "/a"}));
        // Same name + same input = same key.
        assert_eq!(
            ToolResultCache::key_for(&call1),
            ToolResultCache::key_for(&call2)
        );
    }

    #[tokio::test]
    async fn test_tool_result_cache() {
        let cache = ToolResultCache::new(10);
        let call = call("read_file");
        assert!(cache.get(&call).await.is_none());
        cache.insert(&call, &ToolResult::success("c1", "ok")).await;
        assert!(cache.get(&call).await.is_some());
        assert_eq!(cache.len().await, 1);
        cache.clear().await;
        assert!(cache.get(&call).await.is_none());
    }

    #[test]
    fn test_tool_execution_outcome_into_result() {
        let outcome = ToolExecutionOutcome {
            call: call("read_file"),
            result: Some(ToolResult::success("c1", "ok")),
            error: None,
            error_kind: None,
            duration: Duration::from_millis(10),
            attempts: 1,
            from_cache: false,
            truncated: false,
        };
        assert!(outcome.is_success());
        let result = outcome.into_result();
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_execution_outcome_error_into_result() {
        let outcome = ToolExecutionOutcome {
            call: call("read_file"),
            result: None,
            error: Some("failed".to_string()),
            error_kind: Some(ToolErrorKind::ExecutionError("failed".to_string())),
            duration: Duration::from_millis(10),
            attempts: 1,
            from_cache: false,
            truncated: false,
        };
        assert!(outcome.is_error());
        let result = outcome.into_result();
        assert!(result.is_error);
    }

    #[test]
    fn test_validate_tool_calls() {
        let calls = vec![
            ToolCall::new("c1", "read_file", json!({"path": "/a"})),
            ToolCall::new("", "write_file", json!({})),
            ToolCall::new("c3", "", json!({})),
        ];
        let (valid, invalid) = validate_tool_calls(&calls);
        assert_eq!(valid.len(), 1);
        assert_eq!(invalid.len(), 2);
    }

    #[tokio::test]
    async fn test_concurrency_limiter() {
        let limiter = ToolConcurrencyLimiter::new(2);
        let _p1 = limiter.acquire().await;
        let _p2 = limiter.acquire().await;
        // Now we're at capacity; try_acquire should fail.
        assert!(limiter.try_acquire().is_none());
        drop(_p1);
        // Now a permit is available again.
        assert!(limiter.try_acquire().is_some());
    }

    #[test]
    fn test_config_builder() {
        let config = ToolExecutionConfig::default()
            .with_timeout(Duration::from_secs(30))
            .with_max_retries(3)
            .with_backoff(Duration::from_millis(500))
            .with_cache(false)
            .with_streaming(true);
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
        assert!(!config.cache_result);
        assert!(config.stream_output);
    }
}
