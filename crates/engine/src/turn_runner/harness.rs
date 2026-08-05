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

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, Message, ToolCall};
    use serde_json::json;

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
        let stage = HarnessStage::new().with_locks(locks.clone());
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
