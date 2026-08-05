//! Compaction stage.
//!
//! Mirrors the Python `engine/turn_runner/compaction_and_history_stage.py`
//! (the compaction half) and `engine/compaction_control.py`. It runs after
//! bootstrap. It:
//!
//! * estimates the conversation's token footprint,
//! * decides whether compaction is warranted via
//!   [`crate::compaction_control::decide_compaction`],
//! * when it is, applies the strategy (summarize / drop-oldest / truncate)
//!   and then repairs the tool-call pairing so the provider never receives an
//!   orphaned `tool_result`,
//! * fires [`crate::hooks::CompactionHook`] around the message surgery,
//! * enforces the context-window budget by re-truncating when the result is
//!   still over budget.
//!
//! When the `session` feature is enabled and a [`opensquilla_session::SessionStorage`]
//! is attached, the stage also drives the session crate's
//! [`CompactionPlanner`]/[`CompactionExecutor`] so persisted transcripts are
//! compacted in the same turn.

use crate::agent::TurnGenerator;
use crate::compaction_control::{self, CompactionDecision, CompactionInput, CompactionStrategy};
use crate::hooks::CompactionHook;
use crate::stages::{Stage, StageContext, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use std::fmt;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// The outcome of a compaction run, for observability.
#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    /// Number of messages before compaction.
    pub before: usize,
    /// Number of messages after compaction.
    pub after: usize,
    /// The strategy that was applied.
    pub strategy: CompactionStrategy,
    /// Estimated tokens before compaction.
    pub tokens_before: u64,
    /// Estimated tokens after compaction.
    pub tokens_after: u64,
    /// Whether the tool-call pairing was repaired after surgery.
    pub repaired_tool_pairs: bool,
}

impl CompactionOutcome {
    /// Number of messages reclaimed.
    pub fn reclaimed(&self) -> usize {
        self.before.saturating_sub(self.after)
    }
}

/// The compaction stage in the turn pipeline.
pub struct CompactionStage {
    /// Message count above which compaction is considered even when the token
    /// estimate is low.
    max_messages: usize,
    /// Whether compaction is enabled at all.
    enabled: bool,
    /// The model context window in tokens, used by the decision.
    context_window_tokens: u64,
    /// Optional compaction hooks fired around the surgery.
    hooks: Vec<Arc<dyn CompactionHook + Send + Sync>>,
    /// Whether the context window budget is enforced after surgery by
    /// re-truncating an over-budget result.
    enforce_budget: bool,
    /// The most recent compaction outcome.
    last_outcome: std::sync::Mutex<Option<CompactionOutcome>>,
    /// Optional session storage driving persisted compaction (feature `session`).
    #[cfg(feature = "session")]
    session_storage: Option<opensquilla_session::SessionStorage>,
}

impl fmt::Debug for CompactionStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompactionStage")
            .field("max_messages", &self.max_messages)
            .field("enabled", &self.enabled)
            .field("context_window_tokens", &self.context_window_tokens)
            .field("hooks", &self.hooks.len())
            .field("enforce_budget", &self.enforce_budget)
            .finish_non_exhaustive()
    }
}

impl CompactionStage {
    /// Create a new compaction stage with the given message limit.
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages,
            enabled: true,
            context_window_tokens: 128_000,
            hooks: Vec::new(),
            enforce_budget: true,
            last_outcome: std::sync::Mutex::new(None),
            #[cfg(feature = "session")]
            session_storage: None,
        }
    }

    /// Enable or disable compaction.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the model context window (tokens) used by the decision.
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window_tokens = tokens.max(1);
        self
    }

    /// Register compaction hooks fired around the surgery.
    pub fn with_hooks(mut self, hooks: Vec<Arc<dyn CompactionHook + Send + Sync>>) -> Self {
        self.hooks = hooks;
        self
    }

    /// Enable or disable post-surgery context-window budget enforcement.
    pub fn with_budget_enforcement(mut self, enforce: bool) -> Self {
        self.enforce_budget = enforce;
        self
    }

    /// Attach a session storage so persisted transcripts are compacted too.
    ///
    /// Only available when the `session` feature is enabled.
    #[cfg(feature = "session")]
    pub fn with_session_storage(mut self, storage: opensquilla_session::SessionStorage) -> Self {
        self.session_storage = Some(storage);
        self
    }

    /// The most recent compaction outcome, if any.
    pub fn last_outcome(&self) -> Option<CompactionOutcome> {
        self.last_outcome
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Estimate the token footprint of the message list.
    ///
    /// A cheap heuristic: roughly one token per four content characters
    /// (matching the provider stage's estimate).
    pub fn estimate_tokens(&self, messages: &[Message]) -> u64 {
        messages
            .iter()
            .map(|m| m.text_content().chars().count() as u64 / 4)
            .sum()
    }

    /// Decide whether and how to compact the given message list.
    pub fn decide(&self, messages: &[Message]) -> CompactionDecision {
        let input = CompactionInput {
            token_count: self.estimate_tokens(messages),
            message_count: messages.len(),
            context_window_tokens: self.context_window_tokens,
            ..Default::default()
        };
        compaction_control::decide_compaction(&input)
    }

    /// Apply a compaction strategy to the message list.
    ///
    /// `Summarize` keeps system messages and replaces the middle with a single
    /// summary placeholder; `DropOldest` and `Truncate` both keep the most
    /// recent messages within the budget.
    pub fn apply(&self, messages: &[Message], strategy: CompactionStrategy) -> Vec<Message> {
        let budget = compaction_control::truncation_budget(self.max_messages);
        let system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .cloned()
            .collect();
        let rest: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .cloned()
            .collect();

        let keep = match strategy {
            CompactionStrategy::Summarize => {
                if rest.is_empty() {
                    rest
                } else {
                    let mut kept: Vec<Message> = Vec::new();
                    kept.extend(rest.iter().take(1).cloned());
                    let tail_start = rest.len().saturating_sub(budget.saturating_sub(1));
                    if tail_start > 1 {
                        kept.push(Message::system(format!(
                            "[{n} earlier messages summarized]",
                            n = tail_start - 1
                        )));
                    }
                    kept.extend(rest.iter().skip(tail_start).cloned());
                    kept
                }
            }
            CompactionStrategy::DropOldest | CompactionStrategy::Truncate => {
                let tail_start = rest.len().saturating_sub(budget);
                rest.iter().skip(tail_start).cloned().collect()
            }
        };

        let mut out = system;
        out.extend(keep);
        out
    }

    /// Enforce the context-window budget by hard-truncating an over-budget
    /// result to fit within the window.
    ///
    /// Uses [`crate::history::truncate_to_budget`] with a conservative
    /// four-characters-per-token estimate so the result always fits.
    pub fn enforce_context_window(&self, messages: Vec<Message>) -> Vec<Message> {
        let estimated = self.estimate_tokens(&messages);
        if estimated <= self.context_window_tokens {
            return messages;
        }
        let budget_tokens = self.context_window_tokens.saturating_sub(1);
        crate::history::truncate_to_budget(&messages, budget_tokens, 0.25)
    }

    /// Perform the full message surgery and return the outcome.
    ///
    /// The surgery is: fire before-hooks, apply the strategy, repair the
    /// tool-call pairing, enforce the context-window budget, and fire
    /// after-hooks. Hook failures never break the turn.
    pub async fn compact_messages(
        &self,
        turn_id: &str,
        messages: Vec<Message>,
        strategy: CompactionStrategy,
    ) -> (Vec<Message>, CompactionOutcome) {
        let before = messages.len();
        let tokens_before = self.estimate_tokens(&messages);

        for hook in &self.hooks {
            let _ = hook.before_compaction(turn_id, before).await;
        }

        let mut compacted = self.apply(&messages, strategy);
        let after = compacted.len();

        // Repair the tool-call pairing so the provider never sees an orphaned
        // tool_result after the surgery drops its tool_use.
        let mut repaired_tool_pairs = false;
        if compacted.iter().any(|m| {
            matches!(m.role, MessageRole::Tool)
                || m.content.iter().any(|b| matches!(b, opensquilla_core::types::ContentBlock::ToolUse(_)))
        }) {
            let outcome = crate::history::repair_tool_pairs(&compacted);
            if outcome.removed_results > 0 || outcome.unpaired_calls > 0 {
                repaired_tool_pairs = true;
            }
            compacted = outcome.messages;
        }

        // Enforce the context-window budget.
        if self.enforce_budget {
            compacted = self.enforce_context_window(compacted);
        }

        let tokens_after = self.estimate_tokens(&compacted);

        for hook in &self.hooks {
            let _ = hook.after_compaction(turn_id, before, compacted.len()).await;
        }

        let outcome = CompactionOutcome {
            before,
            after: compacted.len(),
            strategy,
            tokens_before,
            tokens_after,
            repaired_tool_pairs,
        };
        *self.last_outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome.clone());
        (compacted, outcome)
    }

    /// Drive persisted compaction through the session crate when a storage is
    /// attached and the `session` feature is enabled.
    ///
    /// Returns `Ok(true)` when a persisted compaction ran, `Ok(false)` when no
    /// storage is attached or the session row is absent (the caller falls back
    /// to in-memory compaction).
    #[cfg(feature = "session")]
    pub async fn run_session_compaction(
        &self,
        turn_id: &str,
    ) -> Result<bool> {
        use opensquilla_core::error::Error;
        let Some(storage) = &self.session_storage else {
            return Ok(false);
        };
        // The turn id encodes the session id as its first UUID component when
        // the runtime routes through the session manager; fall back to hashing
        // the turn id into a stable UUID so the storage path is always usable.
        let session_id = parse_session_id(turn_id).unwrap_or_else(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hasher::write(&mut hasher, turn_id.as_bytes());
            uuid::Uuid::from_u64_pair(hasher.finish(), 0)
        });

        let planner = opensquilla_session::CompactionPlanner::new()
            .with_threshold(self.context_window_tokens.saturating_mul(3) / 4);

        // rusqlite queries are short and the storage is internally synchronized;
        // calling them from async context blocks the executor only briefly.
        let Some(session) = storage
            .get_session(&session_id)
            .map_err(|e| Error::Storage(e.to_string()))?
        else {
            // Session row absent: fall back to in-memory compaction.
            return Ok(false);
        };
        let entries = storage
            .list_by_session(&session_id)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let plan = planner
            .plan_compaction(&session, &entries)
            .map_err(|e| Error::Storage(e.to_string()))?;
        if plan.is_noop() {
            return Ok(false);
        }

        // Mark every compacted entry (up to the newest compacted row) and
        // persist a marker summary so the session history reflects the run.
        let up_to = plan
            .entries_to_compact
            .iter()
            .map(|e| e.created_at)
            .max();
        if let Some(up_to) = up_to {
            let count = storage
                .mark_entries_compacted(&session_id, &up_to)
                .map_err(|e| Error::Storage(e.to_string()))?;
            let summary = opensquilla_session::SessionSummary {
                id: uuid::Uuid::new_v4(),
                session_id,
                summary: format!(
                    "[{}] compacted {} entries, estimated savings {} tokens",
                    plan.strategy.label(),
                    count,
                    plan.estimated_savings
                ),
                created_at: chrono::Utc::now(),
                token_count: plan.estimated_savings.min(1024),
                is_active: true,
            };
            storage
                .insert_summary(&summary)
                .map_err(|e| Error::Storage(e.to_string()))?;
        }

        debug!(
            turn_id = %turn_id,
            session = %session_id,
            strategy = %plan.strategy.label(),
            entries = plan.entries_to_compact.len(),
            savings = plan.estimated_savings,
            "session compaction finished"
        );
        Ok(true)
    }
}

impl Default for CompactionStage {
    fn default() -> Self {
        Self::new(50)
    }
}

#[async_trait]
impl Stage for CompactionStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        if !self.enabled {
            debug!("compaction disabled, skipping");
            return Ok(StageOutput::Continue);
        }

        let decision = self.decide(&ctx.messages);
        let strategy = match decision {
            CompactionDecision::NoCompaction => {
                debug!(
                    turn_id = %ctx.turn_id,
                    message_count = ctx.messages.len(),
                    "compaction not needed"
                );
                return Ok(StageOutput::Continue);
            }
            CompactionDecision::Compact(s) => s,
            CompactionDecision::UrgentCompaction(s) => s,
        };

        // Attempt persisted compaction first; when the storage path runs, the
        // in-memory surgery still applies to the live context.
        #[cfg(feature = "session")]
        let _ = self.run_session_compaction(&ctx.turn_id).await;

        let turn_id = ctx.turn_id.clone();
        let (compacted, outcome) = self
            .compact_messages(&turn_id, ctx.messages.clone(), strategy)
            .await;
        ctx.messages = compacted;

        info!(
            turn_id = %ctx.turn_id,
            before = outcome.before,
            after = outcome.after,
            reclaimed = outcome.reclaimed(),
            strategy = ?strategy,
            tokens_before = outcome.tokens_before,
            tokens_after = outcome.tokens_after,
            repaired = outcome.repaired_tool_pairs,
            "compaction stage complete"
        );

        if outcome.reclaimed() == 0 {
            warn!(
                turn_id = %ctx.turn_id,
                "compaction ran but reclaimed no messages"
            );
        }

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "compaction"
    }
}

/// Parse a session UUID from a turn id when the id embeds one.
///
/// The runtime's turn ids are bare v4 UUIDs, so this returns `Some` when the
/// caller passes a session id directly.
pub fn session_uuid_of(turn_id: &str) -> Option<uuid::Uuid> {
    uuid::Uuid::parse_str(turn_id).ok()
}

/// Parse a session UUID from a turn id (alias for [`session_uuid_of`]).
fn parse_session_id(turn_id: &str) -> Option<uuid::Uuid> {
    session_uuid_of(turn_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, ToolCall, ToolResult};
    use serde_json::json;

    #[test]
    fn test_decide_no_compaction_under_threshold() {
        let stage = CompactionStage::new(50);
        let msgs = vec![Message::user("hello")];
        assert!(matches!(
            stage.decide(&msgs),
            CompactionDecision::NoCompaction
        ));
    }

    #[test]
    fn test_apply_summarize_keeps_system() {
        let stage = CompactionStage::new(50);
        let mut msgs = vec![Message::system("instructions")];
        for i in 0..100 {
            msgs.push(Message::user(format!("message {i}")));
        }
        let out = stage.apply(&msgs, CompactionStrategy::Summarize);
        assert_eq!(out[0].role, MessageRole::System);
        assert!(out.len() < msgs.len());
    }

    #[test]
    fn test_apply_truncate_keeps_recent() {
        let stage = CompactionStage::new(10);
        let mut msgs = Vec::new();
        for i in 0..30 {
            msgs.push(Message::user(format!("m{i}")));
        }
        let out = stage.apply(&msgs, CompactionStrategy::Truncate);
        // Keeps the tail budget (max_messages - 1 for non-system).
        assert_eq!(out.len(), 9);
        assert_eq!(out[0].text_content(), "m21");
    }

    #[test]
    fn test_enforce_context_window_truncates() {
        let stage = CompactionStage::new(1000).with_context_window(100);
        let mut msgs = Vec::new();
        for i in 0..100 {
            msgs.push(Message::user(format!("a fairly long message body {i}")));
        }
        let out = stage.enforce_context_window(msgs);
        assert!(stage.estimate_tokens(&out) <= 100);
    }

    #[test]
    fn test_repair_tool_pairs_after_surgery() {
        let stage = CompactionStage::new(10);
        let mut msgs = vec![Message::user("run")];
        msgs.push(Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall::new(
                "c1",
                "shell",
                json!({"cmd": "ls"}),
            ))],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        });
        // An orphaned result whose tool_use was dropped by truncation.
        msgs.push(Message {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult::success("gone", "ok"))],
            name: Some("shell".into()),
            tool_call_id: Some("gone".into()),
            tool_calls: None,
            tool_result: None,
        });
        let out = stage.apply(&msgs, CompactionStrategy::Truncate);
        let repair = crate::history::repair_tool_pairs(&out);
        assert_eq!(repair.removed_results, 1);
        assert_eq!(repair.unpaired_calls, 0);
    }

    #[tokio::test]
    async fn test_compact_messages_reclaims() {
        let stage = CompactionStage::new(10);
        let mut msgs = vec![Message::system("instructions")];
        for i in 0..30 {
            msgs.push(Message::user(format!("m{i}")));
        }
        let (out, outcome) = stage
            .compact_messages("t1", msgs, CompactionStrategy::Truncate)
            .await;
        assert!(out.len() < 30);
        assert!(outcome.reclaimed() > 0);
        assert!(stage.last_outcome().is_some());
    }
}
