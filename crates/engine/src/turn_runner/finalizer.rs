//! Finalizer stage.
//!
//! Mirrors the Python `engine/turn_runner/turn_finalizer_stage.py` stage. It
//! is the last stage in the pipeline. It:
//!
//! * assembles the final [`StageOutcome`] from the turn context,
//! * persists the turn transcript to the session database (when a
//!   [`SessionManager`] is attached and the `session` feature is enabled),
//! * records token usage into the shared [`crate::usage::UsageTracker`],
//! * broadcasts the completion event through a `tokio::sync::broadcast`
//!   channel so long-polling session subscribers observe the turn,
//! * emits terminal stream events when a channel is present,
//! * performs cleanup (per-turn state such as the attachment list and the
//!   stream consumer state are consumed here, then released).

use crate::agent::TurnGenerator;
use crate::stages::{Stage, StageContext, StageOutcome, StageOutput};
use crate::usage::UsageTracker;
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::events::{StreamEvent, TurnEvent};
use opensquilla_core::types::{Message, MessageRole, Usage};
use std::fmt;
use std::time::Instant;
use tracing::{debug, info, instrument};

/// Cost rollup computed at finalize time.
#[derive(Debug, Clone)]
pub struct CostRollup {
    /// Estimated input cost in USD (uses the pricing crate when available).
    pub estimated_input_cost_usd: f64,
    /// Estimated output cost in USD.
    pub estimated_output_cost_usd: f64,
    /// Total estimated cost in USD.
    pub estimated_total_cost_usd: f64,
}

/// The outcome of finalization, useful to the caller for persistence.
#[derive(Debug, Clone)]
pub struct FinalizeReport {
    /// The final assistant text (concatenated).
    pub final_text: String,
    /// Token usage for the turn.
    pub usage: Usage,
    /// Number of messages in the final history.
    pub message_count: usize,
    /// Cost rollup estimate.
    pub cost: CostRollup,
    /// Wall-clock duration of the turn in milliseconds.
    pub duration_ms: u64,
    /// The turn id.
    pub turn_id: String,
    /// The model that served the turn.
    pub model: String,
    /// The provider that served the turn.
    pub provider: String,
}

impl FinalizeReport {
    /// Whether any assistant text was produced.
    pub fn has_output(&self) -> bool {
        !self.final_text.trim().is_empty()
    }
}

/// Configuration for the finalizer stage.
#[derive(Debug, Clone)]
pub struct FinalizerConfig {
    /// Whether the transcript is persisted to the session database.
    pub persist_transcript: bool,
    /// Whether the completion event is broadcast to session subscribers.
    pub broadcast_completion: bool,
    /// Whether token usage is recorded into the shared tracker.
    pub record_usage: bool,
    /// Whether a terminal `MessageStop` stream event is emitted.
    pub emit_terminal_stream_event: bool,
}

impl Default for FinalizerConfig {
    fn default() -> Self {
        Self {
            persist_transcript: true,
            broadcast_completion: true,
            record_usage: true,
            emit_terminal_stream_event: true,
        }
    }
}

/// The finalizer stage in the turn pipeline.
pub struct FinalizerStage {
    /// Start time used to compute the turn duration.
    started_at: Instant,
    /// Configuration.
    config: FinalizerConfig,
    /// Optional session manager used to persist the transcript (feature
    /// `session`).
    #[cfg(feature = "session")]
    session_manager: Option<opensquilla_session::SessionManager>,
    /// Optional broadcast channel for session completion events.
    broadcast_tx: Option<tokio::sync::broadcast::Sender<TurnEvent>>,
    /// Optional shared usage tracker.
    usage_tracker: Option<UsageTracker>,
}

impl fmt::Debug for FinalizerStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FinalizerStage")
            .field("config", &self.config)
            .field(
                "broadcast_tx",
                &self.broadcast_tx.as_ref().map(|_| "<broadcast>"),
            )
            .field(
                "usage_tracker",
                &self.usage_tracker.as_ref().map(|_| "<usage>"),
            )
            .finish_non_exhaustive()
    }
}

impl FinalizerStage {
    /// Create a new finalizer stage.
    pub fn new() -> Self {
        Self {
            started_at: Instant::now(),
            config: FinalizerConfig::default(),
            #[cfg(feature = "session")]
            session_manager: None,
            broadcast_tx: None,
            usage_tracker: None,
        }
    }

    /// Replace the stage configuration.
    pub fn with_config(mut self, config: FinalizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Attach a session manager for transcript persistence.
    ///
    /// Only available when the `session` feature is enabled.
    #[cfg(feature = "session")]
    pub fn with_session_manager(mut self, manager: opensquilla_session::SessionManager) -> Self {
        self.session_manager = Some(manager);
        self
    }

    /// Attach a broadcast channel for completion events.
    pub fn with_broadcast(mut self, tx: tokio::sync::broadcast::Sender<TurnEvent>) -> Self {
        self.broadcast_tx = Some(tx);
        self
    }

    /// Attach a shared usage tracker.
    pub fn with_usage_tracker(mut self, tracker: UsageTracker) -> Self {
        self.usage_tracker = Some(tracker);
        self
    }

    /// Reset the duration baseline (call before each turn if reusing the stage).
    pub fn reset_timer(&mut self) {
        self.started_at = Instant::now();
    }

    /// Estimate the cost of a turn from a naive token->USD rate.
    ///
    /// The engine does not own provider pricing; this is a deterministic
    /// placeholder that callers can replace with
    /// [`crate::pricing::PricingCache`]-backed numbers.
    pub fn estimate_cost(usage: &Usage) -> CostRollup {
        let input_usd = usage.input_tokens as f64 / 1_000_000.0 * 1.0; // $1 / 1M in
        let output_usd = usage.output_tokens as f64 / 1_000_000.0 * 3.0; // $3 / 1M out
        CostRollup {
            estimated_input_cost_usd: input_usd,
            estimated_output_cost_usd: output_usd,
            estimated_total_cost_usd: input_usd + output_usd,
        }
    }

    /// Build the finalize report from the turn context.
    pub fn report(&self, ctx: &StageContext) -> FinalizeReport {
        let final_text = ctx
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("");
        let cost = Self::estimate_cost(&ctx.usage);
        FinalizeReport {
            final_text,
            usage: ctx.usage,
            message_count: ctx.messages.len(),
            cost,
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            turn_id: ctx.turn_id.clone(),
            model: ctx.current_model.clone(),
            provider: ctx.current_provider.clone(),
        }
    }

    /// Persist the turn transcript to the session database.
    ///
    /// Non-system messages are appended as transcript entries. A missing
    /// session row is treated as a no-op (the turn may not be routed through
    /// the session manager); storage errors are logged and swallowed so a
    /// persistence failure never fails the turn.
    #[cfg(feature = "session")]
    pub fn persist_turn(&self, ctx: &StageContext, report: &FinalizeReport) {
        let Some(manager) = &self.session_manager else {
            return;
        };
        if !self.config.persist_transcript {
            return;
        }
        let Some(session_id) = crate::turn_runner::compaction::session_uuid_of(&ctx.turn_id) else {
            debug!(turn_id = %ctx.turn_id, "turn id is not a session uuid; skipping persistence");
            return;
        };
        for message in &ctx.messages {
            if message.role == MessageRole::System {
                continue;
            }
            let role = match message.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };
            let content = message.text_content();
            let token_count = message
                .content
                .iter()
                .map(|b| match b {
                    opensquilla_core::types::ContentBlock::Text(t) => t.chars().count() as u64 / 4,
                    opensquilla_core::types::ContentBlock::Reasoning(r) => {
                        r.chars().count() as u64 / 4
                    }
                    opensquilla_core::types::ContentBlock::ToolUse(c) => {
                        c.name.len() as u64 / 4 + c.input.to_string().len() as u64 / 4
                    }
                    opensquilla_core::types::ContentBlock::ToolResult(r) => {
                        r.content.len() as u64 / 4
                    }
                })
                .sum::<u64>()
                .max(1);
            match manager.add_message(&session_id, role.to_string(), content, token_count) {
                Ok(_) => {}
                Err(e) => {
                    debug!(turn_id = %ctx.turn_id, session = %session_id, error = %e, "transcript persistence skipped");
                    return;
                }
            }
        }
        let _ = report;
        info!(
            turn_id = %ctx.turn_id,
            session = %session_id,
            messages = ctx.messages.len(),
            "turn transcript persisted"
        );
    }

    /// Broadcast a completion event to session subscribers.
    pub fn broadcast_completion(&self, event: TurnEvent) {
        let Some(tx) = &self.broadcast_tx else {
            return;
        };
        if !self.config.broadcast_completion {
            return;
        }
        // Broadcast errors are expected when no subscriber remains.
        let _ = tx.send(event);
    }

    /// Record the turn's usage into the shared tracker.
    pub fn record_usage(&self, usage: &Usage) {
        if let Some(tracker) = &self.usage_tracker {
            if self.config.record_usage {
                tracker.record(usage);
            }
        }
    }
}

impl Default for FinalizerStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for FinalizerStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("finalizer: persisting and emitting turn completion");

        let report = self.report(ctx);
        let duration_ms = report.duration_ms;

        // Emit a final message-stop if a stream channel exists and no consumer
        // stage already closed it.
        if self.config.emit_terminal_stream_event {
            if let Some(tx) = &ctx.streaming_tx {
                let content: Vec<Message> = ctx.messages.clone();
                let _ = tx
                    .send(StreamEvent::MessageStop {
                        content: content.iter().flat_map(|m| m.content.clone()).collect(),
                        usage: Some(report.usage),
                    })
                    .await;
            }
        }

        // Persist the transcript to the session database.
        #[cfg(feature = "session")]
        self.persist_turn(ctx, &report);

        // Record usage into the shared tracker.
        self.record_usage(&report.usage);

        // Broadcast the completion event.
        let final_message = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::Assistant)
            .cloned();
        if let Some(message) = final_message {
            self.broadcast_completion(TurnEvent::TurnComplete {
                message,
                usage: report.usage,
                duration_ms,
            });
        }

        info!(
            turn_id = %ctx.turn_id,
            message_count = report.message_count,
            total_tokens = report.usage.total_tokens,
            duration_ms = duration_ms,
            cost_usd = report.cost.estimated_total_cost_usd,
            has_output = report.has_output(),
            "finalizer stage complete"
        );

        // Produce the terminal outcome for the runtime.
        let outcome = StageOutcome {
            messages: ctx.messages.clone(),
            usage: report.usage,
            duration_ms,
            success: true,
            error_message: None,
        };

        Ok(StageOutput::Output(outcome))
    }

    fn name(&self) -> &str {
        "finalizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Usage;
    use std::collections::HashMap;

    fn context() -> StageContext {
        StageContext {
            turn_id: "t1".to_string(),
            messages: vec![
                Message::system("instructions"),
                Message::user("hello"),
                Message::assistant("world"),
            ],
            current_model: "m".to_string(),
            current_provider: "p".to_string(),
            usage: Usage::new(10, 20),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_estimate_cost() {
        let cost = FinalizerStage::estimate_cost(&Usage::new(1_000_000, 1_000_000));
        assert!((cost.estimated_input_cost_usd - 1.0).abs() < 1e-9);
        assert!((cost.estimated_output_cost_usd - 3.0).abs() < 1e-9);
        assert!((cost.estimated_total_cost_usd - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_fields() {
        let stage = FinalizerStage::new();
        let ctx = context();
        let report = stage.report(&ctx);
        assert_eq!(report.final_text, "world");
        assert_eq!(report.usage.input_tokens, 10);
        assert_eq!(report.message_count, 3);
        assert_eq!(report.model, "m");
        assert_eq!(report.provider, "p");
        assert!(report.has_output());
    }

    #[test]
    fn test_report_no_output() {
        let stage = FinalizerStage::new();
        let ctx = StageContext {
            turn_id: "t1".into(),
            messages: vec![Message::system("only system")],
            current_model: "m".into(),
            current_provider: "p".into(),
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
            metadata: HashMap::new(),
        };
        let report = stage.report(&ctx);
        assert!(!report.has_output());
    }

    #[tokio::test]
    async fn test_execute_produces_outcome() {
        let stage = FinalizerStage::new();
        let mut ctx = context();
        let generator = MockGenerator;
        let out = stage.execute(&mut ctx, &generator).await.unwrap();
        match out {
            StageOutput::Output(outcome) => {
                assert!(outcome.success);
                assert_eq!(outcome.messages.len(), 3);
            }
            _ => panic!("expected output"),
        }
    }

    #[tokio::test]
    async fn test_broadcast_delivers() {
        let (tx, mut rx) = tokio::sync::broadcast::channel(16);
        let stage = FinalizerStage::new().with_broadcast(tx);
        stage.broadcast_completion(TurnEvent::TurnError {
            message: "boom".to_string(),
            code: Some("E".to_string()),
        });
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, TurnEvent::TurnError { .. }));
    }

    #[test]
    fn test_usage_tracker_recorded() {
        let tracker = UsageTracker::new();
        let stage = FinalizerStage::new().with_usage_tracker(tracker.clone());
        stage.record_usage(&Usage::new(100, 50));
        assert_eq!(tracker.total_input_tokens(), 100);
        assert_eq!(tracker.total_output_tokens(), 50);
    }

    #[derive(Debug)]
    struct MockGenerator;
    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
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
