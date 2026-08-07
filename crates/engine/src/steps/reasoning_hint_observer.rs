//! Reasoning-hint observation step.
//!
//! Mirrors the Python `engine/steps/reasoning_hint_observer.py` step. It runs
//! right after model resolution and records nullable reasoning-hint telemetry
//! (`reasoning_hint_resolved`) without changing the prompt. The hint is a pure
//! function of the resolved model id ([`crate::reasoning_hint::reasoning_tag_hint`]),
//! so the step never alters messages or halts the pipeline.

use crate::pipeline::PipelineContext;
use crate::reasoning_hint::reasoning_tag_hint;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use tracing::instrument;

/// Pre-turn pipeline step that records reasoning-hint telemetry.
///
/// The resolved model id is read from the `resolved_model` metadata written by
/// [`crate::steps::ModelSelectStep`] (falling back to the raw `model` override).
#[derive(Debug, Default)]
pub struct ReasoningHintObserverStep;

impl ReasoningHintObserverStep {
    /// Create a new reasoning-hint observer step.
    pub fn new() -> Self {
        Self
    }

    /// Observe the resolved model and record the reasoning hint, if any.
    pub fn observe(&self, ctx: &mut PipelineContext) {
        let model = ctx
            .get_metadata("resolved_model")
            .cloned()
            .or_else(|| ctx.get_metadata("model").cloned())
            .unwrap_or_default();
        if let Some(hint) = reasoning_tag_hint(&model) {
            ctx.set_metadata("reasoning_hint_resolved", hint);
        }
    }
}

#[async_trait]
impl PipelineStep for ReasoningHintObserverStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        self.observe(ctx);
        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "reasoning_hint_observer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    fn ctx() -> PipelineContext {
        PipelineContext::new("t1".into(), vec![Message::user("hello")])
    }

    #[test]
    fn test_records_hint_for_reasoning_family() {
        let step = ReasoningHintObserverStep::new();
        let mut pipeline = ctx();
        pipeline.set_metadata("resolved_model", "deepseek-r1-0528");
        step.observe(&mut pipeline);
        assert!(pipeline.get_metadata("reasoning_hint_resolved").is_some());
        assert!(
            pipeline
                .get_metadata("reasoning_hint_resolved")
                .map(|h| h.contains("<think>"))
                .unwrap_or(false)
        );
    }

    #[test]
    fn test_no_hint_for_regular_model() {
        let step = ReasoningHintObserverStep::new();
        let mut pipeline = ctx();
        pipeline.set_metadata("resolved_model", "gpt-4o");
        step.observe(&mut pipeline);
        assert!(pipeline.get_metadata("reasoning_hint_resolved").is_none());
    }

    #[test]
    fn test_falls_back_to_model_override() {
        let step = ReasoningHintObserverStep::new();
        let mut pipeline = ctx();
        pipeline.set_metadata("model", "openai/gpt-5");
        step.observe(&mut pipeline);
        assert!(pipeline.get_metadata("reasoning_hint_resolved").is_some());
    }

    #[tokio::test]
    async fn test_execute_is_continue() {
        let step = ReasoningHintObserverStep::new();
        let mut pipeline = ctx();
        pipeline.set_metadata("resolved_model", "codex-1");
        let action = step.execute(&mut pipeline).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert!(pipeline.get_metadata("reasoning_hint_resolved").is_some());
    }
}
