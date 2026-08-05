//! Pre-turn pipeline steps.
//!
//! Mirrors the Python backend's `engine/steps/` package. Each module owns one
//! pipeline step that transforms the [`PipelineContext`] before the turn
//! stages run. Steps are executed in order by a [`StepChain`] with fail-open
//! semantics: a step that returns `Skip` or `Continue` lets the chain proceed,
//! and a step that returns `Halt` stops the entire pipeline.
//!
//! The five step implementations are:
//!
//! * [`meta_resolution::MetaResolutionStep`] — resolve meta-instruction
//!   triggers in the user message and leave a soft hint for the skills filter.
//! * [`model_select::ModelSelectStep`] — resolve the model (and provider) used
//!   for the turn.
//! * [`skills_filter::SkillsFilterStep`] — gate and filter available skills,
//!   then inject `<available_skills>` into the system prompt.
//! * [`context_assembly::ContextAssemblyStep`] — assemble context fragments
//!   from workspace instruction files (SOUL.md, AGENTS.md, ...).
//! * [`attachment_loader::AttachmentLoaderStep`] — load and validate turn
//!   attachments into the context.

pub mod attachment_loader;
pub mod context_assembly;
pub mod meta_resolution;
pub mod model_select;
pub mod skills_filter;

use async_trait::async_trait;
use opensquilla_core::error::Result;
use std::fmt;
use tracing::instrument;

pub use crate::pipeline::{PipelineContext, StepAction};

/// A trait for steps in the pre-turn pipeline.
///
/// Pipeline steps are executed before the main turn stages. They can
/// transform messages, validate inputs, inject system prompts, or halt the
/// pipeline entirely. Unlike `crate::pipeline::PipelineStep`, this trait
/// carries a `name()` accessor so logs and decision records can attribute
/// effects to a specific step.
#[async_trait]
pub trait PipelineStep: Send + Sync + fmt::Debug {
    /// Execute this step, returning an action indicating what to do next.
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction>;

    /// The human-readable name of this step, for logging and error reporting.
    fn name(&self) -> &str;
}

/// An ordered chain of pipeline steps with fail-open semantics.
///
/// Steps execute in insertion order. `Continue` and `Skip` both proceed to the
/// next step (the distinction is purely informational); `Halt(reason)`
/// short-circuits the chain and is returned to the caller.
#[derive(Debug, Default)]
pub struct StepChain {
    /// The ordered list of steps in this chain.
    steps: Vec<Box<dyn PipelineStep + Send + Sync>>,
}

impl StepChain {
    /// Create a new empty step chain.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a step to the end of the chain.
    pub fn add_step(&mut self, step: Box<dyn PipelineStep + Send + Sync>) {
        self.steps.push(step);
    }

    /// Convenience: append a concrete step to the end of the chain.
    pub fn push<S>(&mut self, step: S)
    where
        S: PipelineStep + Send + Sync + 'static,
    {
        self.steps.push(Box::new(step));
    }

    /// The number of steps in the chain.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the chain contains no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Execute all steps in the chain, in order.
    ///
    /// Returns the first `Halt` action produced by any step, or
    /// `StepAction::Continue` if every step completed normally.
    #[instrument(skip(self), fields(step_count = self.steps.len()))]
    pub async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        for step in &self.steps {
            match step.execute(ctx).await? {
                StepAction::Continue | StepAction::Skip => continue,
                halt @ StepAction::Halt(_) => return Ok(halt),
            }
        }
        Ok(StepAction::Continue)
    }
}

/// A convenience wrapper that adapts a plain async closure into a
/// [`PipelineStep`], for test fixtures and ad-hoc pipelines.
pub struct ClosureStep<F> {
    name: String,
    f: F,
}

impl<F> ClosureStep<F> {
    /// Create a closure-backed step with the given name.
    pub fn new(name: impl Into<String>, f: F) -> Self {
        Self {
            name: name.into(),
            f,
        }
    }
}

impl<F> fmt::Debug for ClosureStep<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosureStep")
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl<F> PipelineStep for ClosureStep<F>
where
    F: for<'a> Fn(
            &'a mut PipelineContext,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<StepAction>> + Send + 'a>,
        >
        + Send
        + Sync,
{
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        (self.f)(ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Re-export the concrete step types at the module root.
pub use attachment_loader::{AttachmentDescriptor, AttachmentLoaderStep};
pub use context_assembly::ContextAssemblyStep;
pub use meta_resolution::{MetaResolutionConfig, MetaResolutionStep};
pub use model_select::{ModelSelectConfig, ModelSelectStep};
pub use skills_filter::{SkillSpec, SkillsFilterConfig, SkillsFilterStep};

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::error::Result;
    use opensquilla_core::types::Message;
    use std::pin::Pin;

    type BoxedStepFuture<'a> = Pin<Box<dyn std::future::Future<Output = Result<StepAction>> + Send + 'a>>;

    #[tokio::test]
    async fn test_chain_runs_in_order() {
        let mut chain = StepChain::new();
        chain.push(ClosureStep::new("first", |ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
            Box::pin(async move {
                ctx.set_metadata("order", "first");
                Ok(StepAction::Continue)
            })
        }));
        chain.push(ClosureStep::new("second", |ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
            Box::pin(async move {
                ctx.set_metadata("order", "second");
                Ok(StepAction::Continue)
            })
        }));

        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let action = chain.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert_eq!(ctx.get_metadata("order").map(|s| s.as_str()), Some("second"));
    }

    #[tokio::test]
    async fn test_halt_short_circuits() {
        let mut chain = StepChain::new();
        chain.push(ClosureStep::new("halt", |_ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
            Box::pin(async move { Ok(StepAction::Halt("stop".into())) })
        }));
        chain.push(ClosureStep::new("never", |ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
            Box::pin(async move {
                ctx.set_metadata("ran", "true");
                Ok(StepAction::Continue)
            })
        }));

        let mut ctx = PipelineContext::new("t1".into(), Vec::new());
        let action = chain.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Halt(ref r) if r == "stop"));
        assert!(ctx.get_metadata("ran").is_none());
    }
}
