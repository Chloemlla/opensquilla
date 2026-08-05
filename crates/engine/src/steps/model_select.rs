//! Model selection step.
//!
//! Mirrors the Python `engine/steps/resolve_model.py` step. It resolves the
//! model (and provider) used for the turn from, in priority order:
//!
//! 1. an explicit per-turn override in pipeline metadata (`model`,
//!    `provider_name`),
//! 2. a routing decision produced by the routing policy engine (stored as
//!    `routing_decision` metadata),
//! 3. the configured default model / provider.
//!
//! The resolved values are written back into `resolved_model` and
//! `provider_name` metadata so downstream stages (prompt assembly, provider
//! resolution) consume a single, pinned decision.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// Configuration for the model selection step.
#[derive(Debug, Clone)]
pub struct ModelSelectConfig {
    /// The model used when no override or routing decision is present.
    pub default_model: String,
    /// The provider serving the default model.
    pub default_provider: String,
    /// Whether to honor a routing decision found in pipeline metadata.
    pub honor_routing: bool,
}

impl Default for ModelSelectConfig {
    fn default() -> Self {
        Self {
            default_model: String::new(),
            default_provider: String::new(),
            honor_routing: true,
        }
    }
}

/// A serializable routing decision carried through pipeline metadata.
///
/// This is a lightweight projection of `crate::routing::RoutingDecision`,
/// kept here so the pipeline step does not force a dependency on the routing
/// module's JSON value machinery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingMetadata {
    /// The routed text tier (e.g. `c2`).
    pub tier: String,
    /// The model bound to that tier.
    pub model: String,
    /// The classifier confidence for the decision.
    pub confidence: f64,
    /// The decision source (e.g. `v4_phase3`, `heuristic`, `default`).
    pub source: String,
}

/// The outcome of model resolution, recorded in metadata.
#[derive(Debug, Clone)]
pub struct ModelSelectOutcome {
    /// The resolved model id.
    pub model: String,
    /// The provider serving the model.
    pub provider: String,
    /// The routing tier that produced the model, if routing was honored.
    pub routed_tier: Option<String>,
    /// The source of the resolution (`override`, `routing`, or `default`).
    pub source: String,
}

/// Pre-turn pipeline step that resolves the turn's model.
#[derive(Debug)]
pub struct ModelSelectStep {
    config: ModelSelectConfig,
}

impl ModelSelectStep {
    /// Create a new step with the given default model and provider.
    pub fn new(default_model: impl Into<String>, default_provider: impl Into<String>) -> Self {
        Self {
            config: ModelSelectConfig {
                default_model: default_model.into(),
                default_provider: default_provider.into(),
                ..Default::default()
            },
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: ModelSelectConfig) -> Self {
        Self { config }
    }
}

impl ModelSelectStep {
    /// Resolve the model and provider for the turn from the pipeline context.
    ///
    /// Priority: explicit metadata override, then routing decision, then the
    /// configured default. The result is written into `resolved_model`,
    /// `provider_name`, `routed_tier`, and `model_source` metadata.
    pub fn resolve(&self, ctx: &mut PipelineContext) -> ModelSelectOutcome {
        // 1. Explicit per-turn override.
        if let Some(model) = ctx.get_metadata("model") {
            let provider = ctx
                .get_metadata("provider_name")
                .cloned()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| self.config.default_provider.clone());
            debug!(model = %model, provider = %provider, "model selected via override");
            ctx.set_metadata("resolved_model", model);
            ctx.set_metadata("provider_name", &provider);
            ctx.set_metadata("model_source", "override");
            return ModelSelectOutcome {
                model: model.clone(),
                provider,
                routed_tier: None,
                source: "override".to_string(),
            };
        }

        // 2. Routing decision (e.g. produced by the squilla-router step).
        if self.config.honor_routing {
            if let Some(raw) = ctx.get_metadata("routing_decision") {
                if let Ok(decision) = serde_json::from_str::<RoutingMetadata>(raw) {
                    let provider = ctx
                        .get_metadata("provider_name")
                        .cloned()
                        .filter(|p| !p.is_empty())
                        .unwrap_or_else(|| self.config.default_provider.clone());
                    debug!(
                        tier = %decision.tier,
                        model = %decision.model,
                        source = %decision.source,
                        "model selected via routing"
                    );
                    ctx.set_metadata("resolved_model", &decision.model);
                    ctx.set_metadata("provider_name", &provider);
                    ctx.set_metadata("routed_tier", &decision.tier);
                    ctx.set_metadata("model_source", &decision.source);
                    return ModelSelectOutcome {
                        model: decision.model,
                        provider,
                        routed_tier: Some(decision.tier),
                        source: decision.source,
                    };
                }
            }
        }

        // 3. Configured default.
        debug!(
            model = %self.config.default_model,
            provider = %self.config.default_provider,
            "model selected via default"
        );
        ctx.set_metadata("resolved_model", &self.config.default_model);
        ctx.set_metadata("provider_name", &self.config.default_provider);
        ctx.set_metadata("model_source", "default");
        ModelSelectOutcome {
            model: self.config.default_model.clone(),
            provider: self.config.default_provider.clone(),
            routed_tier: None,
            source: "default".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStep for ModelSelectStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        self.resolve(ctx);
        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "model_select"
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
    fn test_default_selection() {
        let step = ModelSelectStep::new("gpt-4o", "openai");
        let mut pipeline = ctx();
        let outcome = step.resolve(&mut pipeline);
        assert_eq!(outcome.model, "gpt-4o");
        assert_eq!(outcome.source, "default");
        assert_eq!(pipeline.get_metadata("resolved_model").map(String::as_str), Some("gpt-4o"));
    }

    #[test]
    fn test_override_selection() {
        let step = ModelSelectStep::new("gpt-4o", "openai");
        let mut pipeline = ctx();
        pipeline.set_metadata("model", "claude-sonnet");
        let outcome = step.resolve(&mut pipeline);
        assert_eq!(outcome.model, "claude-sonnet");
        assert_eq!(outcome.source, "override");
    }

    #[test]
    fn test_routing_selection() {
        let step = ModelSelectStep::new("gpt-4o", "openai");
        let mut pipeline = ctx();
        let decision = RoutingMetadata {
            tier: "c2".into(),
            model: "deepseek-chat".into(),
            confidence: 0.6,
            source: "heuristic".into(),
        };
        pipeline.set_metadata("routing_decision", serde_json::to_string(&decision).unwrap());
        let outcome = step.resolve(&mut pipeline);
        assert_eq!(outcome.model, "deepseek-chat");
        assert_eq!(outcome.routed_tier.as_deref(), Some("c2"));
        assert_eq!(pipeline.get_metadata("routed_tier").map(String::as_str), Some("c2"));
    }
}
