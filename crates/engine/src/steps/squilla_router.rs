//! Squilla router step.
//!
//! Mirrors the Python `engine/steps/squilla_router.py` step. It classifies
//! message complexity and routes to an appropriate model tier, running
//! post-classifier heuristics (confidence gate, anti-downgrade, budget gate)
//! on top of the routing output.
//!
//! The Python step is ~1490 lines with an ML classifier strategy, a policy
//! engine, provider-mismatch veto, budget gate, and routing history. The Rust
//! port implements the deterministic routing paths (image routing, default
//! tier, router-control hold) and the metadata bookkeeping. The ML classifier
//! strategy and policy engine are left as `TODO(parity)` because the Rust
//! `PipelineContext` has no config, provider, or strategy handle.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use tracing::{debug, instrument};

/// The default text tier when no routing config is available.
pub const DEFAULT_TEXT_TIER: &str = "c0";

/// The canonical text-tier ladder, ordered low-to-high.
pub const TEXT_TIERS: &[&str] = &["c0", "c1", "c2", "c3"];

/// A serializable routing decision, matching the Python `RoutingDecision`.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The routed text tier (e.g. `c2`).
    pub tier: String,
    /// The model bound to that tier.
    pub model: String,
    /// The classifier confidence for the decision.
    pub confidence: f64,
    /// The decision source (e.g. `image_route`, `default`, `heuristic`).
    pub source: String,
}

/// A tier definition, mirroring the Python `tiers` dict entries.
#[derive(Debug, Clone, Default)]
pub struct TierConfig {
    /// The model id for this tier.
    pub model: String,
    /// The provider serving this tier (optional).
    pub provider: String,
    /// Whether this tier supports image inputs.
    pub supports_image: bool,
    /// Whether this tier is image-only (excluded from text routing).
    pub image_only: bool,
    /// Whether this tier supports thinking/reasoning.
    pub supports_thinking: bool,
}

/// Configuration for the squilla router step.
#[derive(Debug, Clone)]
pub struct SquillaRouterConfig {
    /// Master switch. When false the step is a complete no-op.
    pub enabled: bool,
    /// The rollout phase: `"observe"`, `"prompt_only"`, or `"full"`.
    pub rollout_phase: String,
    /// The default tier when the classifier cannot decide.
    pub default_tier: String,
    /// The configured text tiers.
    pub tiers: Vec<(String, TierConfig)>,
    /// Whether auto-thinking is enabled.
    pub auto_thinking: bool,
}

impl Default for SquillaRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            rollout_phase: "observe".to_string(),
            default_tier: DEFAULT_TEXT_TIER.to_string(),
            tiers: Vec::new(),
            auto_thinking: true,
        }
    }
}

/// Pre-turn pipeline step that routes to an appropriate model tier.
#[derive(Debug)]
pub struct SquillaRouterStep {
    config: SquillaRouterConfig,
}

impl SquillaRouterStep {
    /// Create a new step with default configuration (disabled).
    pub fn new() -> Self {
        Self {
            config: SquillaRouterConfig::default(),
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: SquillaRouterConfig) -> Self {
        Self { config }
    }

    /// Return the canonical index of a tier name in [`TEXT_TIERS`], or `None`
    /// for unknown/custom tier names.
    fn tier_index(name: &str) -> Option<usize> {
        TEXT_TIERS.iter().position(|t| *t == name)
    }

    /// Return the valid (non-image-only) tier names in canonical ladder order.
    fn valid_text_tiers(&self) -> Vec<String> {
        let mut tiers: Vec<String> = self
            .config
            .tiers
            .iter()
            .filter(|(_, cfg)| !cfg.image_only)
            .map(|(name, _)| name.clone())
            .collect();
        // Sort by canonical ladder position; unknown/custom tiers sort after.
        tiers.sort_by_key(|name| {
            Self::tier_index(name)
                .map(|i| (0u8, i))
                .unwrap_or((1u8, 0))
        });
        tiers
    }

    /// Look up a tier config by name.
    fn tier_cfg(&self, name: &str) -> Option<&TierConfig> {
        self.config
            .tiers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, cfg)| cfg)
    }

    /// Check whether the current turn has an image attachment.
    ///
    /// TODO(parity): the Python step checks `ctx.attachments` for image media
    /// types. The Rust PipelineContext has no attachments field; we check the
    /// `current_turn_has_image` metadata flag instead.
    fn current_turn_has_image(&self, ctx: &PipelineContext) -> bool {
        ctx.get_metadata("current_turn_has_image")
            .map(|v| v == "true")
            .unwrap_or(false)
    }

    /// Build a fallback chain of lower tiers for the selected tier.
    fn text_fallback_chain(&self, selected_tier: &str) -> Vec<String> {
        let selected_index = match Self::tier_index(selected_tier) {
            Some(i) => i,
            None => return Vec::new(),
        };
        let mut chain = Vec::new();
        for tier_name in TEXT_TIERS[..selected_index].iter().rev() {
            let Some(cfg) = self.tier_cfg(tier_name) else {
                continue;
            };
            if cfg.image_only || cfg.model.is_empty() {
                continue;
            }
            chain.push(cfg.model.clone());
        }
        chain
    }

    /// Record routing metadata into the context.
    fn record_routing_metadata(
        &self,
        ctx: &mut PipelineContext,
        decision: &RoutingDecision,
        routing_applied: bool,
    ) {
        ctx.set_metadata("routed_tier", &decision.tier);
        ctx.set_metadata("routed_model", &decision.model);
        ctx.set_metadata("routing_applied", if routing_applied { "true" } else { "false" });
        ctx.set_metadata("rollout_phase", &self.config.rollout_phase);
        ctx.set_metadata("applied_model", &decision.model);
        ctx.set_metadata("routing_confidence", &decision.confidence.to_string());
        ctx.set_metadata("routing_source", &decision.source);
        let chain = self.text_fallback_chain(&decision.tier);
        ctx.set_metadata("router_fallback_chain", chain.join(","));
    }

    /// Record thinking metadata for the routed tier.
    fn record_thinking_metadata(&self, ctx: &mut PipelineContext, tier_name: &str) {
        if !self.config.auto_thinking {
            return;
        }
        if let Some(cfg) = self.tier_cfg(tier_name) {
            if cfg.supports_thinking {
                ctx.set_metadata("thinking_requested", "true");
                ctx.set_metadata("thinking_level", "medium");
            }
        }
    }
}

impl Default for SquillaRouterStep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStep for SquillaRouterStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if !self.config.enabled {
            debug!("squilla_router disabled, skipping");
            return Ok(StepAction::Continue);
        }

        if self.config.tiers.is_empty() {
            debug!("squilla_router: no tiers configured, skipping");
            return Ok(StepAction::Continue);
        }

        // TODO(parity): the Python step skips subagent sessions
        // (`":subagent:" in ctx.session_key`). The Rust PipelineContext has
        // no session_key; we check the `is_subagent` metadata flag instead.
        if ctx.get_metadata("is_subagent").map(|v| v == "true").unwrap_or(false) {
            debug!("squilla_router: skipping subagent session");
            return Ok(StepAction::Continue);
        }

        let current_turn_has_image = self.current_turn_has_image(ctx);
        let history_gate_needs_image = ctx
            .get_metadata("router_vision_followup_needs_image")
            .map(|v| v == "true")
            .unwrap_or(false);
        let turn_needs_image = current_turn_has_image || history_gate_needs_image;

        // Image-aware routing: pick directly from supports_image tiers.
        if turn_needs_image {
            let image_tiers: Vec<&(String, TierConfig)> = self
                .config
                .tiers
                .iter()
                .filter(|(_, cfg)| cfg.supports_image)
                .collect();

            if image_tiers.is_empty() {
                // TODO(parity): the Python step raises a RuntimeError here.
                // The Rust step records the error and continues, since
                // halting the pipeline on a missing image tier would block
                // the turn entirely.
                debug!("squilla_router: image detected but no supports_image tier");
                ctx.set_metadata("router_error", "no_image_tier_configured");
                return Ok(StepAction::Continue);
            }

            let (tier_name, tier_cfg) = image_tiers[0];
            let baseline_model = ctx
                .get_metadata("resolved_model")
                .cloned()
                .or_else(|| ctx.get_metadata("model").cloned())
                .unwrap_or_default();
            ctx.set_metadata("baseline_model", &baseline_model);

            let decision = RoutingDecision {
                tier: tier_name.clone(),
                model: tier_cfg.model.clone(),
                confidence: 1.0,
                source: "image_route".to_string(),
            };

            let routing_applied = true;
            self.record_routing_metadata(ctx, &decision, routing_applied);
            ctx.set_metadata("image_route_reason", if current_turn_has_image { "current_turn" } else { "gate_history" });
            self.record_thinking_metadata(ctx, tier_name);

            // Record the routing decision as JSON for ModelSelectStep.
            let decision_json = serde_json::json!({
                "tier": decision.tier,
                "model": decision.model,
                "confidence": decision.confidence,
                "source": decision.source,
            });
            ctx.set_metadata("routing_decision", &decision_json.to_string());

            debug!(
                tier = %decision.tier,
                model = %decision.model,
                "squilla_router.image_routed"
            );
            return Ok(StepAction::Continue);
        }

        // Empty-text guard for the ML text classifier.
        let semantic_message = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == opensquilla_core::types::MessageRole::User)
            .map(|m| m.text_content())
            .unwrap_or_default();
        if semantic_message.trim().is_empty() {
            return Ok(StepAction::Continue);
        }

        let valid_tiers = self.valid_text_tiers();
        if valid_tiers.is_empty() {
            return Ok(StepAction::Continue);
        }

        // TODO(parity): router-control hold path. The Python step checks a
        // `RouterControlHoldStore` in `ctx.metadata["router_control_hold_store"]`
        // for a valid hold. The Rust PipelineContext has no such store; this
        // path is skipped.

        // TODO(parity): ML classification strategy. The Python step loads a
        // `V4Phase3Strategy` (or heuristic fallback) and calls
        // `strategy.classify(message, valid_tiers, routing_history=...)`.
        // The Rust engine has no ML runtime; we fall back to the default tier.

        let default_tier = if self.config.default_tier.is_empty() {
            DEFAULT_TEXT_TIER.to_string()
        } else {
            self.config.default_tier.clone()
        };

        let tier_name = if self.tier_cfg(&default_tier).is_some() {
            default_tier.clone()
        } else {
            valid_tiers.first().cloned().unwrap_or_default()
        };

        if tier_name.is_empty() {
            return Ok(StepAction::Continue);
        }

        let tier_cfg = self.tier_cfg(&tier_name).cloned().unwrap_or_default();
        let baseline_model = ctx
            .get_metadata("resolved_model")
            .cloned()
            .or_else(|| ctx.get_metadata("model").cloned())
            .unwrap_or_default();
        ctx.set_metadata("baseline_model", &baseline_model);

        let decision = RoutingDecision {
            tier: tier_name.clone(),
            model: tier_cfg.model.clone(),
            confidence: 0.0,
            source: "default".to_string(),
        };

        // TODO(parity): the full policy engine pipeline (confidence gate,
        // anti-downgrade, large-context floor, budget gate, provider-mismatch
        // veto, controller thinking/prompt-policy) is not implemented here.
        // The Rust routing crate has the types but the pipeline step does not
        // own a policy engine instance. When the policy engine is wired in,
        // it should run between classification and metadata recording.

        let routing_applied = self.config.rollout_phase != "observe";
        self.record_routing_metadata(ctx, &decision, routing_applied);
        self.record_thinking_metadata(ctx, &tier_name);

        // Record the routing decision as JSON for ModelSelectStep.
        let decision_json = serde_json::json!({
            "tier": decision.tier,
            "model": decision.model,
            "confidence": decision.confidence,
            "source": decision.source,
        });
        ctx.set_metadata("routing_decision", &decision_json.to_string());

        debug!(
            tier = %decision.tier,
            model = %decision.model,
            routing_applied = routing_applied,
            "squilla_router.routed"
        );

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "squilla_router"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    fn ctx_with_user(text: &str) -> PipelineContext {
        PipelineContext::new("t1".into(), vec![Message::user(text)])
    }

    fn sample_config() -> SquillaRouterConfig {
        SquillaRouterConfig {
            enabled: true,
            rollout_phase: "full".into(),
            default_tier: "c0".into(),
            tiers: vec![
                (
                    "c0".into(),
                    TierConfig {
                        model: "gpt-4o-mini".into(),
                        ..Default::default()
                    },
                ),
                (
                    "c1".into(),
                    TierConfig {
                        model: "gpt-4o".into(),
                        ..Default::default()
                    },
                ),
                (
                    "image_model".into(),
                    TierConfig {
                        model: "gpt-4o-vision".into(),
                        supports_image: true,
                        image_only: true,
                        ..Default::default()
                    },
                ),
            ],
            auto_thinking: true,
        }
    }

    #[tokio::test]
    async fn test_disabled_is_noop() {
        let step = SquillaRouterStep::new();
        let mut ctx = ctx_with_user("hello");
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert!(ctx.get_metadata("routed_tier").is_none());
    }

    #[tokio::test]
    async fn test_empty_text_skips() {
        let step = SquillaRouterStep::with_config(sample_config());
        let mut ctx = ctx_with_user("   ");
        step.execute(&mut ctx).await.unwrap();
        assert!(ctx.get_metadata("routed_tier").is_none());
    }

    #[tokio::test]
    async fn test_default_tier_routing() {
        let step = SquillaRouterStep::with_config(sample_config());
        let mut ctx = ctx_with_user("hello world");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("routed_tier").map(String::as_str),
            Some("c0")
        );
        assert_eq!(
            ctx.get_metadata("routed_model").map(String::as_str),
            Some("gpt-4o-mini")
        );
        assert_eq!(
            ctx.get_metadata("routing_source").map(String::as_str),
            Some("default")
        );
    }

    #[tokio::test]
    async fn test_image_routing() {
        let step = SquillaRouterStep::with_config(sample_config());
        let mut ctx = ctx_with_user("what is this");
        ctx.set_metadata("current_turn_has_image", "true");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("routed_tier").map(String::as_str),
            Some("image_model")
        );
        assert_eq!(
            ctx.get_metadata("routed_model").map(String::as_str),
            Some("gpt-4o-vision")
        );
        assert_eq!(
            ctx.get_metadata("image_route_reason").map(String::as_str),
            Some("current_turn")
        );
    }

    #[tokio::test]
    async fn test_subagent_skips() {
        let step = SquillaRouterStep::with_config(sample_config());
        let mut ctx = ctx_with_user("hello");
        ctx.set_metadata("is_subagent", "true");
        step.execute(&mut ctx).await.unwrap();
        assert!(ctx.get_metadata("routed_tier").is_none());
    }

    #[tokio::test]
    async fn test_observe_does_not_apply() {
        let mut config = sample_config();
        config.rollout_phase = "observe".into();
        let step = SquillaRouterStep::with_config(config);
        let mut ctx = ctx_with_user("hello");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("routing_applied").map(String::as_str),
            Some("false")
        );
    }

    #[tokio::test]
    async fn test_routing_decision_json() {
        let step = SquillaRouterStep::with_config(sample_config());
        let mut ctx = ctx_with_user("hello");
        step.execute(&mut ctx).await.unwrap();
        let raw = ctx.get_metadata("routing_decision").unwrap();
        let json: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(json["tier"], "c0");
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["source"], "default");
    }
}
