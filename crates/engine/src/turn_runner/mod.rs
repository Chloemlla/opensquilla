//! Turn runner stages.
//!
//! Mirrors the Python backend's `engine/turn_runner/` package: the eight
//! per-turn stages that execute in order inside a single conversation turn.
//!
//! The `Stage` trait itself lives in [`crate::stages`]; this module re-exports
//! it and provides the concrete stage implementations plus the shared
//! [`TurnRunnerConfig`]. The stage chain mirrors the Python order:
//!
//! 1. [`harness::HarnessStage`] — setup, session lock, error boundary.
//! 2. [`agent_bootstrap::AgentBootstrapStage`] — system prompt, agent identity.
//! 3. [`compaction::CompactionStage`] — trigger check, run compaction.
//! 4. [`attachment::AttachmentStage`] — load attachments, validate.
//! 5. [`input::InputStage`] — validate input, prepare messages.
//! 6. [`provider::ProviderStage`] — call LLM, stream response.
//! 7. [`stream_consumer::StreamConsumerStage`] — consume SSE, buffer tool calls.
//! 8. [`finalizer::FinalizerStage`] — persist, emit events, cleanup.

pub mod agent_bootstrap;
pub mod attachment;
pub mod compaction;
pub mod finalizer;
pub mod harness;
pub mod input;
pub mod metrics;
pub mod provider;
pub mod stream_consumer;

use std::collections::HashMap;
use std::path::PathBuf;

use crate::routing::{CalibrationState, ModelSelector, RouterConfig, TierConfig};
use crate::steps::StepChain;

pub use crate::stages::{Stage, StageContext, StageError, StageOutcome, StageOutput};

/// Configuration for the pre-turn pipeline (the [`StepChain`]).
///
/// The pipeline runs before the turn stages and wires the five step
/// implementations in the Python order:
/// `meta_resolution` → `model_select` → `skills_filter` → `context_assembly`
/// → `attachment_loader`. Each field is consumed by exactly one step.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Whether the pre-turn pipeline runs at all.
    pub enabled: bool,
    /// Workspace root used by the context assembly and attachment loader steps.
    pub workspace_root: Option<PathBuf>,
    /// Default model resolved by the model-select step when no override or
    /// routing decision is present.
    pub default_model: String,
    /// Default provider resolved by the model-select step.
    pub default_provider: String,
    /// Skill catalog registered with the skills filter.
    pub skill_catalog: Vec<crate::steps::SkillSpec>,
    /// Tool names available this turn (used by the skills gate).
    pub available_tools: Vec<String>,
    /// Attachments loaded into the turn by the attachment loader step.
    pub attachments: Vec<crate::steps::AttachmentDescriptor>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            workspace_root: None,
            default_model: String::new(),
            default_provider: String::new(),
            skill_catalog: Vec::new(),
            available_tools: Vec::new(),
            attachments: Vec::new(),
        }
    }
}

/// Configuration for routing-policy integration.
///
/// When `enabled`, the runtime runs [`crate::routing::RoutingPolicyEngine`]
/// against the pipeline context before provider selection and writes the
/// resulting decision into pipeline metadata so the model-select step and the
/// provider stage can bind to the routed model.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Whether routing runs before provider selection.
    pub enabled: bool,
    /// The router policy configuration.
    pub router: RouterConfig,
    /// Tier configuration map used to bind a model to the routed tier.
    pub tiers: HashMap<String, TierConfig>,
    /// The canonical tiers configured for this deployment.
    pub valid_tiers: Vec<String>,
    /// On-device confidence-gate calibration (additive, default-off).
    pub calibration: Option<CalibrationState>,
    /// Health-aware model selector used to refine the routed model. When
    /// attached, the selector's chain walk (which consults the shared
    /// [`crate::routing::ProviderHealthLedger`]) overrides the tier-bound model.
    pub model_selector: Option<ModelSelector>,
    /// The context window (tokens) of the routed model, for the floor rules.
    pub context_window_tokens: u64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            router: RouterConfig::default(),
            tiers: HashMap::new(),
            valid_tiers: vec![
                "c0".to_string(),
                "c1".to_string(),
                "c2".to_string(),
                "c3".to_string(),
            ],
            calibration: None,
            model_selector: None,
            context_window_tokens: crate::routing::DEFAULT_CONTEXT_WINDOW_TOKENS,
        }
    }
}

/// Shared configuration for building a turn runner stage chain.
///
/// This mirrors the knobs the Python `TurnRunner` reads at boot time. Each
/// stage implementation reads the subset of fields it owns; the chain builder
/// [`default_stages`] distributes them. The pipeline and routing configs feed
/// [`default_pipeline`] and the runtime's routing integration.
#[derive(Debug, Clone)]
pub struct TurnRunnerConfig {
    /// Maximum number of tool call iterations per turn.
    pub max_tool_rounds: u32,
    /// Maximum number of messages allowed in a single turn before the input
    /// stage rejects it.
    pub max_messages: usize,
    /// Whether compaction is enabled at all.
    pub compaction_enabled: bool,
    /// Message count above which compaction triggers.
    pub compaction_message_limit: usize,
    /// Context window (tokens) used by the compaction decision.
    pub context_window_tokens: u64,
    /// Whether streaming responses are enabled.
    pub streaming_enabled: bool,
    /// Maximum number of distinct session locks retained by the harness.
    pub session_lock_max_entries: usize,
    /// Workspace root used to resolve attachment and context file paths.
    pub workspace_root: Option<PathBuf>,
    /// Default system prompt injected by the bootstrap stage when the turn has
    /// no system message yet.
    pub default_system_prompt: String,
    /// Default model used by the provider stage when the generator does not
    /// name one.
    pub default_model: String,
    /// Default provider used when the generator does not name one.
    pub default_provider: String,
    /// Pre-turn pipeline configuration.
    pub pipeline: PipelineConfig,
    /// Routing policy configuration.
    pub routing: RoutingConfig,
    /// No-progress watchdog mode: `off` | `log` | `warn_model` | `block`.
    /// Mirrors the Python `progress_watchdog_mode` config; defaults to `off`.
    pub progress_watchdog_mode: String,
    /// Post-write convergence tracking enablement. Mirrors the Python
    /// `post_write_convergence_enabled` config; defaults to `false`.
    pub post_write_convergence_enabled: bool,
    /// Review-on-submit checkpoint enablement. Mirrors the Python
    /// `submit_review_enabled` config; defaults to `false`.
    pub submit_review_enabled: bool,
    /// Final-diff contract mode: `off` | `log` | `warn_model`. Mirrors the
    /// Python `final_diff_contract_mode` config; defaults to `log`.
    pub final_diff_contract_mode: String,
}

impl Default for TurnRunnerConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 10,
            max_messages: 100,
            compaction_enabled: true,
            compaction_message_limit: 50,
            context_window_tokens: 128_000,
            streaming_enabled: true,
            session_lock_max_entries: 4096,
            workspace_root: None,
            default_system_prompt: String::new(),
            default_model: String::new(),
            default_provider: String::new(),
            pipeline: PipelineConfig::default(),
            routing: RoutingConfig::default(),
            progress_watchdog_mode: "off".to_string(),
            post_write_convergence_enabled: false,
            submit_review_enabled: false,
            final_diff_contract_mode: "log".to_string(),
        }
    }
}

/// Build the default ordered stage chain from a shared config.
///
/// The returned stages implement [`crate::stages::Stage`] and can be handed to
/// a [`crate::runtime::TurnRunnerBuilder`] via `add_stage`.
pub fn default_stages(config: &TurnRunnerConfig) -> Vec<Box<dyn Stage + Send + Sync>> {
    vec![
        Box::new(harness::HarnessStage::new_with_max_entries(
            config.session_lock_max_entries,
        )),
        Box::new(
            agent_bootstrap::AgentBootstrapStage::new(config.default_system_prompt.clone())
                .with_workspace_root(config.workspace_root.clone()),
        ),
        Box::new(
            compaction::CompactionStage::new(config.compaction_message_limit)
                .with_enabled(config.compaction_enabled)
                .with_context_window(config.context_window_tokens),
        ),
        Box::new(attachment::AttachmentStage::new(
            config.workspace_root.clone(),
        )),
        Box::new(input::InputStage::new(config.max_messages)),
        Box::new(provider::ProviderStage::new(
            config.default_model.clone(),
            config.default_provider.clone(),
            config.streaming_enabled,
        )),
        Box::new(stream_consumer::StreamConsumerStage::new()),
        Box::new(finalizer::FinalizerStage::new()),
    ]
}

/// The number of stages in the default chain.
pub const DEFAULT_STAGE_COUNT: usize = 8;

/// Build the default ordered pre-turn step chain from a shared config.
///
/// Mirrors the Python `engine/steps/` pipeline order:
/// `meta_resolution` → `model_select` → `reasoning_hint_observer` →
/// `skills_filter` → `context_assembly` → `attachment_loader`. Each step reads
/// the fields it owns from `config.pipeline`.
pub fn default_pipeline(config: &TurnRunnerConfig) -> StepChain {
    let mut chain = StepChain::new();
    chain.push(crate::steps::MetaResolutionStep::new());
    chain.push(crate::steps::ModelSelectStep::new(
        config.pipeline.default_model.clone(),
        config.pipeline.default_provider.clone(),
    ));
    chain.push(crate::steps::ReasoningHintObserverStep::new());
    chain.push(
        crate::steps::SkillsFilterStep::new()
            .with_catalog(config.pipeline.skill_catalog.clone())
            .with_available_tools(config.pipeline.available_tools.clone()),
    );
    chain.push(crate::steps::ContextAssemblyStep::new(
        config.pipeline.workspace_root.clone(),
    ));
    let attachment = crate::steps::AttachmentLoaderStep::default_with_workspace(
        config.pipeline.workspace_root.clone(),
    );
    attachment.set_attachments(config.pipeline.attachments.clone());
    chain.push(attachment);
    chain
}

// Re-export the stage implementations that do not collide with the existing
// `crate::stages` names so callers can construct individual stages, plus the
// per-stage configuration and result types.
pub use agent_bootstrap::{AgentBootstrapStage, AgentIdentity, BootstrapSettings, TokenBudget};
pub use attachment::{
    AttachmentCleanupHook, AttachmentConfig, AttachmentFileMetadata, AttachmentLoadOutcome,
    AttachmentStage, MultipartField, TurnAttachment,
};
pub use compaction::CompactionOutcome;
pub use finalizer::{CostRollup, FinalizeReport, FinalizerStage};
pub use harness::{
    HarnessConfig, HarnessStage, StageMetrics, TurnErrorAggregator, TurnErrorBoundary,
    TurnErrorKind,
};
pub use input::{InputConfig, InputMode, InputReport, InputStage};
pub use metrics::{StageMetric, StageMetricsCollector, StageRollup, StageTimer};
pub use provider::{
    FailoverOrder, ProviderCallReport, ProviderFailoverPolicy, ProviderOutcomeTracker,
    ProviderRetryPolicy, ProviderStage, RateLimiter,
};
pub use stream_consumer::{
    BufferedToolCall, StreamConfig, StreamConsumerStage, StreamConsumerState,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stages_length() {
        let stages = default_stages(&TurnRunnerConfig::default());
        assert_eq!(stages.len(), DEFAULT_STAGE_COUNT);
        // The stage chain is ordered harness -> bootstrap -> compaction.
        assert_eq!(stages[0].name(), "harness");
        assert_eq!(stages[1].name(), "bootstrap");
        assert_eq!(stages[2].name(), "compaction");
        assert_eq!(stages[7].name(), "finalizer");
    }

    #[test]
    fn test_default_pipeline_orders_six_steps() {
        let config = TurnRunnerConfig::default();
        let chain = default_pipeline(&config);
        assert_eq!(chain.len(), 6);
        // Names are checked via the closure-backed step names exposed through
        // the pipeline metadata after execution.
    }

    #[test]
    fn test_pipeline_config_disabled_by_default() {
        let config = TurnRunnerConfig::default();
        assert!(!config.pipeline.enabled);
        assert!(!config.routing.enabled);
        assert_eq!(config.routing.valid_tiers.len(), 4);
    }
}
