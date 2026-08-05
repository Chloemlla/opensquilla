//! # OpenSquilla Engine
//!
//! The agent engine that orchestrates conversation turns, manages the agent
//! state machine, and coordinates the pipeline of processing stages.
//! This crate implements the core runtime for executing agent interactions
//! with LLM providers, tool execution, and context management.

/// TurnRunner and AgentRuntime for managing the agent lifecycle.
pub mod runtime;

/// Agent state machine and the TurnGenerator trait for producing responses.
pub mod agent;

/// Pre-turn pipeline for processing messages before they reach the model.
pub mod pipeline;

/// Pre-turn pipeline step implementations (meta resolution, model selection,
/// skills filtering, context assembly, attachment loading).
pub mod steps;

/// Turn runner stage implementations and shared configuration.
pub mod turn_runner;

/// Routing policy engine: tiers, calibration, health ledger, model selector.
pub mod routing;

/// Turn runner stages: Harness, Bootstrap, Compaction, Input, Provider,
/// Stream, Finalizer.
pub mod stages;

/// Hook traits for extending turn behavior: TurnHook, CompactionHook, ToolHook.
pub mod hooks;

/// Usage tracking and accumulation across turns.
pub mod usage;

/// Context assembly: load SOUL.md, AGENTS.md, and workspace files.
pub mod context;

/// Pure turn-control decisions: continue / stop / error classification.
pub mod turn_control;

/// Conversation history management: truncation, tool-pair repair, rebuild.
pub mod history;

/// Reasoning cleanup for cross-provider compatibility.
pub mod thinking;

/// Sub-agent management: process-internal agents and shared providers.
pub mod subagent;

/// Slash command registry and command metadata enums.
pub mod commands;

/// Empty / no-progress turn recovery decisions.
pub mod runtime_recovery;

/// Per-session asynchronous locks with automatic cleanup.
pub mod session_lock;

/// Compaction continuation decisions.
pub mod compaction_control;

/// Model pricing cache and cost calculation.
pub mod pricing;

/// Re-export the most commonly used types at the crate root for convenience.
pub use runtime::{AgentRuntime, TurnRunner, TurnRunnerBuilder};
pub use agent::{
    Agent, AgentConfig, AgentError, AgentRegistry, AgentState, BackgroundProcess, CommandResult,
    GitOperation, GitResult, RecoveryAction, TurnContext, TurnGenerator, TurnOutcome, UsageEvent,
    UsageStats,
};
pub use session_lock::{SessionLockGuard, SessionLockSet, with_session_lock};
pub use compaction_control::{CompactionDecision, CompactionInput, CompactionStrategy};
pub use pricing::{ModelPricing, ModelPrice, PricingCache, PricingResult};
pub use runtime::{NoopToolExecutor, ToolExecutor};

// Pre-turn pipeline steps. The `PipelineStep` trait and `StepAction` live in
// the `pipeline` module (re-exported above); the concrete step types and the
// step chain are re-exported here.
pub use steps::{
    AttachmentDescriptor, AttachmentLoaderStep, ContextAssemblyStep, MetaResolutionConfig,
    MetaResolutionStep, ModelSelectConfig, ModelSelectStep, SkillSpec, SkillsFilterConfig,
    SkillsFilterStep, StepChain,
};

// Turn runner stages and shared configuration. The stage types whose names
// collide with `crate::stages` (harness, compaction, input, provider,
// finalizer) remain reachable through `turn_runner::<stage>::<Stage>`.
pub use turn_runner::{
    AgentBootstrapStage, AgentIdentity, AttachmentCleanupHook, AttachmentConfig,
    AttachmentFileMetadata, AttachmentLoadOutcome, AttachmentStage, BootstrapSettings,
    BufferedToolCall, CompactionOutcome, CostRollup, FinalizeReport, HarnessConfig,
    InputConfig, InputMode, InputReport, MultipartField, PipelineConfig, ProviderCallReport,
    ProviderRetryPolicy, RateLimiter, RoutingConfig, StageMetrics, StreamConfig,
    StreamConsumerStage, StreamConsumerState, TokenBudget, TurnAttachment, TurnRunnerConfig,
};

// Routing policy engine, calibration, health ledger, and model selector.
pub use routing::{
    BudgetGateInput, BudgetGateResult, CalibrationState, ModelSelector, PolicyInputs, PolicyResult,
    ProviderConfig, ProviderFailureKind, ProviderHealthLedger, RouterConfig, RoutingDecision,
    RoutingPolicyEngine, SelectorConfig, TierCapability, TierConfig,
};