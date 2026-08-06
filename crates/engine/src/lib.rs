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

/// Context window management: system prompt assembly, token budget allocation.
pub mod context_builder;

/// Tool execution lifecycle: dispatch, timeout, result capture, retry.
pub mod tool_executor;

/// Token/cost budget enforcement, rate limiting, and circuit breakers.
pub mod budget;

/// Crash recovery, state reconstruction, and partial turn replay.
pub mod recovery;

pub use agent::{
    Agent, AgentBuilder, AgentConfig, AgentError, AgentRegistry, AgentSnapshot, AgentState,
    BackgroundProcess, BackgroundProcessManager, CommandResult, ErrorCategory,
    ErrorClassification, GitOpResult, GitOperation, GitResult, PlaceholderGenerator,
    RecoveryAction, ToolCallBudget, ToolCallDecision, ToolRoundResult, TurnContext,
    TurnGenerator, TurnOutcome, TurnPhase, UsageEvent, UsageStats, classify_error,
};
pub use compaction_control::{
    CompactionDecision, CompactionInput, CompactionPlan, CompactionStrategy,
    average_message_tokens, plan_compaction, should_compact_before_generation,
};
pub use pricing::{ModelPrice, ModelPricing, PricingCache, PricingResult};
/// Re-export the most commonly used types at the crate root for convenience.
pub use runtime::{AgentRuntime, TurnRunner, TurnRunnerBuilder};
pub use runtime::{NoopToolExecutor, ToolExecutor};
pub use session_lock::{SessionLockGuard, SessionLockSet, with_session_lock};

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
    BufferedToolCall, CompactionOutcome, CostRollup, FailoverOrder, FinalizeReport, HarnessConfig,
    InputConfig, InputMode, InputReport, MultipartField, PipelineConfig, ProviderCallReport,
    ProviderFailoverPolicy, ProviderOutcomeTracker, ProviderRetryPolicy, RateLimiter, RoutingConfig,
    StageMetric, StageMetrics, StageMetricsCollector, StageRollup, StageTimer, StreamConfig,
    StreamConsumerStage, StreamConsumerState, TurnAttachment, TurnErrorAggregator,
    TurnErrorBoundary, TurnErrorKind, TurnRunnerConfig,
};

// Routing policy engine, calibration, health ledger, and model selector.
pub use routing::{
    BudgetGateInput, BudgetGateResult, CalibrationState, ModelSelector, PolicyInputs, PolicyResult,
    ProviderConfig, ProviderFailureKind, ProviderHealthLedger, RouterConfig, RoutingDecision,
    RoutingPolicyEngine, SelectorConfig, TierCapability, TierConfig,
};

// Context window management and system prompt assembly.
pub use context_builder::{
    CharTokenEstimator, ContextPromptBuilder, ContextWindowManager, FragmentBuilder,
    SystemPromptAssembler, SystemPromptSection, TokenBudgetAllocation, TokenEstimator,
};

// Tool execution lifecycle.
pub use tool_executor::{
    ToolConcurrencyLimiter, ToolErrorKind, ToolExecutionConfig, ToolExecutionEngine,
    ToolExecutionEngineBuilder, ToolExecutionHook, ToolExecutionOutcome, ToolExecutorRegistry,
    ToolPermission, ToolPermissionRule, ToolPolicy, ToolResultCache, ToolOutputStream,
};

// Budget enforcement, rate limiting, and circuit breakers.
pub use budget::{
    BudgetCheckResult, BudgetConfig, BudgetManager, CostBudget, ModelRateLimiter,
    RequestAdmission, SessionBudgetReport, SessionBudgetTracker,
};

// Crash recovery, state reconstruction, and replay.
pub use recovery::{
    CrashRecovery, CrashRecoveryConfig, CrashSnapshot, RecoveryStatus, ReconstructedState,
    ReplayCheckpoint, ReplayDecision, ReplayOutcome, StateReconstructor, TransactionOutcome,
    TransactionStatus, TransactionalUpdate, TurnReplay,
    transactional::{TransactionJournal, TransactionJournalEntry},
};
