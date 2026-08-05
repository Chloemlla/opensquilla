use crate::agent::{AgentState, TurnGenerator, TurnOutcome};
use crate::pipeline::{PipelineContext, PipelineStep, StepAction};
use crate::routing::{PolicyInputs, RoutingDecision, RoutingPolicyEngine};
use crate::stages::{Stage, StageContext, StageOutcome, StageOutput};
use crate::steps::StepChain;
use crate::turn_control;
use crate::turn_runner::{PipelineConfig, RoutingConfig, TurnRunnerConfig};
use crate::usage::UsageTracker;
use async_trait::async_trait;
use dashmap::DashMap;
use opensquilla_core::error::Result;
use opensquilla_core::events::TurnEvent;
use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall, ToolResult, Usage};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// A trait for executing a tool call during the agent loop.
///
/// The runtime calls `execute` for every `tool_use` block returned by the
/// model, appends the resulting `ToolResult` back into the message history,
/// and loops back to the provider stage. Implementations should be shared
/// (via `Arc`) so the same executor can be used by the top-level runner and
/// any sub-agents.
#[async_trait]
pub trait ToolExecutor: Send + Sync + fmt::Debug {
    /// Execute a single tool call, returning its result.
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult>;

    /// A human-readable name for this executor, for logging.
    fn name(&self) -> &str {
        "tool-executor"
    }
}

/// A tool executor that always fails with a "not configured" error.
///
/// This is a safe default: tool calls are reported as errors rather than
/// silently dropped, and the agent loop terminates.
#[derive(Debug)]
pub struct NoopToolExecutor;

#[async_trait]
impl ToolExecutor for NoopToolExecutor {
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        Ok(ToolResult::error(
            &call.id,
            format!("No tool executor configured for '{}'", call.name),
        ))
    }
}

/// A handle to a running agent, allowing interaction with it.
#[derive(Debug, Clone)]
pub struct AgentHandle {
    /// The unique identifier for this agent.
    pub agent_id: String,
    /// The current state of the agent.
    pub state: AgentState,
    /// Total token usage across all turns.
    pub total_usage: Usage,
    /// Number of turns completed.
    pub turn_count: u64,
}

/// Builder for constructing a TurnRunner with optional stages and hooks.
#[derive(Debug)]
pub struct TurnRunnerBuilder {
    /// The ordered list of stages to execute during a turn.
    stages: Vec<Box<dyn Stage + Send + Sync>>,
    /// The pipeline steps to execute before the turn.
    pipeline_steps: Vec<Box<dyn PipelineStep + Send + Sync>>,
    /// The ordered pre-turn step chain (meta_resolution → ... → attachment).
    pipeline: StepChain,
    /// Optional routing policy configuration.
    routing: Option<RoutingConfig>,
    /// The usage tracker for accumulating token counts.
    usage_tracker: UsageTracker,
    /// The maximum number of tool call iterations per turn.
    max_tool_rounds: u32,
    /// Whether to enable streaming responses.
    #[allow(dead_code)]
    streaming_enabled: bool,
    /// The shared tool executor used to run tool calls in the agent loop.
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    /// Optional channel for emitting turn lifecycle events.
    event_tx: Option<mpsc::Sender<TurnEvent>>,
}

impl Default for TurnRunnerBuilder {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            pipeline_steps: Vec::new(),
            pipeline: StepChain::new(),
            routing: None,
            usage_tracker: UsageTracker::new(),
            max_tool_rounds: 10,
            streaming_enabled: true,
            tool_executor: None,
            event_tx: None,
        }
    }
}

impl TurnRunnerBuilder {
    /// Create a new TurnRunnerBuilder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a stage to the turn execution pipeline.
    pub fn add_stage(mut self, stage: Box<dyn Stage + Send + Sync>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add a pipeline step to run before the turn.
    pub fn add_pipeline_step(mut self, step: Box<dyn PipelineStep + Send + Sync>) -> Self {
        self.pipeline_steps.push(step);
        self
    }

    /// Add a pre-turn step to the [`StepChain`] pipeline.
    ///
    /// This accepts any [`crate::steps::PipelineStep`] implementation — e.g.
    /// the concrete `MetaResolutionStep`, `ModelSelectStep`, `SkillsFilterStep`,
    /// `ContextAssemblyStep`, or `AttachmentLoaderStep` — and appends it to the
    /// ordered chain executed before the turn stages.
    pub fn add_chain_step<S>(mut self, step: S) -> Self
    where
        S: crate::steps::PipelineStep + Send + Sync + 'static,
    {
        self.pipeline.push(step);
        self
    }

    /// Enable routing-policy integration for turns run through this runner.
    pub fn with_routing(mut self, routing: RoutingConfig) -> Self {
        self.routing = Some(routing);
        self
    }

    /// Enable routing-policy integration from a config flag.
    pub fn routing_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.routing = Some(RoutingConfig::default());
        } else {
            self.routing = None;
        }
        self
    }

    /// Set the pipeline (pre-turn step chain) configuration.
    pub fn with_pipeline_config(mut self, config: PipelineConfig) -> Self {
        let chain = crate::turn_runner::default_pipeline(&TurnRunnerConfig {
            pipeline: config,
            ..TurnRunnerConfig::default()
        });
        self.pipeline = chain;
        self
    }

    /// Set the maximum number of tool call iterations per turn.
    pub fn max_tool_rounds(mut self, max: u32) -> Self {
        self.max_tool_rounds = max;
        self
    }

    /// Enable or disable streaming responses.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming_enabled = enabled;
        self
    }

    /// Set the tool executor used to run tool calls in the agent loop.
    ///
    /// Without an executor, tool calls returned by the model end the turn
    /// without being executed.
    pub fn tool_executor(mut self, executor: Arc<dyn ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    /// Set the channel used to emit turn lifecycle events.
    pub fn event_sender(mut self, tx: mpsc::Sender<TurnEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Build the TurnRunner with the configured options.
    pub fn build(self) -> TurnRunner {
        TurnRunner {
            stages: self.stages,
            pipeline_steps: self.pipeline_steps,
            pipeline: self.pipeline,
            routing: self.routing,
            usage_tracker: self.usage_tracker,
            max_tool_rounds: self.max_tool_rounds,
            streaming_enabled: self.streaming_enabled,
            tool_executor: self.tool_executor,
            event_tx: self.event_tx,
        }
    }
}

/// Runs a single conversation turn, orchestrating the pipeline and stages.
#[derive(Debug)]
pub struct TurnRunner {
    /// The ordered list of stages to execute during a turn.
    stages: Vec<Box<dyn Stage + Send + Sync>>,
    /// The pipeline steps to execute before the turn.
    pipeline_steps: Vec<Box<dyn PipelineStep + Send + Sync>>,
    /// The ordered pre-turn step chain (meta_resolution → ... → attachment).
    pipeline: StepChain,
    /// Optional routing policy configuration.
    routing: Option<RoutingConfig>,
    /// The usage tracker for accumulating token counts.
    usage_tracker: UsageTracker,
    /// The maximum number of tool call iterations per turn.
    max_tool_rounds: u32,
    /// Whether to enable streaming responses.
    #[allow(dead_code)]
    streaming_enabled: bool,
    /// The shared tool executor used to run tool calls in the agent loop.
    tool_executor: Option<Arc<dyn ToolExecutor>>,
    /// Optional channel for emitting turn lifecycle events.
    event_tx: Option<mpsc::Sender<TurnEvent>>,
}

impl TurnRunner {
    /// Execute a full turn with the given messages and generator.
    ///
    /// This method runs the pipeline, then drives the agent loop:
    ///
    /// 1. Run the pre-provider stages (harness, bootstrap, compaction, input).
    /// 2. Enter the tool-call loop around the provider stage:
    ///    - Call the generator (via the provider stage).
    ///    - Inspect the response for tool calls.
    ///    - If tool calls are present, execute them with the configured
    ///      `ToolExecutor`, append the results, and loop back to the provider.
    ///    - Otherwise (`end_turn` / final text) break out of the loop.
    /// 3. Run the post-provider stages (stream, finalizer).
    ///
    /// Turn lifecycle events (`TurnStart`, `GenerationStart`, `TurnComplete`,
    /// `TurnError`) are emitted through the configured event channel.
    #[instrument(skip(self, generator), fields(turn_id))]
    pub async fn run_turn(
        &self,
        mut messages: Vec<Message>,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        let turn_id = Uuid::new_v4().to_string();
        let start = Instant::now();

        info!(turn_id = %turn_id, "Starting turn execution");

        self.emit_event(
            &turn_id,
            TurnEvent::TurnStart {
                turn_id: turn_id.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
        );

        // Build the pipeline context.
        let mut ctx = PipelineContext::new(turn_id.clone(), messages.clone());

        // Run pipeline steps before the turn.
        for step in &self.pipeline_steps {
            debug!("Executing pipeline step");
            match step.execute(&mut ctx).await {
                Ok(StepAction::Continue) => continue,
                Ok(StepAction::Skip) => {
                    debug!("Pipeline step requested skip, continuing");
                    continue;
                }
                Ok(StepAction::Halt(reason)) => {
                    info!(reason = %reason, "Pipeline step halted execution");
                    return Ok(TurnOutcome::Halted {
                        reason,
                        messages: ctx.messages,
                        usage: self.usage_tracker.current(),
                    });
                }
                Err(e) => {
                    error!(turn_id = %turn_id, error = %e, "Pipeline step failed");
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: ctx.messages,
                        usage: self.usage_tracker.current(),
                    });
                }
            }
        }

        // Use the messages from the pipeline context for the stages.
        messages = ctx.messages;

        // Build the stage context.
        let mut stage_ctx = StageContext {
            turn_id: turn_id.clone(),
            messages,
            current_model: String::new(),
            current_provider: String::new(),
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: self.max_tool_rounds,
        };

        // Locate the provider stage to drive the agent loop. If no stage is
        // named "provider", fall back to a linear pass through all stages.
        let provider_idx = self.stages.iter().position(|s| s.name() == "provider");

        let outcome = if let Some(pidx) = provider_idx {
            self.run_agent_loop(&mut stage_ctx, generator, pidx).await
        } else {
            self.run_stages_linearly(&mut stage_ctx, generator).await
        };

        self.usage_tracker.record(&stage_ctx.usage);

        // Patch in the real duration and emit terminal events.
        let duration_ms = start.elapsed().as_millis() as u64;
        let outcome = match outcome {
            Ok(TurnOutcome::Complete { messages, usage, .. }) => {
                info!(
                    turn_id = %turn_id,
                    duration_ms = duration_ms,
                    message_count = messages.len(),
                    "Turn completed"
                );
                let final_message = messages
                    .iter()
                    .rev()
                    .find(|m| matches!(m.role, MessageRole::Assistant))
                    .cloned();
                if let Some(message) = final_message {
                    self.emit_event(
                        &turn_id,
                        TurnEvent::TurnComplete {
                            message,
                            usage,
                            duration_ms,
                        },
                    );
                }
                TurnOutcome::Complete {
                    messages,
                    usage,
                    duration_ms,
                }
            }
            Ok(TurnOutcome::Error { message, .. }) => {
                error!(turn_id = %turn_id, error = %message, "Turn failed");
                self.emit_event(
                    &turn_id,
                    TurnEvent::TurnError {
                        message: message.clone(),
                        code: Some("TURN_ERROR".to_string()),
                    },
                );
                TurnOutcome::Error {
                    message,
                    messages: stage_ctx.messages.clone(),
                    usage: self.usage_tracker.current(),
                }
            }
            Ok(other) => other,
            Err(e) => {
                error!(turn_id = %turn_id, error = %e, "Stage execution failed");
                TurnOutcome::Error {
                    message: e.to_string(),
                    messages: stage_ctx.messages.clone(),
                    usage: self.usage_tracker.current(),
                }
            }
        };

        Ok(outcome)
    }

    /// Run the agent loop: pre-provider stages, then the tool-call loop around
    /// the provider stage, then the post-provider stages.
    async fn run_agent_loop(
        &self,
        stage_ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
        provider_idx: usize,
    ) -> Result<TurnOutcome> {
        // Run stages before the provider.
        for stage in &self.stages[..provider_idx] {
            debug!(stage = %stage.name(), "Executing stage");
            match stage.execute(stage_ctx, generator).await? {
                StageOutput::Continue => {}
                StageOutput::Output(outcome) => return Ok(stage_outcome_to_turn(outcome)),
                StageOutput::Error(e) => {
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: stage_ctx.messages.clone(),
                        usage: stage_ctx.usage,
                    });
                }
            }
        }

        let provider_stage = &self.stages[provider_idx];
        let executor: Option<&dyn ToolExecutor> =
            self.tool_executor.as_ref().map(|e| e.as_ref() as &dyn ToolExecutor);

        loop {
            // Emit generatoreration start for each provider round.
            debug!(
                turn_id = %stage_ctx.turn_id,
                tool_round = stage_ctx.tool_round,
                "Running provider stage"
            );
            self.emit_event(
                &stage_ctx.turn_id,
                TurnEvent::GenerationStart {
                    model: stage_ctx.current_model.clone(),
                    provider: stage_ctx.current_provider.clone(),
                },
            );

            let prev_len = stage_ctx.messages.len();
            // Preserve usage accumulated in earlier rounds; the provider stage
            // overwrites `ctx.usage` with the current round's snapshot.
            let prev_usage = stage_ctx.usage;

            match provider_stage.execute(stage_ctx, generator).await? {
                StageOutput::Continue => {}
                StageOutput::Output(outcome) => return Ok(stage_outcome_to_turn(outcome)),
                StageOutput::Error(e) => {
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: stage_ctx.messages.clone(),
                        usage: stage_ctx.usage,
                    });
                }
            }

            // Accumulate this round's usage into the running total.
            stage_ctx.usage.accumulate(&prev_usage);

            // Inspect the newly appended messages for tool calls.
            let new_messages = &stage_ctx.messages[prev_len..];
            let calls = turn_control::pending_tool_calls(new_messages);

            if calls.is_empty() {
                // end_turn / final-text surface: no more tool calls to run.
                info!(
                    turn_id = %stage_ctx.turn_id,
                    tool_round = stage_ctx.tool_round,
                    "Provider response has no tool calls, ending loop"
                );
                break;
            }

            if stage_ctx.tool_round >= stage_ctx.max_tool_rounds {
                warn!(
                    turn_id = %stage_ctx.turn_id,
                    tool_round = stage_ctx.tool_round,
                    max_tool_rounds = stage_ctx.max_tool_rounds,
                    "Max tool rounds reached, ending loop"
                );
                break;
            }

            let executor = match executor {
                Some(e) => e,
                None => {
                    warn!(
                        turn_id = %stage_ctx.turn_id,
                        "Tool calls present but no ToolExecutor configured, ending turn"
                    );
                    break;
                }
            };

            // Execute each tool call and append its result.
            for call in calls {
                info!(
                    turn_id = %stage_ctx.turn_id,
                    tool = %call.name,
                    call_id = %call.id,
                    "Executing tool call"
                );
                let result = match executor.execute(&call).await {
                    Ok(result) => result,
                    Err(e) => {
                        error!(
                            turn_id = %stage_ctx.turn_id,
                            tool = %call.name,
                            error = %e,
                            "Tool execution failed"
                        );
                        ToolResult::error(&call.id, e.to_string())
                    }
                };
                stage_ctx.messages.push(tool_result_message(call, result));
            }

            stage_ctx.tool_round += 1;
        }

        // Run the remaining stages (stream, finalizer).
        for stage in &self.stages[provider_idx + 1..] {
            debug!(stage = %stage.name(), "Executing stage");
            match stage.execute(stage_ctx, generator).await? {
                StageOutput::Continue => {}
                StageOutput::Output(outcome) => return Ok(stage_outcome_to_turn(outcome)),
                StageOutput::Error(e) => {
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: stage_ctx.messages.clone(),
                        usage: stage_ctx.usage,
                    });
                }
            }
        }

        // No stage produced an explicit output; synthesize a completion.
        Ok(TurnOutcome::Complete {
            messages: stage_ctx.messages.clone(),
            usage: stage_ctx.usage,
            duration_ms: 0,
        })
    }

    /// Run all stages exactly once, in order (used when no provider stage
    /// exists to drive the agent loop).
    async fn run_stages_linearly(
        &self,
        stage_ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        for stage in &self.stages {
            debug!(stage = %stage.name(), "Executing stage");
            match stage.execute(stage_ctx, generator).await? {
                StageOutput::Continue => continue,
                StageOutput::Output(outcome) => return Ok(stage_outcome_to_turn(outcome)),
                StageOutput::Error(e) => {
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: stage_ctx.messages.clone(),
                        usage: stage_ctx.usage,
                    });
                }
            }
        }
        Ok(TurnOutcome::Complete {
            messages: stage_ctx.messages.clone(),
            usage: stage_ctx.usage,
            duration_ms: 0,
        })
    }

    /// Emit a turn lifecycle event through the configured channel.
    ///
    /// Emission is best-effort and non-blocking: if the channel is full or
    /// closed, the event is dropped with a debug log.
    fn emit_event(&self, turn_id: &str, event: TurnEvent) {
        if let Some(tx) = &self.event_tx {
            if let Err(e) = tx.try_send(event) {
                debug!(turn_id = %turn_id, error = %e, "Failed to emit turn event");
            }
        }
    }

    /// Get a reference to the usage tracker.
    pub fn usage_tracker(&self) -> &UsageTracker {
        &self.usage_tracker
    }

    /// Get a mutable reference to the usage tracker.
    pub fn usage_tracker_mut(&mut self) -> &mut UsageTracker {
        &mut self.usage_tracker
    }

    /// Set the tool executor used to run tool calls in the agent loop.
    pub fn set_tool_executor(&mut self, executor: Arc<dyn ToolExecutor>) {
        self.tool_executor = Some(executor);
    }

    /// Get a reference to the current tool executor, if set.
    pub fn tool_executor(&self) -> Option<&dyn ToolExecutor> {
        self.tool_executor.as_ref().map(|e| e.as_ref() as &dyn ToolExecutor)
    }
}

impl TurnRunner {
    /// Construct a TurnRunner from a [`TurnRunnerConfig`].
    ///
    /// This builds the 8-stage chain via [`crate::turn_runner::default_stages`],
    /// the 5-step pre-turn pipeline via [`crate::turn_runner::default_pipeline`],
    /// and carries the routing configuration into the runner so
    /// [`TurnRunner::run_with_routing`] can drive the policy engine.
    pub fn from_config(config: &TurnRunnerConfig) -> Self {
        let mut builder = TurnRunnerBuilder::new()
            .max_tool_rounds(config.max_tool_rounds)
            .streaming(config.streaming_enabled)
            .with_routing(config.routing.clone());
        for stage in crate::turn_runner::default_stages(config) {
            builder = builder.add_stage(stage);
        }
        let mut runner = builder.build();
        runner.pipeline = crate::turn_runner::default_pipeline(config);
        runner
    }

    /// Execute the pre-turn pipeline over the given context.
    ///
    /// The pipeline is composed of two layers, both executed in order:
    ///
    /// 1. the legacy [`PipelineStep`] objects (from
    ///    [`TurnRunnerBuilder::add_pipeline_step`]), kept for backward
    ///    compatibility;
    /// 2. the new [`StepChain`] (from [`TurnRunnerBuilder::add_chain_step`] or
    ///    [`crate::turn_runner::default_pipeline`]).
    ///
    /// The first `Halt(reason)` short-circuits and is returned; any `Err`
    /// propagates to the caller.
    #[instrument(skip(self), fields(turn_id = %ctx.turn_id))]
    pub async fn run_pipeline(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        for step in &self.pipeline_steps {
            debug!("Executing legacy pipeline step");
            match step.execute(ctx).await? {
                StepAction::Continue | StepAction::Skip => continue,
                halt @ StepAction::Halt(_) => return Ok(halt),
            }
        }
        if !self.pipeline.is_empty() {
            debug!(step_count = self.pipeline.len(), "Executing step chain");
        }
        self.pipeline.execute(ctx).await
    }

    /// Run the routing policy engine over the current pipeline context.
    ///
    /// When routing is not configured or disabled, returns `None`. Otherwise it
    /// builds [`PolicyInputs`] from the pipeline metadata (tier, model,
    /// confidence, source) plus the configured tier map, runs the
    /// [`RoutingPolicyEngine`], and returns the final [`RoutingDecision`].
    ///
    /// The decision is not applied here; callers use [`TurnRunner::apply_routing`]
    /// or the higher-level [`TurnRunner::run_with_routing`] to bind it.
    pub fn run_routing_decision(&self, ctx: &PipelineContext) -> Option<RoutingDecision> {
        let routing = self.routing.as_ref()?;
        if !routing.enabled {
            return None;
        }

        let tier = ctx
            .get_metadata("routing_tier")
            .cloned()
            .filter(|t| !t.is_empty())
            .or_else(|| routing.valid_tiers.first().cloned())
            .unwrap_or_else(|| crate::routing::DEFAULT_TEXT_TIER.to_string());
        let model = ctx
            .get_metadata("routing_model")
            .cloned()
            .filter(|m| !m.is_empty())
            .unwrap_or_default();
        let confidence = ctx
            .get_metadata("routing_confidence")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.5);
        let source = ctx
            .get_metadata("routing_source")
            .cloned()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "heuristic".to_string());

        let decision = RoutingDecision::new(tier, model, confidence, source);

        // The most recent user message drives complaint detection.
        let message = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content())
            .unwrap_or_default();

        let inputs = PolicyInputs {
            decision,
            message,
            router_cfg: routing.router.clone(),
            tiers: routing.tiers.clone(),
            valid_tiers: routing.valid_tiers.clone(),
            routing_history: None,
            extra: Some(HashMap::new()),
            thinking_mode: ctx.get_metadata("thinking_mode").cloned(),
            prompt_policy: ctx.get_metadata("prompt_policy").cloned(),
            history_strategy: true,
            material_estimated_tokens: ctx
                .get_metadata("material_estimated_tokens")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0),
            context_window_tokens: routing.context_window_tokens,
            now: None,
            turn_has_image: false,
            tier_capabilities: None,
            calibration: routing.calibration.clone(),
            budget: None,
        };

        let engine = RoutingPolicyEngine::new();
        let result = engine.run(&inputs);
        let mut decision = result.decision;

        // Refine the tier-bound model through the health-aware model selector
        // (its chain walk consults the shared provider health ledger, so a
        // benched deployment is skipped when an alternative exists).
        if let Some(selector) = &routing.model_selector {
            if let Some(selected) = selector.select_model() {
                if !selected.is_empty() {
                    decision.model = selected;
                }
            }
        }

        Some(decision)
    }

    /// Write a routing decision into pipeline metadata so the model-select step
    /// and provider stage can bind to it.
    fn apply_routing(&self, ctx: &mut PipelineContext) {
        let Some(decision) = self.run_routing_decision(ctx) else {
            return;
        };
        let metadata = crate::steps::model_select::RoutingMetadata {
            tier: decision.tier.clone(),
            model: decision.model.clone(),
            confidence: decision.confidence,
            source: decision.source.clone(),
        };
        if let Ok(json) = serde_json::to_string(&metadata) {
            ctx.set_metadata("routing_decision", json);
        }
        ctx.set_metadata("routed_tier", &decision.tier);
        ctx.set_metadata("routing_source", &decision.source);
        if !decision.model.is_empty() {
            ctx.set_metadata("resolved_model", &decision.model);
        }
        debug!(
            tier = %decision.tier,
            model = %decision.model,
            source = %decision.source,
            "Routing decision applied to pipeline"
        );
    }

    /// The shared turn execution core.
    ///
    /// `run_pipeline` and `run_routing` select the execution mode:
    ///
    /// * `(false, false)` — standalone: stages run directly.
    /// * `(true, false)` — pipeline + stages.
    /// * `(true, true)` — pipeline + routing + stages.
    ///
    /// Emits the same lifecycle events as [`TurnRunner::run_turn`].
    #[instrument(skip(self, generator), fields(turn_id))]
    async fn run_turn_inner(
        &self,
        messages: Vec<Message>,
        generator: &dyn TurnGenerator,
        run_pipeline: bool,
        run_routing: bool,
    ) -> Result<TurnOutcome> {
        let turn_id = Uuid::new_v4().to_string();
        let start = Instant::now();
        info!(turn_id = %turn_id, "Starting turn execution");

        self.emit_event(
            &turn_id,
            TurnEvent::TurnStart {
                turn_id: turn_id.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
        );

        let mut ctx = PipelineContext::new(turn_id.clone(), messages.clone());

        if run_pipeline {
            match self.run_pipeline(&mut ctx).await {
                Ok(StepAction::Halt(reason)) => {
                    info!(reason = %reason, "Pipeline step halted execution");
                    return Ok(TurnOutcome::Halted {
                        reason,
                        messages: ctx.messages,
                        usage: self.usage_tracker.current(),
                    });
                }
                Ok(_) => {}
                Err(e) => {
                    error!(turn_id = %turn_id, error = %e, "Pipeline step failed");
                    return Ok(TurnOutcome::Error {
                        message: e.to_string(),
                        messages: ctx.messages,
                        usage: self.usage_tracker.current(),
                    });
                }
            }
        }

        if run_routing {
            self.apply_routing(&mut ctx);
        }

        // Seed the stage context with the routed model when available.
        let routed_model = ctx.get_metadata("resolved_model").cloned().unwrap_or_default();
        let routed_provider = ctx
            .get_metadata("provider_name")
            .cloned()
            .filter(|p| !p.is_empty())
            .unwrap_or_default();

        let mut stage_ctx = StageContext {
            turn_id: turn_id.clone(),
            messages: ctx.messages,
            current_model: routed_model,
            current_provider: routed_provider,
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: self.max_tool_rounds,
        };

        let provider_idx = self.stages.iter().position(|s| s.name() == "provider");
        let outcome = if let Some(pidx) = provider_idx {
            self.run_agent_loop(&mut stage_ctx, generator, pidx).await
        } else {
            self.run_stages_linearly(&mut stage_ctx, generator).await
        };

        self.usage_tracker.record(&stage_ctx.usage);

        let duration_ms = start.elapsed().as_millis() as u64;
        let outcome = match outcome {
            Ok(TurnOutcome::Complete { messages, usage, .. }) => {
                info!(
                    turn_id = %turn_id,
                    duration_ms = duration_ms,
                    message_count = messages.len(),
                    "Turn completed"
                );
                let final_message = messages
                    .iter()
                    .rev()
                    .find(|m| matches!(m.role, MessageRole::Assistant))
                    .cloned();
                if let Some(message) = final_message {
                    self.emit_event(
                        &turn_id,
                        TurnEvent::TurnComplete {
                            message,
                            usage,
                            duration_ms,
                        },
                    );
                }
                TurnOutcome::Complete {
                    messages,
                    usage,
                    duration_ms,
                }
            }
            Ok(TurnOutcome::Error { message, .. }) => {
                error!(turn_id = %turn_id, error = %message, "Turn failed");
                self.emit_event(
                    &turn_id,
                    TurnEvent::TurnError {
                        message: message.clone(),
                        code: Some("TURN_ERROR".to_string()),
                    },
                );
                TurnOutcome::Error {
                    message,
                    messages: stage_ctx.messages.clone(),
                    usage: self.usage_tracker.current(),
                }
            }
            Ok(other) => other,
            Err(e) => {
                error!(turn_id = %turn_id, error = %e, "Stage execution failed");
                TurnOutcome::Error {
                    message: e.to_string(),
                    messages: stage_ctx.messages.clone(),
                    usage: self.usage_tracker.current(),
                }
            }
        };

        Ok(outcome)
    }

    /// Execute a turn with no pipeline and no routing: the stages run directly
    /// against the provided messages.
    pub async fn run_standalone(
        &self,
        messages: Vec<Message>,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        self.run_turn_inner(messages, generator, false, false).await
    }

    /// Execute a turn with the pre-turn pipeline (steps + step chain) but no
    /// routing policy.
    pub async fn run_with_pipeline(
        &self,
        messages: Vec<Message>,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        self.run_turn_inner(messages, generator, true, false).await
    }

    /// Execute a turn with the full flow: pre-turn pipeline, routing policy,
    /// and then the turn stages.
    pub async fn run_with_routing(
        &self,
        messages: Vec<Message>,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        self.run_turn_inner(messages, generator, true, true).await
    }

    /// Whether routing policy integration is configured and enabled.
    pub fn routing_enabled(&self) -> bool {
        self.routing.as_ref().map(|r| r.enabled).unwrap_or(false)
    }

    /// Get a reference to the configured routing config, if any.
    pub fn routing_config(&self) -> Option<&RoutingConfig> {
        self.routing.as_ref()
    }
}

/// Convert a `StageOutcome` into a `TurnOutcome::Complete`.
fn stage_outcome_to_turn(outcome: StageOutcome) -> TurnOutcome {
    TurnOutcome::Complete {
        messages: outcome.messages,
        usage: outcome.usage,
        duration_ms: outcome.duration_ms,
    }
}

/// Build a tool-result message from a call and its result.
fn tool_result_message(call: ToolCall, result: ToolResult) -> Message {
    Message {
        role: MessageRole::Tool,
        content: vec![ContentBlock::ToolResult(result.clone())],
        name: Some(call.name),
        tool_call_id: Some(result.tool_use_id.clone()),
        tool_calls: None,
        tool_result: Some(result),
    }
}

/// The top-level agent runtime that manages multiple agents and their turns.
#[derive(Debug)]
pub struct AgentRuntime {
    /// The registry of active agent handles.
    agents: Arc<DashMap<String, AgentHandle>>,
    /// The default TurnRunner configuration for executing turns.
    runner: Arc<RwLock<TurnRunner>>,
    /// Channel for broadcasting turn events to subscribers.
    event_tx: mpsc::Sender<TurnEvent>,
    /// The tool executor shared across turns, synced onto the runner before
    /// each execution.
    tool_executor: std::sync::Mutex<Option<Arc<dyn ToolExecutor>>>,
    /// Whether the runtime is currently running.
    running: Arc<RwLock<bool>>,
}

impl AgentRuntime {
    /// Create a new AgentRuntime with the given TurnRunner and event channel.
    ///
    /// The event channel is also injected into the runner so every turn emits
    /// lifecycle events through it.
    pub fn new(runner: TurnRunner, event_tx: mpsc::Sender<TurnEvent>) -> Self {
        let mut runner = runner;
        runner.event_tx = Some(event_tx.clone());
        Self {
            agents: Arc::new(DashMap::new()),
            runner: Arc::new(RwLock::new(runner)),
            event_tx,
            tool_executor: std::sync::Mutex::new(None),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Set the tool executor used to run tool calls in the agent loop.
    ///
    /// The executor is applied to the underlying TurnRunner on the next
    /// `execute_turn` call.
    pub fn set_tool_executor(&self, executor: Arc<dyn ToolExecutor>) {
        *self.tool_executor.lock().unwrap() = Some(executor);
    }

    /// Get the currently configured tool executor, if any.
    pub fn tool_executor(&self) -> Option<Arc<dyn ToolExecutor>> {
        self.tool_executor.lock().unwrap().clone()
    }

    /// Start the runtime, allowing it to process turns.
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = true;
        info!("AgentRuntime started");
        Ok(())
    }

    /// Gracefully shut down the runtime, stopping all agent processing.
    pub async fn shutdown(&self) -> Result<()> {
        let mut running = self.running.write().await;
        *running = false;
        self.agents.clear();
        info!("AgentRuntime shut down");
        Ok(())
    }

    /// Check whether the runtime is currently running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Register a new agent in the runtime.
    pub fn register_agent(&self, agent_id: String, handle: AgentHandle) {
        self.agents.insert(agent_id, handle);
        debug!("Agent registered");
    }

    /// Remove an agent from the runtime by its ID.
    pub fn unregister_agent(&self, agent_id: &str) -> Option<AgentHandle> {
        let handle = self.agents.remove(agent_id).map(|(_, v)| v);
        if handle.is_some() {
            debug!(agent_id = %agent_id, "Agent unregistered");
        }
        handle
    }

    /// Get a handle to a registered agent by ID.
    pub fn get_agent(&self, agent_id: &str) -> Option<AgentHandle> {
        self.agents.get(agent_id).map(|r| r.value().clone())
    }

    /// List all registered agent IDs.
    pub fn list_agents(&self) -> Vec<String> {
        self.agents.iter().map(|r| r.key().clone()).collect()
    }

    /// Get the total number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Get the current TurnRunner configuration.
    pub async fn get_runner(&self) -> tokio::sync::RwLockReadGuard<'_, TurnRunner> {
        self.runner.read().await
    }

    /// Update the TurnRunner configuration.
    ///
    /// The runtime's event channel is injected into the new runner so turn
    /// lifecycle events keep flowing.
    pub async fn set_runner(&self, mut runner: TurnRunner) {
        runner.event_tx = Some(self.event_tx.clone());
        let mut lock = self.runner.write().await;
        *lock = runner;
    }

    /// Get a sender for publishing turn events.
    pub fn event_sender(&self) -> mpsc::Sender<TurnEvent> {
        self.event_tx.clone()
    }

    /// Execute a turn for a specific agent, using a registered generator.
    pub async fn execute_turn(
        &self,
        agent_id: &str,
        messages: Vec<Message>,
        generator: &dyn TurnGenerator,
    ) -> Result<TurnOutcome> {
        if !*self.running.read().await {
            return Err(opensquilla_core::error::Error::Internal(
                "Runtime is not running".to_string(),
            ));
        }

        // Briefly take a write lock to sync the latest tool executor onto the
        // runner, then run the turn under a read lock so concurrent turns for
        // different agents are not serialized.
        {
            let mut runner = self.runner.write().await;
            if let Some(executor) = self.tool_executor.lock().unwrap().clone() {
                runner.tool_executor = Some(executor);
            }
        }
        let runner = self.runner.read().await;
        let outcome = runner.run_turn(messages, generator).await?;

        // Update the agent handle with the outcome.
        if let Some(mut handle) = self.agents.get_mut(agent_id) {
            match &outcome {
                TurnOutcome::Complete { usage, .. } => {
                    handle.total_usage.accumulate(usage);
                    handle.turn_count += 1;
                }
                _ => {}
            }
        }

        Ok(outcome)
    }
}

impl Default for TurnRunner {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            pipeline_steps: Vec::new(),
            pipeline: StepChain::new(),
            routing: None,
            usage_tracker: UsageTracker::new(),
            max_tool_rounds: 10,
            streaming_enabled: true,
            tool_executor: None,
            event_tx: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::TierConfig;
    use crate::steps::ClosureStep;
    use opensquilla_core::types::Message;
    use std::pin::Pin;

    type BoxedStepFuture<'a> =
        Pin<Box<dyn std::future::Future<Output = Result<StepAction>> + Send + 'a>>;

    #[derive(Debug)]
    struct MockGenerator {
        model: String,
        provider: String,
    }

    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _messages: &[Message]) -> Result<Vec<Message>> {
            Ok(vec![Message::assistant("mock response")])
        }

        fn model_name(&self) -> &str {
            &self.model
        }

        fn provider_name(&self) -> &str {
            &self.provider
        }
    }

    fn config_with_system() -> TurnRunnerConfig {
        let mut config = TurnRunnerConfig::default();
        config.default_system_prompt = "You are a test agent.".to_string();
        config
    }

    #[tokio::test]
    async fn test_standalone_runs_stage_chain() {
        let runner = TurnRunner::from_config(&config_with_system());
        let generator = MockGenerator {
            model: "test-model".into(),
            provider: "test-provider".into(),
        };
        let outcome = runner
            .run_standalone(vec![Message::user("hello")], &generator)
            .await
            .unwrap();
        assert!(matches!(outcome, TurnOutcome::Complete { .. }));
        let messages = outcome.messages();
        // The system prompt is injected by the bootstrap stage.
        assert!(messages.iter().any(|m| m.role == MessageRole::System));
    }

    #[tokio::test]
    async fn test_from_config_builds_stages_and_pipeline() {
        let mut config = config_with_system();
        config.pipeline.enabled = true;
        let runner = TurnRunner::from_config(&config);
        assert_eq!(runner.stages.len(), crate::turn_runner::DEFAULT_STAGE_COUNT);
        assert_eq!(runner.pipeline.len(), 5);
        assert_eq!(runner.stages[0].name(), "harness");
        assert_eq!(runner.stages[7].name(), "finalizer");
    }

    #[tokio::test]
    async fn test_run_with_pipeline_executes_step_chain() {
        let runner = TurnRunnerBuilder::new()
            .add_chain_step(ClosureStep::new(
                "inject",
                |ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
                    Box::pin(async move {
                        ctx.add_message(Message::system("pipeline ran"));
                        Ok(StepAction::Continue)
                    })
                },
            ))
            .build();
        let generator = MockGenerator {
            model: "test-model".into(),
            provider: "test-provider".into(),
        };
        let outcome = runner
            .run_with_pipeline(vec![Message::user("hello")], &generator)
            .await
            .unwrap();
        assert!(matches!(outcome, TurnOutcome::Complete { .. }));
        assert!(outcome
            .messages()
            .iter()
            .any(|m| m.role == MessageRole::System && m.text_content() == "pipeline ran"));
    }

    #[tokio::test]
    async fn test_run_pipeline_halt_short_circuits() {
        let runner = TurnRunnerBuilder::new()
            .add_chain_step(ClosureStep::new(
                "halt",
                |_ctx: &mut PipelineContext| -> BoxedStepFuture<'_> {
                    Box::pin(async move { Ok(StepAction::Halt("stop".to_string())) })
                },
            ))
            .build();
        let generator = MockGenerator {
            model: "m".into(),
            provider: "p".into(),
        };
        let outcome = runner
            .run_with_pipeline(vec![Message::user("hello")], &generator)
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            TurnOutcome::Halted { reason, .. } if reason == "stop"
        ));
    }

    #[test]
    fn test_run_routing_decision_binds_model() {
        let mut config = config_with_system();
        config.routing.enabled = true;
        config.routing.tiers.insert(
            "c1".to_string(),
            TierConfig {
                model: "deepseek-chat".to_string(),
                ..Default::default()
            },
        );
        let runner = TurnRunner::from_config(&config);
        assert!(runner.routing_enabled());
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        ctx.set_metadata("routing_tier", "c1");
        let decision = runner.run_routing_decision(&ctx).unwrap();
        assert_eq!(decision.tier, "c1");
        assert_eq!(decision.model, "deepseek-chat");
    }

    #[test]
    fn test_routing_disabled_returns_none() {
        let runner = TurnRunnerBuilder::new().build();
        let ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        assert!(runner.run_routing_decision(&ctx).is_none());
        assert!(!runner.routing_enabled());
    }
}