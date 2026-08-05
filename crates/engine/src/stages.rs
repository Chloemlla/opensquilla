use crate::agent::TurnGenerator;
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::events::StreamEvent;
use opensquilla_core::types::{Message, Usage};
use std::fmt;
use tokio::sync::mpsc;
use tracing::{debug, info, instrument};

/// The context passed through each stage of the turn execution pipeline.
#[derive(Debug, Clone)]
pub struct StageContext {
    /// The unique identifier for the current turn.
    pub turn_id: String,
    /// The messages accumulated so far in the turn.
    pub messages: Vec<Message>,
    /// The name of the model currently being used.
    pub current_model: String,
    /// The name of the provider currently being used.
    pub current_provider: String,
    /// Token usage accumulated so far in this turn.
    pub usage: Usage,
    /// Optional sender for streaming events.
    pub streaming_tx: Option<mpsc::Sender<StreamEvent>>,
    /// The current tool call round (0-based).
    pub tool_round: u32,
    /// The maximum number of tool call rounds allowed.
    pub max_tool_rounds: u32,
}

/// The output of a stage after execution.
#[derive(Debug)]
pub enum StageOutput {
    /// Continue to the next stage in the pipeline.
    Continue,
    /// A complete output was produced; stop executing further stages.
    Output(StageOutcome),
    /// An error occurred during stage execution.
    Error(StageError),
}

impl StageOutput {
    /// Create a StageOutput from a StageOutcome.
    pub fn from_outcome(outcome: StageOutcome) -> Self {
        StageOutput::Output(outcome)
    }
}

/// A simplified turn outcome for stage output.
#[derive(Debug, Clone)]
pub struct StageOutcome {
    /// The final messages produced by this turn.
    pub messages: Vec<Message>,
    /// Token usage for this turn.
    pub usage: Usage,
    /// Duration of the turn in milliseconds.
    pub duration_ms: u64,
    /// Whether the turn completed successfully.
    pub success: bool,
    /// Optional error message if the turn failed.
    pub error_message: Option<String>,
}

impl StageOutcome {
    /// Create a new successful turn outcome.
    pub fn success(messages: Vec<Message>, usage: Usage, duration_ms: u64) -> Self {
        Self {
            messages,
            usage,
            duration_ms,
            success: true,
            error_message: None,
        }
    }

    /// Create a new failed turn outcome.
    pub fn failure(
        messages: Vec<Message>,
        usage: Usage,
        duration_ms: u64,
        error: impl Into<String>,
    ) -> Self {
        Self {
            messages,
            usage,
            duration_ms,
            success: false,
            error_message: Some(error.into()),
        }
    }
}

/// An error that occurred during stage execution.
#[derive(Debug, Clone)]
pub struct StageError {
    /// A human-readable error message.
    pub message: String,
    /// An optional error code for programmatic handling.
    pub code: Option<String>,
    /// The stage name where the error occurred.
    pub stage: String,
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = &self.code {
            write!(f, "[{}] {}: {}", self.stage, code, self.message)
        } else {
            write!(f, "{}: {}", self.stage, self.message)
        }
    }
}

impl std::error::Error for StageError {}

/// The trait that all turn stages must implement.
///
/// Stages are executed in order during a conversation turn. Each stage
/// can modify the context, produce output, or halt the pipeline.
#[async_trait]
pub trait Stage: Send + Sync + fmt::Debug {
    /// Execute this stage with the given context and generator.
    async fn execute(
        &self,
        ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<StageOutput>;

    /// The name of this stage for logging and error reporting.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Stage Implementations
// ---------------------------------------------------------------------------

/// The Harness stage initializes the turn context and prepares the environment.
///
/// This is the first stage in the pipeline. It resets the tool round counter,
/// validates the context, and sets up any initial state required for the turn.
#[derive(Debug)]
pub struct HarnessStage;

impl HarnessStage {
    /// Create a new HarnessStage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for HarnessStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for HarnessStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("Harness stage: initializing turn context");

        // Reset the tool round counter for this turn.
        ctx.tool_round = 0;

        // Validate that we have messages to process.
        if ctx.messages.is_empty() {
            return Ok(StageOutput::Error(StageError {
                message: "No messages provided for the turn".to_string(),
                code: Some("EMPTY_MESSAGES".to_string()),
                stage: self.name().to_string(),
            }));
        }

        info!(
            turn_id = %ctx.turn_id,
            message_count = ctx.messages.len(),
            "Harness stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "harness"
    }
}

/// The Bootstrap stage loads the generator's model and provider information.
///
/// This stage queries the generator for the model name and provider,
/// populating the context for downstream stages.
#[derive(Debug)]
pub struct BootstrapStage;

impl BootstrapStage {
    /// Create a new BootstrapStage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BootstrapStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for BootstrapStage {
    #[instrument(skip(self, ctx, generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("Bootstrap stage: loading model and provider info");

        ctx.current_model = generator.model_name().to_string();
        ctx.current_provider = generator.provider_name().to_string();

        info!(
            turn_id = %ctx.turn_id,
            model = %ctx.current_model,
            provider = %ctx.current_provider,
            "Bootstrap stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "bootstrap"
    }
}

/// The Compaction stage manages context window limits.
///
/// This stage checks if the message history exceeds the context window
/// and compacts it if necessary (e.g., by summarizing older messages or
/// dropping the least relevant ones).
#[derive(Debug)]
pub struct CompactionStage {
    /// The maximum number of messages allowed before compaction is triggered.
    max_messages: usize,
    /// Whether to enable automatic compaction.
    enabled: bool,
}

impl CompactionStage {
    /// Create a new CompactionStage with the given message limit.
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages,
            enabled: true,
        }
    }

    /// Enable or disable compaction.
    pub fn set_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
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
            debug!("Compaction disabled, skipping");
            return Ok(StageOutput::Continue);
        }

        if ctx.messages.len() <= self.max_messages {
            debug!(
                "Message count {} within limit {}, skipping compaction",
                ctx.messages.len(),
                self.max_messages
            );
            return Ok(StageOutput::Continue);
        }

        let before = ctx.messages.len();
        // Simple compaction: keep the first (system) message and the last N messages.
        let keep = self.max_messages.saturating_sub(1);
        let system_messages: Vec<Message> = ctx
            .messages
            .iter()
            .filter(|m| m.role == opensquilla_core::types::MessageRole::System)
            .cloned()
            .collect();

        let recent: Vec<Message> = ctx
            .messages
            .iter()
            .rev()
            .take(keep)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        ctx.messages = system_messages;
        ctx.messages.extend(recent);

        let after = ctx.messages.len();
        info!(
            turn_id = %ctx.turn_id,
            before = before,
            after = after,
            compacted = before - after,
            "Compaction stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "compaction"
    }
}

/// The Input stage processes the user's input messages before sending to the model.
///
/// This stage can perform input validation, transform messages, or inject
/// additional context.
#[derive(Debug)]
pub struct InputStage;

impl InputStage {
    /// Create a new InputStage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for InputStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for InputStage {
    #[instrument(skip(self), fields(stage = %self.name()))]
    async fn execute(
        &self,
        _ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("Input stage: processing input messages");
        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "input"
    }
}

/// The Provider stage sends the messages to the LLM provider and gets a response.
///
/// This is the core stage that interacts with the model. It calls the generator
/// and collects the response messages.
#[derive(Debug)]
pub struct ProviderStage;

impl ProviderStage {
    /// Create a new ProviderStage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ProviderStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for ProviderStage {
    #[instrument(skip(self, ctx, generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("Provider stage: sending messages to model");

        let response = generator.generate(&ctx.messages).await?;

        let input_tokens: u64 = ctx
            .messages
            .iter()
            .map(|m| m.text_content().len() as u64 / 4)
            .sum();

        let output_tokens: u64 = response
            .iter()
            .map(|m| m.text_content().len() as u64 / 4)
            .sum();

        ctx.usage = Usage::new(input_tokens, output_tokens);
        ctx.messages.extend(response);

        info!(
            turn_id = %ctx.turn_id,
            input_tokens = ctx.usage.input_tokens,
            output_tokens = ctx.usage.output_tokens,
            "Provider stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "provider"
    }
}

/// The Stream stage handles streaming responses from the provider.
///
/// This stage processes streaming events and forwards them to the
/// registered event sender.
#[derive(Debug)]
pub struct StreamStage;

impl StreamStage {
    /// Create a new StreamStage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for StreamStage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Stage for StreamStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        if let Some(tx) = &ctx.streaming_tx {
            debug!("Stream stage: processing streaming events");

            // Send a message stop event to indicate streaming is complete.
            let _ = tx
                .send(StreamEvent::MessageStop {
                    content: ctx
                        .messages
                        .iter()
                        .flat_map(|m| m.content.clone())
                        .collect(),
                    usage: Some(ctx.usage),
                })
                .await;

            info!(
                turn_id = %ctx.turn_id,
                "Stream stage: streaming events sent"
            );
        } else {
            debug!("Stream stage: no streaming channel configured, skipping");
        }

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "stream"
    }
}

/// The Finalizer stage performs post-processing on the turn output.
///
/// This is the last stage in the pipeline. It can perform cleanup,
/// logging, metrics collection, or any other finalization tasks.
#[derive(Debug)]
pub struct FinalizerStage;

impl FinalizerStage {
    /// Create a new FinalizerStage.
    pub fn new() -> Self {
        Self
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
        debug!("Finalizer stage: post-processing turn output");

        info!(
            turn_id = %ctx.turn_id,
            message_count = ctx.messages.len(),
            total_tokens = ctx.usage.total_tokens,
            "Finalizer stage complete"
        );

        // Return the final output to signal completion.
        let outcome = StageOutcome {
            messages: ctx.messages.clone(),
            usage: ctx.usage,
            duration_ms: 0, // Will be set by the runtime.
            success: true,
            error_message: None,
        };

        Ok(StageOutput::Output(outcome))
    }

    fn name(&self) -> &str {
        "finalizer"
    }
}

/// A helper function to create a default set of stages for a TurnRunner.
pub fn default_stages() -> Vec<Box<dyn Stage + Send + Sync>> {
    vec![
        Box::new(HarnessStage::new()) as Box<dyn Stage + Send + Sync>,
        Box::new(BootstrapStage::new()) as Box<dyn Stage + Send + Sync>,
        Box::new(CompactionStage::new(50)) as Box<dyn Stage + Send + Sync>,
        Box::new(InputStage::new()) as Box<dyn Stage + Send + Sync>,
        Box::new(ProviderStage::new()) as Box<dyn Stage + Send + Sync>,
        Box::new(StreamStage::new()) as Box<dyn Stage + Send + Sync>,
        Box::new(FinalizerStage::new()) as Box<dyn Stage + Send + Sync>,
    ]
}
