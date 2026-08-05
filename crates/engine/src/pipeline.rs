use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::Message;
use std::fmt;
use tracing::instrument;

/// The action to take after a pipeline step completes.
#[derive(Debug, Clone)]
pub enum StepAction {
    /// Continue to the next step in the pipeline.
    Continue,
    /// Skip this step's effect and continue to the next step.
    Skip,
    /// Halt the entire pipeline with a reason, preventing further processing.
    Halt(String),
}

impl fmt::Display for StepAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepAction::Continue => write!(f, "continue"),
            StepAction::Skip => write!(f, "skip"),
            StepAction::Halt(reason) => write!(f, "halt({})", reason),
        }
    }
}

/// The context passed through the pipeline, containing messages and metadata.
#[derive(Debug, Clone)]
pub struct PipelineContext {
    /// The unique identifier for the current turn.
    pub turn_id: String,
    /// The messages being processed in the pipeline.
    pub messages: Vec<Message>,
    /// Metadata key-value pairs that pipeline steps can read and write.
    pub metadata: std::collections::HashMap<String, String>,
}

impl PipelineContext {
    /// Create a new pipeline context with the given turn ID and messages.
    pub fn new(turn_id: String, messages: Vec<Message>) -> Self {
        Self {
            turn_id,
            messages,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a message to the pipeline context.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Set a metadata value for this pipeline context.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// A trait for steps in the pre-turn pipeline.
///
/// Pipeline steps are executed before the main turn stages. They can
/// transform messages, validate inputs, inject system prompts, or
/// halt the pipeline entirely.
#[async_trait]
pub trait PipelineStep: Send + Sync + fmt::Debug {
    /// Execute this pipeline step, returning an action indicating what to do next.
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction>;
}

/// A pipeline that chains multiple steps together, executing them in order.
#[derive(Debug, Default)]
pub struct Pipeline {
    /// The ordered list of steps in this pipeline.
    steps: Vec<Box<dyn PipelineStep + Send + Sync>>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the end of the pipeline.
    pub fn add_step(&mut self, step: Box<dyn PipelineStep + Send + Sync>) {
        self.steps.push(step);
    }

    /// Execute all steps in the pipeline in order.
    ///
    /// Returns the final action from the last step that returned a non-Continue
    /// action, or StepAction::Continue if all steps completed normally.
    #[instrument(skip(self), fields(turn_id = %ctx.turn_id))]
    pub async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        for step in &self.steps {
            match step.execute(ctx).await? {
                StepAction::Continue => continue,
                StepAction::Skip => continue,
                halt @ StepAction::Halt(_) => return Ok(halt),
            }
        }
        Ok(StepAction::Continue)
    }

    /// Get the number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the pipeline contains no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// A pipeline step that injects a system message at the beginning of the
/// message list, if one does not already exist.
#[derive(Debug)]
pub struct SystemPromptInjector {
    /// The system prompt text to inject.
    prompt: String,
    /// Whether to prepend (true) or append (false) the system prompt.
    prepend: bool,
}

impl SystemPromptInjector {
    /// Create a new system prompt injector.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            prepend: true,
        }
    }

    /// Set whether to prepend or append the system prompt.
    pub fn position(mut self, prepend: bool) -> Self {
        self.prepend = prepend;
        self
    }
}

#[async_trait]
impl PipelineStep for SystemPromptInjector {
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        // Check if any message already has the System role.
        let has_system = ctx.messages.iter().any(|m| {
            matches!(m.role, opensquilla_core::types::MessageRole::System)
        });

        if !has_system {
            let system_msg = Message::system(&self.prompt);
            if self.prepend {
                ctx.messages.insert(0, system_msg);
            } else {
                ctx.messages.push(system_msg);
            }
        }

        Ok(StepAction::Continue)
    }
}

/// A pipeline step that validates message count and size limits.
#[derive(Debug)]
pub struct MessageValidator {
    /// Maximum number of messages allowed in a single turn.
    max_messages: usize,
    /// Maximum total content length across all messages.
    max_total_length: usize,
}

impl MessageValidator {
    /// Create a new message validator with the given limits.
    pub fn new(max_messages: usize, max_total_length: usize) -> Self {
        Self {
            max_messages,
            max_total_length,
        }
    }
}

#[async_trait]
impl PipelineStep for MessageValidator {
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if ctx.messages.len() > self.max_messages {
            return Ok(StepAction::Halt(format!(
                "Message count {} exceeds maximum of {}",
                ctx.messages.len(),
                self.max_messages
            )));
        }

        let total_length: usize = ctx
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|block| match block {
                opensquilla_core::types::ContentBlock::Text(t) => Some(t.len()),
                _ => None,
            })
            .sum();

        if total_length > self.max_total_length {
            return Ok(StepAction::Halt(format!(
                "Total content length {} exceeds maximum of {}",
                total_length, self.max_total_length
            )));
        }

        Ok(StepAction::Continue)
    }
}

impl Default for MessageValidator {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_total_length: 100_000,
        }
    }
}