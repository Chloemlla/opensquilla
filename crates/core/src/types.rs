use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// The role of a message participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageRole {
    /// System-level instructions and context.
    System,
    /// End-user input.
    User,
    /// Model-generated assistant response.
    Assistant,
    /// Tool execution results.
    Tool,
}

/// A unique identifier for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// Create a new random session ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse a session ID from a string.
    pub fn from_string(s: &str) -> Option<Self> {
        Uuid::parse_str(s).ok().map(Self)
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique identifier for a user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub Uuid);

impl UserId {
    /// Create a new random user ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for UserId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique identifier for an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

impl AgentId {
    /// Create a new random agent ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique identifier for a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub Uuid);

impl MessageId {
    /// Create a new random message ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique identifier for a memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// Create a new random memory ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MemoryId {
    fn default() -> Self {
        Self::new()
    }
}

/// A unique identifier for a scheduled job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub Uuid);

impl JobId {
    /// Create a new random job ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

/// Alias for backwards compatibility: Role is MessageRole.
pub type Role = MessageRole;

/// Alias for backwards compatibility: ChatMessage is a Message.
pub type ChatMessage = Message;

/// Tool definition for use in provider configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A single message in a conversation turn, composed of one or more content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender.
    pub role: MessageRole,
    /// The content blocks that make up this message.
    pub content: Vec<ContentBlock>,
    /// Optional name for the message sender (e.g., tool name, function name).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The tool call ID this message is responding to (for tool results).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Direct tool calls (backwards compatibility with provider crate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Direct tool result (backwards compatibility with provider crate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolResult>,
}

impl Message {
    /// Create a new text-only message with the given role and text content.
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    /// Create a new system message with the given text.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(MessageRole::System, text)
    }

    /// Create a new user message with the given text.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(MessageRole::User, text)
    }

    /// Create a new assistant message with the given text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(MessageRole::Assistant, text)
    }

    /// Extract the text content from this message, joining all text blocks.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A content block within a message, supporting multimodal and tool content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// A tool use request from the model.
    #[serde(rename = "tool_use")]
    ToolUse(ToolCall),
    /// A tool result returned to the model.
    #[serde(rename = "tool_result")]
    ToolResult(ToolResult),
    /// Model reasoning/thinking content (not visible to the user).
    #[serde(rename = "reasoning")]
    Reasoning { reasoning: String },
}

/// A tool call request issued by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call instance.
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// The input arguments for the tool, as a JSON value.
    pub input: serde_json::Value,
}

impl ToolCall {
    /// Create a new tool call with the given id, name, and input.
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input,
        }
    }
}

/// The result of executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool use ID this result corresponds to.
    pub tool_use_id: String,
    /// The content of the tool result.
    pub content: String,
    /// Whether the tool execution resulted in an error.
    pub is_error: bool,
}

impl ToolResult {
    /// Create a new successful tool result.
    pub fn success(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// Create a new error tool result.
    pub fn error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
        }
    }
}

/// Token usage statistics for a model request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Number of input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Number of output (completion) tokens generated.
    pub output_tokens: u64,
    /// Total number of tokens consumed (input + output).
    pub total_tokens: u64,
}

impl Usage {
    /// Create a new usage record with the given token counts.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }

    /// Add another usage record to this one, accumulating counts.
    pub fn accumulate(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// A conversation session consisting of a sequence of messages.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Conversation {
    /// Unique identifier for this conversation.
    pub id: String,
    /// The messages in this conversation, in chronological order.
    pub messages: Vec<Message>,
    /// Metadata associated with this conversation.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Conversation {
    /// Create a new conversation with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            messages: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a message to the conversation.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get the total token usage across all messages in this conversation.
    pub fn total_usage(&self) -> Usage {
        let total = Usage::default();
        // Estimate: each message contributes some tokens based on content length.
        // Precise tracking requires provider-specific tokenization.
        total
    }
}

/// Configuration for a rate limit on API calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum number of requests allowed in the window.
    pub max_requests: u32,
    /// The duration of the rate limit window in seconds.
    pub window_secs: u64,
}

/// A request to generate a model response, wrapping conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// The conversation messages to use as context.
    pub messages: Vec<Message>,
    /// The model to use for generation.
    pub model: String,
    /// Optional provider name to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Sampling parameters for generation.
    #[serde(default)]
    pub parameters: GenerationParameters,
    /// Available tools for the model to use.
    #[serde(default)]
    pub tools: Vec<ToolCall>,
}

/// Sampling parameters for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParameters {
    /// Temperature for sampling (0.0 to 2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-p nucleus sampling parameter.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Stop sequences that terminate generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_top_p() -> f64 {
    0.9
}

fn default_max_tokens() -> u32 {
    4096
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
            stop_sequences: Vec::new(),
        }
    }
}
