use crate::types::{ContentBlock, Message, ToolCall, ToolResult, Usage};
use serde::{Deserialize, Serialize};

/// Events emitted during a streaming model response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum StreamEvent {
    /// A new content block has started.
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        /// The index of the content block.
        index: usize,
        /// The content block that started.
        block: ContentBlock,
    },
    /// A delta update to an existing content block.
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        /// The index of the content block being updated.
        index: usize,
        /// The delta update to apply.
        delta: ContentBlockDelta,
    },
    /// A content block has been completed.
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        /// The index of the content block that stopped.
        index: usize,
    },
    /// A message delta update (e.g., stop reason change).
    #[serde(rename = "message_delta")]
    MessageDelta {
        /// The delta to apply to the message.
        delta: MessageDelta,
        /// Updated usage information.
        usage: Option<Usage>,
    },
    /// The message has been fully streamed and is complete.
    #[serde(rename = "message_stop")]
    MessageStop {
        /// The complete message content blocks.
        content: Vec<ContentBlock>,
        /// Final usage information.
        usage: Option<Usage>,
    },
    /// An error occurred during streaming.
    #[serde(rename = "error")]
    Error {
        /// The error message.
        message: String,
        /// Optional error code.
        code: Option<String>,
    },
    /// A ping keepalive event.
    #[serde(rename = "ping")]
    Ping,
}

/// A delta update to a content block during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlockDelta {
    /// A text delta: partial text content.
    #[serde(rename = "text_delta")]
    TextDelta {
        /// The partial text content.
        text: String,
    },
    /// A reasoning delta: partial reasoning content.
    #[serde(rename = "reasoning_delta")]
    ReasoningDelta {
        /// The partial reasoning content.
        reasoning: String,
    },
    /// A tool call delta: partial tool call arguments.
    #[serde(rename = "input_json_delta")]
    InputJsonDelta {
        /// The partial JSON input string.
        partial_json: String,
    },
}

/// A delta update to a message during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    /// The stop reason for the message, if changed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// The stop sequence that terminated generation, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Events emitted during a conversation turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum TurnEvent {
    /// The turn has started.
    #[serde(rename = "turn_start")]
    TurnStart {
        /// The turn identifier.
        turn_id: String,
        /// The timestamp when the turn started.
        timestamp: String,
    },
    /// A message has been received from the user.
    #[serde(rename = "user_message")]
    UserMessage {
        /// The user message.
        message: Message,
    },
    /// The model has started generating a response.
    #[serde(rename = "generation_start")]
    GenerationStart {
        /// The model being used.
        model: String,
        /// The provider serving the request.
        provider: String,
    },
    /// A stream event from the model response.
    #[serde(rename = "stream_event")]
    StreamEvent {
        /// The stream event from the model.
        event: StreamEvent,
    },
    /// The turn has completed successfully.
    #[serde(rename = "turn_complete")]
    TurnComplete {
        /// The final assistant message.
        message: Message,
        /// Token usage for this turn.
        usage: Usage,
        /// Duration of the turn in milliseconds.
        duration_ms: u64,
    },
    /// The turn encountered an error.
    #[serde(rename = "turn_error")]
    TurnError {
        /// The error message.
        message: String,
        /// Optional error code.
        code: Option<String>,
    },
    /// Context compaction has been triggered.
    #[serde(rename = "compaction")]
    Compaction {
        /// Number of messages before compaction.
        before_count: usize,
        /// Number of messages after compaction.
        after_count: usize,
        /// Whether compaction was successful.
        success: bool,
    },
}

/// Events emitted during tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum ToolEvent {
    /// A tool call has been initiated.
    #[serde(rename = "tool_call_start")]
    ToolCallStart {
        /// The tool call details.
        call: ToolCall,
        /// The timestamp when the tool call started.
        timestamp: String,
    },
    /// A tool call has completed successfully.
    #[serde(rename = "tool_call_complete")]
    ToolCallComplete {
        /// The tool call that was executed.
        call: ToolCall,
        /// The result of the tool execution.
        result: ToolResult,
        /// Duration of the tool execution in milliseconds.
        duration_ms: u64,
    },
    /// A tool call has failed.
    #[serde(rename = "tool_call_error")]
    ToolCallError {
        /// The tool call that failed.
        call: ToolCall,
        /// The error message.
        error: String,
        /// Duration of the failed attempt in milliseconds.
        duration_ms: u64,
    },
    /// Multiple tool calls are being executed in parallel.
    #[serde(rename = "tool_parallel_start")]
    ToolParallelStart {
        /// The tool calls being executed.
        calls: Vec<ToolCall>,
    },
    /// All parallel tool calls have completed.
    #[serde(rename = "tool_parallel_complete")]
    ToolParallelComplete {
        /// The results of all tool executions.
        results: Vec<ToolResult>,
        /// Total duration in milliseconds.
        total_duration_ms: u64,
    },
    /// A tool call has been rejected (e.g., due to safety checks).
    #[serde(rename = "tool_rejected")]
    ToolRejected {
        /// The tool call that was rejected.
        call: ToolCall,
        /// The reason for rejection.
        reason: String,
    },
}

/// Combine multiple event types into a single observable event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GatewayEvent {
    /// A stream event from model generation.
    Stream(StreamEvent),
    /// A turn lifecycle event.
    Turn(TurnEvent),
    /// A tool execution event.
    Tool(ToolEvent),
}
