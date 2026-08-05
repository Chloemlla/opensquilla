//! Stream consumer stage.
//!
//! Mirrors the Python `engine/turn_runner/stream_consumer_stage.py` stage. It
//! runs after the provider stage. It consumes the streaming output surface:
//!
//! * extracts and buffers tool-call deltas into complete
//!   [`BufferedToolCall`] structures,
//! * accumulates the streamed text parts of the assistant messages,
//! * extracts reasoning/thinking deltas into a parallel channel,
//! * forwards `ContentBlockStart`/`ContentBlockStop`/`MessageStop` events
//!   through the context's stream channel so WebSocket/HTTP clients observe a
//!   normal SSE lifecycle,
//! * manages stream timeouts and surfaces stream errors as structured state.
//!
//! The stage is observable but non-mutating: it reads the messages the
//! provider appended and records its findings in a per-turn state that the
//! finalizer can consume for persistence.

use crate::agent::TurnGenerator;
use crate::stages::{Stage, StageContext, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::events::{ContentBlockDelta, StreamEvent};
use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall};
use std::sync::Mutex;
use std::time::Duration;
use tracing::{debug, info, instrument, warn};

/// Configuration for the stream consumer stage.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum time to wait for the stream to deliver its terminal event.
    pub timeout: Duration,
    /// Maximum number of deltas buffered per text part before warning.
    pub max_delta_budget: usize,
    /// Whether reasoning deltas are extracted into a separate accumulator.
    pub extract_reasoning: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            max_delta_budget: 100_000,
            extract_reasoning: true,
        }
    }
}

/// A tool call buffered from streamed deltas.
#[derive(Debug, Clone)]
pub struct BufferedToolCall {
    /// The tool use id.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// Accumulated JSON input (may be partial; parse lazily).
    pub partial_json: String,
}

impl BufferedToolCall {
    /// Create a new buffered tool call with empty input.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            partial_json: String::new(),
        }
    }

    /// Append an input-json delta.
    pub fn push_json(&mut self, delta: &str) {
        self.partial_json.push_str(delta);
    }

    /// Try to parse the accumulated input as JSON.
    pub fn parsed_input(&self) -> Option<serde_json::Value> {
        serde_json::from_str(&self.partial_json).ok()
    }

    /// Convert into a concrete [`ToolCall`] when the input parses.
    pub fn into_tool_call(self) -> Option<ToolCall> {
        let input = self.parsed_input()?;
        Some(ToolCall::new(self.id, self.name, input))
    }
}

/// Per-turn streaming state accumulated by the consumer stage.
#[derive(Debug, Clone, Default)]
pub struct StreamConsumerState {
    /// Text parts streamed so far, in order.
    pub text_parts: Vec<String>,
    /// Tool calls buffered from streamed deltas.
    pub buffered_tool_calls: Vec<BufferedToolCall>,
    /// Reasoning deltas accumulated from the stream.
    pub reasoning_parts: Vec<String>,
    /// Whether a message-stop event was emitted.
    pub message_stopped: bool,
    /// Any error surfaced while consuming the stream.
    pub error_message: Option<String>,
    /// A machine-readable error code, when the stream failed.
    pub error_code: Option<String>,
    /// Whether the stream timed out before its terminal event.
    pub timed_out: bool,
    /// Number of deltas consumed from the channel.
    pub delta_count: u64,
}

impl StreamConsumerState {
    /// The concatenated streamed text.
    pub fn final_text(&self) -> String {
        self.text_parts.join("")
    }

    /// The concatenated reasoning text.
    pub fn reasoning_text(&self) -> String {
        self.reasoning_parts.join("")
    }

    /// True when any tool calls were buffered.
    pub fn has_tool_calls(&self) -> bool {
        !self.buffered_tool_calls.is_empty()
    }

    /// Convert the buffered tool calls into concrete `ToolCall`s, dropping any
    /// that do not parse as JSON.
    pub fn assembled_tool_calls(&self) -> Vec<ToolCall> {
        self.buffered_tool_calls
            .iter()
            .filter_map(|b| {
                Some(ToolCall::new(
                    b.id.clone(),
                    b.name.clone(),
                    b.parsed_input()?,
                ))
            })
            .collect()
    }
}

/// The stream consumer stage in the turn pipeline.
#[derive(Debug)]
pub struct StreamConsumerStage {
    /// Per-turn state (the stage is shared across turns, so the state is a
    /// mutable slot replaced on each execution).
    state: Mutex<StreamConsumerState>,
    /// Configuration.
    config: StreamConfig,
}

impl StreamConsumerStage {
    /// Create a new stream consumer stage.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(StreamConsumerState::default()),
            config: StreamConfig::default(),
        }
    }

    /// Replace the stage configuration.
    pub fn with_config(mut self, config: StreamConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the stream timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// The stage configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// The current consumer state (after the latest execution).
    pub fn state(&self) -> StreamConsumerState {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Extract tool calls from the assistant messages and buffer them.
    fn buffer_tool_calls(&self, messages: &[Message]) -> Vec<BufferedToolCall> {
        let mut buffered: Vec<BufferedToolCall> = Vec::new();
        for message in messages {
            if message.role != MessageRole::Assistant {
                continue;
            }
            // Direct tool_calls field (provider-crate compatibility).
            if let Some(calls) = &message.tool_calls {
                for call in calls {
                    let mut b = BufferedToolCall::new(&call.id, &call.name);
                    if let Some(s) = call.input.as_str() {
                        b.partial_json = s.to_string();
                    } else {
                        b.partial_json = call.input.to_string();
                    }
                    buffered.push(b);
                }
            }
            // Content-block tool_use.
            for block in &message.content {
                if let ContentBlock::ToolUse(call) = block {
                    let mut b = BufferedToolCall::new(&call.id, &call.name);
                    if let Some(s) = call.input.as_str() {
                        b.partial_json = s.to_string();
                    } else {
                        b.partial_json = call.input.to_string();
                    }
                    buffered.push(b);
                }
            }
        }
        buffered
    }

    /// Accumulate the text parts of the assistant messages.
    fn collect_text(&self, messages: &[Message]) -> Vec<String> {
        let mut parts = Vec::new();
        for message in messages {
            if message.role != MessageRole::Assistant {
                continue;
            }
            let text = message.text_content();
            if !text.is_empty() {
                parts.push(text);
            }
        }
        parts
    }

    /// Consume a live stream from an mpsc channel of core [`StreamEvent`]s.
    ///
    /// Waits for the terminal `MessageStop` (or a timeout) while accumulating
    /// text, reasoning, and tool-call deltas. Stream errors and timeouts are
    /// recorded in the returned state.
    pub async fn consume_channel(
        &self,
        mut rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    ) -> StreamConsumerState {
        let mut state = StreamConsumerState::default();
        let mut tool_by_id: std::collections::HashMap<String, BufferedToolCall> =
            std::collections::HashMap::new();
        let mut tool_order: Vec<String> = Vec::new();

        loop {
            let event = match tokio::time::timeout(self.config.timeout, rx.recv()).await {
                Ok(Some(event)) => event,
                Ok(None) => break, // channel closed cleanly
                Err(_) => {
                    state.timed_out = true;
                    state.error_message = Some(format!(
                        "stream timed out after {}s",
                        self.config.timeout.as_secs()
                    ));
                    state.error_code = Some("STREAM_TIMEOUT".to_string());
                    warn!(
                        timeout_s = self.config.timeout.as_secs(),
                        "stream consumer timed out"
                    );
                    break;
                }
            };

            state.delta_count += 1;
            match event {
                StreamEvent::ContentBlockStart { index, block } => {
                    let _ = index;
                    // A tool-use block starts a new buffered tool call; the
                    // subsequent InputJsonDelta fragments accumulate onto it.
                    if let ContentBlock::ToolUse(call) = block {
                        let mut b = BufferedToolCall::new(&call.id, &call.name);
                        if let Some(s) = call.input.as_str() {
                            b.partial_json = s.to_string();
                        } else {
                            b.partial_json = call.input.to_string();
                        }
                        tool_by_id.insert(call.id.clone(), b);
                        tool_order.push(call.id.clone());
                    }
                }
                StreamEvent::ContentBlockStop { .. } => {}
                StreamEvent::ContentBlockDelta { index, delta } => {
                    let _ = index;
                    match delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                state.text_parts.push(text);
                            }
                        }
                        ContentBlockDelta::ReasoningDelta { reasoning } => {
                            if self.config.extract_reasoning && !reasoning.is_empty() {
                                state.reasoning_parts.push(reasoning);
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            // Without an explicit tool-call id in the delta, we
                            // append to the most recently started tool call.
                            if let Some(last_id) = tool_order.last().cloned() {
                                if let Some(call) = tool_by_id.get_mut(&last_id) {
                                    call.push_json(&partial_json);
                                }
                            }
                        }
                    }
                }
                StreamEvent::MessageDelta { .. } => {}
                StreamEvent::MessageStop { content, usage } => {
                    // The terminal event carries the fully assembled content.
                    let _ = usage;
                    // Extract any tool calls embedded in the final content.
                    for block in &content {
                        if let ContentBlock::ToolUse(call) = block {
                            let mut b = BufferedToolCall::new(&call.id, &call.name);
                            if let Some(s) = call.input.as_str() {
                                b.partial_json = s.to_string();
                            } else {
                                b.partial_json = call.input.to_string();
                            }
                            tool_by_id.insert(call.id.clone(), b);
                            tool_order.push(call.id.clone());
                        }
                        if let ContentBlock::Text(t) = block {
                            if !t.is_empty() {
                                state.text_parts.push(t.clone());
                            }
                        }
                        if let ContentBlock::Reasoning(r) = block {
                            if self.config.extract_reasoning && !r.is_empty() {
                                state.reasoning_parts.push(r.clone());
                            }
                        }
                    }
                    state.message_stopped = true;
                    break;
                }
                StreamEvent::Error { message, code } => {
                    state.error_message = Some(message);
                    state.error_code = code;
                    break;
                }
                StreamEvent::Ping => {}
            }
        }

        // Preserve tool-call ordering.
        state.buffered_tool_calls = tool_order
            .into_iter()
            .filter_map(|id| tool_by_id.remove(&id))
            .collect();
        state
    }

    /// Snapshot the accumulated per-turn state from already-appended messages.
    pub fn snapshot(&self, ctx: &StageContext) -> StreamConsumerState {
        let assistant_messages: Vec<Message> = ctx
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .cloned()
            .collect();
        StreamConsumerState {
            text_parts: self.collect_text(&assistant_messages),
            buffered_tool_calls: self.buffer_tool_calls(&assistant_messages),
            reasoning_parts: extract_reasoning_from_messages(&assistant_messages),
            message_stopped: false,
            error_message: None,
            error_code: None,
            timed_out: false,
            delta_count: 0,
        }
    }
}

impl Default for StreamConsumerStage {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract reasoning text from assistant messages.
fn extract_reasoning_from_messages(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|m| m.role == MessageRole::Assistant)
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::Reasoning(r) if !r.is_empty() => Some(r.clone()),
            _ => None,
        })
        .collect()
}

#[async_trait]
impl Stage for StreamConsumerStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("stream_consumer: consuming provider stream output");

        let assistant_messages: Vec<Message> = ctx
            .messages
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .cloned()
            .collect();

        let text_parts = self.collect_text(&assistant_messages);
        let buffered_tool_calls = self.buffer_tool_calls(&assistant_messages);
        let reasoning_parts = extract_reasoning_from_messages(&assistant_messages);

        // Forward stream events when a channel exists.
        let mut message_stopped = false;
        if let Some(tx) = &ctx.streaming_tx {
            let mut index = 0usize;
            for message in &assistant_messages {
                for block in &message.content {
                    let _ = tx
                        .send(StreamEvent::ContentBlockStart {
                            index,
                            block: block.clone(),
                        })
                        .await;
                    let _ = tx.send(StreamEvent::ContentBlockStop { index }).await;
                    index += 1;
                }
            }
            let _ = tx
                .send(StreamEvent::MessageStop {
                    content: assistant_messages
                        .iter()
                        .flat_map(|m| m.content.clone())
                        .collect(),
                    usage: Some(ctx.usage),
                })
                .await;
            message_stopped = true;
        }

        let state = StreamConsumerState {
            text_parts,
            buffered_tool_calls,
            reasoning_parts,
            message_stopped,
            error_message: None,
            error_code: None,
            timed_out: false,
            delta_count: 0,
        };
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();

        info!(
            turn_id = %ctx.turn_id,
            text_parts = state.text_parts.len(),
            tool_calls = state.buffered_tool_calls.len(),
            reasoning_parts = state.reasoning_parts.len(),
            message_stopped = state.message_stopped,
            "stream_consumer stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "stream_consumer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_buffer_tool_calls_from_content_blocks() {
        let stage = StreamConsumerStage::new();
        let call = ToolCall::new("call_1", "read_file", json!({"path": "/a"}));
        let msg = Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse(call.clone())],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let buffered = stage.buffer_tool_calls(&[msg]);
        assert_eq!(buffered.len(), 1);
        assert_eq!(buffered[0].id, "call_1");
        assert_eq!(buffered[0].name, "read_file");
        assert!(buffered[0].parsed_input().is_some());
    }

    #[test]
    fn test_buffered_tool_call_into_tool_call() {
        let mut b = BufferedToolCall::new("c1", "web_search");
        b.push_json("{\"q\": \"rust\"}");
        let tc = b.into_tool_call().expect("parses");
        assert_eq!(tc.id, "c1");
        assert_eq!(tc.input["q"], "rust");
    }

    #[test]
    fn test_collect_text_and_reasoning() {
        let stage = StreamConsumerStage::new();
        let msg = Message {
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Reasoning("thinking".into()),
                ContentBlock::Text("answer".into()),
            ],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let text = stage.collect_text(&[msg.clone()]);
        assert_eq!(text, vec!["answer".to_string()]);
        let reasoning = extract_reasoning_from_messages(&[msg]);
        assert_eq!(reasoning, vec!["thinking".to_string()]);
    }

    #[tokio::test]
    async fn test_consume_channel_assembles_events() {
        let stage = StreamConsumerStage::new();
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        tx.send(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: "Hello".to_string(),
            },
        })
        .await
        .unwrap();
        tx.send(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::TextDelta {
                text: " world".to_string(),
            },
        })
        .await
        .unwrap();
        tx.send(StreamEvent::ContentBlockDelta {
            index: 0,
            delta: ContentBlockDelta::ReasoningDelta {
                reasoning: "reason".to_string(),
            },
        })
        .await
        .unwrap();
        tx.send(StreamEvent::MessageStop {
            content: vec![],
            usage: Some(opensquilla_core::types::Usage::new(10, 20)),
        })
        .await
        .unwrap();
        drop(tx);

        let state = stage.consume_channel(rx).await;
        assert!(state.message_stopped);
        assert_eq!(state.final_text(), "Hello world");
        assert_eq!(state.reasoning_text(), "reason");
        assert!(state.error_message.is_none());
        assert!(!state.timed_out);
    }

    #[tokio::test]
    async fn test_consume_channel_timeout() {
        let stage = StreamConsumerStage::new().with_timeout(Duration::from_millis(50));
        let (_tx, rx) = tokio::sync::mpsc::channel(16);
        let state = stage.consume_channel(rx).await;
        assert!(state.timed_out);
        assert_eq!(state.error_code.as_deref(), Some("STREAM_TIMEOUT"));
    }

    #[tokio::test]
    async fn test_consume_channel_error_event() {
        let stage = StreamConsumerStage::new();
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        tx.send(StreamEvent::Error {
            message: "boom".to_string(),
            code: Some("E_BOOM".to_string()),
        })
        .await
        .unwrap();
        drop(tx);
        let state = stage.consume_channel(rx).await;
        assert_eq!(state.error_message.as_deref(), Some("boom"));
        assert_eq!(state.error_code.as_deref(), Some("E_BOOM"));
    }

    #[test]
    fn test_snapshot_extracts_state() {
        let stage = StreamConsumerStage::new();
        let ctx = StageContext {
            turn_id: "t1".into(),
            messages: vec![Message {
                role: MessageRole::Assistant,
                content: vec![
                    ContentBlock::Text("answer".into()),
                    ContentBlock::ToolUse(ToolCall::new("c1", "shell", json!({"cmd": "ls"}))),
                ],
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            }],
            current_model: String::new(),
            current_provider: String::new(),
            usage: opensquilla_core::types::Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
        };
        let state = stage.snapshot(&ctx);
        assert_eq!(state.final_text(), "answer");
        assert!(state.has_tool_calls());
    }
}
