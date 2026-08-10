//! SSE stream assembly.
//!
//! Provides a stateful [`StreamAssembler`] that consumes typed [`SseDelta`]
//! events (text, reasoning, and tool-call fragments) and produces assembled
//! [`ContentBlock`] values. The assembler handles reasoning buffering, tool
//! call argument accumulation, and content block assembly for OpenAI,
//! Anthropic, and DeepSeek delta formats.
//!
//! Providers parse their wire format into [`SseDelta`] values (via
//! [`SseDelta::from_json`] / [`SseDelta::from_json_many`] for the common
//! OpenAI-compatible and Anthropic chunk shapes) and feed them to
//! [`StreamAssembler::push_delta`]. Text deltas are merged into a single
//! content block; reasoning deltas are buffered separately; tool-call
//! argument fragments are accumulated until the arguments form complete JSON,
//! at which point a complete `ContentBlock::ToolUse` is emitted.

use crate::stream::ToolCallBuffer;
use opensquilla_core::types::{ContentBlock, ToolCall};
use serde_json::Value;

/// A single normalized stream delta.
///
/// This is the format-agnostic input event consumed by [`StreamAssembler`].
/// Adapters map their wire protocol (OpenAI Chat Completions, Anthropic
/// Messages, DeepSeek) into these deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseDelta {
    /// A text content delta.
    Text(String),
    /// A reasoning/thinking delta.
    Reasoning(String),
    /// The start of a tool call, carrying its stream-local id and name.
    ToolCallBegin { id: String, name: String },
    /// A partial tool call arguments delta (streamed JSON fragment).
    ToolCallDelta(String),
    /// The end of the current tool call.
    ToolCallEnd,
    /// The stream has completed.
    Done,
}

impl SseDelta {
    /// Parse a raw SSE `data:` payload (a JSON string) into a single delta.
    ///
    /// Returns `None` for chunks that carry no content (e.g. a role-only
    /// delta, `message_start`, or an empty completion chunk).
    pub fn from_json(data: &str) -> Option<SseDelta> {
        let value: Value = serde_json::from_str(data).ok()?;
        Self::from_value(&value)
    }

    /// Parse a raw SSE `data:` payload into a list of deltas.
    ///
    /// A single OpenAI-compatible chunk may carry multiple parallel tool-call
    /// entries; this returns one delta per entry so the caller can route each
    /// index independently. When the chunk carries no tool calls, it falls
    /// back to [`Self::from_value`].
    pub fn from_json_many(data: &str) -> Vec<SseDelta> {
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(delta) = choice.get("delta") {
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let args = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if !id.is_empty() || !name.is_empty() {
                                out.push(SseDelta::ToolCallBegin {
                                    id: id.to_string(),
                                    name: name.to_string(),
                                });
                            } else if !args.is_empty() {
                                out.push(SseDelta::ToolCallDelta(args.to_string()));
                            }
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            if let Some(delta) = Self::from_value(&value) {
                out.push(delta);
            }
        }
        out
    }

    /// Convert an already-parsed JSON chunk into a single delta.
    ///
    /// Handles the OpenAI Chat Completions shape (`choices[0].delta` with
    /// `content`, DeepSeek `reasoning_content`, and `tool_calls`), and the
    /// Anthropic Messages shape (`content_block_start`,
    /// `content_block_delta`, `content_block_stop`, `message_delta`,
    /// `message_stop`).
    pub fn from_value(value: &Value) -> Option<SseDelta> {
        // OpenAI-compatible chat completion chunk.
        if let Some(choices) = value.get("choices").and_then(|v| v.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(delta) = choice.get("delta") {
                    if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                        if !content.is_empty() {
                            return Some(SseDelta::Text(content.to_string()));
                        }
                    }
                    // DeepSeek / OpenAI reasoning fields.
                    for key in [
                        "reasoning_content",
                        "reasoning",
                        "reasoning_text",
                        "thinking",
                        "thinking_content",
                    ] {
                        if let Some(r) = delta.get(key).and_then(|v| v.as_str()) {
                            if !r.is_empty() {
                                return Some(SseDelta::Reasoning(r.to_string()));
                            }
                        }
                    }
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tool_calls {
                            let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let args = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if !id.is_empty() || !name.is_empty() {
                                return Some(SseDelta::ToolCallBegin {
                                    id: id.to_string(),
                                    name: name.to_string(),
                                });
                            }
                            if !args.is_empty() {
                                return Some(SseDelta::ToolCallDelta(args.to_string()));
                            }
                        }
                    }
                }
                if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    if !finish.is_empty() && finish != "null" {
                        return Some(SseDelta::Done);
                    }
                }
            }
        }

        // Anthropic Messages chunk.
        if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
            match event_type {
                "content_block_start" => {
                    if let Some(block) = value.get("content_block") {
                        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match block_type {
                            "text" => {
                                let text = block.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                if !text.is_empty() {
                                    return Some(SseDelta::Text(text.to_string()));
                                }
                            }
                            "tool_use" => {
                                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                if !id.is_empty() || !name.is_empty() {
                                    return Some(SseDelta::ToolCallBegin {
                                        id: id.to_string(),
                                        name: name.to_string(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_delta" => {
                    if let Some(delta) = value.get("delta") {
                        let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match delta_type {
                            "text_delta" => {
                                let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                                if !text.is_empty() {
                                    return Some(SseDelta::Text(text.to_string()));
                                }
                            }
                            "input_json_delta" => {
                                let partial = delta
                                    .get("partial_json")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if !partial.is_empty() {
                                    return Some(SseDelta::ToolCallDelta(partial.to_string()));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                "content_block_stop" => return Some(SseDelta::ToolCallEnd),
                "message_delta" | "message_stop" => return Some(SseDelta::Done),
                _ => {}
            }
        }

        // OpenAI Responses API chunk.
        if let Some(event_type) = value.get("type").and_then(|v| v.as_str()) {
            match event_type {
                "response.output_text.delta" => {
                    let text = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                    if !text.is_empty() {
                        return Some(SseDelta::Text(text.to_string()));
                    }
                }
                "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                    let reasoning = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                    if !reasoning.is_empty() {
                        return Some(SseDelta::Reasoning(reasoning.to_string()));
                    }
                }
                "response.output_item.added" => {
                    if let Some(item) = value.get("item") {
                        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if item_type == "function_call" {
                            let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            if !id.is_empty() || !name.is_empty() {
                                return Some(SseDelta::ToolCallBegin {
                                    id: id.to_string(),
                                    name: name.to_string(),
                                });
                            }
                        }
                    }
                }
                "response.function_call_arguments.delta" => {
                    let delta = value.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                    if !delta.is_empty() {
                        return Some(SseDelta::ToolCallDelta(delta.to_string()));
                    }
                }
                "response.function_call_arguments.done" => return Some(SseDelta::ToolCallEnd),
                "response.completed" => return Some(SseDelta::Done),
                _ => {}
            }
        }
        None
    }
}

/// The current assembly phase of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblerState {
    /// No content is currently being received.
    Idle,
    /// Text content is being accumulated.
    ReceivingContent,
    /// Reasoning content is being accumulated.
    ReceivingReasoning,
    /// A tool call is being assembled.
    ReceivingToolCall {
        /// The stream-local tool call id.
        id: String,
        /// The tool name.
        name: String,
        /// The accumulated raw JSON arguments.
        args: String,
    },
    /// The stream has finished.
    Complete,
}

/// Assembles SSE deltas into content blocks.
///
/// Text deltas are merged into a single `ContentBlock::Text`; reasoning
/// deltas are accumulated into a `ContentBlock::Reasoning`; tool-call
/// argument fragments are buffered (via the shared [`ToolCallBuffer`]) and
/// emitted as a complete `ContentBlock::ToolUse` once the arguments form
/// valid JSON or the call is closed by `SseDelta::ToolCallEnd`.
#[derive(Debug)]
pub struct StreamAssembler {
    buffer: ToolCallBuffer,
    reasoning_buffer: String,
    content_buffer: String,
    tool_calls: Vec<ToolCall>,
    state: AssemblerState,
    finished: bool,
}

impl Default for StreamAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamAssembler {
    /// Create a new stream assembler.
    pub fn new() -> Self {
        Self {
            buffer: ToolCallBuffer::default(),
            reasoning_buffer: String::new(),
            content_buffer: String::new(),
            tool_calls: Vec::new(),
            state: AssemblerState::Idle,
            finished: false,
        }
    }

    /// Process a single SSE event delta, updating internal state and emitting
    /// any complete content blocks.
    pub fn push_delta(&mut self, delta: &SseDelta) -> Vec<ContentBlock> {
        if self.finished {
            return Vec::new();
        }
        let mut blocks = Vec::new();
        match delta {
            SseDelta::Text(text) => {
                blocks.extend(self.flush_current());
                self.content_buffer.push_str(text);
                self.state = AssemblerState::ReceivingContent;
            }
            SseDelta::Reasoning(reasoning) => {
                blocks.extend(self.flush_current());
                self.reasoning_buffer.push_str(reasoning);
                self.state = AssemblerState::ReceivingReasoning;
            }
            SseDelta::ToolCallBegin { id, name } => {
                blocks.extend(self.flush_current());
                self.state = AssemblerState::ReceivingToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    args: String::new(),
                };
            }
            SseDelta::ToolCallDelta(args) => {
                if let AssemblerState::ReceivingToolCall {
                    id,
                    name,
                    args: state_args,
                } = &mut self.state
                {
                    state_args.push_str(args);
                    if !self.tool_calls.iter().any(|tc| tc.id == *id) {
                        if let Some(tc) = self.buffer.accumulate(id, name, args) {
                            self.tool_calls.push(tc.clone());
                            blocks.push(ContentBlock::ToolUse(tc));
                        }
                    }
                }
            }
            SseDelta::ToolCallEnd => {
                if let Some(block) = self.finalize_active_tool_call() {
                    blocks.push(block);
                }
            }
            SseDelta::Done => {
                blocks.extend(self.finalize());
            }
        }
        blocks
    }

    /// Finalize the stream, flushing any remaining buffers.
    pub fn finalize(&mut self) -> Vec<ContentBlock> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut blocks = self.flush_current();
        // Flush any remaining tool-call buffers not already emitted.
        let flushed = self.buffer.flush();
        for tc in flushed {
            if !self.tool_calls.iter().any(|existing| existing.id == tc.id) {
                self.tool_calls.push(tc.clone());
                blocks.push(ContentBlock::ToolUse(tc));
            }
        }
        self.state = AssemblerState::Complete;
        blocks
    }

    /// Return the assembled tool calls.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Flush whatever buffer is currently active into content blocks.
    fn flush_current(&mut self) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        if matches!(self.state, AssemblerState::ReceivingContent) {
            let text = std::mem::take(&mut self.content_buffer);
            if !text.is_empty() {
                blocks.push(ContentBlock::Text { text });
            }
        } else if matches!(self.state, AssemblerState::ReceivingReasoning) {
            let reasoning = std::mem::take(&mut self.reasoning_buffer);
            if !reasoning.is_empty() {
                blocks.push(ContentBlock::Reasoning { reasoning });
            }
        } else if matches!(self.state, AssemblerState::ReceivingToolCall { .. }) {
            if let Some(block) = self.finalize_active_tool_call() {
                blocks.push(block);
            }
        }
        blocks
    }

    /// Finalize the currently active tool call (if any), emitting a complete
    /// `ToolUse` block when its identity is known.
    fn finalize_active_tool_call(&mut self) -> Option<ContentBlock> {
        if !matches!(self.state, AssemblerState::ReceivingToolCall { .. }) {
            return None;
        }
        if let AssemblerState::ReceivingToolCall { id, name, args } =
            std::mem::replace(&mut self.state, AssemblerState::Idle)
        {
            if !self.tool_calls.iter().any(|tc| tc.id == id) {
                let parsed = serde_json::from_str(&args).unwrap_or_else(|_| serde_json::json!({}));
                let tc = ToolCall::new(id, name, parsed);
                if !tc.name.is_empty() {
                    self.tool_calls.push(tc.clone());
                    return Some(ContentBlock::ToolUse(tc));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_block(block: &ContentBlock) -> &str {
        match block {
            ContentBlock::Text { text: ref t } => t,
            other => panic!("expected Text block, got {other:?}"),
        }
    }

    #[test]
    fn test_text_delta_accumulation() {
        let mut asm = StreamAssembler::new();
        assert!(asm.push_delta(&SseDelta::Text("Hello ".into())).is_empty());
        assert!(asm.push_delta(&SseDelta::Text("world".into())).is_empty());
        let blocks = asm.finalize();
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_block(&blocks[0]), "Hello world");
    }

    #[test]
    fn test_reasoning_then_text_flushes_reasoning() {
        let mut asm = StreamAssembler::new();
        assert!(
            asm.push_delta(&SseDelta::Reasoning("think".into()))
                .is_empty()
        );
        let blocks = asm.push_delta(&SseDelta::Text("answer".into()));
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Reasoning { reasoning: ref r } => assert_eq!(r, "think"),
            other => panic!("expected Reasoning block, got {other:?}"),
        }
        let blocks = asm.finalize();
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_block(&blocks[0]), "answer");
    }

    #[test]
    fn test_tool_call_assembly_across_deltas() {
        let mut asm = StreamAssembler::new();
        assert!(
            asm.push_delta(&SseDelta::ToolCallBegin {
                id: "call_1".into(),
                name: "get_weather".into(),
            })
            .is_empty()
        );
        assert!(
            asm.push_delta(&SseDelta::ToolCallDelta(r#"{"loc"#.into()))
                .is_empty()
        );
        let blocks = asm.push_delta(&SseDelta::ToolCallDelta(r#"ation":"NYC"}"#.into()));
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.id, "call_1");
                assert_eq!(tc.name, "get_weather");
                assert_eq!(tc.input["location"], "NYC");
            }
            other => panic!("expected ToolUse block, got {other:?}"),
        }
        assert!(asm.push_delta(&SseDelta::ToolCallEnd).is_empty());
        assert_eq!(asm.tool_calls().len(), 1);
        assert_eq!(asm.tool_calls()[0].input["location"], "NYC");
    }

    #[test]
    fn test_tool_call_completed_by_delta_no_duplicate_on_end() {
        let mut asm = StreamAssembler::new();
        asm.push_delta(&SseDelta::ToolCallBegin {
            id: "c1".into(),
            name: "ping".into(),
        });
        let blocks = asm.push_delta(&SseDelta::ToolCallDelta("{}".into()));
        assert_eq!(blocks.len(), 1); // emitted as soon as JSON completes
        assert!(asm.push_delta(&SseDelta::ToolCallEnd).is_empty());
        assert_eq!(asm.tool_calls().len(), 1);
    }

    #[test]
    fn test_finalize_flushes_pending_buffers() {
        let mut asm = StreamAssembler::new();
        assert!(
            asm.push_delta(&SseDelta::Text("pending ".into()))
                .is_empty()
        );
        let blocks = asm.push_delta(&SseDelta::Reasoning("reason".into()));
        assert_eq!(blocks.len(), 1); // the pending text is flushed
        let blocks = asm.finalize();
        assert_eq!(blocks.len(), 1); // the reasoning buffer is flushed
        match &blocks[0] {
            ContentBlock::Reasoning { reasoning: ref r } => assert_eq!(r, "reason"),
            other => panic!("expected Reasoning block, got {other:?}"),
        }
    }

    #[test]
    fn test_interleaved_content_tool_call_content() {
        let mut asm = StreamAssembler::new();
        assert!(asm.push_delta(&SseDelta::Text("before ".into())).is_empty());
        let blocks = asm.push_delta(&SseDelta::ToolCallBegin {
            id: "t1".into(),
            name: "f".into(),
        });
        assert_eq!(blocks.len(), 1); // pending text is flushed
        assert_eq!(text_block(&blocks[0]), "before ");
        let blocks = asm.push_delta(&SseDelta::ToolCallDelta(r#"{"x":1}"#.into()));
        assert_eq!(blocks.len(), 1); // tool use emitted when JSON completes
        let blocks = asm.push_delta(&SseDelta::Text(" after".into()));
        assert_eq!(blocks.len(), 0); // nothing flushed yet
        let blocks = asm.finalize();
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_block(&blocks[0]), " after");
        assert_eq!(asm.tool_calls().len(), 1);
    }

    #[test]
    fn test_finalize_emits_incomplete_tool_call() {
        let mut asm = StreamAssembler::new();
        asm.push_delta(&SseDelta::ToolCallBegin {
            id: "t1".into(),
            name: "f".into(),
        });
        asm.push_delta(&SseDelta::ToolCallDelta(r#"{"partial"#.into()));
        let blocks = asm.finalize();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse(tc) => {
                assert_eq!(tc.name, "f");
                assert!(tc.input.is_object());
            }
            other => panic!("expected ToolUse block, got {other:?}"),
        }
    }

    #[test]
    fn test_done_finalizes() {
        let mut asm = StreamAssembler::new();
        asm.push_delta(&SseDelta::Text("done".into()));
        let blocks = asm.push_delta(&SseDelta::Done);
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_block(&blocks[0]), "done");
        // Further pushes are ignored after Done.
        assert!(asm.push_delta(&SseDelta::Text("ignored".into())).is_empty());
    }

    #[test]
    fn test_from_json_openai_text() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::Text("Hello".into()))
        );
    }

    #[test]
    fn test_from_json_deepseek_reasoning() {
        let data =
            r#"{"choices":[{"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::Reasoning("thinking".into()))
        );
    }

    #[test]
    fn test_from_json_openai_tool_call_begin() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallBegin {
                id: "call_1".into(),
                name: "get_weather".into(),
            })
        );
    }

    #[test]
    fn test_from_json_openai_tool_call_delta() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc\""}}]},"finish_reason":null}]}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallDelta(r#"{"loc"#.into()))
        );
    }

    #[test]
    fn test_from_json_openai_finish() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::Done));
    }

    #[test]
    fn test_from_json_anthropic_deltas() {
        let data =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::Text("Hi".into())));

        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallBegin {
                id: "toolu_1".into(),
                name: "get_weather".into(),
            })
        );

        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"loc\""}}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallDelta(r#"{"loc"#.into()))
        );

        let data = r#"{"type":"content_block_stop","index":1}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::ToolCallEnd));

        let data = r#"{"type":"message_stop"}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::Done));
    }

    #[test]
    fn test_from_json_openai_responses_api() {
        let data = r#"{"type":"response.output_text.delta","delta":"Hello"}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::Text("Hello".into()))
        );

        let data = r#"{"type":"response.reasoning_summary_text.delta","delta":"think"}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::Reasoning("think".into()))
        );

        let data = r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","name":"get_weather","arguments":""}}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallBegin {
                id: "fc_1".into(),
                name: "get_weather".into(),
            })
        );

        let data = r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"loc\""}"#;
        assert_eq!(
            SseDelta::from_json(data),
            Some(SseDelta::ToolCallDelta(r#"{"loc"#.into()))
        );

        let data = r#"{"type":"response.function_call_arguments.done","item_id":"fc_1"}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::ToolCallEnd));

        let data = r#"{"type":"response.completed"}"#;
        assert_eq!(SseDelta::from_json(data), Some(SseDelta::Done));
    }

    #[test]
    fn test_from_json_many_parallel_tool_calls() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f1","arguments":""}},{"index":1,"id":"b","function":{"name":"f2","arguments":""}}]},"finish_reason":null}]}"#;
        let deltas = SseDelta::from_json_many(data);
        assert_eq!(deltas.len(), 2);
        assert_eq!(
            deltas[0],
            SseDelta::ToolCallBegin {
                id: "a".into(),
                name: "f1".into(),
            }
        );
        assert_eq!(
            deltas[1],
            SseDelta::ToolCallBegin {
                id: "b".into(),
                name: "f2".into(),
            }
        );
    }

    #[test]
    fn test_from_json_role_only_chunk_is_none() {
        let data = r#"{"choices":[{"delta":{"role":"assistant"},"index":0}]}"#;
        assert_eq!(SseDelta::from_json(data), None);
    }
}
