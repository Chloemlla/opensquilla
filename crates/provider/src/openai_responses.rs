//! OpenAI Responses API backend.
//!
//! Unlike the chat completions API (`/v1/chat/completions`), the Responses API
//! (`/v1/responses`) uses an *input* array of items and returns an *output*
//! array of items. Items include message items (with `output_text`, `refusal`,
//! `input_image`, and `input_file` content parts), reasoning items,
//! function-call items, computer-use items, web-search items, file-search
//! items, and code-interpreter items.
//!
//! This backend translates between the canonical [`crate::types::Provider`]
//! interface and the Responses API's item-oriented shape. It supports:
//!
//! * **Three provider kinds** — the hosted OpenAI Responses API plus the
//!   Volcengine and BytePlus Ark coding-plan endpoints, which are
//!   Responses-API compatible ([`OpenAiResponsesProvider`]).
//! * **Conversation threading** — `previous_response_id` for stateless
//!   continuation of a prior response.
//! * **Typed tools** — `function`, `computer_use`, `web_search`,
//!   `file_search`, and `code_interpreter` tools.
//! * **Streaming** — SSE events such as `response.output_text.delta`,
//!   `response.function_call_arguments.delta`, and `response.completed`.
//!
//! The streaming path exposes both a [`StreamEvent`] mapper (used by the
//! [`SseStream`] adapter) and an [`SseDelta`] mapper for the shared
//! [`crate::stream_assembly::StreamAssembler`].

use std::collections::HashMap;

use async_trait::async_trait;
use futures::Stream;
use opensquilla_core::types::{
    ChatMessage, ContentBlock, MessageRole, ToolCall, ToolDefinition, Usage,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::stream::SseStream;
use crate::stream_assembly::SseDelta;
use crate::types::{
    ChatConfig, ChatProvider, Provider, ProviderError, ProviderResponse, ProviderResult,
    StreamEvent,
};

// ---------------------------------------------------------------------------
// Provider kinds and configuration
// ---------------------------------------------------------------------------

/// The Responses-API backend variant.
///
/// This is distinct from [`OpenAIResponsesProvider`] (the concrete provider
/// struct). It selects which hosted endpoint and default model catalog a
/// provider instance targets. All three variants speak the same Responses
/// wire protocol and differ only in base URL and model defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiResponsesProvider {
    /// OpenAI's hosted Responses API (`https://api.openai.com/v1/responses`).
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    /// Volcengine Ark coding-plan Responses-compatible endpoint
    /// (`https://ark.cn-beijing.volces.com/api/v3/responses`).
    #[serde(rename = "volcengine_coding_plan")]
    VolcengineCodingPlan,
    /// BytePlus Ark coding-plan Responses-compatible endpoint
    /// (`https://ark.byteplus.com/api/v3/responses`).
    #[serde(rename = "byteplus_coding_plan")]
    ByteplusCodingPlan,
}

impl OpenAiResponsesProvider {
    /// The canonical registry id for this provider kind.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai_responses",
            Self::VolcengineCodingPlan => "volcengine_coding_plan",
            Self::ByteplusCodingPlan => "byteplus_coding_plan",
        }
    }

    /// Parse a provider kind from its registry id.
    pub fn parse(id: &str) -> Option<Self> {
        match id {
            "openai_responses" => Some(Self::OpenAiResponses),
            "volcengine_coding_plan" => Some(Self::VolcengineCodingPlan),
            "byteplus_coding_plan" => Some(Self::ByteplusCodingPlan),
            _ => None,
        }
    }

    /// The default Responses-API base URL for this provider kind.
    pub const fn default_base_url(&self) -> &'static str {
        match self {
            Self::OpenAiResponses => "https://api.openai.com/v1",
            Self::VolcengineCodingPlan => "https://ark.cn-beijing.volces.com/api/v3",
            Self::ByteplusCodingPlan => "https://ark.byteplus.com/api/v3",
        }
    }

    /// Well-known model ids for this provider kind.
    pub const fn default_models(&self) -> &'static [&'static str] {
        match self {
            Self::OpenAiResponses => &[
                "o1",
                "o1-mini",
                "o3",
                "o3-mini",
                "o4-mini",
                "gpt-4o",
                "gpt-4o-mini",
            ],
            Self::VolcengineCodingPlan => &["doubao-coding"],
            Self::ByteplusCodingPlan => &["doubao-coding"],
        }
    }

    /// The default model used when a request does not specify one.
    pub const fn default_model(&self) -> &'static str {
        match self {
            Self::OpenAiResponses => "o3",
            Self::VolcengineCodingPlan => "doubao-coding",
            Self::ByteplusCodingPlan => "doubao-coding",
        }
    }
}

impl std::fmt::Display for OpenAiResponsesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for a Responses-API backend.
///
/// Carries the provider kind, the API key, and the base URL. The base URL may
/// be left empty to use the provider kind's default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponsesConfig {
    /// The backend variant this configuration targets.
    pub provider: OpenAiResponsesProvider,
    /// The API key used for `Authorization: Bearer`.
    pub api_key: String,
    /// The base URL (e.g. `https://api.openai.com/v1`). When empty, the
    /// provider kind's default is used.
    pub base_url: String,
}

impl OpenAiResponsesConfig {
    /// Create a config for the given provider kind and API key.
    ///
    /// An empty `base_url` selects the kind's default endpoint.
    pub fn new(provider: OpenAiResponsesProvider, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            base_url: String::new(),
        }
    }

    /// Create a config with an explicit base URL override.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Build a config for the hosted OpenAI Responses API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiResponsesProvider::OpenAiResponses, api_key)
    }

    /// Build a config for the Volcengine Ark coding-plan endpoint.
    pub fn volcengine_coding_plan(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiResponsesProvider::VolcengineCodingPlan, api_key)
    }

    /// Build a config for the BytePlus Ark coding-plan endpoint.
    pub fn byteplus_coding_plan(api_key: impl Into<String>) -> Self {
        Self::new(OpenAiResponsesProvider::ByteplusCodingPlan, api_key)
    }

    /// The effective base URL, falling back to the provider kind's default.
    pub fn resolved_base_url(&self) -> &str {
        if self.base_url.trim().is_empty() {
            self.provider.default_base_url()
        } else {
            self.base_url.trim_end_matches('/')
        }
    }
}

// ---------------------------------------------------------------------------
// Responses API input items
// ---------------------------------------------------------------------------

/// A single content part inside a Responses message item.
///
/// Mirrors the Responses API content part types: `input_text`, `output_text`,
/// `input_image`, `input_file`, and `refusal`.
#[derive(Debug, Clone)]
pub enum ContentPart {
    /// Plain text content. `is_assistant` selects `output_text` vs
    /// `input_text`, matching the Responses API's role-symmetric part types.
    Text { text: String, is_assistant: bool },
    /// An inline or URL image. URLs are used as-is; local data must be encoded
    /// as a `data:` URL by the caller.
    InputImage { image_url: String },
    /// An uploaded file part. `file_data` carries the raw bytes and
    /// `file_content_type` the MIME type.
    InputFile {
        filename: String,
        file_data: String,
        file_content_type: String,
    },
    /// A refusal part (assistant output only).
    Refusal { refusal: String },
}

impl ContentPart {
    /// Create a user/system text part (`input_text`).
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            is_assistant: false,
        }
    }

    /// Create an assistant text part (`output_text`).
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            is_assistant: true,
        }
    }

    /// Create an input image part.
    pub fn input_image(image_url: impl Into<String>) -> Self {
        Self::InputImage {
            image_url: image_url.into(),
        }
    }

    /// Create an input file part.
    pub fn input_file(
        filename: impl Into<String>,
        file_data: impl Into<String>,
        file_content_type: impl Into<String>,
    ) -> Self {
        Self::InputFile {
            filename: filename.into(),
            file_data: file_data.into(),
            file_content_type: file_content_type.into(),
        }
    }

    /// Create a refusal part.
    pub fn refusal(text: impl Into<String>) -> Self {
        Self::Refusal {
            refusal: text.into(),
        }
    }

    /// Convert a core [`ContentBlock`] into a content part.
    ///
    /// Tool-use and tool-result blocks are handled at the message-item level
    /// (they become `function_call` / `function_call_output` items), so this
    /// returns `None` for them. Reasoning blocks are never sent upstream.
    pub fn from_block(block: &ContentBlock, is_assistant: bool) -> Option<Self> {
        match block {
            ContentBlock::Text(text) => Some(Self::Text {
                text: text.clone(),
                is_assistant,
            }),
            ContentBlock::Reasoning(_) => None,
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => None,
        }
    }

    /// Serialize this part to its Responses API JSON shape.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Text { text, is_assistant } => serde_json::json!({
                "type": if *is_assistant { "output_text" } else { "input_text" },
                "text": text,
            }),
            Self::InputImage { image_url } => serde_json::json!({
                "type": "input_image",
                "image_url": image_url,
            }),
            Self::InputFile {
                filename,
                file_data,
                file_content_type,
            } => serde_json::json!({
                "type": "input_file",
                "filename": filename,
                "file_data": file_data,
                "file_content_type": file_content_type,
            }),
            Self::Refusal { refusal } => serde_json::json!({
                "type": "refusal",
                "refusal": refusal,
            }),
        }
    }
}

/// A Responses API input item.
///
/// The input array is an ordered list of message items, function-call items,
/// and function-call-output items. Message items carry a role and content
/// parts; function-call and function-call-output items carry tool identity.
#[derive(Debug, Clone)]
pub enum InputItem {
    /// A message item (`type: "message"`).
    Message {
        role: MessageRole,
        content: Vec<ContentPart>,
    },
    /// A function-call item (`type: "function_call"`).
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    /// A function-call-output item (`type: "function_call_output"`).
    FunctionCallOutput { call_id: String, output: String },
}

impl InputItem {
    /// Create a message item with the given role and content parts.
    pub fn message(role: MessageRole, content: Vec<ContentPart>) -> Self {
        Self::Message { role, content }
    }

    /// Create a text-only message item.
    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self::Message {
            role,
            content: vec![ContentPart::text(text)],
        }
    }

    /// Create a function-call item.
    pub fn function_call(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::FunctionCall {
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Create a function-call-output item.
    pub fn function_call_output(call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self::FunctionCallOutput {
            call_id: call_id.into(),
            output: output.into(),
        }
    }

    /// The Responses API `type` discriminator for this item.
    pub fn item_type(&self) -> &'static str {
        match self {
            Self::Message { .. } => "message",
            Self::FunctionCall { .. } => "function_call",
            Self::FunctionCallOutput { .. } => "function_call_output",
        }
    }

    /// Serialize this item to its Responses API JSON shape.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Message { role, content } => serde_json::json!({
                "type": "message",
                "role": role_str(role),
                "content": content.iter().map(ContentPart::to_json).collect::<Vec<_>>(),
            }),
            Self::FunctionCall {
                call_id,
                name,
                arguments,
            } => serde_json::json!({
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }),
            Self::FunctionCallOutput { call_id, output } => serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }),
        }
    }
}

/// Convert a [`MessageRole`] to its Responses API wire role string.
pub fn role_str(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Flatten a list of canonical messages into Responses API input items.
///
/// Text blocks are accumulated into message items; tool-use and tool-result
/// blocks become `function_call` / `function_call_output` items, flushing any
/// pending text first (mirroring the Python `_responses_input`). Reasoning
/// blocks are dropped.
pub fn build_responses_input_items(messages: &[ChatMessage]) -> Vec<InputItem> {
    let mut items: Vec<InputItem> = Vec::new();
    for msg in messages {
        let mut pending: Vec<ContentPart> = Vec::new();
        let is_assistant = msg.role == MessageRole::Assistant;
        for block in &msg.content {
            match block {
                ContentBlock::Text(text) => pending.push(ContentPart::Text {
                    text: text.clone(),
                    is_assistant,
                }),
                ContentBlock::Reasoning(_) => {}
                ContentBlock::ToolUse(tc) => {
                    if !pending.is_empty() {
                        items.push(InputItem::Message {
                            role: msg.role.clone(),
                            content: std::mem::take(&mut pending),
                        });
                    }
                    let arguments =
                        serde_json::to_string(&tc.input).unwrap_or_else(|_| "{}".to_string());
                    items.push(InputItem::FunctionCall {
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments,
                    });
                }
                ContentBlock::ToolResult(tr) => {
                    if !pending.is_empty() {
                        items.push(InputItem::Message {
                            role: msg.role.clone(),
                            content: std::mem::take(&mut pending),
                        });
                    }
                    items.push(InputItem::FunctionCallOutput {
                        call_id: tr.tool_use_id.clone(),
                        output: tr.content.clone(),
                    });
                }
            }
        }
        if !pending.is_empty() {
            items.push(InputItem::Message {
                role: msg.role.clone(),
                content: pending,
            });
        }
    }
    items
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// The tool types the Responses API recognizes.
///
/// Function tools carry a name, description, and JSON Schema parameters;
/// built-in tools (`computer_use`, `web_search`, `file_search`,
/// `code_interpreter`) are enabled by type alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesToolType {
    /// A user-defined function tool.
    Function,
    /// The `computer_use` tool for GUI automation.
    ComputerUse,
    /// The `web_search` tool for web retrieval.
    WebSearch,
    /// The `file_search` tool for workspace search.
    FileSearch,
    /// The `code_interpreter` tool for code execution.
    CodeInterpreter,
}

impl ResponsesToolType {
    /// The Responses API wire name for this tool type.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::ComputerUse => "computer_use",
            Self::WebSearch => "web_search",
            Self::FileSearch => "file_search",
            Self::CodeInterpreter => "code_interpreter",
        }
    }

    /// Parse a tool type from its wire name.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "function" => Some(Self::Function),
            "computer_use" => Some(Self::ComputerUse),
            "web_search" => Some(Self::WebSearch),
            "file_search" => Some(Self::FileSearch),
            "code_interpreter" => Some(Self::CodeInterpreter),
            _ => None,
        }
    }

    /// Whether this is a built-in (non-function) tool type.
    pub const fn is_builtin(&self) -> bool {
        !matches!(self, Self::Function)
    }
}

/// A Responses API tool definition.
///
/// Function tools carry full schema metadata; built-in tools are identified by
/// type alone and ignore `name`, `description`, and `parameters`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesTool {
    /// The tool type.
    pub tool_type: ResponsesToolType,
    /// The function name (function tools only).
    pub name: String,
    /// The function description (function tools only).
    pub description: String,
    /// The JSON Schema for the function parameters (function tools only).
    pub parameters: serde_json::Value,
    /// Whether the schema should be enforced strictly (function tools only).
    #[serde(default)]
    pub strict: bool,
}

impl ResponsesTool {
    /// Create a function tool with the given schema.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: ResponsesToolType::Function,
            name: name.into(),
            description: description.into(),
            parameters,
            strict: false,
        }
    }

    /// Create a built-in tool (no schema metadata).
    pub fn builtin(tool_type: ResponsesToolType) -> Self {
        Self {
            tool_type,
            name: String::new(),
            description: String::new(),
            parameters: serde_json::Value::Object(Default::default()),
            strict: false,
        }
    }

    /// Build a tool from a canonical [`ToolDefinition`].
    ///
    /// If the tool name matches a built-in tool type (`computer_use`,
    /// `web_search`, `file_search`, `code_interpreter`) it is emitted as that
    /// built-in; otherwise it is emitted as a function tool.
    pub fn from_tool_definition(tool: &ToolDefinition) -> Self {
        if let Some(builtin) = ResponsesToolType::parse(&tool.name) {
            if builtin.is_builtin() {
                return Self::builtin(builtin);
            }
        }
        Self::function(&tool.name, &tool.description, tool.input_schema.clone())
    }

    /// Serialize this tool to its Responses API JSON shape.
    pub fn to_json(&self) -> serde_json::Value {
        let mut json = serde_json::json!({ "type": self.tool_type.as_str() });
        if let Some(obj) = json.as_object_mut() {
            if self.tool_type == ResponsesToolType::Function {
                obj.insert("name".into(), serde_json::json!(self.name));
                obj.insert("description".into(), serde_json::json!(self.description));
                obj.insert("strict".into(), serde_json::json!(self.strict));
                obj.insert(
                    "parameters".into(),
                    if self.parameters.is_null() {
                        serde_json::json!({"type": "object"})
                    } else {
                        self.parameters.clone()
                    },
                );
            }
            // Built-in tools are emitted by type alone; per-tool options can be
            // injected through `config.extra` at request build time.
        }
        json
    }
}

// ---------------------------------------------------------------------------
// Responses API output items
// ---------------------------------------------------------------------------

/// Token usage reported by a Responses response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponsesUsage {
    /// Input (prompt) tokens.
    pub input_tokens: u64,
    /// Output (completion) tokens.
    pub output_tokens: u64,
    /// Reasoning tokens, from `output_tokens_details.reasoning_tokens`.
    pub reasoning_tokens: u64,
    /// Cached input tokens, from `input_tokens_details.cached_tokens`.
    pub cached_tokens: u64,
}

impl ResponsesUsage {
    /// Parse usage from a Responses `usage` object.
    pub fn from_json(usage: &serde_json::Value) -> Self {
        let input_tokens = usage
            .get("input_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let output_tokens = usage
            .get("output_tokens")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let reasoning_tokens = usage
            .get("output_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let cached_tokens = usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        Self {
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
        }
    }

    /// Convert to the canonical [`Usage`] (drops reasoning/cached detail).
    pub fn to_usage(&self) -> Usage {
        Usage::new(self.input_tokens, self.output_tokens)
    }
}

/// A parsed item from a Responses `output` array.
#[derive(Debug, Clone)]
pub enum ResponsesOutputItem {
    /// A message item with rendered text and refusals.
    Message {
        role: String,
        text: Vec<String>,
        refusals: Vec<String>,
    },
    /// A reasoning item with a summary and full reasoning content.
    Reasoning {
        summary: String,
        content: Vec<String>,
    },
    /// A function-call item.
    FunctionCall {
        call_id: String,
        item_id: String,
        name: String,
        arguments: String,
    },
    /// A computer-use item (`type: "computer_call"`).
    ComputerCall {
        call_id: String,
        action: serde_json::Value,
    },
    /// A web-search item (`type: "web_search_call"`).
    WebSearchCall { id: String },
    /// A file-search item (`type: "file_search_call"`).
    FileSearchCall { id: String },
    /// A code-interpreter item (`type: "code_interpreter_call"`).
    CodeInterpreterCall { id: String, input: String },
    /// An unrecognized output item type.
    Unknown { item_type: String },
}

impl ResponsesOutputItem {
    /// Parse a raw output item into a typed value.
    pub fn parse(item: &serde_json::Value) -> Self {
        let item_type = item
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        match item_type {
            "message" => {
                let role = item
                    .get("role")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("assistant")
                    .to_string();
                let mut text = Vec::new();
                let mut refusals = Vec::new();
                if let Some(content) = item.get("content").and_then(serde_json::Value::as_array) {
                    for part in content {
                        match part
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("")
                        {
                            "output_text" | "input_text" => {
                                if let Some(t) =
                                    part.get("text").and_then(serde_json::Value::as_str)
                                {
                                    text.push(t.to_string());
                                }
                            }
                            "refusal" => {
                                if let Some(r) =
                                    part.get("refusal").and_then(serde_json::Value::as_str)
                                {
                                    refusals.push(r.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Self::Message {
                    role,
                    text,
                    refusals,
                }
            }
            "reasoning" => {
                let mut summary = String::new();
                if let Some(summaries) = item.get("summary").and_then(serde_json::Value::as_array) {
                    for s in summaries {
                        if let Some(t) = s.get("text").and_then(serde_json::Value::as_str) {
                            summary.push_str(t);
                        }
                    }
                }
                let mut content = Vec::new();
                if let Some(c) = item.get("content").and_then(serde_json::Value::as_array) {
                    for part in c {
                        if let Some(t) = part.get("text").and_then(serde_json::Value::as_str) {
                            content.push(t.to_string());
                        }
                    }
                }
                Self::Reasoning { summary, content }
            }
            "function_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let item_id = item
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let arguments = item
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
                Self::FunctionCall {
                    call_id,
                    item_id,
                    name,
                    arguments,
                }
            }
            "computer_call" => {
                let call_id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let action = item
                    .get("action")
                    .cloned()
                    .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
                Self::ComputerCall { call_id, action }
            }
            "web_search_call" => {
                let id = item
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Self::WebSearchCall { id }
            }
            "file_search_call" => {
                let id = item
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Self::FileSearchCall { id }
            }
            "code_interpreter_call" => {
                let id = item
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let input = item
                    .get("input")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                Self::CodeInterpreterCall { id, input }
            }
            other => Self::Unknown {
                item_type: other.to_string(),
            },
        }
    }
}

/// A fully assembled Responses request body.
///
/// This builder mirrors the wire payload produced by
/// [`OpenAIResponsesProvider::build_request_body`] but exposes each field
/// explicitly, which makes request construction testable and lets callers
/// assemble advanced requests (computer-use, JSON schema output, threading)
/// without a provider instance.
#[derive(Debug, Clone)]
pub struct ResponsesRequest {
    /// The model identifier.
    pub model: String,
    /// The input items (messages, function calls, function-call outputs).
    pub input: Vec<InputItem>,
    /// System instructions sent as the top-level `instructions` field.
    pub instructions: Option<String>,
    /// Tool definitions.
    pub tools: Vec<ResponsesTool>,
    /// Tool-choice policy (`"auto"`, `"none"`, `"required"`, or a
    /// `{"type": "function", "name": ...}` object). Defaults to `"auto"`.
    pub tool_choice: serde_json::Value,
    /// Optional `previous_response_id` for conversation threading.
    pub previous_response_id: Option<String>,
    /// Whether to store the response server-side (`store: false` by default).
    pub store: bool,
    /// Maximum output tokens (`max_output_tokens`).
    pub max_output_tokens: u32,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Whether the request is streamed.
    pub stream: bool,
    /// JSON Schema output (`text.format`) when structured output is requested.
    pub output_schema: Option<OutputSchema>,
    /// Extra top-level fields passed through verbatim.
    pub extra: HashMap<String, serde_json::Value>,
}

/// Structured output configuration for a Responses request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSchema {
    /// The JSON Schema used to validate the structured output.
    pub schema: serde_json::Value,
    /// Whether the schema is enforced strictly.
    pub strict: bool,
    /// Optional named schema entry (defaults to `"structured_output"`).
    pub name: Option<String>,
}

impl Default for ResponsesRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            input: Vec::new(),
            instructions: None,
            tools: Vec::new(),
            tool_choice: serde_json::json!("auto"),
            previous_response_id: None,
            store: false,
            max_output_tokens: 4096,
            temperature: None,
            top_p: None,
            stop: Vec::new(),
            stream: false,
            output_schema: None,
            extra: HashMap::new(),
        }
    }
}

impl ResponsesRequest {
    /// Build a request from a [`ChatConfig`] and messages.
    pub fn from_config(
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Self {
        let input = build_responses_input_items(messages);
        let instructions = {
            let system = messages
                .iter()
                .filter(|m| m.role == MessageRole::System)
                .map(|m| m.text_content())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            if system.is_empty() {
                None
            } else {
                Some(system)
            }
        };
        let tool_choice = config
            .extra
            .get("tool_choice")
            .cloned()
            .unwrap_or_else(|| serde_json::json!("auto"));
        let previous_response_id = config
            .extra
            .get("previous_response_id")
            .and_then(serde_json::Value::as_str)
            .map(String::from);
        let output_schema = config
            .extra
            .get("output_json_schema")
            .cloned()
            .map(|schema| OutputSchema {
                schema,
                strict: config
                    .extra
                    .get("output_json_schema_strict")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false),
                name: config
                    .extra
                    .get("output_json_schema_name")
                    .and_then(serde_json::Value::as_str)
                    .map(String::from),
            });
        let mut extra = config.extra.clone();
        extra.remove("tool_choice");
        extra.remove("previous_response_id");
        extra.remove("output_json_schema");
        extra.remove("output_json_schema_strict");
        extra.remove("output_json_schema_name");

        Self {
            model: config.model.clone(),
            input,
            instructions,
            tools: tools
                .iter()
                .map(ResponsesTool::from_tool_definition)
                .collect(),
            tool_choice,
            previous_response_id,
            store: false,
            max_output_tokens: config.max_tokens,
            temperature: (config.temperature != 0.0).then_some(config.temperature),
            top_p: (config.top_p != 0.0).then_some(config.top_p),
            stop: config.stop_sequences.clone(),
            stream,
            output_schema,
            extra,
        }
    }

    /// Serialize this request to its Responses API JSON payload.
    pub fn to_json(&self) -> serde_json::Value {
        let input: Vec<serde_json::Value> = self.input.iter().map(InputItem::to_json).collect();
        let mut body = serde_json::json!({
            "model": self.model,
            "input": input,
            "store": self.store,
            "stream": self.stream,
        });
        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "max_output_tokens".into(),
                serde_json::json!(self.max_output_tokens),
            );
            if let Some(instructions) = &self.instructions {
                obj.insert("instructions".into(), serde_json::json!(instructions));
            }
            if !self.tools.is_empty() {
                obj.insert(
                    "tools".into(),
                    serde_json::json!(
                        self.tools
                            .iter()
                            .map(ResponsesTool::to_json)
                            .collect::<Vec<_>>()
                    ),
                );
                obj.insert("tool_choice".into(), self.tool_choice.clone());
            }
            if let Some(prev) = &self.previous_response_id {
                obj.insert("previous_response_id".into(), serde_json::json!(prev));
            }
            if let Some(t) = self.temperature {
                obj.insert("temperature".into(), serde_json::json!(t));
            }
            if let Some(p) = self.top_p {
                obj.insert("top_p".into(), serde_json::json!(p));
            }
            if !self.stop.is_empty() {
                obj.insert("stop".into(), serde_json::json!(self.stop));
            }
            if let Some(schema) = &self.output_schema {
                let name = schema
                    .name
                    .clone()
                    .unwrap_or_else(|| "structured_output".into());
                obj.insert(
                    "text".into(),
                    serde_json::json!({
                        "format": {
                            "type": "json_schema",
                            "name": name,
                            "strict": schema.strict,
                            "schema": schema.schema,
                        }
                    }),
                );
            }
            for (k, v) in &self.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }
}

// ---------------------------------------------------------------------------
// The concrete provider
// ---------------------------------------------------------------------------

/// Provider for the OpenAI Responses API (`/v1/responses`).
///
/// This struct implements the canonical [`Provider`] trait and the
/// [`ChatProvider`] streaming entrypoint. It also exposes the Responses
/// building blocks ([`build_responses_input_items`],
/// [`Self::build_request_body`], [`Self::parse_response`]) so callers can
/// assemble item-level requests directly.
pub struct OpenAIResponsesProvider {
    name: String,
    api_base: String,
    api_key: String,
    provider_kind: OpenAiResponsesProvider,
    org_id: Option<String>,
    default_model: String,
    client: Client,
}

impl std::fmt::Debug for OpenAIResponsesProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesProvider")
            .field("name", &self.name)
            .field("api_base", &self.api_base)
            .field("api_key", &"[redacted]")
            .field("provider_kind", &self.provider_kind)
            .field("org_id", &self.org_id)
            .field("default_model", &self.default_model)
            .finish_non_exhaustive()
    }
}

impl OpenAIResponsesProvider {
    /// Create a new Responses-API provider targeting the OpenAI endpoint.
    ///
    /// * `name` – A label for this provider instance (e.g.
    ///   `"openai_responses"`).
    /// * `api_base` – The base URL (e.g. `"https://api.openai.com/v1"`).
    /// * `api_key` – The API key for `Authorization: Bearer`.
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let provider_kind = OpenAiResponsesProvider::parse(&name)
            .unwrap_or(OpenAiResponsesProvider::OpenAiResponses);
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .expect("Failed to create reqwest Client");
        Self {
            name,
            api_base: api_base.into(),
            api_key: api_key.into(),
            provider_kind,
            org_id: None,
            default_model: provider_kind.default_model().to_string(),
            client,
        }
    }

    /// Create a provider from a [`OpenAiResponsesConfig`].
    ///
    /// The provider kind drives the default base URL and model when the config
    /// does not override them.
    pub fn from_config(config: OpenAiResponsesConfig) -> Self {
        let base_url = config.resolved_base_url().to_string();
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .expect("Failed to create reqwest Client");
        Self {
            name: config.provider.as_str().to_string(),
            api_base: base_url,
            api_key: config.api_key,
            provider_kind: config.provider,
            org_id: None,
            default_model: config.provider.default_model().to_string(),
            client,
        }
    }

    /// Set the `OpenAI-Organization` header for this provider.
    pub fn with_org_id(mut self, org_id: impl Into<String>) -> Self {
        self.org_id = Some(org_id.into());
        self
    }

    /// Override the default model used when a request omits one.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// The backend variant this provider targets.
    pub fn provider_kind(&self) -> OpenAiResponsesProvider {
        self.provider_kind
    }

    /// The configured API base URL.
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    /// The API key used for authentication.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// The default model for this provider.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// The effective request model, preferring the config's model.
    #[allow(dead_code)]
    fn effective_model<'a>(&'a self, config: &'a ChatConfig) -> &'a str {
        if config.model.is_empty() {
            &self.default_model
        } else {
            &config.model
        }
    }

    /// The full Responses endpoint URL.
    pub fn responses_url(&self) -> String {
        format!("{}/responses", self.api_base.trim_end_matches('/'))
    }

    /// Build the Responses API request body from canonical messages.
    ///
    /// Convenience wrapper around [`ResponsesRequest::from_config`] that
    /// returns the serialized payload directly.
    pub fn build_request_body(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut request = ResponsesRequest::from_config(config, messages, tools, stream);
        if request.model.is_empty() {
            request.model = self.default_model.clone();
        }
        request.to_json()
    }

    /// Parse a non-streaming Responses API output into a [`ProviderResponse`].
    pub fn parse_response(&self, data: &serde_json::Value, model: &str) -> ProviderResponse {
        let output_items = self.parse_output_items(data);

        let mut content: Vec<ContentBlock> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut reasoning_text = String::new();

        for item in &output_items {
            match item {
                ResponsesOutputItem::Message { text, refusals, .. } => {
                    for t in text {
                        if !t.is_empty() {
                            content.push(ContentBlock::Text(t.clone()));
                        }
                    }
                    for r in refusals {
                        if !r.is_empty() {
                            content.push(ContentBlock::Text(r.clone()));
                        }
                    }
                }
                ResponsesOutputItem::Reasoning {
                    summary,
                    content: rc,
                } => {
                    reasoning_text.push_str(summary);
                    for t in rc {
                        reasoning_text.push_str(t);
                    }
                }
                ResponsesOutputItem::FunctionCall {
                    call_id,
                    item_id,
                    name,
                    arguments,
                } => {
                    let id = if call_id.is_empty() { item_id } else { call_id };
                    let args: serde_json::Value = serde_json::from_str(arguments)
                        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                    if !name.is_empty() {
                        tool_calls.push(ToolCall::new(id.clone(), name.clone(), args));
                    }
                }
                ResponsesOutputItem::ComputerCall { call_id, action } => {
                    tool_calls.push(ToolCall::new(
                        call_id.clone(),
                        "computer_use",
                        action.clone(),
                    ));
                }
                ResponsesOutputItem::WebSearchCall { id } => {
                    tool_calls.push(ToolCall::new(
                        id.clone(),
                        "web_search",
                        serde_json::json!({}),
                    ));
                }
                ResponsesOutputItem::FileSearchCall { id } => {
                    tool_calls.push(ToolCall::new(
                        id.clone(),
                        "file_search",
                        serde_json::json!({}),
                    ));
                }
                ResponsesOutputItem::CodeInterpreterCall { id, input } => {
                    tool_calls.push(ToolCall::new(
                        id.clone(),
                        "code_interpreter",
                        serde_json::json!({ "input": input }),
                    ));
                }
                ResponsesOutputItem::Unknown { .. } => {}
            }
        }

        let mut msg = ChatMessage {
            role: MessageRole::Assistant,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        if !tool_calls.is_empty() {
            msg.tool_calls = Some(tool_calls);
        }
        if !reasoning_text.is_empty() {
            msg.content
                .insert(0, ContentBlock::Reasoning(reasoning_text));
        }

        let usage_data = &data["usage"];
        let usage = Usage::new(
            usage_data["input_tokens"].as_u64().unwrap_or(0),
            usage_data["output_tokens"].as_u64().unwrap_or(0),
        );

        let stop_reason = self.resolve_stop_reason(data, &msg);

        ProviderResponse {
            content: vec![msg],
            usage,
            model: model.to_string(),
            stop_reason,
        }
    }

    /// Parse a Responses `output` array into typed output items.
    pub fn parse_output_items(&self, data: &serde_json::Value) -> Vec<ResponsesOutputItem> {
        let mut items = Vec::new();
        if let Some(output) = data.get("output").and_then(serde_json::Value::as_array) {
            for item in output {
                items.push(ResponsesOutputItem::parse(item));
            }
        }
        items
    }

    /// Parse the `usage` object of a Responses response.
    pub fn parse_usage(&self, data: &serde_json::Value) -> ResponsesUsage {
        ResponsesUsage::from_json(&data["usage"])
    }

    /// Resolve the canonical stop reason from a Responses response.
    fn resolve_stop_reason(&self, data: &serde_json::Value, msg: &ChatMessage) -> Option<String> {
        let status = data.get("status").and_then(serde_json::Value::as_str);
        let has_tool = msg
            .tool_calls
            .as_ref()
            .map(|tc| !tc.is_empty())
            .unwrap_or(false);
        match status {
            Some("completed") => Some(if has_tool {
                "tool_use".to_string()
            } else {
                "end_turn".to_string()
            }),
            Some("incomplete") => {
                let reason = data
                    .get("incomplete_details")
                    .and_then(|d| d.get("reason"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("incomplete");
                Some(if reason == "max_output_tokens" {
                    "length".to_string()
                } else {
                    "incomplete".to_string()
                })
            }
            Some("failed") => Some("failed".to_string()),
            Some("cancelled") => Some("cancelled".to_string()),
            Some(other) => Some(other.to_string()),
            None => data
                .get("stop_reason")
                .and_then(serde_json::Value::as_str)
                .map(String::from),
        }
    }

    /// Send the request body to the Responses endpoint and normalize errors.
    async fn post(&self, body: &serde_json::Value) -> ProviderResult<reqwest::Response> {
        let mut request = self
            .client
            .post(self.responses_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(body);
        if let Some(org) = &self.org_id {
            request = request.header("OpenAI-Organization", org);
        }
        let resp = request.send().await.map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProviderError::Auth(error_text));
            }
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Err(ProviderError::RateLimited(error_text));
            }
            return Err(ProviderError::Provider(format!(
                "HTTP {status}: {error_text}"
            )));
        }
        Ok(resp)
    }

    /// List models available from the provider's `/models` endpoint.
    ///
    /// Degrades to the static well-known model list on any transport or auth
    /// failure (mirroring the Python `list_models` contract).
    pub async fn list_models(&self) -> Vec<String> {
        let url = format!("{}/models", self.api_base.trim_end_matches('/'));
        let resp = match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => resp,
            _ => {
                return self
                    .provider_kind
                    .default_models()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
            }
        };
        match resp.json::<serde_json::Value>().await {
            Ok(data) => {
                let mut models = Vec::new();
                if let Some(rows) = data.get("data").and_then(serde_json::Value::as_array) {
                    for row in rows {
                        if let Some(id) = row.get("id").and_then(serde_json::Value::as_str) {
                            models.push(id.to_string());
                        }
                    }
                }
                if models.is_empty() {
                    self.provider_kind
                        .default_models()
                        .iter()
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    models
                }
            }
            Err(_) => self
                .provider_kind
                .default_models()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

#[async_trait]
impl Provider for OpenAIResponsesProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        self.provider_kind
            .default_models()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let mut request = ResponsesRequest::from_config(config, messages, tools, false);
        if request.model.is_empty() {
            request.model = self.default_model.clone();
        }
        let model = request.model.clone();
        let body = request.to_json();

        debug!(
            target = "provider",
            provider = %self.name,
            model = %model,
            input_items = request.input.len(),
            "Sending Responses API request"
        );

        let resp = self.post(&body).await?;
        let data: serde_json::Value = resp.json().await.map_err(ProviderError::Network)?;

        info!(
            target = "provider",
            provider = %self.name,
            model = %model,
            "Responses API response received"
        );

        Ok(self.parse_response(&data, &model))
    }

    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.chat(config, messages, tools).await
    }
}

#[async_trait]
impl ChatProvider for OpenAIResponsesProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let mut request = ResponsesRequest::from_config(config, messages, tools, true);
        if request.model.is_empty() {
            request.model = self.default_model.clone();
        }
        let body = request.to_json();

        debug!(
            target = "provider",
            provider = %self.name,
            model = %request.model,
            input_items = request.input.len(),
            "Sending Responses API streaming request"
        );

        let resp = self.post(&body).await?;
        let stream = SseStream::new(resp, parse_responses_sse_event);
        Ok(Box::new(stream))
    }
}

// ---------------------------------------------------------------------------
// Streaming SSE parsing
// ---------------------------------------------------------------------------

/// Parse a Responses API SSE `data:` line into a [`StreamEvent`].
///
/// The Responses API streaming protocol emits typed events:
/// `response.output_text.delta`, `response.reasoning.delta`,
/// `response.reasoning_summary_text.delta`, `response.reasoning_text.delta`,
/// `response.output_item.added`, `response.function_call_arguments.delta`,
/// `response.output_item.done`, `response.completed`, `response.failed`,
/// `response.incomplete`, and `response.cancelled`.
pub fn parse_responses_sse_event(data: &str) -> Option<ProviderResult<StreamEvent>> {
    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse Responses SSE JSON: {e}");
            return None;
        }
    };
    parse_responses_sse_value(&value)
}

/// Parse an already-deserialized Responses SSE event value.
fn parse_responses_sse_value(value: &serde_json::Value) -> Option<ProviderResult<StreamEvent>> {
    let event_type = value.get("type").and_then(serde_json::Value::as_str)?;

    match event_type {
        "response.output_text.delta" => {
            let text = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if text.is_empty() {
                None
            } else {
                Some(Ok(StreamEvent::Text {
                    text: text.to_string(),
                }))
            }
        }
        "response.reasoning.delta"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_text.delta" => {
            let reasoning = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if reasoning.is_empty() {
                None
            } else {
                Some(Ok(StreamEvent::Reasoning {
                    reasoning: reasoning.to_string(),
                }))
            }
        }
        "response.output_item.added" => {
            let item = value.get("item")?;
            parse_output_item_start(item)
        }
        "response.function_call_arguments.delta" => {
            let id = value
                .get("item_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let name = value
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let args = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments: args,
            }))
        }
        "response.output_item.done" => {
            let item = value.get("item")?;
            if item.get("type").and_then(serde_json::Value::as_str) == Some("function_call") {
                let id = item
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| item.get("id").and_then(serde_json::Value::as_str))
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let args = item
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("{}")
                    .to_string();
                Some(Ok(StreamEvent::ToolCall {
                    id,
                    name,
                    arguments: args,
                }))
            } else {
                None
            }
        }
        "response.completed" | "response.done" => {
            let response = value
                .get("response")
                .cloned()
                .unwrap_or_else(|| value.clone());
            let usage = ResponsesUsage::from_json(&response["usage"]);
            let stop_reason = response
                .get("status")
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    response
                        .get("stop_reason")
                        .and_then(serde_json::Value::as_str)
                })
                .map(String::from);
            Some(Ok(StreamEvent::Done {
                usage: Some(usage.to_usage()),
                stop_reason,
            }))
        }
        "response.failed" | "response.incomplete" | "response.cancelled" | "error" => {
            let error = value
                .get("error")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
            let msg = error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| value.get("message").and_then(serde_json::Value::as_str))
                .unwrap_or("Responses API error")
                .to_string();
            Some(Ok(StreamEvent::Error { message: msg }))
        }
        _ => None,
    }
}

/// Map the `item` of an `response.output_item.added` event to a tool-call
/// start event, covering function, computer, web-search, file-search, and
/// code-interpreter items.
fn parse_output_item_start(item: &serde_json::Value) -> Option<ProviderResult<StreamEvent>> {
    let item_type = item
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    match item_type {
        "function_call" => {
            let id = item
                .get("call_id")
                .and_then(serde_json::Value::as_str)
                .or_else(|| item.get("id").and_then(serde_json::Value::as_str))
                .unwrap_or("")
                .to_string();
            let name = item
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments: String::new(),
            }))
        }
        "computer_call" => {
            let id = item
                .get("call_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let action = item
                .get("action")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
            Some(Ok(StreamEvent::ToolCall {
                id,
                name: "computer_use".to_string(),
                arguments: action.to_string(),
            }))
        }
        "web_search_call" => {
            let id = item
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Ok(StreamEvent::ToolCall {
                id,
                name: "web_search".to_string(),
                arguments: String::new(),
            }))
        }
        "file_search_call" => {
            let id = item
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Ok(StreamEvent::ToolCall {
                id,
                name: "file_search".to_string(),
                arguments: String::new(),
            }))
        }
        "code_interpreter_call" => {
            let id = item
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let input = item
                .get("input")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            Some(Ok(StreamEvent::ToolCall {
                id,
                name: "code_interpreter".to_string(),
                arguments: input,
            }))
        }
        _ => None,
    }
}

/// Map a Responses API SSE data line to an [`SseDelta`] for the shared
/// [`crate::stream_assembly::StreamAssembler`].
///
/// This complements [`SseDelta::from_json`], which already recognizes the
/// core Responses events; this function additionally covers the newer
/// built-in tool item events (computer, web-search, file-search,
/// code-interpreter).
pub fn responses_sse_to_delta(data: &str) -> Option<SseDelta> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    responses_sse_value_to_delta(&value)
}

/// Map an already-deserialized Responses SSE event to an [`SseDelta`].
pub fn responses_sse_value_to_delta(value: &serde_json::Value) -> Option<SseDelta> {
    let event_type = value.get("type").and_then(serde_json::Value::as_str)?;
    match event_type {
        "response.output_text.delta" => {
            let text = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if text.is_empty() {
                None
            } else {
                Some(SseDelta::Text(text.to_string()))
            }
        }
        "response.reasoning.delta"
        | "response.reasoning_summary_text.delta"
        | "response.reasoning_text.delta" => {
            let reasoning = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if reasoning.is_empty() {
                None
            } else {
                Some(SseDelta::Reasoning(reasoning.to_string()))
            }
        }
        "response.output_item.added" => {
            let item = value.get("item")?;
            let item_type = item
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            match item_type {
                "function_call"
                | "computer_call"
                | "web_search_call"
                | "file_search_call"
                | "code_interpreter_call" => {
                    let id = item
                        .get("call_id")
                        .and_then(serde_json::Value::as_str)
                        .or_else(|| item.get("id").and_then(serde_json::Value::as_str))
                        .unwrap_or("")
                        .to_string();
                    let name = item
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_else(|| {
                            if item_type == "function_call" {
                                ""
                            } else {
                                item_type.trim_end_matches("_call")
                            }
                        })
                        .to_string();
                    Some(SseDelta::ToolCallBegin { id, name })
                }
                _ => None,
            }
        }
        "response.function_call_arguments.delta" => {
            let delta = value
                .get("delta")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if delta.is_empty() {
                None
            } else {
                Some(SseDelta::ToolCallDelta(delta.to_string()))
            }
        }
        "response.function_call_arguments.done" | "response.output_item.done" => {
            Some(SseDelta::ToolCallEnd)
        }
        "response.completed" | "response.done" => Some(SseDelta::Done),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> OpenAIResponsesProvider {
        OpenAIResponsesProvider::new("openai_responses", "https://api.openai.com/v1", "sk-test")
    }

    // -----------------------------------------------------------------------
    // Input item building
    // -----------------------------------------------------------------------

    #[test]
    fn test_role_str_maps_all_roles() {
        assert_eq!(role_str(&MessageRole::System), "system");
        assert_eq!(role_str(&MessageRole::User), "user");
        assert_eq!(role_str(&MessageRole::Assistant), "assistant");
        assert_eq!(role_str(&MessageRole::Tool), "tool");
    }

    #[test]
    fn test_build_input_items_text_only() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
        ];
        let items = build_responses_input_items(&messages);
        assert_eq!(items.len(), 3);
        let first = items[0].to_json();
        assert_eq!(first["type"], "message");
        assert_eq!(first["role"], "system");
        assert_eq!(first["content"][0]["text"], "You are helpful.");
        assert_eq!(first["content"][0]["type"], "input_text");
        // Assistant text is output_text.
        let third = items[2].to_json();
        assert_eq!(third["content"][0]["type"], "output_text");
    }

    #[test]
    fn test_build_input_items_with_tool_use_and_result() {
        let mut user_msg = ChatMessage::user("what is the weather?");
        user_msg.content.push(ContentBlock::ToolUse(ToolCall::new(
            "call_1",
            "get_weather",
            serde_json::json!({"city": "NYC"}),
        )));
        let tool_msg = ChatMessage::text(MessageRole::Tool, "72F and sunny");
        let tool_msg = ChatMessage {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(
                opensquilla_core::types::ToolResult::success("call_1", "72F and sunny"),
            )],
            ..tool_msg
        };

        let items = build_responses_input_items(&[user_msg, tool_msg]);
        assert_eq!(items.len(), 3);
        // message item with the text, then function_call, then function_call_output
        assert_eq!(items[0].item_type(), "message");
        assert_eq!(items[1].item_type(), "function_call");
        assert_eq!(items[2].item_type(), "function_call_output");

        let fc = items[1].to_json();
        assert_eq!(fc["call_id"], "call_1");
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["arguments"], r#"{"city":"NYC"}"#);

        let out = items[2].to_json();
        assert_eq!(out["call_id"], "call_1");
        assert_eq!(out["output"], "72F and sunny");
    }

    #[test]
    fn test_build_input_items_drops_reasoning() {
        let assistant = ChatMessage {
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Reasoning("hidden".into()),
                ContentBlock::Text("visible".into()),
            ],
            ..ChatMessage::assistant("")
        };
        let items = build_responses_input_items(&[assistant]);
        assert_eq!(items.len(), 1);
        let json = items[0].to_json();
        let content = json["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "visible");
    }

    #[test]
    fn test_content_part_image_and_file_json() {
        let image = ContentPart::input_image("https://example.com/a.png");
        assert_eq!(image.to_json()["type"], "input_image");
        assert_eq!(image.to_json()["image_url"], "https://example.com/a.png");

        let file = ContentPart::input_file("notes.txt", "aGVsbG8=", "text/plain");
        assert_eq!(file.to_json()["type"], "input_file");
        assert_eq!(file.to_json()["filename"], "notes.txt");
        assert_eq!(file.to_json()["file_data"], "aGVsbG8=");
    }

    // -----------------------------------------------------------------------
    // Tools
    // -----------------------------------------------------------------------

    #[test]
    fn test_responses_tool_function() {
        let tool = ResponsesTool::function(
            "get_weather",
            "Get the weather",
            serde_json::json!({"type": "object", "properties": {}}),
        );
        let json = tool.to_json();
        assert_eq!(json["type"], "function");
        assert_eq!(json["name"], "get_weather");
        assert_eq!(json["description"], "Get the weather");
        assert!(json.get("parameters").is_some());
    }

    #[test]
    fn test_responses_tool_builtins() {
        for t in [
            ResponsesToolType::ComputerUse,
            ResponsesToolType::WebSearch,
            ResponsesToolType::FileSearch,
            ResponsesToolType::CodeInterpreter,
        ] {
            let json = ResponsesTool::builtin(t).to_json();
            assert_eq!(json["type"], t.as_str());
            assert!(json.get("name").is_none());
        }
    }

    #[test]
    fn test_tool_from_tool_definition_detects_builtins() {
        let web = ToolDefinition {
            name: "web_search".into(),
            description: "".into(),
            input_schema: serde_json::json!({}),
        };
        let built = ResponsesTool::from_tool_definition(&web);
        assert_eq!(built.tool_type, ResponsesToolType::WebSearch);

        let custom = ToolDefinition {
            name: "my_function".into(),
            description: "desc".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let func = ResponsesTool::from_tool_definition(&custom);
        assert_eq!(func.tool_type, ResponsesToolType::Function);
        assert_eq!(func.name, "my_function");
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_basic() {
        let p = provider();
        let config = ChatConfig {
            model: "o3".into(),
            max_tokens: 2048,
            temperature: 0.4,
            ..Default::default()
        };
        let messages = vec![ChatMessage::system("Be concise."), ChatMessage::user("Hi")];
        let body = p.build_request_body(&config, &messages, &[], false);
        assert_eq!(body["model"], "o3");
        assert_eq!(body["max_output_tokens"].as_u64(), Some(2048));
        assert_eq!(body["store"].as_bool(), Some(false));
        assert_eq!(body["instructions"], "Be concise.");
        assert_eq!(body["temperature"].as_f64(), Some(0.4));
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        // max_tokens is NOT the chat-completions key.
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn test_build_request_body_tools_and_choice() {
        let p = provider();
        let config = ChatConfig {
            model: "o3".into(),
            extra: HashMap::from([("tool_choice".into(), serde_json::json!("required"))]),
            ..Default::default()
        };
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        }];
        let body = p.build_request_body(&config, &[ChatMessage::user("find")], &tools, false);
        assert_eq!(body["tool_choice"], "required");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "search");
    }

    #[test]
    fn test_build_request_body_previous_response_id() {
        let p = provider();
        let config = ChatConfig {
            model: "o3".into(),
            extra: HashMap::from([(
                "previous_response_id".into(),
                serde_json::json!("resp_abc123"),
            )]),
            ..Default::default()
        };
        let body = p.build_request_body(&config, &[ChatMessage::user("again")], &[], false);
        assert_eq!(body["previous_response_id"], "resp_abc123");
    }

    #[test]
    fn test_build_request_body_json_schema_output() {
        let p = provider();
        let config = ChatConfig {
            model: "o3".into(),
            extra: HashMap::from([
                (
                    "output_json_schema".into(),
                    serde_json::json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
                ),
                ("output_json_schema_strict".into(), serde_json::json!(true)),
            ]),
            ..Default::default()
        };
        let body = p.build_request_body(&config, &[ChatMessage::user("structured")], &[], false);
        let format = &body["text"]["format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["name"], "structured_output");
        assert_eq!(format["strict"].as_bool(), Some(true));
        assert_eq!(format["schema"]["type"], "object");
    }

    #[test]
    fn test_build_request_body_default_model_fallback() {
        let p = provider().with_default_model("o4-mini");
        let config = ChatConfig::default(); // empty model
        let body = p.build_request_body(&config, &[ChatMessage::user("hi")], &[], false);
        assert_eq!(body["model"], "o4-mini");
    }

    #[test]
    fn test_build_request_body_empty_input_no_tools() {
        let p = provider();
        let config = ChatConfig {
            model: "o3".into(),
            ..Default::default()
        };
        let body = p.build_request_body(&config, &[], &[], false);
        assert!(body.get("tools").is_none());
        assert!(body.get("instructions").is_none());
        assert_eq!(body["input"].as_array().unwrap().len(), 0);
    }

    // -----------------------------------------------------------------------
    // Response parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_response_with_message_and_function_call() {
        let p = provider();
        let data = serde_json::json!({
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hello world"}]},
                {"type": "function_call", "call_id": "c1", "id": "fc_1", "name": "search", "arguments": "{\"q\": \"rust\"}"}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 10}
        });
        let resp = p.parse_response(&data, "o3");
        assert_eq!(resp.model, "o3");
        assert_eq!(resp.usage.input_tokens, 5);
        assert_eq!(resp.usage.output_tokens, 10);
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
        let msg = &resp.content[0];
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.text_content(), "Hello world");
        let tcs = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "c1");
        assert_eq!(tcs[0].name, "search");
        assert_eq!(tcs[0].input["q"], "rust");
    }

    #[test]
    fn test_parse_response_end_turn_stop_reason() {
        let p = provider();
        let data = serde_json::json!({
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Just text"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = p.parse_response(&data, "o3");
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert!(resp.content[0].tool_calls.is_none());
    }

    #[test]
    fn test_parse_response_length_truncation() {
        let p = provider();
        let data = serde_json::json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "partial"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = p.parse_response(&data, "o3");
        assert_eq!(resp.stop_reason.as_deref(), Some("length"));
    }

    #[test]
    fn test_parse_response_reasoning_block() {
        let p = provider();
        let data = serde_json::json!({
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought carefully"}], "content": [{"type": "reasoning_text", "text": "let me think"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]}
            ],
            "usage": {"input_tokens": 3, "output_tokens": 4}
        });
        let resp = p.parse_response(&data, "o3");
        let msg = &resp.content[0];
        match &msg.content[0] {
            ContentBlock::Reasoning(r) => assert_eq!(r, "thought carefullylet me think"),
            other => panic!("expected reasoning block, got {other:?}"),
        }
        match &msg.content[1] {
            ContentBlock::Text(t) => assert_eq!(t, "answer"),
            other => panic!("expected text block, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_builtin_calls() {
        let p = provider();
        let data = serde_json::json!({
            "status": "completed",
            "output": [
                {"type": "computer_call", "call_id": "cc_1", "action": {"type": "click", "x": 100, "y": 200}},
                {"type": "web_search_call", "id": "ws_1"},
                {"type": "file_search_call", "id": "fs_1"},
                {"type": "code_interpreter_call", "id": "ci_1", "input": "print(1)"}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let resp = p.parse_response(&data, "o3");
        let msg = &resp.content[0];
        let tcs = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tcs.len(), 4);
        assert_eq!(tcs[0].name, "computer_use");
        assert_eq!(tcs[0].input["type"], "click");
        assert_eq!(tcs[1].name, "web_search");
        assert_eq!(tcs[2].name, "file_search");
        assert_eq!(tcs[3].name, "code_interpreter");
        assert_eq!(tcs[3].input["input"], "print(1)");
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn test_parse_usage_details() {
        let p = provider();
        let data = serde_json::json!({
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 7}
            }
        });
        let usage = p.parse_usage(&data);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cached_tokens, 4);
        assert_eq!(usage.reasoning_tokens, 7);
    }

    #[test]
    fn test_parse_output_items_unknown_type() {
        let p = provider();
        let data = serde_json::json!({
            "output": [{"type": "some_new_item", "foo": 1}]
        });
        let items = p.parse_output_items(&data);
        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponsesOutputItem::Unknown { item_type } => assert_eq!(item_type, "some_new_item"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // SSE parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_text_delta() {
        let data = r#"{"type":"response.output_text.delta","delta":"Hello"}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text event"),
        }
    }

    #[test]
    fn test_parse_reasoning_deltas() {
        for etype in [
            "response.reasoning.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ] {
            let data = format!(r#"{{"type":"{etype}","delta":"thinking"}}"#);
            let event = parse_responses_sse_event(&data);
            match event.unwrap() {
                Ok(StreamEvent::Reasoning { reasoning }) => assert_eq!(reasoning, "thinking"),
                _ => panic!("Expected Reasoning event for {etype}"),
            }
        }
    }

    #[test]
    fn test_parse_completed() {
        let data = r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":20}}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                let usage = usage.unwrap();
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 20);
                assert_eq!(stop_reason, Some("completed".into()));
            }
            _ => panic!("Expected Done event"),
        }
    }

    #[test]
    fn test_parse_function_call_item_added() {
        let data = r#"{"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_1","name":"get_weather"}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert!(arguments.is_empty());
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_builtin_item_added() {
        let data = r#"{"type":"response.output_item.added","item":{"type":"computer_call","call_id":"cc_1","action":{"type":"click","x":1,"y":2}}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "cc_1");
                assert_eq!(name, "computer_use");
                assert!(arguments.contains("\"click\""));
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_function_call_arguments_delta() {
        let data = r#"{"type":"response.function_call_arguments.delta","item_id":"call_1","name":"get_weather","delta":"{\"city\":"}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"#);
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_output_item_done_authoritative_args() {
        let data = r#"{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"NYC\"}"}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"NYC"}"#);
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_parse_failed() {
        let data = r#"{"type":"response.failed","error":{"message":"rate limited","code":"rate_limit_exceeded"}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Error { message }) => assert_eq!(message, "rate limited"),
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_error_event() {
        let data = r#"{"type":"error","error":{"message":"bad request"}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Error { message }) => assert_eq!(message, "bad request"),
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_empty_delta_is_none() {
        let data = r#"{"type":"response.output_text.delta","delta":""}"#;
        assert!(parse_responses_sse_event(data).is_none());
    }

    #[test]
    fn test_parse_invalid_json_is_none() {
        assert!(parse_responses_sse_event("not json").is_none());
    }

    #[test]
    fn test_parse_unknown_event_is_none() {
        let data = r#"{"type":"response.created","response":{"id":"resp_1"}}"#;
        assert!(parse_responses_sse_event(data).is_none());
    }

    // -----------------------------------------------------------------------
    // SseDelta integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_responses_sse_to_delta_text() {
        let data = r#"{"type":"response.output_text.delta","delta":"hi"}"#;
        assert_eq!(
            responses_sse_to_delta(data),
            Some(SseDelta::Text("hi".into()))
        );
    }

    #[test]
    fn test_responses_sse_to_delta_tool_begin() {
        let data = r#"{"type":"response.output_item.added","item":{"type":"function_call","call_id":"c","name":"f"}}"#;
        assert_eq!(
            responses_sse_to_delta(data),
            Some(SseDelta::ToolCallBegin {
                id: "c".into(),
                name: "f".into()
            })
        );
    }

    #[test]
    fn test_responses_sse_to_delta_tool_delta() {
        let data =
            r#"{"type":"response.function_call_arguments.delta","item_id":"c","delta":"{\"a\""}"#;
        assert_eq!(
            responses_sse_to_delta(data),
            Some(SseDelta::ToolCallDelta("{\"a\"".into()))
        );
    }

    #[test]
    fn test_responses_sse_to_delta_done() {
        let data = r#"{"type":"response.completed"}"#;
        assert_eq!(responses_sse_to_delta(data), Some(SseDelta::Done));
    }

    // -----------------------------------------------------------------------
    // Provider kinds and configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_provider_kind_parsing() {
        assert_eq!(
            OpenAiResponsesProvider::parse("openai_responses"),
            Some(OpenAiResponsesProvider::OpenAiResponses)
        );
        assert_eq!(
            OpenAiResponsesProvider::parse("volcengine_coding_plan"),
            Some(OpenAiResponsesProvider::VolcengineCodingPlan)
        );
        assert_eq!(
            OpenAiResponsesProvider::parse("byteplus_coding_plan"),
            Some(OpenAiResponsesProvider::ByteplusCodingPlan)
        );
        assert_eq!(OpenAiResponsesProvider::parse("nope"), None);
        assert_eq!(
            OpenAiResponsesProvider::OpenAiResponses.as_str(),
            "openai_responses"
        );
    }

    #[test]
    fn test_provider_kind_base_urls() {
        assert_eq!(
            OpenAiResponsesProvider::OpenAiResponses.default_base_url(),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            OpenAiResponsesProvider::VolcengineCodingPlan.default_base_url(),
            "https://ark.cn-beijing.volces.com/api/v3"
        );
        assert_eq!(
            OpenAiResponsesProvider::ByteplusCodingPlan.default_base_url(),
            "https://ark.byteplus.com/api/v3"
        );
    }

    #[test]
    fn test_config_resolves_default_base_url() {
        let cfg = OpenAiResponsesConfig::volcengine_coding_plan("key");
        assert_eq!(
            cfg.resolved_base_url(),
            "https://ark.cn-beijing.volces.com/api/v3"
        );
        let cfg = cfg.with_base_url("https://example.com/v2");
        assert_eq!(cfg.resolved_base_url(), "https://example.com/v2");
    }

    #[test]
    fn test_provider_from_config_sets_endpoint() {
        let p = OpenAIResponsesProvider::from_config(OpenAiResponsesConfig::byteplus_coding_plan(
            "key",
        ));
        assert_eq!(p.name(), "byteplus_coding_plan");
        assert_eq!(p.api_base(), "https://ark.byteplus.com/api/v3");
        assert_eq!(
            p.responses_url(),
            "https://ark.byteplus.com/api/v3/responses"
        );
        assert_eq!(
            p.provider_kind(),
            OpenAiResponsesProvider::ByteplusCodingPlan
        );
    }

    #[test]
    fn test_new_detects_kind_from_name() {
        let p = OpenAIResponsesProvider::new("volcengine_coding_plan", "", "key");
        assert_eq!(p.name(), "volcengine_coding_plan");
        assert_eq!(
            p.provider_kind(),
            OpenAiResponsesProvider::VolcengineCodingPlan
        );
        // Non-matching names keep their label and fall back to the OpenAI kind.
        let p2 = OpenAIResponsesProvider::new("custom", "https://x", "key");
        assert_eq!(p2.name(), "custom");
        assert_eq!(p2.provider_kind(), OpenAiResponsesProvider::OpenAiResponses);
    }

    #[test]
    fn test_supported_models() {
        let p = provider();
        let models = p.supported_models();
        assert!(models.contains(&"o3".to_string()));
        assert!(models.contains(&"o4-mini".to_string()));
    }
}
