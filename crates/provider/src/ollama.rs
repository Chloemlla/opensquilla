//! Ollama provider.
//!
//! Communicates with a local (or cloud) Ollama instance using the Ollama HTTP
//! API (`http://localhost:11434` by default). Ollama is a local-first provider:
//! no API key is required and no rate limiting is applied.
//!
//! Chat requests hit `POST /api/chat` with JSON-per-line streaming. Model
//! management endpoints (`/api/tags`, `/api/show`, `/api/pull`,
//! `/api/delete`) and local embeddings (`/api/embed`) are exposed as methods on
//! [`OllamaProvider`].

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use opensquilla_core::types::{ChatMessage, ContentBlock, Role, ToolCall, ToolDefinition, Usage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{debug, info, warn};

use crate::types::{
    ChatConfig, ChatProvider, Provider, ProviderError, ProviderResponse, ProviderResult,
    StreamEvent,
};
use crate::util::truncate;

/// Default Ollama base URL.
const DEFAULT_OLLAMA_BASE: &str = "http://localhost:11434";
/// Ollama's server default `num_ctx` is 2048, which silently truncates the
/// front of an agent prompt (system prompt + tool schemas). Default to a window
/// large enough for real agent turns.
const DEFAULT_OLLAMA_NUM_CTX: u32 = 8192;

// ---------------------------------------------------------------------------
// Configuration and model types
// ---------------------------------------------------------------------------

/// Provider-level configuration for the Ollama backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Base URL of the Ollama server (e.g. `http://localhost:11434`).
    #[serde(default = "default_ollama_base")]
    pub base_url: String,
    /// Default model used when a request's `ChatConfig.model` is empty.
    #[serde(default = "default_ollama_model")]
    pub default_model: String,
    /// `keep_alive` value sent with chat requests.
    #[serde(default)]
    pub keep_alive: Option<String>,
    /// Context window size (`num_ctx` in the request `options`).
    #[serde(default = "default_num_ctx")]
    pub num_ctx: Option<u32>,
}

fn default_ollama_base() -> String {
    DEFAULT_OLLAMA_BASE.into()
}

fn default_ollama_model() -> String {
    "llama3.1".into()
}

fn default_num_ctx() -> Option<u32> {
    Some(DEFAULT_OLLAMA_NUM_CTX)
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_ollama_base(),
            default_model: default_ollama_model(),
            keep_alive: None,
            num_ctx: Some(DEFAULT_OLLAMA_NUM_CTX),
        }
    }
}

/// A model listed by `GET /api/tags`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    /// Model id, e.g. `llama3.1:latest`.
    pub name: String,
    /// ISO-8601 timestamp of the last modification.
    pub modified_at: String,
    /// Model size in bytes.
    pub size: i64,
    /// Model digest.
    pub digest: String,
    /// Model metadata.
    pub details: OllamaModelDetails,
}

/// Model metadata returned in `details`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    /// Parent model for a quantized/derived model.
    pub parent_model: String,
    /// File format (e.g. `gguf`).
    pub format: String,
    /// Model family (e.g. `llama`).
    pub family: String,
    /// All families the model belongs to.
    pub families: Vec<String>,
    /// Parameter size label (e.g. `8.0B`).
    pub parameter_size: String,
    /// Quantization level (e.g. `Q4_0`).
    pub quantization_level: String,
}

impl Default for OllamaModelDetails {
    fn default() -> Self {
        Self {
            parent_model: String::new(),
            format: String::new(),
            family: String::new(),
            families: Vec::new(),
            parameter_size: String::new(),
            quantization_level: String::new(),
        }
    }
}

/// A single progress frame from `POST /api/pull`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgress {
    /// Status label (e.g. `pulling manifest`, `verifying sha256 digest`).
    pub status: String,
    /// Layer digest, present while a layer is being pulled.
    #[serde(default)]
    pub digest: Option<String>,
    /// Total bytes for the active layer.
    #[serde(default)]
    pub total: Option<i64>,
    /// Bytes completed for the active layer.
    #[serde(default)]
    pub completed: Option<i64>,
}

// ---------------------------------------------------------------------------
// Request building helpers
// ---------------------------------------------------------------------------

/// Build an Ollama tool definition in OpenAI-function shape.
fn build_ollama_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

/// Parse a single Ollama `tool_calls` entry into a [`ToolCall`].
fn parse_ollama_tool_call(tc: &Value, index: usize) -> Option<ToolCall> {
    let fn_obj = tc.get("function")?;
    let name = fn_obj
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = fn_obj
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let id = tc
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let id = if id.is_empty() {
        format!("call_{index}")
    } else {
        id
    };
    Some(ToolCall::new(id, name, arguments))
}

/// Convert one internal message into one or more Ollama chat messages.
///
/// A single message may expand into several Ollama messages: assistant turns
/// carry their `tool_calls` so the model keeps a record of what it invoked,
/// and each `tool_result` block becomes its own `tool`-role message tagged with
/// `tool_name` so the model can correlate the result with the call.
fn build_ollama_messages(msg: &ChatMessage, tool_names: &HashMap<String, String>) -> Vec<Value> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut images: Vec<String> = Vec::new();
    let mut tool_messages: Vec<Value> = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text(t) => text_parts.push(t.clone()),
            ContentBlock::ToolUse(tc) => {
                tool_calls.push(json!({"function": {"name": tc.name, "arguments": tc.input}}));
            }
            ContentBlock::ToolResult(tr) => {
                let mut tm = json!({
                    "role": "tool",
                    "content": tr.content,
                });
                if let Some(name) = tool_names.get(&tr.tool_use_id) {
                    tm["tool_name"] = json!(name);
                }
                tool_messages.push(tm);
            }
            ContentBlock::Reasoning(_) => {
                // Ollama has no thinking replay channel; drop reasoning blocks.
            }
        }
    }
    // Backwards-compat fields.
    if let Some(calls) = &msg.tool_calls {
        for tc in calls {
            tool_calls.push(json!({"function": {"name": tc.name, "arguments": tc.input}}));
        }
    }
    if let Some(tr) = &msg.tool_result {
        let mut tm = json!({
            "role": "tool",
            "content": tr.content,
        });
        if let Some(name) = tool_names.get(&tr.tool_use_id) {
            tm["tool_name"] = json!(name);
        }
        tool_messages.push(tm);
    }

    let mut out: Vec<Value> = Vec::new();
    if !text_parts.is_empty() || !tool_calls.is_empty() || !images.is_empty() {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let mut main = json!({
            "role": role,
            "content": text_parts.join(" "),
        });
        if !tool_calls.is_empty() {
            main["tool_calls"] = json!(tool_calls);
        }
        if !images.is_empty() {
            main["images"] = json!(images);
        }
        out.push(main);
    }
    out.extend(tool_messages);
    out
}

/// Convert all messages into the Ollama wire format.
///
/// A first pass maps `tool_use` ids to their tool names so tool results can be
/// correlated.
fn convert_ollama_messages(messages: &[ChatMessage]) -> Vec<Value> {
    let mut tool_names: HashMap<String, String> = HashMap::new();
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse(tc) = block {
                tool_names.insert(tc.id.clone(), tc.name.clone());
            }
        }
        if let Some(calls) = &msg.tool_calls {
            for tc in calls {
                tool_names.insert(tc.id.clone(), tc.name.clone());
            }
        }
    }

    let mut out: Vec<Value> = Vec::new();
    for msg in messages {
        out.extend(build_ollama_messages(msg, &tool_names));
    }
    out
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Provider for a local or cloud Ollama instance.
pub struct OllamaProvider {
    name: String,
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider.
    ///
    /// * `name` – a label for this provider instance (e.g. `"ollama"`).
    /// * `api_base` – the base URL of the Ollama server.
    pub fn new(name: impl Into<String>, api_base: impl Into<String>) -> Self {
        let config = OllamaConfig {
            base_url: api_base.into(),
            default_model: default_ollama_model(),
            keep_alive: None,
            num_ctx: Some(DEFAULT_OLLAMA_NUM_CTX),
        };
        Self::from_config(name, config)
    }

    /// Create a provider from an explicit [`OllamaConfig`].
    pub fn from_config(name: impl Into<String>, config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to create reqwest Client");

        Self {
            name: name.into(),
            config,
            client,
        }
    }

    /// Build the Ollama chat request body.
    fn build_request_body(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Value {
        let model = if config.model.is_empty() {
            self.config.default_model.clone()
        } else {
            config.model.clone()
        };

        let ollama_messages = convert_ollama_messages(messages);

        let mut options = json!({
            "num_predict": config.max_tokens,
            "num_ctx": self.config.num_ctx.unwrap_or(DEFAULT_OLLAMA_NUM_CTX),
        });
        if let Some(opts) = options.as_object_mut() {
            if !config.extra.contains_key("temperature") {
                opts.insert("temperature".into(), json!(config.temperature));
            }
        }

        let mut body = json!({
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
            "options": options,
        });

        if let Some(keep_alive) = &self.config.keep_alive {
            body["keep_alive"] = json!(keep_alive);
        }
        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools.iter().map(build_ollama_tool).collect();
            body["tools"] = json!(tool_defs);
        }

        // Merge extra parameters, letting them override defaults.
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &config.extra {
                if k == "temperature" {
                    continue;
                }
                obj.insert(k.clone(), v.clone());
            }
        }

        body
    }

    /// Map a connect error to a friendly diagnostic.
    fn map_network_error(&self, e: reqwest::Error) -> ProviderError {
        if e.is_connect() {
            ProviderError::Provider(format!(
                "Failed to connect to Ollama at {} — is the Ollama server running? ({})",
                self.config.base_url, e
            ))
        } else {
            ProviderError::Network(e)
        }
    }

    /// POST a chat request body to `/api/chat`, normalizing errors.
    async fn post_chat(&self, body: &Value) -> ProviderResult<reqwest::Response> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "chat"));
        }
        Ok(resp)
    }

    /// Stream a chat response from `/api/chat`.
    ///
    /// This is the streaming analogue of `chat()`, surfaced as a method so
    /// callers that need the raw Ollama JSONL stream can use it directly.
    pub async fn stream_ollama_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let body = self.build_request_body(config, messages, tools, true);
        let resp = self.post_chat(&body).await?;
        let stream = OllamaStream::new(resp);
        Ok(Box::new(stream))
    }

    /// List installed models (`GET /api/tags`).
    pub async fn list_models(&self) -> ProviderResult<Vec<OllamaModel>> {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "list models"));
        }
        let data: Value = resp.json().await.map_err(ProviderError::Network)?;
        let models: Vec<OllamaModel> = data["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| serde_json::from_value(m.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        Ok(models)
    }

    /// Show a model's metadata (`POST /api/show`).
    pub async fn show_model(&self, name: &str) -> ProviderResult<OllamaModel> {
        let url = format!("{}/api/show", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(url)
            .json(&json!({"name": name}))
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "show model"));
        }
        let data: Value = resp.json().await.map_err(ProviderError::Network)?;
        let details: OllamaModelDetails =
            serde_json::from_value(data["details"].clone()).unwrap_or_default();
        Ok(OllamaModel {
            name: name.to_string(),
            modified_at: data["modified_at"].as_str().unwrap_or("").to_string(),
            size: data["size"].as_i64().unwrap_or(0),
            digest: data["digest"].as_str().unwrap_or("").to_string(),
            details,
        })
    }

    /// Pull a model, returning the accumulated streaming progress
    /// (`POST /api/pull`).
    pub async fn pull_model(&self, name: &str) -> ProviderResult<Vec<PullProgress>> {
        let url = format!("{}/api/pull", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(url)
            .json(&json!({"name": name, "stream": true}))
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "pull model"));
        }
        let bytes = resp.bytes().await.map_err(ProviderError::Network)?;
        let text = String::from_utf8_lossy(&bytes);
        let mut progress: Vec<PullProgress> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(p) = serde_json::from_str::<PullProgress>(line) {
                progress.push(p);
            }
        }
        Ok(progress)
    }

    /// Delete a model (`DELETE /api/delete`).
    pub async fn delete_model(&self, name: &str) -> ProviderResult<()> {
        let url = format!("{}/api/delete", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .delete(url)
            .json(&json!({"name": name}))
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "delete model"));
        }
        Ok(())
    }

    /// Embed a single text via `POST /api/embed`.
    pub async fn embed(&self, model: &str, text: &str) -> ProviderResult<Vec<f32>> {
        let mut embeddings = self.embed_batch(model, &[text.to_string()]).await?;
        embeddings
            .pop()
            .ok_or_else(|| ProviderError::Provider("Ollama embed returned no embeddings".into()))
    }

    /// Embed a batch of texts via `POST /api/embed`.
    pub async fn embed_batch(
        &self,
        model: &str,
        texts: &[String],
    ) -> ProviderResult<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(url)
            .json(&json!({"model": model, "input": texts}))
            .send()
            .await
            .map_err(|e| self.map_network_error(e))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_ollama_error(status, &text, "embed"));
        }
        let data: Value = resp.json().await.map_err(ProviderError::Network)?;
        let embeddings: Vec<Vec<f32>> = data["embeddings"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|item| {
                        item.as_array()
                            .map(|vec| {
                                vec.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|v| v as f32)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();

        if embeddings.is_empty() {
            // Older Ollama versions expose a single embedding via the legacy
            // `/api/embeddings` shape.
            let legacy: Vec<f32> = data["embedding"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|v| v as f32)
                        .collect()
                })
                .unwrap_or_default();
            if !legacy.is_empty() {
                return Ok(vec![legacy]);
            }
        }
        Ok(embeddings)
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "llama3.2".into(),
            "llama3.1".into(),
            "mistral".into(),
            "codellama".into(),
            "mixtral".into(),
            "deepseek-coder".into(),
            "qwen2.5".into(),
        ]
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let body = self.build_request_body(config, messages, tools, false);

        debug!(target = "provider", provider = %self.name, "Sending non-streaming request to Ollama");

        let resp = self.post_chat(&body).await?;
        let data: Value = resp.json().await.map_err(ProviderError::Network)?;

        let msg = &data["message"];
        let content = msg["content"].as_str().unwrap_or("").to_string();
        let mut assistant = ChatMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Text(content)],
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
            name: None,
        };
        if let Some(calls) = msg["tool_calls"].as_array() {
            let tool_calls: Vec<ToolCall> = calls
                .iter()
                .enumerate()
                .filter_map(|(i, tc)| parse_ollama_tool_call(tc, i))
                .collect();
            if !tool_calls.is_empty() {
                assistant.tool_calls = Some(tool_calls);
            }
        }

        let usage = Usage::new(
            data["prompt_eval_count"].as_u64().unwrap_or(0),
            data["eval_count"].as_u64().unwrap_or(0),
        );
        let done_reason = data["done_reason"].as_str().map(String::from);
        let model = data["model"].as_str().unwrap_or("").to_string();

        info!(
            target = "provider",
            provider = %self.name,
            model = %config.model,
            "Non-streaming Ollama response received"
        );

        Ok(ProviderResponse {
            content: vec![assistant],
            usage,
            model,
            stop_reason: done_reason,
        })
    }

    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.stream_ollama_chat(config, messages, tools).await
    }
}

#[async_trait]
impl ChatProvider for OllamaProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.stream_ollama_chat(config, messages, tools).await
    }
}

// ---------------------------------------------------------------------------
// Ollama stream — each line is a complete JSON object
// ---------------------------------------------------------------------------

/// Map an Ollama HTTP error response to a [`ProviderError`].
///
/// Ollama error bodies carry `{"error": "..."}`; a 404 (model not found) maps
/// to [`ProviderError::UnsupportedModel`] and auth failures to
/// [`ProviderError::Auth`].
fn map_ollama_error(status: reqwest::StatusCode, body: &str, action: &str) -> ProviderError {
    let msg = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"].as_str().map(String::from))
        .unwrap_or_else(|| body.to_string());
    let msg = truncate(&msg, 1000);
    match status.as_u16() {
        401 | 403 => ProviderError::Auth(format!("{action} auth failed: {msg}")),
        404 => ProviderError::UnsupportedModel(msg),
        429 => ProviderError::RateLimited(format!("{action} rate limited: {msg}")),
        _ => ProviderError::Provider(format!("HTTP {status}: {msg}")),
    }
}

/// Stream adapter for Ollama's JSON-per-line streaming format.
///
/// Each line is a complete JSON object. Text deltas, tool calls, a top-level
/// `error` field, and the terminal `done=true` frame are all handled here.
pub struct OllamaStream {
    body: futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    buffer: Vec<u8>,
    pending: VecDeque<ProviderResult<StreamEvent>>,
    done: bool,
}

impl OllamaStream {
    /// Create a stream from a `reqwest::Response`.
    fn new(response: reqwest::Response) -> Self {
        Self::from_body(response.bytes_stream().boxed())
    }

    /// Create a stream from a raw byte stream. Exposed for tests.
    fn from_body(
        body: futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    ) -> Self {
        Self {
            body,
            buffer: Vec::new(),
            pending: VecDeque::new(),
            done: false,
        }
    }

    /// Process one buffered JSON line.
    fn process_line(&mut self, line: &str) -> Option<Vec<ProviderResult<StreamEvent>>> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse Ollama JSON line: {e}");
                return None;
            }
        };
        let events = self.process_frame(&value);
        if events.is_empty() {
            None
        } else {
            Some(events)
        }
    }

    /// Convert one Ollama chunk into stream events, updating state.
    fn process_frame(&mut self, value: &Value) -> Vec<ProviderResult<StreamEvent>> {
        let mut out = Vec::new();

        if let Some(err) = value.get("error") {
            let message = if let Some(s) = err.as_str() {
                s.to_string()
            } else {
                err.to_string()
            };
            out.push(Ok(StreamEvent::Error { message }));
            return out;
        }

        let msg_chunk = &value["message"];
        if let Some(text) = msg_chunk["content"].as_str() {
            if !text.is_empty() {
                out.push(Ok(StreamEvent::Text {
                    text: text.to_string(),
                }));
            }
        }

        if let Some(calls) = msg_chunk["tool_calls"].as_array() {
            for (i, tc) in calls.iter().enumerate() {
                if let Some(call) = parse_ollama_tool_call(tc, i) {
                    let arguments =
                        serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".into());
                    out.push(Ok(StreamEvent::ToolCall {
                        id: call.id,
                        name: call.name,
                        arguments,
                    }));
                }
            }
        }

        if value.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
            let usage = Usage::new(
                value["prompt_eval_count"].as_u64().unwrap_or(0),
                value["eval_count"].as_u64().unwrap_or(0),
            );
            let done_reason = value["done_reason"].as_str().map(String::from);
            self.done = true;
            out.push(Ok(StreamEvent::Done {
                usage: Some(usage),
                stop_reason: done_reason,
            }));
        }

        out
    }
}

impl Stream for OllamaStream {
    type Item = ProviderResult<StreamEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(ev) = this.pending.pop_front() {
                return Poll::Ready(Some(ev));
            }
            if this.done {
                return Poll::Ready(None);
            }

            match Pin::new(&mut this.body).poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(ProviderError::Network(e))));
                }
                Poll::Ready(None) => {
                    this.done = true;
                    let remaining = String::from_utf8_lossy(&this.buffer).to_string();
                    this.buffer.clear();
                    if !remaining.is_empty() {
                        if let Some(events) = this.process_line(&remaining) {
                            this.pending.extend(events);
                        }
                    }
                    if let Some(ev) = this.pending.pop_front() {
                        return Poll::Ready(Some(ev));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }

            let text = String::from_utf8_lossy(&this.buffer).to_string();
            if let Some(pos) = text.find('\n') {
                let line = text[..pos].to_string();
                this.buffer = text[pos + 1..].as_bytes().to_vec();
                if let Some(events) = this.process_line(&line) {
                    this.pending.extend(events);
                }
            } else {
                return Poll::Pending;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_config(model: &str) -> ChatConfig {
        ChatConfig {
            model: model.to_string(),
            max_tokens: 512,
            temperature: 0.3,
            ..Default::default()
        }
    }

    fn ollama_config() -> OllamaConfig {
        OllamaConfig {
            base_url: "http://localhost:11434".into(),
            default_model: "llama3.1".into(),
            keep_alive: Some("5m".into()),
            num_ctx: Some(8192),
        }
    }

    #[test]
    fn test_config_default() {
        let c = OllamaConfig::default();
        assert_eq!(c.base_url, "http://localhost:11434");
        assert_eq!(c.default_model, "llama3.1");
        assert_eq!(c.num_ctx, Some(8192));
        assert!(c.keep_alive.is_none());
    }

    #[test]
    fn test_build_request_basic() {
        let provider = OllamaProvider::from_config("ollama", ollama_config());
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let body = provider.build_request_body(&chat_config("llama3.1"), &messages, &[], false);
        assert_eq!(body["model"], "llama3.1");
        assert_eq!(body["stream"], false);
        assert_eq!(body["keep_alive"], "5m");
        assert_eq!(body["options"]["num_ctx"], 8192);
        assert_eq!(body["options"]["num_predict"], 512);
        assert_eq!(body["options"]["temperature"], 0.3);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hello");
    }

    #[test]
    fn test_build_request_default_model() {
        let provider = OllamaProvider::from_config("ollama", ollama_config());
        let body = provider.build_request_body(&chat_config(""), &[], &[], false);
        assert_eq!(body["model"], "llama3.1");
    }

    #[test]
    fn test_build_request_tools() {
        let provider = OllamaProvider::from_config("ollama", ollama_config());
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        };
        let body = provider.build_request_body(&chat_config("llama3.1"), &[], &[tool], false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn test_build_request_tool_result_expansion() {
        let provider = OllamaProvider::from_config("ollama", ollama_config());
        let messages = vec![ChatMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall::new(
                "call_1",
                "get_weather",
                json!({"city": "NYC"}),
            ))],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }];
        let body = provider.build_request_body(&chat_config("llama3.1"), &messages, &[], false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["arguments"]["city"],
            "NYC"
        );
    }

    #[test]
    fn test_build_request_tool_result_message() {
        let provider = OllamaProvider::from_config("ollama", ollama_config());
        let messages = vec![
            ChatMessage {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse(ToolCall::new(
                    "call_1",
                    "get_weather",
                    json!({"city": "NYC"}),
                ))],
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            },
            ChatMessage {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult(
                    opensquilla_core::types::ToolResult::success("call_1", "70F"),
                )],
                name: None,
                tool_call_id: Some("call_1".into()),
                tool_calls: None,
                tool_result: None,
            },
        ];
        let body = provider.build_request_body(&chat_config("llama3.1"), &messages, &[], false);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["content"], "70F");
        assert_eq!(msgs[1]["tool_name"], "get_weather");
    }

    #[test]
    fn test_parse_ollama_tool_call() {
        let tc = json!({
            "id": "call_abc",
            "function": {"name": "get_weather", "arguments": {"city": "NYC"}}
        });
        let call = parse_ollama_tool_call(&tc, 0).unwrap();
        assert_eq!(call.id, "call_abc");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.input["city"], "NYC");
    }

    #[test]
    fn test_parse_ollama_tool_call_generates_id() {
        let tc = json!({
            "function": {"name": "f", "arguments": {}}
        });
        let call = parse_ollama_tool_call(&tc, 3).unwrap();
        assert_eq!(call.id, "call_3");
    }

    #[test]
    fn test_ollama_stream_text_and_done() {
        let mut stream = OllamaStream::from_body(futures::stream::empty().boxed());
        let evs = stream
            .process_line(r#"{"model":"llama3","message":{"role":"assistant","content":"Hello"},"done":false}"#)
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            Ok(StreamEvent::Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text event"),
        }

        let evs = stream
            .process_line(r#"{"model":"llama3","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":20}"#)
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input_tokens, 10);
                assert_eq!(u.output_tokens, 20);
                assert_eq!(stop_reason.as_deref(), Some("stop"));
            }
            _ => panic!("Expected Done event"),
        }
        assert!(stream.done);
    }

    #[test]
    fn test_ollama_stream_tool_calls() {
        let mut stream = OllamaStream::from_body(futures::stream::empty().boxed());
        let evs = stream
            .process_line(r#"{"model":"llama3","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"get_weather","arguments":{"city":"NYC"}}}]},"done":false}"#)
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
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
    fn test_ollama_stream_error() {
        let mut stream = OllamaStream::from_body(futures::stream::empty().boxed());
        let evs = stream
            .process_line(r#"{"error":"model 'nope' not found"}"#)
            .unwrap();
        match &evs[0] {
            Ok(StreamEvent::Error { message }) => {
                assert_eq!(message, "model 'nope' not found")
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_map_ollama_error_404() {
        let err = map_ollama_error(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error":"model 'x' not found, try pulling it first"}"#,
            "chat",
        );
        assert!(matches!(err, ProviderError::UnsupportedModel(_)));
    }

    #[test]
    fn test_map_ollama_error_auth() {
        let err = map_ollama_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error":"unauthorized"}"#,
            "chat",
        );
        assert!(matches!(err, ProviderError::Auth(_)));
    }

    #[test]
    fn test_parse_ollama_list_models() {
        let data = json!({
            "models": [
                {
                    "name": "llama3.1:latest",
                    "modified_at": "2024-07-22T18:17:54.123Z",
                    "size": 3825936993u32,
                    "digest": "abc123",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "8.0B",
                        "quantization_level": "Q4_0"
                    }
                }
            ]
        });
        let models: Vec<OllamaModel> = data["models"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| serde_json::from_value(m.clone()).ok())
            .collect();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "llama3.1:latest");
        assert_eq!(models[0].details.family, "llama");
        assert_eq!(models[0].details.parameter_size, "8.0B");
    }

    #[test]
    fn test_pull_progress_deserialize() {
        let line = r#"{"status":"pulling manifest"}"#;
        let p: PullProgress = serde_json::from_str(line).unwrap();
        assert_eq!(p.status, "pulling manifest");
        assert!(p.digest.is_none());

        let line = r#"{"status":"downloading","digest":"sha256:abc","total":100,"completed":40}"#;
        let p: PullProgress = serde_json::from_str(line).unwrap();
        assert_eq!(p.digest.as_deref(), Some("sha256:abc"));
        assert_eq!(p.total, Some(100));
        assert_eq!(p.completed, Some(40));
    }
}
