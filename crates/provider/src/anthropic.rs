//! Anthropic provider.
//!
//! Communicates with the Anthropic Messages API (and Anthropic-compatible
//! endpoints such as MiniMax and the Qwen token-plan gateway) using reqwest
//! for HTTP and a stateful SSE stream for token streaming.
//!
//! The adapter implements the shared [`Provider`] trait. Request bodies are
//! built by [`build_anthropic_request`], non-streaming responses are parsed by
//! [`parse_anthropic_response`], and streaming responses are consumed by
//! [`AnthropicStream`], which understands the full Anthropic Messages SSE event
//! sequence (`message_start`, `content_block_start`, `content_block_delta`,
//! `content_block_stop`, `message_delta`, `message_stop`).

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

/// The Anthropic API version header sent by default.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Cap for non-thinking `max_tokens` values.
const DEFAULT_MAX_TOKENS_CAP: u32 = 8192;
/// Extra keys handled explicitly by [`build_anthropic_request`] and therefore
/// not merged blindly into the request payload.
const RESERVED_EXTRA_KEYS: &[&str] = &[
    "thinking",
    "thinking_budget_tokens",
    "cache_breakpoints",
    "tool_cache_control",
];

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// The Anthropic-compatible backend family a provider instance targets.
///
/// The three variants share the Messages wire format but differ in default
/// model, auth style, and base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicProviderKind {
    /// Anthropic proper (`api.anthropic.com`).
    Anthropic,
    /// MiniMax's Anthropic-compatible endpoint (Bearer auth).
    Minimax,
    /// Qwen token-plan gateway exposing an Anthropic-compatible surface.
    QwenTokenPlanAnthropic,
}

impl AnthropicProviderKind {
    /// The registry / spec id for this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Minimax => "minimax",
            Self::QwenTokenPlanAnthropic => "qwen_token_plan_anthropic",
        }
    }

    /// Infer the kind from a provider id. Unknown ids fall back to Anthropic.
    pub fn from_id(id: &str) -> Self {
        match id {
            "minimax" => Self::Minimax,
            "qwen_token_plan_anthropic" => Self::QwenTokenPlanAnthropic,
            _ => Self::Anthropic,
        }
    }

    /// The default model advertised for this kind.
    pub fn default_model(&self) -> &'static str {
        match self {
            Self::Anthropic => "claude-3-5-sonnet-20241022",
            Self::Minimax => "abab6.5-chat",
            Self::QwenTokenPlanAnthropic => "qwen-plus",
        }
    }

    /// Default auth header style for this kind. MiniMax's Anthropic-compatible
    /// endpoint requires a Bearer header; the others use `x-api-key`.
    pub fn default_auth_style(&self) -> AuthHeaderStyle {
        match self {
            Self::Minimax => AuthHeaderStyle::Bearer,
            _ => AuthHeaderStyle::XApiKey,
        }
    }
}

/// How the API key is presented to the upstream endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthHeaderStyle {
    /// `x-api-key: <key>` (Anthropic proper).
    #[default]
    XApiKey,
    /// `Authorization: Bearer <key>`.
    Bearer,
}

/// Configuration for Anthropic extended thinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Whether thinking is enabled.
    pub enabled: bool,
    /// Token budget for `{"type": "enabled"}` thinking.
    #[serde(default = "default_thinking_budget")]
    pub budget_tokens: u32,
    /// Use `{"type": "adaptive"}` (Claude Sonnet 4.6 / Opus 4.6 families).
    #[serde(default)]
    pub adaptive: bool,
}

fn default_thinking_budget() -> u32 {
    4096
}

/// A system-prompt breakpoint for prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheBreakpoint {
    /// The text segment to cache.
    pub text: String,
    /// Whether this segment is marked with `cache_control: ephemeral`.
    #[serde(default)]
    pub cache: bool,
}

/// Provider-level configuration for the Anthropic backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// Which Anthropic-compatible family this instance targets.
    pub provider: AnthropicProviderKind,
    /// The API key.
    pub api_key: String,
    /// Base URL (e.g. `https://api.anthropic.com` or `https://api.anthropic.com/v1`).
    pub base_url: String,
    /// Override the `anthropic-version` header.
    #[serde(default)]
    pub anthropic_version: Option<String>,
    /// Additional `anthropic-beta` header values.
    #[serde(default)]
    pub anthropic_beta: Option<Vec<String>>,
    /// Default model used when a request's `ChatConfig.model` is empty.
    #[serde(default = "default_anthropic_model")]
    pub default_model: String,
    /// How the API key is presented.
    #[serde(default)]
    pub auth_style: AuthHeaderStyle,
    /// Minimum temperature for models in `temperature_floor_model_ids`.
    #[serde(default)]
    pub temperature_floor: f64,
    /// Model ids (last path segment) that are subject to `temperature_floor`.
    #[serde(default)]
    pub temperature_floor_model_ids: Vec<String>,
    /// Replay provider-private thinking/signature blocks on request.
    #[serde(default = "default_true")]
    pub replay_provider_state: bool,
}

fn default_anthropic_model() -> String {
    "claude-3-5-sonnet-20241022".into()
}

fn default_true() -> bool {
    true
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            provider: AnthropicProviderKind::Anthropic,
            api_key: String::new(),
            base_url: "https://api.anthropic.com/v1".into(),
            anthropic_version: None,
            anthropic_beta: None,
            default_model: default_anthropic_model(),
            auth_style: AuthHeaderStyle::XApiKey,
            temperature_floor: 0.0,
            temperature_floor_model_ids: Vec::new(),
            replay_provider_state: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

/// Join all system-message text into a single Anthropic `system` string.
fn extract_system(messages: &[ChatMessage]) -> Option<String> {
    let parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.text_content())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// Build the `system` payload, applying `cache_breakpoints` from `extra` when
/// present (mirrors the Python `_build_system_payload`).
fn build_system_payload(system: &str, extra: &HashMap<String, Value>) -> Value {
    if let Some(bps) = extra.get("cache_breakpoints").and_then(|v| v.as_array()) {
        let mut blocks: Vec<Value> = Vec::new();
        for bp in bps {
            let text = bp
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if text.is_empty() {
                continue;
            }
            let cache = bp.get("cache").and_then(|v| v.as_bool()).unwrap_or(false);
            let mut block = json!({"type": "text", "text": text});
            if cache {
                block["cache_control"] = json!({"type": "ephemeral"});
            }
            blocks.push(block);
        }
        if !blocks.is_empty() {
            return Value::Array(blocks);
        }
    }
    json!(system)
}

/// Build the `tools` payload, optionally marking indices from
/// `tool_cache_control` in `extra` with ephemeral `cache_control`.
fn build_tools_payload(tools: &[ToolDefinition], extra: &HashMap<String, Value>) -> Vec<Value> {
    let cache_tools: std::collections::HashSet<usize> = extra
        .get("tool_cache_control")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64())
                .map(|i| i as usize)
                .collect()
        })
        .unwrap_or_default();

    tools
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut tool = json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            });
            if cache_tools.contains(&i) {
                tool["cache_control"] = json!({"type": "ephemeral"});
            }
            tool
        })
        .collect()
}

/// Read thinking configuration out of `ChatConfig.extra`.
///
/// Supported shapes:
/// - `{"thinking": true, "thinking_budget_tokens": 4096}`
/// - `{"thinking": {"enabled": true, "budget_tokens": 4096, "type": "adaptive"}}`
fn extract_thinking(cfg: &ChatConfig) -> Option<ThinkingConfig> {
    let thinking = cfg.extra.get("thinking")?;
    let enabled = match thinking {
        Value::Bool(b) => *b,
        Value::Object(_) => thinking
            .get("enabled")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        Value::Number(n) => n.as_f64().unwrap_or(0.0) > 0.0,
        _ => false,
    };
    if !enabled {
        return None;
    }
    let budget = cfg
        .extra
        .get("thinking_budget_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| thinking.get("budget_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(4096) as u32;
    let adaptive = thinking.get("type").and_then(|v| v.as_str()) == Some("adaptive");
    Some(ThinkingConfig {
        enabled: true,
        budget_tokens: budget,
        adaptive,
    })
}

/// Build the `thinking` parameter, raising `max_tokens` when the budget would
/// otherwise exceed it (Anthropic requires `max_tokens > budget_tokens`).
fn build_thinking_payload(
    thinking: &ThinkingConfig,
    model: &str,
    max_tokens: &mut u32,
) -> Option<Value> {
    if !thinking.enabled {
        return None;
    }
    let model_lower = model.to_lowercase();
    let adaptive = thinking.adaptive
        || model_lower.contains("claude-sonnet-4-6")
        || model_lower.contains("claude-opus-4-6");
    if adaptive {
        return Some(json!({"type": "adaptive"}));
    }
    let budget = thinking.budget_tokens.max(1);
    if budget >= *max_tokens {
        *max_tokens = budget.saturating_add(4096);
    }
    Some(json!({"type": "enabled", "budget_tokens": budget}))
}

/// Convert a single internal message into an Anthropic Messages content array.
///
/// System messages are handled separately by [`build_anthropic_request`];
/// tool results are emitted as `tool_result` content blocks on a `user`-role
/// message, matching the Anthropic wire contract.
fn convert_message(msg: &ChatMessage, replay_provider_state: bool) -> Value {
    let role = match msg.role {
        Role::System => "user",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "user",
    };

    let mut parts: Vec<Value> = Vec::new();
    for block in &msg.content {
        match block {
            ContentBlock::Text(t) => parts.push(json!({"type": "text", "text": t})),
            ContentBlock::ToolUse(tc) => parts.push(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            })),
            ContentBlock::ToolResult(tr) => parts.push(json!({
                "type": "tool_result",
                "tool_use_id": tr.tool_use_id,
                "content": tr.content,
                "is_error": tr.is_error,
            })),
            ContentBlock::Reasoning(r) => {
                // Thinking/signature replay is only valid for the exact minting
                // turn; a foreign signature is rejected by the API.
                if replay_provider_state {
                    parts.push(json!({"type": "thinking", "thinking": r}));
                }
            }
        }
    }
    // Backwards-compat fields.
    if let Some(calls) = &msg.tool_calls {
        for tc in calls {
            parts.push(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.input,
            }));
        }
    }
    if let Some(tr) = &msg.tool_result {
        parts.push(json!({
            "type": "tool_result",
            "tool_use_id": tr.tool_use_id,
            "content": tr.content,
            "is_error": tr.is_error,
        }));
    }

    // Legacy tool messages: plain text on a Tool role with a `tool_call_id`
    // becomes an Anthropic `tool_result` block so the model can correlate it.
    let has_tool_result = parts
        .iter()
        .any(|p| p.get("type").and_then(|v| v.as_str()) == Some("tool_result"));
    if msg.role == Role::Tool && !has_tool_result && !msg.text_content().is_empty() {
        let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
        parts.push(json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": msg.text_content(),
            "is_error": false,
        }));
    }

    if parts.is_empty() {
        parts.push(json!({"type": "text", "text": ""}));
    }

    json!({"role": role, "content": parts})
}

/// Build an Anthropic Messages API request body.
///
/// System messages are extracted into the top-level `system` parameter,
/// tool/thinking definitions into their own parameters, and the remaining
/// messages into the `messages` array. Provider-specific knobs (`thinking`,
/// `cache_breakpoints`, `tool_cache_control`) are read from `cfg.extra`.
pub fn build_anthropic_request(
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    cfg: &ChatConfig,
    anthropic_cfg: &AnthropicConfig,
    stream: bool,
) -> Value {
    let model = if cfg.model.is_empty() {
        anthropic_cfg.default_model.clone()
    } else {
        cfg.model.clone()
    };

    let mut max_tokens = cfg.max_tokens.clamp(1, DEFAULT_MAX_TOKENS_CAP);
    let thinking = extract_thinking(cfg);
    let thinking_payload = thinking
        .as_ref()
        .and_then(|t| build_thinking_payload(t, &model, &mut max_tokens));

    let system = extract_system(messages);
    let system_payload = system
        .as_deref()
        .map(|s| build_system_payload(s, &cfg.extra));

    let anthropic_messages: Vec<Value> = messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| convert_message(m, anthropic_cfg.replay_provider_state))
        .collect();

    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "stream": stream,
    });

    if let Some(sys) = system_payload {
        body["system"] = sys;
    }
    if !cfg.stop_sequences.is_empty() {
        body["stop_sequences"] = json!(cfg.stop_sequences);
    }
    if !tools.is_empty() {
        body["tools"] = json!(build_tools_payload(tools, &cfg.extra));
    }
    if let Some(t) = thinking_payload {
        body["thinking"] = t;
    } else if !cfg.extra.contains_key("temperature") {
        body["temperature"] = json!(cfg.temperature);
    }

    // Merge remaining extra parameters, skipping the reserved keys handled above.
    for (k, v) in &cfg.extra {
        if RESERVED_EXTRA_KEYS.contains(&k.as_str()) {
            continue;
        }
        body[k] = v.clone();
    }

    body
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// Read Anthropic cache-creation token counts, handling both the direct
/// `cache_creation_input_tokens` field and the per-breakpoint
/// `cache_creation` map.
fn cache_creation_input_tokens(usage: &Value) -> u64 {
    if let Some(direct) = usage["cache_creation_input_tokens"].as_u64() {
        if direct > 0 {
            return direct;
        }
    }
    usage["cache_creation"]
        .as_object()
        .map(|obj| obj.values().filter_map(|v| v.as_u64()).sum())
        .unwrap_or(0)
}

/// Build a [`Usage`] from an Anthropic usage object, folding cache read and
/// cache creation tokens into the input count.
fn anthropic_usage(usage: &Value) -> Usage {
    let base = usage["input_tokens"].as_u64().unwrap_or(0);
    let cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
    let cache_creation = cache_creation_input_tokens(usage);
    let output = usage["output_tokens"].as_u64().unwrap_or(0);
    Usage::new(base + cache_read + cache_creation, output)
}

/// Parse a non-streaming Anthropic Messages response into a [`ProviderResponse`].
///
/// Content blocks of type `text`, `tool_use`, and `thinking` are preserved;
/// tool calls are attached to the assistant message's `tool_calls` field.
pub fn parse_anthropic_response(data: &Value) -> ProviderResponse {
    let content = data["content"]
        .as_array()
        .map(|blocks| {
            let mut message = ChatMessage {
                role: Role::Assistant,
                content: Vec::new(),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            };
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            for block in blocks {
                match block["type"].as_str() {
                    Some("text") => {
                        let text = block["text"].as_str().unwrap_or("").to_string();
                        message.content.push(ContentBlock::Text(text));
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        tool_calls.push(ToolCall::new(id, name, input));
                    }
                    Some("thinking") => {
                        let thinking = block["thinking"].as_str().unwrap_or("").to_string();
                        message.content.push(ContentBlock::Reasoning(thinking));
                    }
                    _ => {}
                }
            }
            if !tool_calls.is_empty() {
                message.tool_calls = Some(tool_calls);
            }
            vec![message]
        })
        .unwrap_or_default();

    ProviderResponse {
        content,
        usage: anthropic_usage(&data["usage"]),
        model: data["model"].as_str().unwrap_or("").to_string(),
        stop_reason: data["stop_reason"].as_str().map(String::from),
    }
}

/// Map an Anthropic HTTP error body to a [`ProviderError`].
///
/// Recognizes Anthropic's structured `error: { type, message }` bodies and maps
/// `authentication_error`/`permission_error` to [`ProviderError::Auth`],
/// `rate_limit_error` to [`ProviderError::RateLimited`], and the overload /
/// internal classes to retryable provider errors.
fn map_anthropic_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        let err = &value["error"];
        let err_type = err["type"].as_str().unwrap_or("");
        let msg = err["message"].as_str().unwrap_or(body).to_string();
        match err_type {
            "authentication_error" | "permission_error" => ProviderError::Auth(msg),
            "rate_limit_error" => ProviderError::RateLimited(msg),
            "overloaded_error" | "api_error" | "internal_error" => {
                ProviderError::Provider(format!("Anthropic {err_type}: {msg}"))
            }
            _ => ProviderError::Provider(format!("HTTP {status}: {}", truncate(body, 1000))),
        }
    } else {
        let text = truncate(body, 1000);
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            ProviderError::Auth(text)
        } else if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            ProviderError::RateLimited(text)
        } else {
            ProviderError::Provider(format!("HTTP {status}: {text}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Provider for the Anthropic Messages API.
pub struct AnthropicProvider {
    name: String,
    config: AnthropicConfig,
    client: Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    ///
    /// * `name` – provider id (e.g. `"anthropic"`, `"minimax"`,
    ///   `"qwen_token_plan_anthropic"`). Used to infer the backend family.
    /// * `api_base` – the base URL of the API.
    /// * `api_key` – the API key for authentication.
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let kind = AnthropicProviderKind::from_id(&name);
        let config = AnthropicConfig {
            provider: kind,
            api_key: api_key.into(),
            base_url: api_base.into(),
            default_model: kind.default_model().into(),
            auth_style: kind.default_auth_style(),
            ..Default::default()
        };
        Self::from_config(name, config)
    }

    /// Create a provider from an explicit [`AnthropicConfig`].
    pub fn from_config(name: impl Into<String>, config: AnthropicConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(180))
            .build()
            .expect("Failed to create reqwest Client");

        Self {
            name: name.into(),
            config,
            client,
        }
    }

    /// Build a request body from a request config, applying the temperature
    /// floor for configured model ids.
    fn build_request_body(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Value {
        let mut effective = config.clone();
        if effective.model.is_empty() {
            effective.model = self.config.default_model.clone();
        }
        let last = effective
            .model
            .rsplit('/')
            .next()
            .unwrap_or(&effective.model)
            .to_lowercase();
        if self.config.temperature_floor > 0.0
            && self
                .config
                .temperature_floor_model_ids
                .iter()
                .any(|m| m.eq_ignore_ascii_case(&last))
            && !effective.extra.contains_key("temperature")
        {
            effective.temperature = effective.temperature.max(self.config.temperature_floor);
        }
        build_anthropic_request(messages, tools, &effective, &self.config, stream)
    }

    /// Build the full URL for an API path, avoiding a duplicated `/v1` prefix.
    fn api_url(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/v1") && path.starts_with("/v1/") {
            format!("{}{}", base, &path[3..])
        } else {
            format!("{}{}", base, path)
        }
    }

    /// Authentication and version headers for a Messages request.
    fn request_headers(&self) -> Vec<(String, String)> {
        let mut headers = Vec::new();
        if !self.config.api_key.is_empty() {
            match self.config.auth_style {
                AuthHeaderStyle::Bearer => {
                    headers.push((
                        "Authorization".into(),
                        format!("Bearer {}", self.config.api_key),
                    ));
                }
                AuthHeaderStyle::XApiKey => {
                    headers.push(("x-api-key".into(), self.config.api_key.clone()));
                }
            }
        }
        let version = self
            .config
            .anthropic_version
            .as_deref()
            .unwrap_or(ANTHROPIC_VERSION);
        headers.push(("anthropic-version".into(), version.to_string()));
        if let Some(beta) = &self.config.anthropic_beta {
            if !beta.is_empty() {
                headers.push(("anthropic-beta".into(), beta.join(",")));
            }
        }
        headers
    }

    /// POST a request body to `/v1/messages`, normalizing errors.
    async fn post_messages(&self, body: &Value) -> ProviderResult<reqwest::Response> {
        let url = self.api_url("/v1/messages");
        let mut req = self
            .client
            .post(url)
            .header("Content-Type", "application/json");
        for (k, v) in self.request_headers() {
            req = req.header(k, v);
        }
        let resp = req
            .json(body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(map_anthropic_error(status, &text));
        }
        Ok(resp)
    }

    /// Stream a chat response over the Anthropic Messages SSE protocol.
    ///
    /// This is the streaming analogue of `chat()`, surfaced as a method so
    /// callers that need the raw Anthropic stream (rather than the generic
    /// [`Provider`] stream) can use it directly.
    pub async fn stream_anthropic_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let body = self.build_request_body(config, messages, tools, true);
        let resp = self.post_messages(&body).await?;
        let stream = AnthropicStream::new(resp);
        Ok(Box::new(stream))
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        match self.config.provider {
            AnthropicProviderKind::Minimax => vec!["abab6.5-chat".into(), "abab6.5s-chat".into()],
            AnthropicProviderKind::QwenTokenPlanAnthropic => {
                vec!["qwen-plus".into(), "qwen-max".into()]
            }
            AnthropicProviderKind::Anthropic => vec![
                "claude-opus-4-6".into(),
                "claude-sonnet-4-6".into(),
                "claude-haiku-4-5-20251001".into(),
                "claude-3-5-sonnet-20241022".into(),
                "claude-3-5-haiku-20241022".into(),
                "claude-3-opus-20240229".into(),
            ],
        }
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let body = self.build_request_body(config, messages, tools, false);

        debug!(target = "provider", provider = %self.name, "Sending non-streaming request");

        let resp = self.post_messages(&body).await?;
        let data: Value = resp.json().await.map_err(ProviderError::Network)?;

        let parsed = parse_anthropic_response(&data);

        info!(
            target = "provider",
            provider = %self.name,
            model = %config.model,
            input_tokens = parsed.usage.input_tokens,
            output_tokens = parsed.usage.output_tokens,
            "Non-streaming response received"
        );

        Ok(parsed)
    }

    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.stream_anthropic_chat(config, messages, tools).await
    }
}

#[async_trait]
impl ChatProvider for AnthropicProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.stream_anthropic_chat(config, messages, tools).await
    }
}

// ---------------------------------------------------------------------------
// Stateful Anthropic SSE stream
// ---------------------------------------------------------------------------

/// Parse a single Anthropic SSE data line into a [`StreamEvent`].
///
/// This is the stateless parser used by callers that already have access to a
/// frame-oriented feed. For production streaming use [`AnthropicStream`],
/// which additionally tracks tool-call identity and merges usage across
/// `message_start` / `message_delta` frames.
#[allow(dead_code)]
fn parse_anthropic_sse_event(data: &str) -> Option<ProviderResult<StreamEvent>> {
    if data == "[DONE]" {
        return Some(Ok(StreamEvent::Done {
            usage: None,
            stop_reason: None,
        }));
    }

    let value: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse Anthropic SSE JSON: {e}");
            return None;
        }
    };

    let event_type = value["type"].as_str()?;

    match event_type {
        "content_block_delta" => {
            let delta = &value["delta"];
            match delta["type"].as_str() {
                Some("text_delta") => {
                    let text = delta["text"].as_str().unwrap_or("").to_string();
                    if text.is_empty() {
                        None
                    } else {
                        Some(Ok(StreamEvent::Text { text }))
                    }
                }
                Some("input_json_delta") => {
                    let partial = delta["partial_json"].as_str().unwrap_or("").to_string();
                    Some(Ok(StreamEvent::ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: partial,
                    }))
                }
                Some("thinking_delta") => {
                    let reasoning = delta["thinking"].as_str().unwrap_or("").to_string();
                    if reasoning.is_empty() {
                        None
                    } else {
                        Some(Ok(StreamEvent::Reasoning { reasoning }))
                    }
                }
                _ => None,
            }
        }
        "content_block_start" => {
            let block = &value["content_block"];
            match block["type"].as_str() {
                Some("tool_use") => {
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    Some(Ok(StreamEvent::ToolCall {
                        id,
                        name,
                        arguments: String::new(),
                    }))
                }
                Some("thinking") => {
                    let reasoning = block["thinking"].as_str().unwrap_or("").to_string();
                    if reasoning.is_empty() {
                        None
                    } else {
                        Some(Ok(StreamEvent::Reasoning { reasoning }))
                    }
                }
                _ => None,
            }
        }
        "message_delta" => {
            let delta = &value["delta"];
            let stop_reason = delta["stop_reason"].as_str().map(String::from);
            let usage = anthropic_usage(&value["usage"]);
            Some(Ok(StreamEvent::Done {
                usage: Some(usage),
                stop_reason,
            }))
        }
        "message_stop" => None,
        "error" => {
            let msg = value["error"]["message"]
                .as_str()
                .unwrap_or("Unknown Anthropic error")
                .to_string();
            Some(Ok(StreamEvent::Error { message: msg }))
        }
        "ping" => None,
        _ => None,
    }
}

/// Stateful SSE stream adapter for the Anthropic Messages protocol.
///
/// Unlike the stateless `parse_anthropic_sse_event`, this stream tracks
/// tool-call identity by content-block index, accumulates token usage across
/// `message_start` and `message_delta` frames, and emits a terminal `Done`
/// event from `message_stop`.
pub struct AnthropicStream {
    body: futures::stream::BoxStream<'static, Result<bytes::Bytes, reqwest::Error>>,
    buffer: Vec<u8>,
    pending: VecDeque<ProviderResult<StreamEvent>>,
    done: bool,
    message_started: bool,
    input_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
    output_tokens: u64,
    stop_reason: String,
    /// Content-block index -> (tool_use id, tool name) for active tool calls.
    tool_ids: HashMap<i64, (String, String)>,
}

impl AnthropicStream {
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
            message_started: false,
            input_tokens: 0,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            output_tokens: 0,
            stop_reason: "end_turn".to_string(),
            tool_ids: HashMap::new(),
        }
    }

    /// Process one buffered line (which may carry an SSE `data:` prefix).
    fn process_line(&mut self, line: &str) -> Option<Vec<ProviderResult<StreamEvent>>> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }
        let data = trimmed
            .strip_prefix("data: ")
            .or_else(|| trimmed.strip_prefix("data:"))?
            .trim();
        if data.is_empty() || data == "[DONE]" {
            return None;
        }
        let value: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse Anthropic SSE JSON: {e}");
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

    /// Convert one Anthropic event frame into stream events, updating state.
    fn process_frame(&mut self, value: &Value) -> Vec<ProviderResult<StreamEvent>> {
        let mut out = Vec::new();
        let event_type = value["type"].as_str().unwrap_or("");
        match event_type {
            "message_start" => {
                self.message_started = true;
                let usage = &value["message"]["usage"];
                self.input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
                self.cached_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
                self.cache_creation_tokens = cache_creation_input_tokens(usage);
            }
            "content_block_start" => {
                let index = value["index"].as_i64().unwrap_or(-1);
                let block = &value["content_block"];
                match block["type"].as_str() {
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        self.tool_ids.insert(index, (id.clone(), name.clone()));
                        out.push(Ok(StreamEvent::ToolCall {
                            id,
                            name,
                            arguments: String::new(),
                        }));
                    }
                    Some("text") => {
                        let text = block["text"].as_str().unwrap_or("");
                        if !text.is_empty() {
                            out.push(Ok(StreamEvent::Text {
                                text: text.to_string(),
                            }));
                        }
                    }
                    Some("thinking") => {
                        let thinking = block["thinking"].as_str().unwrap_or("");
                        if !thinking.is_empty() {
                            out.push(Ok(StreamEvent::Reasoning {
                                reasoning: thinking.to_string(),
                            }));
                        }
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let delta = &value["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        let text = delta["text"].as_str().unwrap_or("");
                        if !text.is_empty() {
                            out.push(Ok(StreamEvent::Text {
                                text: text.to_string(),
                            }));
                        }
                    }
                    Some("input_json_delta") => {
                        let index = value["index"].as_i64().unwrap_or(-1);
                        let partial = delta["partial_json"].as_str().unwrap_or("").to_string();
                        let (id, name) = self.tool_ids.get(&index).cloned().unwrap_or_default();
                        out.push(Ok(StreamEvent::ToolCall {
                            id,
                            name,
                            arguments: partial,
                        }));
                    }
                    Some("thinking_delta") => {
                        let thinking = delta["thinking"].as_str().unwrap_or("");
                        if !thinking.is_empty() {
                            out.push(Ok(StreamEvent::Reasoning {
                                reasoning: thinking.to_string(),
                            }));
                        }
                    }
                    Some("signature_delta") => {
                        // The signature has no StreamEvent representation; it is
                        // retained only in the non-streaming parse path.
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                // Tool-call completion is signalled by the terminal Done event;
                // the consumer's ToolCallBuffer flushes on JSON validity.
            }
            "message_delta" => {
                let delta = &value["delta"];
                if let Some(reason) = delta["stop_reason"].as_str() {
                    self.stop_reason = reason.to_string();
                }
                let usage = &value["usage"];
                if let Some(o) = usage["output_tokens"].as_u64() {
                    self.output_tokens = o;
                }
                if let Some(c) = usage["cache_read_input_tokens"].as_u64() {
                    self.cached_tokens = c.max(self.cached_tokens);
                }
                let cc = cache_creation_input_tokens(usage);
                if cc > 0 {
                    self.cache_creation_tokens = cc.max(self.cache_creation_tokens);
                }
            }
            "message_stop" => {
                self.done = true;
                let input = self.input_tokens + self.cached_tokens + self.cache_creation_tokens;
                let usage = Usage::new(input, self.output_tokens);
                let stop = if self.stop_reason.is_empty() {
                    None
                } else {
                    Some(self.stop_reason.clone())
                };
                out.push(Ok(StreamEvent::Done {
                    usage: Some(usage),
                    stop_reason: stop,
                }));
            }
            "error" => {
                let msg = value["error"]["message"]
                    .as_str()
                    .unwrap_or("Unknown Anthropic error")
                    .to_string();
                out.push(Ok(StreamEvent::Error { message: msg }));
            }
            "ping" => {}
            _ => {}
        }
        out
    }
}

impl Stream for AnthropicStream {
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
                    let mut events: Vec<ProviderResult<StreamEvent>> = Vec::new();
                    let remaining = String::from_utf8_lossy(&this.buffer).to_string();
                    this.buffer.clear();
                    if !remaining.is_empty() {
                        if let Some(ev) = this.process_line(&remaining) {
                            events.extend(ev);
                        }
                    }
                    // A truncated stream (no message_stop) still yields a
                    // best-effort terminal with the accumulated usage.
                    if !this.done && this.message_started {
                        this.done = true;
                        let input =
                            this.input_tokens + this.cached_tokens + this.cache_creation_tokens;
                        let usage = Usage::new(input, this.output_tokens);
                        let stop = if this.stop_reason.is_empty() {
                            None
                        } else {
                            Some(this.stop_reason.clone())
                        };
                        events.push(Ok(StreamEvent::Done {
                            usage: Some(usage),
                            stop_reason: stop,
                        }));
                    } else {
                        this.done = true;
                    }
                    this.pending.extend(events);
                    if let Some(ev) = this.pending.pop_front() {
                        return Poll::Ready(Some(ev));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }

            // Extract complete lines from the buffer.
            let text = String::from_utf8_lossy(&this.buffer).to_string();
            if let Some(pos) = text.find('\n') {
                let line = text[..pos].to_string();
                this.buffer = text.as_bytes()[pos + 1..].to_vec();
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
    use opensquilla_core::types::ToolResult;

    fn chat_config(model: &str) -> ChatConfig {
        ChatConfig {
            model: model.to_string(),
            max_tokens: 1024,
            temperature: 0.7,
            ..Default::default()
        }
    }

    fn anthropic_config() -> AnthropicConfig {
        AnthropicConfig {
            provider: AnthropicProviderKind::Anthropic,
            api_key: "sk-test".into(),
            base_url: "https://api.anthropic.com/v1".into(),
            anthropic_version: None,
            anthropic_beta: None,
            default_model: "claude-3-5-sonnet-20241022".into(),
            auth_style: AuthHeaderStyle::XApiKey,
            temperature_floor: 0.0,
            temperature_floor_model_ids: Vec::new(),
            replay_provider_state: true,
        }
    }

    #[test]
    fn test_kind_from_id() {
        assert_eq!(
            AnthropicProviderKind::from_id("anthropic"),
            AnthropicProviderKind::Anthropic
        );
        assert_eq!(
            AnthropicProviderKind::from_id("minimax"),
            AnthropicProviderKind::Minimax
        );
        assert_eq!(
            AnthropicProviderKind::from_id("qwen_token_plan_anthropic"),
            AnthropicProviderKind::QwenTokenPlanAnthropic
        );
        assert_eq!(
            AnthropicProviderKind::from_id("unknown"),
            AnthropicProviderKind::Anthropic
        );
    }

    #[test]
    fn test_minimax_default_auth_is_bearer() {
        assert_eq!(
            AnthropicProviderKind::Minimax.default_auth_style(),
            AuthHeaderStyle::Bearer
        );
        assert_eq!(
            AnthropicProviderKind::Anthropic.default_auth_style(),
            AuthHeaderStyle::XApiKey
        );
    }

    #[test]
    fn test_extract_system() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        assert_eq!(
            extract_system(&messages),
            Some("You are helpful.".to_string())
        );
        assert_eq!(extract_system(&[ChatMessage::user("x")]), None);
    }

    #[test]
    fn test_build_request_system_and_messages() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there"),
        ];
        let body = build_anthropic_request(
            &messages,
            &[],
            &chat_config("claude-3-5-sonnet-20241022"),
            &anthropic_config(),
            false,
        );
        assert_eq!(body["system"], "You are helpful.");
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][0]["type"], "text");
        assert_eq!(msgs[0]["content"][0]["text"], "Hello");
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[test]
    fn test_build_request_default_model_when_empty() {
        let mut cfg = chat_config("");
        cfg.model = String::new();
        let body = build_anthropic_request(&[], &[], &cfg, &anthropic_config(), false);
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_build_request_tools() {
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather for a city".into(),
            input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        };
        let body = build_anthropic_request(
            &[ChatMessage::user("what's the weather?")],
            &[tool],
            &chat_config("claude-3-5-sonnet-20241022"),
            &anthropic_config(),
            false,
        );
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["description"], "Get weather for a city");
        assert_eq!(
            tools[0]["input_schema"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn test_build_request_tool_cache_control() {
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather".into(),
            input_schema: json!({"type": "object"}),
        };
        let mut cfg = chat_config("claude-3-5-sonnet-20241022");
        cfg.extra.insert("tool_cache_control".into(), json!([0]));
        let body = build_anthropic_request(
            &[ChatMessage::user("hi")],
            &[tool],
            &cfg,
            &anthropic_config(),
            false,
        );
        assert_eq!(
            body["tools"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[test]
    fn test_build_request_thinking_enabled() {
        let mut cfg = chat_config("claude-3-5-sonnet-20241022");
        cfg.extra.insert("thinking".into(), json!(true));
        cfg.extra
            .insert("thinking_budget_tokens".into(), json!(4096));
        let body = build_anthropic_request(&[], &[], &cfg, &anthropic_config(), false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
    }

    #[test]
    fn test_build_request_thinking_adaptive() {
        let mut cfg = chat_config("claude-sonnet-4-6");
        cfg.extra.insert("thinking".into(), json!(true));
        let body = build_anthropic_request(&[], &[], &cfg, &anthropic_config(), false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        // Adaptive thinking must not leak temperature.
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn test_build_request_thinking_raises_max_tokens() {
        let mut cfg = chat_config("claude-3-5-sonnet-20241022");
        cfg.max_tokens = 1024;
        cfg.extra.insert("thinking".into(), json!(true));
        cfg.extra
            .insert("thinking_budget_tokens".into(), json!(8192));
        let body = build_anthropic_request(&[], &[], &cfg, &anthropic_config(), false);
        // budget (8192) >= max_tokens (1024), so max_tokens is raised to
        // budget + 4096.
        assert_eq!(body["max_tokens"], 8192 + 4096);
    }

    #[test]
    fn test_build_request_cache_breakpoints() {
        let messages = vec![ChatMessage::system("prefix text")];
        let mut cfg = chat_config("claude-3-5-sonnet-20241022");
        cfg.extra.insert(
            "cache_breakpoints".into(),
            json!([
                {"text": "first segment", "cache": true},
                {"text": "second segment", "cache": false},
            ]),
        );
        let body = build_anthropic_request(&messages, &[], &cfg, &anthropic_config(), false);
        let sys = body["system"].as_array().unwrap();
        assert_eq!(sys.len(), 2);
        assert_eq!(sys[0]["text"], "first segment");
        assert_eq!(sys[0]["cache_control"], json!({"type": "ephemeral"}));
        assert!(sys[1].get("cache_control").is_none());
    }

    #[test]
    fn test_convert_tool_use_and_result() {
        let msg = ChatMessage {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("thinking out loud".into()),
                ContentBlock::ToolUse(ToolCall::new(
                    "toolu_1",
                    "get_weather",
                    json!({"city": "NYC"}),
                )),
            ],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let value = convert_message(&msg, true);
        assert_eq!(value["role"], "assistant");
        let parts = value["content"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["type"], "tool_use");
        assert_eq!(parts[1]["id"], "toolu_1");
        assert_eq!(parts[1]["input"]["city"], "NYC");
    }

    #[test]
    fn test_convert_tool_result_message() {
        let msg = ChatMessage {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult::success(
                "toolu_1", "70F",
            ))],
            name: None,
            tool_call_id: Some("toolu_1".into()),
            tool_calls: None,
            tool_result: None,
        };
        let value = convert_message(&msg, true);
        assert_eq!(value["role"], "user");
        let parts = value["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "tool_result");
        assert_eq!(parts[0]["tool_use_id"], "toolu_1");
        assert_eq!(parts[0]["content"], "70F");
    }

    #[test]
    fn test_convert_legacy_tool_text_message() {
        let msg = ChatMessage {
            role: Role::Tool,
            content: vec![ContentBlock::Text("42".into())],
            name: None,
            tool_call_id: Some("toolu_1".into()),
            tool_calls: None,
            tool_result: None,
        };
        let value = convert_message(&msg, true);
        let parts = value["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "tool_result");
        assert_eq!(parts[0]["tool_use_id"], "toolu_1");
    }

    #[test]
    fn test_convert_thinking_respects_replay_flag() {
        let msg = ChatMessage {
            role: Role::Assistant,
            content: vec![ContentBlock::Reasoning("secret".into())],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let with_replay = convert_message(&msg, true);
        assert_eq!(with_replay["content"][0]["type"], "thinking");
        let without_replay = convert_message(&msg, false);
        assert_eq!(
            without_replay["content"].as_array().unwrap().len(),
            1,
            "empty-thinking messages still carry a text placeholder"
        );
        assert_eq!(without_replay["content"][0]["type"], "text");
    }

    #[test]
    fn test_parse_anthropic_response_text_and_thinking() {
        let data = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Hello"},
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        });
        let resp = parse_anthropic_response(&data);
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0].content.len(), 2);
        assert!(matches!(
            resp.content[0].content[0],
            ContentBlock::Reasoning(_)
        ));
        assert_eq!(resp.content[0].text_content(), "Hello");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 20);
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_parse_anthropic_response_tool_use() {
        let data = json!({
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 8},
        });
        let resp = parse_anthropic_response(&data);
        let calls = resp.content[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["city"], "NYC");
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn test_parse_anthropic_response_cache_usage() {
        let data = json!({
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30,
            },
        });
        let resp = parse_anthropic_response(&data);
        assert_eq!(resp.usage.input_tokens, 180);
        assert_eq!(resp.usage.output_tokens, 20);
        assert_eq!(resp.usage.total_tokens, 200);
    }

    #[test]
    fn test_cache_creation_input_tokens_map() {
        let usage = json!({"cache_creation": {"0": 10, "1": 20}});
        assert_eq!(cache_creation_input_tokens(&usage), 30);
        let usage = json!({"cache_creation_input_tokens": 40});
        assert_eq!(cache_creation_input_tokens(&usage), 40);
    }

    #[test]
    fn test_parse_anthropic_sse_text_delta() {
        let data = r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#;
        let event = parse_anthropic_sse_event(data);
        assert!(event.is_some());
        match event.unwrap() {
            Ok(StreamEvent::Text { text }) => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text event"),
        }
    }

    #[test]
    fn test_parse_anthropic_sse_thinking_delta() {
        let data =
            r#"{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}"#;
        let event = parse_anthropic_sse_event(data);
        assert!(event.is_some());
        match event.unwrap() {
            Ok(StreamEvent::Reasoning { reasoning }) => assert_eq!(reasoning, "hmm"),
            _ => panic!("Expected Reasoning event"),
        }
    }

    #[test]
    fn test_parse_anthropic_sse_done() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":10,"output_tokens":20}}"#;
        let event = parse_anthropic_sse_event(data);
        assert!(event.is_some());
        match event.unwrap() {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                assert_eq!(stop_reason, Some("end_turn".into()));
                let u = usage.unwrap();
                assert_eq!(u.input_tokens, 10);
                assert_eq!(u.output_tokens, 20);
            }
            _ => panic!("Expected Done event"),
        }
    }

    #[test]
    fn test_parse_anthropic_sse_error() {
        let data = r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#;
        let event = parse_anthropic_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Error { message }) => assert_eq!(message, "overloaded"),
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_anthropic_stream_tool_use_events() {
        let mut stream = AnthropicStream::from_body(futures::stream::empty().boxed());
        let evs = stream
            .process_line(r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#)
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert!(arguments.is_empty());
            }
            _ => panic!("Expected ToolCall event"),
        }
        let evs = stream
            .process_line(r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc\""}}"#)
            .unwrap();
        match &evs[0] {
            Ok(StreamEvent::ToolCall {
                id,
                name,
                arguments,
            }) => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"loc"#);
            }
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_anthropic_stream_message_stop_emits_done() {
        let mut stream = AnthropicStream::from_body(futures::stream::empty().boxed());
        stream.process_line(r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":5}}}"#);
        stream.process_line(r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}"#);
        let events = stream
            .process_line(r#"data: {"type":"message_stop"}"#)
            .unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input_tokens, 15);
                assert_eq!(u.output_tokens, 20);
                assert_eq!(stop_reason.as_deref(), Some("end_turn"));
            }
            _ => panic!("Expected Done event"),
        }
        // After message_stop the stream is finished.
        assert!(stream.done);
    }

    #[test]
    fn test_anthropic_stream_thinking_delta() {
        let mut stream = AnthropicStream::from_body(futures::stream::empty().boxed());
        let evs = stream
            .process_line(r#"data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}"#)
            .unwrap();
        match &evs[0] {
            Ok(StreamEvent::Reasoning { reasoning }) => assert_eq!(reasoning, "hmm"),
            _ => panic!("Expected Reasoning event"),
        }
    }

    #[test]
    fn test_map_anthropic_error_auth() {
        let err = map_anthropic_error(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"type":"error","error":{"type":"authentication_error","message":"bad key"}}"#,
        );
        assert!(matches!(err, ProviderError::Auth(_)));
    }

    #[test]
    fn test_map_anthropic_error_rate_limit() {
        let err = map_anthropic_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#,
        );
        assert!(matches!(err, ProviderError::RateLimited(_)));
    }

    #[test]
    fn test_map_anthropic_error_overloaded() {
        let err = map_anthropic_error(
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
        );
        assert!(matches!(err, ProviderError::Provider(_)));
    }
}
