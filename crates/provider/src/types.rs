//! Provider trait, chat configuration, stream events, and shared types.

use async_trait::async_trait;
use futures::Stream;
use opensquilla_core::types::{ChatMessage, ToolDefinition, Usage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Provider-specific errors.
#[derive(Error, Debug)]
pub enum ProviderError {
    /// Authentication failure (invalid API key, expired token, etc.).
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Rate limit exceeded.
    #[error("Rate limited: {0}")]
    RateLimited(String),

    /// The provider returned an error response.
    #[error("Provider error: {0}")]
    Provider(String),

    /// Request timed out.
    #[error("Request timed out: {0}")]
    Timeout(String),

    /// Network or transport-level error.
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// Serialization / deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Invalid configuration.
    #[error("Configuration error: {0}")]
    Config(String),

    /// The model is not supported by this provider.
    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),

    /// An internal (non-recoverable) error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<opensquilla_core::error::Error> for ProviderError {
    fn from(e: opensquilla_core::error::Error) -> Self {
        ProviderError::Internal(e.to_string())
    }
}

/// Convenience result alias for provider operations.
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Configuration for a chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// The model identifier to use (e.g. "gpt-4o", "claude-3-opus-20240229").
    pub model: String,
    /// Sampling temperature (0.0 – 2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Top-p nucleus sampling.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,
    /// Provider-specific extra parameters.
    #[serde(default)]
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

fn default_temperature() -> f64 {
    0.7
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_top_p() -> f64 {
    0.9
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            top_p: default_top_p(),
            stop_sequences: Vec::new(),
            stream: false,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// A non-streaming response from a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    /// The generated message content.
    pub content: Vec<ChatMessage>,
    /// Token usage statistics.
    pub usage: Usage,
    /// The model that generated the response.
    pub model: String,
    /// The reason the generation stopped.
    pub stop_reason: Option<String>,
}

/// Events emitted during a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    /// A text delta: partial text content.
    #[serde(rename = "text")]
    Text {
        /// The partial text content.
        text: String,
    },
    /// A reasoning/thinking delta.
    #[serde(rename = "reasoning")]
    Reasoning {
        /// The partial reasoning content.
        reasoning: String,
    },
    /// A tool call delta: partial tool call arguments.
    #[serde(rename = "tool_call")]
    ToolCall {
        /// The tool call identifier (may be partial for streaming).
        id: String,
        /// The tool name.
        name: String,
        /// Partial JSON arguments.
        arguments: String,
    },
    /// The stream has completed.
    #[serde(rename = "done")]
    Done {
        /// Final usage information.
        usage: Option<Usage>,
        /// The reason generation stopped.
        stop_reason: Option<String>,
    },
    /// An error occurred during streaming.
    #[serde(rename = "error")]
    Error {
        /// The error message.
        message: String,
    },
}

/// The core provider trait.
///
/// All LLM providers (OpenAI, Anthropic, Ollama, etc.) implement this trait.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Return a human-readable name for this provider (e.g. "openai").
    fn name(&self) -> &str;

    /// Return the list of models this provider supports.
    fn supported_models(&self) -> Vec<String>;

    /// Send a non-streaming chat completion request.
    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse>;

    /// Send a streaming chat completion request.
    ///
    /// Returns a `Stream` of `StreamEvent` items.
    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>>;
}

/// A provider with a unified `chat()` entrypoint that streams events.
///
/// This is the ergonomic entrypoint used by the Responses and Codex backends:
/// every request is streamed and the caller consumes `StreamEvent`s. The
/// [`Provider`] trait's `stream_chat` delegates to this method, and its
/// `send_message` collects the stream into a single [`ProviderResponse`].
#[async_trait]
pub trait ChatProvider: Send + Sync {
    /// Stream a chat completion request, emitting partial events.
    ///
    /// The returned stream yields [`StreamEvent`] values (text, reasoning,
    /// tool-call fragments, done, error). Providers may use the OpenAI
    /// Responses item protocol, a chat-completions protocol, or a code
    /// execution endpoint behind this unified surface.
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>>;
}
