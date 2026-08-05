//! IPC contract types.
//!
//! All request and response types that cross the Tauri `invoke()` boundary
//! between the Vue 3 frontend and the Rust agent runtime. These types are
//! serde-serialized with camelCase field names to match the frontend's
//! TypeScript conventions.

use opensquilla_core::events::{StreamEvent, ToolEvent, TurnEvent};
use opensquilla_core::types::{ContentBlock, Message, MessageRole, Usage};
use opensquilla_core::{ModelCapabilities, ModelInfo, ProviderSpec};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Session types
// ---------------------------------------------------------------------------

/// Request to create a new session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionCreateRequest {
    /// Optional title for the session. Defaults to "New Session".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The model to use for this session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The agent ID to associate with this session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Optional system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Session mode: "chat", "plan", "agent", or "batch".
    #[serde(default = "default_session_mode")]
    pub mode: String,
}

fn default_session_mode() -> String {
    "chat".to_string()
}

/// A session entry returned in list/detail responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub id: String,
    pub title: String,
    pub model: String,
    pub agent_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub state: String,
    pub mode: String,
    pub message_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub total_tokens: u64,
}

/// Response containing a list of sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionListResponse {
    pub sessions: Vec<SessionInfo>,
    pub count: usize,
}

/// Response containing a single session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionResponse {
    pub session: SessionInfo,
}

// ---------------------------------------------------------------------------
// Message / chat types
// ---------------------------------------------------------------------------

/// A simplified content block for the IPC boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlockDto {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// A tool use request from the model.
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool result returned to the model.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    /// Model reasoning/thinking content.
    #[serde(rename = "reasoning")]
    Reasoning { reasoning: String },
}

impl From<&ContentBlock> for ContentBlockDto {
    fn from(block: &ContentBlock) -> Self {
        match block {
            ContentBlock::Text(text) => ContentBlockDto::Text { text: text.clone() },
            ContentBlock::ToolUse(call) => ContentBlockDto::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.input.clone(),
            },
            ContentBlock::ToolResult(result) => ContentBlockDto::ToolResult {
                tool_use_id: result.tool_use_id.clone(),
                content: result.content.clone(),
                is_error: result.is_error,
            },
            ContentBlock::Reasoning(reasoning) => ContentBlockDto::Reasoning {
                reasoning: reasoning.clone(),
            },
        }
    }
}

/// A message DTO used in requests and responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageDto {
    pub role: String,
    pub content: Vec<ContentBlockDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl From<&Message> for MessageDto {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        let content: Vec<ContentBlockDto> = msg.content.iter().map(ContentBlockDto::from).collect();
        MessageDto {
            role: role.to_string(),
            content,
            name: msg.name.clone(),
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

/// Request to send a message to the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendRequest {
    /// The session ID to send the message to.
    pub session_id: String,
    /// The text content of the user's message.
    pub message: String,
    /// Optional conversation history to include (for stateless mode).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<MessageDto>,
    /// Override the model for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Override the provider for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Whether to stream the response. Defaults to true.
    #[serde(default = "default_true")]
    pub stream: bool,
}

fn default_true() -> bool {
    true
}

/// A streaming event sent from the backend to the frontend via Tauri events.
///
/// Each variant maps to an engine event that the Vue frontend subscribes to
/// via `listen()`. The events are emitted on the channel named
/// `agent:stream:{session_id}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MessageStreamEvent {
    /// The turn has started.
    #[serde(rename = "turn_start")]
    TurnStart {
        turn_id: String,
        session_id: String,
        timestamp: String,
    },
    /// The model has started generating a response.
    #[serde(rename = "generation_start")]
    GenerationStart { model: String, provider: String },
    /// A streaming delta event from the model.
    #[serde(rename = "stream_event")]
    StreamEvent { event: StreamEventPayload },
    /// A tool call has been initiated.
    #[serde(rename = "tool_call_start")]
    ToolCallStart {
        tool_call_id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool call has completed.
    #[serde(rename = "tool_call_complete")]
    ToolCallComplete {
        tool_call_id: String,
        name: String,
        content: String,
        is_error: bool,
        duration_ms: u64,
    },
    /// The turn has completed successfully.
    #[serde(rename = "turn_complete")]
    TurnComplete {
        turn_id: String,
        session_id: String,
        messages: Vec<MessageDto>,
        usage: UsagePayload,
        duration_ms: u64,
    },
    /// The turn encountered an error.
    #[serde(rename = "turn_error")]
    TurnError {
        turn_id: String,
        session_id: String,
        message: String,
        code: Option<String>,
    },
    /// Context compaction was triggered.
    #[serde(rename = "compaction")]
    Compaction {
        before_count: usize,
        after_count: usize,
        success: bool,
    },
}

/// A serializable representation of a core `StreamEvent`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum StreamEventPayload {
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        block: ContentBlockDto,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: usize,
        delta: StreamDeltaPayload,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        stop_reason: Option<String>,
        usage: Option<UsagePayload>,
    },
    #[serde(rename = "message_stop")]
    MessageStop {
        content: Vec<ContentBlockDto>,
        usage: Option<UsagePayload>,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        code: Option<String>,
    },
    #[serde(rename = "ping")]
    Ping,
}

/// A streaming delta payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StreamDeltaPayload {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "reasoning_delta")]
    ReasoningDelta { reasoning: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// Token usage payload.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UsagePayload {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl From<Usage> for UsagePayload {
    fn from(u: Usage) -> Self {
        UsagePayload {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
        }
    }
}

/// Convert a core `StreamEvent` into a `StreamEventPayload`.
impl From<StreamEvent> for StreamEventPayload {
    fn from(event: StreamEvent) -> Self {
        match event {
            StreamEvent::ContentBlockStart { index, block } => {
                StreamEventPayload::ContentBlockStart {
                    index,
                    block: ContentBlockDto::from(&block),
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                let delta_payload = match delta {
                    opensquilla_core::events::ContentBlockDelta::TextDelta { text } => {
                        StreamDeltaPayload::TextDelta { text }
                    }
                    opensquilla_core::events::ContentBlockDelta::ReasoningDelta { reasoning } => {
                        StreamDeltaPayload::ReasoningDelta { reasoning }
                    }
                    opensquilla_core::events::ContentBlockDelta::InputJsonDelta {
                        partial_json,
                    } => StreamDeltaPayload::InputJsonDelta { partial_json },
                };
                StreamEventPayload::ContentBlockDelta {
                    index,
                    delta: delta_payload,
                }
            }
            StreamEvent::ContentBlockStop { index } => {
                StreamEventPayload::ContentBlockStop { index }
            }
            StreamEvent::MessageDelta { delta, usage } => StreamEventPayload::MessageDelta {
                stop_reason: delta.stop_reason,
                usage: usage.map(UsagePayload::from),
            },
            StreamEvent::MessageStop { content, usage } => StreamEventPayload::MessageStop {
                content: content.iter().map(ContentBlockDto::from).collect(),
                usage: usage.map(UsagePayload::from),
            },
            StreamEvent::Error { message, code } => StreamEventPayload::Error { message, code },
            StreamEvent::Ping => StreamEventPayload::Ping,
        }
    }
}

/// Convert a core `TurnEvent` into a `MessageStreamEvent`.
impl From<TurnEvent> for MessageStreamEvent {
    fn from(event: TurnEvent) -> Self {
        match event {
            TurnEvent::TurnStart { turn_id, timestamp } => MessageStreamEvent::TurnStart {
                turn_id,
                session_id: String::new(),
                timestamp,
            },
            TurnEvent::UserMessage { message } => {
                // UserMessage is not directly streamed; convert to a turn start
                // variant as a no-op passthrough — the frontend already has
                // the user's message from the send request.
                let _ = message;
                MessageStreamEvent::TurnStart {
                    turn_id: String::new(),
                    session_id: String::new(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                }
            }
            TurnEvent::GenerationStart { model, provider } => {
                MessageStreamEvent::GenerationStart { model, provider }
            }
            TurnEvent::StreamEvent { event } => MessageStreamEvent::StreamEvent {
                event: StreamEventPayload::from(event),
            },
            TurnEvent::TurnComplete {
                message,
                usage,
                duration_ms,
            } => MessageStreamEvent::TurnComplete {
                turn_id: String::new(),
                session_id: String::new(),
                messages: vec![MessageDto::from(&message)],
                usage: UsagePayload::from(usage),
                duration_ms,
            },
            TurnEvent::TurnError { message, code } => MessageStreamEvent::TurnError {
                turn_id: String::new(),
                session_id: String::new(),
                message,
                code,
            },
            TurnEvent::Compaction {
                before_count,
                after_count,
                success,
            } => MessageStreamEvent::Compaction {
                before_count,
                after_count,
                success,
            },
        }
    }
}

/// Convert a core `ToolEvent` into a `MessageStreamEvent`.
impl From<ToolEvent> for MessageStreamEvent {
    fn from(event: ToolEvent) -> Self {
        match event {
            ToolEvent::ToolCallStart { call, .. } => MessageStreamEvent::ToolCallStart {
                tool_call_id: call.id,
                name: call.name,
                input: call.input,
            },
            ToolEvent::ToolCallComplete {
                call,
                result,
                duration_ms,
            } => MessageStreamEvent::ToolCallComplete {
                tool_call_id: call.id,
                name: call.name,
                content: result.content,
                is_error: result.is_error,
                duration_ms,
            },
            ToolEvent::ToolCallError {
                call,
                error,
                duration_ms,
            } => MessageStreamEvent::ToolCallComplete {
                tool_call_id: call.id,
                name: call.name,
                content: error,
                is_error: true,
                duration_ms,
            },
            ToolEvent::ToolParallelStart { calls } => {
                // Emit the first call; parallel completion is handled separately.
                if let Some(call) = calls.into_iter().next() {
                    MessageStreamEvent::ToolCallStart {
                        tool_call_id: call.id,
                        name: call.name,
                        input: call.input,
                    }
                } else {
                    MessageStreamEvent::TurnError {
                        turn_id: String::new(),
                        session_id: String::new(),
                        message: "Empty parallel tool call batch".to_string(),
                        code: Some("EMPTY_TOOL_BATCH".to_string()),
                    }
                }
            }
            ToolEvent::ToolParallelComplete {
                results,
                total_duration_ms,
            } => {
                if let Some(result) = results.into_iter().next() {
                    MessageStreamEvent::ToolCallComplete {
                        tool_call_id: result.tool_use_id,
                        name: String::new(),
                        content: result.content,
                        is_error: result.is_error,
                        duration_ms: total_duration_ms,
                    }
                } else {
                    MessageStreamEvent::TurnError {
                        turn_id: String::new(),
                        session_id: String::new(),
                        message: "Empty parallel tool result batch".to_string(),
                        code: Some("EMPTY_TOOL_RESULTS".to_string()),
                    }
                }
            }
            ToolEvent::ToolRejected { call, reason } => MessageStreamEvent::ToolCallComplete {
                tool_call_id: call.id,
                name: call.name,
                content: reason,
                is_error: true,
                duration_ms: 0,
            },
        }
    }
}

/// Response to a message send request (non-streaming mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageSendResponse {
    pub turn_id: String,
    pub session_id: String,
    pub messages: Vec<MessageDto>,
    pub usage: UsagePayload,
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// Provider / Model / Skill types
// ---------------------------------------------------------------------------

/// Response containing the list of configured providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderListResponse {
    pub providers: Vec<ProviderInfo>,
    pub default_provider: Option<String>,
    pub count: usize,
}

/// Provider information for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderInfo {
    pub name: String,
    pub provider_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    pub max_retries: u32,
    pub timeout_secs: u64,
}

impl From<&opensquilla_core::config::ProviderConfig> for ProviderInfo {
    fn from(config: &opensquilla_core::config::ProviderConfig) -> Self {
        ProviderInfo {
            name: config.name.clone(),
            provider_type: config.provider_type.clone(),
            base_url: config.base_url.clone(),
            models: config.models.clone(),
            default_model: config.default_model.clone(),
            max_retries: config.max_retries,
            timeout_secs: config.timeout_secs,
        }
    }
}

/// Response containing the list of available models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelListResponse {
    pub models: Vec<ModelInfoDto>,
    pub default_model: Option<String>,
    pub count: usize,
}

/// Model information for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfoDto {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub context_window: u64,
    pub max_output_tokens: u64,
    pub capabilities: ModelCapabilities,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
}

impl From<&ModelInfo> for ModelInfoDto {
    fn from(info: &ModelInfo) -> Self {
        ModelInfoDto {
            id: info.id.clone(),
            name: info.name.clone(),
            provider: info.provider.clone(),
            context_window: info.context_window,
            max_output_tokens: info.max_output_tokens,
            capabilities: info.capabilities.clone(),
            display_name: None,
        }
    }
}

impl From<&ProviderSpec> for ProviderInfo {
    fn from(spec: &ProviderSpec) -> Self {
        ProviderInfo {
            name: spec.name.clone(),
            provider_type: spec.name.clone(),
            base_url: Some(spec.base_url.clone()),
            models: spec.models.iter().map(|m| m.id.clone()).collect(),
            default_model: spec.models.first().map(|m| m.id.clone()),
            max_retries: 3,
            timeout_secs: 120,
        }
    }
}

/// Response containing the list of available skills.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillListResponse {
    pub skills: Vec<SkillInfo>,
    pub count: usize,
}

/// Skill information for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillInfo {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub description: String,
    pub layer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub is_meta: bool,
    pub disabled: bool,
}

// ---------------------------------------------------------------------------
// Health / Config types
// ---------------------------------------------------------------------------

/// Health report returned by the doctor health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthReport {
    pub status: String,
    pub uptime_seconds: u64,
    pub timestamp: String,
    pub components: Vec<HealthComponent>,
    pub issues: Vec<HealthIssue>,
    pub gateway_running: bool,
    pub gateway_url: Option<String>,
}

/// A single component's health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthComponent {
    pub name: String,
    pub status: String,
    pub description: String,
    pub latency_ms: u64,
    pub details: std::collections::HashMap<String, String>,
}

/// A health issue found during the check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthIssue {
    pub component: String,
    pub severity: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Response containing the full configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigGetResponse {
    pub config: serde_json::Value,
}

/// Request to set a configuration value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigSetRequest {
    pub key: String,
    pub value: serde_json::Value,
}

/// Response to a config set operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigSetResponse {
    pub key: String,
    pub value: serde_json::Value,
    pub status: String,
}

/// Gateway status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayStatusResponse {
    pub running: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response from the workbench surface creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkbenchSurfaceResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_id: Option<String>,
}

/// Request to create a workbench surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkbenchSurfaceCreateRequest {
    pub surface_id: String,
    pub kind: String,
    pub scope_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default)]
    pub allow_remote_resources: bool,
}

/// Request to set the surface rect.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkbenchSurfaceRectRequest {
    pub surface_id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub visible: bool,
}

/// Request to navigate a workbench surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkbenchNavigationRequest {
    pub surface_id: String,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// Artifact preview lease grant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactPreviewLeaseGrant {
    pub launch_url: String,
    pub expected_origin: String,
    pub scope_id: String,
    pub mode: String,
}

/// Artifact preview lease create request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactPreviewLeaseCreateRequest {
    pub artifact_id: String,
    pub scope_id: String,
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
}

/// Artifact preview lease payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactPreviewLeasePayload {
    pub lease_id: String,
    pub effective_mode: String,
    pub launch_url: String,
    pub entrypoint: String,
    pub expires_at: String,
    pub preview_origin: String,
    pub idle_timeout_seconds: u64,
}

/// A generic operation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationResult {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_create_request_deserialize() {
        let json = r#"{"title": "Test", "model": "gpt-4", "mode": "chat"}"#;
        let req: SessionCreateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.title.as_deref(), Some("Test"));
        assert_eq!(req.model.as_deref(), Some("gpt-4"));
        assert_eq!(req.mode, "chat");
    }

    #[test]
    fn test_message_send_request_deserialize() {
        let json = r#"{"sessionId": "abc123", "message": "Hello", "stream": true}"#;
        let req: MessageSendRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.session_id, "abc123");
        assert_eq!(req.message, "Hello");
        assert!(req.stream);
    }

    #[test]
    fn test_usage_payload_from_usage() {
        let usage = Usage::new(100, 50);
        let payload = UsagePayload::from(usage);
        assert_eq!(payload.input_tokens, 100);
        assert_eq!(payload.output_tokens, 50);
        assert_eq!(payload.total_tokens, 150);
    }

    #[test]
    fn test_message_dto_from_message() {
        let msg = Message::user("Hello");
        let dto = MessageDto::from(&msg);
        assert_eq!(dto.role, "user");
        assert_eq!(dto.content.len(), 1);
        match &dto.content[0] {
            ContentBlockDto::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("Expected Text block"),
        }
    }
}
