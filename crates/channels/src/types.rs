use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingMessage {
    pub id: Uuid,
    pub channel_id: String,
    pub channel_type: ChannelType,
    pub user_id: String,
    pub user_name: Option<String>,
    pub text: String,
    pub thread_id: Option<String>,
    pub attachments: Vec<MessageAttachment>,
    pub timestamp: DateTime<Utc>,
    pub raw: serde_json::Value,
    /// Provider-normalized metadata (conversation_kind, is_group, interaction_type, …).
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Whether the transport authenticated the sender's provenance.
    #[serde(default)]
    pub provenance_authenticated: bool,
    /// Whether the event is explicitly addressed to the bot (adapter hook).
    #[serde(default)]
    pub sender_is_group_mentioned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutgoingMessage {
    pub id: Uuid,
    pub channel_id: String,
    pub channel_type: ChannelType,
    pub text: String,
    pub thread_id: Option<String>,
    pub attachments: Vec<MessageAttachment>,
    pub metadata: serde_json::Value,
}

impl OutgoingMessage {
    pub fn new(channel_id: String, channel_type: ChannelType, text: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            channel_id,
            channel_type,
            text,
            thread_id: None,
            attachments: Vec::new(),
            metadata: serde_json::Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAttachment {
    pub attachment_type: String,
    pub url: Option<String>,
    pub data: Option<serde_json::Value>,
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    Slack,
    Discord,
    Telegram,
    Feishu,
    DingTalk,
    QQ,
    WeCom,
    Matrix,
    MSTeams,
    Terminal,
    WebSocket,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub channel_type: ChannelType,
    pub channel_id: String,
    pub name: String,
    pub enabled: bool,
    pub config: serde_json::Value,
}

#[async_trait::async_trait]
pub trait Channel: Send + Sync {
    fn channel_type(&self) -> ChannelType;
    fn channel_id(&self) -> &str;
    fn name(&self) -> &str;

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String>;
    async fn send_typing(&self, channel_id: &str) -> Result<(), String>;
    async fn set_webhook(&self, url: &str) -> Result<(), String>;
}

/// Reference-counted channel handle
pub type ChannelHandle = Arc<dyn Channel>;
