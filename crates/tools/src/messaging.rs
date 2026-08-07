//! Messaging tools: send_message.
//!
//! Sends a text message to a registered channel (Slack, Discord, Telegram,
//! etc.) through `opensquilla-channels`' [`ChannelManager`].
//!
//! Channels must be registered on the manager before the tool is invoked
//! (e.g. by calling `manager.register(handle)` or
//! `manager.init_channel(config)` during bootstrap). The tool resolves the
//! channel by its `channel_id` and delivers via the channel's real
//! `send_message` implementation so delivery failures surface to the caller
//! (rather than the fire-and-forget `ChannelManager::send`).

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_channels::Channel;
use opensquilla_channels::OutgoingMessage;
use opensquilla_channels::manager::ChannelManager;
use serde_json::Value;
use std::collections::HashMap;

/// Tool for sending a message to a registered channel.
pub struct SendMessageTool {
    manager: ChannelManager,
}

impl SendMessageTool {
    /// Create a new send_message tool backed by the given channel manager.
    pub fn new(manager: ChannelManager) -> Self {
        Self { manager }
    }
}

impl Default for SendMessageTool {
    fn default() -> Self {
        Self::new(ChannelManager::new())
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "send_message",
                concat!(
                    "Send a text message to a registered channel (e.g. Slack, Discord, Telegram). ",
                    "The channel must already be registered and enabled.",
                ),
                HashMap::from([
                    (
                        "channel_id".to_string(),
                        ParameterDefinition::required_string(
                            "The registered channel ID to send to",
                        ),
                    ),
                    (
                        "text".to_string(),
                        ParameterDefinition::required_string("The message text to send"),
                    ),
                    (
                        "thread_id".to_string(),
                        ParameterDefinition::string("Optional thread ID to send the message into"),
                    ),
                ]),
            )
            .category("messaging")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let channel_id = params["channel_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'channel_id'"))?
            .to_string();
        let text = params["text"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'text'"))?
            .to_string();
        if text.trim().is_empty() {
            return Err(ToolError::invalid_args("'text' must not be empty"));
        }
        let thread_id = params["thread_id"].as_str().map(String::from);

        // Resolve the channel by its registered ID.
        let handle = self.manager.get(&channel_id).ok_or_else(|| {
            ToolError::new(
                "CHANNEL_NOT_FOUND",
                format!(
                    "Channel '{}' is not registered. Available channels: {}",
                    channel_id,
                    self.manager.list_channels().join(", ")
                ),
            )
        })?;

        // Build the outgoing message. The channel type is derived from the
        // registered handle, so the caller does not need to supply it.
        let channel_type = handle.channel_type();
        let mut message =
            OutgoingMessage::new(channel_id.clone(), channel_type.clone(), text.clone());
        message.thread_id = thread_id.clone();

        // Deliver through the channel's real send_message so delivery errors
        // (auth, rate limit, network) are returned to the caller instead of
        // being silently logged by the fire-and-forget manager path.
        handle.send_message(&message).await.map_err(|e| {
            ToolError::new(
                "SEND_FAILED",
                format!("Failed to send to '{}': {}", channel_id, e),
            )
        })?;

        let data = serde_json::json!({
            "channel_id": channel_id,
            "channel_type": channel_type,
            "thread_id": thread_id,
            "message_id": message.id.to_string(),
            "text_length": text.len(),
        });

        Ok(ToolOutput::success_with_data(
            format!("Sent message to channel '{}'", channel_id),
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_send_message_channel_not_registered() {
        let tool = SendMessageTool::default();
        let result = tool
            .execute(serde_json::json!({
                "channel_id": "no-such-channel",
                "text": "hello",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "CHANNEL_NOT_FOUND");
    }
}
