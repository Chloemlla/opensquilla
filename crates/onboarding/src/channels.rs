use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Configuration for a messaging channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSetup {
    /// The channel type (e.g., "discord", "slack", "telegram").
    pub channel_type: String,
    /// User-friendly display name.
    pub display_name: String,
    /// Whether this channel is enabled.
    pub enabled: bool,
    /// Channel-specific settings (tokens, webhooks, etc.).
    pub settings: HashMap<String, String>,
    /// Instructions for setting up this channel.
    pub setup_instructions: Vec<String>,
}

/// Discover available channels.
pub fn discover_channels() -> Vec<ChannelSetup> {
    vec![
        ChannelSetup {
            channel_type: "discord".to_string(),
            display_name: "Discord".to_string(),
            enabled: false,
            settings: HashMap::new(),
            setup_instructions: vec![
                "1. Create a Discord application at https://discord.com/developers/applications".to_string(),
                "2. Go to the Bot section and create a bot".to_string(),
                "3. Copy the bot token".to_string(),
                "4. Invite the bot to your server using the OAuth2 URL generator".to_string(),
                "5. Set the bot token in config: opensquilla config set channel.discord.token <token>".to_string(),
            ],
        },
        ChannelSetup {
            channel_type: "slack".to_string(),
            display_name: "Slack".to_string(),
            enabled: false,
            settings: HashMap::new(),
            setup_instructions: vec![
                "1. Create a Slack app at https://api.slack.com/apps".to_string(),
                "2. Enable Event Subscriptions and add the bot token scope".to_string(),
                "3. Install the app to your workspace".to_string(),
                "4. Copy the Bot User OAuth Token".to_string(),
                "5. Set the token in config: opensquilla config set channel.slack.token <token>".to_string(),
            ],
        },
        ChannelSetup {
            channel_type: "telegram".to_string(),
            display_name: "Telegram".to_string(),
            enabled: false,
            settings: HashMap::new(),
            setup_instructions: vec![
                "1. Start a chat with @BotFather on Telegram".to_string(),
                "2. Send /newbot and follow the prompts".to_string(),
                "3. Copy the bot token from BotFather".to_string(),
                "4. Set the token in config: opensquilla config set channel.telegram.token <token>".to_string(),
            ],
        },
        ChannelSetup {
            channel_type: "matrix".to_string(),
            display_name: "Matrix".to_string(),
            enabled: false,
            settings: HashMap::new(),
            setup_instructions: vec![
                "1. Register a bot account on a Matrix homeserver".to_string(),
                "2. Obtain the access token for the bot account".to_string(),
                "3. Set the config values: homeserver, user_id, access_token".to_string(),
            ],
        },
        ChannelSetup {
            channel_type: "web".to_string(),
            display_name: "Web Interface".to_string(),
            enabled: false,
            settings: HashMap::new(),
            setup_instructions: vec![
                "1. Enable the web channel in config".to_string(),
                "2. The web interface will be available at the configured port".to_string(),
            ],
        },
    ]
}

/// Get a specific channel by type.
pub fn get_channel(channel_type: &str) -> Option<ChannelSetup> {
    discover_channels()
        .into_iter()
        .find(|c| c.channel_type == channel_type)
}