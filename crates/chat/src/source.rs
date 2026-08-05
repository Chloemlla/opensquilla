use serde::{Deserialize, Serialize};

/// Where a conversation/chat session came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionSource {
    /// Interactive terminal.
    Terminal,
    /// WebSocket connection.
    WebSocket,
    /// HTTP request.
    Http,
    /// Messaging channel (Slack, Telegram, etc.).
    Channel,
    /// Programmatic API call.
    Api,
}

impl SessionSource {
    /// A stable string identifier for this source.
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionSource::Terminal => "terminal",
            SessionSource::WebSocket => "websocket",
            SessionSource::Http => "http",
            SessionSource::Channel => "channel",
            SessionSource::Api => "api",
        }
    }
}

impl std::fmt::Display for SessionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Default for SessionSource {
    fn default() -> Self {
        SessionSource::Api
    }
}
