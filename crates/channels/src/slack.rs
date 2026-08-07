//! Slack channel adapter — Events API webhooks, Socket Mode, and the REST API.
//!
//! Slack bots can receive events two ways and send messages via the Web API.
//! This module implements the raw protocol without the `slack-sdk` crate:
//!
//! 1. **Events API**: incoming messages arrive as signed HTTP POSTs to a
//!    public webhook URL. Signature verification, challenge answering and
//!    payload parsing live in [`crate::webhook`]; this module builds the
//!    axum route ([`SlackChannel::webhook_route`]) and reuses the parser.
//! 2. **Socket Mode**: for local development behind NAT, the app connects a
//!    persistent WebSocket obtained from `apps.connections.open`. Events are
//!    delivered as envelopes that must be acknowledged.
//! 3. **Web API**: outbound messages use `chat.postMessage` with Slack
//!    Blocks, `conversations.history` / `users.info` for context, and file
//!    upload via `files.upload`.
//!
//! Token management: [`SlackOAuth`] drives the `oauth.v2` install/refresh
//! flow and [`SlackTokenStore`] caches the resulting credentials.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, OutgoingMessage,
};
use crate::webhook::{WebhookError, WebhookMethod, WebhookRoute};
use chrono::{DateTime, Utc};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use regex::Regex;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::{debug, error, info, warn};

/// The WebSocket stream type used by Socket Mode.
pub type SlackWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Default Slack Web API base.
pub const DEFAULT_API_BASE: &str = "https://slack.com/api";
/// Default Socket Mode WebSocket base (ticket URLs come from the API).
pub const DEFAULT_SOCKET_BASE: &str = "https://wss-primary.slack.com/link";

const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;

// ---------------------------------------------------------------------------
// OAuth token management
// ---------------------------------------------------------------------------

/// A set of OAuth v2 credentials for a Slack app installation.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SlackOAuthTokens {
    /// Bot user access token (`xoxb-...`) used to call the Web API.
    pub bot_token: Option<String>,
    /// App-level token (`xapp-...`) used for Socket Mode.
    pub app_token: Option<String>,
    /// Refresh token, when the app is configured with token rotation.
    pub refresh_token: Option<String>,
    /// Expiry of the access token; `None` for long-lived legacy tokens.
    pub expires_at: Option<DateTime<Utc>>,
    /// The workspace the app was installed into.
    pub team_id: Option<String>,
    /// The workspace display name.
    pub team_name: Option<String>,
    /// The Slack app id.
    pub app_id: Option<String>,
    /// Granted OAuth scopes.
    pub scope: Option<String>,
}

impl SlackOAuthTokens {
    /// Whether a usable bot token is present.
    pub fn has_bot_token(&self) -> bool {
        self.bot_token.as_deref().is_some_and(|t| !t.is_empty())
    }

    /// Whether a usable app token is present.
    pub fn has_app_token(&self) -> bool {
        self.app_token.as_deref().is_some_and(|t| !t.is_empty())
    }

    /// Whether the stored token is still valid by expiry.
    pub fn is_expired(&self) -> bool {
        self.expires_at.map(|e| Utc::now() >= e).unwrap_or(false)
    }
}

/// A thread-safe cache of OAuth tokens with expiry checking.
#[derive(Clone, Default)]
pub struct SlackTokenStore {
    inner: Arc<Mutex<SlackOAuthTokens>>,
}

impl SlackTokenStore {
    /// Create a store seeded with the given tokens.
    pub fn new(tokens: SlackOAuthTokens) -> Self {
        Self {
            inner: Arc::new(Mutex::new(tokens)),
        }
    }

    /// Create an empty store.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The current cached tokens.
    pub async fn get(&self) -> SlackOAuthTokens {
        self.inner.lock().await.clone()
    }

    /// The bot token, if any and not expired.
    pub async fn bot_token(&self) -> Option<String> {
        let guard = self.inner.lock().await;
        if guard.is_expired() {
            None
        } else {
            guard.bot_token.clone()
        }
    }

    /// The app-level token, if any.
    pub async fn app_token(&self) -> Option<String> {
        self.inner.lock().await.app_token.clone()
    }

    /// Atomically replace the cached tokens (e.g. after a refresh).
    pub async fn set(&self, tokens: SlackOAuthTokens) {
        *self.inner.lock().await = tokens;
    }

    /// Replace just the bot token (used when a new token arrives).
    pub async fn set_bot_token(&self, token: impl Into<String>) {
        let mut guard = self.inner.lock().await;
        guard.bot_token = Some(token.into());
        guard.expires_at = None;
    }
}

/// OAuth v2 install / refresh flow helpers.
#[derive(Debug, Clone)]
pub struct SlackOAuth {
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    scopes: String,
}

impl SlackOAuth {
    /// Create an OAuth helper from an installed app's credentials.
    pub fn new(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        redirect_uri: impl Into<String>,
        scopes: impl Into<String>,
    ) -> Self {
        Self {
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            redirect_uri: redirect_uri.into(),
            scopes: scopes.into(),
        }
    }

    /// Build the "Add to Slack" authorization URL.
    pub fn authorize_url(&self, state: &str) -> String {
        format!(
            "https://slack.com/oauth/v2/authorize?client_id={}&scope={}&redirect_uri={}&state={}",
            self.client_id,
            urlencode(&self.scopes),
            urlencode(&self.redirect_uri),
            urlencode(state)
        )
    }

    /// Exchange an authorization `code` for tokens.
    pub async fn exchange_code(
        &self,
        client: &reqwest::Client,
        code: &str,
    ) -> Result<SlackOAuthTokens, String> {
        let resp = client
            .post(format!("{DEFAULT_API_BASE}/oauth.v2.access"))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("code", code),
                ("redirect_uri", self.redirect_uri.as_str()),
            ])
            .send()
            .await
            .map_err(|e| format!("Slack OAuth exchange request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Slack OAuth exchange parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(SlackOAuthTokens {
                bot_token: body
                    .get("access_token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                app_token: body
                    .get("app_token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                refresh_token: body
                    .get("refresh_token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                expires_at: body
                    .get("expires_in")
                    .and_then(|v| v.as_i64())
                    .map(|secs| Utc::now() + chrono::Duration::seconds(secs)),
                team_id: body
                    .get("team")
                    .and_then(|t| t.get("id"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
                team_name: body
                    .get("team")
                    .and_then(|t| t.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
                app_id: body
                    .get("app_id")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                scope: body.get("scope").and_then(|v| v.as_str()).map(String::from),
            })
        } else {
            Err(format!(
                "Slack OAuth error: {}",
                body["error"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Refresh an expiring token via `oauth.v2.refresh`.
    pub async fn refresh(
        &self,
        client: &reqwest::Client,
        refresh_token: &str,
    ) -> Result<SlackOAuthTokens, String> {
        let resp = client
            .post(format!("{DEFAULT_API_BASE}/oauth.v2.refresh"))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", refresh_token),
            ])
            .send()
            .await
            .map_err(|e| format!("Slack OAuth refresh request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Slack OAuth refresh parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(SlackOAuthTokens {
                bot_token: body
                    .get("access_token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                app_token: None,
                refresh_token: body
                    .get("refresh_token")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                expires_at: body
                    .get("expires_in")
                    .and_then(|v| v.as_i64())
                    .map(|secs| Utc::now() + chrono::Duration::seconds(secs)),
                ..SlackOAuthTokens::default()
            })
        } else {
            Err(format!(
                "Slack OAuth refresh error: {}",
                body["error"].as_str().unwrap_or("unknown")
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Block kit builder
// ---------------------------------------------------------------------------

/// Helpers for building Slack Block Kit payloads.
///
/// All methods return [`Value`] blocks that can be passed to
/// [`SlackClient::post_message`]. A typical message is a list of blocks.
pub struct SlackBlockBuilder;

impl SlackBlockBuilder {
    /// A `section` block with `mrkdwn` text.
    pub fn section_mrkdwn(text: impl Into<String>) -> Value {
        json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text.into(),
            }
        })
    }

    /// A `section` block with plain text.
    pub fn section_plain(text: impl Into<String>) -> Value {
        json!({
            "type": "section",
            "text": {
                "type": "plain_text",
                "text": text.into(),
            }
        })
    }

    /// A `section` block with an array of `fields`.
    pub fn section_fields(text: impl Into<String>, fields: Vec<Value>) -> Value {
        json!({
            "type": "section",
            "text": Self::mrkdwn(text),
            "fields": fields,
        })
    }

    /// A `divider` block.
    pub fn divider() -> Value {
        json!({ "type": "divider" })
    }

    /// A `context` block carrying small elements.
    pub fn context(elements: Vec<Value>) -> Value {
        json!({ "type": "context", "elements": elements })
    }

    /// An `actions` block of interactive elements.
    pub fn actions(elements: Vec<Value>) -> Value {
        json!({ "type": "actions", "elements": elements })
    }

    /// A `header` block (Slack requires the header text to be plain_text).
    pub fn header(text: impl Into<String>) -> Value {
        json!({
            "type": "header",
            "text": Self::plain_text(text),
        })
    }

    /// An `image` block.
    pub fn image(image_url: impl Into<String>, alt_text: impl Into<String>) -> Value {
        json!({
            "type": "image",
            "image_url": image_url.into(),
            "alt_text": alt_text.into(),
        })
    }

    /// An `mrkdwn` text element.
    pub fn mrkdwn(text: impl Into<String>) -> Value {
        json!({ "type": "mrkdwn", "text": text.into() })
    }

    /// A `plain_text` text element.
    pub fn plain_text(text: impl Into<String>) -> Value {
        json!({ "type": "plain_text", "text": text.into() })
    }

    /// A button element.
    pub fn button(
        text: impl Into<String>,
        action_id: impl Into<String>,
        value: impl Into<String>,
    ) -> Value {
        json!({
            "type": "button",
            "text": Self::plain_text(text),
            "action_id": action_id.into(),
            "value": value.into(),
        })
    }

    /// Build the default block list for an outgoing message: a section with
    /// the message text followed by one image block per attachment.
    pub fn from_message(message: &OutgoingMessage) -> Vec<Value> {
        let mut blocks = vec![Self::section_mrkdwn(&message.text)];
        for att in &message.attachments {
            if let Some(url) = &att.url {
                blocks.push(Self::image(url.clone(), att.attachment_type.clone()));
            }
        }
        blocks
    }
}

// ---------------------------------------------------------------------------
// REST client
// ---------------------------------------------------------------------------

/// A small, token-aware Slack Web API client.
#[derive(Clone)]
pub struct SlackClient {
    http: reqwest::Client,
    tokens: SlackTokenStore,
    api_base: String,
}

impl SlackClient {
    /// Create a client backed by a token store.
    pub fn new(tokens: SlackTokenStore) -> Self {
        Self::with_base(tokens, DEFAULT_API_BASE.to_string())
    }

    /// Create a client with a custom API base (for proxies / enterprise grids).
    pub fn with_base(tokens: SlackTokenStore, api_base: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client builder cannot fail");
        Self {
            http,
            tokens,
            api_base,
        }
    }

    /// Create a client authenticated with a bare bot token.
    pub fn with_bot_token(token: impl Into<String>) -> Self {
        Self::new(SlackTokenStore::new(SlackOAuthTokens {
            bot_token: Some(token.into()),
            ..SlackOAuthTokens::default()
        }))
    }

    /// The backing token store (allows swapping tokens at runtime).
    pub fn token_store(&self) -> &SlackTokenStore {
        &self.tokens
    }

    async fn api_url(&self, method: &str) -> String {
        format!("{}/{}", self.api_base, method)
    }

    /// Perform an authenticated JSON POST to a Web API method.
    async fn authed_post(&self, method: &str, body: Value) -> Result<Value, String> {
        let token = self
            .tokens
            .bot_token()
            .await
            .ok_or("No bot token configured")?;
        let resp = self
            .http
            .post(self.api_url(method).await)
            .header("Authorization", format!("Bearer {token}"))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Slack {method} request: {e}"))?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| format!("Slack {method} parse: {e}"))?;
        if value["ok"].as_bool().unwrap_or(false) {
            Ok(value)
        } else {
            Err(format!(
                "Slack {method} error: {}",
                value["error"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Verify the bot token with `auth.test`.
    pub async fn auth_test(&self) -> Result<Value, String> {
        self.authed_post("auth.test", json!({})).await
    }

    /// Post a message to a channel, optionally with blocks and a thread reply.
    ///
    /// `blocks` may be empty, in which case only the plain text is sent.
    pub async fn post_message(
        &self,
        channel: &str,
        text: &str,
        blocks: Vec<Value>,
        thread_ts: Option<&str>,
        mrkdwn: bool,
    ) -> Result<Value, String> {
        let mut payload = json!({
            "channel": channel,
            "text": text,
            "mrkdwn": mrkdwn,
        });
        if !blocks.is_empty() {
            payload["blocks"] = Value::Array(blocks);
        }
        if let Some(ts) = thread_ts {
            payload["thread_ts"] = Value::String(ts.to_string());
        }
        self.authed_post("chat.postMessage", payload).await
    }

    /// Post an ephemeral message visible only to one user.
    pub async fn post_ephemeral(
        &self,
        channel: &str,
        user: &str,
        text: &str,
        blocks: Vec<Value>,
    ) -> Result<Value, String> {
        let mut payload = json!({
            "channel": channel,
            "user": user,
            "text": text,
        });
        if !blocks.is_empty() {
            payload["blocks"] = Value::Array(blocks);
        }
        self.authed_post("chat.postEphemeral", payload).await
    }

    /// Update a previously posted message.
    pub async fn update_message(
        &self,
        channel: &str,
        ts: &str,
        text: &str,
        blocks: Vec<Value>,
    ) -> Result<Value, String> {
        let mut payload = json!({
            "channel": channel,
            "ts": ts,
            "text": text,
        });
        if !blocks.is_empty() {
            payload["blocks"] = Value::Array(blocks);
        }
        self.authed_post("chat.update", payload).await
    }

    /// Delete a message by its timestamp.
    pub async fn delete_message(&self, channel: &str, ts: &str) -> Result<(), String> {
        self.authed_post("chat.delete", json!({ "channel": channel, "ts": ts }))
            .await?;
        Ok(())
    }

    /// Add a reaction emoji to a message.
    pub async fn add_reaction(&self, channel: &str, ts: &str, emoji: &str) -> Result<(), String> {
        self.authed_post(
            "reactions.add",
            json!({ "channel": channel, "timestamp": ts, "name": emoji }),
        )
        .await?;
        Ok(())
    }

    /// Fetch the most recent messages in a channel.
    pub async fn conversations_history(
        &self,
        channel: &str,
        limit: u32,
        cursor: Option<&str>,
    ) -> Result<Value, String> {
        let mut params = json!({ "channel": channel, "limit": limit });
        if let Some(c) = cursor {
            params["cursor"] = Value::String(c.to_string());
        }
        self.authed_post("conversations.history", params).await
    }

    /// Fetch the replies in a thread.
    pub async fn conversations_replies(
        &self,
        channel: &str,
        ts: &str,
        limit: u32,
    ) -> Result<Value, String> {
        self.authed_post(
            "conversations.replies",
            json!({ "channel": channel, "ts": ts, "limit": limit }),
        )
        .await
    }

    /// Look up a user's profile.
    pub async fn users_info(&self, user: &str) -> Result<Value, String> {
        self.authed_post("users.info", json!({ "user": user }))
            .await
    }

    /// Upload a file to a channel using multipart form data.
    pub async fn files_upload(
        &self,
        channel: &str,
        file_name: &str,
        content: &[u8],
        mime: Option<&str>,
    ) -> Result<Value, String> {
        let token = self
            .tokens
            .bot_token()
            .await
            .ok_or("No bot token configured")?;
        let part = reqwest::multipart::Part::bytes(content.to_vec())
            .file_name(file_name.to_string())
            .mime_str(mime.unwrap_or("application/octet-stream"))
            .map_err(|e| format!("Mime parse: {e}"))?;
        let form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("channels", channel.to_string());
        let resp = self
            .http
            .post(self.api_url("files.upload").await)
            .header("Authorization", format!("Bearer {token}"))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("Slack files.upload request: {e}"))?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| format!("Slack files.upload parse: {e}"))?;
        if value["ok"].as_bool().unwrap_or(false) {
            Ok(value)
        } else {
            Err(format!(
                "Slack files.upload error: {}",
                value["error"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a raw payload to an incoming-webhook URL.
    pub async fn post_webhook(&self, url: &str, payload: Value) -> Result<(), String> {
        let resp = self
            .http
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Slack webhook request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Slack webhook returned {}", resp.status()))
        }
    }

    /// Open a Socket Mode connection and return the WSS URL.
    ///
    /// Uses the app-level token when present, otherwise falls back to the bot
    /// token. Requires the `connections:write` scope.
    pub async fn open_socket_connection(&self) -> Result<String, String> {
        let token = match self.tokens.app_token().await {
            Some(t) => t,
            None => self
                .tokens
                .bot_token()
                .await
                .ok_or("No app or bot token for Socket Mode")?,
        };
        let resp = self
            .http
            .post(self.api_url("apps.connections.open").await)
            .header("Authorization", format!("Bearer {token}"))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| format!("apps.connections.open request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("apps.connections.open parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            body["url"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| "No URL in apps.connections.open response".to_string())
        } else {
            Err(format!(
                "apps.connections.open error: {}",
                body["error"].as_str().unwrap_or("unknown")
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Socket Mode envelope parsing
// ---------------------------------------------------------------------------

/// An envelope received over a Socket Mode WebSocket.
#[derive(Debug, Clone, serde::Deserialize)]
struct SocketEnvelope {
    #[serde(rename = "type")]
    envelope_type: String,
    #[serde(default)]
    envelope_id: Option<String>,
    #[serde(default)]
    payload: Option<Value>,
    #[serde(default)]
    #[allow(dead_code)] // kept for API/serialization compatibility
    error: Option<Value>,
}

/// The type of a Socket Mode envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocketEnvelopeType {
    Hello,
    EventsApi,
    SlashCommands,
    Interactive,
    Unknown,
}

impl SocketEnvelope {
    /// Classify the envelope.
    pub fn kind(&self) -> SocketEnvelopeType {
        match self.envelope_type.as_str() {
            "hello" => SocketEnvelopeType::Hello,
            "events_api" => SocketEnvelopeType::EventsApi,
            "slash_commands" => SocketEnvelopeType::SlashCommands,
            "interactive" => SocketEnvelopeType::Interactive,
            _ => SocketEnvelopeType::Unknown,
        }
    }
}

/// Parse a Socket Mode `events_api` payload into an [`IncomingMessage`].
///
/// The payload has the same shape as an Events API `event_callback`, so it
/// delegates to [`crate::webhook::parse_slack_payload`].
pub fn parse_socket_event(payload: &Value) -> Result<IncomingMessage, WebhookError> {
    crate::webhook::parse_slack_payload(payload)
}

// ---------------------------------------------------------------------------
// Channel adapter
// ---------------------------------------------------------------------------

/// Slack channel adapter backed by the REST API, Events API and Socket Mode.
pub struct SlackChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    tokens: SlackTokenStore,
    webhook_url: Arc<Mutex<Option<String>>>,
    signing_secret: Option<String>,
    socket_mode: bool,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    socket_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl SlackChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let bot_token = config
            .config
            .get("bot_token")
            .and_then(|v| v.as_str())
            .map(String::from);
        let app_token = config
            .config
            .get("app_token")
            .and_then(|v| v.as_str())
            .map(String::from);
        let refresh_token = config
            .config
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(String::from);
        let expires_in = config
            .config
            .get("expires_in")
            .and_then(|v| v.as_i64())
            .map(|secs| Utc::now() + chrono::Duration::seconds(secs));
        let signing_secret = config
            .config
            .get("signing_secret")
            .and_then(|v| v.as_str())
            .map(String::from);
        let webhook_url = config
            .config
            .get("webhook_url")
            .and_then(|v| v.as_str())
            .map(String::from);
        let socket_mode = config
            .config
            .get("socket_mode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let tokens = SlackTokenStore::new(SlackOAuthTokens {
            bot_token,
            app_token,
            refresh_token,
            expires_at: expires_in,
            team_id: config
                .config
                .get("team_id")
                .and_then(|v| v.as_str())
                .map(String::from),
            team_name: config
                .config
                .get("team_name")
                .and_then(|v| v.as_str())
                .map(String::from),
            app_id: config
                .config
                .get("app_id")
                .and_then(|v| v.as_str())
                .map(String::from),
            scope: config
                .config
                .get("scope")
                .and_then(|v| v.as_str())
                .map(String::from),
        });

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

        Ok(Self {
            config,
            client,
            tokens,
            webhook_url: Arc::new(Mutex::new(webhook_url)),
            signing_secret,
            socket_mode,
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
            socket_task: Arc::new(Mutex::new(None)),
        })
    }

    /// Access the token store (for swapping tokens at runtime).
    pub fn token_store(&self) -> &SlackTokenStore {
        &self.tokens
    }

    /// The signing secret used to verify Events API requests.
    pub fn signing_secret(&self) -> Option<&str> {
        self.signing_secret.as_deref()
    }

    /// Whether Socket Mode is enabled in config.
    pub fn socket_mode_enabled(&self) -> bool {
        self.socket_mode
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Push a parsed incoming message onto the internal queue.
    ///
    /// Used by the Socket Mode loop and the webhook route callback.
    pub fn push_incoming(&self, message: IncomingMessage) {
        self.incoming.blocking_lock().push_back(message);
    }

    /// Start the Socket Mode loop in the background.
    pub async fn start_socket_mode(&self) -> Result<(), String> {
        if !self.socket_mode {
            return Ok(());
        }
        {
            let mut running = self.running.lock().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }
        let running = self.running.clone();
        let incoming = self.incoming.clone();
        let client = self.client.clone();
        let tokens = self.tokens.clone();
        let task = tokio::spawn(async move {
            let api_client = SlackClient {
                http: client,
                tokens,
                api_base: DEFAULT_API_BASE.to_string(),
            };
            run_socket_loop(running, incoming, api_client).await;
        });
        *self.socket_task.lock().await = Some(task);
        Ok(())
    }

    /// Stop the Socket Mode loop.
    pub async fn stop_socket_mode(&self) {
        *self.running.lock().await = false;
        if let Some(task) = self.socket_task.lock().await.take() {
            task.abort();
        }
    }

    /// Build the axum [`WebhookRoute`] for the Slack Events API.
    ///
    /// The route verifies the `X-Slack-Signature` header (when a signing
    /// secret is configured), answers `url_verification` challenges, and
    /// forwards parsed messages to [`SlackChannel::push_incoming`].
    pub fn webhook_route(&self, path: impl Into<String>) -> WebhookRoute {
        let incoming = self.incoming.clone();
        let handler = crate::webhook::SlackWebhookHandler::new().on_message(move |msg| {
            incoming.blocking_lock().push_back(msg);
            Ok(())
        });
        let mut route = WebhookRoute::new(path, WebhookMethod::Post, ChannelType::Slack, handler);
        if let Some(secret) = &self.signing_secret {
            route = route.with_secret(secret.clone());
        }
        route
    }

    // -- send helpers ------------------------------------------------------

    async fn send_with_bot(&self, message: &OutgoingMessage) -> Result<(), String> {
        let api = SlackClient {
            http: self.client.clone(),
            tokens: self.tokens.clone(),
            api_base: DEFAULT_API_BASE.to_string(),
        };
        let text = to_mrkdwn(&message.text);
        let blocks = if message.metadata.get("blocks").is_some() {
            message
                .metadata
                .get("blocks")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else {
            SlackBlockBuilder::from_message(message)
        };
        api.post_message(
            &message.channel_id,
            &text,
            blocks,
            message.thread_id.as_deref(),
            true,
        )
        .await?;
        Ok(())
    }

    async fn send_with_webhook(&self, message: &OutgoingMessage) -> Result<(), String> {
        let url = self
            .webhook_url
            .lock()
            .await
            .clone()
            .ok_or("No webhook URL configured")?;
        let payload = json!({
            "text": to_mrkdwn(&message.text),
            "channel": message.channel_id,
        });
        let resp = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Slack webhook error: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Slack webhook returned {}", resp.status()))
        }
    }

    /// Send an ephemeral message to a specific user.
    pub async fn send_ephemeral(
        &self,
        channel: &str,
        user: &str,
        text: &str,
    ) -> Result<(), String> {
        let api = SlackClient {
            http: self.client.clone(),
            tokens: self.tokens.clone(),
            api_base: DEFAULT_API_BASE.to_string(),
        };
        api.post_ephemeral(channel, user, text, Vec::new()).await?;
        Ok(())
    }
}

/// Run the Socket Mode connection loop with reconnect + backoff.
async fn run_socket_loop(
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    api: SlackClient,
) {
    info!("Slack Socket Mode loop starting");
    let mut attempt: u32 = 0;
    loop {
        if !*running.lock().await {
            return;
        }
        let ws_url = match api.open_socket_connection().await {
            Ok(url) => url,
            Err(e) => {
                error!("Slack Socket Mode open failed: {e}");
                attempt = attempt.saturating_add(1).min(10);
                tokio::time::sleep(slack_reconnect_delay(attempt)).await;
                continue;
            }
        };
        run_socket_cycle(running.clone(), incoming.clone(), &ws_url).await;
        attempt = attempt.saturating_add(1).min(10);
        if !*running.lock().await {
            return;
        }
        let delay = slack_reconnect_delay(attempt);
        info!("Slack Socket Mode reconnecting in {}s", delay.as_secs());
        tokio::time::sleep(delay).await;
    }
}

/// Run one Socket Mode WebSocket session.
async fn run_socket_cycle(
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    ws_url: &str,
) {
    let (ws_stream, _) = match connect_async(ws_url).await {
        Ok(pair) => pair,
        Err(e) => {
            error!("Slack Socket Mode connect failed: {e}");
            return;
        }
    };
    let (mut sink, mut read): (
        SplitSink<SlackWsStream, WsMessage>,
        SplitStream<SlackWsStream>,
    ) = ws_stream.split();
    info!("Slack Socket Mode connected");

    while let Some(frame) = read.next().await {
        if !*running.lock().await {
            return;
        }
        match frame {
            Ok(WsMessage::Text(text)) => {
                let envelope: SocketEnvelope = match serde_json::from_str(&text) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("Slack Socket Mode frame parse error: {e}");
                        continue;
                    }
                };
                match envelope.kind() {
                    SocketEnvelopeType::Hello => {
                        info!("Slack Socket Mode hello received");
                    }
                    SocketEnvelopeType::EventsApi => {
                        if let Some(payload) = &envelope.payload {
                            match parse_socket_event(payload) {
                                Ok(msg) => incoming.lock().await.push_back(msg),
                                Err(WebhookError::UnsupportedEvent(t)) if t == "bot_message" => {}
                                Err(e) => {
                                    debug!("Slack Socket Mode event skipped: {e}");
                                }
                            }
                        }
                        ack_socket_envelope(&mut sink, &envelope).await;
                    }
                    SocketEnvelopeType::SlashCommands | SocketEnvelopeType::Interactive => {
                        ack_socket_envelope(&mut sink, &envelope).await;
                    }
                    SocketEnvelopeType::Unknown => {
                        // Legacy envelopes do not require acks.
                        debug!(
                            "Slack Socket Mode unknown envelope {}",
                            envelope.envelope_type
                        );
                    }
                }
            }
            Ok(WsMessage::Ping(payload)) => {
                if sink.send(WsMessage::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Ok(WsMessage::Close(_)) => {
                info!("Slack Socket Mode connection closed");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                warn!("Slack Socket Mode error: {e}");
                break;
            }
        }
    }
}

/// Acknowledge a Socket Mode envelope so Slack keeps delivering.
async fn ack_socket_envelope(
    sink: &mut SplitSink<SlackWsStream, WsMessage>,
    envelope: &SocketEnvelope,
) {
    let Some(envelope_id) = &envelope.envelope_id else {
        return;
    };
    let ack = json!({ "envelope_id": envelope_id, "payload": {} });
    if sink
        .send(WsMessage::Text(ack.to_string().into()))
        .await
        .is_err()
    {
        warn!("Slack Socket Mode ack send failed");
    }
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn slack_reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

// ---------------------------------------------------------------------------
// Markdown formatting
// ---------------------------------------------------------------------------

fn bold_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\*\*([^*]+)\*\*").unwrap())
}

fn link_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap())
}

fn bullet_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^\s*\*\s+").unwrap())
}

/// Convert a subset of GitHub-flavored Markdown to Slack `mrkdwn`.
///
/// Handles `**bold**`, `[text](url)`, and `* item` bullets. Inline code
/// fences and `_italics_` pass through unchanged because Slack uses the same
/// syntax for code and italics.
pub fn to_mrkdwn(text: &str) -> String {
    // Heading lines become bold + prefix.
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str("*");
            out.push_str(rest);
            out.push_str("*\n");
        } else if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str("*");
            out.push_str(rest);
            out.push_str("*\n");
        } else if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str("*");
            out.push_str(rest);
            out.push_str("*\n");
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    let out = bold_re().replace_all(&out, "*$1*");
    let out = link_re().replace_all(&out, "<$2|$1>");
    let out = bullet_re().replace_all(&out, "• ");
    out.into_owned()
}

/// Percent-encode a URL query component.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[async_trait::async_trait]
impl Channel for SlackChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Slack
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        if self.webhook_url.lock().await.is_some() {
            self.send_with_webhook(message).await
        } else {
            self.send_with_bot(message).await
        }
    }

    async fn send_typing(&self, channel_id: &str) -> Result<(), String> {
        // The Slack Web API has no server-driven typing indicator for bots;
        // post an ephemeral placeholder that is immediately replaced by the
        // real response. This keeps the "typing" semantic without polling.
        let token = self
            .tokens
            .bot_token()
            .await
            .ok_or("No bot token configured")?;
        let resp = self
            .client
            .post(format!("{DEFAULT_API_BASE}/chat.postEphemeral"))
            .header("Authorization", format!("Bearer {token}"))
            .json(&json!({ "channel": channel_id, "text": "..." }))
            .send()
            .await
            .map_err(|e| format!("Slack typing request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Typing indicator failed: {}", resp.status()))
        }
    }

    async fn set_webhook(&self, url: &str) -> Result<(), String> {
        *self.webhook_url.lock().await = Some(url.to_string());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> SlackChannel {
        SlackChannel::new(ChannelConfig {
            channel_type: ChannelType::Slack,
            channel_id: "C123".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({ "bot_token": "xoxb-test" }),
        })
        .unwrap()
    }

    #[test]
    fn test_mrkdwn_bold_and_links() {
        assert_eq!(to_mrkdwn("**bold** here"), "*bold* here");
        assert_eq!(to_mrkdwn("[site](https://x.com)"), "<https://x.com|site>");
        assert_eq!(to_mrkdwn("* item"), "• item");
    }

    #[test]
    fn test_mrkdwn_heading() {
        assert_eq!(to_mrkdwn("# Title"), "*Title*\n");
        assert_eq!(to_mrkdwn("## Sub"), "*Sub*\n");
        assert_eq!(to_mrkdwn("### Deep"), "*Deep*\n");
    }

    #[test]
    fn test_urlencode() {
        assert_eq!(urlencode("a b&c=d"), "a+b%26c%3Dd");
        assert_eq!(urlencode("simple"), "simple");
    }

    #[test]
    fn test_blocks_from_message() {
        let msg = OutgoingMessage::new("C1".to_string(), ChannelType::Slack, "hi".to_string());
        let blocks = SlackBlockBuilder::from_message(&msg);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "section");
        assert_eq!(blocks[0]["text"]["text"], "hi");
    }

    #[test]
    fn test_oauth_authorize_url() {
        let oauth = SlackOAuth::new("id", "secret", "http://cb", "chat:write");
        let url = oauth.authorize_url("state1");
        assert!(url.starts_with("https://slack.com/oauth/v2/authorize?"));
        assert!(url.contains("scope=chat%3Awrite"));
        assert!(url.contains("state=state1"));
    }

    #[test]
    fn test_token_store_roundtrip() {
        let store = SlackTokenStore::new(SlackOAuthTokens {
            bot_token: Some("xoxb-1".to_string()),
            app_token: Some("xapp-1".to_string()),
            ..SlackOAuthTokens::default()
        });
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            assert_eq!(store.bot_token().await.as_deref(), Some("xoxb-1"));
            assert_eq!(store.app_token().await.as_deref(), Some("xapp-1"));
            store.set_bot_token("xoxb-2").await;
            assert_eq!(store.bot_token().await.as_deref(), Some("xoxb-2"));
        });
    }

    #[test]
    fn test_expired_token_rejected() {
        let mut tokens = SlackOAuthTokens::default();
        tokens.bot_token = Some("xoxb".to_string());
        tokens.expires_at = Some(Utc::now() - chrono::Duration::seconds(10));
        assert!(tokens.is_expired());
        let store = SlackTokenStore::new(tokens);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            assert_eq!(store.bot_token().await, None);
        });
    }

    #[test]
    fn test_parse_socket_event() {
        let payload = json!({
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C456",
                "user": "U789",
                "text": "hello socket",
                "ts": "1625241600.000001"
            }
        });
        let msg = parse_socket_event(&payload).unwrap();
        assert_eq!(msg.channel_id, "C456");
        assert_eq!(msg.text, "hello socket");
    }

    #[test]
    fn test_envelope_classification() {
        let env: SocketEnvelope = serde_json::from_value(json!({
            "type": "events_api",
            "envelope_id": "e1",
            "payload": {}
        }))
        .unwrap();
        assert_eq!(env.kind(), SocketEnvelopeType::EventsApi);
        assert_eq!(env.envelope_id.as_deref(), Some("e1"));
    }

    #[test]
    fn test_reconnect_backoff_capped() {
        assert_eq!(slack_reconnect_delay(0).as_secs(), 1);
        assert_eq!(slack_reconnect_delay(1).as_secs(), 2);
        assert!(slack_reconnect_delay(20).as_secs() <= MAX_RECONNECT_DELAY_SECS);
    }
}
