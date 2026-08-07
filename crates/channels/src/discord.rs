//! Discord channel adapter — Gateway WebSocket + raw REST API.
//!
//! Discord bots receive events over a persistent WebSocket Gateway and send
//! messages via the REST API. This module implements the raw protocol without
//! the `serenity` / `twilight` SDK crates:
//!
//! 1. **Gateway**: `GET /gateway/bot` returns a `wss://...` URL.
//! 2. **Identify**: after connecting and receiving `op:10` Hello, the client
//!    sends `op:2` with the bot token, intents and client properties. On
//!    disconnect it may `op:6` Resume with its `session_id` and last sequence.
//! 3. **Heartbeat**: the client sends `op:1` every `heartbeat_interval`; the
//!    server replies `op:11`.
//! 4. **Dispatch**: `op:0` events (e.g. `MESSAGE_CREATE`, `READY`) carry
//!    payloads; the `s` field is the sequence used for resuming.
//! 5. **Reconnect**: `op:7` requests reconnect; `op:9` signals an invalid
//!    session (resume if `d == true`, otherwise re-identify).
//!
//! Outbound messages use `POST /channels/{channel_id}/messages` with the bot
//! token. REST calls are throttled by [`DiscordRateLimiter`] per the API
//! guidelines (global 50 req/s, per-route buckets, 429 `Retry-After`).

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use chrono::{DateTime, Utc};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// The WebSocket stream type used by the Discord Gateway.
pub type DiscordWsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub const DEFAULT_API_BASE: &str = "https://discord.com/api/v10";
/// Fallback heartbeat interval; overridden by `op:10`.
const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 41_250;
const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;
/// Minimum spacing between REST requests (Discord global limit is 50 req/s).
const GLOBAL_MIN_SPACING: Duration = Duration::from_millis(20);

// Gateway intents (bitmask).
/// Guild / channel lifecycle events.
pub const INTENT_GUILDS: u64 = 1 << 0;
/// Guild member events.
pub const INTENT_GUILD_MEMBERS: u64 = 1 << 1;
/// Guild message events.
pub const INTENT_GUILD_MESSAGES: u64 = 1 << 9;
/// Guild message reaction events.
pub const INTENT_GUILD_MESSAGE_REACTIONS: u64 = 1 << 10;
/// Direct message events.
pub const INTENT_DIRECT_MESSAGES: u64 = 1 << 12;
/// Message content (required since the 2022 privileged-intent change).
pub const INTENT_MESSAGE_CONTENT: u64 = 1 << 15;

/// An envelope received from the Discord Gateway.
#[derive(Debug, Clone, serde::Deserialize)]
struct GatewayEnvelope {
    op: i64,
    #[serde(default)]
    s: Option<u64>,
    #[serde(default)]
    t: Option<String>,
    #[serde(default)]
    d: Option<Value>,
}

/// Resume state captured from the `READY` dispatch.
#[derive(Debug, Clone)]
struct DiscordSession {
    session_id: String,
    last_seq: u64,
    #[allow(dead_code)]
    resume_url: String,
}

// ---------------------------------------------------------------------------
// Rate limiting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BucketState {
    limit: u32,
    remaining: u32,
    reset_at: Instant,
}

impl BucketState {
    fn new() -> Self {
        Self {
            limit: u32::MAX,
            remaining: u32::MAX,
            reset_at: Instant::now(),
        }
    }
}

#[derive(Debug, Default)]
struct RateLimiterState {
    global_last: Option<Instant>,
    buckets: HashMap<String, BucketState>,
}

/// A per-route REST rate limiter honoring Discord `x-ratelimit-*` headers and
/// 429 `Retry-After` responses.
#[derive(Clone, Default)]
pub struct DiscordRateLimiter {
    inner: Arc<Mutex<RateLimiterState>>,
}

impl DiscordRateLimiter {
    /// Create an empty rate limiter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wait until a request on `route` is allowed (global spacing + bucket).
    pub async fn acquire(&self, route: &str) {
        loop {
            let need_sleep = {
                let mut state = self.inner.lock().await;
                let now = Instant::now();
                if let Some(last) = state.global_last {
                    let elapsed = now.duration_since(last);
                    if elapsed < GLOBAL_MIN_SPACING {
                        drop(state);
                        tokio::time::sleep(GLOBAL_MIN_SPACING - elapsed).await;
                        continue;
                    }
                }
                state.global_last = Some(now);
                let bucket = state
                    .buckets
                    .entry(route.to_string())
                    .or_insert_with(BucketState::new);
                if bucket.remaining == 0 {
                    let wait = bucket.reset_at.saturating_duration_since(now);
                    if !wait.is_zero() {
                        drop(state);
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                }
                false
            };
            let _ = need_sleep;
            break;
        }
    }

    /// Record response headers for `route` to update the bucket.
    pub async fn record_response(&self, route: &str, headers: &HeaderMap) {
        let limit = parse_header_u32(headers, "x-ratelimit-limit");
        let remaining = parse_header_u32(headers, "x-ratelimit-remaining");
        let reset_epoch = parse_header_f64(headers, "x-ratelimit-reset");
        if limit.is_none() && remaining.is_none() && reset_epoch.is_none() {
            return;
        }
        let mut state = self.inner.lock().await;
        let bucket = state
            .buckets
            .entry(route.to_string())
            .or_insert_with(BucketState::new);
        if let Some(l) = limit {
            bucket.limit = l;
        }
        if let Some(r) = remaining {
            bucket.remaining = r;
        }
        if let Some(epoch) = reset_epoch {
            bucket.reset_at = now_or_dur(epoch);
        }
    }

    /// Handle a 429 response: wait `Retry-After` seconds.
    pub async fn handle_rate_limited(&self, headers: &HeaderMap) {
        let retry_after = parse_header_f64(headers, "retry-after").unwrap_or(1.0);
        let wait = Duration::from_secs_f64(retry_after.max(0.05));
        warn!("Discord rate limited; waiting {}ms", wait.as_millis());
        tokio::time::sleep(wait).await;
    }
}

fn parse_header_u32(headers: &HeaderMap, name: &str) -> Option<u32> {
    headers
        .get(name)
        .and_then(|v: &HeaderValue| v.to_str().ok())
        .and_then(|v| v.parse::<u32>().ok())
}

fn parse_header_f64(headers: &HeaderMap, name: &str) -> Option<f64> {
    headers
        .get(name)
        .and_then(|v: &HeaderValue| v.to_str().ok())
        .and_then(|v| v.parse::<f64>().ok())
}

/// Convert a `x-ratelimit-reset` unix epoch into an [`Instant`] (or now if
/// the epoch is in the past).
fn now_or_dur(epoch: f64) -> Instant {
    let now = Utc::now().timestamp() as f64;
    let diff = (epoch - now).max(0.0);
    Instant::now() + Duration::from_secs_f64(diff)
}

// ---------------------------------------------------------------------------
// Channel adapter
// ---------------------------------------------------------------------------

/// Discord channel adapter using the Gateway WebSocket for events and the
/// REST API for outbound messages.
pub struct DiscordChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    bot_token: String,
    application_id: Option<String>,
    api_base: String,
    intents: u64,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    session: Arc<Mutex<Option<DiscordSession>>>,
    rate_limiter: DiscordRateLimiter,
    gateway_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl DiscordChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let bot_token = config
            .config
            .get("bot_token")
            .and_then(|v| v.as_str())
            .ok_or("No bot_token configured for Discord")?
            .to_string();
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .to_string();
        let application_id = config
            .config
            .get("application_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let intents = config
            .config
            .get("intents")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(Self::default_intents);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
        Ok(Self {
            config,
            client,
            bot_token,
            application_id,
            api_base,
            intents,
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
            session: Arc::new(Mutex::new(None)),
            rate_limiter: DiscordRateLimiter::new(),
            gateway_task: Arc::new(Mutex::new(None)),
        })
    }

    /// The default intent mask: guilds, guild messages + reactions, direct
    /// messages and message content.
    pub fn default_intents() -> u64 {
        INTENT_GUILDS
            | INTENT_GUILD_MEMBERS
            | INTENT_GUILD_MESSAGES
            | INTENT_GUILD_MESSAGE_REACTIONS
            | INTENT_DIRECT_MESSAGES
            | INTENT_MESSAGE_CONTENT
    }

    /// Fetch the Gateway WebSocket URL.
    pub async fn get_gateway_url(&self) -> Result<String, String> {
        let resp = self
            .client
            .get(format!("{}/gateway/bot", self.api_base))
            .header("Authorization", format!("Bot {}", self.bot_token))
            .send()
            .await
            .map_err(|e| format!("Discord gateway request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Discord gateway parse: {e}"))?;
        if let Some(url) = body["url"].as_str() {
            Ok(url.to_string())
        } else {
            Err(format!(
                "No gateway URL: {}",
                body["message"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Connect to the Gateway WebSocket.
    pub async fn connect(&self) -> Result<DiscordWsStream, String> {
        let url = self.get_gateway_url().await?;
        let ws_url = format!("{}?v=10&encoding=json", url.trim_end_matches('/'));
        let (ws, _) = connect_async(&ws_url)
            .await
            .map_err(|e| format!("Discord Gateway connect failed: {e}"))?;
        info!("Discord Gateway WS connected");
        Ok(ws)
    }

    /// Start the Gateway loop (identify/resume, heartbeat, reconnect) in the
    /// background.
    pub async fn start_gateway(&self) -> Result<(), String> {
        {
            let mut running = self.running.lock().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }
        let running = self.running.clone();
        let client = self.client.clone();
        let bot_token = self.bot_token.clone();
        let api_base = self.api_base.clone();
        let intents = self.intents;
        let incoming = self.incoming.clone();
        let session = self.session.clone();
        let task = tokio::spawn(async move {
            info!("Discord Gateway loop starting");
            let mut attempt: u32 = 0;
            loop {
                if !*running.lock().await {
                    return;
                }
                let ws_url = match get_gateway_url_raw(&client, &api_base, &bot_token).await {
                    Ok(url) => url,
                    Err(e) => {
                        error!("Discord gateway error: {e}");
                        format!("{}/?v=10&encoding=json", api_base)
                    }
                };
                run_gateway_cycle(
                    running.clone(),
                    bot_token.clone(),
                    ws_url,
                    intents,
                    incoming.clone(),
                    session.clone(),
                )
                .await;
                attempt = attempt.saturating_add(1).min(10);
                if !*running.lock().await {
                    return;
                }
                let delay = discord_reconnect_delay(attempt);
                info!("Discord reconnecting in {}s", delay.as_secs());
                tokio::time::sleep(delay).await;
            }
        });
        *self.gateway_task.lock().await = Some(task);
        Ok(())
    }

    /// Stop the Gateway loop.
    pub async fn stop_gateway(&self) {
        *self.running.lock().await = false;
        if let Some(task) = self.gateway_task.lock().await.take() {
            task.abort();
        }
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Send a raw message to a channel with the given payload.
    pub async fn create_message_raw(
        &self,
        channel_id: &str,
        payload: Value,
    ) -> Result<Value, String> {
        let route = format!("/channels/{}/messages", channel_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Discord send request: {e}"))?;
        let status = resp.status();
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            self.rate_limiter.handle_rate_limited(resp.headers()).await;
            return Err("Discord rate limited (429)".to_string());
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Discord send parse: {e}"))?;
        if status.is_success() {
            Ok(body)
        } else {
            Err(format!(
                "Discord error: {}",
                body["message"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a message with optional embeds.
    pub async fn send_message_raw(
        &self,
        channel_id: &str,
        content: &str,
        embeds: Vec<Value>,
    ) -> Result<Value, String> {
        let mut payload = json!({ "content": content });
        if !embeds.is_empty() {
            payload["embeds"] = Value::Array(embeds);
        }
        self.create_message_raw(channel_id, payload).await
    }

    /// Edit a previously sent message.
    pub async fn edit_message(
        &self,
        channel_id: &str,
        message_id: &str,
        content: &str,
    ) -> Result<(), String> {
        let route = format!("/channels/{}/messages/{}", channel_id, message_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&json!({ "content": content }))
            .send()
            .await
            .map_err(|e| format!("Discord edit request: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Discord edit failed: {}", resp.status()))
        }
    }

    /// Delete a message.
    pub async fn delete_message(&self, channel_id: &str, message_id: &str) -> Result<(), String> {
        let route = format!("/channels/{}/messages/{}", channel_id, message_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .delete(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .send()
            .await
            .map_err(|e| format!("Discord delete request: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Discord delete failed: {}", resp.status()))
        }
    }

    /// Add a reaction to a message.
    pub async fn add_reaction(
        &self,
        channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> Result<(), String> {
        let encoded = urlencode(emoji);
        let route = format!(
            "/channels/{}/messages/{}/reactions/{}",
            channel_id, message_id, encoded
        );
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .send()
            .await
            .map_err(|e| format!("Discord reaction request: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Discord reaction failed: {}", resp.status()))
        }
    }

    /// Create a DM channel with a user.
    pub async fn create_dm(&self, user_id: &str) -> Result<String, String> {
        let route = "/users/@me/channels";
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(route).await;
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&json!({ "recipient_id": user_id }))
            .send()
            .await
            .map_err(|e| format!("Discord DM request: {e}"))?;
        self.rate_limiter
            .record_response(route, resp.headers())
            .await;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Discord DM parse: {e}"))?;
        body["id"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| "No DM channel id".to_string())
    }

    /// Register global slash commands.
    pub async fn register_global_commands(&self, commands: Vec<Value>) -> Result<(), String> {
        let app_id = self
            .application_id
            .as_deref()
            .ok_or("No application_id configured")?;
        let route = format!("/applications/{}/commands", app_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&Value::Array(commands))
            .send()
            .await
            .map_err(|e| format!("Discord command registration: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "Discord command registration failed: {}",
                resp.status()
            ))
        }
    }

    /// Register slash commands scoped to a single guild.
    pub async fn register_guild_commands(
        &self,
        guild_id: &str,
        commands: Vec<Value>,
    ) -> Result<(), String> {
        let app_id = self
            .application_id
            .as_deref()
            .ok_or("No application_id configured")?;
        let route = format!("/applications/{}/guilds/{}/commands", app_id, guild_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&Value::Array(commands))
            .send()
            .await
            .map_err(|e| format!("Discord guild command registration: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "Discord guild command registration failed: {}",
                resp.status()
            ))
        }
    }

    /// Build a slash command definition.
    pub fn command(name: &str, description: &str) -> Value {
        json!({
            "name": name,
            "description": description,
            "options": [],
        })
    }

    /// Build an embed value from an attachment.
    pub fn embed_from_attachment(att: &MessageAttachment) -> Value {
        json!({
            "title": att.attachment_type,
            "url": att.url,
            "description": att.data.as_ref().and_then(|d| d.get("description")).cloned().unwrap_or(Value::Null),
        })
    }

    /// The internal rate limiter (exposed for testing).
    pub fn rate_limiter(&self) -> &DiscordRateLimiter {
        &self.rate_limiter
    }
}

/// Run one Gateway session with identify/resume, heartbeat and event parsing.
async fn run_gateway_cycle(
    running: Arc<Mutex<bool>>,
    bot_token: String,
    ws_url: String,
    intents: u64,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    session: Arc<Mutex<Option<DiscordSession>>>,
) {
    let connect_url = ws_url;
    let (ws_stream, _) = match connect_async(&connect_url).await {
        Ok(pair) => pair,
        Err(e) => {
            error!("Discord Gateway connect failed: {e}");
            return;
        }
    };
    let (mut sink, mut read): (
        SplitSink<DiscordWsStream, WsMessage>,
        SplitStream<DiscordWsStream>,
    ) = ws_stream.split();

    let last_seq: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    let interval_ms: Arc<Mutex<u64>> = Arc::new(Mutex::new(DEFAULT_HEARTBEAT_INTERVAL_MS));

    // Identify or resume based on saved session state.
    let resume = session.lock().await.clone();
    if let Some(sess) = &resume {
        let resume_payload = json!({
            "op": 6,
            "d": {
                "token": bot_token,
                "session_id": sess.session_id,
                "seq": sess.last_seq,
            }
        });
        if sink
            .send(WsMessage::Text(resume_payload.to_string()))
            .await
            .is_err()
        {
            return;
        }
        info!("Discord attempting resume");
    } else {
        let identify = json!({
            "op": 2,
            "d": {
                "token": bot_token,
                "intents": intents,
                "properties": {
                    "os": "windows",
                    "browser": "opensquilla",
                    "device": "opensquilla",
                }
            }
        });
        if sink
            .send(WsMessage::Text(identify.to_string()))
            .await
            .is_err()
        {
            return;
        }
        info!("Discord identify sent");
    }

    let heartbeat = spawn_discord_heartbeat(sink, last_seq.clone(), interval_ms.clone());

    while let Some(frame) = read.next().await {
        if !*running.lock().await {
            break;
        }
        match frame {
            Ok(WsMessage::Text(text)) => {
                let envelope: GatewayEnvelope = match serde_json::from_str(&text) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("Discord frame parse error: {e}");
                        continue;
                    }
                };
                match envelope.op {
                    0 => {
                        if let Some(seq) = envelope.s {
                            *last_seq.lock().await = seq;
                            if let Some(ref mut sess) = *session.lock().await {
                                sess.last_seq = seq;
                            }
                        }
                        // Capture resume state from the READY dispatch.
                        if envelope.t.as_deref() == Some("READY") {
                            if let Some(d) = &envelope.d {
                                let sess = DiscordSession {
                                    session_id: d
                                        .get("session_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                    last_seq: *last_seq.lock().await,
                                    resume_url: d
                                        .get("resume_gateway_url")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string(),
                                };
                                *session.lock().await = Some(sess);
                            }
                        }
                        if let Some(msg) = parse_dispatch_event(&envelope) {
                            incoming.lock().await.push_back(msg);
                        }
                    }
                    1 => {
                        // Server-requested heartbeat; the periodic heartbeat
                        // task sends the next one on schedule.
                        debug!("Discord heartbeat requested by server");
                    }
                    7 => {
                        info!("Discord reconnect requested by server");
                        break;
                    }
                    9 => {
                        // Invalid session; `d: true` allows resume, false does not.
                        let can_resume = envelope.d.and_then(|d| d.as_bool()).unwrap_or(false);
                        error!("Discord invalid session (resume={})", can_resume);
                        if !can_resume {
                            *session.lock().await = None;
                        }
                        break;
                    }
                    10 => {
                        if let Some(iv) = envelope
                            .d
                            .as_ref()
                            .and_then(|d| d["heartbeat_interval"].as_u64())
                        {
                            *interval_ms.lock().await = iv;
                        }
                    }
                    11 => {
                        debug!("Discord heartbeat ack");
                    }
                    other => {
                        debug!("Discord op {other} ignored");
                    }
                }
            }
            Ok(WsMessage::Close(_)) => {
                info!("Discord Gateway closed");
                break;
            }
            Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {}
            Err(e) => {
                warn!("Discord Gateway error: {e}");
                break;
            }
            _ => {}
        }
    }

    heartbeat.abort();
}

/// Spawn a periodic `op:1` heartbeat that echoes the latest dispatch sequence.
fn spawn_discord_heartbeat(
    mut sink: SplitSink<DiscordWsStream, WsMessage>,
    last_seq: Arc<Mutex<u64>>,
    interval_ms: Arc<Mutex<u64>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let initial = { *interval_ms.lock().await };
        let mut interval = tokio::time::interval(Duration::from_millis(initial.max(1)));
        interval.tick().await; // consume the immediate first tick
        loop {
            interval.tick().await;
            let iv = { *interval_ms.lock().await };
            if iv.max(1) != interval.period().as_millis() as u64 {
                interval = tokio::time::interval(Duration::from_millis(iv.max(1)));
                interval.tick().await;
            }
            let seq = { *last_seq.lock().await };
            let payload = json!({ "op": 1, "d": seq });
            if sink
                .send(WsMessage::Text(payload.to_string()))
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn discord_reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

async fn get_gateway_url_raw(
    client: &reqwest::Client,
    api_base: &str,
    bot_token: &str,
) -> Result<String, String> {
    let resp = client
        .get(format!("{}/gateway/bot", api_base))
        .header("Authorization", format!("Bot {bot_token}"))
        .send()
        .await
        .map_err(|e| format!("Discord gateway request: {e}"))?;
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("Discord gateway parse: {e}"))?;
    body["url"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| format!("No gateway URL in response: {body}"))
}

/// Parse a dispatch event into an [`IncomingMessage`] when it is a message.
fn parse_dispatch_event(envelope: &GatewayEnvelope) -> Option<IncomingMessage> {
    let event_type = envelope.t.as_deref()?;
    match event_type {
        "READY" => {
            let d = envelope.d.as_ref()?;
            let session_id = d.get("session_id").and_then(|v| v.as_str()).unwrap_or("");
            let _resume_url = d
                .get("resume_gateway_url")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            info!("Discord READY received (session={})", session_id);
            None
        }
        "MESSAGE_CREATE" => parse_message_create(envelope.d.as_ref()?),
        _ => None,
    }
}

/// Parse a `MESSAGE_CREATE` event.
fn parse_message_create(d: &Value) -> Option<IncomingMessage> {
    // Skip bot-authored messages to avoid feedback loops.
    let author = d.get("author");
    if author
        .and_then(|a| a.get("bot"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return None;
    }
    let channel_id = d
        .get("channel_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let user_id = author
        .and_then(|a| a.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let user_name = author
        .and_then(|a| a.get("global_name"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            author
                .and_then(|a| a.get("username"))
                .and_then(|v| v.as_str())
                .map(String::from)
        });
    let text = d
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let message_id = d.get("id").and_then(|v| v.as_str()).map(String::from);
    let thread_id = d
        .get("message_reference")
        .and_then(|r| r.get("message_id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = d
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);

    let attachments: Vec<MessageAttachment> = d
        .get("attachments")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .map(|a| MessageAttachment {
                    attachment_type: "file".to_string(),
                    url: a.get("url").and_then(|v| v.as_str()).map(String::from),
                    data: Some(a.clone()),
                    mime_type: a
                        .get("content_type")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                })
                .collect()
        })
        .unwrap_or_default();

    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type: ChannelType::Discord,
        user_id,
        user_name,
        text,
        thread_id: thread_id.or(message_id),
        attachments,
        timestamp,
        raw: d.clone(),
    })
}

/// Percent-encode a path segment (used for emoji in reactions).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

#[async_trait::async_trait]
impl Channel for DiscordChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Discord
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let embeds: Vec<Value> = message
            .attachments
            .iter()
            .map(Self::embed_from_attachment)
            .collect();
        self.send_message_raw(&message.channel_id, &message.text, embeds)
            .await?;
        Ok(())
    }

    async fn send_typing(&self, channel_id: &str) -> Result<(), String> {
        let route = format!("/channels/{}/typing", channel_id);
        let url = format!("{}{}", self.api_base, route);
        self.rate_limiter.acquire(&route).await;
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .send()
            .await
            .map_err(|e| format!("Discord typing request: {e}"))?;
        self.rate_limiter
            .record_response(&route, resp.headers())
            .await;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "Failed to send typing indicator: {}",
                resp.status()
            ))
        }
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        Err(
            "Discord does not use webhooks for bot messages. Use the Gateway and REST API instead."
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> DiscordChannel {
        DiscordChannel::new(ChannelConfig {
            channel_type: ChannelType::Discord,
            channel_id: "guild".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({ "bot_token": "token" }),
        })
        .unwrap()
    }

    #[test]
    fn test_default_intents() {
        let i = DiscordChannel::default_intents();
        assert!(i & INTENT_GUILDS != 0);
        assert!(i & INTENT_GUILD_MESSAGES != 0);
        assert!(i & INTENT_MESSAGE_CONTENT != 0);
    }

    #[test]
    fn test_parse_message_create() {
        let d = json!({
            "id": "m1",
            "channel_id": "c1",
            "guild_id": "g1",
            "author": {"id": "u1", "username": "alice", "bot": false},
            "content": "hello discord",
            "timestamp": "2021-07-02T16:00:00.000000+00:00",
            "attachments": []
        });
        let env = GatewayEnvelope {
            op: 0,
            s: Some(1),
            t: Some("MESSAGE_CREATE".to_string()),
            d: Some(d),
        };
        let msg = parse_dispatch_event(&env).unwrap();
        assert_eq!(msg.channel_type, ChannelType::Discord);
        assert_eq!(msg.channel_id, "c1");
        assert_eq!(msg.user_id, "u1");
        assert_eq!(msg.text, "hello discord");
    }

    #[test]
    fn test_parse_message_skips_bots() {
        let d = json!({
            "id": "m2",
            "channel_id": "c1",
            "author": {"id": "bot1", "username": "bot", "bot": true},
            "content": "ignore me",
            "timestamp": "2021-07-02T16:00:00.000000+00:00"
        });
        let env = GatewayEnvelope {
            op: 0,
            s: Some(2),
            t: Some("MESSAGE_CREATE".to_string()),
            d: Some(d),
        };
        assert!(parse_dispatch_event(&env).is_none());
    }

    #[test]
    fn test_parse_ready_ignored() {
        let env = GatewayEnvelope {
            op: 0,
            s: Some(3),
            t: Some("READY".to_string()),
            d: Some(json!({"session_id": "s1", "resume_gateway_url": "wss://x"})),
        };
        assert!(parse_dispatch_event(&env).is_none());
    }

    #[test]
    fn test_reconnect_backoff_capped() {
        assert_eq!(discord_reconnect_delay(0).as_secs(), 1);
        assert_eq!(discord_reconnect_delay(1).as_secs(), 2);
        assert!(discord_reconnect_delay(20).as_secs() <= MAX_RECONNECT_DELAY_SECS);
    }

    #[test]
    fn test_urlencode_emoji() {
        assert_eq!(urlencode("a b"), "a+b");
        assert_eq!(urlencode("👍"), "%F0%9F%91%8D");
    }

    #[test]
    fn test_command_builder() {
        let cmd = DiscordChannel::command("ping", "pong");
        assert_eq!(cmd["name"], "ping");
        assert_eq!(cmd["description"], "pong");
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire_is_fast() {
        let rl = DiscordRateLimiter::new();
        let start = Instant::now();
        rl.acquire("/channels/x/messages").await;
        assert!(start.elapsed() < Duration::from_millis(100));
    }
}
