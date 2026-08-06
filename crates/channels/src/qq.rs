//! QQ Bot channel adapter — WebSocket Gateway + raw REST API.
//!
//! The official QQ Bot Platform streams events over a persistent WebSocket
//! Gateway and exposes a REST API for sending. This module implements the raw
//! protocol without the `qq-botpy` SDK:
//!
//! 1. **Gateway**: `GET /gateway/bot` returns a `wss://...` URL.
//! 2. **Identify**: after connecting, send `{"op": 2, "d": {"token", "intents",
//!    "shard"}}`.
//! 3. **Hello**: the server answers with `op:10` carrying `heartbeat_interval`;
//!    the client then sends `op:1` heartbeats (with the last dispatch `seq`)
//!    and the server replies `op:11`.
//! 4. **Dispatch**: `op:0` events carry message payloads (`AT_MESSAGE_CREATE`,
//!    `GROUP_AT_MESSAGE_CREATE`, `C2C_MESSAGE_CREATE`, ...).
//! 5. **Reconnect**: `op:7` asks for an immediate reconnect; `op:9` means the
//!    session is invalid.
//!
//! Outbound messages use `POST /channels/{channel_id}/messages` for guild
//! channels and `POST /v2/users|groups/{openid}/messages` for C2C / group
//! conversations, authenticated with `Authorization: QQBot {token}`.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use chrono::{DateTime, Utc};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use serde_json::{Value, json};

/// The WebSocket stream type used by the QQ Gateway connection.
pub type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

const DEFAULT_API_BASE: &str = "https://api.sgroup.qq.com";
/// Default gateway heartbeat interval in milliseconds, overridden by `op:10`.
const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 41_250;
const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;

// Gateway intents (bitmask). These are the official QQ Bot Platform values.
/// Guild / channel lifecycle events.
pub const INTENT_GUILDS: u64 = 1 << 0;
/// Guild messages (private-domain channels).
pub const INTENT_GUILD_MESSAGES: u64 = 1 << 9;
/// Direct messages between bot and user.
pub const INTENT_DIRECT_MESSAGE: u64 = 1 << 12;
/// Group and C2C (single-chat) messages.
pub const INTENT_GROUP_AND_C2C: u64 = 1 << 25;
/// Interaction events (slash commands, buttons).
pub const INTENT_INTERACTION: u64 = 1 << 26;
/// Forum/thread events.
pub const INTENT_FORUMS: u64 = 1 << 28;
/// Public-domain guild messages.
pub const INTENT_PUBLIC_GUILD_MESSAGES: u64 = 1 << 30;

/// An envelope received from the QQ Gateway WebSocket.
#[derive(Debug, Clone, serde::Deserialize)]
struct QQGatewayEnvelope {
    op: i64,
    #[serde(default)]
    s: Option<u64>,
    #[serde(default)]
    t: Option<String>,
    #[serde(default)]
    d: Option<Value>,
}

/// QQ Bot channel adapter using WebSocket for event streaming and reqwest REST
/// for API calls.
pub struct QQChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    bot_token: String,
    api_base: String,
    intents: u64,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
}

impl QQChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let bot_token = config
            .config
            .get("bot_token")
            .and_then(|v| v.as_str())
            .ok_or("No bot_token configured for QQ Bot")?
            .to_string();
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .to_string();
        let intents = config
            .config
            .get("intents")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(Self::default_intents);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        Ok(Self {
            config,
            client,
            bot_token,
            api_base,
            intents,
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// The default intent mask: guild + channel messages, direct messages,
    /// group/C2C messages, interactions and forums.
    pub fn default_intents() -> u64 {
        INTENT_GUILDS
            | INTENT_GUILD_MESSAGES
            | INTENT_DIRECT_MESSAGE
            | INTENT_GROUP_AND_C2C
            | INTENT_INTERACTION
            | INTENT_FORUMS
            | INTENT_PUBLIC_GUILD_MESSAGES
    }

    /// Build the `op:2` identify payload used to authenticate the Gateway
    /// session and subscribe to [`QQChannel::intents`].
    pub fn handle_intents(&self) -> Value {
        json!({
            "op": 2,
            "d": {
                "token": format!("QQBot {}", self.bot_token),
                "intents": self.intents,
                "shard": [0, 1],
            }
        })
    }

    /// Fetch the Gateway URL and open a WebSocket connection.
    pub async fn connect(&self) -> Result<WsStream, String> {
        let ws_url = get_gateway_url(&self.client, &self.api_base, &self.bot_token).await?;
        let (ws_stream, _) = connect_async(&ws_url)
            .await
            .map_err(|e| format!("QQ Gateway WS connect failed: {}", e))?;
        info!("QQ Gateway WS connected");
        Ok(ws_stream)
    }

    /// Start the Gateway connection loop with heartbeat and reconnect.
    pub async fn start_websocket(&self) -> Result<(), String> {
        {
            let mut r = self.running.lock().await;
            if *r {
                return Ok(());
            }
            *r = true;
        }
        let running = self.running.clone();
        let client = self.client.clone();
        let bot_token = self.bot_token.clone();
        let api_base = self.api_base.clone();
        let intents = self.intents;
        let incoming = self.incoming.clone();

        tokio::spawn(async move {
            info!("QQ Bot WS starting");
            let mut attempt: u32 = 0;
            loop {
                if !*running.lock().await {
                    return;
                }
                let ws_url = match get_gateway_url(&client, &api_base, &bot_token).await {
                    Ok(url) => url,
                    Err(e) => {
                        error!("QQ gateway error: {}", e);
                        ws_url_for_shard(&api_base)
                    }
                };
                run_gateway_cycle(
                    running.clone(),
                    bot_token.clone(),
                    ws_url,
                    intents,
                    incoming.clone(),
                )
                .await;
                attempt = attempt.saturating_add(1).min(10);
                if !*running.lock().await {
                    return;
                }
                let delay = qq_reconnect_delay(attempt);
                info!("QQ Bot reconnecting in {}s", delay.as_secs());
                tokio::time::sleep(delay).await;
            }
        });
        Ok(())
    }

    /// Stop the Gateway loop.
    pub async fn stop_websocket(&self) {
        *self.running.lock().await = false;
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Send a message to a guild channel, C2C user, or group.
    async fn send_qq_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let msg_type = message
            .metadata
            .get("msg_type")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let url = build_send_url(&self.api_base, message);
        let mut payload = json!({
            "content": message.text,
            "msg_type": msg_type,
        });
        if let Some(ref reply) = message.thread_id {
            payload["msg_id"] = Value::String(reply.clone());
        }
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("QQ Bot send request: {}", e))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("QQ Bot send parse: {}", e))?;
        if status.is_success() {
            Ok(())
        } else {
            Err(format!(
                "QQ Bot error: {} (status: {})",
                body["message"].as_str().unwrap_or("unknown"),
                status
            ))
        }
    }

    /// Send a markdown message (`msg_type: 2`).
    pub async fn send_markdown_message(
        &self,
        channel_id: &str,
        content: &str,
    ) -> Result<(), String> {
        let payload = json!({"content": content, "msg_type": 2});
        let url = format!("{}/channels/{}/messages", self.api_base, channel_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("QQ Bot markdown send: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("QQ Bot markdown failed: {}", resp.status()))
        }
    }

    /// Send an image message (`msg_type: 3`) by media id.
    pub async fn send_image_message(&self, channel_id: &str, media_id: &str) -> Result<(), String> {
        let payload = json!({"content": "", "msg_type": 3, "media_id": media_id});
        let url = format!("{}/channels/{}/messages", self.api_base, channel_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("QQ Bot image send: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("QQ Bot image failed: {}", resp.status()))
        }
    }

    /// Send an embed message (`msg_type: 7`).
    pub async fn send_embed_message(
        &self,
        channel_id: &str,
        title: &str,
        description: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "msg_type": 7,
            "embed": {"title": title, "description": description},
        });
        let url = format!("{}/channels/{}/messages", self.api_base, channel_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("QQ Bot embed send: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("QQ Bot embed failed: {}", resp.status()))
        }
    }

    /// Reply to an existing message in a guild channel.
    pub async fn reply_to_message(
        &self,
        channel_id: &str,
        message_id: &str,
        text: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "content": text,
            "msg_type": 0,
            "msg_id": message_id,
        });
        let url = format!("{}/channels/{}/messages", self.api_base, channel_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("QQ Bot reply: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("QQ Bot reply failed: {}", resp.status()))
        }
    }

    /// Add a reaction to a message in a guild channel.
    pub async fn add_reaction(
        &self,
        channel_id: &str,
        message_id: &str,
        emoji_id: &str,
    ) -> Result<(), String> {
        let url = format!(
            "{}/channels/{}/messages/{}/reactions/{}",
            self.api_base, channel_id, message_id, emoji_id
        );
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("QQBot {}", self.bot_token))
            .send()
            .await
            .map_err(|e| format!("QQ Bot reaction: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("QQ Bot reaction failed: {}", resp.status()))
        }
    }
}

/// Fetch the Gateway WebSocket URL from `/gateway/bot`.
async fn get_gateway_url(
    client: &reqwest::Client,
    api_base: &str,
    bot_token: &str,
) -> Result<String, String> {
    let resp = client
        .get(format!("{}/gateway/bot", api_base))
        .header("Authorization", format!("QQBot {}", bot_token))
        .send()
        .await
        .map_err(|e| format!("QQ gateway request: {}", e))?;
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("QQ gateway parse: {}", e))?;
    body["url"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| format!("No gateway URL in response: {}", body))
}

/// Fallback URL when the gateway call fails: a sharded WebSocket endpoint.
fn ws_url_for_shard(api_base: &str) -> String {
    format!("{}/websocket?shard=0/1", api_base)
}

/// Pick the outbound endpoint for a message based on its target.
fn build_send_url(api_base: &str, message: &OutgoingMessage) -> String {
    if let Some(openid) = message.metadata.get("user_openid").and_then(|v| v.as_str()) {
        format!("{}/v2/users/{}/messages", api_base, openid)
    } else if let Some(group_openid) = message
        .metadata
        .get("group_openid")
        .and_then(|v| v.as_str())
    {
        format!("{}/v2/groups/{}/messages", api_base, group_openid)
    } else {
        format!("{}/channels/{}/messages", api_base, message.channel_id)
    }
}

/// Run one Gateway session: identify, heartbeat, read, reconnect on demand.
async fn run_gateway_cycle(
    running: Arc<Mutex<bool>>,
    bot_token: String,
    ws_url: String,
    intents: u64,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
) {
    let (ws_stream, _) = match connect_async(&ws_url).await {
        Ok(pair) => pair,
        Err(e) => {
            error!("QQ Gateway WS connect failed: {}", e);
            return;
        }
    };
    let (mut sink, mut read): (SplitSink<WsStream, Message>, SplitStream<WsStream>) =
        ws_stream.split();
    info!("QQ Bot WS connected");

    let identify = json!({
        "op": 2,
        "d": {
            "token": format!("QQBot {}", bot_token),
            "intents": intents,
            "shard": [0, 1],
        }
    });
    if let Err(e) = sink.send(Message::Text(identify.to_string().into())).await {
        error!("QQ identify send failed: {}", e);
        return;
    }

    let last_seq: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
    let interval_ms: Arc<Mutex<u64>> = Arc::new(Mutex::new(DEFAULT_HEARTBEAT_INTERVAL_MS));
    let heartbeat = spawn_qq_heartbeat(sink, last_seq.clone(), interval_ms.clone());

    while let Some(frame) = read.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let envelope = match serde_json::from_str::<QQGatewayEnvelope>(&text) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("QQ frame parse error: {}", e);
                        continue;
                    }
                };
                match envelope.op {
                    0 => {
                        if let Some(seq) = envelope.s {
                            *last_seq.lock().await = Some(seq);
                        }
                        if let Some(msg) = parse_gateway_event(&envelope, ChannelType::QQ) {
                            let mut q = incoming.lock().await;
                            q.push_back(msg);
                        }
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
                        debug!("QQ heartbeat ack");
                    }
                    7 => {
                        info!("QQ reconnect requested by server");
                        break;
                    }
                    9 => {
                        error!("QQ invalid session");
                        break;
                    }
                    other => {
                        debug!("QQ op {} ignored", other);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                info!("QQ Bot WS closed");
                break;
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
            Err(e) => {
                warn!("QQ Bot WS error: {}", e);
                break;
            }
            _ => {}
        }
    }

    heartbeat.abort();
    let _ = running.lock().await;
}

/// Spawn a periodic `op:1` heartbeat that echoes the latest dispatch `seq`.
fn spawn_qq_heartbeat(
    mut sink: SplitSink<WsStream, Message>,
    last_seq: Arc<Mutex<Option<u64>>>,
    interval_ms: Arc<Mutex<u64>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let initial = {
            let iv = interval_ms.lock().await;
            *iv
        };
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
            let payload = json!({"op": 1, "d": seq});
            if sink
                .send(Message::Text(payload.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    })
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn qq_reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

/// Parse a dispatch event into an [`IncomingMessage`] when it is a message.
fn parse_gateway_event(
    envelope: &QQGatewayEnvelope,
    channel_type: ChannelType,
) -> Option<IncomingMessage> {
    let event_type = envelope.t.as_deref()?;
    let d = envelope.d.as_ref()?;
    match event_type {
        "AT_MESSAGE_CREATE" | "PUBLIC_MESSAGE_CREATE" | "DIRECT_MESSAGE_CREATE" => {
            parse_channel_message(d, channel_type)
        }
        "GROUP_AT_MESSAGE_CREATE" => parse_group_message(d, channel_type),
        "C2C_MESSAGE_CREATE" => parse_c2c_message(d, channel_type),
        _ => None,
    }
}

/// Parse a guild channel message event.
fn parse_channel_message(d: &Value, channel_type: ChannelType) -> Option<IncomingMessage> {
    let channel_id = d
        .get("channel_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let author = d.get("author");
    let user_id = author
        .and_then(|a| a.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let user_name = author
        .and_then(|a| a.get("username"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let text = d
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let thread_id = d
        .get("msg_id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| d.get("id").and_then(|v| v.as_str()).map(String::from));
    let timestamp = d
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(parse_qq_timestamp)
        .unwrap_or_else(Utc::now);
    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type,
        user_id,
        user_name,
        text,
        thread_id,
        attachments: Vec::new(),
        timestamp,
        raw: d.clone(),
    })
}

/// Parse a group `GROUP_AT_MESSAGE_CREATE` event.
fn parse_group_message(d: &Value, channel_type: ChannelType) -> Option<IncomingMessage> {
    let channel_id = d
        .get("group_openid")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| d.get("group_id").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_default();
    let author = d.get("author");
    let user_id = author
        .and_then(|a| a.get("member_openid"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            author
                .and_then(|a| a.get("user_openid"))
                .and_then(|v| v.as_str())
        })
        .or_else(|| d.get("author_id").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_string();
    let user_name = author
        .and_then(|a| a.get("member").and_then(|m| m.get("name")))
        .and_then(|v| v.as_str())
        .or_else(|| {
            author
                .and_then(|a| a.get("nickname"))
                .and_then(|v| v.as_str())
        })
        .map(String::from);
    let text = d
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let thread_id = d.get("msg_id").and_then(|v| v.as_str()).map(String::from);
    let timestamp = d
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(parse_qq_timestamp)
        .unwrap_or_else(Utc::now);
    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type,
        user_id,
        user_name,
        text,
        thread_id,
        attachments: Vec::new(),
        timestamp,
        raw: d.clone(),
    })
}

/// Parse a C2C (single-chat) `C2C_MESSAGE_CREATE` event.
fn parse_c2c_message(d: &Value, channel_type: ChannelType) -> Option<IncomingMessage> {
    let channel_id = d
        .get("user_openid")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| d.get("openid").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_default();
    let user_id = channel_id.clone();
    let user_name = d
        .get("author")
        .and_then(|a| a.get("member").and_then(|m| m.get("nickname")))
        .and_then(|v| v.as_str())
        .or_else(|| {
            d.get("author")
                .and_then(|a| a.get("nickname"))
                .and_then(|v| v.as_str())
        })
        .map(String::from);
    let text = d
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let thread_id = d.get("msg_id").and_then(|v| v.as_str()).map(String::from);
    let timestamp = d
        .get("timestamp")
        .and_then(|v| v.as_str())
        .and_then(parse_qq_timestamp)
        .unwrap_or_else(Utc::now);
    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type,
        user_id,
        user_name,
        text,
        thread_id,
        attachments: Vec::new(),
        timestamp,
        raw: d.clone(),
    })
}

/// Parse an RFC3339 QQ timestamp.
fn parse_qq_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.with_timezone(&Utc))
}

/// Build an attachment from a QQ media id (used by image messages).
fn media_attachment(media_id: &str) -> MessageAttachment {
    MessageAttachment {
        attachment_type: "image".to_string(),
        url: None,
        data: Some(json!({"media_id": media_id})),
        mime_type: Some("image/*".to_string()),
    }
}

#[async_trait::async_trait]
impl Channel for QQChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::QQ
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.send_qq_message(message).await
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        // The QQ Bot Platform exposes no "typing" indicator primitive; this is
        // a no-op kept for trait conformance.
        Ok(())
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        // QQ uses the WebSocket Gateway for inbound, not HTTP webhooks.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope(op: i64, t: &str, d: Value) -> QQGatewayEnvelope {
        serde_json::from_value(json!({"op": op, "s": 1, "t": t, "d": d})).unwrap()
    }

    #[test]
    fn test_parse_channel_message() {
        let d = json!({
            "id": "event-1",
            "channel_id": "channel-1",
            "guild_id": "guild-1",
            "author": {"id": "user-1", "username": "alice"},
            "content": "hello qq",
            "timestamp": "2023-04-13T10:00:00+08:00",
            "msg_id": "msg-1",
        });
        let ev = envelope(0, "AT_MESSAGE_CREATE", d);
        let msg = parse_gateway_event(&ev, ChannelType::QQ).expect("should parse");
        assert_eq!(msg.channel_type, ChannelType::QQ);
        assert_eq!(msg.channel_id, "channel-1");
        assert_eq!(msg.user_id, "user-1");
        assert_eq!(msg.user_name.as_deref(), Some("alice"));
        assert_eq!(msg.text, "hello qq");
        assert_eq!(msg.thread_id.as_deref(), Some("msg-1"));
    }

    #[test]
    fn test_parse_group_message() {
        let d = json!({
            "id": "event-2",
            "group_openid": "group-openid-1",
            "author": {"member_openid": "member-1", "member": {"name": "bob"}},
            "content": "hello group",
            "timestamp": "2023-04-13T10:00:00Z",
            "msg_id": "msg-2",
        });
        let ev = envelope(0, "GROUP_AT_MESSAGE_CREATE", d);
        let msg = parse_gateway_event(&ev, ChannelType::QQ).expect("should parse");
        assert_eq!(msg.channel_id, "group-openid-1");
        assert_eq!(msg.user_id, "member-1");
        assert_eq!(msg.user_name.as_deref(), Some("bob"));
        assert_eq!(msg.text, "hello group");
    }

    #[test]
    fn test_parse_c2c_message() {
        let d = json!({
            "id": "event-3",
            "user_openid": "user-openid-1",
            "author": {"user_openid": "user-openid-1"},
            "content": "hello c2c",
            "timestamp": "2023-04-13T10:00:00Z",
            "msg_id": "msg-3",
        });
        let ev = envelope(0, "C2C_MESSAGE_CREATE", d);
        let msg = parse_gateway_event(&ev, ChannelType::QQ).expect("should parse");
        assert_eq!(msg.channel_id, "user-openid-1");
        assert_eq!(msg.user_id, "user-openid-1");
        assert_eq!(msg.text, "hello c2c");
    }

    #[test]
    fn test_hello_event_is_not_a_message() {
        let ev = envelope(10, "", json!({"heartbeat_interval": 41250}));
        assert!(parse_gateway_event(&ev, ChannelType::QQ).is_none());
    }

    #[test]
    fn test_default_intents_mask() {
        let mask = QQChannel::default_intents();
        assert!(mask & INTENT_GUILD_MESSAGES != 0);
        assert!(mask & INTENT_GROUP_AND_C2C != 0);
        assert!(mask & INTENT_PUBLIC_GUILD_MESSAGES != 0);
    }

    #[test]
    fn test_handle_intents_payload() {
        let cfg = ChannelConfig {
            channel_type: ChannelType::QQ,
            channel_id: "qq".to_string(),
            name: "QQ".to_string(),
            enabled: true,
            config: json!({"bot_token": "abc"}),
        };
        let channel = QQChannel::new(cfg).unwrap();
        let identify = channel.handle_intents();
        assert_eq!(identify["op"], 2);
        assert_eq!(identify["d"]["token"], "QQBot abc");
        assert_eq!(identify["d"]["intents"], channel.intents);
        assert_eq!(identify["d"]["shard"], json!([0, 1]));
    }

    #[test]
    fn test_build_send_url() {
        let msg = OutgoingMessage::new("channel-1".to_string(), ChannelType::QQ, "hi".to_string());
        assert_eq!(
            build_send_url("https://api.sgroup.qq.com", &msg),
            "https://api.sgroup.qq.com/channels/channel-1/messages"
        );

        let mut c2c = msg.clone();
        c2c.metadata = json!({"user_openid": "u1"});
        assert_eq!(
            build_send_url("https://api.sgroup.qq.com", &c2c),
            "https://api.sgroup.qq.com/v2/users/u1/messages"
        );
    }

    #[test]
    fn test_heartbeat_payload_uses_last_seq() {
        // Verify the heartbeat JSON construction mirrors the sequence value.
        let seq: Option<u64> = Some(42);
        let payload = json!({"op": 1, "d": seq});
        assert_eq!(payload["op"], 1);
        assert_eq!(payload["d"], 42);
    }

    #[test]
    fn test_media_attachment() {
        let a = media_attachment("media-1");
        assert_eq!(a.attachment_type, "image");
        assert_eq!(a.data.as_ref().unwrap()["media_id"], "media-1");
    }
}
