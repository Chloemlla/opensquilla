//! DingTalk (钉钉) channel adapter — Stream Mode WebSocket + raw REST API.
//!
//! DingTalk bots receive messages over a persistent WebSocket ("Stream Mode")
//! rather than an HTTP webhook. This module implements the raw protocol with
//! no third-party SDK:
//!
//! 1. **Open**: `POST /v1.0/gateway/connections/open` exchanges the app
//!    `clientId`/`clientSecret` for a one-time `ticket` plus a WebSocket
//!    `endpoint`.
//! 2. **Connect**: open the WebSocket at `{endpoint}/connect?ticket=...&clientId=...`.
//! 3. **Receive**: the server pushes JSON envelopes. A `BOT_MESSAGE` envelope
//!    carries a chat message; the client must ack each event via
//!    `POST /v1.0/gateway/connections/ack` so the stream does not redeliver.
//! 4. **Heartbeat**: the client sends WebSocket `Ping` frames every 30 s;
//!    tungstenite answers server pings with pongs automatically.
//! 5. **Send**: outbound messages go through the DingTalk OpenAPI REST
//!    endpoints (`/robot/groupMessages/send`, `/robot/singleMessages/send`)
//!    authenticated with a tenant access token from `/oauth2/accessToken`.
//!
//! The robot webhook security scheme (timestamp + HMAC-SHA256 signature) is
//! implemented for callers that bridge the stream to an outbound webhook.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use base64::Engine;
use chrono::{TimeZone, Utc};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use serde_json::{Value, json};

type HmacSha256 = Hmac<Sha256>;
/// The WebSocket stream type used by the DingTalk Stream Mode connection.
pub type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

const DEFAULT_API_BASE: &str = "https://api.dingtalk.com/v1.0";
const DEFAULT_WS_ENDPOINT: &str = "wss://stream.dingtalk.com";
/// DingTalk Stream Mode expects a WebSocket ping roughly every 30 seconds.
const HEARTBEAT_INTERVAL_SECS: u64 = 30;
const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;

/// Stream-mode event type that carries a chat message.
const BOT_MESSAGE_EVENT: &str = "BOT_MESSAGE";

/// A frame received from the DingTalk Stream Mode WebSocket.
///
/// Two envelope shapes exist in the wild: a headered form
/// (`headers.eventType` / `headers.eventId`) used by older stream clients and
/// a flat form (`eventType` / `eventId`) used by newer gateways. Both are
/// accepted here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DingTalkStreamEnvelope {
    #[serde(default)]
    headers: Option<DingTalkStreamHeaders>,
    #[serde(default)]
    event_id: Option<String>,
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    message_id: Option<String>,
    #[serde(default)]
    message: Option<Value>,
    #[serde(default)]
    payload: Option<Value>,
}

/// Headers present on headered DingTalk stream envelopes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DingTalkStreamHeaders {
    #[serde(default)]
    event_id: Option<String>,
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    message_id: Option<String>,
}

impl DingTalkStreamEnvelope {
    /// The event type, read from either the flat field or the headers.
    fn event_type(&self) -> &str {
        self.event_type
            .as_deref()
            .or_else(|| self.headers.as_ref().and_then(|h| h.event_type.as_deref()))
            .unwrap_or("")
    }

    /// The event id, read from either the flat field or the headers.
    fn event_id(&self) -> Option<&str> {
        self.event_id
            .as_deref()
            .or_else(|| self.headers.as_ref().and_then(|h| h.event_id.as_deref()))
    }

    /// The embedded message payload, if any. The payload is sometimes a
    /// JSON-encoded string; when it is, it is decoded.
    fn message(&self) -> Option<Value> {
        if let Some(msg) = self.message.clone() {
            return Some(msg);
        }
        if let Some(payload) = &self.payload {
            if let Some(s) = payload.as_str() {
                return serde_json::from_str(s)
                    .ok()
                    .or_else(|| Some(payload.clone()));
            }
            return Some(payload.clone());
        }
        None
    }
}

/// DingTalk channel adapter using Stream Mode WebSocket and raw HTTP REST API.
///
/// The struct is shared behind an `Arc` by the channel manager. Incoming
/// messages are parsed by a background task spawned from [`DingTalkChannel::start_stream`]
/// and queued for polling via [`DingTalkChannel::receive`].
pub struct DingTalkChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    client_id: String,
    client_secret: Option<String>,
    api_base: String,
    ws_endpoint: String,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
}

impl DingTalkChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let client_id = config
            .config
            .get("client_id")
            .and_then(|v| v.as_str())
            .ok_or("No client_id configured for DingTalk")?
            .to_string();
        let client_secret = config
            .config
            .get("client_secret")
            .and_then(|v| v.as_str())
            .map(String::from);
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .to_string();
        let ws_endpoint = config
            .config
            .get("ws_endpoint")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_WS_ENDPOINT)
            .to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        Ok(Self {
            config,
            client,
            client_id,
            client_secret,
            api_base,
            ws_endpoint,
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Obtain a tenant access token for the DingTalk OpenAPI.
    async fn get_access_token(&self) -> Result<String, String> {
        let secret = self
            .client_secret
            .as_ref()
            .ok_or("No client_secret configured")?;
        let resp = self
            .client
            .post(format!("{}/oauth2/accessToken", self.api_base))
            .json(&json!({"appKey": self.client_id, "appSecret": secret}))
            .send()
            .await
            .map_err(|e| format!("DingTalk auth request failed: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("DingTalk auth parse: {}", e))?;
        body["accessToken"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| format!("No accessToken in response: {}", body))
    }

    /// Open a connection ticket and return the WebSocket stream.
    ///
    /// This establishes the persistent Stream Mode connection. The returned
    /// stream should be handed to [`DingTalkChannel::handle_heartbeat`] and a
    /// read loop.
    pub async fn connect(&self) -> Result<WsStream, String> {
        let secret = self.client_secret.as_deref().unwrap_or("");
        let (ticket, endpoint) =
            open_connection_ticket(&self.client, &self.api_base, &self.client_id, secret).await?;
        let endpoint = if endpoint.is_empty() {
            self.ws_endpoint.clone()
        } else {
            endpoint
        };
        let ws_url = build_ws_url(&endpoint, &self.client_id, &ticket);
        let (ws_stream, _) = connect_async(&ws_url)
            .await
            .map_err(|e| format!("DingTalk WS connect failed: {}", e))?;
        info!("DingTalk stream connected");
        Ok(ws_stream)
    }

    /// Spawn a periodic WebSocket `Ping` heartbeat for the given sink.
    ///
    /// Returns the task handle; callers should `abort()` it when the socket
    /// closes so the underlying sink is dropped.
    pub fn handle_heartbeat(mut sink: SplitSink<WsStream, Message>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(HEARTBEAT_INTERVAL_SECS));
            interval.tick().await; // consume the immediate first tick
            loop {
                interval.tick().await;
                if sink.send(Message::Ping(Vec::new())).await.is_err() {
                    break;
                }
            }
        })
    }

    /// Start the persistent Stream Mode connection loop.
    ///
    /// A background task obtains a ticket, connects, heartbeats, acks and
    /// parses `BOT_MESSAGE` events, and reconnects with backoff on failure.
    pub async fn start_stream(&self) -> Result<(), String> {
        {
            let mut r = self.running.lock().await;
            if *r {
                return Ok(());
            }
            *r = true;
        }
        let running = self.running.clone();
        let client = self.client.clone();
        let client_id = self.client_id.clone();
        let client_secret = self.client_secret.clone().unwrap_or_default();
        let api_base = self.api_base.clone();
        let ws_endpoint = self.ws_endpoint.clone();
        let incoming = self.incoming.clone();

        tokio::spawn(async move {
            info!("DingTalk stream starting");
            let mut attempt: u32 = 0;
            loop {
                if !*running.lock().await {
                    return;
                }
                match open_connection_ticket(&client, &api_base, &client_id, &client_secret).await {
                    Ok((ticket, endpoint)) => {
                        let endpoint = if endpoint.is_empty() {
                            ws_endpoint.clone()
                        } else {
                            endpoint
                        };
                        let ws_url = build_ws_url(&endpoint, &client_id, &ticket);
                        run_stream_cycle(
                            running.clone(),
                            client.clone(),
                            client_id.clone(),
                            api_base.clone(),
                            ws_url,
                            incoming.clone(),
                        )
                        .await;
                        attempt = 0;
                    }
                    Err(e) => {
                        error!("DingTalk ticket error: {}", e);
                    }
                }
                if !*running.lock().await {
                    return;
                }
                let delay = reconnect_delay(attempt);
                attempt = attempt.saturating_add(1).min(10);
                info!("DingTalk reconnecting in {}s", delay.as_secs());
                tokio::time::sleep(delay).await;
            }
        });
        Ok(())
    }

    /// Stop the stream loop. In-flight sockets close on their next error.
    pub async fn stop_stream(&self) {
        *self.running.lock().await = false;
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Send a chat message to a single user or group conversation.
    async fn send_dingtalk_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let token = self.get_access_token().await?;
        let conversation_type = message
            .metadata
            .get("conversation_type")
            .and_then(|v| v.as_str())
            .unwrap_or("single");
        let msg_type = message
            .metadata
            .get("msg_type")
            .and_then(|v| v.as_str())
            .unwrap_or("text");
        if conversation_type == "group" {
            self.send_group_message(&token, message, msg_type).await
        } else {
            self.send_single_message(&token, message, msg_type).await
        }
    }

    /// Send a text or markdown message to a group conversation.
    async fn send_group_message(
        &self,
        token: &str,
        message: &OutgoingMessage,
        msg_type: &str,
    ) -> Result<(), String> {
        let (msg_key, msg_param) = build_msg_key_param(msg_type, &message.text);
        let payload = json!({
            "conversationId": message.channel_id,
            "msgKey": msg_key,
            "msgParam": msg_param,
        });
        let resp = self
            .client
            .post(format!("{}/robot/groupMessages/send", self.api_base))
            .header("x-acs-dingtalk-access-token", token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("DingTalk group send: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("DingTalk parse: {}", e))?;
        if body["processQueryKey"].is_string() {
            Ok(())
        } else {
            Err(format!("DingTalk group error: {}", body))
        }
    }

    /// Send a text or markdown message to a single chat.
    async fn send_single_message(
        &self,
        token: &str,
        message: &OutgoingMessage,
        msg_type: &str,
    ) -> Result<(), String> {
        let (msg_key, msg_param) = build_msg_key_param(msg_type, &message.text);
        let payload = json!({
            "userId": message.channel_id,
            "msgKey": msg_key,
            "msgParam": msg_param,
        });
        let resp = self
            .client
            .post(format!("{}/robot/singleMessages/send", self.api_base))
            .header("x-acs-dingtalk-access-token", token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("DingTalk single send: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("DingTalk parse: {}", e))?;
        if body["processQueryKey"].is_string() {
            Ok(())
        } else {
            Err(format!("DingTalk single error: {}", body))
        }
    }

    /// Send a markdown message to a conversation.
    pub async fn send_markdown_message(
        &self,
        channel_id: &str,
        title: &str,
        text: &str,
        is_group: bool,
    ) -> Result<(), String> {
        let token = self.get_access_token().await?;
        let (url, id_key) = if is_group {
            (
                format!("{}/robot/groupMessages/send", self.api_base),
                "conversationId",
            )
        } else {
            (
                format!("{}/robot/singleMessages/send", self.api_base),
                "userId",
            )
        };
        let msg_param = json!({"title": title, "text": text}).to_string();
        let mut payload = json!({"msgKey": "sampleMarkdown", "msgParam": msg_param});
        payload[id_key] = Value::String(channel_id.to_string());
        let resp = self
            .client
            .post(url)
            .header("x-acs-dingtalk-access-token", token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("DingTalk markdown send: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("DingTalk parse: {}", e))?;
        if body["processQueryKey"].is_string() {
            Ok(())
        } else {
            Err(format!("DingTalk markdown error: {}", body))
        }
    }
}

/// One reconnect cycle: connect, heartbeat, read, parse, ack.
async fn run_stream_cycle(
    running: Arc<Mutex<bool>>,
    client: reqwest::Client,
    client_id: String,
    api_base: String,
    ws_url: String,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
) {
    let (ws_stream, _) = match connect_async(&ws_url).await {
        Ok(pair) => pair,
        Err(e) => {
            error!("DingTalk WS connect failed: {}", e);
            return;
        }
    };
    let (sink, mut read): (SplitSink<WsStream, Message>, SplitStream<WsStream>) = ws_stream.split();
    info!("DingTalk WS connected");

    let heartbeat = DingTalkChannel::handle_heartbeat(sink);

    while let Some(frame) = read.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let envelope = match serde_json::from_str::<DingTalkStreamEnvelope>(&text) {
                    Ok(e) => e,
                    Err(e) => {
                        warn!("DingTalk frame parse error: {}", e);
                        continue;
                    }
                };
                let event_type = envelope.event_type().to_string();
                if event_type == BOT_MESSAGE_EVENT {
                    if let Some(event_id) = envelope.event_id() {
                        if let Err(e) =
                            acknowledge_event(&client, &api_base, &client_id, event_id).await
                        {
                            warn!("DingTalk ack failed: {}", e);
                        }
                    }
                    if let Some(value) = envelope.message() {
                        match parse_message_event(
                            &value,
                            envelope.event_id(),
                            ChannelType::DingTalk,
                        ) {
                            Ok(msg) => {
                                let mut q = incoming.lock().await;
                                q.push_back(msg);
                            }
                            Err(e) => warn!("DingTalk message parse error: {}", e),
                        }
                    }
                } else {
                    debug!("DingTalk event: {}", event_type);
                }
            }
            Ok(Message::Close(_)) => {
                info!("DingTalk WS closed");
                break;
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
            Err(e) => {
                warn!("DingTalk WS error: {}", e);
                break;
            }
            _ => {}
        }
    }

    heartbeat.abort();
    let _ = running.lock().await;
}

/// Exchange `clientId`/`clientSecret` for a Stream Mode ticket and endpoint.
async fn open_connection_ticket(
    client: &reqwest::Client,
    api_base: &str,
    client_id: &str,
    client_secret: &str,
) -> Result<(String, String), String> {
    let resp = client
        .post(format!("{}/gateway/connections/open", api_base))
        .json(&json!({"clientId": client_id, "clientSecret": client_secret}))
        .send()
        .await
        .map_err(|e| format!("DingTalk ticket request: {}", e))?;
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("DingTalk ticket parse: {}", e))?;
    let ticket = body["ticket"]
        .as_str()
        .ok_or_else(|| format!("No ticket in response: {}", body))?
        .to_string();
    let endpoint = body["endpoint"].as_str().unwrap_or("").to_string();
    Ok((ticket, endpoint))
}

/// Build the Stream Mode WebSocket URL for a ticket.
fn build_ws_url(ws_endpoint: &str, client_id: &str, ticket: &str) -> String {
    let base = ws_endpoint.trim_end_matches('/');
    format!(
        "{}/connect?ticket={}&clientId={}&protocol=websocket&version=1.0",
        base, ticket, client_id
    )
}

/// Ack an event so the stream does not redeliver it.
async fn acknowledge_event(
    client: &reqwest::Client,
    api_base: &str,
    client_id: &str,
    event_id: &str,
) -> Result<(), String> {
    let resp = client
        .post(format!("{}/gateway/connections/ack", api_base))
        .json(&json!({"clientId": client_id, "eventId": event_id}))
        .send()
        .await
        .map_err(|e| format!("DingTalk ack request: {}", e))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("DingTalk ack returned {}", resp.status()))
    }
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

/// Parse a DingTalk `BOT_MESSAGE` payload into an [`IncomingMessage`].
///
/// The message object follows the DingTalk robot message schema:
///
/// ```json
/// {
///   "msgtype": "text",
///   "msgId": "xxx",
///   "conversationId": "cidxxx",
///   "conversationType": "1",
///   "senderId": "uidxxx",
///   "senderNick": "Alice",
///   "text": { "content": "hello" },
///   "createAt": 1625241600000
/// }
/// ```
fn parse_message_event(
    message: &Value,
    event_id: Option<&str>,
    channel_type: ChannelType,
) -> Result<IncomingMessage, String> {
    let msg_type = message
        .get("msgtype")
        .and_then(|v| v.as_str())
        .unwrap_or("text");
    let text = extract_message_text(message, msg_type);
    let channel_id = message
        .get("conversationId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let user_id = message
        .get("senderId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let user_name = message
        .get("senderNick")
        .and_then(|v| v.as_str())
        .map(String::from);
    let thread_id = message
        .get("msgId")
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = message
        .get("createAt")
        .and_then(|v| v.as_i64())
        .map(|ms| Utc.timestamp_millis_opt(ms))
        .and_then(|ts| ts.single())
        .unwrap_or_else(Utc::now);

    let attachments = parse_message_attachments(message, msg_type);

    Ok(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type,
        user_id,
        user_name,
        text,
        thread_id,
        attachments,
        timestamp,
        raw: json!({
            "event_id": event_id,
            "msgtype": msg_type,
            "message": message,
        }),
    })
}

/// Extract the human-readable text from a DingTalk message body.
fn extract_message_text(message: &Value, msg_type: &str) -> String {
    match msg_type {
        "text" => message
            .get("text")
            .and_then(|t| t.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "markdown" => message
            .get("markdown")
            .and_then(|m| m.get("text"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                message
                    .get("markdown")
                    .and_then(|m| m.get("content"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("")
            .to_string(),
        "picture" | "image" => "[image]".to_string(),
        "audio" | "voice" => "[audio]".to_string(),
        "file" => {
            let name = message
                .get("content")
                .and_then(|c| c.get("fileName"))
                .and_then(|v| v.as_str())
                .unwrap_or("file");
            format!("[file] {}", name)
        }
        _ => format!("[{}]", msg_type),
    }
}

/// Extract attachment metadata from a DingTalk message body.
fn parse_message_attachments(message: &Value, msg_type: &str) -> Vec<MessageAttachment> {
    if msg_type == "picture" || msg_type == "image" {
        // The image content is either a JSON object or a JSON-encoded string
        // carrying `picMediaId` / `downloadCode`.
        let raw = message.get("content").cloned();
        let obj = raw
            .as_ref()
            .and_then(|c| c.as_str())
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .or(raw);
        if let Some(obj) = obj {
            return vec![MessageAttachment {
                attachment_type: "image".to_string(),
                url: obj
                    .get("downloadCode")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                data: Some(json!({"picMediaId": obj.get("picMediaId").and_then(|v| v.as_str())})),
                mime_type: Some("image/*".to_string()),
            }];
        }
    }
    Vec::new()
}

/// Map an outbound message type to the DingTalk `msgKey` / `msgParam` pair.
fn build_msg_key_param(msg_type: &str, text: &str) -> (String, String) {
    match msg_type {
        "markdown" => (
            "sampleMarkdown".to_string(),
            json!({"title": "OpenSquilla", "text": text}).to_string(),
        ),
        "image" => (
            "sampleImage".to_string(),
            json!({"photo": text}).to_string(),
        ),
        _ => (
            "sampleText".to_string(),
            json!({"content": text}).to_string(),
        ),
    }
}

/// Compute the DingTalk robot webhook signature.
///
/// The scheme signs `"{timestamp}\n{secret}"` with HMAC-SHA256 using the app
/// secret and base64-encodes the digest. `timestamp` is epoch milliseconds.
pub fn compute_webhook_signature(secret: &str, timestamp_ms: &str) -> String {
    let string_to_sign = format!("{}\n{}", timestamp_ms, secret);
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC accepts keys of any size");
    mac.update(string_to_sign.as_bytes());
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

/// Verify a DingTalk robot webhook signature (constant-time).
pub fn verify_webhook_signature(secret: &str, timestamp_ms: &str, sign: &str) -> bool {
    let expected = compute_webhook_signature(secret, timestamp_ms);
    if expected.len() != sign.len() {
        return false;
    }
    let a = expected.as_bytes();
    let b = sign.as_bytes();
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[async_trait::async_trait]
impl Channel for DingTalkChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::DingTalk
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.send_dingtalk_message(message).await
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        // DingTalk's OpenAPI exposes no bot "typing" indicator; this is a
        // no-op kept for trait conformance.
        Ok(())
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        // DingTalk uses Stream Mode rather than webhooks for inbound messages.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_text_message() {
        let envelope = json!({
            "eventId": "event-1",
            "eventType": "BOT_MESSAGE",
            "message": {
                "msgtype": "text",
                "msgId": "msg-1",
                "conversationId": "cid123",
                "senderId": "uid456",
                "senderNick": "Alice",
                "text": {"content": "hello dingtalk"},
                "createAt": 1625241600000i64,
            }
        });
        let parsed = serde_json::from_value::<DingTalkStreamEnvelope>(envelope).unwrap();
        assert_eq!(parsed.event_type(), "BOT_MESSAGE");
        assert_eq!(parsed.event_id(), Some("event-1"));
        let msg = parsed
            .message()
            .and_then(|m| parse_message_event(&m, parsed.event_id(), ChannelType::DingTalk).ok())
            .expect("parse should succeed");
        assert_eq!(msg.channel_type, ChannelType::DingTalk);
        assert_eq!(msg.channel_id, "cid123");
        assert_eq!(msg.user_id, "uid456");
        assert_eq!(msg.user_name.as_deref(), Some("Alice"));
        assert_eq!(msg.text, "hello dingtalk");
        assert_eq!(msg.thread_id.as_deref(), Some("msg-1"));
    }

    #[test]
    fn test_parse_headered_envelope() {
        let envelope = json!({
            "headers": {
                "eventId": "event-9",
                "eventType": "BOT_MESSAGE",
                "topic": "/v1.0/im/bot/messages/get",
            },
            "payload": {
                "msgtype": "markdown",
                "conversationId": "cid9",
                "senderId": "uid9",
                "markdown": {"text": "**bold**"},
            }
        });
        let parsed = serde_json::from_value::<DingTalkStreamEnvelope>(envelope).unwrap();
        assert_eq!(parsed.event_type(), "BOT_MESSAGE");
        assert_eq!(parsed.event_id(), Some("event-9"));
        let msg = parsed
            .message()
            .and_then(|m| parse_message_event(&m, parsed.event_id(), ChannelType::DingTalk).ok())
            .expect("parse should succeed");
        assert_eq!(msg.text, "**bold**");
    }

    #[test]
    fn test_parse_payload_json_string() {
        let envelope = json!({
            "eventId": "e1",
            "eventType": "BOT_MESSAGE",
            "payload": r#"{"msgtype":"text","conversationId":"cid1","senderId":"uid1","text":{"content":"from string"}}"#,
        });
        let parsed = serde_json::from_value::<DingTalkStreamEnvelope>(envelope).unwrap();
        let msg = parsed
            .message()
            .and_then(|m| parse_message_event(&m, parsed.event_id(), ChannelType::DingTalk).ok())
            .expect("parse should succeed");
        assert_eq!(msg.text, "from string");
    }

    #[test]
    fn test_parse_image_message_attachment() {
        let message = json!({
            "msgtype": "picture",
            "conversationId": "cid1",
            "senderId": "uid1",
            "content": r#"{"picMediaId":"@media1","picType":"gif"}"#,
        });
        let msg = parse_message_event(&message, None, ChannelType::DingTalk).unwrap();
        assert_eq!(msg.text, "[image]");
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(msg.attachments[0].attachment_type, "image");
    }

    #[test]
    fn test_non_message_event_is_not_a_message() {
        let envelope = json!({"eventId": "e2", "eventType": "CHATBOT_TOPIC_UPDATE"});
        let parsed = serde_json::from_value::<DingTalkStreamEnvelope>(envelope).unwrap();
        assert_ne!(parsed.event_type(), BOT_MESSAGE_EVENT);
        assert!(parsed.message().is_none());
    }

    #[test]
    fn test_build_ws_url() {
        let url = build_ws_url("wss://stream.dingtalk.com", "appid", "ticket123");
        assert!(
            url.starts_with("wss://stream.dingtalk.com/connect?ticket=ticket123&clientId=appid")
        );
        assert!(url.contains("protocol=websocket"));
        assert!(url.contains("version=1.0"));
    }

    #[test]
    fn test_webhook_signature_roundtrip() {
        let secret = "SEC0001abc";
        let timestamp = "1625241600000";
        let sign = compute_webhook_signature(secret, timestamp);
        assert!(verify_webhook_signature(secret, timestamp, &sign));
        assert!(!verify_webhook_signature(secret, "1625241600001", &sign));
        assert!(!verify_webhook_signature("wrong-secret", timestamp, &sign));
    }

    #[test]
    fn test_webhook_signature_is_hmac_sha256_base64() {
        let secret = "secret";
        let timestamp = "1234567890";
        let sign = compute_webhook_signature(secret, timestamp);
        let string_to_sign = format!("{}\n{}", timestamp, secret);
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(string_to_sign.as_bytes());
        let expected =
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
        assert_eq!(sign, expected);
    }

    #[test]
    fn test_msg_key_param_builder() {
        let (key, param) = build_msg_key_param("text", "hi");
        assert_eq!(key, "sampleText");
        assert_eq!(
            serde_json::from_str::<Value>(&param).unwrap()["content"],
            "hi"
        );

        let (key, param) = build_msg_key_param("markdown", "# title");
        assert_eq!(key, "sampleMarkdown");
        assert_eq!(
            serde_json::from_str::<Value>(&param).unwrap()["text"],
            "# title"
        );
    }

    #[test]
    fn test_reconnect_backoff_is_capped() {
        assert_eq!(reconnect_delay(0).as_secs(), 1);
        assert_eq!(reconnect_delay(1).as_secs(), 2);
        assert!(reconnect_delay(10).as_secs() <= MAX_RECONNECT_DELAY_SECS);
    }
}
