//! Matrix channel adapter — HTTP long polling (Client-Server API).
//!
//! Matrix does not use WebSocket for bots. Instead, events are received by
//! long-polling the `/_matrix/client/v3/sync` endpoint, and messages are sent
//! via the Room Send endpoint. This module implements the raw protocol without
//! the `matrix-nio` SDK:
//!
//! 1. **Sync**: `GET /_matrix/client/v3/sync?timeout=30000&since=...` blocks
//!    up to 30 s and returns a `next_batch` token plus room events.
//! 2. **Send**: `PUT /_matrix/client/v3/rooms/{roomId}/send/m.room.message/{txnId}`
//!    with an `m.text` / `m.notice` / `m.image` / `m.file` payload.
//! 3. **Rooms**: `join` and `leave` room management, and member listing.
//!
//! End-to-end encryption (`m.room.encrypted`) is a stub: the adapter cannot
//! decrypt Olm/Megolm traffic, so encrypted events are logged and skipped.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use chrono::{TimeZone, Utc};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

use serde_json::{json, Value};
use uuid::Uuid;

const DEFAULT_SYNC_TIMEOUT_MS: u64 = 30_000;
const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;
const TYPING_TIMEOUT_MS: u64 = 12_000;

/// Matrix channel adapter using HTTP long polling.
///
/// Incoming messages are parsed by a background task spawned from
/// [`MatrixChannel::start_sync`] and queued for polling via
/// [`MatrixChannel::receive`].
pub struct MatrixChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    homeserver_url: String,
    access_token: String,
    user_id: String,
    sync_token: Arc<Mutex<Option<String>>>,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
}

impl MatrixChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let homeserver_url = config
            .config
            .get("homeserver_url")
            .and_then(|v| v.as_str())
            .ok_or("No homeserver_url configured for Matrix")?
            .trim_end_matches('/')
            .to_string();

        let access_token = config
            .config
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or("No access_token configured for Matrix")?
            .to_string();

        let user_id = config
            .config
            .get("user_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            config,
            client,
            homeserver_url,
            access_token,
            user_id,
            sync_token: Arc::new(Mutex::new(None)),
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Perform a single long-poll sync. Returns the messages parsed from this
    /// batch and advances the stored `since` token.
    pub async fn sync_once(&self) -> Result<Vec<IncomingMessage>, String> {
        let url = self.build_sync_url().await;
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| format!("Matrix sync request failed: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Matrix sync parse error: {}", e))?;
        let (next_batch, messages) = parse_sync_response(&body);
        if let Some(next) = next_batch {
            let mut token = self.sync_token.lock().await;
            *token = Some(next);
        }
        Ok(messages)
    }

    /// Start the long-polling sync loop in the background.
    pub async fn start_sync(&self) -> Result<(), String> {
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
        let homeserver_url = self.homeserver_url.clone();
        let access_token = self.access_token.clone();
        let sync_token = self.sync_token.clone();

        tokio::spawn(async move {
            info!("Matrix sync loop starting");
            let mut attempt: u32 = 0;
            loop {
                if !*running.lock().await {
                    return;
                }
                let mut url = format!(
                    "{}/_matrix/client/v3/sync?timeout={}",
                    homeserver_url, DEFAULT_SYNC_TIMEOUT_MS
                );
                {
                    let token = sync_token.lock().await;
                    if let Some(ref since) = *token {
                        url.push_str(&format!("&since={}", since));
                    }
                }
                match client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", access_token))
                    .send()
                    .await
                {
                    Ok(resp) => match resp.json::<Value>().await {
                        Ok(body) => {
                            let (next_batch, messages) = parse_sync_response(&body);
                            if let Some(next) = next_batch {
                                *sync_token.lock().await = Some(next);
                            }
                            if !messages.is_empty() {
                                let mut q = incoming.lock().await;
                                q.extend(messages);
                            }
                            attempt = 0;
                            continue; // long poll already waited; poll again
                        }
                        Err(e) => {
                            warn!("Matrix sync parse error: {}", e);
                        }
                    },
                    Err(e) => {
                        warn!("Matrix sync request failed: {}", e);
                    }
                }
                if !*running.lock().await {
                    return;
                }
                let delay = matrix_reconnect_delay(attempt);
                attempt = attempt.saturating_add(1).min(10);
                tokio::time::sleep(delay).await;
            }
        });
        Ok(())
    }

    /// Stop the sync loop.
    pub async fn stop_sync(&self) {
        *self.running.lock().await = false;
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    async fn build_sync_url(&self) -> String {
        let mut url = format!(
            "{}/_matrix/client/v3/sync?timeout={}",
            self.homeserver_url, DEFAULT_SYNC_TIMEOUT_MS
        );
        let token = self.sync_token.lock().await;
        if let Some(ref since) = *token {
            url.push_str(&format!("&since={}", since));
        }
        url
    }

    /// Send a room message, selecting the `msgtype` from the attachment when
    /// present (m.text by default, m.image / m.file / m.audio / m.video).
    async fn send_matrix_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let txn_id = Uuid::new_v4().to_string();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
            self.homeserver_url, message.channel_id, txn_id
        );
        let payload = build_room_message_payload(message);
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Matrix send error: {}", e))?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Matrix send parse error: {}", e))?;
        if status.is_success() && body["event_id"].is_string() {
            Ok(())
        } else {
            Err(format!(
                "Matrix error: {}",
                body["error"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send an `m.notice` message (used for non-user-visible bot output).
    pub async fn send_notice(&self, room_id: &str, text: &str) -> Result<(), String> {
        let mut msg = OutgoingMessage::new(room_id.to_string(), ChannelType::Matrix, text.to_string());
        msg.metadata = json!({"msgtype": "m.notice"});
        self.send_matrix_message(&msg).await
    }

    /// Send an `m.image` message by `mxc://` URL.
    pub async fn send_image(&self, room_id: &str, mxc_url: &str, alt_text: &str) -> Result<(), String> {
        let mut msg = OutgoingMessage::new(room_id.to_string(), ChannelType::Matrix, alt_text.to_string());
        msg.attachments.push(MessageAttachment {
            attachment_type: "m.image".to_string(),
            url: Some(mxc_url.to_string()),
            data: Some(json!({"mimetype": "image/*"})),
            mime_type: Some("image/*".to_string()),
        });
        self.send_matrix_message(&msg).await
    }

    /// Join a room by ID or alias (`#alias:server`).
    pub async fn join_room(&self, room_id_or_alias: &str) -> Result<String, String> {
        let url = format!(
            "{}/_matrix/client/v3/join/{}",
            self.homeserver_url,
            urlencode_path(room_id_or_alias)
        );
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| format!("Matrix join request: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Matrix join parse: {}", e))?;
        body["room_id"].as_str().map(String::from).ok_or_else(|| {
            format!(
                "Matrix join error: {}",
                body["error"].as_str().unwrap_or("unknown")
            )
        })
    }

    /// Leave a room by ID.
    pub async fn leave_room(&self, room_id: &str) -> Result<(), String> {
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/leave",
            self.homeserver_url, room_id
        );
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| format!("Matrix leave request: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Matrix leave failed: {}", resp.status()))
        }
    }

    /// List the members of a room.
    pub async fn get_room_members(&self, room_id: &str) -> Result<Vec<Value>, String> {
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/members",
            self.homeserver_url, room_id
        );
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .send()
            .await
            .map_err(|e| format!("Matrix members request: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Matrix members parse: {}", e))?;
        Ok(body
            .get("chunk")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default())
    }

    /// Whether the adapter can decrypt end-to-end encrypted messages.
    ///
    /// Always `false` for now; E2EE (Olm/Megolm) is a future work item.
    pub fn e2ee_supported() -> bool {
        false
    }
}

/// Parse a `/sync` response into the next batch token and incoming messages.
fn parse_sync_response(body: &Value) -> (Option<String>, Vec<IncomingMessage>) {
    let next_batch = body.get("next_batch").and_then(|v| v.as_str()).map(String::from);
    let mut messages = Vec::new();
    let Some(rooms) = body.get("rooms").and_then(|r| r.as_object()) else {
        return (next_batch, messages);
    };
    if let Some(join) = rooms.get("join").and_then(|j| j.as_object()) {
        for (room_id, room_data) in join {
            let Some(events) = room_data
                .get("timeline")
                .and_then(|t| t.get("events"))
                .and_then(|e| e.as_array())
            else {
                continue;
            };
            for event in events {
                match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "m.room.message" => {
                        if let Some(msg) = parse_room_message(room_id, event) {
                            messages.push(msg);
                        }
                    }
                    "m.room.encrypted" => handle_encrypted_event(event),
                    "m.room.member" => log_member_event(event),
                    _ => {}
                }
            }
        }
    }
    if let Some(invite) = rooms.get("invite").and_then(|i| i.as_object()) {
        for (room_id, _room_data) in invite {
            info!("Matrix bot invited to room {}", room_id);
        }
    }
    (next_batch, messages)
}

/// Parse a single `m.room.message` event into an [`IncomingMessage`].
fn parse_room_message(room_id: &str, event: &Value) -> Option<IncomingMessage> {
    let content = event.get("content")?;
    let msgtype = content.get("msgtype").and_then(|v| v.as_str()).unwrap_or("m.text");
    let body_text = content.get("body").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let sender = event.get("sender").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let event_id = event.get("event_id").and_then(|v| v.as_str()).map(String::from);
    let thread_id = content
        .get("m.relates_to")
        .and_then(|r| r.get("event_id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = event
        .get("origin_server_ts")
        .and_then(|v| v.as_i64())
        .map(|ms| Utc.timestamp_millis_opt(ms))
        .and_then(|ts| ts.single())
        .unwrap_or_else(Utc::now);
    let attachments = parse_message_attachments(content, msgtype);
    let text = if attachments.is_empty() {
        body_text
    } else {
        format!("[{}] {}", msgtype.trim_start_matches("m."), body_text)
    };
    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id: room_id.to_string(),
        channel_type: ChannelType::Matrix,
        user_id: sender,
        user_name: None,
        text,
        thread_id: thread_id.or(event_id),
        attachments,
        timestamp,
        raw: event.clone(),
    })
}

/// Extract attachments from a message body for non-text `msgtype`s.
fn parse_message_attachments(content: &Value, msgtype: &str) -> Vec<MessageAttachment> {
    if msgtype == "m.text" || msgtype == "m.notice" || msgtype == "m.emote" {
        return Vec::new();
    }
    let url = content.get("url").and_then(|v| v.as_str()).map(String::from);
    let mime = content
        .get("info")
        .and_then(|i| i.get("mimetype"))
        .and_then(|v| v.as_str())
        .map(String::from);
    vec![MessageAttachment {
        attachment_type: msgtype.trim_start_matches("m.").to_string(),
        url,
        data: Some(content.clone()),
        mime_type: mime,
    }]
}

/// E2EE stub: log the encryption algorithm and skip the event.
fn handle_encrypted_event(event: &Value) {
    let algorithm = event
        .get("content")
        .and_then(|c| c.get("algorithm"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    warn!(
        "Matrix E2EE message not decrypted (algorithm={}); E2EE support is not yet implemented",
        algorithm
    );
}

/// Log an `m.room.member` membership change.
fn log_member_event(event: &Value) {
    let sender = event.get("sender").and_then(|v| v.as_str()).unwrap_or("");
    let membership = event
        .get("content")
        .and_then(|c| c.get("membership"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    info!("Matrix member event: {} -> {}", sender, membership);
}

/// Build the `m.room.message` payload for an outbound message.
fn build_room_message_payload(message: &OutgoingMessage) -> Value {
    if message.attachments.is_empty() {
        let msgtype = message
            .metadata
            .get("msgtype")
            .and_then(|v| v.as_str())
            .unwrap_or("m.text");
        return json!({"msgtype": msgtype, "body": message.text});
    }
    let att = &message.attachments[0];
    let msgtype = match att.attachment_type.as_str() {
        "image" | "m.image" => "m.image",
        "audio" | "m.audio" => "m.audio",
        "video" | "m.video" => "m.video",
        "file" | "m.file" => "m.file",
        _ => "m.file",
    };
    json!({
        "msgtype": msgtype,
        "body": message.text,
        "url": att.url.as_deref().unwrap_or(""),
        "info": att.data.clone().unwrap_or(Value::Null),
    })
}

/// Percent-encode the reserved characters in a Matrix room id/alias path
/// segment (`#`, `?`, `%`). Room ids like `!abc:server` pass through.
fn urlencode_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'#' | b'?' | b'%' | b' ' => {
                out.push_str(&format!("%{:02X}", b));
            }
            _ => out.push(b as char),
        }
    }
    out
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn matrix_reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

#[async_trait::async_trait]
impl Channel for MatrixChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Matrix
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.send_matrix_message(message).await
    }

    async fn send_typing(&self, channel_id: &str) -> Result<(), String> {
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/typing/{}",
            self.homeserver_url, channel_id, self.user_id
        );
        let resp = self
            .client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.access_token))
            .json(&json!({"typing": true, "timeout": TYPING_TIMEOUT_MS}))
            .send()
            .await
            .map_err(|e| format!("Matrix typing request: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Matrix typing failed: {}", resp.status()))
        }
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        // Matrix uses long-polling, not HTTP webhooks.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_sync_text_message() {
        let body = json!({
            "next_batch": "s72595_4483_1934",
            "rooms": {
                "join": {
                    "!room:server": {
                        "timeline": {
                            "events": [{
                                "type": "m.room.message",
                                "sender": "@alice:server",
                                "event_id": "$evt1",
                                "origin_server_ts": 1625241600000,
                                "content": {"msgtype": "m.text", "body": "hello matrix"}
                            }]
                        }
                    }
                }
            }
        });
        let (next_batch, messages) = parse_sync_response(&body);
        assert_eq!(next_batch.as_deref(), Some("s72595_4483_1934"));
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.channel_type, ChannelType::Matrix);
        assert_eq!(msg.channel_id, "!room:server");
        assert_eq!(msg.user_id, "@alice:server");
        assert_eq!(msg.text, "hello matrix");
    }

    #[test]
    fn test_parse_sync_image_message() {
        let body = json!({
            "next_batch": "s2",
            "rooms": {
                "join": {
                    "!room:server": {
                        "timeline": {
                            "events": [{
                                "type": "m.room.message",
                                "sender": "@bob:server",
                                "event_id": "$evt2",
                                "origin_server_ts": 1625241600000,
                                "content": {
                                    "msgtype": "m.image",
                                    "body": "photo.png",
                                    "url": "mxc://server/abc",
                                    "info": {"mimetype": "image/png", "size": 1024}
                                }
                            }]
                        }
                    }
                }
            }
        });
        let (_, messages) = parse_sync_response(&body);
        assert_eq!(messages.len(), 1);
        let msg = &messages[0];
        assert_eq!(msg.text, "[image] photo.png");
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(msg.attachments[0].url.as_deref(), Some("mxc://server/abc"));
        assert_eq!(msg.attachments[0].mime_type.as_deref(), Some("image/png"));
    }

    #[test]
    fn test_parse_sync_skips_encrypted_and_member() {
        let body = json!({
            "next_batch": "s3",
            "rooms": {
                "join": {
                    "!room:server": {
                        "timeline": {
                            "events": [
                                {
                                    "type": "m.room.encrypted",
                                    "content": {"algorithm": "m.megolm.v1.aes-sha2"}
                                },
                                {
                                    "type": "m.room.member",
                                    "sender": "@carol:server",
                                    "content": {"membership": "join"}
                                }
                            ]
                        }
                    }
                }
            }
        });
        let (_, messages) = parse_sync_response(&body);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_parse_sync_notice_message() {
        let body = json!({
            "next_batch": "s4",
            "rooms": {
                "join": {
                    "!r:server": {
                        "timeline": {
                            "events": [{
                                "type": "m.room.message",
                                "sender": "@bot:server",
                                "event_id": "$e",
                                "origin_server_ts": 1625241600000,
                                "content": {"msgtype": "m.notice", "body": "notice text"}
                            }]
                        }
                    }
                }
            }
        });
        let (_, messages) = parse_sync_response(&body);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "notice text");
        assert!(messages[0].attachments.is_empty());
    }

    #[test]
    fn test_build_text_payload() {
        let msg = OutgoingMessage::new("!r:server".to_string(), ChannelType::Matrix, "hi".to_string());
        let payload = build_room_message_payload(&msg);
        assert_eq!(payload["msgtype"], "m.text");
        assert_eq!(payload["body"], "hi");
    }

    #[test]
    fn test_build_notice_payload() {
        let mut msg = OutgoingMessage::new("!r:server".to_string(), ChannelType::Matrix, "note".to_string());
        msg.metadata = json!({"msgtype": "m.notice"});
        let payload = build_room_message_payload(&msg);
        assert_eq!(payload["msgtype"], "m.notice");
    }

    #[test]
    fn test_build_image_payload() {
        let mut msg = OutgoingMessage::new("!r:server".to_string(), ChannelType::Matrix, "pic".to_string());
        msg.attachments.push(MessageAttachment {
            attachment_type: "image".to_string(),
            url: Some("mxc://server/key".to_string()),
            data: Some(json!({"mimetype": "image/png"})),
            mime_type: Some("image/png".to_string()),
        });
        let payload = build_room_message_payload(&msg);
        assert_eq!(payload["msgtype"], "m.image");
        assert_eq!(payload["url"], "mxc://server/key");
        assert_eq!(payload["info"]["mimetype"], "image/png");
    }

    #[test]
    fn test_urlencode_alias() {
        assert_eq!(urlencode_path("#alias:server"), "%23alias:server");
        assert_eq!(urlencode_path("!room:server"), "!room:server");
        assert_eq!(urlencode_path("a b%c"), "a%20b%25c");
    }

    #[test]
    fn test_reconnect_backoff_is_capped() {
        assert_eq!(matrix_reconnect_delay(0).as_secs(), 1);
        assert_eq!(matrix_reconnect_delay(1).as_secs(), 2);
        assert!(matrix_reconnect_delay(10).as_secs() <= MAX_RECONNECT_DELAY_SECS);
    }

    #[test]
    fn test_e2ee_stub() {
        assert!(!MatrixChannel::e2ee_supported());
    }
}
