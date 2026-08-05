//! Feishu (Lark) channel adapter — raw HTTP OpenAPI + event subscription.
//!
//! This module implements the Feishu Open Platform protocol without the
//! `lark-oapi` SDK:
//!
//! 1. **Auth**: `POST /auth/v3/tenant_access_token/internal` exchanges the app
//!    `app_id`/`app_secret` for a tenant access token that is cached until it
//!    expires (with a safety margin).
//! 2. **Send**: `POST /im/v1/messages?receive_id_type=...` with a
//!    JSON-encoded `content` string. Text, post (rich text), image and
//!    interactive card messages are supported.
//! 3. **Receive**: Feishu pushes events to an HTTP webhook. The subscription
//!    URL-verification challenge is answered with the echoed `challenge` and
//!    `im.message.receive_v1` callbacks are parsed into [`IncomingMessage`]s.
//! 4. **Signature**: event callbacks can be verified with
//!    `HMAC-SHA256(verification_token, timestamp + nonce + body)`.
//!
//! Token refresh is transparent: a response carrying a token error code
//! (e.g. `99991663`) triggers a refresh and a single retry.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use base64::Engine;
use chrono::{DateTime, TimeZone, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::warn;

use serde_json::{Value, json};
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_API_BASE: &str = "https://open.feishu.cn/open-apis";
/// Token refresh margin: treat the token as expired this many seconds early.
const TOKEN_EXPIRE_MARGIN_SECS: i64 = 300;
const DEFAULT_TOKEN_EXPIRE_SECS: i64 = 7_200;

/// Feishu error codes that indicate a stale tenant access token.
fn is_token_error(code: i64) -> bool {
    matches!(code, 99_991_661 | 99_991_663 | 99_991_668)
}

#[derive(Debug, Clone)]
struct TenantToken {
    token: String,
    expires_at: DateTime<Utc>,
}

/// Feishu/Lark channel adapter using the raw HTTP REST API.
///
/// Incoming webhook events are parsed by [`FeishuChannel::handle_event_callback`]
/// and queued for polling via [`FeishuChannel::receive`].
pub struct FeishuChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    app_id: String,
    app_secret: String,
    verification_token: Option<String>,
    tenant_token: Arc<Mutex<Option<TenantToken>>>,
    api_base: String,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
}

impl FeishuChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let app_id = config
            .config
            .get("app_id")
            .and_then(|v| v.as_str())
            .ok_or("No app_id configured for Feishu")?
            .to_string();
        let app_secret = config
            .config
            .get("app_secret")
            .and_then(|v| v.as_str())
            .ok_or("No app_secret configured for Feishu")?
            .to_string();
        let verification_token = config
            .config
            .get("verification_token")
            .and_then(|v| v.as_str())
            .map(String::from);
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        Ok(Self {
            config,
            client,
            app_id,
            app_secret,
            verification_token,
            tenant_token: Arc::new(Mutex::new(None)),
            api_base,
            incoming: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Obtain (and cache) a tenant access token.
    async fn get_tenant_token(&self) -> Result<String, String> {
        let cached = self.tenant_token.lock().await.clone();
        if let Some(cached) = &cached {
            if Utc::now() < cached.expires_at {
                return Ok(cached.token.clone());
            }
        }
        let fresh = self.request_tenant_token().await?;
        let token = fresh.token.clone();
        *self.tenant_token.lock().await = Some(fresh);
        Ok(token)
    }

    /// Force a token refresh on the next call.
    async fn refresh_tenant_token(&self) -> Result<String, String> {
        *self.tenant_token.lock().await = None;
        self.get_tenant_token().await
    }

    /// Fetch a fresh tenant token from the auth endpoint.
    async fn request_tenant_token(&self) -> Result<TenantToken, String> {
        let resp = self
            .client
            .post(format!(
                "{}/auth/v3/tenant_access_token/internal",
                self.api_base
            ))
            .json(&json!({"app_id": self.app_id, "app_secret": self.app_secret}))
            .send()
            .await
            .map_err(|e| format!("Feishu auth request failed: {}", e))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Feishu auth parse error: {}", e))?;
        let code = body["code"].as_i64().unwrap_or(-1);
        if code != 0 {
            return Err(format!(
                "Feishu auth error: {} (code: {})",
                body["msg"].as_str().unwrap_or("unknown"),
                code
            ));
        }
        let token = body["tenant_access_token"]
            .as_str()
            .ok_or("No tenant_access_token in response")?
            .to_string();
        let expire_secs = body["expire"].as_i64().unwrap_or(DEFAULT_TOKEN_EXPIRE_SECS);
        let expires_at = Utc::now()
            .checked_add_signed(chrono::Duration::seconds(
                expire_secs - TOKEN_EXPIRE_MARGIN_SECS,
            ))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::hours(2));
        Ok(TenantToken { token, expires_at })
    }

    /// Send a message, transparently refreshing the token once on a token error.
    async fn send_feishu_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let token = self.get_tenant_token().await?;
        let body = self.post_message(&token, message).await?;
        let code = body["code"].as_i64().unwrap_or(-1);
        if code == 0 {
            return Ok(());
        }
        if is_token_error(code) {
            let token = self.refresh_tenant_token().await?;
            let body = self.post_message(&token, message).await?;
            if body["code"].as_i64().unwrap_or(-1) == 0 {
                Ok(())
            } else {
                Err(feishu_error(&body))
            }
        } else {
            Err(feishu_error(&body))
        }
    }

    /// POST a message to `im/v1/messages` and return the response body.
    async fn post_message(&self, token: &str, message: &OutgoingMessage) -> Result<Value, String> {
        let msg_type = message
            .metadata
            .get("msg_type")
            .and_then(|v| v.as_str())
            .unwrap_or("text");
        let receive_id_type = receive_id_type(&message.channel_id);
        let content = build_send_content(msg_type, &message.text);
        let payload = json!({
            "receive_id": message.channel_id,
            "msg_type": msg_type,
            "content": content,
        });
        let resp = self
            .client
            .post(format!("{}/im/v1/messages", self.api_base))
            .query(&[("receive_id_type", receive_id_type)])
            .header("Authorization", format!("Bearer {}", token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Feishu send error: {}", e))?;
        resp.json()
            .await
            .map_err(|e| format!("Feishu response parse: {}", e))
    }

    /// Send a plain-text message.
    pub async fn send_text_message(&self, receive_id: &str, text: &str) -> Result<(), String> {
        self.send_feishu_message(&OutgoingMessage::new(
            receive_id.to_string(),
            ChannelType::Feishu,
            text.to_string(),
        ))
        .await
    }

    /// Send a rich-text (post) message.
    pub async fn send_post_message(
        &self,
        receive_id: &str,
        title: &str,
        text: &str,
    ) -> Result<(), String> {
        let mut msg = OutgoingMessage::new(
            receive_id.to_string(),
            ChannelType::Feishu,
            text.to_string(),
        );
        msg.metadata = json!({"msg_type": "post", "post_title": title});
        self.send_feishu_message(&msg).await
    }

    /// Send an image by `image_key`.
    pub async fn send_image_message(
        &self,
        receive_id: &str,
        image_key: &str,
    ) -> Result<(), String> {
        let mut msg = OutgoingMessage::new(
            receive_id.to_string(),
            ChannelType::Feishu,
            image_key.to_string(),
        );
        msg.metadata = json!({"msg_type": "image"});
        self.send_feishu_message(&msg).await
    }

    /// Send an interactive card.
    pub async fn send_card_message(&self, receive_id: &str, card: &Value) -> Result<(), String> {
        let text = serde_json::to_string(card).unwrap_or_default();
        let mut msg = OutgoingMessage::new(receive_id.to_string(), ChannelType::Feishu, text);
        msg.metadata = json!({"msg_type": "interactive"});
        self.send_feishu_message(&msg).await
    }

    /// Build an interactive card JSON body.
    pub fn build_interactive_card(
        &self,
        title: &str,
        text: &str,
        buttons: &[(String, String)],
    ) -> Value {
        let mut elements = vec![json!({"tag": "div", "text": {"tag": "lark_md", "content": text}})];
        if !buttons.is_empty() {
            let actions: Vec<Value> = buttons
                .iter()
                .map(|(label, value)| {
                    json!({
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": label},
                        "type": "primary",
                        "value": {"value": value},
                    })
                })
                .collect();
            elements.push(json!({"tag": "hr"}));
            elements.push(json!({"tag": "action", "actions": actions}));
        }
        json!({
            "config": {"wide_screen_mode": true},
            "header": {"title": {"tag": "plain_text", "content": title}, "template": "blue"},
            "elements": elements,
        })
    }

    /// Handle a webhook POST body.
    ///
    /// Returns `Some(challenge)` when the payload is the subscription
    /// URL-verification challenge (which the caller should echo verbatim), and
    /// `None` for ordinary event callbacks (which are parsed and queued).
    pub fn receive_webhook(&self, payload: &Value) -> Result<Option<String>, String> {
        if payload.get("type").and_then(|v| v.as_str()) == Some("url_verification") {
            let challenge = payload
                .get("challenge")
                .and_then(|v| v.as_str())
                .ok_or("No challenge in url_verification payload")?
                .to_string();
            if let Some(expected) = &self.verification_token {
                if !expected.is_empty() {
                    let token = payload.get("token").and_then(|v| v.as_str()).unwrap_or("");
                    if token != expected {
                        return Err("Feishu verification token mismatch".to_string());
                    }
                }
            }
            return Ok(Some(challenge));
        }
        Ok(None)
    }

    /// Handle an event subscription callback. Verifies the header token when a
    /// verification token is configured, parses the message, and queues it.
    pub async fn handle_event_callback(&self, payload: &Value) -> Option<IncomingMessage> {
        if let Some(expected) = &self.verification_token {
            if !expected.is_empty() {
                let token = payload
                    .get("header")
                    .and_then(|h| h.get("token"))
                    .and_then(|v| v.as_str());
                if token != Some(expected.as_str()) {
                    warn!("Feishu event token mismatch; ignoring callback");
                    return None;
                }
            }
        }
        let msg = parse_event_callback(payload)?;
        let mut q = self.incoming.lock().await;
        q.push_back(msg.clone());
        Some(msg)
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Verify a Feishu event callback signature.
    ///
    /// The signature is `base64(HMAC-SHA256(token, timestamp + nonce + body))`
    /// carried in the `X-Lark-Signature` header. Returns `true` when no
    /// verification token is configured (verification disabled).
    pub fn verify_event_signature(
        &self,
        timestamp: &str,
        nonce: &str,
        body: &str,
        signature: &str,
    ) -> bool {
        let Some(token) = &self.verification_token else {
            return true;
        };
        if token.is_empty() {
            return true;
        }
        verify_feishu_signature(token, timestamp, nonce, body, signature)
    }
}

/// Map a Feishu id prefix to the `receive_id_type` query parameter.
fn receive_id_type(channel_id: &str) -> &'static str {
    if channel_id.starts_with("ou_") {
        "open_id"
    } else if channel_id.starts_with("on_") {
        "union_id"
    } else if channel_id.starts_with("oi_") {
        "user_id"
    } else {
        "chat_id"
    }
}

/// Build the JSON-encoded `content` field for a Feishu message.
fn build_send_content(msg_type: &str, text: &str) -> String {
    match msg_type {
        "post" => serde_json::to_string(&json!({
            "post": {
                "zh_cn": {
                    "title": "OpenSquilla",
                    "content": [[{"tag": "text", "text": text}]],
                }
            }
        }))
        .unwrap_or_default(),
        "image" => serde_json::to_string(&json!({"image_key": text})).unwrap_or_default(),
        "interactive" => serde_json::to_string(&json!({
            "config": {"wide_screen_mode": true},
            "header": {"title": {"tag": "plain_text", "content": "OpenSquilla"}, "template": "blue"},
            "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": text}}],
        }))
        .unwrap_or_default(),
        _ => serde_json::to_string(&json!({"text": text})).unwrap_or_default(),
    }
}

/// Format a Feishu API error from the response body.
fn feishu_error(body: &Value) -> String {
    format!(
        "Feishu API error: {} (code: {})",
        body["msg"].as_str().unwrap_or("unknown"),
        body["code"]
    )
}

/// Pure HMAC-SHA256 event signature verification.
pub fn verify_feishu_signature(
    token: &str,
    timestamp: &str,
    nonce: &str,
    body: &str,
    signature: &str,
) -> bool {
    let string_to_sign = format!("{}{}{}", timestamp, nonce, body);
    let mut mac = HmacSha256::new_from_slice(token.as_bytes()).expect("HMAC accepts any key size");
    mac.update(string_to_sign.as_bytes());
    let expected = base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes());
    if expected.len() != signature.len() {
        return false;
    }
    let a = expected.as_bytes();
    let b = signature.as_bytes();
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Parse an `im.message.receive_v1` event callback into an [`IncomingMessage`].
fn parse_event_callback(payload: &Value) -> Option<IncomingMessage> {
    let header = payload.get("header")?;
    let event_type = header.get("event_type").and_then(|v| v.as_str())?;
    if event_type != "im.message.receive_v1" {
        return None;
    }
    let event = payload.get("event")?;
    let message = event.get("message")?;
    let message_type = message
        .get("message_type")
        .and_then(|v| v.as_str())
        .unwrap_or("text");
    let content_str = message
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    let content: Value = serde_json::from_str(content_str).unwrap_or(Value::Null);

    let text = match message_type {
        "text" => content
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "post" => extract_post_text(&content),
        "image" => "[image]".to_string(),
        _ => format!("[{}]", message_type),
    };

    let channel_id = message
        .get("chat_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let sender = event.get("sender");
    let user_id = sender
        .and_then(|s| s.get("sender_id"))
        .and_then(|id| id.get("open_id"))
        .and_then(|v| v.as_str())
        .or_else(|| {
            sender
                .and_then(|s| s.get("sender_id"))
                .and_then(|id| id.get("user_id"))
                .and_then(|v| v.as_str())
        })
        .unwrap_or("")
        .to_string();
    let user_name = sender
        .and_then(|s| s.get("sender_type"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let thread_id = message
        .get("message_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let timestamp = message
        .get("create_time")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<i64>().ok())
        .map(|ms| Utc.timestamp_millis_opt(ms))
        .and_then(|ts| ts.single())
        .unwrap_or_else(Utc::now);

    let attachments = if message_type == "image" {
        content
            .get("image_key")
            .and_then(|v| v.as_str())
            .map(|key| {
                vec![MessageAttachment {
                    attachment_type: "image".to_string(),
                    url: None,
                    data: Some(json!({"image_key": key})),
                    mime_type: Some("image/*".to_string()),
                }]
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type: ChannelType::Feishu,
        user_id,
        user_name,
        text,
        thread_id,
        attachments,
        timestamp,
        raw: payload.clone(),
    })
}

/// Flatten the text nodes of a Feishu post message.
fn extract_post_text(content: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(post) = content.get("post") {
        for lang in ["zh_cn", "en_us", "ja_jp"] {
            let Some(lang_obj) = post.get(lang) else {
                continue;
            };
            let Some(rows) = lang_obj.get("content").and_then(|c| c.as_array()) else {
                break;
            };
            for row in rows {
                let Some(cells) = row.as_array() else {
                    continue;
                };
                for cell in cells {
                    if let Some(text) = cell.get("text").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    }
                }
            }
            break;
        }
    }
    parts.join(" ")
}

#[async_trait::async_trait]
impl Channel for FeishuChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Feishu
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.send_feishu_message(message).await
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        // Feishu exposes no bot "typing" indicator over the OpenAPI; no-op.
        Ok(())
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_channel() -> FeishuChannel {
        let cfg = ChannelConfig {
            channel_type: ChannelType::Feishu,
            channel_id: "feishu".to_string(),
            name: "Feishu".to_string(),
            enabled: true,
            config: json!({
                "app_id": "cli_abc",
                "app_secret": "secret",
                "verification_token": "vtoken",
            }),
        };
        FeishuChannel::new(cfg).unwrap()
    }

    fn text_callback() -> Value {
        json!({
            "schema": "2.0",
            "header": {
                "event_id": "evt-1",
                "event_type": "im.message.receive_v1",
                "token": "vtoken",
                "app_id": "cli_abc",
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_123", "user_id": "uid_123"},
                    "sender_type": "user",
                },
                "message": {
                    "message_id": "om_1",
                    "chat_id": "oc_456",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": "{\"text\":\"hello feishu\"}",
                    "create_time": "1625241600000",
                },
            },
        })
    }

    #[test]
    fn test_parse_text_callback() {
        let msg = parse_event_callback(&text_callback()).expect("should parse");
        assert_eq!(msg.channel_type, ChannelType::Feishu);
        assert_eq!(msg.channel_id, "oc_456");
        assert_eq!(msg.user_id, "ou_123");
        assert_eq!(msg.text, "hello feishu");
        assert_eq!(msg.thread_id.as_deref(), Some("om_1"));
    }

    #[test]
    fn test_parse_post_callback() {
        let payload = json!({
            "header": {"event_type": "im.message.receive_v1", "token": "vtoken"},
            "event": {
                "message": {
                    "message_id": "om_2",
                    "chat_id": "oc_1",
                    "message_type": "post",
                    "content": r#"{"post":{"zh_cn":{"title":"t","content":[[{"tag":"text","text":"hello"},{"tag":"text","text":" world"}]]}}}"#,
                    "create_time": "1625241600000",
                }
            }
        });
        let msg = parse_event_callback(&payload).expect("should parse");
        assert_eq!(msg.text, "hello world");
    }

    #[test]
    fn test_parse_image_callback_attachment() {
        let payload = json!({
            "header": {"event_type": "im.message.receive_v1", "token": "vtoken"},
            "event": {
                "message": {
                    "message_id": "om_3",
                    "chat_id": "oc_1",
                    "message_type": "image",
                    "content": r#"{"image_key":"img_v2_xxx"}"#,
                    "create_time": "1625241600000",
                }
            }
        });
        let msg = parse_event_callback(&payload).expect("should parse");
        assert_eq!(msg.text, "[image]");
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(
            msg.attachments[0].data.as_ref().unwrap()["image_key"],
            "img_v2_xxx"
        );
    }

    #[test]
    fn test_non_message_event_ignored() {
        let payload = json!({"header": {"event_type": "im.chat.updated_v1"}});
        assert!(parse_event_callback(&payload).is_none());
    }

    #[test]
    fn test_url_verification_challenge() {
        let channel = test_channel();
        let payload = json!({
            "type": "url_verification",
            "challenge": "challenge-str",
            "token": "vtoken",
        });
        let result = channel.receive_webhook(&payload).unwrap();
        assert_eq!(result.as_deref(), Some("challenge-str"));
    }

    #[test]
    fn test_url_verification_bad_token() {
        let channel = test_channel();
        let payload = json!({
            "type": "url_verification",
            "challenge": "c",
            "token": "wrong",
        });
        assert!(channel.receive_webhook(&payload).is_err());
    }

    #[test]
    fn test_card_builder() {
        let channel = test_channel();
        let card = channel.build_interactive_card(
            "Title",
            "Body text",
            &[("OK".to_string(), "ok-value".to_string())],
        );
        assert_eq!(card["config"]["wide_screen_mode"], true);
        assert_eq!(card["header"]["title"]["content"], "Title");
        assert_eq!(card["elements"][0]["text"]["content"], "Body text");
        assert_eq!(card["elements"][2]["tag"], "action");
        assert_eq!(
            card["elements"][2]["actions"][0]["value"]["value"],
            "ok-value"
        );
    }

    #[test]
    fn test_card_builder_no_buttons() {
        let channel = test_channel();
        let card = channel.build_interactive_card("T", "B", &[]);
        assert_eq!(card["elements"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_send_content() {
        assert_eq!(
            serde_json::from_str::<Value>(&build_send_content("text", "hi")).unwrap()["text"],
            "hi"
        );
        let post: Value = serde_json::from_str(&build_send_content("post", "hello")).unwrap();
        assert_eq!(post["post"]["zh_cn"]["content"][0][0]["text"], "hello");
        let image: Value = serde_json::from_str(&build_send_content("image", "img_1")).unwrap();
        assert_eq!(image["image_key"], "img_1");
        let card: Value = serde_json::from_str(&build_send_content("interactive", "hi")).unwrap();
        assert!(card["elements"].is_array());
    }

    #[test]
    fn test_receive_id_type() {
        assert_eq!(receive_id_type("ou_1"), "open_id");
        assert_eq!(receive_id_type("on_1"), "union_id");
        assert_eq!(receive_id_type("oi_1"), "user_id");
        assert_eq!(receive_id_type("oc_1"), "chat_id");
    }

    #[test]
    fn test_signature_roundtrip() {
        let token = "vtoken";
        let body = r#"{"header":{}}"#;
        let sign = {
            let string_to_sign = format!("{}{}{}", "1700000000", "nonce", body);
            let mut mac = HmacSha256::new_from_slice(token.as_bytes()).unwrap();
            mac.update(string_to_sign.as_bytes());
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
        };
        assert!(verify_feishu_signature(
            token,
            "1700000000",
            "nonce",
            body,
            &sign
        ));
        assert!(!verify_feishu_signature(
            token,
            "1700000001",
            "nonce",
            body,
            &sign
        ));
        assert!(!verify_feishu_signature(
            "wrong",
            "1700000000",
            "nonce",
            body,
            &sign
        ));
    }

    #[test]
    fn test_token_error_codes() {
        assert!(is_token_error(99_991_663));
        assert!(!is_token_error(0));
        assert!(!is_token_error(190_001));
    }
}
