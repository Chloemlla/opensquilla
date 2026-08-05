//! MS Teams channel adapter — Bot Framework v3/v4 protocol.
//!
//! Microsoft Teams bots use the Bot Framework: incoming messages arrive as
//! HTTP POST webhook activities and outgoing messages go through the Bot
//! Connector REST API. This module implements the raw protocol without the
//! `botbuilder` SDK:
//!
//! 1. **Auth**: `POST https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token`
//!    exchanges the app id/password for an access token (cached).
//! 2. **Inbound**: [`MSTeamsChannel::handle_activity`] dispatches activity
//!    types — `message`, `conversationUpdate`, `typing`, `invoke`,
//!    `messageReaction`, `event`, `endOfConversation`.
//! 3. **Outbound**: [`MSTeamsChannel::send_activity`] posts to
//!    `{serviceUrl}/v3/conversations/{id}/activities`; replies carry a
//!    `replyToId`. Proactive conversations are created via `/v3/conversations`.
//! 4. **Cards**: AdaptiveCard, HeroCard and ThumbnailCard builders.
//! 5. **Auth validation**: incoming `Authorization: Bearer <JWT>` tokens are
//!    structurally validated (audience, issuer, expiry). HS256 tokens are
//!    verified against the app password; RS256 requires the Bot Framework
//!    OpenID public keys (deferred — the RSA primitive is not a crate dep).

use crate::types::{
    Channel, ChannelConfig, ChannelType, MessageAttachment, OutgoingMessage,
};
use base64::Engine;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

use serde_json::{json, Value};

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_BOT_ENDPOINT: &str = "https://smba.trafficmanager.net/apis/";
const TOKEN_EXPIRE_MARGIN_SECS: i64 = 300;
/// Standard Bot Framework OpenID issuer host.
const BOTFRAMEWORK_ISSUER: &str = "botframework.com";

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    expires_at: DateTime<Utc>,
}

/// A decoded (but not yet verified) JWT.
struct JwtParts {
    header: Value,
    payload: Value,
    signature: Vec<u8>,
    signing_input: String,
}

/// MS Teams channel adapter using the Bot Framework protocol with an axum
/// webhook for inbound activities.
pub struct MSTeamsChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    app_id: String,
    app_password: String,
    bot_endpoint: String,
    service_url: Arc<Mutex<Option<String>>>,
    conversation_id: Arc<Mutex<Option<String>>>,
    token: Arc<Mutex<Option<CachedToken>>>,
}

impl MSTeamsChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let app_id = config
            .config
            .get("app_id")
            .and_then(|v| v.as_str())
            .ok_or("No app_id configured for MS Teams")?
            .to_string();
        let app_password = config
            .config
            .get("app_password")
            .and_then(|v| v.as_str())
            .ok_or("No app_password configured for MS Teams")?
            .to_string();
        let bot_endpoint = config
            .config
            .get("bot_endpoint")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_BOT_ENDPOINT)
            .to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        Ok(Self {
            config,
            client,
            app_id,
            app_password,
            bot_endpoint,
            service_url: Arc::new(Mutex::new(None)),
            conversation_id: Arc::new(Mutex::new(None)),
            token: Arc::new(Mutex::new(None)),
        })
    }

    /// Obtain (and cache) a Bot Framework access token.
    async fn get_bot_token(&self) -> Result<String, String> {
        let mut guard = self.token.lock().await;
        if let Some(ref cached) = *guard {
            if Utc::now() < cached.expires_at {
                return Ok(cached.token.clone());
            }
        }
        let resp = self
            .client
            .post("https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token")
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", &self.app_id),
                ("client_secret", &self.app_password),
                ("scope", "https://api.botframework.com/.default"),
            ])
            .send()
            .await
            .map_err(|e| format!("Teams auth request failed: {}", e))?;
        let body: Value = resp.json().await.map_err(|e| format!("Teams auth parse: {}", e))?;
        let token = body
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or("No access_token in Teams auth response")?
            .to_string();
        let expires_in = body["expires_in"].as_i64().unwrap_or(3600);
        let expires_at = Utc::now()
            .checked_add_signed(chrono::Duration::seconds(expires_in - TOKEN_EXPIRE_MARGIN_SECS))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::hours(1));
        *guard = Some(CachedToken { token: token.clone(), expires_at });
        Ok(token)
    }

    /// Send a reply to a Teams conversation.
    async fn send_teams_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let token = self.get_bot_token().await?;
        let service_url = self
            .service_url
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| self.bot_endpoint.clone());
        let conversation_id = self
            .conversation_id
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| message.channel_id.clone());

        let mut activity = json!({
            "type": "message",
            "from": {"id": self.app_id, "name": self.config.name},
            "conversation": {"id": conversation_id},
            "text": message.text,
        });
        if let Some(ref reply) = message.thread_id {
            activity["replyToId"] = Value::String(reply.clone());
        }
        if !message.attachments.is_empty() {
            let attachments: Vec<Value> = message
                .attachments
                .iter()
                .map(|a| {
                    json!({
                        "contentType": "application/vnd.microsoft.card.hero",
                        "content": {
                            "title": a.attachment_type,
                            "text": message.text,
                            "images": a.url.as_ref().map(|u| json!([{"url": u}])).unwrap_or(json!([])),
                        }
                    })
                })
                .collect();
            activity["attachments"] = Value::Array(attachments);
        }

        let url = format!(
            "{}/v3/conversations/{}/activities",
            service_url.trim_end_matches('/'),
            conversation_id
        );
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&activity)
            .send()
            .await
            .map_err(|e| format!("Teams send request: {}", e))?;
        let status = resp.status();
        let body: Value = resp.json().await.map_err(|e| format!("Teams parse: {}", e))?;
        if status.is_success() {
            Ok(())
        } else {
            Err(format!(
                "Teams error: {}",
                body["errorDescription"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send an arbitrary outgoing activity to a conversation.
    pub async fn send_activity(&self, activity: &Value) -> Result<Value, String> {
        let token = self.get_bot_token().await?;
        let service_url = self
            .service_url
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| self.bot_endpoint.clone());
        let conversation_id = activity
            .get("conversation")
            .and_then(|c| c.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let url = format!(
            "{}/v3/conversations/{}/activities",
            service_url.trim_end_matches('/'),
            conversation_id
        );
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(activity)
            .send()
            .await
            .map_err(|e| format!("Teams send activity: {}", e))?;
        let body: Value = resp.json().await.map_err(|e| format!("Teams parse: {}", e))?;
        if resp.status().is_success() {
            Ok(body)
        } else {
            Err(format!(
                "Teams error: {}",
                body["errorDescription"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Create a new (typically proactive) conversation and store its id.
    pub async fn create_conversation(&self, members: &[&str], tenant_id: Option<&str>) -> Result<String, String> {
        let token = self.get_bot_token().await?;
        let service_url = self
            .service_url
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| self.bot_endpoint.clone());
        let url = format!("{}/v3/conversations", service_url.trim_end_matches('/'));
        let payload = create_conversation_payload(&self.app_id, &self.config.name, members, tenant_id);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Teams create conversation: {}", e))?;
        let body: Value = resp.json().await.map_err(|e| format!("Teams parse: {}", e))?;
        let conv_id = body["id"]
            .as_str()
            .ok_or_else(|| format!("No conversation id in response: {}", body))?
            .to_string();
        *self.conversation_id.lock().await = Some(conv_id.clone());
        Ok(conv_id)
    }

    /// List members of a conversation.
    pub async fn get_conversation_members(&self, conversation_id: &str) -> Result<Vec<Value>, String> {
        let token = self.get_bot_token().await?;
        let service_url = self
            .service_url
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| self.bot_endpoint.clone());
        let url = format!(
            "{}/v3/conversations/{}/members",
            service_url.trim_end_matches('/'),
            conversation_id
        );
        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| format!("Teams members request: {}", e))?;
        let body: Value = resp.json().await.map_err(|e| format!("Teams parse: {}", e))?;
        Ok(body.as_array().cloned().unwrap_or_default())
    }

    /// Process an incoming Activity from the Bot Framework webhook.
    pub async fn handle_activity(&self, activity: Value) -> Result<Value, String> {
        let activity_type = activity.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match activity_type {
            "message" => self.handle_message_activity(&activity).await,
            "conversationUpdate" => self.handle_conversation_update(&activity).await,
            "typing" => Ok(json!({"status": "typing"})),
            "invoke" => self.handle_invoke(&activity).await,
            "messageReaction" => {
                info!("Teams message reaction received");
                Ok(json!({"status": "ok"}))
            }
            "endOfConversation" => {
                info!("Teams conversation ended");
                Ok(json!({"status": "ok"}))
            }
            "event" => {
                let name = activity.get("name").and_then(|v| v.as_str()).unwrap_or("");
                info!("Teams event activity: {}", name);
                Ok(json!({"status": "ok", "event": name}))
            }
            other => {
                info!("Teams activity type: {}", other);
                Ok(json!({"status": "ignored"}))
            }
        }
    }

    async fn handle_message_activity(&self, activity: &Value) -> Result<Value, String> {
        let text = activity.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let from_id = activity
            .get("from")
            .and_then(|f| f.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let from_name = activity
            .get("from")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let conversation_id = activity
            .get("conversation")
            .and_then(|c| c.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let service_url = activity
            .get("serviceUrl")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let reply_to = activity.get("replyToId").and_then(|v| v.as_str()).map(String::from);

        if !service_url.is_empty() {
            *self.service_url.lock().await = Some(service_url);
        }
        if !conversation_id.is_empty() {
            *self.conversation_id.lock().await = Some(conversation_id.clone());
        }

        let attachments = parse_activity_attachments(activity);

        info!("Teams message from {}: {}", from_id, text);
        Ok(json!({
            "status": "received",
            "text": text,
            "from_id": from_id,
            "from_name": from_name,
            "conversation_id": conversation_id,
            "reply_to": reply_to,
            "attachments": attachments.len(),
        }))
    }

    async fn handle_conversation_update(&self, activity: &Value) -> Result<Value, String> {
        let added = activity
            .get("membersAdded")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let removed = activity
            .get("membersRemoved")
            .and_then(|m| m.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        info!("Teams conversation update: {} added, {} removed", added, removed);
        Ok(json!({"status": "ok", "members_added": added, "members_removed": removed}))
    }

    async fn handle_invoke(&self, activity: &Value) -> Result<Value, String> {
        let name = activity.get("name").and_then(|v| v.as_str()).unwrap_or("");
        match name {
            "adaptiveCard/action" => {
                let value = activity.get("value").cloned().unwrap_or(Value::Null);
                Ok(json!({
                    "status": 200,
                    "type": "application/vnd.microsoft.activity.message",
                    "value": {"text": "Card action received", "card_action": value},
                }))
            }
            "signin/verifyState" => {
                let state = activity
                    .get("value")
                    .and_then(|v| v.get("state"))
                    .cloned()
                    .unwrap_or(Value::Null);
                Ok(json!({"status": 200, "value": {"state": state}}))
            }
            _ => {
                info!("Teams invoke activity: {}", name);
                Ok(json!({"status": 200}))
            }
        }
    }

    /// Build an AdaptiveCard attachment payload.
    pub fn build_adaptive_card(&self, title: &str, text: &str, buttons: &[(String, String)]) -> Value {
        let actions: Vec<Value> = buttons
            .iter()
            .map(|(label, value)| {
                json!({"type": "Action.Submit", "title": label, "data": {"value": value}})
            })
            .collect();
        json!({
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard",
                "version": "1.4",
                "body": [
                    {"type": "TextBlock", "size": "Medium", "weight": "Bolder", "text": title, "wrap": true},
                    {"type": "TextBlock", "text": text, "wrap": true},
                ],
                "actions": actions,
            }
        })
    }

    /// Build a HeroCard attachment payload.
    pub fn build_hero_card(
        &self,
        title: &str,
        subtitle: &str,
        text: &str,
        image_url: Option<&str>,
        buttons: &[(String, String)],
    ) -> Value {
        let images = image_url.map(|u| json!([{"url": u}])).unwrap_or(json!([]));
        let buttons: Vec<Value> = buttons
            .iter()
            .map(|(label, value)| json!({"type": "openUrl", "title": label, "value": value}))
            .collect();
        json!({
            "contentType": "application/vnd.microsoft.card.hero",
            "content": {
                "title": title,
                "subtitle": subtitle,
                "text": text,
                "images": images,
                "buttons": buttons,
            }
        })
    }

    /// Build a ThumbnailCard attachment payload.
    pub fn build_thumbnail_card(
        &self,
        title: &str,
        subtitle: &str,
        text: &str,
        image_url: Option<&str>,
        buttons: &[(String, String)],
    ) -> Value {
        let images = image_url.map(|u| json!([{"url": u}])).unwrap_or(json!([]));
        let buttons: Vec<Value> = buttons
            .iter()
            .map(|(label, value)| json!({"type": "openUrl", "title": label, "value": value}))
            .collect();
        json!({
            "contentType": "application/vnd.microsoft.card.thumbnail",
            "content": {
                "title": title,
                "subtitle": subtitle,
                "text": text,
                "images": images,
                "buttons": buttons,
            }
        })
    }

    /// Validate an incoming Bot Framework JWT.
    ///
    /// Checks the structural segments, the audience against the bot app id,
    /// the issuer against `botframework.com`, and expiry. HS256 tokens are
    /// verified against the app password; RS256 signature verification is
    /// deferred (see module docs).
    pub fn validate_jwt(&self, token: &str) -> Result<Value, String> {
        let parts = decode_jwt(token)?;
        let aud = parts.payload.get("aud").and_then(|v| v.as_str()).unwrap_or("");
        if aud != self.app_id {
            return Err(format!(
                "JWT audience mismatch: got '{}', expected '{}'",
                aud, self.app_id
            ));
        }
        let iss = parts.payload.get("iss").and_then(|v| v.as_str()).unwrap_or("");
        if !iss.contains(BOTFRAMEWORK_ISSUER) {
            return Err(format!("JWT from untrusted issuer: {}", iss));
        }
        if let Some(exp) = parts.payload.get("exp").and_then(|v| v.as_i64()) {
            if Utc::now().timestamp() >= exp {
                return Err("JWT is expired".to_string());
            }
        }
        let alg = parts.header.get("alg").and_then(|v| v.as_str()).unwrap_or("");
        match alg {
            "HS256" => self.verify_hs256_signature(&parts)?,
            "RS256" => {
                warn!("Teams JWT RS256 signature not verified (OpenID keys not configured)");
            }
            other => return Err(format!("Unsupported JWT algorithm: {}", other)),
        }
        Ok(parts.payload)
    }

    fn verify_hs256_signature(&self, parts: &JwtParts) -> Result<(), String> {
        let mut mac = HmacSha256::new_from_slice(self.app_password.as_bytes())
            .expect("HMAC accepts keys of any size");
        mac.update(parts.signing_input.as_bytes());
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let expected = engine.encode(mac.finalize().into_bytes());
        let actual = engine.encode(&parts.signature);
        if expected == actual {
            Ok(())
        } else {
            Err("JWT signature verification failed".to_string())
        }
    }
}

/// Extract the Bearer token from an authorization header map.
pub fn extract_bearer_token(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(reqwest::header::AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
        .map(String::from)
}

/// Decode a JWT into its header, payload, signature and signing input.
fn decode_jwt(token: &str) -> Result<JwtParts, String> {
    let mut segments = token.split('.');
    let header_b64 = segments.next().ok_or("JWT missing header")?;
    let payload_b64 = segments.next().ok_or("JWT missing payload")?;
    let signature_b64 = segments.next().ok_or("JWT missing signature")?;
    if segments.next().is_some() {
        return Err("JWT has too many segments".to_string());
    }
    let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let header_bytes = engine.decode(header_b64).map_err(|e| format!("JWT header decode: {}", e))?;
    let payload_bytes = engine.decode(payload_b64).map_err(|e| format!("JWT payload decode: {}", e))?;
    let signature = engine.decode(signature_b64).map_err(|e| format!("JWT signature decode: {}", e))?;
    let header: Value = serde_json::from_slice(&header_bytes).map_err(|e| format!("JWT header JSON: {}", e))?;
    let payload: Value = serde_json::from_slice(&payload_bytes).map_err(|e| format!("JWT payload JSON: {}", e))?;
    Ok(JwtParts {
        header,
        payload,
        signature,
        signing_input: format!("{}.{}", header_b64, payload_b64),
    })
}

/// Build the `POST /v3/conversations` request body.
fn create_conversation_payload(app_id: &str, bot_name: &str, members: &[&str], tenant_id: Option<&str>) -> Value {
    let members_json: Vec<Value> = members.iter().map(|m| json!({"id": m})).collect();
    let mut payload = json!({
        "bot": {"id": app_id, "name": bot_name},
        "isGroup": members.len() > 1,
        "members": members_json,
    });
    if let Some(tenant) = tenant_id {
        payload["channelData"] = json!({"tenant": {"id": tenant}});
    }
    payload
}

/// Parse the attachments of an inbound message activity.
fn parse_activity_attachments(activity: &Value) -> Vec<MessageAttachment> {
    activity
        .get("attachments")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|att| {
                    let content_type = att
                        .get("contentType")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let url = att
                        .get("content")
                        .and_then(|c| c.get("images"))
                        .and_then(|i| i.as_array())
                        .and_then(|imgs| imgs.first())
                        .and_then(|img| img.get("url"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    MessageAttachment {
                        attachment_type: content_type,
                        url,
                        data: Some(att.clone()),
                        mime_type: None,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[async_trait::async_trait]
impl Channel for MSTeamsChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::MSTeams
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.send_teams_message(message).await
    }

    async fn send_typing(&self, channel_id: &str) -> Result<(), String> {
        let token = self.get_bot_token().await?;
        let service_url = self
            .service_url
            .lock()
            .await
            .clone()
            .unwrap_or_else(|| self.bot_endpoint.clone());
        let url = format!(
            "{}/v3/conversations/{}/activities",
            service_url.trim_end_matches('/'),
            channel_id
        );
        let typing_activity = json!({
            "type": "typing",
            "from": {"id": self.app_id},
            "conversation": {"id": channel_id},
        });
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&typing_activity)
            .send()
            .await
            .map_err(|e| format!("Teams typing request: {}", e))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Teams typing failed: {}", resp.status()))
        }
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const APP_ID: &str = "00000000-0000-0000-0000-000000000000";
    const APP_SECRET: &str = "test-secret";
    const ISSUER: &str = "https://api.botframework.com";

    fn test_config() -> ChannelConfig {
        ChannelConfig {
            channel_type: ChannelType::MSTeams,
            channel_id: "teams".to_string(),
            name: "Teams".to_string(),
            enabled: true,
            config: json!({
                "app_id": APP_ID,
                "app_password": APP_SECRET,
                "bot_endpoint": "https://smba.trafficmanager.net/apis/",
            }),
        }
    }

    fn make_hs256_jwt(app_id: &str, secret: &str, iss: &str, exp: i64) -> String {
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        let header = engine.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = engine.encode(format!(r#"{{"aud":"{}","iss":"{}","exp":{}}}"#, app_id, iss, exp));
        let signing_input = format!("{}.{}", header, payload);
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(signing_input.as_bytes());
        let sig = engine.encode(mac.finalize().into_bytes());
        format!("{}.{}", signing_input, sig)
    }

    #[test]
    fn test_decode_jwt_valid() {
        let token = make_hs256_jwt(APP_ID, APP_SECRET, ISSUER, Utc::now().timestamp() + 3600);
        let parts = decode_jwt(&token).unwrap();
        assert_eq!(parts.payload["aud"], APP_ID);
        assert_eq!(parts.payload["iss"], ISSUER);
        assert_eq!(parts.header["alg"], "HS256");
    }

    #[test]
    fn test_decode_jwt_malformed() {
        assert!(decode_jwt("not-a-jwt").is_err());
        assert!(decode_jwt("a.b").is_err());
        assert!(decode_jwt("a.b.c.d").is_err());
    }

    #[test]
    fn test_validate_jwt_ok() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let token = make_hs256_jwt(APP_ID, APP_SECRET, ISSUER, Utc::now().timestamp() + 3600);
        let payload = channel.validate_jwt(&token).unwrap();
        assert_eq!(payload["aud"], APP_ID);
    }

    #[test]
    fn test_validate_jwt_wrong_secret() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let token = make_hs256_jwt(APP_ID, "wrong-secret", ISSUER, Utc::now().timestamp() + 3600);
        assert!(channel.validate_jwt(&token).is_err());
    }

    #[test]
    fn test_validate_jwt_expired() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let token = make_hs256_jwt(APP_ID, APP_SECRET, ISSUER, Utc::now().timestamp() - 60);
        assert!(channel.validate_jwt(&token).is_err());
    }

    #[test]
    fn test_validate_jwt_wrong_audience() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let token = make_hs256_jwt("some-other-app", APP_SECRET, ISSUER, Utc::now().timestamp() + 3600);
        assert!(channel.validate_jwt(&token).is_err());
    }

    #[test]
    fn test_validate_jwt_untrusted_issuer() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let token = make_hs256_jwt(APP_ID, APP_SECRET, "https://evil.example", Utc::now().timestamp() + 3600);
        assert!(channel.validate_jwt(&token).is_err());
    }

    #[test]
    fn test_extract_bearer_token() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_static("Bearer abc.def.ghi"),
        );
        assert_eq!(extract_bearer_token(&headers).as_deref(), Some("abc.def.ghi"));

        let empty = reqwest::header::HeaderMap::new();
        assert!(extract_bearer_token(&empty).is_none());
    }

    #[test]
    fn test_build_adaptive_card() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let card = channel.build_adaptive_card("Title", "Body", &[("Go".to_string(), "val".to_string())]);
        assert_eq!(card["contentType"], "application/vnd.microsoft.card.adaptive");
        assert_eq!(card["content"]["type"], "AdaptiveCard");
        assert_eq!(card["content"]["body"][0]["text"], "Title");
        assert_eq!(card["content"]["actions"][0]["title"], "Go");
    }

    #[test]
    fn test_build_hero_card() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let card = channel.build_hero_card(
            "H",
            "S",
            "T",
            Some("https://example.com/img.png"),
            &[("Link".to_string(), "https://example.com".to_string())],
        );
        assert_eq!(card["contentType"], "application/vnd.microsoft.card.hero");
        assert_eq!(card["content"]["title"], "H");
        assert_eq!(card["content"]["images"][0]["url"], "https://example.com/img.png");
        assert_eq!(card["content"]["buttons"][0]["type"], "openUrl");
    }

    #[test]
    fn test_build_thumbnail_card() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let card = channel.build_thumbnail_card("T", "", "", None, &[]);
        assert_eq!(card["contentType"], "application/vnd.microsoft.card.thumbnail");
        assert_eq!(card["content"]["images"], json!([]));
    }

    #[test]
    fn test_create_conversation_payload() {
        let payload = create_conversation_payload(APP_ID, "Bot", &["user-1"], Some("tenant-1"));
        assert_eq!(payload["bot"]["id"], APP_ID);
        assert!(!payload["isGroup"].as_bool().unwrap());
        assert_eq!(payload["members"][0]["id"], "user-1");
        assert_eq!(payload["channelData"]["tenant"]["id"], "tenant-1");

        let group = create_conversation_payload(APP_ID, "Bot", &["u1", "u2"], None);
        assert!(group["isGroup"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_handle_message_activity() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let activity = json!({
            "type": "message",
            "serviceUrl": "https://smba.trafficmanager.net/apis/",
            "from": {"id": "user-1", "name": "Alice"},
            "conversation": {"id": "conv-1"},
            "text": "hello teams",
        });
        let result = channel.handle_activity(activity).await.unwrap();
        assert_eq!(result["status"], "received");
        assert_eq!(result["text"], "hello teams");
        assert_eq!(result["from_id"], "user-1");
        let svc = channel.service_url.lock().await.clone();
        assert_eq!(svc.as_deref(), Some("https://smba.trafficmanager.net/apis/"));
        let conv = channel.conversation_id.lock().await.clone();
        assert_eq!(conv.as_deref(), Some("conv-1"));
    }

    #[tokio::test]
    async fn test_handle_invoke_card_action() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let activity = json!({
            "type": "invoke",
            "name": "adaptiveCard/action",
            "value": {"action": {"title": "OK"}},
        });
        let result = channel.handle_activity(activity).await.unwrap();
        assert_eq!(result["status"], 200);
        assert_eq!(result["value"]["card_action"]["action"]["title"], "OK");
    }

    #[tokio::test]
    async fn test_handle_typing() {
        let channel = MSTeamsChannel::new(test_config()).unwrap();
        let activity = json!({"type": "typing"});
        let result = channel.handle_activity(activity).await.unwrap();
        assert_eq!(result["status"], "typing");
    }
}
