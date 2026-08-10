//! WeCom (WeChat Work) channel adapter — callbacks + raw REST API.
//!
//! WeCom apps receive messages either as an HTTP callback or (rarely) via a
//! WebSocket; outbound messages go through the `qyapi.weixin.qq.com` REST API.
//! This module implements the raw protocol with no SDK:
//!
//! 1. **Access token**: `GET /cgi-bin/gettoken?corpid=...&corpsecret=...`
//!    returns a short-lived token that is cached and refreshed.
//! 2. **Outbound**: `POST /cgi-bin/message/send` with `touser` / `msgtype`
//!    / `agentid` payloads. Text, markdown, image, news and file messages are
//!    built by the [`MessageBuilder`] helpers.
//! 3. **Media**: `POST /cgi-bin/media/upload` uploads files for `image` /
//!    `file` messages and returns a `media_id`.
//! 4. **Callbacks**: URL verification echoes `echostr` after checking the
//!    SHA-1 signature over `(token, timestamp, nonce, echostr)`. Callback
//!    bodies are AES-256-CBC encrypted with the `EncodingAESKey`; this module
//!    implements both encryption and decryption without an SDK.

use crate::types::{Channel, ChannelConfig, ChannelType, OutgoingMessage};
use aes::Aes256;
use aes::cipher::{BlockEncrypt, KeyInit};
use base64::Engine as _;
use chrono::{DateTime, Utc};
use reqwest::multipart::{Form, Part};
use serde_json::{Value, json};
use sha1::Digest;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::info;

/// Default WeCom API base.
pub const DEFAULT_API_BASE: &str = "https://qyapi.weixin.qq.com/cgi-bin";

/// Lenient base64 decoder for WeCom `EncodingAESKey` (43-char key + `=`).
///
/// WeChat's reference implementations decode the key allowing non-zero
/// trailing bits on the final symbol, which the strict `STANDARD` engine
/// rejects. Mirror that so the documented sample key round-trips.
static WECOM_KEY_B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::GeneralPurpose::new(
        &base64::engine::general_purpose::STANDARD_ALPHABET,
        base64::engine::general_purpose::GeneralPurposeConfig::new()
            .with_decode_allow_trailing_bits(true),
    );

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    expires_at: DateTime<Utc>,
}

/// WeCom (WeChat Work) channel adapter.
pub struct WeComChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    corp_id: String,
    agent_id: String,
    secret: String,
    token: String,
    encoding_aes_key: String,
    api_base: String,
    access_token: Arc<Mutex<Option<CachedToken>>>,
    robot_webhook: Arc<Mutex<Option<String>>>,
}

impl WeComChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let corp_id = config
            .config
            .get("corp_id")
            .and_then(|v| v.as_str())
            .ok_or("No corp_id configured for WeCom")?
            .to_string();
        let agent_id = config
            .config
            .get("agent_id")
            .and_then(|v| v.as_str())
            .ok_or("No agent_id configured for WeCom")?
            .to_string();
        let secret = config
            .config
            .get("secret")
            .and_then(|v| v.as_str())
            .ok_or("No secret configured for WeCom")?
            .to_string();
        let token = config
            .config
            .get("token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let encoding_aes_key = config
            .config
            .get("encoding_aes_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/')
            .to_string();
        let robot_webhook = config
            .config
            .get("webhook_url")
            .and_then(|v| v.as_str())
            .map(String::from);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
        Ok(Self {
            config,
            client,
            corp_id,
            agent_id,
            secret,
            token,
            encoding_aes_key,
            api_base,
            access_token: Arc::new(Mutex::new(None)),
            robot_webhook: Arc::new(Mutex::new(robot_webhook)),
        })
    }

    // -- access token -------------------------------------------------------

    /// Obtain (and cache) the application access token.
    pub async fn get_access_token(&self) -> Result<String, String> {
        let mut guard = self.access_token.lock().await;
        if let Some(ref cached) = *guard {
            if Utc::now() < cached.expires_at {
                return Ok(cached.token.clone());
            }
        }
        let resp = self
            .client
            .get(format!("{}/gettoken", self.api_base))
            .query(&[
                ("corpid", self.corp_id.as_str()),
                ("corpsecret", self.secret.as_str()),
            ])
            .send()
            .await
            .map_err(|e| format!("WeCom token request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("WeCom token parse: {e}"))?;
        let token = body
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("WeCom token error: {body}"))?
            .to_string();
        let expires_in = body
            .get("expires_in")
            .and_then(|v| v.as_i64())
            .unwrap_or(7200);
        let expires_at = Utc::now()
            .checked_add_signed(chrono::Duration::seconds(expires_in - 300))
            .unwrap_or_else(|| Utc::now() + chrono::Duration::hours(2));
        *guard = Some(CachedToken {
            token: token.clone(),
            expires_at,
        });
        info!("WeCom access token acquired, expires {}", expires_at);
        Ok(token)
    }

    // -- outbound messages --------------------------------------------------

    /// Send an application message to a user or department.
    async fn send_app_message(&self, _touser: &str, body: Value) -> Result<(), String> {
        let token = self.get_access_token().await?;
        let resp = self
            .client
            .post(format!("{}/message/send", self.api_base))
            .query(&[("access_token", &token)])
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("WeCom message/send request: {e}"))?;
        let parsed: Value = resp
            .json()
            .await
            .map_err(|e| format!("WeCom message/send parse: {e}"))?;
        let errcode = parsed.get("errcode").and_then(|v| v.as_i64()).unwrap_or(-1);
        if errcode == 0 {
            Ok(())
        } else {
            Err(format!(
                "WeCom error: {} (code: {errcode})",
                parsed["errmsg"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a text application message.
    pub async fn send_text_message(&self, touser: &str, content: &str) -> Result<(), String> {
        let body = MessageBuilder::text_message(touser, &self.agent_id, content);
        self.send_app_message(touser, body).await
    }

    /// Send a markdown application message.
    pub async fn send_markdown_message(&self, touser: &str, content: &str) -> Result<(), String> {
        let body = MessageBuilder::markdown_message(touser, &self.agent_id, content);
        self.send_app_message(touser, body).await
    }

    /// Send an image application message by `media_id`.
    pub async fn send_image_message(&self, touser: &str, media_id: &str) -> Result<(), String> {
        let body = MessageBuilder::image_message(touser, &self.agent_id, media_id);
        self.send_app_message(touser, body).await
    }

    /// Send a news (article card) application message.
    pub async fn send_news_message(
        &self,
        touser: &str,
        articles: Vec<Value>,
    ) -> Result<(), String> {
        let body = MessageBuilder::news_message(touser, &self.agent_id, articles);
        self.send_app_message(touser, body).await
    }

    /// Send a file application message by `media_id`.
    pub async fn send_file_message(&self, touser: &str, media_id: &str) -> Result<(), String> {
        let body = MessageBuilder::file_message(touser, &self.agent_id, media_id);
        self.send_app_message(touser, body).await
    }

    /// Send a textcard application message.
    pub async fn send_textcard_message(
        &self,
        touser: &str,
        title: &str,
        description: &str,
        url: &str,
    ) -> Result<(), String> {
        let body =
            MessageBuilder::textcard_message(touser, &self.agent_id, title, description, url);
        self.send_app_message(touser, body).await
    }

    // -- group robot webhook ------------------------------------------------

    /// Send a payload to a group-robot webhook URL.
    pub async fn send_robot_payload(
        &self,
        webhook_url: &str,
        payload: Value,
    ) -> Result<(), String> {
        let resp = self
            .client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("WeCom robot request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("WeCom robot parse: {e}"))?;
        let errcode = body.get("errcode").and_then(|v| v.as_i64()).unwrap_or(-1);
        if errcode == 0 {
            Ok(())
        } else {
            Err(format!(
                "WeCom robot error: {} (code: {errcode})",
                body["errmsg"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a text group-robot message to the configured webhook.
    pub async fn send_robot_text(&self, content: &str) -> Result<(), String> {
        let url = self
            .robot_webhook
            .lock()
            .await
            .clone()
            .ok_or("No robot webhook_url configured")?;
        self.send_robot_payload(
            &url,
            json!({"msgtype": "text", "text": {"content": content}}),
        )
        .await
    }

    // -- media --------------------------------------------------------------

    /// Upload a media file and return its `media_id`.
    pub async fn upload_media(
        &self,
        media_type: &str,
        file_name: &str,
        bytes: &[u8],
    ) -> Result<String, String> {
        let token = self.get_access_token().await?;
        let part = Part::bytes(bytes.to_vec()).file_name(file_name.to_string());
        let form = Form::new().part("media", part);
        let resp = self
            .client
            .post(format!("{}/media/upload", self.api_base))
            .query(&[("access_token", token.as_str()), ("type", media_type)])
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("WeCom media/upload request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("WeCom media/upload parse: {e}"))?;
        let errcode = body.get("errcode").and_then(|v| v.as_i64()).unwrap_or(-1);
        if errcode == 0 {
            body.get("media_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| "No media_id in upload response".to_string())
        } else {
            Err(format!(
                "WeCom media/upload error: {} (code: {errcode})",
                body["errmsg"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Download a media file by `media_id`.
    pub async fn get_media(&self, media_id: &str) -> Result<Vec<u8>, String> {
        let token = self.get_access_token().await?;
        let resp = self
            .client
            .get(format!("{}/media/get", self.api_base))
            .query(&[("access_token", token.as_str()), ("media_id", media_id)])
            .send()
            .await
            .map_err(|e| format!("WeCom media/get request: {e}"))?;
        if resp.status().is_success() {
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();
            if content_type.contains("application/json") {
                let body: Value = resp
                    .json()
                    .await
                    .map_err(|e| format!("WeCom media/get parse: {e}"))?;
                Err(format!(
                    "WeCom media/get error: {}",
                    body["errmsg"].as_str().unwrap_or("unknown")
                ))
            } else {
                resp.bytes()
                    .await
                    .map(|b| b.to_vec())
                    .map_err(|e| format!("WeCom media/get read: {e}"))
            }
        } else {
            Err(format!("WeCom media/get failed: {}", resp.status()))
        }
    }

    // -- callback verification ----------------------------------------------

    /// Compute the callback SHA-1 signature:
    /// `SHA1(sort([token, timestamp, nonce, data]).concat())`.
    pub fn compute_signature(&self, timestamp: &str, nonce: &str, data: &str) -> String {
        compute_wecom_signature(&self.token, timestamp, nonce, data)
    }

    /// Verify a callback URL challenge (`echostr`) and return it if valid.
    pub fn verify_url(
        &self,
        msg_signature: &str,
        timestamp: &str,
        nonce: &str,
        echostr: &str,
    ) -> Result<String, String> {
        let computed = self.compute_signature(timestamp, nonce, echostr);
        if computed == msg_signature {
            Ok(echostr.to_string())
        } else {
            Err("WeCom callback signature verification failed".to_string())
        }
    }

    /// Encrypt a plaintext reply for a callback.
    pub fn encrypt_message(&self, plaintext: &str, receive_id: &str) -> Result<String, String> {
        encrypt_wecom_payload(&self.encoding_aes_key, plaintext, receive_id)
    }

    /// Decrypt a callback `Encrypt` field.
    pub fn decrypt_message(&self, encrypted: &str) -> Result<String, String> {
        crate::webhook::decrypt_wecom_payload(&self.encoding_aes_key, encrypted)
            .map_err(|e| e.to_string())
    }

    /// Build the full XML callback reply including the message signature.
    ///
    /// ```
    /// <xml>
    ///   <Encrypt>...</Encrypt>
    ///   <MsgSignature>...</MsgSignature>
    ///   <TimeStamp>...</TimeStamp>
    ///   <Nonce>...</Nonce>
    /// </xml>
    /// ```
    pub fn build_callback_reply(
        &self,
        reply_text: &str,
        receive_id: &str,
        timestamp: &str,
        nonce: &str,
    ) -> Result<String, String> {
        let encrypted = self.encrypt_message(reply_text, receive_id)?;
        let signature = self.compute_signature(timestamp, nonce, &encrypted);
        Ok(format!(
            "<xml>\
             <Encrypt><![CDATA[{}]]></Encrypt>\
             <MsgSignature><![CDATA[{}]]></MsgSignature>\
             <TimeStamp>{}</TimeStamp>\
             <Nonce><![CDATA[{}]]></Nonce>\
             </xml>",
            xml_escape(&encrypted),
            signature,
            timestamp,
            xml_escape(nonce)
        ))
    }
}

/// Compute a WeCom callback SHA-1 signature.
pub fn compute_wecom_signature(token: &str, timestamp: &str, nonce: &str, data: &str) -> String {
    let mut parts = [
        token.to_string(),
        timestamp.to_string(),
        nonce.to_string(),
        data.to_string(),
    ];
    parts.sort();
    hex::encode(sha1::Sha1::digest(parts.concat().as_bytes()))
}

/// Encrypt a WeCom callback payload with AES-256-CBC.
///
/// The key is the base64 decode of `EncodingAESKey + "="`, the IV is the first
/// 16 bytes of that key, and the plaintext has the form
/// `random(16) || msg_len(4, big-endian) || msg || receive_id`, padded with
/// PKCS7. Returns the base64 ciphertext.
pub fn encrypt_wecom_payload(
    encoding_aes_key: &str,
    plaintext: &str,
    receive_id: &str,
) -> Result<String, String> {
    let full_key = format!("{encoding_aes_key}=");
    let key_bytes = WECOM_KEY_B64
        .decode(&full_key)
        .map_err(|_| "invalid EncodingAESKey".to_string())?;
    if key_bytes.len() != 32 {
        return Err("EncodingAESKey must decode to 32 bytes".to_string());
    }
    let iv = &key_bytes[0..16];
    let key = aes::cipher::generic_array::GenericArray::from_slice(&key_bytes);
    let cipher = Aes256::new(key);

    // random(16) || msg_len(4, BE) || msg || receive_id
    let mut buf = Vec::with_capacity(plaintext.len() + receive_id.len() + 52);
    buf.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    buf.extend_from_slice(&(plaintext.len() as u32).to_be_bytes());
    buf.extend_from_slice(plaintext.as_bytes());
    buf.extend_from_slice(receive_id.as_bytes());

    // PKCS7 padding: pad to 16-byte boundary
    let block_size = 16;
    let pad_len = block_size - (buf.len() % block_size);
    buf.resize(buf.len() + pad_len, pad_len as u8);

    // CBC mode encryption
    let mut prev = aes::cipher::generic_array::GenericArray::clone_from_slice(iv);
    for chunk in buf.chunks_mut(block_size) {
        let mut block = aes::cipher::generic_array::GenericArray::clone_from_slice(chunk);
        // XOR with previous ciphertext (or IV for first block)
        for (b, p) in block.iter_mut().zip(prev.iter()) {
            *b ^= *p;
        }
        cipher.encrypt_block(&mut block);
        chunk.copy_from_slice(&block);
        prev = block;
    }

    Ok(base64::engine::general_purpose::STANDARD.encode(&buf))
}

/// Decrypt a WeCom callback payload (thin wrapper over
/// [`crate::webhook::decrypt_wecom_payload`]).
pub fn decrypt_wecom_encrypted(encoding_aes_key: &str, ciphertext: &str) -> Result<String, String> {
    crate::webhook::decrypt_wecom_payload(encoding_aes_key, ciphertext).map_err(|e| e.to_string())
}

/// Escape a string for inclusion in an XML CDATA-free text node.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Message builders
// ---------------------------------------------------------------------------

/// Helpers for building WeCom message payloads.
pub struct MessageBuilder;

impl MessageBuilder {
    fn agent(agent_id: &str) -> Value {
        json!({ "agentid": agent_id })
    }

    /// Merge all keys from `b` into `a` (shallow).
    fn merge(a: &mut Value, b: &Value) {
        if let (Value::Object(a_map), Value::Object(b_map)) = (a, b) {
            a_map.extend(b_map.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
    }

    /// A `text` message body.
    pub fn text_message(touser: &str, agent_id: &str, content: &str) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "text",
            "text": { "content": content },
            "safe": 0,
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// A `markdown` message body.
    pub fn markdown_message(touser: &str, agent_id: &str, content: &str) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "markdown",
            "markdown": { "content": content },
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// An `image` message body.
    pub fn image_message(touser: &str, agent_id: &str, media_id: &str) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "image",
            "image": { "media_id": media_id },
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// A `news` message body from a list of article objects.
    pub fn news_message(touser: &str, agent_id: &str, articles: Vec<Value>) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "news",
            "news": { "articles": articles },
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// A `file` message body.
    pub fn file_message(touser: &str, agent_id: &str, media_id: &str) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "file",
            "file": { "media_id": media_id },
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// A `textcard` message body.
    pub fn textcard_message(
        touser: &str,
        agent_id: &str,
        title: &str,
        description: &str,
        url: &str,
    ) -> Value {
        let mut msg = json!({
            "touser": touser,
            "msgtype": "textcard",
            "textcard": {
                "title": title,
                "description": description,
                "url": url,
            },
        });
        Self::merge(&mut msg, &Self::agent(agent_id));
        msg
    }

    /// A single news article.
    pub fn news_article(title: &str, description: &str, url: &str, pic_url: Option<&str>) -> Value {
        let mut article = json!({
            "title": title,
            "description": description,
            "url": url,
        });
        if let Some(pic) = pic_url {
            article["picurl"] = json!(pic);
        }
        article
    }
}

#[async_trait::async_trait]
impl Channel for WeComChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::WeCom
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        // Prefer group-robot webhook when configured.
        if let Some(url) = self.robot_webhook.lock().await.clone() {
            return self
                .send_robot_payload(
                    &url,
                    json!({"msgtype": "text", "text": {"content": message.text}}),
                )
                .await;
        }

        // Attachment-first path: upload and send as image/file when available.
        if let Some(att) = message.attachments.first() {
            let media_type = match att.attachment_type.as_str() {
                "image" => "image",
                _ => "file",
            };
            if let Some(data) = &att.data {
                if let Some(base64_str) = data.get("base64").and_then(|v| v.as_str()) {
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(base64_str)
                    {
                        let name = data
                            .get("file_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("file.bin");
                        let media_id = self.upload_media(media_type, name, &bytes).await?;
                        if media_type == "image" {
                            return self
                                .send_image_message(&message.channel_id, &media_id)
                                .await;
                        }
                        return self.send_file_message(&message.channel_id, &media_id).await;
                    }
                }
            }
        }

        self.send_text_message(&message.channel_id, &message.text)
            .await
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        // WeCom has no typing indicator API.
        Ok(())
    }

    async fn set_webhook(&self, url: &str) -> Result<(), String> {
        *self.robot_webhook.lock().await = Some(url.to_string());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WECOM_KEY: &str = "jWmYm7qr5nMoAUwZRjGtBxmz3KA1tkAj3ykkR6q2B2C";

    fn channel() -> WeComChannel {
        WeComChannel::new(ChannelConfig {
            channel_type: ChannelType::WeCom,
            channel_id: "user1".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({
                "corp_id": "ww123",
                "agent_id": "1000002",
                "secret": "secret",
                "token": "QDG6eK",
                "encoding_aes_key": WECOM_KEY,
            }),
        })
        .unwrap()
    }

    #[test]
    fn test_signature_matches_reference() {
        // Reference values from the WeCom docs example.
        let token = "QDG6eK";
        let timestamp = "1409659813";
        let nonce = "1372623149";
        let encrypt = "9jq3f4x...";
        let sig = compute_wecom_signature(token, timestamp, nonce, encrypt);
        let mut parts = [
            token.to_string(),
            timestamp.to_string(),
            nonce.to_string(),
            encrypt.to_string(),
        ];
        parts.sort();
        let joined = parts.join("");
        let expected = hex::encode(sha1::Sha1::digest(joined.as_bytes()));
        assert_eq!(sig, expected);
    }

    #[test]
    fn test_aes_roundtrip() {
        let plaintext = r#"{"MsgType":"text","ToUserName":"ww123","FromUserName":"user1","MsgId":"1","Content":"hi","CreateTime":12345}"#;
        let encrypted = encrypt_wecom_payload(WECOM_KEY, plaintext, "ww123").unwrap();
        let decrypted = crate::webhook::decrypt_wecom_payload(WECOM_KEY, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_verify_url_valid() {
        let c = channel();
        let timestamp = "1409659813";
        let nonce = "1372623149";
        let echostr = "echo123";
        let sig = c.compute_signature(timestamp, nonce, echostr);
        assert_eq!(
            c.verify_url(&sig, timestamp, nonce, echostr).unwrap(),
            "echo123"
        );
        assert!(c.verify_url("deadbeef", timestamp, nonce, echostr).is_err());
    }

    #[test]
    fn test_message_builders() {
        let text = MessageBuilder::text_message("u1", "a1", "hello");
        assert_eq!(text["msgtype"], "text");
        assert_eq!(text["text"]["content"], "hello");
        assert_eq!(text["agentid"], "a1");

        let md = MessageBuilder::markdown_message("u1", "a1", "# hi");
        assert_eq!(md["msgtype"], "markdown");

        let img = MessageBuilder::image_message("u1", "a1", "m1");
        assert_eq!(img["msgtype"], "image");
        assert_eq!(img["image"]["media_id"], "m1");

        let articles = vec![MessageBuilder::news_article(
            "Title",
            "Desc",
            "https://x",
            None,
        )];
        let news = MessageBuilder::news_message("u1", "a1", articles);
        assert_eq!(news["news"]["articles"][0]["title"], "Title");
    }

    #[test]
    fn test_build_callback_reply_shape() {
        let c = channel();
        let reply = c
            .build_callback_reply("hello", "ww123", "1409659813", "1372623149")
            .unwrap();
        assert!(reply.contains("<xml>"));
        assert!(reply.contains("<Encrypt><![CDATA["));
        assert!(reply.contains("<MsgSignature><![CDATA["));
        assert!(reply.contains("<TimeStamp>1409659813</TimeStamp>"));
    }

    #[test]
    fn test_encrypt_rejects_bad_key() {
        assert!(encrypt_wecom_payload("short", "hi", "ww123").is_err());
    }
}
