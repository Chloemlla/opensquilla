//! Telegram channel adapter — long polling / webhook + raw Bot API.
//!
//! Telegram bots receive updates either by long-polling `getUpdates` or by
//! registering a webhook URL, and send messages via the Bot API. This module
//! implements the raw protocol without the `teloxide` / `telegram-bot` SDK
//! crates:
//!
//! 1. **Long polling**: `GET /bot{token}/getUpdates?timeout=30&offset=...`
//!    blocks up to 30 s; each update is parsed into an [`IncomingMessage`]
//!    and the `offset` advanced to acknowledge delivery.
//! 2. **Webhook**: [`TelegramChannel::webhook_route`] builds an axum route
//!    for the Bot API webhook; [`TelegramChannel::set_webhook`] registers it.
//! 3. **Outbound**: `sendMessage` with `parse_mode`, `reply_to_message_id`
//!    thread replies, and `reply_markup` inline keyboards. File messages use
//!    multipart uploads and `getFile` / file download.

use crate::types::{
    Channel, ChannelConfig, ChannelType, IncomingMessage, MessageAttachment, OutgoingMessage,
};
use crate::webhook::{WebhookError, WebhookMethod, WebhookRoute};
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};

/// Default Telegram Bot API base.
pub const DEFAULT_API_BASE: &str = "https://api.telegram.org";
const DEFAULT_POLL_TIMEOUT_SECS: u64 = 30;
const DEFAULT_POLL_LIMIT: u32 = 100;
const BASE_RECONNECT_DELAY_SECS: u64 = 1;
const MAX_RECONNECT_DELAY_SECS: u64 = 60;

// ---------------------------------------------------------------------------
// Inline keyboards
// ---------------------------------------------------------------------------

/// Builder for `InlineKeyboardMarkup` payloads.
///
/// ```
/// use opensquilla_channels::telegram::InlineKeyboardBuilder;
/// let keyboard = InlineKeyboardBuilder::new()
///     .row(vec![
///         InlineKeyboardBuilder::callback_button("Yes", "yes"),
///         InlineKeyboardBuilder::callback_button("No", "no"),
///     ])
///     .row(vec![
///         InlineKeyboardBuilder::url_button("Docs", "https://docs.opensquilla.dev"),
///     ])
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct InlineKeyboardBuilder {
    rows: Vec<Vec<Value>>,
}

impl InlineKeyboardBuilder {
    /// Create an empty keyboard.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a row of buttons.
    pub fn row(mut self, buttons: Vec<Value>) -> Self {
        self.rows.push(buttons);
        self
    }

    /// A button that sends a callback query.
    pub fn callback_button(text: impl Into<String>, callback_data: impl Into<String>) -> Value {
        json!({
            "text": text.into(),
            "callback_data": callback_data.into(),
        })
    }

    /// A button that opens a URL.
    pub fn url_button(text: impl Into<String>, url: impl Into<String>) -> Value {
        json!({
            "text": text.into(),
            "url": url.into(),
        })
    }

    /// A button that launches a Web App.
    pub fn web_app_button(text: impl Into<String>, url: impl Into<String>) -> Value {
        json!({
            "text": text.into(),
            "web_app": { "url": url.into() },
        })
    }

    /// Build the `reply_markup` payload.
    pub fn build(self) -> Value {
        json!({ "inline_keyboard": self.rows })
    }
}

// ---------------------------------------------------------------------------
// Channel adapter
// ---------------------------------------------------------------------------

/// Telegram channel adapter backed by the Bot API.
pub struct TelegramChannel {
    config: ChannelConfig,
    client: reqwest::Client,
    bot_token: String,
    api_base: String,
    running: Arc<Mutex<bool>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    offset: Arc<Mutex<u64>>,
    poll_task: Arc<Mutex<Option<JoinHandle<()>>>>,
    default_parse_mode: String,
}

impl TelegramChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let bot_token = config
            .config
            .get("bot_token")
            .and_then(|v| v.as_str())
            .ok_or("No bot_token configured for Telegram")?
            .to_string();
        let api_base = config
            .config
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_API_BASE)
            .to_string();
        let default_parse_mode = config
            .config
            .get("parse_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("HTML")
            .to_string();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;
        Ok(Self {
            config,
            client,
            bot_token,
            api_base,
            running: Arc::new(Mutex::new(false)),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
            offset: Arc::new(Mutex::new(0)),
            poll_task: Arc::new(Mutex::new(None)),
            default_parse_mode,
        })
    }

    fn api_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", self.api_base, self.bot_token, method)
    }

    /// The bot's own profile (`getMe`).
    pub async fn get_me(&self) -> Result<Value, String> {
        let resp = self
            .client
            .get(self.api_url("getMe"))
            .send()
            .await
            .map_err(|e| format!("Telegram getMe request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram getMe parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram getMe error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    // -- outbound -----------------------------------------------------------

    /// Send a text message with a parse mode and optional reply markup.
    pub async fn send_text(
        &self,
        chat_id: &str,
        text: &str,
        parse_mode: Option<&str>,
        reply_to_message_id: Option<i64>,
        reply_markup: Option<Value>,
    ) -> Result<Value, String> {
        let mut payload = json!({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode.unwrap_or(&self.default_parse_mode),
        });
        if let Some(id) = reply_to_message_id {
            payload["reply_to_message_id"] = json!(id);
        }
        if let Some(markup) = reply_markup {
            payload["reply_markup"] = markup;
        }
        let resp = self
            .client
            .post(self.api_url("sendMessage"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Telegram sendMessage request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram sendMessage parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram sendMessage error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a photo by bytes (multipart upload).
    pub async fn send_photo(
        &self,
        chat_id: &str,
        photo: &[u8],
        file_name: &str,
        caption: &str,
    ) -> Result<Value, String> {
        let part = Part::bytes(photo.to_vec()).file_name(file_name.to_string());
        let form = Form::new()
            .part("photo", part)
            .text("chat_id", chat_id.to_string())
            .text("caption", caption.to_string())
            .text("parse_mode", self.default_parse_mode.clone());
        let resp = self
            .client
            .post(self.api_url("sendPhoto"))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("Telegram sendPhoto request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram sendPhoto parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram sendPhoto error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Send a document by bytes (multipart upload).
    pub async fn send_document(
        &self,
        chat_id: &str,
        data: &[u8],
        file_name: &str,
        caption: &str,
    ) -> Result<Value, String> {
        let part = Part::bytes(data.to_vec()).file_name(file_name.to_string());
        let form = Form::new()
            .part("document", part)
            .text("chat_id", chat_id.to_string())
            .text("caption", caption.to_string());
        let resp = self
            .client
            .post(self.api_url("sendDocument"))
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("Telegram sendDocument request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram sendDocument parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram sendDocument error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Edit the text of a previously sent message.
    pub async fn edit_message_text(
        &self,
        chat_id: &str,
        message_id: i64,
        text: &str,
    ) -> Result<(), String> {
        let payload = json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": self.default_parse_mode,
        });
        let resp = self
            .client
            .post(self.api_url("editMessageText"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Telegram editMessageText request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram editMessageText parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(())
        } else {
            Err(format!(
                "Telegram editMessageText error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Delete a message.
    pub async fn delete_message(&self, chat_id: &str, message_id: i64) -> Result<(), String> {
        let resp = self
            .client
            .post(self.api_url("deleteMessage"))
            .json(&json!({ "chat_id": chat_id, "message_id": message_id }))
            .send()
            .await
            .map_err(|e| format!("Telegram deleteMessage request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Telegram deleteMessage failed: {}", resp.status()))
        }
    }

    /// Answer an inline keyboard callback query.
    pub async fn answer_callback_query(
        &self,
        callback_query_id: &str,
        text: Option<&str>,
    ) -> Result<(), String> {
        let mut payload = json!({ "callback_query_id": callback_query_id });
        if let Some(t) = text {
            payload["text"] = json!(t);
        }
        let resp = self
            .client
            .post(self.api_url("answerCallbackQuery"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Telegram answerCallbackQuery request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!(
                "Telegram answerCallbackQuery failed: {}",
                resp.status()
            ))
        }
    }

    /// Resolve a file id to a downloadable path (`getFile`).
    pub async fn get_file(&self, file_id: &str) -> Result<Value, String> {
        let resp = self
            .client
            .post(self.api_url("getFile"))
            .json(&json!({ "file_id": file_id }))
            .send()
            .await
            .map_err(|e| format!("Telegram getFile request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram getFile parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram getFile error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Download a file by its `file_path` (from [`TelegramChannel::get_file`]).
    pub async fn download_file(&self, file_path: &str) -> Result<Vec<u8>, String> {
        let url = format!("{}/file/bot{}/{}", self.api_base, self.bot_token, file_path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Telegram file download request: {e}"))?;
        if resp.status().is_success() {
            resp.bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| format!("Telegram file download read: {e}"))
        } else {
            Err(format!("Telegram file download failed: {}", resp.status()))
        }
    }

    /// Send a chat action (typing, upload_photo, ...).
    pub async fn send_chat_action(&self, chat_id: &str, action: &str) -> Result<(), String> {
        let resp = self
            .client
            .post(self.api_url("sendChatAction"))
            .json(&json!({ "chat_id": chat_id, "action": action }))
            .send()
            .await
            .map_err(|e| format!("Telegram sendChatAction request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Telegram sendChatAction failed: {}", resp.status()))
        }
    }

    // -- webhook ------------------------------------------------------------

    /// Register a webhook URL with the Bot API.
    pub async fn set_webhook(&self, url: &str, secret_token: Option<&str>) -> Result<(), String> {
        let mut payload = json!({ "url": url });
        if let Some(token) = secret_token {
            payload["secret_token"] = json!(token);
        }
        let resp = self
            .client
            .post(self.api_url("setWebhook"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Telegram setWebhook request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram setWebhook parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(())
        } else {
            Err(format!(
                "Telegram setWebhook error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Remove the webhook and fall back to long polling.
    pub async fn delete_webhook(&self) -> Result<(), String> {
        let resp = self
            .client
            .post(self.api_url("deleteWebhook"))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| format!("Telegram deleteWebhook request: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(format!("Telegram deleteWebhook failed: {}", resp.status()))
        }
    }

    /// Inspect the current webhook configuration.
    pub async fn get_webhook_info(&self) -> Result<Value, String> {
        let resp = self
            .client
            .post(self.api_url("getWebhookInfo"))
            .json(&json!({}))
            .send()
            .await
            .map_err(|e| format!("Telegram getWebhookInfo request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram getWebhookInfo parse: {e}"))?;
        if body["ok"].as_bool().unwrap_or(false) {
            Ok(body["result"].clone())
        } else {
            Err(format!(
                "Telegram getWebhookInfo error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ))
        }
    }

    /// Build the axum [`WebhookRoute`] for the Bot API webhook.
    ///
    /// Verifies the `X-Telegram-Bot-Api-Secret-Token` header when a secret is
    /// configured and forwards parsed updates to the internal queue.
    pub fn webhook_route(&self, path: impl Into<String>, secret_token: Option<String>) -> WebhookRoute {
        let incoming = self.incoming.clone();
        let handler = crate::webhook::TelegramWebhookHandler::new().on_message(move |msg| {
            incoming.blocking_lock().push_back(msg);
            Ok(())
        });
        let mut route = WebhookRoute::new(path, WebhookMethod::Post, ChannelType::Telegram, handler);
        if let Some(secret) = secret_token {
            route = route.with_secret(secret);
        }
        route
    }

    // -- long polling -------------------------------------------------------

    /// Perform a single `getUpdates` long poll.
    pub async fn poll_once(&self) -> Result<Vec<IncomingMessage>, String> {
        let offset = { *self.offset.lock().await };
        let resp = self
            .client
            .post(self.api_url("getUpdates"))
            .json(&json!({
                "offset": offset,
                "timeout": DEFAULT_POLL_TIMEOUT_SECS,
                "limit": DEFAULT_POLL_LIMIT,
            }))
            .send()
            .await
            .map_err(|e| format!("Telegram getUpdates request: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Telegram getUpdates parse: {e}"))?;
        if !body["ok"].as_bool().unwrap_or(false) {
            return Err(format!(
                "Telegram getUpdates error: {}",
                body["description"].as_str().unwrap_or("unknown")
            ));
        }
        let mut messages = Vec::new();
        let Some(updates) = body["result"].as_array() else {
            return Ok(messages);
        };
        let mut max_update_id = offset;
        for update in updates {
            if let Some(uid) = update.get("update_id").and_then(|v| v.as_u64()) {
                max_update_id = max_update_id.max(uid);
            }
            match crate::webhook::parse_telegram_payload(update) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    debug!("Telegram update skipped: {e}");
                }
            }
        }
        // Ack processed updates so they are not redelivered.
        *self.offset.lock().await = max_update_id.saturating_add(1);
        Ok(messages)
    }

    /// Start the long-polling loop in the background.
    pub async fn start_polling(&self) -> Result<(), String> {
        {
            let mut running = self.running.lock().await;
            if *running {
                return Ok(());
            }
            *running = true;
        }
        let running = self.running.clone();
        let incoming = self.incoming.clone();
        let offset = self.offset.clone();
        let bot_token = self.bot_token.clone();
        let api_base = self.api_base.clone();
        let client = self.client.clone();
        let task = tokio::spawn(async move {
            info!("Telegram long-polling loop starting");
            let mut attempt: u32 = 0;
            loop {
                if !*running.lock().await {
                    return;
                }
                match poll_updates(
                    &client,
                    &api_base,
                    &bot_token,
                    offset.clone(),
                    incoming.clone(),
                )
                .await
                {
                    Ok(()) => {
                        attempt = 0;
                    }
                    Err(e) => {
                        error!("Telegram polling error: {e}");
                        if !*running.lock().await {
                            return;
                        }
                        let delay = telegram_reconnect_delay(attempt);
                        attempt = attempt.saturating_add(1).min(10);
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        });
        *self.poll_task.lock().await = Some(task);
        Ok(())
    }

    /// Stop the long-polling loop.
    pub async fn stop_polling(&self) {
        *self.running.lock().await = false;
        if let Some(task) = self.poll_task.lock().await.take() {
            task.abort();
        }
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }
}

/// One long-poll iteration shared between the loop and tests.
async fn poll_updates(
    client: &reqwest::Client,
    api_base: &str,
    bot_token: &str,
    offset: Arc<Mutex<u64>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
) -> Result<(), String> {
    let current = { *offset.lock().await };
    let url = format!("{}/bot{}/getUpdates", api_base, bot_token);
    let resp = client
        .post(&url)
        .json(&json!({
            "offset": current,
            "timeout": DEFAULT_POLL_TIMEOUT_SECS,
            "limit": DEFAULT_POLL_LIMIT,
        }))
        .send()
        .await
        .map_err(|e| format!("Telegram getUpdates request: {e}"))?;
    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("Telegram getUpdates parse: {e}"))?;
    if !body["ok"].as_bool().unwrap_or(false) {
        return Err(format!(
            "Telegram getUpdates error: {}",
            body["description"].as_str().unwrap_or("unknown")
        ));
    }
    let Some(updates) = body["result"].as_array() else {
        return Ok(());
    };
    let mut max_update_id = current;
    for update in updates {
        if let Some(uid) = update.get("update_id").and_then(|v| v.as_u64()) {
            max_update_id = max_update_id.max(uid);
        }
        match crate::webhook::parse_telegram_payload(update) {
            Ok(msg) => incoming.lock().await.push_back(msg),
            Err(e) => {
                debug!("Telegram update skipped: {e}");
            }
        }
    }
    *offset.lock().await = max_update_id.saturating_add(1);
    Ok(())
}

/// Exponential backoff capped at [`MAX_RECONNECT_DELAY_SECS`].
fn telegram_reconnect_delay(attempt: u32) -> Duration {
    let exp = BASE_RECONNECT_DELAY_SECS.saturating_mul(1u64 << attempt.min(5));
    Duration::from_secs(exp.min(MAX_RECONNECT_DELAY_SECS))
}

#[async_trait::async_trait]
impl Channel for TelegramChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Telegram
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let chat_id = message.channel_id.clone();
        let parse_mode = message
            .metadata
            .get("parse_mode")
            .and_then(|v| v.as_str())
            .or_else(|| Some(self.default_parse_mode.as_str()));
        let reply_to = message
            .thread_id
            .as_deref()
            .and_then(|t| t.parse::<i64>().ok());
        let reply_markup = message
            .metadata
            .get("reply_markup")
            .cloned()
            .or_else(|| {
                message
                    .metadata
                    .get("keyboard")
                    .cloned()
                    .map(|k| json!({ "inline_keyboard": k }))
            });

        // File upload path: send the first attachment as a document.
        if let Some(att) = message.attachments.first() {
            if let Some(data) = &att.data {
                if let Some(base64_str) = data.get("base64").and_then(|v| v.as_str()) {
                    use base64::Engine;
                    if let Ok(bytes) =
                        base64::engine::general_purpose::STANDARD.decode(base64_str)
                    {
                        let name = data
                            .get("file_name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("file.bin");
                        return self
                            .send_document(&chat_id, &bytes, name, &message.text)
                            .await
                            .map(|_| ());
                    }
                }
            }
            if let Some(url) = &att.url {
                // A URL attachment is quoted in the text as a fallback.
                let text = format!("{}\n{}", message.text, url);
                return self
                    .send_text(&chat_id, &text, parse_mode, reply_to, reply_markup)
                    .await
                    .map(|_| ());
            }
        }

        self.send_text(&chat_id, &message.text, parse_mode, reply_to, reply_markup)
            .await
            .map(|_| ())
    }

    async fn send_typing(&self, channel_id: &str) -> Result<(), String> {
        self.send_chat_action(channel_id, "typing").await
    }

    async fn set_webhook(&self, url: &str) -> Result<(), String> {
        self.set_webhook(url, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> TelegramChannel {
        TelegramChannel::new(ChannelConfig {
            channel_type: ChannelType::Telegram,
            channel_id: "123".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({ "bot_token": "tok" }),
        })
        .unwrap()
    }

    #[test]
    fn test_api_url() {
        let c = channel();
        assert_eq!(c.api_url("getMe"), "https://api.telegram.org/bottok/getMe");
    }

    #[test]
    fn test_inline_keyboard_builder() {
        let kb = InlineKeyboardBuilder::new()
            .row(vec![
                InlineKeyboardBuilder::callback_button("Yes", "yes"),
                InlineKeyboardBuilder::url_button("Docs", "https://x.dev"),
            ])
            .build();
        assert_eq!(kb["inline_keyboard"][0][0]["text"], "Yes");
        assert_eq!(kb["inline_keyboard"][0][0]["callback_data"], "yes");
        assert_eq!(kb["inline_keyboard"][0][1]["url"], "https://x.dev");
    }

    #[tokio::test]
    async fn test_poll_once_parses_updates() {
        // A mock server returning a single update.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let body = r#"{
                "ok": true,
                "result": [{
                    "update_id": 42,
                    "message": {
                        "message_id": 7,
                        "chat": {"id": 456, "type": "private"},
                        "from": {"id": 789, "first_name": "Alice"},
                        "text": "hi",
                        "date": 1625241600
                    }
                }]
            }"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = socket.write_all(resp.as_bytes()).await;
        });

        let offset = Arc::new(Mutex::new(0));
        let incoming = Arc::new(Mutex::new(VecDeque::new()));
        let client = reqwest::Client::new();
        let api_base = format!("http://{}", addr);
        poll_updates(&client, &api_base, "tok", offset.clone(), incoming.clone())
            .await
            .unwrap();
        assert_eq!(*offset.lock().await, 43);
        let msg = incoming.lock().await.pop_front().unwrap();
        assert_eq!(msg.text, "hi");
        assert_eq!(msg.user_name.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_reconnect_backoff_capped() {
        assert_eq!(telegram_reconnect_delay(0).as_secs(), 1);
        assert_eq!(telegram_reconnect_delay(1).as_secs(), 2);
        assert!(telegram_reconnect_delay(20).as_secs() <= MAX_RECONNECT_DELAY_SECS);
    }
}
