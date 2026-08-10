//! Webhook route integration for the OpenSquilla gateway.
//!
//! The Python channels system registers webhook routes with Starlette. This
//! module provides the Rust/axum equivalent: a [`WebhookRegistry`] that
//! manages [`WebhookRoute`]s and produces an axum [`Router`], plus handlers
//! for Slack, Telegram, and WeCom with HMAC / token signature verification.
//!
//! ```
//! use opensquilla_channels::webhook::{
//!     WebhookRegistry, WebhookRoute, WebhookMethod, SlackWebhookHandler,
//! };
//!
//! let registry = WebhookRegistry::new();
//! assert!(registry.register(WebhookRoute::new(
//!     "/webhooks/slack",
//!     WebhookMethod::Post,
//!     opensquilla_channels::types::ChannelType::Slack,
//!     SlackWebhookHandler::new().on_message(|msg| {
//!         println!("received: {}", msg.text);
//!         Ok(())
//!     }),
//! )).is_ok());
//! ```

use crate::types::{ChannelType, IncomingMessage, MessageAttachment};
use aes::Aes256;
use aes::cipher::{BlockDecrypt, KeyInit};
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::any;
use axum::{Json, Router};
use chrono::{DateTime, TimeZone, Utc};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};
use uuid::Uuid;

/// Lenient base64 decoder for WeCom `EncodingAESKey` (43-char key + `=`).
///
/// WeChat's reference implementations decode the key allowing non-zero
/// trailing bits on the final symbol, which the strict `STANDARD` engine
/// rejects. Mirror that so the documented sample key round-trips.
static WECOM_KEY_B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::GeneralPurpose::new(
        &base64::alphabet::STANDARD,
        base64::engine::general_purpose::GeneralPurposeConfig::new()
            .with_decode_allow_trailing_bits(true),
    );

/// AES-256-CBC decryptor used by the WeCom adapter.
/// The HTTP method a webhook route accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl WebhookMethod {
    /// Convert from an axum `Method`.
    pub fn from_http(method: &Method) -> Option<Self> {
        match *method {
            Method::GET => Some(Self::Get),
            Method::POST => Some(Self::Post),
            Method::PUT => Some(Self::Put),
            Method::PATCH => Some(Self::Patch),
            Method::DELETE => Some(Self::Delete),
            _ => None,
        }
    }

    /// The HTTP method as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
        }
    }

    /// Whether this route accepts the given HTTP method.
    pub fn matches(&self, method: &Method) -> bool {
        matches!(Self::from_http(method), Some(m) if m == *self)
    }
}

/// An error that can occur while processing a webhook.
#[derive(Debug, thiserror::Error)]
pub enum WebhookError {
    #[error("Missing required field: {0}")]
    MissingField(String),
    #[error("Invalid payload: {0}")]
    InvalidPayload(String),
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Unsupported event type: {0}")]
    UnsupportedEvent(String),
    #[error("Method not allowed: {0}")]
    MethodNotAllowed(String),
    #[error("Webhook handler error: {0}")]
    HandlerError(String),
    #[error("Webhook echo challenge: {0}")]
    Challenge(String),
    #[error("Route already registered: {0}")]
    RouteAlreadyRegistered(String),
}

/// The HTTP response produced by a webhook handler.
#[derive(Debug, Clone)]
pub struct WebhookResponse {
    pub status: StatusCode,
    pub body: Value,
}

impl WebhookResponse {
    /// A successful empty response.
    pub fn ok() -> Self {
        Self {
            status: StatusCode::OK,
            body: json!({ "ok": true }),
        }
    }

    /// A successful response with no body.
    pub fn empty() -> Self {
        Self {
            status: StatusCode::OK,
            body: Value::Null,
        }
    }

    /// A challenge response (Slack URL verification).
    pub fn challenge(challenge: impl Into<String>) -> Self {
        Self {
            status: StatusCode::OK,
            body: json!({ "challenge": challenge.into() }),
        }
    }

    /// Build a response from a status and body.
    pub fn json(status: StatusCode, body: Value) -> Self {
        Self { status, body }
    }
}

/// Handles incoming webhook requests for a specific channel adapter.
#[async_trait::async_trait]
pub trait WebhookHandler: Send + Sync {
    /// The channel type this handler services.
    fn channel_type(&self) -> ChannelType;

    /// Optionally answer a provider verification challenge (e.g. Slack
    /// `url_verification`). Returns the challenge string when this payload is
    /// such a challenge.
    fn verify_challenge(&self, _payload: &Value) -> Option<String> {
        None
    }

    /// Parse the webhook payload into an [`IncomingMessage`].
    ///
    /// `raw_body` is the raw request body (used for signature verification),
    /// `headers` are the request headers, and `signature_secret` is the
    /// configured shared secret for this route (e.g. a Slack signing secret or
    /// a WeCom EncodingAESKey).
    fn parse_payload(
        &self,
        payload: &Value,
        raw_body: &str,
        headers: &HeaderMap,
        signature_secret: Option<&str>,
    ) -> Result<IncomingMessage, WebhookError>;

    /// Process the parsed incoming message.
    async fn handle(&self, message: IncomingMessage) -> Result<WebhookResponse, WebhookError>;
}

type ParseFn = dyn Fn(&Value, &str, &HeaderMap, Option<&str>) -> Result<IncomingMessage, WebhookError>
    + Send
    + Sync;
type MessageCallback = dyn Fn(IncomingMessage) -> Result<(), String> + Send + Sync;

/// A webhook handler backed by plain functions.
pub struct FunctionWebhookHandler {
    channel_type: ChannelType,
    parse: Arc<ParseFn>,
    handle: Arc<dyn Fn(IncomingMessage) -> Result<WebhookResponse, WebhookError> + Send + Sync>,
}

impl FunctionWebhookHandler {
    /// Build a handler from a channel type, a parse function, and a handle
    /// function.
    pub fn new<F, G>(channel_type: ChannelType, parse: F, handle: G) -> Self
    where
        F: Fn(&Value, &str, &HeaderMap, Option<&str>) -> Result<IncomingMessage, WebhookError>
            + Send
            + Sync
            + 'static,
        G: Fn(IncomingMessage) -> Result<WebhookResponse, WebhookError> + Send + Sync + 'static,
    {
        Self {
            channel_type,
            parse: Arc::new(parse),
            handle: Arc::new(handle),
        }
    }
}

#[async_trait::async_trait]
impl WebhookHandler for FunctionWebhookHandler {
    fn channel_type(&self) -> ChannelType {
        self.channel_type.clone()
    }

    fn parse_payload(
        &self,
        payload: &Value,
        raw_body: &str,
        headers: &HeaderMap,
        signature_secret: Option<&str>,
    ) -> Result<IncomingMessage, WebhookError> {
        (self.parse)(payload, raw_body, headers, signature_secret)
    }

    async fn handle(&self, message: IncomingMessage) -> Result<WebhookResponse, WebhookError> {
        (self.handle)(message)
    }
}

/// A webhook route registered with the [`WebhookRegistry`].
#[derive(Clone)]
pub struct WebhookRoute {
    /// The path this route is mounted at (e.g. `/webhooks/slack`).
    pub path: String,
    /// The HTTP method this route accepts.
    pub method: WebhookMethod,
    /// The channel type serviced by this route.
    pub channel_type: ChannelType,
    /// The handler that processes requests.
    pub handler: Arc<dyn WebhookHandler + Send + Sync>,
    /// Optional shared secret used for signature verification.
    pub signature_secret: Option<String>,
}

impl WebhookRoute {
    /// Create a new webhook route.
    pub fn new(
        path: impl Into<String>,
        method: WebhookMethod,
        channel_type: ChannelType,
        handler: impl WebhookHandler + 'static,
    ) -> Self {
        Self {
            path: path.into(),
            method,
            channel_type,
            handler: Arc::new(handler),
            signature_secret: None,
        }
    }

    /// Set the shared secret used for signature verification.
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.signature_secret = Some(secret.into());
        self
    }
}

/// The axum router state for webhook routes.
#[derive(Clone)]
pub struct WebhookState {
    /// The registry backing the mounted routes.
    pub registry: Arc<WebhookRegistry>,
}

/// Registry that manages webhook routes and generates an axum [`Router`].
#[derive(Clone, Default)]
pub struct WebhookRegistry {
    routes: Arc<RwLock<Vec<WebhookRoute>>>,
}

impl WebhookRegistry {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a webhook route. Returns an error if a route with the same
    /// path is already registered.
    pub fn register(&self, route: WebhookRoute) -> Result<(), WebhookError> {
        let mut routes = self
            .routes
            .write()
            .map_err(|_| WebhookError::HandlerError("registry lock poisoned".into()))?;
        if routes.iter().any(|r| r.path == route.path) {
            return Err(WebhookError::RouteAlreadyRegistered(route.path.clone()));
        }
        info!(path = %route.path, "Registered webhook route");
        routes.push(route);
        Ok(())
    }

    /// Register a route, replacing any route with the same path.
    pub fn upsert(&self, route: WebhookRoute) {
        let mut routes = self.routes.write().expect("webhook registry lock poisoned");
        routes.retain(|r| r.path != route.path);
        routes.push(route);
    }

    /// Remove a route by path. Returns whether a route was removed.
    pub fn unregister(&self, path: &str) -> bool {
        let mut routes = self.routes.write().expect("webhook registry lock poisoned");
        let before = routes.len();
        routes.retain(|r| r.path != path);
        let removed = routes.len() != before;
        if removed {
            info!(path = %path, "Unregistered webhook route");
        }
        removed
    }

    /// Get a route by path.
    pub fn get(&self, path: &str) -> Option<WebhookRoute> {
        self.routes
            .read()
            .expect("webhook registry lock poisoned")
            .iter()
            .find(|r| r.path == path)
            .cloned()
    }

    /// List all registered routes.
    pub fn routes(&self) -> Vec<WebhookRoute> {
        self.routes
            .read()
            .expect("webhook registry lock poisoned")
            .clone()
    }

    /// List all registered route paths.
    pub fn paths(&self) -> Vec<String> {
        self.routes().into_iter().map(|r| r.path).collect()
    }

    /// Number of registered routes.
    pub fn len(&self) -> usize {
        self.routes
            .read()
            .expect("webhook registry lock poisoned")
            .len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build an axum [`Router`] mounting all registered webhook routes.
    ///
    /// Each route is mounted at its configured path; the handler dispatches on
    /// the HTTP method, verifies the channel signature, parses the payload into
    /// an [`IncomingMessage`], and forwards it to the route handler.
    pub fn router(&self) -> Router {
        let registry = Arc::new(self.clone());
        let state = WebhookState {
            registry: registry.clone(),
        };
        let routes = self.routes();
        let mut router = Router::new();
        for route in routes {
            let handler = route.handler.clone();
            let secret = route.signature_secret.clone();
            let expected_method = route.method;
            let path = route.path.clone();
            let route_handler = any(
                move |State(_): State<WebhookState>,
                      actual_method: Method,
                      headers: HeaderMap,
                      query: Query<HashMap<String, String>>,
                      body: Bytes| async move {
                    dispatch_webhook(
                        handler.clone(),
                        secret.clone(),
                        expected_method,
                        actual_method,
                        headers,
                        query.0,
                        body,
                    )
                    .await
                },
            );
            router = router.route(&path, route_handler);
        }
        router.with_state(state)
    }
}

/// Dispatch an incoming webhook request to its handler.
async fn dispatch_webhook(
    handler: Arc<dyn WebhookHandler + Send + Sync>,
    secret: Option<String>,
    expected_method: WebhookMethod,
    actual_method: Method,
    headers: HeaderMap,
    query: HashMap<String, String>,
    body: Bytes,
) -> Response {
    if !expected_method.matches(&actual_method) {
        warn!(
            expected = expected_method.as_str(),
            actual = %actual_method,
            "Webhook method mismatch"
        );
        return (
            StatusCode::METHOD_NOT_ALLOWED,
            Json(json!({ "error": "method not allowed" })),
        )
            .into_response();
    }

    let raw_bytes = body.to_vec();
    let raw_text = String::from_utf8_lossy(&raw_bytes).to_string();
    let payload: Value = if raw_text.trim().is_empty() {
        if !query.is_empty() {
            serde_json::to_value(&query).unwrap_or(Value::Null)
        } else {
            Value::Null
        }
    } else {
        serde_json::from_str(&raw_text).unwrap_or_else(|_| json!({ "raw": raw_text }))
    };

    // 1. Provider verification challenge (e.g. Slack url_verification).
    if let Some(challenge) = handler.verify_challenge(&payload) {
        return Json(json!({ "challenge": challenge })).into_response();
    }

    // 2. Channel-specific signature verification.
    if let Err(err) = verify_webhook_signature(
        &handler.channel_type(),
        &payload,
        &raw_bytes,
        &headers,
        &query,
        secret.as_deref(),
    ) {
        return match err {
            WebhookError::SignatureVerificationFailed => (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "signature verification failed" })),
            )
                .into_response(),
            other => (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": other.to_string() })),
            )
                .into_response(),
        };
    }

    // 3. Parse the payload into an IncomingMessage.
    match handler.parse_payload(&payload, &raw_text, &headers, secret.as_deref()) {
        Ok(message) => match handler.handle(message).await {
            Ok(resp) => (resp.status, Json(resp.body)).into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": err.to_string() })),
            )
                .into_response(),
        },
        Err(WebhookError::Challenge(challenge)) => {
            Json(json!({ "echostr": challenge })).into_response()
        }
        Err(WebhookError::SignatureVerificationFailed) => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "signature verification failed" })),
        )
            .into_response(),
        Err(err) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": err.to_string() })),
        )
            .into_response(),
    }
}

/// Verify a webhook signature using the channel-appropriate scheme.
///
/// When no secret is configured, verification is skipped (the caller has opted
/// out of signature enforcement).
pub fn verify_webhook_signature(
    channel_type: &ChannelType,
    payload: &Value,
    raw_body: &[u8],
    headers: &HeaderMap,
    query: &HashMap<String, String>,
    secret: Option<&str>,
) -> Result<(), WebhookError> {
    let Some(secret) = secret else {
        return Ok(());
    };

    match channel_type {
        ChannelType::Slack => {
            let timestamp = headers
                .get("x-slack-request-timestamp")
                .and_then(|v| v.to_str().ok())
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            let signature = headers
                .get("x-slack-signature")
                .and_then(|v| v.to_str().ok())
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            let body = String::from_utf8_lossy(raw_body);
            if verify_slack_signature(secret, timestamp, &body, signature) {
                Ok(())
            } else {
                Err(WebhookError::SignatureVerificationFailed)
            }
        }
        ChannelType::Telegram => {
            if verify_telegram_secret_token(headers, secret) {
                Ok(())
            } else {
                Err(WebhookError::SignatureVerificationFailed)
            }
        }
        ChannelType::WeCom => {
            // Callback verification echo requests carry `echostr` as a query
            // parameter; decryption and verification are handled in
            // `WeComWebhookHandler::parse_payload`.
            if payload.get("echostr").is_some() {
                return Ok(());
            }
            let timestamp = query
                .get("timestamp")
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            let nonce = query
                .get("nonce")
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            let msg_signature = query
                .get("msg_signature")
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            let encrypt = payload
                .get("Encrypt")
                .and_then(|v| v.as_str())
                .ok_or(WebhookError::SignatureVerificationFailed)?;
            if verify_wecom_signature(secret, timestamp, nonce, encrypt, msg_signature) {
                Ok(())
            } else {
                Err(WebhookError::SignatureVerificationFailed)
            }
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Signature verification primitives
// ---------------------------------------------------------------------------

/// Compute an HMAC-SHA256 digest over `message` with `secret`, hex-encoded.
pub fn hmac_sha256_hex(secret: &[u8], message: &[u8]) -> String {
    use hmac::Mac;
    let mut mac = <hmac::Hmac<sha2::Sha256> as hmac::Mac>::new_from_slice(secret)
        .expect("HMAC accepts keys of any size");
    mac.update(message);
    hex::encode(mac.finalize().into_bytes())
}

/// Constant-time HMAC-SHA256 verification of a hex-encoded signature.
pub fn verify_hmac_sha256(secret: &str, message: &[u8], signature_hex: &str) -> bool {
    use hmac::Mac;
    let Ok(decoded) = hex::decode(signature_hex) else {
        return false;
    };
    let mut mac = match <hmac::Hmac<sha2::Sha256> as hmac::Mac>::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(message);
    mac.verify_slice(&decoded).is_ok()
}

/// Verify a Slack Events API signature.
///
/// Slack signs the body as `v0=HMAC_SHA256(signing_secret, "v0:{timestamp}:{body}")`.
pub fn verify_slack_signature(
    signing_secret: &str,
    timestamp: &str,
    body: &str,
    signature: &str,
) -> bool {
    let Some(expected_hex) = signature.strip_prefix("v0=") else {
        return false;
    };
    let message = format!("v0:{timestamp}:{body}");
    verify_hmac_sha256(signing_secret, message.as_bytes(), expected_hex)
}

/// Verify a Telegram Bot API secret token header.
pub fn verify_telegram_secret_token(headers: &HeaderMap, secret: &str) -> bool {
    headers
        .get("x-telegram-bot-api-secret-token")
        .and_then(|v| v.to_str().ok())
        .map(|v| v == secret)
        .unwrap_or(false)
}

/// Compute the WeCom callback SHA1 signature:
/// `SHA1(sort(token, timestamp, nonce, encrypt).join(""))`.
pub fn verify_wecom_signature(
    token: &str,
    timestamp: &str,
    nonce: &str,
    encrypt: &str,
    signature: &str,
) -> bool {
    use sha1::Digest;

    let mut parts = [
        token.to_string(),
        timestamp.to_string(),
        nonce.to_string(),
        encrypt.to_string(),
    ];
    parts.sort();
    let joined = parts.join("");
    let digest = hex::encode(sha1::Sha1::digest(joined.as_bytes()));
    digest == signature
}

/// Decrypt a WeCom callback payload (`Encrypt` field).
///
/// WeCom encrypts the callback body with AES-256-CBC. The key is the base64
/// decode of `EncodingAESKey + "="`, the IV is the first 16 bytes of that key,
/// and the plaintext has the form
/// `random(16) || msg_len(4, big-endian) || msg || receive_id`.
pub fn decrypt_wecom_payload(
    encoding_aes_key: &str,
    ciphertext: &str,
) -> Result<String, WebhookError> {
    use base64::Engine;

    let full_key = format!("{encoding_aes_key}=");
    let key_bytes = WECOM_KEY_B64
        .decode(&full_key)
        .map_err(|_| WebhookError::InvalidPayload("invalid EncodingAESKey".into()))?;
    if key_bytes.len() != 32 {
        return Err(WebhookError::InvalidPayload(
            "EncodingAESKey must decode to 32 bytes".into(),
        ));
    }

    let iv: [u8; 16] = key_bytes[0..16].try_into().unwrap();
    let key = aes::cipher::generic_array::GenericArray::from_slice(&key_bytes);
    let cipher = Aes256::new(key);

    let encrypted = base64::engine::general_purpose::STANDARD
        .decode(ciphertext)
        .map_err(|_| WebhookError::InvalidPayload("invalid base64 ciphertext".into()))?;
    let mut buf = encrypted.clone();
    let mut prev = iv;
    for chunk in buf.chunks_mut(16) {
        let mut block: [u8; 16] = chunk.try_into().unwrap();
        let mut ga_block = aes::cipher::generic_array::GenericArray::from(block);
        cipher.decrypt_block(&mut ga_block);
        block = ga_block.into();
        for (b, p) in block.iter_mut().zip(prev.iter()) {
            *b ^= *p;
        }
        chunk.copy_from_slice(&block);
        prev = block;
    }
    let pad_len = buf[buf.len() - 1] as usize;
    if pad_len == 0 || pad_len > 16 {
        return Err(WebhookError::InvalidPayload("invalid PKCS7 padding".into()));
    }
    let plaintext = &buf[..buf.len() - pad_len];

    if plaintext.len() < 20 {
        return Err(WebhookError::InvalidPayload(
            "decrypted payload too short".into(),
        ));
    }
    let msg_len =
        u32::from_be_bytes([plaintext[16], plaintext[17], plaintext[18], plaintext[19]]) as usize;
    let msg_bytes = plaintext
        .get(20..20 + msg_len)
        .ok_or_else(|| WebhookError::InvalidPayload("message length out of range".into()))?;
    String::from_utf8(msg_bytes.to_vec())
        .map_err(|_| WebhookError::InvalidPayload("decrypted message is not UTF-8".into()))
}

// ---------------------------------------------------------------------------
// Payload parsing
// ---------------------------------------------------------------------------

fn ts_float_to_datetime(ts: f64) -> DateTime<Utc> {
    let secs = ts.floor() as i64;
    let nanos = ((ts - ts.floor()) * 1e9) as u32;
    Utc.timestamp_opt(secs, nanos)
        .single()
        .unwrap_or_else(Utc::now)
}

fn ts_seconds_to_datetime(secs: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(secs, 0).single().unwrap_or_else(Utc::now)
}

/// Parse a Slack Events API message event into an [`IncomingMessage`].
pub fn parse_slack_payload(payload: &Value) -> Result<IncomingMessage, WebhookError> {
    let event = payload
        .get("event")
        .ok_or_else(|| WebhookError::MissingField("event".into()))?;
    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if event_type != "message" {
        return Err(WebhookError::UnsupportedEvent(event_type.into()));
    }
    // Ignore bot messages to avoid feedback loops.
    if event.get("subtype").and_then(|v| v.as_str()) == Some("bot_message") {
        return Err(WebhookError::UnsupportedEvent("bot_message".into()));
    }

    let channel_id = event
        .get("channel")
        .and_then(|v| v.as_str())
        .ok_or_else(|| WebhookError::MissingField("event.channel".into()))?
        .to_string();
    let user_id = event
        .get("user")
        .and_then(|v| v.as_str())
        .ok_or_else(|| WebhookError::MissingField("event.user".into()))?
        .to_string();
    let text = event
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let thread_id = event
        .get("thread_ts")
        .and_then(|v| v.as_str())
        .map(String::from);
    let ts = event
        .get("ts")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let timestamp = ts.map(ts_float_to_datetime).unwrap_or_else(Utc::now);

    let attachments: Vec<MessageAttachment> = event
        .get("files")
        .and_then(|v| v.as_array())
        .map(|files| {
            files
                .iter()
                .map(|f| MessageAttachment {
                    attachment_type: "file".to_string(),
                    url: f
                        .get("url_private")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    data: Some(f.clone()),
                    mime_type: f.get("mimetype").and_then(|v| v.as_str()).map(String::from),
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type: ChannelType::Slack,
        user_id,
        user_name: event
            .get("username")
            .and_then(|v| v.as_str())
            .map(String::from),
        text,
        thread_id,
        attachments,
        timestamp,
        raw: payload.clone(),
        metadata: serde_json::Value::Null,
        provenance_authenticated: false,
        sender_is_group_mentioned: false,
    })
}

/// Parse a Telegram Bot API update into an [`IncomingMessage`].
pub fn parse_telegram_payload(payload: &Value) -> Result<IncomingMessage, WebhookError> {
    let message = payload
        .get("message")
        .or_else(|| payload.get("edited_message"))
        .ok_or_else(|| WebhookError::MissingField("message".into()))?;
    let chat = message
        .get("chat")
        .ok_or_else(|| WebhookError::MissingField("message.chat".into()))?;
    let chat_id = chat
        .get("id")
        .and_then(|v| v.as_i64())
        .map(|i| i.to_string())
        .or_else(|| chat.get("id").and_then(|v| v.as_str()).map(String::from))
        .ok_or_else(|| WebhookError::MissingField("message.chat.id".into()))?;

    let from = message.get("from");
    let user_id = from
        .and_then(|f| f.get("id"))
        .and_then(|v| v.as_i64())
        .map(|i| i.to_string())
        .or_else(|| {
            from.and_then(|f| f.get("id"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    let user_name = from
        .and_then(|f| f.get("first_name"))
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            from.and_then(|f| f.get("username"))
                .and_then(|v| v.as_str())
                .map(String::from)
        });

    let text = message
        .get("text")
        .and_then(|v| v.as_str())
        .map(String::from)
        .or_else(|| {
            message
                .get("caption")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    let thread_id = message
        .get("message_thread_id")
        .and_then(|v| v.as_i64())
        .map(|i| i.to_string());
    let timestamp = message
        .get("date")
        .and_then(|v| v.as_i64())
        .map(ts_seconds_to_datetime)
        .unwrap_or_else(Utc::now);

    Ok(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id: chat_id,
        channel_type: ChannelType::Telegram,
        user_id,
        user_name,
        text,
        thread_id,
        attachments: Vec::new(),
        timestamp,
        raw: payload.clone(),
        metadata: serde_json::Value::Null,
        provenance_authenticated: false,
        sender_is_group_mentioned: false,
    })
}

/// Parse a (decrypted) WeCom callback payload into an [`IncomingMessage`].
pub fn parse_wecom_payload(payload: &Value) -> Result<IncomingMessage, WebhookError> {
    let msg_type = payload
        .get("MsgType")
        .and_then(|v| v.as_str())
        .ok_or_else(|| WebhookError::MissingField("MsgType".into()))?;
    if msg_type == "event" {
        return Err(WebhookError::UnsupportedEvent("event".into()));
    }

    let channel_id = payload
        .get("ToUserName")
        .and_then(|v| v.as_str())
        .ok_or_else(|| WebhookError::MissingField("ToUserName".into()))?
        .to_string();
    let user_id = payload
        .get("FromUserName")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let text = match msg_type {
        "text" => payload
            .get("Content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "image" => "[image]".to_string(),
        "voice" => "[voice]".to_string(),
        "video" => "[video]".to_string(),
        "location" => format!(
            "Location: {}",
            payload.get("Label").and_then(|v| v.as_str()).unwrap_or("")
        ),
        _ => String::new(),
    };

    let timestamp = payload
        .get("CreateTime")
        .and_then(|v| v.as_i64())
        .map(ts_seconds_to_datetime)
        .unwrap_or_else(Utc::now);

    Ok(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id,
        channel_type: ChannelType::WeCom,
        user_id,
        user_name: None,
        text,
        thread_id: None,
        attachments: Vec::new(),
        timestamp,
        raw: payload.clone(),
        metadata: serde_json::Value::Null,
        provenance_authenticated: false,
        sender_is_group_mentioned: false,
    })
}

/// Parse a generic webhook payload into an [`IncomingMessage`] with the given
/// channel identity.
pub fn parse_incoming_message(
    channel_type: ChannelType,
    channel_id: impl Into<String>,
    user_id: impl Into<String>,
    user_name: Option<String>,
    text: impl Into<String>,
    thread_id: Option<String>,
    raw: Value,
) -> IncomingMessage {
    IncomingMessage {
        id: Uuid::new_v4(),
        channel_id: channel_id.into(),
        channel_type,
        user_id: user_id.into(),
        user_name,
        text: text.into(),
        thread_id,
        attachments: Vec::new(),
        timestamp: Utc::now(),
        raw,
        metadata: serde_json::Value::Null,
        provenance_authenticated: false,
        sender_is_group_mentioned: false,
    }
}

// ---------------------------------------------------------------------------
// Channel adapters
// ---------------------------------------------------------------------------

/// Handler for Slack Events API webhooks.
pub struct SlackWebhookHandler {
    on_message: Option<Arc<MessageCallback>>,
}

impl SlackWebhookHandler {
    /// Create a new Slack webhook handler.
    pub fn new() -> Self {
        Self { on_message: None }
    }

    /// Attach a callback invoked for each parsed incoming message.
    pub fn on_message<F>(mut self, callback: F) -> Self
    where
        F: Fn(IncomingMessage) -> Result<(), String> + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }
}

impl Default for SlackWebhookHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl WebhookHandler for SlackWebhookHandler {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Slack
    }

    fn verify_challenge(&self, payload: &Value) -> Option<String> {
        if payload.get("type").and_then(|v| v.as_str()) == Some("url_verification") {
            payload
                .get("challenge")
                .and_then(|v| v.as_str())
                .map(String::from)
        } else {
            None
        }
    }

    fn parse_payload(
        &self,
        payload: &Value,
        _raw_body: &str,
        _headers: &HeaderMap,
        _signature_secret: Option<&str>,
    ) -> Result<IncomingMessage, WebhookError> {
        parse_slack_payload(payload)
    }

    async fn handle(&self, message: IncomingMessage) -> Result<WebhookResponse, WebhookError> {
        if let Some(callback) = &self.on_message {
            callback(message).map_err(WebhookError::HandlerError)?;
        }
        Ok(WebhookResponse::ok())
    }
}

/// Handler for Telegram Bot API webhooks.
pub struct TelegramWebhookHandler {
    on_message: Option<Arc<MessageCallback>>,
}

impl TelegramWebhookHandler {
    /// Create a new Telegram webhook handler.
    pub fn new() -> Self {
        Self { on_message: None }
    }

    /// Attach a callback invoked for each parsed incoming message.
    pub fn on_message<F>(mut self, callback: F) -> Self
    where
        F: Fn(IncomingMessage) -> Result<(), String> + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }
}

impl Default for TelegramWebhookHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl WebhookHandler for TelegramWebhookHandler {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Telegram
    }

    fn parse_payload(
        &self,
        payload: &Value,
        _raw_body: &str,
        _headers: &HeaderMap,
        _signature_secret: Option<&str>,
    ) -> Result<IncomingMessage, WebhookError> {
        parse_telegram_payload(payload)
    }

    async fn handle(&self, message: IncomingMessage) -> Result<WebhookResponse, WebhookError> {
        if let Some(callback) = &self.on_message {
            callback(message).map_err(WebhookError::HandlerError)?;
        }
        Ok(WebhookResponse::ok())
    }
}

/// Handler for WeCom callback webhooks (encrypted or plaintext).
pub struct WeComWebhookHandler {
    on_message: Option<Arc<MessageCallback>>,
}

impl WeComWebhookHandler {
    /// Create a new WeCom webhook handler.
    pub fn new() -> Self {
        Self { on_message: None }
    }

    /// Attach a callback invoked for each parsed incoming message.
    pub fn on_message<F>(mut self, callback: F) -> Self
    where
        F: Fn(IncomingMessage) -> Result<(), String> + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }
}

impl Default for WeComWebhookHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl WebhookHandler for WeComWebhookHandler {
    fn channel_type(&self) -> ChannelType {
        ChannelType::WeCom
    }

    fn parse_payload(
        &self,
        payload: &Value,
        _raw_body: &str,
        _headers: &HeaderMap,
        signature_secret: Option<&str>,
    ) -> Result<IncomingMessage, WebhookError> {
        // Callback verification echo: `echostr` is an encrypted challenge sent
        // as a query parameter. Decrypt it and surface it via Challenge so the
        // dispatcher can answer with the plaintext.
        if let Some(echostr) = payload.get("echostr").and_then(|v| v.as_str()) {
            let key = signature_secret.ok_or(WebhookError::SignatureVerificationFailed)?;
            let decrypted = decrypt_wecom_payload(key, echostr)?;
            return Err(WebhookError::Challenge(decrypted));
        }
        if let Some(encrypt) = payload.get("Encrypt").and_then(|v| v.as_str()) {
            let key = signature_secret.ok_or(WebhookError::SignatureVerificationFailed)?;
            let plaintext = decrypt_wecom_payload(key, encrypt)?;
            let plain_json: Value = serde_json::from_str(&plaintext)
                .map_err(|e| WebhookError::InvalidPayload(e.to_string()))?;
            if let Some(echostr) = plain_json.get("echostr").and_then(|v| v.as_str()) {
                return Err(WebhookError::Challenge(echostr.to_string()));
            }
            return parse_wecom_payload(&plain_json);
        }
        parse_wecom_payload(payload)
    }

    async fn handle(&self, message: IncomingMessage) -> Result<WebhookResponse, WebhookError> {
        if let Some(callback) = &self.on_message {
            callback(message).map_err(WebhookError::HandlerError)?;
        }
        Ok(WebhookResponse::ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes::cipher::BlockEncrypt;
    use axum::body::Body;
    use axum::http::Request;
    use base64::Engine as _;

    /// A valid 43-character WeCom EncodingAESKey (documented sample).
    const WECOM_KEY: &str = "jWmYm7qr5nMoAUwZRjGtBxmz3KA1tkAj3ykkR6q2B2C";

    fn slack_event_payload() -> Value {
        json!({
            "token": "abc123",
            "team_id": "T0001",
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C123",
                "user": "U456",
                "text": "hello from slack",
                "ts": "1625241600.000001",
                "thread_ts": "1625241600.000000"
            }
        })
    }

    #[test]
    fn test_registry_register_get_unregister() {
        let registry = WebhookRegistry::new();
        let route = WebhookRoute::new(
            "/webhooks/slack",
            WebhookMethod::Post,
            ChannelType::Slack,
            SlackWebhookHandler::new(),
        );
        registry.register(route).unwrap();
        assert_eq!(registry.len(), 1);
        assert!(registry.get("/webhooks/slack").is_some());
        assert_eq!(registry.paths(), vec!["/webhooks/slack".to_string()]);

        assert!(registry.unregister("/webhooks/slack"));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_duplicate_rejected() {
        let registry = WebhookRegistry::new();
        let route = WebhookRoute::new(
            "/webhooks/slack",
            WebhookMethod::Post,
            ChannelType::Slack,
            SlackWebhookHandler::new(),
        );
        registry.register(route).unwrap();
        let duplicate = WebhookRoute::new(
            "/webhooks/slack",
            WebhookMethod::Post,
            ChannelType::Slack,
            SlackWebhookHandler::new(),
        );
        let err = registry.register(duplicate).unwrap_err();
        assert!(matches!(err, WebhookError::RouteAlreadyRegistered(_)));
    }

    #[test]
    fn test_verify_hmac_sha256() {
        let secret = "secret";
        let digest = hmac_sha256_hex(secret.as_bytes(), b"message");
        assert!(verify_hmac_sha256(secret, b"message", &digest));
        assert!(!verify_hmac_sha256(secret, b"other", &digest));
        assert!(!verify_hmac_sha256(secret, b"message", "not-hex"));
    }

    #[test]
    fn test_verify_slack_signature() {
        let signing_secret = "8f742231b10e2628b0a4c21e04c3e5f1";
        let timestamp = "1531420618";
        let body = "token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2H3EV&...";
        let signature = hmac_sha256_hex(
            signing_secret.as_bytes(),
            format!("v0:{timestamp}:{body}").as_bytes(),
        );
        let full = format!("v0={signature}");
        assert!(verify_slack_signature(
            signing_secret,
            timestamp,
            body,
            &full
        ));
        assert!(!verify_slack_signature(
            signing_secret,
            "1531420619",
            body,
            &full
        ));
        assert!(!verify_slack_signature(
            signing_secret,
            timestamp,
            body,
            "v0=deadbeef"
        ));
    }

    #[test]
    fn test_verify_wecom_signature() {
        let token = "QDG6eK";
        let timestamp = "1409659813";
        let nonce = "1372623149";
        let encrypt = "9jq3f...";
        let signature = {
            use sha1::Digest;
            let mut parts = [
                token.to_string(),
                timestamp.to_string(),
                nonce.to_string(),
                encrypt.to_string(),
            ];
            parts.sort();
            let joined = parts.join("");
            hex::encode(sha1::Sha1::digest(joined.as_bytes()))
        };
        assert!(verify_wecom_signature(
            token, timestamp, nonce, encrypt, &signature
        ));
        assert!(!verify_wecom_signature(
            token, timestamp, nonce, encrypt, "deadbeef"
        ));
    }

    #[test]
    fn test_wecom_aes_roundtrip() {
        let plaintext = r#"{"MsgType":"text","ToUserName":"ww123","FromUserName":"user1","MsgId":"1","Content":"hi","CreateTime":12345}"#;
        let encrypted = {
            let full_key = format!("{WECOM_KEY}=");
            let key_bytes = WECOM_KEY_B64
                .decode(&full_key)
                .unwrap();
            let iv = &key_bytes[0..16];
            let key = aes::cipher::generic_array::GenericArray::from_slice(&key_bytes);
            let cipher = aes::Aes256::new(key);

            let mut buf = plaintext.as_bytes().to_vec();
            let block_size = 16;
            let pad_len = block_size - (buf.len() % block_size);
            buf.resize(buf.len() + pad_len, pad_len as u8);

            let mut prev = aes::cipher::generic_array::GenericArray::clone_from_slice(iv);
            for chunk in buf.chunks_mut(block_size) {
                let mut block = aes::cipher::generic_array::GenericArray::clone_from_slice(chunk);
                for (b, p) in block.iter_mut().zip(prev.iter()) {
                    *b ^= *p;
                }
                cipher.encrypt_block(&mut block);
                chunk.copy_from_slice(&block);
                prev = block;
            }
            base64::engine::general_purpose::STANDARD.encode(&buf)
        };
        let decrypted = decrypt_wecom_payload(WECOM_KEY, &encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_parse_slack_payload() {
        let message = parse_slack_payload(&slack_event_payload()).unwrap();
        assert_eq!(message.channel_type, ChannelType::Slack);
        assert_eq!(message.channel_id, "C123");
        assert_eq!(message.user_id, "U456");
        assert_eq!(message.text, "hello from slack");
        assert_eq!(message.thread_id.as_deref(), Some("1625241600.000000"));
    }

    #[test]
    fn test_parse_slack_challenge_handled_outside() {
        let payload = json!({ "type": "url_verification", "challenge": "challenge-token" });
        let handler = SlackWebhookHandler::new();
        assert_eq!(
            handler.verify_challenge(&payload).as_deref(),
            Some("challenge-token")
        );
    }

    #[test]
    fn test_parse_telegram_payload() {
        let payload = json!({
            "update_id": 123,
            "message": {
                "message_id": 1,
                "chat": { "id": 456, "type": "private" },
                "from": { "id": 789, "first_name": "Alice", "username": "alice" },
                "text": "hello telegram",
                "date": 1625241600
            }
        });
        let message = parse_telegram_payload(&payload).unwrap();
        assert_eq!(message.channel_type, ChannelType::Telegram);
        assert_eq!(message.channel_id, "456");
        assert_eq!(message.user_id, "789");
        assert_eq!(message.user_name.as_deref(), Some("Alice"));
        assert_eq!(message.text, "hello telegram");
    }

    #[test]
    fn test_parse_wecom_payload() {
        let payload = json!({
            "MsgType": "text",
            "ToUserName": "ww123",
            "FromUserName": "user1",
            "MsgId": "1",
            "Content": "hello wecom",
            "CreateTime": 1625241600
        });
        let message = parse_wecom_payload(&payload).unwrap();
        assert_eq!(message.channel_type, ChannelType::WeCom);
        assert_eq!(message.channel_id, "ww123");
        assert_eq!(message.user_id, "user1");
        assert_eq!(message.text, "hello wecom");
    }

    #[tokio::test]
    async fn test_dispatch_parses_and_handles() {
        let handler = SlackWebhookHandler::new().on_message(|msg| {
            assert_eq!(msg.text, "hello from slack");
            Ok(())
        });
        let response = dispatch_webhook(
            Arc::new(handler),
            None,
            WebhookMethod::Post,
            Method::POST,
            HeaderMap::new(),
            HashMap::new(),
            Bytes::from(serde_json::to_string(&slack_event_payload()).unwrap()),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_dispatch_method_mismatch() {
        let handler = SlackWebhookHandler::new();
        let response = dispatch_webhook(
            Arc::new(handler),
            None,
            WebhookMethod::Post,
            Method::GET,
            HeaderMap::new(),
            HashMap::new(),
            Bytes::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn test_dispatch_signature_failure() {
        let handler = TelegramWebhookHandler::new();
        let response = dispatch_webhook(
            Arc::new(handler),
            Some("expected-token".to_string()),
            WebhookMethod::Post,
            Method::POST,
            HeaderMap::new(), // no X-Telegram-Bot-Api-Secret-Token
            HashMap::new(),
            Bytes::from(r#"{"update_id":1,"message":{}}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_router_end_to_end() {
        use tower::ServiceExt;

        let registry = WebhookRegistry::new();
        registry
            .register(WebhookRoute::new(
                "/webhooks/slack",
                WebhookMethod::Post,
                ChannelType::Slack,
                SlackWebhookHandler::new().on_message(|_| Ok(())),
            ))
            .unwrap();
        let app = registry.router();

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/webhooks/slack")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_string(&slack_event_payload()).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // A route that is not registered yields 404.
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/webhooks/unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
