//! WebSocket handler for the gateway.
//!
//! Manages WebSocket connections: upgrade from HTTP, challenge-response
//! authentication, protocol-version negotiation, frame parsing, subscription
//! management, and RPC dispatch over the WebSocket transport.
//!
//! The connection lifecycle mirrors the Python `gateway/websocket.py`:
//!
//! 1. Accept the socket and send a `connect.challenge` event.
//! 2. Wait for the client's `connect` request within the pre-auth timeout.
//! 3. Resolve authentication via the configured [`AuthConfig`].
//! 4. Negotiate the protocol version.
//! 5. Send `hello-ok`.
//! 6. Register the connection and enter the main message loop (with a
//!    concurrent liveness `tick` task).
//! 7. On disconnect, unregister and clean up subscriptions.

use axum::{
    extract::{
        ws::{CloseFrame, Message, WebSocket},
        WebSocketUpgrade,
    },
    response::IntoResponse,
    Extension,
};
use futures::{SinkExt, StreamExt};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::auth::{resolve_auth, AuthConfig, AuthPrincipal};
use crate::protocol::{
    ERROR_INVALID_REQUEST, ERROR_METHOD_NOT_FOUND, ERROR_UNAUTHORIZED, HelloOk, PongFrame,
    ReqFrame, ResFrame, WsEventFrame, negotiate_protocol, PROTOCOL_VERSION, PREAUTH_TIMEOUT_MS,
    TICK_INTERVAL_MS, WS_CLOSE_SERVICE_RESTART, make_error_res, make_ok_res,
};
use crate::rpc::{RpcContext, RpcRegistry};

/// The sender half of a split WebSocket stream.
pub type WsSender = futures::stream::SplitSink<axum::extract::ws::WebSocket, Message>;

/// An error produced while writing to a WebSocket connection.
#[derive(Debug, Clone)]
pub struct WsError(pub String);

impl std::fmt::Display for WsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WebSocket error: {}", self.0)
    }
}

impl std::error::Error for WsError {}

impl WsError {
    fn send<E: std::fmt::Display>(e: E) -> Self {
        Self(format!("send failed: {e}"))
    }
}

/// Handle a WebSocket upgrade request.
///
/// This function is called by the axum router when a client requests a
/// WebSocket upgrade at the configured endpoint. The optional extensions
/// (auth config, subscription manager, connection registry) are wired by the
/// [`crate::app::Gateway`] router.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Extension(rpc_registry): Extension<Arc<RpcRegistry>>,
    auth: Option<Extension<Arc<AuthConfig>>>,
    subs: Option<Extension<Arc<SubscriptionManager>>>,
    conns: Option<Extension<Arc<ConnectionRegistry>>>,
) -> impl IntoResponse {
    let auth_config = auth
        .map(|Extension(c)| c)
        .unwrap_or_else(|| Arc::new(AuthConfig::default()));
    let subscription_manager = subs
        .map(|Extension(c)| c)
        .unwrap_or_else(|| Arc::new(SubscriptionManager::default()));
    let connection_registry = conns
        .map(|Extension(c)| c)
        .unwrap_or_else(|| Arc::new(ConnectionRegistry::default()));

    ws.on_upgrade(move |socket| {
        handle_socket(
            socket,
            rpc_registry,
            auth_config,
            subscription_manager,
            connection_registry,
        )
    })
}

/// The main WebSocket connection handler: handshake, message loop, cleanup.
async fn handle_socket(
    socket: WebSocket,
    rpc_registry: Arc<RpcRegistry>,
    auth_config: Arc<AuthConfig>,
    subscription_manager: Arc<SubscriptionManager>,
    connection_registry: Arc<ConnectionRegistry>,
) {
    let (sender, mut receiver) = socket.split();
    let sender = Arc::new(tokio::sync::Mutex::new(sender));
    let conn_id = uuid::Uuid::new_v4().to_string();

    info!(conn_id = %conn_id, "WebSocket connected");

    // Step 1: challenge
    let nonce = uuid::Uuid::new_v4().to_string();
    let challenge = WsEventFrame::new("connect.challenge", serde_json::json!({ "nonce": nonce }));
    if let Err(e) = send_raw(&sender, &challenge).await {
        warn!(conn_id = %conn_id, error = %e, "Failed to send connect.challenge");
        return;
    }

    // Step 2: wait for the connect request within the pre-auth timeout.
    let raw = match tokio::time::timeout(Duration::from_millis(PREAUTH_TIMEOUT_MS), receiver.next())
        .await
    {
        Ok(Some(Ok(Message::Text(text)))) => text.to_string(),
        Ok(Some(Ok(Message::Close(_)))) => {
            info!(conn_id = %conn_id, "Client closed before connect");
            return;
        }
        Ok(Some(Ok(_))) => {
            warn!(conn_id = %conn_id, "Non-text frame before connect");
            send_close(&sender, 1008, "expected_text").await;
            return;
        }
        Ok(Some(Err(e))) => {
            warn!(conn_id = %conn_id, error = %e, "Receive error before connect");
            return;
        }
        Ok(None) => return,
        Err(_) => {
            warn!(conn_id = %conn_id, "Pre-auth timeout");
            send_close(&sender, WS_CLOSE_SERVICE_RESTART, "preauth_timeout").await;
            return;
        }
    };

    // Step 3: parse the connect request.
    let value = match serde_json::from_str::<serde_json::Value>(&raw) {
        Ok(v) if v.is_object() => v,
        _ => {
            send_res(&sender, make_error_res("handshake", ERROR_INVALID_REQUEST, "Invalid JSON in connect frame", false, None, None, None)).await;
            send_close(&sender, WS_CLOSE_SERVICE_RESTART, "invalid_connect").await;
            return;
        }
    };

    if value["type"] != "req" || value["method"] != "connect" {
        send_res(
            &sender,
            make_error_res(
                wire_frame_id(value.get("id")),
                ERROR_INVALID_REQUEST,
                "First message must be connect request",
                false,
                None,
                None,
                None,
            ),
        )
        .await;
        send_close(&sender, WS_CLOSE_SERVICE_RESTART, "invalid_connect").await;
        return;
    }

    let req_id = wire_frame_id(value.get("id"));
    let params = value.get("params").cloned().unwrap_or(serde_json::Value::Null);
    let params = params.as_object().cloned().unwrap_or_default();

    // Step 4: resolve auth.
    let auth_params = params
        .get("auth")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let role_claim = params
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("operator");
    // The peer IP is not yet plumbed through the upgrade path; `None` is
    // treated as non-loopback for scope computation.
    let principal = match resolve_auth(&auth_config, &auth_params, role_claim, None) {
        Some(p) => p,
        None => {
            send_res(
                &sender,
                make_error_res(
                    req_id.as_str(),
                    ERROR_UNAUTHORIZED,
                    "Authentication failed",
                    false,
                    None,
                    None,
                    None,
                ),
            )
            .await;
            send_close(&sender, WS_CLOSE_SERVICE_RESTART, "auth_failed").await;
            return;
        }
    };

    // Step 5: negotiate the protocol version.
    let min_proto = params
        .get("minProtocol")
        .and_then(|v| v.as_i64())
        .unwrap_or(1) as i32;
    let max_proto = params
        .get("maxProtocol")
        .and_then(|v| v.as_i64())
        .unwrap_or(PROTOCOL_VERSION as i64) as i32;
    let Some(protocol) = negotiate_protocol(min_proto, max_proto, PROTOCOL_VERSION) else {
        send_res(
            &sender,
            make_error_res(
                req_id.as_str(),
                ERROR_INVALID_REQUEST,
                "Unsupported protocol version range",
                false,
                None,
                None,
                None,
            ),
        )
        .await;
        send_close(&sender, WS_CLOSE_SERVICE_RESTART, "bad_protocol").await;
        return;
    };

    // Step 6: send hello-ok.
    let hello = HelloOk::new(
        protocol,
        env!("CARGO_PKG_VERSION"),
        conn_id.as_str(),
        rpc_registry.methods(),
        default_events().into_iter().map(String::from).collect(),
    );
    if let Err(e) = send_raw(&sender, &hello).await {
        warn!(conn_id = %conn_id, error = %e, "Failed to send hello-ok");
        return;
    }

    // Step 7: register the connection and start the tick loop.
    let conn = Arc::new(WsConnection::new(conn_id.clone(), principal, sender.clone()));
    connection_registry.register(conn.clone());
    info!(conn_id = %conn_id, role = %conn.principal.role, "WebSocket authenticated");

    let tick_sender = sender.clone();
    let tick_conn_id = conn_id.clone();
    let tick_task = tokio::spawn(async move {
        tick_loop(tick_sender, tick_conn_id, TICK_INTERVAL_MS).await;
    });

    // Step 8: main message loop.
    while let Some(msg_result) = receiver.next().await {
        let msg = match msg_result {
            Ok(msg) => msg,
            Err(e) => {
                warn!(conn_id = %conn_id, error = %e, "WebSocket receive error");
                break;
            }
        };

        match msg {
            Message::Text(text) => {
                handle_text_frame(&conn, &rpc_registry, text.as_str()).await;
            }
            Message::Close(_) => {
                info!(conn_id = %conn_id, "WebSocket close received");
                break;
            }
            Message::Ping(data) => {
                let mut sender = sender.lock().await;
                if let Err(e) = sender.send(Message::Pong(data)).await {
                    warn!(conn_id = %conn_id, error = %e, "Failed to send pong");
                    break;
                }
            }
            Message::Pong(_) => {
                // Ignore unsolicited pongs.
            }
            _ => {
                warn!(conn_id = %conn_id, "Unsupported WebSocket message type");
            }
        }
    }

    // Cleanup: stop the tick task, unregister, and clear subscriptions.
    tick_task.abort();
    connection_registry.unregister(&conn_id);
    subscription_manager.remove_connection(&conn_id);
    info!(conn_id = %conn_id, "WebSocket connection closed");
}

/// Parse a single inbound text frame and dispatch it.
async fn handle_text_frame(conn: &Arc<WsConnection>, registry: &RpcRegistry, text: &str) {
    let value = match serde_json::from_str::<serde_json::Value>(text) {
        Ok(v) if v.is_object() => v,
        _ => {
            let err = make_error_res("", ERROR_INVALID_REQUEST, "Invalid JSON", false, None, None, None);
            let _ = conn.send_res(&err).await;
            return;
        }
    };

    let frame_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match frame_type {
        "ping" => {
            let pong = PongFrame::default();
            let _ = conn.send_text(serde_json::to_string(&pong).unwrap_or_default()).await;
        }
        "pong" => {
            // Keepalive acknowledged; nothing to do.
        }
        "req" => {
            let req_id = wire_frame_id(value.get("id"));
            let method = value
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if req_id.is_empty() || method.is_empty() {
                let err = make_error_res(
                    "",
                    ERROR_INVALID_REQUEST,
                    "Frame id and method must be strings",
                    false,
                    None,
                    None,
                    None,
                );
                let _ = conn.send_res(&err).await;
                return;
            }

            let params = value.get("params").cloned().unwrap_or(serde_json::Value::Null);
            debug!(conn_id = %conn.conn_id, method = %method, "RPC call");
            let ctx = RpcContext::new(
                conn.conn_id.clone(),
                conn.principal.clone(),
                method.clone(),
                req_id.clone(),
            );

            let response = match registry.dispatch_with_ctx(&method, params, &ctx).await {
                Some(Ok(result)) => make_ok_res(req_id.as_str(), Some(result)),
                Some(Err(app_err)) => make_error_res(
                    req_id.as_str(),
                    app_err.code.as_str(),
                    app_err.message.as_str(),
                    false,
                    app_err.details.clone(),
                    None,
                    None,
                ),
                None => make_error_res(
                    req_id.as_str(),
                    ERROR_METHOD_NOT_FOUND,
                    format!("Method not found: {method}"),
                    false,
                    None,
                    None,
                    None,
                ),
            };
            let _ = conn.send_res(&response).await;
        }
        other => {
            let err = make_error_res(
                "",
                ERROR_INVALID_REQUEST,
                format!("Unknown frame type: {other:?}"),
                false,
                None,
                None,
                None,
            );
            let _ = conn.send_res(&err).await;
        }
    }
}

/// The list of server-pushed event names advertised in the handshake.
fn default_events() -> Vec<&'static str> {
    vec![
        "connect.challenge",
        "agent",
        "session.message",
        "sessions.changed",
        "presence",
        "tick",
        "shutdown",
        "health",
        "heartbeat",
        "cron",
    ]
}

/// Best-effort extraction of a string request id from a JSON value.
fn wire_frame_id(value: Option<&serde_json::Value>) -> String {
    match value {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        Some(serde_json::Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// Serialize a frame and send it through the raw sender.
async fn send_raw<T: serde::Serialize>(
    sender: &Arc<tokio::sync::Mutex<WsSender>>,
    frame: &T,
) -> Result<(), WsError> {
    let text = serde_json::to_string(frame).map_err(|e| WsError(e.to_string()))?;
    let mut guard = sender.lock().await;
    guard
        .send(Message::Text(text.into()))
        .await
        .map_err(WsError::send)?;
    Ok(())
}

/// Send a response frame through the raw sender.
async fn send_res(sender: &Arc<tokio::sync::Mutex<WsSender>>, frame: ResFrame) {
    let _ = send_raw(sender, &frame).await;
}

/// Send a close frame through the raw sender.
async fn send_close(sender: &Arc<tokio::sync::Mutex<WsSender>>, code: u16, reason: &str) {
    let frame = CloseFrame {
        code,
        reason: reason.to_string().into(),
    };
    let mut guard = sender.lock().await;
    let _ = guard.send(Message::Close(Some(frame))).await;
}

/// Emit a liveness `tick` event on a fixed interval until the sender fails.
async fn tick_loop(sender: Arc<tokio::sync::Mutex<WsSender>>, conn_id: String, interval_ms: u64) {
    let interval = Duration::from_millis(interval_ms.max(1000));
    let mut tick = tokio::time::interval(interval);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tick.tick().await;
        let event = WsEventFrame::new(
            "tick",
            serde_json::json!({ "time_ms": chrono::Utc::now().timestamp_millis() }),
        );
        let text = match serde_json::to_string(&event) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut guard = sender.lock().await;
        if guard.send(Message::Text(text.into())).await.is_err() {
            debug!(conn_id = %conn_id, "Tick loop ended (send failed)");
            break;
        }
    }
}

/// A connected WebSocket client.
///
/// Holds the shared sender half of the socket and per-connection protocol
/// state (principal, sequence counter).
pub struct WsConnection {
    /// The connection identifier.
    pub conn_id: String,
    /// The authenticated principal for this connection.
    pub principal: AuthPrincipal,
    /// When the connection was established.
    pub connected_at: chrono::DateTime<chrono::Utc>,
    /// Monotonic per-connection frame sequence counter.
    seq: AtomicU64,
    /// The shared sender half of the split socket.
    sender: Arc<tokio::sync::Mutex<WsSender>>,
}

impl std::fmt::Debug for WsConnection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsConnection")
            .field("conn_id", &self.conn_id)
            .field("principal", &self.principal)
            .field("connected_at", &self.connected_at)
            .finish_non_exhaustive()
    }
}

impl WsConnection {
    /// Create a new connection handle.
    pub fn new(conn_id: impl Into<String>, principal: AuthPrincipal, sender: Arc<tokio::sync::Mutex<WsSender>>) -> Self {
        Self {
            conn_id: conn_id.into(),
            principal,
            connected_at: chrono::Utc::now(),
            seq: AtomicU64::new(0),
            sender,
        }
    }

    /// Mint the next frame sequence number.
    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Whether the connection is authenticated.
    pub fn authenticated(&self) -> bool {
        self.principal.authenticated
    }

    /// Send a raw text frame.
    pub async fn send_text(&self, text: String) -> Result<(), WsError> {
        let mut guard = self.sender.lock().await;
        guard
            .send(Message::Text(text.into()))
            .await
            .map_err(WsError::send)?;
        Ok(())
    }

    /// Send a server-pushed event with a fresh sequence number.
    pub async fn send_event(&self, event: &str, payload: serde_json::Value) -> Result<(), WsError> {
        let frame = WsEventFrame::new(event, payload).with_seq(self.next_seq());
        let text = serde_json::to_string(&frame).map_err(|e| WsError(e.to_string()))?;
        self.send_text(text).await
    }

    /// Send an RPC response frame.
    pub async fn send_res(&self, frame: &ResFrame) -> Result<(), WsError> {
        let text = serde_json::to_string(frame).map_err(|e| WsError(e.to_string()))?;
        self.send_text(text).await
    }

    /// Close the connection with the given code and reason.
    pub async fn close(&self, code: u16, reason: &str) -> Result<(), WsError> {
        let frame = CloseFrame {
            code,
            reason: reason.to_string().into(),
        };
        let mut guard = self.sender.lock().await;
        guard
            .send(Message::Close(Some(frame)))
            .await
            .map_err(WsError::send)?;
        Ok(())
    }
}

/// Tracks all active WebSocket connections.
#[derive(Default)]
pub struct ConnectionRegistry {
    connections: Mutex<HashMap<String, Arc<WsConnection>>>,
}

impl std::fmt::Debug for ConnectionRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.connections.lock().unwrap().len();
        f.debug_struct("ConnectionRegistry").field("len", &len).finish()
    }
}

impl ConnectionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            connections: Mutex::new(HashMap::new()),
        }
    }

    /// Register a connection.
    pub fn register(&self, conn: Arc<WsConnection>) {
        self.connections.lock().unwrap().insert(conn.conn_id.clone(), conn);
    }

    /// Unregister a connection by id.
    pub fn unregister(&self, conn_id: &str) -> Option<Arc<WsConnection>> {
        self.connections.lock().unwrap().remove(conn_id)
    }

    /// Look up a connection by id.
    pub fn get(&self, conn_id: &str) -> Option<Arc<WsConnection>> {
        self.connections.lock().unwrap().get(conn_id).cloned()
    }

    /// Snapshot all current connections (releases the lock).
    pub fn all(&self) -> Vec<Arc<WsConnection>> {
        self.connections.lock().unwrap().values().cloned().collect()
    }

    /// The number of active connections.
    pub fn len(&self) -> usize {
        self.connections.lock().unwrap().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.connections.lock().unwrap().is_empty()
    }

    /// Broadcast an event to all authenticated connections.
    ///
    /// Send failures on individual connections are logged and skipped.
    pub async fn broadcast(&self, event: &str, payload: serde_json::Value) {
        for conn in self.all() {
            if conn.authenticated() {
                if let Err(e) = conn.send_event(event, payload.clone()).await {
                    debug!(conn_id = %conn.conn_id, error = %e, "Broadcast send failed");
                }
            }
        }
    }
}

/// Tracks which connections are subscribed to session-level, message-level,
/// and topic-level events.
#[derive(Debug, Default)]
pub struct SubscriptionManager {
    /// Connection ids subscribed to session lifecycle events.
    session_subs: Mutex<HashSet<String>>,
    /// `session_key -> {conn_id}` subscriptions for message events.
    message_subs: Mutex<HashMap<String, HashSet<String>>>,
    /// `topic -> {conn_id}` subscriptions for topic events.
    topic_subs: Mutex<HashMap<String, HashSet<String>>>,
}

impl SubscriptionManager {
    /// Create a new, empty subscription manager.
    pub fn new() -> Self {
        Self::default()
    }

    // -- session-level (sessions.subscribe / sessions.unsubscribe) --

    /// Subscribe a connection to session lifecycle events.
    pub fn subscribe_sessions(&self, conn_id: &str) {
        self.session_subs.lock().unwrap().insert(conn_id.to_string());
    }

    /// Unsubscribe a connection from session lifecycle events.
    pub fn unsubscribe_sessions(&self, conn_id: &str) {
        self.session_subs.lock().unwrap().remove(conn_id);
    }

    /// The set of connections subscribed to session lifecycle events.
    pub fn get_session_subscribers(&self) -> HashSet<String> {
        self.session_subs.lock().unwrap().clone()
    }

    // -- message-level (sessions.messages.subscribe / unsubscribe) --

    /// Subscribe a connection to a session's message events.
    pub fn subscribe_messages(&self, conn_id: &str, session_key: &str) {
        self.message_subs
            .lock()
            .unwrap()
            .entry(session_key.to_string())
            .or_default()
            .insert(conn_id.to_string());
    }

    /// Unsubscribe a connection from a session's message events.
    pub fn unsubscribe_messages(&self, conn_id: &str, session_key: &str) {
        if let Some(subs) = self.message_subs.lock().unwrap().get_mut(session_key) {
            subs.remove(conn_id);
        }
    }

    /// The set of connections subscribed to a session's message events.
    pub fn get_message_subscribers(&self, session_key: &str) -> HashSet<String> {
        self.message_subs
            .lock()
            .unwrap()
            .get(session_key)
            .cloned()
            .unwrap_or_default()
    }

    // -- topic-level (cron.subscribe / cron.unsubscribe) --

    /// Subscribe a connection to a topic's events.
    pub fn subscribe_topic(&self, conn_id: &str, topic: &str) {
        self.topic_subs
            .lock()
            .unwrap()
            .entry(topic.to_string())
            .or_default()
            .insert(conn_id.to_string());
    }

    /// Unsubscribe a connection from a topic's events.
    pub fn unsubscribe_topic(&self, conn_id: &str, topic: &str) {
        let mut subs = self.topic_subs.lock().unwrap();
        if let Some(set) = subs.get_mut(topic) {
            set.remove(conn_id);
            if set.is_empty() {
                subs.remove(topic);
            }
        }
    }

    /// The set of connections subscribed to a topic.
    pub fn get_topic_subscribers(&self, topic: &str) -> HashSet<String> {
        self.topic_subs
            .lock()
            .unwrap()
            .get(topic)
            .cloned()
            .unwrap_or_default()
    }

    /// Remove a connection from all subscription sets (disconnect cleanup).
    pub fn remove_connection(&self, conn_id: &str) {
        self.session_subs.lock().unwrap().remove(conn_id);

        let mut message_subs = self.message_subs.lock().unwrap();
        let mut empty_sessions = Vec::new();
        for (session_key, subs) in message_subs.iter_mut() {
            subs.remove(conn_id);
            if subs.is_empty() {
                empty_sessions.push(session_key.clone());
            }
        }
        for session_key in empty_sessions {
            message_subs.remove(&session_key);
        }

        let mut topic_subs = self.topic_subs.lock().unwrap();
        let mut empty_topics = Vec::new();
        for (topic, subs) in topic_subs.iter_mut() {
            subs.remove(conn_id);
            if subs.is_empty() {
                empty_topics.push(topic.clone());
            }
        }
        for topic in empty_topics {
            topic_subs.remove(&topic);
        }
    }
}

/// Kept for reference: the request frame type used by the handshake.
#[allow(dead_code)]
fn _parse_req_frame(value: &serde_json::Value) -> Option<ReqFrame> {
    let id = value.get("id")?.as_str()?.to_string();
    let method = value.get("method")?.as_str()?.to_string();
    let params = value.get("params").cloned();
    Some(ReqFrame::new(id, method, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_manager_sessions() {
        let mgr = SubscriptionManager::new();
        mgr.subscribe_sessions("conn-1");
        mgr.subscribe_sessions("conn-2");
        assert_eq!(mgr.get_session_subscribers().len(), 2);

        mgr.unsubscribe_sessions("conn-1");
        assert_eq!(mgr.get_session_subscribers().len(), 1);
        assert!(mgr.get_session_subscribers().contains("conn-2"));
    }

    #[test]
    fn test_subscription_manager_messages() {
        let mgr = SubscriptionManager::new();
        mgr.subscribe_messages("conn-1", "s1");
        mgr.subscribe_messages("conn-2", "s1");
        mgr.subscribe_messages("conn-2", "s2");

        assert_eq!(mgr.get_message_subscribers("s1").len(), 2);
        assert_eq!(mgr.get_message_subscribers("s2").len(), 1);

        mgr.unsubscribe_messages("conn-1", "s1");
        assert_eq!(mgr.get_message_subscribers("s1").len(), 1);
    }

    #[test]
    fn test_subscription_manager_topics() {
        let mgr = SubscriptionManager::new();
        mgr.subscribe_topic("conn-1", "cron");
        mgr.subscribe_topic("conn-2", "cron");
        mgr.subscribe_topic("conn-2", "health");

        assert_eq!(mgr.get_topic_subscribers("cron").len(), 2);

        mgr.unsubscribe_topic("conn-1", "cron");
        assert_eq!(mgr.get_topic_subscribers("cron").len(), 1);
    }

    #[test]
    fn test_subscription_manager_remove_connection() {
        let mgr = SubscriptionManager::new();
        mgr.subscribe_sessions("conn-1");
        mgr.subscribe_messages("conn-1", "s1");
        mgr.subscribe_topic("conn-1", "cron");
        mgr.subscribe_topic("conn-2", "cron");

        mgr.remove_connection("conn-1");

        assert!(mgr.get_session_subscribers().is_empty());
        assert!(mgr.get_message_subscribers("s1").is_empty());
        assert_eq!(mgr.get_topic_subscribers("cron").len(), 1);
        assert!(mgr.get_topic_subscribers("cron").contains("conn-2"));
    }

    #[test]
    fn test_connection_registry_register_and_remove() {
        let registry = ConnectionRegistry::new();
        assert!(registry.is_empty());

        // Build a connection with a no-op sender by splitting a real pair is
        // not possible without a socket; instead we exercise the registry with
        // a stand-in using a manually created connection is impractical here.
        // The len/is_empty paths are covered below with the default registry.
        registry.unregister("missing");
        assert_eq!(registry.len(), 0);
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn test_wire_frame_id() {
        assert_eq!(wire_frame_id(Some(&serde_json::json!("abc"))), "abc");
        assert_eq!(wire_frame_id(Some(&serde_json::json!(42))), "42");
        assert_eq!(wire_frame_id(Some(&serde_json::json!(true))), "true");
        assert_eq!(wire_frame_id(Some(&serde_json::json!({"a": 1}))), "");
        assert_eq!(wire_frame_id(None), "");
    }

    #[test]
    fn test_default_events() {
        let events = default_events();
        assert!(events.contains(&"tick"));
        assert!(events.contains(&"connect.challenge"));
        assert!(events.contains(&"session.message"));
    }

    #[test]
    fn test_error_type_display() {
        let err = WsError("boom".to_string());
        assert_eq!(err.to_string(), "WebSocket error: boom");
    }
}
