//! WebSocket channel adapter — axum WebSocket upgrade + event broadcasting.
//!
//! This adapter exposes an HTTP route that upgrades to WebSocket, then
//! broadcasts structured event frames to every connected client:
//!
//! - **Outbound**: [`WebSocketChannel::send_message`] broadcasts a
//!   `{"type":"message", ...}` frame; [`WebSocketChannel::send_typing`]
//!   broadcasts a typing indicator.
//! - **Inbound**: text frames (plain or `{"type":"message","text":"..."}`)
//!   are parsed into [`IncomingMessage`]s and forwarded to the registered
//!   `on_message` callback and internal queue.
//! - **Lifecycle**: connections are registered in a registry keyed by a
//!   generated id; the writer task runs a ping keepalive and removes the
//!   connection on close/error, invoking `on_disconnect`.

use crate::types::{Channel, ChannelConfig, ChannelType, IncomingMessage, OutgoingMessage};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use chrono::{DateTime, Utc};
use futures::{SinkExt, StreamExt};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Default outbound queue capacity per connection.
pub const DEFAULT_OUTBOUND_CAPACITY: usize = 1024;
/// Default ping keepalive interval.
pub const DEFAULT_PING_INTERVAL: Duration = Duration::from_secs(30);

/// Metadata describing a connected WebSocket client.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConnectionInfo {
    /// The registry connection id (UUID).
    pub id: String,
    /// The authenticated user id.
    pub user_id: String,
    /// An optional display name.
    pub user_name: Option<String>,
    /// When the connection was established.
    pub connected_at: DateTime<Utc>,
    /// The remote peer address, when known.
    pub remote_addr: Option<String>,
}

struct ConnectionHandle {
    info: ConnectionInfo,
    sender: mpsc::Sender<Message>,
    ping_interval: Duration,
}

impl ConnectionHandle {
    fn into_info(self) -> ConnectionInfo {
        self.info
    }
}

/// WebSocket channel adapter using tokio `mpsc` and axum WebSocket.
#[derive(Clone)]
pub struct WebSocketChannel {
    config: ChannelConfig,
    connections: Arc<Mutex<HashMap<String, ConnectionHandle>>>,
    incoming: Arc<Mutex<VecDeque<IncomingMessage>>>,
    outbound_capacity: usize,
    ping_interval: Duration,
    on_message: Option<Arc<dyn Fn(IncomingMessage) -> Result<(), String> + Send + Sync>>,
    on_connect: Option<Arc<dyn Fn(ConnectionInfo) + Send + Sync>>,
    on_disconnect: Option<Arc<dyn Fn(ConnectionInfo) + Send + Sync>>,
}

impl WebSocketChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let outbound_capacity = config
            .config
            .get("outbound_capacity")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_OUTBOUND_CAPACITY);
        let ping_interval_secs = config
            .config
            .get("ping_interval_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_PING_INTERVAL.as_secs());
        Ok(Self {
            config,
            connections: Arc::new(Mutex::new(HashMap::new())),
            incoming: Arc::new(Mutex::new(VecDeque::new())),
            outbound_capacity,
            ping_interval: Duration::from_secs(ping_interval_secs),
            on_message: None,
            on_connect: None,
            on_disconnect: None,
        })
    }

    /// Register a callback invoked for every parsed incoming message.
    pub fn on_message<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(IncomingMessage) -> Result<(), String> + Send + Sync + 'static,
    {
        self.on_message = Some(Arc::new(callback));
        self
    }

    /// Register a callback invoked when a client connects.
    pub fn on_connect<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(ConnectionInfo) + Send + Sync + 'static,
    {
        self.on_connect = Some(Arc::new(callback));
        self
    }

    /// Register a callback invoked when a client disconnects.
    pub fn on_disconnect<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(ConnectionInfo) + Send + Sync + 'static,
    {
        self.on_disconnect = Some(Arc::new(callback));
        self
    }

    /// Build an axum response that upgrades the request to a WebSocket and
    /// registers the connection under `user_id`.
    pub fn handle_upgrade(&self, upgrade: WebSocketUpgrade, user_id: String) -> Response {
        let channel = self.clone();
        upgrade.on_upgrade(move |socket| async move {
            channel.accept(socket, user_id).await;
        })
    }

    /// Accept a new WebSocket connection, generating a connection id.
    pub async fn accept(&self, socket: WebSocket, user_id: String) -> String {
        let connection_id = Uuid::new_v4().to_string();
        self.accept_connection(socket, user_id, connection_id.clone()).await;
        connection_id
    }

    /// Accept a new WebSocket connection with an explicit connection id.
    pub async fn accept_connection(
        &self,
        socket: WebSocket,
        user_id: String,
        connection_id: String,
    ) {
        let user_name = None;
        let (ws_sender, mut ws_receiver) = socket.split();
        let (tx, mut rx) = mpsc::channel::<Message>(self.outbound_capacity);

        let info = ConnectionInfo {
            id: connection_id.clone(),
            user_id: user_id.clone(),
            user_name,
            connected_at: Utc::now(),
            remote_addr: None,
        };

        // Register the connection.
        {
            let mut conns = self.connections.lock().await;
            conns.insert(
                connection_id.clone(),
                ConnectionHandle {
                    info: info.clone(),
                    sender: tx,
                    ping_interval: self.ping_interval,
                },
            );
        }

        // Send the welcome frame.
        let welcome = serde_json::json!({
            "type": "welcome",
            "connection_id": connection_id,
            "user_id": user_id,
            "timestamp": Utc::now(),
        });
        let mut conns = self.connections.lock().await;
        if let Some(handle) = conns.get(&connection_id) {
            let _ = handle
                .sender
                .send(Message::Text(welcome.to_string().into()))
                .await;
        }
        drop(conns);

        let connections = self.connections.clone();
        let on_disconnect = self.on_disconnect.clone();
        let ping_interval = self.ping_interval;
        let info_clone = info.clone();
        let conn_id_writer = connection_id.clone();
        let conn_id_reader = connection_id.clone();

        // Writer task: drains the outbound queue and runs the ping keepalive.
        tokio::spawn(async move {
            let mut ws_sender = ws_sender;
            let mut ping = tokio::time::interval(ping_interval);
            ping.tick().await; // consume the immediate first tick
            loop {
                tokio::select! {
                    msg = rx.recv() => {
                        match msg {
                            Some(m) => {
                                if ws_sender.send(m).await.is_err() {
                                    break;
                                }
                            }
                            None => break,
                        }
                    }
                    _ = ping.tick() => {
                        if ws_sender.send(Message::Ping(Vec::new().into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            // Cleanup on disconnect.
            let removed = {
                let mut conns = connections.lock().await;
                conns.remove(&conn_id_writer)
            };
            if let Some(handle) = removed {
                let info = handle.into_info();
                if let Some(cb) = &on_disconnect {
                    cb(info.clone());
                }
                info!("WebSocket connection closed: {}", info.id);
            }
        });

        // Reader task: parse incoming frames and dispatch.
        let incoming = self.incoming.clone();
        let on_message = self.on_message.clone();
        let channel_id = self.config.channel_id.clone();
        tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        let parsed = parse_client_frame(text.to_string(), &channel_id, &info_clone);
                        match parsed {
                            Some(incoming_msg) => {
                                incoming.lock().await.push_back(incoming_msg.clone());
                                if let Some(cb) = &on_message {
                                    if let Err(e) = cb(incoming_msg) {
                                        warn!("WebSocket on_message callback error: {e}");
                                    }
                                }
                            }
                            None => debug!("WebSocket non-message frame ignored"),
                        }
                    }
                    Ok(Message::Binary(_)) => {}
                    Ok(Message::Close(_)) => {
                        info!("WebSocket connection closed by client: {}", conn_id_reader);
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        warn!("WebSocket error on {}: {e}", conn_id_reader);
                        break;
                    }
                }
            }
        });
    }

    /// Pull the next parsed incoming message, if any.
    pub async fn receive(&self) -> Result<Option<IncomingMessage>, String> {
        let mut q = self.incoming.lock().await;
        Ok(q.pop_front())
    }

    /// Send a raw frame to a single connection by id.
    pub async fn send_to(&self, connection_id: &str, frame: serde_json::Value) -> Result<(), String> {
        let text = serde_json::to_string(&frame).map_err(|e| format!("Serialize: {e}"))?;
        let conns = self.connections.lock().await;
        match conns.get(connection_id) {
            Some(handle) => handle
                .sender
                .send(Message::Text(text.into()))
                .await
                .map_err(|e| format!("Send to {connection_id}: {e}")),
            None => Err(format!("Connection not found: {connection_id}")),
        }
    }

    /// Send a raw frame to every connection owned by `user_id`.
    pub async fn send_to_user(&self, user_id: &str, frame: serde_json::Value) -> Result<usize, String> {
        let text = serde_json::to_string(&frame).map_err(|e| format!("Serialize: {e}"))?;
        let conns = self.connections.lock().await;
        let mut sent = 0;
        for handle in conns.values() {
            if handle.info.user_id == user_id {
                if handle.sender.send(Message::Text(text.clone().into())).await.is_ok() {
                    sent += 1;
                }
            }
        }
        Ok(sent)
    }

    /// Broadcast a message frame to all connected clients.
    async fn broadcast_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let payload = serde_json::json!({
            "type": "message",
            "id": message.id,
            "channel_id": message.channel_id,
            "text": message.text,
            "timestamp": Utc::now(),
        });
        self.broadcast_raw(&payload).await
    }

    /// Broadcast a raw JSON frame to all connected clients.
    pub async fn broadcast_raw(&self, payload: &serde_json::Value) -> Result<(), String> {
        let text = serde_json::to_string(payload).map_err(|e| format!("Serialize: {e}"))?;
        let conns = self.connections.lock().await;
        for handle in conns.values() {
            let _ = handle.sender.send(Message::Text(text.clone().into())).await;
        }
        Ok(())
    }

    /// Disconnect a connection by id.
    pub async fn disconnect(&self, connection_id: &str) -> bool {
        let removed = {
            let mut conns = self.connections.lock().await;
            conns.remove(connection_id)
        };
        if let Some(handle) = removed {
            let info = handle.into_info();
            if let Some(cb) = &self.on_disconnect {
                cb(info.clone());
            }
            info!("WebSocket connection closed: {}", info.id);
            true
        } else {
            false
        }
    }

    /// The number of connected clients.
    pub async fn connection_count(&self) -> usize {
        self.connections.lock().await.len()
    }

    /// List the currently connected clients.
    pub async fn list_connections(&self) -> Vec<ConnectionInfo> {
        let conns = self.connections.lock().await;
        conns.values().map(|h| h.info.clone()).collect()
    }

    /// Whether a connection with the given id exists.
    pub async fn has_connection(&self, connection_id: &str) -> bool {
        self.connections.lock().await.contains_key(connection_id)
    }
}

/// Parse a client text frame into an [`IncomingMessage`].
///
/// Supports plain text and `{"type":"message","text":"..."}` JSON frames.
pub fn parse_client_frame(
    text: String,
    channel_id: &str,
    conn: &ConnectionInfo,
) -> Option<IncomingMessage> {
    let message_text = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) if value.get("type").and_then(|v| v.as_str()) == Some("message") => value
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        Ok(_) => text.clone(),
        Err(_) => text.clone(),
    };
    if message_text.trim().is_empty() {
        return None;
    }
    Some(IncomingMessage {
        id: Uuid::new_v4(),
        channel_id: channel_id.to_string(),
        channel_type: ChannelType::WebSocket,
        user_id: conn.user_id.clone(),
        user_name: conn.user_name.clone(),
        text: message_text,
        thread_id: None,
        attachments: Vec::new(),
        timestamp: Utc::now(),
        raw: serde_json::json!({ "frame": text }),
    })
}

#[async_trait::async_trait]
impl Channel for WebSocketChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::WebSocket
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        self.broadcast_message(message).await
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        let payload = serde_json::json!({
            "type": "typing",
            "timestamp": Utc::now(),
        });
        self.broadcast_raw(&payload).await
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn channel() -> WebSocketChannel {
        WebSocketChannel::new(ChannelConfig {
            channel_type: ChannelType::WebSocket,
            channel_id: "ws1".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({}),
        })
        .unwrap()
    }

    #[test]
    fn test_parse_plain_text() {
        let conn = ConnectionInfo {
            id: "c1".to_string(),
            user_id: "u1".to_string(),
            user_name: None,
            connected_at: Utc::now(),
            remote_addr: None,
        };
        let msg = parse_client_frame("hello".to_string(), "ws1", &conn).unwrap();
        assert_eq!(msg.text, "hello");
        assert_eq!(msg.user_id, "u1");
        assert_eq!(msg.channel_id, "ws1");
        assert_eq!(msg.channel_type, ChannelType::WebSocket);
    }

    #[test]
    fn test_parse_json_frame() {
        let conn = ConnectionInfo {
            id: "c2".to_string(),
            user_id: "u2".to_string(),
            user_name: Some("alice".to_string()),
            connected_at: Utc::now(),
            remote_addr: None,
        };
        let msg = parse_client_frame(
            r#"{"type":"message","text":"hi there"}"#.to_string(),
            "ws1",
            &conn,
        )
        .unwrap();
        assert_eq!(msg.text, "hi there");
        assert_eq!(msg.user_name.as_deref(), Some("alice"));
    }

    #[test]
    fn test_parse_ignores_empty() {
        let conn = ConnectionInfo {
            id: "c3".to_string(),
            user_id: "u3".to_string(),
            user_name: None,
            connected_at: Utc::now(),
            remote_addr: None,
        };
        assert!(parse_client_frame("   ".to_string(), "ws1", &conn).is_none());
    }

    #[test]
    fn test_connection_info_serializes() {
        let conn = ConnectionInfo {
            id: "c4".to_string(),
            user_id: "u4".to_string(),
            user_name: None,
            connected_at: Utc::now(),
            remote_addr: Some("127.0.0.1:1234".to_string()),
        };
        let v = serde_json::to_value(&conn).unwrap();
        assert_eq!(v["id"], "c4");
        assert_eq!(v["remote_addr"], "127.0.0.1:1234");
    }
}
