//! Gateway-backed [`SessionBridge`] for the inbound MCP server.
//!
//! Implements the `opensquilla_mcp::SessionBridge` trait by wiring to the
//! gateway's real [`SessionStore`] and [`ChatStore`]. This is the Rust
//! equivalent of the Python `OpenSquillaMCPBridge`, which bridged MCP
//! session-operation tools to gateway RPCs over WebSocket. Here the bridge
//! calls the stores directly since the MCP server runs in-process with the
//! gateway.

use opensquilla_mcp::{McpServerError, SessionBridge};
use serde_json::{Value, json};

use crate::chat::ChatStore;
use crate::session_export::ExportFormat;
use crate::sessions::SessionStore;

/// A [`SessionBridge`] backed by the gateway's in-process session and chat
/// stores.
///
/// Construct with [`GatewayMcpBridge::new`] and attach to an [`McpServer`] via
/// `with_bridge(Arc::new(bridge))`.
///
/// [`McpServer`]: opensquilla_mcp::McpServer
#[derive(Clone)]
pub struct GatewayMcpBridge {
    sessions: SessionStore,
    chat: ChatStore,
}

impl GatewayMcpBridge {
    /// Create a new bridge over the given session and chat stores.
    pub fn new(sessions: SessionStore, chat: ChatStore) -> Self {
        Self { sessions, chat }
    }
}

#[async_trait::async_trait]
impl SessionBridge for GatewayMcpBridge {
    async fn conversations_list(&self, limit: Option<u32>) -> Result<Value, McpServerError> {
        let mut sessions = self.sessions.list();
        if let Some(n) = limit {
            sessions.truncate(n as usize);
        }
        let count = sessions.len();
        Ok(json!({
            "sessions": sessions,
            "count": count,
            "limit": limit,
        }))
    }

    async fn session_resolve(&self, key: &str) -> Result<Value, McpServerError> {
        let session = self.sessions.get_str(key);
        Ok(json!({
            "key": key,
            "session": session,
        }))
    }

    async fn messages_read(&self, key: &str, limit: Option<u32>) -> Result<Value, McpServerError> {
        let mut messages = self.sessions.messages(key);
        if let Some(n) = limit {
            messages.truncate(n as usize);
        }
        let count = messages.len();
        Ok(json!({
            "key": key,
            "messages": messages,
            "count": count,
            "limit": limit,
        }))
    }

    async fn messages_send(
        &self,
        key: &str,
        message: &str,
        intent: &str,
    ) -> Result<Value, McpServerError> {
        // Enqueue the turn through the chat store's ingress pipeline so the
        // message is validated, deduplicated, and a worker is spawned to
        // process it.
        let turn = self
            .chat
            .enqueue_turn(key, message, Vec::new(), None)
            .await
            .map_err(|e| McpServerError::SessionOp(e.to_string()))?;

        self.chat.spawn_worker_if_needed(key).await;

        Ok(json!({
            "key": key,
            "sent": true,
            "message": message,
            "intent": intent,
            "turn": turn,
            "status": "queued",
        }))
    }

    async fn events_wait(
        &self,
        key: &str,
        since_stream_seq: Option<i64>,
        timeout_ms: u64,
        max_events: u32,
        terminal_only: bool,
    ) -> Result<Value, McpServerError> {
        // TODO(parity): The Python bridge subscribes via
        // `sessions.messages.subscribe` with a `since_stream_seq` cursor and
        // receives framed events carrying `stream_seq` and terminal event
        // names (`session.event.done`, `task.cancelled`, etc.). The Rust
        // gateway's `SessionEventBroadcaster` is a simple broadcast channel
        // with no replay/seq cursor and no terminal-event taxonomy. We poll
        // `ChatStore::next_event` up to `max_events` times, but
        // `since_stream_seq` and `terminal_only` are not honored.
        let max_events = max_events.max(1);
        let mut events: Vec<Value> = Vec::new();
        let mut timed_out = false;

        let per_event_timeout = std::cmp::max(timeout_ms / max_events as u64, 50);

        while events.len() < max_events as usize {
            match self.chat.next_event(key, per_event_timeout).await {
                Ok(Some(event)) => {
                    events.push(serde_json::to_value(&event).unwrap_or_else(|_| {
                        json!({
                            "kind": format!("{:?}", event.kind),
                            "session_id": event.session_id,
                        })
                    }));
                }
                Ok(None) => {
                    timed_out = true;
                    break;
                }
                Err(e) => {
                    return Err(McpServerError::SessionOp(e.to_string()));
                }
            }
        }

        if events.is_empty() {
            timed_out = true;
        }

        Ok(json!({
            "key": key,
            "events": events,
            "since": since_stream_seq,
            "timeout_ms": timeout_ms,
            "max_events": max_events,
            "terminal_only": terminal_only,
            "timed_out": timed_out,
        }))
    }

    async fn transcript_jsonl(
        &self,
        key: &str,
        limit: Option<u32>,
    ) -> Result<String, McpServerError> {
        // Use the session store's export in JSONL format. The export produces
        // one JSON object per message line. When a limit is given, we export
        // the full transcript then truncate the output lines to match.
        let export = self
            .sessions
            .export(key, ExportFormat::Jsonl)
            .map_err(|e| McpServerError::SessionOp(e.to_string()))?;

        let content = export
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if let Some(n) = limit {
            let truncated: String = content
                .lines()
                .take(n as usize)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(truncated)
        } else {
            Ok(content)
        }
    }
}
