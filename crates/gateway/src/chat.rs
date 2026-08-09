//! Chat RPC handlers.
//!
//! Provides RPC handlers for sending messages (through the turn ingress
//! pipeline), retrieving paginated chat history, searching transcripts,
//! editing/deleting messages, managing attachments, and subscribing to turn
//! events. The store wraps a [`TurnIngress`] so chat sends flow through the
//! same validation/deduplication/queueing pipeline as engine-originated turns.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::attachments::{AttachmentMeta, AttachmentStore, AttachmentUpload, store_attachment};
use crate::rpc::{RpcRegistry, rpc_handler};
use crate::session_events::{SessionEvent, SessionEventBroadcaster, SessionEventKind};
use crate::session_search::{
    IndexedMessage, SessionSearchIndex, SessionSearchOptions, SessionSearchResult,
};
use crate::session_services::SessionServices;
use crate::turn_ingress::{InboundTurn, TurnIngress};

/// Maximum number of messages returned per history page.
pub const DEFAULT_PAGE_SIZE: usize = 50;

/// A chat message record returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageResponse {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl ChatMessageResponse {
    /// Build a message view from a role string and content.
    fn new(session_id: &str, role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: role.into(),
            content: content.into(),
            timestamp: Utc::now(),
            model: None,
        }
    }
}

/// A chat store with turn ingress, search indexing, and attachment support.
///
/// All state is shared behind `Arc`s so clones are cheap. Sending a message
/// enqueues an [`InboundTurn`] through the shared [`TurnIngress`], records the
/// user message, and lazily spawns a per-session worker that drains the queue.
#[derive(Clone)]
pub struct ChatStore {
    conversations: Arc<Mutex<HashMap<String, Vec<ChatMessageResponse>>>>,
    turns: Arc<Mutex<HashMap<String, Vec<InboundTurn>>>>,
    broadcaster: SessionEventBroadcaster,
    ingress: TurnIngress,
    search_index: SessionSearchIndex,
    attachments: AttachmentStore,
}

impl Default for ChatStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatStore {
    /// Create a new chat store with its own event broadcaster and turn ingress.
    pub fn new() -> Self {
        let broadcaster = SessionEventBroadcaster::new();
        let services = SessionServices::builder().build();
        let ingress = TurnIngress::new(services, broadcaster.clone());
        Self {
            conversations: Arc::new(Mutex::new(HashMap::new())),
            turns: Arc::new(Mutex::new(HashMap::new())),
            broadcaster,
            ingress,
            search_index: SessionSearchIndex::new(),
            attachments: AttachmentStore::new(),
        }
    }

    /// Return the event broadcaster (for external stream subscribers).
    pub fn broadcaster(&self) -> SessionEventBroadcaster {
        self.broadcaster.clone()
    }

    /// Return the turn ingress pipeline.
    pub fn ingress(&self) -> TurnIngress {
        self.ingress.clone()
    }

    // -----------------------------------------------------------------------
    // Messages
    // -----------------------------------------------------------------------

    /// Append a message to a session's history and index it for search.
    pub fn add_message(&self, session_id: &str, msg: ChatMessageResponse) {
        {
            let mut conversations = self.conversations.lock();
            conversations
                .entry(session_id.to_string())
                .or_default()
                .push(msg.clone());
        }
        self.search_index.index(IndexedMessage {
            session_id: session_id.to_string(),
            message_id: msg.id.clone(),
            role: msg.role.clone(),
            content: msg.content.clone(),
            model: msg.model.clone(),
            timestamp: msg.timestamp,
        });
        self.broadcaster
            .publish_simple(SessionEventKind::MessageAdded, session_id);
    }

    /// Get a page of messages for a session, newest-last within the page.
    pub fn get_history(
        &self,
        session_id: &str,
        limit: usize,
        offset: usize,
    ) -> Vec<ChatMessageResponse> {
        let conversations = self.conversations.lock();
        conversations
            .get(session_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect()
    }

    /// Return the total number of messages for a session.
    pub fn history_len(&self, session_id: &str) -> usize {
        self.conversations
            .lock()
            .get(session_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Get a single message by id.
    pub fn get_message(&self, session_id: &str, message_id: &str) -> Option<ChatMessageResponse> {
        self.conversations
            .lock()
            .get(session_id)?
            .iter()
            .find(|m| m.id == message_id)
            .cloned()
    }

    /// Replace a message's content in place.
    pub fn update_message(
        &self,
        session_id: &str,
        message_id: &str,
        content: String,
    ) -> Option<ChatMessageResponse> {
        let mut conversations = self.conversations.lock();
        let list = conversations.get_mut(session_id)?;
        let msg = list.iter_mut().find(|m| m.id == message_id)?;
        msg.content = content.clone();
        msg.timestamp = Utc::now();
        let updated = msg.clone();
        drop(conversations);
        // Re-index the updated content so search stays consistent.
        self.search_index.index(IndexedMessage {
            session_id: session_id.to_string(),
            message_id: message_id.to_string(),
            role: updated.role.clone(),
            content: updated.content.clone(),
            model: updated.model.clone(),
            timestamp: updated.timestamp,
        });
        Some(updated)
    }

    /// Delete a message by id, removing it from search too.
    pub fn delete_message(&self, session_id: &str, message_id: &str) -> bool {
        let mut conversations = self.conversations.lock();
        let list = conversations.get_mut(session_id);
        let removed = match list {
            Some(list) => {
                let len_before = list.len();
                list.retain(|m| m.id != message_id);
                list.len() < len_before
            }
            None => false,
        };
        if removed {
            self.search_index.remove(session_id, message_id);
        }
        removed
    }

    /// Clear all messages for a session.
    pub fn clear(&self, session_id: &str) -> usize {
        let mut conversations = self.conversations.lock();
        let count = conversations.get(session_id).map(|v| v.len()).unwrap_or(0);
        conversations.remove(session_id);
        drop(conversations);
        self.search_index.remove_session(session_id);
        count
    }

    // -----------------------------------------------------------------------
    // Turn ingress
    // -----------------------------------------------------------------------

    /// Validate and enqueue a turn through the ingress pipeline, then record it.
    pub async fn enqueue_turn(
        &self,
        session_id: &str,
        message: &str,
        attachment_ids: Vec<String>,
        turn_id: Option<Uuid>,
    ) -> Result<InboundTurn, AppError> {
        let mut turn = InboundTurn::new(session_id, message, turn_id);
        turn.attachment_ids = attachment_ids;
        let recorded = self.ingress.enqueue(turn).await?;
        // Only record unique turns; retries with the same turn id are dropped.
        if !matches!(recorded.status, crate::turn_ingress::TurnStatus::Duplicate) {
            let mut turns = self.turns.lock();
            turns
                .entry(session_id.to_string())
                .or_default()
                .push(recorded.clone());
        }
        Ok(recorded)
    }

    /// Spawn the per-session worker if one is not already running.
    ///
    /// The worker drains the session's queue one turn at a time, appending an
    /// assistant acknowledgement for each processed turn. A single worker per
    /// session is created lazily on first send; later sends only enqueue.
    pub async fn spawn_worker_if_needed(&self, session_id: &str) {
        if self.ingress.is_session_active(session_id) {
            return;
        }
        let store = self.clone();
        let ingress = self.ingress.clone();
        let _ = ingress
            .spawn_worker(session_id, move |turn, _services| {
                let store = store.clone();
                async move {
                    // Simulate the engine answering. A real deployment would
                    // forward the turn to the provider runtime instead.
                    let content = format!("[agent] {}", turn.message);
                    store.add_message(
                        &turn.session_id,
                        ChatMessageResponse::new(&turn.session_id, "assistant", content),
                    );
                    Ok(())
                }
            })
            .await;
    }

    /// Return all recorded turns for a session, most recent first.
    pub fn list_turns(&self, session_id: &str) -> Vec<InboundTurn> {
        let mut turns = self
            .turns
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        turns.sort_by_key(|b| std::cmp::Reverse(b.received_at));
        turns
    }

    /// Cancel the active (pending/queued/running) turn for a session by
    /// marking it failed with an abort reason. Returns the aborted `turn_id`,
    /// or `None` if no cancellable turn was found.
    pub fn abort_active_turn(&self, session_id: &str) -> Option<Uuid> {
        let mut turns = self.turns.lock();
        let list = turns.get_mut(session_id)?;
        let target = list.iter_mut().find(|t| {
            matches!(
                t.status,
                crate::turn_ingress::TurnStatus::Pending
                    | crate::turn_ingress::TurnStatus::Queued
                    | crate::turn_ingress::TurnStatus::Running
            )
        })?;
        target.status = crate::turn_ingress::TurnStatus::Failed;
        target.error = Some("aborted by user".to_string());
        Some(target.turn_id)
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Search the transcript index, optionally scoped to a session.
    pub fn search(
        &self,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> SessionSearchResult {
        let result = self.search_index.search(
            query,
            SessionSearchOptions {
                limit,
                session_id: None,
            },
        );
        if let Some(sid) = session_id {
            let mut filtered = result;
            filtered.hits.retain(|h| h.session_id == sid);
            filtered.total = filtered.hits.len();
            filtered
        } else {
            result
        }
    }

    // -----------------------------------------------------------------------
    // Attachments
    // -----------------------------------------------------------------------

    /// Persist an uploaded attachment for a session.
    pub fn upload_attachment(
        &self,
        session_id: &str,
        upload: AttachmentUpload,
    ) -> Result<AttachmentMeta, AppError> {
        let dir = default_attachment_dir();
        store_attachment(&self.attachments, session_id, upload, &dir)
    }

    /// List attachments for a session.
    pub fn attachments_for(&self, session_id: &str) -> Vec<AttachmentMeta> {
        self.attachments.list_for_session(session_id)
    }

    /// Delete an attachment by id.
    pub fn delete_attachment(&self, attachment_id: &str) -> Result<bool, AppError> {
        self.attachments.delete(attachment_id)
    }

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------

    /// Wait for the next event for a session, or `None` on timeout.
    pub async fn next_event(
        &self,
        session_id: &str,
        timeout_ms: u64,
    ) -> Result<Option<SessionEvent>, AppError> {
        let mut rx = self.broadcaster.subscribe();
        let session_id = session_id.to_string();
        let wait = tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), async {
            loop {
                let event = rx
                    .recv()
                    .await
                    .map_err(|e| AppError::internal(e.to_string()))?;
                if event.session_id == session_id {
                    return Ok::<_, AppError>(event);
                }
            }
        });
        match wait.await {
            Ok(Ok(event)) => Ok(Some(event)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None),
        }
    }
}

/// Default directory used for RPC attachment uploads.
fn default_attachment_dir() -> PathBuf {
    std::env::temp_dir().join("opensquilla-attachments")
}

/// Parse a session id from a `session_id` parameter.
///
/// Chat history is keyed by an arbitrary session string (not necessarily a
/// UUID), so no format validation is applied.
fn parse_session_id(params: &serde_json::Value) -> Result<String, AppError> {
    params
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))
        .map(|s| s.to_string())
}

/// Register chat RPC handlers on the given registry.
pub fn register_chat_handlers(registry: &mut RpcRegistry, chat_store: ChatStore) {
    let chat_store = Arc::new(chat_store);

    // chat.send — submit a message through the turn ingress pipeline
    registry.register(rpc_handler("chat.send", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let message = params
                    .get("message")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'message' parameter"))?;
                let attachment_ids: Vec<String> = params
                    .get("attachment_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|a| a.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let turn_id = params
                    .get("turn_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());

                let session_id_str = session_id.to_string();
                let recorded = store
                    .enqueue_turn(&session_id_str, message, attachment_ids, turn_id)
                    .await?;

                // Record the user message locally and start the session worker
                // on first send so the pipeline is exercised end to end.
                if !matches!(recorded.status, crate::turn_ingress::TurnStatus::Duplicate) {
                    let user_msg = ChatMessageResponse::new(&session_id_str, "user", message);
                    store.add_message(&session_id_str, user_msg.clone());
                    store.spawn_worker_if_needed(&session_id_str).await;
                    return Ok(serde_json::json!({
                        "turn": recorded,
                        "message": user_msg,
                        "status": "queued",
                    }));
                }

                Ok(serde_json::json!({
                    "turn": recorded,
                    "status": "duplicate",
                }))
            }
        }
    }));

    // chat.history — paginated conversation history
    registry.register(rpc_handler("chat.history", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let limit = params
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(DEFAULT_PAGE_SIZE as u64) as usize;
                let offset = params.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let session_id_str = session_id.to_string();
                let total = store.history_len(&session_id_str);
                let messages = store.get_history(&session_id_str, limit, offset);
                let has_more = offset + messages.len() < total;
                Ok(serde_json::json!({
                    "messages": messages,
                    "count": messages.len(),
                    "total": total,
                    "has_more": has_more,
                }))
            }
        }
    }));

    // chat.clear — clear conversation history
    registry.register(rpc_handler("chat.clear", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let cleared = store.clear(&session_id.to_string());
                Ok(serde_json::json!({"cleared": cleared}))
            }
        }
    }));

    // chat.message.get — fetch a single message by id
    registry.register(rpc_handler("chat.message.get", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let message_id = params
                    .get("message_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'message_id' parameter"))?;
                match store.get_message(&session_id.to_string(), message_id) {
                    Some(msg) => {
                        Ok(serde_json::to_value(msg)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!(
                        "Message '{message_id}' not found"
                    ))),
                }
            }
        }
    }));

    // chat.message.update — edit a message's content
    registry.register(rpc_handler("chat.message.update", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let message_id = params
                    .get("message_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'message_id' parameter"))?;
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'content' parameter"))?;
                match store.update_message(&session_id.to_string(), message_id, content.to_string())
                {
                    Some(msg) => {
                        Ok(serde_json::to_value(msg)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!(
                        "Message '{message_id}' not found"
                    ))),
                }
            }
        }
    }));

    // chat.message.delete — delete a message by id
    registry.register(rpc_handler("chat.message.delete", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let message_id = params
                    .get("message_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'message_id' parameter"))?;
                if store.delete_message(&session_id.to_string(), message_id) {
                    Ok(serde_json::json!({"deleted": true, "message_id": message_id}))
                } else {
                    Err(AppError::not_found(format!(
                        "Message '{message_id}' not found"
                    )))
                }
            }
        }
    }));

    // chat.search — full-text search over chat transcripts
    registry.register(rpc_handler("chat.search", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'query' parameter"))?;
                let session_id = params.get("session_id").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let result = store.search(query, session_id, limit);
                serde_json::to_value(result).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // chat.turns — list turns recorded for a session
    registry.register(rpc_handler("chat.turns", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let turns = store.list_turns(&session_id.to_string());
                Ok(serde_json::json!({
                    "turns": turns,
                    "count": turns.len(),
                }))
            }
        }
    }));

    // chat.attachments.upload — store a base64-encoded attachment
    registry.register(rpc_handler("chat.attachments.upload", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let filename = params
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'filename' parameter"))?
                    .to_string();
                let content_type = params
                    .get("content_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'content_type' parameter"))?
                    .to_string();
                let data = params
                    .get("data")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'data' parameter"))?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .map_err(|_| AppError::bad_request("Invalid base64 attachment data"))?;
                let upload = AttachmentUpload {
                    filename,
                    content_type,
                    bytes,
                };
                let meta = store.upload_attachment(&session_id.to_string(), upload)?;
                serde_json::to_value(meta).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // chat.attachments.list — list attachments for a session
    registry.register(rpc_handler("chat.attachments.list", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let attachments = store.attachments_for(&session_id.to_string());
                Ok(serde_json::json!({
                    "attachments": attachments,
                    "count": attachments.len(),
                }))
            }
        }
    }));

    // chat.attachments.delete — delete an attachment by id
    registry.register(rpc_handler("chat.attachments.delete", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let attachment_id = params
                    .get("attachment_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'attachment_id' parameter"))?;
                if store.delete_attachment(attachment_id)? {
                    Ok(serde_json::json!({"deleted": true, "attachment_id": attachment_id}))
                } else {
                    Err(AppError::not_found(format!(
                        "Attachment '{attachment_id}' not found"
                    )))
                }
            }
        }
    }));

    // chat.stream — wait for the next event for a session (short poll)
    registry.register(rpc_handler("chat.stream", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                let timeout_ms = params
                    .get("timeout_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(200);
                match store
                    .next_event(&session_id.to_string(), timeout_ms)
                    .await?
                {
                    Some(event) => Ok(serde_json::to_value(event)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Ok(serde_json::json!({"timeout": true})),
                }
            }
        }
    }));

    // chat.abort — cancel the active turn for a session
    registry.register(rpc_handler("chat.abort", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_session_id(&params)?;
                match store.abort_active_turn(&session_id.to_string()) {
                    Some(turn_id) => Ok(serde_json::json!({
                        "ok": true,
                        "session_key": session_id,
                        "aborted": turn_id.to_string(),
                    })),
                    None => Ok(serde_json::json!({
                        "ok": false,
                        "session_key": session_id,
                        "aborted": null,
                        "message": "No active turn to abort",
                    })),
                }
            }
        }
    }));

    // chat.inject — inject a message directly into a session's history
    registry.register(rpc_handler("chat.inject", {
        let store = chat_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_key = params
                    .get("sessionKey")
                    .or_else(|| params.get("session_id"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'sessionKey' parameter"))?;
                let role = params
                    .get("role")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'role' parameter"))?;
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'content' parameter"))?;
                let msg = ChatMessageResponse::new(session_key, role, content);
                store.add_message(session_key, msg.clone());
                Ok(serde_json::json!({
                    "ok": true,
                    "session_key": session_key,
                    "message": msg,
                }))
            }
        }
    }));

    // chat.clarify_submit — not implemented (engine has no clarification flow)
    registry.register(rpc_handler("chat.clarify_submit", {
        move |_params| async move {
            Err(AppError::new(
                "RPC_UNAVAILABLE",
                "chat.clarify_submit requires a clarification engine that is not available",
            )
            .with_status(501))
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::SessionId;

    #[tokio::test]
    async fn test_chat_send_and_history() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let session_id = SessionId::new();

        // Send a message
        let params = serde_json::json!({
            "session_id": session_id.to_string(),
            "message": "Hello, world!",
        });
        let result = registry.dispatch("chat.send", params).await;
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["status"], "queued");
        assert_eq!(resp["message"]["role"], "user");

        // Wait for the worker to append the assistant message.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        // Get history
        let params = serde_json::json!({"session_id": session_id.to_string()});
        let result = registry.dispatch("chat.history", params).await;
        let resp = result.unwrap().unwrap();
        let messages: Vec<ChatMessageResponse> =
            serde_json::from_value(resp["messages"].clone()).unwrap();
        assert_eq!(messages.len(), 2); // user + assistant
        assert_eq!(messages[0].role, "user");
        assert_eq!(resp["total"], 2);
    }

    #[tokio::test]
    async fn test_chat_history_pagination() {
        let store = ChatStore::new();
        for i in 0..5 {
            store.add_message(
                "s1",
                ChatMessageResponse::new("s1", "user", format!("msg {i}")),
            );
        }
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store);

        let r = registry
            .dispatch(
                "chat.history",
                serde_json::json!({"session_id": "s1", "limit": 2, "offset": 0}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 2);
        assert_eq!(resp["total"], 5);
        assert_eq!(resp["has_more"], true);

        let r = registry
            .dispatch(
                "chat.history",
                serde_json::json!({"session_id": "s1", "limit": 2, "offset": 4}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
        assert_eq!(resp["has_more"], false);
    }

    #[tokio::test]
    async fn test_chat_message_edit_delete() {
        let store = ChatStore::new();
        let msg = ChatMessageResponse::new("s1", "user", "original");
        store.add_message("s1", msg.clone());

        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch(
                "chat.message.update",
                serde_json::json!({"session_id": "s1", "message_id": msg.id, "content": "edited"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["content"], "edited");

        let r = registry
            .dispatch(
                "chat.message.delete",
                serde_json::json!({"session_id": "s1", "message_id": msg.id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["deleted"], true);

        assert_eq!(store.history_len("s1"), 0);
    }

    #[tokio::test]
    async fn test_chat_search() {
        let store = ChatStore::new();
        store.add_message(
            "s1",
            ChatMessageResponse::new("s1", "user", "How do I configure the API key?"),
        );
        store.add_message(
            "s1",
            ChatMessageResponse::new("s1", "assistant", "It goes in the TOML file."),
        );
        store.add_message(
            "s2",
            ChatMessageResponse::new("s2", "user", "What is the weather today?"),
        );

        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store);

        let r = registry
            .dispatch("chat.search", serde_json::json!({"query": "configure"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["total"].as_u64().unwrap() >= 1);

        // Scoped search should only return s1 hits.
        let r = registry
            .dispatch(
                "chat.search",
                serde_json::json!({"query": "weather", "session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["total"], 0);
    }

    #[tokio::test]
    async fn test_chat_turns_and_duplicate() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let session_id = SessionId::new();
        let turn_id = Uuid::new_v4().to_string();

        let params = serde_json::json!({
            "session_id": session_id.to_string(),
            "message": "first",
            "turn_id": turn_id,
        });
        let r = registry.dispatch("chat.send", params).await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "queued");

        // Same turn_id retried → duplicate, no user message added.
        let params = serde_json::json!({
            "session_id": session_id.to_string(),
            "message": "first",
            "turn_id": turn_id,
        });
        let r = registry.dispatch("chat.send", params).await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "duplicate");

        let r = registry
            .dispatch(
                "chat.turns",
                serde_json::json!({"session_id": session_id.to_string()}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_chat_attachment_upload() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let data = base64::engine::general_purpose::STANDARD.encode(b"attachment data");
        let r = registry
            .dispatch(
                "chat.attachments.upload",
                serde_json::json!({
                    "session_id": "s1",
                    "filename": "file.txt",
                    "content_type": "text/plain",
                    "data": data,
                }),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["session_id"], "s1");
        assert_eq!(resp["filename"], "file.txt");

        let r = registry
            .dispatch(
                "chat.attachments.list",
                serde_json::json!({"session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_chat_stream_publishes_event() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        // Publish an event concurrently; chat.stream should observe it.
        let broadcaster = store.broadcaster();
        let publish = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            broadcaster.publish_simple(SessionEventKind::TurnStarted, "s1");
        });

        let r = registry
            .dispatch(
                "chat.stream",
                serde_json::json!({"session_id": "s1", "timeout_ms": 500}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["session_id"], "s1");
        assert_eq!(resp["kind"], "turn_started");
        publish.await.unwrap();
    }

    #[tokio::test]
    async fn test_chat_stream_timeout() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store);

        let r = registry
            .dispatch(
                "chat.stream",
                serde_json::json!({"session_id": "s1", "timeout_ms": 30}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["timeout"], true);
    }

    #[tokio::test]
    async fn test_chat_abort_cancels_active_turn() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let session_id = SessionId::new().to_string();
        store
            .enqueue_turn(&session_id, "hello", vec![], None)
            .await
            .unwrap();

        let r = registry
            .dispatch(
                "chat.abort",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["ok"], true);
        assert!(resp["aborted"].as_str().is_some());

        // No active turn left → ok=false.
        let r = registry
            .dispatch(
                "chat.abort",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["ok"], false);
    }

    #[tokio::test]
    async fn test_chat_inject_adds_message() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch(
                "chat.inject",
                serde_json::json!({
                    "sessionKey": "s1",
                    "role": "assistant",
                    "content": "injected",
                }),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["ok"], true);
        assert_eq!(resp["message"]["role"], "assistant");
        assert_eq!(store.history_len("s1"), 1);
    }

    #[tokio::test]
    async fn test_chat_clarify_submit_unavailable() {
        let store = ChatStore::new();
        let mut registry = RpcRegistry::new();
        register_chat_handlers(&mut registry, store);

        let r = registry
            .dispatch("chat.clarify_submit", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap_err();
        assert_eq!(resp.code, "RPC_UNAVAILABLE");
    }
}
