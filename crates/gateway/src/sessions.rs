//! Session lifecycle RPC handlers.
//!
//! Provides RPC handlers for creating, listing, retrieving, updating,
//! archiving, exporting, and deleting sessions, plus turn management,
//! compaction, and attachment operations. The store is wired to the session
//! lifecycle state machine, the event broadcaster, the search index, and the
//! disk archive/export layers so the handlers exercise the full session
//! subsystem rather than a bare in-memory list.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use opensquilla_core::types::{Message, MessageRole, SessionId};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::attachments::{AttachmentMeta, AttachmentStore, AttachmentUpload, store_attachment};
use crate::rpc::{RpcRegistry, rpc_handler};
use crate::session_archive::SessionArchiver;
use crate::session_events::{SessionEventBroadcaster, SessionEventKind};
use crate::session_export::{ExportFormat, ExportedMessage, SessionExporter};
use crate::session_lifecycle::{
    LifecycleRecord, LifecycleTransition, SessionLifecycleManager, SessionState,
};
use crate::session_search::{IndexedMessage, SessionSearchIndex, SessionSearchResult};

/// A single recorded turn in a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    /// Unique turn id.
    pub turn_id: Uuid,
    /// The session this turn belongs to.
    pub session_id: String,
    /// One of `reserved`, `queued`, `running`, `completed`, `cancelled`, `failed`.
    pub status: String,
    /// The user message text, if one was provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// When the turn was created.
    pub started_at: DateTime<Utc>,
    /// When the turn finished (completed/cancelled/failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    /// Error detail if the turn failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A record of a session compaction run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionRecord {
    /// Unique compaction id.
    pub id: Uuid,
    /// The session being compacted.
    pub session_id: String,
    /// One of `requested`, `running`, `completed`, `failed`.
    pub status: String,
    /// Optional caller-supplied reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// When compaction was requested.
    pub requested_at: DateTime<Utc>,
    /// When compaction finished.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    /// Resulting summary text (filled on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Estimated input tokens consumed (filled on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    /// Estimated output tokens produced (filled on completion).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    /// Error detail if compaction failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A session entry stored in the `SessionStore`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: SessionId,
    pub title: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Current lifecycle state (see [`SessionState`]).
    pub state: String,
    /// Number of messages currently in the transcript.
    #[serde(default)]
    pub message_count: u64,
    /// Number of turns that have been started for this session.
    #[serde(default)]
    pub turn_count: u64,
    /// Arbitrary caller-supplied metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SessionEntry {
    /// Parse the lifecycle state field.
    pub fn state_enum(&self) -> Result<SessionState, AppError> {
        SessionState::parse(&self.state)
    }
}

/// A session store with lifecycle, turn, compaction, attachment, search, and
/// archive/export support.
///
/// All state is shared behind `Arc`s so clones are cheap and every clone sees
/// the same data. The lifecycle manager drives the Created → Active → Paused →
/// Archived → Deleted state machine and publishes events on the broadcaster.
#[derive(Clone)]
pub struct SessionStore {
    sessions: Arc<Mutex<HashMap<String, SessionEntry>>>,
    transcripts: Arc<Mutex<HashMap<String, Vec<Message>>>>,
    turns: Arc<Mutex<HashMap<String, Vec<TurnRecord>>>>,
    compactions: Arc<Mutex<HashMap<String, Vec<CompactionRecord>>>>,
    lifecycle: SessionLifecycleManager,
    broadcaster: SessionEventBroadcaster,
    attachments: AttachmentStore,
    archiver: Option<SessionArchiver>,
    search_index: SessionSearchIndex,
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionStore {
    /// Create a new empty session store.
    pub fn new() -> Self {
        let broadcaster = SessionEventBroadcaster::new();
        let lifecycle = SessionLifecycleManager::new(broadcaster.clone());
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            transcripts: Arc::new(Mutex::new(HashMap::new())),
            turns: Arc::new(Mutex::new(HashMap::new())),
            compactions: Arc::new(Mutex::new(HashMap::new())),
            lifecycle,
            broadcaster,
            attachments: AttachmentStore::new(),
            archiver: None,
            search_index: SessionSearchIndex::new(),
        }
    }

    /// Attach a disk archiver rooted at `dir`. Subsequent archive calls write
    /// a JSON snapshot for the session.
    pub fn with_archive_dir(mut self, dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        self.archiver = Some(SessionArchiver::new(dir)?);
        Ok(self)
    }

    /// Return the event broadcaster (for external subscribers).
    pub fn broadcaster(&self) -> SessionEventBroadcaster {
        self.broadcaster.clone()
    }

    /// Return the search index (for cross-session transcript search).
    pub fn search_index(&self) -> SessionSearchIndex {
        self.search_index.clone()
    }

    /// Return the attachment store.
    pub fn attachments(&self) -> AttachmentStore {
        self.attachments.clone()
    }

    // -----------------------------------------------------------------------
    // Session metadata CRUD
    // -----------------------------------------------------------------------

    /// Create a new session in the `Created` lifecycle state.
    pub fn create(&self, title: String, model: String) -> SessionEntry {
        let now = Utc::now();
        let entry = SessionEntry {
            id: SessionId::new(),
            title,
            model,
            created_at: now,
            updated_at: now,
            state: SessionState::Created.as_str().to_string(),
            message_count: 0,
            turn_count: 0,
            metadata: HashMap::new(),
        };
        let id_str = entry.id.to_string();
        {
            let mut sessions = self.sessions.lock();
            sessions.insert(id_str.clone(), entry.clone());
        }
        // Register in the lifecycle manager (publishes a Created event).
        let _ = self.lifecycle.register(id_str);
        entry
    }

    /// List all sessions, optionally filtered by lifecycle state.
    pub fn list(&self) -> Vec<SessionEntry> {
        let mut entries: Vec<SessionEntry> = self.sessions.lock().values().cloned().collect();
        entries.sort_by_key(|b| std::cmp::Reverse(b.updated_at));
        entries
    }

    /// List sessions matching a lifecycle state filter.
    pub fn list_by_state(&self, state: Option<&SessionState>) -> Vec<SessionEntry> {
        let entries = self.list();
        match state {
            Some(s) => entries
                .into_iter()
                .filter(|e| e.state == s.as_str())
                .collect(),
            None => entries,
        }
    }

    /// Get a session by ID.
    pub fn get(&self, id: &SessionId) -> Option<SessionEntry> {
        self.sessions.lock().get(&id.to_string()).cloned()
    }

    /// Get a session by string ID.
    pub fn get_str(&self, id: &str) -> Option<SessionEntry> {
        self.sessions.lock().get(id).cloned()
    }

    /// Update a session's title and/or model and/or metadata.
    pub fn update(
        &self,
        id: &SessionId,
        title: Option<String>,
        model: Option<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> Option<SessionEntry> {
        let id_str = id.to_string();
        let updated = {
            let mut sessions = self.sessions.lock();
            let entry = sessions.get_mut(&id_str)?;
            if let Some(t) = title {
                entry.title = t;
            }
            if let Some(m) = model {
                entry.model = m;
            }
            if let Some(meta) = metadata {
                entry.metadata.extend(meta);
            }
            entry.updated_at = Utc::now();
            entry.clone()
        };
        self.broadcaster
            .publish_simple(SessionEventKind::Updated, id_str);
        Some(updated)
    }

    /// Delete a session (soft delete): transitions to `Deleted` and removes it
    /// from the active set, cleaning up attachments.
    pub fn delete(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        let id_str = id.to_string();
        let entry = self
            .get(id)
            .ok_or_else(|| AppError::not_found(format!("Session {id} not found")))?;
        self.lifecycle.delete(&id_str)?;
        self.sessions.lock().remove(&id_str);
        self.attachments.delete_session(&id_str);
        self.search_index.remove_session(&id_str);
        Ok(entry)
    }

    // -----------------------------------------------------------------------
    // Lifecycle transitions
    // -----------------------------------------------------------------------

    /// Archive a session (transition to `Archived`) and snapshot it to disk
    /// when an archiver is configured.
    pub fn archive(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        let id_str = id.to_string();
        self.lifecycle.archive(&id_str)?;
        if let Some(archiver) = &self.archiver {
            let payload = self.snapshot(&id_str);
            archiver.archive(&id_str, payload)?;
        }
        self.refresh_state(&id_str);
        self.get(id)
            .ok_or_else(|| AppError::not_found(format!("Session {id} not found")))
    }

    /// Activate a session (Created → Active).
    pub fn activate(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        self.transition(id, LifecycleTransition::Activate)
    }

    /// Pause an active session (Active → Paused).
    pub fn pause(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        self.transition(id, LifecycleTransition::Pause)
    }

    /// Resume a paused session (Paused → Active).
    pub fn resume(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        self.transition(id, LifecycleTransition::Resume)
    }

    /// Restore an archived session (Archived → Active).
    pub fn restore(&self, id: &SessionId) -> Result<SessionEntry, AppError> {
        self.transition(id, LifecycleTransition::Restore)
    }

    /// Apply an arbitrary lifecycle transition and sync the stored entry.
    pub fn transition(
        &self,
        id: &SessionId,
        transition: LifecycleTransition,
    ) -> Result<SessionEntry, AppError> {
        let id_str = id.to_string();
        self.lifecycle.transition(&id_str, transition)?;
        self.refresh_state(&id_str);
        self.get(id)
            .ok_or_else(|| AppError::not_found(format!("Session {id} not found")))
    }

    /// Return the lifecycle state of a session, if known.
    pub fn state(&self, id: &SessionId) -> Option<SessionState> {
        self.lifecycle.state(&id.to_string())
    }

    /// Return the full lifecycle record for a session.
    pub fn lifecycle_history(&self, id: &SessionId) -> Option<Vec<LifecycleRecord>> {
        self.lifecycle.get(&id.to_string()).map(|l| l.history)
    }

    /// Copy the lifecycle manager's state back into the stored entry.
    fn refresh_state(&self, id: &str) {
        if let Some(state) = self.lifecycle.state(id) {
            let mut sessions = self.sessions.lock();
            if let Some(entry) = sessions.get_mut(id) {
                entry.state = state.as_str().to_string();
                entry.updated_at = Utc::now();
            }
        }
    }

    // -----------------------------------------------------------------------
    // Transcript
    // -----------------------------------------------------------------------

    /// Append a message to a session's transcript and index it for search.
    pub fn add_message(&self, session_id: &str, msg: Message) {
        let role = format!("{:?}", msg.role).to_lowercase();
        let message_id = Uuid::new_v4().to_string();
        {
            let mut transcripts = self.transcripts.lock();
            transcripts
                .entry(session_id.to_string())
                .or_default()
                .push(msg.clone());
        }
        {
            let mut sessions = self.sessions.lock();
            if let Some(entry) = sessions.get_mut(session_id) {
                entry.message_count += 1;
                entry.updated_at = Utc::now();
            }
        }
        self.search_index.index(IndexedMessage {
            session_id: session_id.to_string(),
            message_id,
            role,
            content: msg.text_content(),
            model: None,
            timestamp: Utc::now(),
        });
        self.broadcaster
            .publish_simple(SessionEventKind::MessageAdded, session_id);
    }

    /// Return all messages for a session, in order.
    pub fn messages(&self, session_id: &str) -> Vec<Message> {
        self.transcripts
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Return a page of messages for a session.
    pub fn messages_page(&self, session_id: &str, limit: usize, offset: usize) -> Vec<Message> {
        let all = self.messages(session_id);
        all.into_iter().skip(offset).take(limit).collect()
    }

    /// Build a JSON snapshot of a session for archiving.
    fn snapshot(&self, session_id: &str) -> serde_json::Value {
        let session = self.get_str(session_id);
        let messages = self.messages(session_id);
        serde_json::json!({
            "session": session,
            "messages": messages,
        })
    }

    // -----------------------------------------------------------------------
    // Turn management
    // -----------------------------------------------------------------------

    /// Reserve a turn slot for a session without a message yet. Used to
    /// allocate a turn id up front (e.g. before streaming input arrives).
    pub fn reserve_turn(&self, session_id: &str) -> TurnRecord {
        let record = TurnRecord {
            turn_id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            status: "reserved".to_string(),
            message: None,
            started_at: Utc::now(),
            completed_at: None,
            error: None,
        };
        self.turns
            .lock()
            .entry(session_id.to_string())
            .or_default()
            .push(record.clone());
        record
    }

    /// Start a turn. If a reserved turn id is supplied, promote it to
    /// `queued`; otherwise create a new turn with the given message.
    pub fn start_turn(
        &self,
        session_id: &str,
        message: Option<String>,
        turn_id: Option<Uuid>,
    ) -> TurnRecord {
        let mut turns = self.turns.lock();
        let list = turns.entry(session_id.to_string()).or_default();
        let record = match turn_id {
            Some(tid) => {
                if let Some(existing) = list.iter_mut().find(|t| t.turn_id == tid) {
                    existing.status = "queued".to_string();
                    existing.message = message.or_else(|| existing.message.clone());
                    existing.clone()
                } else {
                    let record = TurnRecord {
                        turn_id: tid,
                        session_id: session_id.to_string(),
                        status: "queued".to_string(),
                        message,
                        started_at: Utc::now(),
                        completed_at: None,
                        error: None,
                    };
                    list.push(record.clone());
                    record
                }
            }
            None => {
                let record = TurnRecord {
                    turn_id: Uuid::new_v4(),
                    session_id: session_id.to_string(),
                    status: "queued".to_string(),
                    message,
                    started_at: Utc::now(),
                    completed_at: None,
                    error: None,
                };
                list.push(record.clone());
                record
            }
        };
        drop(turns);
        {
            let mut sessions = self.sessions.lock();
            if let Some(entry) = sessions.get_mut(session_id) {
                entry.turn_count = entry.turn_count.saturating_add(1);
                entry.updated_at = Utc::now();
            }
        }
        self.broadcaster
            .publish_simple(SessionEventKind::TurnStarted, session_id);
        record
    }

    /// Cancel a reserved/queued/running turn by id.
    pub fn cancel_turn(&self, session_id: &str, turn_id: Uuid) -> Result<TurnRecord, AppError> {
        let mut turns = self.turns.lock();
        let list = turns
            .get_mut(session_id)
            .ok_or_else(|| AppError::not_found(format!("No turns for session '{session_id}'")))?;
        let record = list
            .iter_mut()
            .find(|t| t.turn_id == turn_id)
            .ok_or_else(|| AppError::not_found(format!("Turn '{turn_id}' not found")))?;
        if record.status == "completed" {
            return Err(AppError::bad_request("Cannot cancel a completed turn"));
        }
        record.status = "cancelled".to_string();
        record.completed_at = Some(Utc::now());
        Ok(record.clone())
    }

    /// Mark a turn as running.
    pub fn mark_turn_running(
        &self,
        session_id: &str,
        turn_id: Uuid,
    ) -> Result<TurnRecord, AppError> {
        let mut turns = self.turns.lock();
        let list = turns
            .get_mut(session_id)
            .ok_or_else(|| AppError::not_found(format!("No turns for session '{session_id}'")))?;
        let record = list
            .iter_mut()
            .find(|t| t.turn_id == turn_id)
            .ok_or_else(|| AppError::not_found(format!("Turn '{turn_id}' not found")))?;
        record.status = "running".to_string();
        Ok(record.clone())
    }

    /// Mark a turn as completed.
    pub fn complete_turn(&self, session_id: &str, turn_id: Uuid) -> Result<TurnRecord, AppError> {
        let mut turns = self.turns.lock();
        let list = turns
            .get_mut(session_id)
            .ok_or_else(|| AppError::not_found(format!("No turns for session '{session_id}'")))?;
        let record = list
            .iter_mut()
            .find(|t| t.turn_id == turn_id)
            .ok_or_else(|| AppError::not_found(format!("Turn '{turn_id}' not found")))?;
        record.status = "completed".to_string();
        record.completed_at = Some(Utc::now());
        Ok(record.clone())
    }

    /// List all turns for a session, most recent first.
    pub fn list_turns(&self, session_id: &str) -> Vec<TurnRecord> {
        let mut turns = self
            .turns
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default();
        turns.sort_by_key(|b| std::cmp::Reverse(b.started_at));
        turns
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    /// Request a compaction for a session. The record is created in the
    /// `requested` state and a background task completes it after a short
    /// delay with a summary of the transcript.
    pub fn trigger_compaction(&self, session_id: &str, reason: Option<String>) -> CompactionRecord {
        let record = CompactionRecord {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            status: "requested".to_string(),
            reason,
            requested_at: Utc::now(),
            completed_at: None,
            summary: None,
            input_tokens: None,
            output_tokens: None,
            error: None,
        };
        {
            let mut compactions = self.compactions.lock();
            compactions
                .entry(session_id.to_string())
                .or_default()
                .push(record.clone());
        }
        let store = self.clone();
        let session_id = session_id.to_string();
        let id = record.id;
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let summary = store.build_compaction_summary(&session_id);
            let input = store.estimate_tokens(&session_id);
            let _ = store.complete_compaction(&session_id, id, summary, input);
        });
        record
    }

    /// Complete a pending compaction record.
    pub fn complete_compaction(
        &self,
        session_id: &str,
        id: Uuid,
        summary: String,
        input_tokens: u64,
    ) -> Result<CompactionRecord, AppError> {
        let mut compactions = self.compactions.lock();
        let list = compactions.get_mut(session_id).ok_or_else(|| {
            AppError::not_found(format!("No compactions for session '{session_id}'"))
        })?;
        let record = list
            .iter_mut()
            .find(|c| c.id == id)
            .ok_or_else(|| AppError::not_found(format!("Compaction '{id}' not found")))?;
        record.status = "completed".to_string();
        record.completed_at = Some(Utc::now());
        record.summary = Some(summary);
        record.input_tokens = Some(input_tokens);
        // Rough estimate: 4 chars ≈ 1 output token for the summary.
        let out = (record
            .summary
            .as_ref()
            .map(|s| s.chars().count() as u64)
            .unwrap_or(0))
            / 4;
        record.output_tokens = Some(out);
        Ok(record.clone())
    }

    /// Return the most recent compaction record for a session.
    pub fn latest_compaction(&self, session_id: &str) -> Option<CompactionRecord> {
        self.compactions
            .lock()
            .get(session_id)
            .and_then(|list| list.last())
            .cloned()
    }

    /// Return all compaction records for a session.
    pub fn compaction_history(&self, session_id: &str) -> Vec<CompactionRecord> {
        self.compactions
            .lock()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Build a summary string from the transcript.
    fn build_compaction_summary(&self, session_id: &str) -> String {
        let messages = self.messages(session_id);
        let user_count = messages
            .iter()
            .filter(|m| m.role == MessageRole::User)
            .count();
        let total_chars: usize = messages
            .iter()
            .map(|m| m.text_content().chars().count())
            .sum();
        format!(
            "Compacted session with {count} messages ({users} user turns, ~{chars} chars).",
            count = messages.len(),
            users = user_count,
            chars = total_chars,
        )
    }

    /// Rough token estimate for a transcript (4 chars per token).
    fn estimate_tokens(&self, session_id: &str) -> u64 {
        let total_chars: usize = self
            .messages(session_id)
            .iter()
            .map(|m| m.text_content().chars().count())
            .sum();
        (total_chars / 4) as u64
    }

    // -----------------------------------------------------------------------
    // Attachments
    // -----------------------------------------------------------------------

    /// List attachments for a session.
    pub fn attachments_for(&self, session_id: &str) -> Vec<AttachmentMeta> {
        self.attachments.list_for_session(session_id)
    }

    /// Delete an attachment by id.
    pub fn delete_attachment(&self, attachment_id: &str) -> Result<bool, AppError> {
        self.attachments.delete(attachment_id)
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Export a session transcript in the requested format.
    pub fn export(
        &self,
        session_id: &str,
        format: ExportFormat,
    ) -> Result<serde_json::Value, AppError> {
        let session = self
            .get_str(session_id)
            .ok_or_else(|| AppError::not_found(format!("Session {session_id} not found")))?;
        let messages = self.messages(session_id);
        let exported: Vec<ExportedMessage> = messages
            .iter()
            .map(|m| ExportedMessage {
                role: format!("{:?}", m.role).to_lowercase(),
                content: m.text_content(),
                model: None,
                timestamp: Some(Utc::now()),
                meta: None,
            })
            .collect();
        let export = SessionExporter::build_export(
            session_id,
            serde_json::to_value(&session).map_err(|e| AppError::internal(e.to_string()))?,
            exported,
        );
        let content = String::from_utf8_lossy(&export.to_bytes(format)?).to_string();
        Ok(serde_json::json!({
            "session_id": session_id,
            "format": format_name(format),
            "content": content,
            "message_count": messages.len(),
        }))
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    /// Full-text search over all session transcripts.
    pub fn search_transcripts(&self, query: &str, limit: usize) -> SessionSearchResult {
        self.search_index.search(
            query,
            crate::session_search::SessionSearchOptions {
                limit,
                session_id: None,
            },
        )
    }
}

/// Stable string name for an export format.
fn format_name(format: ExportFormat) -> &'static str {
    match format {
        ExportFormat::Json => "json",
        ExportFormat::Markdown => "markdown",
        ExportFormat::Jsonl => "jsonl",
    }
}

/// Parse a session id parameter, erroring with a clear message.
fn parse_session_id(params: &serde_json::Value) -> Result<SessionId, AppError> {
    let id_str = params
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
    SessionId::from_string(id_str)
        .ok_or_else(|| AppError::bad_request(format!("Invalid session ID '{id_str}'")))
}

/// Parse a `session_id` parameter that may be keyed as `session_id` too.
///
/// Unlike [`parse_session_id`], this accepts arbitrary session keys (not just
/// UUIDs) because turn/compaction/attachment/export handlers operate on the
/// store by string key.
fn parse_any_session_id(params: &serde_json::Value) -> Result<String, AppError> {
    params
        .get("session_id")
        .or_else(|| params.get("id"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))
        .map(|s| s.to_string())
}

/// Register session RPC handlers on the given registry.
pub fn register_session_handlers(registry: &mut RpcRegistry, store: SessionStore) {
    let store = Arc::new(store);

    // sessions.create — create a new session
    registry.register(rpc_handler("sessions.create", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let title = params
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("New Session")
                    .to_string();
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let metadata: HashMap<String, serde_json::Value> = params
                    .get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                let mut entry = store.create(title, model);
                if !metadata.is_empty() {
                    if let Some(updated) = store.update(&entry.id, None, None, Some(metadata)) {
                        entry = updated;
                    }
                }
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.list — list sessions, optionally filtered by state
    registry.register(rpc_handler("sessions.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let state_filter = params
                    .get("state")
                    .and_then(|v| v.as_str())
                    .map(SessionState::parse)
                    .transpose()?;
                let sessions = store.list_by_state(state_filter.as_ref());
                Ok(serde_json::json!({
                    "sessions": sessions,
                    "count": sessions.len(),
                }))
            }
        }
    }));

    // sessions.get — fetch a session by id
    registry.register(rpc_handler("sessions.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                match store.get(&id) {
                    Some(entry) => Ok(serde_json::to_value(entry)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Session {id} not found"))),
                }
            }
        }
    }));

    // sessions.update — update title/model/metadata
    registry.register(rpc_handler("sessions.update", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let title = params
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let metadata: Option<HashMap<String, serde_json::Value>> = params
                    .get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
                match store.update(&id, title, model, metadata) {
                    Some(entry) => Ok(serde_json::to_value(entry)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Session {id} not found"))),
                }
            }
        }
    }));

    // sessions.delete — delete a session
    registry.register(rpc_handler("sessions.delete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.delete(&id)?;
                Ok(serde_json::json!({
                    "deleted": true,
                    "session": entry,
                }))
            }
        }
    }));

    // sessions.archive — transition to archived and snapshot to disk
    registry.register(rpc_handler("sessions.archive", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.archive(&id)?;
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.activate — Created → Active
    registry.register(rpc_handler("sessions.activate", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.activate(&id)?;
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.pause — Active → Paused
    registry.register(rpc_handler("sessions.pause", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.pause(&id)?;
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.resume — Paused → Active
    registry.register(rpc_handler("sessions.resume", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.resume(&id)?;
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.restore — Archived → Active
    registry.register(rpc_handler("sessions.restore", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let entry = store.restore(&id)?;
                serde_json::to_value(entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.state — current lifecycle state and transition history
    registry.register(rpc_handler("sessions.state", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_session_id(&params)?;
                let state = store
                    .state(&id)
                    .ok_or_else(|| AppError::not_found(format!("Session {id} not found")))?;
                let history = store.lifecycle_history(&id).unwrap_or_default();
                Ok(serde_json::json!({
                    "id": id.to_string(),
                    "state": state.as_str(),
                    "accepts_turns": state.accepts_turns(),
                    "read_only": state.is_read_only(),
                    "terminal": state.is_terminal(),
                    "history": history,
                }))
            }
        }
    }));

    // sessions.export — export the transcript
    registry.register(rpc_handler("sessions.export", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let format = params
                    .get("format")
                    .and_then(|v| v.as_str())
                    .map(ExportFormat::parse)
                    .transpose()?
                    .unwrap_or(ExportFormat::Json);
                let result = store.export(&session_id.to_string(), format)?;
                Ok(result)
            }
        }
    }));

    // sessions.turns.list — list turns for a session
    registry.register(rpc_handler("sessions.turns.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let turns = store.list_turns(&session_id.to_string());
                Ok(serde_json::json!({
                    "turns": turns,
                    "count": turns.len(),
                }))
            }
        }
    }));

    // sessions.turn.reserve — reserve a turn slot
    registry.register(rpc_handler("sessions.turn.reserve", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let record = store.reserve_turn(&session_id.to_string());
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.turn.start — start (queue) a turn, optionally promoting a reserved turn
    registry.register(rpc_handler("sessions.turn.start", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let message = params
                    .get("message")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let turn_id = params
                    .get("turn_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());
                let record = store.start_turn(&session_id.to_string(), message, turn_id);
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.turn.cancel — cancel a turn
    registry.register(rpc_handler("sessions.turn.cancel", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let turn_id = params
                    .get("turn_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'turn_id' parameter"))?;
                let turn_id = Uuid::parse_str(turn_id)
                    .map_err(|_| AppError::bad_request("Invalid turn_id"))?;
                let record = store.cancel_turn(&session_id.to_string(), turn_id)?;
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.compaction.trigger — request a compaction
    registry.register(rpc_handler("sessions.compaction.trigger", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let record = store.trigger_compaction(&session_id.to_string(), reason);
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.compaction.status — latest compaction for a session
    registry.register(rpc_handler("sessions.compaction.status", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                match store.latest_compaction(&session_id.to_string()) {
                    Some(record) => Ok(serde_json::to_value(record)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!(
                        "No compaction recorded for session {session_id}"
                    ))),
                }
            }
        }
    }));

    // sessions.compaction.history — compaction history for a session
    registry.register(rpc_handler("sessions.compaction.history", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let history = store.compaction_history(&session_id.to_string());
                Ok(serde_json::json!({
                    "compactions": history,
                    "count": history.len(),
                }))
            }
        }
    }));

    // sessions.attachments.list — list attachments for a session
    registry.register(rpc_handler("sessions.attachments.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
                let attachments = store.attachments_for(&session_id.to_string());
                Ok(serde_json::json!({
                    "attachments": attachments,
                    "count": attachments.len(),
                }))
            }
        }
    }));

    // sessions.attachments.upload — store a base64-encoded attachment
    registry.register(rpc_handler("sessions.attachments.upload", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = parse_any_session_id(&params)?;
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

                let dir = default_attachment_dir();
                let upload = AttachmentUpload {
                    filename,
                    content_type,
                    bytes,
                };
                let meta =
                    store_attachment(&store.attachments, &session_id.to_string(), upload, &dir)?;
                serde_json::to_value(meta).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // sessions.attachments.delete — delete an attachment by id
    registry.register(rpc_handler("sessions.attachments.delete", {
        let store = store.clone();
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

    // sessions.search — full-text search across session transcripts
    registry.register(rpc_handler("sessions.search", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'query' parameter"))?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let result = store.search_transcripts(query, limit);
                serde_json::to_value(result).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));
}

/// Default directory used for RPC attachment uploads.
fn default_attachment_dir() -> PathBuf {
    std::env::temp_dir().join("opensquilla-attachments")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_crud() {
        let store = SessionStore::new();
        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        // Create
        let params = serde_json::json!({"title": "Test Session", "model": "gpt-4"});
        let result = registry.dispatch("sessions.create", params).await;
        assert!(result.is_some());
        let entry: SessionEntry = serde_json::from_value(result.unwrap().unwrap()).unwrap();
        let session_id = entry.id;
        assert_eq!(entry.state, "created");

        // List
        let result = registry
            .dispatch("sessions.list", serde_json::Value::Null)
            .await;
        let resp = result.unwrap().unwrap();
        let sessions: Vec<SessionEntry> = serde_json::from_value(resp["sessions"].clone()).unwrap();
        assert_eq!(sessions.len(), 1);

        // Get
        let params = serde_json::json!({"id": session_id.to_string()});
        let result = registry.dispatch("sessions.get", params).await;
        assert!(result.unwrap().is_ok());

        // Activate then delete
        let result = registry
            .dispatch(
                "sessions.activate",
                serde_json::json!({"id": session_id.to_string()}),
            )
            .await;
        assert!(result.unwrap().is_ok());

        let params = serde_json::json!({"id": session_id.to_string()});
        let result = registry.dispatch("sessions.delete", params).await;
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_session_state_machine() {
        let store = SessionStore::new();
        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let resp = registry
            .dispatch("sessions.create", serde_json::json!({"title": "t"}))
            .await
            .unwrap()
            .unwrap();
        let id = resp["id"].as_str().unwrap().to_string();

        // created → active → paused → active → archived → active
        let r = registry
            .dispatch("sessions.activate", serde_json::json!({"id": id}))
            .await;
        assert_eq!(r.unwrap().unwrap()["state"], "active");

        let r = registry
            .dispatch("sessions.pause", serde_json::json!({"id": id}))
            .await;
        assert_eq!(r.unwrap().unwrap()["state"], "paused");

        // Pausing an already-paused session is illegal.
        let r = registry
            .dispatch("sessions.pause", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_err());

        let r = registry
            .dispatch("sessions.resume", serde_json::json!({"id": id}))
            .await;
        assert_eq!(r.unwrap().unwrap()["state"], "active");

        let r = registry
            .dispatch("sessions.archive", serde_json::json!({"id": id}))
            .await;
        assert_eq!(r.unwrap().unwrap()["state"], "archived");

        let r = registry
            .dispatch("sessions.restore", serde_json::json!({"id": id}))
            .await;
        assert_eq!(r.unwrap().unwrap()["state"], "active");

        let r = registry
            .dispatch("sessions.state", serde_json::json!({"id": id}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["state"], "active");
        assert_eq!(resp["accepts_turns"], true);
        assert!(resp["history"].as_array().unwrap().len() >= 4);
    }

    #[tokio::test]
    async fn test_turn_lifecycle() {
        let store = SessionStore::new();
        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let resp = registry
            .dispatch("sessions.create", serde_json::json!({}))
            .await
            .unwrap()
            .unwrap();
        let session_id = resp["id"].as_str().unwrap().to_string();

        // Reserve a turn slot.
        let r = registry
            .dispatch(
                "sessions.turn.reserve",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "reserved");
        let turn_id = resp["turn_id"].as_str().unwrap().to_string();

        // Start the turn (promotes reserved → queued).
        let r = registry
            .dispatch(
                "sessions.turn.start",
                serde_json::json!({"session_id": session_id, "turn_id": turn_id, "message": "hello"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "queued");
        assert_eq!(resp["message"], "hello");

        // Cancel it.
        let r = registry
            .dispatch(
                "sessions.turn.cancel",
                serde_json::json!({"session_id": session_id, "turn_id": turn_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "cancelled");

        // List turns.
        let r = registry
            .dispatch(
                "sessions.turns.list",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_compaction_trigger_and_complete() {
        let store = SessionStore::new();
        store.add_message("s1", Message::user("What is the weather?"));
        store.add_message("s1", Message::assistant("It is sunny today."));

        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch(
                "sessions.compaction.trigger",
                serde_json::json!({"session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "requested");
        let cid = resp["id"].as_str().unwrap().to_string();

        // Wait for the background task to finish.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        let r = registry
            .dispatch(
                "sessions.compaction.status",
                serde_json::json!({"session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["id"].as_str().unwrap(), cid);
        assert_eq!(resp["status"], "completed");
        assert!(resp["summary"].as_str().unwrap().contains("2 messages"));

        let r = registry
            .dispatch(
                "sessions.compaction.history",
                serde_json::json!({"session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_export_json() {
        let store = SessionStore::new();
        let resp = registry_create(&store).await;
        let session_id = resp["id"].as_str().unwrap().to_string();
        store.add_message(&session_id, Message::user("Hello"));
        store.add_message(&session_id, Message::assistant("Hi there"));

        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch(
                "sessions.export",
                serde_json::json!({"session_id": session_id, "format": "json"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["format"], "json");
        assert_eq!(resp["message_count"], 2);
        assert!(resp["content"].as_str().unwrap().contains("Hi there"));
    }

    #[tokio::test]
    async fn test_attachment_upload_and_delete() {
        let store = SessionStore::new();
        let resp = registry_create(&store).await;
        let session_id = resp["id"].as_str().unwrap().to_string();

        let data = base64::engine::general_purpose::STANDARD.encode(b"hello bytes");
        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch(
                "sessions.attachments.upload",
                serde_json::json!({
                    "session_id": session_id,
                    "filename": "note.txt",
                    "content_type": "text/plain",
                    "data": data,
                }),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["filename"], "note.txt");
        let attachment_id = resp["attachment_id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch(
                "sessions.attachments.list",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        let r = registry
            .dispatch(
                "sessions.attachments.delete",
                serde_json::json!({"attachment_id": attachment_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["deleted"], true);
    }

    #[tokio::test]
    async fn test_search_transcripts() {
        let store = SessionStore::new();
        let resp = registry_create(&store).await;
        let session_id = resp["id"].as_str().unwrap().to_string();
        store.add_message(
            &session_id,
            Message::user("How do I configure the API key?"),
        );
        store.add_message(&session_id, Message::assistant("It goes in the TOML file."));

        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, store.clone());

        let r = registry
            .dispatch("sessions.search", serde_json::json!({"query": "configure"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["total"].as_u64().unwrap() >= 1);
    }

    /// Helper: create a session and return the response value.
    async fn registry_create(store: &SessionStore) -> serde_json::Value {
        let mut registry = RpcRegistry::new();
        register_session_handlers(&mut registry, (*store).clone());
        registry
            .dispatch("sessions.create", serde_json::json!({"title": "t"}))
            .await
            .unwrap()
            .unwrap()
    }
}
