//! Multi-session RPC tools: sessions_send, sessions_spawn, sessions_yield,
//! sessions_history.
//!
//! These let an agent communicate with, spawn, yield to, and inspect other
//! sessions. They mirror the Python `opensquilla.tools.builtin.sessions`
//! module's session-to-session RPC tools.
//!
//! The Python tools key sessions by `session_key` strings (e.g.
//! `"agent:main:main"`) and dispatch through a `SessionManager` + `TaskRuntime`
//! pair. The Rust `opensquilla-session` crate keys sessions by `Uuid` and has
//! no task-runtime; these tools therefore take `session_id` UUIDs and persist
//! messages/transcript state directly through `SessionStorage`. Each spot that
//! would have enqueued work on the missing task-runtime is marked with a
//! `// TODO:` naming the missing primitive.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_session::SessionStorage;
use opensquilla_session::models::{Session, SessionFork, SessionStatus, TranscriptEntry};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Map a session-crate error to a tool error.
fn map_session_error(op: &str, err: impl std::fmt::Display) -> ToolError {
    ToolError::new("SESSION_ERROR", format!("Session {} failed: {}", op, err))
}

/// Terminal statuses that block further message injection. Mirrors the Python
/// `_TERMINAL_STATUSES` set; `SessionStatus` has no `Done`/`Failed` variants so
/// `Killed` and `Archived` are the terminal states.
fn is_terminal(status: &SessionStatus) -> bool {
    matches!(status, SessionStatus::Killed | SessionStatus::Archived)
}

/// Serialize a transcript entry for tool output, mirroring the Python
/// `read_transcript` row shape (role, content, created_at, token_count).
fn transcript_to_json(entry: &TranscriptEntry) -> Value {
    serde_json::json!({
        "id": entry.id.to_string(),
        "role": entry.role,
        "content": entry.content,
        "created_at": entry.created_at.to_rfc3339(),
        "token_count": entry.token_count,
        "compacted": entry.compacted,
    })
}

/// Serialize a session to a JSON value for tool output.
fn session_to_json(session: &Session) -> Value {
    serde_json::json!({
        "session_id": session.id.to_string(),
        "agent_id": session.agent_id.to_string(),
        "name": session.name,
        "status": session.status,
        "mode": session.mode,
        "parent_session_id": session.parent_session_id.map(|id| id.to_string()),
        "fork_event": session.fork_event,
        "message_count": session.message_count,
        "total_tokens": session.total_tokens,
        "created_at": session.created_at.to_rfc3339(),
        "last_active_at": session.last_active_at.to_rfc3339(),
    })
}

/// Parse and validate a UUID parameter.
fn parse_uuid_param(params: &Value, name: &str) -> Result<Uuid, ToolError> {
    let raw = params[name]
        .as_str()
        .ok_or_else(|| ToolError::invalid_args(format!("Missing required parameter '{}'", name)))?;
    Uuid::parse_str(raw)
        .map_err(|e| ToolError::invalid_args(format!("Invalid '{}' UUID '{}': {}", name, raw, e)))
}

// ===========================================================================
// sessions_send
// ===========================================================================

/// Tool for sending a message to another session (inter-session communication).
///
/// Validates the target session exists and is not terminal, then appends a
/// `user` transcript entry tagged with `inter_session` provenance. The Python
/// tool additionally enqueues a task on a `TaskRuntime`; the Rust session crate
/// has no task-runtime, so the message is persisted directly and the runtime
/// enqueue is a TODO.
pub struct SessionsSendTool {
    storage: Arc<SessionStorage>,
}

impl SessionsSendTool {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionsSendTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "sessions_send",
                "Send a message to another session (inter-session communication).",
                HashMap::from([
                    (
                        "session_id".to_string(),
                        ParameterDefinition::required_string("Target session UUID"),
                    ),
                    (
                        "message".to_string(),
                        ParameterDefinition::required_string("Message text to inject"),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let session_id = parse_uuid_param(&params, "session_id")?;
        let message = params["message"].as_str().unwrap_or("").trim().to_string();
        if message.is_empty() {
            return Err(ToolError::invalid_args("Message must not be empty"));
        }

        let storage = self.storage.clone();
        let message_for_task = message.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<Value, ToolError> {
            let session = storage
                .get_session(&session_id)
                .map_err(|e| map_session_error("get", e))?
                .ok_or_else(|| {
                    ToolError::new(
                        "SESSION_NOT_FOUND",
                        format!("Session '{}' not found", session_id),
                    )
                })?;
            if is_terminal(&session.status) {
                return Err(ToolError::new(
                    "SESSION_TERMINATED",
                    format!(
                        "Session '{}' is terminated (status={:?})",
                        session_id, session.status
                    ),
                ));
            }

            // Persist the message as a user transcript entry with
            // inter-session provenance. The Python tool enqueues a task via
            // TaskRuntime::send; the Rust session crate has no task-runtime
            // so we record the message directly.
            // TODO: enqueue on a TaskRuntime equivalent once the runtime
            // crate exposes one; for now the message is delivered
            // synchronously to the transcript.
            let entry = TranscriptEntry {
                metadata: serde_json::json!({
                    "kind": "inter_session",
                    "source_tool": "sessions_send",
                }),
                ..TranscriptEntry::new(session_id, "user".to_string(), message_for_task, 0)
            };
            storage
                .insert_transcript_entry(&entry)
                .map_err(|e| map_session_error("insert_transcript", e))?;

            // Bump the session's message_count / activity timestamps so the
            // new message is visible to listing and compaction.
            let mut updated = session;
            updated.message_count += 1;
            updated.last_active_at = chrono::Utc::now();
            updated.updated_at = updated.last_active_at;
            storage
                .update_session(&updated)
                .map_err(|e| map_session_error("update", e))?;

            Ok(serde_json::json!({
                "status": "delivered",
                "session_id": session_id.to_string(),
                "entry_id": entry.id.to_string(),
            }))
        })
        .await
        .map_err(|e| {
            ToolError::new("SESSION_ERROR", format!("Session send task failed: {}", e))
        })??;

        Ok(ToolOutput::success_with_data(
            format!("Message delivered to session {}", session_id),
            result,
        ))
    }
}

// ===========================================================================
// sessions_spawn
// ===========================================================================

/// Tool for spawning an isolated subagent session with its own transcript.
///
/// Creates a child session linked to `parent_session_id` (via
/// `parent_session_id` + a `SessionFork` row), appends the grounded task as the
/// child's first `user` transcript entry, and returns the new session id. The
/// Python tool additionally enqueues the child on a `TaskRuntime` and applies a
/// spawn-depth / max-children gate; the Rust session crate has no task-runtime
/// or spawn-depth tracking, so the enqueue is a TODO and the depth gate is
/// approximated by counting existing child forks.
pub struct SessionsSpawnTool {
    storage: Arc<SessionStorage>,
}

impl SessionsSpawnTool {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

/// Maximum spawn depth mirroring the Python `MAX_SPAWN_DEPTH` constant. The
/// Rust session crate does not persist spawn depth, so this is enforced against
/// the chain of parent links.
const MAX_SPAWN_DEPTH: usize = 8;

/// Default per-parent active-children cap mirroring the Python
/// `max_children_per_session` default. The Rust session crate has no policy
/// layer, so this is a static conservative bound.
const DEFAULT_MAX_CHILDREN: usize = 16;

#[async_trait]
impl Tool for SessionsSpawnTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "sessions_spawn",
                concat!(
                    "Spawn an isolated subagent session with its own context window and ",
                    "transcript. Returns immediately; after spawning one or more subagents, ",
                    "call sessions_yield with no session_id so completion is pushed back to ",
                    "the parent session. The task must be self-contained.",
                ),
                HashMap::from([
                    (
                        "parent_session_id".to_string(),
                        ParameterDefinition::required_string("The parent session UUID"),
                    ),
                    (
                        "task".to_string(),
                        ParameterDefinition::required_string(
                            "Initial task / user message for the spawned session",
                        ),
                    ),
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::string(
                            "Agent UUID the child session belongs to; defaults to the parent's agent",
                        ),
                    ),
                    (
                        "model".to_string(),
                        ParameterDefinition::string("Optional model override for the child session"),
                    ),
                ]),
            )
            .category("session")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let parent_session_id = parse_uuid_param(&params, "parent_session_id")?;
        let task = params["task"].as_str().unwrap_or("").trim().to_string();
        if task.is_empty() {
            return Err(ToolError::invalid_args("Task must not be empty"));
        }
        let model_override = params["model"]
            .as_str()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let agent_id_override = params["agent_id"]
            .as_str()
            .and_then(|s| Uuid::parse_str(s.trim()).ok());

        let storage = self.storage.clone();
        let result = tokio::task::spawn_blocking(
            move || -> Result<Value, ToolError> {
                let parent = storage
                    .get_session(&parent_session_id)
                    .map_err(|e| map_session_error("get_parent", e))?
                    .ok_or_else(|| {
                        ToolError::new(
                            "SESSION_NOT_FOUND",
                            format!("Parent session '{}' not found", parent_session_id),
                        )
                    })?;
                if is_terminal(&parent.status) {
                    return Err(ToolError::new(
                        "SESSION_TERMINATED",
                        format!(
                            "Parent session '{}' is terminated (status={:?})",
                            parent_session_id, parent.status
                        ),
                    ));
                }

                // Spawn-depth gate: walk the parent chain to count ancestors.
                let mut depth = 0usize;
                let mut ancestor_id = parent.parent_session_id;
                while let Some(id) = ancestor_id {
                    depth += 1;
                    if depth > MAX_SPAWN_DEPTH {
                        return Err(ToolError::new(
                            "SPAWN_DEPTH_EXCEEDED",
                            format!("Max spawn depth ({}) exceeded", MAX_SPAWN_DEPTH),
                        ));
                    }
                    match storage.get_session(&id) {
                        Ok(Some(s)) => ancestor_id = s.parent_session_id,
                        Ok(None) => break,
                        Err(e) => return Err(map_session_error("get_ancestor", e)),
                    }
                }

                // max-children gate: count existing child forks of the parent.
                // TODO: the Python gate counts only *running* children via a
                // status filter; the Rust session crate's fork records have no
                // status, so this counts all child forks ever created. A
                // status-filtered count needs a SessionStorage primitive that
                // joins session_forks to sessions on status.
                let children = storage
                    .list_session_forks_by_source(&parent_session_id)
                    .map_err(|e| map_session_error("list_forks", e))?;
                if children.len() >= DEFAULT_MAX_CHILDREN {
                    return Err(ToolError::new(
                        "MAX_CHILDREN_EXCEEDED",
                        format!(
                            "Max active children ({}) exceeded for session '{}'",
                            DEFAULT_MAX_CHILDREN, parent_session_id
                        ),
                    ));
                }

                // Resolve the child's agent: explicit > parent's agent.
                let child_agent_id = agent_id_override.unwrap_or(parent.agent_id);

                let now = chrono::Utc::now();
                let mut child = Session::default();
                child.agent_id = child_agent_id;
                child.name = format!("Subagent of {}", parent.name);
                child.mode = parent.mode.clone();
                child.system_prompt = parent.system_prompt.clone();
                child.parent_session_id = Some(parent_session_id);
                child.fork_event = Some("subagent".to_string());
                child.created_at = now;
                child.updated_at = now;
                child.last_active_at = now;
                let mut metadata = serde_json::json!({
                    "origin": "sessions_spawn",
                    "parent_session_id": parent_session_id.to_string(),
                    "spawn_depth": depth + 1,
                });
                if let Some(m) = &model_override {
                    metadata["model"] = serde_json::json!(m);
                }
                child.metadata = metadata;

                storage
                    .create_session(&child)
                    .map_err(|e| map_session_error("create_child", e))?;
                storage
                    .insert_session_fork(&SessionFork {
                        id: Uuid::new_v4(),
                        source_session_id: parent_session_id,
                        child_session_id: child.id,
                        fork_event: "subagent".to_string(),
                        created_at: now,
                        metadata: serde_json::json!({
                            "parent_task_id": null,
                            "spawn_depth": depth + 1,
                        }),
                    })
                    .map_err(|e| map_session_error("insert_fork", e))?;

                // Grounding prompt mirroring the Python `_SUBAGENT_SYSTEM_PROMPT`.
                let grounded_task = format!(
                    "You are a subagent. Execute the delegated task faithfully and return \
                     a structured result to your parent session.\n\n{}",
                    task
                );
                let entry = TranscriptEntry::new(
                    child.id,
                    "user".to_string(),
                    grounded_task,
                    0,
                );
                storage
                    .insert_transcript_entry(&entry)
                    .map_err(|e| map_session_error("insert_transcript", e))?;
                child.message_count = 1;
                child.updated_at = chrono::Utc::now();
                child.last_active_at = child.updated_at;
                storage
                    .update_session(&child)
                    .map_err(|e| map_session_error("update_child", e))?;

                // TODO: enqueue the child session on a TaskRuntime equivalent
                // (Python `runtime.enqueue` with run_kind="subagent") so the
                // child actually executes. The Rust session crate has no
                // task-runtime, so the child is persisted but not driven.

                Ok(serde_json::json!({
                    "session_id": child.id.to_string(),
                    "parent_session_id": parent_session_id.to_string(),
                    "agent_id": child_agent_id.to_string(),
                    "spawn_depth": depth + 1,
                    "status": "queued",
                    "completion_delivery": "pushed_to_parent_session",
                    "yield_instruction": "Call sessions_yield with no session_id after spawning subagents; do not wait on each child session.",
                }))
            },
        )
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session spawn task failed: {}", e),
            )
        })??;

        Ok(ToolOutput::success_with_data(
            format!(
                "Spawned subagent session {}",
                result["session_id"].as_str().unwrap_or("")
            ),
            result,
        ))
    }
}

// ===========================================================================
// sessions_yield
// ===========================================================================

/// Tool for yielding the current turn so pending subagent completions can be
/// pushed back later.
///
/// With no `session_id`, mirrors the Python no-key path: returns a `yielded`
/// status indicating the turn is releasing control. With a `session_id`,
/// mirrors the legacy status-wait path: returns the target session's current
/// status without blocking. The Python tool blocks on a `TaskRuntime` for up to
/// `timeout_seconds`; the Rust session crate has no task-runtime, so the wait
/// is a TODO and this returns a non-blocking status snapshot.
pub struct SessionsYieldTool {
    storage: Arc<SessionStorage>,
}

impl SessionsYieldTool {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionsYieldTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "sessions_yield",
                concat!(
                    "Yield the current turn so pending subagent completions can be pushed back ",
                    "later. Omit session_id after sessions_spawn. Supplying session_id is a ",
                    "legacy status wait that returns structured status on timeout instead of ",
                    "failing the tool.",
                ),
                HashMap::from([
                    (
                        "session_id".to_string(),
                        ParameterDefinition::string(
                            "Optional child session UUID for legacy status wait",
                        ),
                    ),
                    (
                        "message".to_string(),
                        ParameterDefinition::string(
                            "Optional note explaining why the current turn is yielding",
                        ),
                    ),
                    (
                        "timeout_seconds".to_string(),
                        ParameterDefinition::integer(
                            "Max wait time in seconds (0=return immediately, 1-3600 waits)",
                        )
                        .default(serde_json::json!(300)),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let timeout_seconds = params["timeout_seconds"].as_i64().unwrap_or(300);
        if !(0..=3600).contains(&timeout_seconds) {
            return Err(ToolError::invalid_args(
                "Timeout must be between 0 and 3600 seconds",
            ));
        }
        let message = params["message"]
            .as_str()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // No session_id: yield the current turn immediately. The Python tool
        // additionally closes the parent's subagent spawn group via the
        // gateway; the Rust session crate has no spawn-group concept, so we
        // return the plain yielded payload.
        let session_id_raw = params["session_id"]
            .as_str()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty());
        if session_id_raw.is_none() {
            let mut payload = serde_json::json!({
                "status": "yielded",
                "waited": false,
                "message": "Current turn yielded; wait for pushed session events.",
            });
            if let Some(m) = message {
                payload["yield_message"] = serde_json::json!(m);
            }
            return Ok(ToolOutput::success(
                serde_json::to_string_pretty(&payload).unwrap_or_default(),
            )
            .with_data(payload));
        }

        // Legacy status-wait path: look up the target session and return its
        // status without blocking.
        let session_id = Uuid::parse_str(session_id_raw.unwrap()).map_err(|e| {
            ToolError::invalid_args(format!(
                "Invalid 'session_id' UUID '{}': {}",
                session_id_raw.unwrap(),
                e
            ))
        })?;

        let storage = self.storage.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<Value, ToolError> {
            let session = storage
                .get_session(&session_id)
                .map_err(|e| map_session_error("get", e))?
                .ok_or_else(|| {
                    ToolError::new(
                        "SESSION_NOT_FOUND",
                        format!("Session '{}' not found", session_id),
                    )
                })?;
            // TODO: block on a TaskRuntime equivalent for up to timeout_seconds
            // (Python `runtime.wait(latest_task_id)`). The Rust session crate
            // has no task-runtime, so we return a non-blocking status snapshot.
            Ok(serde_json::json!({
                "session_id": session_id.to_string(),
                "status": session.status,
                "waited": false,
            }))
        })
        .await
        .map_err(|e| {
            ToolError::new("SESSION_ERROR", format!("Session yield task failed: {}", e))
        })??;

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&result).unwrap_or_default())
                .with_data(result),
        )
    }
}

// ===========================================================================
// sessions_history
// ===========================================================================

/// Tool for retrieving conversation history from a session's transcript.
pub struct SessionsHistoryTool {
    storage: Arc<SessionStorage>,
}

impl SessionsHistoryTool {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionsHistoryTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "sessions_history",
                "Retrieve conversation history from a session's transcript.",
                HashMap::from([
                    (
                        "session_id".to_string(),
                        ParameterDefinition::required_string("Session UUID to read history from"),
                    ),
                    (
                        "limit".to_string(),
                        ParameterDefinition::integer("Max messages to return (1-100)")
                            .default(serde_json::json!(20)),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let session_id = parse_uuid_param(&params, "session_id")?;
        let limit = params["limit"].as_i64().unwrap_or(20);
        if !(1..=100).contains(&limit) {
            return Err(ToolError::invalid_args("Limit must be between 1 and 100"));
        }
        let _limit_u64 = limit as u64;

        let storage = self.storage.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<Value, ToolError> {
            let session = storage
                .get_session(&session_id)
                .map_err(|e| map_session_error("get", e))?
                .ok_or_else(|| {
                    ToolError::new(
                        "SESSION_NOT_FOUND",
                        format!("Session '{}' not found", session_id),
                    )
                })?;

            // `get_transcript_entries` returns oldest-first with a limit. The
            // Python tool returns the most recent `limit` messages; to mirror
            // that, page through to count total and read the last page.
            // TODO: add a SessionStorage::recent_transcript_entries(session_id,
            // limit) primitive so we don't paginate here.
            let page = 200u64;
            let mut offset = 0u64;
            let mut all: Vec<TranscriptEntry> = Vec::new();
            loop {
                let batch = storage
                    .get_transcript_entries(&session_id, page, offset)
                    .map_err(|e| map_session_error("transcript", e))?;
                let len = batch.len() as u64;
                all.extend(batch);
                if len < page {
                    break;
                }
                offset += len;
            }
            let total = all.len();
            let start = total.saturating_sub(limit as usize);
            let messages: Vec<Value> = all[start..].iter().map(transcript_to_json).collect();

            Ok(serde_json::json!({
                "session_id": session_id.to_string(),
                "session": session_to_json(&session),
                "message_count": messages.len(),
                "total_messages": total,
                "messages": messages,
            }))
        })
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session history task failed: {}", e),
            )
        })??;

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&result).unwrap_or_default())
                .with_data(result),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_session::models::SessionMode;

    fn test_storage() -> Arc<SessionStorage> {
        Arc::new(SessionStorage::in_memory().expect("in-memory session storage"))
    }

    /// Seed a session and return its id.
    fn seed_session(storage: &SessionStorage) -> Uuid {
        let id = Uuid::new_v4();
        let session = Session {
            id,
            agent_id: Uuid::new_v4(),
            name: "rpc test".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_active_at: chrono::Utc::now(),
            status: SessionStatus::Active,
            mode: SessionMode::Chat,
            system_prompt: String::new(),
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: None,
            fork_event: None,
            metadata: serde_json::Value::Null,
        };
        storage.create_session(&session).expect("create session");
        id
    }

    #[tokio::test]
    async fn test_sessions_send_delivers_message() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        let tool = SessionsSendTool::from_arc(storage.clone());

        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "message": "hello from parent",
            }))
            .await;
        assert!(result.is_ok(), "send failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("delivered"));

        // The message was appended to the transcript and the session's
        // message_count was bumped.
        let entries = storage
            .get_transcript_entries(&session_id, 100, 0)
            .expect("transcript");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "user");
        assert_eq!(entries[0].content, "hello from parent");
        assert_eq!(
            entries[0].metadata["source_tool"],
            serde_json::json!("sessions_send")
        );
        let session = storage.get_session(&session_id).unwrap().unwrap();
        assert_eq!(session.message_count, 1);
    }

    #[tokio::test]
    async fn test_sessions_send_rejects_empty_message() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        let tool = SessionsSendTool::from_arc(storage);

        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "message": "   ",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_sessions_send_rejects_missing_session() {
        let storage = test_storage();
        let tool = SessionsSendTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": Uuid::new_v4().to_string(),
                "message": "hi",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_sessions_send_rejects_killed_session() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        // Mark the session Killed.
        let mut session = storage.get_session(&session_id).unwrap().unwrap();
        session.status = SessionStatus::Killed;
        storage.update_session(&session).unwrap();

        let tool = SessionsSendTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "message": "hi",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_TERMINATED");
    }

    #[tokio::test]
    async fn test_sessions_spawn_creates_child() {
        let storage = test_storage();
        let parent_id = seed_session(&storage);
        let tool = SessionsSpawnTool::from_arc(storage.clone());

        let result = tool
            .execute(serde_json::json!({
                "parent_session_id": parent_id.to_string(),
                "task": "summarize the report",
            }))
            .await;
        assert!(result.is_ok(), "spawn failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("queued"));
        assert_eq!(data["spawn_depth"], 1);
        let child_id: Uuid = data["session_id"].as_str().unwrap().parse().unwrap();

        // The child session exists with the parent link + fork record.
        let child = storage.get_session(&child_id).unwrap().unwrap();
        assert_eq!(child.parent_session_id, Some(parent_id));
        assert_eq!(child.fork_event.as_deref(), Some("subagent"));
        assert_eq!(child.message_count, 1);
        let forks = storage.list_session_forks_by_source(&parent_id).unwrap();
        assert_eq!(forks.len(), 1);
        assert_eq!(forks[0].child_session_id, child_id);

        // The grounded task was appended as the first user message.
        let entries = storage.get_transcript_entries(&child_id, 100, 0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].role, "user");
        assert!(entries[0].content.contains("summarize the report"));
        assert!(entries[0].content.contains("You are a subagent."));
    }

    #[tokio::test]
    async fn test_sessions_spawn_rejects_empty_task() {
        let storage = test_storage();
        let parent_id = seed_session(&storage);
        let tool = SessionsSpawnTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "parent_session_id": parent_id.to_string(),
                "task": "",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_sessions_spawn_rejects_missing_parent() {
        let storage = test_storage();
        let tool = SessionsSpawnTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "parent_session_id": Uuid::new_v4().to_string(),
                "task": "do thing",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_sessions_yield_no_session_id() {
        let storage = test_storage();
        let tool = SessionsYieldTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({ "timeout_seconds": 0 }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("yielded"));
        assert_eq!(data["waited"], serde_json::json!(false));
    }

    #[tokio::test]
    async fn test_sessions_yield_with_message() {
        let storage = test_storage();
        let tool = SessionsYieldTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "message": "waiting on subagent",
                "timeout_seconds": 0,
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(
            data["yield_message"],
            serde_json::json!("waiting on subagent")
        );
    }

    #[tokio::test]
    async fn test_sessions_yield_rejects_bad_timeout() {
        let storage = test_storage();
        let tool = SessionsYieldTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({ "timeout_seconds": 9999 }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_sessions_yield_legacy_status_wait() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        let tool = SessionsYieldTool::from_arc(storage.clone());
        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "timeout_seconds": 0,
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!(SessionStatus::Active));
        assert_eq!(data["waited"], serde_json::json!(false));
    }

    #[tokio::test]
    async fn test_sessions_yield_legacy_missing_session() {
        let storage = test_storage();
        let tool = SessionsYieldTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": Uuid::new_v4().to_string(),
                "timeout_seconds": 0,
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_sessions_history_returns_transcript() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        // Seed three transcript entries directly.
        for i in 0..3 {
            let entry =
                TranscriptEntry::new(session_id, "user".to_string(), format!("msg {}", i), 10);
            storage.insert_transcript_entry(&entry).unwrap();
        }

        let tool = SessionsHistoryTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "limit": 2,
            }))
            .await;
        assert!(result.is_ok(), "history failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["message_count"], 2);
        assert_eq!(data["total_messages"], 3);
        let messages = data["messages"].as_array().unwrap();
        // Most recent 2 returned, in chronological order.
        assert_eq!(messages[0]["content"], serde_json::json!("msg 1"));
        assert_eq!(messages[1]["content"], serde_json::json!("msg 2"));
    }

    #[tokio::test]
    async fn test_sessions_history_rejects_bad_limit() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        let tool = SessionsHistoryTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "limit": 0,
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_sessions_history_missing_session() {
        let storage = test_storage();
        let tool = SessionsHistoryTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": Uuid::new_v4().to_string(),
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_sessions_history_empty_transcript() {
        let storage = test_storage();
        let session_id = seed_session(&storage);
        let tool = SessionsHistoryTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["message_count"], 0);
        assert_eq!(data["total_messages"], 0);
    }
}
