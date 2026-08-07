//! Session tools: session_create, session_list, session_get, session_switch,
//! session_export, session_delete.
//!
//! Backed by `opensquilla-session`'s [`SessionStorage`]. All storage methods are
//! synchronous SQLite calls and are wrapped in `tokio::task::spawn_blocking`.
//!
//! Note on `session_switch`: the Rust `opensquilla-session` crate has no
//! global "active session" concept, so the tool tracks the *current* session
//! per agent in its own in-memory map (mirroring the Python layer's
//! `current_session_key`). The switch is validated against storage so it can
//! only point at an existing session.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_session::SessionStorage;
use opensquilla_session::models::{Session, SessionMode, TranscriptEntry};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use uuid::Uuid;

/// Map a session-crate error to a tool error.
fn map_session_error(op: &str, err: impl std::fmt::Display) -> ToolError {
    ToolError::new("SESSION_ERROR", format!("Session {} failed: {}", op, err))
}

/// Parse a `mode` parameter into a [`SessionMode`].
fn parse_mode(raw: Option<&str>) -> SessionMode {
    match raw.unwrap_or("chat").to_lowercase().as_str() {
        "plan" => SessionMode::Plan,
        "agent" => SessionMode::Agent,
        "batch" => SessionMode::Batch,
        _ => SessionMode::Chat,
    }
}

/// Serialize a session to a JSON value for tool output.
fn session_to_json(session: &Session) -> Value {
    serde_json::json!({
        "session_id": session.id.to_string(),
        "agent_id": session.agent_id.to_string(),
        "name": session.name,
        "status": session.status,
        "mode": session.mode,
        "system_prompt": session.system_prompt,
        "message_count": session.message_count,
        "total_tokens": session.total_tokens,
        "total_cost_usd": session.total_cost_usd,
        "created_at": session.created_at.to_rfc3339(),
        "updated_at": session.updated_at.to_rfc3339(),
        "last_active_at": session.last_active_at.to_rfc3339(),
        "parent_session_id": session.parent_session_id.map(|id| id.to_string()),
        "metadata": session.metadata,
    })
}

/// Tool for creating a new session.
pub struct SessionCreateTool {
    storage: Arc<SessionStorage>,
}

impl SessionCreateTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionCreateTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_create",
                "Create a new session for an agent. Returns the new session's ID.",
                HashMap::from([
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::required_string(
                            "The agent UUID the session belongs to",
                        ),
                    ),
                    (
                        "name".to_string(),
                        ParameterDefinition::required_string(
                            "A human-readable name for the session",
                        ),
                    ),
                    (
                        "system_prompt".to_string(),
                        ParameterDefinition::string("Optional system prompt for the session"),
                    ),
                    (
                        "mode".to_string(),
                        ParameterDefinition::string("Session mode: chat, plan, agent, batch")
                            .enum_values(vec![
                                "chat".into(),
                                "plan".into(),
                                "agent".into(),
                                "batch".into(),
                            ])
                            .default(serde_json::json!("chat")),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let agent_raw = params["agent_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'agent_id'"))?;
        let agent_id = Uuid::parse_str(agent_raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'agent_id' UUID '{}': {}", agent_raw, e))
        })?;
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?
            .to_string();
        let system_prompt = params["system_prompt"].as_str().unwrap_or("").to_string();
        let mode = parse_mode(params["mode"].as_str());

        // Build the Session by hand — SessionStorage::create_session persists a
        // fully-constructed Session rather than building one for us.
        let mut session = Session::default();
        session.agent_id = agent_id;
        session.name = name.clone();
        session.system_prompt = system_prompt.clone();
        session.mode = mode.clone();

        let storage = self.storage.clone();
        let session_for_task = session.clone();
        tokio::task::spawn_blocking(move || -> Result<(), ToolError> {
            storage
                .create_session(&session_for_task)
                .map_err(|e| map_session_error("create", e))
        })
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session create task failed: {}", e),
            )
        })??;

        let data = serde_json::json!({
            "session_id": session.id.to_string(),
            "agent_id": agent_id.to_string(),
            "name": name,
            "mode": mode,
            "session": session_to_json(&session),
        });

        Ok(ToolOutput::success_with_data(
            format!(
                "Created session '{}' ({}) for agent {}",
                name, session.id, agent_id
            ),
            data,
        ))
    }
}

/// Tool for listing sessions.
pub struct SessionListTool {
    storage: Arc<SessionStorage>,
}

impl SessionListTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionListTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_list",
                "List sessions for an agent, most recently active first, with pagination.",
                HashMap::from([
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::required_string("The agent UUID to list sessions for"),
                    ),
                    (
                        "limit".to_string(),
                        ParameterDefinition::integer("Maximum number of sessions")
                            .default(serde_json::json!(20)),
                    ),
                    (
                        "offset".to_string(),
                        ParameterDefinition::integer("Result offset for pagination")
                            .default(serde_json::json!(0)),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let agent_raw = params["agent_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'agent_id'"))?;
        let agent_id = Uuid::parse_str(agent_raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'agent_id' UUID '{}': {}", agent_raw, e))
        })?;
        let limit = params["limit"].as_i64().unwrap_or(20).max(1).min(200) as u64;
        let offset = params["offset"].as_i64().unwrap_or(0).max(0) as u64;

        let storage = self.storage.clone();
        let sessions = tokio::task::spawn_blocking(move || -> Result<Vec<Session>, ToolError> {
            storage
                .list_sessions(&agent_id, limit, offset)
                .map_err(|e| map_session_error("list", e))
        })
        .await
        .map_err(|e| {
            ToolError::new("SESSION_ERROR", format!("Session list task failed: {}", e))
        })??;

        let items: Vec<Value> = sessions.iter().map(session_to_json).collect();
        let data = serde_json::json!({
            "agent_id": agent_id.to_string(),
            "count": items.len(),
            "limit": limit,
            "offset": offset,
            "sessions": items,
        });

        let content = if items.is_empty() {
            format!("No sessions found for agent {}", agent_id)
        } else {
            serde_json::to_string_pretty(&data).unwrap_or_default()
        };
        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for getting a single session.
pub struct SessionGetTool {
    storage: Arc<SessionStorage>,
}

impl SessionGetTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionGetTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_get",
                "Get a session's details by its ID.",
                HashMap::from([(
                    "session_id".to_string(),
                    ParameterDefinition::required_string("The session UUID to look up"),
                )]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let raw = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?;
        let id = Uuid::parse_str(raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'session_id' UUID '{}': {}", raw, e))
        })?;

        let storage = self.storage.clone();
        let session = tokio::task::spawn_blocking(move || -> Result<Option<Session>, ToolError> {
            storage
                .get_session(&id)
                .map_err(|e| map_session_error("get", e))
        })
        .await
        .map_err(|e| {
            ToolError::new("SESSION_ERROR", format!("Session get task failed: {}", e))
        })??;

        let session = session.ok_or_else(|| {
            ToolError::new("SESSION_NOT_FOUND", format!("Session '{}' not found", raw))
        })?;

        let data = session_to_json(&session);
        Ok(ToolOutput::success_with_data(
            format!("Session '{}': {}", session.name, session.id),
            data,
        ))
    }
}

/// Tool for switching the current session for an agent.
///
/// The Rust session crate has no active-session concept; this tool maintains
/// a per-agent "current session" pointer in memory and validates the target
/// session exists in storage.
pub struct SessionSwitchTool {
    storage: Arc<SessionStorage>,
    /// agent_id -> current session_id.
    current: Mutex<HashMap<String, Uuid>>,
}

impl SessionSwitchTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
            current: Mutex::new(HashMap::new()),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self {
            storage,
            current: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Tool for SessionSwitchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_switch",
                concat!(
                    "Set the current session for an agent. The session must already exist. ",
                    "Returns the now-current session.",
                ),
                HashMap::from([
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::required_string(
                            "The agent UUID whose current session to switch",
                        ),
                    ),
                    (
                        "session_id".to_string(),
                        ParameterDefinition::required_string("The session UUID to switch to"),
                    ),
                ]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let agent_raw = params["agent_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'agent_id'"))?;
        let _agent_id = Uuid::parse_str(agent_raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'agent_id' UUID '{}': {}", agent_raw, e))
        })?;
        let raw = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?;
        let session_id = Uuid::parse_str(raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'session_id' UUID '{}': {}", raw, e))
        })?;

        // Validate the target session exists before switching to it.
        let storage = self.storage.clone();
        let exists = tokio::task::spawn_blocking(move || -> Result<bool, ToolError> {
            let s = storage
                .get_session(&session_id)
                .map_err(|e| map_session_error("get", e))?;
            Ok(s.is_some())
        })
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session switch task failed: {}", e),
            )
        })??;

        if !exists {
            return Err(ToolError::new(
                "SESSION_NOT_FOUND",
                format!("Cannot switch to non-existent session '{}'", raw),
            ));
        }

        self.current
            .lock()
            .map_err(|_| {
                ToolError::new("SESSION_ERROR", "Session switch lock poisoned".to_string())
            })?
            .insert(agent_raw.to_string(), session_id);

        let data = serde_json::json!({
            "agent_id": agent_raw,
            "session_id": raw,
            "current_session_id": raw,
        });
        Ok(ToolOutput::success_with_data(
            format!("Switched agent {} to session {}", agent_raw, raw),
            data,
        ))
    }
}

/// Tool for exporting a session's transcript.
pub struct SessionExportTool {
    storage: Arc<SessionStorage>,
}

impl SessionExportTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionExportTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_export",
                concat!(
                    "Export a session's transcript as plain text. Includes session metadata ",
                    "and every transcript entry in chronological order.",
                ),
                HashMap::from([(
                    "session_id".to_string(),
                    ParameterDefinition::required_string("The session UUID to export"),
                )]),
            )
            .category("session")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let raw = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?;
        let id = Uuid::parse_str(raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'session_id' UUID '{}': {}", raw, e))
        })?;

        let storage = self.storage.clone();
        let (session, entries) = tokio::task::spawn_blocking(
            move || -> Result<(Option<Session>, Vec<TranscriptEntry>), ToolError> {
                let session = storage
                    .get_session(&id)
                    .map_err(|e| map_session_error("get", e))?;
                // Paginate through the full transcript.
                let mut all = Vec::new();
                let page = 200u64;
                let mut offset = 0u64;
                loop {
                    let batch = storage
                        .get_transcript_entries(&id, page, offset)
                        .map_err(|e| map_session_error("export", e))?;
                    let len = batch.len() as u64;
                    all.extend(batch);
                    if len < page {
                        break;
                    }
                    offset += len;
                }
                Ok((session, all))
            },
        )
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session export task failed: {}", e),
            )
        })??;

        let session = session.ok_or_else(|| {
            ToolError::new("SESSION_NOT_FOUND", format!("Session '{}' not found", raw))
        })?;

        let mut out = String::new();
        out.push_str(&format!("Session: {}\n", session.name));
        out.push_str(&format!("Session ID: {}\n", session.id));
        out.push_str(&format!("Agent ID: {}\n", session.agent_id));
        out.push_str(&format!("Mode: {:?}\n", session.mode));
        out.push_str(&format!("Status: {:?}\n", session.status));
        out.push_str(&format!("Created: {}\n", session.created_at.to_rfc3339()));
        out.push_str(&format!("Messages: {}\n", session.message_count));
        out.push('\n');
        out.push_str("----- TRANSCRIPT -----\n");
        for entry in &entries {
            let role = if entry.role.is_empty() {
                "unknown"
            } else {
                &entry.role
            };
            let compacted_note = if entry.compacted { " (compacted)" } else { "" };
            out.push_str(&format!(
                "\n[{}]{} {}\n",
                role,
                compacted_note,
                entry.created_at.to_rfc3339()
            ));
            out.push_str(entry.content.trim());
            out.push('\n');
        }
        out.push_str("\n----- END TRANSCRIPT -----\n");
        out.push_str(&format!("{} entries exported\n", entries.len()));

        let data = serde_json::json!({
            "session_id": raw,
            "entries": entries.len(),
            "exported_at": chrono::Utc::now().to_rfc3339(),
        });
        Ok(ToolOutput::success(out)
            .with_data(data)
            .with_mime_type("text/plain"))
    }
}

/// Tool for deleting a session.
pub struct SessionDeleteTool {
    storage: Arc<SessionStorage>,
}

impl SessionDeleteTool {
    /// Create the tool from a shared storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SessionDeleteTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "session_delete",
                "Permanently delete a session and its transcript by its ID.",
                HashMap::from([(
                    "session_id".to_string(),
                    ParameterDefinition::required_string("The session UUID to delete"),
                )]),
            )
            .category("session")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let raw = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?;
        let id = Uuid::parse_str(raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'session_id' UUID '{}': {}", raw, e))
        })?;

        let storage = self.storage.clone();
        let deleted = tokio::task::spawn_blocking(move || -> Result<bool, ToolError> {
            // delete_session returns Ok even for a missing id; distinguish a
            // no-op from a real delete by checking existence first.
            let exists = storage
                .get_session(&id)
                .map_err(|e| map_session_error("get", e))?
                .is_some();
            if exists {
                storage
                    .delete_session(&id)
                    .map_err(|e| map_session_error("delete", e))?;
            }
            Ok(exists)
        })
        .await
        .map_err(|e| {
            ToolError::new(
                "SESSION_ERROR",
                format!("Session delete task failed: {}", e),
            )
        })??;

        if !deleted {
            return Err(ToolError::new(
                "SESSION_NOT_FOUND",
                format!("Session '{}' not found", raw),
            ));
        }

        let data = serde_json::json!({ "session_id": raw, "deleted": true });
        Ok(ToolOutput::success_with_data(
            format!("Deleted session '{}'", raw),
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SessionStorage` is not `Clone`, so tools share one handle through an
    /// `Arc<SessionStorage>` (via `from_arc`).
    fn test_storage() -> Arc<SessionStorage> {
        Arc::new(SessionStorage::in_memory().expect("in-memory session storage"))
    }

    #[tokio::test]
    async fn test_session_create_list_get() {
        let storage = test_storage();
        let create = SessionCreateTool::from_arc(storage.clone());
        let list = SessionListTool::from_arc(storage.clone());
        let get = SessionGetTool::from_arc(storage.clone());
        let agent = Uuid::new_v4();

        let result = create
            .execute(serde_json::json!({
                "agent_id": agent.to_string(),
                "name": "test session",
            }))
            .await;
        assert!(result.is_ok(), "create failed: {:?}", result.err());
        let session_id = result.unwrap().data.unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        let result = list
            .execute(serde_json::json!({ "agent_id": agent.to_string() }))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.unwrap()["count"].as_u64(), Some(1));

        let result = get
            .execute(serde_json::json!({ "session_id": session_id }))
            .await;
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().data.unwrap()["name"].as_str(),
            Some("test session")
        );
    }

    #[tokio::test]
    async fn test_session_switch_requires_existing() {
        let storage = test_storage();
        let sw = SessionSwitchTool::from_arc(storage);
        let result = sw
            .execute(serde_json::json!({
                "agent_id": Uuid::new_v4().to_string(),
                "session_id": Uuid::new_v4().to_string(),
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_session_export_empty_transcript() {
        let storage = test_storage();
        let create = SessionCreateTool::from_arc(storage.clone());
        let export = SessionExportTool::from_arc(storage.clone());
        let agent = Uuid::new_v4();

        let result = create
            .execute(serde_json::json!({
                "agent_id": agent.to_string(),
                "name": "export me",
            }))
            .await
            .unwrap();
        let session_id = result.data.unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        let result = export
            .execute(serde_json::json!({ "session_id": session_id }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("0 entries exported"));
    }

    #[tokio::test]
    async fn test_session_delete() {
        let storage = test_storage();
        let create = SessionCreateTool::from_arc(storage.clone());
        let get = SessionGetTool::from_arc(storage.clone());
        let delete = SessionDeleteTool::from_arc(storage.clone());
        let agent = Uuid::new_v4();

        let result = create
            .execute(serde_json::json!({
                "agent_id": agent.to_string(),
                "name": "delete me",
            }))
            .await
            .unwrap();
        let session_id = result.data.unwrap()["session_id"]
            .as_str()
            .unwrap()
            .to_string();

        let result = delete
            .execute(serde_json::json!({ "session_id": session_id }))
            .await;
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().data.unwrap()["deleted"],
            serde_json::json!(true)
        );

        // The session is gone, and deleting it again reports not-found.
        let result = get
            .execute(serde_json::json!({ "session_id": session_id }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");

        let result = delete
            .execute(serde_json::json!({ "session_id": session_id }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SESSION_NOT_FOUND");
    }
}
