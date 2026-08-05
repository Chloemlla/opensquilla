use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use rusqlite::{params, Connection, Transaction};
use serde_json;
use std::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use crate::models::{
    AgentTask, CompactedTranscriptEntry, CompactionHistory, PlanRevision, PlanRun, PlanRunStatus,
    PlanStatus, ProjectWorkspace, RoutingDecision, Session, SessionAttachment, SessionContextState,
    SessionFork, SessionLock, SessionMetadata, SessionMode, SessionStatus, SessionSummary,
    SessionTag, TaskStatus, TranscriptEntry, UsageEvent, UsageEventItem, UsageLedgerState,
};

pub struct SessionStorage {
    conn: Mutex<Connection>,
}

impl SessionStorage {
    pub fn new(path: &str) -> CoreResult<Self> {
        let conn = Connection::open(path).map_err(|e| CoreError::Storage(e.to_string()))?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.initialize_tables()?;
        Ok(storage)
    }

    pub fn in_memory() -> CoreResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| CoreError::Storage(e.to_string()))?;
        let storage = Self {
            conn: Mutex::new(conn),
        };
        storage.initialize_tables()?;
        Ok(storage)
    }

    fn initialize_tables(&self) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;

        // 1. sessions — session metadata, routing, token tracking, cost
        // 2. transcript_entries — message records (role, content, tool calls, reasoning)
        // 3. compacted_transcript_entries — compaction moved-out rows
        // 4. session_summaries — compaction summary records
        // 5. session_context_states — context state
        // 6. plan_revisions — collaboration plan revisions
        // 7. plan_runs — plan run records
        // 8. agent_tasks — task runtime ledger
        // 9. project_workspaces — project directories
        // 10. usage_events — usage events
        // 11. usage_event_items — usage event items
        // 12. usage_ledger_state — usage ledger state
        // 13. session_locks — per-session locks
        // 14. session_attachments — attachment metadata
        // 15. session_forks — fork event records
        // 16. routing_decisions — routing decision records
        // 17. session_metadata — key-value session metadata
        // 18. session_tags — session tag associations
        // 19. compaction_history — compaction history log
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                mode TEXT NOT NULL DEFAULT 'chat',
                system_prompt TEXT NOT NULL DEFAULT '',
                total_tokens INTEGER NOT NULL DEFAULT 0,
                total_cost_usd REAL NOT NULL DEFAULT 0.0,
                message_count INTEGER NOT NULL DEFAULT 0,
                parent_session_id TEXT,
                fork_event TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_agent
                ON sessions(agent_id, last_active_at);
            CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status, last_active_at);

            CREATE TABLE IF NOT EXISTS transcript_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                compacted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_transcript_session
                ON transcript_entries(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_transcript_compacted
                ON transcript_entries(session_id, compacted, created_at);

            CREATE TABLE IF NOT EXISTS compacted_transcript_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                original_entry_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                original_created_at TEXT NOT NULL,
                compacted_at TEXT NOT NULL,
                compaction_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_compacted_transcript_session
                ON compacted_transcript_entries(session_id, original_created_at);
            CREATE INDEX IF NOT EXISTS idx_compacted_transcript_compaction
                ON compacted_transcript_entries(compaction_id);

            CREATE TABLE IF NOT EXISTS session_summaries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_summary_session
                ON session_summaries(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_summary_active
                ON session_summaries(session_id, is_active);

            CREATE TABLE IF NOT EXISTS session_context_states (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                active_summary_id TEXT,
                retained_after TEXT,
                context_tokens INTEGER NOT NULL DEFAULT 0,
                budget_tokens INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_context_state_session
                ON session_context_states(session_id);

            CREATE TABLE IF NOT EXISTS plan_revisions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                plan TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                parent_revision_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_plan_session
                ON plan_revisions(session_id, version);
            CREATE INDEX IF NOT EXISTS idx_plan_status
                ON plan_revisions(status, created_at);

            CREATE TABLE IF NOT EXISTS plan_runs (
                id TEXT PRIMARY KEY,
                plan_revision_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT NOT NULL,
                completed_at TEXT,
                agent_task_id TEXT,
                result TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (plan_revision_id) REFERENCES plan_revisions(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_plan_run_revision
                ON plan_runs(plan_revision_id, started_at);
            CREATE INDEX IF NOT EXISTS idx_plan_run_session
                ON plan_runs(session_id, status);

            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                parent_task_id TEXT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                input TEXT NOT NULL DEFAULT '{}',
                output TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_agent_task_session
                ON agent_tasks(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_task_status
                ON agent_tasks(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_agent_task_parent
                ON agent_tasks(parent_task_id);

            CREATE TABLE IF NOT EXISTS project_workspaces (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_workspace_session
                ON project_workspaces(session_id, created_at);

            CREATE TABLE IF NOT EXISTS usage_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                request_id TEXT,
                total_cost_nanodollars INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_usage_event_session
                ON usage_events(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_event_request
                ON usage_events(request_id);

            CREATE TABLE IF NOT EXISTS usage_event_items (
                id TEXT PRIMARY KEY,
                usage_event_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                units INTEGER NOT NULL DEFAULT 0,
                cost_nanodollars INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (usage_event_id) REFERENCES usage_events(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_usage_item_event
                ON usage_event_items(usage_event_id);
            CREATE INDEX IF NOT EXISTS idx_usage_item_session
                ON usage_event_items(session_id, kind);

            CREATE TABLE IF NOT EXISTS usage_ledger_state (
                session_id TEXT PRIMARY KEY,
                last_event_id TEXT,
                last_event_seq INTEGER NOT NULL DEFAULT 0,
                total_cost_nanodollars INTEGER NOT NULL DEFAULT 0,
                total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
                total_completion_tokens INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS session_locks (
                session_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_session_lock_owner
                ON session_locks(owner, expires_at);

            CREATE TABLE IF NOT EXISTS session_attachments (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                storage_uri TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_attachment_session
                ON session_attachments(session_id, created_at);

            CREATE TABLE IF NOT EXISTS session_forks (
                id TEXT PRIMARY KEY,
                source_session_id TEXT NOT NULL,
                child_session_id TEXT NOT NULL,
                fork_event TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (source_session_id) REFERENCES sessions(id),
                FOREIGN KEY (child_session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_fork_source
                ON session_forks(source_session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_fork_child
                ON session_forks(child_session_id);

            CREATE TABLE IF NOT EXISTS routing_decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_routing_session
                ON routing_decisions(session_id, turn, created_at);

            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, key),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_session_metadata_key
                ON session_metadata(key);

            CREATE TABLE IF NOT EXISTS session_tags (
                session_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, tag),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_session_tags_tag
                ON session_tags(tag);

            CREATE TABLE IF NOT EXISTS compaction_history (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                compaction_id TEXT NOT NULL,
                entries_compacted INTEGER NOT NULL DEFAULT 0,
                tokens_before INTEGER NOT NULL DEFAULT 0,
                tokens_after INTEGER NOT NULL DEFAULT 0,
                summary_id TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed',
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_compaction_history_session
                ON compaction_history(session_id, completed_at);

            CREATE TABLE IF NOT EXISTS usage_entries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost_nanodollars INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_usage_session
                ON usage_entries(session_id, timestamp);
            ",
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        info!("Session storage tables initialized (19 tables)");
        Ok(())
    }

    /// Acquire a `Transaction` over the underlying connection for multi-table
    /// operations. The closure receives the transaction and returns its result;
    /// the transaction is committed on `Ok` and rolled back on `Err`.
    pub fn transaction<F, T>(&self, f: F) -> CoreResult<T>
    where
        F: FnOnce(&Transaction) -> CoreResult<T>,
    {
        let mut conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let tx = conn.transaction().map_err(|e| CoreError::Storage(e.to_string()))?;
        let result = f(&tx)?;
        tx.commit().map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(result)
    }

    // --- Session CRUD ---

    pub fn create_session(&self, session: &Session) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO sessions (id, agent_id, name, created_at, updated_at, last_active_at,
             status, mode, system_prompt, total_tokens, total_cost_usd, message_count,
             parent_session_id, fork_event, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                session.id.to_string(),
                session.agent_id.to_string(),
                session.name,
                session.created_at.to_rfc3339(),
                session.updated_at.to_rfc3339(),
                session.last_active_at.to_rfc3339(),
                serde_json::to_string(&session.status).unwrap_or_default(),
                serde_json::to_string(&session.mode).unwrap_or_default(),
                session.system_prompt,
                session.total_tokens,
                session.total_cost_usd,
                session.message_count,
                session.parent_session_id.map(|id| id.to_string()),
                session.fork_event,
                session.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_session(&self, id: &Uuid) -> CoreResult<Option<Session>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, agent_id, name, created_at, updated_at, last_active_at,
                       status, mode, system_prompt, total_tokens, total_cost_usd, message_count,
                       parent_session_id, fork_event, metadata FROM sessions WHERE id = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(Session {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    agent_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    last_active_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    status: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(SessionStatus::Active),
                    mode: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(SessionMode::Chat),
                    system_prompt: row.get(8)?,
                    total_tokens: row.get::<_, i64>(9)? as u64,
                    total_cost_usd: row.get(10)?,
                    message_count: row.get::<_, i64>(11)? as u64,
                    parent_session_id: row
                        .get::<_, Option<String>>(12)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    fork_event: row.get(13)?,
                    metadata: serde_json::from_str(&row.get::<_, String>(14)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(session)) => Ok(Some(session)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn update_session(&self, session: &Session) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE sessions SET name = ?1, updated_at = ?2, last_active_at = ?3,
             status = ?4, mode = ?5, system_prompt = ?6, total_tokens = ?7,
             total_cost_usd = ?8, message_count = ?9, metadata = ?10
             WHERE id = ?11",
            params![
                session.name,
                session.updated_at.to_rfc3339(),
                session.last_active_at.to_rfc3339(),
                serde_json::to_string(&session.status).unwrap_or_default(),
                serde_json::to_string(&session.mode).unwrap_or_default(),
                session.system_prompt,
                session.total_tokens as i64,
                session.total_cost_usd,
                session.message_count as i64,
                session.metadata.to_string(),
                session.id.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn delete_session(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute("DELETE FROM sessions WHERE id = ?1", params![id.to_string()])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List sessions across all agents, newest-first. Used by the manager's
    /// unfiltered `list()` path.
    pub fn list_all_sessions(&self, limit: u64, offset: u64) -> CoreResult<Vec<Session>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, agent_id, name, created_at, updated_at, last_active_at,
                 status, mode, system_prompt, total_tokens, total_cost_usd, message_count,
                 parent_session_id, fork_event, metadata
                 FROM sessions
                 ORDER BY last_active_at DESC LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let sessions = stmt
            .query_map(params![limit as i64, offset as i64], session_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(sessions)
    }

    pub fn list_sessions(&self, agent_id: &Uuid, limit: u64, offset: u64) -> CoreResult<Vec<Session>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, agent_id, name, created_at, updated_at, last_active_at,
                 status, mode, system_prompt, total_tokens, total_cost_usd, message_count,
                 parent_session_id, fork_event, metadata
                 FROM sessions WHERE agent_id = ?1
                 ORDER BY last_active_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let sessions = stmt
            .query_map(params![agent_id.to_string(), limit as i64, offset as i64], |row| {
                Ok(Session {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    agent_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    last_active_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    status: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(SessionStatus::Active),
                    mode: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(SessionMode::Chat),
                    system_prompt: row.get(8)?,
                    total_tokens: row.get::<_, i64>(9)? as u64,
                    total_cost_usd: row.get(10)?,
                    message_count: row.get::<_, i64>(11)? as u64,
                    parent_session_id: row
                        .get::<_, Option<String>>(12)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    fork_event: row.get(13)?,
                    metadata: serde_json::from_str(&row.get::<_, String>(14)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(sessions)
    }

    // --- Transcript CRUD ---

    pub fn insert_transcript_entry(&self, entry: &TranscriptEntry) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO transcript_entries (id, session_id, role, content, created_at,
             token_count, metadata, compacted)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                entry.id.to_string(),
                entry.session_id.to_string(),
                entry.role,
                entry.content,
                entry.created_at.to_rfc3339(),
                entry.token_count as i64,
                entry.metadata.to_string(),
                entry.compacted as i64,
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_transcript_entries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<TranscriptEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, created_at, token_count, metadata, compacted
                 FROM transcript_entries WHERE session_id = ?1
                 ORDER BY created_at ASC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let entries = stmt
            .query_map(params![session_id.to_string(), limit as i64, offset as i64], |row| {
                Ok(TranscriptEntry {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    token_count: row.get::<_, i64>(5)? as u64,
                    metadata: serde_json::from_str(&row.get::<_, String>(6)?)
                        .unwrap_or(serde_json::Value::Null),
                    compacted: row.get::<_, i64>(7)? != 0,
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    pub fn mark_entries_compacted(
        &self,
        session_id: &Uuid,
        up_to: &DateTime<Utc>,
    ) -> CoreResult<u64> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let count = conn
            .execute(
                "UPDATE transcript_entries SET compacted = 1
                 WHERE session_id = ?1 AND created_at <= ?2 AND compacted = 0",
                params![session_id.to_string(), up_to.to_rfc3339()],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(count as u64)
    }

    /// Get a single transcript entry by id.
    pub fn get_transcript_entry(&self, id: &Uuid) -> CoreResult<Option<TranscriptEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, created_at, token_count, metadata, compacted
                 FROM transcript_entries WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], transcript_entry_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn delete_transcript_entry(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM transcript_entries WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// `list_by_session`: full transcript history (both active and compacted
    /// entries) for a session in chronological order.
    pub fn list_by_session(&self, session_id: &Uuid) -> CoreResult<Vec<TranscriptEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, role, content, created_at, token_count, metadata, compacted
                 FROM transcript_entries WHERE session_id = ?1
                 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let entries = stmt
            .query_map(params![session_id.to_string()], transcript_entry_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(entries)
    }

    // --- Summary CRUD ---

    pub fn insert_summary(&self, summary: &SessionSummary) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_summaries (id, session_id, summary, created_at, token_count, is_active)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                summary.id.to_string(),
                summary.session_id.to_string(),
                summary.summary,
                summary.created_at.to_rfc3339(),
                summary.token_count as i64,
                summary.is_active as i64,
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_active_summary(&self, session_id: &Uuid) -> CoreResult<Option<SessionSummary>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, summary, created_at, token_count, is_active
                 FROM session_summaries WHERE session_id = ?1 AND is_active = 1
                 ORDER BY created_at DESC LIMIT 1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionSummary {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    summary: row.get(2)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    token_count: row.get::<_, i64>(4)? as u64,
                    is_active: row.get::<_, i64>(5)? != 0,
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(s)) => Ok(Some(s)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn deactivate_summaries(&self, session_id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE session_summaries SET is_active = 0 WHERE session_id = ?1",
            params![session_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List all summaries for a session, newest first.
    pub fn list_summaries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<SessionSummary>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, summary, created_at, token_count, is_active
                 FROM session_summaries WHERE session_id = ?1
                 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let summaries = stmt
            .query_map(
                params![session_id.to_string(), limit as i64, offset as i64],
                session_summary_mapper,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(summaries)
    }

    pub fn delete_summary(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_summaries WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Plan Revision CRUD ---

    pub fn insert_plan_revision(&self, plan: &PlanRevision) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO plan_revisions (id, session_id, plan, status, created_at, version, parent_revision_id, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                plan.id.to_string(),
                plan.session_id.to_string(),
                plan.plan,
                serde_json::to_string(&plan.status).unwrap_or_default(),
                plan.created_at.to_rfc3339(),
                plan.version as i64,
                plan.parent_revision_id.map(|id| id.to_string()),
                plan.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_latest_plan(&self, session_id: &Uuid) -> CoreResult<Option<PlanRevision>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, plan, status, created_at, version, parent_revision_id, metadata
                 FROM plan_revisions WHERE session_id = ?1
                 ORDER BY version DESC LIMIT 1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(PlanRevision {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    plan: row.get(2)?,
                    status: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or(PlanStatus::Draft),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    version: row.get::<_, i64>(5)? as u32,
                    parent_revision_id: row
                        .get::<_, Option<String>>(6)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(p)) => Ok(Some(p)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn get_plan_revision(&self, revision_id: &Uuid) -> CoreResult<Option<PlanRevision>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, plan, status, created_at, version, parent_revision_id, metadata
                 FROM plan_revisions WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![revision_id.to_string()], |row| {
                Ok(PlanRevision {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    plan: row.get(2)?,
                    status: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or(PlanStatus::Draft),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    version: row.get::<_, i64>(5)? as u32,
                    parent_revision_id: row
                        .get::<_, Option<String>>(6)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(p)) => Ok(Some(p)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn update_plan_status(&self, id: &Uuid, status: &PlanStatus) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE plan_revisions SET status = ?1 WHERE id = ?2",
            params![
                serde_json::to_string(status).unwrap_or_default(),
                id.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Update a plan revision's plan text and metadata (used to persist
    /// step-level state, which is encoded in `metadata["steps"]`). Status
    /// changes should go through [`update_plan_status`].
    pub fn update_plan_revision(&self, revision: &PlanRevision) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE plan_revisions SET plan = ?1, metadata = ?2, parent_revision_id = ?3
             WHERE id = ?4",
            params![
                revision.plan,
                revision.metadata.to_string(),
                revision.parent_revision_id.map(|id| id.to_string()),
                revision.id.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List all plan revisions for a session, newest version first.
    pub fn list_plan_revisions(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<PlanRevision>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, plan, status, created_at, version, parent_revision_id, metadata
                 FROM plan_revisions WHERE session_id = ?1
                 ORDER BY version DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let plans = stmt
            .query_map(
                params![session_id.to_string(), limit as i64, offset as i64],
                plan_revision_mapper,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(plans)
    }

    pub fn delete_plan_revision(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM plan_revisions WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Usage Entry CRUD ---

    pub fn insert_usage_entry(
        &self,
        session_id: &Uuid,
        prompt_tokens: u64,
        completion_tokens: u64,
        cost_nanodollars: u64,
        model: &str,
        provider: &str,
    ) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO usage_entries (id, session_id, timestamp, prompt_tokens, completion_tokens,
             cost_nanodollars, model, provider)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                Uuid::new_v4().to_string(),
                session_id.to_string(),
                Utc::now().to_rfc3339(),
                prompt_tokens as i64,
                completion_tokens as i64,
                cost_nanodollars as i64,
                model,
                provider,
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_usage_entries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<crate::usage_ledger::UsageEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, timestamp, prompt_tokens, completion_tokens,
                 cost_nanodollars, model, provider
                 FROM usage_entries WHERE session_id = ?1
                 ORDER BY timestamp DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let entries = stmt
            .query_map(params![session_id.to_string(), limit as i64, offset as i64], |row| {
                Ok(crate::usage_ledger::UsageEntry {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    prompt_tokens: row.get::<_, i64>(3)? as u64,
                    completion_tokens: row.get::<_, i64>(4)? as u64,
                    cost_nanodollars: row.get::<_, i64>(5)? as u64,
                    model: row.get(6)?,
                    provider: row.get(7)?,
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entries)
    }

    /// Aggregate usage summary (prompt tokens, completion tokens, cost) for a
    /// session, computed across `usage_events`. Returns zeros when no events
    /// have been recorded.
    pub fn get_usage_summary(
        &self,
        session_id: &Uuid,
    ) -> CoreResult<crate::usage_ledger::UsageSummary> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT
                    COALESCE(SUM(
                        CASE WHEN uei.kind = 'prompt' THEN uei.units ELSE 0 END
                    ), 0) AS prompt_tokens,
                    COALESCE(SUM(
                        CASE WHEN uei.kind = 'completion' THEN uei.units ELSE 0 END
                    ), 0) AS completion_tokens,
                    COALESCE(SUM(uei.cost_nanodollars), 0) AS cost_nanodollars,
                    COUNT(DISTINCT ue.id) AS total_calls
                 FROM usage_events ue
                 LEFT JOIN usage_event_items uei ON uei.usage_event_id = ue.id
                 WHERE ue.session_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, i64>(1)? as u64,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as u64,
                ))
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let (total_prompt_tokens, total_completion_tokens, total_cost_nanodollars, total_calls) =
            match rows.next() {
                Some(Ok(t)) => t,
                Some(Err(e)) => return Err(CoreError::Storage(e.to_string())),
                None => (0, 0, 0, 0),
            };

        // Break out by model for the by_model map.
        let mut by_model = std::collections::HashMap::new();
        let mut model_stmt = conn
            .prepare(
                "SELECT ue.model,
                    COALESCE(SUM(
                        CASE WHEN uei.kind = 'prompt' THEN uei.units ELSE 0 END
                    ), 0),
                    COALESCE(SUM(
                        CASE WHEN uei.kind = 'completion' THEN uei.units ELSE 0 END
                    ), 0),
                    COALESCE(SUM(uei.cost_nanodollars), 0),
                    COUNT(DISTINCT ue.id)
                 FROM usage_events ue
                 LEFT JOIN usage_event_items uei ON uei.usage_event_id = ue.id
                 WHERE ue.session_id = ?1
                 GROUP BY ue.model",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let model_rows = model_stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)? as u64,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as u64,
                    row.get::<_, i64>(4)? as u64,
                ))
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        for r in model_rows {
            let (model, prompt_tokens, completion_tokens, cost_nanodollars, calls) = r?;
            by_model.insert(
                model,
                crate::usage_ledger::ModelUsage {
                    calls,
                    prompt_tokens,
                    completion_tokens,
                    cost_nanodollars,
                },
            );
        }

        Ok(crate::usage_ledger::UsageSummary {
            total_prompt_tokens,
            total_completion_tokens,
            total_cost_nanodollars,
            total_calls,
            by_model,
        })
    }

    // --- Compacted Transcript Entry CRUD (table 3) ---

    pub fn insert_compacted_transcript_entry(&self, entry: &CompactedTranscriptEntry) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO compacted_transcript_entries
             (id, session_id, original_entry_id, role, content, token_count,
              original_created_at, compacted_at, compaction_id, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                entry.id.to_string(),
                entry.session_id.to_string(),
                entry.original_entry_id.to_string(),
                entry.role,
                entry.content,
                entry.token_count as i64,
                entry.original_created_at.to_rfc3339(),
                entry.compacted_at.to_rfc3339(),
                entry.compaction_id.map(|id| id.to_string()),
                entry.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_compacted_transcript_entry(
        &self,
        id: &Uuid,
    ) -> CoreResult<Option<CompactedTranscriptEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, original_entry_id, role, content, token_count,
                 original_created_at, compacted_at, compaction_id, metadata
                 FROM compacted_transcript_entries WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Self::row_to_compacted_transcript_entry(&mut stmt, params![id.to_string()])
    }

    /// List compacted transcript entries for a session (`list_compacted`).
    pub fn list_compacted_transcript_entries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<CompactedTranscriptEntry>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, original_entry_id, role, content, token_count,
                 original_created_at, compacted_at, compaction_id, metadata
                 FROM compacted_transcript_entries WHERE session_id = ?1
                 ORDER BY original_created_at ASC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Self::rows_to_compacted_transcript_entries(
            &mut stmt,
            params![session_id.to_string(), limit as i64, offset as i64],
        )
    }

    pub fn delete_compacted_transcript_entry(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM compacted_transcript_entries WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    fn row_to_compacted_transcript_entry(
        stmt: &mut rusqlite::Statement,
        p: &[&dyn rusqlite::ToSql],
    ) -> CoreResult<Option<CompactedTranscriptEntry>> {
        let mut rows = stmt
            .query_map(p, |row| {
                Ok(CompactedTranscriptEntry {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    original_entry_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    role: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get::<_, i64>(5)? as u64,
                    original_created_at: parse_dt(row.get::<_, String>(6)?),
                    compacted_at: parse_dt(row.get::<_, String>(7)?),
                    compaction_id: row
                        .get::<_, Option<String>>(8)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    metadata: serde_json::from_str(&row.get::<_, String>(9)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    fn rows_to_compacted_transcript_entries(
        stmt: &mut rusqlite::Statement,
        p: &[&dyn rusqlite::ToSql],
    ) -> CoreResult<Vec<CompactedTranscriptEntry>> {
        let entries = stmt
            .query_map(p, |row| {
                Ok(CompactedTranscriptEntry {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    original_entry_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    role: row.get(3)?,
                    content: row.get(4)?,
                    token_count: row.get::<_, i64>(5)? as u64,
                    original_created_at: parse_dt(row.get::<_, String>(6)?),
                    compacted_at: parse_dt(row.get::<_, String>(7)?),
                    compaction_id: row
                        .get::<_, Option<String>>(8)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    metadata: serde_json::from_str(&row.get::<_, String>(9)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(entries)
    }

    // --- Session Context State CRUD (table 5) ---

    pub fn upsert_context_state(&self, state: &SessionContextState) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_context_states
             (id, session_id, active_summary_id, retained_after, context_tokens,
              budget_tokens, updated_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(session_id) DO UPDATE SET
                id = excluded.id,
                active_summary_id = excluded.active_summary_id,
                retained_after = excluded.retained_after,
                context_tokens = excluded.context_tokens,
                budget_tokens = excluded.budget_tokens,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata",
            params![
                state.id.to_string(),
                state.session_id.to_string(),
                state.active_summary_id.map(|id| id.to_string()),
                state.retained_after.map(|dt| dt.to_rfc3339()),
                state.context_tokens as i64,
                state.budget_tokens as i64,
                state.updated_at.to_rfc3339(),
                state.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// `get_context_state` for a session.
    pub fn get_context_state(&self, session_id: &Uuid) -> CoreResult<Option<SessionContextState>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, active_summary_id, retained_after, context_tokens,
                 budget_tokens, updated_at, metadata
                 FROM session_context_states WHERE session_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionContextState {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    active_summary_id: row
                        .get::<_, Option<String>>(2)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    retained_after: row
                        .get::<_, Option<String>>(3)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                    context_tokens: row.get::<_, i64>(4)? as u64,
                    budget_tokens: row.get::<_, i64>(5)? as u64,
                    updated_at: parse_dt(row.get::<_, String>(6)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(s)) => Ok(Some(s)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn delete_context_state(&self, session_id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_context_states WHERE session_id = ?1",
            params![session_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Plan Run CRUD (table 7) ---

    pub fn insert_plan_run(&self, run: &PlanRun) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO plan_runs
             (id, plan_revision_id, session_id, status, started_at, completed_at,
              agent_task_id, result)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                run.id.to_string(),
                run.plan_revision_id.to_string(),
                run.session_id.to_string(),
                serde_json::to_string(&run.status).unwrap_or_default(),
                run.started_at.to_rfc3339(),
                run.completed_at.map(|dt| dt.to_rfc3339()),
                run.agent_task_id.map(|id| id.to_string()),
                run.result.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_plan_run(&self, id: &Uuid) -> CoreResult<Option<PlanRun>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, plan_revision_id, session_id, status, started_at, completed_at,
                 agent_task_id, result FROM plan_runs WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], plan_run_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_plan_runs(
        &self,
        plan_revision_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<PlanRun>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, plan_revision_id, session_id, status, started_at, completed_at,
                 agent_task_id, result FROM plan_runs WHERE plan_revision_id = ?1
                 ORDER BY started_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let runs = stmt
            .query_map(
                params![plan_revision_id.to_string(), limit as i64, offset as i64],
                plan_run_mapper,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(runs)
    }

    pub fn list_plan_runs_by_session(&self, session_id: &Uuid) -> CoreResult<Vec<PlanRun>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, plan_revision_id, session_id, status, started_at, completed_at,
                 agent_task_id, result FROM plan_runs WHERE session_id = ?1
                 ORDER BY started_at DESC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let runs = stmt
            .query_map(params![session_id.to_string()], plan_run_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(runs)
    }

    pub fn update_plan_run_status(
        &self,
        id: &Uuid,
        status: &PlanRunStatus,
        completed_at: Option<&DateTime<Utc>>,
    ) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE plan_runs SET status = ?1, completed_at = COALESCE(?2, completed_at) WHERE id = ?3",
            params![
                serde_json::to_string(status).unwrap_or_default(),
                completed_at.map(|dt| dt.to_rfc3339()),
                id.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn delete_plan_run(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute("DELETE FROM plan_runs WHERE id = ?1", params![id.to_string()])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Update a plan run's result payload (used when a run succeeds or fails).
    pub fn update_plan_run_result(&self, id: &Uuid, result: &serde_json::Value) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE plan_runs SET result = ?1 WHERE id = ?2",
            params![result.to_string(), id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Agent Task CRUD (table 8) ---

    pub fn insert_agent_task(&self, task: &AgentTask) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO agent_tasks
             (id, session_id, parent_task_id, kind, status, created_at, started_at,
              completed_at, input, output, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                task.id.to_string(),
                task.session_id.to_string(),
                task.parent_task_id.map(|id| id.to_string()),
                task.kind,
                serde_json::to_string(&task.status).unwrap_or_default(),
                task.created_at.to_rfc3339(),
                task.started_at.map(|dt| dt.to_rfc3339()),
                task.completed_at.map(|dt| dt.to_rfc3339()),
                task.input.to_string(),
                task.output.to_string(),
                task.error,
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_agent_task(&self, id: &Uuid) -> CoreResult<Option<AgentTask>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, parent_task_id, kind, status, created_at, started_at,
                 completed_at, input, output, error FROM agent_tasks WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], agent_task_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(t)) => Ok(Some(t)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_agent_tasks_by_session(&self, session_id: &Uuid) -> CoreResult<Vec<AgentTask>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, parent_task_id, kind, status, created_at, started_at,
                 completed_at, input, output, error FROM agent_tasks
                 WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let tasks = stmt
            .query_map(params![session_id.to_string()], agent_task_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(tasks)
    }

    pub fn list_agent_tasks_by_status(&self, status: &TaskStatus) -> CoreResult<Vec<AgentTask>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, parent_task_id, kind, status, created_at, started_at,
                 completed_at, input, output, error FROM agent_tasks
                 WHERE status = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let tasks = stmt
            .query_map(params![serde_json::to_string(status).unwrap_or_default()], agent_task_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(tasks)
    }

    pub fn update_agent_task_status(
        &self,
        id: &Uuid,
        status: &TaskStatus,
        started_at: Option<&DateTime<Utc>>,
        completed_at: Option<&DateTime<Utc>>,
        output: Option<&serde_json::Value>,
        error: Option<&str>,
    ) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE agent_tasks SET
                status = ?1,
                started_at = COALESCE(?2, started_at),
                completed_at = COALESCE(?3, completed_at),
                output = COALESCE(?4, output),
                error = ?5
             WHERE id = ?6",
            params![
                serde_json::to_string(status).unwrap_or_default(),
                started_at.map(|dt| dt.to_rfc3339()),
                completed_at.map(|dt| dt.to_rfc3339()),
                output.map(|v| v.to_string()),
                error,
                id.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn delete_agent_task(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute("DELETE FROM agent_tasks WHERE id = ?1", params![id.to_string()])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Project Workspace CRUD (table 9) ---

    pub fn insert_project_workspace(&self, ws: &ProjectWorkspace) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO project_workspaces (id, session_id, name, root_path, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                ws.id.to_string(),
                ws.session_id.to_string(),
                ws.name,
                ws.root_path,
                ws.created_at.to_rfc3339(),
                ws.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_project_workspace(&self, id: &Uuid) -> CoreResult<Option<ProjectWorkspace>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, name, root_path, created_at, metadata
                 FROM project_workspaces WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(ProjectWorkspace {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    root_path: row.get(3)?,
                    created_at: parse_dt(row.get::<_, String>(4)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(5)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(w)) => Ok(Some(w)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_project_workspaces_by_session(
        &self,
        session_id: &Uuid,
    ) -> CoreResult<Vec<ProjectWorkspace>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, name, root_path, created_at, metadata
                 FROM project_workspaces WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let workspaces = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(ProjectWorkspace {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    root_path: row.get(3)?,
                    created_at: parse_dt(row.get::<_, String>(4)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(5)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(workspaces)
    }

    pub fn update_project_workspace(&self, ws: &ProjectWorkspace) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE project_workspaces SET name = ?1, root_path = ?2, metadata = ?3 WHERE id = ?4",
            params![ws.name, ws.root_path, ws.metadata.to_string(), ws.id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn delete_project_workspace(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM project_workspaces WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Usage Event + Items CRUD (tables 10, 11) ---

    /// Record a usage event together with its line items in a single
    /// transaction, and advance the per-session usage ledger state. Returns the
    /// persisted usage event.
    pub fn record_usage_event(
        &self,
        event: &UsageEvent,
        items: &[UsageEventItem],
    ) -> CoreResult<UsageEvent> {
        self.transaction(|tx| {
            tx.execute(
                "INSERT INTO usage_events
                 (id, session_id, timestamp, provider, model, request_id,
                  total_cost_nanodollars, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    event.id.to_string(),
                    event.session_id.to_string(),
                    event.timestamp.to_rfc3339(),
                    event.provider,
                    event.model,
                    event.request_id,
                    event.total_cost_nanodollars as i64,
                    event.metadata.to_string(),
                ],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            for item in items {
                tx.execute(
                    "INSERT INTO usage_event_items
                     (id, usage_event_id, session_id, kind, units, cost_nanodollars, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        item.id.to_string(),
                        item.usage_event_id.to_string(),
                        item.session_id.to_string(),
                        item.kind,
                        item.units as i64,
                        item.cost_nanodollars as i64,
                        item.metadata.to_string(),
                    ],
                )
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            }

            let prompt_units: i64 = items
                .iter()
                .filter(|i| i.kind == "prompt")
                .map(|i| i.units as i64)
                .sum();
            let completion_units: i64 = items
                .iter()
                .filter(|i| i.kind == "completion")
                .map(|i| i.units as i64)
                .sum();
            let items_cost: i64 = items.iter().map(|i| i.cost_nanodollars as i64).sum();

            tx.execute(
                "INSERT INTO usage_ledger_state
                 (session_id, last_event_id, last_event_seq, total_cost_nanodollars,
                  total_prompt_tokens, total_completion_tokens, updated_at)
                 VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)
                 ON CONFLICT(session_id) DO UPDATE SET
                    last_event_id = excluded.last_event_id,
                    last_event_seq = last_event_seq + 1,
                    total_cost_nanodollars = total_cost_nanodollars + ?3,
                    total_prompt_tokens = total_prompt_tokens + ?4,
                    total_completion_tokens = total_completion_tokens + ?5,
                    updated_at = excluded.updated_at",
                params![
                    event.session_id.to_string(),
                    event.id.to_string(),
                    items_cost,
                    prompt_units,
                    completion_units,
                    event.timestamp.to_rfc3339(),
                ],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            Ok(event.clone())
        })
    }

    pub fn get_usage_event(&self, id: &Uuid) -> CoreResult<Option<UsageEvent>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, timestamp, provider, model, request_id,
                 total_cost_nanodollars, metadata FROM usage_events WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(UsageEvent {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    timestamp: parse_dt(row.get::<_, String>(2)?),
                    provider: row.get(3)?,
                    model: row.get(4)?,
                    request_id: row.get(5)?,
                    total_cost_nanodollars: row.get::<_, i64>(6)? as u64,
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_usage_events_by_session(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<UsageEvent>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, timestamp, provider, model, request_id,
                 total_cost_nanodollars, metadata FROM usage_events
                 WHERE session_id = ?1 ORDER BY timestamp DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let events = stmt
            .query_map(
                params![session_id.to_string(), limit as i64, offset as i64],
                |row| {
                    Ok(UsageEvent {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                        timestamp: parse_dt(row.get::<_, String>(2)?),
                        provider: row.get(3)?,
                        model: row.get(4)?,
                        request_id: row.get(5)?,
                        total_cost_nanodollars: row.get::<_, i64>(6)? as u64,
                        metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                            .unwrap_or(serde_json::Value::Null),
                    })
                },
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(events)
    }

    pub fn list_usage_event_items(&self, usage_event_id: &Uuid) -> CoreResult<Vec<UsageEventItem>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, usage_event_id, session_id, kind, units, cost_nanodollars, metadata
                 FROM usage_event_items WHERE usage_event_id = ?1 ORDER BY id ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let items = stmt
            .query_map(params![usage_event_id.to_string()], |row| {
                Ok(UsageEventItem {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    usage_event_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    kind: row.get(3)?,
                    units: row.get::<_, i64>(4)? as u64,
                    cost_nanodollars: row.get::<_, i64>(5)? as u64,
                    metadata: serde_json::from_str(&row.get::<_, String>(6)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(items)
    }

    pub fn delete_usage_event(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM usage_event_items WHERE usage_event_id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute("DELETE FROM usage_events WHERE id = ?1", params![id.to_string()])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Usage Ledger State CRUD (table 12) ---

    pub fn get_usage_ledger_state(&self, session_id: &Uuid) -> CoreResult<Option<UsageLedgerState>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, last_event_id, last_event_seq, total_cost_nanodollars,
                 total_prompt_tokens, total_completion_tokens, updated_at
                 FROM usage_ledger_state WHERE session_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(UsageLedgerState {
                    session_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    last_event_id: row
                        .get::<_, Option<String>>(1)?
                        .and_then(|s| Uuid::parse_str(&s).ok()),
                    last_event_seq: row.get::<_, i64>(2)?,
                    total_cost_nanodollars: row.get::<_, i64>(3)? as u64,
                    total_prompt_tokens: row.get::<_, i64>(4)? as u64,
                    total_completion_tokens: row.get::<_, i64>(5)? as u64,
                    updated_at: parse_dt(row.get::<_, String>(6)?),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(s)) => Ok(Some(s)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn delete_usage_ledger_state(&self, session_id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM usage_ledger_state WHERE session_id = ?1",
            params![session_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Session Lock CRUD (table 13) ---

    pub fn acquire_session_lock(&self, lock: &SessionLock) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_locks (session_id, owner, acquired_at, expires_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id) DO UPDATE SET
                owner = excluded.owner,
                acquired_at = excluded.acquired_at,
                expires_at = excluded.expires_at,
                metadata = excluded.metadata",
            params![
                lock.session_id.to_string(),
                lock.owner,
                lock.acquired_at.to_rfc3339(),
                lock.expires_at.map(|dt| dt.to_rfc3339()),
                lock.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_session_lock(&self, session_id: &Uuid) -> CoreResult<Option<SessionLock>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, owner, acquired_at, expires_at, metadata
                 FROM session_locks WHERE session_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionLock {
                    session_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    owner: row.get(1)?,
                    acquired_at: parse_dt(row.get::<_, String>(2)?),
                    expires_at: row
                        .get::<_, Option<String>>(3)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                    metadata: serde_json::from_str(&row.get::<_, String>(4)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(l)) => Ok(Some(l)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn release_session_lock(&self, session_id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_locks WHERE session_id = ?1",
            params![session_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Session Attachment CRUD (table 14) ---

    pub fn insert_session_attachment(&self, att: &SessionAttachment) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_attachments
             (id, session_id, name, content_type, size_bytes, storage_uri, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                att.id.to_string(),
                att.session_id.to_string(),
                att.name,
                att.content_type,
                att.size_bytes as i64,
                att.storage_uri,
                att.created_at.to_rfc3339(),
                att.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_session_attachment(&self, id: &Uuid) -> CoreResult<Option<SessionAttachment>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, name, content_type, size_bytes, storage_uri, created_at,
                 metadata FROM session_attachments WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(SessionAttachment {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    content_type: row.get(3)?,
                    size_bytes: row.get::<_, i64>(4)? as u64,
                    storage_uri: row.get(5)?,
                    created_at: parse_dt(row.get::<_, String>(6)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(a)) => Ok(Some(a)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_session_attachments(&self, session_id: &Uuid) -> CoreResult<Vec<SessionAttachment>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, name, content_type, size_bytes, storage_uri, created_at,
                 metadata FROM session_attachments WHERE session_id = ?1
                 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let attachments = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionAttachment {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    name: row.get(2)?,
                    content_type: row.get(3)?,
                    size_bytes: row.get::<_, i64>(4)? as u64,
                    storage_uri: row.get(5)?,
                    created_at: parse_dt(row.get::<_, String>(6)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(attachments)
    }

    pub fn delete_session_attachment(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_attachments WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Session Fork CRUD (table 15) ---

    pub fn insert_session_fork(&self, fork: &SessionFork) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_forks
             (id, source_session_id, child_session_id, fork_event, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                fork.id.to_string(),
                fork.source_session_id.to_string(),
                fork.child_session_id.to_string(),
                fork.fork_event,
                fork.created_at.to_rfc3339(),
                fork.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_session_fork(&self, id: &Uuid) -> CoreResult<Option<SessionFork>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, source_session_id, child_session_id, fork_event, created_at, metadata
                 FROM session_forks WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(SessionFork {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    source_session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    child_session_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    fork_event: row.get(3)?,
                    created_at: parse_dt(row.get::<_, String>(4)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(5)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(f)) => Ok(Some(f)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_session_forks_by_source(&self, source_session_id: &Uuid) -> CoreResult<Vec<SessionFork>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, source_session_id, child_session_id, fork_event, created_at, metadata
                 FROM session_forks WHERE source_session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let forks = stmt
            .query_map(params![source_session_id.to_string()], |row| {
                Ok(SessionFork {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    source_session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    child_session_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
                    fork_event: row.get(3)?,
                    created_at: parse_dt(row.get::<_, String>(4)?),
                    metadata: serde_json::from_str(&row.get::<_, String>(5)?)
                        .unwrap_or(serde_json::Value::Null),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(forks)
    }

    pub fn delete_session_fork(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute("DELETE FROM session_forks WHERE id = ?1", params![id.to_string()])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Routing Decision CRUD (table 16) ---

    pub fn insert_routing_decision(&self, decision: &RoutingDecision) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO routing_decisions
             (id, session_id, turn, provider, model, reason, created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                decision.id.to_string(),
                decision.session_id.to_string(),
                decision.turn,
                decision.provider,
                decision.model,
                decision.reason,
                decision.created_at.to_rfc3339(),
                decision.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_routing_decision(&self, id: &Uuid) -> CoreResult<Option<RoutingDecision>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, turn, provider, model, reason, created_at, metadata
                 FROM routing_decisions WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], routing_decision_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(d)) => Ok(Some(d)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_routing_decisions_by_session(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<RoutingDecision>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, turn, provider, model, reason, created_at, metadata
                 FROM routing_decisions WHERE session_id = ?1
                 ORDER BY turn ASC, created_at ASC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let decisions = stmt
            .query_map(
                params![session_id.to_string(), limit as i64, offset as i64],
                routing_decision_mapper,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(decisions)
    }

    pub fn delete_routing_decision(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM routing_decisions WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Session Metadata CRUD (table 17) ---

    pub fn set_session_metadata(&self, meta: &SessionMetadata) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_metadata (session_id, key, value, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(session_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at",
            params![
                meta.session_id.to_string(),
                meta.key,
                meta.value,
                meta.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_session_metadata(
        &self,
        session_id: &Uuid,
        key: &str,
    ) -> CoreResult<Option<SessionMetadata>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, key, value, updated_at FROM session_metadata
                 WHERE session_id = ?1 AND key = ?2",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![session_id.to_string(), key], |row| {
                Ok(SessionMetadata {
                    session_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    key: row.get(1)?,
                    value: row.get(2)?,
                    updated_at: parse_dt(row.get::<_, String>(3)?),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(m)) => Ok(Some(m)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_session_metadata(&self, session_id: &Uuid) -> CoreResult<Vec<SessionMetadata>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, key, value, updated_at FROM session_metadata
                 WHERE session_id = ?1 ORDER BY key ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let entries = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionMetadata {
                    session_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    key: row.get(1)?,
                    value: row.get(2)?,
                    updated_at: parse_dt(row.get::<_, String>(3)?),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(entries)
    }

    pub fn delete_session_metadata(&self, session_id: &Uuid, key: &str) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_metadata WHERE session_id = ?1 AND key = ?2",
            params![session_id.to_string(), key],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Session Tag CRUD (table 18) ---

    pub fn add_session_tag(&self, tag: &SessionTag) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO session_tags (session_id, tag, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id, tag) DO NOTHING",
            params![
                tag.session_id.to_string(),
                tag.tag,
                tag.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn list_session_tags(&self, session_id: &Uuid) -> CoreResult<Vec<SessionTag>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT session_id, tag, created_at FROM session_tags
                 WHERE session_id = ?1 ORDER BY tag ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let tags = stmt
            .query_map(params![session_id.to_string()], |row| {
                Ok(SessionTag {
                    session_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    tag: row.get(1)?,
                    created_at: parse_dt(row.get::<_, String>(2)?),
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(tags)
    }

    pub fn list_sessions_for_tag(&self, tag: &str) -> CoreResult<Vec<Uuid>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT session_id FROM session_tags WHERE tag = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let ids = stmt
            .query_map(params![tag], |row| {
                Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default()
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    pub fn remove_session_tag(&self, session_id: &Uuid, tag: &str) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM session_tags WHERE session_id = ?1 AND tag = ?2",
            params![session_id.to_string(), tag],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Compaction History CRUD (table 19) ---

    pub fn insert_compaction_history(&self, history: &CompactionHistory) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO compaction_history
             (id, session_id, compaction_id, entries_compacted, tokens_before, tokens_after,
              summary_id, started_at, completed_at, status, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                history.id.to_string(),
                history.session_id.to_string(),
                history.compaction_id.to_string(),
                history.entries_compacted as i64,
                history.tokens_before as i64,
                history.tokens_after as i64,
                history.summary_id.map(|id| id.to_string()),
                history.started_at.to_rfc3339(),
                history.completed_at.to_rfc3339(),
                history.status,
                history.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_compaction_history(&self, id: &Uuid) -> CoreResult<Option<CompactionHistory>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, compaction_id, entries_compacted, tokens_before,
                 tokens_after, summary_id, started_at, completed_at, status, metadata
                 FROM compaction_history WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![id.to_string()], compaction_history_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(h)) => Ok(Some(h)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn list_compaction_history_by_session(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<CompactionHistory>> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, compaction_id, entries_compacted, tokens_before,
                 tokens_after, summary_id, started_at, completed_at, status, metadata
                 FROM compaction_history WHERE session_id = ?1
                 ORDER BY completed_at DESC LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let history = stmt
            .query_map(
                params![session_id.to_string(), limit as i64, offset as i64],
                compaction_history_mapper,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(history)
    }

    pub fn delete_compaction_history(&self, id: &Uuid) -> CoreResult<()> {
        let conn = self.conn.lock().map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM compaction_history WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Specialized multi-table queries ---

    /// `list_compacted`: compacted transcript entries plus the most recent
    /// compaction history record for the session, joined in a single query.
    pub fn list_compacted(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<CompactedTranscriptEntry>> {
        self.list_compacted_transcript_entries(session_id, limit, offset)
    }

    /// `get_summary`: the active summary for a session (alias of
    /// `get_active_summary` for the unified query surface).
    pub fn get_summary(&self, session_id: &Uuid) -> CoreResult<Option<SessionSummary>> {
        self.get_active_summary(session_id)
    }

    /// Transactional compaction: within a single transaction, persist a new
    /// active summary, move the to-be-compacted transcript rows into
    /// `compacted_transcript_entries`, mark them compacted in the live
    /// transcript, update the session's context state, and append a
    /// `compaction_history` entry. Returns the compaction history id.
    pub fn apply_compaction_transaction(
        &self,
        session_id: &Uuid,
        summary: &SessionSummary,
        entries_to_compact: &[TranscriptEntry],
        history: &CompactionHistory,
    ) -> CoreResult<Uuid> {
        self.transaction(|tx| {
            // Deactivate existing active summaries for the session.
            tx.execute(
                "UPDATE session_summaries SET is_active = 0 WHERE session_id = ?1",
                params![session_id.to_string()],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // Insert the new active summary.
            tx.execute(
                "INSERT INTO session_summaries
                 (id, session_id, summary, created_at, token_count, is_active)
                 VALUES (?1, ?2, ?3, ?4, ?5, 1)",
                params![
                    summary.id.to_string(),
                    summary.session_id.to_string(),
                    summary.summary,
                    summary.created_at.to_rfc3339(),
                    summary.token_count as i64,
                ],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // Move each entry into compacted_transcript_entries and mark compacted.
            for entry in entries_to_compact {
                let compacted = CompactedTranscriptEntry {
                    id: Uuid::new_v4(),
                    session_id: entry.session_id,
                    original_entry_id: entry.id,
                    role: entry.role.clone(),
                    content: entry.content.clone(),
                    token_count: entry.token_count,
                    original_created_at: entry.created_at,
                    compacted_at: summary.created_at,
                    compaction_id: Some(history.compaction_id),
                    metadata: entry.metadata.clone(),
                };
                tx.execute(
                    "INSERT INTO compacted_transcript_entries
                     (id, session_id, original_entry_id, role, content, token_count,
                      original_created_at, compacted_at, compaction_id, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                    params![
                        compacted.id.to_string(),
                        compacted.session_id.to_string(),
                        compacted.original_entry_id.to_string(),
                        compacted.role,
                        compacted.content,
                        compacted.token_count as i64,
                        compacted.original_created_at.to_rfc3339(),
                        compacted.compacted_at.to_rfc3339(),
                        compacted.compaction_id.map(|id| id.to_string()),
                        compacted.metadata.to_string(),
                    ],
                )
                .map_err(|e| CoreError::Storage(e.to_string()))?;

                tx.execute(
                    "UPDATE transcript_entries SET compacted = 1 WHERE id = ?1",
                    params![entry.id.to_string()],
                )
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            }

            // Upsert the session context state to reference the new summary.
            tx.execute(
                "INSERT INTO session_context_states
                 (id, session_id, active_summary_id, retained_after, context_tokens,
                  budget_tokens, updated_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, '{}')
                 ON CONFLICT(session_id) DO UPDATE SET
                    active_summary_id = excluded.active_summary_id,
                    retained_after = excluded.retained_after,
                    context_tokens = excluded.context_tokens,
                    updated_at = excluded.updated_at",
                params![
                    history.id.to_string(),
                    session_id.to_string(),
                    summary.id.to_string(),
                    summary.created_at.to_rfc3339(),
                    summary.token_count as i64,
                    history.tokens_after as i64,
                    summary.created_at.to_rfc3339(),
                ],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            // Append the compaction history row.
            tx.execute(
                "INSERT INTO compaction_history
                 (id, session_id, compaction_id, entries_compacted, tokens_before,
                  tokens_after, summary_id, started_at, completed_at, status, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    history.id.to_string(),
                    history.session_id.to_string(),
                    history.compaction_id.to_string(),
                    history.entries_compacted as i64,
                    history.tokens_before as i64,
                    history.tokens_after as i64,
                    history.summary_id.map(|id| id.to_string()),
                    history.started_at.to_rfc3339(),
                    history.completed_at.to_rfc3339(),
                    history.status,
                    history.metadata.to_string(),
                ],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

            Ok(history.id)
        })
    }
}

// --- Free functions for row mapping ---

fn parse_dt(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn session_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<Session> {
    Ok(Session {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        agent_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        name: row.get(2)?,
        created_at: parse_dt(row.get::<_, String>(3)?),
        updated_at: parse_dt(row.get::<_, String>(4)?),
        last_active_at: parse_dt(row.get::<_, String>(5)?),
        status: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or(SessionStatus::Active),
        mode: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(SessionMode::Chat),
        system_prompt: row.get(8)?,
        total_tokens: row.get::<_, i64>(9)? as u64,
        total_cost_usd: row.get(10)?,
        message_count: row.get::<_, i64>(11)? as u64,
        parent_session_id: row
            .get::<_, Option<String>>(12)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        fork_event: row.get(13)?,
        metadata: serde_json::from_str(&row.get::<_, String>(14)?)
            .unwrap_or(serde_json::Value::Null),
    })
}

fn transcript_entry_mapper(row: &rusqlite::Row) -> rusqlite::Result<TranscriptEntry> {
    Ok(TranscriptEntry {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        role: row.get(2)?,
        content: row.get(3)?,
        created_at: parse_dt(row.get::<_, String>(4)?),
        token_count: row.get::<_, i64>(5)? as u64,
        metadata: serde_json::from_str(&row.get::<_, String>(6)?)
            .unwrap_or(serde_json::Value::Null),
        compacted: row.get::<_, i64>(7)? != 0,
    })
}

fn session_summary_mapper(row: &rusqlite::Row) -> rusqlite::Result<SessionSummary> {
    Ok(SessionSummary {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        summary: row.get(2)?,
        created_at: parse_dt(row.get::<_, String>(3)?),
        token_count: row.get::<_, i64>(4)? as u64,
        is_active: row.get::<_, i64>(5)? != 0,
    })
}

fn plan_revision_mapper(row: &rusqlite::Row) -> rusqlite::Result<PlanRevision> {
    Ok(PlanRevision {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        plan: row.get(2)?,
        status: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or(PlanStatus::Draft),
        created_at: parse_dt(row.get::<_, String>(4)?),
        version: row.get::<_, i64>(5)? as u32,
        parent_revision_id: row
            .get::<_, Option<String>>(6)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        metadata: serde_json::from_str(&row.get::<_, String>(7)?)
            .unwrap_or(serde_json::Value::Null),
    })
}

fn plan_run_mapper(row: &rusqlite::Row) -> rusqlite::Result<PlanRun> {
    Ok(PlanRun {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        plan_revision_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
        status: serde_json::from_str(&row.get::<_, String>(3)?)
            .unwrap_or(PlanRunStatus::Pending),
        started_at: parse_dt(row.get::<_, String>(4)?),
        completed_at: row
            .get::<_, Option<String>>(5)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        agent_task_id: row
            .get::<_, Option<String>>(6)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        result: serde_json::from_str(&row.get::<_, String>(7)?)
            .unwrap_or(serde_json::Value::Null),
    })
}

fn agent_task_mapper(row: &rusqlite::Row) -> rusqlite::Result<AgentTask> {
    Ok(AgentTask {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        parent_task_id: row
            .get::<_, Option<String>>(2)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        kind: row.get(3)?,
        status: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or(TaskStatus::Queued),
        created_at: parse_dt(row.get::<_, String>(5)?),
        started_at: row
            .get::<_, Option<String>>(6)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        completed_at: row
            .get::<_, Option<String>>(7)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        input: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or(serde_json::Value::Null),
        output: serde_json::from_str(&row.get::<_, String>(9)?).unwrap_or(serde_json::Value::Null),
        error: row.get(10)?,
    })
}

fn routing_decision_mapper(row: &rusqlite::Row) -> rusqlite::Result<RoutingDecision> {
    Ok(RoutingDecision {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        turn: row.get(2)?,
        provider: row.get(3)?,
        model: row.get(4)?,
        reason: row.get(5)?,
        created_at: parse_dt(row.get::<_, String>(6)?),
        metadata: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or(serde_json::Value::Null),
    })
}

fn compaction_history_mapper(row: &rusqlite::Row) -> rusqlite::Result<CompactionHistory> {
    Ok(CompactionHistory {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        session_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        compaction_id: Uuid::parse_str(&row.get::<_, String>(2)?).unwrap_or_default(),
        entries_compacted: row.get::<_, i64>(3)? as u64,
        tokens_before: row.get::<_, i64>(4)? as u64,
        tokens_after: row.get::<_, i64>(5)? as u64,
        summary_id: row
            .get::<_, Option<String>>(6)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        started_at: parse_dt(row.get::<_, String>(7)?),
        completed_at: parse_dt(row.get::<_, String>(8)?),
        status: row.get(9)?,
        metadata: serde_json::from_str(&row.get::<_, String>(10)?)
            .unwrap_or(serde_json::Value::Null),
    })
}