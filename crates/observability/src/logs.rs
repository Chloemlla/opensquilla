//! Structured per-turn log tables.
//!
//! Sqlite-backed port of the Python observability log streams:
//! `decision_log`, `safety_log`, `turn_call_log`, `prompt_report`,
//! `tool_result_log`, `compaction_log`, `cost_log`. Each table mirrors the
//! Python dataclass shape (column names match the JSONL field names) and is
//! written once per triggering event then queried by session_id/turn_id.

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use tracing::info;

// ---------------------------------------------------------------------------
// decision_log — one row per completed turn (DecisionEntry in Python)
// ---------------------------------------------------------------------------

/// One row in the `decision_log` table (router/agent decision record).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLogRow {
    pub id: String,
    pub turn_id: String,
    pub session_key: String,
    pub prompt_hash: String,
    pub system_prompt_hash: String,
    pub tool_list_hash: String,
    pub tool_choice: String,
    pub tokens_input: i64,
    pub tokens_output: i64,
    pub model: String,
    pub provider: String,
    pub latency_ms: i64,
    pub ts: String,
    pub session_id: Option<String>,
    pub session_intent: Option<String>,
    pub intent_summary: Option<String>,
    pub trace_id: Option<String>,
    pub decision_id: Option<String>,
    pub tool_profile: Option<String>,
    pub system_chars: i64,
    pub tool_count: i64,
    pub tools_schema_chars: i64,
    pub skill_count: i64,
    pub skills_prompt_chars: i64,
    pub memory_md_present: bool,
    pub schema_version: i64,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// safety_log — safety/injection events (SafetyEvent in Python)
// ---------------------------------------------------------------------------

/// Closed enum of safety-event categories (mirrors Python SafetyEventType).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyEventType {
    RefusedTool,
    TruncatedOutput,
    RateLimit,
    InjectionBlocked,
    TierDenied,
    SandboxViolation,
}

impl SafetyEventType {
    /// Stable snake_case code stored in the `event_type` column.
    pub fn as_code(self) -> &'static str {
        match self {
            SafetyEventType::RefusedTool => "refused_tool",
            SafetyEventType::TruncatedOutput => "truncated_output",
            SafetyEventType::RateLimit => "rate_limit",
            SafetyEventType::InjectionBlocked => "injection_blocked",
            SafetyEventType::TierDenied => "tier_denied",
            SafetyEventType::SandboxViolation => "sandbox_violation",
        }
    }

    /// Parse a stored code back into the enum; unknown codes map to None.
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "refused_tool" => Some(SafetyEventType::RefusedTool),
            "truncated_output" => Some(SafetyEventType::TruncatedOutput),
            "rate_limit" => Some(SafetyEventType::RateLimit),
            "injection_blocked" => Some(SafetyEventType::InjectionBlocked),
            "tier_denied" => Some(SafetyEventType::TierDenied),
            "sandbox_violation" => Some(SafetyEventType::SandboxViolation),
            _ => None,
        }
    }
}

/// One row in the `safety_log` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyLogRow {
    pub id: String,
    pub event_type: String,
    pub session_id: String,
    pub reason: String,
    pub ts: String,
    pub tool_name: Option<String>,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// turn_call_log — per-tool-call invocation log (TurnCallLogger record)
// ---------------------------------------------------------------------------

/// One row in the `turn_call_log` table (raw per-turn call audit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCallLogRow {
    pub id: String,
    pub schema_version: i64,
    pub ts: String,
    pub privacy: String,
    pub trace_id: String,
    pub seq: i64,
    pub turn_id: String,
    pub session_key: String,
    pub session_id: Option<String>,
    pub session_intent: Option<String>,
    pub agent_id: String,
    pub provider: String,
    pub model: String,
    pub source: String,
    pub kind: String,
    pub payload: String,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// prompt_report — prompt assembly report (PromptReport in Python)
// ---------------------------------------------------------------------------

/// Per-tool prompt/schema footprint (ToolEntry in Python).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEntry {
    pub name: String,
    pub summary_chars: i64,
    pub schema_chars: i64,
    pub properties_count: Option<i64>,
}

/// One row in the `prompt_report` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptReportRow {
    pub id: String,
    pub turn_id: String,
    pub session_key: String,
    pub session_id: Option<String>,
    pub agent_id: String,
    pub system_chars: i64,
    pub system_hash: String,
    pub tool_count: i64,
    pub tool_profile: Option<String>,
    pub tools_schema_chars: i64,
    pub skill_count: i64,
    pub skills_prompt_chars: i64,
    pub memory_md_present: bool,
    pub daily_notes_omitted: bool,
    pub daily_notes_count_before_omit: i64,
    pub daily_notes_policy_reason: Option<String>,
    pub injected_workspace_files_count: i64,
    pub retrieval_mode: Option<String>,
    pub cache_mode: Option<String>,
    pub cache_base_hash: Option<String>,
    pub cache_dynamic_hash: Option<String>,
    pub cache_legacy_hash: Option<String>,
    pub cache_shadow_final_hash: Option<String>,
    pub cache_key_collision: bool,
    pub resolved_model: Option<String>,
    pub provider_after_rewrite: Option<String>,
    pub reasoning_hint_resolved: Option<String>,
    pub cache_base_chars: i64,
    pub cache_dynamic_chars: i64,
    pub tool_entries: String,
    pub schema_version: i64,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// tool_result_log — tool result compression/reduction stats
//                  (ToolResultRecord in Python)
// ---------------------------------------------------------------------------

/// One row in the `tool_result_log` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultLogRow {
    pub id: String,
    pub handle: String,
    pub tool_use_id: String,
    pub tool_name: String,
    pub session_id: String,
    pub session_key: String,
    pub agent_id: String,
    pub sha256: String,
    pub chars: i64,
    pub size_bytes: i64,
    pub stored_size_bytes: Option<i64>,
    pub storage_encoding: String,
    pub created_at_text: String,
    pub recorded_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// compaction_log — compaction events (CompactionReport in Python)
// ---------------------------------------------------------------------------

/// One row in the `compaction_log` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionLogRow {
    pub id: String,
    pub session_id: Option<String>,
    pub session_key: Option<String>,
    pub compaction_id: Option<String>,
    pub trigger_reason: Option<String>,
    pub tokens_before: Option<i64>,
    pub tokens_after: Option<i64>,
    pub removed_count: i64,
    pub kept_count: i64,
    pub chunk_count: i64,
    pub summary_source: String,
    pub flush_receipt_status: String,
    pub coverage_status: String,
    pub state_kind: String,
    pub provider_state_valid: Option<bool>,
    pub persisted_summary_id: Option<i64>,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// cost_log — per-turn cost records (SavingsTelemetry + DecisionEntry fields)
// ---------------------------------------------------------------------------

/// One row in the `cost_log` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostLogRow {
    pub id: String,
    pub turn_id: String,
    pub session_id: Option<String>,
    pub session_key: String,
    pub model: String,
    pub provider: String,
    pub tokens_input: i64,
    pub tokens_output: i64,
    pub latency_ms: i64,
    pub routed_model: Option<String>,
    pub baseline_model: Option<String>,
    pub routing_confidence: Option<f64>,
    pub routing_savings_pct: Option<f64>,
    pub routing_savings_usd_estimated_vs_baseline: Option<f64>,
    pub tool_projection_applied: bool,
    pub tool_projection_tokens_saved: i64,
    pub cache_hit_active: bool,
    pub cache_hit_tokens_saved: i64,
    pub cache_hit_usd_estimated_vs_baseline: Option<f64>,
    pub billed_cost_usd: Option<f64>,
    pub cost_usd: Option<f64>,
    pub cost_source: Option<String>,
    pub total_savings_pct: Option<f64>,
    pub total_savings_usd: Option<f64>,
    pub ts: String,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// LogStore — sqlite-backed store for all structured log tables
// ---------------------------------------------------------------------------

/// Sqlite-backed store for the seven structured log tables.
pub struct LogStore {
    conn: Arc<Mutex<Connection>>,
}

impl LogStore {
    /// Open or create a log store at `path`.
    pub fn new(path: &str) -> CoreResult<Self> {
        let conn = Connection::open(path).map_err(|e| CoreError::Storage(e.to_string()))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize_tables()?;
        info!("Observability log store initialized at {path}");
        Ok(store)
    }

    /// Create an in-memory log store (used by tests and ephemeral runs).
    pub fn in_memory() -> CoreResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| CoreError::Storage(e.to_string()))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    fn initialize_tables(&self) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS decision_log (
                id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                system_prompt_hash TEXT NOT NULL,
                tool_list_hash TEXT NOT NULL,
                tool_choice TEXT NOT NULL,
                tokens_input INTEGER NOT NULL,
                tokens_output INTEGER NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                ts TEXT NOT NULL,
                session_id TEXT,
                session_intent TEXT,
                intent_summary TEXT,
                trace_id TEXT,
                decision_id TEXT,
                tool_profile TEXT,
                system_chars INTEGER NOT NULL DEFAULT 0,
                tool_count INTEGER NOT NULL DEFAULT 0,
                tools_schema_chars INTEGER NOT NULL DEFAULT 0,
                skill_count INTEGER NOT NULL DEFAULT 0,
                skills_prompt_chars INTEGER NOT NULL DEFAULT 0,
                memory_md_present INTEGER NOT NULL DEFAULT 0,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decision_log_turn
                ON decision_log(turn_id);
            CREATE INDEX IF NOT EXISTS idx_decision_log_session
                ON decision_log(session_id, created_at);

            CREATE TABLE IF NOT EXISTS safety_log (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                ts TEXT NOT NULL,
                tool_name TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_safety_log_session
                ON safety_log(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_safety_log_type
                ON safety_log(event_type, created_at);

            CREATE TABLE IF NOT EXISTS turn_call_log (
                id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                ts TEXT NOT NULL,
                privacy TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                turn_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT,
                session_intent TEXT,
                agent_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turn_call_log_turn
                ON turn_call_log(turn_id, seq);
            CREATE INDEX IF NOT EXISTS idx_turn_call_log_session
                ON turn_call_log(session_id, created_at);

            CREATE TABLE IF NOT EXISTS prompt_report (
                id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                session_id TEXT,
                agent_id TEXT NOT NULL,
                system_chars INTEGER NOT NULL DEFAULT 0,
                system_hash TEXT NOT NULL DEFAULT '',
                tool_count INTEGER NOT NULL DEFAULT 0,
                tool_profile TEXT,
                tools_schema_chars INTEGER NOT NULL DEFAULT 0,
                skill_count INTEGER NOT NULL DEFAULT 0,
                skills_prompt_chars INTEGER NOT NULL DEFAULT 0,
                memory_md_present INTEGER NOT NULL DEFAULT 0,
                daily_notes_omitted INTEGER NOT NULL DEFAULT 0,
                daily_notes_count_before_omit INTEGER NOT NULL DEFAULT 0,
                daily_notes_policy_reason TEXT,
                injected_workspace_files_count INTEGER NOT NULL DEFAULT 0,
                retrieval_mode TEXT,
                cache_mode TEXT,
                cache_base_hash TEXT,
                cache_dynamic_hash TEXT,
                cache_legacy_hash TEXT,
                cache_shadow_final_hash TEXT,
                cache_key_collision INTEGER NOT NULL DEFAULT 0,
                resolved_model TEXT,
                provider_after_rewrite TEXT,
                reasoning_hint_resolved TEXT,
                cache_base_chars INTEGER NOT NULL DEFAULT 0,
                cache_dynamic_chars INTEGER NOT NULL DEFAULT 0,
                tool_entries TEXT NOT NULL DEFAULT '[]',
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prompt_report_turn
                ON prompt_report(turn_id);
            CREATE INDEX IF NOT EXISTS idx_prompt_report_session
                ON prompt_report(session_id, created_at);

            CREATE TABLE IF NOT EXISTS tool_result_log (
                id TEXT PRIMARY KEY,
                handle TEXT NOT NULL,
                tool_use_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                session_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                chars INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                stored_size_bytes INTEGER,
                storage_encoding TEXT NOT NULL DEFAULT 'utf-8',
                created_at_text TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_result_log_handle
                ON tool_result_log(handle);
            CREATE INDEX IF NOT EXISTS idx_tool_result_log_session
                ON tool_result_log(session_id, recorded_at);

            CREATE TABLE IF NOT EXISTS compaction_log (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                session_key TEXT,
                compaction_id TEXT,
                trigger_reason TEXT,
                tokens_before INTEGER,
                tokens_after INTEGER,
                removed_count INTEGER NOT NULL DEFAULT 0,
                kept_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                summary_source TEXT NOT NULL DEFAULT 'unknown',
                flush_receipt_status TEXT NOT NULL DEFAULT 'unknown',
                coverage_status TEXT NOT NULL DEFAULT 'unknown',
                state_kind TEXT NOT NULL DEFAULT 'structured_summary_v1',
                provider_state_valid INTEGER,
                persisted_summary_id INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compaction_log_session
                ON compaction_log(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_compaction_log_compaction
                ON compaction_log(compaction_id);

            CREATE TABLE IF NOT EXISTS cost_log (
                id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                session_id TEXT,
                session_key TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                tokens_input INTEGER NOT NULL,
                tokens_output INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                routed_model TEXT,
                baseline_model TEXT,
                routing_confidence REAL,
                routing_savings_pct REAL,
                routing_savings_usd_estimated_vs_baseline REAL,
                tool_projection_applied INTEGER NOT NULL DEFAULT 0,
                tool_projection_tokens_saved INTEGER NOT NULL DEFAULT 0,
                cache_hit_active INTEGER NOT NULL DEFAULT 0,
                cache_hit_tokens_saved INTEGER NOT NULL DEFAULT 0,
                cache_hit_usd_estimated_vs_baseline REAL,
                billed_cost_usd REAL,
                cost_usd REAL,
                cost_source TEXT,
                total_savings_pct REAL,
                total_savings_usd REAL,
                ts TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cost_log_turn
                ON cost_log(turn_id);
            CREATE INDEX IF NOT EXISTS idx_cost_log_session
                ON cost_log(session_id, created_at);
            ",
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- decision_log -------------------------------------------------------

    /// Append a decision-log row.
    pub fn insert_decision(&self, row: &DecisionLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO decision_log (id, turn_id, session_key, prompt_hash, system_prompt_hash,
             tool_list_hash, tool_choice, tokens_input, tokens_output, model, provider, latency_ms,
             ts, session_id, session_intent, intent_summary, trace_id, decision_id, tool_profile,
             system_chars, tool_count, tools_schema_chars, skill_count, skills_prompt_chars,
             memory_md_present, schema_version, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
              ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27)",
            params![
                row.id,
                row.turn_id,
                row.session_key,
                row.prompt_hash,
                row.system_prompt_hash,
                row.tool_list_hash,
                row.tool_choice,
                row.tokens_input,
                row.tokens_output,
                row.model,
                row.provider,
                row.latency_ms,
                row.ts,
                row.session_id,
                row.session_intent,
                row.intent_summary,
                row.trace_id,
                row.decision_id,
                row.tool_profile,
                row.system_chars,
                row.tool_count,
                row.tools_schema_chars,
                row.skill_count,
                row.skills_prompt_chars,
                row.memory_md_present as i64,
                row.schema_version,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Recent decision-log rows, newest first.
    pub fn recent_decisions(&self, limit: u64) -> CoreResult<Vec<DecisionLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_key, prompt_hash, system_prompt_hash, tool_list_hash,
                 tool_choice, tokens_input, tokens_output, model, provider, latency_ms, ts,
                 session_id, session_intent, intent_summary, trace_id, decision_id, tool_profile,
                 system_chars, tool_count, tools_schema_chars, skill_count, skills_prompt_chars,
                 memory_md_present, schema_version, created_at
                 FROM decision_log ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], decision_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Decision-log rows for a session, oldest first.
    pub fn decisions_by_session(&self, session_id: &str) -> CoreResult<Vec<DecisionLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_key, prompt_hash, system_prompt_hash, tool_list_hash,
                 tool_choice, tokens_input, tokens_output, model, provider, latency_ms, ts,
                 session_id, session_intent, intent_summary, trace_id, decision_id, tool_profile,
                 system_chars, tool_count, tools_schema_chars, skill_count, skills_prompt_chars,
                 memory_md_present, schema_version, created_at
                 FROM decision_log WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![session_id], decision_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// The decision-log row for a turn, if any.
    pub fn decision_by_turn(&self, turn_id: &str) -> CoreResult<Option<DecisionLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_key, prompt_hash, system_prompt_hash, tool_list_hash,
                 tool_choice, tokens_input, tokens_output, model, provider, latency_ms, ts,
                 session_id, session_intent, intent_summary, trace_id, decision_id, tool_profile,
                 system_chars, tool_count, tools_schema_chars, skill_count, skills_prompt_chars,
                 memory_md_present, schema_version, created_at
                 FROM decision_log WHERE turn_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![turn_id], decision_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    // -- safety_log ---------------------------------------------------------

    /// Append a safety-event row.
    pub fn insert_safety_event(&self, row: &SafetyLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO safety_log (id, event_type, session_id, reason, ts, tool_name, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                row.id,
                row.event_type,
                row.session_id,
                row.reason,
                row.ts,
                row.tool_name,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Safety events for a session, oldest first.
    pub fn safety_events_by_session(&self, session_id: &str) -> CoreResult<Vec<SafetyLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, event_type, session_id, reason, ts, tool_name, created_at
                 FROM safety_log WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![session_id], safety_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Recent safety events, newest first.
    pub fn recent_safety_events(&self, limit: u64) -> CoreResult<Vec<SafetyLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, event_type, session_id, reason, ts, tool_name, created_at
                 FROM safety_log ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], safety_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    // -- turn_call_log ------------------------------------------------------

    /// Append a turn-call-log row.
    pub fn insert_turn_call(&self, row: &TurnCallLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO turn_call_log (id, schema_version, ts, privacy, trace_id, seq, turn_id,
             session_key, session_id, session_intent, agent_id, provider, model, source, kind,
             payload, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                row.id,
                row.schema_version,
                row.ts,
                row.privacy,
                row.trace_id,
                row.seq,
                row.turn_id,
                row.session_key,
                row.session_id,
                row.session_intent,
                row.agent_id,
                row.provider,
                row.model,
                row.source,
                row.kind,
                row.payload,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Turn-call rows for a turn, in sequence order.
    pub fn turn_calls_by_turn(&self, turn_id: &str) -> CoreResult<Vec<TurnCallLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, schema_version, ts, privacy, trace_id, seq, turn_id, session_key,
                 session_id, session_intent, agent_id, provider, model, source, kind, payload,
                 created_at
                 FROM turn_call_log WHERE turn_id = ?1 ORDER BY seq ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![turn_id], turn_call_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    // -- prompt_report ------------------------------------------------------

    /// Append a prompt-report row.
    pub fn insert_prompt_report(&self, row: &PromptReportRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO prompt_report (id, turn_id, session_key, session_id, agent_id,
             system_chars, system_hash, tool_count, tool_profile, tools_schema_chars, skill_count,
             skills_prompt_chars, memory_md_present, daily_notes_omitted,
             daily_notes_count_before_omit, daily_notes_policy_reason,
             injected_workspace_files_count, retrieval_mode, cache_mode, cache_base_hash,
             cache_dynamic_hash, cache_legacy_hash, cache_shadow_final_hash,
             cache_key_collision, resolved_model, provider_after_rewrite,
             reasoning_hint_resolved, cache_base_chars, cache_dynamic_chars, tool_entries,
             schema_version, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
              ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32)",
            params![
                row.id,
                row.turn_id,
                row.session_key,
                row.session_id,
                row.agent_id,
                row.system_chars,
                row.system_hash,
                row.tool_count,
                row.tool_profile,
                row.tools_schema_chars,
                row.skill_count,
                row.skills_prompt_chars,
                row.memory_md_present as i64,
                row.daily_notes_omitted as i64,
                row.daily_notes_count_before_omit,
                row.daily_notes_policy_reason,
                row.injected_workspace_files_count,
                row.retrieval_mode,
                row.cache_mode,
                row.cache_base_hash,
                row.cache_dynamic_hash,
                row.cache_legacy_hash,
                row.cache_shadow_final_hash,
                row.cache_key_collision as i64,
                row.resolved_model,
                row.provider_after_rewrite,
                row.reasoning_hint_resolved,
                row.cache_base_chars,
                row.cache_dynamic_chars,
                row.tool_entries,
                row.schema_version,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// The prompt-report row for a turn, if any.
    pub fn prompt_report_by_turn(&self, turn_id: &str) -> CoreResult<Option<PromptReportRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_key, session_id, agent_id, system_chars, system_hash,
                 tool_count, tool_profile, tools_schema_chars, skill_count, skills_prompt_chars,
                 memory_md_present, daily_notes_omitted, daily_notes_count_before_omit,
                 daily_notes_policy_reason, injected_workspace_files_count, retrieval_mode,
                 cache_mode, cache_base_hash, cache_dynamic_hash, cache_legacy_hash,
                 cache_shadow_final_hash, cache_key_collision, resolved_model,
                 provider_after_rewrite, reasoning_hint_resolved, cache_base_chars,
                 cache_dynamic_chars, tool_entries, schema_version, created_at
                 FROM prompt_report WHERE turn_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![turn_id], prompt_report_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    // -- tool_result_log ----------------------------------------------------

    /// Append a tool-result-log row.
    pub fn insert_tool_result(&self, row: &ToolResultLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO tool_result_log (id, handle, tool_use_id, tool_name, session_id,
             session_key, agent_id, sha256, chars, size_bytes, stored_size_bytes,
             storage_encoding, created_at_text, recorded_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                row.id,
                row.handle,
                row.tool_use_id,
                row.tool_name,
                row.session_id,
                row.session_key,
                row.agent_id,
                row.sha256,
                row.chars,
                row.size_bytes,
                row.stored_size_bytes,
                row.storage_encoding,
                row.created_at_text,
                row.recorded_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Tool-result rows for a session, oldest first.
    pub fn tool_results_by_session(&self, session_id: &str) -> CoreResult<Vec<ToolResultLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, handle, tool_use_id, tool_name, session_id, session_key, agent_id,
                 sha256, chars, size_bytes, stored_size_bytes, storage_encoding, created_at_text,
                 recorded_at
                 FROM tool_result_log WHERE session_id = ?1 ORDER BY recorded_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![session_id], tool_result_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// The tool-result row for a handle, if any.
    pub fn tool_result_by_handle(&self, handle: &str) -> CoreResult<Option<ToolResultLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, handle, tool_use_id, tool_name, session_id, session_key, agent_id,
                 sha256, chars, size_bytes, stored_size_bytes, storage_encoding, created_at_text,
                 recorded_at
                 FROM tool_result_log WHERE handle = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![handle], tool_result_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    // -- compaction_log -----------------------------------------------------

    /// Append a compaction-log row.
    pub fn insert_compaction(&self, row: &CompactionLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO compaction_log (id, session_id, session_key, compaction_id,
             trigger_reason, tokens_before, tokens_after, removed_count, kept_count, chunk_count,
             summary_source, flush_receipt_status, coverage_status, state_kind,
             provider_state_valid, persisted_summary_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                row.id,
                row.session_id,
                row.session_key,
                row.compaction_id,
                row.trigger_reason,
                row.tokens_before,
                row.tokens_after,
                row.removed_count,
                row.kept_count,
                row.chunk_count,
                row.summary_source,
                row.flush_receipt_status,
                row.coverage_status,
                row.state_kind,
                row.provider_state_valid.map(|b| b as i64),
                row.persisted_summary_id,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Compaction rows for a session, oldest first.
    pub fn compactions_by_session(&self, session_id: &str) -> CoreResult<Vec<CompactionLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_id, session_key, compaction_id, trigger_reason, tokens_before,
                 tokens_after, removed_count, kept_count, chunk_count, summary_source,
                 flush_receipt_status, coverage_status, state_kind, provider_state_valid,
                 persisted_summary_id, created_at
                 FROM compaction_log WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![session_id], compaction_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    // -- cost_log -----------------------------------------------------------

    /// Append a cost-log row.
    pub fn insert_cost(&self, row: &CostLogRow) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO cost_log (id, turn_id, session_id, session_key, model, provider,
             tokens_input, tokens_output, latency_ms, routed_model, baseline_model,
             routing_confidence, routing_savings_pct, routing_savings_usd_estimated_vs_baseline,
             tool_projection_applied, tool_projection_tokens_saved, cache_hit_active,
             cache_hit_tokens_saved, cache_hit_usd_estimated_vs_baseline, billed_cost_usd,
             cost_usd, cost_source, total_savings_pct, total_savings_usd, ts, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17,
              ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            params![
                row.id,
                row.turn_id,
                row.session_id,
                row.session_key,
                row.model,
                row.provider,
                row.tokens_input,
                row.tokens_output,
                row.latency_ms,
                row.routed_model,
                row.baseline_model,
                row.routing_confidence,
                row.routing_savings_pct,
                row.routing_savings_usd_estimated_vs_baseline,
                row.tool_projection_applied as i64,
                row.tool_projection_tokens_saved,
                row.cache_hit_active as i64,
                row.cache_hit_tokens_saved,
                row.cache_hit_usd_estimated_vs_baseline,
                row.billed_cost_usd,
                row.cost_usd,
                row.cost_source,
                row.total_savings_pct,
                row.total_savings_usd,
                row.ts,
                row.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Cost rows for a session, oldest first.
    pub fn costs_by_session(&self, session_id: &str) -> CoreResult<Vec<CostLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_id, session_key, model, provider, tokens_input,
                 tokens_output, latency_ms, routed_model, baseline_model, routing_confidence,
                 routing_savings_pct, routing_savings_usd_estimated_vs_baseline,
                 tool_projection_applied, tool_projection_tokens_saved, cache_hit_active,
                 cache_hit_tokens_saved, cache_hit_usd_estimated_vs_baseline, billed_cost_usd,
                 cost_usd, cost_source, total_savings_pct, total_savings_usd, ts, created_at
                 FROM cost_log WHERE session_id = ?1 ORDER BY created_at ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![session_id], cost_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// The cost row for a turn, if any.
    pub fn cost_by_turn(&self, turn_id: &str) -> CoreResult<Option<CostLogRow>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, turn_id, session_id, session_key, model, provider, tokens_input,
                 tokens_output, latency_ms, routed_model, baseline_model, routing_confidence,
                 routing_savings_pct, routing_savings_usd_estimated_vs_baseline,
                 tool_projection_applied, tool_projection_tokens_saved, cache_hit_active,
                 cache_hit_tokens_saved, cache_hit_usd_estimated_vs_baseline, billed_cost_usd,
                 cost_usd, cost_source, total_savings_pct, total_savings_usd, ts, created_at
                 FROM cost_log WHERE turn_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![turn_id], cost_row_mapper)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    /// Total billed cost across all rows.
    pub fn total_billed_cost(&self) -> CoreResult<f64> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let total: f64 = conn
            .query_row(
                "SELECT COALESCE(SUM(billed_cost_usd), 0.0) FROM cost_log",
                [],
                |row| row.get(0),
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(total)
    }
}

// ---------------------------------------------------------------------------
// Row mappers
// ---------------------------------------------------------------------------

fn parse_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn decision_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<DecisionLogRow> {
    Ok(DecisionLogRow {
        id: row.get(0)?,
        turn_id: row.get(1)?,
        session_key: row.get(2)?,
        prompt_hash: row.get(3)?,
        system_prompt_hash: row.get(4)?,
        tool_list_hash: row.get(5)?,
        tool_choice: row.get(6)?,
        tokens_input: row.get(7)?,
        tokens_output: row.get(8)?,
        model: row.get(9)?,
        provider: row.get(10)?,
        latency_ms: row.get(11)?,
        ts: row.get(12)?,
        session_id: row.get(13)?,
        session_intent: row.get(14)?,
        intent_summary: row.get(15)?,
        trace_id: row.get(16)?,
        decision_id: row.get(17)?,
        tool_profile: row.get(18)?,
        system_chars: row.get(19)?,
        tool_count: row.get(20)?,
        tools_schema_chars: row.get(21)?,
        skill_count: row.get(22)?,
        skills_prompt_chars: row.get(23)?,
        memory_md_present: row.get::<_, i64>(24)? != 0,
        schema_version: row.get(25)?,
        created_at: parse_ts(&row.get::<_, String>(26)?),
    })
}

fn safety_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<SafetyLogRow> {
    Ok(SafetyLogRow {
        id: row.get(0)?,
        event_type: row.get(1)?,
        session_id: row.get(2)?,
        reason: row.get(3)?,
        ts: row.get(4)?,
        tool_name: row.get(5)?,
        created_at: parse_ts(&row.get::<_, String>(6)?),
    })
}

fn turn_call_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<TurnCallLogRow> {
    Ok(TurnCallLogRow {
        id: row.get(0)?,
        schema_version: row.get(1)?,
        ts: row.get(2)?,
        privacy: row.get(3)?,
        trace_id: row.get(4)?,
        seq: row.get(5)?,
        turn_id: row.get(6)?,
        session_key: row.get(7)?,
        session_id: row.get(8)?,
        session_intent: row.get(9)?,
        agent_id: row.get(10)?,
        provider: row.get(11)?,
        model: row.get(12)?,
        source: row.get(13)?,
        kind: row.get(14)?,
        payload: row.get(15)?,
        created_at: parse_ts(&row.get::<_, String>(16)?),
    })
}

fn prompt_report_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<PromptReportRow> {
    Ok(PromptReportRow {
        id: row.get(0)?,
        turn_id: row.get(1)?,
        session_key: row.get(2)?,
        session_id: row.get(3)?,
        agent_id: row.get(4)?,
        system_chars: row.get(5)?,
        system_hash: row.get(6)?,
        tool_count: row.get(7)?,
        tool_profile: row.get(8)?,
        tools_schema_chars: row.get(9)?,
        skill_count: row.get(10)?,
        skills_prompt_chars: row.get(11)?,
        memory_md_present: row.get::<_, i64>(12)? != 0,
        daily_notes_omitted: row.get::<_, i64>(13)? != 0,
        daily_notes_count_before_omit: row.get(14)?,
        daily_notes_policy_reason: row.get(15)?,
        injected_workspace_files_count: row.get(16)?,
        retrieval_mode: row.get(17)?,
        cache_mode: row.get(18)?,
        cache_base_hash: row.get(19)?,
        cache_dynamic_hash: row.get(20)?,
        cache_legacy_hash: row.get(21)?,
        cache_shadow_final_hash: row.get(22)?,
        cache_key_collision: row.get::<_, i64>(23)? != 0,
        resolved_model: row.get(24)?,
        provider_after_rewrite: row.get(25)?,
        reasoning_hint_resolved: row.get(26)?,
        cache_base_chars: row.get(27)?,
        cache_dynamic_chars: row.get(28)?,
        tool_entries: row.get(29)?,
        schema_version: row.get(30)?,
        created_at: parse_ts(&row.get::<_, String>(31)?),
    })
}

fn tool_result_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<ToolResultLogRow> {
    Ok(ToolResultLogRow {
        id: row.get(0)?,
        handle: row.get(1)?,
        tool_use_id: row.get(2)?,
        tool_name: row.get(3)?,
        session_id: row.get(4)?,
        session_key: row.get(5)?,
        agent_id: row.get(6)?,
        sha256: row.get(7)?,
        chars: row.get(8)?,
        size_bytes: row.get(9)?,
        stored_size_bytes: row.get(10)?,
        storage_encoding: row.get(11)?,
        created_at_text: row.get(12)?,
        recorded_at: parse_ts(&row.get::<_, String>(13)?),
    })
}

fn compaction_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<CompactionLogRow> {
    Ok(CompactionLogRow {
        id: row.get(0)?,
        session_id: row.get(1)?,
        session_key: row.get(2)?,
        compaction_id: row.get(3)?,
        trigger_reason: row.get(4)?,
        tokens_before: row.get(5)?,
        tokens_after: row.get(6)?,
        removed_count: row.get(7)?,
        kept_count: row.get(8)?,
        chunk_count: row.get(9)?,
        summary_source: row.get(10)?,
        flush_receipt_status: row.get(11)?,
        coverage_status: row.get(12)?,
        state_kind: row.get(13)?,
        provider_state_valid: row.get::<_, Option<i64>>(14)?.map(|b| b != 0),
        persisted_summary_id: row.get(15)?,
        created_at: parse_ts(&row.get::<_, String>(16)?),
    })
}

fn cost_row_mapper(row: &rusqlite::Row) -> rusqlite::Result<CostLogRow> {
    Ok(CostLogRow {
        id: row.get(0)?,
        turn_id: row.get(1)?,
        session_id: row.get(2)?,
        session_key: row.get(3)?,
        model: row.get(4)?,
        provider: row.get(5)?,
        tokens_input: row.get(6)?,
        tokens_output: row.get(7)?,
        latency_ms: row.get(8)?,
        routed_model: row.get(9)?,
        baseline_model: row.get(10)?,
        routing_confidence: row.get(11)?,
        routing_savings_pct: row.get(12)?,
        routing_savings_usd_estimated_vs_baseline: row.get(13)?,
        tool_projection_applied: row.get::<_, i64>(14)? != 0,
        tool_projection_tokens_saved: row.get(15)?,
        cache_hit_active: row.get::<_, i64>(16)? != 0,
        cache_hit_tokens_saved: row.get(17)?,
        cache_hit_usd_estimated_vs_baseline: row.get(18)?,
        billed_cost_usd: row.get(19)?,
        cost_usd: row.get(20)?,
        cost_source: row.get(21)?,
        total_savings_pct: row.get(22)?,
        total_savings_usd: row.get(23)?,
        ts: row.get(24)?,
        created_at: parse_ts(&row.get::<_, String>(25)?),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn new_id() -> String {
        Uuid::new_v4().to_string()
    }

    #[test]
    fn test_decision_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = DecisionLogRow {
            id: new_id(),
            turn_id: "turn-1".to_string(),
            session_key: "sess-key-1".to_string(),
            prompt_hash: "abc123".to_string(),
            system_prompt_hash: "sys123".to_string(),
            tool_list_hash: "tool123".to_string(),
            tool_choice: "auto".to_string(),
            tokens_input: 1000,
            tokens_output: 500,
            model: "claude-sonnet".to_string(),
            provider: "anthropic".to_string(),
            latency_ms: 1234,
            ts: "2026-01-01T00:00:00Z".to_string(),
            session_id: Some("sess-1".to_string()),
            session_intent: None,
            intent_summary: Some("sum".to_string()),
            trace_id: Some("turn-1".to_string()),
            decision_id: None,
            tool_profile: Some("full".to_string()),
            system_chars: 800,
            tool_count: 4,
            tools_schema_chars: 2000,
            skill_count: 2,
            skills_prompt_chars: 300,
            memory_md_present: true,
            schema_version: 16,
            created_at: Utc::now(),
        };
        store.insert_decision(&row).unwrap();

        let by_turn = store.decision_by_turn("turn-1").unwrap().unwrap();
        assert_eq!(by_turn.model, "claude-sonnet");
        assert_eq!(by_turn.tokens_input, 1000);
        assert!(by_turn.memory_md_present);
        assert_eq!(by_turn.tool_profile.as_deref(), Some("full"));

        let by_session = store.decisions_by_session("sess-1").unwrap();
        assert_eq!(by_session.len(), 1);

        let recent = store.recent_decisions(10).unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_safety_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = SafetyLogRow {
            id: new_id(),
            event_type: SafetyEventType::InjectionBlocked.as_code().to_string(),
            session_id: "sess-1".to_string(),
            reason: "prompt injection detected".to_string(),
            ts: "2026-01-01T00:00:00Z".to_string(),
            tool_name: Some("bash".to_string()),
            created_at: Utc::now(),
        };
        store.insert_safety_event(&row).unwrap();

        let by_session = store.safety_events_by_session("sess-1").unwrap();
        assert_eq!(by_session.len(), 1);
        assert_eq!(by_session[0].event_type, "injection_blocked");
        assert_eq!(by_session[0].tool_name.as_deref(), Some("bash"));
        assert!(SafetyEventType::from_code("injection_blocked").is_some());
        assert!(SafetyEventType::from_code("bogus").is_none());

        let recent = store.recent_safety_events(10).unwrap();
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_turn_call_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = TurnCallLogRow {
            id: new_id(),
            schema_version: 1,
            ts: "2026-01-01T00:00:00Z".to_string(),
            privacy: "raw".to_string(),
            trace_id: "turn-1".to_string(),
            seq: 1,
            turn_id: "turn-1".to_string(),
            session_key: "sess-key-1".to_string(),
            session_id: Some("sess-1".to_string()),
            session_intent: None,
            agent_id: "agent-1".to_string(),
            provider: "anthropic".to_string(),
            model: "claude-sonnet".to_string(),
            source: "{}".to_string(),
            kind: "tool_call".to_string(),
            payload: "{\"tool\":\"bash\"}".to_string(),
            created_at: Utc::now(),
        };
        store.insert_turn_call(&row).unwrap();

        let by_turn = store.turn_calls_by_turn("turn-1").unwrap();
        assert_eq!(by_turn.len(), 1);
        assert_eq!(by_turn[0].seq, 1);
        assert_eq!(by_turn[0].kind, "tool_call");
    }

    #[test]
    fn test_prompt_report_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = PromptReportRow {
            id: new_id(),
            turn_id: "turn-1".to_string(),
            session_key: "sess-key-1".to_string(),
            session_id: Some("sess-1".to_string()),
            agent_id: "agent-1".to_string(),
            system_chars: 1200,
            system_hash: "deadbeef".to_string(),
            tool_count: 3,
            tool_profile: Some("full".to_string()),
            tools_schema_chars: 4500,
            skill_count: 1,
            skills_prompt_chars: 200,
            memory_md_present: true,
            daily_notes_omitted: false,
            daily_notes_count_before_omit: 0,
            daily_notes_policy_reason: None,
            injected_workspace_files_count: 2,
            retrieval_mode: Some("vector".to_string()),
            cache_mode: Some("on".to_string()),
            cache_base_hash: Some("h1".to_string()),
            cache_dynamic_hash: None,
            cache_legacy_hash: None,
            cache_shadow_final_hash: None,
            cache_key_collision: false,
            resolved_model: Some("claude-sonnet-4".to_string()),
            provider_after_rewrite: None,
            reasoning_hint_resolved: None,
            cache_base_chars: 1000,
            cache_dynamic_chars: 200,
            tool_entries: "[]".to_string(),
            schema_version: 4,
            created_at: Utc::now(),
        };
        store.insert_prompt_report(&row).unwrap();

        let by_turn = store.prompt_report_by_turn("turn-1").unwrap().unwrap();
        assert_eq!(by_turn.system_chars, 1200);
        assert_eq!(by_turn.system_hash, "deadbeef");
        assert!(by_turn.memory_md_present);
        assert_eq!(by_turn.resolved_model.as_deref(), Some("claude-sonnet-4"));
        assert_eq!(by_turn.cache_dynamic_hash, None);
    }

    #[test]
    fn test_tool_result_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = ToolResultLogRow {
            id: new_id(),
            handle: "tr-0123456789abcdef0123456789abcdef".to_string(),
            tool_use_id: "tu-1".to_string(),
            tool_name: "bash".to_string(),
            session_id: "sess-1".to_string(),
            session_key: "sess-key-1".to_string(),
            agent_id: "agent-1".to_string(),
            sha256: "abc".to_string(),
            chars: 5000,
            size_bytes: 5000,
            stored_size_bytes: Some(1200),
            storage_encoding: "gzip+utf-8".to_string(),
            created_at_text: "2026-01-01T00:00:00Z".to_string(),
            recorded_at: Utc::now(),
        };
        store.insert_tool_result(&row).unwrap();

        let by_handle = store
            .tool_result_by_handle("tr-0123456789abcdef0123456789abcdef")
            .unwrap()
            .unwrap();
        assert_eq!(by_handle.tool_name, "bash");
        assert_eq!(by_handle.chars, 5000);
        assert_eq!(by_handle.stored_size_bytes, Some(1200));
        assert_eq!(by_handle.storage_encoding, "gzip+utf-8");

        let by_session = store.tool_results_by_session("sess-1").unwrap();
        assert_eq!(by_session.len(), 1);
    }

    #[test]
    fn test_compaction_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = CompactionLogRow {
            id: new_id(),
            session_id: Some("sess-1".to_string()),
            session_key: Some("sess-key-1".to_string()),
            compaction_id: Some("c-1".to_string()),
            trigger_reason: Some("token_budget".to_string()),
            tokens_before: Some(10000),
            tokens_after: Some(4000),
            removed_count: 6,
            kept_count: 4,
            chunk_count: 2,
            summary_source: "local".to_string(),
            flush_receipt_status: "success".to_string(),
            coverage_status: "pass".to_string(),
            state_kind: "structured_summary_v1".to_string(),
            provider_state_valid: Some(true),
            persisted_summary_id: Some(42),
            created_at: Utc::now(),
        };
        store.insert_compaction(&row).unwrap();

        let by_session = store.compactions_by_session("sess-1").unwrap();
        assert_eq!(by_session.len(), 1);
        assert_eq!(by_session[0].tokens_before, Some(10000));
        assert_eq!(by_session[0].tokens_after, Some(4000));
        assert_eq!(by_session[0].provider_state_valid, Some(true));
        assert_eq!(by_session[0].persisted_summary_id, Some(42));
        assert_eq!(
            by_session[0].trigger_reason.as_deref(),
            Some("token_budget")
        );
    }

    #[test]
    fn test_cost_log_round_trip() {
        let store = LogStore::in_memory().unwrap();
        let row = CostLogRow {
            id: new_id(),
            turn_id: "turn-1".to_string(),
            session_id: Some("sess-1".to_string()),
            session_key: "sess-key-1".to_string(),
            model: "claude-sonnet".to_string(),
            provider: "anthropic".to_string(),
            tokens_input: 1000,
            tokens_output: 500,
            latency_ms: 1234,
            routed_model: Some("claude-haiku".to_string()),
            baseline_model: Some("claude-sonnet".to_string()),
            routing_confidence: Some(0.92),
            routing_savings_pct: Some(40.0),
            routing_savings_usd_estimated_vs_baseline: Some(0.004),
            tool_projection_applied: true,
            tool_projection_tokens_saved: 3000,
            cache_hit_active: false,
            cache_hit_tokens_saved: 0,
            cache_hit_usd_estimated_vs_baseline: None,
            billed_cost_usd: Some(0.006),
            cost_usd: Some(0.006),
            cost_source: Some("provider_telemetry".to_string()),
            total_savings_pct: Some(45.0),
            total_savings_usd: Some(0.005),
            ts: "2026-01-01T00:00:00Z".to_string(),
            created_at: Utc::now(),
        };
        store.insert_cost(&row).unwrap();

        let by_turn = store.cost_by_turn("turn-1").unwrap().unwrap();
        assert_eq!(by_turn.model, "claude-sonnet");
        assert_eq!(by_turn.tokens_input, 1000);
        assert!(by_turn.tool_projection_applied);
        assert_eq!(by_turn.routing_confidence, Some(0.92));
        assert_eq!(by_turn.billed_cost_usd, Some(0.006));

        let by_session = store.costs_by_session("sess-1").unwrap();
        assert_eq!(by_session.len(), 1);

        let total = store.total_billed_cost().unwrap();
        assert!((total - 0.006).abs() < 1e-9);
    }

    #[test]
    fn test_nullable_session_id_filters() {
        let store = LogStore::in_memory().unwrap();
        let row = DecisionLogRow {
            id: new_id(),
            turn_id: "turn-2".to_string(),
            session_key: "sess-key-2".to_string(),
            prompt_hash: "h".to_string(),
            system_prompt_hash: "h".to_string(),
            tool_list_hash: "h".to_string(),
            tool_choice: "auto".to_string(),
            tokens_input: 0,
            tokens_output: 0,
            model: "m".to_string(),
            provider: "p".to_string(),
            latency_ms: 0,
            ts: "2026-01-01T00:00:00Z".to_string(),
            session_id: None,
            session_intent: None,
            intent_summary: None,
            trace_id: None,
            decision_id: None,
            tool_profile: None,
            system_chars: 0,
            tool_count: 0,
            tools_schema_chars: 0,
            skill_count: 0,
            skills_prompt_chars: 0,
            memory_md_present: false,
            schema_version: 16,
            created_at: Utc::now(),
        };
        store.insert_decision(&row).unwrap();

        // Null session_id should not match a session-scoped query.
        let by_session = store.decisions_by_session("sess-2").unwrap();
        assert!(by_session.is_empty());
        // But the turn-scoped lookup still works.
        let by_turn = store.decision_by_turn("turn-2").unwrap().unwrap();
        assert!(by_turn.session_id.is_none());
    }

    #[test]
    fn test_in_memory_store_is_fresh() {
        let store = LogStore::in_memory().unwrap();
        assert!(store.recent_decisions(10).unwrap().is_empty());
        assert!(store.recent_safety_events(10).unwrap().is_empty());
        assert_eq!(store.total_billed_cost().unwrap(), 0.0);
    }
}
