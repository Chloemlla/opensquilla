use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub agent_id: Uuid,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
    pub status: SessionStatus,
    pub mode: SessionMode,
    pub system_prompt: String,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub message_count: u64,
    pub parent_session_id: Option<Uuid>,
    pub fork_event: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Paused,
    Archived,
    Compacting,
    Killed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    Chat,
    Plan,
    Agent,
    Batch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub token_count: u64,
    pub metadata: serde_json::Value,
    pub compacted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: Uuid,
    pub session_id: Uuid,
    pub summary: String,
    pub created_at: DateTime<Utc>,
    pub token_count: u64,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRevision {
    pub id: Uuid,
    pub session_id: Uuid,
    pub plan: String,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    pub version: u32,
    pub parent_revision_id: Option<Uuid>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Active,
    Completed,
    Cancelled,
    Superseded,
}

/// 3. `compacted_transcript_entries` — rows moved out of the active transcript
/// during compaction. Mirrors `TranscriptEntry` plus provenance about the
/// compaction event that relocated it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactedTranscriptEntry {
    pub id: Uuid,
    pub session_id: Uuid,
    pub original_entry_id: Uuid,
    pub role: String,
    pub content: String,
    pub token_count: u64,
    pub original_created_at: DateTime<Utc>,
    pub compacted_at: DateTime<Utc>,
    pub compaction_id: Option<Uuid>,
    pub metadata: serde_json::Value,
}

/// 5. `session_context_states` — snapshot of a session's context window
/// (active summary, token budget, retained entry range) used to reconstruct
/// state on resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionContextState {
    pub id: Uuid,
    pub session_id: Uuid,
    pub active_summary_id: Option<Uuid>,
    pub retained_after: Option<DateTime<Utc>>,
    pub context_tokens: u64,
    pub budget_tokens: u64,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// 7. `plan_runs` — a single execution run against a plan revision, tracking
/// runtime status and the agent task it dispatched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRun {
    pub id: Uuid,
    pub plan_revision_id: Uuid,
    pub session_id: Uuid,
    pub status: PlanRunStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub agent_task_id: Option<Uuid>,
    pub result: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanRunStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// 8. `agent_tasks` — runtime ledger of tasks spawned by an agent turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: Uuid,
    pub session_id: Uuid,
    pub parent_task_id: Option<Uuid>,
    pub kind: String,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// 9. `project_workspaces` — project directories associated with a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectWorkspace {
    pub id: Uuid,
    pub session_id: Uuid,
    pub name: String,
    pub root_path: String,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// 10. `usage_events` — a usage event groups one or more line items
/// (see `UsageEventItem`) billed together for a single LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub id: Uuid,
    pub session_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub request_id: Option<String>,
    pub total_cost_nanodollars: u64,
    pub metadata: serde_json::Value,
}

/// 11. `usage_event_items` — individual line items within a `UsageEvent`
/// (e.g. input tokens, cached input, output tokens).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEventItem {
    pub id: Uuid,
    pub usage_event_id: Uuid,
    pub session_id: Uuid,
    pub kind: String,
    pub units: u64,
    pub cost_nanodollars: u64,
    pub metadata: serde_json::Value,
}

/// 12. `usage_ledger_state` — monotonic ledger watermark per session,
/// supporting idempotent appends and reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageLedgerState {
    pub session_id: Uuid,
    pub last_event_id: Option<Uuid>,
    pub last_event_seq: i64,
    pub total_cost_nanodollars: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub updated_at: DateTime<Utc>,
}

/// 13. `session_locks` — per-session advisory locks held by a runtime owner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLock {
    pub session_id: Uuid,
    pub owner: String,
    pub acquired_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
}

/// 14. `session_attachments` — metadata for files attached to a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAttachment {
    pub id: Uuid,
    pub session_id: Uuid,
    pub name: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub storage_uri: String,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// 15. `session_forks` — fork event records linking a child session to its
/// source and the triggering event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionFork {
    pub id: Uuid,
    pub source_session_id: Uuid,
    pub child_session_id: Uuid,
    pub fork_event: String,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// 16. `routing_decisions` — records of model-routing decisions per turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub id: Uuid,
    pub session_id: Uuid,
    pub turn: i64,
    pub provider: String,
    pub model: String,
    pub reason: String,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// 17. `session_metadata` — key-value session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: Uuid,
    pub key: String,
    pub value: String,
    pub updated_at: DateTime<Utc>,
}

/// 18. `session_tags` — session/tag association rows (tags themselves are
/// stored as plain strings; this table is the join).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTag {
    pub session_id: Uuid,
    pub tag: String,
    pub created_at: DateTime<Utc>,
}

/// 19. `compaction_history` — audit log of compaction runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionHistory {
    pub id: Uuid,
    pub session_id: Uuid,
    pub compaction_id: Uuid,
    pub entries_compacted: u64,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub summary_id: Option<Uuid>,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub status: String,
    pub metadata: serde_json::Value,
}

impl Default for Session {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            name: String::new(),
            created_at: now,
            updated_at: now,
            last_active_at: now,
            status: SessionStatus::Active,
            mode: SessionMode::Chat,
            system_prompt: String::new(),
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: None,
            fork_event: None,
            metadata: serde_json::Value::Null,
        }
    }
}

impl TranscriptEntry {
    pub fn new(session_id: Uuid, role: String, content: String, token_count: u64) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            role,
            content,
            created_at: Utc::now(),
            token_count,
            metadata: serde_json::Value::Null,
            compacted: false,
        }
    }
}

impl Default for SessionContextState {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            active_summary_id: None,
            retained_after: None,
            context_tokens: 0,
            budget_tokens: 0,
            updated_at: now,
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for PlanRun {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            plan_revision_id: Uuid::nil(),
            session_id: Uuid::nil(),
            status: PlanRunStatus::Pending,
            started_at: Utc::now(),
            completed_at: None,
            agent_task_id: None,
            result: serde_json::Value::Null,
        }
    }
}

impl Default for AgentTask {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            parent_task_id: None,
            kind: String::new(),
            status: TaskStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            input: serde_json::Value::Null,
            output: serde_json::Value::Null,
            error: None,
        }
    }
}

impl Default for ProjectWorkspace {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            name: String::new(),
            root_path: String::new(),
            created_at: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for UsageEvent {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            timestamp: Utc::now(),
            provider: String::new(),
            model: String::new(),
            request_id: None,
            total_cost_nanodollars: 0,
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for UsageEventItem {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            usage_event_id: Uuid::nil(),
            session_id: Uuid::nil(),
            kind: String::new(),
            units: 0,
            cost_nanodollars: 0,
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for UsageLedgerState {
    fn default() -> Self {
        Self {
            session_id: Uuid::nil(),
            last_event_id: None,
            last_event_seq: 0,
            total_cost_nanodollars: 0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
            updated_at: Utc::now(),
        }
    }
}

impl Default for SessionLock {
    fn default() -> Self {
        Self {
            session_id: Uuid::nil(),
            owner: String::new(),
            acquired_at: Utc::now(),
            expires_at: None,
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for SessionAttachment {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            name: String::new(),
            content_type: String::new(),
            size_bytes: 0,
            storage_uri: String::new(),
            created_at: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for SessionFork {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            source_session_id: Uuid::nil(),
            child_session_id: Uuid::nil(),
            fork_event: String::new(),
            created_at: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for RoutingDecision {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            turn: 0,
            provider: String::new(),
            model: String::new(),
            reason: String::new(),
            created_at: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }
}

impl Default for SessionMetadata {
    fn default() -> Self {
        Self {
            session_id: Uuid::nil(),
            key: String::new(),
            value: String::new(),
            updated_at: Utc::now(),
        }
    }
}

impl Default for SessionTag {
    fn default() -> Self {
        Self {
            session_id: Uuid::nil(),
            tag: String::new(),
            created_at: Utc::now(),
        }
    }
}

impl Default for CompactionHistory {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            compaction_id: Uuid::new_v4(),
            entries_compacted: 0,
            tokens_before: 0,
            tokens_after: 0,
            summary_id: None,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            status: String::from("completed"),
            metadata: serde_json::Value::Null,
        }
    }
}