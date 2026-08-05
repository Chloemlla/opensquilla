use chrono::{DateTime, Utc};
use dashmap::DashMap;
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

use crate::compaction::{CompactionPlanner, CompactionReport, extractive_summary};
use crate::models::{
    AgentTask, CompactedTranscriptEntry, CompactionHistory, ProjectWorkspace, RoutingDecision,
    Session, SessionAttachment, SessionContextState, SessionFork, SessionLock, SessionMetadata,
    SessionMode, SessionStatus, SessionSummary, SessionTag, TaskStatus, TranscriptEntry,
};
use crate::storage::SessionStorage;
use crate::usage_ledger::UsageLedger;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Domain errors raised by [`SessionManager`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    NotFound(Uuid),
    InvalidTransition {
        from: LifecyclePhase,
        to: LifecyclePhase,
    },
    AlreadyCompacting(Uuid),
    EmptyTranscript,
    Locked(String),
    CompactionFailed(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::NotFound(id) => write!(f, "Session {} not found", id),
            SessionError::InvalidTransition { from, to } => {
                write!(f, "Invalid session transition {:?} -> {:?}", from, to)
            }
            SessionError::AlreadyCompacting(id) => {
                write!(f, "Session {} is already compacting", id)
            }
            SessionError::EmptyTranscript => write!(f, "Session has no transcript entries"),
            SessionError::Locked(owner) => write!(f, "Session is locked by '{}'", owner),
            SessionError::CompactionFailed(msg) => write!(f, "Compaction failed: {}", msg),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<SessionError> for CoreError {
    fn from(err: SessionError) -> Self {
        match err {
            SessionError::NotFound(id) => CoreError::NotFound(format!("Session {}", id)),
            SessionError::InvalidTransition { from, to } => CoreError::InvalidInput(format!(
                "Invalid session transition {:?} -> {:?}",
                from, to
            )),
            SessionError::AlreadyCompacting(id) => {
                CoreError::InvalidInput(format!("Session {} is already compacting", id))
            }
            SessionError::EmptyTranscript => {
                CoreError::InvalidInput("Session has no transcript entries".into())
            }
            SessionError::Locked(owner) => {
                CoreError::InvalidInput(format!("Session is locked by '{}'", owner))
            }
            SessionError::CompactionFailed(msg) => CoreError::Session(msg),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Optional model-routing preferences recorded in session metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingPrefs {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
}

/// Builder-style configuration for [`SessionManager::create`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionConfig {
    pub agent_id: Uuid,
    pub name: Option<String>,
    pub system_prompt: String,
    pub mode: SessionMode,
    pub metadata: serde_json::Value,
    pub tags: Vec<String>,
    pub routing: RoutingPrefs,
    pub parent_session_id: Option<Uuid>,
    pub fork_event: Option<String>,
}

impl CreateSessionConfig {
    pub fn new(agent_id: Uuid) -> Self {
        Self {
            agent_id,
            name: None,
            system_prompt: String::new(),
            mode: SessionMode::Chat,
            metadata: serde_json::Value::Null,
            tags: Vec::new(),
            routing: RoutingPrefs::default(),
            parent_session_id: None,
            fork_event: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    pub fn with_mode(mut self, mode: SessionMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_routing(mut self, routing: RoutingPrefs) -> Self {
        self.routing = routing;
        self
    }

    pub fn with_parent(mut self, parent_session_id: Uuid, fork_event: impl Into<String>) -> Self {
        self.parent_session_id = Some(parent_session_id);
        self.fork_event = Some(fork_event.into());
        self
    }
}

/// Configuration for forking a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkConfig {
    pub name: Option<String>,
    pub mode: Option<SessionMode>,
    pub system_prompt: Option<String>,
    pub metadata: serde_json::Value,
    pub copy_transcript: bool,
    pub tags: Vec<String>,
    pub fork_event: String,
}

impl Default for ForkConfig {
    fn default() -> Self {
        Self {
            name: None,
            mode: None,
            system_prompt: None,
            metadata: serde_json::Value::Null,
            copy_transcript: false,
            tags: Vec::new(),
            fork_event: "fork".to_string(),
        }
    }
}

/// Filter used by [`SessionManager::list`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionFilter {
    pub agent_id: Option<Uuid>,
    pub status: Option<SessionStatus>,
    pub mode: Option<SessionMode>,
    pub date_from: Option<DateTime<Utc>>,
    pub date_to: Option<DateTime<Utc>>,
    pub search: Option<String>,
    pub tags: Vec<String>,
    pub limit: u64,
    pub offset: u64,
}

impl SessionFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_agent(mut self, agent_id: Uuid) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    pub fn with_status(mut self, status: SessionStatus) -> Self {
        self.status = Some(status);
        self
    }

    pub fn with_mode(mut self, mode: SessionMode) -> Self {
        self.mode = Some(mode);
        self
    }

    pub fn with_search(mut self, search: impl Into<String>) -> Self {
        self.search = Some(search.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_limit(mut self, limit: u64) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }
}

// ---------------------------------------------------------------------------
// Lifecycle state machine
// ---------------------------------------------------------------------------

/// Manager-level lifecycle phase. Extends the persisted [`SessionStatus`]
/// with transient phases (`Created`, `Deleted`) that are not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecyclePhase {
    Created,
    Active,
    Paused,
    Compacting,
    Archived,
    Killed,
    Deleted,
}

impl LifecyclePhase {
    pub fn from_status(status: &SessionStatus) -> Self {
        match status {
            SessionStatus::Active => Self::Active,
            SessionStatus::Paused => Self::Paused,
            SessionStatus::Archived => Self::Archived,
            SessionStatus::Compacting => Self::Compacting,
            SessionStatus::Killed => Self::Killed,
        }
    }

    pub fn to_status(self) -> Option<SessionStatus> {
        match self {
            Self::Created | Self::Active => Some(SessionStatus::Active),
            Self::Paused => Some(SessionStatus::Paused),
            Self::Compacting => Some(SessionStatus::Compacting),
            Self::Archived => Some(SessionStatus::Archived),
            Self::Killed => Some(SessionStatus::Killed),
            Self::Deleted => None,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Deleted)
    }

    /// Phases reachable in a single step from `self`.
    pub fn allowed_transitions(self) -> Vec<LifecyclePhase> {
        match self {
            Self::Created => vec![Self::Active, Self::Killed, Self::Deleted],
            Self::Active => vec![
                Self::Paused,
                Self::Compacting,
                Self::Archived,
                Self::Killed,
                Self::Deleted,
            ],
            Self::Paused => vec![Self::Active, Self::Archived, Self::Killed, Self::Deleted],
            Self::Compacting => vec![Self::Active, Self::Paused, Self::Killed, Self::Deleted],
            Self::Archived => vec![Self::Active, Self::Killed, Self::Deleted],
            Self::Killed => vec![Self::Deleted],
            Self::Deleted => vec![],
        }
    }

    pub fn can_transition(self, next: LifecyclePhase) -> bool {
        self.allowed_transitions().contains(&next)
    }
}

/// A requested lifecycle change; see [`SessionManager::transition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionTransition {
    Pause,
    Resume,
    Archive,
    Restore,
    Kill,
    Delete,
}

impl SessionTransition {
    pub fn target(self) -> LifecyclePhase {
        match self {
            Self::Pause => LifecyclePhase::Paused,
            Self::Resume | Self::Restore => LifecyclePhase::Active,
            Self::Archive => LifecyclePhase::Archived,
            Self::Kill => LifecyclePhase::Killed,
            Self::Delete => LifecyclePhase::Deleted,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// High-level session lifecycle manager. Coordinates persistence, the
/// in-memory active-session cache, usage tracking, and compaction triggers.
pub struct SessionManager {
    storage: SessionStorage,
    ledger: UsageLedger,
    active_sessions: DashMap<Uuid, Session>,
}

impl SessionManager {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage,
            ledger: UsageLedger::new(),
            active_sessions: DashMap::new(),
        }
    }

    pub fn storage(&self) -> &SessionStorage {
        &self.storage
    }

    pub fn ledger(&self) -> &UsageLedger {
        &self.ledger
    }

    // --- Creation ---

    /// Create a new session from a full configuration.
    pub fn create(&self, config: CreateSessionConfig) -> CoreResult<Session> {
        let now = Utc::now();
        let metadata = merge_routing_metadata(config.metadata, &config.routing);

        let session = Session {
            id: Uuid::new_v4(),
            agent_id: config.agent_id,
            name: config.name.unwrap_or_default(),
            created_at: now,
            updated_at: now,
            last_active_at: now,
            status: SessionStatus::Active,
            mode: config.mode,
            system_prompt: config.system_prompt,
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: config.parent_session_id,
            fork_event: config.fork_event,
            metadata,
        };

        self.storage.create_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());

        for tag in &config.tags {
            self.storage.add_session_tag(&SessionTag {
                session_id: session.id,
                tag: tag.clone(),
                created_at: now,
            })?;
        }

        info!("Created session {}", session.id);
        Ok(session)
    }

    /// Create a new session (legacy signature).
    pub fn create_session(
        &self,
        agent_id: Uuid,
        name: String,
        system_prompt: String,
        mode: SessionMode,
    ) -> CoreResult<Session> {
        let config = CreateSessionConfig::new(agent_id)
            .with_name(name)
            .with_system_prompt(system_prompt)
            .with_mode(mode);
        self.create(config)
    }

    // --- Fetch / update ---

    /// Fetch a session by ID.
    pub fn get(&self, session_id: &Uuid) -> CoreResult<Option<Session>> {
        self.storage.get_session(session_id)
    }

    /// Fetch a session by ID and refresh its activity timestamps. Intended for
    /// callers that are about to operate on the session.
    pub fn get_mut(&self, session_id: &Uuid) -> CoreResult<Option<Session>> {
        let mut session = match self.storage.get_session(session_id)? {
            Some(s) => s,
            None => return Ok(None),
        };
        session.updated_at = Utc::now();
        session.last_active_at = Utc::now();
        self.storage.update_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());
        Ok(Some(session))
    }

    /// Fetch a session by ID (legacy alias).
    pub fn get_session(&self, session_id: &Uuid) -> CoreResult<Option<Session>> {
        self.storage.get_session(session_id)
    }

    /// Touch a session's activity timestamps without changing status.
    pub fn touch(&self, session_id: &Uuid) -> CoreResult<()> {
        let mut session = self.require_session(session_id)?;
        session.last_active_at = Utc::now();
        session.updated_at = Utc::now();
        self.storage.update_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());
        Ok(())
    }

    // --- Lifecycle ---

    /// Apply a lifecycle transition. `Delete` removes the session entirely;
    /// every other transition persists the mapped status.
    pub fn transition(
        &self,
        session_id: &Uuid,
        transition: SessionTransition,
    ) -> CoreResult<Session> {
        let target = transition.target();
        let session = self.require_session(session_id)?;
        let from = LifecyclePhase::from_status(&session.status);

        if !from.can_transition(target) {
            return Err(SessionError::InvalidTransition { from, to: target }.into());
        }

        if target == LifecyclePhase::Deleted {
            self.storage.delete_session(session_id)?;
            self.active_sessions.remove(session_id);
            info!("Deleted session {}", session_id);
            return Ok(session);
        }

        let status = target.to_status().unwrap_or(session.status);
        self.set_status(session_id, status)?;
        self.require_session(session_id)
    }

    /// Pause an active session.
    pub fn pause_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.transition(session_id, SessionTransition::Pause)
    }

    /// Resume a session (Active). Returns `None` for a killed session to
    /// preserve legacy behavior.
    pub fn resume_session(&self, session_id: &Uuid) -> CoreResult<Option<Session>> {
        let session = self.storage.get_session(session_id)?;
        match session {
            Some(s) if s.status == SessionStatus::Killed => {
                warn!("Cannot resume killed session {}", session_id);
                Ok(None)
            }
            Some(s) if s.status == SessionStatus::Active => {
                self.touch(session_id)?;
                self.storage.get_session(session_id)
            }
            Some(_) => Ok(Some(
                self.transition(session_id, SessionTransition::Resume)?,
            )),
            None => Ok(None),
        }
    }

    /// Archive an active or paused session.
    pub fn archive_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.transition(session_id, SessionTransition::Archive)
    }

    /// Restore an archived session.
    pub fn restore_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.transition(session_id, SessionTransition::Restore)
    }

    /// Kill a running session (terminal unless deleted).
    pub fn kill_session(&self, session_id: &Uuid) -> CoreResult<()> {
        self.transition(session_id, SessionTransition::Kill)?;
        Ok(())
    }

    /// Kill alias.
    pub fn kill(&self, session_id: &Uuid) -> CoreResult<()> {
        self.kill_session(session_id)
    }

    /// Permanently delete a session and its transcript.
    pub fn delete_session(&self, session_id: &Uuid) -> CoreResult<()> {
        self.transition(session_id, SessionTransition::Delete)?;
        Ok(())
    }

    // --- Fork / branch ---

    /// Fork a session from an existing one.
    pub fn fork(&self, source_id: &Uuid, config: ForkConfig) -> CoreResult<Session> {
        let source = self.require_session(source_id)?;
        let now = Utc::now();

        let metadata = merge_fork_metadata(config.metadata, &source.id);

        let session = Session {
            id: Uuid::new_v4(),
            agent_id: source.agent_id,
            name: config
                .name
                .unwrap_or_else(|| format!("Fork of {}", source.name)),
            created_at: now,
            updated_at: now,
            last_active_at: now,
            status: SessionStatus::Active,
            mode: config.mode.unwrap_or(source.mode),
            system_prompt: config
                .system_prompt
                .unwrap_or_else(|| source.system_prompt.clone()),
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: Some(source.id),
            fork_event: Some(config.fork_event.clone()),
            metadata,
        };

        self.storage.create_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());

        self.storage.insert_session_fork(&SessionFork {
            id: Uuid::new_v4(),
            source_session_id: source.id,
            child_session_id: session.id,
            fork_event: config.fork_event.clone(),
            created_at: now,
            metadata: serde_json::json!({ "copy_transcript": config.copy_transcript }),
        })?;

        if config.copy_transcript {
            self.copy_transcript(&source.id, &session.id, None)?;
        }
        for tag in &config.tags {
            self.storage.add_session_tag(&SessionTag {
                session_id: session.id,
                tag: tag.clone(),
                created_at: now,
            })?;
        }

        info!("Forked session {} from {}", session.id, source_id);
        Ok(session)
    }

    /// Legacy fork: no transcript copy, arbitrary fork event label.
    pub fn fork_session(
        &self,
        source_session_id: &Uuid,
        fork_event: String,
        new_name: Option<String>,
    ) -> CoreResult<Session> {
        let config = ForkConfig {
            name: new_name,
            fork_event,
            ..Default::default()
        };
        self.fork(source_session_id, config)
    }

    /// Branch a session from a specific transcript turn. The child session
    /// inherits every entry up to and including `turn_id`.
    pub fn branch(&self, source_id: &Uuid, turn_id: &Uuid) -> CoreResult<Session> {
        let source = self.require_session(source_id)?;
        let entries = self.storage.list_by_session(source_id)?;
        let index = entries
            .iter()
            .position(|e| e.id == *turn_id)
            .ok_or_else(|| CoreError::NotFound(format!("Transcript entry {}", turn_id)))?;

        let now = Utc::now();
        let branch = Session {
            id: Uuid::new_v4(),
            agent_id: source.agent_id,
            name: format!("Branch of {} @ {}", source.name, short_id(turn_id)),
            created_at: now,
            updated_at: now,
            last_active_at: now,
            status: SessionStatus::Active,
            mode: source.mode,
            system_prompt: source.system_prompt.clone(),
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: Some(source.id),
            fork_event: Some(format!("branch:{}", turn_id)),
            metadata: serde_json::json!({ "branch_turn": turn_id.to_string() }),
        };
        self.storage.create_session(&branch)?;
        self.active_sessions.insert(branch.id, branch.clone());

        self.storage.insert_session_fork(&SessionFork {
            id: Uuid::new_v4(),
            source_session_id: source.id,
            child_session_id: branch.id,
            fork_event: format!("branch:{}", turn_id),
            created_at: now,
            metadata: serde_json::json!({ "turn_id": turn_id.to_string() }),
        })?;

        self.copy_transcript(&source.id, &branch.id, Some(index))?;
        info!(
            "Branched session {} from {} at turn {}",
            branch.id, source_id, turn_id
        );
        Ok(branch)
    }

    /// List fork records for a source session.
    pub fn fork_history(&self, source_session_id: &Uuid) -> CoreResult<Vec<SessionFork>> {
        self.storage.list_session_forks_by_source(source_session_id)
    }

    // --- Compaction ---

    /// Trigger compaction for a session. The session is moved to `Compacting`
    /// for the duration of the run and restored to `Active` afterwards. Uses a
    /// deterministic extractive summary when no LLM summarizer is wired.
    pub fn compact(&self, session_id: &Uuid) -> CoreResult<CompactionReport> {
        let session = self.require_session(session_id)?;
        if session.status == SessionStatus::Compacting {
            return Err(SessionError::AlreadyCompacting(*session_id).into());
        }

        self.set_status(session_id, SessionStatus::Compacting)?;
        let result = self.run_compaction(session_id);
        let _ = self.set_status(session_id, SessionStatus::Active);
        result
    }

    fn run_compaction(&self, session_id: &Uuid) -> CoreResult<CompactionReport> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let planner = CompactionPlanner::new();
        let plan = planner.plan_compaction(&session, &entries)?;

        if plan.is_noop() {
            return Ok(plan.to_report("no_op"));
        }

        let now = Utc::now();
        let summary_text = extractive_summary(&plan.entries_to_compact, 500);
        let summary = SessionSummary {
            id: Uuid::new_v4(),
            session_id: *session_id,
            summary: summary_text.clone(),
            created_at: now,
            token_count: crate::compaction::estimate_token_count(&summary_text),
            is_active: true,
        };
        let history = CompactionHistory {
            id: Uuid::new_v4(),
            session_id: *session_id,
            compaction_id: Uuid::new_v4(),
            entries_compacted: plan.entries_to_compact.len() as u64,
            tokens_before: plan.current_tokens,
            tokens_after: plan.estimated_after,
            summary_id: Some(summary.id),
            started_at: now,
            completed_at: now,
            status: "completed".to_string(),
            metadata: serde_json::json!({ "strategy": plan.strategy.label() }),
        };

        self.storage.apply_compaction_transaction(
            session_id,
            &summary,
            &plan.entries_to_compact,
            &history,
        )?;

        info!("Compacted session {}", session_id);
        Ok(CompactionReport {
            compaction_id: history.compaction_id,
            session_id: *session_id,
            strategy: plan.strategy,
            entries_compacted: history.entries_compacted,
            tokens_before: history.tokens_before,
            tokens_after: history.tokens_after,
            summary_id: Some(summary.id),
            started_at: now,
            completed_at: now,
            status: "completed".to_string(),
        })
    }

    // --- Transcript ---

    /// Add a message to a session's transcript.
    pub fn add_message(
        &self,
        session_id: &Uuid,
        role: String,
        content: String,
        token_count: u64,
    ) -> CoreResult<TranscriptEntry> {
        let mut session = self.require_session(session_id)?;

        let entry = TranscriptEntry::new(*session_id, role, content, token_count);
        self.storage.insert_transcript_entry(&entry)?;

        session.message_count += 1;
        session.total_tokens += token_count;
        session.last_active_at = Utc::now();
        session.updated_at = Utc::now();
        self.storage.update_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());

        Ok(entry)
    }

    /// Get transcript entries for a session.
    pub fn get_transcript(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<TranscriptEntry>> {
        self.storage
            .get_transcript_entries(session_id, limit, offset)
    }

    /// Full transcript history (active + compacted) for a session.
    pub fn full_transcript(&self, session_id: &Uuid) -> CoreResult<Vec<TranscriptEntry>> {
        self.storage.list_by_session(session_id)
    }

    /// Rows moved out of the active transcript by compaction.
    pub fn compacted_entries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<CompactedTranscriptEntry>> {
        self.storage.list_compacted(session_id, limit, offset)
    }

    /// Delete a single transcript entry.
    pub fn delete_transcript_entry(&self, entry_id: &Uuid) -> CoreResult<()> {
        self.storage.delete_transcript_entry(entry_id)
    }

    // --- Listing ---

    /// List sessions matching a filter. Applies in-memory filtering on top of
    /// the storage index and returns a page of results.
    pub fn list(&self, filter: &SessionFilter) -> CoreResult<Vec<Session>> {
        let mut sessions = match filter.agent_id {
            Some(agent_id) => self.storage.list_sessions(&agent_id, u64::MAX, 0)?,
            None => self.storage.list_all_sessions(u64::MAX, 0)?,
        };

        if let Some(status) = &filter.status {
            sessions.retain(|s| &s.status == status);
        }
        if let Some(mode) = &filter.mode {
            sessions.retain(|s| &s.mode == mode);
        }
        if let Some(from) = filter.date_from {
            sessions.retain(|s| s.created_at >= from);
        }
        if let Some(to) = filter.date_to {
            sessions.retain(|s| s.created_at <= to);
        }
        if let Some(needle) = &filter.search {
            let needle = needle.to_lowercase();
            sessions.retain(|s| {
                s.name.to_lowercase().contains(&needle)
                    || s.system_prompt.to_lowercase().contains(&needle)
                    || s.id.to_string().contains(&needle)
            });
        }
        if !filter.tags.is_empty() {
            let mut tagged: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
            for tag in &filter.tags {
                for id in self.storage.list_sessions_for_tag(tag)? {
                    tagged.insert(id);
                }
            }
            sessions.retain(|s| tagged.contains(&s.id));
        }

        let limit = if filter.limit == 0 {
            usize::MAX
        } else {
            filter.limit as usize
        };
        let skip = filter.offset as usize;
        Ok(sessions.into_iter().skip(skip).take(limit).collect())
    }

    /// List sessions for an agent (legacy signature).
    pub fn list_sessions(
        &self,
        agent_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<Session>> {
        let filter = SessionFilter {
            agent_id: Some(*agent_id),
            limit,
            offset,
            ..Default::default()
        };
        self.list(&filter)
    }

    /// Total number of sessions in the store.
    pub fn count(&self) -> CoreResult<usize> {
        Ok(self.storage.list_all_sessions(u64::MAX, 0)?.len())
    }

    /// Remove archived sessions idle for longer than `max_idle`.
    pub fn prune_idle(&self, max_idle: chrono::Duration) -> CoreResult<usize> {
        let sessions = self.storage.list_all_sessions(u64::MAX, 0)?;
        let cutoff = Utc::now() - max_idle;
        let mut pruned = 0usize;
        for session in sessions {
            if session.status == SessionStatus::Archived && session.last_active_at < cutoff {
                self.storage.delete_session(&session.id)?;
                self.active_sessions.remove(&session.id);
                pruned += 1;
            }
        }
        Ok(pruned)
    }

    // --- Metadata ---

    pub fn set_metadata(&self, session_id: &Uuid, key: &str, value: &str) -> CoreResult<()> {
        self.require_session(session_id)?;
        self.storage.set_session_metadata(&SessionMetadata {
            session_id: *session_id,
            key: key.to_string(),
            value: value.to_string(),
            updated_at: Utc::now(),
        })
    }

    pub fn get_metadata(&self, session_id: &Uuid, key: &str) -> CoreResult<Option<String>> {
        Ok(self
            .storage
            .get_session_metadata(session_id, key)?
            .map(|m| m.value))
    }

    pub fn list_metadata(&self, session_id: &Uuid) -> CoreResult<Vec<(String, String)>> {
        Ok(self
            .storage
            .list_session_metadata(session_id)?
            .into_iter()
            .map(|m| (m.key, m.value))
            .collect())
    }

    pub fn delete_metadata(&self, session_id: &Uuid, key: &str) -> CoreResult<()> {
        self.storage.delete_session_metadata(session_id, key)
    }

    // --- Tags ---

    pub fn add_tag(&self, session_id: &Uuid, tag: &str) -> CoreResult<()> {
        self.require_session(session_id)?;
        self.storage.add_session_tag(&SessionTag {
            session_id: *session_id,
            tag: tag.to_string(),
            created_at: Utc::now(),
        })
    }

    pub fn remove_tag(&self, session_id: &Uuid, tag: &str) -> CoreResult<()> {
        self.storage.remove_session_tag(session_id, tag)
    }

    pub fn list_tags(&self, session_id: &Uuid) -> CoreResult<Vec<String>> {
        Ok(self
            .storage
            .list_session_tags(session_id)?
            .into_iter()
            .map(|t| t.tag)
            .collect())
    }

    pub fn sessions_for_tag(&self, tag: &str) -> CoreResult<Vec<Uuid>> {
        self.storage.list_sessions_for_tag(tag)
    }

    // --- Attachments ---

    pub fn attach(
        &self,
        session_id: &Uuid,
        name: &str,
        content_type: &str,
        size_bytes: u64,
        storage_uri: &str,
        metadata: serde_json::Value,
    ) -> CoreResult<SessionAttachment> {
        self.require_session(session_id)?;
        let attachment = SessionAttachment {
            id: Uuid::new_v4(),
            session_id: *session_id,
            name: name.to_string(),
            content_type: content_type.to_string(),
            size_bytes,
            storage_uri: storage_uri.to_string(),
            created_at: Utc::now(),
            metadata,
        };
        self.storage.insert_session_attachment(&attachment)?;
        Ok(attachment)
    }

    pub fn get_attachment(&self, id: &Uuid) -> CoreResult<Option<SessionAttachment>> {
        self.storage.get_session_attachment(id)
    }

    pub fn list_attachments(&self, session_id: &Uuid) -> CoreResult<Vec<SessionAttachment>> {
        self.storage.list_session_attachments(session_id)
    }

    pub fn remove_attachment(&self, id: &Uuid) -> CoreResult<()> {
        self.storage.delete_session_attachment(id)
    }

    // --- Locks ---

    /// Acquire (or replace) an advisory lock on a session. `ttl_secs` of
    /// `None` means the lock never expires.
    pub fn lock_session(
        &self,
        session_id: &Uuid,
        owner: &str,
        ttl_secs: Option<u64>,
    ) -> CoreResult<()> {
        self.require_session(session_id)?;
        let expires_at = ttl_secs.map(|s| Utc::now() + chrono::Duration::seconds(s as i64));
        self.storage.acquire_session_lock(&SessionLock {
            session_id: *session_id,
            owner: owner.to_string(),
            acquired_at: Utc::now(),
            expires_at,
            metadata: serde_json::Value::Null,
        })
    }

    pub fn unlock_session(&self, session_id: &Uuid) -> CoreResult<()> {
        self.storage.release_session_lock(session_id)
    }

    pub fn get_lock(&self, session_id: &Uuid) -> CoreResult<Option<SessionLock>> {
        self.storage.get_session_lock(session_id)
    }

    /// Whether a session currently holds an un-expired lock. Expired locks are
    /// cleaned up as a side effect.
    pub fn is_locked(&self, session_id: &Uuid) -> CoreResult<bool> {
        match self.storage.get_session_lock(session_id)? {
            Some(lock) => {
                if let Some(expires_at) = lock.expires_at {
                    if expires_at < Utc::now() {
                        self.storage.release_session_lock(session_id)?;
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            None => Ok(false),
        }
    }

    // --- Context state & summaries ---

    pub fn get_context_state(&self, session_id: &Uuid) -> CoreResult<Option<SessionContextState>> {
        self.storage.get_context_state(session_id)
    }

    pub fn set_context_state(&self, state: SessionContextState) -> CoreResult<()> {
        self.storage.upsert_context_state(&state)
    }

    pub fn get_active_summary(&self, session_id: &Uuid) -> CoreResult<Option<SessionSummary>> {
        self.storage.get_active_summary(session_id)
    }

    pub fn list_summaries(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<SessionSummary>> {
        self.storage.list_summaries(session_id, limit, offset)
    }

    pub fn list_compaction_history(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<CompactionHistory>> {
        self.storage
            .list_compaction_history_by_session(session_id, limit, offset)
    }

    // --- Routing ---

    pub fn record_routing_decision(
        &self,
        session_id: &Uuid,
        turn: i64,
        provider: &str,
        model: &str,
        reason: &str,
        metadata: serde_json::Value,
    ) -> CoreResult<RoutingDecision> {
        self.require_session(session_id)?;
        let decision = RoutingDecision {
            id: Uuid::new_v4(),
            session_id: *session_id,
            turn,
            provider: provider.to_string(),
            model: model.to_string(),
            reason: reason.to_string(),
            created_at: Utc::now(),
            metadata,
        };
        self.storage.insert_routing_decision(&decision)?;
        Ok(decision)
    }

    pub fn list_routing_decisions(
        &self,
        session_id: &Uuid,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<RoutingDecision>> {
        self.storage
            .list_routing_decisions_by_session(session_id, limit, offset)
    }

    // --- Workspaces ---

    pub fn add_workspace(
        &self,
        session_id: &Uuid,
        name: &str,
        root_path: &str,
        metadata: serde_json::Value,
    ) -> CoreResult<ProjectWorkspace> {
        self.require_session(session_id)?;
        let workspace = ProjectWorkspace {
            id: Uuid::new_v4(),
            session_id: *session_id,
            name: name.to_string(),
            root_path: root_path.to_string(),
            created_at: Utc::now(),
            metadata,
        };
        self.storage.insert_project_workspace(&workspace)?;
        Ok(workspace)
    }

    pub fn list_workspaces(&self, session_id: &Uuid) -> CoreResult<Vec<ProjectWorkspace>> {
        self.storage.list_project_workspaces_by_session(session_id)
    }

    // --- Agent tasks ---

    pub fn spawn_task(
        &self,
        session_id: &Uuid,
        kind: &str,
        input: serde_json::Value,
    ) -> CoreResult<AgentTask> {
        self.require_session(session_id)?;
        let task = AgentTask {
            id: Uuid::new_v4(),
            session_id: *session_id,
            parent_task_id: None,
            kind: kind.to_string(),
            status: TaskStatus::Queued,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            input,
            output: serde_json::Value::Null,
            error: None,
        };
        self.storage.insert_agent_task(&task)?;
        Ok(task)
    }

    pub fn list_tasks(&self, session_id: &Uuid) -> CoreResult<Vec<AgentTask>> {
        self.storage.list_agent_tasks_by_session(session_id)
    }

    pub fn update_task_status(
        &self,
        task_id: &Uuid,
        status: TaskStatus,
        output: Option<serde_json::Value>,
        error: Option<&str>,
    ) -> CoreResult<()> {
        let now = Utc::now();
        let started_at = if status == TaskStatus::Running {
            Some(now)
        } else {
            None
        };
        let completed_at = if matches!(
            status,
            TaskStatus::Succeeded | TaskStatus::Failed | TaskStatus::Cancelled
        ) {
            Some(now)
        } else {
            None
        };
        self.storage.update_agent_task_status(
            task_id,
            &status,
            started_at.as_ref(),
            completed_at.as_ref(),
            output.as_ref(),
            error,
        )
    }

    // --- Export ---

    /// Export a session and its transcript to JSON.
    pub fn export(&self, session_id: &Uuid) -> CoreResult<serde_json::Value> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let tags = self.list_tags(session_id)?;
        let metadata = self.list_metadata(session_id)?;
        Ok(serde_json::json!({
            "session": session,
            "transcript": entries,
            "tags": tags,
            "metadata": metadata,
        }))
    }

    // --- Active-cache accessors ---

    /// Number of sessions currently held in the active cache.
    pub fn active_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// IDs of sessions currently held in the active cache.
    pub fn active_session_ids(&self) -> Vec<Uuid> {
        self.active_sessions.iter().map(|e| *e.key()).collect()
    }

    // --- Internal helpers ---

    fn require_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.storage
            .get_session(session_id)?
            .ok_or_else(|| SessionError::NotFound(*session_id).into())
    }

    fn set_status(&self, session_id: &Uuid, status: SessionStatus) -> CoreResult<()> {
        let mut session = self.require_session(session_id)?;
        session.status = status;
        session.updated_at = Utc::now();
        self.storage.update_session(&session)?;
        self.active_sessions.insert(session.id, session.clone());
        Ok(())
    }

    fn copy_transcript(
        &self,
        source_id: &Uuid,
        child_id: &Uuid,
        up_to_index: Option<usize>,
    ) -> CoreResult<usize> {
        let entries = self.storage.list_by_session(source_id)?;
        let cutoff = up_to_index.map(|i| i + 1).unwrap_or(entries.len());
        let mut copied = 0usize;
        for entry in entries.into_iter().take(cutoff) {
            let copy = TranscriptEntry {
                id: Uuid::new_v4(),
                session_id: *child_id,
                role: entry.role,
                content: entry.content,
                created_at: entry.created_at,
                token_count: entry.token_count,
                metadata: entry.metadata,
                compacted: entry.compacted,
            };
            self.storage.insert_transcript_entry(&copy)?;
            copied += 1;
        }
        Ok(copied)
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn merge_routing_metadata(
    metadata: serde_json::Value,
    routing: &RoutingPrefs,
) -> serde_json::Value {
    if let Some(obj) = metadata.as_object() {
        let mut out = obj.clone();
        if let Some(provider) = &routing.provider {
            out.insert("routing_provider".into(), serde_json::json!(provider));
        }
        if let Some(model) = &routing.model {
            out.insert("routing_model".into(), serde_json::json!(model));
        }
        if let Some(temperature) = routing.temperature {
            out.insert("routing_temperature".into(), serde_json::json!(temperature));
        }
        if let Some(max_tokens) = routing.max_tokens {
            out.insert("routing_max_tokens".into(), serde_json::json!(max_tokens));
        }
        serde_json::Value::Object(out)
    } else {
        serde_json::json!({
            "routing_provider": routing.provider,
            "routing_model": routing.model,
            "routing_temperature": routing.temperature,
            "routing_max_tokens": routing.max_tokens,
        })
    }
}

fn merge_fork_metadata(metadata: serde_json::Value, source_id: &Uuid) -> serde_json::Value {
    let mut merged = metadata;
    if let Some(obj) = merged.as_object_mut() {
        obj.insert(
            "forked_from".into(),
            serde_json::json!(source_id.to_string()),
        );
        merged = serde_json::Value::Object(obj.clone());
    } else {
        merged = serde_json::json!({ "forked_from": source_id.to_string() });
    }
    merged
}

fn short_id(id: &Uuid) -> String {
    id.to_string().chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn manager() -> SessionManager {
        SessionManager::new(SessionStorage::in_memory().unwrap())
    }

    fn create(manager: &SessionManager, name: &str) -> Session {
        manager
            .create_session(
                Uuid::new_v4(),
                name.to_string(),
                "you are helpful".into(),
                SessionMode::Chat,
            )
            .unwrap()
    }

    #[test]
    fn create_and_get_roundtrip() {
        let m = manager();
        let s = create(&m, "hello world");
        let fetched = m.get(&s.id).unwrap().unwrap();
        assert_eq!(fetched.id, s.id);
        assert_eq!(fetched.name, "hello world");
        assert_eq!(fetched.status, SessionStatus::Active);
        assert_eq!(fetched.mode, SessionMode::Chat);
        assert!(m.active_session_ids().contains(&s.id));
    }

    #[test]
    fn lifecycle_pause_archive_restore_kill_delete() {
        let m = manager();
        let s = create(&m, "lifecycle");

        let paused = m.pause_session(&s.id).unwrap();
        assert_eq!(paused.status, SessionStatus::Paused);

        let resumed = m.resume_session(&s.id).unwrap().unwrap();
        assert_eq!(resumed.status, SessionStatus::Active);

        let archived = m.archive_session(&s.id).unwrap();
        assert_eq!(archived.status, SessionStatus::Archived);

        let restored = m.restore_session(&s.id).unwrap();
        assert_eq!(restored.status, SessionStatus::Active);

        m.kill_session(&s.id).unwrap();
        assert_eq!(m.get(&s.id).unwrap().unwrap().status, SessionStatus::Killed);

        // Resuming a killed session returns None (legacy behavior).
        assert!(m.resume_session(&s.id).unwrap().is_none());

        m.delete_session(&s.id).unwrap();
        assert!(m.get(&s.id).unwrap().is_none());
    }

    #[test]
    fn invalid_transition_is_rejected() {
        let m = manager();
        let s = create(&m, "invalid");

        // Killed -> Paused is invalid.
        m.kill_session(&s.id).unwrap();
        let err = m.pause_session(&s.id).unwrap_err();
        assert!(matches!(
            err,
            CoreError::InvalidInput(_) | CoreError::Session(_)
        ));

        // Deleting twice is a NotFound.
        m.delete_session(&s.id).unwrap();
        let err = m.delete_session(&s.id).unwrap_err();
        assert!(matches!(err, CoreError::NotFound(_)));
    }

    #[test]
    fn fork_without_and_with_transcript_copy() {
        let m = manager();
        let s = create(&m, "source");
        for i in 0..3 {
            m.add_message(&s.id, "user".into(), format!("msg {}", i), 10)
                .unwrap();
        }

        // Default fork: no transcript.
        let f = m
            .fork(
                &s.id,
                ForkConfig {
                    fork_event: "test".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(f.parent_session_id, Some(s.id));
        assert_eq!(f.name, "Fork of source");
        assert!(m.full_transcript(&f.id).unwrap().is_empty());

        // Copy fork: transcript inherited.
        let f2 = m
            .fork(
                &s.id,
                ForkConfig {
                    fork_event: "copy".into(),
                    copy_transcript: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(m.full_transcript(&f2.id).unwrap().len(), 3);

        // Fork record was persisted.
        assert_eq!(m.fork_history(&s.id).unwrap().len(), 2);
    }

    #[test]
    fn branch_copies_through_turn() {
        let m = manager();
        let s = create(&m, "branchable");
        let mut ids = Vec::new();
        for i in 0..5 {
            let e = m
                .add_message(&s.id, "user".into(), format!("msg {}", i), 10)
                .unwrap();
            ids.push(e.id);
        }

        let branch = m.branch(&s.id, &ids[2]).unwrap();
        assert_eq!(branch.parent_session_id, Some(s.id));
        let copied = m.full_transcript(&branch.id).unwrap();
        assert_eq!(copied.len(), 3);
        assert_eq!(copied[0].content, "msg 0");
        assert_eq!(copied[2].content, "msg 2");
    }

    #[test]
    fn branch_rejects_foreign_turn() {
        let m = manager();
        let a = create(&m, "a");
        let b = create(&m, "b");
        let e = m
            .add_message(&b.id, "user".into(), "hello".into(), 5)
            .unwrap();
        assert!(m.branch(&a.id, &e.id).is_err());
    }

    #[test]
    fn list_filters_by_status_and_search() {
        let m = manager();
        let agent = Uuid::new_v4();
        let s1 = m
            .create_session(
                agent,
                "Alpha analysis".into(),
                String::new(),
                SessionMode::Chat,
            )
            .unwrap();
        let s2 = m
            .create_session(
                agent,
                "Beta coding".into(),
                String::new(),
                SessionMode::Plan,
            )
            .unwrap();

        m.archive_session(&s2.id).unwrap();

        let active = m
            .list(
                &SessionFilter::new()
                    .with_agent(agent)
                    .with_status(SessionStatus::Active),
            )
            .unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, s1.id);

        let archived = m
            .list(
                &SessionFilter::new()
                    .with_agent(agent)
                    .with_status(SessionStatus::Archived),
            )
            .unwrap();
        assert_eq!(archived.len(), 1);
        assert_eq!(archived[0].id, s2.id);

        let searched = m
            .list(&SessionFilter::new().with_agent(agent).with_search("coding"))
            .unwrap();
        assert_eq!(searched.len(), 1);
        assert_eq!(searched[0].id, s2.id);
    }

    #[test]
    fn metadata_and_tags_roundtrip() {
        let m = manager();
        let s = create(&m, "meta");
        m.set_metadata(&s.id, "color", "blue").unwrap();
        assert_eq!(m.get_metadata(&s.id, "color").unwrap().unwrap(), "blue");

        m.add_tag(&s.id, "important").unwrap();
        m.add_tag(&s.id, "wip").unwrap();
        assert_eq!(m.list_tags(&s.id).unwrap().len(), 2);
        assert_eq!(m.sessions_for_tag("important").unwrap()[0], s.id);

        m.remove_tag(&s.id, "wip").unwrap();
        assert_eq!(m.list_tags(&s.id).unwrap().len(), 1);
    }

    #[test]
    fn locking_obeys_ttl() {
        let m = manager();
        let s = create(&m, "locked");
        m.lock_session(&s.id, "runner", Some(3600)).unwrap();
        assert!(m.is_locked(&s.id).unwrap());
        m.unlock_session(&s.id).unwrap();
        assert!(!m.is_locked(&s.id).unwrap());
    }

    #[test]
    fn compact_produces_report_and_summary() {
        let m = manager();
        let s = create(&m, "compact-me");
        // 60 entries of 100 tokens => 6,000 tokens, above the 4,096 threshold.
        for i in 0..60 {
            m.add_message(&s.id, "user".into(), format!("line {}", i), 100)
                .unwrap();
        }

        let report = m.compact(&s.id).unwrap();
        assert_eq!(report.status, "completed");
        assert!(report.entries_compacted > 0);
        assert!(report.summary_id.is_some());

        // Session restored to Active after the run.
        assert_eq!(m.get(&s.id).unwrap().unwrap().status, SessionStatus::Active);

        // An active summary + context state now exist.
        assert!(m.get_active_summary(&s.id).unwrap().is_some());
        assert!(m.get_context_state(&s.id).unwrap().is_some());
        assert!(!m.list_compaction_history(&s.id, 10, 0).unwrap().is_empty());
    }

    #[test]
    fn add_message_updates_session_counts() {
        let m = manager();
        let s = create(&m, "counts");
        m.add_message(&s.id, "user".into(), "hello world".into(), 7)
            .unwrap();
        m.add_message(&s.id, "assistant".into(), "hi there".into(), 5)
            .unwrap();
        let fetched = m.get(&s.id).unwrap().unwrap();
        assert_eq!(fetched.message_count, 2);
        assert_eq!(fetched.total_tokens, 12);
    }

    #[test]
    fn prune_idle_removes_old_archived() {
        let m = manager();
        let s = create(&m, "old");
        m.archive_session(&s.id).unwrap();
        // Backdate the session.
        let mut session = m.get(&s.id).unwrap().unwrap();
        session.last_active_at = Utc::now() - Duration::days(30);
        m.storage.update_session(&session).unwrap();

        let pruned = m.prune_idle(Duration::days(7)).unwrap();
        assert_eq!(pruned, 1);
        assert!(m.get(&s.id).unwrap().is_none());
    }

    #[test]
    fn export_serializes_session_and_transcript() {
        let m = manager();
        let s = create(&m, "export-me");
        m.add_message(&s.id, "user".into(), "data".into(), 3)
            .unwrap();
        let value = m.export(&s.id).unwrap();
        assert!(value.get("session").is_some());
        assert!(value.get("transcript").is_some());
        assert_eq!(value["transcript"].as_array().unwrap().len(), 1);
    }
}
