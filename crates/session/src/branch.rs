//! # Session branching
//!
//! Session branching forks a child session from a specific point in a parent
//! session's transcript. Unlike a plain fork (which copies the full parent
//! transcript or nothing), a branch copies the transcript up to and including
//! the chosen entry, producing a divergent conversation line.
//!
//! This module also provides:
//!
//! - [`SessionBrancher`] — the high-level branch/fork/diff orchestrator.
//! - [`BranchDiff`] — a structured diff between two sessions' transcripts.
//! - [`MergeResult`] — the outcome of merging a child session back into its
//!   parent.
//!
//! Branching is a read-only operation on the parent: it never mutates the
//! parent's transcript. Merging, by contrast, appends the child's divergent
//! entries to the parent (see [`SessionBrancher::merge_branch`]).

use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::manager::{ForkConfig, SessionManager};
use crate::models::{Session, SessionFork, SessionStatus, TranscriptEntry};

// ---------------------------------------------------------------------------
// Branch diff
// ---------------------------------------------------------------------------

/// A structured diff between two sessions' transcripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDiff {
    /// The parent session id.
    pub parent_session_id: Uuid,
    /// The child session id.
    pub child_session_id: Uuid,
    /// The entry id where the branch diverged.
    pub branch_point: Uuid,
    /// Entries common to both sessions (in order).
    pub common_prefix: Vec<TranscriptEntry>,
    /// Entries only in the parent session (after the branch point).
    pub parent_only: Vec<TranscriptEntry>,
    /// Entries only in the child session (after the branch point).
    pub child_only: Vec<TranscriptEntry>,
    /// Token count of the common prefix.
    pub common_tokens: u64,
    /// Token count of the parent-only entries.
    pub parent_only_tokens: u64,
    /// Token count of the child-only entries.
    pub child_only_tokens: u64,
    /// When the diff was computed.
    pub computed_at: DateTime<Utc>,
}

impl BranchDiff {
    /// Whether the two sessions diverge at all.
    pub fn diverges(&self) -> bool {
        !self.parent_only.is_empty() || !self.child_only.is_empty()
    }

    /// The total number of divergent entries.
    pub fn divergent_count(&self) -> usize {
        self.parent_only.len() + self.child_only.len()
    }

    /// The total token count of the divergent entries.
    pub fn divergent_tokens(&self) -> u64 {
        self.parent_only_tokens + self.child_only_tokens
    }

    /// Whether the child is strictly ahead of the parent (parent has no
    /// divergent entries).
    pub fn child_ahead(&self) -> bool {
        self.parent_only.is_empty() && !self.child_only.is_empty()
    }

    /// Whether the parent is strictly ahead of the child.
    pub fn parent_ahead(&self) -> bool {
        !self.parent_only.is_empty() && self.child_only.is_empty()
    }

    /// A human-readable summary of the diff.
    pub fn summary(&self) -> String {
        format!(
            "common={} parent_only={} child_only={} tokens_divergent={}",
            self.common_prefix.len(),
            self.parent_only.len(),
            self.child_only.len(),
            self.divergent_tokens()
        )
    }
}

// ---------------------------------------------------------------------------
// Merge result
// ---------------------------------------------------------------------------

/// The outcome of merging a child session's divergent entries into its parent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// The parent session that received the merge.
    pub parent_session_id: Uuid,
    /// The child session that was merged in.
    pub child_session_id: Uuid,
    /// The entries appended to the parent.
    pub merged_entries: Vec<TranscriptEntry>,
    /// The number of entries appended.
    pub entries_merged: usize,
    /// The token count added to the parent.
    pub tokens_added: u64,
    /// Whether the child session was archived after the merge.
    pub child_archived: bool,
    /// When the merge completed.
    pub merged_at: DateTime<Utc>,
}

impl MergeResult {
    /// Whether the merge was a no-op (nothing to merge).
    pub fn is_noop(&self) -> bool {
        self.entries_merged == 0
    }
}

// ---------------------------------------------------------------------------
// Branch configuration
// ---------------------------------------------------------------------------

/// Configuration for branching a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    /// Optional name for the child session.
    pub name: Option<String>,
    /// Whether to copy the transcript up to the branch point.
    pub copy_transcript: bool,
    /// Whether to copy the parent's tags into the child.
    pub copy_tags: bool,
    /// Whether to copy the parent's metadata into the child.
    pub copy_metadata: bool,
    /// The fork event label recorded for the branch.
    pub fork_event: String,
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            name: None,
            copy_transcript: true,
            copy_tags: true,
            copy_metadata: true,
            fork_event: "branch".to_string(),
        }
    }
}

impl BranchConfig {
    /// Create a config with a custom child name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Create a config that does not copy the transcript.
    pub fn without_transcript(mut self) -> Self {
        self.copy_transcript = false;
        self
    }
}

// ---------------------------------------------------------------------------
// SessionBrancher
// ---------------------------------------------------------------------------

/// High-level session branching, diffing, and merging.
///
/// Wraps a [`SessionManager`] and adds branch-specific operations on top of
/// the manager's `fork` and `branch` primitives. The brancher never touches
/// storage directly — it goes through the manager so all operations benefit
/// from the manager's active-session cache and lifecycle enforcement.
pub struct SessionBrancher {
    manager: SessionManager,
}

impl SessionBrancher {
    /// Create a new brancher wrapping the given manager.
    pub fn new(manager: SessionManager) -> Self {
        Self { manager }
    }

    /// Borrow the underlying manager.
    pub fn manager(&self) -> &SessionManager {
        &self.manager
    }

    /// Consume the brancher and return the underlying manager.
    pub fn into_manager(self) -> SessionManager {
        self.manager
    }

    /// Branch a session from a specific transcript entry. The child session
    /// inherits every entry up to and including `turn_id`.
    pub fn branch_from(
        &self,
        source_id: &Uuid,
        turn_id: &Uuid,
        config: &BranchConfig,
    ) -> CoreResult<Session> {
        let source = self.manager.get(source_id)?.ok_or_else(|| {
            CoreError::NotFound(format!("Session {}", source_id))
        })?;

        let entries = self.manager.full_transcript(source_id)?;
        let branch_idx = entries
            .iter()
            .position(|e| e.id == *turn_id)
            .ok_or_else(|| CoreError::NotFound(format!("Transcript entry {}", turn_id)))?;

        let now = Utc::now();
        let name = config.name.clone().unwrap_or_else(|| {
            format!("Branch of {} @ {}", source.name, short_id(turn_id))
        });

        let mut metadata = serde_json::json!({
            "branch_turn": turn_id.to_string(),
            "branched_at": now.to_rfc3339(),
        });
        if config.copy_metadata {
            if let Some(obj) = metadata.as_object_mut() {
                if let Some(src_meta) = source.metadata.as_object() {
                    for (k, v) in src_meta {
                        if k != "branch_turn" && k != "branched_at" {
                            obj.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        }

        let fork_config = ForkConfig {
            name: Some(name),
            mode: Some(source.mode),
            system_prompt: Some(source.system_prompt.clone()),
            metadata,
            copy_transcript: false, // we copy manually below
            tags: if config.copy_tags {
                self.manager.list_tags(source_id)?
            } else {
                Vec::new()
            },
            fork_event: config.fork_event.clone(),
        };

        let child = self.manager.fork(source_id, fork_config)?;

        if config.copy_transcript {
            self.copy_transcript_range(source_id, &child.id, Some(branch_idx))?;
        }

        info!(
            "Branched session {} from {} at turn {} ({} entries copied)",
            child.id,
            source_id,
            turn_id,
            branch_idx + 1
        );
        Ok(child)
    }

    /// Branch from the latest entry in the parent session (equivalent to a
    /// full fork).
    pub fn branch_from_latest(&self, source_id: &Uuid, config: &BranchConfig) -> CoreResult<Session> {
        let entries = self.manager.full_transcript(source_id)?;
        let latest = entries
            .last()
            .ok_or_else(|| CoreError::InvalidInput("Session has no transcript entries".into()))?;
        self.branch_from(source_id, &latest.id, config)
    }

    /// Branch from the beginning of the session (an empty child).
    pub fn branch_from_start(&self, source_id: &Uuid, config: &BranchConfig) -> CoreResult<Session> {
        let source = self.manager.get(source_id)?.ok_or_else(|| {
            CoreError::NotFound(format!("Session {}", source_id))
        })?;
        let now = Utc::now();
        let name = config
            .name
            .clone()
            .unwrap_or_else(|| format!("Branch of {} (start)", source.name));

        let fork_config = ForkConfig {
            name: Some(name),
            mode: Some(source.mode),
            system_prompt: Some(source.system_prompt.clone()),
            metadata: serde_json::json!({ "branched_at": now.to_rfc3339() }),
            copy_transcript: false,
            tags: if config.copy_tags {
                self.manager.list_tags(source_id)?
            } else {
                Vec::new()
            },
            fork_event: format!("branch:start:{}", source.id),
        };

        self.manager.fork(source_id, fork_config)
    }

    /// Compute a diff between a parent session and one of its branches.
    pub fn diff(&self, parent_id: &Uuid, child_id: &Uuid) -> CoreResult<BranchDiff> {
        let parent_fork = self.find_fork_record(parent_id, child_id)?;
        let branch_point = parent_fork
            .metadata
            .get("turn_id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .or_else(|| {
                parent_fork
                    .metadata
                    .get("branch_turn")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok())
            })
            .unwrap_or_else(Uuid::nil);

        let parent_entries = self.manager.full_transcript(parent_id)?;
        let child_entries = self.manager.full_transcript(child_id)?;

        let branch_idx = if branch_point.is_nil() {
            0
        } else {
            parent_entries
                .iter()
                .position(|e| e.id == branch_point)
                .map(|i| i + 1)
                .unwrap_or(0)
        };

        let common_prefix: Vec<TranscriptEntry> = parent_entries[..branch_idx.min(parent_entries.len())].to_vec();
        let parent_only: Vec<TranscriptEntry> = parent_entries[branch_idx.min(parent_entries.len())..].to_vec();

        let common_count = common_prefix.len();
        let child_only: Vec<TranscriptEntry> = child_entries.iter().skip(common_count).cloned().collect();

        let common_tokens: u64 = common_prefix.iter().map(|e| e.token_count).sum();
        let parent_only_tokens: u64 = parent_only.iter().map(|e| e.token_count).sum();
        let child_only_tokens: u64 = child_only.iter().map(|e| e.token_count).sum();

        Ok(BranchDiff {
            parent_session_id: *parent_id,
            child_session_id: *child_id,
            branch_point,
            common_prefix,
            parent_only,
            child_only,
            common_tokens,
            parent_only_tokens,
            child_only_tokens,
            computed_at: Utc::now(),
        })
    }

    /// Merge a child session's divergent entries back into its parent.
    /// The parent must be active (or resumed), and the child is archived
    /// after the merge unless `archive_child` is `false`.
    pub fn merge_branch(
        &self,
        parent_id: &Uuid,
        child_id: &Uuid,
        archive_child: bool,
    ) -> CoreResult<MergeResult> {
        let parent = self.manager.get(parent_id)?.ok_or_else(|| {
            CoreError::NotFound(format!("Session {}", parent_id))
        })?;
        if parent.status == SessionStatus::Killed {
            return Err(CoreError::InvalidInput(
                "Cannot merge into a killed session".into(),
            ));
        }
        if parent.status == SessionStatus::Archived {
            self.manager.resume_session(parent_id)?;
        }

        let diff = self.diff(parent_id, child_id)?;
        let now = Utc::now();

        let mut merged_entries = Vec::new();
        let mut tokens_added = 0u64;

        for entry in &diff.child_only {
            let new_entry = TranscriptEntry {
                id: Uuid::new_v4(),
                session_id: *parent_id,
                role: entry.role.clone(),
                content: entry.content.clone(),
                created_at: now,
                token_count: entry.token_count,
                metadata: serde_json::json!({
                    "merged_from": child_id.to_string(),
                    "original_entry_id": entry.id.to_string(),
                }),
                compacted: false,
            };
            self.manager.add_message(
                parent_id,
                new_entry.role.clone(),
                new_entry.content.clone(),
                new_entry.token_count,
            )?;
            tokens_added += new_entry.token_count;
            merged_entries.push(new_entry);
        }

        if archive_child && !merged_entries.is_empty() {
            self.manager.archive_session(child_id)?;
        }

        info!(
            "Merged {} entries ({} tokens) from child {} into parent {}",
            merged_entries.len(),
            tokens_added,
            child_id,
            parent_id
        );

        let entries_merged = merged_entries.len();
        let child_archived = archive_child && !merged_entries.is_empty();

        Ok(MergeResult {
            parent_session_id: *parent_id,
            child_session_id: *child_id,
            merged_entries,
            entries_merged,
            tokens_added,
            child_archived,
            merged_at: now,
        })
    }

    /// List all branches (child sessions) of a parent session.
    pub fn list_branches(&self, parent_id: &Uuid) -> CoreResult<Vec<SessionFork>> {
        self.manager.fork_history(parent_id)
    }

    /// Find the fork record connecting a parent and child.
    fn find_fork_record(&self, parent_id: &Uuid, child_id: &Uuid) -> CoreResult<SessionFork> {
        let forks = self.manager.fork_history(parent_id)?;
        forks
            .into_iter()
            .find(|f| f.child_session_id == *child_id)
            .ok_or_else(|| {
                CoreError::NotFound(format!(
                    "No fork record connecting parent {} and child {}",
                    parent_id, child_id
                ))
            })
    }

    /// Copy a range of transcript entries from one session to another.
    fn copy_transcript_range(
        &self,
        source_id: &Uuid,
        child_id: &Uuid,
        up_to_index: Option<usize>,
    ) -> CoreResult<usize> {
        let entries = self.manager.full_transcript(source_id)?;
        let cutoff = up_to_index.map(|i| i + 1).unwrap_or(entries.len());
        let mut copied = 0usize;
        for entry in entries.into_iter().take(cutoff) {
            self.manager.add_message(
                child_id,
                entry.role.clone(),
                entry.content.clone(),
                entry.token_count,
            )?;
            copied += 1;
        }
        Ok(copied)
    }
}

/// Truncate a UUID to its first 8 characters for display.
fn short_id(id: &Uuid) -> String {
    id.to_string().chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::CreateSessionConfig;
    use crate::models::SessionMode;

    fn manager() -> SessionManager {
        SessionManager::new(crate::storage::SessionStorage::in_memory().unwrap())
    }

    fn seed_messages(manager: &SessionManager, session_id: &Uuid, count: usize) -> Vec<Uuid> {
        let mut ids = Vec::new();
        for i in 0..count {
            let entry = manager
                .add_message(session_id, "user".into(), format!("msg {}", i), 10)
                .unwrap();
            ids.push(entry.id);
        }
        ids
    }

    #[test]
    fn branch_from_copies_prefix() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_system_prompt("sys")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 5);

        let child = brancher
            .branch_from(&parent.id, &ids[2], &BranchConfig::default())
            .unwrap();
        let child_entries = mgr.full_transcript(&child.id).unwrap();
        assert_eq!(child_entries.len(), 3);
        assert_eq!(child_entries[0].content, "msg 0");
        assert_eq!(child_entries[2].content, "msg 2");
    }

    #[test]
    fn branch_from_start_copies_nothing() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        seed_messages(&mgr, &parent.id, 3);

        let child = brancher
            .branch_from_start(&parent.id, &BranchConfig::default())
            .unwrap();
        assert!(mgr.full_transcript(&child.id).unwrap().is_empty());
    }

    #[test]
    fn branch_from_latest_copies_all() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        seed_messages(&mgr, &parent.id, 4);

        let child = brancher
            .branch_from_latest(&parent.id, &BranchConfig::default())
            .unwrap();
        assert_eq!(mgr.full_transcript(&child.id).unwrap().len(), 4);
    }

    #[test]
    fn diff_detects_divergence() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 3);

        let child = brancher
            .branch_from(&parent.id, &ids[1], &BranchConfig::default())
            .unwrap();

        // Diverge both sessions.
        mgr.add_message(&parent.id, "user".into(), "parent msg".into(), 5)
            .unwrap();
        mgr.add_message(&child.id, "user".into(), "child msg".into(), 5)
            .unwrap();

        let diff = brancher.diff(&parent.id, &child.id).unwrap();
        assert!(diff.diverges());
        assert!(!diff.parent_only.is_empty());
        assert!(!diff.child_only.is_empty());
        assert!(diff.divergent_tokens() > 0);
    }

    #[test]
    fn diff_child_ahead() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 2);

        let child = brancher
            .branch_from(&parent.id, &ids[1], &BranchConfig::default())
            .unwrap();

        // Only the child gets new messages.
        mgr.add_message(&child.id, "user".into(), "child only".into(), 3)
            .unwrap();

        let diff = brancher.diff(&parent.id, &child.id).unwrap();
        assert!(diff.child_ahead());
        assert!(!diff.parent_ahead());
    }

    #[test]
    fn merge_appends_child_entries() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 2);

        let child = brancher
            .branch_from(&parent.id, &ids[1], &BranchConfig::default())
            .unwrap();

        mgr.add_message(&child.id, "user".into(), "child divergent".into(), 7)
            .unwrap();

        let before = mgr.full_transcript(&parent.id).unwrap().len();
        let result = brancher
            .merge_branch(&parent.id, &child.id, true)
            .unwrap();
        let after = mgr.full_transcript(&parent.id).unwrap().len();

        assert!(!result.is_noop());
        assert_eq!(result.entries_merged, 1);
        assert_eq!(result.tokens_added, 7);
        assert!(after > before);
        assert!(result.child_archived);
    }

    #[test]
    fn merge_noop_when_no_divergence() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 2);

        let child = brancher
            .branch_from(&parent.id, &ids[1], &BranchConfig::default())
            .unwrap();

        let result = brancher.merge_branch(&parent.id, &child.id, false).unwrap();
        assert!(result.is_noop());
    }

    #[test]
    fn list_branches_returns_forks() {
        let mgr = manager();
        let brancher = SessionBrancher::new(mgr);
        let mgr = brancher.manager();
        let agent = Uuid::new_v4();
        let parent = mgr
            .create(
                CreateSessionConfig::new(agent)
                    .with_name("parent")
                    .with_mode(SessionMode::Chat),
            )
            .unwrap();
        let ids = seed_messages(&mgr, &parent.id, 3);

        brancher
            .branch_from(&parent.id, &ids[0], &BranchConfig::default())
            .unwrap();
        brancher
            .branch_from(&parent.id, &ids[2], &BranchConfig::default())
            .unwrap();

        let branches = brancher.list_branches(&parent.id).unwrap();
        assert_eq!(branches.len(), 2);
    }

    #[test]
    fn branch_config_builder_works() {
        let config = BranchConfig::default()
            .with_name("custom branch")
            .without_transcript();
        assert_eq!(config.name.as_deref(), Some("custom branch"));
        assert!(!config.copy_transcript);
    }

    #[test]
    fn diff_summary_is_human_readable() {
        let diff = BranchDiff {
            parent_session_id: Uuid::new_v4(),
            child_session_id: Uuid::new_v4(),
            branch_point: Uuid::new_v4(),
            common_prefix: Vec::new(),
            parent_only: vec![],
            child_only: vec![],
            common_tokens: 0,
            parent_only_tokens: 0,
            child_only_tokens: 0,
            computed_at: Utc::now(),
        };
        let summary = diff.summary();
        assert!(summary.contains("common=0"));
    }
}
