//! Crash recovery: detect interrupted turns and reconstruct state.
//!
//! Mirrors the Python `engine/recovery/crash.py`. When the process restarts
//! after a crash, the recovery module scans for in-progress turns that were
//! not finalized and reconstructs their state so the agent can resume or
//! surface the interruption cleanly.

use crate::agent::AgentState;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, Usage};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

/// Configuration for crash recovery.
#[derive(Debug, Clone)]
pub struct CrashRecoveryConfig {
    /// The directory where turn snapshots are persisted.
    pub snapshot_dir: PathBuf,
    /// The maximum age of a snapshot before it's considered stale and ignored.
    pub max_snapshot_age: Duration,
    /// Whether to automatically resume interrupted turns.
    pub auto_resume: bool,
    /// Whether to delete snapshots after successful recovery.
    pub cleanup_after_recovery: bool,
}

impl Default for CrashRecoveryConfig {
    fn default() -> Self {
        Self {
            snapshot_dir: PathBuf::from(".opensquilla/snapshots"),
            max_snapshot_age: Duration::from_secs(3600),
            auto_resume: false,
            cleanup_after_recovery: true,
        }
    }
}

/// The status of a crash recovery operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStatus {
    /// No interrupted turns were found.
    None,
    /// An interrupted turn was found and its state was reconstructed.
    Recovered {
        /// The turn ID that was recovered.
        turn_id: String,
        /// The number of messages reconstructed.
        message_count: usize,
    },
    /// The snapshot was stale and ignored.
    Stale {
        /// The turn ID of the stale snapshot.
        turn_id: String,
    },
    /// Recovery failed with an error.
    Failed {
        /// The turn ID that failed to recover.
        turn_id: String,
        /// The error message.
        error: String,
    },
}

impl RecoveryStatus {
    /// Whether recovery succeeded.
    pub fn is_recovered(&self) -> bool {
        matches!(self, RecoveryStatus::Recovered { .. })
    }

    /// Whether recovery failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, RecoveryStatus::Failed { .. })
    }

    /// The turn ID associated with this status, if any.
    pub fn turn_id(&self) -> Option<&str> {
        match self {
            RecoveryStatus::Recovered { turn_id, .. }
            | RecoveryStatus::Stale { turn_id }
            | RecoveryStatus::Failed { turn_id, .. } => Some(turn_id),
            RecoveryStatus::None => None,
        }
    }
}

/// A persisted snapshot of an in-progress turn.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CrashSnapshot {
    /// The turn ID.
    pub turn_id: String,
    /// The session ID.
    pub session_id: String,
    /// The agent ID.
    pub agent_id: String,
    /// The agent state at the time of the snapshot.
    pub agent_state: String,
    /// The model being used.
    pub model: String,
    /// The provider serving the turn.
    pub provider: String,
    /// The current tool round.
    pub tool_round: u32,
    /// The maximum tool rounds.
    pub max_tool_rounds: u32,
    /// The messages accumulated so far, serialized as JSON.
    pub messages_json: String,
    /// The token usage so far.
    pub usage: Usage,
    /// When the snapshot was created (epoch ms).
    pub created_at_ms: i64,
    /// When the snapshot was last updated (epoch ms).
    pub updated_at_ms: i64,
    /// Whether the turn was finalized.
    pub finalized: bool,
}

impl CrashSnapshot {
    /// Create a new snapshot for a turn.
    pub fn new(turn_id: impl Into<String>, session_id: impl Into<String>) -> Self {
        let now = current_epoch_ms();
        Self {
            turn_id: turn_id.into(),
            session_id: session_id.into(),
            agent_id: String::new(),
            agent_state: AgentState::Thinking.to_string(),
            model: String::new(),
            provider: String::new(),
            tool_round: 0,
            max_tool_rounds: 10,
            messages_json: "[]".to_string(),
            usage: Usage::default(),
            created_at_ms: now,
            updated_at_ms: now,
            finalized: false,
        }
    }

    /// Update the messages in the snapshot.
    pub fn update_messages(&mut self, messages: &[Message]) -> Result<()> {
        self.messages_json = serde_json::to_string(messages)?;
        self.updated_at_ms = current_epoch_ms();
        Ok(())
    }

    /// Deserialize the messages from the snapshot.
    pub fn messages(&self) -> Result<Vec<Message>> {
        Ok(serde_json::from_str(&self.messages_json)?)
    }

    /// The age of the snapshot in milliseconds.
    pub fn age_ms(&self) -> i64 {
        current_epoch_ms() - self.updated_at_ms
    }

    /// Whether the snapshot is stale (older than the given duration).
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.age_ms() > max_age.as_millis() as i64
    }

    /// Mark the snapshot as finalized.
    pub fn finalize(&mut self) {
        self.finalized = true;
        self.updated_at_ms = current_epoch_ms();
    }

    /// The snapshot file path.
    pub fn snapshot_path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("{}.json", self.turn_id))
    }
}

/// The crash recovery manager.
#[derive(Debug)]
pub struct CrashRecovery {
    /// The recovery configuration.
    config: CrashRecoveryConfig,
}

impl CrashRecovery {
    /// Create a new crash recovery manager.
    pub fn new(config: CrashRecoveryConfig) -> Self {
        Self { config }
    }

    /// The recovery configuration.
    pub fn config(&self) -> &CrashRecoveryConfig {
        &self.config
    }

    /// Persist a snapshot to disk.
    pub fn save_snapshot(&self, snapshot: &CrashSnapshot) -> Result<PathBuf> {
        let path = snapshot.snapshot_path(&self.config.snapshot_dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let payload = serde_json::to_string_pretty(snapshot)?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &payload)?;
        std::fs::rename(&tmp, &path)?;
        debug!(turn_id = %snapshot.turn_id, path = %path.display(), "snapshot saved");
        Ok(path)
    }

    /// Load a snapshot by turn ID.
    pub fn load_snapshot(&self, turn_id: &str) -> Result<Option<CrashSnapshot>> {
        let path = self.config.snapshot_dir.join(format!("{turn_id}.json"));
        match std::fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<CrashSnapshot>(&raw) {
                Ok(snapshot) => Ok(Some(snapshot)),
                Err(e) => {
                    warn!(turn_id = %turn_id, error = %e, "failed to parse snapshot");
                    Ok(None)
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(opensquilla_core::error::Error::Io(e)),
        }
    }

    /// List all snapshot turn IDs in the snapshot directory.
    pub fn list_snapshots(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        if !self.config.snapshot_dir.exists() {
            return Ok(ids);
        }
        for entry in std::fs::read_dir(&self.config.snapshot_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(id) = name.strip_suffix(".json") {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }

    /// Delete a snapshot file.
    pub fn delete_snapshot(&self, turn_id: &str) -> Result<()> {
        let path = self.config.snapshot_dir.join(format!("{turn_id}.json"));
        if path.exists() {
            std::fs::remove_file(&path)?;
            debug!(turn_id = %turn_id, "snapshot deleted");
        }
        Ok(())
    }

    /// Scan for interrupted (non-finalized) snapshots.
    pub fn find_interrupted(&self) -> Result<Vec<CrashSnapshot>> {
        let ids = self.list_snapshots()?;
        let mut interrupted = Vec::new();
        for id in ids {
            if let Some(snapshot) = self.load_snapshot(&id)? {
                if !snapshot.finalized && !snapshot.is_stale(self.config.max_snapshot_age) {
                    interrupted.push(snapshot);
                }
            }
        }
        Ok(interrupted)
    }

    /// Recover from a crash by finding and reconstructing interrupted turns.
    ///
    /// Returns the recovery status for each interrupted snapshot found.
    pub fn recover(&self) -> Result<Vec<RecoveryStatus>> {
        let interrupted = self.find_interrupted()?;
        if interrupted.is_empty() {
            return Ok(vec![RecoveryStatus::None]);
        }

        let mut statuses = Vec::new();
        for snapshot in interrupted {
            info!(
                turn_id = %snapshot.turn_id,
                session_id = %snapshot.session_id,
                agent_state = %snapshot.agent_state,
                tool_round = snapshot.tool_round,
                age_ms = snapshot.age_ms(),
                "found interrupted turn, recovering"
            );

            // Check staleness.
            if snapshot.is_stale(self.config.max_snapshot_age) {
                warn!(
                    turn_id = %snapshot.turn_id,
                    age_ms = snapshot.age_ms(),
                    "snapshot is stale, skipping"
                );
                statuses.push(RecoveryStatus::Stale {
                    turn_id: snapshot.turn_id.clone(),
                });
                continue;
            }

            // Reconstruct the messages.
            match snapshot.messages() {
                Ok(messages) => {
                    let message_count = messages.len();
                    statuses.push(RecoveryStatus::Recovered {
                        turn_id: snapshot.turn_id.clone(),
                        message_count,
                    });

                    if self.config.cleanup_after_recovery {
                        self.delete_snapshot(&snapshot.turn_id)?;
                    }
                }
                Err(e) => {
                    warn!(
                        turn_id = %snapshot.turn_id,
                        error = %e,
                        "failed to reconstruct messages from snapshot"
                    );
                    statuses.push(RecoveryStatus::Failed {
                        turn_id: snapshot.turn_id.clone(),
                        error: e.to_string(),
                    });
                }
            }
        }
        Ok(statuses)
    }

    /// Update a snapshot with the latest messages and state.
    pub fn update_snapshot(
        &self,
        snapshot: &mut CrashSnapshot,
        messages: &[Message],
        agent_state: AgentState,
        usage: Usage,
        tool_round: u32,
    ) -> Result<()> {
        snapshot.update_messages(messages)?;
        snapshot.agent_state = agent_state.to_string();
        snapshot.usage = usage;
        snapshot.tool_round = tool_round;
        snapshot.updated_at_ms = current_epoch_ms();
        self.save_snapshot(snapshot)?;
        Ok(())
    }

    /// Mark a snapshot as finalized and optionally clean it up.
    pub fn finalize_snapshot(&self, turn_id: &str) -> Result<()> {
        if let Some(mut snapshot) = self.load_snapshot(turn_id)? {
            snapshot.finalize();
            if self.config.cleanup_after_recovery {
                self.delete_snapshot(turn_id)?;
            } else {
                self.save_snapshot(&snapshot)?;
            }
        }
        Ok(())
    }
}

/// The current epoch time in milliseconds.
pub fn current_epoch_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_test_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("opensquilla-test-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn config_with_tempdir() -> CrashRecoveryConfig {
        CrashRecoveryConfig {
            snapshot_dir: unique_test_dir(),
            max_snapshot_age: Duration::from_secs(3600),
            auto_resume: false,
            cleanup_after_recovery: true,
        }
    }

    #[test]
    fn test_snapshot_creation() {
        let snapshot = CrashSnapshot::new("t1", "s1");
        assert_eq!(snapshot.turn_id, "t1");
        assert_eq!(snapshot.session_id, "s1");
        assert!(!snapshot.finalized);
        assert_eq!(snapshot.tool_round, 0);
    }

    #[test]
    fn test_snapshot_messages_roundtrip() {
        let mut snapshot = CrashSnapshot::new("t1", "s1");
        let messages = vec![Message::user("hello"), Message::assistant("world")];
        snapshot.update_messages(&messages).unwrap();
        let recovered = snapshot.messages().unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].text_content(), "hello");
    }

    #[test]
    fn test_snapshot_staleness() {
        let mut snapshot = CrashSnapshot::new("t1", "s1");
        snapshot.updated_at_ms = current_epoch_ms() - 7_200_000; // 2 hours ago
        assert!(snapshot.is_stale(Duration::from_secs(3600)));
        assert!(!snapshot.is_stale(Duration::from_secs(14400)));
    }

    #[test]
    fn test_save_and_load_snapshot() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        let snapshot = CrashSnapshot::new("t1", "s1");
        let path = recovery.save_snapshot(&snapshot).unwrap();
        assert!(path.exists());

        let loaded = recovery.load_snapshot("t1").unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().turn_id, "t1");
    }

    #[test]
    fn test_load_missing_snapshot() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        let loaded = recovery.load_snapshot("nonexistent").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_list_snapshots() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        recovery.save_snapshot(&CrashSnapshot::new("t1", "s1")).unwrap();
        recovery.save_snapshot(&CrashSnapshot::new("t2", "s1")).unwrap();
        let ids = recovery.list_snapshots().unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_find_interrupted() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        let mut s1 = CrashSnapshot::new("t1", "s1");
        s1.finalized = false;
        recovery.save_snapshot(&s1).unwrap();

        let mut s2 = CrashSnapshot::new("t2", "s1");
        s2.finalized = true;
        recovery.save_snapshot(&s2).unwrap();

        let interrupted = recovery.find_interrupted().unwrap();
        assert_eq!(interrupted.len(), 1);
        assert_eq!(interrupted[0].turn_id, "t1");
    }

    #[test]
    fn test_recover_none_when_no_snapshots() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        let statuses = recovery.recover().unwrap();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0], RecoveryStatus::None);
    }

    #[test]
    fn test_recover_interrupted() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        let mut snapshot = CrashSnapshot::new("t1", "s1");
        let messages = vec![Message::user("hello"), Message::assistant("world")];
        snapshot.update_messages(&messages).unwrap();
        recovery.save_snapshot(&snapshot).unwrap();

        let statuses = recovery.recover().unwrap();
        assert_eq!(statuses.len(), 1);
        assert!(statuses[0].is_recovered());
    }

    #[test]
    fn test_finalize_snapshot() {
        let config = config_with_tempdir();
        let recovery = CrashRecovery::new(config);
        recovery.save_snapshot(&CrashSnapshot::new("t1", "s1")).unwrap();
        recovery.finalize_snapshot("t1").unwrap();
        // With cleanup, the snapshot should be deleted.
        let loaded = recovery.load_snapshot("t1").unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_recovery_status_is_methods() {
        let recovered = RecoveryStatus::Recovered {
            turn_id: "t1".to_string(),
            message_count: 5,
        };
        assert!(recovered.is_recovered());
        assert!(!recovered.is_failed());
        assert_eq!(recovered.turn_id(), Some("t1"));
    }
}
