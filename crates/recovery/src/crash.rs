use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use tracing::{debug, error, info};

/// A crash snapshot for recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashSnapshot {
    /// Unique snapshot ID.
    pub id: String,
    /// Timestamp when the crash occurred.
    pub timestamp: DateTime<Utc>,
    /// The error message.
    pub error_message: String,
    /// The session ID that was active during the crash.
    pub session_id: Option<String>,
    /// Checksum of the crash data for integrity.
    pub checksum: String,
    /// Serialized crash context.
    pub context: String,
}

/// Crash recovery manager.
#[derive(Debug, Clone)]
pub struct CrashRecovery {
    config: Config,
    snapshots: Arc<RwLock<Vec<CrashSnapshot>>>,
    snapshot_dir: PathBuf,
    max_snapshots: usize,
}

impl CrashRecovery {
    /// Create a new crash recovery manager.
    pub fn new(config: &Config) -> Self {
        let snapshot_dir = config
            .get("recovery.snapshot_dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let mut dir = dirs_next::data_dir().unwrap_or_else(|| PathBuf::from("."));
                dir.push("opensquilla");
                dir.push("snapshots");
                dir
            });

        let max_snapshots = config
            .get("recovery.max_snapshots")
            .unwrap_or_else(|| "10".to_string())
            .parse::<usize>()
            .unwrap_or(10);

        // Create snapshot directory
        std::fs::create_dir_all(&snapshot_dir).ok();

        info!(
            "Crash recovery initialized (dir: {}, max snapshots: {max_snapshots})",
            snapshot_dir.display()
        );

        Self {
            config: config.clone(),
            snapshots: Arc::new(RwLock::new(Vec::new())),
            snapshot_dir,
            max_snapshots,
        }
    }

    /// Create a crash recovery manager with a specific snapshot directory (for testing).
    pub fn with_snapshot_dir(config: &Config, snapshot_dir: PathBuf) -> Self {
        let max_snapshots = config
            .get("recovery.max_snapshots")
            .unwrap_or_else(|| "10".to_string())
            .parse::<usize>()
            .unwrap_or(10);

        std::fs::create_dir_all(&snapshot_dir).ok();

        Self {
            config: config.clone(),
            snapshots: Arc::new(RwLock::new(Vec::new())),
            snapshot_dir,
            max_snapshots,
        }
    }

    /// Record a crash snapshot.
    pub async fn record_crash(
        &self,
        error_message: &str,
        session_id: Option<&str>,
        context: &str,
    ) -> Result<CrashSnapshot, RecoveryError> {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();

        // Compute checksum for integrity verification
        let checksum_data = format!("{id}{timestamp}{error_message}{context}");
        let checksum = format!("{:x}", Sha256::digest(checksum_data.as_bytes()));

        let snapshot = CrashSnapshot {
            id,
            timestamp,
            error_message: error_message.to_string(),
            session_id: session_id.map(|s| s.to_string()),
            checksum,
            context: context.to_string(),
        };

        // Save to file
        let snapshot_path = self.snapshot_dir.join(format!("{}.json", snapshot.id));
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| RecoveryError::SerializationError(e.to_string()))?;
        std::fs::write(&snapshot_path, &json).map_err(|e| RecoveryError::IoError(e.to_string()))?;

        // Keep in memory
        let mut snapshots = self.snapshots.write().await;
        snapshots.push(snapshot.clone());

        // Trim to max
        while snapshots.len() > self.max_snapshots {
            let removed = snapshots.remove(0);
            // Remove file
            let old_path = self.snapshot_dir.join(format!("{}.json", removed.id));
            std::fs::remove_file(&old_path).ok();
        }

        error!(
            "Crash recorded: {error_message} (snapshot: {})",
            snapshot.id
        );

        Ok(snapshot)
    }

    /// Load a crash snapshot from file.
    pub async fn load_snapshot(&self, snapshot_id: &str) -> Result<CrashSnapshot, RecoveryError> {
        let snapshot_path = self.snapshot_dir.join(format!("{snapshot_id}.json"));
        let json = std::fs::read_to_string(&snapshot_path)
            .map_err(|_e| RecoveryError::SnapshotNotFound(snapshot_id.to_string()))?;
        let snapshot: CrashSnapshot = serde_json::from_str(&json)
            .map_err(|e| RecoveryError::DeserializationError(e.to_string()))?;

        // Verify checksum
        let checksum_data = format!(
            "{}{}{}{}",
            snapshot.id, snapshot.timestamp, snapshot.error_message, snapshot.context
        );
        let expected_checksum = format!("{:x}", Sha256::digest(checksum_data.as_bytes()));
        if snapshot.checksum != expected_checksum {
            return Err(RecoveryError::IntegrityError(
                "Snapshot checksum mismatch".to_string(),
            ));
        }

        Ok(snapshot)
    }

    /// List all crash snapshots.
    pub async fn list_snapshots(&self) -> Vec<CrashSnapshot> {
        let snapshots = self.snapshots.read().await;
        let mut result = snapshots.clone();

        // Also load from disk
        if let Ok(entries) = std::fs::read_dir(&self.snapshot_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(json) = std::fs::read_to_string(&path) {
                        if let Ok(snapshot) = serde_json::from_str::<CrashSnapshot>(&json) {
                            if !result.iter().any(|s| s.id == snapshot.id) {
                                result.push(snapshot);
                            }
                        }
                    }
                }
            }
        }

        result.sort_by_key(|b| std::cmp::Reverse(b.timestamp));
        result
    }

    /// Attempt to recover from a crash snapshot.
    ///
    /// This verifies the snapshot's integrity (checksum), re-hydrates the
    /// session id from the snapshot, and classifies the recovery outcome. It
    /// does NOT replay the interrupted turn — turn replay is the engine's
    /// responsibility (`crates/engine/src/recovery/replay.rs`), which has
    /// access to the session store. Here we surface the verified snapshot so a
    /// caller can decide whether to replay, resume, or mark the session for
    /// manual review.
    pub async fn recover_from_snapshot(
        &self,
        snapshot: &CrashSnapshot,
    ) -> Result<RecoveryResult, RecoveryError> {
        info!("Attempting recovery from snapshot {}", snapshot.id);

        // Re-verify integrity before trusting the snapshot.
        let checksum_data = format!(
            "{}{}{}{}",
            snapshot.id, snapshot.timestamp, snapshot.error_message, snapshot.context
        );
        let expected = format!("{:x}", Sha256::digest(checksum_data.as_bytes()));
        if snapshot.checksum != expected {
            return Err(RecoveryError::IntegrityError(format!(
                "snapshot {} checksum mismatch — refusing to recover tampered snapshot",
                snapshot.id
            )));
        }

        let session_recovered = snapshot.session_id.is_some();
        let message = if session_recovered {
            format!(
                "Snapshot {} verified; session {} captured — replay/restore must be driven by the session manager.",
                snapshot.id,
                snapshot.session_id.as_ref().expect("checked above")
            )
        } else {
            format!(
                "Snapshot {} verified; no active session was captured — context preserved for manual review.",
                snapshot.id
            )
        };

        Ok(RecoveryResult {
            snapshot_id: snapshot.id.clone(),
            recovered: session_recovered,
            message,
            timestamp: Utc::now(),
        })
    }

    /// Delete a crash snapshot.
    pub async fn delete_snapshot(&self, snapshot_id: &str) -> Result<(), RecoveryError> {
        let snapshot_path = self.snapshot_dir.join(format!("{snapshot_id}.json"));
        if snapshot_path.exists() {
            std::fs::remove_file(&snapshot_path)
                .map_err(|e| RecoveryError::IoError(e.to_string()))?;
        }

        let mut snapshots = self.snapshots.write().await;
        snapshots.retain(|s| s.id != snapshot_id);

        debug!("Deleted snapshot {snapshot_id}");
        Ok(())
    }

    /// Clear all crash snapshots.
    pub async fn clear_snapshots(&self) -> Result<(), RecoveryError> {
        let mut snapshots = self.snapshots.write().await;
        snapshots.clear();

        if let Ok(entries) = std::fs::read_dir(&self.snapshot_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    std::fs::remove_file(&path).ok();
                }
            }
        }

        info!("All crash snapshots cleared");
        Ok(())
    }

    /// Get the snapshot directory path.
    pub fn snapshot_dir(&self) -> &PathBuf {
        &self.snapshot_dir
    }

    /// Get a reference to the underlying config.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// A lightweight session state used for integrity validation.
///
/// The recovery crate does not depend on the session crate, so this is a plain
/// projection of the session's turn history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub id: String,
    pub messages: Vec<crate::merge::SessionMessage>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// The result of validating a session's integrity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionValidation {
    pub session_id: String,
    pub valid: bool,
    pub issues: Vec<String>,
    pub message_count: usize,
}

/// A report covering a crash-recovery pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashReport {
    pub crash_id: String,
    pub timestamp: DateTime<Utc>,
    pub error_message: String,
    pub affected_sessions: Vec<String>,
    pub recovery_actions: Vec<String>,
    pub recovered: bool,
    pub recovered_session_ids: Vec<String>,
}

/// The roles accepted by session validation.
const VALID_ROLES: &[&str] = &["user", "assistant", "system", "tool"];

impl CrashRecovery {
    /// Detect any outstanding crash snapshots and attempt to recover from the
    /// most recent ones.
    ///
    /// Every recoverable snapshot is replayed through
    /// [`CrashRecovery::recover_from_snapshot`]; successful recoveries are
    /// rolled into the returned [`CrashReport`].
    pub async fn recover_from_crash(&self) -> Result<CrashReport, RecoveryError> {
        let snapshots = self.list_snapshots().await;
        let mut affected_sessions = Vec::new();
        let mut recovery_actions = Vec::new();
        let mut recovered_session_ids = Vec::new();
        let mut recovered = false;

        for snapshot in &snapshots {
            if let Some(session_id) = &snapshot.session_id {
                affected_sessions.push(session_id.clone());
            }
            match self.recover_from_snapshot(snapshot).await {
                Ok(result) if result.recovered => {
                    recovered = true;
                    if let Some(session_id) = &snapshot.session_id {
                        recovered_session_ids.push(session_id.clone());
                    }
                    recovery_actions.push(format!("recovered_snapshot:{}", snapshot.id));
                }
                Ok(_) => recovery_actions.push(format!("snapshot_noop:{}", snapshot.id)),
                Err(e) => {
                    recovery_actions.push(format!("snapshot_failed:{}:{}", snapshot.id, e));
                }
            }
        }

        if snapshots.is_empty() {
            recovery_actions.push("no_snapshots_found".to_string());
        }

        Ok(CrashReport {
            crash_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            error_message: snapshots
                .first()
                .map(|s| s.error_message.clone())
                .unwrap_or_default(),
            affected_sessions,
            recovery_actions,
            recovered,
            recovered_session_ids,
        })
    }

    /// Validate a session's integrity: role sanity, orphan tool results, and
    /// timestamp ordering.
    pub async fn validate_session_state(&self, session: &SessionState) -> SessionValidation {
        let mut issues = Vec::new();
        let mut tool_calls: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (i, message) in session.messages.iter().enumerate() {
            if !VALID_ROLES.contains(&message.role.as_str()) {
                issues.push(format!("message[{i}] has unknown role '{}'", message.role));
            }
            if message.content.trim().is_empty() {
                issues.push(format!("message[{i}] has empty content"));
            }
            if message.role == "tool" {
                match message.metadata.get("tool_call_id") {
                    Some(id) if !id.is_empty() => {
                        tool_calls.insert(id.clone());
                    }
                    _ => issues.push(format!(
                        "message[{i}] is a tool result without a tool_call_id"
                    )),
                }
            }
        }

        if session.messages.len() > 1 {
            for pair in session.messages.windows(2) {
                if pair[0].timestamp > pair[1].timestamp {
                    issues.push("messages are not sorted by timestamp".to_string());
                    break;
                }
            }
        }

        if session.messages.is_empty() {
            issues.push("session has no messages".to_string());
        }

        SessionValidation {
            session_id: session.id.clone(),
            valid: issues.is_empty(),
            issues,
            message_count: session.messages.len(),
        }
    }

    /// Repair a corrupted session in place, returning the list of fixes applied.
    ///
    /// Fixes are additive and deterministic:
    ///
    /// 1. empty-content messages are dropped,
    /// 2. tool results missing a `tool_call_id` get a synthesized one,
    /// 3. messages are re-sorted by timestamp.
    pub async fn fix_corrupted_session(
        &self,
        session: &mut SessionState,
    ) -> Result<Vec<String>, RecoveryError> {
        let mut fixes = Vec::new();

        let before = session.messages.len();
        session.messages.retain(|m| !m.content.trim().is_empty());
        if session.messages.len() != before {
            fixes.push(format!(
                "dropped {} empty-content message(s)",
                before - session.messages.len()
            ));
        }

        for (i, message) in session.messages.iter_mut().enumerate() {
            if message.role == "tool"
                && message
                    .metadata
                    .get("tool_call_id")
                    .is_none_or(|v| v.is_empty())
            {
                let id = format!("repair_{i}_{}", uuid::Uuid::new_v4());
                message.metadata.insert("tool_call_id".to_string(), id);
                fixes.push(format!("message[{i}] synthesized missing tool_call_id"));
            }
        }

        session.messages.sort_by_key(|a| a.timestamp);

        Ok(fixes)
    }
}

/// Result of a recovery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    pub snapshot_id: String,
    pub recovered: bool,
    pub message: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    #[error("Snapshot not found: {0}")]
    SnapshotNotFound(String),

    #[error("Integrity error: {0}")]
    IntegrityError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("IO error: {0}")]
    IoError(String),
}

// Simple dirs_next function equivalent
mod dirs_next {
    use std::path::PathBuf;

    pub fn data_dir() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            std::env::var("XDG_DATA_HOME")
                .ok()
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var("HOME")
                        .ok()
                        .map(|h| PathBuf::from(h).join(".local").join("share"))
                })
        }
        #[cfg(target_os = "macos")]
        {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
        }
        #[cfg(target_os = "windows")]
        {
            std::env::var("APPDATA").ok().map(PathBuf::from)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            std::env::var("HOME").ok().map(PathBuf::from)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::config::GatewayConfig;

    fn test_config() -> Config {
        Config {
            gateway: GatewayConfig::default(),
            providers: Vec::new(),
            channels: Vec::new(),
            models: None,
            sandbox: None,
            skills: None,
            scheduler: None,
            observability: None,
            control_ui: None,
        }
    }

    fn temp_recovery() -> CrashRecovery {
        let dir =
            std::env::temp_dir().join(format!("opensquilla-crash-test-{}", uuid::Uuid::new_v4()));
        CrashRecovery::with_snapshot_dir(&test_config(), dir)
    }

    #[tokio::test]
    async fn test_record_and_load_snapshot() {
        let recovery = temp_recovery();
        let snapshot = recovery
            .record_crash("test error", Some("session-123"), "context data")
            .await
            .unwrap();

        assert!(!snapshot.id.is_empty());
        assert_eq!(snapshot.error_message, "test error");
        assert_eq!(snapshot.session_id.as_deref(), Some("session-123"));

        let loaded = recovery.load_snapshot(&snapshot.id).await.unwrap();
        assert_eq!(loaded.id, snapshot.id);
        assert_eq!(loaded.error_message, "test error");
    }

    #[tokio::test]
    async fn test_list_snapshots() {
        let recovery = temp_recovery();
        recovery
            .record_crash("error 1", None, "ctx1")
            .await
            .unwrap();
        recovery
            .record_crash("error 2", None, "ctx2")
            .await
            .unwrap();

        let snapshots = recovery.list_snapshots().await;
        assert_eq!(snapshots.len(), 2);
        // Should be sorted by timestamp descending
        assert!(snapshots[0].timestamp >= snapshots[1].timestamp);
    }

    #[tokio::test]
    async fn test_delete_snapshot() {
        let recovery = temp_recovery();
        let snapshot = recovery
            .record_crash("to delete", None, "ctx")
            .await
            .unwrap();

        recovery.delete_snapshot(&snapshot.id).await.unwrap();
        let snapshots = recovery.list_snapshots().await;
        assert!(snapshots.iter().all(|s| s.id != snapshot.id));
    }

    #[tokio::test]
    async fn test_checksum_verification() {
        let recovery = temp_recovery();
        let snapshot = recovery
            .record_crash("checksum test", None, "data")
            .await
            .unwrap();

        // Loading should succeed (checksum matches)
        let result = recovery.load_snapshot(&snapshot.id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_clear_snapshots() {
        let recovery = temp_recovery();
        recovery.record_crash("a", None, "x").await.unwrap();
        recovery.record_crash("b", None, "y").await.unwrap();

        recovery.clear_snapshots().await.unwrap();
        let snapshots = recovery.list_snapshots().await;
        assert!(snapshots.is_empty());
    }

    #[tokio::test]
    async fn test_recover_from_snapshot() {
        let recovery = temp_recovery();
        let snapshot = recovery
            .record_crash("crash error", Some("session-456"), "context")
            .await
            .unwrap();

        let result = recovery.recover_from_snapshot(&snapshot).await.unwrap();
        assert!(result.recovered);
        assert_eq!(result.snapshot_id, snapshot.id);
    }
}

#[cfg(test)]
mod crash_expansion_tests {
    use super::*;
    use crate::merge::SessionMessage;

    fn session_state() -> SessionState {
        let now = Utc::now();
        SessionState {
            id: "session-1".to_string(),
            messages: vec![
                SessionMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    timestamp: now,
                    metadata: Default::default(),
                },
                SessionMessage {
                    role: "tool".to_string(),
                    content: "result".to_string(),
                    timestamp: now,
                    metadata: Default::default(), // missing tool_call_id
                },
            ],
            created_at: now,
            updated_at: now,
            metadata: Default::default(),
        }
    }

    fn temp_recovery() -> CrashRecovery {
        let dir =
            std::env::temp_dir().join(format!("opensquilla-crash-exp-{}", uuid::Uuid::new_v4()));
        CrashRecovery::with_snapshot_dir(&opensquilla_core::config::Config::default(), dir)
    }

    #[tokio::test]
    async fn test_validate_flags_orphan_tool_result() {
        let recovery = temp_recovery();
        let validation = recovery.validate_session_state(&session_state()).await;
        assert!(!validation.valid);
        assert!(validation.issues.iter().any(|i| i.contains("tool_call_id")));
    }

    #[tokio::test]
    async fn test_fix_corrupted_session() {
        let recovery = temp_recovery();
        let mut session = session_state();
        let fixes = recovery.fix_corrupted_session(&mut session).await.unwrap();
        assert!(fixes.iter().any(|f| f.contains("tool_call_id")));
        // The tool message now carries a synthesized id.
        let tool = session.messages.iter().find(|m| m.role == "tool").unwrap();
        assert!(tool.metadata.contains_key("tool_call_id"));
    }

    #[tokio::test]
    async fn test_fix_drops_empty_messages() {
        let recovery = temp_recovery();
        let mut session = session_state();
        session.messages.push(SessionMessage {
            role: "assistant".to_string(),
            content: "   ".to_string(),
            timestamp: Utc::now(),
            metadata: Default::default(),
        });
        let before = session.messages.len();
        let fixes = recovery.fix_corrupted_session(&mut session).await.unwrap();
        assert_eq!(session.messages.len(), before - 1);
        assert!(fixes.iter().any(|f| f.contains("empty")));
    }

    #[tokio::test]
    async fn test_recover_from_crash_empty() {
        let recovery = temp_recovery();
        let report = recovery.recover_from_crash().await.unwrap();
        assert!(!report.recovered);
        assert!(
            report
                .recovery_actions
                .iter()
                .any(|a| a == "no_snapshots_found")
        );
    }

    #[tokio::test]
    async fn test_recover_from_crash_with_snapshot() {
        let recovery = temp_recovery();
        recovery
            .record_crash("boom", Some("session-9"), "ctx")
            .await
            .unwrap();
        let report = recovery.recover_from_crash().await.unwrap();
        assert!(report.recovered);
        assert!(
            report
                .recovered_session_ids
                .contains(&"session-9".to_string())
        );
    }
}
