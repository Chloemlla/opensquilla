//! Session archiving.
//!
//! Archives sessions to durable storage (JSON snapshots) so that long-running
//! installs can reclaim the in-memory/active session set while preserving the
//! ability to restore a session later. Mirrors the Python archiving helpers
//! that snapshotted session state to disk.
//!
//! The archive format is a versioned JSON file per session. The session id is
//! used as the file stem so restores are deterministic.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Current archive format version.
pub const ARCHIVE_VERSION: u32 = 1;
/// Directory name used under the data dir for archives.
pub const DEFAULT_ARCHIVE_DIR: &str = "archives";

/// Metadata recorded when a session is archived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveMeta {
    /// Archive format version.
    pub version: u32,
    /// When the archive was written.
    pub archived_at: DateTime<Utc>,
    /// The archived session id.
    pub session_id: String,
    /// Number of messages in the snapshot.
    pub message_count: usize,
    /// Number of bytes in the snapshot.
    pub size_bytes: usize,
}

/// A full archive snapshot of a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionArchive {
    /// Archive metadata.
    pub meta: ArchiveMeta,
    /// Session snapshot payload (transcript, settings, etc).
    pub payload: serde_json::Value,
}

impl SessionArchive {
    /// Build a snapshot with the current version.
    pub fn new(session_id: impl Into<String>, payload: serde_json::Value) -> Self {
        let message_count = payload
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);
        Self {
            meta: ArchiveMeta {
                version: ARCHIVE_VERSION,
                archived_at: Utc::now(),
                session_id: session_id.into(),
                message_count,
                size_bytes: 0,
            },
            payload,
        }
    }

    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AppError> {
        serde_json::to_vec(self)
            .map_err(|e| AppError::internal(format!("Failed to serialize archive: {e}")))
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AppError> {
        serde_json::from_slice(bytes)
            .map_err(|e| AppError::internal(format!("Failed to deserialize archive: {e}")))
    }
}

/// File-based session archive store.
///
/// Thread-safe via an internal `RwLock` over an in-memory index; the actual
/// archive files live on disk under the configured directory.
#[derive(Clone)]
pub struct SessionArchiver {
    dir: PathBuf,
    index: std::sync::Arc<RwLock<std::collections::HashMap<String, ArchiveMeta>>>,
}

impl SessionArchiver {
    /// Create an archiver rooted at the given directory, creating it if
    /// missing.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| AppError::internal(format!("Cannot create archive dir: {e}")))?;
        Ok(Self {
            dir,
            index: std::sync::Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Create an archiver using the default archive directory under the data
    /// local dir.
    pub fn default_path() -> Result<Self, AppError> {
        let base = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."));
        Self::new(base.join(DEFAULT_ARCHIVE_DIR))
    }

    /// Path on disk for a session archive.
    pub fn archive_path(&self, session_id: &str) -> PathBuf {
        let safe = sanitize_session_id(session_id);
        self.dir.join(format!("{safe}.json"))
    }

    /// Archive a session snapshot. Overwrites any existing archive.
    pub fn archive(
        &self,
        session_id: &str,
        payload: serde_json::Value,
    ) -> Result<ArchiveMeta, AppError> {
        let mut snapshot = SessionArchive::new(session_id, payload);
        // Compute the serialized size so the metadata is accurate.
        snapshot.meta.size_bytes = serde_json::to_vec(&snapshot)
            .map_err(|e| AppError::internal(format!("Failed to serialize archive: {e}")))?
            .len();
        let bytes = snapshot.to_bytes()?;

        let path = self.archive_path(session_id);
        std::fs::write(&path, bytes)
            .map_err(|e| AppError::internal(format!("Failed to write archive: {e}")))?;

        let meta = snapshot.meta.clone();
        self.index.write().insert(session_id.to_string(), meta.clone());
        info!(session_id = %session_id, path = %path.display(), "Session archived");
        Ok(meta)
    }

    /// Restore a session snapshot from its archive.
    pub fn restore(&self, session_id: &str) -> Result<SessionArchive, AppError> {
        let path = self.archive_path(session_id);
        let bytes = std::fs::read(&path)
            .map_err(|e| AppError::not_found(format!("Archive for '{session_id}': {e}")))?;
        let snapshot = SessionArchive::from_bytes(&bytes)?;
        debug!(session_id = %session_id, "Session restored from archive");
        Ok(snapshot)
    }

    /// Delete a session archive.
    pub fn delete(&self, session_id: &str) -> Result<bool, AppError> {
        self.index.write().remove(session_id);
        let path = self.archive_path(session_id);
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| AppError::internal(format!("Failed to delete archive: {e}")))?;
            info!(session_id = %session_id, "Session archive deleted");
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Return the archived metadata for a session, if present.
    pub fn meta(&self, session_id: &str) -> Option<ArchiveMeta> {
        self.index.read().get(session_id).cloned()
    }

    /// Return metadata for all archived sessions.
    pub fn list(&self) -> Vec<ArchiveMeta> {
        let mut metas: Vec<ArchiveMeta> = self.index.read().values().cloned().collect();
        metas.sort_by(|a, b| a.archived_at.cmp(&b.archived_at));
        metas
    }

    /// Return the number of archived sessions.
    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    /// Return `true` if no sessions are archived.
    pub fn is_empty(&self) -> bool {
        self.index.read().is_empty()
    }

    /// Load the on-disk archive index. Call this at startup if archives may
    /// exist from a previous run.
    pub fn scan_disk(&self) -> Result<usize, AppError> {
        let mut count = 0;
        let entries = std::fs::read_dir(&self.dir)
            .map_err(|e| AppError::internal(format!("Cannot scan archive dir: {e}")))?;
        let mut index = self.index.write();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        warn!(path = %path.display(), error = %e, "Skipping unreadable archive");
                        continue;
                    }
                };
                if let Ok(snapshot) = SessionArchive::from_bytes(&bytes) {
                    index.insert(stem.to_string(), snapshot.meta.clone());
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

/// Sanitize a session id for use as a filename.
fn sanitize_session_id(id: &str) -> String {
    let mut safe = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            safe.push(ch);
        } else {
            safe.push('_');
        }
    }
    safe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("opensquilla-archive-test-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_archive_and_restore_roundtrip() {
        let dir = temp_dir("roundtrip");
        let archiver = SessionArchiver::new(&dir).unwrap();
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "title": "Test",
        });
        let meta = archiver.archive("s1", payload).unwrap();
        assert_eq!(meta.message_count, 1);
        assert!(meta.size_bytes > 0);

        let restored = archiver.restore("s1").unwrap();
        assert_eq!(restored.meta.session_id, "s1");
        assert_eq!(restored.payload["title"], "Test");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_archive_overwrites() {
        let dir = temp_dir("overwrite");
        let archiver = SessionArchiver::new(&dir).unwrap();
        archiver
            .archive("s1", serde_json::json!({"v": 1}))
            .unwrap();
        archiver
            .archive("s1", serde_json::json!({"v": 2}))
            .unwrap();
        let restored = archiver.restore("s1").unwrap();
        assert_eq!(restored.payload["v"], 2);
        assert_eq!(archiver.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_archive() {
        let dir = temp_dir("delete");
        let archiver = SessionArchiver::new(&dir).unwrap();
        archiver
            .archive("s1", serde_json::json!({}))
            .unwrap();
        assert!(archiver.delete("s1").unwrap());
        assert!(!archiver.delete("s1").unwrap());
        assert!(archiver.restore("s1").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scan_disk_rebuilds_index() {
        let dir = temp_dir("scan");
        let archiver = SessionArchiver::new(&dir).unwrap();
        archiver
            .archive("s1", serde_json::json!({"messages": []}))
            .unwrap();
        // Drop the in-memory index by creating a fresh archiver over the same dir.
        let fresh = SessionArchiver::new(&dir).unwrap();
        assert_eq!(fresh.len(), 0);
        let count = fresh.scan_disk().unwrap();
        assert_eq!(count, 1);
        assert!(fresh.meta("s1").is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sanitize_session_id() {
        assert_eq!(sanitize_session_id("abc-123_def.json"), "abc-123_def.json");
        assert_eq!(sanitize_session_id("a/b\\c"), "a_b_c");
    }

    #[test]
    fn test_archive_path_uses_sanitized_id() {
        let dir = temp_dir("path");
        let archiver = SessionArchiver::new(&dir).unwrap();
        let path = archiver.archive_path("../evil");
        assert!(path.ends_with("_evil.json"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_restore_missing_archive() {
        let dir = temp_dir("missing");
        let archiver = SessionArchiver::new(&dir).unwrap();
        assert!(archiver.restore("nope").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
