//! File-system synchronization into the memory store.
//!
//! The [`SyncManager`] provides unified synchronization triggers: a debounced
//! file-system watcher ([`FileWatcher`]) via the `notify` crate, a periodic
//! sync timer, and TTL-based expiry. Single files and whole workspaces can be
//! indexed on demand, and every supported file is recorded both as a memory
//! entry and in the chunk store for retrieval.

use chrono::Utc;
use dashmap::DashMap;
use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, new_debouncer};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use crate::store::MemoryStore;

type EventCallback = Arc<dyn Fn(Vec<PathBuf>) + Send + Sync>;
type Debouncer = notify_debouncer_mini::Debouncer<RecommendedWatcher>;

/// Configuration for the [`SyncManager`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Directories watched by the file-system watcher.
    pub watch_dirs: Vec<String>,
    /// Number of seconds between periodic sync ticks.
    pub interval_seconds: u64,
    /// TTL in seconds after which old, low-importance memories expire.
    pub ttl_seconds: u64,
    /// Minimum importance a memory needs to survive TTL expiry.
    pub min_importance: f64,
    /// File patterns to sync (e.g. `["*.md", "*.txt"]`). Empty means "all".
    pub file_patterns: Vec<String>,
    /// Debounce window for the file-system watcher, in milliseconds.
    pub debounce_ms: u64,
    /// Whether workspace sync recurses into subdirectories.
    pub recursive: bool,
    /// Approximate character size of each chunk produced when indexing a file.
    pub chunk_size: usize,
    /// Number of overlapping characters between consecutive chunks.
    pub chunk_overlap: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            watch_dirs: Vec::new(),
            interval_seconds: 300,
            ttl_seconds: 7 * 24 * 3600,
            min_importance: 0.3,
            file_patterns: vec![
                "*.md".to_string(),
                "*.txt".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            debounce_ms: 2_000,
            recursive: true,
            chunk_size: 1_500,
            chunk_overlap: 200,
        }
    }
}

/// Aggregate statistics from a workspace sync.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Number of files successfully synced.
    pub files_synced: u64,
    /// Number of files skipped (no matching pattern).
    pub files_skipped: u64,
    /// Paths that failed to sync.
    pub failed_files: Vec<String>,
    /// Total number of chunks produced across all indexed files.
    pub total_chunks: u64,
}

/// A debounced file-system watcher.
///
/// Wraps a `notify` watcher behind a debouncer, spawning a background thread
/// that drains debounced events and invokes the registered callback with the
/// set of changed paths. The watcher stays alive as long as this struct is
/// alive.
pub struct FileWatcher {
    /// Directories being watched recursively.
    pub paths: Vec<PathBuf>,
    /// Debounce window applied to raw events.
    pub debounce: Duration,
    callback: Option<EventCallback>,
    handle: Option<std::thread::JoinHandle<()>>,
    _debouncer: Option<Debouncer>,
}

impl std::fmt::Debug for FileWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileWatcher")
            .field("paths", &self.paths)
            .field("debounce", &self.debounce)
            .finish()
    }
}

impl FileWatcher {
    pub fn new(paths: Vec<PathBuf>, debounce: Duration) -> Self {
        Self {
            paths,
            debounce,
            callback: None,
            handle: None,
            _debouncer: None,
        }
    }

    /// Register the callback invoked with the set of changed paths.
    pub fn with_callback(mut self, callback: EventCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Start watching. Returns `self` so the caller can keep the watcher alive.
    pub fn start(mut self) -> CoreResult<Self> {
        let (tx, rx) = mpsc::channel::<DebounceEventResult>();
        let mut debouncer = new_debouncer(self.debounce, tx)
            .map_err(|e| CoreError::Internal(format!("Failed to create debouncer: {}", e)))?;

        for path in &self.paths {
            debouncer
                .watcher()
                .watch(path, RecursiveMode::Recursive)
                .map_err(|e| {
                    CoreError::Internal(format!("Failed to watch {}: {}", path.display(), e))
                })?;
        }

        let callback = self.callback.clone();
        let watched = self.paths.clone();
        self.handle = Some(std::thread::spawn(move || {
            for result in rx {
                match result {
                    Ok(events) => {
                        let changed: Vec<PathBuf> = events.iter().map(|e| e.path.clone()).collect();
                        if !changed.is_empty() {
                            info!("File watcher detected {} changes", changed.len());
                            if let Some(ref cb) = callback {
                                cb(changed);
                            }
                        }
                    }
                    Err(e) => {
                        error!("File watcher error: {}", e);
                    }
                }
            }
            debug!("File watcher for {:?} stopped", watched);
        }));
        self._debouncer = Some(debouncer);
        info!("File watcher started for {} paths", self.paths.len());
        Ok(self)
    }

    /// Stop the watcher thread.
    pub fn stop(&mut self) {
        // Drop the debouncer first so its senders go away and the receiver
        // loop observes a disconnect; joining before that would deadlock.
        self._debouncer = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Manages all synchronization triggers for the memory store.
pub struct SyncManager {
    store: MemoryStore,
    watch_paths: DashMap<PathBuf, bool>,
    callback: Option<EventCallback>,
    config: SyncConfig,
    _watcher: Option<FileWatcher>,
}

impl SyncManager {
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            watch_paths: DashMap::new(),
            callback: None,
            config: SyncConfig::default(),
            _watcher: None,
        }
    }

    pub fn with_config(mut self, config: SyncConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Register a callback for file change events.
    pub fn on_change<F>(&mut self, callback: F)
    where
        F: Fn(Vec<PathBuf>) + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
    }

    /// Watch a directory for file changes.
    pub fn watch_directory(&mut self, path: &str) -> CoreResult<()> {
        let path = PathBuf::from(path);
        if !path.exists() {
            std::fs::create_dir_all(&path)
                .map_err(|e| CoreError::Storage(format!("Cannot create watch dir: {}", e)))?;
        }

        self.watch_paths.insert(path.clone(), true);
        info!("Watching directory for memory sync: {}", path.display());
        Ok(())
    }

    /// Watch a set of workspace directories and start the watcher.
    pub fn watch_files(&mut self, workspace_dirs: &[String]) -> CoreResult<()> {
        for dir in workspace_dirs {
            self.watch_directory(dir)?;
        }
        self.start_watcher()
    }

    /// Start the file watcher.
    pub fn start_watcher(&mut self) -> CoreResult<()> {
        let paths: Vec<PathBuf> = self.watch_paths.iter().map(|e| e.key().clone()).collect();
        if paths.is_empty() {
            warn!("No paths to watch");
            return Ok(());
        }

        let debounce = Duration::from_millis(self.config.debounce_ms);
        let mut watcher = FileWatcher::new(paths, debounce);
        if let Some(ref cb) = self.callback {
            watcher = watcher.with_callback(cb.clone());
        }
        let watcher = watcher.start()?;
        self._watcher = Some(watcher);
        info!("File watcher started for {} paths", self.watch_paths.len());
        Ok(())
    }

    /// Stop the file watcher.
    pub fn stop_watcher(&mut self) {
        if let Some(mut watcher) = self._watcher.take() {
            watcher.stop();
        }
    }

    /// Sync a single file into the memory store.
    ///
    /// Skips files that do not match [`SyncConfig::file_patterns`] and files
    /// whose checksum is unchanged since the last index. The file is recorded
    /// both as a memory entry and in the chunk store.
    pub fn sync_file(&self, file_path: &str) -> CoreResult<()> {
        self.sync_file_internal(file_path).map(|_| ())
    }

    /// Like [`sync_file`][Self::sync_file], but returns the number of chunks
    /// produced for the file (0 when skipped).
    fn sync_file_internal(&self, file_path: &str) -> CoreResult<usize> {
        let path = PathBuf::from(file_path);
        if !path.is_file() {
            return Err(CoreError::InvalidInput(format!(
                "Not a file: {}",
                file_path
            )));
        }
        if !self.is_supported(&path) {
            debug!("Skipping file not matching sync patterns: {}", file_path);
            return Ok(0);
        }

        let bytes = std::fs::read(file_path).map_err(CoreError::Io)?;
        let content = String::from_utf8_lossy(&bytes).to_string();
        if content.trim().is_empty() {
            return Ok(0);
        }

        if self.store.file_is_unchanged(file_path, &bytes)? {
            debug!("File unchanged, skipping sync: {}", file_path);
            return Ok(0);
        }

        let file_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let metadata = serde_json::json!({
            "file_path": file_path,
            "file_name": file_name,
            "synced_at": Utc::now().to_rfc3339(),
            "size": bytes.len(),
        });

        let mut entry = crate::types::MemoryEntry::new(
            opensquilla_core::types::MemoryId(uuid::Uuid::new_v4()),
            uuid::Uuid::nil(),
            content.clone(),
            String::from("file_sync"),
            String::from("document"),
            0.5,
            metadata,
        );
        entry.tags = vec![format!("file:{}", file_name)];
        entry.metadata["file_name"] = serde_json::json!(file_name);

        self.store.insert_memory(&entry)?;

        // Also index into the chunk store for RAG retrieval.
        let chunk_count = self.index_file(file_path)?;

        info!("Synced file into memory: {}", file_path);
        Ok(chunk_count)
    }

    /// Index a single file into the chunk store, splitting its text into
    /// overlapping chunks. Returns the number of chunks created (0 when the
    /// file is unchanged since the last index).
    pub fn index_file(&self, file_path: &str) -> CoreResult<usize> {
        let bytes = std::fs::read(file_path).map_err(CoreError::Io)?;
        let content = String::from_utf8_lossy(&bytes).to_string();
        if self.store.file_is_unchanged(file_path, &bytes)? {
            return Ok(0);
        }
        let indexed = self.store.index_file(file_path, &bytes)?;

        let chunks = chunk_text(&content, self.config.chunk_size, self.config.chunk_overlap);
        for (i, chunk) in chunks.iter().enumerate() {
            let token_count = chunk.split_whitespace().count() as u32;
            let memory_chunk =
                crate::types::MemoryChunk::new(indexed.id, i as u32, chunk.clone(), token_count);
            self.store.insert_chunk(&memory_chunk, &indexed.id)?;
        }
        Ok(chunks.len())
    }

    /// Sync all files from a directory (non-recursive).
    pub fn sync_directory(&self, dir_path: &str) -> CoreResult<u64> {
        let mut count = 0u64;
        let dir = PathBuf::from(dir_path);

        if !dir.is_dir() {
            return Err(CoreError::InvalidInput(format!(
                "Not a directory: {}",
                dir_path
            )));
        }

        let entries = std::fs::read_dir(&dir)
            .map_err(|e| CoreError::Storage(format!("Cannot read directory: {}", e)))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && self.is_supported(&path) {
                match self.sync_file(&path.to_string_lossy()) {
                    Ok(()) => count += 1,
                    Err(e) => warn!("Failed to sync {}: {}", path.display(), e),
                }
            }
        }

        info!("Synced {} files from directory {}", count, dir_path);
        Ok(count)
    }

    /// Sync an entire workspace recursively.
    pub fn sync_workspace(&self, workspace_path: &str) -> CoreResult<SyncStats> {
        let mut stats = SyncStats::default();
        let root = PathBuf::from(workspace_path);

        if !root.is_dir() {
            return Err(CoreError::InvalidInput(format!(
                "Not a directory: {}",
                workspace_path
            )));
        }

        let mut stack: Vec<PathBuf> = vec![root];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .map_err(|e| CoreError::Storage(format!("Cannot read directory: {}", e)))?;
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if self.config.recursive {
                        stack.push(path);
                    }
                } else if self.is_supported(&path) {
                    match self.sync_file_internal(&path.to_string_lossy()) {
                        Ok(chunk_count) => {
                            stats.files_synced += 1;
                            stats.total_chunks += chunk_count as u64;
                        }
                        Err(e) => {
                            warn!("Failed to sync {}: {}", path.display(), e);
                            stats.failed_files.push(path.to_string_lossy().to_string());
                        }
                    }
                } else {
                    stats.files_skipped += 1;
                }
            }
        }

        info!(
            "Workspace sync complete: {} synced, {} skipped, {} failed",
            stats.files_synced,
            stats.files_skipped,
            stats.failed_files.len()
        );
        Ok(stats)
    }

    /// Schedule periodic workspace syncs on a tokio interval.
    ///
    /// The returned handle can be used to cancel the task. Each tick syncs all
    /// configured watch directories and expires old memories.
    pub fn schedule_periodic_sync(
        &self,
        interval: Duration,
    ) -> CoreResult<tokio::task::JoinHandle<()>> {
        let store = self.store.clone();
        let config = self.config.clone();
        let workspace_dirs: Vec<String> = config.watch_dirs.clone();
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                for dir in &workspace_dirs {
                    let manager = SyncManager::new(store.clone());
                    if let Err(e) = manager.sync_workspace(dir) {
                        error!("Periodic sync failed for {}: {}", dir, e);
                    }
                }
                if let Err(e) = store.expire_old_memories(
                    chrono::Duration::seconds(config.ttl_seconds as i64),
                    config.min_importance,
                ) {
                    error!("Periodic TTL expiry failed: {}", e);
                }
            }
        });
        Ok(handle)
    }

    /// Expire old, low-importance memories based on the configured TTL.
    pub fn handle_ttl_expiry(&self) -> CoreResult<u64> {
        let ttl = chrono::Duration::seconds(self.config.ttl_seconds as i64);
        let deleted = self
            .store
            .expire_old_memories(ttl, self.config.min_importance)?;
        info!("TTL expiry removed {} memories", deleted);
        Ok(deleted)
    }

    /// Whether a path matches the configured file patterns.
    fn is_supported(&self, path: &Path) -> bool {
        if self.config.file_patterns.is_empty() {
            return true;
        }
        self.config
            .file_patterns
            .iter()
            .any(|pat| matches_pattern(path, pat))
    }
}

/// Match a path against a pattern.
///
/// Supports extension patterns (`*.md`), wildcards (`*` and `?`), and exact
/// file-name matches.
fn matches_pattern(path: &Path, pattern: &str) -> bool {
    let pattern = pattern.trim();
    if pattern.is_empty() {
        return false;
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return path
            .extension()
            .map(|e| e.to_string_lossy().eq_ignore_ascii_case(ext))
            .unwrap_or(false);
    }
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    wildcard_match(pattern, file_name)
}

/// Classic wildcard matcher: `*` matches any sequence, `?` matches one char.
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (plen, tlen) = (p.len(), t.len());
    let mut dp = vec![vec![false; tlen + 1]; plen + 1];
    dp[0][0] = true;
    for i in 1..=plen {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=plen {
        for j in 1..=tlen {
            match p[i - 1] {
                '*' => dp[i][j] = dp[i - 1][j] || dp[i][j - 1],
                '?' => dp[i][j] = dp[i - 1][j - 1],
                c => dp[i][j] = dp[i - 1][j - 1] && c == t[j - 1],
            }
        }
    }
    dp[plen][tlen]
}

/// Split text into overlapping character chunks.
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        if end >= chars.len() {
            break;
        }
        start += step;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("osq_memory_sync_{}_{}", name, uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn test_wildcard_match() {
        assert!(wildcard_match("*.rs", "lib.rs"));
        assert!(!wildcard_match("*.rs", "lib.txt"));
        assert!(wildcard_match("a?c", "abc"));
        assert!(!wildcard_match("a?c", "ac"));
        assert!(wildcard_match("README*", "README.md"));
        assert!(wildcard_match("README*", "README"));
        assert!(!wildcard_match("README*", "notes.md"));
    }

    #[test]
    fn test_matches_pattern() {
        assert!(matches_pattern(Path::new("a.md"), "*.md"));
        assert!(!matches_pattern(Path::new("a.rs"), "*.md"));
        assert!(matches_pattern(Path::new("lib.rs"), "*.RS"));
        assert!(matches_pattern(Path::new("settings.json"), "settings.json"));
    }

    #[test]
    fn test_chunk_text() {
        let long: String = "a".repeat(100);
        let chunks = chunk_text(&long, 30, 5);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| !c.is_empty()));

        let single = chunk_text("short", 30, 5);
        assert_eq!(single.len(), 1);

        assert!(chunk_text("", 30, 5).is_empty());
    }

    #[test]
    fn test_sync_file_creates_memory() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store.clone());
        let dir = temp_dir("single");
        let path = dir.join("notes.md");
        write(&path, "# Hello memory");
        manager.sync_file(path.to_str().unwrap()).unwrap();
        let all = store
            .list_memories(&uuid::Uuid::nil(), None, 100, 0)
            .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].source, "file_sync");
    }

    #[test]
    fn test_sync_file_skips_unchanged() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store.clone());
        let dir = temp_dir("dedup");
        let path = dir.join("doc.md");
        write(&path, "same content");
        manager.sync_file(path.to_str().unwrap()).unwrap();
        manager.sync_file(path.to_str().unwrap()).unwrap();
        let all = store
            .list_memories(&uuid::Uuid::nil(), None, 100, 0)
            .unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_sync_file_ignores_unmatched_pattern() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store.clone());
        let dir = temp_dir("unmatched");
        let path = dir.join("code.rs");
        write(&path, "fn main() {}");
        manager.sync_file(path.to_str().unwrap()).unwrap();
        let all = store
            .list_memories(&uuid::Uuid::nil(), None, 100, 0)
            .unwrap();
        assert!(all.is_empty());
    }

    #[test]
    fn test_sync_directory_counts_supported() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store.clone());
        let dir = temp_dir("dir");
        write(&dir.join("a.md"), "alpha");
        write(&dir.join("b.txt"), "beta");
        write(&dir.join("c.rs"), "fn main() {}");
        let count = manager.sync_directory(dir.to_str().unwrap()).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_sync_workspace_recursive() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store.clone());
        let dir = temp_dir("ws");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        write(&dir.join("root.md"), "root");
        write(&dir.join("sub").join("nested.md"), "nested");
        let stats = manager.sync_workspace(dir.to_str().unwrap()).unwrap();
        assert_eq!(stats.files_synced, 2);
        assert_eq!(stats.failed_files.len(), 0);
        assert!(stats.total_chunks >= 1);
    }

    #[test]
    fn test_handle_ttl_expiry() {
        let store = MemoryStore::in_memory().unwrap();
        let mut config = SyncConfig::default();
        config.ttl_seconds = 0;
        config.min_importance = 1.0;
        let manager = SyncManager::new(store.clone()).with_config(config);
        let agent = uuid::Uuid::new_v4();
        let entry = crate::types::MemoryEntry::new(
            opensquilla_core::types::MemoryId(uuid::Uuid::new_v4()),
            agent,
            "old memory".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        store.insert_memory(&entry).unwrap();
        let deleted = manager.handle_ttl_expiry().unwrap();
        assert!(deleted >= 1);
    }

    #[tokio::test]
    async fn test_schedule_periodic_sync_returns_handle() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = SyncManager::new(store);
        let handle = manager
            .schedule_periodic_sync(Duration::from_secs(3600))
            .unwrap();
        handle.abort();
    }
}
