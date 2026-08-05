//! Denial-aware stale-output cache.
//!
//! When the approval gate denies an action, we must prevent the agent from
//! reusing the cached output of a *previous* successful run of the same
//! dangerous action. This module implements the concrete hygiene rule: the
//! last successful sandboxed output, keyed by a fingerprint derived from the
//! command + input hash, is cached to disk with a TTL so a denied command's
//! stale output cannot influence the next turn.
//!
//! The cache is file-based: entries live under `<root_dir>/<key>.bin` and an
//! in-memory index (guarded by a `tokio::sync::RwLock`) tracks their
//! metadata. The index is persisted to `<root_dir>/index.json` so it survives
//! process restarts. Purging removes an entry unconditionally — we prefer a
//! false negative (losing a legitimate cached result) over a false positive
//! (letting a denied command's stale output leak into a later run).
//!
//! Two keying styles are supported:
//! - command/input keys via [`StaleOutputCache::cache_key`] / [`get`], and
//! - session-scoped keys via [`record_success`] / [`get_session`] / [`purge`]
//!   / [`clear_session`], matching the no-op variant's surface.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Metadata for a single cached entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub command: String,
    pub input_hash: String,
    pub stored_at: DateTime<Utc>,
    pub ttl_secs: u64,
    /// Session the entry belongs to, when stored session-scoped.
    pub session_id: Option<String>,
    /// Fingerprint used to derive a session-scoped key.
    pub fingerprint: Option<String>,
}

/// File-based cache of the last successful sandboxed output per cache key.
///
/// Keys are derived from `(command, input_hash)` via SHA-256 so identical
/// sandboxed operations reuse the cached output instead of re-executing.
#[derive(Debug)]
pub struct StaleOutputCache {
    root_dir: PathBuf,
    default_ttl_secs: u64,
    index: Arc<RwLock<HashMap<String, CacheEntry>>>,
}

impl StaleOutputCache {
    /// Create a cache rooted at `root_dir` with the default TTL of 1 hour.
    ///
    /// Any persisted index at `<root_dir>/index.json` is loaded synchronously.
    pub fn new(root_dir: PathBuf) -> Self {
        Self::with_ttl(root_dir, 3600)
    }

    /// Create a cache rooted at `root_dir` with a custom default TTL.
    pub fn with_ttl(root_dir: PathBuf, default_ttl_secs: u64) -> Self {
        let cache = Self {
            root_dir,
            default_ttl_secs,
            index: Arc::new(RwLock::new(HashMap::new())),
        };
        let _ = cache.load_index_sync();
        cache
    }

    /// Derive a stable cache key from a command and its input bytes.
    pub fn cache_key(command: &str, input: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(command.as_bytes());
        hasher.update([0u8]);
        hasher.update(input);
        hex::encode(hasher.finalize())
    }

    /// Derive a stable cache key from a command and a precomputed input hash.
    pub fn key_from_hash(command: &str, input_hash: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(command.as_bytes());
        hasher.update([0u8]);
        hasher.update(input_hash.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Derive a stable cache key from a session id and an operation
    /// fingerprint.
    pub fn session_key(session_id: &str, fingerprint: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"session");
        hasher.update([0u8]);
        hasher.update(session_id.as_bytes());
        hasher.update([0u8]);
        hasher.update(fingerprint.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// The on-disk path for a cache key.
    pub fn entry_path(&self, key: &str) -> PathBuf {
        self.root_dir.join(format!("{}.bin", key))
    }

    /// The on-disk path for the persisted index.
    pub fn index_path(&self) -> PathBuf {
        self.root_dir.join("index.json")
    }

    /// Fetch a cached output, honoring the entry TTL.
    ///
    /// Returns `None` on a miss or when the entry has expired (the expired
    /// file is removed eagerly). If the in-memory index does not know the key
    /// (e.g. after a cold start) the file is read directly and its
    /// modified-time is used to rebuild the index entry.
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        if let Some(meta) = self.index.read().await.get(key).cloned() {
            if self.is_expired(&meta) {
                self.invalidate(key).await;
                return None;
            }
            return match tokio::fs::read(self.entry_path(key)).await {
                Ok(bytes) => Some(bytes),
                Err(e) => {
                    warn!("stale_output_cache: read failed for '{}': {}", key, e);
                    None
                }
            };
        }

        // Fallback: file exists but the index is cold.
        let path = self.entry_path(key);
        let data = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(_) => return None,
        };
        if let Ok(metadata) = tokio::fs::metadata(&path).await {
            if let Ok(modified) = metadata.modified() {
                let stored_at: DateTime<Utc> = modified.into();
                if Utc::now() > stored_at + chrono::Duration::seconds(self.default_ttl_secs as i64)
                {
                    let _ = tokio::fs::remove_file(&path).await;
                    return None;
                }
                self.index.write().await.insert(
                    key.to_string(),
                    CacheEntry {
                        key: key.to_string(),
                        command: String::new(),
                        input_hash: String::new(),
                        stored_at,
                        ttl_secs: self.default_ttl_secs,
                        session_id: None,
                        fingerprint: None,
                    },
                );
            }
        }
        Some(data)
    }

    /// Store an output under the given key with the default TTL.
    pub async fn set(
        &self,
        key: &str,
        command: &str,
        input_hash: &str,
        output: Vec<u8>,
    ) -> Result<(), String> {
        self.set_with_meta(key, command, input_hash, None, None, output)
            .await
    }

    /// Store an output under the given key with optional session metadata.
    async fn set_with_meta(
        &self,
        key: &str,
        command: &str,
        input_hash: &str,
        session_id: Option<String>,
        fingerprint: Option<String>,
        output: Vec<u8>,
    ) -> Result<(), String> {
        let meta = CacheEntry {
            key: key.to_string(),
            command: command.to_string(),
            input_hash: input_hash.to_string(),
            stored_at: Utc::now(),
            ttl_secs: self.default_ttl_secs,
            session_id,
            fingerprint,
        };
        let path = self.entry_path(key);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("stale_output_cache: mkdir failed: {}", e))?;
        }
        tokio::fs::write(&path, &output)
            .await
            .map_err(|e| format!("stale_output_cache: write failed: {}", e))?;
        self.index.write().await.insert(key.to_string(), meta);
        self.persist_index().await;
        debug!(
            "stale_output_cache: cached '{}' ({} bytes)",
            key,
            output.len()
        );
        Ok(())
    }

    /// Remove an entry by key. Returns `true` if an entry was removed.
    pub async fn invalidate(&self, key: &str) -> bool {
        let removed = self.index.write().await.remove(key).is_some();
        let path = self.entry_path(key);
        if tokio::fs::try_exists(&path).await.unwrap_or(false) {
            let _ = tokio::fs::remove_file(&path).await;
        }
        if removed {
            self.persist_index().await;
        }
        removed
    }

    /// Remove every expired entry. Returns the number of entries removed.
    pub async fn clear_expired(&self) -> usize {
        let expired: Vec<String> = {
            let index = self.index.read().await;
            index
                .iter()
                .filter(|(_, meta)| self.is_expired(meta))
                .map(|(key, _)| key.clone())
                .collect()
        };
        let mut count = 0usize;
        for key in expired {
            if self.invalidate(&key).await {
                count += 1;
            }
        }
        if count > 0 {
            debug!("stale_output_cache: cleared {} expired entries", count);
        }
        count
    }

    /// Return the cached output for `(command, input)` or execute `f` and
    /// cache its result.
    pub async fn get_or_execute<F, Fut>(
        &self,
        command: &str,
        input: &[u8],
        f: F,
    ) -> Result<Vec<u8>, String>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>, String>> + Send,
    {
        let key = Self::cache_key(command, input);
        if let Some(cached) = self.get(&key).await {
            debug!("stale_output_cache: hit for '{}'", command);
            return Ok(cached);
        }
        let output = f().await?;
        let input_hash = hex::encode(Sha256::digest(input));
        self.set(&key, command, &input_hash, output.clone()).await?;
        Ok(output)
    }

    /// Record a successful sandboxed execution under a session fingerprint.
    pub async fn record_success(&self, session_id: &str, fingerprint: &str, payload: Vec<u8>) {
        let key = Self::session_key(session_id, fingerprint);
        if let Err(e) = self
            .set_with_meta(
                &key,
                fingerprint,
                "",
                Some(session_id.to_string()),
                Some(fingerprint.to_string()),
                payload,
            )
            .await
        {
            warn!("stale_output_cache: record_success failed: {e}");
        }
    }

    /// Fetch a session-scoped cached output.
    pub async fn get_session(&self, session_id: &str, fingerprint: &str) -> Option<Vec<u8>> {
        let key = Self::session_key(session_id, fingerprint);
        self.get(&key).await
    }

    /// Purge a session-scoped cached output. Returns `true` if it existed.
    pub async fn purge(&self, session_id: &str, fingerprint: &str) -> bool {
        let key = Self::session_key(session_id, fingerprint);
        self.invalidate(&key).await
    }

    /// Remove every cached entry belonging to a session. Returns the number
    /// removed.
    pub async fn clear_session(&self, session_id: &str) -> usize {
        let keys: Vec<String> = {
            let index = self.index.read().await;
            index
                .iter()
                .filter(|(_, meta)| meta.session_id.as_deref() == Some(session_id))
                .map(|(key, _)| key.clone())
                .collect()
        };
        let mut count = 0usize;
        for key in keys {
            if self.invalidate(&key).await {
                count += 1;
            }
        }
        count
    }

    /// Persist the in-memory index to `<root_dir>/index.json`.
    pub async fn persist_index(&self) {
        if let Err(e) = self.persist_index_impl().await {
            warn!("stale_output_cache: index persist failed: {e}");
        }
    }

    async fn persist_index_impl(&self) -> Result<(), String> {
        let index = self.index.read().await;
        let json =
            serde_json::to_vec(&*index).map_err(|e| format!("index serialization failed: {e}"))?;
        if let Some(parent) = self.index_path().parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("index mkdir failed: {e}"))?;
        }
        let tmp = self.root_dir.join("index.json.tmp");
        tokio::fs::write(&tmp, &json)
            .await
            .map_err(|e| format!("index write failed: {e}"))?;
        tokio::fs::rename(&tmp, self.index_path())
            .await
            .map_err(|e| format!("index rename failed: {e}"))?;
        Ok(())
    }

    /// Synchronously load a persisted index (called from the constructor).
    fn load_index_sync(&self) -> Result<(), String> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(());
        }
        let content =
            std::fs::read_to_string(&path).map_err(|e| format!("index read failed: {e}"))?;
        let entries: HashMap<String, CacheEntry> =
            serde_json::from_str(&content).map_err(|e| format!("index parse failed: {e}"))?;
        // Only keep entries whose files still exist; drop the rest. Uses
        // try_write so a cache constructed inside a running runtime never
        // blocks the current thread.
        let mut index = match self.index.try_write() {
            Ok(guard) => guard,
            Err(_) => return Ok(()),
        };
        for (key, meta) in entries {
            if self.entry_path(&key).exists() {
                index.insert(key, meta);
            }
        }
        Ok(())
    }

    /// A plain-data view of cached keys (debug/test helper). The payload is
    /// intentionally omitted so the snapshot is safe to log.
    pub async fn snapshot(&self) -> Vec<serde_json::Value> {
        let index = self.index.read().await;
        index
            .values()
            .map(|meta| {
                serde_json::json!({
                    "key": meta.key,
                    "command": meta.command,
                    "input_hash": meta.input_hash,
                    "stored_at": meta.stored_at.to_rfc3339(),
                    "ttl_secs": meta.ttl_secs,
                    "session_id": meta.session_id,
                    "fingerprint": meta.fingerprint,
                })
            })
            .collect()
    }

    /// Number of entries currently tracked (in-memory).
    pub async fn len(&self) -> usize {
        self.index.read().await.len()
    }

    /// Is the cache empty?
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Spawn a background task that periodically removes expired entries and
    /// persists the index. Returns the task handle.
    pub fn spawn_sweeper(self: Arc<Self>, interval_secs: u64) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                ticker.tick().await;
                let removed = self.clear_expired().await;
                if removed > 0 {
                    debug!("stale_output_cache: sweeper removed {removed} entry(ies)");
                }
            }
        })
    }

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        Utc::now() > entry.stored_at + chrono::Duration::seconds(entry.ttl_secs as i64)
    }
}

impl Default for StaleOutputCache {
    fn default() -> Self {
        Self::new(PathBuf::from("stale_output_cache"))
    }
}

/// No-op variant for call sites that don't need the denial-hygiene pathway.
#[derive(Debug, Default, Clone)]
pub struct NullStaleOutputCache;

impl NullStaleOutputCache {
    pub async fn record_success(&self, _session_id: &str, _fingerprint: &str, _payload: Vec<u8>) {}
    pub async fn purge(&self, _session_id: &str, _fingerprint: &str) -> bool {
        false
    }
    pub async fn get(&self, _session_id: &str, _fingerprint: &str) -> Option<Vec<u8>> {
        None
    }
    pub async fn clear_session(&self, _session_id: &str) -> usize {
        0
    }
}
