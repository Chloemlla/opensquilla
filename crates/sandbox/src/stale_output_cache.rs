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
//! metadata. Purging removes an entry unconditionally — we prefer a false
//! negative (losing a legitimate cached result) over a false positive
//! (letting a denied command's stale output leak into a later run).

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Metadata for a single cached entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub command: String,
    pub input_hash: String,
    pub stored_at: DateTime<Utc>,
    pub ttl_secs: u64,
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
    pub fn new(root_dir: PathBuf) -> Self {
        Self::with_ttl(root_dir, 3600)
    }

    /// Create a cache rooted at `root_dir` with a custom default TTL.
    pub fn with_ttl(root_dir: PathBuf, default_ttl_secs: u64) -> Self {
        Self {
            root_dir,
            default_ttl_secs,
            index: Arc::new(RwLock::new(HashMap::new())),
        }
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

    /// The on-disk path for a cache key.
    pub fn entry_path(&self, key: &str) -> PathBuf {
        self.root_dir.join(format!("{}.bin", key))
    }

    /// Fetch a cached output, honoring the entry TTL.
    ///
    /// Returns `None` on a miss or when the entry has expired (the expired
    /// file is removed eagerly). If the in-memory index does not know the key
    /// (e.g. after a process restart) the file is read directly and its
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
        let meta = CacheEntry {
            key: key.to_string(),
            command: command.to_string(),
            input_hash: input_hash.to_string(),
            stored_at: Utc::now(),
            ttl_secs: self.default_ttl_secs,
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
                })
            })
            .collect()
    }

    /// Number of entries currently tracked (in-memory).
    pub async fn len(&self) -> usize {
        self.index.read().await.len()
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
