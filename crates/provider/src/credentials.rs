//! Credential pool: API key rotation, 429 cooldown tracking, round-robin selection.
//!
//! Many providers allow multiple API keys to be associated with one account for
//! higher aggregate rate limits. [`CredentialPool`] rotates through a set of
//! keys round-robin and cools down keys that receive HTTP 429 responses so
//! subsequent requests avoid them until the cooldown window elapses.
//!
//! The pool is thread-safe via [`DashMap`](dashmap::DashMap) and an atomic
//! round-robin counter.

use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Default cooldown applied to a key that receives a 429 when no `Retry-After`
/// header is present.
pub const DEFAULT_429_COOLDOWN: Duration = Duration::from_secs(30);

/// Internal per-key state.
#[derive(Debug)]
struct KeyState {
    /// The API key value.
    key: String,
    /// When the key's current cooldown expires, if any.
    cooldown_until: Option<Instant>,
    /// Number of times this key has been selected.
    selections: u64,
    /// Number of times this key has been rate-limited (429).
    rate_limited_count: u64,
}

impl KeyState {
    fn new(key: String) -> Self {
        Self {
            key,
            cooldown_until: None,
            selections: 0,
            rate_limited_count: 0,
        }
    }

    fn is_available(&self, now: Instant) -> bool {
        match self.cooldown_until {
            None => true,
            Some(until) => now >= until,
        }
    }
}

/// A pool of API keys for a single provider, with round-robin rotation and
/// 429 cooldown tracking.
#[derive(Clone)]
pub struct CredentialPool {
    keys: Arc<DashMap<usize, KeyState>>,
    /// Round-robin cursor.
    cursor: Arc<AtomicUsize>,
    /// Default cooldown duration.
    default_cooldown: Duration,
}

impl CredentialPool {
    /// Create a new credential pool from a list of API keys.
    ///
    /// Keys are deduplicated (case-sensitive) and empty strings are dropped.
    /// Returns `Err` if no usable keys remain.
    pub fn new<I, S>(keys: I) -> Result<Self, CredentialPoolError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::with_cooldown(keys, DEFAULT_429_COOLDOWN)
    }

    /// Create a new credential pool with a custom default cooldown duration.
    pub fn with_cooldown<I, S>(
        keys: I,
        default_cooldown: Duration,
    ) -> Result<Self, CredentialPoolError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen = std::collections::HashSet::new();
        let map: DashMap<usize, KeyState> = DashMap::new();
        let mut idx = 0usize;
        for k in keys {
            let k = k.into();
            if k.trim().is_empty() {
                continue;
            }
            if !seen.insert(k.clone()) {
                continue;
            }
            map.insert(idx, KeyState::new(k));
            idx += 1;
        }
        if map.is_empty() {
            return Err(CredentialPoolError::NoKeys);
        }
        Ok(Self {
            keys: Arc::new(map),
            cursor: Arc::new(AtomicUsize::new(0)),
            default_cooldown,
        })
    }

    /// Create a single-key pool (the common case).
    pub fn single(key: impl Into<String>) -> Result<Self, CredentialPoolError> {
        Self::new([key])
    }

    /// Returns the number of keys in the pool (including cooled-down ones).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the pool has no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Select the next available API key, skipping keys that are in cooldown.
    ///
    /// If all keys are in cooldown, returns the one that will become available
    /// soonest (so the caller can decide to wait or fail fast).
    pub fn select(&self) -> Option<SelectedKey> {
        if self.keys.is_empty() {
            return None;
        }
        let now = Instant::now();
        let n = self.keys.len();

        // Try each key once in round-robin order starting from the cursor.
        let start = self.cursor.load(Ordering::Relaxed) % n;
        let mut best_cooled: Option<(usize, Instant)> = None;

        for offset in 0..n {
            let idx = (start + offset) % n;
            if let Some(mut entry) = self.keys.get_mut(&idx) {
                let state = entry.value_mut();
                if state.is_available(now) {
                    state.selections += 1;
                    state.cooldown_until = None;
                    // Advance the cursor past the one we just picked.
                    self.cursor.store((idx + 1) % n, Ordering::Relaxed);
                    debug!(target = "provider", key_index = idx, "Selected credential");
                    return Some(SelectedKey {
                        index: idx,
                        key: state.key.clone(),
                        available_in: None,
                    });
                } else if let Some(until) = state.cooldown_until {
                    match best_cooled {
                        None => best_cooled = Some((idx, until)),
                        Some((_, best_until)) if until < best_until => {
                            best_cooled = Some((idx, until));
                        }
                        _ => {}
                    }
                }
            }
        }

        // All keys are in cooldown. Return the soonest-available one with a flag.
        if let Some((idx, until)) = best_cooled {
            let wait = until.saturating_duration_since(now);
            if let Some(mut entry) = self.keys.get_mut(&idx) {
                let state = entry.value_mut();
                return Some(SelectedKey {
                    index: idx,
                    key: state.key.clone(),
                    available_in: Some(wait),
                });
            }
        }
        None
    }

    /// Mark the key at `index` as rate-limited (HTTP 429).
    ///
    /// If `retry_after` is provided (from a `Retry-After` header), it is used
    /// as the cooldown duration; otherwise the pool's default cooldown applies.
    pub fn mark_rate_limited(&self, index: usize, retry_after: Option<Duration>) {
        if let Some(mut entry) = self.keys.get_mut(&index) {
            let state = entry.value_mut();
            let cooldown = retry_after.unwrap_or(self.default_cooldown);
            state.cooldown_until = Some(Instant::now() + cooldown);
            state.rate_limited_count += 1;
            warn!(
                target = "provider",
                key_index = index,
                cooldown_secs = cooldown.as_secs(),
                "Credential marked rate-limited (429)"
            );
        }
    }

    /// Mark the key at `index` as having succeeded, clearing any cooldown.
    pub fn mark_success(&self, index: usize) {
        if let Some(mut entry) = self.keys.get_mut(&index) {
            let state = entry.value_mut();
            state.cooldown_until = None;
        }
    }

    /// Mark the key at `index` as invalid (e.g. HTTP 401), cooling it down for
    /// a long time so it is avoided.
    pub fn mark_invalid(&self, index: usize) {
        if let Some(mut entry) = self.keys.get_mut(&index) {
            let state = entry.value_mut();
            // Cool down for a long time (1 hour) to effectively disable the key.
            state.cooldown_until = Some(Instant::now() + Duration::from_secs(3600));
            warn!(target = "provider", key_index = index, "Credential marked invalid (401)");
        }
    }

    /// Return the number of keys currently available (not in cooldown).
    pub fn available_count(&self) -> usize {
        let now = Instant::now();
        self.keys
            .iter()
            .filter(|entry| entry.value().is_available(now))
            .count()
    }

    /// Return a snapshot of per-key statistics for diagnostics.
    pub fn stats(&self) -> Vec<KeyStats> {
        let mut out = Vec::with_capacity(self.keys.len());
        for entry in self.keys.iter() {
            let state = entry.value();
            out.push(KeyStats {
                index: *entry.key(),
                selections: state.selections,
                rate_limited_count: state.rate_limited_count,
                cooldown_remaining: state
                    .cooldown_until
                    .map(|until| until.saturating_duration_since(Instant::now())),
            });
        }
        out.sort_by_key(|s| s.index);
        out
    }
}

/// A key selected from the pool.
#[derive(Debug, Clone)]
pub struct SelectedKey {
    /// The index of the key within the pool.
    pub index: usize,
    /// The API key value.
    pub key: String,
    /// If the key is currently in cooldown, how long until it is available.
    pub available_in: Option<Duration>,
}

impl SelectedKey {
    /// Returns `true` if this key is immediately usable.
    pub fn is_available(&self) -> bool {
        self.available_in.is_none()
    }
}

/// Per-key statistics snapshot.
#[derive(Debug, Clone)]
pub struct KeyStats {
    /// The key index.
    pub index: usize,
    /// Number of times this key was selected.
    pub selections: u64,
    /// Number of times this key received a 429.
    pub rate_limited_count: u64,
    /// Remaining cooldown duration, if any.
    pub cooldown_remaining: Option<Duration>,
}

/// Errors returned by [`CredentialPool`].
#[derive(Debug, thiserror::Error)]
pub enum CredentialPoolError {
    /// No usable keys were provided.
    #[error("no API keys provided")]
    NoKeys,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_key() {
        let pool = CredentialPool::single("sk-test").unwrap();
        assert_eq!(pool.len(), 1);
        let sel = pool.select().unwrap();
        assert_eq!(sel.key, "sk-test");
        assert!(sel.is_available());
    }

    #[test]
    fn test_dedup_and_empty() {
        let pool = CredentialPool::new(["a", "a", "", "b"]).unwrap();
        assert_eq!(pool.len(), 2);
        assert!(CredentialPool::new::<[&str; 0], &str>([]).is_err());
    }

    #[test]
    fn test_round_robin() {
        let pool = CredentialPool::new(["k1", "k2", "k3"]).unwrap();
        let a = pool.select().unwrap().key;
        let b = pool.select().unwrap().key;
        let c = pool.select().unwrap().key;
        let d = pool.select().unwrap().key;
        assert_eq!(a, "k1");
        assert_eq!(b, "k2");
        assert_eq!(c, "k3");
        assert_eq!(d, "k1");
    }

    #[test]
    fn test_cooldown_skips_key() {
        let pool = CredentialPool::new(["k1", "k2"]).unwrap();
        let first = pool.select().unwrap();
        pool.mark_rate_limited(first.index, Some(Duration::from_secs(60)));
        // k1 is now in cooldown; next selection must be k2.
        let second = pool.select().unwrap();
        assert_eq!(second.key, "k2");
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_all_cooldown_returns_soonest() {
        let pool = CredentialPool::new(["k1", "k2"]).unwrap();
        pool.mark_rate_limited(0, Some(Duration::from_secs(60)));
        pool.mark_rate_limited(1, Some(Duration::from_secs(10)));
        let sel = pool.select().unwrap();
        assert!(sel.available_in.is_some());
        // Should return the soonest-available key (k2).
        assert_eq!(sel.index, 1);
    }

    #[test]
    fn test_mark_success_clears_cooldown() {
        let pool = CredentialPool::single("k1").unwrap();
        pool.mark_rate_limited(0, Some(Duration::from_secs(60)));
        assert_eq!(pool.available_count(), 0);
        pool.mark_success(0);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_stats() {
        let pool = CredentialPool::new(["k1", "k2"]).unwrap();
        let _ = pool.select();
        let _ = pool.select();
        pool.mark_rate_limited(0, Some(Duration::from_secs(30)));
        let stats = pool.stats();
        assert_eq!(stats.len(), 2);
        assert!(stats[0].rate_limited_count >= 1);
        assert!(stats[0].cooldown_remaining.is_some());
        assert!(stats[1].selections >= 1);
    }
}
