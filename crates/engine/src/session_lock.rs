//! Per-session asynchronous locks.
//!
//! Mirrors the Python backend's `session_lock.py`: one `Mutex` per session key,
//! stored in a shared map. Locks are acquired and released by session key, and
//! unused locks are automatically cleaned up when the last reference (handle or
//! guard) is dropped.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{debug, trace};

/// A guard that holds a per-session lock.
///
/// Releasing is automatic: dropping the guard unlocks the mutex and prunes the
/// lock entry from the map if no other handle/guard references it.
#[derive(Debug)]
pub struct SessionLockGuard {
    /// The owned mutex guard, taken (dropped) first during cleanup.
    guard: Option<OwnedMutexGuard<()>>,
    /// The set this lock belongs to, used for cleanup bookkeeping.
    owner: Arc<LockSetInner>,
    /// The session key this guard protects.
    key: String,
}

impl Drop for SessionLockGuard {
    fn drop(&mut self) {
        // Drop the owned guard first so the `Arc<Mutex<()>>` it holds is
        // released; only then can cleanup see the entry as unreferenced.
        self.guard.take();
        let removed = self.owner.cleanup();
        trace!(session = %self.key, removed = removed, "Session lock released");
    }
}

/// A handle used to acquire a session lock.
///
/// Cloning a `SessionLockHandle` yields another handle to the *same* logical
/// per-session mutex (both share an `Arc<Mutex<()>>`).
#[derive(Debug, Clone)]
pub struct SessionLockHandle {
    /// The session key this lock protects.
    key: String,
    /// The shared mutex, taken (dropped) first during cleanup.
    inner: Option<Arc<Mutex<()>>>,
    /// The set this handle belongs to, used for cleanup bookkeeping.
    owner: Arc<LockSetInner>,
}

impl Drop for SessionLockHandle {
    fn drop(&mut self) {
        // Drop the shared Arc first so cleanup can see the entry as
        // unreferenced.
        self.inner.take();
        let removed = self.owner.cleanup();
        trace!(session = %self.key, removed = removed, "Session lock handle dropped");
    }
}

impl SessionLockHandle {
    /// Acquire the lock this handle points to.
    pub async fn lock_owned(self) -> SessionLockGuard {
        let inner = self
            .inner
            .clone()
            .expect("session lock handle already consumed");
        let guard = inner.lock_owned().await;
        debug!(session = %self.key, "Session lock acquired");
        SessionLockGuard {
            guard: Some(guard),
            owner: self.owner.clone(),
            key: self.key.clone(),
        }
    }

    /// Try to acquire the lock without waiting.
    pub fn try_lock_owned(self) -> Option<SessionLockGuard> {
        let inner = self
            .inner
            .clone()
            .expect("session lock handle already consumed");
        let guard = inner.clone().try_lock_owned().ok()?;
        debug!(session = %self.key, "Session lock acquired (try)");
        Some(SessionLockGuard {
            guard: Some(guard),
            owner: self.owner.clone(),
            key: self.key.clone(),
        })
    }

    /// The session key this lock protects.
    pub fn key(&self) -> &str {
        &self.key
    }
}

/// A set of per-session locks.
///
/// Locks are created lazily on first acquire and removed automatically when no
/// longer referenced by any guard or handle.
#[derive(Debug)]
pub struct SessionLockSet {
    inner: Arc<LockSetInner>,
}

#[derive(Debug, Default)]
struct LockSetInner {
    /// The per-session mutexes.
    locks: std::sync::Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// The maximum number of distinct lock entries retained before pruning.
    max_entries: AtomicUsize,
}

impl LockSetInner {
    /// Remove all tracked locks that have no outstanding references.
    ///
    /// Returns the number of entries removed.
    fn cleanup(&self) -> usize {
        let removed = {
            let mut locks = self.locks.lock().unwrap_or_else(|e| e.into_inner());
            // A lock entry is referenced by the map itself plus every live
            // handle/guard. Only entries with strong-count == 1 (map only)
            // are safe to drop.
            let before = locks.len();
            locks.retain(|_, lock| Arc::strong_count(lock) > 1);
            before - locks.len()
        };
        if removed > 0 {
            debug!(removed = removed, "Session locks cleaned up");
        }
        removed
    }
}

impl Default for SessionLockSet {
    fn default() -> Self {
        let inner = LockSetInner {
            locks: std::sync::Mutex::new(HashMap::new()),
            max_entries: AtomicUsize::new(4096),
        };
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl SessionLockSet {
    /// Create a new empty lock set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of distinct lock entries retained.
    ///
    /// When the map grows beyond this size, unused entries are pruned lazily
    /// on the next acquire.
    pub fn with_max_entries(self, max: usize) -> Self {
        self.inner.max_entries.store(max.max(1), Ordering::Relaxed);
        self
    }

    /// Acquire the lock for the given session key.
    ///
    /// Returns a guard that releases the lock (and triggers cleanup) when
    /// dropped. The caller typically holds the guard across the critical
    /// section:
    ///
    /// ```ignore
    /// let _guard = locks.acquire(session_key).await;
    /// // ... exclusive section ...
    /// ```
    pub async fn acquire(&self, key: &str) -> SessionLockGuard {
        let handle = self.handle(key);
        handle.lock_owned().await
    }

    /// Create a handle to the session lock without acquiring it.
    pub fn handle(&self, key: &str) -> SessionLockHandle {
        self.maybe_prune();

        let inner = {
            let mut locks = self.inner.locks.lock().unwrap_or_else(|e| e.into_inner());
            locks
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        trace!(session = %key, "Session lock handle created");
        SessionLockHandle {
            key: key.to_string(),
            inner: Some(inner),
            owner: self.inner.clone(),
        }
    }

    /// The number of distinct session locks currently tracked.
    pub fn tracked_count(&self) -> usize {
        self.inner
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// The number of outstanding references to session locks (live handles
    /// and held guards), derived from the strong reference counts.
    pub fn active_handles(&self) -> usize {
        self.inner
            .locks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|lock| Arc::strong_count(lock).saturating_sub(1))
            .sum()
    }

    /// Remove all tracked locks that have no outstanding references.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup(&self) -> usize {
        self.inner.cleanup()
    }

    /// Prune entries when the tracked count exceeds the configured maximum.
    fn maybe_prune(&self) {
        let max = self.inner.max_entries.load(Ordering::Relaxed);
        let len = self.tracked_count();
        if len <= max {
            return;
        }
        let removed = self.cleanup();
        debug!(
            tracked = len,
            max = max,
            removed = removed,
            "Session lock pruning"
        );
    }
}

/// Acquire a session lock for the duration of a closure.
///
/// Convenience wrapper: acquires the lock, runs `f`, then releases.
pub async fn with_session_lock<T, F, Fut>(locks: &SessionLockSet, key: &str, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let _guard = locks.acquire(key).await;
    f().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_acquire_release() {
        let locks = SessionLockSet::new();
        let guard = locks.acquire("session-1").await;
        assert_eq!(locks.tracked_count(), 1);
        assert_eq!(locks.active_handles(), 1);
        drop(guard);
        assert_eq!(locks.tracked_count(), 0);
        assert_eq!(locks.active_handles(), 0);
    }

    #[tokio::test]
    async fn test_serialized_access() {
        let locks = Arc::new(SessionLockSet::new());
        let key = "shared";

        let l1 = locks.clone();
        let t1 = tokio::spawn(async move {
            let _g = l1.acquire(key).await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        });

        let l2 = locks.clone();
        let t2 = tokio::spawn(async move {
            // This must wait until t1 releases the lock.
            let _g = l2.acquire(key).await;
        });

        let start = std::time::Instant::now();
        let _ = tokio::join!(t1, t2);
        assert!(start.elapsed() >= std::time::Duration::from_millis(40));
    }

    #[tokio::test]
    async fn test_distinct_sessions_parallel() {
        let locks = Arc::new(SessionLockSet::new());
        let mut tasks = Vec::new();
        for i in 0..10 {
            let locks = locks.clone();
            tasks.push(tokio::spawn(async move {
                let _g = locks.acquire(&format!("session-{i}")).await;
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }));
        }
        let start = std::time::Instant::now();
        for t in tasks {
            t.await.unwrap();
        }
        // All ran concurrently; total time should be well under the serial sum.
        assert!(start.elapsed() < std::time::Duration::from_millis(100));
    }

    #[test]
    fn test_handle_clone_shares_lock() {
        let locks = SessionLockSet::new();
        let h1 = locks.handle("k");
        let h2 = h1.clone();
        assert_eq!(locks.tracked_count(), 1);
        assert_eq!(locks.active_handles(), 2);
        drop(h1);
        assert_eq!(locks.tracked_count(), 1);
        drop(h2);
        assert_eq!(locks.tracked_count(), 0);
        assert_eq!(locks.active_handles(), 0);
    }

    #[tokio::test]
    async fn test_with_session_lock() {
        let locks = SessionLockSet::new();
        let result = with_session_lock(&locks, "k", || async { 42 }).await;
        assert_eq!(result, 42);
        assert_eq!(locks.tracked_count(), 0);
    }
}
