//! Process PID lock.
//!
//! Mirrors the Python `pidlock.py` module. Writes the current process id to a
//! lock file on startup, refuses to start if another live gateway holds the
//! lock, detects and clears stale locks left behind by crashed processes, and
//! removes the lock file on clean shutdown.
//!
//! The lock file is created atomically with `create_new`, so concurrent
//! starts race safely: exactly one wins.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// How old a lock file must be before it is considered stale even if the
/// process check is inconclusive (used on platforms where process liveness
/// cannot be probed cheaply).
pub const DEFAULT_STALE_AFTER_SECS: u64 = 24 * 60 * 60;

/// Contents of the lock file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockRecord {
    /// Process id that holds the lock.
    pub pid: u32,
    /// When the lock was created.
    pub created_at_unix_secs: u64,
    /// Optional hostname for diagnostics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

/// A held PID lock. Dropping it without calling [`PidLock::release`] leaves
/// the lock file in place (stale-lock detection handles cleanup later); call
/// [`PidLock::release`] explicitly during graceful shutdown.
#[derive(Debug)]
pub struct PidLock {
    path: PathBuf,
    pid: u32,
}

impl PidLock {
    /// Acquire the lock at the given path.
    ///
    /// Creates the lock file atomically. If it already exists, the existing
    /// record is read and checked:
    /// - if the process is alive and not us, acquisition fails;
    /// - if the lock is stale (process dead, or older than the stale
    ///   threshold on platforms that cannot probe), it is cleared and the
    ///   acquisition is retried once.
    pub fn acquire(path: &Path) -> Result<Self, AppError> {
        match Self::try_create(path) {
            Ok(lock) => return Ok(lock),
            Err(e) => {
                // If creation failed for a reason other than "already
                // exists", propagate it.
                if e.code != "LOCK_EXISTS" {
                    return Err(e);
                }
            }
        }

        // The lock exists. Determine whether it is stale.
        let record = read_lock(path)?;
        debug!(path = %path.display(), pid = record.pid, "Existing lock found");
        if is_stale(path, &record) {
            warn!(path = %path.display(), pid = record.pid, "Clearing stale PID lock");
            std::fs::remove_file(path)
                .map_err(|e| AppError::internal(format!("Failed to clear stale lock: {e}")))?;
            return Self::try_create(path).map_err(|e| {
                AppError::internal(format!(
                    "Failed to re-acquire lock after clearing stale: {e}"
                ))
            });
        }

        Err(AppError::internal(format!(
            "Another gateway instance is running (pid {}); refusing to start",
            record.pid
        )))
    }

    /// Try to create the lock file. Returns `LOCK_EXISTS` error if present.
    fn try_create(path: &Path) -> Result<Self, AppError> {
        let pid = std::process::id();
        let record = LockRecord {
            pid,
            created_at_unix_secs: now_unix_secs(),
            hostname: hostname(),
        };
        let json = serde_json::to_string_pretty(&record)
            .map_err(|e| AppError::internal(format!("Failed to serialize lock record: {e}")))?;

        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    AppError::new("LOCK_EXISTS", "Lock file already exists")
                } else {
                    AppError::internal(format!("Failed to create lock file: {e}"))
                }
            })?;
        writeln!(file, "{json}")
            .map_err(|e| AppError::internal(format!("Failed to write lock file: {e}")))?;
        drop(file);

        info!(path = %path.display(), pid = pid, "PID lock acquired");
        Ok(Self {
            path: path.to_path_buf(),
            pid,
        })
    }

    /// Release the lock by removing the lock file.
    ///
    /// Only removes the file if it still contains our pid, to avoid removing
    /// a lock that was re-acquired by another process after a crash.
    pub fn release(&self) -> Result<(), AppError> {
        match read_lock(&self.path) {
            Ok(record) if record.pid == self.pid => {
                std::fs::remove_file(&self.path)
                    .map_err(|e| AppError::internal(format!("Failed to remove lock file: {e}")))?;
                info!(path = %self.path.display(), "PID lock released");
                Ok(())
            }
            Ok(record) => {
                warn!(
                    path = %self.path.display(),
                    our_pid = self.pid,
                    holder_pid = record.pid,
                    "Refusing to remove lock owned by another pid"
                );
                Err(AppError::internal(
                    "Lock file is owned by a different process; not removing",
                ))
            }
            Err(_) => {
                // Lock file gone (e.g. removed externally); treat as released.
                debug!(path = %self.path.display(), "Lock file already absent");
                Ok(())
            }
        }
    }

    /// Return the pid holding this lock.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// Return the lock file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Read and parse the lock file.
fn read_lock(path: &Path) -> Result<LockRecord, AppError> {
    let mut contents = String::new();
    let mut file = File::open(path)
        .map_err(|e| AppError::internal(format!("Failed to open lock file: {e}")))?;
    file.read_to_string(&mut contents)
        .map_err(|e| AppError::internal(format!("Failed to read lock file: {e}")))?;
    serde_json::from_str(&contents)
        .map_err(|e| AppError::internal(format!("Lock file has invalid contents: {e}")))
}

/// Determine whether an existing lock is stale.
///
/// A lock is stale if its process is no longer alive, or (when process
/// liveness cannot be probed) if the lock file is older than the stale
/// threshold.
fn is_stale(path: &Path, record: &LockRecord) -> bool {
    match probe_process_alive(record.pid) {
        Some(true) => false,
        Some(false) => true,
        None => {
            // Cannot probe (unsupported platform or probe failed). Fall back
            // to file age.
            let age = file_age_secs(path).unwrap_or(0);
            age >= DEFAULT_STALE_AFTER_SECS
        }
    }
}

/// Best-effort process liveness probe. Returns `Some(true)`/`Some(false)` if
/// the check ran, or `None` if the platform/command is unavailable.
fn probe_process_alive(pid: u32) -> Option<bool> {
    #[cfg(target_os = "windows")]
    {
        windows_process_alive(pid)
    }
    #[cfg(unix)]
    {
        unix_process_alive(pid)
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        let _ = pid;
        None
    }
}

/// Probe process liveness on Unix via `ps -p <pid> -o pid=`.
#[cfg(unix)]
fn unix_process_alive(pid: u32) -> Option<bool> {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "pid="])
        .output()
        .ok()?;
    if !output.status.success() {
        // `ps` exits non-zero when the pid is not found.
        return Some(false);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    Some(trimmed == pid.to_string())
}

/// Probe process liveness on Windows via `tasklist /FI "PID eq <pid>"`.
#[cfg(target_os = "windows")]
fn windows_process_alive(pid: u32) -> Option<bool> {
    let filter = format!("PID eq {pid}");
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &filter, "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A found process produces a CSV row containing the pid.
    Some(stdout.contains(&format!("\"{pid}\"")))
}

/// Age of a file in whole seconds, or `None` if it cannot be read.
fn file_age_secs(path: &Path) -> Option<u64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    let now = SystemTime::now();
    let age = now.duration_since(modified).unwrap_or(Duration::ZERO);
    Some(age.as_secs())
}

/// Current unix timestamp in seconds.
fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Best-effort hostname.
fn hostname() -> Option<String> {
    std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_lock(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "opensquilla-pidlock-test-{tag}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn test_acquire_release() {
        let path = temp_lock("roundtrip");
        {
            let lock = PidLock::acquire(&path).unwrap();
            assert!(path.exists());
            assert_eq!(lock.pid(), std::process::id());
            let record = read_lock(&path).unwrap();
            assert_eq!(record.pid, std::process::id());
            lock.release().unwrap();
        }
        assert!(!path.exists());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_double_acquire_fails() {
        let path = temp_lock("double");
        let lock = PidLock::acquire(&path).unwrap();
        // A second acquisition while we hold the lock must fail because our
        // process is alive.
        assert!(PidLock::acquire(&path).is_err());
        lock.release().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_stale_lock_cleared() {
        let path = temp_lock("stale");
        // Write a lock owned by a pid that does not exist.
        let record = LockRecord {
            pid: u32::MAX,
            created_at_unix_secs: now_unix_secs(),
            hostname: None,
        };
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();
        // u32::MAX almost certainly does not map to a live process; the stale
        // path should clear the file and acquire the lock.
        let lock = PidLock::acquire(&path).unwrap();
        assert_eq!(lock.pid(), std::process::id());
        lock.release().unwrap();
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_release_refuses_foreign_lock() {
        let path = temp_lock("foreign");
        let lock = PidLock::acquire(&path).unwrap();
        // Simulate another process taking over the lock file.
        let record = LockRecord {
            pid: 1,
            created_at_unix_secs: now_unix_secs(),
            hostname: None,
        };
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();
        assert!(lock.release().is_err());
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_release_absent_ok() {
        let path = temp_lock("absent");
        // Never acquired; release should succeed (no-op).
        let lock = PidLock {
            path: path.clone(),
            pid: std::process::id(),
        };
        assert!(lock.release().is_ok());
    }

    #[test]
    fn test_lock_record_roundtrip() {
        let record = LockRecord {
            pid: 42,
            created_at_unix_secs: 123,
            hostname: Some("box".into()),
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: LockRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pid, 42);
        assert_eq!(parsed.hostname.as_deref(), Some("box"));
    }

    #[test]
    fn test_invalid_lock_contents() {
        let path = temp_lock("invalid");
        std::fs::write(&path, "not-json").unwrap();
        // read_lock should surface an error; acquire will treat an existing
        // but unreadable lock as not stale and refuse to proceed only after
        // read_lock errors propagate.
        assert!(read_lock(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
