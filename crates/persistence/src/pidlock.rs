use std::path::{Path, PathBuf};

/// A simple PID lock file guarding against multiple concurrent processes.
///
/// The lock is released automatically when the value is dropped.
pub struct PidLock {
    path: PathBuf,
    pid: u32,
}

impl PidLock {
    /// Try to acquire the lock at the given path.
    ///
    /// If a lock file exists and references a live process, acquisition fails.
    /// Stale lock files from dead processes are overwritten.
    pub fn acquire(path: impl Into<PathBuf>) -> crate::Result<Self> {
        let path = path.into();
        let pid = std::process::id();

        if path.exists() {
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            let existing: u32 = contents.trim().parse().unwrap_or(0);
            if existing != 0 && process_exists(existing) {
                return Err(crate::Error::InvalidState(format!(
                    "Another OpenSquilla process (PID {existing}) holds the lock {}",
                    path.display()
                )));
            }
        }

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, pid.to_string())?;
        tracing::debug!(pid, path = %path.display(), "acquired pid lock");

        Ok(Self { path, pid })
    }

    /// The PID holding this lock.
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The lock file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Release the lock, removing the file only if it still belongs to us.
    pub fn release(&self) {
        if let Ok(contents) = std::fs::read_to_string(&self.path) {
            if contents.trim().parse::<u32>().unwrap_or(0) == self.pid {
                let _ = std::fs::remove_file(&self.path);
                tracing::debug!(pid = self.pid, "released pid lock");
            }
        }
    }
}

impl Drop for PidLock {
    fn drop(&mut self) {
        self.release();
    }
}

/// Best-effort check whether a process with the given PID is alive.
#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    use std::process::Command;
    let filter = format!("PID eq {pid}");
    Command::new("tasklist")
        .args(["/FI", filter.as_str(), "/NH"])
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(not(windows))]
fn process_exists(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}
