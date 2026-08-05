use crate::policy::{SandboxPolicy, SandboxResult, AuditEntry};
use std::process::Stdio;
use tokio::process::Command;
use tracing::info;

/// Noop sandbox backend: no isolation, direct subprocess execution.
///
/// This backend provides no security isolation and is intended for:
/// - Development/testing environments
/// - Platforms not supported by other backends
/// - Operations that don't require sandboxing
pub struct NoopSandbox {
    audit_log: Vec<AuditEntry>,
}

impl NoopSandbox {
    pub fn new() -> Self {
        Self { audit_log: Vec::new() }
    }

    /// Execute a command directly with no sandbox isolation, but with resource limits.
    pub async fn execute(&mut self, command: &str, args: &[&str], policy: &SandboxPolicy) -> Result<SandboxResult, String> {
        let start = std::time::Instant::now();

        info!("Noop sandbox executing: {} {}", command, args.join(" "));

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Apply environment variable allowlist
        for var in &policy.env_allowlist {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
        }

        let child = cmd.spawn().map_err(|e| format!("Failed to spawn process: {}", e))?;
        let child_id = child.id().ok_or("No child process ID")?;

        // Apply timeout if configured
        let output = if let Some(cpu_limit) = policy.resource_limits.cpu_time_secs {
            let timeout = tokio::time::Duration::from_secs(cpu_limit);
            let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

            match result {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => return Err(format!("Process wait: {}", e)),
                Err(_elapsed) => {
                    // Timeout occurred: kill the process
                    #[cfg(unix)]
                    {
                        use nix::sys::signal::{kill, Signal};
                        use nix::unistd::Pid;
                        let _ = kill(Pid::from_raw(child_id as i32), Signal::SIGKILL);
                    }
                    #[cfg(windows)]
                    {
                        let _ = Command::new("taskkill")
                            .args(["/F", "/PID", &child_id.to_string()])
                            .output().await;
                    }

                    let duration = start.elapsed();
                    self.audit_log.push(AuditEntry {
                        timestamp: chrono::Utc::now(),
                        action: "noop_execute_timeout".to_string(),
                        details: format!("command={}, timed out after {}s", command, cpu_limit),
                    });
                    return Ok(SandboxResult {
                        exit_code: -1,
                        stdout: String::new(),
                        stderr: format!("Process timed out after {} seconds", cpu_limit),
                        duration_ms: duration.as_millis() as u64,
                        audit_log: self.audit_log.clone(),
                    });
                }
            }
        } else {
            child.wait_with_output().await.map_err(|e| format!("Process wait: {}", e))?
        };

        let duration = start.elapsed();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        self.audit_log.push(AuditEntry {
            timestamp: chrono::Utc::now(),
            action: "noop_execute".to_string(),
            details: format!("command={}, exit={}, duration={:?}", command, output.status.code().unwrap_or(-1), duration),
        });

        Ok(SandboxResult {
            exit_code: output.status.code().unwrap_or(-1),
            stdout,
            stderr,
            duration_ms: duration.as_millis() as u64,
            audit_log: self.audit_log.clone(),
        })
    }
}

impl Default for NoopSandbox {
    fn default() -> Self {
        Self::new()
    }
}