//! Noop sandbox backend: no isolation, direct subprocess execution.
//!
//! Intended for development/testing environments, platforms not supported by
//! other backends, and operations that don't require sandboxing. It still
//! applies the policy's resource limits (via `setrlimit` on Unix, Job Object
//! hints via process group control on Windows), an environment allowlist, a
//! working directory, and a CPU-time timeout with process-group kill.

use crate::policy::{AuditEntry, SandboxPolicy, SandboxResult};
use std::collections::HashMap;
use std::process::Stdio;
use tokio::process::Command;
use tracing::info;

/// Noop sandbox backend.
pub struct NoopSandbox {
    audit_log: Vec<AuditEntry>,
}

impl NoopSandbox {
    /// Create a new noop sandbox backend.
    pub fn new() -> Self {
        Self {
            audit_log: Vec::new(),
        }
    }

    /// Execute a command directly with resource limits and a timeout.
    pub async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        self.run(command, args, None, None, policy).await
    }

    /// Execute a command directly with an explicit environment and working
    /// directory.
    pub async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        self.run(command, args, Some(env), working_dir, policy).await
    }

    /// The noop sandbox is always "available".
    pub fn health_check(&self) -> bool {
        true
    }

    /// Snapshot of recorded audit entries.
    pub fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }

    async fn run(
        &mut self,
        command: &str,
        args: &[&str],
        env: Option<HashMap<String, String>>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        let start = std::time::Instant::now();
        let filtered_env = filter_env(env.as_ref(), policy);
        let limits = RlimitSpec::from_policy(policy);

        info!("noop sandbox executing: {} {}", command, args.join(" "));

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        cmd.envs(&filtered_env);
        if let Some(cwd) = working_dir {
            cmd.current_dir(cwd);
        }

        apply_process_controls(&mut cmd, limits)?;

        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn process: {e}"))?;
        let child_id = child.id();

        let cpu_time = policy.resource_limits.cpu_time_secs.unwrap_or(300).max(1);
        let timeout = tokio::time::Duration::from_secs(cpu_time);
        let output = tokio::time::timeout(timeout, child.wait_with_output()).await;

        let (exit_code, stdout, stderr, timed_out) = match output {
            Ok(Ok(o)) => (
                o.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&o.stdout).to_string(),
                String::from_utf8_lossy(&o.stderr).to_string(),
                false,
            ),
            Ok(Err(e)) => return Err(format!("process wait: {e}")),
            Err(_) => {
                kill_process_group(child_id).await;
                (
                    -1,
                    String::new(),
                    format!("Process timed out after {cpu_time} seconds"),
                    true,
                )
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        self.audit_log.push(AuditEntry {
            timestamp: chrono::Utc::now(),
            action: if timed_out {
                "noop_execute_timeout".to_string()
            } else {
                "noop_execute".to_string()
            },
            details: format!(
                "command={command}, exit={exit_code}, duration_ms={duration_ms}, timed_out={timed_out}"
            ),
        });

        Ok(SandboxResult {
            exit_code,
            stdout,
            stderr,
            duration_ms,
            audit_log: self.audit_log.clone(),
        })
    }
}

impl Default for NoopSandbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource limit spec derived from a policy.
///
/// Only consumed on Unix (via `pre_exec`); suppressed elsewhere.
#[cfg_attr(not(unix), allow(dead_code))]
#[derive(Clone, Copy)]
struct RlimitSpec {
    cpu: Option<u64>,
    mem: Option<u64>,
    nproc: Option<u64>,
    fsize: Option<u64>,
}

impl RlimitSpec {
    fn from_policy(p: &SandboxPolicy) -> Self {
        Self {
            cpu: p.resource_limits.cpu_time_secs,
            mem: p.resource_limits.memory_bytes,
            nproc: p.resource_limits.max_processes,
            fsize: p.resource_limits.file_size_bytes,
        }
    }
}

/// Intersect an environment map with the policy allowlist.
fn filter_env(
    env: Option<&HashMap<String, String>>,
    policy: &SandboxPolicy,
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let supplied = env.cloned().unwrap_or_default();
    for var in &policy.env_allowlist {
        if let Some(v) = supplied.get(var) {
            out.insert(var.clone(), v.clone());
        } else if let Ok(v) = std::env::var(var) {
            out.insert(var.clone(), v);
        }
    }
    out
}

/// Configure process-group management and resource limits on the command.
fn apply_process_controls(cmd: &mut Command, limits: RlimitSpec) -> Result<(), String> {
    #[cfg(unix)]
    {
        unsafe {
            cmd.pre_exec(move || {
                // New session so the child leads its own process group (and is
                // not in the parent's group), enabling group-wide kill.
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                apply_rlimits(&limits)
            });
        }
    }
    #[cfg(windows)]
    {
        use windows::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP.0 as u32);
    }
    Ok(())
}

/// Apply `setrlimit` limits to the current process (Unix).
#[cfg(unix)]
fn apply_rlimits(limits: &RlimitSpec) -> Result<(), std::io::Error> {
    unsafe {
        if let Some(cpu) = limits.cpu {
            let lim = libc::rlimit {
                rlim_cur: cpu,
                rlim_max: cpu,
            };
            if libc::setrlimit(libc::RLIMIT_CPU, &lim) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        if let Some(mem) = limits.mem {
            let lim = libc::rlimit {
                rlim_cur: mem,
                rlim_max: mem,
            };
            if libc::setrlimit(libc::RLIMIT_AS, &lim) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        if let Some(nproc) = limits.nproc {
            let lim = libc::rlimit {
                rlim_cur: nproc,
                rlim_max: nproc,
            };
            if libc::setrlimit(libc::RLIMIT_NPROC, &lim) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        if let Some(fsize) = limits.fsize {
            let lim = libc::rlimit {
                rlim_cur: fsize,
                rlim_max: fsize,
            };
            if libc::setrlimit(libc::RLIMIT_FSIZE, &lim) != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
    }
    Ok(())
}

/// Kill the child's process group (or the process tree) after a timeout.
async fn kill_process_group(child_id: Option<u32>) {
    #[cfg(unix)]
    {
        if let Some(id) = child_id {
            use nix::sys::signal::{killpg, Signal};
            use nix::unistd::Pid;
            let _ = killpg(Pid::from_raw(id as i32), Signal::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        if let Some(id) = child_id {
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &id.to_string()])
                .output()
                .await;
        }
    }
}

#[async_trait::async_trait]
impl crate::Sandbox for NoopSandbox {
    async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        NoopSandbox::execute(self, command, args, policy).await
    }

    async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        NoopSandbox::execute_with_env(self, command, args, env, working_dir, policy).await
    }

    fn health_check(&self) -> bool {
        NoopSandbox::health_check(self)
    }

    fn name(&self) -> &'static str {
        "noop"
    }

    fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }
}
