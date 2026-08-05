//! macOS sandbox backend using `sandbox-exec` (Seatbelt).
//!
//! Generates an SBPL (SandBox Profile Language) profile at runtime, writes it
//! to a temporary file, and launches the target command through
//! `sandbox-exec -f <profile> <command>`. The profile is deny-by-default for
//! both filesystem and network access, with explicit allow rules derived from
//! the [`SandboxPolicy`].
//!
//! At STRICT/LOCKED levels the target binary is additionally checked with
//! `codesign --verify --strict` before it is allowed to run, and resource
//! limits are applied via `setrlimit` in a `pre_exec` hook.
//!
//! Note: `sandbox-exec` is deprecated by Apple but remains present on current
//! macOS releases; this backend intentionally uses it because it is the
//! standard Seatbelt interface and requires no third-party tooling.

use crate::policy::{AuditEntry, SandboxPolicy, SandboxResult};
use std::collections::HashMap;

#[cfg(target_os = "macos")]
mod backend {
    use super::*;
    use std::process::Stdio;
    use tokio::process::Command;
    use tracing::{info, warn};

    pub struct MacOsBackend {
        sandbox_exec_path: String,
        codesign_path: Option<String>,
    }

    impl MacOsBackend {
        pub fn new() -> Self {
            Self {
                sandbox_exec_path: which("sandbox-exec").unwrap_or_else(|| "sandbox-exec".into()),
                codesign_path: which("codesign"),
            }
        }

        pub fn available(&self) -> bool {
            !self.sandbox_exec_path.is_empty()
        }

        pub async fn execute(
            &self,
            command: &str,
            args: &[&str],
            env: Option<&HashMap<String, String>>,
            working_dir: Option<&str>,
            policy: &SandboxPolicy,
        ) -> Result<SandboxResult, String> {
            let start = std::time::Instant::now();

            if policy.level >= crate::policy::SandboxLevel::Strict {
                if let Some(binary) = resolve_binary(command) {
                    self.verify_code_signature(&binary).await?;
                } else {
                    warn!(
                        "macos sandbox: could not resolve '{}' for signature check",
                        command
                    );
                }
            }

            let sbpl = build_sbpl(policy);
            let temp_dir = std::env::temp_dir();
            let sbpl_path = temp_dir.join(format!("sandbox_{}.sbpl", uuid::Uuid::new_v4()));
            tokio::fs::write(&sbpl_path, sbpl.as_bytes())
                .await
                .map_err(|e| format!("failed to write SBPL file: {e}"))?;

            info!(
                "macos sandbox executing: sandbox-exec -f {} {} {}",
                sbpl_path.display(),
                command,
                args.join(" ")
            );

            let mut cmd = Command::new(&self.sandbox_exec_path);
            cmd.arg("-f")
                .arg(&sbpl_path)
                .arg(command)
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let env_filtered = filter_env(env, policy);
            cmd.envs(&env_filtered);
            if let Some(cwd) = working_dir {
                cmd.current_dir(cwd);
            }
            apply_rlimits_pre_exec(&mut cmd, policy)?;

            let output = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn sandbox-exec: {e}"))?
                .wait_with_output()
                .await
                .map_err(|e| format!("process wait: {e}"))?;

            let _ = tokio::fs::remove_file(&sbpl_path).await;
            let duration = start.elapsed();

            Ok(SandboxResult {
                exit_code: output.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                duration_ms: duration.as_millis() as u64,
                audit_log: Vec::new(),
            })
        }

        /// Verify the target binary carries a valid, strict code signature.
        async fn verify_code_signature(&self, binary: &str) -> Result<(), String> {
            let codesign = self.codesign_path.as_deref().unwrap_or("codesign");
            let output = Command::new(codesign)
                .args(["--verify", "--strict", "--deep", binary])
                .output()
                .await
                .map_err(|e| format!("codesign invocation failed: {e}"))?;
            if output.status.success() {
                info!("macos sandbox: code signature verified for {}", binary);
                Ok(())
            } else {
                Err(format!(
                    "code signature verification failed for '{}': {}",
                    binary,
                    String::from_utf8_lossy(&output.stderr).trim()
                ))
            }
        }
    }

    impl Default for MacOsBackend {
        fn default() -> Self {
            Self::new()
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

    fn io_err(msg: String) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::Other, msg)
    }

    /// Apply `setrlimit` limits in a `pre_exec` hook so the sandboxed program
    /// inherits them across `exec`.
    fn apply_rlimits_pre_exec(cmd: &mut Command, policy: &SandboxPolicy) -> Result<(), String> {
        let cpu = policy.resource_limits.cpu_time_secs;
        let mem = policy.resource_limits.memory_bytes;
        let nproc = policy.resource_limits.max_processes;
        let fsize = policy.resource_limits.file_size_bytes;
        unsafe {
            cmd.pre_exec(move || {
                unsafe {
                    if let Some(cpu) = cpu {
                        let lim = libc::rlimit {
                            rlim_cur: cpu,
                            rlim_max: cpu,
                        };
                        if libc::setrlimit(libc::RLIMIT_CPU, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_CPU): {}",
                                std::io::Error::last_os_error()
                            )));
                        }
                    }
                    if let Some(mem) = mem {
                        let lim = libc::rlimit {
                            rlim_cur: mem,
                            rlim_max: mem,
                        };
                        if libc::setrlimit(libc::RLIMIT_AS, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_AS): {}",
                                std::io::Error::last_os_error()
                            )));
                        }
                    }
                    if let Some(nproc) = nproc {
                        let lim = libc::rlimit {
                            rlim_cur: nproc,
                            rlim_max: nproc,
                        };
                        if libc::setrlimit(libc::RLIMIT_NPROC, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_NPROC): {}",
                                std::io::Error::last_os_error()
                            )));
                        }
                    }
                    if let Some(fsize) = fsize {
                        let lim = libc::rlimit {
                            rlim_cur: fsize,
                            rlim_max: fsize,
                        };
                        if libc::setrlimit(libc::RLIMIT_FSIZE, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_FSIZE): {}",
                                std::io::Error::last_os_error()
                            )));
                        }
                    }
                }
                Ok(())
            });
        }
        Ok(())
    }

    /// Escape a path for inclusion in an SBPL string literal.
    fn sbpl_escape(path: &str) -> String {
        path.replace('\\', "\\\\").replace('"', "\\\"")
    }

    /// Compile an SBPL (SandBox Profile Language) profile from the policy.
    fn build_sbpl(policy: &SandboxPolicy) -> String {
        let mut sbpl = String::new();

        // Version header and deny-by-default.
        sbpl.push_str("(version 1)\n");
        sbpl.push_str("(deny default)\n");

        // Basic process operations.
        sbpl.push_str("(allow process-fork)\n");
        sbpl.push_str("(allow process-exec)\n");
        sbpl.push_str("(allow process-info*)\n");
        sbpl.push_str("(allow signal (target self))\n");
        sbpl.push_str("(allow sysctl-read)\n");
        sbpl.push_str("(allow ipc-posix-semaphore)\n");
        sbpl.push_str("(allow ipc-posix-shm)\n");
        sbpl.push_str("(allow mach-lookup (global-name \"com.apple.system.logger\"))\n");
        sbpl.push_str("(allow mach-task-self)\n");
        sbpl.push_str("(allow mach-privilege-task-port)\n");

        // Filesystem: allow metadata traversal of the root, then explicit
        // read/write rules.
        sbpl.push_str("(allow file-read-metadata (subpath \"/\") (subpath \"/private/\"))\n");
        sbpl.push_str(
            "(allow file-read* (subpath \"/usr\") (subpath \"/System\") (subpath \"/Library\") (subpath \"/private/var/tmp\"))\n",
        );

        for path in &policy.filesystem.read_allowed {
            if !path.is_empty() {
                sbpl.push_str(&format!(
                    "(allow file-read* (subpath \"{}\"))\n",
                    sbpl_escape(path)
                ));
            }
        }
        for path in &policy.filesystem.write_allowed {
            if !path.is_empty() {
                sbpl.push_str(&format!(
                    "(allow file-read* file-write* (subpath \"{}\"))\n",
                    sbpl_escape(path)
                ));
            }
        }

        if policy.filesystem.home_readable {
            if let Some(home) = std::env::var_os("HOME") {
                let home = home.to_string_lossy();
                sbpl.push_str(&format!(
                    "(allow file-read* (subpath \"{}\"))\n",
                    sbpl_escape(&home)
                ));
            }
        }
        if policy.filesystem.tmp_writable {
            sbpl.push_str("(allow file-read* file-write* (subpath \"/tmp\"))\n");
            sbpl.push_str("(allow file-read* file-write* (subpath \"/private/tmp\"))\n");
        }

        // Explicitly deny write access everywhere else.
        sbpl.push_str("(deny file-write*)\n");

        // Network rules.
        match &policy.network {
            crate::policy::NetworkPolicy::None => {
                sbpl.push_str("(deny network*)\n");
            }
            crate::policy::NetworkPolicy::ProxyAllowlist(domains) => {
                sbpl.push_str("(allow network* (local ip \"127.0.0.1\"))\n");
                sbpl.push_str("(allow network-outbound (remote ip \"::1\"))\n");
                for domain in domains {
                    sbpl.push_str(&format!(
                        "(allow network-outbound (remote name \"{}\"))\n",
                        sbpl_escape(domain)
                    ));
                }
            }
            crate::policy::NetworkPolicy::Host => {
                sbpl.push_str("(allow network*)\n");
            }
        }

        sbpl
    }
}

#[cfg(target_os = "macos")]
fn which(name: &str) -> Option<String> {
    std::env::var_os("PATH").and_then(|paths| {
        for path in std::env::split_paths(&paths) {
            let candidate = path.join(name);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
        None
    })
}

/// Resolve a bare command name to an absolute path using `PATH`.
#[cfg(target_os = "macos")]
fn resolve_binary(command: &str) -> Option<String> {
    let path = std::path::Path::new(command);
    if path.is_absolute() {
        return Some(command.to_string());
    }
    std::env::var_os("PATH").and_then(|paths| {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(command);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
        None
    })
}

/// macOS sandbox backend.
///
/// The struct exists on every platform so callers can reference `MacOsSandbox`
/// unconditionally, but on non-macOS targets `execute` returns an
/// `UnsupportedPlatform` error.
pub struct MacOsSandbox {
    audit_log: Vec<AuditEntry>,
    #[cfg(target_os = "macos")]
    backend: backend::MacOsBackend,
}

impl MacOsSandbox {
    /// Create a new macOS sandbox backend.
    pub fn new() -> Self {
        Self {
            audit_log: Vec::new(),
            #[cfg(target_os = "macos")]
            backend: backend::MacOsBackend::new(),
        }
    }

    /// Execute a command inside a macOS sandbox-exec sandbox.
    pub async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        self.run(command, args, None, None, policy).await
    }

    /// Execute a command inside a macOS sandbox-exec sandbox with an explicit
    /// environment and working directory.
    pub async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        self.run(command, args, Some(env), working_dir, policy)
            .await
    }

    /// Check whether sandbox-exec is available on this host.
    pub fn health_check(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.backend.available()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
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
        #[cfg(target_os = "macos")]
        {
            let result = self
                .backend
                .execute(command, args, env.as_ref(), working_dir, policy)
                .await;
            let (ok, duration_ms) = match &result {
                Ok(r) => (true, r.duration_ms),
                Err(_) => (false, 0),
            };
            self.audit_log.push(AuditEntry {
                timestamp: chrono::Utc::now(),
                action: if ok {
                    "macos_execute".to_string()
                } else {
                    "macos_execute_failed".to_string()
                },
                details: format!("command={command}, ok={ok}, duration_ms={duration_ms}"),
            });
            let mut res = result?;
            res.audit_log = self.audit_log.clone();
            Ok(res)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (&args, &env, &working_dir, &policy);
            self.audit_log.push(AuditEntry {
                timestamp: chrono::Utc::now(),
                action: "macos_execute_unsupported".to_string(),
                details: format!("command={command}"),
            });
            Err("macOS sandbox is not available on this platform".to_string())
        }
    }
}

impl Default for MacOsSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl crate::Sandbox for MacOsSandbox {
    async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        MacOsSandbox::execute(self, command, args, policy).await
    }

    async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        MacOsSandbox::execute_with_env(self, command, args, env, working_dir, policy).await
    }

    fn health_check(&self) -> bool {
        MacOsSandbox::health_check(self)
    }

    fn name(&self) -> &'static str {
        "macos"
    }

    fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }
}
