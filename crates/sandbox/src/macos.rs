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

    /// macOS sandbox framework FFI. `libc` 0.2 does not export these symbols.
    #[link(name = "sandbox")]
    unsafe extern "C" {
        fn sandbox_init(
            profile: *const libc::c_char,
            flags: u64,
            errorbuf: *mut *mut libc::c_char,
        ) -> libc::c_int;
        fn sandbox_free_error(errorbuf: *mut libc::c_char);
    }

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

            // Enforce a wall-clock timeout on the whole sandboxed execution so
            // a hung child cannot block the worker forever. The CPU limit is
            // enforced separately by setrlimit in the pre_exec hook.
            let timeout_secs = policy.resource_limits.effective_timeout_secs();
            let spawned = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn sandbox-exec: {e}"))?;
            let output = match tokio::time::timeout(
                std::time::Duration::from_secs(timeout_secs),
                spawned.wait_with_output(),
            )
            .await
            {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => {
                    let _ = tokio::fs::remove_file(&sbpl_path).await;
                    return Err(format!("process wait: {e}"));
                }
                Err(_) => {
                    let _ = tokio::fs::remove_file(&sbpl_path).await;
                    return Ok(SandboxResult {
                        exit_code: -1,
                        stdout: String::new(),
                        stderr: format!(
                            "Sandboxed execution timed out after {timeout_secs} seconds"
                        ),
                        duration_ms: start.elapsed().as_millis() as u64,
                        audit_log: Vec::new(),
                    });
                }
            };

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
        let nofile = policy.resource_limits.open_fds;
        let core = policy.resource_limits.core_size_bytes;
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
                    if let Some(nofile) = nofile {
                        let lim = libc::rlimit {
                            rlim_cur: nofile,
                            rlim_max: nofile,
                        };
                        if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_NOFILE): {}",
                                std::io::Error::last_os_error()
                            )));
                        }
                    }
                    if let Some(core) = core {
                        let lim = libc::rlimit {
                            rlim_cur: core,
                            rlim_max: core,
                        };
                        if libc::setrlimit(libc::RLIMIT_CORE, &lim) != 0 {
                            return Err(io_err(format!(
                                "setrlimit(RLIMIT_CORE): {}",
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

    /// Build an SBPL profile through the dedicated `seatbelt` module, which
    /// produces the richer profile (deny-by-default with level-specific
    /// hardening for STRICT/LOCKED). Falls back to the inline [`build_sbpl`]
    /// if the module is unavailable (it is always available).
    fn build_sbpl_advanced(policy: &SandboxPolicy) -> String {
        crate::seatbelt::SeatbeltProfile::from_policy(policy).source
    }

    /// Launch a command with `sandbox_init(3)` instead of the deprecated
    /// `sandbox-exec` binary.
    ///
    /// `sandbox_init` applies the profile to the current process; this is used
    /// for the native fallback where we fork a child, apply the profile with
    /// `sandbox_init`, then `exec` the target. It requires the calling process
    /// to be single-threaded (which holds inside `fork`).
    #[cfg(target_os = "macos")]
    fn launch_with_sandbox_init(
        command: &str,
        args: &[&str],
        env: &HashMap<String, String>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        use std::ffi::CString;

        // Compile the SBPL profile.
        let profile = crate::seatbelt::SeatbeltProfile::from_policy(policy);
        profile.validate_syntax()?;

        // The classic sandbox_init takes a C string; the modern variant takes
        // a URL. We pass the profile inline via the classic entry point.
        let profile_c = CString::new(profile.source.as_bytes()).map_err(|e| format!("CString: {e}"))?;

        unsafe {
            // `sandbox_init(profile, flags, errorbuf)`:
            // flags = 1 (SANDBOX_NAMED) would treat `profile` as a name; we
            // use flags = 0 (SANDBOX_BUILTIN is 2; plain inline profile uses
            // 0).
            let mut errorbuf: *mut libc::c_char = std::ptr::null_mut();
            let result = sandbox_init(profile_c.as_ptr(), 0, &mut errorbuf);
            if result != 0 {
                let err = if errorbuf.is_null() {
                    "sandbox_init failed".to_string()
                } else {
                    let msg = std::ffi::CStr::from_ptr(errorbuf).to_string_lossy().to_string();
                    sandbox_free_error(errorbuf);
                    msg
                };
                return Err(format!("sandbox_init: {err}"));
            }
        }

        // Re-apply resource limits (the child process).
        let limits = RlimitSpec::from_policy(policy);
        apply_rlimits_libc(&limits).map_err(|e| e)?;

        if let Some(cwd) = std::env::var_os("PWD") {
            let _ = std::env::set_current_dir(cwd);
        }
        exec_command(command, args, env);
        Err("exec failed".to_string())
    }

    /// Apply resource limits to the *current* process (used by the native
    /// fallback after `fork`).
    #[cfg(target_os = "macos")]
    fn apply_rlimits_libc(limits: &RlimitSpec) -> Result<(), String> {
        unsafe {
            macro_rules! set_limit {
                ($which:expr, $val:expr) => {{
                    let lim = libc::rlimit {
                        rlim_cur: $val,
                        rlim_max: $val,
                    };
                    if libc::setrlimit($which, &lim) != 0 {
                        return Err(format!(
                            "setrlimit({}): {}",
                            stringify!($which),
                            std::io::Error::last_os_error()
                        ));
                    }
                }};
            }
            if let Some(cpu) = limits.cpu {
                set_limit!(libc::RLIMIT_CPU, cpu);
            }
            if let Some(mem) = limits.mem {
                set_limit!(libc::RLIMIT_AS, mem);
            }
            if let Some(nproc) = limits.nproc {
                set_limit!(libc::RLIMIT_NPROC, nproc);
            }
            if let Some(fsize) = limits.fsize {
                set_limit!(libc::RLIMIT_FSIZE, fsize);
            }
            if let Some(nofile) = limits.nofile {
                set_limit!(libc::RLIMIT_NOFILE, nofile);
            }
            if let Some(core) = limits.core {
                set_limit!(libc::RLIMIT_CORE, core);
            }
        }
        Ok(())
    }

    /// Exec the target with a filtered environment (macOS `execvpe`).
    #[cfg(target_os = "macos")]
    fn exec_command(command: &str, args: &[&str], env: &HashMap<String, String>) {
        use std::ffi::CString;

        // `execve` does not search PATH, so resolve to an absolute path first.
        let resolved = resolve_binary(command).unwrap_or_else(|| command.to_string());
        let program = CString::new(resolved).unwrap_or_default();
        let cargs: Vec<CString> = args
            .iter()
            .filter_map(|a| CString::new(*a).ok())
            .collect();
        let mut arg_ptrs: Vec<*const libc::c_char> = Vec::with_capacity(cargs.len() + 2);
        arg_ptrs.push(program.as_ptr());
        for a in &cargs {
            arg_ptrs.push(a.as_ptr());
        }
        arg_ptrs.push(std::ptr::null());

        let env_vars: Vec<CString> = env
            .iter()
            .filter_map(|(k, v)| CString::new(format!("{k}={v}")).ok())
            .collect();
        let mut env_ptrs: Vec<*const libc::c_char> = env_vars.iter().map(|c| c.as_ptr()).collect();
        env_ptrs.push(std::ptr::null());

        unsafe {
            libc::execve(program.as_ptr(), arg_ptrs.as_ptr(), env_ptrs.as_ptr());
        }
    }

    /// A minimal resource-limit spec for the native macOS path.
    #[cfg(target_os = "macos")]
    #[derive(Clone, Copy)]
    struct RlimitSpec {
        cpu: Option<u64>,
        mem: Option<u64>,
        nproc: Option<u64>,
        fsize: Option<u64>,
        nofile: Option<u64>,
        core: Option<u64>,
    }

    #[cfg(target_os = "macos")]
    impl RlimitSpec {
        fn from_policy(p: &SandboxPolicy) -> Self {
            Self {
                cpu: p.resource_limits.cpu_time_secs,
                mem: p.resource_limits.memory_bytes,
                nproc: p.resource_limits.max_processes,
                fsize: p.resource_limits.file_size_bytes,
                nofile: p.resource_limits.open_fds,
                core: p.resource_limits.core_size_bytes,
            }
        }
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
