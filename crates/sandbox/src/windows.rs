//! Windows sandbox backend.
//!
//! Layered isolation using the `windows` crate:
//!
//! 1. **Job Object** — created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` plus
//!    process-count, per-process-memory and CPU-time limits derived from the
//!    policy. Every sandboxed process is assigned to the job.
//! 2. **Restricted token (hardening)** — `CreateRestrictedToken` with
//!    `DISABLE_MAX_PRIVILEGE` and a low-integrity label. When the host runs
//!    elevated, the process is spawned with `CreateProcessAsUserW` using this
//!    token; otherwise execution falls back to a `tokio` spawn assigned to the
//!    Job Object. The restricted-token path needs elevation because
//!    `CreateProcessAsUserW` requires the `SE_ASSIGNPRIMARYTOKEN_NAME`
//!    privilege.
//! 3. **Network isolation (best-effort)** — Windows Firewall rules
//!    (`INetFwPolicy2`, which is WFP-backed) block the sandboxed program's
//!    outbound traffic in `NONE` mode. Rule installation requires elevation;
//!    failures degrade to a logged warning and the rule is removed when the
//!    execution finishes.
//!
//! Filesystem control is enforced through the restricted token + Job Object
//! (process working set), complemented by the policy's path allow/deny lists
//! applied by the network proxy layer where relevant.

use crate::policy::{AuditEntry, SandboxPolicy, SandboxResult};
use std::collections::HashMap;

#[cfg(windows)]
mod backend {
    use super::*;
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::process::Stdio;
    use tokio::process::Command;
    use tracing::warn;
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::Foundation::{CloseHandle, BOOL, HANDLE, STILL_ACTIVE};
    use windows::Win32::Security::{
        CreateRestrictedToken, CreateWellKnownSid, SetTokenInformation,
        CREATE_RESTRICTED_TOKEN_FLAGS, PSID, SECURITY_ATTRIBUTES, SID, SID_AND_ATTRIBUTES,
        TOKEN_ACCESS_MASK, TOKEN_INFORMATION_CLASS, TOKEN_MANDATORY_LABEL, WELL_KNOWN_SID_TYPE,
    };
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_PROCESS_TIME, JOBOBJECTINFOCLASS,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    };
    use windows::Win32::System::Pipes::CreatePipe;
    use windows::Win32::System::Threading::{
        CreateProcessAsUserW, GetCurrentProcess, GetExitCodeProcess, OpenProcessToken, ResumeThread,
        TerminateProcess, CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
        PROCESS_CREATION_FLAGS, PROCESS_INFORMATION, STARTUPINFOW, STARTUPINFOW_FLAGS,
    };

    /// `CLSID_NetFwPolicy2` (`{E2B3C97F-6AE1-41AC-817A-F6F92166D7DD}`). The `windows` crate
    /// 0.58 does not expose this constant, so the class GUID is passed to `CoCreateInstance`
    /// directly.
    const CLSID_NET_FW_POLICY2: windows::core::GUID =
        windows::core::GUID::from_u128(0xE2B3C97F_6AE1_41AC_817A_F6F92166D7DD);
    /// `CLSID_NetFwRule` (`{2C5BC43E-3369-4C33-AB0C-BE9469677AF4}`). The `windows` crate
    /// 0.58 does not expose this constant, so the class GUID is passed to `CoCreateInstance`
    /// directly.
    const CLSID_NET_FW_RULE: windows::core::GUID =
        windows::core::GUID::from_u128(0x2C5BC43E_3369_4C33_AB0C_BE9469677AF4);

    pub struct WindowsBackend;

    impl WindowsBackend {
        pub fn new() -> Self {
            Self
        }

        pub fn available(&self) -> bool {
            // Job Objects are available on every supported Windows version.
            match create_job_object(&crate::policy::ResourceLimits::default()) {
                Ok(job) => {
                    let _ = unsafe { CloseHandle(job) };
                    true
                }
                Err(_) => false,
            }
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
            let filtered_env = filter_env(env, policy);

            // Job Object for process-group/resource management.
            let job = create_job_object(&policy.resource_limits)?;

            // Best-effort network isolation.
            let mut firewall_rule: Option<String> = None;
            if policy.network.is_none() {
                if let Some(program) = resolve_binary(command) {
                    match upsert_firewall_rule(
                        &format!("OpenSquilla Sandbox Block {}", uuid::Uuid::new_v4()),
                        &program,
                        windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_RULE_DIRECTION::NET_FW_RULE_DIR_OUT,
                        windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_ACTION::NET_FW_ACTION_BLOCK,
                    ) {
                        Ok(name) => firewall_rule = Some(name),
                        Err(e) => warn!("network isolation rule unavailable: {e}"),
                    }
                }
            }

            // Hardened path: restricted token + CreateProcessAsUserW.
            let mut outcome: Option<Result<(i32, Vec<u8>, Vec<u8>), String>> = None;
            match create_restricted_token() {
                Ok(token) => {
                    match spawn_restricted(command, args, &filtered_env, working_dir, token, job) {
                        Ok(proc) => {
                            let result = wait_restricted(proc, policy.resource_limits.cpu_time_secs)
                                .await;
                            outcome = Some(result);
                        }
                        Err(e) => {
                            warn!("restricted-token spawn failed ({e}); using job-object sandbox");
                        }
                    }
                    let _ = unsafe { CloseHandle(token) };
                }
                Err(e) => {
                    warn!("restricted token unavailable ({e}); using job-object sandbox");
                }
            }

            // Fallback: tokio spawn + Job Object assignment.
            if outcome.is_none() {
                let result = self
                    .spawn_tokio(command, args, &filtered_env, working_dir, policy, job)
                    .await;
                outcome = Some(result);
            }

            // Remove the temporary firewall rule once execution finishes.
            if let Some(name) = firewall_rule {
                if let Err(e) = remove_firewall_rule(&name) {
                    debug_network_rule_removal_failed(&name, &e);
                }
            }

            let _ = unsafe { CloseHandle(job) };

            let (exit_code, stdout, stderr) = match outcome {
                Some(Ok(t)) => t,
                Some(Err(e)) => return Err(e),
                None => return Err("no execution path produced a result".to_string()),
            };

            Ok(SandboxResult {
                exit_code,
                stdout: String::from_utf8_lossy(&stdout).to_string(),
                stderr: String::from_utf8_lossy(&stderr).to_string(),
                duration_ms: start.elapsed().as_millis() as u64,
                audit_log: Vec::new(),
            })
        }

        /// Fallback spawn: `tokio::process::Command` with a new process group,
        /// assigned to the Job Object immediately after spawn.
        async fn spawn_tokio(
            &self,
            command: &str,
            args: &[&str],
            env: &HashMap<String, String>,
            working_dir: Option<&str>,
            policy: &SandboxPolicy,
            job: HANDLE,
        ) -> Result<(i32, Vec<u8>, Vec<u8>), String> {
            let flags = (CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT).0 as u32;
            let mut cmd = Command::new(command);
            cmd.args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true)
                .creation_flags(flags);
            cmd.envs(env);
            if let Some(cwd) = working_dir {
                cmd.current_dir(cwd);
            }

            let mut child = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn process: {e}"))?;

            unsafe {
                AssignProcessToJobObject(job, HANDLE(child.as_raw_handle()))
                    .map_err(|e| format!("AssignProcessToJobObject: {e}"))?;
            }

            let timeout_dur =
                std::time::Duration::from_secs(policy.resource_limits.cpu_time_secs.unwrap_or(300));
            match tokio::time::timeout(timeout_dur, child.wait_with_output()).await {
                Ok(Ok(output)) => Ok((
                    output.status.code().unwrap_or(-1),
                    output.stdout,
                    output.stderr,
                )),
                Ok(Err(e)) => Err(format!("process wait: {e}")),
                Err(_) => {
                    let _ = child.kill().await;
                    Ok((
                        -1,
                        Vec::new(),
                        format!("Process timed out after {} seconds", timeout_dur.as_secs())
                            .into_bytes(),
                    ))
                }
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

    /// Create a Job Object with limits derived from the policy and
    /// `KILL_ON_JOB_CLOSE` so no sandboxed process outlives the job.
    fn create_job_object(limits: &crate::policy::ResourceLimits) -> Result<HANDLE, String> {
        unsafe {
            let job = CreateJobObjectW(None, PCWSTR::null())
                .map_err(|e| format!("CreateJobObjectW: {e}"))?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            let mut flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if let Some(mem) = limits.memory_bytes {
                info.BasicLimitInformation.ProcessMemoryLimit = mem as usize;
                flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            }
            if let Some(cpu) = limits.cpu_time_secs {
                // 100-nanosecond units.
                info.BasicLimitInformation.PerProcessUserTimeLimit =
                    (cpu as i64).saturating_mul(10_000_000);
                flags |= JOB_OBJECT_LIMIT_PROCESS_TIME;
            }
            if let Some(nproc) = limits.max_processes {
                info.BasicLimitInformation.ActiveProcessLimit = nproc as u32;
                flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
            }
            info.BasicLimitInformation.LimitFlags = flags;

            SetInformationJobObject(
                job,
                JOBOBJECTINFOCLASS::JobObjectExtendedLimitInformation,
                &info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(|e| format!("SetInformationJobObject: {e}"))?;
            Ok(job)
        }
    }

    /// Create a restricted token from the current process token: all
    /// privileges disabled (`DISABLE_MAX_PRIVILEGE`) and low integrity.
    fn create_restricted_token() -> Result<HANDLE, String> {
        unsafe {
            let mut process_token: HANDLE = HANDLE(std::ptr::null_mut());
            OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ACCESS_MASK::TOKEN_ASSIGN_PRIMARY
                    | TOKEN_ACCESS_MASK::TOKEN_DUPLICATE
                    | TOKEN_ACCESS_MASK::TOKEN_QUERY
                    | TOKEN_ACCESS_MASK::TOKEN_ADJUST_DEFAULT,
                &mut process_token,
            )
            .map_err(|e| format!("OpenProcessToken: {e}"))?;

            let mut restricted: HANDLE = HANDLE(std::ptr::null_mut());
            CreateRestrictedToken(
                process_token,
                CREATE_RESTRICTED_TOKEN_FLAGS::DISABLE_MAX_PRIVILEGE,
                None,
                None,
                None,
                &mut restricted,
            )
            .map_err(|e| format!("CreateRestrictedToken: {e}"))?;

            // Label the token low-integrity.
            let mut sid_buf = [0u8; 256];
            let sid_ptr = sid_buf.as_mut_ptr() as *mut SID;
            let mut sid_size = sid_buf.len() as u32;
            CreateWellKnownSid(
                WELL_KNOWN_SID_TYPE::WinLowLabelSid,
                None,
                PSID(sid_ptr as *mut c_void),
                &mut sid_size,
            )
            .map_err(|e| format!("CreateWellKnownSid: {e}"))?;

            let label = TOKEN_MANDATORY_LABEL {
                Label: SID_AND_ATTRIBUTES {
                    Sid: PSID(sid_ptr as *mut c_void),
                    // SE_GROUP_INTEGRITY
                    Attributes: 0x20,
                },
            };
            SetTokenInformation(
                restricted,
                TOKEN_INFORMATION_CLASS::TokenIntegrityLevel,
                &label as *const TOKEN_MANDATORY_LABEL as *const c_void,
                std::mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
            )
            .map_err(|e| format!("SetTokenInformation(low integrity): {e}"))?;

            let _ = CloseHandle(process_token);
            Ok(restricted)
        }
    }

    /// A process spawned through `CreateProcessAsUserW`.
    struct RestrictedProcess {
        process: HANDLE,
        thread: HANDLE,
        stdout_read: HANDLE,
        stderr_read: HANDLE,
    }

    /// Spawn a process using the restricted token. Creates pipes for stdout /
    /// stderr, launches suspended, assigns to the Job Object, then resumes.
    fn spawn_restricted(
        command: &str,
        args: &[&str],
        env: &HashMap<String, String>,
        working_dir: Option<&str>,
        token: HANDLE,
        job: HANDLE,
    ) -> Result<RestrictedProcess, String> {
        unsafe {
            let mut sa = SECURITY_ATTRIBUTES {
                nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                lpSecurityDescriptor: std::ptr::null_mut(),
                bInheritHandle: BOOL(1),
            };

            let mut out_read: HANDLE = HANDLE(std::ptr::null_mut());
            let mut out_write: HANDLE = HANDLE(std::ptr::null_mut());
            let mut err_read: HANDLE = HANDLE(std::ptr::null_mut());
            let mut err_write: HANDLE = HANDLE(std::ptr::null_mut());
            CreatePipe(&mut out_read, &mut out_write, Some(&sa), 0)
                .map_err(|e| format!("CreatePipe(stdout): {e}"))?;
            CreatePipe(&mut err_read, &mut err_write, Some(&sa), 0)
                .map_err(|e| format!("CreatePipe(stderr): {e}"))?;

            let cmdline = build_command_line(command, args);
            let mut cmdline_wide: Vec<u16> =
                cmdline.encode_utf16().chain(std::iter::once(0)).collect();
            let env_block = build_environment_block(env);
            let cwd_wide = working_dir
                .map(|c| c.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>());
            let cwd_ptr = cwd_wide
                .as_ref()
                .map(|v| PCWSTR::from_raw(v.as_ptr()))
                .unwrap_or(PCWSTR::null());

            let mut si = STARTUPINFOW {
                cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                dwFlags: STARTUPINFOW_FLAGS::STARTF_USESTDHANDLES,
                hStdInput: HANDLE(std::ptr::null_mut()),
                hStdOutput: out_write,
                hStdError: err_write,
                ..Default::default()
            };
            let mut pi: PROCESS_INFORMATION = PROCESS_INFORMATION::default();

            let creation_flags: PROCESS_CREATION_FLAGS =
                CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT;

            CreateProcessAsUserW(
                token,
                PCWSTR::null(),
                PWSTR(cmdline_wide.as_mut_ptr()),
                None,
                None,
                BOOL(1),
                creation_flags,
                Some(env_block.as_ptr() as *const c_void),
                cwd_ptr,
                &si,
                &mut pi,
            )
            .map_err(|e| format!("CreateProcessAsUserW: {e}"))?;

            if let Err(e) = AssignProcessToJobObject(job, pi.hProcess) {
                let _ = CloseHandle(pi.hProcess);
                let _ = CloseHandle(pi.hThread);
                return Err(format!("AssignProcessToJobObject: {e}"));
            }
            if let Err(e) = ResumeThread(pi.hThread) {
                let _ = CloseHandle(pi.hProcess);
                let _ = CloseHandle(pi.hThread);
                return Err(format!("ResumeThread: {e}"));
            }

            // The parent does not need the write ends.
            let _ = CloseHandle(out_write);
            let _ = CloseHandle(err_write);

            Ok(RestrictedProcess {
                process: pi.hProcess,
                thread: pi.hThread,
                stdout_read: out_read,
                stderr_read: err_read,
            })
        }
    }

    /// Wait for a restricted process to exit while draining its output pipes,
    /// enforcing the policy CPU-time timeout.
    async fn wait_restricted(
        proc: RestrictedProcess,
        cpu_time_secs: Option<u64>,
    ) -> Result<(i32, Vec<u8>, Vec<u8>), String> {
        let out_file = unsafe { std::fs::File::from_raw_handle(proc.stdout_read.0) };
        let err_file = unsafe { std::fs::File::from_raw_handle(proc.stderr_read.0) };
        let mut out_reader = tokio::fs::File::from_std(out_file);
        let mut err_reader = tokio::fs::File::from_std(err_file);
        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();

        let read_out = tokio::spawn(async move {
            let _ = tokio::io::AsyncReadExt::read_to_end(&mut out_reader, &mut out_buf).await;
            out_buf
        });
        let read_err = tokio::spawn(async move {
            let _ = tokio::io::AsyncReadExt::read_to_end(&mut err_reader, &mut err_buf).await;
            err_buf
        });

        let timeout_dur =
            std::time::Duration::from_secs(cpu_time_secs.unwrap_or(300).max(1));
        let start = std::time::Instant::now();
        let mut exit_code: i32 = -1;
        loop {
            let code = unsafe {
                let mut code: u32 = 0;
                let _ = GetExitCodeProcess(proc.process, &mut code);
                code
            };
            if code != STILL_ACTIVE.0 as u32 {
                exit_code = code as i32;
                break;
            }
            if start.elapsed() >= timeout_dur {
                unsafe {
                    let _ = TerminateProcess(proc.process, 1);
                }
                exit_code = -1;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        let (out_res, err_res) = tokio::join!(read_out, read_err);
        unsafe {
            let _ = CloseHandle(proc.process);
            let _ = CloseHandle(proc.thread);
        }

        Ok((exit_code, out_res.unwrap_or_default(), err_res.unwrap_or_default()))
    }

    /// Build a quoted Windows command line from the program and arguments.
    fn build_command_line(command: &str, args: &[&str]) -> String {
        let mut line = format!("\"{}\"", command.replace('"', "\\\""));
        for a in args {
            if a.is_empty() || a.contains([' ', '\t', '"']) {
                line.push_str(&format!(" \"{}\"", a.replace('"', "\\\"")));
            } else {
                line.push_str(&format!(" {a}"));
            }
        }
        line
    }

    /// Build a double-null-terminated Windows environment block.
    fn build_environment_block(env: &HashMap<String, String>) -> Vec<u16> {
        let mut block = Vec::new();
        let mut pairs: Vec<(&String, &String)> = env.iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));
        for (k, v) in pairs {
            block.extend(k.encode_utf16());
            block.push(b'=' as u16);
            block.extend(v.encode_utf16());
            block.push(0);
        }
        block.push(0);
        block
    }

    /// Resolve a bare command name to an absolute path using `PATH`.
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

    /// Add (or refresh) a Windows Firewall rule for a program.
    fn upsert_firewall_rule(
        name: &str,
        program: &str,
        direction: windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_RULE_DIRECTION,
        action: windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_ACTION,
    ) -> Result<String, String> {
        use windows::core::BSTR;
        use windows::Win32::Foundation::VARIANT_TRUE;
        use windows::Win32::NetworkManagement::WindowsFirewall::{
            INetFwPolicy2, INetFwRule, NET_FW_PROFILE2_ALL,
        };
        use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let policy2: INetFwPolicy2 = CoCreateInstance(&CLSID_NET_FW_POLICY2, None, CLSCTX_ALL)
                .map_err(|e| format!("CoCreateInstance(NetFwPolicy2): {e}"))?;
            let rules = policy2
                .Rules()
                .map_err(|e| format!("INetFwPolicy2::Rules: {e}"))?;

            let rule: INetFwRule = CoCreateInstance(&CLSID_NET_FW_RULE, None, CLSCTX_ALL)
                .map_err(|e| format!("CoCreateInstance(NetFwRule): {e}"))?;
            rule.SetName(&BSTR::from(name))
                .map_err(|e| format!("INetFwRule::Name: {e}"))?;
            rule.SetApplicationName(&BSTR::from(program))
                .map_err(|e| format!("INetFwRule::ApplicationName: {e}"))?;
            rule.SetDirection(direction)
                .map_err(|e| format!("INetFwRule::Direction: {e}"))?;
            rule.SetAction(action)
                .map_err(|e| format!("INetFwRule::Action: {e}"))?;
            rule.SetEnabled(VARIANT_TRUE)
                .map_err(|e| format!("INetFwRule::Enabled: {e}"))?;
            rule.SetProfiles(NET_FW_PROFILE2_ALL.0)
                .map_err(|e| format!("INetFwRule::Profiles: {e}"))?;
            rule.SetInterfaceTypes(&BSTR::from("All")).ok();

            rules.Add(&rule)
                .map_err(|e| format!("INetFwRules::Add: {e}"))?;
            Ok(name.to_string())
        }
    }

    /// Remove a Windows Firewall rule by name.
    fn remove_firewall_rule(name: &str) -> Result<(), String> {
        use windows::core::BSTR;
        use windows::Win32::NetworkManagement::WindowsFirewall::INetFwPolicy2;
        use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_ALL};

        unsafe {
            let policy2: INetFwPolicy2 = CoCreateInstance(&CLSID_NET_FW_POLICY2, None, CLSCTX_ALL)
                .map_err(|e| format!("CoCreateInstance(NetFwPolicy2): {e}"))?;
            let rules = policy2
                .Rules()
                .map_err(|e| format!("INetFwPolicy2::Rules: {e}"))?;
            rules.Remove(&BSTR::from(name))
                .map_err(|e| format!("INetFwRules::Remove: {e}"))?;
            Ok(())
        }
    }

    fn debug_network_rule_removal_failed(name: &str, error: &str) {
        warn!("failed to remove network isolation rule '{name}': {error}");
    }
}

/// Windows sandbox backend.
///
/// The struct exists on every platform so callers can reference
/// `WindowsSandbox` unconditionally, but on non-Windows targets `execute`
/// returns an `UnsupportedPlatform` error.
pub struct WindowsSandbox {
    audit_log: Vec<AuditEntry>,
    #[cfg(windows)]
    backend: backend::WindowsBackend,
}

impl WindowsSandbox {
    /// Create a new Windows sandbox backend.
    pub fn new() -> Self {
        Self {
            audit_log: Vec::new(),
            #[cfg(windows)]
            backend: backend::WindowsBackend::new(),
        }
    }

    /// Execute a command in a Windows sandbox.
    pub async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        self.run(command, args, None, None, policy).await
    }

    /// Execute a command in a Windows sandbox with an explicit environment
    /// and working directory.
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

    /// Check whether the Windows sandbox can operate on this host.
    pub fn health_check(&self) -> bool {
        #[cfg(windows)]
        {
            self.backend.available()
        }
        #[cfg(not(windows))]
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
        #[cfg(windows)]
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
                    "windows_execute".to_string()
                } else {
                    "windows_execute_failed".to_string()
                },
                details: format!("command={command}, ok={ok}, duration_ms={duration_ms}"),
            });
            let mut res = result?;
            res.audit_log = self.audit_log.clone();
            Ok(res)
        }
        #[cfg(not(windows))]
        {
            let _ = (&args, &env, &working_dir, &policy);
            self.audit_log.push(AuditEntry {
                timestamp: chrono::Utc::now(),
                action: "windows_execute_unsupported".to_string(),
                details: format!("command={command}"),
            });
            Err("Windows sandbox is not available on this platform".to_string())
        }
    }
}

impl Default for WindowsSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl crate::Sandbox for WindowsSandbox {
    async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        WindowsSandbox::execute(self, command, args, policy).await
    }

    async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        WindowsSandbox::execute_with_env(self, command, args, env, working_dir, policy).await
    }

    fn health_check(&self) -> bool {
        WindowsSandbox::health_check(self)
    }

    fn name(&self) -> &'static str {
        "windows"
    }

    fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }
}
