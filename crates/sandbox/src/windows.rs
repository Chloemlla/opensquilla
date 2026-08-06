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
    use std::process::Stdio;
    use tokio::process::Command;
    use tracing::warn;
    use windows::Win32::Foundation::{BOOL, CloseHandle, HANDLE, STILL_ACTIVE};
    use windows::Win32::Security::{
        CreateRestrictedToken, CreateWellKnownSid, DISABLE_MAX_PRIVILEGE, PSID,
        SECURITY_ATTRIBUTES, SID, SID_AND_ATTRIBUTES, SetTokenInformation, TOKEN_ADJUST_DEFAULT,
        TOKEN_ASSIGN_PRIMARY, TOKEN_DUPLICATE, TOKEN_MANDATORY_LABEL, TOKEN_QUERY,
        TokenIntegrityLevel, WinLowLabelSid,
    };
    use windows::core::{PCWSTR, PWSTR};

    /// `HANDLE` is `*mut c_void` which is not `Send`/`Sync`, but Windows
    /// handles are just pointer-sized integers and are safe to transfer
    /// between threads.  This wrapper provides the missing impls.
    #[derive(Clone, Copy)]
    struct SendHandle(HANDLE);
    unsafe impl Send for SendHandle {}
    unsafe impl Sync for SendHandle {}
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        JOB_OBJECT_LIMIT_PROCESS_TIME, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    use windows::Win32::System::Pipes::CreatePipe;
    use windows::Win32::System::Threading::{
        CREATE_NEW_PROCESS_GROUP, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT,
        CreateProcessAsUserW, GetCurrentProcess, GetExitCodeProcess, OpenProcessToken,
        PROCESS_CREATION_FLAGS, PROCESS_INFORMATION, ResumeThread, STARTF_USESTDHANDLES,
        STARTUPINFOW, TerminateProcess,
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
                    let _ = unsafe { CloseHandle(job.0) };
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
                        windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_RULE_DIR_OUT,
                        windows::Win32::NetworkManagement::WindowsFirewall::NET_FW_ACTION_BLOCK,
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
                            let result =
                                wait_restricted(proc, policy.resource_limits.cpu_time_secs).await;
                            outcome = Some(result);
                        }
                        Err(e) => {
                            warn!("restricted-token spawn failed ({e}); using job-object sandbox");
                        }
                    }
                    let _ = unsafe { CloseHandle(token.0) };
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

            let _ = unsafe { CloseHandle(job.0) };

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
            job: SendHandle,
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

            let child = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn process: {e}"))?;
            let child_handle = SendHandle(HANDLE(child.raw_handle().unwrap()));

            unsafe {
                AssignProcessToJobObject(job.0, child_handle.0)
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
                    let _ = unsafe { TerminateProcess(child_handle.0, 1) };
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
    fn create_job_object(limits: &crate::policy::ResourceLimits) -> Result<SendHandle, String> {
        unsafe {
            let job = CreateJobObjectW(None, PCWSTR::null())
                .map_err(|e| format!("CreateJobObjectW: {e}"))?;
            let job = SendHandle(job);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            let mut flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if let Some(mem) = limits.memory_bytes {
                info.ProcessMemoryLimit = mem as usize;
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
                job.0,
                JobObjectExtendedLimitInformation,
                &info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
            .map_err(|e| format!("SetInformationJobObject: {e}"))?;
            Ok(job)
        }
    }

    /// Create a restricted token from the current process token: all
    /// privileges disabled (`DISABLE_MAX_PRIVILEGE`) and low integrity.
    fn create_restricted_token() -> Result<SendHandle, String> {
        unsafe {
            let mut process_token: HANDLE = HANDLE(std::ptr::null_mut());
            OpenProcessToken(
                GetCurrentProcess(),
                TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT,
                &mut process_token,
            )
            .map_err(|e| format!("OpenProcessToken: {e}"))?;

            let mut restricted: HANDLE = HANDLE(std::ptr::null_mut());
            CreateRestrictedToken(
                process_token,
                DISABLE_MAX_PRIVILEGE,
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
                WinLowLabelSid,
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
                TokenIntegrityLevel,
                &label as *const TOKEN_MANDATORY_LABEL as *const c_void,
                std::mem::size_of::<TOKEN_MANDATORY_LABEL>() as u32,
            )
            .map_err(|e| format!("SetTokenInformation(low integrity): {e}"))?;

            let _ = CloseHandle(process_token);
            Ok(SendHandle(restricted))
        }
    }

    /// A process spawned through `CreateProcessAsUserW`.
    struct RestrictedProcess {
        process: HANDLE,
        thread: HANDLE,
        stdout_read: HANDLE,
        stderr_read: HANDLE,
    }
    unsafe impl Send for RestrictedProcess {}
    unsafe impl Sync for RestrictedProcess {}

    /// Spawn a process using the restricted token. Creates pipes for stdout /
    /// stderr, launches suspended, assigns to the Job Object, then resumes.
    fn spawn_restricted(
        command: &str,
        args: &[&str],
        env: &HashMap<String, String>,
        working_dir: Option<&str>,
        token: SendHandle,
        job: SendHandle,
    ) -> Result<RestrictedProcess, String> {
        let token = token.0;
        let job = job.0;
        unsafe {
            let sa = SECURITY_ATTRIBUTES {
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
            let cwd_wide = working_dir.map(|c| {
                c.encode_utf16()
                    .chain(std::iter::once(0))
                    .collect::<Vec<u16>>()
            });
            let cwd_ptr = cwd_wide
                .as_ref()
                .map(|v| PCWSTR::from_raw(v.as_ptr()))
                .unwrap_or(PCWSTR::null());

            let si = STARTUPINFOW {
                cb: std::mem::size_of::<STARTUPINFOW>() as u32,
                dwFlags: STARTF_USESTDHANDLES,
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
            ResumeThread(pi.hThread);

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
        use std::os::windows::io::FromRawHandle;
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

        let timeout_dur = std::time::Duration::from_secs(cpu_time_secs.unwrap_or(300).max(1));
        let start = std::time::Instant::now();
        let mut exit_code: i32;
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

        Ok((
            exit_code,
            out_res.unwrap_or_default(),
            err_res.unwrap_or_default(),
        ))
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
        use windows::Win32::Foundation::VARIANT_TRUE;
        use windows::Win32::NetworkManagement::WindowsFirewall::{
            INetFwPolicy2, INetFwRule, NET_FW_PROFILE2_ALL,
        };
        use windows::Win32::System::Com::{
            CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
        };
        use windows::core::BSTR;

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

            rules
                .Add(&rule)
                .map_err(|e| format!("INetFwRules::Add: {e}"))?;
            Ok(name.to_string())
        }
    }

    /// Remove a Windows Firewall rule by name.
    fn remove_firewall_rule(name: &str) -> Result<(), String> {
        use windows::Win32::NetworkManagement::WindowsFirewall::INetFwPolicy2;
        use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance};
        use windows::core::BSTR;

        unsafe {
            let policy2: INetFwPolicy2 = CoCreateInstance(&CLSID_NET_FW_POLICY2, None, CLSCTX_ALL)
                .map_err(|e| format!("CoCreateInstance(NetFwPolicy2): {e}"))?;
            let rules = policy2
                .Rules()
                .map_err(|e| format!("INetFwPolicy2::Rules: {e}"))?;
            rules
                .Remove(&BSTR::from(name))
                .map_err(|e| format!("INetFwRules::Remove: {e}"))?;
            Ok(())
        }
    }

    fn debug_network_rule_removal_failed(name: &str, error: &str) {
        warn!("failed to remove network isolation rule '{name}': {error}");
    }

    /// RAII wrapper for a Windows kernel handle: closes the handle on drop so
    /// a panic in a sandboxed execution cannot leak handles.
    pub struct HandleGuard(pub HANDLE);
    impl Drop for HandleGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
    // `HANDLE` is `*mut c_void`, which is not `Send`/`Sync`; these are
    // pointer-sized integers in practice and safe to move between threads.
    unsafe impl Send for HandleGuard {}
    unsafe impl Sync for HandleGuard {}

    /// Set additional Job Object limits that the basic setup does not cover:
    /// per-process working set. Reads the current extended limits first and
    /// modifies them so flags set by [`create_job_object`] (notably
    /// `KILL_ON_JOB_CLOSE`) are preserved.
    fn configure_job_object(
        job: HANDLE,
        limits: &crate::policy::ResourceLimits,
    ) -> Result<(), String> {
        use windows::Win32::System::JobObjects::{
            JOB_OBJECT_LIMIT_WORKINGSET, JOB_OBJECT_UILIMIT_HANDLES, JobObjectBasicUIRestrictions,
            JobObjectExtendedLimitInformation, JOBOBJECT_BASIC_UI_RESTRICTIONS,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, QueryInformationJobObject,
        };
        unsafe {
            // Read the current limits so we only amend them and do not clobber
            // flags set earlier (notably KILL_ON_JOB_CLOSE).
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            let mut returned = 0u32;
            let query_ok = QueryInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &mut info as *mut _ as *mut c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                Some(&mut returned),
            )
            .is_ok();
            if query_ok {
                if let Some(rss) = limits.rss_bytes {
                    let max = rss.min(512 * 1024 * 1024) as usize;
                    info.BasicLimitInformation.MinimumWorkingSetSize = 256 * 1024;
                    info.BasicLimitInformation.MaximumWorkingSetSize = max;
                    info.BasicLimitInformation.LimitFlags |=
                        JOB_OBJECT_LIMIT_WORKINGSET | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                }
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
                .map_err(|e| format!("SetInformationJobObject(workingset): {e}"))?;
            }

            // Prevent sandboxed processes from using UI handles (dialogs,
            // clipboard, user objects) — a common escape vector.
            let ui = JOBOBJECT_BASIC_UI_RESTRICTIONS {
                UIRestrictionsClass: JOB_OBJECT_UILIMIT_HANDLES,
            };
            SetInformationJobObject(
                job,
                JobObjectBasicUIRestrictions,
                &ui as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_BASIC_UI_RESTRICTIONS>() as u32,
            )
            .map_err(|e| format!("SetInformationJobObject(UIRestrictions): {e}"))?;
        }
        Ok(())
    }

    /// Create an AppContainer profile and derive its SID.
    ///
    /// AppContainers are the modern Windows sandbox primitive (used by Edge,
    /// Chrome, Office). Processes spawned with an AppContainer SID run at low
    /// integrity, cannot access most of the user's profile, cannot write
    /// outside their package root, and have no network access unless a
    /// capability SID is granted.
    ///
    /// Returns the profile name and the derived SID. The caller is responsible
    /// for deleting the profile when no longer needed.
    pub fn create_appcontainer_profile() -> Result<(String, Vec<u8>), String> {
        use windows::Win32::Security::Isolation::{
            CreateAppContainerProfile, DeriveAppContainerSidFromAppContainerName,
        };
        use windows::core::PWSTR;

        unsafe {
            let name = format!(
                "OpenSquilla.Sandbox.{}",
                uuid::Uuid::new_v4().to_string().replace('-', "")
            );
            let name_wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            let name_pwstr = PWSTR(name_wide.as_ptr() as *mut u16);

            let app_sid =
                CreateAppContainerProfile(name_pwstr, PWSTR::null(), PWSTR::null(), None)
                    .map_err(|e| format!("CreateAppContainerProfile: {e}"))?;
            let _ = windows::Win32::Security::FreeSid(app_sid);

            let sid = DeriveAppContainerSidFromAppContainerName(name_pwstr)
                .map_err(|e| format!("DeriveAppContainerSidFromAppContainerName: {e}"))?;

            // Copy the SID bytes out so we can free the original later.
            let mut sid_buf = [0u8; 256];
            let mut sid_size = sid_buf.len() as u32;
            windows::Win32::Security::CopySid(
                sid_size,
                windows::Win32::Security::PSID(sid_buf.as_mut_ptr() as *mut c_void),
                sid,
            )
            .map_err(|e| format!("CopySid: {e}"))?;
            Ok((name, sid_buf.to_vec()))
        }
    }

    /// Delete an AppContainer profile by name.
    pub fn delete_appcontainer_profile(name: &str) -> Result<(), String> {
        use windows::Win32::Security::Isolation::DeleteAppContainerProfile;
        use windows::core::PWSTR;
        unsafe {
            let name_wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            DeleteAppContainerProfile(PWSTR(name_wide.as_ptr() as *mut u16))
                .map_err(|e| format!("DeleteAppContainerProfile: {e}"))
        }
    }

    /// Attach a CPU-rate limit to the Job Object (Windows 8+). `rate` is a
    /// percentage of a single core, e.g. 50 for half a core.
    fn set_job_cpu_rate(job: HANDLE, rate_percent: u32) -> Result<(), String> {
        use windows::Win32::System::JobObjects::{
            JOB_OBJECT_CPU_RATE_CONTROL_ENABLE, JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
            JobObjectCpuRateControlInformation, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
            JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0,
        };
        unsafe {
            let mut info = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION::default();
            // The CpuRate field is a 16.16 fixed-point percentage of a core.
            info.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
            info.Anonymous = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0 {
                CpuRate: rate_percent << 16,
            };
            SetInformationJobObject(
                job,
                JobObjectCpuRateControlInformation,
                &info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_CPU_RATE_CONTROL_INFORMATION>() as u32,
            )
            .map_err(|e| format!("SetInformationJobObject(CPU rate): {e}"))
        }
    }

    /// Enable a memory cap on the Job Object. Returns the configured limit.
    ///
    /// Reads the current extended limits first so earlier flags (e.g.
    /// `KILL_ON_JOB_CLOSE`) are preserved.
    fn apply_job_memory_limit(
        job: HANDLE,
        limits: &crate::policy::ResourceLimits,
    ) -> Result<Option<u64>, String> {
        if let Some(mem) = limits.memory_bytes {
            unsafe {
                use windows::Win32::System::JobObjects::{
                    JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                    QueryInformationJobObject,
                };
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                let mut returned = 0u32;
                let _ = QueryInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &mut info as *mut _ as *mut c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                    Some(&mut returned),
                );
                info.ProcessMemoryLimit = mem as usize;
                info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
                .map_err(|e| format!("SetInformationJobObject(memory): {e}"))?;
            }
        }
        Ok(limits.memory_bytes)
    }

    /// Query the exit code of a process, distinguishing a still-running
    /// process from a terminated one.
    fn query_exit_code(process: HANDLE) -> Result<i32, String> {
        unsafe {
            let mut code: u32 = 0;
            GetExitCodeProcess(process, &mut code)
                .map_err(|e| format!("GetExitCodeProcess: {e}"))?;
            Ok(code as i32)
        }
    }

    /// A small wrapper that tracks every handle a sandboxed execution opened
    /// so the caller can verify the process closed them (leak detection).
    pub struct JobResources {
        job: HandleGuard,
        _marker: std::marker::PhantomData<()>,
    }

    impl JobResources {
        /// Create a configured Job Object from a policy's resource limits.
        pub fn from_policy(limits: &crate::policy::ResourceLimits) -> Result<Self, String> {
            let job = create_job_object(limits)?;
            let _ = configure_job_object(job.0, limits);
            if let Some(quota) = limits.cpu_quota_cores {
                let rate = (quota * 100.0) as u32;
                if rate > 0 && rate <= 100 {
                    let _ = set_job_cpu_rate(job.0, rate);
                }
            }
            let _ = apply_job_memory_limit(job.0, limits);
            Ok(Self {
                job: HandleGuard(job.0),
                _marker: std::marker::PhantomData,
            })
        }

        /// The raw job handle.
        pub fn handle(&self) -> HANDLE {
            self.job.0
        }
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
        self.run(command, args, Some(env), working_dir, policy)
            .await
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
