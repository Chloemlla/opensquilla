//! Linux sandbox backend.
//!
//! Two execution strategies are provided:
//!
//! 1. **bubblewrap (`bwrap`)** — the primary path. The `bwrap` binary is
//!    invoked with user/PID/IPC/UTS/mount namespace isolation, a tmpfs root,
//!    read-only host binds, `--cap-drop ALL`, and (at STRICT/LOCKED levels) a
//!    seccomp-BPF filter compiled by `seccompiler` and passed through
//!    `--seccomp`. Resource limits are applied via `setrlimit` in a
//!    `pre_exec` hook so the target inherits them across `exec`.
//!
//! 2. **native (nix + libc)** — a fallback used when `bwrap` is absent. It
//!    forks a child which unshares user/net/IPC/UTS/mount namespaces through
//!    `nix::sched::unshare`, writes uid/gid maps, applies `setrlimit` limits
//!    and `prctl(PR_SET_SECCOMP)` filtering, then `exec`s the target.
//!
//! The native path cannot use the PID namespace (which would require a
//! double-fork) and is inherently weaker than bwrap; it exists to keep the
//! sandbox usable on hosts without bubblewrap.

use crate::policy::{AuditEntry, SandboxPolicy, SandboxResult};
use std::collections::{BTreeMap, HashMap};

/// Execution options shared by both Linux strategies.
///
/// This type is only exercised on Linux; the `allow(dead_code)` attribute is
/// applied to non-Linux builds where the public `execute` methods construct it
/// but the backend that consumes it is compiled out.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct ExecOptions<'a> {
    pub command: &'a str,
    pub args: &'a [&'a str],
    pub env: Option<HashMap<String, String>>,
    pub working_dir: Option<&'a str>,
    pub policy: &'a SandboxPolicy,
}

#[cfg(target_os = "linux")]
mod backend {
    use super::*;
    use crate::policy::SandboxLevel;
    use nix::sched::CloneFlags;
    use nix::sys::wait::WaitStatus;
    use nix::unistd::ForkResult;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use tokio::process::Command;
    use tracing::{info, warn};

    pub struct LinuxSandboxBackend {
        bwrap_path: Option<String>,
    }

    impl LinuxSandboxBackend {
        pub fn new() -> Self {
            Self {
                bwrap_path: which_bwrap(),
            }
        }

        /// Is any sandboxing primitive available?
        pub fn available(&self) -> bool {
            self.bwrap_path.is_some() || userns_accessible()
        }

        pub async fn execute(&self, opts: &ExecOptions<'_>) -> Result<SandboxResult, String> {
            if let Some(bwrap) = &self.bwrap_path {
                // Compile the seccomp filter before spawning so allocation
                // happens in the parent.
                let seccomp_fd = if opts.policy.level >= SandboxLevel::Strict {
                    match compile_seccomp(opts.policy).and_then(|p| create_seccomp_fd(&p)) {
                        Ok(fd) => Some(fd),
                        Err(e) => {
                            warn!("seccomp filter unavailable; continuing without: {e}");
                            None
                        }
                    }
                } else {
                    None
                };
                let raw_fd = seccomp_fd.as_ref().map(|f| f.as_raw_fd());
                let result = self.execute_bwrap(bwrap, opts, raw_fd).await;
                // The OwnedFd is dropped here, after bwrap has consumed it.
                result
            } else {
                self.execute_native(opts).await
            }
        }

        async fn execute_bwrap(
            &self,
            bwrap: &str,
            opts: &ExecOptions<'_>,
            seccomp_fd: Option<RawFd>,
        ) -> Result<SandboxResult, String> {
            let start = std::time::Instant::now();

            let mut args = self.build_bwrap_args(opts.policy, seccomp_fd, opts.working_dir);
            args.push(opts.command.to_string());
            for a in opts.args {
                args.push(a.to_string());
            }

            let env = filter_env(opts.env.as_ref(), opts.policy);
            let limits = RlimitSpec::from_policy(opts.policy);

            let mut cmd = Command::new(bwrap);
            cmd.args(&args)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);
            cmd.envs(&env);
            if let Some(cwd) = opts.working_dir {
                cmd.current_dir(cwd);
            }
            apply_rlimits_pre_exec(&mut cmd, limits)?;

            info!(
                "linux sandbox (bwrap): {} {}",
                opts.command,
                opts.args.join(" ")
            );
            let child = cmd
                .spawn()
                .map_err(|e| format!("failed to spawn bwrap: {e}"))?;
            let output = child
                .wait_with_output()
                .await
                .map_err(|e| format!("process wait: {e}"))?;

            let duration = start.elapsed();
            Ok(SandboxResult {
                exit_code: output.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                duration_ms: duration.as_millis() as u64,
                audit_log: Vec::new(),
            })
        }

        /// Build the `bwrap` argument vector from the policy.
        fn build_bwrap_args(
            &self,
            policy: &SandboxPolicy,
            seccomp_fd: Option<RawFd>,
            working_dir: Option<&str>,
        ) -> Vec<String> {
            let mut args = vec![
                "--unshare-user".to_string(),
                "--unshare-pid".to_string(),
                "--unshare-ipc".to_string(),
                "--unshare-uts".to_string(),
            ];

            // Network isolation: NONE blocks at the namespace level; proxy and
            // host modes share the host network namespace (the proxy enforces
            // allowlisting in proxy mode).
            if policy.network.is_none() {
                args.push("--unshare-net".to_string());
            }

            // Fresh root filesystem.
            args.push("--dir".to_string());
            args.push("/".to_string());
            args.push("--dir".to_string());
            args.push("/usr".to_string());
            args.push("--dir".to_string());
            args.push("/tmp".to_string());

            if !policy.filesystem.blocks("/proc") {
                args.push("--proc".to_string());
                args.push("/proc".to_string());
            }
            if !policy.filesystem.blocks("/dev") {
                args.push("--dev".to_string());
                args.push("/dev".to_string());
            }
            if !policy.filesystem.blocks("/dev/shm") {
                args.push("--tmpfs".to_string());
                args.push("/dev/shm".to_string());
            }
            if !policy.filesystem.blocks("/tmp") {
                args.push("--tmpfs".to_string());
                args.push("/tmp".to_string());
            }
            if !policy.filesystem.blocks("/run") {
                args.push("--tmpfs".to_string());
                args.push("/run".to_string());
            }
            if !policy.filesystem.blocks("/var") {
                args.push("--tmpfs".to_string());
                args.push("/var".to_string());
            }

            // Read-only host system directories (skipped when denied).
            for dir in [
                "/usr",
                "/etc",
                "/lib",
                "/lib64",
                "/bin",
                "/sbin",
                "/opt",
                "/nix",
                "/usr/local",
            ] {
                if policy.filesystem.blocks(dir) {
                    continue;
                }
                args.push("--ro-bind-try".to_string());
                args.push(dir.to_string());
                args.push(dir.to_string());
            }

            // Policy filesystem rules.
            for path in &policy.filesystem.read_allowed {
                if !path.is_empty() {
                    args.push("--ro-bind".to_string());
                    args.push(path.clone());
                    args.push(path.clone());
                }
            }
            for path in &policy.filesystem.write_allowed {
                if !path.is_empty() {
                    args.push("--bind".to_string());
                    args.push(path.clone());
                    args.push(path.clone());
                }
            }

            // Home directory.
            if policy.filesystem.home_readable {
                if let Some(home) = std::env::var_os("HOME") {
                    let home = home.to_string_lossy().to_string();
                    args.push("--ro-bind-try".to_string());
                    args.push(home.clone());
                    args.push(home);
                }
            }

            // A non-writable /tmp is remounted read-only.
            if !policy.filesystem.tmp_writable && !policy.filesystem.blocks("/tmp") {
                args.push("--remount-ro".to_string());
                args.push("/tmp".to_string());
            }

            // Drop every capability.
            args.push("--cap-drop".to_string());
            args.push("ALL".to_string());

            // Working directory.
            if let Some(cwd) = working_dir {
                args.push("--chdir".to_string());
                args.push(cwd.to_string());
            }

            // seccomp filter.
            if let Some(fd) = seccomp_fd {
                args.push("--seccomp".to_string());
                args.push(fd.to_string());
            }

            // Die with the parent process.
            args.push("--die-with-parent".to_string());

            args
        }

        /// Native fallback: fork a child, unshare namespaces via nix, apply
        /// resource limits and seccomp, then exec.
        async fn execute_native(&self, opts: &ExecOptions<'_>) -> Result<SandboxResult, String> {
            let start = std::time::Instant::now();

            let mut out_fds = [0i32; 2];
            let mut err_fds = [0i32; 2];
            let ret_out = unsafe { libc::pipe(out_fds.as_mut_ptr()) };
            let ret_err = unsafe { libc::pipe(err_fds.as_mut_ptr()) };
            if ret_out != 0 || ret_err != 0 {
                return Err(format!("pipe: {}", nix::errno::Errno::last()));
            }
            let (out_read, out_write) = (out_fds[0], out_fds[1]);
            let (err_read, err_write) = (err_fds[0], err_fds[1]);

            // Compile seccomp in the parent; the forked child sees the result.
            let seccomp_prog = if opts.policy.level >= SandboxLevel::Strict {
                match compile_seccomp(opts.policy) {
                    Ok(prog) => Some(prog),
                    Err(e) => {
                        warn!("seccomp compile failed: {e}");
                        None
                    }
                }
            } else {
                None
            };
            let env = filter_env(opts.env.as_ref(), opts.policy);
            let limits = RlimitSpec::from_policy(opts.policy);

            let pid = unsafe { nix::unistd::fork() }.map_err(|e| format!("fork: {e}"))?;

            match pid {
                ForkResult::Child => {
                    unsafe {
                        libc::close(out_read);
                        libc::close(err_read);
                        libc::dup2(out_write, 1);
                        libc::dup2(err_write, 2);
                        libc::close(out_write);
                        libc::close(err_write);
                    }

                    // Point stdin at /dev/null.
                    if let Ok(null) = std::fs::File::open("/dev/null") {
                        unsafe { libc::dup2(null.as_raw_fd(), 0) };
                    }

                    // User namespace first, then write uid/gid maps, then the
                    // remaining namespaces. PID namespaces are omitted because
                    // that would require a double-fork.
                    match nix::sched::unshare(CloneFlags::CLONE_NEWUSER) {
                        Ok(()) => {
                            let _ = write_uid_gid_maps();
                            let _ = nix::sched::unshare(
                                CloneFlags::CLONE_NEWNET
                                    | CloneFlags::CLONE_NEWIPC
                                    | CloneFlags::CLONE_NEWUTS
                                    | CloneFlags::CLONE_NEWNS,
                            );
                            // Best-effort read-only /proc remount.
                            if !opts.policy.filesystem.blocks("/proc") {
                                let _ = nix::mount::mount(
                                    Some("proc"),
                                    "/proc",
                                    Some("proc"),
                                    nix::mount::MsFlags::MS_NOSUID
                                        | nix::mount::MsFlags::MS_NOEXEC
                                        | nix::mount::MsFlags::MS_NODEV,
                                    None::<&str>,
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "sandbox: unshare(CLONE_NEWUSER) failed ({e}); \
                                 continuing with reduced isolation"
                            );
                        }
                    }

                    let _ = apply_rlimits_libc(&limits);
                    if let Some(prog) = &seccomp_prog {
                        let _ = apply_seccomp(prog);
                    }
                    // Landlock LSM filesystem confinement (best-effort; only
                    // applies on x86_64 kernels with Landlock support).
                    let _ = apply_landlock_rules(
                        &opts.policy.filesystem.read_allowed,
                        &opts.policy.filesystem.write_allowed,
                    );
                    if let Some(cwd) = opts.working_dir {
                        let _ = std::env::set_current_dir(cwd);
                    }

                    // Close inherited descriptors so the sandboxed program
                    // cannot reach into the parent's sockets/files.
                    for fd in 3..=255 {
                        unsafe { libc::close(fd) };
                    }

                    exec_command(opts.command, opts.args, &env);
                    // exec_command only returns when exec fails; exit the
                    // child. Diverges, so this arm typechecks against the
                    // parent arm's `Result`.
                    unsafe { libc::_exit(127) }
                }
                ForkResult::Parent { child } => {
                    unsafe {
                        libc::close(out_write);
                        libc::close(err_write);
                    }

                    let out_file = unsafe { std::fs::File::from_raw_fd(out_read) };
                    let err_file = unsafe { std::fs::File::from_raw_fd(err_read) };
                    let mut out_reader = tokio::fs::File::from_std(out_file);
                    let mut err_reader = tokio::fs::File::from_std(err_file);
                    let mut out_buf = Vec::new();
                    let mut err_buf = Vec::new();
                    let (out_res, err_res) = tokio::join!(
                        tokio::io::AsyncReadExt::read_to_end(&mut out_reader, &mut out_buf),
                        tokio::io::AsyncReadExt::read_to_end(&mut err_reader, &mut err_buf),
                    );
                    let _ = (out_res, err_res);

                    let status = nix::sys::wait::waitpid(child, None)
                        .map_err(|e| format!("waitpid: {e}"))?;
                    let exit_code = match status {
                        WaitStatus::Exited(_, code) => code,
                        WaitStatus::Signaled(_, sig, _) => 128 + sig as i32,
                        _ => -1,
                    };

                    Ok(SandboxResult {
                        exit_code,
                        stdout: String::from_utf8_lossy(&out_buf).to_string(),
                        stderr: String::from_utf8_lossy(&err_buf).to_string(),
                        duration_ms: start.elapsed().as_millis() as u64,
                        audit_log: Vec::new(),
                    })
                }
            }
        }
    }

    impl Default for LinuxSandboxBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Resource limit spec derived from a policy. `Copy` so it can be moved
    /// into a `pre_exec` closure.
    #[derive(Clone, Copy)]
    struct RlimitSpec {
        cpu: Option<u64>,
        mem: Option<u64>,
        nproc: Option<u64>,
        fsize: Option<u64>,
        nofile: Option<u64>,
        core: Option<u64>,
    }

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

    /// Intersect an environment map with the policy's allowlist.
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

    /// Apply `setrlimit` limits to the current process.
    fn apply_rlimits_libc(limits: &RlimitSpec) -> Result<(), String> {
        unsafe {
            if let Some(cpu) = limits.cpu {
                let lim = libc::rlimit {
                    rlim_cur: cpu,
                    rlim_max: cpu,
                };
                if libc::setrlimit(libc::RLIMIT_CPU, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_CPU): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            if let Some(mem) = limits.mem {
                let lim = libc::rlimit {
                    rlim_cur: mem,
                    rlim_max: mem,
                };
                if libc::setrlimit(libc::RLIMIT_AS, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_AS): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            if let Some(nproc) = limits.nproc {
                let lim = libc::rlimit {
                    rlim_cur: nproc,
                    rlim_max: nproc,
                };
                if libc::setrlimit(libc::RLIMIT_NPROC, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_NPROC): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            if let Some(fsize) = limits.fsize {
                let lim = libc::rlimit {
                    rlim_cur: fsize,
                    rlim_max: fsize,
                };
                if libc::setrlimit(libc::RLIMIT_FSIZE, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_FSIZE): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            if let Some(nofile) = limits.nofile {
                let lim = libc::rlimit {
                    rlim_cur: nofile,
                    rlim_max: nofile,
                };
                if libc::setrlimit(libc::RLIMIT_NOFILE, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_NOFILE): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            if let Some(core) = limits.core {
                let lim = libc::rlimit {
                    rlim_cur: core,
                    rlim_max: core,
                };
                if libc::setrlimit(libc::RLIMIT_CORE, &lim) != 0 {
                    return Err(format!(
                        "setrlimit(RLIMIT_CORE): {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Attach a `pre_exec` hook that applies resource limits in the child.
    fn apply_rlimits_pre_exec(cmd: &mut Command, limits: RlimitSpec) -> Result<(), String> {
        unsafe {
            cmd.pre_exec(move || apply_rlimits_libc(&limits).map_err(io_err));
        }
        Ok(())
    }

    fn io_err(msg: String) -> std::io::Error {
        std::io::Error::other(msg)
    }

    /// Write the uid/gid maps for a freshly created user namespace so the
    /// caller maps to root (uid 0) inside the namespace.
    fn write_uid_gid_maps() -> Result<(), String> {
        let uid = nix::unistd::getuid().as_raw();
        let gid = nix::unistd::getgid().as_raw();
        let _ = std::fs::write("/proc/self/setgroups", "deny");
        std::fs::write("/proc/self/uid_map", format!("0 {uid} 1"))
            .map_err(|e| format!("uid_map write: {e}"))?;
        std::fs::write("/proc/self/gid_map", format!("0 {gid} 1"))
            .map_err(|e| format!("gid_map write: {e}"))?;
        Ok(())
    }

    /// Compile a seccomp-BPF allowlist filter with `seccompiler`. Deny-by-
    /// default: any syscall not allowlisted returns `EPERM`.
    fn compile_seccomp(_policy: &SandboxPolicy) -> Result<seccompiler::BpfProgram, String> {
        use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, SeccompRule, TargetArch};

        #[cfg(target_arch = "x86_64")]
        let target_arch = TargetArch::x86_64;
        #[cfg(target_arch = "aarch64")]
        let target_arch = TargetArch::aarch64;

        let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
        let syscalls: &[libc::c_long] = &[
            libc::SYS_execve,
            libc::SYS_execveat,
            libc::SYS_exit,
            libc::SYS_exit_group,
            libc::SYS_read,
            libc::SYS_write,
            libc::SYS_close,
            libc::SYS_lseek,
            libc::SYS_mmap,
            libc::SYS_mprotect,
            libc::SYS_munmap,
            libc::SYS_brk,
            libc::SYS_openat,
            libc::SYS_fstat,
            libc::SYS_newfstatat,
            libc::SYS_readlink,
            libc::SYS_readlinkat,
            libc::SYS_getdents64,
            libc::SYS_access,
            libc::SYS_faccessat,
            libc::SYS_getuid,
            libc::SYS_geteuid,
            libc::SYS_getgid,
            libc::SYS_getegid,
            libc::SYS_getpid,
            libc::SYS_getppid,
            libc::SYS_gettid,
            libc::SYS_prctl,
            libc::SYS_set_robust_list,
            libc::SYS_set_tid_address,
            libc::SYS_futex,
            libc::SYS_clock_gettime,
            libc::SYS_gettimeofday,
            libc::SYS_nanosleep,
            libc::SYS_clock_nanosleep,
            libc::SYS_pipe,
            libc::SYS_pipe2,
            libc::SYS_dup,
            libc::SYS_dup3,
            libc::SYS_fcntl,
            libc::SYS_ioctl,
            libc::SYS_uname,
            libc::SYS_getrandom,
            libc::SYS_rt_sigaction,
            libc::SYS_rt_sigprocmask,
            libc::SYS_rt_sigreturn,
            libc::SYS_sigaltstack,
            libc::SYS_fstatfs,
            libc::SYS_statfs,
            libc::SYS_ftruncate,
            libc::SYS_truncate,
            libc::SYS_pwrite64,
            libc::SYS_pread64,
            libc::SYS_writev,
            libc::SYS_readv,
            libc::SYS_poll,
            libc::SYS_ppoll,
            libc::SYS_epoll_create1,
            libc::SYS_epoll_ctl,
            libc::SYS_epoll_wait,
            libc::SYS_eventfd2,
            libc::SYS_timerfd_create,
            libc::SYS_timerfd_settime,
            libc::SYS_getcwd,
            libc::SYS_chdir,
            libc::SYS_fchdir,
            libc::SYS_mkdir,
            libc::SYS_unlink,
            libc::SYS_rmdir,
            libc::SYS_rename,
            libc::SYS_link,
            libc::SYS_symlink,
            libc::SYS_socket,
            libc::SYS_connect,
            libc::SYS_sendto,
            libc::SYS_recvfrom,
            libc::SYS_sendmsg,
            libc::SYS_recvmsg,
            libc::SYS_getsockopt,
            libc::SYS_setsockopt,
            libc::SYS_getpeername,
            libc::SYS_getsockname,
            libc::SYS_shutdown,
            libc::SYS_bind,
            libc::SYS_listen,
            libc::SYS_accept4,
            libc::SYS_socketpair,
            libc::SYS_setsid,
            libc::SYS_setpgid,
            libc::SYS_umask,
            libc::SYS_prlimit64,
            libc::SYS_getrlimit,
            libc::SYS_setrlimit,
            libc::SYS_madvise,
            libc::SYS_sched_yield,
            libc::SYS_tgkill,
            libc::SYS_wait4,
            libc::SYS_clone,
            libc::SYS_fchmodat,
            libc::SYS_utimensat,
            libc::SYS_getrusage,
            libc::SYS_times,
            libc::SYS_sysinfo,
        ];
        for s in syscalls {
            if let Ok(rule) = SeccompRule::new(vec![]) {
                rules.insert(*s, vec![rule]);
            }
        }

        #[cfg(target_arch = "x86_64")]
        if let Ok(rule) = SeccompRule::new(vec![]) {
            rules.insert(libc::SYS_arch_prctl, vec![rule]);
        }

        let filter = SeccompFilter::new(
            rules,
            SeccompAction::Errno(1),
            SeccompAction::Allow,
            target_arch,
        )
        .map_err(|e| format!("seccomp filter build: {e}"))?;
        BpfProgram::try_from(filter).map_err(|e| format!("seccomp bpf compile: {e}"))
    }

    /// Size in bytes of the `struct sock_fprog` header for the host
    /// architecture. `{ u16 len; padding; void *filter }`.
    fn sock_fprog_size() -> usize {
        let ptr_size = std::mem::size_of::<usize>();
        align_up(2, ptr_size) + ptr_size
    }

    fn align_up(value: usize, align: usize) -> usize {
        (value + align - 1) & !(align - 1)
    }

    /// Serialize a `BpfProgram` into the on-disk `sock_fprog` layout that
    /// `bwrap --seccomp` and `prctl(PR_SET_SECCOMP)` expect. The filter
    /// pointer field is left zeroed; the `prctl` path patches it to the
    /// actual buffer address before calling.
    fn serialize_sock_fprog(prog: &seccompiler::BpfProgram) -> Vec<u8> {
        let ptr_size = std::mem::size_of::<usize>();
        let filter_offset = align_up(2, ptr_size);
        let header_size = sock_fprog_size();

        let mut insns = Vec::with_capacity(prog.len() * 8);
        for insn in prog {
            insns.extend_from_slice(&insn.code.to_le_bytes());
            insns.push(insn.jt);
            insns.push(insn.jf);
            insns.extend_from_slice(&insn.k.to_le_bytes());
        }

        let mut buf = Vec::with_capacity(header_size + insns.len());
        buf.extend_from_slice(&(prog.len() as u16).to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, filter_offset - 2));
        buf.extend(std::iter::repeat_n(0u8, ptr_size));
        buf.extend_from_slice(&insns);
        buf
    }

    /// Create a memfd containing the serialized seccomp filter for bwrap.
    ///
    /// The memfd intentionally lacks `MFD_CLOEXEC` so it survives the `exec`
    /// into bwrap; it is dropped once bwrap has read it.
    fn create_seccomp_fd(prog: &seccompiler::BpfProgram) -> Result<OwnedFd, String> {
        use nix::sys::memfd::{MemFdCreateFlag, memfd_create};
        use nix::unistd::{Whence, lseek, write};

        let serialized = serialize_sock_fprog(prog);
        let fd = memfd_create(c"osq-seccomp", MemFdCreateFlag::empty())
            .map_err(|e| format!("memfd_create: {e}"))?;
        let mut written = 0usize;
        while written < serialized.len() {
            let n = write(&fd, &serialized[written..]).map_err(|e| format!("memfd write: {e}"))?;
            written += n;
        }
        lseek(fd.as_raw_fd(), 0, Whence::SeekSet).map_err(|e| format!("memfd seek: {e}"))?;
        Ok(fd)
    }

    /// Install a compiled seccomp filter on the current process.
    fn apply_seccomp(prog: &seccompiler::BpfProgram) -> Result<(), String> {
        let header_size = sock_fprog_size();
        let filter_offset = header_size - std::mem::size_of::<usize>();
        let mut serialized = serialize_sock_fprog(prog);

        // Patch the filter pointer to point into this buffer (absolute address).
        let addr = serialized.as_mut_ptr().wrapping_add(header_size) as usize;
        let bytes = addr.to_ne_bytes();
        serialized[filter_offset..filter_offset + bytes.len()].copy_from_slice(&bytes);

        let ret = unsafe {
            libc::prctl(
                libc::PR_SET_SECCOMP,
                libc::SECCOMP_MODE_FILTER,
                serialized.as_ptr() as *const libc::c_void,
            )
        };
        if ret != 0 {
            return Err(format!(
                "prctl(PR_SET_SECCOMP) failed: {}",
                nix::errno::Errno::last()
            ));
        }
        Ok(())
    }

    /// Exec the target program with a filtered environment.
    fn exec_command(command: &str, args: &[&str], env: &HashMap<String, String>) {
        use std::ffi::CString;

        let program = CString::new(command).unwrap_or_default();
        let cargs: Vec<CString> = args.iter().filter_map(|a| CString::new(*a).ok()).collect();

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
            libc::execvpe(program.as_ptr(), arg_ptrs.as_ptr(), env_ptrs.as_ptr());
            // Fall back to PATH lookup with the inherited environment.
            libc::execvp(program.as_ptr(), arg_ptrs.as_ptr());
        }
    }

    /// Can unprivileged user namespaces be created on this kernel?
    fn userns_accessible() -> bool {
        if let Ok(v) = std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone") {
            if let Ok(n) = v.trim().parse::<u32>() {
                return n != 0;
            }
        }
        if let Ok(v) = std::fs::read_to_string("/proc/sys/user/max_user_namespaces") {
            if let Ok(n) = v.trim().parse::<u32>() {
                return n != 0;
            }
        }
        true
    }

    /// Join an existing namespace by fd (`setns`). Useful when the sandbox
    /// needs to re-enter a pre-created container namespace.
    #[allow(dead_code)]
    fn setns(ns_fd: RawFd, nstype: CloneFlags) -> Result<(), String> {
        let ret = unsafe { libc::setns(ns_fd, nstype.bits()) };
        if ret != 0 {
            return Err(format!("setns: {}", nix::errno::Errno::last()));
        }
        Ok(())
    }

    /// cgroup v2 controller for CPU, memory and process-count limits.
    ///
    /// Creates a dedicated child cgroup under `/sys/fs/cgroup/opensquilla/`
    /// (creating the parent on first use), writes the controller limits, and
    /// exposes the process PID to be moved into it before the sandboxed
    /// program starts. Dropping the controller removes the child cgroup.
    ///
    /// cgroup v2 is read from `/sys/fs/cgroup/cgroup.controllers`; if it is
    /// not present the controller reports `available() == false` and callers
    /// should fall back to `setrlimit`.
    #[allow(dead_code)]
    pub struct CgroupV2Controller {
        path: PathBuf,
    }

    #[allow(dead_code)]
    impl CgroupV2Controller {
        /// The base directory under which per-sandbox cgroups are created.
        pub fn base_dir() -> PathBuf {
            PathBuf::from("/sys/fs/cgroup/opensquilla")
        }

        /// Is cgroup v2 available and mounted?
        pub fn available() -> bool {
            let controllers = Self::base_dir().join("cgroup.controllers");
            // The parent may not exist yet; check the v2 root marker instead.
            Path::new("/sys/fs/cgroup/cgroup.controllers").exists() || controllers.exists()
        }

        /// Create a new controller with limits from a policy.
        pub fn create(limits: &crate::policy::ResourceLimits) -> Result<Self, String> {
            let base = Self::base_dir();
            if !base.exists() {
                std::fs::create_dir_all(&base).map_err(|e| format!("cgroup mkdir: {e}"))?;
                // Enable controllers the kernel allows.
                if let Ok(controllers) =
                    std::fs::read_to_string("/sys/fs/cgroup/cgroup.subtree_control")
                {
                    let mut enabled = String::new();
                    for c in ["cpu", "memory", "pids"] {
                        if controllers.contains(c) {
                            enabled.push_str(c);
                            enabled.push(' ');
                        }
                    }
                    let _ = std::fs::write(base.join("cgroup.subtree_control"), enabled.trim());
                }
            }
            let name = format!("sbx_{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
            let path = base.join(&name);
            std::fs::create_dir(&path).map_err(|e| format!("cgroup create: {e}"))?;

            let controller = Self { path };
            controller.apply_limits(limits)?;
            Ok(controller)
        }

        fn apply_limits(&self, limits: &crate::policy::ResourceLimits) -> Result<(), String> {
            // cpu.max is "quota period"; quota of 100000 us == one core.
            let period_us = 100_000u64;
            let quota_us = match limits.cpu_quota_cores {
                Some(q) => ((q * period_us as f64) as u64).max(1),
                None => period_us,
            };
            self.write("cpu.max", &format!("{quota_us} {period_us}"))?;

            if let Some(mem) = limits.memory_bytes {
                self.write("memory.max", &mem.to_string())?;
            } else {
                self.write("memory.max", "max")?;
            }

            if let Some(nproc) = limits.max_processes {
                self.write("pids.max", &nproc.to_string())?;
            } else {
                self.write("pids.max", "max")?;
            }

            // Disable swap so the memory limit is hard.
            let _ = self.write("memory.swap.max", "0");
            Ok(())
        }

        fn write(&self, file: &str, value: &str) -> Result<(), String> {
            let path = self.path.join(file);
            std::fs::write(&path, value).map_err(|e| format!("cgroup write {}: {e}", file))
        }

        /// Move a process into this cgroup by PID.
        pub fn attach(&self, pid: i32) -> Result<(), String> {
            let path = self.path.join("cgroup.procs");
            std::fs::write(&path, pid.to_string()).map_err(|e| format!("cgroup attach: {e}"))
        }

        /// The cgroup directory path.
        pub fn path(&self) -> &Path {
            &self.path
        }

        /// The leaf name of the cgroup.
        pub fn name(&self) -> &str {
            self.path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("sbx")
        }
    }

    impl Drop for CgroupV2Controller {
        fn drop(&mut self) {
            // Freeze and release the cgroup; best-effort.
            let _ = std::fs::write(self.path.join("cgroup.freeze"), "1");
            let _ = std::fs::remove_dir(&self.path);
        }
    }

    /// Landlock LSM rules for the native fallback.
    ///
    /// Landlock (Linux 5.13+) lets an unprivileged process restrict its own
    /// filesystem access with no root. This module builds a ruleset that
    /// denies writes outside the allowed paths and denies reads outside the
    /// allowed read paths, then restricts the calling process. Must be called
    /// in the forked child before `exec`.
    pub fn apply_landlock_rules(
        read_allowed: &[String],
        write_allowed: &[String],
    ) -> Result<(), String> {
        // Syscall numbers are only stable on x86_64; fall back gracefully
        // elsewhere.
        #[cfg(target_arch = "x86_64")]
        {
            const LANDLOCK_CREATE_RULESET_VERSION: usize = 0x0000_0001;
            const LANDLOCK_ACCESS_FS_EXECUTE: u64 = 1 << 0;
            const LANDLOCK_ACCESS_FS_WRITE_FILE: u64 = 1 << 1;
            const LANDLOCK_ACCESS_FS_READ_FILE: u64 = 1 << 2;
            const LANDLOCK_ACCESS_FS_READ_DIR: u64 = 1 << 3;
            const LANDLOCK_ACCESS_FS_REMOVE_DIR: u64 = 1 << 4;
            const LANDLOCK_ACCESS_FS_REMOVE_FILE: u64 = 1 << 5;
            const LANDLOCK_ACCESS_FS_MAKE_CHAR: u64 = 1 << 6;
            const LANDLOCK_ACCESS_FS_MAKE_DIR: u64 = 1 << 7;
            const LANDLOCK_ACCESS_FS_MAKE_REG: u64 = 1 << 8;
            const LANDLOCK_ACCESS_FS_MAKE_SOCK: u64 = 1 << 9;
            const LANDLOCK_ACCESS_FS_MAKE_FIFO: u64 = 1 << 10;
            const LANDLOCK_ACCESS_FS_MAKE_BLOCK: u64 = 1 << 11;
            const LANDLOCK_ACCESS_FS_MAKE_SYM: u64 = 1 << 12;
            const LANDLOCK_ACCESS_FS_REFER: u64 = 1 << 13;
            const LANDLOCK_ACCESS_FS_TRUNCATE: u64 = 1 << 14;

            #[repr(C)]
            #[derive(Default, Clone, Copy)]
            struct LandlockRulesetAttr {
                handled_access_fs: u64,
            }
            #[repr(C)]
            #[derive(Default, Clone, Copy)]
            struct LandlockPathBeneathAttr {
                allowed_access: u64,
                parent_fd: i32,
            }

            unsafe fn syscall3(nr: libc::c_long, a1: usize, a2: usize, a3: usize) -> libc::c_long {
                unsafe { libc::syscall(nr, a1, a2, a3) }
            }

            unsafe fn syscall4(
                nr: libc::c_long,
                a1: usize,
                a2: usize,
                a3: usize,
                a4: usize,
            ) -> libc::c_long {
                unsafe { libc::syscall(nr, a1, a2, a3, a4) }
            }

            unsafe {
                // Query the ABI version. If the kernel lacks Landlock, this
                // returns -1 with EOPNOTSUPP/ENOSYS and we degrade silently.
                let version = syscall3(
                    libc::SYS_landlock_create_ruleset,
                    LANDLOCK_CREATE_RULESET_VERSION,
                    0,
                    std::ptr::null::<u8>() as usize,
                );
                if version < 0 {
                    return Ok(());
                }

                let handled = LANDLOCK_ACCESS_FS_EXECUTE
                    | LANDLOCK_ACCESS_FS_WRITE_FILE
                    | LANDLOCK_ACCESS_FS_READ_FILE
                    | LANDLOCK_ACCESS_FS_READ_DIR
                    | LANDLOCK_ACCESS_FS_REMOVE_DIR
                    | LANDLOCK_ACCESS_FS_REMOVE_FILE
                    | LANDLOCK_ACCESS_FS_MAKE_CHAR
                    | LANDLOCK_ACCESS_FS_MAKE_DIR
                    | LANDLOCK_ACCESS_FS_MAKE_REG
                    | LANDLOCK_ACCESS_FS_MAKE_SOCK
                    | LANDLOCK_ACCESS_FS_MAKE_FIFO
                    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
                    | LANDLOCK_ACCESS_FS_MAKE_SYM
                    | LANDLOCK_ACCESS_FS_REFER
                    | LANDLOCK_ACCESS_FS_TRUNCATE;

                let attr = LandlockRulesetAttr {
                    handled_access_fs: handled,
                };
                let ruleset_fd = syscall3(
                    libc::SYS_landlock_create_ruleset,
                    0,
                    &attr as *const LandlockRulesetAttr as usize,
                    std::mem::size_of::<LandlockRulesetAttr>(),
                );
                if ruleset_fd < 0 {
                    return Err(format!(
                        "landlock_create_ruleset: {}",
                        nix::errno::Errno::last()
                    ));
                }

                // Allow-read access: read file + dir + execute + truncate (so
                // programs can execute binaries from read paths).
                let read_access = LANDLOCK_ACCESS_FS_READ_FILE
                    | LANDLOCK_ACCESS_FS_READ_DIR
                    | LANDLOCK_ACCESS_FS_EXECUTE;
                let write_access = LANDLOCK_ACCESS_FS_WRITE_FILE
                    | LANDLOCK_ACCESS_FS_REMOVE_DIR
                    | LANDLOCK_ACCESS_FS_REMOVE_FILE
                    | LANDLOCK_ACCESS_FS_MAKE_CHAR
                    | LANDLOCK_ACCESS_FS_MAKE_DIR
                    | LANDLOCK_ACCESS_FS_MAKE_REG
                    | LANDLOCK_ACCESS_FS_MAKE_SOCK
                    | LANDLOCK_ACCESS_FS_MAKE_FIFO
                    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
                    | LANDLOCK_ACCESS_FS_MAKE_SYM
                    | LANDLOCK_ACCESS_FS_TRUNCATE;

                let add_rule = |path: &Path, access: u64| -> Result<(), String> {
                    // OsStrExt::as_ptr was removed from std; build a NUL-terminated
                    // C string from the platform-encoded bytes instead.
                    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
                        .map_err(|e| {
                            format!(
                                "landlock add_rule({}): path has interior NUL: {e}",
                                path.display()
                            )
                        })?;
                    let fd = libc::open(c_path.as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
                    if fd < 0 {
                        return Ok(()); // missing path: skip
                    }
                    let beneath = LandlockPathBeneathAttr {
                        allowed_access: access,
                        parent_fd: fd,
                    };
                    let ret = syscall4(
                        libc::SYS_landlock_add_rule,
                        ruleset_fd as usize,
                        0x1, // LANDLOCK_RULE_PATH_BENEATH
                        &beneath as *const LandlockPathBeneathAttr as usize,
                        0,
                    );
                    libc::close(fd);
                    if ret != 0 {
                        return Err(format!(
                            "landlock_add_rule({}): {}",
                            path.display(),
                            nix::errno::Errno::last()
                        ));
                    }
                    Ok(())
                };

                // Allow read on system directories + explicit read paths.
                for sysdir in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/opt", "/etc"] {
                    let _ = add_rule(Path::new(sysdir), read_access);
                }
                for p in read_allowed {
                    let _ = add_rule(Path::new(p), read_access);
                }
                for p in write_allowed {
                    let _ = add_rule(Path::new(p), read_access | write_access);
                }

                // Restrict the current process.
                let ret = syscall3(libc::SYS_landlock_restrict_self, ruleset_fd as usize, 0, 0);
                libc::close(ruleset_fd as i32);
                if ret != 0 {
                    return Err(format!(
                        "landlock_restrict_self: {}",
                        nix::errno::Errno::last()
                    ));
                }
            }
            Ok(())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (read_allowed, write_allowed);
            Ok(())
        }
    }

    /// Apply cgroup limits to the current process (used by the native
    /// fallback when a cgroup controller is available).
    #[allow(dead_code)]
    pub fn apply_cgroup_for_self(
        limits: &crate::policy::ResourceLimits,
    ) -> Option<CgroupV2Controller> {
        if !CgroupV2Controller::available() {
            return None;
        }
        match CgroupV2Controller::create(limits) {
            Ok(c) => {
                let pid = nix::unistd::getpid().as_raw();
                let _ = c.attach(pid);
                Some(c)
            }
            Err(_) => None,
        }
    }
}

#[cfg(target_os = "linux")]
fn which_bwrap() -> Option<String> {
    std::env::var_os("PATH").and_then(|paths| {
        for path in std::env::split_paths(&paths) {
            let candidate = path.join("bwrap");
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().to_string());
            }
        }
        None
    })
}

/// Linux sandbox backend.
///
/// The struct exists on every platform so callers can reference
/// `LinuxSandbox` unconditionally, but on non-Linux targets `execute` returns
/// an `UnsupportedPlatform` error.
pub struct LinuxSandbox {
    audit_log: Vec<AuditEntry>,
    #[cfg(target_os = "linux")]
    backend: backend::LinuxSandboxBackend,
}

impl LinuxSandbox {
    /// Create a new Linux sandbox backend.
    pub fn new() -> Self {
        Self {
            audit_log: Vec::new(),
            #[cfg(target_os = "linux")]
            backend: backend::LinuxSandboxBackend::new(),
        }
    }

    /// Execute a command inside a bubblewrap sandbox.
    pub async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        let opts = ExecOptions {
            command,
            args,
            env: None,
            working_dir: None,
            policy,
        };
        self.run(opts).await
    }

    /// Execute a command inside a bubblewrap sandbox with an explicit
    /// environment and working directory.
    pub async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        let opts = ExecOptions {
            command,
            args,
            env: Some(env),
            working_dir,
            policy,
        };
        self.run(opts).await
    }

    /// Check whether the Linux sandbox can operate on this host.
    pub fn health_check(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.backend.available()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    /// Snapshot of recorded audit entries.
    pub fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }

    async fn run(&mut self, opts: ExecOptions<'_>) -> Result<SandboxResult, String> {
        #[cfg(target_os = "linux")]
        {
            let result = self.backend.execute(&opts).await;
            let (ok, duration_ms) = match &result {
                Ok(r) => (true, r.duration_ms),
                Err(_) => (false, 0),
            };
            self.audit_log.push(AuditEntry {
                timestamp: chrono::Utc::now(),
                action: if ok {
                    "linux_execute".to_string()
                } else {
                    "linux_execute_failed".to_string()
                },
                details: format!(
                    "command={}, ok={ok}, duration_ms={duration_ms}",
                    opts.command
                ),
            });
            let mut res = result?;
            res.audit_log = self.audit_log.clone();
            Ok(res)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = &opts;
            self.audit_log.push(AuditEntry {
                timestamp: chrono::Utc::now(),
                action: "linux_execute_unsupported".to_string(),
                details: format!("command={}", opts.command),
            });
            Err("Linux sandbox is not available on this platform".to_string())
        }
    }
}

impl Default for LinuxSandbox {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl crate::Sandbox for LinuxSandbox {
    async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        LinuxSandbox::execute(self, command, args, policy).await
    }

    async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String> {
        LinuxSandbox::execute_with_env(self, command, args, env, working_dir, policy).await
    }

    fn health_check(&self) -> bool {
        LinuxSandbox::health_check(self)
    }

    fn name(&self) -> &'static str {
        "linux"
    }

    fn audit_log(&self) -> Vec<AuditEntry> {
        self.audit_log.clone()
    }
}
