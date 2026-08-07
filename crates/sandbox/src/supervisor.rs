//! Process supervision and resource monitoring.
//!
//! Provides a cross-platform supervisor that spawns a sandboxed child, enforces
//! wall-clock and CPU timeouts, monitors resource usage, and produces a
//! [`crate::metrics::ExecutionMetrics`] snapshot for the backends.
//!
//! The supervisor is deliberately backend-agnostic: it takes a command, args,
//! environment, working directory and policy, and returns a
//! [`SupervisedRun`] containing the exit status, captured output and metrics.
//! The platform backends use it to avoid duplicating timeout and metrics logic.

use std::collections::HashMap;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::process::{Child, Command};
use tracing::debug;

use crate::metrics::{ExecutionMetrics, Rusage, read_self_rusage};
use crate::policy::SandboxPolicy;

/// A completed supervised run.
#[derive(Debug, Clone)]
pub struct SupervisedRun {
    /// Exit code of the process (`-1` when killed by timeout).
    pub exit_code: i32,
    /// Captured stdout.
    pub stdout: String,
    /// Captured stderr.
    pub stderr: String,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Whether the run was killed by the supervisor's timeout.
    pub timed_out: bool,
    /// The resource-usage delta for the child.
    pub rusage: Option<Rusage>,
    /// Number of policy denials observed (best-effort; backends may populate).
    pub policy_denials: u64,
    /// Number of network requests attempted (best-effort).
    pub network_requests: u64,
    /// Number of network requests blocked (best-effort).
    pub network_blocked: u64,
}

impl SupervisedRun {
    /// Convert into an [`ExecutionMetrics`] snapshot.
    pub fn into_metrics(&self, execution_id: &str, backend: &str, level: &str) -> ExecutionMetrics {
        let mut m = ExecutionMetrics::new(execution_id, backend)
            .with_duration(self.duration)
            .with_exit(self.exit_code, self.timed_out)
            .with_level(level);
        if let Some(r) = &self.rusage {
            r.apply_to(&mut m);
        }
        m.policy_denials = self.policy_denials;
        m.network_requests = self.network_requests;
        m.network_blocked = self.network_blocked;
        m
    }

    /// Whether this failed run was likely denied by the sandbox, for a given
    /// backend name.
    ///
    /// Convenience bridge to
    /// [`crate::denial_attribution::is_likely_sandbox_denied`]; the noop/host
    /// backends are never attributed.
    pub fn likely_sandbox_denied(&self, backend: &str) -> bool {
        crate::denial_attribution::is_likely_sandbox_denied(
            &crate::denial_attribution::SandboxRunOutcome::new(
                self.exit_code,
                &self.stdout,
                &self.stderr,
                backend,
            ),
        )
    }
}

/// Options for a supervised spawn.
#[derive(Debug, Clone)]
pub struct SpawnOptions<'a> {
    pub command: &'a str,
    pub args: &'a [&'a str],
    pub env: Option<HashMap<String, String>>,
    pub working_dir: Option<&'a str>,
    pub policy: &'a SandboxPolicy,
    /// Extra creation flags passed through to the platform spawn (e.g. process
    /// group flags). Backend-specific; `None` uses the backend default.
    #[cfg(windows)]
    pub creation_flags: Option<u32>,
    /// Kill the process group on timeout (Unix). Defaults to `true`.
    pub kill_process_group_on_timeout: bool,
}

/// A running supervised child.
pub struct SupervisedChild {
    child: Child,
    child_id: Option<u32>,
    start: Instant,
    timeout: Duration,
    kill_group_on_timeout: bool,
    rusage_before: Option<Rusage>,
}

impl SupervisedChild {
    /// Wait for the child to finish, enforcing the configured timeout and
    /// capturing output. The child must have been spawned with piped stdout /
    /// stderr.
    pub async fn wait(self) -> Result<SupervisedRun, String> {
        let rusage_before = self.rusage_before;
        let start = self.start;
        let child_id = self.child_id;
        let kill_group = self.kill_group_on_timeout;
        let timed_out;

        let output = match tokio::time::timeout(self.timeout, self.child.wait_with_output()).await {
            Ok(Ok(o)) => {
                timed_out = false;
                o
            }
            Ok(Err(e)) => return Err(format!("process wait: {e}")),
            Err(_) => {
                timed_out = true;
                debug!(
                    "supervisor: killing child after {}ms timeout",
                    self.timeout.as_millis()
                );
                kill_child(child_id, kill_group).await;
                let duration = start.elapsed();
                let rusage_after = read_self_rusage();
                let rusage = match (rusage_before.as_ref(), rusage_after.as_ref()) {
                    (Some(b), Some(a)) => Some(Rusage::delta(b, a)),
                    _ => None,
                };
                return Ok(SupervisedRun {
                    exit_code: -1,
                    stdout: String::new(),
                    stderr: format!("Process timed out after {}ms", self.timeout.as_millis()),
                    duration,
                    timed_out: true,
                    rusage,
                    policy_denials: 0,
                    network_requests: 0,
                    network_blocked: 0,
                });
            }
        };

        let duration = start.elapsed();
        let rusage_after = read_self_rusage();
        let rusage = match (rusage_before.as_ref(), rusage_after.as_ref()) {
            (Some(b), Some(a)) => Some(Rusage::delta(b, a)),
            _ => None,
        };

        Ok(SupervisedRun {
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration,
            timed_out,
            rusage,
            policy_denials: 0,
            network_requests: 0,
            network_blocked: 0,
        })
    }
}

/// Spawn a child process with the given options, returning a [`SupervisedChild`].
///
/// The child is spawned with piped stdout/stderr and null stdin. On Unix, a
/// new process group is created so a timeout can kill the whole group; on
/// Windows, a new process group is requested via creation flags. The policy's
/// resource limits are applied in a `pre_exec` hook on Unix (see
/// [`apply_rlimits_pre_exec`]).
pub fn spawn_supervised(opts: &SpawnOptions<'_>) -> Result<SupervisedChild, String> {
    let mut cmd = Command::new(opts.command);
    cmd.args(opts.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Environment filtering is the caller's responsibility (backends differ in
    // how they intersect with the allowlist); we simply pass what we're given.
    if let Some(env) = &opts.env {
        cmd.envs(env);
    }
    if let Some(cwd) = opts.working_dir {
        cmd.current_dir(cwd);
    }

    configure_process_group(&mut cmd, opts)?;

    let child = cmd
        .spawn()
        .map_err(|e| format!("supervisor: spawn '{}' failed: {e}", opts.command))?;
    let child_id = child_id(&child);

    let timeout = Duration::from_secs(opts.policy.resource_limits.effective_timeout_secs());
    let rusage_before = read_self_rusage();

    Ok(SupervisedChild {
        child,
        child_id,
        start: Instant::now(),
        timeout,
        kill_group_on_timeout: opts.kill_process_group_on_timeout,
        rusage_before,
    })
}

/// Configure process-group creation and resource limits on the command.
fn configure_process_group(cmd: &mut Command, opts: &SpawnOptions<'_>) -> Result<(), String> {
    #[cfg(unix)]
    {
        let limits = RlimitSpec::from_policy(opts.policy);
        unsafe {
            cmd.pre_exec(move || {
                // New session so the child leads its own process group.
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
        let flags = opts
            .creation_flags
            .unwrap_or(CREATE_NEW_PROCESS_GROUP.0 as u32);
        cmd.creation_flags(flags);
    }
    Ok(())
}

#[cfg(unix)]
fn child_id(child: &Child) -> Option<u32> {
    child.id()
}

#[cfg(windows)]
fn child_id(child: &Child) -> Option<u32> {
    child.id()
}

#[cfg(not(any(unix, windows)))]
fn child_id(child: &Child) -> Option<u32> {
    child.id()
}

/// Kill the child's process group (Unix) or process tree (Windows).
async fn kill_child(child_id: Option<u32>, kill_group: bool) {
    let _ = kill_group;
    #[cfg(unix)]
    {
        if let Some(id) = child_id {
            use nix::sys::signal::{Signal, killpg};
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

/// Resource limit spec (Unix only).
#[cfg(unix)]
#[derive(Clone, Copy)]
struct RlimitSpec {
    cpu: Option<u64>,
    mem: Option<u64>,
    nproc: Option<u64>,
    fsize: Option<u64>,
    nofile: Option<u64>,
    core: Option<u64>,
}

#[cfg(unix)]
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

/// Apply `setrlimit` limits to the current process (Unix).
#[cfg(unix)]
fn apply_rlimits(limits: &RlimitSpec) -> Result<(), std::io::Error> {
    unsafe {
        macro_rules! set {
            ($which:expr, $val:expr) => {{
                let lim = libc::rlimit {
                    rlim_cur: $val,
                    rlim_max: $val,
                };
                if libc::setrlimit($which, &lim) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
            }};
        }
        if let Some(cpu) = limits.cpu {
            set!(libc::RLIMIT_CPU, cpu);
        }
        if let Some(mem) = limits.mem {
            set!(libc::RLIMIT_AS, mem);
        }
        if let Some(nproc) = limits.nproc {
            set!(libc::RLIMIT_NPROC, nproc);
        }
        if let Some(fsize) = limits.fsize {
            set!(libc::RLIMIT_FSIZE, fsize);
        }
        if let Some(nofile) = limits.nofile {
            set!(libc::RLIMIT_NOFILE, nofile);
        }
        if let Some(core) = limits.core {
            set!(libc::RLIMIT_CORE, core);
        }
    }
    Ok(())
}

/// A periodic poller that samples a child's resource usage from the OS.
///
/// On Linux this reads `/proc/<pid>/stat` and `/proc/<pid>/status`; on other
/// platforms it is a no-op returning the parent's rusage delta.
pub struct ResourcePoller {
    #[cfg(target_os = "linux")]
    pid: i32,
    started: Instant,
}

impl ResourcePoller {
    /// Create a poller for a child PID.
    pub fn new(pid: i32) -> Self {
        Self {
            #[cfg(target_os = "linux")]
            pid,
            started: Instant::now(),
        }
    }

    /// Sample the child's current CPU time and RSS.
    pub fn sample(&self) -> Option<ResourceSample> {
        #[cfg(target_os = "linux")]
        {
            let stat_path = format!("/proc/{}/stat", self.pid);
            let stat = std::fs::read_to_string(&stat_path).ok()?;
            // Comm may contain spaces/parens; parse after the last ')'.
            let close = stat.rfind(')')?;
            let rest = &stat[close + 1..];
            let fields: Vec<&str> = rest.split_whitespace().collect();
            // After the ')' fields start at utime (field 14 → index 11 in rest).
            let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
            if clk_tck == 0 {
                return None;
            }
            let utime = fields.get(11)?.parse::<u64>().ok()?;
            let stime = fields.get(12)?.parse::<u64>().ok()?;
            let rss_pages = fields.get(21)?.parse::<u64>().ok()?;
            Some(ResourceSample {
                user_cpu_ms: utime * 1000 / clk_tck,
                system_cpu_ms: stime * 1000 / clk_tck,
                rss_bytes: rss_pages * 4096,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Elapsed wall time since the poller was created.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// A sampled point of resource usage.
#[derive(Debug, Clone, Copy)]
pub struct ResourceSample {
    pub user_cpu_ms: u64,
    pub system_cpu_ms: u64,
    pub rss_bytes: u64,
}

/// Convenience helper: run a command with a timeout, returning captured
/// output, without building a full policy. Useful for health checks and
/// backend probes.
pub async fn run_with_timeout(
    command: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<(i32, String, String), String> {
    let mut cmd = Command::new(command);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let child = cmd
        .spawn()
        .map_err(|e| format!("spawn '{}': {e}", command))?;
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(o)) => Ok((
            o.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&o.stdout).to_string(),
            String::from_utf8_lossy(&o.stderr).to_string(),
        )),
        Ok(Err(e)) => Err(format!("wait: {e}")),
        Err(_) => Err(format!("'{}' timed out after {:?}", command, timeout)),
    }
}

/// Track policy-denial counters so the metrics snapshot can fold them in.
/// Backends call [`DenialTracker::record`] whenever a rule blocks an action.
#[derive(Debug, Default, Clone)]
pub struct DenialTracker {
    denials: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl DenialTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one denial of the given category.
    pub fn record(&self) {
        self.denials
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The total number of denials recorded.
    pub fn count(&self) -> u64 {
        self.denials.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denial_tracker_counts() {
        let t = DenialTracker::new();
        assert_eq!(t.count(), 0);
        t.record();
        t.record();
        assert_eq!(t.count(), 2);
    }

    #[test]
    fn supervised_run_denial_attribution() {
        let run = SupervisedRun {
            exit_code: 1,
            stdout: String::new(),
            stderr: "sandbox: permission denied".to_string(),
            duration: Duration::from_secs(1),
            timed_out: false,
            rusage: None,
            policy_denials: 0,
            network_requests: 0,
            network_blocked: 0,
        };
        assert!(run.likely_sandbox_denied("linux"));
        assert!(!run.likely_sandbox_denied("noop"));
        let ok = SupervisedRun {
            exit_code: 0,
            ..run
        };
        assert!(!ok.likely_sandbox_denied("linux"));
    }

    #[tokio::test]
    async fn run_with_timeout_works() {
        // `echo` is not an executable on Windows, so use the platform shell.
        #[cfg(windows)]
        let (cmd, args) = ("cmd.exe", vec!["/C", "echo", "hi"]);
        #[cfg(not(windows))]
        let (cmd, args) = ("sh", vec!["-c", "echo hi"]);
        let (code, out, _) = run_with_timeout(cmd, &args, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(code, 0);
        assert!(out.contains("hi"));
    }

    #[tokio::test]
    async fn run_with_timeout_honors_limit() {
        let result = run_with_timeout("sleep", &["5"], Duration::from_millis(50)).await;
        assert!(result.is_err());
    }
}
