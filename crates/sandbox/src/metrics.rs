//! Sandbox execution metrics: CPU time, memory peak, syscall counts.
//!
//! Provides:
//!
//! - [`ExecutionMetrics`] — a snapshot of resource usage for a single
//!   sandboxed execution.
//! - [`MetricsCollector`] — a thread-safe accumulator that backends push
//!   per-execution samples to and that callers snapshot for dashboards.
//! - [`SyscallCounter`] — a counting tally of syscall classes observed by a
//!   seccomp/audit backend (Linux) or ETW/Seatbelt (macOS).
//!
//! The collector is `Send + Sync` and cheap to clone (state behind `Arc`),
//! so a single instance can be shared across all sandbox backends in a process.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// A snapshot of resource usage for a single sandboxed execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Opaque execution id (correlates with audit entries).
    pub execution_id: String,
    /// When the execution started.
    pub started_at: DateTime<Utc>,
    /// When the execution ended.
    pub ended_at: DateTime<Utc>,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// User CPU time in milliseconds.
    pub user_cpu_ms: u64,
    /// System (kernel) CPU time in milliseconds.
    pub system_cpu_ms: u64,
    /// Peak resident set size in bytes.
    pub peak_rss_bytes: u64,
    /// Peak virtual memory in bytes.
    pub peak_vmem_bytes: u64,
    /// Number of bytes read from disk.
    pub io_read_bytes: u64,
    /// Number of bytes written to disk.
    pub io_write_bytes: u64,
    /// Number of child processes spawned.
    pub child_processes: u64,
    /// Number of file descriptors opened.
    pub open_fds: u64,
    /// Exit code of the process.
    pub exit_code: i32,
    /// Whether the execution was killed by a timeout.
    pub timed_out: bool,
    /// Per-syscall-class counts (Linux seccomp audit / macOS Seatbelt).
    pub syscalls: HashMap<String, u64>,
    /// Number of policy denials (filesystem, network, syscall).
    pub policy_denials: u64,
    /// Number of network requests attempted.
    pub network_requests: u64,
    /// Number of network requests blocked by the proxy.
    pub network_blocked: u64,
    /// Free-form backend tags.
    pub backend: String,
    /// Security level used.
    pub level: String,
}

impl ExecutionMetrics {
    /// Create a new metrics snapshot with the given execution id and backend.
    pub fn new(execution_id: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            execution_id: execution_id.into(),
            started_at: Utc::now(),
            ended_at: Utc::now(),
            duration_ms: 0,
            user_cpu_ms: 0,
            system_cpu_ms: 0,
            peak_rss_bytes: 0,
            peak_vmem_bytes: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
            child_processes: 0,
            open_fds: 0,
            exit_code: 0,
            timed_out: false,
            syscalls: HashMap::new(),
            policy_denials: 0,
            network_requests: 0,
            network_blocked: 0,
            backend: backend.into(),
            level: String::new(),
        }
    }

    /// Record the start time.
    pub fn mark_start(&mut self) {
        self.started_at = Utc::now();
    }

    /// Record the end time and compute the wall-clock duration.
    pub fn mark_end(&mut self) {
        self.ended_at = Utc::now();
        self.duration_ms = (self.ended_at - self.started_at).num_milliseconds().max(0) as u64;
    }

    /// Set the wall-clock duration from a `Duration`.
    pub fn with_duration(mut self, d: Duration) -> Self {
        self.duration_ms = d.as_millis() as u64;
        self
    }

    /// Set CPU times.
    pub fn with_cpu(mut self, user_ms: u64, system_ms: u64) -> Self {
        self.user_cpu_ms = user_ms;
        self.system_cpu_ms = system_ms;
        self
    }

    /// Set memory peaks.
    pub fn with_memory(mut self, peak_rss: u64, peak_vmem: u64) -> Self {
        self.peak_rss_bytes = peak_rss;
        self.peak_vmem_bytes = peak_vmem;
        self
    }

    /// Set I/O counters.
    pub fn with_io(mut self, read: u64, write: u64) -> Self {
        self.io_read_bytes = read;
        self.io_write_bytes = write;
        self
    }

    /// Set the exit code and timeout flag.
    pub fn with_exit(mut self, code: i32, timed_out: bool) -> Self {
        self.exit_code = code;
        self.timed_out = timed_out;
        self
    }

    /// Set the security level.
    pub fn with_level(mut self, level: impl Into<String>) -> Self {
        self.level = level.into();
        self
    }

    /// Record a syscall observation.
    pub fn record_syscall(&mut self, name: &str) {
        *self.syscalls.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Total CPU time (user + system) in milliseconds.
    pub fn total_cpu_ms(&self) -> u64 {
        self.user_cpu_ms + self.system_cpu_ms
    }

    /// CPU utilisation as a fraction of wall-clock time (0.0–N.0, where N is
    /// the number of cores). Returns 0.0 when duration is zero.
    pub fn cpu_utilization(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        self.total_cpu_ms() as f64 / self.duration_ms as f64
    }

    /// Total syscall count across all classes.
    pub fn total_syscalls(&self) -> u64 {
        self.syscalls.values().copied().sum()
    }
}

/// Classification of syscalls into coarse categories for metrics dashboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyscallClass {
    Filesystem,
    Network,
    Process,
    Memory,
    Signal,
    Time,
    System,
    Other,
}

impl SyscallClass {
    /// Classify a syscall name into a coarse category.
    pub fn classify(name: &str) -> SyscallClass {
        match name {
            "read" | "write" | "open" | "openat" | "close" | "lseek" | "stat" | "fstat"
            | "newfstatat" | "statfs" | "fstatfs" | "readlink" | "readlinkat" | "getdents"
            | "getdents64" | "mkdir" | "rmdir" | "unlink" | "unlinkat" | "rename" | "renameat"
            | "link" | "linkat" | "symlink" | "symlinkat" | "chmod" | "fchmod" | "fchmodat"
            | "chown" | "fchown" | "fchownat" | "truncate" | "ftruncate" | "pwrite64"
            | "pread64" | "writev" | "readv" | "access" | "faccessat" | "utimensat" | "getcwd"
            | "chdir" | "fchdir" => SyscallClass::Filesystem,
            "socket" | "connect" | "accept" | "accept4" | "bind" | "listen" | "sendto"
            | "recvfrom" | "sendmsg" | "recvmsg" | "getsockopt" | "setsockopt" | "getpeername"
            | "getsockname" | "shutdown" | "socketpair" => SyscallClass::Network,
            "fork" | "vfork" | "clone" | "clone3" | "execve" | "execveat" | "exit"
            | "exit_group" | "wait4" | "waitid" | "setpgid" | "setsid" | "getpid" | "getppid"
            | "gettid" | "getuid" | "geteuid" | "getgid" | "getegid" | "prctl" => {
                SyscallClass::Process
            }
            "mmap" | "munmap" | "mprotect" | "brk" | "madvise" | "mremap" | "mincore" => {
                SyscallClass::Memory
            }
            "rt_sigaction" | "rt_sigprocmask" | "rt_sigreturn" | "sigaltstack" | "kill"
            | "tgkill" | "tkill" | "pause" => SyscallClass::Signal,
            "clock_gettime" | "gettimeofday" | "nanosleep" | "clock_nanosleep" | "time"
            | "times" | "timerfd_create" | "timerfd_settime" => SyscallClass::Time,
            "uname" | "sysinfo" | "getrlimit" | "setrlimit" | "prlimit64" | "getrandom"
            | "sched_yield" | "sched_getaffinity" | "sched_setaffinity" | "umask" | "ioctl"
            | "fcntl" | "dup" | "dup2" | "dup3" | "pipe" | "pipe2" | "poll" | "ppoll"
            | "epoll_create1" | "epoll_ctl" | "epoll_wait" | "eventfd2" | "futex"
            | "set_robust_list" | "set_tid_address" => SyscallClass::System,
            _ => SyscallClass::Other,
        }
    }

    /// Human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            SyscallClass::Filesystem => "filesystem",
            SyscallClass::Network => "network",
            SyscallClass::Process => "process",
            SyscallClass::Memory => "memory",
            SyscallClass::Signal => "signal",
            SyscallClass::Time => "time",
            SyscallClass::System => "system",
            SyscallClass::Other => "other",
        }
    }
}

/// A counting tally of syscalls, keyed by name.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyscallCounter {
    counts: HashMap<String, u64>,
}

impl SyscallCounter {
    /// Create an empty counter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one observation of a syscall.
    pub fn record(&mut self, name: &str) {
        *self.counts.entry(name.to_string()).or_insert(0) += 1;
    }

    /// Record `n` observations of a syscall.
    pub fn record_n(&mut self, name: &str, n: u64) {
        *self.counts.entry(name.to_string()).or_insert(0) += n;
    }

    /// Get the count for a syscall.
    pub fn get(&self, name: &str) -> u64 {
        self.counts.get(name).copied().unwrap_or(0)
    }

    /// Total observations across all syscalls.
    pub fn total(&self) -> u64 {
        self.counts.values().copied().sum()
    }

    /// Aggregate counts by [`SyscallClass`].
    pub fn by_class(&self) -> HashMap<SyscallClass, u64> {
        let mut out = HashMap::new();
        for (name, count) in &self.counts {
            let class = SyscallClass::classify(name);
            *out.entry(class).or_insert(0) += count;
        }
        out
    }

    /// All recorded syscall names and their counts.
    pub fn entries(&self) -> &HashMap<String, u64> {
        &self.counts
    }

    /// Merge another counter into this one.
    pub fn merge(&mut self, other: &SyscallCounter) {
        for (name, count) in &other.counts {
            *self.counts.entry(name.clone()).or_insert(0) += count;
        }
    }

    /// Reset all counts.
    pub fn clear(&mut self) {
        self.counts.clear();
    }
}

/// Aggregate metrics over many executions, for dashboards and reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Total executions observed.
    pub total_executions: u64,
    /// Executions that exited zero.
    pub successful_executions: u64,
    /// Executions that exited non-zero.
    pub failed_executions: u64,
    /// Executions killed by timeout.
    pub timed_out_executions: u64,
    /// Total wall-clock time in milliseconds.
    pub total_duration_ms: u64,
    /// Total user CPU time in milliseconds.
    pub total_user_cpu_ms: u64,
    /// Total system CPU time in milliseconds.
    pub total_system_cpu_ms: u64,
    /// Peak RSS across all executions, in bytes.
    pub peak_rss_bytes: u64,
    /// Total bytes read from disk.
    pub total_io_read_bytes: u64,
    /// Total bytes written to disk.
    pub total_io_write_bytes: u64,
    /// Total policy denials.
    pub total_policy_denials: u64,
    /// Total network requests attempted.
    pub total_network_requests: u64,
    /// Total network requests blocked.
    pub total_network_blocked: u64,
    /// Aggregate syscall counts.
    pub syscalls: SyscallCounter,
    /// Per-backend execution counts.
    pub per_backend: HashMap<String, u64>,
    /// Per-level execution counts.
    pub per_level: HashMap<String, u64>,
}

impl AggregateMetrics {
    /// Fold a single execution's metrics into the aggregate.
    pub fn fold(&mut self, m: &ExecutionMetrics) {
        self.total_executions += 1;
        if m.timed_out {
            self.timed_out_executions += 1;
        } else if m.exit_code == 0 {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }
        self.total_duration_ms += m.duration_ms;
        self.total_user_cpu_ms += m.user_cpu_ms;
        self.total_system_cpu_ms += m.system_cpu_ms;
        if m.peak_rss_bytes > self.peak_rss_bytes {
            self.peak_rss_bytes = m.peak_rss_bytes;
        }
        self.total_io_read_bytes += m.io_read_bytes;
        self.total_io_write_bytes += m.io_write_bytes;
        self.total_policy_denials += m.policy_denials;
        self.total_network_requests += m.network_requests;
        self.total_network_blocked += m.network_blocked;
        for (name, count) in &m.syscalls {
            self.syscalls.record_n(name, *count);
        }
        *self.per_backend.entry(m.backend.clone()).or_insert(0) += 1;
        if !m.level.is_empty() {
            *self.per_level.entry(m.level.clone()).or_insert(0) += 1;
        }
    }

    /// Average wall-clock duration in milliseconds.
    pub fn avg_duration_ms(&self) -> u64 {
        self.total_duration_ms.checked_div(self.total_executions).unwrap_or(0)
    }

    /// Success rate as a fraction in [0.0, 1.0].
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }
}

/// Thread-safe accumulator that backends push samples to.
#[derive(Clone)]
pub struct MetricsCollector {
    inner: Arc<RwLock<Vec<ExecutionMetrics>>>,
    counters: Arc<Counters>,
}

#[derive(Default)]
struct Counters {
    total_executions: AtomicU64,
    successful: AtomicU64,
    failed: AtomicU64,
    timed_out: AtomicU64,
    policy_denials: AtomicU64,
    network_requests: AtomicU64,
    network_blocked: AtomicU64,
}

impl MetricsCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Vec::new())),
            counters: Arc::new(Counters::default()),
        }
    }

    /// Record a completed execution.
    pub async fn record(&self, metrics: ExecutionMetrics) {
        self.counters
            .total_executions
            .fetch_add(1, Ordering::Relaxed);
        if metrics.timed_out {
            self.counters.timed_out.fetch_add(1, Ordering::Relaxed);
        } else if metrics.exit_code == 0 {
            self.counters.successful.fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters.failed.fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .policy_denials
            .fetch_add(metrics.policy_denials, Ordering::Relaxed);
        self.counters
            .network_requests
            .fetch_add(metrics.network_requests, Ordering::Relaxed);
        self.counters
            .network_blocked
            .fetch_add(metrics.network_blocked, Ordering::Relaxed);
        self.inner.write().await.push(metrics);
    }

    /// Snapshot the raw per-execution metrics (newest last).
    pub async fn samples(&self) -> Vec<ExecutionMetrics> {
        self.inner.read().await.clone()
    }

    /// Compute aggregate metrics over all recorded executions.
    pub async fn aggregate(&self) -> AggregateMetrics {
        let mut agg = AggregateMetrics::default();
        for m in self.inner.read().await.iter() {
            agg.fold(m);
        }
        agg
    }

    /// Retain only the last `n` executions, dropping older samples.
    pub async fn retain_last(&self, n: usize) {
        let mut guard = self.inner.write().await;
        if guard.len() > n {
            let start = guard.len() - n;
            guard.drain(0..start);
        }
    }

    /// Total executions recorded (atomic, no lock).
    pub fn total_executions(&self) -> u64 {
        self.counters.total_executions.load(Ordering::Relaxed)
    }

    /// Successful executions (atomic).
    pub fn successful(&self) -> u64 {
        self.counters.successful.load(Ordering::Relaxed)
    }

    /// Failed executions (atomic).
    pub fn failed(&self) -> u64 {
        self.counters.failed.load(Ordering::Relaxed)
    }

    /// Timed-out executions (atomic).
    pub fn timed_out(&self) -> u64 {
        self.counters.timed_out.load(Ordering::Relaxed)
    }

    /// Total policy denials (atomic).
    pub fn policy_denials(&self) -> u64 {
        self.counters.policy_denials.load(Ordering::Relaxed)
    }

    /// Clear all recorded metrics.
    pub async fn clear(&self) {
        self.inner.write().await.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Read resource usage for the current process from `/proc/self` (Linux).
/// Returns `None` on non-Linux or on read failure.
#[cfg(target_os = "linux")]
pub fn read_self_rusage() -> Option<Rusage> {
    let mut user_ms = 0u64;
    let mut system_ms = 0u64;
    let mut rss_bytes = 0u64;
    let mut vmem_bytes = 0u64;

    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        let fields: Vec<&str> = stat.split_whitespace().collect();
        // Fields (1-indexed in man proc): utime=14, stime=15, vsize=23, rss=24.
        if fields.len() >= 24 {
            let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as u64;
            if clk_tck > 0 {
                user_ms = (fields[13].parse::<u64>().ok()? * 1000)
                    .checked_div(clk_tck)
                    .unwrap_or(0);
                system_ms = (fields[14].parse::<u64>().ok()? * 1000)
                    .checked_div(clk_tck)
                    .unwrap_or(0);
            }
            vmem_bytes = fields[22].parse::<u64>().ok()?;
            let rss_pages = fields[23].parse::<u64>().ok()?;
            let page_size = 4096u64;
            rss_bytes = rss_pages * page_size;
        }
    }

    let mut io_read = 0u64;
    let mut io_write = 0u64;
    if let Ok(io) = std::fs::read_to_string("/proc/self/io") {
        for line in io.lines() {
            if let Some(rest) = line.strip_prefix("read_bytes:") {
                io_read = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("write_bytes:") {
                io_write = rest.trim().parse().unwrap_or(0);
            }
        }
    }

    Some(Rusage {
        user_ms,
        system_ms,
        rss_bytes,
        vmem_bytes,
        io_read_bytes: io_read,
        io_write_bytes: io_write,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn read_self_rusage() -> Option<Rusage> {
    None
}

/// Resource usage snapshot used to compute deltas for a sandboxed child.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Rusage {
    pub user_ms: u64,
    pub system_ms: u64,
    pub rss_bytes: u64,
    pub vmem_bytes: u64,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
}

impl Rusage {
    /// Compute the delta between a "before" and "after" snapshot.
    pub fn delta(before: &Rusage, after: &Rusage) -> Rusage {
        Rusage {
            user_ms: after.user_ms.saturating_sub(before.user_ms),
            system_ms: after.system_ms.saturating_sub(before.system_ms),
            rss_bytes: after.rss_bytes.max(before.rss_bytes),
            vmem_bytes: after.vmem_bytes.max(before.vmem_bytes),
            io_read_bytes: after.io_read_bytes.saturating_sub(before.io_read_bytes),
            io_write_bytes: after.io_write_bytes.saturating_sub(before.io_write_bytes),
        }
    }

    /// Apply this rusage to a metrics snapshot.
    pub fn apply_to(&self, m: &mut ExecutionMetrics) {
        m.user_cpu_ms = self.user_ms;
        m.system_cpu_ms = self.system_ms;
        m.peak_rss_bytes = self.rss_bytes;
        m.peak_vmem_bytes = self.vmem_bytes;
        m.io_read_bytes = self.io_read_bytes;
        m.io_write_bytes = self.io_write_bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_classification() {
        assert_eq!(SyscallClass::classify("openat"), SyscallClass::Filesystem);
        assert_eq!(SyscallClass::classify("connect"), SyscallClass::Network);
        assert_eq!(SyscallClass::classify("execve"), SyscallClass::Process);
        assert_eq!(SyscallClass::classify("mmap"), SyscallClass::Memory);
        assert_eq!(
            SyscallClass::classify("unknown_syscall"),
            SyscallClass::Other
        );
    }

    #[test]
    fn aggregate_fold() {
        let mut agg = AggregateMetrics::default();
        let m1 = ExecutionMetrics::new("e1", "linux")
            .with_cpu(10, 5)
            .with_memory(1000, 2000)
            .with_exit(0, false);
        let m2 = ExecutionMetrics::new("e2", "linux")
            .with_cpu(20, 10)
            .with_memory(1500, 3000)
            .with_exit(1, false);
        let m3 = ExecutionMetrics::new("e3", "linux").with_exit(-1, true);
        agg.fold(&m1);
        agg.fold(&m2);
        agg.fold(&m3);
        assert_eq!(agg.total_executions, 3);
        assert_eq!(agg.successful_executions, 1);
        assert_eq!(agg.failed_executions, 1);
        assert_eq!(agg.timed_out_executions, 1);
        assert_eq!(agg.total_user_cpu_ms, 30);
        assert_eq!(agg.peak_rss_bytes, 1500);
    }

    #[tokio::test]
    async fn collector_record_and_aggregate() {
        let c = MetricsCollector::new();
        c.record(
            ExecutionMetrics::new("e1", "linux")
                .with_exit(0, false)
                .with_level("strict"),
        )
        .await;
        c.record(
            ExecutionMetrics::new("e2", "linux")
                .with_exit(1, false)
                .with_level("strict"),
        )
        .await;
        assert_eq!(c.total_executions(), 2);
        assert_eq!(c.successful(), 1);
        assert_eq!(c.failed(), 1);
        let agg = c.aggregate().await;
        assert_eq!(agg.per_level.get("strict"), Some(&2));
    }

    #[test]
    fn syscall_counter_by_class() {
        let mut counter = SyscallCounter::new();
        counter.record("openat");
        counter.record("openat");
        counter.record("read");
        counter.record("connect");
        let by_class = counter.by_class();
        assert_eq!(by_class.get(&SyscallClass::Filesystem), Some(&3));
        assert_eq!(by_class.get(&SyscallClass::Network), Some(&1));
    }

    #[test]
    fn cpu_utilization() {
        let m = ExecutionMetrics::new("e1", "linux")
            .with_duration(Duration::from_millis(1000))
            .with_cpu(500, 500);
        assert!((m.cpu_utilization() - 1.0).abs() < 0.001);
    }
}
