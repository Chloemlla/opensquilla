use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Security level for sandbox isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SandboxLevel {
    /// Standard isolation: basic namespace isolation, filesystem restrictions, network proxy
    Standard,
    /// Strict isolation: all of STANDARD + seccomp-BPF syscall filtering, resource limits
    Strict,
    /// Locked isolation: all of STRICT + no network, no persistent filesystem, read-only root
    Locked,
}

impl SandboxLevel {
    /// The most restrictive of two levels.
    pub fn max(self, other: SandboxLevel) -> SandboxLevel {
        if self >= other { self } else { other }
    }

    /// The least restrictive of two levels.
    pub fn min(self, other: SandboxLevel) -> SandboxLevel {
        if self <= other { self } else { other }
    }
}

/// Classification of the operation a sandboxed execution will perform. Used by
/// [`SandboxPolicy::select_level`] to recommend an isolation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationClass {
    /// Compiling or interpreting arbitrary code (python, node, …).
    CodeExecution,
    /// Executing a shell command line.
    ShellCommand,
    /// Reading files from disk.
    FileRead,
    /// Writing or deleting files on disk.
    FileWrite,
    /// Making network requests.
    NetworkRequest,
    /// Running git operations.
    GitOperation,
    /// Package installation or management.
    PackageInstall,
    /// Modifying system state (services, registry, boot config).
    SystemModification,
    /// Accessing credentials, keys, or secrets.
    CredentialAccess,
    /// Anything not covered above.
    Unknown,
}

/// Map an operation descriptor to a classification.
pub fn classify_operation(operation: &str) -> OperationClass {
    match operation {
        "code_execution" | "code_interpreter" | "execute_code" => OperationClass::CodeExecution,
        "shell_command" | "shell" | "command_execution" | "bash" => OperationClass::ShellCommand,
        "file_read" | "read_file" | "read" => OperationClass::FileRead,
        "file_write" | "write_file" | "write" | "delete_file" | "remove" => {
            OperationClass::FileWrite
        }
        "network_request" | "http_request" | "fetch" | "web_request" => {
            OperationClass::NetworkRequest
        }
        "git_operation" | "git" => OperationClass::GitOperation,
        "package_install" | "install_package" | "pip_install" | "npm_install" => {
            OperationClass::PackageInstall
        }
        "system_modification" | "service_management" | "system_config" | "shutdown" | "reboot" => {
            OperationClass::SystemModification
        }
        "credential_access" | "key_access" | "secret_read" | "password" => {
            OperationClass::CredentialAccess
        }
        _ => OperationClass::Unknown,
    }
}

/// Filesystem access policy for the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemPolicy {
    /// Directories allowed for read access
    pub read_allowed: Vec<String>,
    /// Directories allowed for read-write access
    pub write_allowed: Vec<String>,
    /// Directories explicitly denied
    pub denied: Vec<String>,
    /// Whether /tmp is writable
    pub tmp_writable: bool,
    /// Whether the home directory is readable
    pub home_readable: bool,
}

impl Default for FilesystemPolicy {
    fn default() -> Self {
        Self {
            read_allowed: Vec::new(),
            write_allowed: Vec::new(),
            denied: Vec::new(),
            tmp_writable: true,
            home_readable: true,
        }
    }
}

/// Return `true` when `path` is equal to `base` or a strict subpath of it.
///
/// Both sides are normalised by trimming trailing slashes. Windows drive
/// letters are compared case-insensitively; POSIX paths are case-sensitive.
fn path_is_within(path: &str, base: &str) -> bool {
    if base.is_empty() {
        return true;
    }
    if base == "/" {
        return true;
    }
    let p = path.trim_end_matches(['/', '\\']);
    let b = base.trim_end_matches(['/', '\\']);

    let eq = |a: &str, c: &str| -> bool {
        if is_windows_path(a) {
            a.eq_ignore_ascii_case(c)
        } else {
            a == c
        }
    };

    if eq(p, b) {
        return true;
    }
    if let Some(rest) = strip_prefix_case(p, b) {
        return rest.starts_with('/') || rest.starts_with('\\') || rest.is_empty();
    }
    false
}

fn is_windows_path(path: &str) -> bool {
    path.len() >= 2 && path.as_bytes()[1] == b':' && (path.as_bytes()[0].is_ascii_alphabetic())
}

fn strip_prefix_case<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if is_windows_path(s) && is_windows_path(prefix) {
        if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
            return Some(&s[prefix.len()..]);
        }
        None
    } else {
        s.strip_prefix(prefix)
    }
}

impl FilesystemPolicy {
    /// Is `path` readable under this policy? Denied paths always win.
    pub fn allows_read(&self, path: &str) -> bool {
        if self.blocks(path) {
            return false;
        }
        if self.read_allowed.iter().any(|a| path_is_within(path, a)) {
            return true;
        }
        if self.write_allowed.iter().any(|a| path_is_within(path, a)) {
            return true;
        }
        if self.home_readable {
            if let Some(home) = std::env::var_os("HOME") {
                let home = home.to_string_lossy();
                if path_is_within(path, &home) {
                    return true;
                }
            }
        }
        if self.tmp_writable && path_is_within(path, "/tmp") {
            return true;
        }
        false
    }

    /// Is `path` writable under this policy? Denied paths always win.
    pub fn allows_write(&self, path: &str) -> bool {
        if self.blocks(path) {
            return false;
        }
        if self.write_allowed.iter().any(|a| path_is_within(path, a)) {
            return true;
        }
        if self.tmp_writable && path_is_within(path, "/tmp") {
            return true;
        }
        false
    }

    /// Is `path` explicitly denied?
    pub fn blocks(&self, path: &str) -> bool {
        self.denied.iter().any(|d| path_is_within(path, d))
    }

    /// Builder: add a read-allowed path.
    pub fn with_read_allowed(mut self, path: impl Into<String>) -> Self {
        let p = path.into();
        if !self.read_allowed.contains(&p) {
            self.read_allowed.push(p);
        }
        self
    }

    /// Builder: add a write-allowed path.
    pub fn with_write_allowed(mut self, path: impl Into<String>) -> Self {
        let p = path.into();
        if !self.write_allowed.contains(&p) {
            self.write_allowed.push(p);
        }
        self
    }

    /// Builder: add a denied path.
    pub fn with_denied(mut self, path: impl Into<String>) -> Self {
        let p = path.into();
        if !self.denied.contains(&p) {
            self.denied.push(p);
        }
        self
    }

    /// Merge another filesystem policy into this one (union semantics).
    fn merge(&self, other: &FilesystemPolicy) -> FilesystemPolicy {
        let mut read_allowed = self.read_allowed.clone();
        for p in &other.read_allowed {
            if !read_allowed.contains(p) {
                read_allowed.push(p.clone());
            }
        }
        let mut write_allowed = self.write_allowed.clone();
        for p in &other.write_allowed {
            if !write_allowed.contains(p) {
                write_allowed.push(p.clone());
            }
        }
        let mut denied = self.denied.clone();
        for p in &other.denied {
            if !denied.contains(p) {
                denied.push(p.clone());
            }
        }
        FilesystemPolicy {
            read_allowed,
            write_allowed,
            denied,
            tmp_writable: other.tmp_writable,
            home_readable: other.home_readable,
        }
    }
}

/// Network access policy for the sandbox.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// No network access
    None,
    /// Only allowlisted domains via local proxy
    ProxyAllowlist(Vec<String>),
    /// Direct host network access (no restrictions)
    Host,
}

impl NetworkPolicy {
    /// Is all network access blocked?
    pub fn is_none(&self) -> bool {
        matches!(self, NetworkPolicy::None)
    }

    /// Is unrestricted host networking permitted?
    pub fn is_host(&self) -> bool {
        matches!(self, NetworkPolicy::Host)
    }

    /// The domain allowlist when using the proxy mode.
    pub fn allowlist(&self) -> Option<&[String]> {
        match self {
            NetworkPolicy::ProxyAllowlist(domains) => Some(domains),
            _ => None,
        }
    }

    /// The allowlist entries that would be rejected by
    /// [`crate::domain_validation::validate_domain_pattern`].
    ///
    /// Audit-style (never fails the policy): the proxy still honours these
    /// entries as configured, but callers can surface the list as warnings so
    /// an operator notices IP literals, non-FQDNs or broad wildcards that
    /// slipped into the allowlist.
    pub fn invalid_allowlist_entries(&self) -> Vec<String> {
        match self {
            NetworkPolicy::ProxyAllowlist(domains) => domains
                .iter()
                .filter(|d| {
                    crate::domain_validation::validate_domain_pattern(d).status
                        != crate::domain_validation::DomainStatus::Allowed
                })
                .cloned()
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Resource limits for the sandboxed process.
///
/// Every field is optional: `None` means "inherit the parent's limit". The
/// [`ResourceLimits::default`] applies conservative caps suitable for an
/// untrusted subprocess.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU time in seconds (RLIMIT_CPU).
    pub cpu_time_secs: Option<u64>,
    /// Maximum address space in bytes (RLIMIT_AS).
    pub memory_bytes: Option<u64>,
    /// Maximum number of child processes (RLIMIT_NPROC).
    pub max_processes: Option<u64>,
    /// Maximum file size in bytes (RLIMIT_FSIZE).
    pub file_size_bytes: Option<u64>,
    /// Maximum number of open file descriptors (RLIMIT_NOFILE).
    pub open_fds: Option<u64>,
    /// Maximum wall-clock time in seconds. Enforced by the backend with a
    /// timeout-and-kill, not by `setrlimit`.
    pub wall_time_secs: Option<u64>,
    /// Maximum resident set size in bytes (RLIMIT_RSS). Best-effort on most
    /// kernels; prefer `memory_bytes` for hard enforcement.
    pub rss_bytes: Option<u64>,
    /// Maximum number of threads (Linux: via `RLIMIT_NPROC` since threads
    /// count as processes; surfaced separately for clarity).
    pub max_threads: Option<u64>,
    /// Maximum core dump size in bytes (RLIMIT_CORE). Defaults to 0 in
    /// sandboxed executions to avoid leaking memory to disk.
    pub core_size_bytes: Option<u64>,
    /// CPU quota as a fraction of one core in [0.0, N.0]. Enforced via cgroup
    /// CPU bandwidth control on Linux (`cpu.cfs_quota_us`) where available;
    /// ignored on platforms without cgroup support.
    pub cpu_quota_cores: Option<f64>,
    /// Maximum total file locks (RLIMIT_LOCKS). Linux only.
    pub max_file_locks: Option<u64>,
    /// Maximum number of pending signals (RLIMIT_SIGPENDING). Linux only.
    pub max_pending_signals: Option<u64>,
    /// Maximum message queue bytes (RLIMIT_MSGQUEUE). Linux only.
    pub max_msgqueue_bytes: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_time_secs: Some(60),
            memory_bytes: Some(512 * 1024 * 1024),
            max_processes: Some(50),
            file_size_bytes: Some(100 * 1024 * 1024),
            open_fds: Some(256),
            wall_time_secs: Some(120),
            rss_bytes: Some(512 * 1024 * 1024),
            max_threads: Some(50),
            core_size_bytes: Some(0),
            cpu_quota_cores: Some(1.0),
            max_file_locks: Some(32),
            max_pending_signals: Some(64),
            max_msgqueue_bytes: Some(8192),
        }
    }
}

impl ResourceLimits {
    /// Conservative limits for an untrusted code-execution sandbox.
    pub fn strict() -> Self {
        Self {
            cpu_time_secs: Some(30),
            memory_bytes: Some(256 * 1024 * 1024),
            max_processes: Some(20),
            file_size_bytes: Some(50 * 1024 * 1024),
            open_fds: Some(128),
            wall_time_secs: Some(60),
            rss_bytes: Some(256 * 1024 * 1024),
            max_threads: Some(20),
            core_size_bytes: Some(0),
            cpu_quota_cores: Some(0.5),
            max_file_locks: Some(16),
            max_pending_signals: Some(32),
            max_msgqueue_bytes: Some(4096),
        }
    }

    /// Maximum-isolation limits.
    pub fn locked() -> Self {
        Self {
            cpu_time_secs: Some(10),
            memory_bytes: Some(128 * 1024 * 1024),
            max_processes: Some(10),
            file_size_bytes: Some(10 * 1024 * 1024),
            open_fds: Some(64),
            wall_time_secs: Some(20),
            rss_bytes: Some(128 * 1024 * 1024),
            max_threads: Some(10),
            core_size_bytes: Some(0),
            cpu_quota_cores: Some(0.25),
            max_file_locks: Some(8),
            max_pending_signals: Some(16),
            max_msgqueue_bytes: Some(2048),
        }
    }

    /// Limits for a specific level.
    pub fn for_level(level: SandboxLevel) -> Self {
        match level {
            SandboxLevel::Standard => Self::default(),
            SandboxLevel::Strict => Self::strict(),
            SandboxLevel::Locked => Self::locked(),
        }
    }

    /// Merge with another limit set. A non-`None` value in `other` wins;
    /// otherwise the base value is kept.
    fn merge(&self, other: &ResourceLimits) -> ResourceLimits {
        ResourceLimits {
            cpu_time_secs: other.cpu_time_secs.or(self.cpu_time_secs),
            memory_bytes: other.memory_bytes.or(self.memory_bytes),
            max_processes: other.max_processes.or(self.max_processes),
            file_size_bytes: other.file_size_bytes.or(self.file_size_bytes),
            open_fds: other.open_fds.or(self.open_fds),
            wall_time_secs: other.wall_time_secs.or(self.wall_time_secs),
            rss_bytes: other.rss_bytes.or(self.rss_bytes),
            max_threads: other.max_threads.or(self.max_threads),
            core_size_bytes: other.core_size_bytes.or(self.core_size_bytes),
            cpu_quota_cores: other.cpu_quota_cores.or(self.cpu_quota_cores),
            max_file_locks: other.max_file_locks.or(self.max_file_locks),
            max_pending_signals: other.max_pending_signals.or(self.max_pending_signals),
            max_msgqueue_bytes: other.max_msgqueue_bytes.or(self.max_msgqueue_bytes),
        }
    }

    /// Validate that limits are internally consistent.
    pub fn validate(&self) -> Result<(), PolicyValidationError> {
        if let Some(cpu) = self.cpu_time_secs {
            if cpu == 0 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "cpu_time_secs must be non-zero".to_string(),
                });
            }
        }
        if let Some(mem) = self.memory_bytes {
            if mem < 1024 * 1024 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "memory_bytes must be at least 1 MiB".to_string(),
                });
            }
        }
        if let Some(nproc) = self.max_processes {
            if nproc == 0 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "max_processes must be non-zero".to_string(),
                });
            }
        }
        if let Some(fds) = self.open_fds {
            if fds == 0 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "open_fds must be non-zero".to_string(),
                });
            }
        }
        if let Some(wall) = self.wall_time_secs {
            if wall == 0 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "wall_time_secs must be non-zero".to_string(),
                });
            }
        }
        if let Some(threads) = self.max_threads {
            if threads == 0 {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "max_threads must be non-zero".to_string(),
                });
            }
        }
        if let Some(quota) = self.cpu_quota_cores {
            if quota < 0.0 || !quota.is_finite() {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: "cpu_quota_cores must be non-negative and finite".to_string(),
                });
            }
        }
        // Wall time should be >= CPU time (a process cannot use more CPU than
        // wall time).
        if let (Some(cpu), Some(wall)) = (self.cpu_time_secs, self.wall_time_secs) {
            if cpu > wall {
                return Err(PolicyValidationError::InvalidResourceLimit {
                    message: format!(
                        "cpu_time_secs ({cpu}) must not exceed wall_time_secs ({wall})"
                    ),
                });
            }
        }
        Ok(())
    }

    /// The effective wall-clock timeout, falling back to the CPU time when no
    /// explicit wall time is set.
    pub fn effective_timeout_secs(&self) -> u64 {
        self.wall_time_secs
            .or(self.cpu_time_secs)
            .unwrap_or(300)
            .max(1)
    }

    /// Human-readable summary for audit logs.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(cpu) = self.cpu_time_secs {
            parts.push(format!("cpu={cpu}s"));
        }
        if let Some(wall) = self.wall_time_secs {
            parts.push(format!("wall={wall}s"));
        }
        if let Some(mem) = self.memory_bytes {
            parts.push(format!("mem={}", format_bytes(mem)));
        }
        if let Some(nproc) = self.max_processes {
            parts.push(format!("nproc={nproc}"));
        }
        if let Some(fds) = self.open_fds {
            parts.push(format!("fds={fds}"));
        }
        if let Some(fsize) = self.file_size_bytes {
            parts.push(format!("fsize={}", format_bytes(fsize)));
        }
        if let Some(quota) = self.cpu_quota_cores {
            parts.push(format!("quota={quota}c"));
        }
        parts.join(", ")
    }
}

/// Format a byte count as a human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}

/// Static rule set associated with a security level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelRules {
    /// The level these rules describe.
    pub level: SandboxLevel,
    /// Network is fully isolated (no host connectivity at the kernel level).
    pub network_isolated: bool,
    /// The root filesystem is read-only.
    pub read_only_root: bool,
    /// The user's home directory is readable.
    pub home_readable: bool,
    /// `/tmp` is writable.
    pub tmp_writable: bool,
    /// seccomp-BPF syscall filtering is enabled.
    pub seccomp_enabled: bool,
    /// The target binary must pass a code-signature check (macOS).
    pub require_code_signature: bool,
    /// A restricted token / least-privilege token is required (Windows).
    pub restricted_token: bool,
    /// Paths denied by default at this level.
    pub default_denied_paths: Vec<String>,
}

/// Complete sandbox policy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Isolation level
    pub level: SandboxLevel,
    /// Filesystem restrictions
    pub filesystem: FilesystemPolicy,
    /// Network access policy
    pub network: NetworkPolicy,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Whether to enable audit logging
    pub audit_enabled: bool,
    /// Additional environment variables to set
    pub env_allowlist: Vec<String>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy {
                read_allowed: vec![],
                write_allowed: vec![],
                denied: vec![],
                tmp_writable: true,
                home_readable: true,
            },
            network: NetworkPolicy::ProxyAllowlist(vec![]),
            resource_limits: ResourceLimits::default(),
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "USER".to_string(),
                "LANG".to_string(),
                "TMPDIR".to_string(),
            ],
        }
    }
}

impl SandboxPolicy {
    /// Select an appropriate policy level based on the operation type.
    ///
    /// Pure function: given an operation descriptor, returns the recommended
    /// isolation level. No I/O, no side effects — safe to call anywhere.
    pub fn select_level(operation: &str) -> SandboxLevel {
        match classify_operation(operation) {
            OperationClass::CodeExecution | OperationClass::ShellCommand => SandboxLevel::Strict,
            OperationClass::SystemModification
            | OperationClass::CredentialAccess
            | OperationClass::PackageInstall => SandboxLevel::Locked,
            OperationClass::FileRead
            | OperationClass::FileWrite
            | OperationClass::NetworkRequest
            | OperationClass::GitOperation
            | OperationClass::Unknown => SandboxLevel::Standard,
        }
    }

    /// Build a complete policy from a base level and optional overrides.
    ///
    /// Pure function: composes the level-specific defaults with caller
    /// overrides. Overrides merge additively for filesystem paths and
    /// environment variables; scalar settings (network mode, resource limits,
    /// audit flag) are replaced by the override when present.
    pub fn build_policy(level: SandboxLevel, overrides: Option<SandboxPolicy>) -> Self {
        let base = match level {
            SandboxLevel::Standard => Self::standard(),
            SandboxLevel::Strict => Self::strict(),
            SandboxLevel::Locked => Self::locked(),
        };

        match overrides {
            Some(o) => base.merge(&o),
            None => base,
        }
    }

    /// Build a policy for an operation, applying optional overrides.
    ///
    /// Equivalent to `build_policy(select_level(operation), overrides)`.
    pub fn for_operation(operation: &str, overrides: Option<&SandboxPolicy>) -> Self {
        let level = Self::select_level(operation);
        Self::build_policy(level, overrides.cloned())
    }

    /// Compose this policy with an override set.
    ///
    /// Merge semantics:
    /// - `level` is the most restrictive of the two.
    /// - filesystem `read_allowed` / `write_allowed` / `denied` are unioned.
    /// - filesystem booleans and network mode are taken from the override.
    /// - resource limits prefer the override's non-`None` values.
    /// - environment allowlists are unioned.
    pub fn merge(&self, other: &SandboxPolicy) -> SandboxPolicy {
        let mut env_allowlist = self.env_allowlist.clone();
        for var in &other.env_allowlist {
            if !env_allowlist.contains(var) {
                env_allowlist.push(var.clone());
            }
        }

        SandboxPolicy {
            level: self.level.max(other.level),
            filesystem: self.filesystem.merge(&other.filesystem),
            network: other.network.clone(),
            resource_limits: self.resource_limits.merge(&other.resource_limits),
            audit_enabled: other.audit_enabled,
            env_allowlist,
        }
    }

    /// Validate the policy for internal consistency.
    ///
    /// Checks:
    /// - resource limits are sane
    /// - no path is simultaneously denied and allowed
    /// - the network mode is compatible with the isolation level
    /// - allowlist entries are well-formed
    pub fn validate(&self) -> Result<(), PolicyValidationError> {
        self.resource_limits.validate()?;

        for denied in &self.filesystem.denied {
            if self.filesystem.read_allowed.iter().any(|a| a == denied) {
                return Err(PolicyValidationError::DeniedPathAlsoAllowed {
                    path: denied.clone(),
                });
            }
            if self.filesystem.write_allowed.iter().any(|a| a == denied) {
                return Err(PolicyValidationError::DeniedPathAlsoWritable {
                    path: denied.clone(),
                });
            }
        }

        if self.level == SandboxLevel::Locked && self.network.is_host() {
            return Err(PolicyValidationError::LevelNetworkConflict { level: self.level });
        }

        if let NetworkPolicy::ProxyAllowlist(domains) = &self.network {
            for domain in domains {
                if domain.trim().is_empty() {
                    return Err(PolicyValidationError::InvalidAllowedDomain {
                        domain: domain.clone(),
                    });
                }
                if domain.contains('/') || domain.contains(' ') {
                    return Err(PolicyValidationError::InvalidAllowedDomain {
                        domain: domain.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// The effective environment allowlist, deduplicated and sorted.
    pub fn effective_env_allowlist(&self) -> Vec<String> {
        let mut list = self.env_allowlist.clone();
        list.sort();
        list.dedup();
        list
    }

    /// The rule set that applies at this policy's level.
    pub fn rules(&self) -> LevelRules {
        Self::level_rules(self.level)
    }

    /// The static rule set for a given level.
    pub fn level_rules(level: SandboxLevel) -> LevelRules {
        match level {
            SandboxLevel::Standard => LevelRules {
                level,
                network_isolated: false,
                read_only_root: false,
                home_readable: true,
                tmp_writable: true,
                seccomp_enabled: false,
                require_code_signature: false,
                restricted_token: false,
                default_denied_paths: vec!["/proc".to_string(), "/sys".to_string()],
            },
            SandboxLevel::Strict => LevelRules {
                level,
                network_isolated: false,
                read_only_root: false,
                home_readable: true,
                tmp_writable: true,
                seccomp_enabled: true,
                require_code_signature: false,
                restricted_token: true,
                default_denied_paths: vec![
                    "/etc".to_string(),
                    "/var".to_string(),
                    "/sys".to_string(),
                    "/proc".to_string(),
                ],
            },
            SandboxLevel::Locked => LevelRules {
                level,
                network_isolated: true,
                read_only_root: true,
                home_readable: false,
                tmp_writable: false,
                seccomp_enabled: true,
                require_code_signature: true,
                restricted_token: true,
                default_denied_paths: vec![
                    "/etc".to_string(),
                    "/var".to_string(),
                    "/sys".to_string(),
                    "/proc".to_string(),
                    "/dev".to_string(),
                ],
            },
        }
    }

    /// Serialize to a JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Deserialize from a JSON value.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        serde_json::from_value(value.clone())
            .map_err(|e| format!("invalid sandbox policy JSON: {e}"))
    }

    /// Standard policy: basic isolation with network proxy.
    fn standard() -> Self {
        Self {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy {
                read_allowed: vec![],
                write_allowed: vec![],
                denied: vec![],
                tmp_writable: true,
                home_readable: true,
            },
            network: NetworkPolicy::ProxyAllowlist(vec![]),
            resource_limits: ResourceLimits::default(),
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "USER".to_string(),
                "LANG".to_string(),
                "TMPDIR".to_string(),
            ],
        }
    }

    /// Strict policy: syscall filtering, tighter resource limits.
    fn strict() -> Self {
        Self {
            level: SandboxLevel::Strict,
            filesystem: FilesystemPolicy {
                read_allowed: vec![],
                write_allowed: vec![],
                denied: vec![
                    "/etc".to_string(),
                    "/var".to_string(),
                    "/sys".to_string(),
                    "/proc".to_string(),
                ],
                tmp_writable: true,
                home_readable: true,
            },
            network: NetworkPolicy::ProxyAllowlist(vec![]),
            resource_limits: ResourceLimits::strict(),
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string(), "HOME".to_string(), "TMPDIR".to_string()],
        }
    }

    /// Locked policy: maximum isolation, no network, read-only.
    fn locked() -> Self {
        Self {
            level: SandboxLevel::Locked,
            filesystem: FilesystemPolicy {
                read_allowed: vec![],
                write_allowed: vec![],
                denied: vec![
                    "/etc".to_string(),
                    "/var".to_string(),
                    "/sys".to_string(),
                    "/proc".to_string(),
                    "/dev".to_string(),
                ],
                tmp_writable: false,
                home_readable: false,
            },
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits::locked(),
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string()],
        }
    }
}

/// Result of a sandbox execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SandboxResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
    pub audit_log: Vec<AuditEntry>,
}

/// A single audit record produced by a sandbox execution.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub action: String,
    pub details: String,
}

/// Validation errors reported by [`SandboxPolicy::validate`].
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PolicyValidationError {
    #[error("filesystem denied path '{path}' is also listed as read-allowed")]
    DeniedPathAlsoAllowed { path: String },
    #[error("filesystem denied path '{path}' is also listed as write-allowed")]
    DeniedPathAlsoWritable { path: String },
    #[error("invalid resource limit: {message}")]
    InvalidResourceLimit { message: String },
    #[error(
        "level '{level:?}' requires full network isolation but network policy allows host access"
    )]
    LevelNetworkConflict { level: SandboxLevel },
    #[error("invalid domain in network allowlist: '{domain}'")]
    InvalidAllowedDomain { domain: String },
    #[error("invalid environment variable name: '{name}'")]
    InvalidEnvVar { name: String },
}

/// Environment-variable filtering policy.
///
/// Controls which environment variables are passed to the sandboxed process.
/// The default behaviour is an allowlist: only variables listed in
/// `allowlist` (plus any in `set`) are inherited from the parent, and `set`
/// injects explicit values. Variables in `denylist` are always stripped,
/// even if they appear in the allowlist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentPolicy {
    /// Variables allowed to be inherited from the parent process.
    pub allowlist: Vec<String>,
    /// Variables to explicitly strip (always wins over allowlist).
    pub denylist: Vec<String>,
    /// Variables to set to explicit values.
    pub set: std::collections::HashMap<String, String>,
    /// Whether to clear the entire environment before applying the allowlist.
    /// When `true`, only `set` plus the allowlisted-and-present variables
    /// survive.
    pub clear: bool,
}

impl Default for EnvironmentPolicy {
    fn default() -> Self {
        Self {
            allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "USER".to_string(),
                "LANG".to_string(),
                "LC_ALL".to_string(),
                "TMPDIR".to_string(),
            ],
            denylist: vec![
                // Never leak secrets into the sandbox.
                "AWS_SECRET_ACCESS_KEY".to_string(),
                "GITHUB_TOKEN".to_string(),
                "OPENAI_API_KEY".to_string(),
                "ANTHROPIC_API_KEY".to_string(),
                "DATABASE_URL".to_string(),
                "SECRET".to_string(),
                "TOKEN".to_string(),
            ],
            set: std::collections::HashMap::new(),
            clear: false,
        }
    }
}

impl EnvironmentPolicy {
    /// Create an empty policy (nothing inherited).
    pub fn empty() -> Self {
        Self {
            allowlist: Vec::new(),
            denylist: Vec::new(),
            set: std::collections::HashMap::new(),
            clear: true,
        }
    }

    /// Builder: add an allowed variable.
    pub fn allow(mut self, var: impl Into<String>) -> Self {
        let v = var.into();
        if !self.allowlist.contains(&v) {
            self.allowlist.push(v);
        }
        self
    }

    /// Builder: add a denied variable.
    pub fn deny(mut self, var: impl Into<String>) -> Self {
        let v = var.into();
        if !self.denylist.contains(&v) {
            self.denylist.push(v);
        }
        self
    }

    /// Builder: set an explicit variable.
    pub fn set(mut self, var: impl Into<String>, value: impl Into<String>) -> Self {
        self.set.insert(var.into(), value.into());
        self
    }

    /// Filter a supplied environment map through this policy.
    ///
    /// `supplied` is the environment the caller wants to pass; the parent
    /// process's environment is consulted for allowlisted variables that are
    /// not in `supplied`.
    pub fn filter(&self, supplied: &std::collections::HashMap<String, String>) -> std::collections::HashMap<String, String> {
        let mut out = std::collections::HashMap::new();
        if !self.clear {
            // Inherit allowlisted variables from the parent when not supplied.
            for var in &self.allowlist {
                if !supplied.contains_key(var) {
                    if let Ok(v) = std::env::var(var) {
                        out.insert(var.clone(), v);
                    }
                }
            }
        }
        // Apply supplied values for allowlisted variables.
        for (k, v) in supplied {
            if self.denylist.iter().any(|d| d == k) {
                continue;
            }
            if self.allowlist.iter().any(|a| a == k) || self.set.contains_key(k) {
                out.insert(k.clone(), v.clone());
            }
        }
        // Apply explicit sets (always win).
        for (k, v) in &self.set {
            if !self.denylist.iter().any(|d| d == k) {
                out.insert(k.clone(), v.clone());
            }
        }
        // Strip denylist as a final guard.
        for d in &self.denylist {
            out.remove(d);
        }
        out
    }

    /// Validate that variable names are well-formed (alphanumeric + underscore,
    /// not starting with a digit).
    pub fn validate(&self) -> Result<(), PolicyValidationError> {
        let check = |name: &str| -> Result<(), PolicyValidationError> {
            if name.is_empty() {
                return Err(PolicyValidationError::InvalidEnvVar {
                    name: name.to_string(),
                });
            }
            let mut chars = name.chars();
            let first = chars.next().unwrap();
            if !(first.is_ascii_alphabetic() || first == '_') {
                return Err(PolicyValidationError::InvalidEnvVar {
                    name: name.to_string(),
                });
            }
            if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(PolicyValidationError::InvalidEnvVar {
                    name: name.to_string(),
                });
            }
            Ok(())
        };
        for v in &self.allowlist {
            check(v)?;
        }
        for v in &self.set {
            check(v.0)?;
        }
        Ok(())
    }
}

/// Build the default SSRF-prevention CIDR block list.
///
/// Returns the standard private / link-local / loopback ranges that a sandbox
/// proxy should refuse to connect to.
pub fn default_blocked_cidrs() -> Vec<String> {
    vec![
        "0.0.0.0/8".to_string(),
        "10.0.0.0/8".to_string(),
        "100.64.0.0/10".to_string(),
        "127.0.0.0/8".to_string(),
        "169.254.0.0/16".to_string(),
        "172.16.0.0/12".to_string(),
        "192.0.0.0/24".to_string(),
        "192.0.2.0/24".to_string(),
        "192.168.0.0/16".to_string(),
        "198.18.0.0/15".to_string(),
        "198.51.100.0/24".to_string(),
        "203.0.113.0/24".to_string(),
        "224.0.0.0/4".to_string(),
        "240.0.0.0/4".to_string(),
        "::1/128".to_string(),
        "fc00::/7".to_string(),
        "fe80::/10".to_string(),
        "::ffff:0:0/96".to_string(),
    ]
}

/// A network access rule combining a domain pattern and a port range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRule {
    /// Domain pattern (`"example.com"`, `"*.example.com"`).
    pub domain: String,
    /// Allowed ports. Empty means all ports.
    pub ports: Vec<u16>,
}

impl NetworkRule {
    /// Create a rule allowing all ports for a domain.
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            ports: Vec::new(),
        }
    }

    /// Create a rule allowing specific ports.
    pub fn with_ports(domain: impl Into<String>, ports: Vec<u16>) -> Self {
        Self {
            domain: domain.into(),
            ports,
        }
    }

    /// Does this rule match a given host and port?
    pub fn matches(&self, host: &str, port: u16) -> bool {
        let h = host.to_lowercase();
        let d = self.domain.to_lowercase();
        let domain_ok = if let Some(suffix) = d.strip_prefix("*.") {
            h == suffix || h.ends_with(&format!(".{suffix}"))
        } else {
            h == d
        };
        if !domain_ok {
            return false;
        }
        if self.ports.is_empty() {
            return true;
        }
        self.ports.contains(&port)
    }
}

/// Derive the effective resource limits for a level merged with overrides.
pub fn derive_resource_limits(
    level: SandboxLevel,
    overrides: Option<&ResourceLimits>,
) -> ResourceLimits {
    let base = ResourceLimits::for_level(level);
    match overrides {
        Some(o) => base.merge(o),
        None => base,
    }
}

/// A human-readable one-line summary of a policy, for audit logs.
pub fn policy_summary(policy: &SandboxPolicy) -> String {
    format!(
        "level={:?} net={} fs(read={} write={} deny={}) limits[{}] env={}",
        policy.level,
        match &policy.network {
            NetworkPolicy::None => "none".to_string(),
            NetworkPolicy::Host => "host".to_string(),
            NetworkPolicy::ProxyAllowlist(d) => format!("proxy({})", d.len()),
        },
        policy.filesystem.read_allowed.len(),
        policy.filesystem.write_allowed.len(),
        policy.filesystem.denied.len(),
        policy.resource_limits.summary(),
        policy.env_allowlist.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_limits_default_validates() {
        ResourceLimits::default().validate().unwrap();
        ResourceLimits::strict().validate().unwrap();
        ResourceLimits::locked().validate().unwrap();
    }

    #[test]
    fn resource_limits_for_level() {
        assert!(ResourceLimits::for_level(SandboxLevel::Locked).cpu_time_secs.unwrap() < ResourceLimits::for_level(SandboxLevel::Standard).cpu_time_secs.unwrap());
    }

    #[test]
    fn effective_timeout_falls_back_to_cpu() {
        let mut l = ResourceLimits::default();
        l.wall_time_secs = None;
        assert_eq!(l.effective_timeout_secs(), l.cpu_time_secs.unwrap());
    }

    #[test]
    fn environment_policy_filters() {
        let policy = EnvironmentPolicy::default()
            .allow("FOO")
            .deny("SECRET")
            .set("BAR", "baz");
        let mut supplied = std::collections::HashMap::new();
        supplied.insert("FOO".to_string(), "foo_val".to_string());
        supplied.insert("SECRET".to_string(), "leak".to_string());
        let filtered = policy.filter(&supplied);
        assert_eq!(filtered.get("FOO"), Some(&"foo_val".to_string()));
        assert_eq!(filtered.get("BAR"), Some(&"baz".to_string()));
        assert!(!filtered.contains_key("SECRET"));
    }

    #[test]
    fn environment_policy_validates_names() {
        let policy = EnvironmentPolicy::default().allow("1INVALID");
        assert!(policy.validate().is_err());
        let policy = EnvironmentPolicy::default().allow("VALID_NAME");
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn default_blocked_cidrs_includes_loopback() {
        let cidrs = default_blocked_cidrs();
        assert!(cidrs.contains(&"127.0.0.0/8".to_string()));
        assert!(cidrs.contains(&"169.254.0.0/16".to_string()));
    }

    #[test]
    fn network_rule_matches() {
        let rule = NetworkRule::with_ports("*.example.com", vec![443]);
        assert!(rule.matches("api.example.com", 443));
        assert!(!rule.matches("api.example.com", 80));
        assert!(!rule.matches("example.org", 443));
    }

    #[test]
    fn invalid_allowlist_entries_audit() {
        let policy = NetworkPolicy::ProxyAllowlist(vec![
            "github.com".to_string(),
            "127.0.0.1".to_string(),
            "localhost".to_string(),
            "*.com".to_string(),
        ]);
        let invalid = policy.invalid_allowlist_entries();
        assert!(!invalid.contains(&"github.com".to_string()));
        assert!(invalid.contains(&"127.0.0.1".to_string()));
        assert!(invalid.contains(&"localhost".to_string()));
        assert!(invalid.contains(&"*.com".to_string()));
        assert!(NetworkPolicy::None.invalid_allowlist_entries().is_empty());
        assert!(NetworkPolicy::Host.invalid_allowlist_entries().is_empty());
    }

    #[test]
    fn format_bytes_human_readable() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MiB");
    }

    #[test]
    fn policy_summary_nonempty() {
        let policy = SandboxPolicy::default();
        let s = policy_summary(&policy);
        assert!(s.contains("level="));
        assert!(s.contains("net="));
    }

    #[test]
    fn build_policy_for_each_level_validates() {
        for level in [
            SandboxLevel::Standard,
            SandboxLevel::Strict,
            SandboxLevel::Locked,
        ] {
            let policy = SandboxPolicy::build_policy(level, None);
            policy.validate().unwrap();
        }
    }
}
