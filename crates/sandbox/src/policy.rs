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
}

/// Resource limits for the sandboxed process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU time in seconds
    pub cpu_time_secs: Option<u64>,
    /// Maximum memory in bytes
    pub memory_bytes: Option<u64>,
    /// Maximum number of child processes
    pub max_processes: Option<u64>,
    /// Maximum file size in bytes
    pub file_size_bytes: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            cpu_time_secs: Some(60),
            memory_bytes: Some(512 * 1024 * 1024),
            max_processes: Some(50),
            file_size_bytes: Some(100 * 1024 * 1024),
        }
    }
}

impl ResourceLimits {
    /// Merge with another limit set. A non-`None` value in `other` wins;
    /// otherwise the base value is kept.
    fn merge(&self, other: &ResourceLimits) -> ResourceLimits {
        ResourceLimits {
            cpu_time_secs: other.cpu_time_secs.or(self.cpu_time_secs),
            memory_bytes: other.memory_bytes.or(self.memory_bytes),
            max_processes: other.max_processes.or(self.max_processes),
            file_size_bytes: other.file_size_bytes.or(self.file_size_bytes),
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
        Ok(())
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
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(60),
                memory_bytes: Some(512 * 1024 * 1024), // 512 MB
                max_processes: Some(50),
                file_size_bytes: Some(100 * 1024 * 1024), // 100 MB
            },
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
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(60),
                memory_bytes: Some(512 * 1024 * 1024),
                max_processes: Some(50),
                file_size_bytes: Some(100 * 1024 * 1024),
            },
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
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(30),
                memory_bytes: Some(256 * 1024 * 1024),
                max_processes: Some(20),
                file_size_bytes: Some(50 * 1024 * 1024),
            },
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
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(10),
                memory_bytes: Some(128 * 1024 * 1024),
                max_processes: Some(10),
                file_size_bytes: Some(10 * 1024 * 1024),
            },
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string()],
        }
    }
}

/// Result of a sandbox execution.
#[derive(Debug, Clone)]
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
}
