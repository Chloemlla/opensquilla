//! Sandbox profile presets per security level.
//!
//! Each [`SandboxProfile`] bundles a [`SandboxPolicy`]
//! with a human-readable name, description, the set of capabilities it drops,
//! and the platform-specific notes that the backends consult when building
//! their isolation primitives (bwrap args, SBPL text, Job Object flags).
//!
//! The presets are intentionally conservative: they err on the side of more
//! isolation and tighter resource limits. Callers can relax a preset by
//! merging an override policy ([`SandboxProfile::with_overrides`]).
//!
//! In addition to the three core levels (STANDARD, STRICT, LOCKED) this module
//! defines task-specific profiles that compose a base level with operation-
//! specific filesystem and network rules, e.g. a `code_execution` profile that
//! locks down the network but permits a writable workspace and `/tmp`.

use serde::{Deserialize, Serialize};

use crate::policy::{
    FilesystemPolicy, NetworkPolicy, OperationClass, ResourceLimits, SandboxLevel, SandboxPolicy,
    classify_operation,
};

/// A named, reusable sandbox profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxProfile {
    /// Machine-readable profile id, e.g. `"strict_code_execution"`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Longer description for audit logs and UIs.
    pub description: String,
    /// The base security level.
    pub level: SandboxLevel,
    /// The fully-derived policy.
    pub policy: SandboxPolicy,
    /// Linux capabilities to drop (display only; backends drop ALL and only
    /// re-add from this list when the level permits).
    pub allowed_capabilities: Vec<String>,
    /// Platform notes consulted by the backends.
    pub notes: ProfileNotes,
}

/// Platform-specific notes attached to a profile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileNotes {
    /// Whether the bwrap backend should unshare the network namespace.
    pub linux_unshare_net: bool,
    /// Whether the bwrap backend should mount a fresh tmpfs root.
    pub linux_fresh_root: bool,
    /// Whether the macOS backend should apply a deny-by-default SBPL profile.
    pub macos_deny_default: bool,
    /// Whether the Windows backend should create a restricted token.
    pub windows_restricted_token: bool,
    /// Whether the Windows backend should install a firewall block rule.
    pub windows_firewall_block: bool,
    /// Whether seccomp-BPF filtering is enabled.
    pub seccomp_enabled: bool,
    /// Whether the sandbox root filesystem is read-only.
    pub read_only_root: bool,
}

impl SandboxProfile {
    /// Look up the standard preset for a security level.
    pub fn for_level(level: SandboxLevel) -> Self {
        match level {
            SandboxLevel::Standard => Self::standard(),
            SandboxLevel::Strict => Self::strict(),
            SandboxLevel::Locked => Self::locked(),
        }
    }

    /// Select a profile for an operation descriptor, falling back to the level
    /// recommended by [`SandboxPolicy::select_level`].
    pub fn for_operation(operation: &str) -> Self {
        match classify_operation(operation) {
            OperationClass::CodeExecution => Self::code_execution(),
            OperationClass::ShellCommand => Self::shell_command(),
            OperationClass::FileRead => Self::file_read(),
            OperationClass::FileWrite => Self::file_write(),
            OperationClass::NetworkRequest => Self::network_request(),
            OperationClass::GitOperation => Self::git_operation(),
            OperationClass::PackageInstall => Self::package_install(),
            OperationClass::SystemModification => Self::system_modification(),
            OperationClass::CredentialAccess => Self::credential_access(),
            OperationClass::Unknown => {
                let level = SandboxPolicy::select_level(operation);
                Self::for_level(level)
            }
        }
    }

    /// Apply caller overrides by merging an override policy into this
    /// profile's policy.
    pub fn with_overrides(mut self, overrides: &SandboxPolicy) -> Self {
        self.policy = self.policy.merge(overrides);
        self.level = self.policy.level;
        self
    }

    /// Standard profile: basic isolation with network proxy.
    pub fn standard() -> Self {
        Self {
            id: "standard".to_string(),
            name: "Standard".to_string(),
            description: "Basic namespace isolation, filesystem restrictions, network proxy."
                .to_string(),
            level: SandboxLevel::Standard,
            policy: SandboxPolicy::build_policy(SandboxLevel::Standard, None),
            allowed_capabilities: vec![],
            notes: ProfileNotes {
                linux_unshare_net: false,
                linux_fresh_root: false,
                macos_deny_default: false,
                windows_restricted_token: false,
                windows_firewall_block: false,
                seccomp_enabled: false,
                read_only_root: false,
            },
        }
    }

    /// Strict profile: syscall filtering, tighter resource limits.
    pub fn strict() -> Self {
        Self {
            id: "strict".to_string(),
            name: "Strict".to_string(),
            description: "Seccomp-BPF syscall filtering, restricted token, tighter limits."
                .to_string(),
            level: SandboxLevel::Strict,
            policy: SandboxPolicy::build_policy(SandboxLevel::Strict, None),
            allowed_capabilities: vec![],
            notes: ProfileNotes {
                linux_unshare_net: false,
                linux_fresh_root: false,
                macos_deny_default: true,
                windows_restricted_token: true,
                windows_firewall_block: false,
                seccomp_enabled: true,
                read_only_root: false,
            },
        }
    }

    /// Locked profile: maximum isolation, no network, read-only.
    pub fn locked() -> Self {
        Self {
            id: "locked".to_string(),
            name: "Locked".to_string(),
            description: "No network, no persistent filesystem, read-only root, full seccomp."
                .to_string(),
            level: SandboxLevel::Locked,
            policy: SandboxPolicy::build_policy(SandboxLevel::Locked, None),
            allowed_capabilities: vec![],
            notes: ProfileNotes {
                linux_unshare_net: true,
                linux_fresh_root: true,
                macos_deny_default: true,
                windows_restricted_token: true,
                windows_firewall_block: true,
                seccomp_enabled: true,
                read_only_root: true,
            },
        }
    }

    /// Profile for arbitrary code execution: locked network, writable
    /// workspace + `/tmp`, seccomp on.
    pub fn code_execution() -> Self {
        let mut profile = Self::strict();
        profile.id = "strict_code_execution".to_string();
        profile.name = "Code Execution".to_string();
        profile.description =
            "Arbitrary code execution: locked network, writable workspace, seccomp.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Strict,
            filesystem: FilesystemPolicy::default()
                .with_write_allowed("/workspace")
                .with_write_allowed("/tmp"),
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(30),
                memory_bytes: Some(256 * 1024 * 1024),
                max_processes: Some(20),
                file_size_bytes: Some(50 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "TMPDIR".to_string(),
                "LANG".to_string(),
            ],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for shell commands: standard isolation with proxy network.
    pub fn shell_command() -> Self {
        let mut profile = Self::standard();
        profile.id = "standard_shell_command".to_string();
        profile.name = "Shell Command".to_string();
        profile.description = "Shell command execution with standard isolation.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Strict,
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::ProxyAllowlist(vec![]),
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(30),
                memory_bytes: Some(256 * 1024 * 1024),
                max_processes: Some(20),
                file_size_bytes: Some(50 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "USER".to_string(),
                "SHELL".to_string(),
                "LANG".to_string(),
                "TMPDIR".to_string(),
            ],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for read-only file access.
    pub fn file_read() -> Self {
        let mut profile = Self::standard();
        profile.id = "standard_file_read".to_string();
        profile.name = "File Read".to_string();
        profile.description = "Read-only file access with standard isolation.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy::default().with_read_allowed("/"),
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(30),
                memory_bytes: Some(128 * 1024 * 1024),
                max_processes: Some(5),
                file_size_bytes: Some(10 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string(), "HOME".to_string(), "LANG".to_string()],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for file writes: standard isolation, writable workspace.
    pub fn file_write() -> Self {
        let mut profile = Self::standard();
        profile.id = "standard_file_write".to_string();
        profile.name = "File Write".to_string();
        profile.description = "File write access restricted to workspace.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy::default()
                .with_read_allowed("/workspace")
                .with_write_allowed("/workspace")
                .with_write_allowed("/tmp"),
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(30),
                memory_bytes: Some(128 * 1024 * 1024),
                max_processes: Some(5),
                file_size_bytes: Some(100 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string(), "HOME".to_string(), "LANG".to_string()],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for network requests: proxy allowlist, tight memory.
    pub fn network_request() -> Self {
        let mut profile = Self::standard();
        profile.id = "standard_network_request".to_string();
        profile.name = "Network Request".to_string();
        profile.description = "Outbound network via allowlist proxy.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::ProxyAllowlist(vec![]),
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(60),
                memory_bytes: Some(128 * 1024 * 1024),
                max_processes: Some(10),
                file_size_bytes: Some(10 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string(), "HOME".to_string(), "LANG".to_string()],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for git operations: read/write workspace, proxy network for
    /// remotes.
    pub fn git_operation() -> Self {
        let mut profile = Self::standard();
        profile.id = "standard_git_operation".to_string();
        profile.name = "Git Operation".to_string();
        profile.description =
            "Git operations with workspace read/write and proxy network.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Standard,
            filesystem: FilesystemPolicy::default()
                .with_read_allowed("/workspace")
                .with_write_allowed("/workspace")
                .with_read_allowed("/tmp"),
            network: NetworkPolicy::ProxyAllowlist(vec![
                "github.com".to_string(),
                "*.github.com".to_string(),
                "gitlab.com".to_string(),
                "*.gitlab.com".to_string(),
                "bitbucket.org".to_string(),
            ]),
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(120),
                memory_bytes: Some(256 * 1024 * 1024),
                max_processes: Some(20),
                file_size_bytes: Some(500 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "GIT_AUTHOR_NAME".to_string(),
                "GIT_AUTHOR_EMAIL".to_string(),
                "GIT_COMMITTER_NAME".to_string(),
                "GIT_COMMITTER_EMAIL".to_string(),
                "LANG".to_string(),
            ],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for package installation: locked down, requires approval.
    pub fn package_install() -> Self {
        let mut profile = Self::locked();
        profile.id = "locked_package_install".to_string();
        profile.name = "Package Install".to_string();
        profile.description =
            "Package installation: locked isolation, proxy network to registry.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Locked,
            filesystem: FilesystemPolicy::default()
                .with_read_allowed("/usr")
                .with_write_allowed("/workspace")
                .with_write_allowed("/tmp"),
            network: NetworkPolicy::ProxyAllowlist(vec![
                "pypi.org".to_string(),
                "files.pythonhosted.org".to_string(),
                "registry.npmjs.org".to_string(),
                "registry.yarnpkg.com".to_string(),
                "crates.io".to_string(),
                "static.crates.io".to_string(),
                "index.crates.io".to_string(),
            ]),
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(300),
                memory_bytes: Some(512 * 1024 * 1024),
                max_processes: Some(30),
                file_size_bytes: Some(1024 * 1024 * 1024),
                wall_time_secs: Some(300),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "TMPDIR".to_string(),
                "LANG".to_string(),
            ],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for system modifications: maximum isolation, no network.
    pub fn system_modification() -> Self {
        let mut profile = Self::locked();
        profile.id = "locked_system_modification".to_string();
        profile.name = "System Modification".to_string();
        profile.description =
            "System state modification: locked isolation, no network.".to_string();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Locked,
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(10),
                memory_bytes: Some(64 * 1024 * 1024),
                max_processes: Some(5),
                file_size_bytes: Some(10 * 1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec!["PATH".to_string()],
        };
        profile.with_overrides(&overrides)
    }

    /// Profile for credential access: locked, no network, no filesystem.
    pub fn credential_access() -> Self {
        let mut profile = Self::locked();
        profile.id = "locked_credential_access".to_string();
        profile.name = "Credential Access".to_string();
        profile.description =
            "Credential/key access: locked, no network, no filesystem.".to_string();
        let overrides = SandboxPolicy {
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
                    "/root".to_string(),
                    "/home".to_string(),
                ],
                tmp_writable: false,
                home_readable: false,
            },
            network: NetworkPolicy::None,
            resource_limits: ResourceLimits {
                cpu_time_secs: Some(5),
                memory_bytes: Some(32 * 1024 * 1024),
                max_processes: Some(1),
                file_size_bytes: Some(1024 * 1024),
                ..Default::default()
            },
            audit_enabled: true,
            env_allowlist: vec![],
        };
        profile.with_overrides(&overrides)
    }

    /// All built-in profiles in a deterministic order.
    pub fn builtin_profiles() -> Vec<SandboxProfile> {
        vec![
            Self::standard(),
            Self::strict(),
            Self::locked(),
            Self::code_execution(),
            Self::shell_command(),
            Self::file_read(),
            Self::file_write(),
            Self::network_request(),
            Self::git_operation(),
            Self::package_install(),
            Self::system_modification(),
            Self::credential_access(),
        ]
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Look up a builtin profile by id.
    pub fn by_id(id: &str) -> Option<Self> {
        Self::builtin_profiles().into_iter().find(|p| p.id == id)
    }
}

/// Registry of named profiles, allowing runtime registration of custom
/// profiles alongside the builtins.
#[derive(Debug, Clone, Default)]
pub struct ProfileRegistry {
    profiles: Vec<SandboxProfile>,
}

impl ProfileRegistry {
    /// Create a registry preloaded with all builtin profiles.
    pub fn with_builtins() -> Self {
        Self {
            profiles: SandboxProfile::builtin_profiles(),
        }
    }

    /// Register a custom profile. Replaces an existing profile with the same id.
    pub fn register(&mut self, profile: SandboxProfile) {
        if let Some(slot) = self.profiles.iter_mut().find(|p| p.id == profile.id) {
            *slot = profile;
        } else {
            self.profiles.push(profile);
        }
    }

    /// Look up a profile by id.
    pub fn get(&self, id: &str) -> Option<&SandboxProfile> {
        self.profiles.iter().find(|p| p.id == id)
    }

    /// All registered profiles.
    pub fn all(&self) -> &[SandboxProfile] {
        &self.profiles
    }

    /// Resolve a profile for an operation, falling back to the operation's
    /// recommended level.
    pub fn resolve_for_operation(&self, operation: &str) -> SandboxProfile {
        let builtin = SandboxProfile::for_operation(operation);
        // If a registered profile matches the builtin's id, prefer the
        // registered one (it may carry custom overrides).
        self.get(&builtin.id).cloned().unwrap_or(builtin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_profiles_complete() {
        let profiles = SandboxProfile::builtin_profiles();
        assert_eq!(profiles.len(), 12);
        for p in &profiles {
            assert!(!p.id.is_empty());
            assert!(!p.name.is_empty());
            assert!(
                p.policy.validate().is_ok(),
                "profile {} failed validation",
                p.id
            );
        }
    }

    #[test]
    fn for_operation_dispatch() {
        assert_eq!(
            SandboxProfile::for_operation("code_execution").id,
            "strict_code_execution"
        );
        assert_eq!(
            SandboxProfile::for_operation("git").id,
            "standard_git_operation"
        );
        assert_eq!(
            SandboxProfile::for_operation("password").id,
            "locked_credential_access"
        );
    }

    #[test]
    fn registry_lookup() {
        let reg = ProfileRegistry::with_builtins();
        assert!(reg.get("locked").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn override_merges() {
        let base = SandboxProfile::strict();
        let overrides = SandboxPolicy {
            level: SandboxLevel::Locked,
            ..SandboxPolicy::default()
        };
        let merged = base.with_overrides(&overrides);
        assert_eq!(merged.level, SandboxLevel::Locked);
    }
}
