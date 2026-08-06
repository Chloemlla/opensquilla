//! # OpenSquilla Sandbox
//!
//! Multi-platform sandbox for secure subprocess execution with policy-based
//! isolation. Supports Linux (bubblewrap + seccomp + nix namespaces), macOS
//! (sandbox-exec/Seatbelt), Windows (restricted tokens + Job Objects +
//! Firewall/WFP), and a noop fallback for development.
//!
//! Every platform backend implements the [`Sandbox`] trait so callers can
//! dispatch through [`default_sandbox()`] without platform `cfg` blocks. The
//! concrete types (e.g. [`LinuxSandbox`]) additionally keep their inherent
//! `execute` methods for callers that select a backend explicitly.

pub mod command_rules;
pub mod config;
pub mod default_allowlist;
pub mod denial_attribution;
pub mod destructive_intents;
pub mod directory_listing;
pub mod domain_validation;
pub mod error;
pub mod governance;
pub mod linux;
pub mod macos;
pub mod managed_proxy_env;
pub mod metrics;
pub mod network;
pub mod noop;
pub mod package_bundles;
pub mod path_rules;
pub mod policy;
pub mod profile;
pub mod run_mode;
pub mod run_mode_policy;
pub mod sandbox_manager;
pub mod seatbelt;
pub mod seccomp;
pub mod sensitive_paths;
pub mod stale_output_cache;
pub mod supervisor;
pub mod whitelist;
pub mod windows;

pub use command_rules::{
    CommandAssessment, CommandRule, RiskTier, assess_command, assess_with_rules,
    command_basename, default_command_rules, profile_id_for_command,
};
pub use config::{
    ApprovalsReviewer, Backend, EffectiveMode, NetworkDefault, SandboxSettings, SecurityLevel,
};
pub use default_allowlist::{
    DefaultAllowlistGroup, default_allowlist, default_allowlist_domains,
    default_allowlist_network_policy, default_allowlist_payload, default_allowlist_source,
};
pub use denial_attribution::{SandboxRunOutcome, is_likely_sandbox_denied};
pub use destructive_intents::{extract_intent, extract_intents};
pub use directory_listing::format_directory_entry;
pub use domain_validation::{DomainDecision, DomainStatus, domain_matches, normalize_domain, validate_domain_pattern};
pub use error::{SandboxError, SandboxErrorKind};
pub use governance::{
    ApprovalQueue, ApprovalRequest, ApprovalStatus, ApproverChannel, EscalationPolicy,
    GovernanceAuditEntry, GovernanceAuditTrail, GovernanceCoordinator, GovernanceDecision,
    GovernanceEvent, GovernanceEventKind, GovernanceMetrics, RejectionEntry,
};
pub use linux::LinuxSandbox;
pub use macos::MacOsSandbox;
pub use managed_proxy_env::{
    ALLOW_LOCAL_BINDING_ENV_KEY, DEFAULT_NO_PROXY_VALUE, NO_PROXY_ENV_KEYS,
    OPENSQUILLA_NETWORK_ENV_KEY, PROXY_ACTIVE_ENV_KEY, PROXY_CONTROL_ENV, PROXY_ENV_KEYS,
    WINDOWS_GIT_SSL_ENV, extend_env_allowlist_with_proxy_vars, managed_proxy_env,
    managed_proxy_env_allowlist, managed_proxy_env_for_backend, managed_proxy_env_names_upper,
};
pub use metrics::{
    AggregateMetrics, ExecutionMetrics, MetricsCollector, Rusage, SyscallClass, SyscallCounter,
};
pub use network::{
    DnsChecker, DomainAllowlist, DomainCheck, IpRange, NetworkConfig, NetworkMode, NetworkProxy,
    ProxyAuditEntry, ProxyHandle, ProxyRequestLog, RateLimiter, RequestLogBuffer, TokenBucket,
    default_blocked_ranges,
};
pub use noop::NoopSandbox;
pub use package_bundles::{
    PACKAGE_BUNDLES, default_package_bundle_ids, expand_package_bundle, package_bundle_domains,
};
pub use path_rules::{
    PathAccessController, PathDecision, is_readable, is_writable, whitelist_from_filesystem,
};
pub use policy::{
    AuditEntry, FilesystemPolicy, NetworkPolicy, OperationClass, PolicyValidationError,
    ResourceLimits, SandboxLevel, SandboxPolicy, SandboxResult, classify_operation,
};
pub use profile::{ProfileNotes, ProfileRegistry, SandboxProfile};
pub use run_mode::{
    RunMode, RunModeConfigInput, RunModeConfigPatch, RunModeError, approval_behavior,
    config_run_mode, display_name, execution_target, full_mode_is_explicit,
    legacy_state_to_run_mode, normalize_run_mode, project_default_run_mode,
    run_mode_config_patch, sandbox_runtime_capability_mode,
};
pub use run_mode_policy::{
    Principal, allowed_run_modes_for_principal, coerce_run_mode_for_principal,
    default_run_mode_for_principal, hello_auth_payload, principal_payload, run_mode_allowed_for_principal,
    run_mode_policy_payload,
};
pub use sandbox_manager::{
    ManagedSandbox, RunRequest, SandboxBuilder, SandboxManager, SandboxOutcome,
};
pub use seatbelt::operations::{SeatbeltCategory, SeatbeltOperation};
pub use seatbelt::SeatbeltProfile;
pub use seccomp::{
    AllowRule, ArgComparator, BpfProgram, ComparisonOp, SeccompAction, SeccompInstruction,
    SeccompPolicy, SeccompFilterBuilder, syscall_name_to_number,
};
pub use sensitive_paths::{
    build_block_envelope, is_sensitive_path, linux_runtime_sensitive_deny_roots,
    sensitive_path_in_text, sensitive_path_marker, sensitive_target_in_command,
};
pub use stale_output_cache::{
    CacheEntry, NullStaleOutputCache, StaleOutputCache, TtlPolicy, VerifiedEntry,
    VerifiedOutputCache, content_hash,
};
pub use supervisor::{DenialTracker, ResourcePoller, ResourceSample, SpawnOptions, SupervisedChild, SupervisedRun, spawn_supervised};
pub use whitelist::{AccessIntent, AccessMode, AccessVerdict, PathRule, PathWhitelist};
pub use windows::WindowsSandbox;

use std::collections::HashMap;

/// A unified sandbox backend interface implemented by every platform backend.
///
/// The trait is intentionally thin and synchronous-friendly: each backend
/// performs its own subprocess management, policy enforcement and audit
/// capture, and returns the same [`SandboxResult`] shape.
#[async_trait::async_trait]
pub trait Sandbox: Send + Sync {
    /// Execute a command inside the sandbox.
    async fn execute(
        &mut self,
        command: &str,
        args: &[&str],
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String>;

    /// Execute a command inside the sandbox with an explicit environment and
    /// working directory. The `env` map is intersected with the policy's
    /// environment allowlist before being passed to the child.
    async fn execute_with_env(
        &mut self,
        command: &str,
        args: &[&str],
        env: HashMap<String, String>,
        working_dir: Option<&str>,
        policy: &SandboxPolicy,
    ) -> Result<SandboxResult, String>;

    /// Return `true` when the underlying sandbox primitives are available and
    /// believed operational (e.g. the `bwrap` binary exists, or a Job Object
    /// can be created).
    fn health_check(&self) -> bool;

    /// A stable, human-readable backend name (`"linux"`, `"macos"`,
    /// `"windows"`, `"noop"`).
    fn name(&self) -> &'static str;

    /// A snapshot of the audit entries recorded by this backend so far.
    fn audit_log(&self) -> Vec<AuditEntry>;
}

/// Return the platform-appropriate sandbox backend as a boxed trait object.
///
/// This is the primary entry point for callers that do not care about the
/// concrete platform type; the returned object dispatches to the Linux,
/// macOS, Windows or noop backend based on the compile target.
pub fn default_sandbox() -> Box<dyn Sandbox> {
    #[cfg(target_os = "linux")]
    {
        Box::new(LinuxSandbox::new())
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(MacOsSandbox::new())
    }
    #[cfg(target_os = "windows")]
    {
        Box::new(WindowsSandbox::new())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Box::new(NoopSandbox::new())
    }
}
