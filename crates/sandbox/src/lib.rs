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

pub mod error;
pub mod governance;
pub mod linux;
pub mod macos;
pub mod network;
pub mod noop;
pub mod policy;
pub mod stale_output_cache;
pub mod windows;

pub use error::{SandboxError, SandboxErrorKind};
pub use governance::{
    ApprovalQueue, ApprovalRequest, ApprovalStatus, GovernanceEvent, GovernanceEventKind,
    GovernanceMetrics, RejectionEntry,
};
pub use linux::LinuxSandbox;
pub use macos::MacOsSandbox;
pub use network::{IpRange, NetworkConfig, NetworkMode, NetworkProxy, ProxyAuditEntry, ProxyHandle};
pub use noop::NoopSandbox;
pub use policy::{
    AuditEntry, FilesystemPolicy, NetworkPolicy, OperationClass, PolicyValidationError,
    ResourceLimits, SandboxLevel, SandboxPolicy, SandboxResult,
};
pub use stale_output_cache::{CacheEntry, NullStaleOutputCache, StaleOutputCache};
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
/// concrete platform type:
///
/// ```
/// let mut sb = opensquilla_sandbox::default_sandbox();
/// let policy = opensquilla_sandbox::SandboxPolicy::default();
/// let _ = sb.execute("echo", &["hi"], &policy).await;
/// ```
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
