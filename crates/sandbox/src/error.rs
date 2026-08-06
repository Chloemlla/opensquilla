//! Sandbox error types.
//!
//! The public `execute` APIs return `String` errors to stay compatible with
//! the rest of the workspace (which threads `String` through the RPC and CLI
//! layers). Internally, platform backends can use the structured
//! [`SandboxError`] below; it converts losslessly into a `String` for the
//! public boundary.

use std::fmt;

/// Kinds of failures the sandbox subsystem can encounter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxErrorKind {
    /// The backend is not available on the current platform (e.g. a Linux
    /// backend compiled for macOS).
    UnsupportedPlatform,
    /// The sandbox primitive could not be created (namespace, job object,
    /// restricted token, seccomp filter, …).
    SetupFailed,
    /// The subprocess could not be spawned.
    SpawnFailed,
    /// The subprocess exceeded its configured timeout.
    Timeout,
    /// The subprocess exited with a non-zero status.
    NonZeroExit,
    /// A policy rule rejected the execution before it started.
    PolicyDenied,
    /// Network policy enforcement failed.
    Network,
    /// The filesystem layout was rejected (mounts, paths, …).
    Filesystem,
    /// I/O or internal error.
    Io,
    /// The command or binary could not be resolved/executed.
    CommandNotFound,
    /// Resource limits were exceeded by the sandboxed process.
    ResourceExceeded,
    /// A governance approval was required but not granted (or still pending).
    ApprovalRequired,
    /// An approval was rejected (recorded in the rejection ledger).
    ApprovalRejected,
    /// The sandbox profile is malformed or references an unknown profile id.
    InvalidProfile,
    /// The seccomp filter could not be compiled or installed.
    Seccomp,
    /// The SBPL/Seatbelt profile could not be compiled or applied.
    Seatbelt,
    /// The cgroup or container runtime failed.
    Cgroup,
    /// The operation was cancelled (shutdown, dropped request).
    Cancelled,
}

impl SandboxErrorKind {
    /// Short machine-readable tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            SandboxErrorKind::UnsupportedPlatform => "unsupported_platform",
            SandboxErrorKind::SetupFailed => "setup_failed",
            SandboxErrorKind::SpawnFailed => "spawn_failed",
            SandboxErrorKind::Timeout => "timeout",
            SandboxErrorKind::NonZeroExit => "non_zero_exit",
            SandboxErrorKind::PolicyDenied => "policy_denied",
            SandboxErrorKind::Network => "network",
            SandboxErrorKind::Filesystem => "filesystem",
            SandboxErrorKind::Io => "io",
            SandboxErrorKind::CommandNotFound => "command_not_found",
            SandboxErrorKind::ResourceExceeded => "resource_exceeded",
            SandboxErrorKind::ApprovalRequired => "approval_required",
            SandboxErrorKind::ApprovalRejected => "approval_rejected",
            SandboxErrorKind::InvalidProfile => "invalid_profile",
            SandboxErrorKind::Seccomp => "seccomp",
            SandboxErrorKind::Seatbelt => "seatbelt",
            SandboxErrorKind::Cgroup => "cgroup",
            SandboxErrorKind::Cancelled => "cancelled",
        }
    }

    /// An HTTP-style status hint for the RPC boundary (informational).
    pub fn http_status(&self) -> u16 {
        match self {
            SandboxErrorKind::UnsupportedPlatform => 501,
            SandboxErrorKind::SetupFailed => 500,
            SandboxErrorKind::SpawnFailed => 500,
            SandboxErrorKind::Timeout => 504,
            SandboxErrorKind::NonZeroExit => 422,
            SandboxErrorKind::PolicyDenied | SandboxErrorKind::ApprovalRejected => 403,
            SandboxErrorKind::ApprovalRequired => 402,
            SandboxErrorKind::Network => 502,
            SandboxErrorKind::Filesystem => 400,
            SandboxErrorKind::Io => 500,
            SandboxErrorKind::CommandNotFound => 404,
            SandboxErrorKind::ResourceExceeded => 429,
            SandboxErrorKind::InvalidProfile => 400,
            SandboxErrorKind::Seccomp | SandboxErrorKind::Seatbelt | SandboxErrorKind::Cgroup => 500,
            SandboxErrorKind::Cancelled => 499,
        }
    }
}

impl fmt::Display for SandboxErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured sandbox error. Kept deliberately small; most callers convert
/// to `String` at the API boundary.
#[derive(Debug, Clone)]
pub struct SandboxError {
    /// Failure category.
    pub kind: SandboxErrorKind,
    /// Human-readable detail.
    pub message: String,
    /// Optional exit code of a failed child process.
    pub exit_code: Option<i32>,
    /// Optional retry hint (e.g. "increase memory limit").
    pub hint: Option<String>,
}

impl SandboxError {
    /// Create a new sandbox error.
    pub fn new(kind: SandboxErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            exit_code: None,
            hint: None,
        }
    }

    /// Attach an exit code.
    pub fn with_exit_code(mut self, code: i32) -> Self {
        self.exit_code = Some(code);
        self
    }

    /// Attach a retry hint.
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Convenience constructor for `SetupFailed`.
    pub fn setup(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::SetupFailed, msg)
    }

    /// Convenience constructor for `SpawnFailed`.
    pub fn spawn(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::SpawnFailed, msg)
    }

    /// Convenience constructor for `Timeout`.
    pub fn timeout(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::Timeout, msg)
    }

    /// Convenience constructor for `PolicyDenied`.
    pub fn denied(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::PolicyDenied, msg)
    }

    /// Convenience constructor for `Network`.
    pub fn network(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::Network, msg)
    }

    /// Convenience constructor for `Filesystem`.
    pub fn filesystem(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::Filesystem, msg)
    }

    /// Convenience constructor for `CommandNotFound`.
    pub fn command_not_found(command: impl AsRef<str>) -> Self {
        Self::new(
            SandboxErrorKind::CommandNotFound,
            format!("command not found: '{}'", command.as_ref()),
        )
    }

    /// Convenience constructor for `ApprovalRejected`.
    pub fn approval_rejected(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::ApprovalRejected, msg)
    }

    /// Convenience constructor for `ResourceExceeded`.
    pub fn resource_exceeded(msg: impl Into<String>) -> Self {
        Self::new(SandboxErrorKind::ResourceExceeded, msg)
    }

    /// The machine-readable tag.
    pub fn tag(&self) -> &'static str {
        self.kind.as_str()
    }
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.message)
    }
}

impl std::error::Error for SandboxError {}

impl From<SandboxError> for String {
    fn from(err: SandboxError) -> Self {
        err.to_string()
    }
}

impl From<std::io::Error> for SandboxError {
    fn from(e: std::io::Error) -> Self {
        Self::new(SandboxErrorKind::Io, e.to_string())
    }
}

impl From<serde_json::Error> for SandboxError {
    fn from(e: serde_json::Error) -> Self {
        Self::new(SandboxErrorKind::Io, format!("JSON: {e}"))
    }
}
