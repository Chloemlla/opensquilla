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
}

impl SandboxError {
    /// Create a new sandbox error.
    pub fn new(kind: SandboxErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
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
