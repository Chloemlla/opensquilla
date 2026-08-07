//! Linux seccomp-BPF filter generation for syscall allowlisting.
//!
//! This module generates deny-by-default seccomp-BPF filters that allowlist a
//! curated set of syscalls, optionally with argument-level restrictions. The
//! compiled [`BpfProgram`] is consumed by the Linux backend (`bwrap --seccomp`
//! or `prctl(PR_SET_SECCOMP)`).
//!
//! Three preset policies are provided:
//!
//! - [`SeccompPolicy::standard()`] — a broad allowlist suitable for general
//!   programs (glibc, musl, Rust, Go runtimes).
//! - [`SeccompPolicy::strict()`] — the standard set minus network syscalls,
//!   `ptrace`, `clone` with `CLONE_NEW*` flags, and a few others.
//! - [`SeccompPolicy::locked()`] — the strict set minus filesystem write
//!   syscalls and `socket`/`connect`/`bind` entirely.
//!
//! Custom policies can be built via [`SeccompPolicy::custom`].

pub mod builder;
pub mod presets;
pub mod syscalls;

pub use builder::{SeccompFilterBuilder, SeccompPolicy};
pub use presets::preset_allowlist;
pub use syscalls::{SYSCALL_NAMES, syscall_name_to_number};

use serde::{Deserialize, Serialize};

/// The action taken when a syscall matches a filter rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeccompAction {
    /// Allow the syscall.
    Allow,
    /// Deny the syscall and return `errno`.
    Errno(u16),
    /// Kill the process.
    Kill,
    /// Kill the process with a specific signal.
    KillProcess,
    /// Log the syscall and allow it.
    Log,
}

impl Default for SeccompAction {
    fn default() -> Self {
        SeccompAction::Errno(1)
    }
}

/// A compiled seccomp-BPF program (a list of BPF instructions).
pub type BpfProgram = Vec<SeccompInstruction>;

/// A single BPF instruction in the seccomp filter.
///
/// This is a mirror of `struct sock_filter` so the program can be serialised
/// to the kernel layout without depending on `seccompiler` at the public API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeccompInstruction {
    /// Opcode.
    pub code: u16,
    /// Jump-true target (instruction offset).
    pub jt: u8,
    /// Jump-false target (instruction offset).
    pub jf: u8,
    /// Generic argument (constant or field selector).
    pub k: u32,
}

/// A named allowlist entry: syscall name plus optional argument restrictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowRule {
    /// The syscall name (e.g. `"read"`, `"openat"`).
    pub syscall: String,
    /// Optional argument comparators. All must match for the rule to allow.
    pub args: Vec<ArgComparator>,
}

impl AllowRule {
    /// Allow a syscall unconditionally.
    pub fn allow(syscall: impl Into<String>) -> Self {
        Self {
            syscall: syscall.into(),
            args: Vec::new(),
        }
    }

    /// Allow a syscall with argument restrictions.
    pub fn allow_with(syscall: impl Into<String>, args: Vec<ArgComparator>) -> Self {
        Self {
            syscall: syscall.into(),
            args,
        }
    }
}

/// A comparison on a single syscall argument.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ArgComparator {
    /// Argument index (0–5).
    pub index: u8,
    /// The comparison operator.
    pub op: ComparisonOp,
    /// The value to compare against.
    pub value: u64,
    /// Whether `value` is treated as a 32-bit quantity.
    pub is_32bit: bool,
}

impl ArgComparator {
    /// Create a 64-bit equality comparison.
    pub fn eq(index: u8, value: u64) -> Self {
        Self {
            index,
            op: ComparisonOp::Equal,
            value,
            is_32bit: false,
        }
    }

    /// Create a 32-bit equality comparison.
    pub fn eq32(index: u8, value: u32) -> Self {
        Self {
            index,
            op: ComparisonOp::Equal,
            value: value as u64,
            is_32bit: true,
        }
    }

    /// Create a masked-equality comparison (the argument AND `mask` equals
    /// `value`).
    pub fn masked_eq(index: u8, mask: u64, value: u64) -> Self {
        Self {
            index,
            op: ComparisonOp::MaskedEqual(mask),
            value,
            is_32bit: false,
        }
    }
}

/// Comparison operators supported by seccomp-BPF.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// Argument == value.
    Equal,
    /// Argument > value.
    GreaterThan,
    /// Argument >= value.
    GreaterOrEqual,
    /// Argument < value.
    LessThan,
    /// Argument <= value.
    LessOrEqual,
    /// (Argument & mask) == value.
    MaskedEqual(u64),
}
