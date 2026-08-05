//! Seccomp filter builder.
//!
//! Builds a deny-by-default [`BpfProgram`] from a list of [`AllowRule`]s. On
//! Linux the heavy lifting is delegated to `seccompiler`; on other platforms
//! the builder produces a structural representation that can be serialised or
//! inspected but would only be installable on Linux.
//!
//! The builder also exposes a [`SeccompPolicy`] enum that bundles the preset
//! allowlists from [`crate::seccomp::presets`] with a default deny action, so
//! callers can do `SeccompPolicy::Strict.compile()` without managing rule
//! lists directly.

use serde::{Deserialize, Serialize};

use crate::policy::SandboxLevel;
use crate::seccomp::presets;
use crate::seccomp::syscalls::syscall_name_to_number;
use crate::seccomp::{AllowRule, ArgComparator, BpfProgram, ComparisonOp, SeccompAction};

/// A seccomp policy: an allowlist plus a default deny action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeccompPolicy {
    /// Human-readable name.
    pub name: String,
    /// The allowlist of syscalls (with optional arg filters).
    pub allowlist: Vec<AllowRule>,
    /// The action taken for any syscall not in the allowlist.
    pub default_action: SeccompAction,
}

impl SeccompPolicy {
    /// The STANDARD preset.
    pub fn standard() -> Self {
        Self {
            name: "standard".to_string(),
            allowlist: presets::standard_allowlist(),
            default_action: SeccompAction::Errno(1),
        }
    }

    /// The STRICT preset.
    pub fn strict() -> Self {
        Self {
            name: "strict".to_string(),
            allowlist: presets::strict_allowlist(),
            default_action: SeccompAction::Errno(1),
        }
    }

    /// The LOCKED preset.
    pub fn locked() -> Self {
        Self {
            name: "locked".to_string(),
            allowlist: presets::locked_allowlist(),
            default_action: SeccompAction::Kill,
        }
    }

    /// Select a preset for a sandbox level.
    pub fn for_level(level: SandboxLevel) -> Self {
        match level {
            SandboxLevel::Standard => Self::standard(),
            SandboxLevel::Strict => Self::strict(),
            SandboxLevel::Locked => Self::locked(),
        }
    }

    /// Build a custom policy from an explicit allowlist.
    pub fn custom(name: impl Into<String>, allowlist: Vec<AllowRule>) -> Self {
        Self {
            name: name.into(),
            allowlist,
            default_action: SeccompAction::Errno(1),
        }
    }

    /// Set the default (deny) action.
    pub fn with_default_action(mut self, action: SeccompAction) -> Self {
        self.default_action = action;
        self
    }

    /// Add an allow rule.
    pub fn allow(mut self, rule: AllowRule) -> Self {
        self.allowlist.push(rule);
        self
    }

    /// Compile the policy into a BPF program.
    ///
    /// On Linux this delegates to `seccompiler` for correctness. On other
    /// platforms it builds a structural program (the same `BpfProgram` type)
    /// that mirrors the Linux layout but is not installable.
    pub fn compile(&self) -> Result<BpfProgram, String> {
        compile_policy(self)
    }

    /// Validate that every syscall name in the allowlist resolves to a number.
    pub fn validate(&self) -> Result<(), String> {
        for rule in &self.allowlist {
            if syscall_name_to_number(&rule.syscall).is_none() {
                return Err(format!("unknown syscall: {}", rule.syscall));
            }
        }
        Ok(())
    }

    /// Number of allow rules.
    pub fn len(&self) -> usize {
        self.allowlist.len()
    }

    /// Is the allowlist empty?
    pub fn is_empty(&self) -> bool {
        self.allowlist.is_empty()
    }
}

/// Builder for incremental construction of a [`SeccompPolicy`].
#[derive(Debug, Clone)]
pub struct SeccompFilterBuilder {
    name: String,
    allowlist: Vec<AllowRule>,
    default_action: SeccompAction,
}

impl SeccompFilterBuilder {
    /// Start a new builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            allowlist: Vec::new(),
            default_action: SeccompAction::Errno(1),
        }
    }

    /// Start from a preset.
    pub fn from_preset(preset: &SeccompPolicy) -> Self {
        Self {
            name: preset.name.clone(),
            allowlist: preset.allowlist.clone(),
            default_action: preset.default_action,
        }
    }

    /// Allow a syscall unconditionally.
    pub fn allow(mut self, syscall: impl Into<String>) -> Self {
        self.allowlist.push(AllowRule::allow(syscall));
        self
    }

    /// Allow a syscall with argument restrictions.
    pub fn allow_with(mut self, syscall: impl Into<String>, args: Vec<ArgComparator>) -> Self {
        self.allowlist.push(AllowRule::allow_with(syscall, args));
        self
    }

    /// Deny a syscall explicitly (remove it from the allowlist). This is a
    /// no-op if the syscall was not allowed.
    pub fn deny(mut self, syscall: &str) -> Self {
        self.allowlist.retain(|r| r.syscall != syscall);
        self
    }

    /// Set the default (deny) action.
    pub fn default_action(mut self, action: SeccompAction) -> Self {
        self.default_action = action;
        self
    }

    /// Build the policy.
    pub fn build(self) -> SeccompPolicy {
        SeccompPolicy {
            name: self.name,
            allowlist: self.allowlist,
            default_action: self.default_action,
        }
    }
}

/// Compile a policy into a BPF program.
///
/// On Linux, uses `seccompiler` to emit a correct, installable filter. On
/// other platforms, emits a structural program (not installable, but useful
/// for testing, inspection and serialisation).
fn compile_policy(policy: &SeccompPolicy) -> Result<BpfProgram, String> {
    // First validate that all names resolve.
    policy.validate()?;

    #[cfg(target_os = "linux")]
    {
        compile_with_seccompiler(policy)
    }
    #[cfg(not(target_os = "linux"))]
    {
        compile_structural(policy)
    }
}

#[cfg(target_os = "linux")]
fn compile_with_seccompiler(policy: &SeccompPolicy) -> Result<BpfProgram, String> {
    use seccompiler::{SeccompAction as SCAction, SeccompFilter, SeccompRule};

    let mut rules: std::collections::HashMap<u64, Vec<SeccompRule>> =
        std::collections::HashMap::new();

    for allow in &policy.allowlist {
        let nr = syscall_name_to_number(&allow.syscall)
            .ok_or_else(|| format!("unknown syscall: {}", allow.syscall))?;
        // seccompiler 0.4: an empty rule list means "allow unconditionally".
        // We pass a single empty rule so the map entry exists.
        let rule = SeccompRule::new(vec![]).map_err(|e| format!("seccomp rule build: {e}"))?;
        rules.insert(nr as u64, vec![rule]);
    }

    let default = match policy.default_action {
        SeccompAction::Allow => SCAction::Allow,
        SeccompAction::Errno(e) => SCAction::Errno(e),
        SeccompAction::Kill | SeccompAction::KillProcess => SCAction::KillProcess,
        SeccompAction::Log => SCAction::Log,
    };

    let filter = SeccompFilter::new(rules, default)
        .map_err(|e| format!("seccomp filter build: {e}"))?;
    let bpf = filter
        .into_bpf()
        .map_err(|e| format!("seccomp bpf compile: {e}"))?;
    // Convert seccompiler's sock_filter into our SeccompInstruction.
    Ok(bpf
        .into_iter()
        .map(|insn| crate::seccomp::SeccompInstruction {
            code: insn.code,
            jt: insn.jt,
            jf: insn.jf,
            k: insn.k,
        })
        .collect())
}

/// Structural compiler used on non-Linux platforms and as a fallback. Produces
/// a minimal BPF program that is not installable but has the right shape for
/// serialisation and testing.
#[cfg(not(target_os = "linux"))]
fn compile_structural(policy: &SeccompPolicy) -> Result<BpfProgram, String> {
    use crate::seccomp::SeccompInstruction;

    // A minimal program: load the syscall number, compare against each
    // allowlisted number, and jump to allow or default. This is a sketch, not
    // a correct BPF program — it exists so the type can be constructed and
    // serialised on non-Linux.
    let mut prog = Vec::new();
    // BPF_LD | BPF_W | BPF_ABS  (load syscall nr at offset 0)
    prog.push(SeccompInstruction {
        code: 0x0020,
        jt: 0,
        jf: 0,
        k: 0,
    });
    let allow_count = policy.allowlist.len() as u32;
    for allow in &policy.allowlist {
        if let Some(nr) = syscall_name_to_number(&allow.syscall) {
            // BPF_JMP | BPF_JEQ | BPF_K  (compare to syscall nr)
            prog.push(SeccompInstruction {
                code: 0x0015,
                jt: 1, // jump to allow on match
                jf: 0, // fall through on mismatch
                k: nr as u32,
            });
        }
    }
    // Default action.
    let default_k = match policy.default_action {
        SeccompAction::Allow => 0x7fff_0000,
        SeccompAction::Errno(e) => 0x0005_0000 | (e as u32),
        SeccompAction::Kill | SeccompAction::KillProcess => 0,
        SeccompAction::Log => 0x7ffc_0000,
    };
    // RET default.
    prog.push(SeccompInstruction {
        code: 0x0006,
        jt: 0,
        jf: 0,
        k: default_k,
    });
    // RET allow (reached by the jt=1 jumps). seccomp return-allow.
    prog.push(SeccompInstruction {
        code: 0x0006,
        jt: 0,
        jf: 0,
        k: 0x7fff_0000,
    });
    let _ = allow_count; // referenced for clarity
    Ok(prog)
}

/// Serialise a BPF program into the kernel `sock_fprog` layout (a `u16` count
/// followed by the filter instructions). Used by the Linux backend to pass the
/// filter to `bwrap --seccomp` via a memfd.
pub fn serialize_sock_fprog(prog: &BpfProgram) -> Vec<u8> {
    let ptr_size = std::mem::size_of::<usize>();
    let filter_offset = align_up(2, ptr_size);
    let header_size = filter_offset + ptr_size;

    let mut insns = Vec::with_capacity(prog.len() * 8);
    for insn in prog {
        insns.extend_from_slice(&insn.code.to_le_bytes());
        insns.push(insn.jt);
        insns.push(insn.jf);
        insns.extend_from_slice(&insn.k.to_le_bytes());
    }

    let mut buf = Vec::with_capacity(header_size + insns.len());
    buf.extend_from_slice(&(prog.len() as u16).to_le_bytes());
    buf.extend(std::iter::repeat(0u8).take(filter_offset - 2));
    buf.extend(std::iter::repeat(0u8).take(ptr_size)); // filter pointer (patched later)
    buf.extend_from_slice(&insns);
    buf
}

fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// Describe a compiled program as a human-readable summary (for audit logs).
pub fn describe_program(policy: &SeccompPolicy) -> String {
    let mut sorted: Vec<&str> = policy.allowlist.iter().map(|r| r.syscall.as_str()).collect();
    sorted.sort();
    sorted.dedup();
    format!(
        "seccomp policy '{}' ({} allowed syscalls, default={:?})",
        policy.name,
        sorted.len(),
        policy.default_action
    )
}

/// Check whether a given syscall would be allowed by a policy. Pure helper for
/// audit/diagnostics (does not consult arg filters).
pub fn is_allowed(policy: &SeccompPolicy, syscall_name: &str) -> bool {
    policy.allowlist.iter().any(|r| r.syscall == syscall_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate() {
        SeccompPolicy::standard().validate().unwrap();
        SeccompPolicy::strict().validate().unwrap();
        SeccompPolicy::locked().validate().unwrap();
    }

    #[test]
    fn builder_allow_deny() {
        let policy = SeccompFilterBuilder::new("test")
            .allow("read")
            .allow("write")
            .deny("write")
            .build();
        assert!(is_allowed(&policy, "read"));
        assert!(!is_allowed(&policy, "write"));
    }

    #[test]
    fn for_level_dispatch() {
        assert_eq!(SeccompPolicy::for_level(SandboxLevel::Standard).name, "standard");
        assert_eq!(SeccompPolicy::for_level(SandboxLevel::Locked).name, "locked");
    }

    #[test]
    fn unknown_syscall_fails_validation() {
        let policy = SeccompPolicy::custom("bad", vec![AllowRule::allow("not_a_syscall")]);
        assert!(policy.validate().is_err());
    }

    #[test]
    fn serialize_produces_header() {
        let policy = SeccompPolicy::standard();
        let prog = policy.compile().unwrap();
        let bytes = serialize_sock_fprog(&prog);
        // The first two bytes are the little-endian instruction count.
        let count = u16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(count as usize, prog.len());
    }
}
