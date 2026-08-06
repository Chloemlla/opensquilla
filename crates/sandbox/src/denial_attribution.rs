//! Narrow Codex-compatible attribution of command failure to the sandbox.
//!
//! Port of `src/opensquilla/sandbox/denial_attribution.py`. A conservative
//! heuristic deciding whether a failed command was likely denied by the
//! sandbox (rather than failing for an unrelated reason). The heuristic is
//! deliberately biased: it returns `false` whenever there is any doubt, so
//! genuine sandbox denials are not confused with application bugs.

/// The outcome of a sandboxed run, used by [`is_likely_sandbox_denied`].
#[derive(Debug, Clone)]
pub struct SandboxRunOutcome {
    /// Process exit code.
    pub returncode: i32,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// The backend that ran the command (`"bwrap"`, `"seatbelt"`, `"noop"`,
    /// `""`, ...).
    pub backend_used: String,
    /// Backend-provided notes; a non-empty set is treated as direct evidence
    /// of a sandbox denial.
    pub backend_notes: Vec<String>,
}

impl SandboxRunOutcome {
    /// Convenience constructor for callers holding a policy-level result.
    pub fn new(
        returncode: i32,
        stdout: impl Into<String>,
        stderr: impl Into<String>,
        backend_used: impl Into<String>,
    ) -> Self {
        Self {
            returncode,
            stdout: stdout.into(),
            stderr: stderr.into(),
            backend_used: backend_used.into(),
            backend_notes: Vec::new(),
        }
    }
}

const DENIED_KEYWORDS: &[&str] = &[
    "operation not permitted",
    "permission denied",
    "read-only file system",
    "seccomp",
    "sandbox",
    "landlock",
    "failed to write file",
];

const QUICK_REJECT_EXIT_CODES: &[i32] = &[2, 126, 127];

const UNSANDBOXED_BACKENDS: &[&str] = &["", "noop", "none", "host"];

/// Return `true` when the failed run was likely denied by the sandbox.
///
/// Semantics (mirroring the Python port):
/// 1. Backends `""`/`noop`/`none`/`host` are never attributed.
/// 2. Non-empty `backend_notes` are treated as direct evidence.
/// 3. A zero exit code is never attributed.
/// 4. Output containing a denial keyword (seccomp, permission denied, ...) is
///    attributed.
/// 5. Exit codes 2/126/127 are "quick rejects" and never attributed.
/// 6. A `128 + SIGSYS` exit code (process killed by a seccomp violation) is
///    attributed, on platforms where SIGSYS exists.
pub fn is_likely_sandbox_denied(result: &SandboxRunOutcome) -> bool {
    let backend_used = result.backend_used.trim().to_lowercase();
    if UNSANDBOXED_BACKENDS.contains(&backend_used.as_str()) {
        return false;
    }
    if !result.backend_notes.is_empty() {
        return true;
    }
    if result.returncode == 0 {
        return false;
    }
    let combined = format!("{}\n{}", result.stderr, result.stdout).to_lowercase();
    if DENIED_KEYWORDS.iter().any(|k| combined.contains(k)) {
        return true;
    }
    if QUICK_REJECT_EXIT_CODES.contains(&result.returncode) {
        return false;
    }
    match sigsys_signal() {
        Some(sigsys) => result.returncode == 128 + sigsys,
        None => false,
    }
}

/// SIGSYS (31) exists on POSIX platforms; it is absent on Windows.
fn sigsys_signal() -> Option<i32> {
    #[cfg(target_os = "windows")]
    {
        None
    }
    #[cfg(not(target_os = "windows"))]
    {
        Some(31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(returncode: i32, stdout: &str, stderr: &str, backend: &str) -> SandboxRunOutcome {
        SandboxRunOutcome::new(returncode, stdout, stderr, backend)
    }

    #[test]
    fn unsandboxed_backends_never_attributed() {
        for backend in ["", "noop", "none", "host", "  NoOp  "] {
            let r = outcome(1, "", "permission denied", backend);
            assert!(!is_likely_sandbox_denied(&r), "for backend '{backend}'");
        }
    }

    #[test]
    fn backend_notes_are_direct_evidence() {
        let mut r = outcome(1, "", "", "bwrap");
        r.backend_notes = vec!["mount failed: operation not permitted".to_string()];
        assert!(is_likely_sandbox_denied(&r));
    }

    #[test]
    fn zero_exit_is_not_denial() {
        let r = outcome(0, "ok", "", "bwrap");
        assert!(!is_likely_sandbox_denied(&r));
    }

    #[test]
    fn denied_keywords_attributed() {
        for keyword in [
            "operation not permitted",
            "Permission Denied",
            "read-only file system",
            "seccomp",
            "sandbox",
            "landlock",
            "failed to write file",
        ] {
            let r = outcome(1, "", keyword, "bwrap");
            assert!(is_likely_sandbox_denied(&r), "for keyword '{keyword}'");
        }
    }

    #[test]
    fn quick_reject_codes_never_attributed() {
        for code in [2, 126, 127] {
            let r = outcome(code, "", "not found", "bwrap");
            assert!(!is_likely_sandbox_denied(&r), "for code {code}");
        }
    }

    #[test]
    fn seccomp_kill_attributed_on_posix() {
        let r = outcome(128 + 31, "", "", "bwrap");
        if sigsys_signal().is_some() {
            assert!(is_likely_sandbox_denied(&r));
        }
    }

    #[test]
    fn unrelated_failure_not_attributed() {
        let r = outcome(1, "", "TypeError: foo is not defined", "bwrap");
        assert!(!is_likely_sandbox_denied(&r));
    }
}
