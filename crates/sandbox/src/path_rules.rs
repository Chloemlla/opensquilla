//! Filesystem policy → path-whitelist bridge.
//!
//! [`FilesystemPolicy`] uses simple prefix matching; [`crate::whitelist::PathWhitelist`]
//! supports glob patterns with deny-precedence. This module bridges the two:
//! it compiles a [`FilesystemPolicy`] into a [`PathWhitelist`] and provides a
//! higher-level [`PathAccessController`] that consults both, resolving
//! symlinks before deciding.

use crate::policy::FilesystemPolicy;
use crate::whitelist::{AccessIntent, AccessMode, AccessVerdict, PathRule, PathWhitelist};
use std::path::PathBuf;

/// Compile a [`FilesystemPolicy`] into a [`PathWhitelist`].
///
/// Mapping:
/// - every `write_allowed` path becomes an allow rule with [`AccessMode::ReadWrite`],
/// - every `read_allowed` path becomes an allow rule with [`AccessMode::Read`],
/// - every `denied` path becomes a deny rule (deny always wins),
/// - `/tmp` becomes writable when `tmp_writable`,
/// - `$HOME` becomes readable when `home_readable`.
pub fn whitelist_from_filesystem(policy: &FilesystemPolicy) -> PathWhitelist {
    let mut wl = PathWhitelist::new();

    for path in &policy.write_allowed {
        if !path.is_empty() {
            wl.add(PathRule::allow(format!("{}/**", trim_trailing(path)), AccessMode::ReadWrite)
                .with_reason("filesystem policy write_allowed"));
            wl.add(PathRule::allow(path.clone(), AccessMode::ReadWrite)
                .with_reason("filesystem policy write_allowed (root)"));
        }
    }
    for path in &policy.read_allowed {
        if !path.is_empty() {
            wl.add(PathRule::allow(format!("{}/**", trim_trailing(path)), AccessMode::Read)
                .with_reason("filesystem policy read_allowed"));
            wl.add(PathRule::allow(path.clone(), AccessMode::Read)
                .with_reason("filesystem policy read_allowed (root)"));
        }
    }
    for path in &policy.denied {
        if !path.is_empty() {
            wl.add(PathRule::deny(format!("{}/**", trim_trailing(path)))
                .with_reason("filesystem policy denied"));
            wl.add(PathRule::deny(path.clone()).with_reason("filesystem policy denied (root)"));
        }
    }
    if policy.tmp_writable {
        wl.add(PathRule::allow("/tmp/**", AccessMode::ReadWrite).with_reason("tmp writable"));
        wl.add(PathRule::allow("/tmp", AccessMode::ReadWrite).with_reason("tmp writable (root)"));
    }
    if policy.home_readable {
        if let Some(home) = std::env::var_os("HOME") {
            let home = home.to_string_lossy().to_string();
            wl.add(PathRule::allow(format!("{home}/**"), AccessMode::Read)
                .with_reason("home readable"));
            wl.add(PathRule::allow(home, AccessMode::Read).with_reason("home readable (root)"));
        }
    }
    wl
}

fn trim_trailing(path: &str) -> &str {
    path.trim_end_matches(['/', '\\'])
}

/// A combined path-access decision that merges the policy and the whitelist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathDecision {
    /// Access is allowed.
    Allow(AccessMode),
    /// Access is denied by the whitelist deny rules.
    Deny(String),
    /// Access falls back to the policy's prefix matching.
    PolicyFallback(bool),
}

impl PathDecision {
    /// Is access permitted?
    pub fn permitted(&self) -> bool {
        match self {
            PathDecision::Allow(_) => true,
            PathDecision::Deny(_) => false,
            PathDecision::PolicyFallback(ok) => *ok,
        }
    }

    /// The effective access mode when allowed.
    pub fn mode(&self) -> Option<AccessMode> {
        match self {
            PathDecision::Allow(m) => Some(*m),
            _ => None,
        }
    }
}

/// A controller that evaluates paths against both the filesystem policy and
/// the compiled whitelist.
#[derive(Debug, Clone)]
pub struct PathAccessController {
    policy: FilesystemPolicy,
    whitelist: PathWhitelist,
}

impl PathAccessController {
    /// Create a controller from a filesystem policy.
    pub fn new(policy: FilesystemPolicy) -> Self {
        let whitelist = whitelist_from_filesystem(&policy);
        Self { policy, whitelist }
    }

    /// Create from a policy reference.
    pub fn from_policy(policy: &FilesystemPolicy) -> Self {
        Self::new(policy.clone())
    }

    /// Check read access for a path (resolving symlinks first).
    pub fn check_read(&self, path: &str) -> PathDecision {
        let verdict = self.check_on_disk(path, AccessIntent::Read);
        match verdict {
            AccessVerdict::Allow(mode) => PathDecision::Allow(mode),
            AccessVerdict::Deny(reason) => PathDecision::Deny(reason),
            AccessVerdict::NoMatch => PathDecision::PolicyFallback(self.policy.allows_read(path)),
        }
    }

    /// Check write access for a path (resolving symlinks first).
    pub fn check_write(&self, path: &str) -> PathDecision {
        let verdict = self.check_on_disk(path, AccessIntent::Write);
        match verdict {
            AccessVerdict::Allow(mode) => PathDecision::Allow(mode),
            AccessVerdict::Deny(reason) => PathDecision::Deny(reason),
            AccessVerdict::NoMatch => PathDecision::PolicyFallback(self.policy.allows_write(path)),
        }
    }

    fn check_on_disk(&self, path: &str, intent: AccessIntent) -> AccessVerdict {
        // Resolve symlinks so an escape through a link cannot bypass the rules.
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
        match intent {
            AccessIntent::Read => self.whitelist.check_read(&canonical.to_string_lossy()),
            AccessIntent::Write => self.whitelist.check_write(&canonical.to_string_lossy()),
        }
    }

    /// The underlying whitelist (for audit/diagnostics).
    pub fn whitelist(&self) -> &PathWhitelist {
        &self.whitelist
    }

    /// The underlying policy.
    pub fn policy(&self) -> &FilesystemPolicy {
        &self.policy
    }

    /// A summary of the compiled rule set (for audit logs).
    pub fn summary(&self) -> String {
        format!(
            "path rules: {} whitelist rules (fs: {} read, {} write, {} denied)",
            self.whitelist.len(),
            self.policy.read_allowed.len(),
            self.policy.write_allowed.len(),
            self.policy.denied.len(),
        )
    }
}

/// A convenience check: is a path readable under a filesystem policy, with
/// glob-deny semantics applied?
pub fn is_readable(policy: &FilesystemPolicy, path: &str) -> bool {
    let controller = PathAccessController::from_policy(policy);
    controller.check_read(path).permitted()
}

/// A convenience check: is a path writable under a filesystem policy?
pub fn is_writable(policy: &FilesystemPolicy, path: &str) -> bool {
    let controller = PathAccessController::from_policy(policy);
    controller.check_write(path).permitted()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitelist_compiles_write_paths() {
        let policy = FilesystemPolicy::default()
            .with_write_allowed("/workspace")
            .with_denied("/workspace/.git");
        let controller = PathAccessController::new(policy);
        // Deny wins for the denied subpath.
        let d = controller.check_write("/workspace/.git/config");
        assert!(!d.permitted());
    }

    #[test]
    fn denied_path_blocks_even_in_allowed_parent() {
        let policy = FilesystemPolicy::default()
            .with_read_allowed("/home/user")
            .with_denied("/home/user/.ssh");
        let wl = whitelist_from_filesystem(&policy);
        assert!(wl.check_read("/home/user/docs/file").is_allow());
        let d = wl.check_read("/home/user/.ssh/id_rsa");
        assert!(d.is_deny());
    }

    #[test]
    fn tmp_writable_in_whitelist() {
        let policy = FilesystemPolicy::default();
        let wl = whitelist_from_filesystem(&policy);
        assert!(wl.check_write("/tmp/x").is_allow());
    }

    #[test]
    fn access_mode_mapping() {
        let policy = FilesystemPolicy::default()
            .with_read_allowed("/read-only")
            .with_write_allowed("/writable");
        let wl = whitelist_from_filesystem(&policy);
        assert_eq!(
            wl.check_read("/read-only/a"),
            crate::whitelist::AccessVerdict::Allow(AccessMode::Read)
        );
        assert_eq!(
            wl.check_write("/writable/a"),
            crate::whitelist::AccessVerdict::Allow(AccessMode::ReadWrite)
        );
    }
}
