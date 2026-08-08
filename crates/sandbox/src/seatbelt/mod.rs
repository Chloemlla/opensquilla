//! macOS Seatbelt (SBPL) profile generation.
//!
//! Generates [SandBox Profile Language] text from a [`SandboxPolicy`] or a
//! higher-level [`SeatbeltProfile`]. The generated profile is deny-by-default
//! with explicit allow rules for filesystem, network, process and Mach
//! operations.
//!
//! The macOS backend writes the generated profile to a temporary `.sbpl` file
//! and invokes `sandbox-exec -f <profile> <command>`.
//!
//! [SandBox Profile Language]: https://reverse.put.as/2011/09/14/apple-sandbox-guide-v1.0/

pub mod operations;
pub mod presets;

pub use presets::preset_profile;

use serde::{Deserialize, Serialize};

use crate::policy::{NetworkPolicy, SandboxLevel, SandboxPolicy};

/// A compiled Seatbelt profile: the SBPL source text plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeatbeltProfile {
    /// The SBPL source text.
    pub source: String,
    /// A human-readable name.
    pub name: String,
    /// The security level this profile targets.
    pub level: SandboxLevel,
    /// Whether the profile is deny-by-default.
    pub deny_default: bool,
}

impl SeatbeltProfile {
    /// Compile a [`SandboxPolicy`] into an SBPL profile.
    pub fn from_policy(policy: &SandboxPolicy) -> Self {
        let source = compile_policy(policy);
        Self {
            source,
            name: format!("opensquilla_{}", level_tag(policy.level)),
            level: policy.level,
            deny_default: true,
        }
    }

    /// Compile with a custom name.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// The SBPL source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Length of the source in bytes.
    pub fn source_len(&self) -> usize {
        self.source.len()
    }

    /// Validate the profile source by parsing it for balanced parentheses.
    /// This is a syntactic check only; it does not verify that the operations
    /// are accepted by `sandbox-exec`.
    pub fn validate_syntax(&self) -> Result<(), String> {
        let mut depth: i32 = 0;
        let mut in_string = false;
        let mut escape = false;
        for c in self.source.chars() {
            if in_string {
                if escape {
                    escape = false;
                } else if c == '\\' {
                    escape = true;
                } else if c == '"' {
                    in_string = false;
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err("unbalanced ')' in SBPL source".to_string());
                    }
                }
                _ => {}
            }
        }
        if in_string {
            return Err("unterminated string in SBPL source".to_string());
        }
        if depth != 0 {
            return Err(format!(
                "unbalanced parentheses in SBPL source (depth={depth})"
            ));
        }
        Ok(())
    }
}

fn level_tag(level: SandboxLevel) -> &'static str {
    match level {
        SandboxLevel::Standard => "standard",
        SandboxLevel::Strict => "strict",
        SandboxLevel::Locked => "locked",
    }
}

/// Compile a [`SandboxPolicy`] into SBPL source text.
pub fn compile_policy(policy: &SandboxPolicy) -> String {
    let mut sbpl = String::with_capacity(4096);

    // Version header and deny-by-default.
    sbpl.push_str("(version 1)\n");
    sbpl.push_str("(deny default)\n");
    sbpl.push_str("(allow default-pager)\n");

    // Process operations.
    sbpl.push_str(";;; Process operations\n");
    sbpl.push_str("(allow process-fork)\n");
    sbpl.push_str("(allow process-exec)\n");
    sbpl.push_str("(allow process-info*)\n");
    sbpl.push_str("(allow signal (target self))\n");
    sbpl.push_str("(allow sysctl-read)\n");

    // IPC.
    sbpl.push_str(";;; IPC\n");
    sbpl.push_str("(allow ipc-posix-semaphore*)\n");
    sbpl.push_str("(allow ipc-posix-shm*)\n");
    sbpl.push_str("(allow ipc-sysv)\n");

    // Mach.
    sbpl.push_str(";;; Mach\n");
    sbpl.push_str("(allow mach-task-self)\n");
    sbpl.push_str("(allow mach-privilege-task-port)\n");
    sbpl.push_str("(allow mach-lookup (global-name \"com.apple.system.logger\"))\n");
    sbpl.push_str("(allow mach-lookup (global-name \"com.apple.system.notification_center\"))\n");

    // Filesystem.
    sbpl.push_str(";;; Filesystem\n");
    compile_filesystem(&mut sbpl, policy);

    // Network.
    sbpl.push_str(";;; Network\n");
    compile_network(&mut sbpl, &policy.network);

    // Level-specific tightening.
    if policy.level >= SandboxLevel::Strict {
        sbpl.push_str(";;; Strict-level restrictions\n");
        sbpl.push_str("(deny process-exec (literal \"/usr/bin/sudo\"))\n");
        sbpl.push_str("(deny process-exec (literal \"/usr/bin/su\"))\n");
        sbpl.push_str("(deny process-exec (literal \"/bin/launchctl\"))\n");
    }
    if policy.level >= SandboxLevel::Locked {
        sbpl.push_str(";;; Locked-level restrictions\n");
        sbpl.push_str("(deny iokit*)\n");
        sbpl.push_str("(deny sysctl*)\n");
        sbpl.push_str("(deny mach-lookup)\n");
        sbpl.push_str("(allow mach-task-self)\n");
        sbpl.push_str("(allow mach-privilege-task-port)\n");
    }

    sbpl
}

/// Emit filesystem rules.
fn compile_filesystem(sbpl: &mut String, policy: &SandboxPolicy) {
    // Allow metadata traversal of the root (needed for path resolution).
    sbpl.push_str("(allow file-read-metadata (subpath \"/\"))\n");

    // Read-only system directories.
    let system_dirs = [
        "/usr",
        "/System",
        "/Library",
        "/bin",
        "/sbin",
        "/opt",
        "/private/var/tmp",
    ];
    for dir in system_dirs {
        if !policy.filesystem.blocks(dir) {
            sbpl.push_str(&format!(
                "(allow file-read* (subpath \"{}\"))\n",
                sbpl_escape(dir)
            ));
        }
    }

    // Policy read-allowed paths.
    for path in &policy.filesystem.read_allowed {
        if path.is_empty() || policy.filesystem.blocks(path) {
            continue;
        }
        sbpl.push_str(&format!(
            "(allow file-read* (subpath \"{}\"))\n",
            sbpl_escape(path)
        ));
    }

    // Policy write-allowed paths.
    for path in &policy.filesystem.write_allowed {
        if path.is_empty() || policy.filesystem.blocks(path) {
            continue;
        }
        sbpl.push_str(&format!(
            "(allow file-read* file-write* (subpath \"{}\"))\n",
            sbpl_escape(path)
        ));
    }

    // Home directory.
    if policy.filesystem.home_readable {
        if let Some(home) = std::env::var_os("HOME") {
            let home = home.to_string_lossy();
            sbpl.push_str(&format!(
                "(allow file-read* (subpath \"{}\"))\n",
                sbpl_escape(&home)
            ));
        }
    }

    // /tmp.
    if policy.filesystem.tmp_writable {
        sbpl.push_str("(allow file-read* file-write* (subpath \"/tmp\"))\n");
        sbpl.push_str("(allow file-read* file-write* (subpath \"/private/tmp\"))\n");
        sbpl.push_str("(allow file-read* file-write* (subpath \"/private/var/tmp\"))\n");
    }

    // Explicitly denied paths.
    for path in &policy.filesystem.denied {
        if path.is_empty() {
            continue;
        }
        sbpl.push_str(&format!(
            "(deny file-read* file-write* (subpath \"{}\"))\n",
            sbpl_escape(path)
        ));
    }

    // Deny write to the rest.
    sbpl.push_str("(deny file-write*)\n");
}

/// Emit network rules.
fn compile_network(sbpl: &mut String, network: &NetworkPolicy) {
    match network {
        NetworkPolicy::None => {
            sbpl.push_str("(deny network*)\n");
        }
        NetworkPolicy::ProxyAllowlist(domains) => {
            // Allow loopback (for the local proxy).
            sbpl.push_str("(allow network* (local ip \"127.0.0.1\"))\n");
            sbpl.push_str("(allow network* (local ip \"::1\"))\n");
            sbpl.push_str("(allow network-outbound (remote ip \"127.0.0.1\"))\n");
            sbpl.push_str("(allow network-outbound (remote ip \"::1\"))\n");
            for domain in domains {
                if domain.trim().is_empty() {
                    continue;
                }
                sbpl.push_str(&format!(
                    "(allow network-outbound (remote name \"{}\"))\n",
                    sbpl_escape(domain)
                ));
            }
        }
        NetworkPolicy::Host => {
            sbpl.push_str("(allow network*)\n");
        }
    }
}

/// Escape a string for inclusion in an SBPL string literal.
pub fn sbpl_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::FilesystemPolicy;

    #[test]
    fn compiles_for_each_level() {
        for level in [
            SandboxLevel::Standard,
            SandboxLevel::Strict,
            SandboxLevel::Locked,
        ] {
            let policy = SandboxPolicy::build_policy(level, None);
            let profile = SeatbeltProfile::from_policy(&policy);
            profile.validate_syntax().unwrap();
            assert!(profile.source.contains("(version 1)"));
            assert!(profile.source.contains("(deny default)"));
        }
    }

    #[test]
    fn network_none_denies() {
        let policy = SandboxPolicy {
            network: NetworkPolicy::None,
            ..Default::default()
        };
        let profile = SeatbeltProfile::from_policy(&policy);
        assert!(profile.source.contains("(deny network*)"));
    }

    #[test]
    fn allowlist_emits_remote_name() {
        let policy = SandboxPolicy {
            network: NetworkPolicy::ProxyAllowlist(vec!["example.com".to_string()]),
            ..Default::default()
        };
        let profile = SeatbeltProfile::from_policy(&policy);
        assert!(profile.source.contains("remote name \"example.com\""));
    }

    #[test]
    fn denied_paths_emitted() {
        let policy = SandboxPolicy {
            filesystem: FilesystemPolicy::default().with_denied("/secret"),
            ..SandboxPolicy::default()
        };
        let profile = SeatbeltProfile::from_policy(&policy);
        assert!(
            profile
                .source
                .contains("(deny file-read* file-write* (subpath \"/secret\"))")
        );
    }

    #[test]
    fn escape_quotes() {
        assert_eq!(sbpl_escape(r#"a"b\c"#), r#"a\"b\\c"#);
    }
}
