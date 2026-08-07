//! Command risk classification.
//!
//! The sandbox receives arbitrary command lines from the agent. Before
//! execution, the manager classifies the command into a risk tier and decides
//! (a) which sandbox profile to use, and (b) whether governance approval is
//! required. This module implements that classification as pure data + pure
//! functions, so it can be unit-tested in isolation and reused by the RPC,
//! CLI and Tauri layers.

use serde::{Deserialize, Serialize};

use crate::policy::SandboxLevel;

/// A risk tier for a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskTier {
    /// Benign read-only introspection (`ls`, `pwd`, `cat` on safe paths).
    Benign,
    /// Normal read/write operations inside the workspace.
    Low,
    /// Operations with meaningful side effects or network access.
    Medium,
    /// Arbitrary code execution, package installs, system modification.
    High,
    /// Credential access, kernel/device access, exfiltration vectors.
    Critical,
}

impl RiskTier {
    /// The sandbox level recommended for this tier.
    pub fn recommended_level(self) -> SandboxLevel {
        match self {
            RiskTier::Benign | RiskTier::Low => SandboxLevel::Standard,
            RiskTier::Medium | RiskTier::High => SandboxLevel::Strict,
            RiskTier::Critical => SandboxLevel::Locked,
        }
    }

    /// Whether governance approval should be required before running.
    pub fn requires_approval(self) -> bool {
        matches!(self, RiskTier::High | RiskTier::Critical)
    }

    /// Human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            RiskTier::Benign => "benign",
            RiskTier::Low => "low",
            RiskTier::Medium => "medium",
            RiskTier::High => "high",
            RiskTier::Critical => "critical",
        }
    }
}

/// The classification of a single command line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandAssessment {
    /// The command name (first token, basename).
    pub command: String,
    /// The full argument vector.
    pub args: Vec<String>,
    /// The risk tier.
    pub tier: RiskTier,
    /// Human-readable reason for the tier.
    pub reason: String,
    /// Recommended profile id.
    pub recommended_profile: String,
    /// Whether governance approval is required.
    pub approval_required: bool,
    /// Paths the command is believed to touch (for escalation routing).
    pub touched_paths: Vec<String>,
}

/// A static classification rule: a command-name pattern with a tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRule {
    /// The command name, optionally with a `*` suffix wildcard (e.g.
    /// `"python*"`).
    pub pattern: String,
    /// The risk tier for this command.
    pub tier: RiskTier,
    /// Optional argument patterns that trigger a higher tier. Each entry is
    /// an argument substring; if any argument contains it, the tier is
    /// upgraded.
    pub escalation_args: Vec<String>,
    /// Human-readable reason.
    pub reason: &'static str,
}

impl CommandRule {
    /// Create a rule.
    pub fn new(pattern: &'static str, tier: RiskTier, reason: &'static str) -> Self {
        Self {
            pattern: pattern.to_string(),
            tier,
            escalation_args: Vec::new(),
            reason,
        }
    }

    /// Add an argument that escalates this command's tier.
    pub fn with_escalation_arg(mut self, arg: &str) -> Self {
        self.escalation_args.push(arg.to_string());
        self
    }

    /// Does this rule match a command name (basename)?
    pub fn matches(&self, command: &str) -> bool {
        if let Some(prefix) = self.pattern.strip_suffix('*') {
            command.starts_with(prefix)
        } else {
            command == self.pattern
        }
    }
}

/// The default rule table.
///
/// The ordering matters: the first rule whose pattern matches wins, and a
/// matching escalation argument raises the tier to the rule's tier (never
/// lowers).
pub fn default_command_rules() -> Vec<CommandRule> {
    vec![
        // Credential / key access.
        CommandRule::new(
            "ssh",
            RiskTier::High,
            "remote access; possible credential access",
        )
        .with_escalation_arg("-i"),
        CommandRule::new("scp", RiskTier::High, "remote file copy"),
        CommandRule::new("sftp", RiskTier::High, "remote file transfer"),
        CommandRule::new("gpg", RiskTier::Medium, "key management")
            .with_escalation_arg("--export-secret")
            .with_escalation_arg("--decrypt"),
        CommandRule::new("openssl", RiskTier::Medium, "crypto operations")
            .with_escalation_arg("genrsa")
            .with_escalation_arg("genpkey")
            .with_escalation_arg("pkcs12"),
        CommandRule::new("kubectl", RiskTier::High, "cluster control")
            .with_escalation_arg("get secrets")
            .with_escalation_arg("exec"),
        CommandRule::new("docker", RiskTier::High, "container control")
            .with_escalation_arg("exec")
            .with_escalation_arg("run")
            .with_escalation_arg("pull"),
        CommandRule::new("podman", RiskTier::High, "container control"),
        CommandRule::new("aws", RiskTier::Medium, "cloud CLI")
            .with_escalation_arg("secretsmanager")
            .with_escalation_arg("iam")
            .with_escalation_arg("sts"),
        CommandRule::new("gcloud", RiskTier::Medium, "cloud CLI")
            .with_escalation_arg("iam")
            .with_escalation_arg("secrets"),
        CommandRule::new("az", RiskTier::Medium, "cloud CLI"),
        // Package management.
        CommandRule::new("pip", RiskTier::Medium, "package installation")
            .with_escalation_arg("install")
            .with_escalation_arg("--global"),
        CommandRule::new("pip3", RiskTier::Medium, "package installation")
            .with_escalation_arg("install")
            .with_escalation_arg("--global"),
        CommandRule::new("npm", RiskTier::Medium, "package installation")
            .with_escalation_arg("install")
            .with_escalation_arg("publish"),
        CommandRule::new("yarn", RiskTier::Medium, "package installation")
            .with_escalation_arg("add")
            .with_escalation_arg("publish"),
        CommandRule::new("cargo", RiskTier::Medium, "Rust package management")
            .with_escalation_arg("install")
            .with_escalation_arg("publish"),
        CommandRule::new("go", RiskTier::Medium, "Go toolchain")
            .with_escalation_arg("install")
            .with_escalation_arg("get"),
        CommandRule::new("apt", RiskTier::High, "system package manager")
            .with_escalation_arg("install")
            .with_escalation_arg("remove"),
        CommandRule::new("apt-get", RiskTier::High, "system package manager")
            .with_escalation_arg("install")
            .with_escalation_arg("remove"),
        CommandRule::new("yum", RiskTier::High, "system package manager"),
        CommandRule::new("dnf", RiskTier::High, "system package manager"),
        CommandRule::new("brew", RiskTier::Medium, "macOS package manager")
            .with_escalation_arg("install"),
        CommandRule::new("pamac", RiskTier::High, "system package manager"),
        // System modification.
        CommandRule::new("sudo", RiskTier::Critical, "privilege escalation"),
        CommandRule::new("su", RiskTier::Critical, "switch user"),
        CommandRule::new("passwd", RiskTier::Critical, "change credentials"),
        CommandRule::new("chsh", RiskTier::High, "change login shell"),
        CommandRule::new("chmod", RiskTier::Low, "change permissions")
            .with_escalation_arg("4777")
            .with_escalation_arg("777"),
        CommandRule::new("chown", RiskTier::Medium, "change ownership"),
        CommandRule::new("mount", RiskTier::Critical, "mount filesystem"),
        CommandRule::new("umount", RiskTier::Critical, "unmount filesystem"),
        CommandRule::new("fdisk", RiskTier::Critical, "disk partitioning"),
        CommandRule::new("mkfs", RiskTier::Critical, "format disk"),
        CommandRule::new("rm", RiskTier::Low, "remove files")
            .with_escalation_arg("-rf /")
            .with_escalation_arg("--no-preserve-root"),
        CommandRule::new("shutdown", RiskTier::Critical, "system shutdown"),
        CommandRule::new("reboot", RiskTier::Critical, "system reboot"),
        CommandRule::new("systemctl", RiskTier::High, "system service control")
            .with_escalation_arg("start")
            .with_escalation_arg("stop")
            .with_escalation_arg("restart"),
        CommandRule::new("service", RiskTier::High, "system service control"),
        CommandRule::new("crontab", RiskTier::Medium, "schedule jobs"),
        CommandRule::new("at", RiskTier::Medium, "schedule job"),
        CommandRule::new("useradd", RiskTier::Critical, "create user"),
        CommandRule::new("userdel", RiskTier::Critical, "delete user"),
        CommandRule::new("usermod", RiskTier::Critical, "modify user"),
        CommandRule::new("groupadd", RiskTier::High, "create group"),
        // Network tooling.
        CommandRule::new("curl", RiskTier::Low, "HTTP client")
            .with_escalation_arg("--upload-file")
            .with_escalation_arg("-F")
            .with_escalation_arg("--data-binary")
            .with_escalation_arg("--connect-to")
            .with_escalation_arg("--resolve"),
        CommandRule::new("wget", RiskTier::Low, "HTTP client").with_escalation_arg("--post-file"),
        CommandRule::new("nc", RiskTier::Critical, "raw network socket"),
        CommandRule::new("ncat", RiskTier::Critical, "raw network socket"),
        CommandRule::new("socat", RiskTier::Critical, "socket relay"),
        CommandRule::new("telnet", RiskTier::High, "remote shell protocol"),
        CommandRule::new("ssh-keygen", RiskTier::Medium, "generate SSH keys"),
        CommandRule::new("iptables", RiskTier::Critical, "firewall manipulation"),
        CommandRule::new("nft", RiskTier::Critical, "firewall manipulation"),
        // Interpreters (arbitrary code).
        CommandRule::new("python", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("python3", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("node", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("perl", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("ruby", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("php", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("lua", RiskTier::High, "arbitrary code execution"),
        CommandRule::new("sh", RiskTier::Medium, "shell"),
        CommandRule::new("bash", RiskTier::Medium, "shell"),
        CommandRule::new("zsh", RiskTier::Medium, "shell"),
        CommandRule::new("fish", RiskTier::Medium, "shell"),
        CommandRule::new("sqlite3", RiskTier::Medium, "database access")
            .with_escalation_arg(".dump"),
        CommandRule::new("psql", RiskTier::High, "database access"),
        CommandRule::new("mysql", RiskTier::High, "database access"),
        // Git (usually benign).
        CommandRule::new("git", RiskTier::Low, "version control")
            .with_escalation_arg("push")
            .with_escalation_arg("remote add")
            .with_escalation_arg("fetch"),
        // File reads.
        CommandRule::new("cat", RiskTier::Benign, "read file")
            .with_escalation_arg("/etc/passwd")
            .with_escalation_arg("/etc/shadow")
            .with_escalation_arg("/etc/sudoers"),
        CommandRule::new("ls", RiskTier::Benign, "list directory"),
        CommandRule::new("pwd", RiskTier::Benign, "print working directory"),
        CommandRule::new("find", RiskTier::Low, "find files"),
        CommandRule::new("grep", RiskTier::Benign, "search text")
            .with_escalation_arg("/etc/passwd")
            .with_escalation_arg("/etc/shadow"),
        CommandRule::new("head", RiskTier::Benign, "read file start")
            .with_escalation_arg("/etc/passwd")
            .with_escalation_arg("/etc/shadow"),
        CommandRule::new("tail", RiskTier::Benign, "read file end")
            .with_escalation_arg("/etc/passwd")
            .with_escalation_arg("/etc/shadow"),
        CommandRule::new("less", RiskTier::Benign, "page through file"),
        CommandRule::new("more", RiskTier::Benign, "page through file"),
        CommandRule::new("env", RiskTier::Benign, "print environment"),
        CommandRule::new("printenv", RiskTier::Benign, "print environment"),
        // Dev tools.
        CommandRule::new("make", RiskTier::Medium, "build system").with_escalation_arg("install"),
        CommandRule::new("cmake", RiskTier::Medium, "build system")
            .with_escalation_arg("--install"),
        CommandRule::new("ninja", RiskTier::Low, "build tool"),
        CommandRule::new("gcc", RiskTier::High, "compile code"),
        CommandRule::new("g++", RiskTier::High, "compile code"),
        CommandRule::new("clang", RiskTier::High, "compile code"),
        CommandRule::new("rustc", RiskTier::High, "compile code"),
        CommandRule::new("javac", RiskTier::High, "compile code"),
        CommandRule::new("java", RiskTier::High, "run JVM code"),
        CommandRule::new("go run", RiskTier::High, "run Go code"),
    ]
}

/// Raise a risk tier by one step (capped at Critical).
fn bump_tier(tier: RiskTier) -> RiskTier {
    match tier {
        RiskTier::Benign => RiskTier::Low,
        RiskTier::Low => RiskTier::Medium,
        RiskTier::Medium => RiskTier::High,
        RiskTier::High => RiskTier::Critical,
        RiskTier::Critical => RiskTier::Critical,
    }
}

/// Extract the command name (basename) from a full command string.
pub fn command_basename(command: &str) -> String {
    command
        .trim()
        .rsplit('/')
        .next()
        .unwrap_or(command)
        .to_string()
}

/// Classify a command line (command + args) against the default rule table.
pub fn assess_command(command: &str, args: &[&str]) -> CommandAssessment {
    assess_with_rules(command, args, &default_command_rules())
}

/// Classify a command line against an explicit rule table.
pub fn assess_with_rules(command: &str, args: &[&str], rules: &[CommandRule]) -> CommandAssessment {
    let name = command_basename(command);
    let mut tier = RiskTier::Low;
    let mut reason = "no matching rule; treated as low risk".to_string();

    for rule in rules {
        if rule.matches(&name) {
            tier = rule.tier.max(tier);
            reason = rule.reason.to_string();
            // Escalation arguments raise the tier one step (never lower it).
            for esc in &rule.escalation_args {
                let esc_norm = esc.to_lowercase();
                let hit = args.iter().any(|a| a.to_lowercase().contains(&esc_norm));
                if hit {
                    tier = bump_tier(tier);
                    reason = format!("{} (escalation arg '{}')", rule.reason, esc);
                    break;
                }
            }
            break;
        }
    }

    // Special-case dangerous path arguments regardless of rule.
    let dangerous_paths = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/root/.ssh",
        "/etc/ssh",
        "/etc/kubernetes",
        "/var/run/docker.sock",
        "/var/lib/kubelet",
    ];
    let mut touched_paths: Vec<String> = Vec::new();
    for a in args {
        for danger in &dangerous_paths {
            if a.contains(danger) {
                tier = tier.max(RiskTier::High);
                reason = format!("touches sensitive path {danger}");
                touched_paths.push(danger.to_string());
            }
        }
    }

    // A bare shell or interpreter is at least Medium.
    if matches!(name.as_str(), "sh" | "bash" | "zsh" | "fish") && tier < RiskTier::Medium {
        tier = RiskTier::Medium;
        reason = "shell invocation".to_string();
    }

    // Sensitive-path hard-block escalation: a destructive command (rm, or a
    // Python delete via os.remove / shutil.rmtree / Path.unlink) that targets a
    // sensitive host path is treated as at least High risk regardless of the
    // rule table. This is the policy-side guard feeding
    // `crate::sensitive_paths::sensitive_target_in_command`.
    let full_command = format!("{} {}", command, args.join(" "));
    if let Some(marker) =
        crate::sensitive_paths::sensitive_target_in_command(&full_command, None, None)
    {
        tier = tier.max(RiskTier::High);
        reason = format!("destructive command targets sensitive path {marker}");
        if !touched_paths.contains(&marker) {
            touched_paths.push(marker);
        }
    }

    let approval_required = tier.requires_approval();
    CommandAssessment {
        command: name,
        args: args.iter().map(|s| s.to_string()).collect(),
        tier,
        reason,
        recommended_profile: match tier {
            RiskTier::Benign | RiskTier::Low => "standard_file_read".to_string(),
            RiskTier::Medium => "standard".to_string(),
            RiskTier::High => "strict".to_string(),
            RiskTier::Critical => "locked".to_string(),
        },
        approval_required,
        touched_paths,
    }
}

/// Resolve the sandbox profile for a command via classification. This is the
/// bridge between the risk classifier and the profile registry.
pub fn profile_id_for_command(command: &str, args: &[&str]) -> String {
    let assessment = assess_command(command, args);
    assessment.recommended_profile
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benign_read_only() {
        let a = assess_command("ls", &["-la"]);
        assert_eq!(a.tier, RiskTier::Benign);
        assert!(!a.approval_required);
    }

    #[test]
    fn arbitrary_code_is_high() {
        let a = assess_command("python", &["-c", "print(1)"]);
        assert_eq!(a.tier, RiskTier::High);
        assert!(a.approval_required);
        assert_eq!(a.recommended_profile, "strict");
    }

    #[test]
    fn sudo_is_critical() {
        let a = assess_command("sudo", &["rm", "-rf", "/"]);
        assert_eq!(a.tier, RiskTier::Critical);
        assert!(a.approval_required);
    }

    #[test]
    fn sensitive_path_escalates() {
        let a = assess_command("cat", &["/etc/shadow"]);
        assert_eq!(a.tier, RiskTier::High);
        assert!(a.touched_paths.contains(&"/etc/shadow".to_string()));
    }

    #[test]
    fn escalation_arg_raises_tier() {
        // curl is Low; --resolve (SSRF-ish) escalates.
        let a = assess_command(
            "curl",
            &["--resolve", "internal:443:127.0.0.1", "http://internal/"],
        );
        assert!(a.tier >= RiskTier::Medium);
    }

    #[test]
    fn basename_extraction() {
        assert_eq!(command_basename("/usr/bin/python3"), "python3");
        assert_eq!(command_basename("python"), "python");
    }

    #[test]
    fn classification_ordering_is_stable() {
        // git push escalates to Medium because "push" is a listed escalation
        // argument (network egress to a remote).
        let a = assess_command("git", &["push", "origin", "main"]);
        assert_eq!(a.tier, RiskTier::Medium);

        // Plain git status stays Low.
        let b = assess_command("git", &["status"]);
        assert_eq!(b.tier, RiskTier::Low);
    }

    #[test]
    fn destructive_targets_are_escalated() {
        // `rm` targeting a sensitive leaf is at least High, even though the
        // rule table only grades `rm` as Low.
        let a = assess_command("rm", &["/home/u/.ssh/id_rsa"]);
        assert!(a.tier >= RiskTier::High);
        assert!(a.touched_paths.iter().any(|p| p == "/id_rsa"));
        // A plain `rm` inside /tmp stays Low.
        let b = assess_command("rm", &["/tmp/scratch.txt"]);
        assert_eq!(b.tier, RiskTier::Low);
    }

    #[test]
    fn python_delete_sensitive_target_escalated() {
        let a = assess_command(
            "python",
            &["-c", "import shutil; shutil.rmtree('/etc/shadow')"],
        );
        assert!(a.tier >= RiskTier::High);
        assert!(a.touched_paths.iter().any(|p| p == "/etc/shadow"));
    }
}
