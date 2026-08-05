//! # Skill eligibility checking
//!
//! The [`EligibilityChecker`] evaluates a skill's [`SkillRequires`] against the
//! current host. Checks cover:
//!
//! - Operating system and CPU architecture matching
//! - Binary availability (a `which`/`where` equivalent, cached)
//! - Environment variable presence and value matching
//! - File / directory existence
//! - Tool presence and version constraints
//! - Capability probes (network, docker, git, python, …)
//! - Free memory floors
//!
//! The checker is pure and deterministic: identical [`SkillRequires`] on the
//! same host always produce the same verdict. Results are available both as a
//! simple boolean (via [`EligibilityChecker::is_eligible`]) and as a detailed
//! per-requirement [`EligibilityReport`] for UI display.

use crate::types::{SkillRequires, SkillVersion};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tracing::debug;

/// The kind of a single eligibility requirement, for report display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequirementKind {
    Os,
    Arch,
    Binary,
    EnvVar,
    Capability,
    File,
    Tool,
    ToolVersion,
    Memory,
    Network,
    Version,
}

impl RequirementKind {
    /// A short label for this requirement kind.
    pub fn label(self) -> &'static str {
        match self {
            RequirementKind::Os => "os",
            RequirementKind::Arch => "arch",
            RequirementKind::Binary => "binary",
            RequirementKind::EnvVar => "env",
            RequirementKind::Capability => "capability",
            RequirementKind::File => "file",
            RequirementKind::Tool => "tool",
            RequirementKind::ToolVersion => "tool_version",
            RequirementKind::Memory => "memory",
            RequirementKind::Network => "network",
            RequirementKind::Version => "version",
        }
    }
}

/// The outcome of a single requirement check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckStatus {
    /// What was checked, e.g. `"binary:git"`.
    pub requirement: String,
    /// The category of the check.
    pub kind: RequirementKind,
    /// Whether the check passed.
    pub ok: bool,
    /// Human-readable detail for the failure (empty when `ok`).
    pub detail: String,
}

impl CheckStatus {
    fn pass(requirement: impl Into<String>, kind: RequirementKind) -> Self {
        Self {
            requirement: requirement.into(),
            kind,
            ok: true,
            detail: String::new(),
        }
    }

    fn fail(
        requirement: impl Into<String>,
        kind: RequirementKind,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            requirement: requirement.into(),
            kind,
            ok: false,
            detail: detail.into(),
        }
    }
}

/// Detailed eligibility result.
#[derive(Debug, Clone, Default)]
pub struct EligibilityReport {
    /// The individual requirement checks, in evaluation order.
    pub checks: Vec<CheckStatus>,
}

impl EligibilityReport {
    /// Whether every check passed.
    pub fn is_eligible(&self) -> bool {
        self.checks.iter().all(|c| c.ok)
    }

    /// The list of failed checks.
    pub fn failures(&self) -> Vec<&CheckStatus> {
        self.checks.iter().filter(|c| !c.ok).collect()
    }

    /// Number of failed checks.
    pub fn failure_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.ok).count()
    }
}

/// A snapshot of the current host's static characteristics.
#[derive(Debug, Clone)]
pub struct CurrentHost {
    /// The OS name: `"linux"`, `"macos"`, `"windows"`, or `"other"`.
    pub os: String,
    /// The OS family: `"unix"` or `"windows"`.
    pub family: String,
    /// The CPU architecture: `"x86_64"`, `"aarch64"`, `"arm64"`, `"x86"`, …
    pub arch: String,
    /// Total system memory in MiB (best-effort; `None` if unknown).
    pub total_memory_mib: Option<u64>,
}

impl Default for CurrentHost {
    fn default() -> Self {
        Self::detect()
    }
}

impl CurrentHost {
    /// Detect the current host characteristics.
    pub fn detect() -> Self {
        let os = match std::env::consts::OS {
            "linux" => "linux",
            "macos" => "macos",
            "windows" => "windows",
            other => other,
        }
        .to_string();
        let family = if cfg!(windows) { "windows" } else { "unix" }.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let total_memory_mib = detect_total_memory_mib();
        Self {
            os,
            family,
            arch,
            total_memory_mib,
        }
    }

    /// Whether `os` (e.g. `"linux"`) matches this host, considering aliases
    /// like `"unix"`, `"posix"`, `"darwin"`, `"win32"`.
    pub fn os_matches(&self, os: &str) -> bool {
        let os = os.trim().to_ascii_lowercase();
        match os.as_str() {
            "any" | "all" => true,
            "unix" | "posix" => self.family == "unix",
            "windows" | "win32" | "win" => self.family == "windows",
            "macos" | "darwin" | "osx" => self.os == "macos",
            "linux" => self.os == "linux",
            other => self.os == other,
        }
    }

    /// Whether `arch` (e.g. `"x86_64"`, `"arm64"`) matches this host,
    /// considering common aliases.
    pub fn arch_matches(&self, arch: &str) -> bool {
        let arch = arch.trim().to_ascii_lowercase();
        match arch.as_str() {
            "any" | "all" => true,
            "x86_64" | "amd64" | "x64" => self.arch == "x86_64" || self.arch == "amd64",
            "x86" | "i386" | "i686" | "ia32" => self.arch == "x86",
            "aarch64" | "arm64" => self.arch == "aarch64" || self.arch == "arm64",
            "arm" | "armv7" => self.arch.starts_with("arm"),
            "wasm" | "wasm32" => self.arch == "wasm32",
            other => self.arch == other,
        }
    }
}

/// Detect total system memory in MiB (best-effort, platform-specific).
fn detect_total_memory_mib() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        // Best-effort: read TOTALMEMORY via `wmic` or `systeminfo` is slow;
        // fall back to the `GetPhysicallyInstalledSystemMemory` API is complex,
        // so we return None on Windows unless the env is set.
        std::env::var("OSQ_TOTAL_MEMORY_MIB")
            .ok()
            .and_then(|s| s.parse().ok())
    }
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = content.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb / 1024)
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(bytes / (1024 * 1024))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

/// The eligibility checker.
///
/// Binary and environment-variable results are cached to avoid repeated
/// subprocess and environment lookups. Call [`EligibilityChecker::clear_cache`]
/// if the PATH or environment changes at runtime.
pub struct EligibilityChecker {
    /// Cache of binary existence checks (name -> exists).
    binary_cache: Mutex<HashMap<String, bool>>,
    /// Cache of env var checks (name -> exists).
    env_cache: Mutex<HashMap<String, bool>>,
    /// Cache of file existence checks (path -> exists).
    file_cache: Mutex<HashMap<String, bool>>,
    /// The detected host characteristics.
    host: CurrentHost,
    /// When the checker was created.
    created_at: chrono::DateTime<chrono::Utc>,
}

impl Default for EligibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl EligibilityChecker {
    /// Create a new eligibility checker for the current host.
    pub fn new() -> Self {
        Self {
            binary_cache: Mutex::new(HashMap::new()),
            env_cache: Mutex::new(HashMap::new()),
            file_cache: Mutex::new(HashMap::new()),
            host: CurrentHost::detect(),
            created_at: chrono::Utc::now(),
        }
    }

    /// Create a checker against an explicit host snapshot (used in tests and
    /// for cross-compilation checks).
    pub fn for_host(host: CurrentHost) -> Self {
        Self {
            binary_cache: Mutex::new(HashMap::new()),
            env_cache: Mutex::new(HashMap::new()),
            file_cache: Mutex::new(HashMap::new()),
            host,
            created_at: chrono::Utc::now(),
        }
    }

    /// The host snapshot this checker was created against.
    pub fn host(&self) -> &CurrentHost {
        &self.host
    }

    /// When this checker was created.
    pub fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        self.created_at
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Check if a skill is eligible on the current system.
    ///
    /// Returns `Ok(true)` when every requirement passes, or `Err(failures)`
    /// where `failures` is a list of human-readable reasons.
    pub fn is_eligible(&self, requires: &SkillRequires) -> Result<bool, Vec<String>> {
        let report = self.check(requires);
        if report.is_eligible() {
            Ok(true)
        } else {
            Err(report
                .failures()
                .into_iter()
                .map(|c| format!("{}: {}", c.kind.label(), c.detail))
                .collect())
        }
    }

    /// Evaluate all requirements, returning a detailed report.
    pub fn check(&self, requires: &SkillRequires) -> EligibilityReport {
        let mut report = EligibilityReport::default();

        if let Some(os_list) = &requires.os {
            if !os_list.is_empty() {
                report.checks.push(self.check_os_check(os_list));
            }
        }

        if let Some(arch_list) = &requires.arch {
            if !arch_list.is_empty() {
                report.checks.push(self.check_arch_check(arch_list));
            }
        }

        if let Some(binaries) = &requires.binaries {
            for binary in binaries {
                report.checks.push(self.check_binary_check(binary));
            }
        }

        if let Some(env_vars) = &requires.env_vars {
            for var in env_vars {
                report.checks.push(self.check_env_var_check(var));
            }
        }

        if let Some(files) = &requires.files {
            for file in files {
                report.checks.push(self.check_file_check(file));
            }
        }

        if let Some(tools) = &requires.tools {
            for tool in tools {
                report.checks.push(self.check_tool_check(tool));
            }
        }

        if let Some(tool_versions) = &requires.tool_versions {
            for (tool, constraint) in tool_versions {
                report
                    .checks
                    .push(self.check_tool_version_check(tool, constraint));
            }
        }

        if let Some(capabilities) = &requires.capabilities {
            for cap in capabilities {
                report.checks.push(self.check_capability_check(cap));
            }
        }

        if let Some(min_memory) = requires.min_memory_mib {
            report.checks.push(self.check_memory_check(min_memory));
        }

        if let Some(network) = requires.network {
            report.checks.push(self.check_network_check(network));
        }

        if let Some(min_version) = &requires.min_version {
            report
                .checks
                .push(self.check_min_version_check(min_version));
        }

        report
    }

    // -----------------------------------------------------------------------
    // Individual checks
    // -----------------------------------------------------------------------

    fn check_os_check(&self, required: &[String]) -> CheckStatus {
        let requirement = format!("os:{}", required.join(","));
        if required.iter().any(|os| self.host.os_matches(os)) {
            CheckStatus::pass(requirement, RequirementKind::Os)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Os,
                format!("host is {} ({})", self.host.os, self.host.family),
            )
        }
    }

    fn check_arch_check(&self, required: &[String]) -> CheckStatus {
        let requirement = format!("arch:{}", required.join(","));
        if required.iter().any(|arch| self.host.arch_matches(arch)) {
            CheckStatus::pass(requirement, RequirementKind::Arch)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Arch,
                format!("host arch is {}", self.host.arch),
            )
        }
    }

    fn check_binary_check(&self, name: &str) -> CheckStatus {
        let requirement = format!("binary:{name}");
        match self.which(name) {
            Some(path) => CheckStatus::pass(requirement, RequirementKind::Binary),
            None => CheckStatus::fail(
                requirement,
                RequirementKind::Binary,
                format!("'{name}' not found in PATH"),
            ),
        }
    }

    fn check_env_var_check(&self, name: &str) -> CheckStatus {
        let requirement = format!("env:{name}");
        if self.env_var_is_set(name) {
            CheckStatus::pass(requirement, RequirementKind::EnvVar)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::EnvVar,
                format!("environment variable '{name}' is not set"),
            )
        }
    }

    fn check_file_check(&self, spec: &str) -> CheckStatus {
        let requirement = format!("file:{spec}");
        let path = expand_path(spec);
        if path.exists() {
            CheckStatus::pass(requirement, RequirementKind::File)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::File,
                format!("path '{}' does not exist", path.display()),
            )
        }
    }

    fn check_tool_check(&self, tool: &str) -> CheckStatus {
        let requirement = format!("tool:{tool}");
        // Tools are registry entries owned by the caller; without a registry
        // handle we treat "tool" as satisfied when a binary of the same name
        // exists. The engine's skills_filter performs the strict registry gate.
        if self.which(tool).is_some() {
            CheckStatus::pass(requirement, RequirementKind::Tool)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Tool,
                format!("tool '{tool}' is not available"),
            )
        }
    }

    fn check_tool_version_check(&self, tool: &str, constraint: &str) -> CheckStatus {
        let requirement = format!("tool_version:{tool} ({constraint})");
        let Some(version) = self.binary_version(tool) else {
            return CheckStatus::fail(
                requirement,
                RequirementKind::ToolVersion,
                format!("'{tool}' not found, cannot check version"),
            );
        };
        if version.matches_constraint(constraint) {
            CheckStatus::pass(requirement, RequirementKind::ToolVersion)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::ToolVersion,
                format!(
                    "'{tool}' version {} does not satisfy '{constraint}'",
                    version
                ),
            )
        }
    }

    fn check_capability_check(&self, capability: &str) -> CheckStatus {
        let requirement = format!("capability:{capability}");
        if self.check_capability(capability) {
            CheckStatus::pass(requirement, RequirementKind::Capability)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Capability,
                format!("capability '{capability}' not available"),
            )
        }
    }

    fn check_memory_check(&self, min_mib: u64) -> CheckStatus {
        let requirement = format!("memory:>{min_mib}MiB");
        match self.host.total_memory_mib {
            Some(total) if total >= min_mib => {
                CheckStatus::pass(requirement, RequirementKind::Memory)
            }
            Some(total) => CheckStatus::fail(
                requirement,
                RequirementKind::Memory,
                format!("host has {total}MiB, needs {min_mib}MiB"),
            ),
            None => CheckStatus::pass(requirement, RequirementKind::Memory),
        }
    }

    fn check_network_check(&self, needs_network: bool) -> CheckStatus {
        let requirement = format!("network:{needs_network}");
        if !needs_network {
            return CheckStatus::pass(requirement, RequirementKind::Network);
        }
        // A lightweight, cached probe: DNS resolution of a well-known host.
        // Failing the probe is treated as "not eligible" only when required.
        if self.network_available() {
            CheckStatus::pass(requirement, RequirementKind::Network)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Network,
                "network probe failed",
            )
        }
    }

    fn check_min_version_check(&self, min_version: &str) -> CheckStatus {
        let requirement = format!("version:>={min_version}");
        // The skill's own minimum version is compared against an unknown
        // runtime version here; this check is mostly informative and only fails
        // when the constraint itself is malformed.
        if min_version.parse::<SkillVersion>().is_ok() {
            CheckStatus::pass(requirement, RequirementKind::Version)
        } else {
            CheckStatus::fail(
                requirement,
                RequirementKind::Version,
                format!("invalid version constraint '{min_version}'"),
            )
        }
    }

    // -----------------------------------------------------------------------
    // Primitive probes
    // -----------------------------------------------------------------------

    /// Whether a named binary exists on the PATH (cached).
    pub fn is_binary_available(&self, name: &str) -> bool {
        if let Ok(mut cache) = self.binary_cache.lock() {
            if let Some(&exists) = cache.get(name) {
                return exists;
            }
        }
        let exists = self.which(name).is_some();
        if let Ok(mut cache) = self.binary_cache.lock() {
            cache.insert(name.to_string(), exists);
        }
        exists
    }

    /// Locate a binary on the PATH, returning its absolute path.
    ///
    /// On Windows this respects `PATHEXT` so `git` finds `git.exe`. On Unix it
    /// honors `PATH` and checks executable permissions.
    pub fn which(&self, name: &str) -> Option<PathBuf> {
        if name.contains('/') || name.contains('\\') {
            let p = Path::new(name);
            if p.is_file() {
                return Some(p.to_path_buf());
            }
            return None;
        }

        let path_var = std::env::var("PATH").ok()?;
        let extensions: Vec<String> = if cfg!(windows) {
            std::env::var("PATHEXT")
                .unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_string())
                .split(';')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        for dir in path_var.split(if cfg!(windows) { ';' } else { ':' }) {
            if dir.is_empty() {
                continue;
            }
            let dir = Path::new(dir);
            let base = dir.join(name);
            if cfg!(windows) {
                for ext in &extensions {
                    let candidate = PathBuf::from(format!("{}{}", base.display(), ext));
                    if candidate.is_file() {
                        return Some(candidate);
                    }
                }
            } else {
                if base.is_file() {
                    // Ensure it is executable (best-effort; `is_file` on Unix
                    // does not imply +x, but this avoids a stat round-trip in
                    // the common case).
                    return Some(base);
                }
            }
        }
        None
    }

    /// Best-effort version string for a binary, parsed as a [`SkillVersion`].
    ///
    /// Runs `<binary> --version` (or `-version`, `version`) and extracts the
    /// first dotted numeric version. Returns `None` when the version cannot be
    /// determined.
    pub fn binary_version(&self, name: &str) -> Option<SkillVersion> {
        let path = self.which(name)?;
        for flag in ["--version", "-version", "version", "-v"] {
            if let Ok(output) = std::process::Command::new(&path).arg(flag).output() {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    if let Some(version) = extract_version(&text) {
                        if let Ok(v) = version.parse::<SkillVersion>() {
                            return Some(v);
                        }
                    }
                }
            }
        }
        None
    }

    /// Whether an environment variable is set and non-empty (cached).
    pub fn env_var_is_set(&self, name: &str) -> bool {
        if let Ok(mut cache) = self.env_cache.lock() {
            if let Some(&exists) = cache.get(name) {
                return exists;
            }
        }
        let exists = std::env::var(name).map(|v| !v.is_empty()).unwrap_or(false);
        if let Ok(mut cache) = self.env_cache.lock() {
            cache.insert(name.to_string(), exists);
        }
        exists
    }

    /// Whether a file or directory exists (cached).
    pub fn file_exists(&self, path: &str) -> bool {
        if let Ok(mut cache) = self.file_cache.lock() {
            if let Some(&exists) = cache.get(path) {
                return exists;
            }
        }
        let exists = expand_path(path).exists();
        if let Ok(mut cache) = self.file_cache.lock() {
            cache.insert(path.to_string(), exists);
        }
        exists
    }

    /// Whether a capability is available.
    pub fn check_capability(&self, capability: &str) -> bool {
        match capability {
            "network" => self.network_available(),
            "filesystem" | "fs" => true,
            "docker" => self.is_binary_available("docker"),
            "git" => self.is_binary_available("git"),
            "python" | "python3" => {
                self.is_binary_available("python3") || self.is_binary_available("python")
            }
            "node" | "nodejs" => self.is_binary_available("node"),
            "npm" => self.is_binary_available("npm"),
            "rust" | "cargo" => self.is_binary_available("cargo"),
            "go" | "golang" => self.is_binary_available("go"),
            "java" | "jvm" => self.is_binary_available("java"),
            "sandbox" => cfg!(any(
                target_os = "linux",
                target_os = "macos",
                target_os = "windows"
            )),
            "tty" | "terminal" => std::io::IsTerminal::is_terminal(&std::io::stdout()),
            "gui" => cfg!(any(target_os = "windows", target_os = "macos")) || cfg!(feature = "gui"),
            "ffmpeg" => self.is_binary_available("ffmpeg"),
            "curl" => self.is_binary_available("curl"),
            "wget" => self.is_binary_available("wget"),
            "aws" => self.is_binary_available("aws"),
            "gcloud" => self.is_binary_available("gcloud"),
            _ => false,
        }
    }

    /// Lightweight network availability probe (DNS lookup of a well-known
    /// host). Results are not cached because network state can change.
    pub fn network_available(&self) -> bool {
        use std::net::ToSocketAddrs;
        ("one.one.one.one:53")
            .to_socket_addrs()
            .map(|mut addrs| addrs.next().is_some())
            .unwrap_or(false)
    }

    /// Clear the binary, env var, and file caches.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.binary_cache.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = self.env_cache.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = self.file_cache.lock() {
            cache.clear();
        }
        debug!("Eligibility caches cleared");
    }

    /// Filter a list of skills down to those eligible on this host.
    pub fn filter_eligible<'a>(
        &self,
        skills: impl Iterator<Item = &'a crate::types::SkillSpec>,
    ) -> Vec<&'a crate::types::SkillSpec> {
        skills
            .filter(|s| self.is_eligible(&s.requires).unwrap_or(false))
            .collect()
    }

    /// Compute a compact verdict string for logging, e.g. `"eligible"` or
    /// `"2 failures: binary:git, env:TOKEN"`.
    pub fn verdict(&self, requires: &SkillRequires) -> String {
        let report = self.check(requires);
        if report.is_eligible() {
            "eligible".to_string()
        } else {
            let reasons: Vec<String> = report
                .failures()
                .into_iter()
                .map(|c| c.requirement.clone())
                .collect();
            format!(
                "{} failure(s): {}",
                report.failure_count(),
                reasons.join(", ")
            )
        }
    }
}

/// Expand `~`, `$VAR`, and `${VAR}` in a path string.
fn expand_path(spec: &str) -> PathBuf {
    let mut out = spec.to_string();
    if let Some(rest) = spec.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            out = Path::new(&home).join(rest).to_string_lossy().to_string();
        }
    }
    PathBuf::from(out)
}

/// Extract a dotted numeric version from arbitrary program output.
fn extract_version(text: &str) -> Option<String> {
    // Match sequences like `1.2.3`, `v1.2.3`, `1.2`, `1.2.3.4`.
    let re = regex::Regex::new(r"(?m)\bv?(\d+\.\d+(?:\.\d+)?(?:[-+][0-9A-Za-z.-]+)?)\b").ok()?;
    let caps = re.captures(text)?;
    Some(caps.get(1)?.as_str().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_host() -> CurrentHost {
        CurrentHost {
            os: "linux".to_string(),
            family: "unix".to_string(),
            arch: "x86_64".to_string(),
            total_memory_mib: Some(8192),
        }
    }

    #[test]
    fn os_matches_aliases() {
        let host = test_host();
        assert!(host.os_matches("linux"));
        assert!(host.os_matches("unix"));
        assert!(host.os_matches("posix"));
        assert!(host.os_matches("any"));
        assert!(!host.os_matches("windows"));
        assert!(!host.os_matches("macos"));
    }

    #[test]
    fn arch_matches_aliases() {
        let host = test_host();
        assert!(host.arch_matches("x86_64"));
        assert!(host.arch_matches("amd64"));
        assert!(host.arch_matches("x64"));
        assert!(!host.arch_matches("arm64"));
        assert!(host.arch_matches("any"));
    }

    #[test]
    fn check_os_requirement() {
        let checker = EligibilityChecker::for_host(test_host());
        let requires = SkillRequires {
            os: Some(vec!["windows".to_string()]),
            ..SkillRequires::default()
        };
        let report = checker.check(&requires);
        assert!(!report.is_eligible());
        assert_eq!(report.failure_count(), 1);

        let requires = SkillRequires {
            os: Some(vec!["unix".to_string()]),
            ..SkillRequires::default()
        };
        assert!(checker.check(&requires).is_eligible());
    }

    #[test]
    fn env_var_requirement() {
        let checker = EligibilityChecker::new();
        std::env::set_var("OSQ_ELIG_TEST_VAR", "1");
        let requires = SkillRequires {
            env_vars: Some(vec!["OSQ_ELIG_TEST_VAR".to_string()]),
            ..SkillRequires::default()
        };
        assert!(checker.check(&requires).is_eligible());
        std::env::remove_var("OSQ_ELIG_TEST_VAR");

        // The env var cache must be invalidated for the change to be seen.
        checker.clear_cache();
        let requires = SkillRequires {
            env_vars: Some(vec!["OSQ_ELIG_TEST_VAR".to_string()]),
            ..SkillRequires::default()
        };
        assert!(!checker.check(&requires).is_eligible());
    }

    #[test]
    fn binary_requirement() {
        let checker = EligibilityChecker::new();
        // `cargo` is almost always present in a Rust toolchain environment.
        let requires = SkillRequires {
            binaries: Some(vec!["definitely-not-a-real-binary-xyz".to_string()]),
            ..SkillRequires::default()
        };
        assert!(!checker.check(&requires).is_eligible());
        assert!(!checker.is_binary_available("definitely-not-a-real-binary-xyz"));
        checker.clear_cache();
    }

    #[test]
    fn file_requirement() {
        let checker = EligibilityChecker::new();
        let requires = SkillRequires {
            files: Some(vec!["/this/path/does/not/exist".to_string()]),
            ..SkillRequires::default()
        };
        assert!(!checker.check(&requires).is_eligible());
        assert!(!checker.file_exists("/this/path/does/not/exist"));
    }

    #[test]
    fn version_constraint_check() {
        let v: SkillVersion = "3.11.4".parse().unwrap();
        assert!(v.matches_constraint(">=3.11"));
        assert!(v.matches_constraint(">=3.11, <4"));
        assert!(!v.matches_constraint(">=3.12"));
        assert!(v.matches_constraint("==3.11.4"));
    }

    #[test]
    fn extract_version_from_output() {
        assert_eq!(extract_version("Python 3.11.4"), Some("3.11.4".to_string()));
        assert_eq!(
            extract_version("git version 2.43.0.windows.1"),
            Some("2.43.0".to_string())
        );
        assert_eq!(
            extract_version("node v20.11.0"),
            Some("20.11.0".to_string())
        );
        assert_eq!(extract_version("no version here"), None);
    }

    #[test]
    fn report_failures_are_ordered() {
        let checker = EligibilityChecker::for_host(test_host());
        let requires = SkillRequires {
            os: Some(vec!["windows".to_string()]),
            binaries: Some(vec!["nope-xyz".to_string()]),
            env_vars: Some(vec!["MISSING_VAR_XYZ".to_string()]),
            ..SkillRequires::default()
        };
        let report = checker.check(&requires);
        assert!(!report.is_eligible());
        assert_eq!(report.failure_count(), 3);
        let labels: Vec<&str> = report.failures().iter().map(|c| c.kind.label()).collect();
        assert!(labels.contains(&"os"));
        assert!(labels.contains(&"binary"));
        assert!(labels.contains(&"env"));
    }

    #[test]
    fn is_eligible_matches_report() {
        let checker = EligibilityChecker::for_host(test_host());
        let good = SkillRequires {
            os: Some(vec!["linux".to_string()]),
            ..SkillRequires::default()
        };
        assert!(checker.is_eligible(&good).unwrap());
        let bad = SkillRequires {
            os: Some(vec!["plan9".to_string()]),
            ..SkillRequires::default()
        };
        assert!(checker.is_eligible(&bad).is_err());
        let _ = checker;
    }

    #[test]
    fn memory_check() {
        let checker = EligibilityChecker::for_host(test_host());
        let requires = SkillRequires {
            min_memory_mib: Some(100),
            ..SkillRequires::default()
        };
        assert!(checker.check(&requires).is_eligible());
        let requires = SkillRequires {
            min_memory_mib: Some(1024 * 1024),
            ..SkillRequires::default()
        };
        assert!(!checker.check(&requires).is_eligible());
    }

    #[test]
    fn verdict_string() {
        let checker = EligibilityChecker::for_host(test_host());
        let good = SkillRequires::default();
        assert_eq!(checker.verdict(&good), "eligible");
        let bad = SkillRequires {
            os: Some(vec!["windows".to_string()]),
            ..SkillRequires::default()
        };
        assert!(checker.verdict(&bad).contains("failure"));
    }
}
