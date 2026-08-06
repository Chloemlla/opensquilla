//! Sandbox settings model and combination validation.
//!
//! Port of `src/opensquilla/sandbox/config.py`. The settings live in their own
//! module rather than being glued onto a gateway config so the validation
//! rules can be unit-tested without booting the gateway.
//!
//! The four-way truth table for the two feature switches (`sandbox` and
//! `security_grading`) is implemented in [`SandboxSettings::validate_combination`],
//! which returns an [`EffectiveMode`] instead of mutating silently.

use serde::{Deserialize, Serialize};

use crate::run_mode::{RunMode, RunModeConfigInput, normalize_run_mode};

/// Selectable sandbox backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Pick the platform-appropriate backend automatically.
    #[default]
    Auto,
    /// Linux bubblewrap backend.
    Bubblewrap,
    /// macOS Seatbelt backend.
    Seatbelt,
    /// No-op passthrough backend.
    Noop,
    /// Windows default backend.
    WindowsDefault,
}

impl Backend {
    /// The stable string form.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Auto => "auto",
            Backend::Bubblewrap => "bubblewrap",
            Backend::Seatbelt => "seatbelt",
            Backend::Noop => "noop",
            Backend::WindowsDefault => "windows_default",
        }
    }

    /// Parse a backend name. The removed `windows_restricted_token` name is
    /// rejected explicitly (it was merged into `windows_default`).
    pub fn parse(value: &str) -> Result<Backend, String> {
        let value = value.trim().to_lowercase();
        if value == "windows_restricted_token" {
            return Err(
                "windows_restricted_token was removed; use backend='windows_default' or backend='auto'"
                    .to_string(),
            );
        }
        match value.as_str() {
            "auto" => Ok(Backend::Auto),
            "bubblewrap" => Ok(Backend::Bubblewrap),
            "seatbelt" => Ok(Backend::Seatbelt),
            "noop" => Ok(Backend::Noop),
            "windows_default" => Ok(Backend::WindowsDefault),
            other => Err(format!("unknown sandbox backend '{other}'")),
        }
    }
}

/// The network posture applied by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkDefault {
    /// No network access.
    None,
    /// Domain-allowlist via the managed proxy.
    #[default]
    ProxyAllowlist,
}

impl NetworkDefault {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkDefault::None => "none",
            NetworkDefault::ProxyAllowlist => "proxy_allowlist",
        }
    }
}

/// Who reviews approval requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalsReviewer {
    /// The human user.
    User,
    /// Automatic review heuristics.
    #[default]
    AutoReview,
}

/// Security grading levels ordered by strictness.
///
/// Integer ordering is load-bearing: callers can write `level >= Strict` to
/// mean "at least strict".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityLevel {
    /// Legacy/compatibility mode; only reachable when
    /// `allow_legacy_mode` is explicitly true.
    Disabled,
    /// Default for normal agent tool execution.
    #[default]
    Standard,
    /// Higher-risk actions with tighter limits.
    Strict,
    /// Minimum-visibility deny-by-default posture with required approval.
    Locked,
}

impl SecurityLevel {
    /// The short display label (`L0-disabled`, `L1-standard`, ...).
    pub fn label(self) -> &'static str {
        match self {
            SecurityLevel::Disabled => "L0-disabled",
            SecurityLevel::Standard => "L1-standard",
            SecurityLevel::Strict => "L2-strict",
            SecurityLevel::Locked => "L3-locked",
        }
    }
}

/// Resolved runtime posture after combination validation.
///
/// The gateway logs one line containing these fields on boot so operators can
/// see at a glance which way the switches ended up pointing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveMode {
    pub sandbox_enabled: bool,
    pub grading_enabled: bool,
    pub default_level: SecurityLevel,
    pub backend: Backend,
    pub insecure_mode: bool,
    /// Human/machine-readable notes explaining coercions.
    pub notes: Vec<String>,
}

impl EffectiveMode {
    /// The serializable payload shape.
    pub fn as_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "sandbox_enabled": self.sandbox_enabled,
            "grading_enabled": self.grading_enabled,
            "default_level": self.default_level.label(),
            "backend": self.backend.as_str(),
            "insecure_mode": self.insecure_mode,
            "notes": self.notes,
        })
    }
}

/// Top-level sandbox configuration.
///
/// Two independent switches:
/// * `sandbox` — whether isolation is enforced at all.
/// * `security_grading` — whether the level-selection + approval flow is
///   active. When false, the system uses a fixed `STANDARD` policy with no
///   dynamic escalation.
///
/// Both default to `true` so fresh installs start in the Managed Execution
/// posture. Invalid combinations are coerced with an explicit note via
/// [`SandboxSettings::validate_combination`]; the coercion is deliberate so
/// upgrades of existing deployments do not hard-fail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxSettings {
    /// Whether isolation is enforced at all.
    pub sandbox: bool,
    /// Whether level-selection + approval flow is active.
    pub security_grading: bool,
    /// The default security level.
    pub default_level: SecurityLevel,
    /// The selected backend.
    pub backend: Backend,
    /// Whether the legacy `DISABLED` level may be selected.
    pub allow_legacy_mode: bool,
    /// Explicit run mode, when set (overrides the two switches).
    pub run_mode: Option<RunMode>,
    /// Whether automatic sandbox setup is enabled.
    pub auto_setup: bool,
    /// Whether the host root is mounted read-only.
    pub host_root_readonly: bool,
    /// Whether `/tmp` is excluded from the writable set.
    pub exclude_slash_tmp: bool,
    /// Whether `TMPDIR` is excluded from the environment allowlist.
    pub exclude_tmpdir_env_var: bool,
    /// Who reviews approval requests.
    pub approvals_reviewer: ApprovalsReviewer,
    /// The default network posture.
    pub network_default: NetworkDefault,
    /// Denial threshold before a cooldown is enforced.
    pub denial_threshold: u32,
    /// The `permissions.default_mode` string (`"off"`, `"bypass"`, `"full"`,
    /// `"standard"`, ...). `None` means "not configured". Consumed by
    /// [`SandboxSettings::run_mode_input`] so the run-mode helpers can see the
    /// permissions posture, mirroring the Python `config.permissions`.
    #[serde(default)]
    pub permissions_default_mode: Option<String>,
    /// Extra read-only mounts.
    pub extra_ro_mounts: Vec<String>,
    /// Extra read-write mounts.
    pub extra_rw_mounts: Vec<String>,
    /// Denied read roots.
    pub denied_read_roots: Vec<String>,
    /// Denied read globs.
    pub denied_read_globs: Vec<String>,
    /// CPU seconds cap.
    pub cpu_seconds: u32,
    /// Memory cap in MiB.
    pub memory_mb: u32,
    /// Wall-clock cap in seconds.
    pub wall_seconds: u32,
}

impl Default for SandboxSettings {
    fn default() -> Self {
        Self {
            sandbox: true,
            security_grading: true,
            default_level: SecurityLevel::Standard,
            backend: Backend::Auto,
            allow_legacy_mode: false,
            run_mode: None,
            auto_setup: true,
            host_root_readonly: true,
            exclude_slash_tmp: false,
            exclude_tmpdir_env_var: false,
            approvals_reviewer: ApprovalsReviewer::AutoReview,
            network_default: NetworkDefault::ProxyAllowlist,
            denial_threshold: 3,
            permissions_default_mode: None,
            extra_ro_mounts: Vec::new(),
            extra_rw_mounts: Vec::new(),
            denied_read_roots: Vec::new(),
            denied_read_globs: Vec::new(),
            cpu_seconds: 30,
            memory_mb: 1024,
            wall_seconds: 60,
        }
    }
}

impl SandboxSettings {
    /// The effective switch state after applying an explicit `run_mode`.
    fn effective_switches(&self) -> (bool, bool) {
        match self.run_mode {
            Some(RunMode::Full) => (false, false),
            Some(RunMode::Standard) | Some(RunMode::Trusted) => (true, true),
            None => (self.sandbox, self.security_grading),
        }
    }

    /// Validate that the `DISABLED` level is only selected with explicit
    /// `allow_legacy_mode`, and that an explicit run mode is coherent.
    pub fn check_constraints(&self) -> Result<(), String> {
        if self.default_level == SecurityLevel::Disabled && !self.allow_legacy_mode {
            return Err(
                "default_level=DISABLED requires allow_legacy_mode=True; legacy mode must be opted into explicitly".to_string(),
            );
        }
        if let Some(mode) = self.run_mode {
            let _ = normalize_run_mode(Some(mode.as_str()), RunMode::Trusted)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Resolve the two switches into an [`EffectiveMode`].
    ///
    /// Truth table:
    /// * `sandbox=true, grading=true` — full mode, level selection on.
    /// * `sandbox=true, grading=false` — isolation on, fixed `STANDARD`
    ///   policy, approval escalation off.
    /// * `sandbox=false, grading=true` — inconsistent; grading coerced to
    ///   `false` (never silent; a note is recorded).
    /// * `sandbox=false, grading=false` — legacy mode; a note is recorded so
    ///   running without sandbox is never invisible.
    pub fn validate_combination(&self) -> EffectiveMode {
        let mut notes: Vec<String> = Vec::new();
        let (sandbox_enabled, mut grading_enabled) = self.effective_switches();
        let mut level = self.default_level;

        if !sandbox_enabled && grading_enabled {
            grading_enabled = false;
            notes.push("grading_coerced_to_false_because_sandbox_disabled".to_string());
        }
        if !grading_enabled && sandbox_enabled {
            level = SecurityLevel::Standard;
            notes.push("fixed_standard_policy".to_string());
        }

        let insecure = !sandbox_enabled;
        if insecure {
            notes.push("insecure_mode".to_string());
            if !self.allow_legacy_mode {
                notes.push("legacy_flag_missing".to_string());
            }
        }

        EffectiveMode {
            sandbox_enabled,
            grading_enabled,
            default_level: level,
            backend: self.backend,
            insecure_mode: insecure,
            notes,
        }
    }

    /// The run-mode config surface derived from these settings.
    ///
    /// The concrete booleans are treated as explicitly set (Rust settings are
    /// always fully materialized), so the `None`-defaults-to-FULL branch of
    /// [`crate::run_mode::config_run_mode`] does not fire for real settings.
    pub fn run_mode_input(&self) -> RunModeConfigInput {
        RunModeConfigInput {
            run_mode: self.run_mode,
            sandbox: Some(self.sandbox),
            security_grading: Some(self.security_grading),
            permissions_default_mode: self.permissions_default_mode.clone(),
        }
    }

    /// The project-default run mode, using the full config surface.
    pub fn project_default_run_mode(&self) -> RunMode {
        crate::run_mode::project_default_run_mode(&self.run_mode_input())
    }

    /// The run mode the sandbox runtime is capable of executing under.
    pub fn runtime_capability_run_mode(&self) -> RunMode {
        crate::run_mode::sandbox_runtime_capability_mode(&self.run_mode_input())
    }

    /// The resolved run mode for this configuration.
    pub fn effective_run_mode(&self) -> RunMode {
        crate::run_mode::config_run_mode(&self.run_mode_input())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_posture_is_managed_execution() {
        let settings = SandboxSettings::default();
        let mode = settings.validate_combination();
        assert!(mode.sandbox_enabled);
        assert!(mode.grading_enabled);
        assert!(!mode.insecure_mode);
        assert!(mode.notes.is_empty());
    }

    #[test]
    fn grading_off_uses_fixed_standard() {
        let settings = SandboxSettings {
            security_grading: false,
            ..SandboxSettings::default()
        };
        let mode = settings.validate_combination();
        assert!(mode.sandbox_enabled);
        assert!(!mode.grading_enabled);
        assert_eq!(mode.default_level, SecurityLevel::Standard);
        assert!(mode.notes.contains(&"fixed_standard_policy".to_string()));
    }

    #[test]
    fn sandbox_off_coerces_grading() {
        let settings = SandboxSettings {
            sandbox: false,
            ..SandboxSettings::default()
        };
        let mode = settings.validate_combination();
        assert!(!mode.sandbox_enabled);
        assert!(!mode.grading_enabled);
        assert!(mode.insecure_mode);
        assert!(mode
            .notes
            .contains(&"grading_coerced_to_false_because_sandbox_disabled".to_string()));
    }

    #[test]
    fn insecure_mode_notes_legacy_flag() {
        let settings = SandboxSettings {
            sandbox: false,
            security_grading: false,
            ..SandboxSettings::default()
        };
        let mode = settings.validate_combination();
        assert!(mode.insecure_mode);
        assert!(mode.notes.contains(&"insecure_mode".to_string()));
        assert!(mode.notes.contains(&"legacy_flag_missing".to_string()));
    }

    #[test]
    fn run_mode_overrides_switches() {
        let settings = SandboxSettings {
            run_mode: Some(RunMode::Full),
            ..SandboxSettings::default()
        };
        let mode = settings.validate_combination();
        assert!(!mode.sandbox_enabled);
        assert!(!mode.grading_enabled);
        assert!(mode.insecure_mode);
    }

    #[test]
    fn disabled_level_requires_legacy_flag() {
        let settings = SandboxSettings {
            default_level: SecurityLevel::Disabled,
            allow_legacy_mode: false,
            ..SandboxSettings::default()
        };
        assert!(settings.check_constraints().is_err());
        let settings = SandboxSettings {
            default_level: SecurityLevel::Disabled,
            allow_legacy_mode: true,
            ..SandboxSettings::default()
        };
        assert!(settings.check_constraints().is_ok());
    }

    #[test]
    fn backend_parse_rejects_removed_name() {
        assert!(Backend::parse("windows_restricted_token").is_err());
        assert_eq!(Backend::parse("auto").unwrap(), Backend::Auto);
        assert_eq!(Backend::parse("bubblewrap").unwrap(), Backend::Bubblewrap);
        assert!(Backend::parse("nope").is_err());
    }

    #[test]
    fn effective_mode_payload() {
        let settings = SandboxSettings::default();
        let payload = settings.validate_combination().as_dict();
        assert_eq!(payload["default_level"], "L1-standard");
        assert_eq!(payload["backend"], "auto");
        assert_eq!(payload["sandbox_enabled"], serde_json::Value::Bool(true));
    }

    #[test]
    fn run_mode_helpers_resolve_from_settings() {
        let settings = SandboxSettings::default();
        // Default switches (sandbox + grading on, no permission mode) resolve
        // to TRUSTED via the legacy fallback.
        assert_eq!(settings.effective_run_mode(), RunMode::Trusted);
        assert_eq!(settings.project_default_run_mode(), RunMode::Trusted);
        assert_eq!(settings.runtime_capability_run_mode(), RunMode::Trusted);

        let settings = SandboxSettings {
            run_mode: Some(RunMode::Full),
            ..SandboxSettings::default()
        };
        assert_eq!(settings.effective_run_mode(), RunMode::Full);
        assert_eq!(settings.runtime_capability_run_mode(), RunMode::Full);

        let settings = SandboxSettings {
            permissions_default_mode: Some("bypass".to_string()),
            ..SandboxSettings::default()
        };
        assert_eq!(settings.effective_run_mode(), RunMode::Full);
    }

    #[test]
    fn permissions_default_mode_serde_defaults() {
        // Round-trip: serialization includes the field, deserialization keeps
        // `None` when the value was null, and the explicit value survives.
        let settings = SandboxSettings::default();
        let json = serde_json::to_value(&settings).unwrap();
        let back: SandboxSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back.permissions_default_mode, None);
        assert_eq!(back.sandbox, true);

        let settings = SandboxSettings {
            permissions_default_mode: Some("bypass".to_string()),
            ..SandboxSettings::default()
        };
        let json = serde_json::to_value(&settings).unwrap();
        let back: SandboxSettings = serde_json::from_value(json).unwrap();
        assert_eq!(back.permissions_default_mode.as_deref(), Some("bypass"));
    }
}
