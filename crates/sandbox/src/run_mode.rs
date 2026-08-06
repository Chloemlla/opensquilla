//! Shared sandbox run-mode vocabulary.
//!
//! Port of `src/opensquilla/sandbox/run_mode.py`. Defines the three run modes
//! (STANDARD, TRUSTED, FULL) and the pure helpers that normalize a raw
//! configuration value into a mode, describe it for humans, and derive the
//! sandbox/grading switch patch it implies.

use serde::{Deserialize, Serialize};

/// The three sandbox run modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Standard-Sandbox: strictest sandboxing for every execution.
    Standard,
    /// Managed Execution: sandboxing with approval-based escalation.
    Trusted,
    /// Full Host Access: no sandbox, no grading.
    Full,
}

impl RunMode {
    /// The string value (`"standard"`, `"trusted"`, `"full"`).
    pub fn as_str(self) -> &'static str {
        match self {
            RunMode::Standard => "standard",
            RunMode::Trusted => "trusted",
            RunMode::Full => "full",
        }
    }
}

/// The configuration patch implied by a run mode: whether the sandbox and
/// security-grading switches are on, the network default, and the permission
/// default mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunModeConfigPatch {
    pub run_mode: RunMode,
    pub sandbox: bool,
    pub security_grading: bool,
    /// `"none"` or `"proxy_allowlist"`.
    pub network_default: String,
    pub permissions_default_mode: String,
}

const RUN_MODE_ALIASES: &[(&str, RunMode)] = &[
    ("on", RunMode::Standard),
    ("off", RunMode::Standard),
    ("bypass", RunMode::Full),
    ("standard", RunMode::Standard),
    ("standard-sandbox", RunMode::Standard),
    ("standard_sandbox", RunMode::Standard),
    ("trust", RunMode::Trusted),
    ("trusted", RunMode::Trusted),
    ("trusted-sandbox", RunMode::Trusted),
    ("trusted_sandbox", RunMode::Trusted),
    ("full", RunMode::Full),
    ("full-host-access", RunMode::Full),
    ("full_host_access", RunMode::Full),
];

/// Normalize a raw value into a [`RunMode`].
///
/// `None` or blank values resolve to `default` (which itself is normalized).
/// Unknown strings raise a [`RunModeError`] listing the accepted aliases.
pub fn normalize_run_mode(value: Option<&str>, default: RunMode) -> Result<RunMode, RunModeError> {
    if let Some(value) = value {
        let key = value.trim().to_lowercase();
        if key.is_empty() {
            return Ok(default);
        }
        if let Some((_, mode)) = RUN_MODE_ALIASES.iter().find(|(alias, _)| *alias == key) {
            return Ok(*mode);
        }
        let mut allowed: Vec<&str> = RUN_MODE_ALIASES.iter().map(|(a, _)| *a).collect();
        allowed.sort();
        return Err(RunModeError::UnknownAlias {
            value: key,
            allowed: allowed.join(", "),
        });
    }
    Ok(default)
}

/// Error returned when a run-mode string is not a known alias.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunModeError {
    #[error("run_mode must be one of: {allowed} (got '{value}')")]
    UnknownAlias { value: String, allowed: String },
}

/// Human-readable label for a mode.
pub fn display_name(mode: RunMode) -> &'static str {
    match mode {
        RunMode::Standard => "Standard-Sandbox",
        RunMode::Trusted => "Managed Execution",
        RunMode::Full => "Full Host Access",
    }
}

/// The execution target: `"sandbox"` unless the mode is FULL.
pub fn execution_target(mode: RunMode) -> &'static str {
    match mode {
        RunMode::Full => "host",
        RunMode::Standard | RunMode::Trusted => "sandbox",
    }
}

/// The approval behavior key (`"standard"`, `"trusted"`, `"full"`).
pub fn approval_behavior(mode: RunMode) -> &'static str {
    mode.as_str()
}

/// The config patch implied by a mode.
pub fn run_mode_config_patch(mode: RunMode) -> RunModeConfigPatch {
    match mode {
        RunMode::Full => RunModeConfigPatch {
            run_mode: RunMode::Full,
            sandbox: false,
            security_grading: false,
            network_default: "none".to_string(),
            permissions_default_mode: "full".to_string(),
        },
        RunMode::Standard | RunMode::Trusted => RunModeConfigPatch {
            run_mode: mode,
            sandbox: true,
            security_grading: true,
            network_default: "proxy_allowlist".to_string(),
            permissions_default_mode: "off".to_string(),
        },
    }
}

/// Derive a run mode from the legacy `sandbox_enabled` / `grading_enabled` /
/// `permissions_default_mode` switches.
pub fn legacy_state_to_run_mode(
    sandbox_enabled: bool,
    grading_enabled: bool,
    permissions_default_mode: Option<&str>,
) -> RunMode {
    let permission_mode = permissions_default_mode.unwrap_or("").trim().to_lowercase();
    if permission_mode == "bypass" || permission_mode == "full" {
        return RunMode::Full;
    }
    if permission_mode == "off" || permission_mode == "on" || permission_mode.is_empty() {
        return RunMode::Trusted;
    }
    if matches!(
        permission_mode.as_str(),
        "standard" | "standard-sandbox" | "standard_sandbox" | "restricted"
    ) {
        return RunMode::Standard;
    }
    if !sandbox_enabled {
        return RunMode::Trusted;
    }
    if sandbox_enabled && !grading_enabled {
        return RunMode::Standard;
    }
    RunMode::Trusted
}

/// The sandbox-related config surface consumed by [`config_run_mode`] and
/// friends.
///
/// This is the Rust stand-in for the Python duck-typed `config.sandbox`
/// submodel plus `config.permissions.default_mode`. A `None` boolean marks a
/// field that was not explicitly set, mirroring the pydantic
/// `model_fields_set` checks in `run_mode.py`; `None` for
/// `permissions_default_mode` means "not configured" (equivalent to `""`).
#[derive(Debug, Clone, Default)]
pub struct RunModeConfigInput {
    /// The explicit `sandbox.run_mode`, when set.
    pub run_mode: Option<RunMode>,
    /// `sandbox.sandbox`: `Some(b)` when explicitly set, `None` when unset.
    pub sandbox: Option<bool>,
    /// `sandbox.security_grading`: `Some(b)` when explicitly set, `None` when
    /// unset.
    pub security_grading: Option<bool>,
    /// `permissions.default_mode` (`"off"`, `"bypass"`, `"full"`, ...).
    pub permissions_default_mode: Option<String>,
}

/// Resolve the project run mode from a config surface.
///
/// Faithful port of `run_mode.py::config_run_mode`:
/// 1. an explicit `sandbox.run_mode` wins,
/// 2. a `bypass`/`full` permissions default forces FULL,
/// 3. an explicitly-set `sandbox=false` forces FULL,
/// 4. when neither switch was explicitly set the project defaults to FULL
///    (host access is the owner default; the runtime stays sandbox-capable),
/// 5. otherwise the legacy switch combination is normalized.
pub fn config_run_mode(input: &RunModeConfigInput) -> RunMode {
    if let Some(explicit) = input.run_mode {
        return explicit;
    }
    let permission_mode = input
        .permissions_default_mode
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if permission_mode == "bypass" || permission_mode == "full" {
        return RunMode::Full;
    }
    if let Some(sandbox) = input.sandbox {
        if !sandbox {
            return RunMode::Full;
        }
    }
    if input.sandbox.is_none() && input.security_grading.is_none() {
        return RunMode::Full;
    }
    legacy_state_to_run_mode(
        input.sandbox.unwrap_or(false),
        input.security_grading.unwrap_or(false),
        Some(&permission_mode),
    )
}

/// The project-default run mode for a config surface.
///
/// Alias of [`config_run_mode`], matching `run_mode.py::project_default_run_mode`.
pub fn project_default_run_mode(input: &RunModeConfigInput) -> RunMode {
    config_run_mode(input)
}

/// True when the FULL mode was explicitly selected (not merely defaulted).
///
/// Mirrors `run_mode.py::_full_mode_is_explicit`: an explicit
/// `sandbox.run_mode=FULL`, an explicitly-set `sandbox=false`, or a `full`
/// permissions default.
pub fn full_mode_is_explicit(input: &RunModeConfigInput) -> bool {
    if let Some(mode) = input.run_mode {
        return mode == RunMode::Full;
    }
    if let Some(sandbox) = input.sandbox {
        if !sandbox {
            return true;
        }
    }
    input
        .permissions_default_mode
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        == "full"
}

/// The run mode the sandbox runtime is *capable* of executing under.
///
/// When the project default is FULL only because nothing was explicitly
/// configured, the runtime still keeps a sandbox-capable posture (STANDARD) so
/// explicit Standard/Trusted calls can be honored. An explicit FULL selection
/// disables that capability. Mirrors
/// `run_mode.py::sandbox_runtime_capability_mode`.
pub fn sandbox_runtime_capability_mode(input: &RunModeConfigInput) -> RunMode {
    let configured = config_run_mode(input);
    if configured == RunMode::Full && !full_mode_is_explicit(input) {
        return RunMode::Standard;
    }
    configured
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_aliases() {
        assert_eq!(normalize_run_mode(Some("on"), RunMode::Full).unwrap(), RunMode::Standard);
        assert_eq!(normalize_run_mode(Some("bypass"), RunMode::Full).unwrap(), RunMode::Full);
        assert_eq!(normalize_run_mode(Some("trusted-sandbox"), RunMode::Full).unwrap(), RunMode::Trusted);
        assert_eq!(normalize_run_mode(Some("FULL_HOST_ACCESS"), RunMode::Standard).unwrap(), RunMode::Full);
        assert_eq!(normalize_run_mode(None, RunMode::Full).unwrap(), RunMode::Full);
        assert_eq!(normalize_run_mode(Some("  "), RunMode::Trusted).unwrap(), RunMode::Trusted);
        assert!(normalize_run_mode(Some("bogus"), RunMode::Full).is_err());
    }

    #[test]
    fn display_and_target() {
        assert_eq!(display_name(RunMode::Trusted), "Managed Execution");
        assert_eq!(execution_target(RunMode::Standard), "sandbox");
        assert_eq!(execution_target(RunMode::Full), "host");
        assert_eq!(approval_behavior(RunMode::Full), "full");
    }

    #[test]
    fn config_patch_shapes() {
        let full = run_mode_config_patch(RunMode::Full);
        assert!(!full.sandbox);
        assert!(!full.security_grading);
        assert_eq!(full.network_default, "none");
        assert_eq!(full.permissions_default_mode, "full");

        let trusted = run_mode_config_patch(RunMode::Trusted);
        assert!(trusted.sandbox);
        assert!(trusted.security_grading);
        assert_eq!(trusted.network_default, "proxy_allowlist");
        assert_eq!(trusted.permissions_default_mode, "off");
    }

    #[test]
    fn legacy_state_conversions() {
        assert_eq!(
            legacy_state_to_run_mode(true, true, Some("bypass")),
            RunMode::Full
        );
        assert_eq!(
            legacy_state_to_run_mode(true, true, Some("off")),
            RunMode::Trusted
        );
        assert_eq!(
            legacy_state_to_run_mode(true, true, Some("standard")),
            RunMode::Standard
        );
        assert_eq!(legacy_state_to_run_mode(false, true, Some("on")), RunMode::Trusted);
        assert_eq!(legacy_state_to_run_mode(true, false, Some("")), RunMode::Trusted);
        // Fallback when permission mode is unrecognized.
        assert_eq!(legacy_state_to_run_mode(false, false, Some("weird")), RunMode::Trusted);
        assert_eq!(legacy_state_to_run_mode(true, true, Some("weird")), RunMode::Trusted);
    }

    fn input(
        run_mode: Option<RunMode>,
        sandbox: Option<bool>,
        security_grading: Option<bool>,
        permissions_default_mode: Option<&str>,
    ) -> RunModeConfigInput {
        RunModeConfigInput {
            run_mode,
            sandbox,
            security_grading,
            permissions_default_mode: permissions_default_mode.map(|s| s.to_string()),
        }
    }

    #[test]
    fn config_run_mode_explicit_wins() {
        let cfg = input(Some(RunMode::Full), Some(true), Some(true), Some("off"));
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        let cfg = input(Some(RunMode::Standard), Some(false), Some(false), Some("bypass"));
        assert_eq!(config_run_mode(&cfg), RunMode::Standard);
    }

    #[test]
    fn config_run_mode_permissions_bypass_forces_full() {
        // `bypass` forces FULL as the project default, but it is not an
        // *explicit* FULL selection, so the runtime stays sandbox-capable.
        let cfg = input(None, Some(true), Some(true), Some("bypass"));
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        assert!(!full_mode_is_explicit(&cfg));
        assert_eq!(sandbox_runtime_capability_mode(&cfg), RunMode::Standard);
        // An explicit `full` permissions default IS explicit.
        let cfg = input(None, Some(true), Some(true), Some("full"));
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        assert!(full_mode_is_explicit(&cfg));
        assert_eq!(sandbox_runtime_capability_mode(&cfg), RunMode::Full);
    }

    #[test]
    fn config_run_mode_explicit_sandbox_off_is_full() {
        let cfg = input(None, Some(false), Some(true), Some("off"));
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        assert!(full_mode_is_explicit(&cfg));
    }

    #[test]
    fn config_run_mode_nothing_set_is_full_but_not_explicit() {
        let cfg = input(None, None, None, None);
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        assert!(!full_mode_is_explicit(&cfg));
        // The runtime stays sandbox-capable in that case.
        assert_eq!(sandbox_runtime_capability_mode(&cfg), RunMode::Standard);
    }

    #[test]
    fn config_run_mode_legacy_fallback() {
        // Both switches set: legacy normalization applies (off -> TRUSTED).
        let cfg = input(None, Some(true), Some(true), None);
        assert_eq!(config_run_mode(&cfg), RunMode::Trusted);
        let cfg = input(None, Some(true), Some(true), Some("standard"));
        assert_eq!(config_run_mode(&cfg), RunMode::Standard);
    }

    #[test]
    fn project_default_matches_config() {
        let cfg = input(None, Some(true), Some(true), Some("standard"));
        assert_eq!(project_default_run_mode(&cfg), config_run_mode(&cfg));
    }

    #[test]
    fn runtime_capability_explicit_full_stays_full() {
        let cfg = input(Some(RunMode::Full), Some(true), Some(true), None);
        assert_eq!(config_run_mode(&cfg), RunMode::Full);
        assert!(full_mode_is_explicit(&cfg));
        assert_eq!(sandbox_runtime_capability_mode(&cfg), RunMode::Full);
    }
}
