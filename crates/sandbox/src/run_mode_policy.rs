//! Principal-aware sandbox run-mode authorization helpers.
//!
//! Port of `src/opensquilla/sandbox/run_mode_policy.py`. Owners may select any
//! run mode (including FULL host access); non-owners are restricted to
//! STANDARD / TRUSTED. The module replaces the Python duck-typed `principal`
//! with an explicit [`Principal`] value type.

use serde::{Deserialize, Serialize};

use crate::run_mode::{RunMode, RunModeError, normalize_run_mode};

/// The principal (user or service) requesting a run mode.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Principal {
    /// Whether this principal owns the deployment.
    pub is_owner: bool,
    /// Free-form role label, e.g. `"owner"`, `"member"`.
    pub role: Option<String>,
    /// OAuth-style scope strings.
    pub scopes: Vec<String>,
    /// Whether the principal is authenticated.
    pub authenticated: bool,
}

impl Principal {
    /// The run modes this principal may select.
    pub fn allowed_run_modes(&self) -> Vec<RunMode> {
        allowed_run_modes_for_principal(self)
    }

    /// The default run mode for this principal.
    pub fn default_run_mode(&self) -> RunMode {
        default_run_mode_for_principal(self)
    }
}

const OWNER_ALLOWED_RUN_MODES: [RunMode; 3] = [RunMode::Standard, RunMode::Trusted, RunMode::Full];
const NON_OWNER_ALLOWED_RUN_MODES: [RunMode; 2] = [RunMode::Standard, RunMode::Trusted];

/// The run modes a principal may select.
pub fn allowed_run_modes_for_principal(principal: &Principal) -> Vec<RunMode> {
    if principal.is_owner {
        OWNER_ALLOWED_RUN_MODES.to_vec()
    } else {
        NON_OWNER_ALLOWED_RUN_MODES.to_vec()
    }
}

/// The default run mode: owners default to FULL, everyone else to TRUSTED.
pub fn default_run_mode_for_principal(principal: &Principal) -> RunMode {
    if principal.is_owner {
        RunMode::Full
    } else {
        RunMode::Trusted
    }
}

/// Whether `mode` is selectable by `principal`. Invalid aliases are treated as
/// not allowed (fail closed).
pub fn run_mode_allowed_for_principal(mode: Option<&str>, principal: &Principal) -> bool {
    let default = default_run_mode_for_principal(principal);
    let normalized = match normalize_run_mode(mode, default) {
        Ok(m) => m,
        Err(RunModeError::UnknownAlias { .. }) => return false,
    };
    allowed_run_modes_for_principal(principal).contains(&normalized)
}

/// Coerce a requested mode into one the principal may select. Unknown aliases
/// and disallowed modes fall back to the principal's default.
pub fn coerce_run_mode_for_principal(mode: Option<&str>, principal: &Principal) -> RunMode {
    let default = default_run_mode_for_principal(principal);
    let normalized = match normalize_run_mode(mode, default) {
        Ok(m) => m,
        Err(RunModeError::UnknownAlias { .. }) => return default,
    };
    if allowed_run_modes_for_principal(principal).contains(&normalized) {
        normalized
    } else {
        default
    }
}

/// The serializable principal payload consumed by auth/hello endpoints.
pub fn principal_payload(principal: &Principal) -> serde_json::Value {
    let mut scopes: Vec<String> = principal.scopes.clone();
    scopes.sort();
    serde_json::json!({
        "role": principal.role,
        "scopes": scopes,
        "isOwner": principal.is_owner,
        "authenticated": principal.authenticated,
    })
}

/// The serializable run-mode policy payload: allowed modes, the default, and
/// the reason FULL host access is disabled for non-owners.
pub fn run_mode_policy_payload(principal: &Principal) -> serde_json::Value {
    let allowed = allowed_run_modes_for_principal(principal);
    serde_json::json!({
        "allowedRunModes": allowed.iter().map(|m| m.as_str()).collect::<Vec<_>>(),
        "defaultRunMode": default_run_mode_for_principal(principal).as_str(),
        "fullHostAccessDisabledReason": if principal.is_owner { serde_json::Value::Null } else {
            serde_json::Value::String("owner_required".to_string())
        },
    })
}

/// The combined hello endpoint payload.
pub fn hello_auth_payload(principal: &Principal) -> serde_json::Value {
    serde_json::json!({
        "principal": principal_payload(principal),
        "runModePolicy": run_mode_policy_payload(principal),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner() -> Principal {
        Principal {
            is_owner: true,
            role: Some("owner".to_string()),
            scopes: vec!["run:full".to_string()],
            authenticated: true,
        }
    }

    fn member() -> Principal {
        Principal {
            is_owner: false,
            role: Some("member".to_string()),
            scopes: vec!["run:trusted".to_string()],
            authenticated: true,
        }
    }

    #[test]
    fn owners_can_select_full() {
        let p = owner();
        assert!(run_mode_allowed_for_principal(Some("full"), &p));
        assert_eq!(coerce_run_mode_for_principal(Some("full"), &p), RunMode::Full);
        assert_eq!(p.default_run_mode(), RunMode::Full);
    }

    #[test]
    fn non_owners_are_restricted() {
        let p = member();
        assert!(!run_mode_allowed_for_principal(Some("full"), &p));
        assert_eq!(coerce_run_mode_for_principal(Some("full"), &p), RunMode::Trusted);
        assert_eq!(coerce_run_mode_for_principal(Some("standard"), &p), RunMode::Standard);
        assert_eq!(p.default_run_mode(), RunMode::Trusted);
    }

    #[test]
    fn invalid_alias_fails_closed() {
        let p = member();
        assert!(!run_mode_allowed_for_principal(Some("bogus"), &p));
        assert_eq!(coerce_run_mode_for_principal(Some("bogus"), &p), RunMode::Trusted);
    }

    #[test]
    fn payloads_are_well_formed() {
        let p = member();
        let payload = hello_auth_payload(&p);
        assert_eq!(payload["principal"]["isOwner"], serde_json::Value::Bool(false));
        assert_eq!(
            payload["runModePolicy"]["fullHostAccessDisabledReason"],
            serde_json::Value::String("owner_required".to_string())
        );
        assert_eq!(payload["runModePolicy"]["defaultRunMode"], "trusted");
    }
}
