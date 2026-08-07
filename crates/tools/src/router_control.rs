//! Router control tool: router_control.
//!
//! Steers the routing tier at runtime by setting or clearing a per-session
//! routing hold. The Python `router_control.py` delegates to a
//! `RouterControlHoldStore` and a `router_control_config` on the tool context.
//!
//! The tools crate cannot depend on the gateway (where the production
//! `opensquilla_gateway::routing::RoutingStore` lives) without
//! a circular dependency, so this tool carries its own lightweight
//! [`RoutingHoldStore`] that mirrors the gateway's hold semantics. The gateway
//! is expected to bridge its own `RoutingStore` into this tool's store at boot
//! time when full integration is needed; for standalone tool-registry use the
//! in-tool store is self-sufficient.
//!
//! The canonical tier ladder (`c0` < `c1` < `c2` < `c3`) and legacy aliases
//! (`t0`-`t3`) are mirrored inline from the engine's `routing` module so this
//! crate does not need to depend on `opensquilla-engine`.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The canonical text tier ladder, lowest to highest.
pub const TEXT_TIERS: [&str; 4] = ["c0", "c1", "c2", "c3"];

/// Legacy tier aliases (`t0` -> `c0`, ...).
const LEGACY_TEXT_TIER_ALIASES: [(&str, &str); 4] =
    [("t0", "c0"), ("t1", "c1"), ("t2", "c2"), ("t3", "c3")];

/// Normalize a tier value to its canonical text tier id, accepting legacy
/// `t0`-`t3` aliases. Returns `None` for unknown or empty values.
pub fn normalize_text_tier(value: &str) -> Option<String> {
    let tier = value.trim().to_lowercase();
    if tier.is_empty() {
        return None;
    }
    if TEXT_TIERS.contains(&tier.as_str()) {
        return Some(tier);
    }
    LEGACY_TEXT_TIER_ALIASES
        .iter()
        .find(|(alias, _)| *alias == tier.as_str())
        .map(|(_, canonical)| canonical.to_string())
}

/// A per-session routing hold. When a session is held, the router pins the
/// model/provider/tier instead of running the normal selection strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingHold {
    /// The session this hold applies to.
    pub session_id: String,
    /// The canonical tier id (e.g. `c2`) the hold pins.
    pub tier: String,
    /// Short excerpt from the user message that requested the switch.
    pub evidence: String,
    /// Optional concise reason for observability.
    pub reason: Option<String>,
    /// When the hold was set.
    pub held_at: DateTime<Utc>,
}

/// In-memory per-session routing hold store. Mirrors the gateway's
/// `RoutingStore` hold semantics; `Clone` shares one underlying map.
#[derive(Clone, Default)]
pub struct RoutingHoldStore {
    holds: Arc<Mutex<HashMap<String, RoutingHold>>>,
}

impl RoutingHoldStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or replace) the routing hold for a session.
    pub fn set_hold(&self, hold: RoutingHold) {
        self.holds.lock().insert(hold.session_id.clone(), hold);
    }

    /// Get the active hold for a session, if any.
    pub fn get_hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.holds.lock().get(session_id).cloned()
    }

    /// Release (clear) the routing hold for a session, returning the removed
    /// hold if one was present.
    pub fn clear_hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.holds.lock().remove(session_id)
    }

    /// List all active holds.
    pub fn list_holds(&self) -> Vec<RoutingHold> {
        self.holds.lock().values().cloned().collect()
    }
}

/// The configured router-control targets. The Python layer resolves a
/// `target_id` against a `router_control_config` menu; here we accept any
/// canonical tier id as a valid target.
fn resolve_target(target_id: &str) -> Result<String, ToolError> {
    let trimmed = target_id.trim();
    if trimmed.is_empty() {
        return Err(ToolError::new(
            "INVALID_TARGET",
            "target_id is required for set_hold",
        ));
    }
    normalize_text_tier(trimmed).ok_or_else(|| {
        ToolError::new(
            "INVALID_TARGET",
            format!(
                "Unknown router target '{}'. Use a canonical tier id (c0, c1, c2, c3) or legacy alias (t0-t3).",
                trimmed
            ),
        )
    })
}

/// Tool for controlling the Squilla router for a session.
pub struct RouterControlTool {
    store: RoutingHoldStore,
    /// Whether router control is enabled. When false, the tool rejects all
    /// actions with a clear message (mirroring the Python config gate).
    enabled: bool,
}

impl RouterControlTool {
    /// Create a new router_control tool backed by the given hold store.
    ///
    /// When `enabled` is false the tool rejects all actions. Construct with
    /// [`RouterControlTool::enabled`] to allow set/clear.
    pub fn new(store: RoutingHoldStore, enabled: bool) -> Self {
        Self { store, enabled }
    }

    /// Create an enabled tool with a fresh in-memory store.
    pub fn enabled() -> Self {
        Self::new(RoutingHoldStore::new(), true)
    }

    /// Create a disabled tool (rejects all actions). Matches the default
    /// behavior when no router-control config is present.
    pub fn disabled() -> Self {
        Self::new(RoutingHoldStore::new(), false)
    }

    /// Create the tool from an existing store, enabled by default.
    pub fn with_store(store: RoutingHoldStore) -> Self {
        Self::new(store, true)
    }
}

impl Default for RouterControlTool {
    fn default() -> Self {
        Self::disabled()
    }
}

#[async_trait]
impl Tool for RouterControlTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "router_control",
                concat!(
                    "Control the Squilla router for this session. Use only when the user asks ",
                    "to switch to a configured route or restore automatic routing. ",
                    "Supports set_hold (pin a tier) and clear_hold (restore automatic routing).",
                ),
                HashMap::from([
                    (
                        "action".to_string(),
                        ParameterDefinition::string("Set a short-lived router hold or clear it")
                            .enum_values(vec!["set_hold".into(), "clear_hold".into()]),
                    ),
                    (
                        "session_id".to_string(),
                        ParameterDefinition::required_string("The session to apply the hold to"),
                    ),
                    (
                        "target_id".to_string(),
                        ParameterDefinition::string(
                            "Canonical target id (tier: c0, c1, c2, c3 or t0-t3). Required for set_hold.",
                        ),
                    ),
                    (
                        "evidence".to_string(),
                        ParameterDefinition::required_string(
                            "Short excerpt from the user message that requested the switch",
                        ),
                    ),
                    (
                        "reason".to_string(),
                        ParameterDefinition::string("Optional concise reason for observability"),
                    ),
                ]),
            )
            .category("routing")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        if !self.enabled {
            return Err(ToolError::new(
                "ROUTER_CONTROL_DISABLED",
                "squilla router is disabled or unavailable",
            ));
        }

        let session_id = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?
            .trim()
            .to_string();
        if session_id.is_empty() {
            return Err(ToolError::invalid_args("session_id must not be empty"));
        }

        let action = params["action"].as_str().unwrap_or("").trim().to_string();
        let evidence = params["evidence"].as_str().unwrap_or("").trim().to_string();

        if evidence.is_empty() {
            return Err(ToolError::invalid_args("evidence is required"));
        }

        match action.as_str() {
            "clear_hold" => {
                let removed = self.store.clear_hold(&session_id);
                let data = serde_json::json!({
                    "status": "ok",
                    "action": "clear_hold",
                    "session_id": session_id,
                    "evidence": evidence,
                    "had_hold": removed.is_some(),
                    "replay_required": removed.is_some(),
                });
                Ok(ToolOutput::success_with_data(
                    format!("Cleared router hold for session {}", session_id),
                    data,
                ))
            }
            "set_hold" => {
                let target_id = params["target_id"].as_str().unwrap_or("");
                let tier = resolve_target(target_id)?;
                let reason = params["reason"].as_str().map(|s| s.trim().to_string());

                let hold = RoutingHold {
                    session_id: session_id.clone(),
                    tier: tier.clone(),
                    evidence: evidence.clone(),
                    reason,
                    held_at: Utc::now(),
                };
                self.store.set_hold(hold.clone());

                let data = serde_json::json!({
                    "status": "ok",
                    "action": "set_hold",
                    "session_id": session_id,
                    "target": tier,
                    "evidence": evidence,
                    "replay_required": true,
                    "held_at": hold.held_at.to_rfc3339(),
                });
                Ok(ToolOutput::success_with_data(
                    format!(
                        "Set router hold to tier {} for session {}",
                        tier, session_id
                    ),
                    data,
                ))
            }
            other => Err(ToolError::new(
                "UNSUPPORTED_ACTION",
                format!(
                    "Unsupported router_control action '{}'. Use set_hold or clear_hold.",
                    other
                ),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_text_tier_accepts_canonical_and_legacy() {
        assert_eq!(normalize_text_tier("c2"), Some("c2".into()));
        assert_eq!(normalize_text_tier("T3"), Some("c3".into()));
        assert_eq!(normalize_text_tier("t0"), Some("c0".into()));
        assert_eq!(normalize_text_tier(""), None);
        assert_eq!(normalize_text_tier("unknown"), None);
    }

    #[tokio::test]
    async fn test_router_control_disabled_rejects() {
        let tool = RouterControlTool::disabled();
        let result = tool
            .execute(serde_json::json!({
                "action": "set_hold",
                "session_id": "s1",
                "target_id": "c2",
                "evidence": "use c2",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "ROUTER_CONTROL_DISABLED");
    }

    #[tokio::test]
    async fn test_set_and_clear_hold() {
        let store = RoutingHoldStore::new();
        let tool = RouterControlTool::with_store(store.clone());

        let result = tool
            .execute(serde_json::json!({
                "action": "set_hold",
                "session_id": "s1",
                "target_id": "c2",
                "evidence": "user asked for c2",
                "reason": "explicit request",
            }))
            .await;
        assert!(result.is_ok(), "set_hold failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["target"], serde_json::json!("c2"));

        let hold = store.get_hold("s1").expect("hold present");
        assert_eq!(hold.tier, "c2");
        assert_eq!(hold.reason.as_deref(), Some("explicit request"));

        let result = tool
            .execute(serde_json::json!({
                "action": "clear_hold",
                "session_id": "s1",
                "evidence": "restore automatic",
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["had_hold"], serde_json::json!(true));
        assert!(store.get_hold("s1").is_none());
    }

    #[tokio::test]
    async fn test_set_hold_rejects_invalid_target() {
        let tool = RouterControlTool::enabled();
        let result = tool
            .execute(serde_json::json!({
                "action": "set_hold",
                "session_id": "s1",
                "target_id": "c9",
                "evidence": "bad",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_TARGET");
    }

    #[tokio::test]
    async fn test_set_hold_accepts_legacy_alias() {
        let store = RoutingHoldStore::new();
        let tool = RouterControlTool::with_store(store.clone());
        let result = tool
            .execute(serde_json::json!({
                "action": "set_hold",
                "session_id": "s1",
                "target_id": "t3",
                "evidence": "use top tier",
            }))
            .await;
        assert!(result.is_ok());
        assert_eq!(store.get_hold("s1").unwrap().tier, "c3");
    }

    #[tokio::test]
    async fn test_clear_hold_without_existing() {
        let tool = RouterControlTool::enabled();
        let result = tool
            .execute(serde_json::json!({
                "action": "clear_hold",
                "session_id": "ghost",
                "evidence": "nothing",
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["had_hold"], serde_json::json!(false));
    }

    #[tokio::test]
    async fn test_missing_evidence_rejected() {
        let tool = RouterControlTool::enabled();
        let result = tool
            .execute(serde_json::json!({
                "action": "set_hold",
                "session_id": "s1",
                "target_id": "c2",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_unsupported_action_rejected() {
        let tool = RouterControlTool::enabled();
        let result = tool
            .execute(serde_json::json!({
                "action": "frobnicate",
                "session_id": "s1",
                "evidence": "x",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "UNSUPPORTED_ACTION");
    }
}
