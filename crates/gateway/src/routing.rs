//! Routing RPC handlers.
//!
//! Provides `rpc_routing` (per-session routing hold) and `rpc_router`
//! (routing decision records) for inspecting and controlling the model
//! routing layer.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A per-session routing hold. When a session is held, the router pins the
/// model/provider instead of running the normal selection strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingHold {
    pub session_id: String,
    pub model: String,
    pub provider: Option<String>,
    pub reason: String,
    pub held_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// A recorded routing decision for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub id: String,
    pub session_id: String,
    pub requested_model: Option<String>,
    pub selected_model: String,
    pub selected_provider: String,
    pub strategy: String,
    pub reason: String,
    pub fallback_used: bool,
    pub decided_at: DateTime<Utc>,
}

/// In-memory routing state.
#[derive(Clone, Default)]
pub struct RoutingStore {
    holds: Arc<Mutex<HashMap<String, RoutingHold>>>,
    decisions: Arc<Mutex<Vec<RoutingDecision>>>,
}

impl RoutingStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a routing hold for a session.
    pub fn set_hold(&self, hold: RoutingHold) {
        self.holds.lock().insert(hold.session_id.clone(), hold);
    }

    /// Get the active hold for a session.
    pub fn get_hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.holds.lock().get(session_id).cloned()
    }

    /// Release a routing hold.
    pub fn release_hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.holds.lock().remove(session_id)
    }

    /// List all active holds.
    pub fn list_holds(&self) -> Vec<RoutingHold> {
        let holds: Vec<RoutingHold> = self.holds.lock().values().cloned().collect();
        holds
    }

    /// Record a routing decision.
    pub fn record_decision(&self, decision: RoutingDecision) {
        self.decisions.lock().push(decision);
    }

    /// List decisions, optionally filtered by session.
    pub fn list_decisions(&self, session_id: Option<&str>, limit: usize) -> Vec<RoutingDecision> {
        let decisions = self.decisions.lock();
        let filtered: Vec<RoutingDecision> = decisions
            .iter()
            .filter(|d| session_id.is_none_or(|sid| d.session_id == sid))
            .cloned()
            .collect();
        filtered.into_iter().rev().take(limit).collect()
    }
}

/// Register routing RPC handlers on the given registry.
pub fn register_routing_handlers(registry: &mut RpcRegistry, store: RoutingStore) {
    let store = Arc::new(store);

    // routing.hold — pin a model/provider for a session
    registry.register(rpc_handler("routing.hold", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?;
                let provider = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("manual hold")
                    .to_string();
                let expires_at = params
                    .get("expires_at")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));

                let hold = RoutingHold {
                    session_id: session_id.to_string(),
                    model: model.to_string(),
                    provider,
                    reason,
                    held_at: Utc::now(),
                    expires_at,
                };
                store.set_hold(hold.clone());
                serde_json::to_value(hold).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // routing.get_hold — fetch the active hold for a session
    registry.register(rpc_handler("routing.get_hold", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                match store.get_hold(session_id) {
                    Some(hold) => Ok(serde_json::to_value(hold)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Ok(serde_json::json!({
                        "session_id": session_id,
                        "held": false,
                    })),
                }
            }
        }
    }));

    // routing.release — release a routing hold
    registry.register(rpc_handler("routing.release", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                match store.release_hold(session_id) {
                    Some(_) => Ok(serde_json::json!({"released": true, "session_id": session_id})),
                    None => Err(AppError::not_found(format!(
                        "No routing hold for session '{session_id}'"
                    ))),
                }
            }
        }
    }));

    // routing.list_holds — list all active routing holds
    registry.register(rpc_handler("routing.list_holds", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let holds = store.list_holds();
                Ok(serde_json::json!({
                    "holds": holds,
                    "count": holds.len(),
                }))
            }
        }
    }));

    // router.record — record a routing decision
    registry.register(rpc_handler("router.record", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let selected_model = params
                    .get("selected_model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'selected_model' parameter"))?;
                let selected_provider = params
                    .get("selected_provider")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        AppError::bad_request("Missing 'selected_provider' parameter")
                    })?;
                let requested_model = params
                    .get("requested_model")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let strategy = params
                    .get("strategy")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let fallback_used = params
                    .get("fallback_used")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let decision = RoutingDecision {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    requested_model,
                    selected_model: selected_model.to_string(),
                    selected_provider: selected_provider.to_string(),
                    strategy,
                    reason,
                    fallback_used,
                    decided_at: Utc::now(),
                };
                store.record_decision(decision.clone());
                serde_json::to_value(decision).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // router.history — list routing decisions for a session
    registry.register(rpc_handler("router.history", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params.get("session_id").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
                let decisions = store.list_decisions(session_id, limit);
                Ok(serde_json::json!({
                    "decisions": decisions,
                    "count": decisions.len(),
                }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_routing_hold_lifecycle() {
        let store = RoutingStore::new();
        let mut registry = RpcRegistry::new();
        register_routing_handlers(&mut registry, store);

        let params = serde_json::json!({
            "session_id": "s1",
            "model": "gpt-4o",
            "provider": "openai",
            "reason": "user request",
        });
        let r = registry.dispatch("routing.hold", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("routing.get_hold", serde_json::json!({"session_id": "s1"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["model"], "gpt-4o");

        let r = registry
            .dispatch("routing.release", serde_json::json!({"session_id": "s1"}))
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("routing.get_hold", serde_json::json!({"session_id": "s1"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["held"], false);
    }

    #[tokio::test]
    async fn test_router_record_and_history() {
        let store = RoutingStore::new();
        let mut registry = RpcRegistry::new();
        register_routing_handlers(&mut registry, store);

        let params = serde_json::json!({
            "session_id": "s1",
            "selected_model": "claude-sonnet-4",
            "selected_provider": "anthropic",
            "strategy": "default",
            "reason": "routing decision",
        });
        let r = registry.dispatch("router.record", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("router.history", serde_json::json!({"session_id": "s1"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }
}
