//! Sandbox RPC handlers.
//!
//! Provides `rpc_sandbox` for sandbox run-context management: building and
//! inspecting [`SandboxPolicy`] instances, selecting isolation levels for
//! operations, and previewing the policy that would apply to a given run.

use opensquilla_core::error::AppError;
use opensquilla_sandbox::policy::{
    AuditEntry, FilesystemPolicy, NetworkPolicy, ResourceLimits, SandboxLevel, SandboxPolicy,
    SandboxResult,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// An in-memory store of named sandbox policies keyed by run/context id.
#[derive(Clone, Default)]
pub struct SandboxContextStore {
    policies: Arc<Mutex<std::collections::HashMap<String, SandboxPolicy>>>,
    results: Arc<Mutex<std::collections::HashMap<String, SandboxResult>>>,
}

impl SandboxContextStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a policy for a given context id.
    pub fn put_policy(&self, context_id: &str, policy: SandboxPolicy) {
        self.policies.lock().insert(context_id.to_string(), policy);
    }

    /// Retrieve a policy by context id.
    pub fn get_policy(&self, context_id: &str) -> Option<SandboxPolicy> {
        self.policies.lock().get(context_id).cloned()
    }

    /// Remove a policy by context id.
    pub fn remove_policy(&self, context_id: &str) -> bool {
        self.policies.lock().remove(context_id).is_some()
    }

    /// Record a sandbox execution result for later inspection.
    pub fn record_result(&self, context_id: &str, result: SandboxResult) {
        self.results.lock().insert(context_id.to_string(), result);
    }

    /// Retrieve the last recorded result for a context id.
    pub fn get_result(&self, context_id: &str) -> Option<SandboxResult> {
        self.results.lock().get(context_id).cloned()
    }

    /// List all registered context ids.
    pub fn list(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.policies.lock().keys().cloned().collect();
        ids.sort();
        ids
    }
}

/// Serializable view of a sandbox level.
fn level_name(level: SandboxLevel) -> &'static str {
    match level {
        SandboxLevel::Standard => "STANDARD",
        SandboxLevel::Strict => "STRICT",
        SandboxLevel::Locked => "LOCKED",
    }
}

/// Resolve a sandbox level from a string parameter.
fn parse_level(s: &str) -> Result<SandboxLevel, AppError> {
    match s.to_ascii_uppercase().as_str() {
        "STANDARD" => Ok(SandboxLevel::Standard),
        "STRICT" => Ok(SandboxLevel::Strict),
        "LOCKED" => Ok(SandboxLevel::Locked),
        other => Err(AppError::bad_request(format!(
            "Unknown sandbox level '{other}'. Expected STANDARD, STRICT, or LOCKED."
        ))),
    }
}

/// Serialize a policy into a JSON value.
fn policy_to_value(policy: &SandboxPolicy) -> serde_json::Value {
    serde_json::to_value(policy).unwrap_or_else(|_| serde_json::Value::Null)
}

/// Register sandbox RPC handlers on the given registry.
pub fn register_sandbox_handlers(registry: &mut RpcRegistry, store: SandboxContextStore) {
    let store = Arc::new(store);

    // sandbox.select_level — recommend a level for a given operation
    registry.register(rpc_handler("sandbox.select_level", {
        move |params| async move {
            let operation = params
                .get("operation")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::bad_request("Missing 'operation' parameter"))?;
            let level = SandboxPolicy::select_level(operation);
            Ok(serde_json::json!({
                "operation": operation,
                "level": level_name(level),
            }))
        }
    }));

    // sandbox.build_policy — build a complete policy from a base level
    registry.register(rpc_handler("sandbox.build_policy", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let level_str = params
                    .get("level")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'level' parameter"))?;
                let level = parse_level(level_str)?;

                let overrides = if params.get("overrides").is_some() {
                    let policy: SandboxPolicy = serde_json::from_value(
                        params
                            .get("overrides")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null),
                    )
                    .map_err(|e| AppError::bad_request(format!("Invalid overrides: {e}")))?;
                    Some(policy)
                } else {
                    None
                };

                let policy = SandboxPolicy::build_policy(level, overrides);
                let context_id = params
                    .get("context_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                store.put_policy(&context_id, policy.clone());

                Ok(serde_json::json!({
                    "context_id": context_id,
                    "policy": policy_to_value(&policy),
                }))
            }
        }
    }));

    // sandbox.get_policy — retrieve a stored policy by context id
    registry.register(rpc_handler("sandbox.get_policy", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let context_id = params
                    .get("context_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'context_id' parameter"))?;
                match store.get_policy(context_id) {
                    Some(policy) => Ok(serde_json::json!({
                        "context_id": context_id,
                        "policy": policy_to_value(&policy),
                    })),
                    None => Err(AppError::not_found(format!(
                        "No sandbox policy for context '{context_id}'"
                    ))),
                }
            }
        }
    }));

    // sandbox.list_contexts — list all registered run context ids
    registry.register(rpc_handler("sandbox.list_contexts", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let ids = store.list();
                Ok(serde_json::json!({
                    "contexts": ids,
                    "count": ids.len(),
                }))
            }
        }
    }));

    // sandbox.remove_context — drop a stored policy
    registry.register(rpc_handler("sandbox.remove_context", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let context_id = params
                    .get("context_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'context_id' parameter"))?;
                if store.remove_policy(context_id) {
                    Ok(serde_json::json!({"removed": true, "context_id": context_id}))
                } else {
                    Err(AppError::not_found(format!(
                        "No sandbox context '{context_id}'"
                    )))
                }
            }
        }
    }));

    // sandbox.preview — describe the effective policy for an operation without storing it
    registry.register(rpc_handler("sandbox.preview", {
        move |params| async move {
            let operation = params
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("file_read");
            let level = SandboxPolicy::select_level(operation);
            let policy = SandboxPolicy::build_policy(level, None);
            Ok(serde_json::json!({
                "operation": operation,
                "recommended_level": level_name(level),
                "filesystem": serde_json::to_value(&policy.filesystem).unwrap_or(serde_json::Value::Null),
                "network": serde_json::to_value(&policy.network).unwrap_or(serde_json::Value::Null),
                "resource_limits": serde_json::to_value(&policy.resource_limits).unwrap_or(serde_json::Value::Null),
                "audit_enabled": policy.audit_enabled,
            }))
        }
    }));

    // sandbox.record_result — record an execution result for a context
    registry.register(rpc_handler("sandbox.record_result", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let context_id = params
                    .get("context_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'context_id' parameter"))?;
                let exit_code = params
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0) as i32;
                let stdout = params
                    .get("stdout")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let stderr = params
                    .get("stderr")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let duration_ms = params
                    .get("duration_ms")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let result = SandboxResult {
                    exit_code,
                    stdout,
                    stderr,
                    duration_ms,
                    audit_log: Vec::<AuditEntry>::new(),
                };
                store.record_result(context_id, result.clone());

                Ok(serde_json::json!({
                    "context_id": context_id,
                    "exit_code": result.exit_code,
                    "duration_ms": result.duration_ms,
                }))
            }
        }
    }));

    // sandbox.get_result — fetch the last recorded result for a context
    registry.register(rpc_handler("sandbox.get_result", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let context_id = params
                    .get("context_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'context_id' parameter"))?;
                match store.get_result(context_id) {
                    Some(result) => Ok(serde_json::json!({
                        "context_id": context_id,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "duration_ms": result.duration_ms,
                    })),
                    None => Err(AppError::not_found(format!(
                        "No recorded result for context '{context_id}'"
                    ))),
                }
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sandbox_select_and_build() {
        let store = SandboxContextStore::new();
        let mut registry = RpcRegistry::new();
        register_sandbox_handlers(&mut registry, store);

        let r = registry
            .dispatch(
                "sandbox.select_level",
                serde_json::json!({"operation": "code_execution"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["level"], "STRICT");

        let r = registry
            .dispatch(
                "sandbox.build_policy",
                serde_json::json!({"level": "STANDARD", "context_id": "ctx-1"}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch(
                "sandbox.get_policy",
                serde_json::json!({"context_id": "ctx-1"}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("sandbox.list_contexts", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_sandbox_record_result() {
        let store = SandboxContextStore::new();
        let mut registry = RpcRegistry::new();
        register_sandbox_handlers(&mut registry, store);

        let params = serde_json::json!({
            "context_id": "run-1",
            "exit_code": 0,
            "stdout": "hello",
            "duration_ms": 42,
        });
        let r = registry.dispatch("sandbox.record_result", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch(
                "sandbox.get_result",
                serde_json::json!({"context_id": "run-1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["stdout"], "hello");
        assert_eq!(resp["duration_ms"], 42);
    }
}
