//! Approvals RPC handlers.
//!
//! Provides `rpc_approvals` for managing the sandbox approval queue: submit,
//! approve, reject, list pending, and inspect the rejection ledger.

use opensquilla_core::error::AppError;
use opensquilla_sandbox::governance::{ApprovalQueue, ApprovalRequest, RejectionEntry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A shared approval queue service.
#[derive(Clone)]
pub struct ApprovalsService {
    queue: Arc<ApprovalQueue>,
}

impl ApprovalsService {
    /// Create a new service with a default 5-minute timeout queue.
    pub fn new() -> Self {
        Self {
            queue: Arc::new(ApprovalQueue::new()),
        }
    }

    /// Create a service with a custom timeout.
    pub fn with_timeout(timeout_secs: u64) -> Self {
        Self {
            queue: Arc::new(ApprovalQueue::with_timeout(timeout_secs)),
        }
    }

    /// Access the underlying queue.
    pub fn queue(&self) -> &ApprovalQueue {
        &self.queue
    }
}

impl Default for ApprovalsService {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable view of an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalView {
    pub id: String,
    pub operation: String,
    pub command: String,
    pub args: Vec<String>,
    pub reason: String,
    pub requested_at: String,
    pub expires_at: String,
    pub status: String,
}

impl From<&ApprovalRequest> for ApprovalView {
    fn from(r: &ApprovalRequest) -> Self {
        Self {
            id: r.id.clone(),
            operation: r.operation.clone(),
            command: r.command.clone(),
            args: r.args.clone(),
            reason: r.reason.clone(),
            requested_at: r.requested_at.to_rfc3339(),
            expires_at: r.expires_at.to_rfc3339(),
            status: format!("{:?}", r.status).to_lowercase(),
        }
    }
}

/// Serializable view of a rejection ledger entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionView {
    pub request_id: String,
    pub operation: String,
    pub command: String,
    pub rejected_at: String,
    pub reason: String,
    pub rejected_by: String,
    pub operation_hash: String,
}

impl From<&RejectionEntry> for RejectionView {
    fn from(r: &RejectionEntry) -> Self {
        Self {
            request_id: r.request_id.clone(),
            operation: r.operation.clone(),
            command: r.command.clone(),
            rejected_at: r.rejected_at.to_rfc3339(),
            reason: r.reason.clone(),
            rejected_by: r.rejected_by.clone(),
            operation_hash: r.operation_hash.clone(),
        }
    }
}

/// Register approvals RPC handlers on the given registry.
pub fn register_approvals_handlers(registry: &mut RpcRegistry, service: ApprovalsService) {
    let service = Arc::new(service);

    // approvals.submit — submit a new approval request
    registry.register(rpc_handler("approvals.submit", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let operation = params
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'operation' parameter"))?;
                let command = params
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'command' parameter"))?;
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Pending approval")
                    .to_string();
                let args: Vec<String> = params
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let id = service
                    .queue()
                    .submit(operation, command, &args, &reason)
                    .await
                    .map_err(AppError::bad_request)?;
                Ok(serde_json::json!({
                    "id": id,
                    "operation": operation,
                    "command": command,
                    "status": "pending",
                }))
            }
        }
    }));

    // approvals.approve — approve a pending request
    registry.register(rpc_handler("approvals.approve", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                service
                    .queue()
                    .approve(id)
                    .await
                    .map_err(AppError::bad_request)?;
                Ok(serde_json::json!({"id": id, "status": "approved"}))
            }
        }
    }));

    // approvals.reject — reject a pending request and record in the ledger
    registry.register(rpc_handler("approvals.reject", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'reason' parameter"))?;
                let rejected_by = params
                    .get("rejected_by")
                    .and_then(|v| v.as_str())
                    .unwrap_or("system")
                    .to_string();
                service
                    .queue()
                    .reject(id, reason, &rejected_by)
                    .await
                    .map_err(AppError::bad_request)?;
                Ok(serde_json::json!({"id": id, "status": "rejected"}))
            }
        }
    }));

    // approvals.pending — list all pending requests
    registry.register(rpc_handler("approvals.pending", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let pending: Vec<ApprovalView> = service
                    .queue()
                    .get_pending()
                    .await
                    .iter()
                    .map(ApprovalView::from)
                    .collect();
                Ok(serde_json::json!({
                    "pending": pending,
                    "count": pending.len(),
                }))
            }
        }
    }));

    // approvals.ledger — list the rejection ledger
    registry.register(rpc_handler("approvals.ledger", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let ledger: Vec<RejectionView> = service
                    .queue()
                    .get_rejection_ledger()
                    .await
                    .iter()
                    .map(RejectionView::from)
                    .collect();
                Ok(serde_json::json!({
                    "rejections": ledger,
                    "count": ledger.len(),
                }))
            }
        }
    }));

    // approvals.is_rejected — check if an operation has been previously rejected
    registry.register(rpc_handler("approvals.is_rejected", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let operation = params
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'operation' parameter"))?;
                let command = params
                    .get("command")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'command' parameter"))?;
                let args: Vec<String> = params
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let rejected = service.queue().is_rejected(operation, command, &args).await;
                Ok(serde_json::json!({
                    "operation": operation,
                    "command": command,
                    "is_rejected": rejected,
                }))
            }
        }
    }));

    // approvals.cleanup — expire stale pending requests
    registry.register(rpc_handler("approvals.cleanup", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let expired = service.queue().cleanup_expired().await;
                Ok(serde_json::json!({"expired": expired}))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_approvals_submit_approve() {
        let service = ApprovalsService::new();
        let mut registry = RpcRegistry::new();
        register_approvals_handlers(&mut registry, service);

        let params = serde_json::json!({
            "operation": "shell_command",
            "command": "rm",
            "args": ["/tmp/file"],
            "reason": "Destructive operation",
        });
        let r = registry.dispatch("approvals.submit", params).await;
        let resp = r.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch("approvals.approve", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_approvals_reject_and_guard() {
        let service = ApprovalsService::new();
        let mut registry = RpcRegistry::new();
        register_approvals_handlers(&mut registry, service);

        let params = serde_json::json!({
            "operation": "shell_command",
            "command": "rm",
            "args": ["/etc/passwd"],
        });
        let resp = registry
            .dispatch("approvals.submit", params)
            .await
            .unwrap()
            .unwrap();
        let id = resp["id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch(
                "approvals.reject",
                serde_json::json!({"id": id, "reason": "Dangerous", "rejected_by": "admin"}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        // The guard should now block re-submission.
        let r = registry
            .dispatch(
                "approvals.is_rejected",
                serde_json::json!({"operation": "shell_command", "command": "rm", "args": ["/etc/passwd"]}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["is_rejected"], true);

        // Ledger should have one entry.
        let r = registry
            .dispatch("approvals.ledger", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_approvals_pending() {
        let service = ApprovalsService::new();
        let mut registry = RpcRegistry::new();
        register_approvals_handlers(&mut registry, service);

        let params = serde_json::json!({
            "operation": "file_write",
            "command": "cp",
            "args": ["a", "b"],
        });
        let _ = registry.dispatch("approvals.submit", params).await.unwrap();

        let r = registry
            .dispatch("approvals.pending", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }
}
