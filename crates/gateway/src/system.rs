//! System / message RPC handlers.
//!
//! Provides `rpc_system` for system-level RPCs: gateway info, ping/echo, and
//! a message delegation layer that forwards messages to the appropriate
//! subsystem based on type.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// Gateway runtime info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub name: String,
    pub version: String,
    pub started_at: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub pid: u32,
}

/// A system message record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub id: String,
    pub message_type: String,
    pub payload: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// System service holding runtime info and a message log.
#[derive(Clone)]
pub struct SystemService {
    info: Arc<SystemInfo>,
    messages: Arc<Mutex<Vec<SystemMessage>>>,
}

impl SystemService {
    /// Create a new system service.
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            info: Arc::new(SystemInfo {
                name: "opensquilla-gateway".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                started_at: now,
                uptime_seconds: 0,
                pid: std::process::id(),
            }),
            messages: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get the current system info with a refreshed uptime.
    pub fn info(&self) -> SystemInfo {
        let mut info = (*self.info).clone();
        info.uptime_seconds = (Utc::now() - info.started_at).num_seconds().max(0) as u64;
        info
    }

    /// Record a system message.
    pub fn record(&self, message: SystemMessage) {
        self.messages.lock().push(message);
    }

    /// List recorded messages, optionally filtered by type.
    pub fn list_messages(&self, message_type: Option<&str>, limit: usize) -> Vec<SystemMessage> {
        let messages = self.messages.lock();
        let filtered: Vec<SystemMessage> = messages
            .iter()
            .filter(|m| message_type.is_none_or(|t| m.message_type == t))
            .cloned()
            .collect();
        filtered.into_iter().rev().take(limit).collect()
    }
}

impl Default for SystemService {
    fn default() -> Self {
        Self::new()
    }
}

/// Register system RPC handlers on the given registry.
pub fn register_system_handlers(registry: &mut RpcRegistry, service: SystemService) {
    let service = Arc::new(service);

    // system.info — gateway runtime info
    registry.register(rpc_handler("system.info", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let info = service.info();
                serde_json::to_value(info).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // system.ping — liveness probe
    registry.register(rpc_handler("system.ping", {
        move |params| async move {
            let timestamp = Utc::now().to_rfc3339();
            let echoed = params
                .get("echo")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            Ok(serde_json::json!({
                "pong": true,
                "timestamp": timestamp,
                "echo": echoed,
            }))
        }
    }));

    // system.message — record and route a system message
    registry.register(rpc_handler("system.message", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let message_type = params
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'type' parameter"))?;
                let payload = params
                    .get("payload")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let message = SystemMessage {
                    id: Uuid::new_v4().to_string(),
                    message_type: message_type.to_string(),
                    payload,
                    timestamp: Utc::now(),
                };
                service.record(message.clone());
                serde_json::to_value(message).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // system.messages — list recorded system messages
    registry.register(rpc_handler("system.messages", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let message_type = params.get("type").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
                let messages = service.list_messages(message_type, limit);
                Ok(serde_json::json!({
                    "messages": messages,
                    "count": messages.len(),
                }))
            }
        }
    }));

    // system.version — version string only
    registry.register(rpc_handler("system.version", {
        move |_params| async move {
            Ok(serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
            }))
        }
    }));

    // system.uptime — uptime in seconds
    registry.register(rpc_handler("system.uptime", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let info = service.info();
                Ok(serde_json::json!({
                    "uptime_seconds": info.uptime_seconds,
                    "started_at": info.started_at.to_rfc3339(),
                    "pid": info.pid,
                }))
            }
        }
    }));

    // system.echo — pure echo for diagnostics
    registry.register(rpc_handler("system.echo", {
        move |params| async move {
            Ok(serde_json::json!({
                "received": params,
            }))
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_ping() {
        let service = SystemService::new();
        let mut registry = RpcRegistry::new();
        register_system_handlers(&mut registry, service);

        let r = registry
            .dispatch("system.ping", serde_json::json!({"echo": "hello"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["pong"], true);
        assert_eq!(resp["echo"], "hello");
    }

    #[tokio::test]
    async fn test_system_info_and_version() {
        let service = SystemService::new();
        let mut registry = RpcRegistry::new();
        register_system_handlers(&mut registry, service);

        let r = registry
            .dispatch("system.info", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["name"], "opensquilla-gateway");

        let r = registry
            .dispatch("system.version", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["version"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_system_message_record_and_list() {
        let service = SystemService::new();
        let mut registry = RpcRegistry::new();
        register_system_handlers(&mut registry, service);

        let params = serde_json::json!({
            "type": "notification",
            "payload": {"text": "hello"},
        });
        let r = registry.dispatch("system.message", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("system.messages", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_system_echo() {
        let service = SystemService::new();
        let mut registry = RpcRegistry::new();
        register_system_handlers(&mut registry, service);

        let r = registry
            .dispatch("system.echo", serde_json::json!({"any": "thing"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["received"]["any"], "thing");
    }
}
