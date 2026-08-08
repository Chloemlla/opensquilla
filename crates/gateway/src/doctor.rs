//! Doctor / health-check RPC handlers.
//!
//! Provides `rpc_doctor` for unified health checks across subsystems, backed
//! by the recovery crate's [`HealthCheck`] manager.

use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use opensquilla_recovery::health::{HealthCheck, HealthCheckResult, HealthStatus};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A shared health-check service.
#[derive(Clone)]
pub struct DoctorService {
    check: Arc<HealthCheck>,
    diagnostics_enabled: Arc<Mutex<bool>>,
}

impl DoctorService {
    /// Create a new doctor service from the application config.
    pub fn new(config: &Config) -> Self {
        Self {
            check: Arc::new(HealthCheck::new(config)),
            diagnostics_enabled: Arc::new(Mutex::new(false)),
        }
    }

    /// Whether verbose diagnostics are enabled.
    pub fn diagnostics_enabled(&self) -> bool {
        *self.diagnostics_enabled.lock()
    }

    /// Toggle diagnostics on or off.
    pub fn set_diagnostics(&self, enabled: bool) {
        *self.diagnostics_enabled.lock() = enabled;
    }
}

/// Summarized health view for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub status: String,
    pub uptime_seconds: u64,
    pub timestamp: String,
    pub components: Vec<ComponentView>,
    pub issues: Vec<IssueView>,
    pub diagnostics_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentView {
    pub name: String,
    pub status: String,
    pub description: String,
    pub latency_ms: u64,
    pub details: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueView {
    pub component: String,
    pub severity: String,
    pub message: String,
    pub suggestion: Option<String>,
}

fn status_name(s: HealthStatus) -> &'static str {
    match s {
        HealthStatus::Healthy => "healthy",
        HealthStatus::Degraded => "degraded",
        HealthStatus::Unhealthy => "unhealthy",
    }
}

fn from_result(result: HealthCheckResult, diagnostics: bool) -> DoctorReport {
    DoctorReport {
        status: status_name(result.status).to_string(),
        uptime_seconds: result.uptime_seconds,
        timestamp: result.timestamp.to_rfc3339(),
        components: result
            .components
            .into_iter()
            .map(|c| ComponentView {
                name: c.name,
                status: status_name(c.status).to_string(),
                description: c.description,
                latency_ms: c.latency_ms,
                details: c.details,
            })
            .collect(),
        issues: result
            .issues
            .into_iter()
            .map(|i| IssueView {
                component: i.component,
                severity: format!("{:?}", i.severity).to_lowercase(),
                message: i.message,
                suggestion: i.suggestion,
            })
            .collect(),
        diagnostics_enabled: diagnostics,
    }
}

/// Register doctor RPC handlers on the given registry.
pub fn register_doctor_handlers(registry: &mut RpcRegistry, service: DoctorService) {
    let service = Arc::new(service);

    // doctor.full — run a full health check across all components
    registry.register(rpc_handler("doctor.full", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let result = service.check.run_full_check().await;
                let report = from_result(result, service.diagnostics_enabled());
                serde_json::to_value(report).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // doctor.quick — run a quick config-only check
    registry.register(rpc_handler("doctor.quick", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let result = service.check.quick_check().await;
                let report = from_result(result, service.diagnostics_enabled());
                serde_json::to_value(report).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // doctor.uptime — report current uptime
    registry.register(rpc_handler("doctor.uptime", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let uptime = service.check.uptime();
                Ok(serde_json::json!({
                    "uptime_seconds": uptime.num_seconds().max(0) as u64,
                    "start_time": service.check.start_time().to_rfc3339(),
                }))
            }
        }
    }));

    // doctor.diagnostics — toggle verbose diagnostics
    registry.register(rpc_handler("doctor.diagnostics", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let enabled = params
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                service.set_diagnostics(enabled);
                Ok(serde_json::json!({
                    "diagnostics_enabled": enabled,
                }))
            }
        }
    }));

    // doctor.status — one-shot overall status string (healthy/degraded/unhealthy)
    registry.register(rpc_handler("doctor.status", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let result = service.check.quick_check().await;
                Ok(serde_json::json!({
                    "status": status_name(result.status),
                    "uptime_seconds": result.uptime_seconds,
                }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_doctor_quick_check() {
        let config = Config::default();
        let service = DoctorService::new(&config);
        let mut registry = RpcRegistry::new();
        register_doctor_handlers(&mut registry, service);

        let r = registry
            .dispatch("doctor.quick", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["status"].as_str().is_some());
        assert!(resp["components"].is_array());
    }

    #[tokio::test]
    async fn test_doctor_diagnostics_toggle() {
        let config = Config::default();
        let service = DoctorService::new(&config);
        let mut registry = RpcRegistry::new();
        register_doctor_handlers(&mut registry, service);

        let r = registry
            .dispatch("doctor.diagnostics", serde_json::json!({"enabled": true}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["diagnostics_enabled"], true);
    }

    #[tokio::test]
    async fn test_doctor_uptime() {
        let config = Config::default();
        let service = DoctorService::new(&config);
        let mut registry = RpcRegistry::new();
        register_doctor_handlers(&mut registry, service);

        let r = registry
            .dispatch("doctor.uptime", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["uptime_seconds"].as_u64().is_some());
    }
}
