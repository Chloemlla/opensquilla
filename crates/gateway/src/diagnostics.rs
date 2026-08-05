//! Diagnostics RPC handlers.
//!
//! Provides `rpc_diagnostics` for toggling verbose diagnostics across the
//! gateway, including trace-level logging, prompt reporting, and decision
//! logging.

use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// Diagnostics configuration state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsState {
    pub enabled: bool,
    pub verbose: bool,
    pub trace: bool,
    pub prompt_report: bool,
    pub decision_log: bool,
    pub safe_log: bool,
}

impl Default for DiagnosticsState {
    fn default() -> Self {
        Self {
            enabled: false,
            verbose: false,
            trace: false,
            prompt_report: false,
            decision_log: false,
            safe_log: true,
        }
    }
}

/// A shared diagnostics toggle.
#[derive(Clone, Default)]
pub struct DiagnosticsService {
    state: Arc<Mutex<DiagnosticsState>>,
}

impl DiagnosticsService {
    /// Create a new service with diagnostics disabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current state.
    pub fn state(&self) -> DiagnosticsState {
        self.state.lock().clone()
    }

    /// Update the state from a set of optional fields.
    pub fn update(
        &self,
        enabled: Option<bool>,
        verbose: Option<bool>,
        trace: Option<bool>,
        prompt_report: Option<bool>,
        decision_log: Option<bool>,
        safe_log: Option<bool>,
    ) -> DiagnosticsState {
        let mut state = self.state.lock();
        if let Some(e) = enabled {
            state.enabled = e;
        }
        if let Some(v) = verbose {
            state.verbose = v;
        }
        if let Some(t) = trace {
            state.trace = t;
        }
        if let Some(p) = prompt_report {
            state.prompt_report = p;
        }
        if let Some(d) = decision_log {
            state.decision_log = d;
        }
        if let Some(s) = safe_log {
            state.safe_log = s;
        }
        state.clone()
    }
}

/// Register diagnostics RPC handlers on the given registry.
pub fn register_diagnostics_handlers(registry: &mut RpcRegistry, service: DiagnosticsService) {
    let service = Arc::new(service);

    // diagnostics.toggle — enable or disable diagnostics globally
    registry.register(rpc_handler("diagnostics.toggle", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let enabled = params
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let state = service.update(Some(enabled), None, None, None, None, None);
                Ok(serde_json::to_value(state).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // diagnostics.state — read the current diagnostics state
    registry.register(rpc_handler("diagnostics.state", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let state = service.state();
                Ok(serde_json::to_value(state).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // diagnostics.configure — update individual diagnostic flags
    registry.register(rpc_handler("diagnostics.configure", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let enabled = params.get("enabled").and_then(|v| v.as_bool());
                let verbose = params.get("verbose").and_then(|v| v.as_bool());
                let trace = params.get("trace").and_then(|v| v.as_bool());
                let prompt_report = params.get("prompt_report").and_then(|v| v.as_bool());
                let decision_log = params.get("decision_log").and_then(|v| v.as_bool());
                let safe_log = params.get("safe_log").and_then(|v| v.as_bool());

                let state = service.update(
                    enabled,
                    verbose,
                    trace,
                    prompt_report,
                    decision_log,
                    safe_log,
                );
                Ok(serde_json::to_value(state).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // diagnostics.enable_all — turn on all diagnostic flags
    registry.register(rpc_handler("diagnostics.enable_all", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let state = service.update(
                    Some(true),
                    Some(true),
                    Some(true),
                    Some(true),
                    Some(true),
                    Some(true),
                );
                Ok(serde_json::to_value(state).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // diagnostics.disable_all — turn off all diagnostic flags
    registry.register(rpc_handler("diagnostics.disable_all", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let state = service.update(
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(false),
                    Some(false),
                );
                Ok(serde_json::to_value(state).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_diagnostics_toggle() {
        let service = DiagnosticsService::new();
        let mut registry = RpcRegistry::new();
        register_diagnostics_handlers(&mut registry, service);

        let r = registry
            .dispatch("diagnostics.toggle", serde_json::json!({"enabled": true}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["enabled"], true);

        let r = registry
            .dispatch("diagnostics.state", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["enabled"], true);
    }

    #[tokio::test]
    async fn test_diagnostics_enable_all() {
        let service = DiagnosticsService::new();
        let mut registry = RpcRegistry::new();
        register_diagnostics_handlers(&mut registry, service);

        let r = registry
            .dispatch("diagnostics.enable_all", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["verbose"], true);
        assert_eq!(resp["trace"], true);
    }

    #[tokio::test]
    async fn test_diagnostics_configure() {
        let service = DiagnosticsService::new();
        let mut registry = RpcRegistry::new();
        register_diagnostics_handlers(&mut registry, service);

        let r = registry
            .dispatch(
                "diagnostics.configure",
                serde_json::json!({"trace": true, "safe_log": false}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["trace"], true);
        assert_eq!(resp["safe_log"], false);
    }
}
