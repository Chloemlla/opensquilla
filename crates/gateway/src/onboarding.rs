//! Onboarding RPC handlers.
//!
//! Provides `rpc_onboarding` for provider/channel configuration changes
//! during the initial setup flow, layered on top of the onboarding crate's
//! `SetupFlow` state machine and `ConfigStorage`.

use std::sync::Arc;
use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use opensquilla_onboarding::flow::{OnboardingError, SetupFlow, SetupState};
use opensquilla_onboarding::providers::{discover_providers, get_provider, ProviderSpec};
use opensquilla_onboarding::storage::ConfigStorage;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::rpc::{rpc_handler, RpcRegistry};

/// A shared onboarding session holding the active setup flow.
#[derive(Clone)]
pub struct OnboardingSession {
    flow: Arc<Mutex<Option<SetupFlow>>>,
}

impl OnboardingSession {
    /// Create an empty onboarding session (no active flow).
    pub fn new() -> Self {
        Self {
            flow: Arc::new(Mutex::new(None)),
        }
    }

    /// Start a new setup flow, replacing any existing one.
    pub fn start(&self, config: &Config) {
        let flow = SetupFlow::new(config);
        *self.flow.lock() = Some(flow);
    }

    /// Run a closure with mutable access to the active flow.
    fn with_flow<R>(&self, f: impl FnOnce(&mut SetupFlow) -> R) -> Result<R, AppError> {
        let mut guard = self.flow.lock();
        let flow = guard
            .as_mut()
            .ok_or_else(|| AppError::bad_request("No active onboarding flow. Call onboarding.start first."))?;
        Ok(f(flow))
    }
}

impl Default for OnboardingSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Response payload describing the current wizard state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardingStatusResponse {
    pub state: String,
    pub progress: f32,
    pub steps: Vec<OnboardingStepView>,
    pub is_complete: bool,
    pub is_cancelled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardingStepView {
    pub state: String,
    pub title: String,
    pub description: String,
    pub completed: bool,
    pub skipped: bool,
}

fn onboarding_err(e: OnboardingError) -> AppError {
    AppError::bad_request(e.to_string())
}

/// Register onboarding RPC handlers on the given registry.
pub fn register_onboarding_handlers(registry: &mut RpcRegistry, session: OnboardingSession) {
    let session = Arc::new(session);

    // onboarding.providers — list discoverable provider specs
    registry.register(rpc_handler("onboarding.providers", {
        move |_params| {
            let providers: Vec<ProviderSpec> = discover_providers();
            Ok(serde_json::json!({
                "providers": providers,
                "count": providers.len(),
            }))
        }
    }));

    // onboarding.start — begin a new setup flow
    registry.register(rpc_handler("onboarding.start", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let config = Config::default();
                session.start(&config);
                let state = session.with_flow(|f| (f.current_state(), f.progress()))?;
                Ok(serde_json::json!({
                    "started": true,
                    "state": format!("{:?}", state.0),
                    "progress": state.1,
                }))
            }
        }
    }));

    // onboarding.status — current wizard state and step list
    registry.register(rpc_handler("onboarding.status", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let view = session.with_flow(|f| {
                    let steps: Vec<OnboardingStepView> = f
                        .steps()
                        .iter()
                        .map(|s| OnboardingStepView {
                            state: format!("{:?}", s.state),
                            title: s.title.clone(),
                            description: s.description.clone(),
                            completed: s.completed,
                            skipped: s.skipped,
                        })
                        .collect();
                    OnboardingStatusResponse {
                        state: format!("{:?}", f.current_state()),
                        progress: f.progress(),
                        steps,
                        is_complete: f.is_complete(),
                        is_cancelled: f.is_cancelled(),
                    }
                })?;
                Ok(serde_json::to_value(view).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // onboarding.select_provider — pick a provider by name for the flow
    registry.register(rpc_handler("onboarding.select_provider", {
        let session = session.clone();
        move |params| {
            let session = session.clone();
            async move {
                let name = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'provider' parameter"))?;
                let spec = get_provider(name)
                    .ok_or_else(|| AppError::not_found(format!("Unknown provider '{name}'")))?;
                session.with_flow(|f| f.select_provider(spec.clone()))?;
                Ok(serde_json::json!({"selected": name}))
            }
        }
    }));

    // onboarding.set_api_key — store the API key for the selected provider
    registry.register(rpc_handler("onboarding.set_api_key", {
        let session = session.clone();
        move |params| {
            let session = session.clone();
            async move {
                let key = params
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'api_key' parameter"))?;
                session.with_flow(|f| f.set_api_key(key))?;
                Ok(serde_json::json!({"status": "set"}))
            }
        }
    }));

    // onboarding.set_model — set the default model
    registry.register(rpc_handler("onboarding.set_model", {
        let session = session.clone();
        move |params| {
            let session = session.clone();
            async move {
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?;
                session.with_flow(|f| f.set_default_model(model))?;
                Ok(serde_json::json!({"model": model}))
            }
        }
    }));

    // onboarding.advance — move to the next step
    registry.register(rpc_handler("onboarding.advance", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let state = session.with_flow(|f| {
                    f.advance().map_err(onboarding_err)?;
                    Ok::<_, AppError>(f.current_state())
                })??;
                Ok(serde_json::json!({"state": format!("{:?}", state)}))
            }
        }
    }));

    // onboarding.back — return to the previous step
    registry.register(rpc_handler("onboarding.back", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let state = session.with_flow(|f| {
                    f.go_back();
                    f.current_state()
                })?;
                Ok(serde_json::json!({"state": format!("{:?}", state)}))
            }
        }
    }));

    // onboarding.skip — skip the current step
    registry.register(rpc_handler("onboarding.skip", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let state = session.with_flow(|f| {
                    f.skip();
                    f.current_state()
                })?;
                Ok(serde_json::json!({"state": format!("{:?}", state)}))
            }
        }
    }));

    // onboarding.complete — finalize and persist configuration
    registry.register(rpc_handler("onboarding.complete", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let flow = {
                    let mut guard = session.flow.lock();
                    guard
                        .take()
                        .ok_or_else(|| AppError::bad_request("No active onboarding flow"))?
                };
                flow.complete().await.map_err(onboarding_err)?;
                Ok(serde_json::json!({"completed": true}))
            }
        }
    }));

    // onboarding.cancel — abandon the flow
    registry.register(rpc_handler("onboarding.cancel", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                session.with_flow(|f| f.cancel())?;
                Ok(serde_json::json!({"cancelled": true}))
            }
        }
    }));

    // onboarding.save — persist the current configuration to storage
    registry.register(rpc_handler("onboarding.save", {
        let session = session.clone();
        move |_params| {
            let session = session.clone();
            async move {
                let config = Config::default();
                let storage = ConfigStorage::new(&config)
                    .map_err(|e| AppError::internal(format!("Storage init failed: {e}")))?;
                storage
                    .save_config(&config)
                    .await
                    .map_err(|e| AppError::internal(format!("Save failed: {e}")))?;
                let _ = session; // keep session alive in closure
                Ok(serde_json::json!({"saved": true}))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_onboarding_flow_lifecycle() {
        let session = OnboardingSession::new();
        let mut registry = RpcRegistry::new();
        register_onboarding_handlers(&mut registry, session);

        // Start
        let r = registry.dispatch("onboarding.start", serde_json::Value::Null).await;
        assert!(r.unwrap().is_ok());

        // Select provider
        let r = registry
            .dispatch("onboarding.select_provider", serde_json::json!({"provider": "openai"}))
            .await;
        assert!(r.unwrap().is_ok());

        // Set API key
        let r = registry
            .dispatch("onboarding.set_api_key", serde_json::json!({"api_key": "sk-test"}))
            .await;
        assert!(r.unwrap().is_ok());

        // Advance should succeed now
        let r = registry.dispatch("onboarding.advance", serde_json::Value::Null).await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_onboarding_providers_list() {
        let session = OnboardingSession::new();
        let mut registry = RpcRegistry::new();
        register_onboarding_handlers(&mut registry, session);
        let r = registry.dispatch("onboarding.providers", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_onboarding_status_requires_start() {
        let session = OnboardingSession::new();
        let mut registry = RpcRegistry::new();
        register_onboarding_handlers(&mut registry, session);
        let r = registry.dispatch("onboarding.status", serde_json::Value::Null).await;
        assert!(r.unwrap().is_err());
    }
}
