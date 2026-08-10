//! Onboarding wizard RPC handlers.
//!
//! Provides `rpc_wizard` — a dedicated state-machine handler for the
//! onboarding wizard, distinct from the broader `rpc_onboarding`. This
//! module exposes the wizard as a stateless RPC over a shared wizard store,
//! allowing the frontend to drive the wizard step by step.

use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use opensquilla_onboarding::flow::SetupState;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// The wizard state machine view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardState {
    pub current_state: String,
    pub step_index: usize,
    pub total_steps: usize,
    pub progress: f32,
    pub completed_steps: Vec<String>,
    pub skipped_steps: Vec<String>,
}

/// A wizard session holding the active state and collected inputs.
#[derive(Debug, Clone, Default)]
pub struct WizardSession {
    current_state: SetupState,
    completed: Vec<SetupState>,
    skipped: Vec<SetupState>,
    inputs: HashMap<String, serde_json::Value>,
}

impl WizardSession {
    /// Create a new session starting at the welcome state.
    pub fn new() -> Self {
        Self {
            current_state: SetupState::Welcome,
            completed: Vec::new(),
            skipped: Vec::new(),
            inputs: HashMap::new(),
        }
    }

    /// The ordered list of wizard states.
    fn ordered_states() -> Vec<SetupState> {
        vec![
            SetupState::Welcome,
            SetupState::ProviderSelection,
            SetupState::ProviderAuth,
            SetupState::ModelSelection,
            SetupState::ChannelSetup,
            SetupState::SandboxConfig,
            SetupState::Complete,
        ]
    }

    /// Advance to the next state.
    fn advance(&mut self) -> Result<(), AppError> {
        let states = Self::ordered_states();
        let idx = states
            .iter()
            .position(|s| *s == self.current_state)
            .ok_or_else(|| AppError::internal("Invalid wizard state"))?;
        if idx + 1 >= states.len() {
            return Err(AppError::bad_request(
                "Wizard is already at the final state",
            ));
        }
        self.completed.push(self.current_state);
        self.current_state = states[idx + 1];
        Ok(())
    }

    /// Go back to the previous state.
    fn back(&mut self) -> Result<(), AppError> {
        let states = Self::ordered_states();
        let idx = states
            .iter()
            .position(|s| *s == self.current_state)
            .ok_or_else(|| AppError::internal("Invalid wizard state"))?;
        if idx == 0 {
            return Err(AppError::bad_request("Already at the first state"));
        }
        self.current_state = states[idx - 1];
        Ok(())
    }

    /// Skip the current state.
    fn skip(&mut self) -> Result<(), AppError> {
        self.skipped.push(self.current_state);
        self.advance()
    }

    /// Store an input value for a key.
    fn set_input(&mut self, key: &str, value: serde_json::Value) {
        self.inputs.insert(key.to_string(), value);
    }

    /// The current progress as a fraction 0.0–1.0.
    fn progress(&self) -> f32 {
        let states = Self::ordered_states();
        let idx = states
            .iter()
            .position(|s| *s == self.current_state)
            .unwrap_or(0);
        (idx + 1) as f32 / states.len() as f32
    }

    /// Build a serializable view.
    fn to_view(&self) -> WizardState {
        let states = Self::ordered_states();
        let idx = states
            .iter()
            .position(|s| *s == self.current_state)
            .unwrap_or(0);
        WizardState {
            current_state: format!("{:?}", self.current_state),
            step_index: idx,
            total_steps: states.len(),
            progress: self.progress(),
            completed_steps: self.completed.iter().map(|s| format!("{s:?}")).collect(),
            skipped_steps: self.skipped.iter().map(|s| format!("{s:?}")).collect(),
        }
    }
}

/// A shared wizard store keyed by session id.
#[derive(Clone, Default)]
pub struct WizardStore {
    sessions: Arc<Mutex<HashMap<String, WizardSession>>>,
}

impl WizardStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a new wizard session, returning the session id.
    pub fn start(&self) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        self.sessions
            .lock()
            .insert(id.clone(), WizardSession::new());
        id
    }

    /// Get a session by id.
    pub fn get(&self, id: &str) -> Option<WizardSession> {
        self.sessions.lock().get(id).cloned()
    }

    /// Run a closure with mutable access to a session.
    fn with_session<R>(
        &self,
        id: &str,
        f: impl FnOnce(&mut WizardSession) -> R,
    ) -> Result<R, AppError> {
        let mut sessions = self.sessions.lock();
        let session = sessions
            .get_mut(id)
            .ok_or_else(|| AppError::not_found(format!("Wizard session '{id}' not found")))?;
        Ok(f(session))
    }

    /// Remove a session.
    pub fn remove(&self, id: &str) -> bool {
        self.sessions.lock().remove(id).is_some()
    }
}

/// Register wizard RPC handlers on the given registry.
pub fn register_wizard_handlers(registry: &mut RpcRegistry, store: WizardStore) {
    let store = Arc::new(store);

    // wizard.start — start a new wizard session
    registry.register(rpc_handler("wizard.start", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let id = store.start();
                let view = store.get(&id).unwrap().to_view();
                Ok(serde_json::json!({
                    "session_id": id,
                    "state": view,
                }))
            }
        }
    }));

    // wizard.state — get the current state of a session
    registry.register(rpc_handler("wizard.state", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                match store.get(session_id) {
                    Some(session) => Ok(serde_json::json!({
                        "session_id": session_id,
                        "state": session.to_view(),
                    })),
                    None => Err(AppError::not_found(format!(
                        "Wizard session '{session_id}' not found"
                    ))),
                }
            }
        }
    }));

    // wizard.advance — move to the next step
    registry.register(rpc_handler("wizard.advance", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let view =
                    store.with_session(session_id, |s| -> Result<WizardState, AppError> {
                        s.advance()?;
                        Ok(s.to_view())
                    })??;
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "state": view,
                }))
            }
        }
    }));

    // wizard.back — go back to the previous step
    registry.register(rpc_handler("wizard.back", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let view =
                    store.with_session(session_id, |s| -> Result<WizardState, AppError> {
                        s.back()?;
                        Ok(s.to_view())
                    })??;
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "state": view,
                }))
            }
        }
    }));

    // wizard.skip — skip the current step
    registry.register(rpc_handler("wizard.skip", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let view =
                    store.with_session(session_id, |s| -> Result<WizardState, AppError> {
                        s.skip()?;
                        Ok(s.to_view())
                    })??;
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "state": view,
                }))
            }
        }
    }));

    // wizard.input — store a wizard input value
    registry.register(rpc_handler("wizard.input", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                let value = params
                    .get("value")
                    .ok_or_else(|| AppError::bad_request("Missing 'value' parameter"))?
                    .clone();
                store.with_session(session_id, |s| s.set_input(key, value))?;
                Ok(serde_json::json!({"session_id": session_id, "key": key, "stored": true}))
            }
        }
    }));

    // wizard.complete — finalize the wizard and persist config
    registry.register(rpc_handler("wizard.complete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let view =
                    store.with_session(session_id, |s| -> Result<WizardState, AppError> {
                        // Advance to Complete if not already there.
                        while s.current_state != SetupState::Complete {
                            s.advance()?;
                        }
                        Ok(s.to_view())
                    })??;

                // Persist a default config to demonstrate the write path.
                let config = Config::default();
                if let Ok(storage) = opensquilla_onboarding::storage::ConfigStorage::new(&config) {
                    let _ = storage.save_config(&config).await;
                }

                store.remove(session_id);
                Ok(serde_json::json!({
                    "session_id": session_id,
                    "completed": true,
                    "state": view,
                }))
            }
        }
    }));

    // wizard.cancel — abandon a wizard session
    registry.register(rpc_handler("wizard.cancel", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                if store.remove(session_id) {
                    Ok(serde_json::json!({"cancelled": true, "session_id": session_id}))
                } else {
                    Err(AppError::not_found(format!(
                        "Wizard session '{session_id}' not found"
                    )))
                }
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wizard_full_flow() {
        let store = WizardStore::new();
        let mut registry = RpcRegistry::new();
        register_wizard_handlers(&mut registry, store);

        let r = registry
            .dispatch("wizard.start", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        let session_id = resp["session_id"].as_str().unwrap().to_string();
        assert_eq!(resp["state"]["current_state"], "Welcome");

        // Advance through several steps
        for _ in 0..6 {
            let r = registry
                .dispatch(
                    "wizard.advance",
                    serde_json::json!({"session_id": session_id}),
                )
                .await;
            assert!(r.unwrap().is_ok());
        }

        let r = registry
            .dispatch(
                "wizard.state",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["state"]["current_state"], "Complete");
    }

    #[tokio::test]
    async fn test_wizard_input_and_back() {
        let store = WizardStore::new();
        let mut registry = RpcRegistry::new();
        register_wizard_handlers(&mut registry, store);

        let resp = registry
            .dispatch("wizard.start", serde_json::Value::Null)
            .await
            .unwrap()
            .unwrap();
        let session_id = resp["session_id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch(
                "wizard.input",
                serde_json::json!({"session_id": session_id, "key": "provider", "value": "openai"}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch(
                "wizard.advance",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("wizard.back", serde_json::json!({"session_id": session_id}))
            .await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_wizard_cancel() {
        let store = WizardStore::new();
        let mut registry = RpcRegistry::new();
        register_wizard_handlers(&mut registry, store);

        let resp = registry
            .dispatch("wizard.start", serde_json::Value::Null)
            .await
            .unwrap()
            .unwrap();
        let session_id = resp["session_id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch(
                "wizard.cancel",
                serde_json::json!({"session_id": session_id}),
            )
            .await;
        assert!(r.unwrap().is_ok());
    }
}
