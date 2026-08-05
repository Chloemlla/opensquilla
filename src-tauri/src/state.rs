//! App state management.
//!
//! Holds shared application state managed by Tauri's `Manager` trait. The
//! `AppState` struct is inserted into the Tauri app via `app.manage()` and
//! accessed in command handlers via `State<AppState>`.

use crate::error::TauriError;
use crate::workbench::WorkbenchManager;
use opensquilla_core::config::Config;
use opensquilla_engine::AgentRuntime;
use opensquilla_gateway::Gateway;
use opensquilla_session::SessionStorage;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// The central application state shared across all Tauri command handlers.
///
/// Each field is wrapped in `Arc` so the struct can be cheaply cloned (Tauri
///'s `State` returns a reference, but we also need to move handles into
/// spawned tasks). Interior mutability is provided by `RwLock` for read-heavy
/// state (config, runtime) and `Mutex` for write-serialized state (session
/// storage).
#[derive(Clone)]
pub struct AppState {
    /// The agent runtime that manages turn execution and agent handles.
    pub runtime: Arc<AgentRuntime>,
    /// The in-process gateway server. `None` when the gateway has not been
    /// started yet or has been stopped.
    pub gateway: Arc<RwLock<Option<Arc<Gateway>>>>,
    /// The application configuration, loaded from disk at startup.
    pub config: Arc<RwLock<Config>>,
    /// The session storage backend (SQLite). Wrapped in a Mutex because
    /// rusqlite's `Connection` is not `Send + Sync` by default.
    pub session_storage: Arc<Mutex<SessionStorage>>,
    /// The gateway's in-memory session store used for the WebSocket RPC API.
    pub session_store: Arc<opensquilla_gateway::SessionStore>,
    /// The gateway's in-memory chat store.
    pub chat_store: Arc<opensquilla_gateway::ChatStore>,
    /// The gateway's in-memory config store.
    pub config_store: Arc<opensquilla_gateway::ConfigStore>,
    /// The native workbench surface manager.
    pub workbench: Arc<Mutex<WorkbenchManager>>,
    /// The dynamically-assigned gateway URL, set when `start_gateway` succeeds.
    pub gateway_url: Arc<RwLock<Option<String>>>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("runtime", &"AgentRuntime")
            .field("gateway", &self.gateway.is_read_locked())
            .field("config", &"Config")
            .field("session_storage", &"SessionStorage")
            .field("workbench", &"WorkbenchManager")
            .field("gateway_url", &self.gateway_url)
            .finish()
    }
}

impl AppState {
    /// Create a new `AppState` with the given runtime, config, and session
    /// storage. Gateway starts as `None` and must be started via the
    /// `start_gateway` command.
    pub fn new(
        runtime: Arc<AgentRuntime>,
        config: Config,
        session_storage: SessionStorage,
    ) -> Self {
        Self {
            runtime,
            gateway: Arc::new(RwLock::new(None)),
            config: Arc::new(RwLock::new(config)),
            session_storage: Arc::new(Mutex::new(session_storage)),
            session_store: Arc::new(opensquilla_gateway::SessionStore::new()),
            chat_store: Arc::new(opensquilla_gateway::ChatStore::new()),
            config_store: Arc::new(opensquilla_gateway::ConfigStore::new()),
            workbench: Arc::new(Mutex::new(WorkbenchManager::new())),
            gateway_url: Arc::new(RwLock::new(None)),
        }
    }

    /// Get a clone of the runtime.
    pub fn runtime(&self) -> Arc<AgentRuntime> {
        self.runtime.clone()
    }

    /// Get a read guard on the config.
    pub async fn config(&self) -> tokio::sync::RwLockReadGuard<'_, Config> {
        self.config.read().await
    }

    /// Get a write guard on the config.
    pub async fn config_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, Config> {
        self.config.write().await
    }

    /// Get the session storage lock guard.
    pub async fn session_storage(&self) -> tokio::sync::MutexGuard<'_, SessionStorage> {
        self.session_storage.lock().await
    }

    /// Check whether the gateway is currently running.
    pub async fn is_gateway_running(&self) -> bool {
        self.gateway.read().await.is_some()
    }

    /// Get the gateway URL if the gateway is running.
    pub async fn gateway_url(&self) -> Option<String> {
        self.gateway_url.read().await.clone()
    }

    /// Set the gateway URL (called by `start_gateway`).
    pub async fn set_gateway_url(&self, url: Option<String>) {
        let mut guard = self.gateway_url.write().await;
        *guard = url;
    }

    /// Set the gateway instance (called by `start_gateway`).
    pub async fn set_gateway(&self, gateway: Option<Arc<Gateway>>) {
        let mut guard = self.gateway.write().await;
        *guard = gateway;
    }

    /// Get a clone of the gateway if it is running.
    pub async fn get_gateway(&self) -> Option<Arc<Gateway>> {
        self.gateway.read().await.clone()
    }

    /// Get the workbench manager lock guard.
    pub async fn workbench(&self) -> tokio::sync::MutexGuard<'_, WorkbenchManager> {
        self.workbench.lock().await
    }

    /// Ensure the runtime is running, returning an error if it is not.
    pub async fn ensure_runtime_running(&self) -> Result<(), TauriError> {
        if !self.runtime.is_running().await {
            return Err(TauriError::unavailable(
                "Agent runtime is not running. Call start_gateway first.",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_engine::TurnRunnerBuilder;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_app_state_creation() {
        let (event_tx, _event_rx) = mpsc::channel::<opensquilla_core::events::TurnEvent>(128);
        let runner = TurnRunnerBuilder::new().build();
        let runtime = Arc::new(AgentRuntime::new(runner, event_tx));
        let config = Config::default();
        let storage = SessionStorage::in_memory().unwrap();
        let state = AppState::new(runtime, config, storage);

        assert!(!state.is_gateway_running().await);
        assert!(state.gateway_url().await.is_none());
    }
}
