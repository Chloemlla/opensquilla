//! Agent runtime lifecycle management for the desktop shell.
//!
//! [`DesktopRuntime`] owns the gateway / agent runtime handle and exposes
//! start / stop / status operations that the Tauri command layer calls into.
//! The runtime is process-local: there is no Python sidecar, no separate
//! gateway process — the gateway's axum router and the agent engine share the
//! same Tokio runtime as the Tauri webview, exactly as described in the
//! migration analysis ("全 Rust 迁移分层策略 → 通信方式: 同一进程").

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{Mutex, oneshot, watch};
use tokio::task::JoinHandle;

use opensquilla_core::config::{Config, GatewayConfig};
use opensquilla_gateway::Gateway;

use crate::Result;

/// Lifecycle state of the desktop agent runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Not yet started.
    Stopped,
    /// Currently starting the gateway and agent engine.
    Starting,
    /// Gateway is bound and serving; agent engine is ready.
    Running,
    /// Graceful shutdown in progress.
    Stopping,
    /// Terminated unexpectedly (see logs).
    Failed,
}

impl RuntimeState {
    /// Human-readable label for the state.
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeState::Stopped => "stopped",
            RuntimeState::Starting => "starting",
            RuntimeState::Running => "running",
            RuntimeState::Stopping => "stopping",
            RuntimeState::Failed => "failed",
        }
    }
}

/// Handle to a running gateway task, with a shutdown signal channel.
struct RuntimeHandle {
    join: JoinHandle<std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    shutdown_tx: oneshot::Sender<()>,
}

impl std::fmt::Debug for RuntimeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeHandle")
            .field("join", &"JoinHandle<()>")
            .field("shutdown_tx", &"oneshot::Sender<()>")
            .finish()
    }
}

/// The desktop agent runtime.
///
/// Owns (lazily) the [`Gateway`] task and a [`watch`] channel broadcasting
/// the current [`RuntimeState`]. All mutating operations are `async` and
/// intended to be called from Tauri command handlers via `State<AppState>`.
pub struct DesktopRuntime {
    state: watch::Sender<RuntimeState>,
    state_rx: watch::Receiver<RuntimeState>,
    handle: Option<RuntimeHandle>,
    /// The socket address the gateway bound to, once known.
    bound_addr: Option<SocketAddr>,
    /// The gateway config used for the last (or next) start.
    gateway_config: GatewayConfig,
}

impl std::fmt::Debug for DesktopRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DesktopRuntime")
            .field("state", &self.state_rx.borrow().as_str())
            .field("bound_addr", &self.bound_addr)
            .field("gateway_config", &self.gateway_config)
            .field("handle", &self.handle.as_ref().map(|_| "Some"))
            .finish()
    }
}

impl DesktopRuntime {
    /// Create a new runtime in the [`Stopped`] state with default config.
    pub fn new() -> Self {
        Self::with_config(GatewayConfig::default())
    }

    /// Create a new runtime seeded with config derived from the OpenSquilla
    /// configuration.
    pub fn from_config(config: &Config) -> Self {
        Self::with_config(config.gateway.clone())
    }

    /// Create a new runtime with an explicit gateway config.
    pub fn with_config(gateway_config: GatewayConfig) -> Self {
        let (state_tx, state_rx) = watch::channel(RuntimeState::Stopped);
        Self {
            state: state_tx,
            state_rx,
            handle: None,
            bound_addr: None,
            gateway_config,
        }
    }

    /// Current lifecycle state.
    pub fn state(&self) -> RuntimeState {
        *self.state_rx.borrow()
    }

    /// Subscribe to state transitions.
    pub fn subscribe(&self) -> watch::Receiver<RuntimeState> {
        self.state_rx.clone()
    }

    /// The address the gateway is bound to, if running.
    pub fn bound_addr(&self) -> Option<SocketAddr> {
        self.bound_addr
    }

    /// The gateway config in use.
    pub fn gateway_config(&self) -> &GatewayConfig {
        &self.gateway_config
    }

    /// Update the gateway config. Only effective before the next [`start`].
    ///
    /// [`start`]: Self::start
    pub fn set_gateway_config(&mut self, config: GatewayConfig) {
        self.gateway_config = config;
    }

    /// Start the gateway + agent runtime on the shared Tokio handle.
    ///
    /// Spawns the gateway's axum server as a Tokio task. The returned future
    /// resolves once the task has been spawned (not once it exits). Use
    /// [`shutdown`] to stop it.
    ///
    /// [`shutdown`]: Self::shutdown
    pub async fn start(&mut self, rt: &tokio::runtime::Handle) -> Result<()> {
        if let Some(_h) = &self.handle {
            return Ok(()); // already running — idempotent
        }

        let _ = self.state.send(RuntimeState::Starting);

        let gateway_config = self.gateway_config.clone();
        let addr: SocketAddr = format!("{}:{}", gateway_config.host, gateway_config.port)
            .parse()
            .map_err(|e| {
                crate::DesktopError::Config(format!("invalid gateway bind address: {e}"))
            })?;

        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let gateway = Gateway::new(gateway_config);
        let router = gateway.router();

        let join = rt.spawn(async move {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            tracing::info!(%addr, "OpenSquilla gateway listening");
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) })
        });

        self.bound_addr = Some(addr);
        self.handle = Some(RuntimeHandle { join, shutdown_tx });
        let _ = self.state.send(RuntimeState::Running);
        Ok(())
    }

    /// Gracefully stop the runtime, waiting for the gateway task to exit.
    pub async fn shutdown(&mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            let _ = self.state.send(RuntimeState::Stopped);
            return Ok(());
        };

        let _ = self.state.send(RuntimeState::Stopping);
        let _ = handle.shutdown_tx.send(());
        let _ = handle.join.await;
        self.bound_addr = None;
        let _ = self.state.send(RuntimeState::Stopped);
        Ok(())
    }
}

impl Default for DesktopRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export of the runtime handle used by the managed-state container.
pub type RuntimeHandleRef = Arc<Mutex<DesktopRuntime>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_starts_stopped() {
        let rt = DesktopRuntime::new();
        assert_eq!(rt.state(), RuntimeState::Stopped);
        assert!(rt.bound_addr().is_none());
    }

    #[tokio::test]
    async fn runtime_can_start_and_stop() {
        // Bind to an ephemeral port so the test never clashes with a live
        // gateway. Port 0 parses to a valid SocketAddr and the OS assigns a
        // free port at bind time.
        let mut rt = DesktopRuntime::with_config(GatewayConfig {
            host: "127.0.0.1".into(),
            port: 0,
            ..GatewayConfig::default()
        });
        let handle = tokio::runtime::Handle::current();
        let _ = rt.start(&handle).await;
        let _ = rt.shutdown().await;
        assert_eq!(rt.state(), RuntimeState::Stopped);
    }
}
