//! In-process gateway management.
//!
//! Instead of spawning a Python subprocess (as the Electron app did), the
//! Tauri app starts the `opensquilla-gateway` axum server inside the same
//! process using `tokio::spawn`. The server binds to `127.0.0.1` with a
//! dynamically assigned port (port 0 lets the OS choose), so there is no
//! conflict with other instances or services.
//!
//! The frontend can communicate via either:
//! 1. `invoke()` — direct Tauri command calls (preferred, lower latency)
//! 2. HTTP/WS — connecting to the in-process gateway's `/ws` endpoint
//!
//! This dual-path preserves backward compatibility with existing Vue components
//! that use WebSocket, while new code can use the faster `invoke()` path.

use crate::error::{TauriError, TauriResult};
use crate::ipc::GatewayStatusResponse;
use crate::state::AppState;
use opensquilla_core::config::GatewayConfig;
use opensquilla_gateway::{AuthPrincipal, RpcContext};
use opensquilla_gateway::Gateway;
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tracing::{error, info, warn};

/// The event name emitted when the gateway status changes.
pub const GATEWAY_STATUS_EVENT: &str = "gateway:status";

/// The event payload emitted when the gateway status changes.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayStatusEvent {
    pub running: bool,
    pub url: Option<String>,
    pub port: Option<u16>,
    pub error: Option<String>,
}

/// Start the in-process gateway server.
///
/// Binds to `127.0.0.1:0` (ephemeral port) and spawns the axum server as a
/// background tokio task. The actual bound address is stored in `AppState` so
/// the frontend can connect to it.
#[tauri::command]
pub async fn start_gateway(
    app: AppHandle,
    state: State<'_, AppState>,
) -> TauriResult<GatewayStatusResponse> {
    start_gateway_inner(&app, &state).await
}

/// Internal gateway start logic, callable from both Tauri commands and
/// non-command contexts (e.g. `main.rs` setup).
pub async fn start_gateway_inner(
    app: &AppHandle,
    state: &AppState,
) -> TauriResult<GatewayStatusResponse> {
    // If the gateway is already running, return its current status.
    if state.is_gateway_running().await {
        let url = state.gateway_url().await;
        let port = url
            .as_ref()
            .and_then(|u| u.rsplit(':').next().and_then(|p| p.parse::<u16>().ok()));
        return Ok(GatewayStatusResponse {
            running: true,
            url,
            port,
            error: None,
        });
    }

    info!("Starting in-process gateway on 127.0.0.1:0");

    // Build the gateway config bound to loopback with an ephemeral port.
    let gateway_config = GatewayConfig {
        host: "127.0.0.1".to_string(),
        port: 0, // ephemeral
        max_connections: 100,
        request_timeout_secs: 120,
        cors_origins: vec![
            "http://localhost:5173".to_string(),
            "http://127.0.0.1:5173".to_string(),
            "tauri://localhost".to_string(),
            "http://tauri.localhost".to_string(),
        ],
    };

    // Create the gateway instance. Gateway::new registers the default RPC
    // handlers (sessions, chat, config).
    let _gateway = Gateway::new(gateway_config);

    // Build the router and bind a TcpListener to get the actual address.
    // We need the actual port before spawning the serve task so we can store it.
    let bind_addr: SocketAddr = "127.0.0.1:0"
        .parse()
        .map_err(|e| TauriError::internal(format!("Invalid bind address: {e}")))?;

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|e| TauriError::internal(format!("Failed to bind gateway: {e}")))?;

    let actual_addr = listener
        .local_addr()
        .map_err(|e| TauriError::internal(format!("Failed to get bound address: {e}")))?;

    let gateway_url = format!("http://{}", actual_addr);
    let port = actual_addr.port();

    info!(url = %gateway_url, port = port, "Gateway bound to ephemeral port");

    // Build the router from the gateway. The router consumes the gateway, but
    // we keep a separate `Arc<Gateway>` for the state so commands can still
    // access its stores. The router owns the RPC registry internally.
    let gateway_arc = Arc::new(Gateway::new(GatewayConfig {
        host: "127.0.0.1".to_string(),
        port,
        max_connections: 100,
        request_timeout_secs: 120,
        cors_origins: vec![
            "http://localhost:5173".to_string(),
            "http://127.0.0.1:5173".to_string(),
            "tauri://localhost".to_string(),
            "http://tauri.localhost".to_string(),
        ],
    }));

    // Store the gateway and URL in state.
    state.set_gateway(Some(gateway_arc.clone())).await;
    state.set_gateway_url(Some(gateway_url.clone())).await;

    // Wire the `control_ui.default_locale` preference into channel system
    // messages so channels follow the UI language.
    let channel_locale = state.config().await.default_locale().to_string();
    gateway_arc.set_channel_locale(&channel_locale);

    // Now build the actual router that will be served. We need to create
    // another Gateway for the router since router() consumes it.
    let serve_gateway = Gateway::new(GatewayConfig {
        host: "127.0.0.1".to_string(),
        port,
        max_connections: 100,
        request_timeout_secs: 120,
        cors_origins: vec![
            "http://localhost:5173".to_string(),
            "http://127.0.0.1:5173".to_string(),
            "tauri://localhost".to_string(),
            "http://tauri.localhost".to_string(),
        ],
    });
    serve_gateway.set_channel_locale(&channel_locale);
    let router = serve_gateway.router();

    // Spawn the serve task. We use axum::serve with the already-bound listener.
    let app_handle = app.clone();
    tokio::spawn(async move {
        info!("Gateway serve task started on {}", actual_addr);
        if let Err(e) = axum::serve(listener, router).await {
            error!(error = %e, "Gateway server error");
            // Emit a status event indicating the gateway stopped with an error.
            let _ = app_handle.emit(
                GATEWAY_STATUS_EVENT,
                GatewayStatusEvent {
                    running: false,
                    url: None,
                    port: None,
                    error: Some(e.to_string()),
                },
            );
        }
        info!("Gateway serve task ended");
    });

    // Emit a status event to the frontend.
    let _ = app.emit(
        GATEWAY_STATUS_EVENT,
        GatewayStatusEvent {
            running: true,
            url: Some(gateway_url.clone()),
            port: Some(port),
            error: None,
        },
    );

    Ok(GatewayStatusResponse {
        running: true,
        url: Some(gateway_url),
        port: Some(port),
        error: None,
    })
}

/// Stop the in-process gateway server.
///
/// This does not forcefully kill the axum task (axum's `serve` does not expose
/// a graceful shutdown handle in the current version without a shutdown signal).
/// Instead, it clears the gateway from state so command handlers stop using it.
/// The background task will continue running until the app process exits.
///
/// For a true graceful shutdown, we would need to use `axum::serve(...).with_graceful_shutdown(signal)`
/// and signal it here. This is a known limitation documented for future work.
#[tauri::command]
pub async fn stop_gateway(
    app: AppHandle,
    state: State<'_, AppState>,
) -> TauriResult<GatewayStatusResponse> {
    stop_gateway_inner(&app, state.inner()).await
}

/// Internal stop logic.
pub async fn stop_gateway_inner(
    app: &AppHandle,
    state: &AppState,
) -> TauriResult<GatewayStatusResponse> {
    if !state.is_gateway_running().await {
        warn!("stop_gateway called but gateway is not running");
        return Ok(GatewayStatusResponse {
            running: false,
            url: None,
            port: None,
            error: None,
        });
    }

    info!("Stopping in-process gateway");

    // Clear the gateway from state. The background serve task will be dropped
    // when the app process exits. We cannot abort it here because we don't
    // hold the JoinHandle (it's owned by the spawned task).
    state.set_gateway(None).await;
    state.set_gateway_url(None).await;

    // Emit a status event to the frontend.
    let _ = app.emit(
        GATEWAY_STATUS_EVENT,
        GatewayStatusEvent {
            running: false,
            url: None,
            port: None,
            error: None,
        },
    );

    Ok(GatewayStatusResponse {
        running: false,
        url: None,
        port: None,
        error: None,
    })
}

/// Query the current gateway status.
#[tauri::command]
pub async fn gateway_status(state: State<'_, AppState>) -> TauriResult<GatewayStatusResponse> {
    let running = state.is_gateway_running().await;
    let url = state.gateway_url().await;
    let port = url
        .as_ref()
        .and_then(|u| u.rsplit(':').next().and_then(|p| p.parse::<u16>().ok()));

    Ok(GatewayStatusResponse {
        running,
        url,
        port,
        error: None,
    })
}

/// Restart the gateway (stop then start).
#[tauri::command]
pub async fn restart_gateway(
    app: AppHandle,
    state: State<'_, AppState>,
) -> TauriResult<GatewayStatusResponse> {
    stop_gateway_inner(&app, state.inner()).await?;
    start_gateway_inner(&app, state.inner()).await
}

/// Get the gateway URL if the gateway is running.
///
/// This is a convenience command for the frontend to discover the gateway's
/// WebSocket endpoint for direct WS connections.
#[tauri::command]
pub async fn get_gateway_url(state: State<'_, AppState>) -> TauriResult<Option<String>> {
    Ok(state.gateway_url().await)
}

/// Generic RPC dispatch — routes a method name + params through the gateway's
/// RPC registry. The gateway must be running.
#[tauri::command]
pub async fn rpc_dispatch(
    state: State<'_, AppState>,
    method: String,
    params: Value,
) -> TauriResult<Value> {
    let gateway = state
        .get_gateway()
        .await
        .ok_or_else(|| TauriError::unavailable("Gateway is not running"))?;
    let ctx = RpcContext::new(
        "tauri",
        AuthPrincipal::new("operator", &["admin", "read", "write", "sessions", "config", "channels", "tools", "secrets", "sandbox"], true, false),
        &method,
        "",
    );
    match gateway
        .rpc_registry
        .dispatch_with_ctx(&method, params, &ctx)
        .await
    {
        Some(Ok(payload)) => Ok(payload),
        Some(Err(e)) => Err(TauriError::from(e)),
        None => Err(TauriError::not_found(format!(
            "No RPC handler for '{method}'"
        ))),
    }
}

/// Desktop status — returns `{ uptime_ms, version, provider }` without
/// requiring the gateway to be running.
#[tauri::command]
pub async fn get_status(state: State<'_, AppState>) -> TauriResult<Value> {
    let elapsed = state.started_at.elapsed();
    let uptime_ms = elapsed.as_millis() as u64;
    let config = state.config().await;
    let provider = config
        .providers
        .first()
        .map(|p| p.name.clone())
        .unwrap_or_default();
    Ok(serde_json::json!({
        "uptime_ms": uptime_ms,
        "version": env!("CARGO_PKG_VERSION"),
        "provider": provider,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_engine::{AgentRuntime, TurnRunnerBuilder};
    use opensquilla_session::SessionStorage;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_gateway_status_initial() {
        let (event_tx, _event_rx) = mpsc::channel::<opensquilla_core::events::TurnEvent>(128);
        let runner = TurnRunnerBuilder::new().build();
        let runtime = Arc::new(AgentRuntime::new(runner, event_tx));
        let config = opensquilla_core::config::Config::default();
        let storage = SessionStorage::in_memory().unwrap();
        let state = AppState::new(runtime, config, storage);

        // Gateway should not be running initially.
        assert!(!state.is_gateway_running().await);
        assert!(state.gateway_url().await.is_none());
    }
}
