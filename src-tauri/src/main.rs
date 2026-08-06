//! # OpenSquilla Desktop — Tauri v2 entry point
//!
//! Replaces the Electron shell at `desktop/electron/`. This binary is the
//! single-process desktop app: the Tauri webview, the gateway (axum), and the
//! agent runtime all share one Tokio runtime, per the migration analysis's
//! "同一进程" decision.
//!
//! ## Startup sequence
//!
//! 1. Initialize `tracing-subscriber` (JSON in release, pretty otherwise).
//! 2. Build a multi-threaded Tokio runtime.
//! 3. Load the OpenSquilla configuration (falling back to defaults).
//! 4. Register the `opensquilla://` deep-link scheme.
//! 5. Acquire the single-instance lock.
//! 6. Build the Tauri app: all plugins, managed state, command handlers,
//!    system tray, window-state restoration, and the `setup` hook.
//! 7. In `setup`, start the gateway runtime and surface the main window.
//!
//! ## Shutdown
//!
//! The app drains gracefully: the gateway is shut down via its graceful-shutdown
//! channel and the Tokio runtime is dropped cleanly.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use opensquilla_core::config::Config;
use opensquilla_desktop_lib::agent_bridge;
use opensquilla_desktop_lib::gateway;
use opensquilla_desktop_lib::state::AppState;
use opensquilla_desktop_lib::workbench;
use opensquilla_desktop_lib::{
    TrayEvent, commands,
    deep_link,
    tray::{TrayIconState, build_tray_menu, rebuild_menu},
    updater, window,
};
use opensquilla_engine::{AgentRuntime, TurnRunnerBuilder};
use opensquilla_session::SessionStorage;
use std::sync::Arc;
use tauri::{
    Emitter, Manager, WindowEvent,
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};
use tauri_plugin_deep_link::DeepLinkExt;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

// ---------------------------------------------------------------------------
// Inline commands (must be defined in the binary crate so that Tauri v2's
// __cmd__ macros are visible to generate_handler!).
// ---------------------------------------------------------------------------

#[tauri::command]
async fn ping() -> &'static str {
    "pong"
}

#[tauri::command]
async fn app_info(_app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let info = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "name": env!("CARGO_PKG_NAME"),
        "tauri": "2",
    });
    Ok(info)
}

#[tauri::command]
async fn reload_config(
    state: tauri::State<'_, opensquilla_desktop_lib::state::AppState>,
) -> Result<serde_json::Value, String> {
    let guard = state.config().await;
    let config = opensquilla_core::config::Config::clone(&guard);
    drop(guard);
    serde_json::to_value(&config).map_err(|e| e.to_string())
}

/// Entry point. Builds the Tokio runtime and hands control to Tauri.
fn main() {
    init_tracing();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("opensquilla")
        .build()
        .expect("failed to build Tokio runtime");

    let _guard = runtime.enter();

    // Load configuration up front so managed state is ready before Tauri
    // builds the webview. A missing config is non-fatal: we fall back to
    // defaults so the onboarding flow can still render.
    let config = match Config::load() {
        Ok(c) => {
            tracing::info!("loaded OpenSquilla configuration");
            c
        }
        Err(e) => {
            tracing::warn!(error = %e, "no configuration found; using defaults");
            Config {
                gateway: opensquilla_core::config::GatewayConfig::default(),
                providers: Vec::new(),
                channels: Vec::new(),
                models: None,
                sandbox: None,
                skills: None,
                scheduler: None,
                observability: None,
            }
        }
    };

    // Build the app state: agent runtime + config + session storage.
    let app_state = build_app_state(&runtime, config);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            // A second launch was attempted. Surface the running instance.
            tracing::info!("second instance blocked; focusing main window");
            let links = deep_link::deep_link_arguments(&argv);
            if let Some(first) = links.first() {
                match deep_link::parse_deep_link(first) {
                    Ok(_action) => window::activate_main_window(app),
                    Err(e) => tracing::warn!(error = %e, "could not parse forwarded deep link"),
                }
            } else {
                window::activate_main_window(app);
            }
        }))
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            // Basic commands
            ping,
            app_info,
            reload_config,
            // Gateway commands
            gateway::start_gateway,
            gateway::stop_gateway,
            gateway::gateway_status,
            gateway::restart_gateway,
            gateway::get_gateway_url,
            // Agent bridge commands
            agent_bridge::send_message,
            agent_bridge::send_message_sync,
            agent_bridge::cancel_turn,
            agent_bridge::get_chat_history,
            agent_bridge::clear_chat_history,
            // Session commands
            agent_bridge::create_session,
            agent_bridge::list_sessions,
            agent_bridge::get_session,
            agent_bridge::delete_session,
            agent_bridge::archive_session,
            // Provider / Model / Skill commands
            agent_bridge::list_providers,
            agent_bridge::list_models,
            agent_bridge::list_skills,
            // Health / Config commands
            agent_bridge::health_check,
            agent_bridge::get_config,
            agent_bridge::set_config,
            agent_bridge::get_config_value,
            agent_bridge::list_config,
            // Workbench commands
            workbench::create_workbench_surface,
            workbench::destroy_workbench_surface,
            workbench::set_workbench_surface_rect,
            workbench::navigate_workbench_surface,
            workbench::destroy_all_workbench_surfaces,
            workbench::list_workbench_surfaces,
            workbench::create_artifact_preview_lease,
            workbench::revoke_artifact_preview_lease,
            // Desktop-native commands (P1)
            commands::export_session,
            commands::import_session,
            commands::pick_directory,
            commands::save_secret,
            commands::get_secret,
            commands::delete_secret,
            commands::list_secrets,
            commands::rotate_secret_key,
            commands::get_locale,
            commands::set_locale,
            commands::check_updates,
            commands::install_update,
            commands::open_external,
            commands::zoom_in,
            commands::zoom_out,
            commands::zoom_reset,
        ])
        .setup(|app| {
            // Managed state for desktop-native features (P1).
            app.manage(commands::open_secret_store(app.handle()));
            app.manage(updater::UpdateStateHandle::new());

            // Build the system tray.
            setup_tray(app.handle())?;

            // Wire tray menu clicks.
            app.on_menu_event(|app, event| {
                if let Some(tray_event) = TrayEvent::from_menu_id(event.id().as_ref()) {
                    match tray_event {
                        TrayEvent::ShowHide => window::toggle_main_window(app),
                        TrayEvent::NewSession | TrayEvent::Settings | TrayEvent::CheckUpdates => {
                            let _ = app.emit("tray://event", tray_event);
                        }
                        TrayEvent::Quit => {
                            tracing::info!("quit requested from tray");
                            app.exit(0);
                        }
                    }
                }
            });

            // Register the `opensquilla://` scheme and route incoming URLs.
            register_deep_links(app.handle());
            {
                let handle = app.handle().clone();
                app.deep_link().on_open_url(move |event| {
                    for url in event.urls() {
                        handle_deep_link(&handle, url.as_str());
                    }
                });
            }

            // If the app was launched via a deep link (cold start), handle it now.
            if let Ok(Some(urls)) = app.deep_link().get_current() {
                for url in &urls {
                    handle_deep_link(app.handle(), url.as_str());
                }
            }

            // Start the agent runtime inside the Tauri app.
            let runtime_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                start_runtime(&runtime_handle).await;
            });

            // Minimize-to-tray: hide instead of close so background work continues.
            let window = app
                .get_webview_window(window::MAIN_WINDOW_LABEL)
                .expect("main window missing");
            let window_clone = window.clone();
            window.on_window_event(move |event| {
                if let WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window_clone.hide();
                    let _ = window_clone.app_handle().emit("window://hidden", ());
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Restore window state and apply devtools in debug.
            if let WindowEvent::Destroyed = event {
                tracing::info!(label = window.label(), "window destroyed");
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running OpenSquilla desktop app");

    tracing::info!("OpenSquilla desktop shut down cleanly");
}

/// Build the application state from configuration and a Tokio runtime handle.
///
/// Creates:
/// 1. An in-memory or file-based SessionStorage.
/// 2. An AgentRuntime with a default TurnRunner, started on the Tokio runtime.
/// 3. The AppState wrapping all of the above.
fn build_app_state(rt: &tokio::runtime::Runtime, config: Config) -> AppState {
    // Create session storage. Use a file-based database in the app data dir,
    // falling back to in-memory if the directory is not available.
    let storage_path = dirs::data_dir()
        .map(|d| d.join("opensquilla").join("sessions.db"))
        .map(|p| p.to_string_lossy().to_string());

    let session_storage = match &storage_path {
        Some(path) => {
            // Ensure the parent directory exists.
            if let Some(parent) = std::path::Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            SessionStorage::new(path).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to open session database, using in-memory");
                SessionStorage::in_memory().unwrap()
            })
        }
        None => SessionStorage::in_memory().unwrap(),
    };

    // Create the event channel for turn events.
    let (event_tx, mut event_rx) =
        tokio::sync::mpsc::channel::<opensquilla_core::events::TurnEvent>(256);

    // Drain turn events in the background (prevents the channel from blocking
    // when no consumer is attached). The agent_bridge module sets up its own
    // event forwarding per-turn.
    rt.spawn(async move {
        while event_rx.recv().await.is_some() {
            // Events are handled per-turn via the streaming channel.
        }
    });

    // Build the default TurnRunner.
    let runner = TurnRunnerBuilder::new()
        .max_tool_rounds(10)
        .streaming(true)
        .build();

    // Create and start the AgentRuntime.
    let runtime = Arc::new(AgentRuntime::new(runner, event_tx));
    rt.block_on(async {
        runtime
            .start()
            .await
            .expect("failed to start agent runtime");
    });
    tracing::info!("agent runtime started");

    AppState::new(runtime, config, session_storage)
}

/// Initialize structured tracing.
fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,opensquilla=debug"));

    let registry = tracing_subscriber::registry().with(filter);

    if cfg!(debug_assertions) {
        registry.with(fmt::layer().pretty()).init();
    } else {
        registry.with(fmt::layer().json()).init();
    }
}

/// Build the system tray with its initial menu.
fn setup_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let menu = build_tray_menu(app, TrayIconState::Starting)?;

    let _tray = TrayIconBuilder::with_id("main-tray")
        .icon(app.default_window_icon().cloned().expect("no app icon"))
        .tooltip("OpenSquilla")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                window::toggle_main_window(app);
            }
        })
        .build(app)?;

    Ok(())
}

/// Register the `opensquilla://` scheme for this instance (desktop only).
fn register_deep_links(app: &tauri::AppHandle) {
    #[cfg(desktop)]
    {
        let result = app.deep_link().register("opensquilla");
        match result {
            Ok(()) => tracing::info!("registered opensquilla:// deep-link scheme"),
            Err(e) => tracing::warn!(error = %e, "deep-link registration failed"),
        }
    }
}

/// Handle an incoming deep-link URL.
fn handle_deep_link(app: &tauri::AppHandle, url: &str) {
    match deep_link::parse_deep_link(url) {
        Ok(action) => {
            tracing::info!(url = %url, action = ?action, "deep-link received");
            // Dispatch surfaces the main window and emits `deep-link://action`
            // so the frontend can resume the session or start an import.
            deep_link::dispatch(app, url, &action);
        }
        Err(e) => {
            tracing::warn!(url = %url, error = %e, "rejected deep link");
        }
    }
}

/// Start the in-process gateway and emit status updates to the frontend.
async fn start_runtime(app: &tauri::AppHandle) {
    // Auto-start the gateway on app launch.
    let state = app.state::<AppState>();

    match gateway::start_gateway_inner(app, state.inner()).await {
        Ok(status) => {
            tracing::info!(
                running = status.running,
                url = ?status.url,
                "Gateway auto-started"
            );
            // Reflect the running state in the tray menu.
            let _ = rebuild_menu(app, TrayIconState::Running);
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to auto-start gateway");
            let _ = rebuild_menu(app, TrayIconState::Failed);
        }
    }
}
