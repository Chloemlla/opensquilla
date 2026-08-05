//! # OpenSquilla Desktop Library
//!
//! Shared library crate for the Tauri v2 desktop shell. Declares the Tauri
//! command modules, the managed runtime state, and re-exports the public
//! command handlers so `main.rs` (and tests) can register them in one place.
//!
//! The binary entry point lives in [`main`](../../main.rs); this crate owns
//! everything that benefits from being unit-tested in isolation: command
//! handlers, the runtime state container, and the agent runtime bootstrap.
//!
//! ## Modules
//!
//! - [`error`] — Tauri error handling, converts `CoreError` to serializable `TauriError`
//! - [`ipc`] — IPC contract types that cross the Tauri invoke boundary
//! - [`state`] — App state management (`AppState` with `Arc<AgentRuntime>`, `Arc<Gateway>`, etc.)
//! - [`gateway`] — In-process axum gateway server (replaces Python subprocess)
//! - [`agent_bridge`] — Bridge between Tauri commands and the engine (the core integration)
//! - [`workbench`] — Native workbench surface management (replaces Electron native surface)
//! - [`commands`] — Basic Tauri commands (ping, app_info, etc.)
//! - [`runtime`] — Desktop runtime lifecycle management
//! - [`deep_link`] — Deep-link (`opensquilla://`) parsing
//! - [`tray`] — System tray menu
//! - [`window`] — Window management helpers

pub mod agent_bridge;
pub mod commands;
pub mod deep_link;
pub mod error;
pub mod gateway;
pub mod ipc;
pub mod lifecycle;
pub mod locale;
pub mod runtime;
pub mod state;
pub mod storage;
pub mod tray;
pub mod updater;
pub mod window;
pub mod workbench;

use opensquilla_gateway::Gateway;

// Re-export the primary types for convenience.
pub use commands::{
    app_info, ping, reload_config, check_updates, delete_secret, export_session, get_locale,
    get_secret, import_session, install_update, list_secrets, open_external, pick_directory,
    rotate_secret_key, save_secret, set_locale, zoom_in, zoom_out, zoom_reset,
};
pub use deep_link::{DeepLinkAction, DeepLinkError, parse_deep_link};
pub use error::{TauriError, TauriResult};
pub use lifecycle::{
    DesktopCleanupMode, LaunchAuthority, OwnershipRecord, OwnershipRecordLoad,
    ownership_record_path, profile_fingerprint,
};
pub use locale::{BUNDLED_LOCALES, DesktopLocale, detect_system_locale, resolve_locale_from_tags};
pub use runtime::{DesktopRuntime, RuntimeState};
pub use state::AppState;
pub use storage::{SecretStorageBackend, SecretStoragePolicyInput, SharedSecretStore};
pub use tray::{TrayIconState, TrayEvent, build_tray_menu, rebuild_menu};
pub use updater::{UpdateState, UpdateStateHandle};

/// Top-level error type for the desktop shell (legacy, used by runtime module).
#[derive(Debug, thiserror::Error)]
pub enum DesktopError {
    #[error("configuration error: {0}")]
    Config(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tauri error: {0}")]
    Tauri(String),
}

/// Result alias used across the desktop shell (legacy, used by runtime module).
pub type Result<T> = std::result::Result<T, DesktopError>;

/// Convenience alias for the gateway type this shell drives.
pub type DesktopGateway = Gateway;
