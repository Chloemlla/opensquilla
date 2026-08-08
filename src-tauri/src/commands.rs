//! Tauri `#[command]` handlers for the desktop-native surface: secrets, locale,
//! auto-update, zoom, directory picker, external links, and session
//! import/export.
//!
//! These complement the gateway/agent-bridge commands in [`crate::gateway`]
//! and [`crate::agent_bridge`], which own the session/provider/model/config
//! surfaces. This module owns the commands that back the Electron preload's
//! `saveSecret` / `getSecret`, `getOsLocale`, `checkForUpdates`, zoom
//! shortcuts, `chooseProjectDirectory`, and `openExternal` — i.e. the P1
//! desktop-native features that have no agent-runtime analogue.
//!
//! Every handler is thin: it delegates to the shared [`crate::storage`]
//! secret store, the [`crate::locale`] resolver, the [`crate::updater`]
//! state, and the [`crate::window`] zoom helpers, returning
//! JSON-serializable structs so the Vue frontend can consume them directly.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State, WebviewWindow};
use uuid::Uuid;

use crate::error::{TauriError, TauriResult};
use crate::ipc::SessionInfo;
use crate::locale::{self, DesktopLocale};
use crate::state::AppState;
use crate::storage::{self, SharedSecretStore};
use crate::updater::{self, DesktopUpdatePlatform, UpdateStateHandle};
use crate::window::{self, ZoomCommand};

// ---------------------------------------------------------------------------
// Session import / export
// ---------------------------------------------------------------------------

/// Export a session and its transcript as a JSON payload.
#[tauri::command]
pub async fn export_session(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<serde_json::Value> {
    let id = Uuid::parse_str(&session_id)
        .map_err(|e| TauriError::bad_request(format!("invalid session id: {e}")))?;
    let storage = state.session_storage().await;
    let session = storage
        .get_session(&id)
        .map_err(|e| TauriError::internal(e.to_string()))?
        .ok_or_else(|| TauriError::not_found(format!("session {session_id} not found")))?;
    let transcript = storage
        .get_transcript_entries(&id, 10_000, 0)
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(serde_json::json!({
        "session": session,
        "transcript": transcript,
    }))
}

/// Import a session payload previously produced by [`export_session`].
#[tauri::command]
pub async fn import_session(
    state: State<'_, AppState>,
    payload: serde_json::Value,
) -> TauriResult<SessionInfo> {
    let session: opensquilla_session::Session = serde_json::from_value(
        payload
            .get("session")
            .cloned()
            .ok_or_else(|| TauriError::bad_request("missing session field"))?,
    )
    .map_err(|e| TauriError::bad_request(format!("invalid session payload: {e}")))?;
    let storage = state.session_storage().await;
    storage
        .create_session(&session)
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(SessionInfo {
        id: session.id.to_string(),
        title: session.name,
        model: String::new(),
        agent_id: session.agent_id.to_string(),
        created_at: session.created_at.to_rfc3339(),
        updated_at: session.updated_at.to_rfc3339(),
        state: format!("{:?}", session.status).to_lowercase(),
        mode: format!("{:?}", session.mode).to_lowercase(),
        message_count: session.message_count,
        system_prompt: if session.system_prompt.is_empty() {
            None
        } else {
            Some(session.system_prompt)
        },
        total_tokens: session.total_tokens,
    })
}

// ---------------------------------------------------------------------------
// Directory picker
// ---------------------------------------------------------------------------

/// Open a native directory picker and return the chosen path.
///
/// Mirrors the Electron `chooseProjectDirectory` preload entry: an optional
/// `initialPath` seeds the dialog.
#[tauri::command]
pub async fn pick_directory(
    app: AppHandle,
    initial_path: Option<String>,
) -> TauriResult<Option<String>> {
    use tauri_plugin_dialog::DialogExt;
    // The plugin's pick_folder callback is synchronous from the JS side but
    // we surface it as an async command by parking on a oneshot.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut builder = app.dialog().file().set_title("Choose a directory");
    if let Some(path) = initial_path.as_deref() {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            builder = builder.set_directory(trimmed);
        }
    }
    builder.pick_folder(move |path| {
        let _ = tx.send(path);
    });
    let result = rx
        .await
        .map_err(|e| TauriError::internal(format!("directory picker cancelled: {e}")))?
        .and_then(|p| p.into_path().ok())
        .map(|pb| pb.to_string_lossy().into_owned());
    Ok(result)
}

// ---------------------------------------------------------------------------
// Secret commands
// ---------------------------------------------------------------------------

/// Input for [`save_secret`].
#[derive(Debug, Clone, Deserialize)]
pub struct SaveSecretInput {
    pub namespace: String,
    pub value: String,
}

/// The result of [`get_secret`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetSecretResult {
    pub namespace: String,
    pub value: Option<String>,
    pub backend: String,
}

/// The inventory returned by [`list_secrets`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SecretInventory {
    pub namespaces: Vec<String>,
    pub backend: String,
}

/// Store a secret (API key, channel token) under a namespace.
#[tauri::command]
pub async fn save_secret(
    secrets: State<'_, SharedSecretStore>,
    input: SaveSecretInput,
) -> TauriResult<serde_json::Value> {
    secrets
        .set(&input.namespace, &input.value)
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(serde_json::json!({ "namespace": input.namespace, "saved": true }))
}

/// Retrieve a secret. Returns the backend in use so the UI can warn about
/// plaintext storage.
#[tauri::command]
pub async fn get_secret(
    secrets: State<'_, SharedSecretStore>,
    namespace: String,
) -> TauriResult<GetSecretResult> {
    let value = secrets
        .get(&namespace)
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(GetSecretResult {
        namespace,
        value,
        backend: secrets.backend().as_str().to_string(),
    })
}

/// Delete a stored secret.
#[tauri::command]
pub async fn delete_secret(
    secrets: State<'_, SharedSecretStore>,
    namespace: String,
) -> TauriResult<serde_json::Value> {
    let removed = secrets
        .delete(&namespace)
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(serde_json::json!({ "namespace": namespace, "deleted": removed }))
}

/// List every stored secret namespace (values never leave the backend).
#[tauri::command]
pub async fn list_secrets(secrets: State<'_, SharedSecretStore>) -> TauriResult<SecretInventory> {
    Ok(SecretInventory {
        namespaces: secrets.list(),
        backend: secrets.backend().as_str().to_string(),
    })
}

/// Rotate the secret-store master key, re-encrypting every entry.
#[tauri::command]
pub async fn rotate_secret_key(
    secrets: State<'_, SharedSecretStore>,
) -> TauriResult<serde_json::Value> {
    secrets
        .rotate_key()
        .map_err(|e| TauriError::internal(e.to_string()))?;
    Ok(serde_json::json!({ "rotated": true }))
}

// ---------------------------------------------------------------------------
// Locale commands
// ---------------------------------------------------------------------------

/// The locale info returned by [`get_locale`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocaleInfo {
    pub locale: String,
    pub detected: String,
    pub overridden: bool,
    pub bundled: Vec<String>,
}

/// Input for [`set_locale`].
#[derive(Debug, Clone, Deserialize)]
pub struct SetLocaleInput {
    pub locale: String,
}

/// Get the effective locale (override or OS-detected) plus the bundled set.
#[tauri::command]
pub async fn get_locale(app: AppHandle) -> TauriResult<LocaleInfo> {
    let config_dir = config_dir_for(&app);
    let detected = locale::detect_system_locale(&app);
    let overridden = locale::load_override(&config_dir);
    let effective = overridden.unwrap_or(detected);
    Ok(LocaleInfo {
        locale: effective.as_str().to_string(),
        detected: detected.as_str().to_string(),
        overridden: overridden.is_some(),
        bundled: locale::BUNDLED_LOCALES
            .iter()
            .map(|l| l.as_str().to_string())
            .collect(),
    })
}

/// Persist a user locale override.
#[tauri::command]
pub async fn set_locale(app: AppHandle, input: SetLocaleInput) -> TauriResult<serde_json::Value> {
    let locale = DesktopLocale::from_tag(&input.locale)
        .ok_or_else(|| TauriError::bad_request(format!("unsupported locale: {}", input.locale)))?;
    let config_dir = config_dir_for(&app);
    locale::save_override(&config_dir, locale).map_err(|e| TauriError::internal(e.to_string()))?;
    let _ = app.emit(
        "locale://changed",
        serde_json::json!({ "locale": locale.as_str() }),
    );
    Ok(serde_json::json!({ "locale": locale.as_str(), "saved": true }))
}

// ---------------------------------------------------------------------------
// Update commands
// ---------------------------------------------------------------------------

/// The update info returned by [`check_updates`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateInfo {
    pub available: bool,
    pub version: Option<String>,
    pub release_url: Option<String>,
    pub body: Option<String>,
}

/// Check for available updates. Resolves the channel manifest for the current
/// version and compares against the running build.
#[tauri::command]
pub async fn check_updates(
    app: AppHandle,
    state: State<'_, UpdateStateHandle>,
) -> TauriResult<UpdateInfo> {
    let version = app.package_info().version.to_string();
    let Some(url) = updater::channel_manifest_url(&version) else {
        state.with_mut(|s| {
            s.error = Some("current version has no update channel".into());
        });
        return Ok(UpdateInfo {
            available: false,
            version: None,
            release_url: None,
            body: None,
        });
    };

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| TauriError::internal(e.to_string()))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| TauriError::internal(e.to_string()))?;
    if !response.status().is_success() {
        state.with_mut(|s| {
            s.error = Some(format!("channel manifest returned {}", response.status()));
        });
        return Ok(UpdateInfo {
            available: false,
            version: None,
            release_url: None,
            body: None,
        });
    }
    let manifest: serde_json::Value = response
        .json()
        .await
        .map_err(|e| TauriError::internal(e.to_string()))?;
    let candidate =
        updater::candidate_from_channel(&version, &manifest, DesktopUpdatePlatform::current())
            .map_err(|e| TauriError::internal(e.to_string()))?;

    let info = match candidate {
        Some(c) => {
            state.with_mut(|s| {
                s.available = true;
                s.version = Some(c.version.clone());
                s.release_url = Some(c.release_url.clone());
                s.error = None;
            });
            UpdateInfo {
                available: true,
                version: Some(c.version),
                release_url: Some(c.release_url),
                body: Some("A new version is available.".to_string()),
            }
        }
        None => {
            state.with_mut(|s| {
                s.available = false;
                s.error = None;
            });
            UpdateInfo {
                available: false,
                version: None,
                release_url: None,
                body: None,
            }
        }
    };
    let _ = app.emit("update://state", state.snapshot());
    Ok(info)
}

/// Download and install a pending update, then restart the app.
#[tauri::command]
pub async fn install_update(
    app: AppHandle,
    state: State<'_, UpdateStateHandle>,
) -> TauriResult<serde_json::Value> {
    state.with_mut(|s| {
        s.downloading = true;
        s.error = None;
    });
    let _ = app.emit("update://state", state.snapshot());

    // Delegate to the Tauri updater plugin, which performs the verified
    // download + install for the current platform's native updater.
    use tauri_plugin_updater::UpdaterExt;
    let updater = app
        .updater()
        .map_err(|e| TauriError::internal(format!("updater unavailable: {e}")))?;
    let update = updater
        .check()
        .await
        .map_err(|e| TauriError::internal(format!("update check failed: {e}")))?;
    let Some(update) = update else {
        state.with_mut(|s| {
            s.downloading = false;
            s.available = false;
        });
        let _ = app.emit("update://state", state.snapshot());
        return Ok(serde_json::json!({ "installed": false, "reason": "no update available" }));
    };

    state.with_mut(|s| {
        s.version = Some(update.version.clone());
    });
    update
        .download_and_install(
            |progress, total| {
                tracing::debug!(progress, total, "update download progress");
            },
            || {},
        )
        .await
        .map_err(|e| TauriError::internal(format!("update install failed: {e}")))?;

    state.with_mut(|s| {
        s.downloading = false;
        s.downloaded = true;
        s.applying = true;
    });
    let _ = app.emit("update://state", state.snapshot());

    // Restart the app to apply the update. `AppHandle::restart` re-executes
    // the current binary, which the updater plugin uses to launch the new
    // version.
    app.restart();
}

// ---------------------------------------------------------------------------
// External link / open commands
// ---------------------------------------------------------------------------

/// Open a URL or path in the user's default application.
#[tauri::command]
#[allow(deprecated)]
pub async fn open_external(app: AppHandle, target: String) -> TauriResult<()> {
    use tauri_plugin_shell::ShellExt;
    app.shell()
        .open(target, None)
        .map_err(|e| TauriError::internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// Zoom commands
// ---------------------------------------------------------------------------

/// Zoom the calling window in by one step.
#[tauri::command]
pub async fn zoom_in(window: WebviewWindow) -> TauriResult<f64> {
    window::apply_zoom(&window, ZoomCommand::In).map_err(|e| TauriError::internal(e.to_string()))
}

/// Zoom the calling window out by one step.
#[tauri::command]
pub async fn zoom_out(window: WebviewWindow) -> TauriResult<f64> {
    window::apply_zoom(&window, ZoomCommand::Out).map_err(|e| TauriError::internal(e.to_string()))
}

/// Reset the calling window's zoom to 100%.
#[tauri::command]
pub async fn zoom_reset(window: WebviewWindow) -> TauriResult<f64> {
    window::apply_zoom(&window, ZoomCommand::Reset).map_err(|e| TauriError::internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the app config directory from the app handle, falling back to the
/// platform default if the Tauri path resolver is unavailable.
fn config_dir_for(app: &AppHandle) -> PathBuf {
    app.path().app_config_dir().unwrap_or_else(|_| {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("opensquilla")
    })
}

/// Open the shared secret store for the running app. Called by `main.rs`
/// during setup so the store is ready before any command runs.
pub fn open_secret_store(app: &AppHandle) -> SharedSecretStore {
    let config_dir = config_dir_for(app);
    // Tauri 2 does not expose a `packaged()` accessor; a release build
    // (debug_assertions off) is the reliable packaged signal across platforms.
    let app_packaged = !cfg!(debug_assertions);
    storage::open_shared(&config_dir, app_packaged).unwrap_or_else(|e| {
        tracing::error!(error = ?e, "failed to open secret store; using ephemeral fallback");
        let policy = storage::SecretStoragePolicyInput {
            env_mode: Some("plain".into()),
            platform: std::env::consts::OS.into(),
            app_packaged,
            codesign_diagnostic: None,
        };
        let store = storage::SecretStore::open(&config_dir, &policy)
            .expect("plaintext store must always open");
        Arc::new(store)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_secret_input_deserializes() {
        let json = serde_json::json!({ "namespace": "provider:openai", "value": "sk-test" });
        let input: SaveSecretInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.namespace, "provider:openai");
        assert_eq!(input.value, "sk-test");
    }

    #[test]
    fn set_locale_input_deserializes() {
        let json = serde_json::json!({ "locale": "zh-Hans" });
        let input: SetLocaleInput = serde_json::from_value(json).unwrap();
        assert_eq!(input.locale, "zh-Hans");
    }
}
