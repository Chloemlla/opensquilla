//! Window lifecycle: main-window creation, window-state restoration, zoom
//! shortcuts, close-to-tray behavior, and multi-window management.
//!
//! Replaces `desktop-window-lifecycle.ts` and `desktop-zoom-shortcuts.ts`.
//! Window-state (position/size/maximized) is restored by
//! `tauri-plugin-window-state`; this module layers the desktop-specific
//! behavior on top: the close-to-tray decision tree, zoom factor math, and the
//! canonical show/hide/focus helpers shared by the tray and deep-link handler.
//!
//! The close behavior mirrors the Electron shell exactly:
//! - If the OS is ending the user's session, or the app is already committed
//!   to quitting, allow the close.
//! - Background mode (hide instead of quit) is supported on macOS always, and
//!   on Windows only when the tray is ready.
//! - On Linux, closing always quits (no tray-backed background mode).
//! - A deferred/draining exit keeps the recoverable window hidden until the
//!   exit either commits or returns to running.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, WebviewWindow, WindowEvent};

/// Label of the main window, matching `tauri.conf.json`.
pub const MAIN_WINDOW_LABEL: &str = "main";

/// The minimum zoom factor (matches the Electron `DESKTOP_ZOOM_MIN_FACTOR`).
pub const ZOOM_MIN_FACTOR: f64 = 0.5;
/// The maximum zoom factor.
pub const ZOOM_MAX_FACTOR: f64 = 3.0;
/// The zoom step multiplier for each in/out command.
pub const ZOOM_STEP_FACTOR: f64 = 1.2;

// ---------------------------------------------------------------------------
// Exit-phase / close-behavior model (desktop-window-lifecycle.ts)
// ---------------------------------------------------------------------------

/// The phases a desktop exit moves through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExitPhase {
    Running,
    Deferred,
    Draining,
    Committed,
}

impl ExitPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            ExitPhase::Running => "running",
            ExitPhase::Deferred => "deferred",
            ExitPhase::Draining => "draining",
            ExitPhase::Committed => "committed",
        }
    }
}

/// The user-configured close behavior for the main window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MainWindowCloseBehavior {
    Background,
    Quit,
    Ask,
}

impl MainWindowCloseBehavior {
    /// The platform default: macOS and Windows default to background; Linux
    /// defaults to quit.
    pub fn default_for_platform(platform: &str) -> Self {
        match platform {
            "macos" | "windows" => Self::Background,
            _ => Self::Quit,
        }
    }
}

/// The action to take when the user closes the main window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainWindowCloseAction {
    Allow,
    FocusOnboarding,
    Hide,
    Quit,
    Ask,
}

/// The context consulted to decide a close action.
#[derive(Debug, Clone)]
pub struct MainWindowCloseContext {
    pub platform: String,
    pub exit_phase: ExitPhase,
    pub system_session_ending: bool,
    pub onboarding_open: bool,
    pub behavior: MainWindowCloseBehavior,
    pub windows_tray_ready: bool,
}

/// Decide what to do when the user closes the main window.
///
/// Mirrors `mainWindowCloseAction` in the Electron shell. A system session
/// end or a committed exit always allows the close. Background mode requires
/// macOS or a ready Windows tray. A deferred/draining exit hides the window
/// until the exit settles.
pub fn main_window_close_action(ctx: &MainWindowCloseContext) -> MainWindowCloseAction {
    if ctx.system_session_ending || ctx.exit_phase == ExitPhase::Committed {
        return MainWindowCloseAction::Allow;
    }
    let background_supported =
        ctx.platform == "macos" || (ctx.platform == "windows" && ctx.windows_tray_ready);
    if !background_supported {
        return MainWindowCloseAction::Quit;
    }
    if ctx.onboarding_open {
        return MainWindowCloseAction::FocusOnboarding;
    }
    // A deferred or draining exit still owns live renderer/runtime state.
    if ctx.exit_phase != ExitPhase::Running {
        return MainWindowCloseAction::Hide;
    }
    match ctx.behavior {
        MainWindowCloseBehavior::Quit => MainWindowCloseAction::Quit,
        MainWindowCloseBehavior::Ask => MainWindowCloseAction::Ask,
        MainWindowCloseBehavior::Background => MainWindowCloseAction::Hide,
    }
}

/// True only when the app is in a phase where revealing the window is safe.
pub fn can_reveal_desktop_app(phase: ExitPhase) -> bool {
    phase == ExitPhase::Running
}

// ---------------------------------------------------------------------------
// Desktop preferences (the schema-v3 document)
// ---------------------------------------------------------------------------

/// The persisted desktop preferences (schema v3).
///
/// `sandbox_unavailable_warning_suppressed` carries `#[serde(default)]` so
/// legacy schema-v2 files (which predate the field) deserialize cleanly;
/// readers must treat a missing value as `false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopPreferences {
    pub schema_version: u32,
    pub main_window_close_behavior: MainWindowCloseBehavior,
    pub background_close_notice_shown: bool,
    pub workbench_preview_mode: WorkbenchPreviewMode,
    pub workbench_preview_notice_shown: bool,
    #[serde(default)]
    pub sandbox_unavailable_warning_suppressed: bool,
}

/// The workbench preview mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkbenchPreviewMode {
    Full,
    Offline,
}

impl Default for DesktopPreferences {
    fn default() -> Self {
        Self::for_platform(std::env::consts::OS)
    }
}

impl DesktopPreferences {
    /// The default preferences for the given platform.
    pub fn for_platform(platform: &str) -> Self {
        Self {
            schema_version: 3,
            main_window_close_behavior: MainWindowCloseBehavior::default_for_platform(platform),
            background_close_notice_shown: false,
            workbench_preview_mode: WorkbenchPreviewMode::Full,
            workbench_preview_notice_shown: false,
            sandbox_unavailable_warning_suppressed: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Zoom (desktop-zoom-shortcuts.ts)
// ---------------------------------------------------------------------------

/// A zoom command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ZoomCommand {
    In,
    Out,
    Reset,
}

/// Compute the next zoom factor for a command, clamped to `[MIN, MAX]`.
pub fn zoom_factor(current: f64, command: ZoomCommand) -> f64 {
    match command {
        ZoomCommand::Reset => 1.0,
        ZoomCommand::In => {
            let candidate = current * ZOOM_STEP_FACTOR;
            candidate.min(ZOOM_MAX_FACTOR)
        }
        ZoomCommand::Out => {
            let candidate = current / ZOOM_STEP_FACTOR;
            candidate.max(ZOOM_MIN_FACTOR)
        }
    }
}

/// Apply a zoom command to a window, returning the new factor.
///
/// Tauri 2 exposes `set_zoom` but no zoom getter, so the current factor is
/// tracked per-window-label in a process-local map (defaulting to 1.0). This
/// matches the Electron shell's compounding behavior within a session.
pub fn apply_zoom<R: Runtime>(
    window: &WebviewWindow<R>,
    command: ZoomCommand,
) -> tauri::Result<f64> {
    let label = window.label().to_string();
    let current = current_zoom_factor(&label);
    let next = zoom_factor(current, command);
    window.set_zoom(next)?;
    set_zoom_factor(&label, next);
    Ok(next)
}

/// The per-window zoom factor cache. Tauri 2 has no zoom getter, so we
/// remember the last value we set, keyed by window label.
fn zoom_cache() -> &'static dashmap::DashMap<String, f64> {
    static CACHE: once_cell::sync::Lazy<dashmap::DashMap<String, f64>> =
        once_cell::sync::Lazy::new(dashmap::DashMap::new);
    &CACHE
}

fn current_zoom_factor(label: &str) -> f64 {
    zoom_cache().get(label).map(|r| *r).unwrap_or(1.0)
}

fn set_zoom_factor(label: &str, factor: f64) {
    zoom_cache().insert(label.to_string(), factor);
}

// ---------------------------------------------------------------------------
// Window access helpers
// ---------------------------------------------------------------------------

/// Retrieve the main window, if it exists.
pub fn main_window<R: Runtime>(app: &AppHandle<R>) -> Option<WebviewWindow<R>> {
    app.get_webview_window(MAIN_WINDOW_LABEL)
}

/// Show and focus the main window. Logs a warning if the window is not yet
/// ready (e.g. during early deep-link handoff before the webview is built).
pub fn activate_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = main_window(app) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    } else {
        tracing::warn!("main window not found; activate requested before window ready");
    }
}

/// Hide the main window (minimize-to-tray flow). The app keeps running so
/// background tasks, schedules, and connected channels continue.
pub fn hide_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = main_window(app) {
        let _ = window.hide();
        let _ = app.emit("window://hidden", ());
    }
}

/// Toggle the main window's visibility: hide if visible, show+focus if hidden.
pub fn toggle_main_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = main_window(app) {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
            let _ = app.emit("window://hidden", ());
        } else {
            let _ = window.show();
            let _ = window.unminimize();
            let _ = window.set_focus();
        }
    }
}

/// Emit an event to the main window's frontend.
pub fn emit_to_main<R: Runtime, S: Serialize + Clone>(
    app: &AppHandle<R>,
    event: &str,
    payload: S,
) -> tauri::Result<()> {
    app.emit_to(MAIN_WINDOW_LABEL, event, payload)
}

/// Install the close-to-tray behavior on the main window.
///
/// Call this once from the Tauri `setup` hook. The returned closure should be
/// invoked from `on_window_event`. It consults [`main_window_close_action`] to
/// decide whether to prevent the close (hiding instead) or allow it.
pub fn install_close_to_tray<R: Runtime>(
    window: &WebviewWindow<R>,
    ctx_provider: impl Fn() -> MainWindowCloseContext + Send + Sync + 'static,
) {
    // `on_window_event` requires a 'static closure; clone the window handle so
    // the closure owns it rather than borrowing the argument.
    let win = window.clone();
    window.on_window_event(move |event| {
        let WindowEvent::CloseRequested { api, .. } = event else {
            return;
        };
        let ctx = ctx_provider();
        match main_window_close_action(&ctx) {
            MainWindowCloseAction::Allow => {
                // Let Tauri close the window normally.
            }
            MainWindowCloseAction::Hide => {
                api.prevent_close();
                let _ = win.hide();
                let _ = win.app_handle().emit("window://hidden", ());
            }
            MainWindowCloseAction::FocusOnboarding => {
                api.prevent_close();
                // Surface the onboarding window instead of closing. The
                // frontend listens for this event to focus its onboarding view.
                let _ = win.app_handle().emit("window://focus-onboarding", ());
            }
            MainWindowCloseAction::Quit => {
                // Allow the close; the app will quit when the last window
                // closes (Tauri's default behavior on non-macOS).
                #[cfg(not(target_os = "macos"))]
                {
                    win.app_handle().exit(0);
                }
                #[cfg(target_os = "macos")]
                {
                    api.prevent_close();
                    let _ = win.hide();
                }
            }
            MainWindowCloseAction::Ask => {
                api.prevent_close();
                // Ask the frontend to show the close-prompt dialog; it emits
                // a `window://close-response` event with the user's choice.
                let _ = win.app_handle().emit("window://close-prompt", ());
            }
        }
    });
}

/// A context provider backed by shared mutable state.
///
/// `main.rs` constructs one of these and updates it as the exit phase, tray
/// readiness, and onboarding state change.
pub struct SharedCloseContext {
    inner: std::sync::Arc<parking_lot::Mutex<MainWindowCloseContext>>,
}

impl SharedCloseContext {
    /// Create a new shared context starting from the running phase.
    pub fn new(platform: &str) -> Self {
        Self {
            inner: std::sync::Arc::new(parking_lot::Mutex::new(MainWindowCloseContext {
                platform: platform.to_string(),
                exit_phase: ExitPhase::Running,
                system_session_ending: false,
                onboarding_open: false,
                behavior: MainWindowCloseBehavior::default_for_platform(platform),
                windows_tray_ready: false,
            })),
        }
    }

    /// Read the current context.
    pub fn get(&self) -> MainWindowCloseContext {
        self.inner.lock().clone()
    }

    /// Update the exit phase.
    pub fn set_exit_phase(&self, phase: ExitPhase) {
        self.inner.lock().exit_phase = phase;
    }

    /// Update the close behavior.
    pub fn set_behavior(&self, behavior: MainWindowCloseBehavior) {
        self.inner.lock().behavior = behavior;
    }

    /// Update whether the Windows tray is ready.
    pub fn set_windows_tray_ready(&self, ready: bool) {
        self.inner.lock().windows_tray_ready = ready;
    }

    /// Update whether onboarding is open.
    pub fn set_onboarding_open(&self, open: bool) {
        self.inner.lock().onboarding_open = open;
    }

    /// Update whether the OS is ending the user's session.
    pub fn set_system_session_ending(&self, ending: bool) {
        self.inner.lock().system_session_ending = ending;
    }

    /// Get a closure suitable for [`install_close_to_tray`].
    pub fn closure(&self) -> impl Fn() -> MainWindowCloseContext + Send + Sync + 'static {
        let inner = self.inner.clone();
        move || inner.lock().clone()
    }
}

// ---------------------------------------------------------------------------
// Multi-window management
// ---------------------------------------------------------------------------

/// Create a secondary window (e.g. for an onboarding or workbench surface).
///
/// The label must be unique; calling with an existing label returns the
/// existing window. `url` is interpreted as a path into the compiled webui
/// assets (e.g. `"index.html#/settings"`).
pub fn create_secondary_window<R: Runtime>(
    app: &AppHandle<R>,
    label: &str,
    title: &str,
    url: &str,
    width: f64,
    height: f64,
) -> tauri::Result<WebviewWindow<R>> {
    if let Some(existing) = app.get_webview_window(label) {
        return Ok(existing);
    }
    WebviewWindow::builder(app, label, tauri::WebviewUrl::App(url.into()))
        .title(title)
        .inner_size(width, height)
        .build()
}

/// Focus a window by label, creating it if a builder is provided.
pub fn focus_window<R: Runtime>(app: &AppHandle<R>, label: &str) -> bool {
    if let Some(window) = app.get_webview_window(label) {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
        true
    } else {
        false
    }
}

/// Close and destroy a window by label. Returns true if a window was closed.
pub fn destroy_window<R: Runtime>(app: &AppHandle<R>, label: &str) -> bool {
    if let Some(window) = app.get_webview_window(label) {
        let _ = window.close();
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn close_action_allows_committed_exit() {
        let ctx = MainWindowCloseContext {
            platform: "linux".into(),
            exit_phase: ExitPhase::Committed,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Quit,
            windows_tray_ready: false,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Allow);
    }

    #[test]
    fn close_action_linux_quits() {
        let ctx = MainWindowCloseContext {
            platform: "linux".into(),
            exit_phase: ExitPhase::Running,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Background,
            windows_tray_ready: false,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Quit);
    }

    #[test]
    fn close_action_windows_background_with_tray() {
        let ctx = MainWindowCloseContext {
            platform: "windows".into(),
            exit_phase: ExitPhase::Running,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Background,
            windows_tray_ready: true,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Hide);
    }

    #[test]
    fn close_action_windows_quits_without_tray() {
        let ctx = MainWindowCloseContext {
            platform: "windows".into(),
            exit_phase: ExitPhase::Running,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Background,
            windows_tray_ready: false,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Quit);
    }

    #[test]
    fn close_action_deferred_hides() {
        let ctx = MainWindowCloseContext {
            platform: "macos".into(),
            exit_phase: ExitPhase::Deferred,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Background,
            windows_tray_ready: false,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Hide);
    }

    #[test]
    fn close_action_ask_when_configured() {
        let ctx = MainWindowCloseContext {
            platform: "macos".into(),
            exit_phase: ExitPhase::Running,
            system_session_ending: false,
            onboarding_open: false,
            behavior: MainWindowCloseBehavior::Ask,
            windows_tray_ready: false,
        };
        assert_eq!(main_window_close_action(&ctx), MainWindowCloseAction::Ask);
    }

    #[test]
    fn zoom_clamps_and_resets() {
        assert_eq!(zoom_factor(1.0, ZoomCommand::Reset), 1.0);
        let stepped = zoom_factor(1.0, ZoomCommand::In);
        assert!((stepped - 1.2).abs() < 1e-9);
        // Clamp to max.
        let maxed = zoom_factor(ZOOM_MAX_FACTOR, ZoomCommand::In);
        assert_eq!(maxed, ZOOM_MAX_FACTOR);
        // Clamp to min.
        let mined = zoom_factor(ZOOM_MIN_FACTOR, ZoomCommand::Out);
        assert_eq!(mined, ZOOM_MIN_FACTOR);
    }

    #[test]
    fn default_preferences_per_platform() {
        assert_eq!(
            DesktopPreferences::for_platform("linux").main_window_close_behavior,
            MainWindowCloseBehavior::Quit
        );
        assert_eq!(
            DesktopPreferences::for_platform("macos").main_window_close_behavior,
            MainWindowCloseBehavior::Background
        );
        assert_eq!(
            DesktopPreferences::for_platform("windows").main_window_close_behavior,
            MainWindowCloseBehavior::Background
        );
    }

    #[test]
    fn can_reveal_only_when_running() {
        assert!(can_reveal_desktop_app(ExitPhase::Running));
        assert!(!can_reveal_desktop_app(ExitPhase::Deferred));
        assert!(!can_reveal_desktop_app(ExitPhase::Draining));
        assert!(!can_reveal_desktop_app(ExitPhase::Committed));
    }
}
