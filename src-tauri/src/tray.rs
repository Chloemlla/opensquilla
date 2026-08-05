//! System tray: icon with context menu, click-to-toggle, and dynamic states.
//!
//! Replaces the Electron `Tray` + Windows tray menu logic. The menu surface
//! expands the Electron shell's three items to the full set the Tauri shell
//! exposes: Show/Hide, New Session, Settings, Check Updates, and Quit. The
//! tray click toggles main-window visibility (the Electron shell only revealed
//! on click; toggling is the conventional desktop behavior and matches the
//! `menuOnLeftClick: false` config in `tauri.conf.json`).
//!
//! The tray is built once in `main.rs` via [`TrayIconBuilder`]; this module
//! owns the menu construction, the menu-item-id → event mapping, and the
//! dynamic status label that reflects the runtime lifecycle.

use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Manager, Runtime,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconEvent,
};

/// Menu item ids. These are the strings Tauri delivers to `on_menu_event`.
pub const ID_SHOW_HIDE: &str = "tray-show-hide";
pub const ID_NEW_SESSION: &str = "tray-new-session";
pub const ID_SETTINGS: &str = "tray-settings";
pub const ID_CHECK_UPDATES: &str = "tray-check-updates";
pub const ID_QUIT: &str = "tray-quit";
pub const ID_STATUS: &str = "tray-status";

/// The tray icon's label for the running state, used to pick the dynamic icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayIconState {
    Starting,
    Running,
    Stopping,
    Stopped,
    Failed,
}

impl TrayIconState {
    /// The human-readable status line for the menu.
    pub fn status_label(self) -> &'static str {
        match self {
            TrayIconState::Starting => "OpenSquilla is starting",
            TrayIconState::Running => "OpenSquilla is running",
            TrayIconState::Stopping => "OpenSquilla is stopping",
            TrayIconState::Stopped => "OpenSquilla is stopped",
            TrayIconState::Failed => "OpenSquilla failed to start",
        }
    }
}

/// Events emitted by the system tray menu, surfaced to the frontend and
/// dispatched by `main.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrayEvent {
    /// User clicked "Show/Hide" — toggle the main window.
    ShowHide,
    /// User clicked "New Session" — ask the frontend to start a new session.
    NewSession,
    /// User clicked "Settings" — ask the frontend to open settings.
    Settings,
    /// User clicked "Check Updates" — trigger a manual update check.
    CheckUpdates,
    /// User clicked "Quit OpenSquilla" — exit the app.
    Quit,
}

impl TrayEvent {
    /// Map a menu item id to a [`TrayEvent`]. Returns `None` for the read-only
    /// status line and separators.
    pub fn from_menu_id(id: &str) -> Option<Self> {
        match id {
            ID_SHOW_HIDE => Some(Self::ShowHide),
            ID_NEW_SESSION => Some(Self::NewSession),
            ID_SETTINGS => Some(Self::Settings),
            ID_CHECK_UPDATES => Some(Self::CheckUpdates),
            ID_QUIT => Some(Self::Quit),
            _ => None,
        }
    }
}

/// Build the tray context menu.
///
/// The menu has six items: Show/Hide, New Session, Settings, a separator, the
/// disabled status line, a separator, Check Updates, and Quit. The status line
/// is rebuilt whenever the runtime state changes via [`rebuild_menu`].
pub fn build_tray_menu<R: Runtime>(
    app: &AppHandle<R>,
    icon_state: TrayIconState,
) -> tauri::Result<Menu<R>> {
    let show_hide = MenuItem::with_id(app, ID_SHOW_HIDE, "Show / Hide", true, None::<&str>)?;
    let new_session = MenuItem::with_id(app, ID_NEW_SESSION, "New Session", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, ID_SETTINGS, "Settings", true, None::<&str>)?;
    let sep1 = PredefinedMenuItem::separator(app)?;
    let status = MenuItem::with_id(
        app,
        ID_STATUS,
        icon_state.status_label(),
        false,
        None::<&str>,
    )?;
    let sep2 = PredefinedMenuItem::separator(app)?;
    let check_updates = MenuItem::with_id(
        app,
        ID_CHECK_UPDATES,
        "Check for Updates",
        true,
        None::<&str>,
    )?;
    let sep3 = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, ID_QUIT, "Quit OpenSquilla", true, None::<&str>)?;

    Menu::with_items(
        app,
        &[
            &show_hide,
            &new_session,
            &settings,
            &sep1,
            &status,
            &sep2,
            &check_updates,
            &sep3,
            &quit,
        ],
    )
}

/// Rebuild the tray menu after a runtime state transition.
///
/// Looks up the `main-tray` tray icon (the id assigned in `main.rs`) and
/// replaces its menu with a fresh one reflecting `icon_state`.
pub fn rebuild_menu<R: Runtime>(
    app: &AppHandle<R>,
    icon_state: TrayIconState,
) -> tauri::Result<()> {
    let Some(tray) = app.tray_by_id("main-tray") else {
        return Ok(());
    };
    let menu = build_tray_menu(app, icon_state)?;
    tray.set_menu(Some(menu))?;
    Ok(())
}

/// The canonical tray click handler: toggle the main window on left-click.
///
/// Wire this to `TrayIconBuilder::on_tray_icon_event` in `main.rs`.
pub fn handle_tray_icon_event<R: Runtime>(app: &AppHandle<R>, event: TrayIconEvent) {
    if let TrayIconEvent::Click {
        button: tauri::tray::MouseButton::Left,
        button_state: tauri::tray::MouseButtonState::Up,
        ..
    } = event
    {
        crate::window::toggle_main_window(app);
    }
}

/// Dispatch a tray menu event to the frontend.
///
/// Emits a `tray://event` event carrying the [`TrayEvent`] so the Vue UI can
/// react (open settings, start a session, etc.). The `Quit` event is handled
/// directly by `main.rs` (it exits the process) and is not emitted to the
/// frontend.
pub fn dispatch_event<R: Runtime>(app: &AppHandle<R>, event: TrayEvent) {
    use tauri::Emitter;
    match event {
        TrayEvent::ShowHide => crate::window::toggle_main_window(app),
        TrayEvent::NewSession | TrayEvent::Settings | TrayEvent::CheckUpdates => {
            let _ = app.emit("tray://event", event);
        }
        TrayEvent::Quit => {
            // Handled by main.rs before this function is reached.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_menu_ids_to_events() {
        assert_eq!(
            TrayEvent::from_menu_id(ID_SHOW_HIDE),
            Some(TrayEvent::ShowHide)
        );
        assert_eq!(
            TrayEvent::from_menu_id(ID_NEW_SESSION),
            Some(TrayEvent::NewSession)
        );
        assert_eq!(
            TrayEvent::from_menu_id(ID_SETTINGS),
            Some(TrayEvent::Settings)
        );
        assert_eq!(
            TrayEvent::from_menu_id(ID_CHECK_UPDATES),
            Some(TrayEvent::CheckUpdates)
        );
        assert_eq!(TrayEvent::from_menu_id(ID_QUIT), Some(TrayEvent::Quit));
        assert_eq!(TrayEvent::from_menu_id(ID_STATUS), None);
        assert_eq!(TrayEvent::from_menu_id("bogus"), None);
    }

    #[test]
    fn status_labels_are_distinct() {
        let labels = [
            TrayIconState::Starting.status_label(),
            TrayIconState::Running.status_label(),
            TrayIconState::Stopping.status_label(),
            TrayIconState::Stopped.status_label(),
            TrayIconState::Failed.status_label(),
        ];
        let unique: std::collections::HashSet<&str> = labels.iter().copied().collect();
        assert_eq!(unique.len(), labels.len());
    }
}
