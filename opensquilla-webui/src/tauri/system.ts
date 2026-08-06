/**
 * System adapter — Tauri-backed replacement for system/desktop RPCs.
 *
 * The existing frontend reaches desktop-native concerns through the Electron
 * preload bridge exposed on `window` (see `src/platform/desktop.ts`)
 * and through gateway RPCs for health/locale. Under Tauri these collapse into
 * `invoke('check_health')`, `invoke('get_locale')`, etc., plus the Tauri
 * shell plugin for `open_external` / window zoom.
 *
 * The Rust `tauri::command` names targeted here are:
 *
 *   invoke('check_health',  { })               → HealthReport
 *   invoke('get_locale',    { })               → string
 *   invoke('set_locale',    { locale })        → void
 *   invoke('check_updates', { })               → UpdateState
 *   invoke('open_external', { url })           → void
 *   invoke('pick_directory',{ initialPath? })  → { path: string } | null
 *   invoke('zoom_in' | 'zoom_out' | 'zoom_reset', { }) → void
 *
 * These mirror the P1 "桌面原生功能" surface in
 * `docs/tauri-migration-analysis.md`: window/tray/deep-link/secure-storage/
 * auto-update, exposed to WebUI via `tauri::command`.
 */

import { invoke } from './invoke'

/** Health report mirroring Rust `HealthReport` (subset of the doctor output). */
export interface HealthReport {
  /** Overall status. */
  status: 'healthy' | 'degraded' | 'unhealthy' | (string & {})
  /** Per-subsystem checks. */
  checks?: Array<{
    name: string
    status: 'ok' | 'warn' | 'fail' | (string & {})
    message?: string
    detail?: unknown
  }>
  /** Gateway/database/provider availability. */
  gateway?: { reachable?: boolean; latencyMs?: number }
  database?: { reachable?: boolean; migrationsApplied?: boolean }
  providers?: { configured?: number; reachable?: number }
  /** ISO timestamp of the check. */
  checkedAt?: string
  checked_at?: string
  [key: string]: unknown
}

/** Update state mirroring Rust `UpdateState`. Reuses the desktop platform shape. */
export interface UpdateState {
  status: 'idle' | 'checking' | 'available' | 'downloading' | 'downloaded' | 'not-available' | 'error' | 'applying' | (string & {})
  currentVersion: string
  current_version?: string
  latestVersion: string | null
  latest_version?: string | null
  progress: number | null
  checkedAt: string | null
  checked_at?: string | null
  error: string | null
  errorCode?: string | null
  error_code?: string | null
  releaseUrl?: string | null
  release_url?: string | null
  canCheck?: boolean
  can_check?: boolean
  canNativeInstall?: boolean
  can_native_install?: boolean
  [key: string]: unknown
}

/**
 * Check system health. Mirrors the `doctor` / health RPC.
 * Rust: `check_health() -> Result<HealthReport, IpcError>`.
 */
export async function checkHealth(): Promise<HealthReport> {
  return invoke<HealthReport>('check_health')
}

/**
 * Get the host OS locale (BCP-47), e.g. `en-US`, `zh-CN`. Used to seed the
 * initial UI language on first run. Mirrors the Electron
 * `getOsLocale()` bridge. Rust: `get_locale() -> Result<String, IpcError>`.
 */
export async function getLocale(): Promise<string> {
  return invoke<string>('get_locale')
}

/**
 * Persist the UI locale choice. The Rust side stores it in the app config and
 * the frontend applies it reactively through vue-i18n.
 * Rust: `set_locale(locale: String) -> Result<(), IpcError>`.
 */
export async function setLocale(locale: string): Promise<void> {
  return invoke<void>('set_locale', { locale })
}

/**
 * Check for app updates. Mirrors the Electron `checkForUpdates()` bridge.
 * Rust: `check_updates() -> Result<UpdateState, IpcError>`.
 */
export async function checkUpdates(): Promise<UpdateState> {
  return invoke<UpdateState>('check_updates')
}

/**
 * Download the available update, if any. Mirrors `downloadUpdate()`.
 * Rust: `download_update() -> Result<UpdateState, IpcError>`.
 */
export async function downloadUpdate(): Promise<UpdateState> {
  return invoke<UpdateState>('download_update')
}

/**
 * Relaunch the app to apply a downloaded update. Mirrors `relaunchToUpdate()`.
 * Rust: `relaunch_to_update() -> Result<UpdateState, IpcError>`.
 */
export async function relaunchToUpdate(): Promise<UpdateState> {
  return invoke<UpdateState>('relaunch_to_update')
}

/**
 * Open a URL in the user's default external application (browser, mail, etc.).
 * Mirrors the Tauri shell `open` plugin, wrapped as a command so the renderer
 * doesn't depend on `@tauri-apps/plugin-shell` directly.
 * Rust: `open_external(url: String) -> Result<(), IpcError>`.
 */
export async function openExternal(url: string): Promise<void> {
  return invoke<void>('open_external', { url })
}

/** Result of a native directory picker invocation. */
export interface DirectoryPickResult {
  path: string
}

/**
 * Open the native directory picker and return the chosen path, or `null` if
 * the user cancelled. Mirrors the Electron `chooseProjectDirectory()` bridge
 * (see `src/platform/desktop.ts`).
 * Rust: `pick_directory(initial_path: Option<String>) -> Result<Option<String>, IpcError>`.
 */
export async function pickDirectory(
  initialPath?: string,
): Promise<DirectoryPickResult | null> {
  const path = await invoke<string | null>('pick_directory', {
    initialPath,
  })
  return path ? { path } : null
}

/**
 * Zoom the main window in by one step. Mirrors the Electron zoom hotkeys.
 * Rust: `zoom_in() -> Result<(), IpcError>`.
 */
export async function zoomIn(): Promise<void> {
  return invoke<void>('zoom_in')
}

/**
 * Zoom the main window out by one step.
 * Rust: `zoom_out() -> Result<(), IpcError>`.
 */
export async function zoomOut(): Promise<void> {
  return invoke<void>('zoom_out')
}

/**
 * Reset the main window zoom to 100%.
 * Rust: `zoom_reset() -> Result<(), IpcError>`.
 */
export async function zoomReset(): Promise<void> {
  return invoke<void>('zoom_reset')
}
