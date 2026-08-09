/**
 * System adapter — Tauri-backed replacement for system/desktop RPCs.
 *
 * The existing frontend reaches desktop-native concerns through the Electron
 * preload bridge exposed on `window` (see `src/platform/desktop.ts`)
 * and through gateway RPCs for health/locale. Under Tauri these collapse into
 * `invoke('health_check')`, `invoke('get_locale')`, etc.
 *
 * The Rust `tauri::command` names targeted here are (see
 * `src-tauri/src/agent_bridge.rs` + `src-tauri/src/commands.rs`):
 *
 *   invoke('health_check',    { })                       → HealthReport
 *   invoke('get_locale',      { })                       → LocaleInfo
 *   invoke('set_locale',      { input: { locale } })     → { locale, saved }
 *   invoke('check_updates',   { })                       → UpdateInfo
 *   invoke('install_update',  { })                       → JsonValue
 *   invoke('open_external',   { target })                → void
 *   invoke('pick_directory',  { initialPath? })          → string | null
 *   invoke('zoom_in' | 'zoom_out' | 'zoom_reset', { })   → f64
 */

import { invoke } from './invoke'

/** One component's health status. Mirrors Rust `HealthComponent`. */
export interface HealthComponent {
  name: string
  status: 'healthy' | 'degraded' | 'unhealthy' | (string & {})
  description: string
  latencyMs: number
  details: Record<string, string>
}

/** A health issue found during the check. Mirrors Rust `HealthIssue`. */
export interface HealthIssue {
  component: string
  severity: string
  message: string
  suggestion?: string | null
}

/** Health report mirroring Rust `HealthReport` (agent_bridge.rs). */
export interface HealthReport {
  /** Overall status. */
  status: 'healthy' | 'degraded' | 'unhealthy' | (string & {})
  /** Seconds the runtime has been up. */
  uptimeSeconds: number
  /** ISO timestamp of the check. */
  timestamp: string
  /** Per-subsystem component results. */
  components: HealthComponent[]
  /** Issues found during the check. */
  issues: HealthIssue[]
  /** Whether the in-process gateway is running. */
  gatewayRunning: boolean
  gatewayUrl?: string | null
  [key: string]: unknown
}

/** Locale info returned by `get_locale`. Mirrors Rust `LocaleInfo` (commands.rs). */
export interface LocaleInfo {
  /** Effective locale (override or OS-detected), e.g. `en-US`, `zh-Hans`. */
  locale: string
  /** OS-detected locale (BCP-47). */
  detected: string
  /** Whether the user has persisted a locale override. */
  overridden: boolean
  /** Locales bundled with the app. */
  bundled: string[]
}

/** Update info returned by `check_updates`. Mirrors Rust `UpdateInfo`. */
export interface UpdateInfo {
  available: boolean
  version?: string | null
  releaseUrl?: string | null
  body?: string | null
}

/**
 * Check system health. Rust: `health_check() -> Result<HealthReport, TauriError>`.
 */
export async function checkHealth(): Promise<HealthReport> {
  return invoke<HealthReport>('health_check', {})
}

/**
 * Get the effective locale (override or OS-detected) plus the bundled set.
 * Rust: `get_locale() -> Result<LocaleInfo, TauriError>`.
 */
export async function getLocale(): Promise<LocaleInfo> {
  return invoke<LocaleInfo>('get_locale', {})
}

/**
 * Persist a user locale override. Rust:
 * `set_locale(input: SetLocaleInput) -> Result<JsonValue, TauriError>` —
 * the input is wrapped in `{ input: { locale } }`.
 */
export async function setLocale(
  locale: string,
): Promise<{ locale: string; saved: boolean }> {
  return invoke<{ locale: string; saved: boolean }>('set_locale', {
    input: { locale },
  })
}

/**
 * Check for app updates. Rust: `check_updates() -> Result<UpdateInfo, TauriError>`.
 */
export async function checkUpdates(): Promise<UpdateInfo> {
  return invoke<UpdateInfo>('check_updates', {})
}

/**
 * Download and install a pending update, then restart the app. Rust:
 * `install_update() -> Result<JsonValue, TauriError>`.
 */
export async function installUpdate(): Promise<{
  installed: boolean
  reason?: string
}> {
  return invoke<{ installed: boolean; reason?: string }>('install_update', {})
}

/**
 * Open a URL or path in the user's default external application. Rust:
 * `open_external(target: String) -> Result<(), TauriError>` — the argument
 * key is `target`.
 */
export async function openExternal(target: string): Promise<void> {
  return invoke<void>('open_external', { target })
}

/** Result of a native directory picker invocation. */
export interface DirectoryPickResult {
  path: string
}

/**
 * Open the native directory picker and return the chosen path, or `null` if
 * the user cancelled. Rust:
 * `pick_directory(initial_path: Option<String>) -> Result<Option<String>, TauriError>`.
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
 * Zoom the main window in by one step. Rust: `zoom_in() -> Result<f64, TauriError>`.
 */
export async function zoomIn(): Promise<number> {
  return invoke<number>('zoom_in', {})
}

/**
 * Zoom the main window out by one step. Rust: `zoom_out() -> Result<f64, TauriError>`.
 */
export async function zoomOut(): Promise<number> {
  return invoke<number>('zoom_out', {})
}

/**
 * Reset the main window zoom to 100%. Rust: `zoom_reset() -> Result<f64, TauriError>`.
 */
export async function zoomReset(): Promise<number> {
  return invoke<number>('zoom_reset', {})
}
