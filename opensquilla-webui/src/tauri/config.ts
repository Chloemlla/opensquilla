/**
 * Config adapter — Tauri-backed replacement for the config RPCs.
 *
 * The existing frontend reads/writes configuration through the gateway RPCs
 * `config.get`, `config.effective`, `config.patch`, and `config.patch.safe`
 * (see `src/composables/setup/useSetupCatalog.ts` and
 * `src/composables/chat/useChatFeatureToggles.ts`). Under Tauri these become
 * `invoke('get_config')`, `invoke('set_config')`, etc.
 *
 * The Rust `tauri::command` names targeted here are (see
 * `src-tauri/src/agent_bridge.rs` + the in-parallel new-command set):
 *
 *   invoke('get_config',          { })                      → { config }
 *   invoke('get_config_effective',{ })                      → { config }
 *   invoke('set_config',          { request: { key, value }}) → { key, value, status }
 *   invoke('patch_config',        { patches, safe })        → { key, value, status }
 *   invoke('list_config',         { })                      → JsonValue (flat map)
 *   invoke('reset_config',        { key? })                 → { ok, message? }
 *   invoke('get_config_value',    { key })                  → JsonValue
 *
 * `get_config` / `get_config_effective` wrap their payload in `{ config }`
 * (the `ConfigGetResponse` struct has no camelCase rename, so the field is
 * literally `config`); every adapter unwraps it.
 */

import { invoke } from './invoke'

/**
 * A configuration snapshot. The Rust runtime serializes the full `Config`
 * struct as a string-keyed object; the renderer treats this as an opaque
 * record and reads individual fields.
 */
export type ConfigSnapshot = Record<string, unknown>

/** One patch applied to the config tree. Mirrors the `patch_config` payload. */
export interface ConfigPatch {
  /** Dotted path, e.g. `provider.default` or `memory.enabled`. */
  key: string
  /** The value to set. */
  value: unknown
  /** Optional: if true, this patch is applied without restarting. */
  hot?: boolean
}

/** Result of `set_config` / `patch_config`. Mirrors Rust `ConfigSetResponse`. */
export interface ConfigSetResult {
  key: string
  value: unknown
  status: string
}

/** Result of `reset_config`. Mirrors Rust `OperationResult`. */
export interface ConfigResetResult {
  ok: boolean
  message?: string | null
}

/**
 * Get the full config snapshot. Rust: `get_config() -> Result<ConfigGetResponse, TauriError>`.
 * Unwraps the `{ config }` envelope.
 */
export async function getConfig(): Promise<ConfigSnapshot> {
  const result = await invoke<{ config: ConfigSnapshot }>('get_config', {})
  return result.config
}

/**
 * Get the effective (resolved + defaulted) config. Rust:
 * `get_config_effective() -> Result<ConfigGetResponse, TauriError>`.
 * Unwraps the `{ config }` envelope.
 */
export async function getEffectiveConfig(): Promise<ConfigSnapshot> {
  const result = await invoke<{ config: ConfigSnapshot }>('get_config_effective', {})
  return result.config
}

/**
 * Set a single config key. Rust:
 * `set_config(request: ConfigSetRequest) -> Result<ConfigSetResponse, TauriError>`.
 * The request is wrapped in `{ request: { key, value } }`.
 */
export async function setConfig(
  key: string,
  value: unknown,
): Promise<ConfigSetResult> {
  return invoke<ConfigSetResult>('set_config', { request: { key, value } })
}

/**
 * Apply a batch of config patches. Rust:
 * `patch_config(patches: Vec<ConfigPatch>, safe: Option<bool>) -> Result<ConfigSetResponse, TauriError>`.
 * Each `ConfigPatch` (`{ key, value }`) is sent under `{ patches, safe? }`.
 */
export async function patchConfig(
  patches: ConfigPatch[],
  safe = true,
): Promise<ConfigSetResult> {
  const wire = patches.map((p) => ({ key: p.key, value: p.value }))
  const args: Record<string, unknown> = { patches: wire }
  if (safe !== undefined) args.safe = safe
  return invoke<ConfigSetResult>('patch_config', args)
}

/**
 * Apply a single patch (convenience for callers that used the single-patch
 * `config.patch` overload).
 */
export async function patchConfigSingle(
  path: string,
  value: unknown,
  safe = true,
): Promise<ConfigSetResult> {
  return patchConfig([{ key: path, value }], safe)
}

/**
 * List all config values as a flat key → value map. Rust:
 * `list_config() -> Result<JsonValue, TauriError>`.
 */
export async function listConfig(): Promise<Record<string, unknown>> {
  return invoke<Record<string, unknown>>('list_config', {})
}

/**
 * Reset the config to defaults. Rust:
 * `reset_config(key: Option<String>) -> Result<OperationResult, TauriError>`.
 * With `key` present only that key is reset; otherwise the whole config.
 */
export async function resetConfig(key?: string): Promise<ConfigResetResult> {
  return invoke<ConfigResetResult>('reset_config', key !== undefined ? { key } : {})
}

/**
 * Read a single config value by key. Rust:
 * `get_config_value(key: String) -> Result<Option<JsonValue>, TauriError>`.
 */
export async function getConfigValue(key: string): Promise<unknown> {
  return invoke<unknown>('get_config_value', { key })
}
