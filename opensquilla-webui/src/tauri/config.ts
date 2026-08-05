/**
 * Config adapter — Tauri-backed replacement for the config RPCs.
 *
 * The existing frontend reads/writes configuration through the gateway RPCs
 * `config.get`, `config.effective`, `config.patch`, and `config.patch.safe`
 * (see `src/composables/setup/useSetupCatalog.ts` and
 * `src/composables/chat/useChatFeatureToggles.ts`). Under Tauri these become
 * `invoke('get_config')`, `invoke('set_config')`, etc.
 *
 * The Rust `tauri::command` names targeted here are:
 *
 *   invoke('get_config',  { })                       → ConfigSnapshot
 *   invoke('set_config',  { key, value })            → ConfigMutationResult
 *   invoke('patch_config',{ patches, safe })         → ConfigMutationResult
 *   invoke('list_config', { })                       → ConfigEntry[]
 *   invoke('get_config_effective', { })              → EffectiveConfig
 *
 * The config model is a 30-field serde struct on the Rust side (the migration
 * analysis notes "30+ 子配置"); here we model the wire shape as a string-keyed
 * record plus typed helpers for the commonly accessed fields, mirroring how
 * the renderer already treats config as a `Record<string, unknown>` in
 * `useChatFeatureToggles` and `useSetupCatalog`.
 */

import { invoke } from './invoke'

/**
 * A configuration snapshot. The Rust runtime serializes the full `Config`
 * struct (Pydantic `config.py` → serde on the Rust side) as a flat
 * string-keyed map of dotted paths, e.g. `provider.default`, `search.provider`.
 * The renderer treats this as an opaque record and reads individual fields.
 */
export type ConfigSnapshot = Record<string, unknown>

/** The effective (resolved + defaulted) config, mirroring `config.effective`. */
export interface EffectiveConfig {
  fields: Record<string, unknown>
  [key: string]: unknown
}

/** One patch applied to the config tree. Mirrors Rust `ConfigPatch`. */
export interface ConfigPatch {
  /** Dotted path, e.g. `provider.default` or `memory.enabled`. */
  path: string
  /** The value to set. `null` deletes the key. */
  value: unknown
  /** Optional: if true, this patch is applied without restarting the gateway. */
  hot?: boolean
}

/** Result of a mutating config command. Mirrors Rust `ConfigMutationResult`. */
export interface ConfigMutationResult {
  /** Whether the change requires a gateway/runtime restart to take effect. */
  restartRequired?: boolean
  restart_required?: boolean
  /** Whether validation passed (safe-mode only; unsafe always applies). */
  applied?: boolean
  /** Validation errors, keyed by patch path. */
  errors?: Record<string, string>
  /** The resulting snapshot after the mutation. */
  config?: ConfigSnapshot
  [key: string]: unknown
}

/** A single config entry returned by `list_config`. Mirrors Rust `ConfigEntry`. */
export interface ConfigEntry {
  key: string
  value: unknown
  /** Whether the value came from the default layer vs. user override. */
  source?: 'default' | 'user' | 'env' | 'managed'
  /** Human-readable description. */
  description?: string
  /** Whether changing this key requires a restart. */
  restartRequired?: boolean
  restart_required?: boolean
}

/**
 * Get the full config snapshot. Mirrors `config.get` RPC.
 * Rust: `#[tauri::command] async fn get_config() -> Result<ConfigSnapshot, IpcError>`.
 */
export async function getConfig(): Promise<ConfigSnapshot> {
  return invoke<ConfigSnapshot>('get_config')
}

/**
 * Get the effective (resolved) config. Mirrors `config.effective` RPC.
 * Rust: `get_config_effective() -> Result<EffectiveConfig, IpcError>`.
 */
export async function getEffectiveConfig(): Promise<EffectiveConfig> {
  return invoke<EffectiveConfig>('get_config_effective')
}

/**
 * Set a single config key. Mirrors the conceptual `config.set` (the legacy
 * gateway exposed this through `config.patch` with a single-element patch
 * list; Tauri exposes a direct `set_config` command).
 *
 * Rust: `set_config(key: String, value: serde_json::Value) -> Result<ConfigMutationResult, IpcError>`.
 */
export async function setConfig(
  key: string,
  value: unknown,
): Promise<ConfigMutationResult> {
  return invoke<ConfigMutationResult>('set_config', { key, value })
}

/**
 * Apply a batch of config patches. Mirrors the `config.patch` RPC.
 *
 * @param patches  - Patches to apply.
 * @param safe     - When true, validate before applying and roll back on
 *                   error (mirrors `config.patch.safe`); when false, apply
 *                   unconditionally (mirrors `config.patch`).
 */
export async function patchConfig(
  patches: ConfigPatch[],
  safe = true,
): Promise<ConfigMutationResult> {
  return invoke<ConfigMutationResult>('patch_config', { patches, safe })
}

/**
 * Apply a single patch (convenience for callers that used the single-patch
 * `config.patch` overload).
 */
export async function patchConfigSingle(
  path: string,
  value: unknown,
  safe = true,
): Promise<ConfigMutationResult> {
  return patchConfig([{ path, value }], safe)
}

/**
 * List all config entries with their source and metadata. Mirrors the
 * conceptual `config.list` (new under Tauri; the legacy gateway did not
 * expose a flat list, but the Rust runtime does for the settings UI).
 *
 * Rust: `list_config() -> Result<Vec<ConfigEntry>, IpcError>`.
 */
export async function listConfig(): Promise<ConfigEntry[]> {
  return invoke<ConfigEntry[]>('list_config')
}

/**
 * Reset the config to defaults. Mirrors the conceptual `config.reset`.
 * Rust: `reset_config() -> Result<ConfigMutationResult, IpcError>`.
 */
export async function resetConfig(): Promise<ConfigMutationResult> {
  return invoke<ConfigMutationResult>('reset_config')
}
