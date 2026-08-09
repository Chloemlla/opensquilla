/**
 * Provider adapter — Tauri-backed replacement for the provider/model RPCs.
 *
 * The existing frontend discovers LLM providers and models through the
 * gateway RPCs `onboarding.catalog` (which embeds a provider list) and
 * `models.routing.get` (see `src/composables/setup/useSetupCatalog.ts` and
 * `src/composables/chat/useChatFeatureToggles.ts`). Under Tauri the Rust
 * runtime exposes dedicated commands for provider/model enumeration and
 * status.
 *
 * The Rust `tauri::command` names targeted here are (see
 * `src-tauri/src/agent_bridge.rs` + the in-parallel new-command set):
 *
 *   invoke('list_providers',   { })                  → { providers, defaultProvider?, count }
 *   invoke('list_models',      { })                  → { models, defaultModel?, count }
 *   invoke('get_provider_status', { providerId? })   → JsonValue
 *   invoke('get_all_provider_statuses', { })         → JsonValue
 *
 * `ProviderInfo` / `ModelInfo` mirror the camelCase serde structs
 * (`ProviderInfo`, `ModelInfoDto` in `src-tauri/src/ipc.rs`).
 */

import { invoke } from './invoke'

/** A registered LLM provider. Mirrors Rust `ProviderInfo` (ipc.rs). */
export interface ProviderInfo {
  /** Provider id, e.g. `openai`, `deepseek`, `anthropic`. */
  name: string
  /** Backend class: openai_compat / anthropic / ollama / ... */
  providerType: string
  provider_type?: string
  /** Default base URL. */
  baseUrl?: string | null
  base_url?: string | null
  /** Model ids offered by this provider. */
  models: string[]
  /** Default model id for this provider. */
  defaultModel?: string | null
  default_model?: string | null
  maxRetries: number
  timeoutSecs: number
  [key: string]: unknown
}

/** A model offered by a provider. Mirrors Rust `ModelInfoDto` (ipc.rs). */
export interface ModelInfo {
  /** Model id, e.g. `gpt-4o`, `deepseek-chat`. */
  id: string
  name: string
  /** Owning provider id. */
  provider: string
  /** Context window size in tokens. */
  contextWindow: number
  context_window?: number
  /** Max output tokens. */
  maxOutputTokens: number
  max_output_tokens?: number
  /** Model capabilities. */
  capabilities?: unknown
  /** Optional display label. */
  displayName?: string | null
  display_name?: string | null
  [key: string]: unknown
}

/** Live status of a provider's connectivity (Rust returns JsonValue). */
export interface ProviderStatus {
  /** Provider id this status describes. */
  id: string
  providerId?: string
  provider_id?: string
  /** Connectivity state. */
  status: 'online' | 'offline' | 'degraded' | 'unknown' | (string & {})
  /** Whether credentials are configured. */
  configured?: boolean
  /** Whether the provider is currently reachable. */
  reachable?: boolean
  /** Last checked at (epoch ms or ISO string). */
  checkedAt?: number | string
  checked_at?: number | string
  /** Latency of the last probe, in milliseconds. */
  latencyMs?: number
  latency_ms?: number
  /** Error message from the last probe, if any. */
  error?: string
  [key: string]: unknown
}

/**
 * List all registered providers. Rust: `list_providers() -> Result<ProviderListResponse, TauriError>` —
 * unwraps the `{ providers }` field.
 */
export async function listProviders(): Promise<ProviderInfo[]> {
  const result = await invoke<{ providers: ProviderInfo[] }>('list_providers', {})
  return result.providers
}

/**
 * List available models. Rust: `list_models() -> Result<ModelListResponse, TauriError>` —
 * unwraps the `{ models }` field. `providerId` is accepted for surface parity
 * but the current command enumerates all providers.
 */
export async function listModels(_providerId?: string): Promise<ModelInfo[]> {
  const result = await invoke<{ models: ModelInfo[] }>('list_models', {})
  return result.models
}

/**
 * Get the live status of a provider. Mirrors the `providers.status` RPC.
 * Rust: `get_provider_status(provider_id: Option<String>) -> Result<JsonValue, TauriError>`.
 */
export async function getProviderStatus(
  providerId?: string,
): Promise<ProviderStatus> {
  return invoke<ProviderStatus>('get_provider_status', {
    providerId,
  })
}

/**
 * Get the live status of all providers. Rust:
 * `get_all_provider_statuses() -> Result<JsonValue, TauriError>`.
 */
export async function getAllProviderStatuses(): Promise<ProviderStatus[]> {
  return invoke<ProviderStatus[]>('get_all_provider_statuses', {})
}
