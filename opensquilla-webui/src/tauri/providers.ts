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
 * The Rust `tauri::command` names targeted here are:
 *
 *   invoke('list_providers',  { })                       → ProviderInfo[]
 *   invoke('list_models',     { providerId? })           → ModelInfo[]
 *   invoke('get_provider_status', { id })                → ProviderStatus
 *
 * These mirror the Rust `registry::ProviderSpec` (50+ declarative providers)
 * and `model_catalog` types documented in
 * `docs/tauri-migration-analysis.md` §2.2. snake_case fields are kept
 * alongside camelCase aliases because the Rust serde structs emit snake_case
 * by default while the renderer was written against the gateway's mixed
 * conventions.
 */

import { invoke } from './invoke'

/** A registered LLM provider. Mirrors Rust `ProviderSpec` (subset). */
export interface ProviderInfo {
  /** Stable provider id, e.g. `openai`, `deepseek`, `anthropic`. */
  id: string
  providerId?: string
  provider_id?: string
  /** Display label. */
  label?: string
  name?: string
  /** Backend class: openai_compat / anthropic / openai_responses / ollama / ensemble. */
  backend?: string
  kind?: string
  /** Default base URL. */
  baseUrl?: string
  base_url?: string
  /** Whether an API key is required. */
  requiresApiKey?: boolean
  requires_api_key?: boolean
  /** Env var holding the API key, when applicable. */
  apiKeyEnv?: string
  api_key_env?: string
  /** Whether the operator has configured credentials for this provider. */
  apiKeyConfigured?: boolean
  api_key_configured?: boolean
  /** Default model id for this provider. */
  model?: string
  defaultModel?: string
  default_model?: string
  /** Note shown in the setup UI. */
  note?: string
  [key: string]: unknown
}

/** A model offered by a provider. Mirrors Rust `ModelInfo`. */
export interface ModelInfo {
  /** Model id, e.g. `gpt-4o`, `deepseek-chat`. */
  id: string
  modelId?: string
  model_id?: string
  /** Owning provider id. */
  providerId?: string
  provider_id?: string
  /** Display label. */
  label?: string
  name?: string
  /** Context window size in tokens. */
  contextWindow?: number
  context_window?: number
  /** Max output tokens. */
  maxOutput?: number
  max_output?: number
  /** Input price per 1M tokens (USD). */
  inputPricePerMillion?: number
  input_price_per_million?: number
  /** Output price per 1M tokens (USD). */
  outputPricePerMillion?: number
  output_price_per_million?: number
  /** Whether the model supports tool/function calling. */
  supportsTools?: boolean
  supports_tools?: boolean
  /** Whether the model supports vision/image input. */
  supportsVision?: boolean
  supports_vision?: boolean
  /** Whether the model emits reasoning tokens. */
  supportsReasoning?: boolean
  supports_reasoning?: boolean
  [key: string]: unknown
}

/** Live status of a provider's connectivity. Mirrors Rust `ProviderStatus`. */
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
 * List all registered providers. Mirrors the provider portion of
 * `onboarding.catalog`. Rust: `list_providers() -> Result<Vec<ProviderInfo>, IpcError>`.
 */
export async function listProviders(): Promise<ProviderInfo[]> {
  return invoke<ProviderInfo[]>('list_providers')
}

/**
 * List models for a provider (or all providers when `providerId` is omitted).
 * Rust: `list_models(provider_id: Option<String>) -> Result<Vec<ModelInfo>, IpcError>`.
 */
export async function listModels(
  providerId?: string,
): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>('list_models', {
    providerId,
  })
}

/**
 * Get the live status of a provider. Mirrors the `providers.status` RPC.
 * Rust: `get_provider_status(id: String) -> Result<ProviderStatus, IpcError>`.
 */
export async function getProviderStatus(
  id: string,
): Promise<ProviderStatus> {
  return invoke<ProviderStatus>('get_provider_status', { id })
}

/**
 * Get the live status of all providers. Convenience over `list_providers` +
 * `get_provider_status`; the Rust runtime exposes a batched command for this.
 * Rust: `get_all_provider_statuses() -> Result<Vec<ProviderStatus>, IpcError>`.
 */
export async function getAllProviderStatuses(): Promise<ProviderStatus[]> {
  return invoke<ProviderStatus[]>('get_all_provider_statuses')
}
