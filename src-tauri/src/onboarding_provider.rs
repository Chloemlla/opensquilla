//! Onboarding provider + LLM profile write commands (S4 write path).
//!
//! Implements the `onboarding.provider.*`, `onboarding.llmProfile.*` and
//! `onboarding.capability.reset` RPC methods over the Tauri bridge so the
//! WebUI setup wizard can persist provider/profile configuration. Mirrors the
//! Python `opensquilla.onboarding.mutations` semantics in minimal Rust:
//!
//! - `llm` is the single source of truth for the primary provider. The engine
//!   re-reads it per message (D2), so no runtime refresh is required.
//! - `Config.providers` is kept consistent as a best-effort upsert so the
//!   legacy `list_providers` / `resolve_provider` surfaces stay coherent.
//! - Persist first (`Config::save`), then hot-apply in memory. A save failure
//!   leaves the live config untouched.

use crate::error::{TauriError, TauriResult};
use crate::state::AppState;
use opensquilla_core::config::{
    AudioConfig, Config, ImageGenerationConfig, LlmConfig, LlmProfile, MemoryConfig,
    MemoryEmbeddingConfig, ProviderConfig,
};
use opensquilla_provider::registry::{AuthScheme, ProviderSpec, ProviderSpecTable};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tauri::State;
use tracing::warn;

/// Round-tripped secret mask echoed back to a write surface.
const REDACTED: &str = "***";

/// `DEFAULT_SEARCH_MAX_RESULTS` from the Python search config.
const DEFAULT_SEARCH_MAX_RESULTS: u32 = 10;

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

/// Uniform write response for every onboarding mutation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnboardingWriteResponse {
    pub changed: bool,
    pub restart_required: bool,
    pub config_path: Option<String>,
    pub entry: serde_json::Value,
    pub warnings: Vec<String>,
}

/// `onboarding.provider.configure` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfigureRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub preserve_api_key: bool,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
    pub preset_id: Option<String>,
    pub router_action: Option<String>,
    pub image_generation_intent: Option<String>,
}

/// Request for the single-argument provider/profile credential methods.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderCredentialRequest {
    pub provider_id: String,
}

/// `onboarding.provider.credential.reveal` response.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderCredentialRevealResponse {
    pub ok: bool,
    pub provider: String,
    pub source: String,
    pub env_key: String,
    pub api_key: String,
}

/// `onboarding.llmProfile.upsert` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileUpsertRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_key_env_pool: Option<Vec<String>>,
    #[serde(default)]
    pub keep_current_secret: bool,
    #[serde(default)]
    pub preserve_api_key: bool,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
}

/// `onboarding.llmProfile.activate` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileActivateRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub router_action: Option<String>,
    pub image_generation_intent: Option<String>,
}

/// `onboarding.llmProfile.active.remove` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileActiveRemoveRequest {
    pub provider_id: String,
    pub replacement_provider_id: String,
    pub replacement_model: Option<String>,
    pub router_action: Option<String>,
    pub image_generation_intent: Option<String>,
}

/// `onboarding.capability.reset` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityResetRequest {
    pub capability_id: String,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn config_path() -> Option<String> {
    Config::discover_path().ok().map(|p| p.to_string_lossy().to_string())
}

fn env_var_set(name: &str) -> bool {
    !name.is_empty() && std::env::var_os(name).map(|v| !v.is_empty()).unwrap_or(false)
}

/// True when a secret value is a round-tripped redaction mask (all asterisks).
fn is_redacted_secret_sentinel(value: &str) -> bool {
    let text = value.trim();
    !text.is_empty() && text.chars().all(|c| c == '*')
}

fn provider_spec(provider_id: &str) -> Option<ProviderSpec> {
    ProviderSpecTable::get(provider_id)
}

fn requires_api_key(spec: &ProviderSpec) -> bool {
    spec.auth != AuthScheme::None
}

/// Provider-id -> registry environment-variable name (Python `registry.py`).
fn provider_env_key(provider_id: &str) -> String {
    match provider_id {
        "openrouter" => "OPENROUTER_API_KEY",
        "openai" | "openai_responses" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        "gemini" => "GEMINI_API_KEY",
        "dashscope" => "DASHSCOPE_API_KEY",
        "zhipu" => "ZAI_API_KEY",
        "moonshot" => "MOONSHOT_API_KEY",
        "minimax" | "minimax_text" => "MINIMAX_API_KEY",
        "mistral" => "MISTRAL_API_KEY",
        "groq" => "GROQ_API_KEY",
        "azure" => "AZURE_OPENAI_API_KEY",
        "ollama" => "OLLAMA_API_KEY",
        "siliconflow" => "SILICONFLOW_API_KEY",
        "volcengine" | "volcengine_coding_plan" | "byteplus_coding_plan" => "VOLCENGINE_API_KEY",
        _ => "",
    }
    .to_string()
}

/// Map a provider id onto a `resolve_provider`-recognized backend type.
fn provider_type_for(provider_id: &str) -> String {
    match provider_id {
        "anthropic" => "anthropic",
        "ollama" => "ollama",
        _ => "openai_compat",
    }
    .to_string()
}

/// Set a dotted key, or remove it when the value is empty (keeps TOML clean).
fn set_or_remove(cfg: &mut Config, key: &str, value: &str) -> Result<(), TauriError> {
    if value.is_empty() {
        cfg.remove(key);
        Ok(())
    } else {
        cfg.set(key, value).map_err(|e| TauriError::bad_request(format!("failed to set {key}: {e}")))
    }
}

/// Case-insensitive lookup keys for one provider profile.
fn profile_storage_keys<'a>(profiles: &'a HashMap<String, LlmProfile>, provider: &str) -> Vec<&'a str> {
    profiles
        .keys()
        .filter(|k| k.eq_ignore_ascii_case(provider))
        .map(|k| k.as_str())
        .collect()
}

/// Look up a profile by case-insensitive provider id.
fn profile_lookup<'a>(config: &'a Config, provider: &str) -> Option<&'a LlmProfile> {
    let profiles = config.llm_profiles.as_ref()?;
    let key = profile_storage_keys(profiles, provider).into_iter().next()?;
    profiles.get(key)
}

/// Describe post-clear credential availability without exposing a value.
fn credential_clear_effective(config: &Config, provider_id: &str, active: bool) -> (bool, String, String) {
    if active {
        let Some(llm) = config.llm.as_ref() else {
            return (false, "none".to_string(), String::new());
        };
        let has_key = llm.api_key.as_deref().map(|k| !k.is_empty()).unwrap_or(false);
        let env_name = llm.api_key_env.clone().unwrap_or_default();
        let source = if has_key {
            "explicit"
        } else if !env_name.is_empty() {
            if env_var_set(&env_name) { "env" } else { "missing_env" }
        } else {
            "none"
        };
        let available = matches!(source, "explicit" | "env" | "not_required");
        (available, source.to_string(), env_name)
    } else {
        let Some(profile) = profile_lookup(config, provider_id) else {
            return (false, "none".to_string(), String::new());
        };
        let has_key = profile.api_key.as_deref().map(|k| !k.is_empty()).unwrap_or(false);
        let env_name = profile.api_key_env.clone().unwrap_or_default();
        let source = if has_key {
            "explicit"
        } else if !env_name.is_empty() {
            if env_var_set(&env_name) { "env" } else { "missing_env" }
        } else {
            // A provider registry default env var can still be active after a clear.
            let registry = provider_env_key(provider_id);
            if !registry.is_empty() && env_var_set(&registry) { "env" } else { "none" }
        };
        let available = matches!(source, "explicit" | "env" | "not_required");
        (available, source.to_string(), env_name)
    }
}

/// Upsert the primary into `Config.providers` so the legacy provider-list and
/// resolver surfaces see the same deployment the engine reads from `llm`.
/// An existing entry is updated in place (name-keyed); a brand-new primary is
/// inserted at the front so `providers.first()` stays the active provider (D3).
fn upsert_providers_entry(cfg: &mut Config, provider_id: &str, model: &str, base_url: &str, api_key: &str) {
    let provider_type = provider_type_for(provider_id);
    if let Some(existing) = cfg.providers.iter_mut().find(|p| p.name == provider_id) {
        existing.provider_type = provider_type;
        if !api_key.is_empty() {
            existing.api_key = Some(api_key.to_string());
        }
        if !base_url.is_empty() {
            existing.base_url = Some(base_url.to_string());
        }
        if !model.is_empty() {
            if !existing.models.contains(&model.to_string()) {
                existing.models.push(model.to_string());
            }
            existing.default_model = Some(model.to_string());
        }
        return;
    }
    cfg.providers.insert(0, ProviderConfig {
        name: provider_id.to_string(),
        provider_type,
        api_key: if api_key.is_empty() { None } else { Some(api_key.to_string()) },
        base_url: if base_url.is_empty() { None } else { Some(base_url.to_string()) },
        models: if model.is_empty() { Vec::new() } else { vec![model.to_string()] },
        default_model: if model.is_empty() { None } else { Some(model.to_string()) },
        max_retries: 3,
        timeout_secs: 120,
    });
}

/// Apply the Router ownership contract on a primary-provider switch.
///
/// Rust `SquillaRouterConfig` has no inline tier ladder, so managed presets
/// only update `preset_binding`/`tier_profile`; tier-level conflict detection
/// is a documented deviation from the Python `_apply_primary_provider_router_policy`.
fn apply_router_primary_policy(cfg: &mut Config, provider: &str, router_action: &str) {
    match router_action {
        "disable" => {
            let _ = cfg.set_value("squilla_router.enabled", &serde_json::Value::Bool(false));
            let _ = cfg.set("squilla_router.preset_binding", "custom");
        }
        "enable_cross_provider" => {
            let _ = cfg.set_value(
                "squilla_router.cross_provider_tiers",
                &serde_json::Value::Bool(true),
            );
            let _ = cfg.set("squilla_router.preset_binding", "custom");
        }
        "use_recommended" => {
            let _ = cfg.set("squilla_router.preset_binding", "follow_primary");
            let _ = cfg.set("squilla_router.tier_profile", provider);
        }
        _ => {
            let binding = cfg
                .squilla_router
                .as_ref()
                .and_then(|r| r.preset_binding.clone())
                .unwrap_or_default();
            if binding == "follow_primary" {
                let _ = cfg.set("squilla_router.tier_profile", provider);
            }
        }
    }
}

/// Apply an explicit OpenRouter image-default intent (Python
/// `apply_image_generation_intent`). Only the official, unowned OpenRouter
/// route is materialized as `follow_llm`; anything else is preserved.
fn apply_image_generation_intent_minimal(
    cfg: &mut Config,
    provider: &str,
    intent: &str,
) -> Option<serde_json::Value> {
    if intent != "enable_provider_default" || provider != "openrouter" {
        return None;
    }
    let image_owned = match cfg.image_generation.as_ref() {
        Some(ig) => !(ig.binding == "follow_llm" && ig.enabled),
        None => false,
    };
    if image_owned {
        return None;
    }
    let _ = cfg.set_value("image_generation.enabled", &serde_json::Value::Bool(true));
    let _ = cfg.set("image_generation.binding", "follow_llm");
    let _ = cfg.set("image_generation.primary", "openrouter/gpt-image-1");
    let _ = cfg.set_value("image_generation.fallbacks", &serde_json::json!([]));
    Some(serde_json::json!({
        "applied": true,
        "reason": "enabled_provider_default",
        "binding": "follow_llm",
        "primary": "openrouter/gpt-image-1",
    }))
}

// ---------------------------------------------------------------------------
// onboarding.provider.configure
// ---------------------------------------------------------------------------

/// `onboarding.provider.configure` — save the active LLM provider.
#[tauri::command]
pub async fn onboarding_provider_configure(
    state: State<'_, AppState>,
    request: ProviderConfigureRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider_id = request.provider_id.trim().to_lowercase();
    if provider_id.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = provider_spec(&provider_id);
    let requires_key = spec.as_ref().map(requires_api_key).unwrap_or(false);

    let (effective_model, effective_base_url, effective_api_key, effective_api_key_env, effective_proxy, stored_provider_routing) = {
        let cfg = state.config().await;
        let llm = cfg.llm.as_ref();
        let same_provider = llm
            .map(|l| l.provider.trim().to_lowercase() == provider_id)
            .unwrap_or(false);

        // model: explicit wins; blank keeps stored (same provider) or registry default.
        let model_param = request.model.as_deref().unwrap_or("").trim().to_string();
        let effective_model = if !model_param.is_empty() {
            model_param
        } else if same_provider {
            llm.and_then(|l| if l.model.trim().is_empty() { None } else { Some(l.model.clone()) })
                .unwrap_or_else(|| {
                    spec.as_ref()
                        .map(|s| s.default_model.to_string())
                        .unwrap_or_else(|| LlmConfig::default().model)
                })
        } else {
            spec.as_ref()
                .map(|s| s.default_model.to_string())
                .unwrap_or_else(|| LlmConfig::default().model)
        };
        if effective_model.trim().is_empty() {
            return Err(TauriError::bad_request("model is required"));
        }

        // base_url: explicit wins; blank keeps stored (same provider) or registry default.
        let base_url_param = request.base_url.as_deref().unwrap_or("").trim().to_string();
        let effective_base_url = if !base_url_param.is_empty() {
            base_url_param
        } else if same_provider {
            llm.and_then(|l| if l.base_url.trim().is_empty() { None } else { Some(l.base_url.clone()) })
                .unwrap_or_else(|| {
                    spec.as_ref()
                        .map(|s| s.api_base.to_string())
                        .unwrap_or_else(|| LlmConfig::default().base_url)
                })
        } else {
            spec.as_ref()
                .map(|s| s.api_base.to_string())
                .unwrap_or_else(|| LlmConfig::default().base_url)
        };

        // proxy: None keeps stored only on a same-provider re-save (a provider
        // switch never carries the old proxy); an explicit value, even empty, wins.
        let effective_proxy = match request.proxy.as_deref() {
            Some(p) => p.trim().to_string(),
            None => {
                if same_provider {
                    llm.and_then(|l| l.proxy.clone()).unwrap_or_default()
                } else {
                    String::new()
                }
            }
        };

        // credentials
        let api_key_param = request.api_key.as_deref().unwrap_or("").trim().to_string();
        let api_key_env_param = request.api_key_env.as_deref().unwrap_or("").trim().to_string();
        let api_key_is_mask = request.api_key.is_some() && is_redacted_secret_sentinel(&api_key_param);
        let preserve_key = request.preserve_api_key || api_key_is_mask;

        let mut effective_api_key = String::new();
        let mut effective_api_key_env = String::new();
        if api_key_is_mask {
            // Round-tripped mask: keep the stored key.
            if let Some(l) = llm {
                effective_api_key = l.api_key.clone().unwrap_or_default();
            }
        } else if !api_key_param.is_empty() {
            effective_api_key = api_key_param;
        } else if !api_key_env_param.is_empty() {
            effective_api_key_env = api_key_env_param;
        } else if same_provider {
            // Blank credential fields keep stored sources on a same-provider re-save.
            if let Some(l) = llm {
                if let Some(env) = l.api_key_env.as_deref().filter(|e| !e.is_empty()) {
                    effective_api_key_env = env.to_string();
                }
                if let Some(key) = l.api_key.as_deref().filter(|k| !k.is_empty()) {
                    if preserve_key || effective_api_key_env.is_empty() {
                        effective_api_key = key.to_string();
                    }
                }
            }
        }
        if !effective_api_key.is_empty() && !effective_api_key_env.is_empty() {
            effective_api_key_env.clear();
        }
        if requires_key && effective_api_key.is_empty() && effective_api_key_env.is_empty() {
            return Err(TauriError::bad_request(format!(
                "provider {provider_id:?} requires an api_key or api_key_env"
            )));
        }

        let stored_provider_routing = if same_provider {
            llm.map(|l| l.provider_routing.clone()).unwrap_or_default()
        } else {
            HashMap::new()
        };

        (
            effective_model,
            effective_base_url,
            effective_api_key,
            effective_api_key_env,
            effective_proxy,
            stored_provider_routing,
        )
    };

    let capability_changes;
    {
        let mut cfg = state.config_mut().await;
        set_or_remove(&mut cfg, "llm.provider", &provider_id)?;
        set_or_remove(&mut cfg, "llm.model", &effective_model)?;
        set_or_remove(&mut cfg, "llm.api_key", &effective_api_key)?;
        set_or_remove(&mut cfg, "llm.api_key_env", &effective_api_key_env)?;
        set_or_remove(&mut cfg, "llm.base_url", &effective_base_url)?;
        set_or_remove(&mut cfg, "llm.proxy", &effective_proxy)?;
        cfg.set_value("llm.provider_routing", &serde_json::json!(stored_provider_routing))
            .map_err(|e| TauriError::bad_request(format!("failed to set llm.provider_routing: {e}")))?;

        let router_action = request.router_action.as_deref().unwrap_or("preserve");
        let preset_requested = request
            .preset_id
            .as_deref()
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .is_some();
        match router_action {
            "disable" | "enable_cross_provider" | "use_recommended" => {
                apply_router_primary_policy(&mut cfg, &provider_id, router_action);
            }
            _ if preset_requested => {
                let _ = cfg.set("squilla_router.preset_binding", "follow_primary");
                let _ = cfg.set("squilla_router.tier_profile", &provider_id);
            }
            _ => {
                let binding = cfg
                    .squilla_router
                    .as_ref()
                    .and_then(|r| r.preset_binding.clone())
                    .unwrap_or_default();
                if binding == "follow_primary" {
                    let _ = cfg.set("squilla_router.tier_profile", &provider_id);
                }
            }
        }

        let intent = request.image_generation_intent.as_deref().unwrap_or("preserve");
        capability_changes = apply_image_generation_intent_minimal(&mut cfg, &provider_id, intent);

        upsert_providers_entry(&mut cfg, &provider_id, &effective_model, &effective_base_url, &effective_api_key);
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist provider configure");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }

    let api_key_source = if !effective_api_key.is_empty() {
        "explicit"
    } else if !effective_api_key_env.is_empty() {
        "env"
    } else {
        "none"
    };

    let mut entry = serde_json::json!({
        "provider": provider_id,
        "model": effective_model,
        "apiKey": if effective_api_key.is_empty() { "" } else { REDACTED },
        "apiKeyEnv": effective_api_key_env,
        "apiKeySource": api_key_source,
        "baseUrl": effective_base_url,
        "proxy": effective_proxy,
        "providerRouting": serde_json::json!(stored_provider_routing),
    });
    if let Some(changes) = capability_changes {
        entry["capabilityChanges"] = serde_json::json!({ "imageGeneration": changes });
    }

    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// onboarding.provider.credential.reveal / clear
// ---------------------------------------------------------------------------

/// `onboarding.provider.credential.reveal` — reveal the active provider key.
#[tauri::command]
pub async fn onboarding_provider_credential_reveal(
    state: State<'_, AppState>,
    request: ProviderCredentialRequest,
) -> TauriResult<ProviderCredentialRevealResponse> {
    let config = state.config().await;
    let Some(llm) = config.llm.as_ref() else {
        return Err(TauriError::bad_request("no active LLM provider configured"));
    };
    let active_provider = llm.provider.trim().to_lowercase();
    let requested = request.provider_id.trim().to_lowercase();
    if requested != active_provider {
        return Err(TauriError::bad_request(format!(
            "credential reveal only supports the active provider (expected {active_provider:?})"
        )));
    }
    let registry_env_key = provider_spec(&active_provider)
        .map(|s| provider_env_key(s.id))
        .unwrap_or_default();

    if let Some(key) = llm.api_key.as_deref() {
        if !key.is_empty() {
            let env_key = llm
                .api_key_env
                .clone()
                .filter(|e| !e.is_empty())
                .unwrap_or_else(|| registry_env_key.clone());
            return Ok(ProviderCredentialRevealResponse {
                ok: true,
                provider: active_provider,
                source: "explicit".to_string(),
                env_key,
                api_key: key.to_string(),
            });
        }
    }
    for env_name in [
        llm.api_key_env.clone().unwrap_or_default(),
        registry_env_key,
    ] {
        if !env_name.is_empty() {
            if let Ok(value) = std::env::var(&env_name) {
                if !value.is_empty() {
                    return Ok(ProviderCredentialRevealResponse {
                        ok: true,
                        provider: active_provider,
                        source: "env".to_string(),
                        env_key: env_name,
                        api_key: value,
                    });
                }
            }
        }
    }
    Err(TauriError::bad_request(
        "no revealable credential is available for the active provider",
    ))
}

/// `onboarding.provider.credential.clear` — clear active-provider credentials.
#[tauri::command]
pub async fn onboarding_provider_credential_clear(
    state: State<'_, AppState>,
    request: ProviderCredentialRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    {
        let mut cfg = state.config_mut().await;
        let active = cfg.llm.as_ref().map(|l| l.provider.trim().to_lowercase()).unwrap_or_default();
        if provider != active {
            return Err(TauriError::bad_request(format!(
                "credential clear only supports the active provider (active: {active:?})"
            )));
        }
        cfg.remove("llm.api_key");
        cfg.remove("llm.api_key_env");
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist provider credential clear");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }
    let (available, source, env_key) = {
        let cfg = state.config().await;
        credential_clear_effective(&cfg, &provider, true)
    };
    let entry = serde_json::json!({
        "provider": provider,
        "active": true,
        "storedCredentialsCleared": true,
        "credentialAvailable": available,
        "credentialSource": source,
        "credentialEnv": env_key,
        "externalCredentialActive": source == "env",
    });
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// onboarding.llmProfile.*
// ---------------------------------------------------------------------------

/// `onboarding.llmProfile.upsert` — create/update a non-primary profile.
#[tauri::command]
pub async fn onboarding_llm_profile_upsert(
    state: State<'_, AppState>,
    request: LlmProfileUpsertRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let preserve = request.keep_current_secret || request.preserve_api_key;
    // `api_key_env_pool` is validated as an array of names by serde; empty
    // entries are dropped during normalization below.

    let (effective_model, effective_base_url, effective_proxy, effective_api_key, effective_api_key_env, effective_pool) = {
        let cfg = state.config().await;
        let existing = profile_lookup(&cfg, &provider);
        let api_key_param = request.api_key.as_deref().unwrap_or("").trim().to_string();
        let api_key_is_mask = request.api_key.is_some() && is_redacted_secret_sentinel(&api_key_param);

        // None = keep current; Some = explicit (empty clears).
        let effective_model = match request.model.as_deref() {
            Some(m) => m.trim().to_string(),
            None => existing.as_ref().and_then(|p| p.model.clone()).unwrap_or_default(),
        };
        let effective_base_url = match request.base_url.as_deref() {
            Some(b) => b.trim().to_string(),
            None => existing.as_ref().and_then(|p| p.base_url.clone()).unwrap_or_default(),
        };
        let effective_proxy = match request.proxy.as_deref() {
            Some(p) => p.trim().to_string(),
            None => existing.as_ref().and_then(|p| p.proxy.clone()).unwrap_or_default(),
        };

        let api_key_env_param = request.api_key_env.as_deref().unwrap_or("").trim().to_string();
        let mut effective_api_key = String::new();
        let mut effective_api_key_env = String::new();
        if api_key_is_mask {
            effective_api_key = existing.as_ref().and_then(|p| p.api_key.clone()).unwrap_or_default();
        } else if !api_key_param.is_empty() {
            effective_api_key = api_key_param;
        } else if !api_key_env_param.is_empty() {
            effective_api_key_env = api_key_env_param;
        } else if let Some(existing) = existing.as_ref() {
            if preserve {
                effective_api_key = existing.api_key.clone().unwrap_or_default();
            }
            effective_api_key_env = existing.api_key_env.clone().unwrap_or_default();
        }

        let effective_pool = match request.api_key_env_pool.clone() {
            Some(pool) => {
                let mut seen = HashSet::new();
                let mut out = Vec::new();
                for name in pool {
                    let name = name.trim().to_string();
                    if !name.is_empty() && seen.insert(name.clone()) {
                        out.push(name);
                    }
                }
                out
            }
            None => existing.as_ref().map(|p| p.api_key_env_pool.clone()).unwrap_or_default(),
        };

        (
            effective_model,
            effective_base_url,
            effective_proxy,
            effective_api_key,
            effective_api_key_env,
            effective_pool,
        )
    };

    {
        let mut cfg = state.config_mut().await;
        // Remove historical case-variant keys, then write the canonical key.
        if let Some(profiles) = cfg.llm_profiles.as_mut() {
            let variants: Vec<String> = profiles
                .keys()
                .filter(|k| **k != *provider && k.eq_ignore_ascii_case(&provider))
                .cloned()
                .collect();
            for v in variants {
                profiles.remove(&v);
            }
        }
        let profile_json = serde_json::json!({
            "model": effective_model,
            "api_key": effective_api_key,
            "api_key_env": effective_api_key_env,
            "api_key_env_pool": effective_pool,
            "base_url": effective_base_url,
            "proxy": effective_proxy,
        });
        cfg.set_value(&format!("llm_profiles.{provider}"), &profile_json)
            .map_err(|e| TauriError::bad_request(format!("failed to set llm_profiles.{provider}: {e}")))?;
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist LLM profile upsert");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }

    let entry = serde_json::json!({
        "provider": provider,
        "model": effective_model,
        "apiKey": if effective_api_key.is_empty() { "" } else { REDACTED },
        "apiKeyEnv": effective_api_key_env,
        "apiKeyEnvPool": effective_pool,
        "baseUrl": effective_base_url,
        "proxy": effective_proxy,
    });
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

/// `onboarding.llmProfile.credential.clear` — clear stored profile credentials.
#[tauri::command]
pub async fn onboarding_llm_profile_credential_clear(
    state: State<'_, AppState>,
    request: ProviderCredentialRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    {
        let cfg = state.config().await;
        let active = cfg.llm.as_ref().map(|l| l.provider.trim().to_lowercase()).unwrap_or_default();
        if provider == active {
            return Err(TauriError::bad_request(
                "active provider credentials use the provider clear operation",
            ));
        }
        if profile_lookup(&cfg, &provider).is_none() {
            return Err(TauriError::not_found(format!("LLM profile {provider:?} does not exist")));
        }
    }
    {
        let mut cfg = state.config_mut().await;
        // Clear every case-variant key via typed mutation so a historical
        // malformed key cannot retain a duplicate secret.
        if let Some(profiles) = cfg.llm_profiles.as_mut() {
            let keys: Vec<String> = profile_storage_keys(profiles, &provider)
                .into_iter()
                .map(|k| k.to_string())
                .collect();
            for key in keys {
                if let Some(profile) = profiles.get_mut(&key) {
                    profile.api_key = None;
                    profile.api_key_env = None;
                    profile.api_key_env_pool.clear();
                }
            }
        }
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist LLM profile credential clear");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }
    let (available, source, env_key) = {
        let cfg = state.config().await;
        credential_clear_effective(&cfg, &provider, false)
    };
    let entry = serde_json::json!({
        "provider": provider,
        "active": false,
        "storedCredentialsCleared": true,
        "credentialAvailable": available,
        "credentialSource": source,
        "credentialEnv": env_key,
        "externalCredentialActive": source == "env",
    });
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

/// `onboarding.llmProfile.remove` — remove an unused provider profile.
#[tauri::command]
pub async fn onboarding_llm_profile_remove(
    state: State<'_, AppState>,
    request: ProviderCredentialRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    {
        let cfg = state.config().await;
        if profile_lookup(&cfg, &provider).is_none() {
            return Err(TauriError::not_found(format!("LLM profile {provider:?} does not exist")));
        }
        if let Some(ensemble) = cfg.llm_ensemble.as_ref() {
            for (index, candidate) in ensemble.candidates.iter().enumerate() {
                if candidate.provider.trim().to_lowercase() == provider {
                    return Err(TauriError::bad_request(format!(
                        "LLM profile {provider:?} is still referenced by llm_ensemble.candidates.{index}"
                    )));
                }
            }
        }
    }
    {
        let mut cfg = state.config_mut().await;
        if let Some(profiles) = cfg.llm_profiles.as_mut() {
            let keys: Vec<String> = profile_storage_keys(profiles, &provider)
                .into_iter()
                .map(|k| k.to_string())
                .collect();
            for key in keys {
                profiles.remove(&key);
            }
        }
        // Drop the section entirely when the last profile is removed so the
        // config file stays clean instead of persisting `llm_profiles = {}`.
        if cfg.llm_profiles.as_ref().map(|m| m.is_empty()).unwrap_or(false) {
            cfg.llm_profiles = None;
        }
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist LLM profile removal");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }
    let entry = serde_json::json!({ "provider": provider, "removed": true });
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

/// `onboarding.llmProfile.activate` — promote a stored profile to primary.
#[tauri::command]
pub async fn onboarding_llm_profile_activate(
    state: State<'_, AppState>,
    request: LlmProfileActivateRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = provider_spec(&provider);
    let entry_router_binding;
    let capability_changes;

    let (previous_provider, effective_model) = {
        let mut cfg = state.config_mut().await;
        let active = cfg.llm.as_ref().map(|l| l.provider.trim().to_lowercase()).unwrap_or_default();
        if provider == active {
            return Err(TauriError::bad_request(format!("provider {provider:?} is already active")));
        }

        let profile = {
            let profiles = cfg.llm_profiles.clone().unwrap_or_default();
            let key = profile_storage_keys(&profiles, &provider)
                .into_iter()
                .next()
                .map(|k| k.to_string());
            key.as_ref().and_then(|k| profiles.get(k).cloned())
        };
        let Some(profile) = profile else {
            return Err(TauriError::not_found(format!("LLM profile {provider:?} does not exist")));
        };
        if !profile.api_key_env_pool.is_empty() {
            return Err(TauriError::bad_request(
                "the primary provider does not support api_key_env_pool",
            ));
        }

        let effective_model = request
            .model
            .as_deref()
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .or_else(|| profile.model.clone())
            .filter(|m| !m.is_empty())
            .or_else(|| spec.as_ref().map(|s| s.default_model.to_string()))
            .unwrap_or_else(|| LlmConfig::default().model);

        // Demote the current primary into a stored profile, then remove the
        // promoted profile (and any case-variant duplicate keys).
        let mut profiles = cfg.llm_profiles.clone().unwrap_or_default();
        if let Some(llm) = cfg.llm.as_ref() {
            if !active.is_empty() {
                profiles.insert(
                    active.clone(),
                    LlmProfile {
                        model: Some(llm.model.clone()),
                        api_key: llm.api_key.clone(),
                        api_key_env: llm.api_key_env.clone(),
                        api_key_env_pool: Vec::new(),
                        base_url: Some(llm.base_url.clone()),
                        proxy: llm.proxy.clone(),
                    },
                );
            }
        }
        let variants: Vec<String> = profiles
            .keys()
            .filter(|k| k.eq_ignore_ascii_case(&provider))
            .cloned()
            .collect();
        for v in variants {
            profiles.remove(&v);
        }
        cfg.llm_profiles = Some(profiles);
        if cfg.llm_profiles.as_ref().map(|m| m.is_empty()).unwrap_or(false) {
            cfg.llm_profiles = None;
        }

        let api_key = profile.api_key.clone().unwrap_or_default();
        let api_key_env = profile.api_key_env.clone().unwrap_or_default();
        let base_url = profile
            .base_url
            .clone()
            .filter(|b| !b.is_empty())
            .or_else(|| spec.as_ref().map(|s| s.api_base.to_string()))
            .unwrap_or_else(|| LlmConfig::default().base_url);
        let proxy = profile.proxy.clone().unwrap_or_default();
        set_or_remove(&mut cfg, "llm.provider", &provider)?;
        set_or_remove(&mut cfg, "llm.model", &effective_model)?;
        set_or_remove(&mut cfg, "llm.api_key", &api_key)?;
        set_or_remove(&mut cfg, "llm.api_key_env", &api_key_env)?;
        set_or_remove(&mut cfg, "llm.base_url", &base_url)?;
        set_or_remove(&mut cfg, "llm.proxy", &proxy)?;

        let router_action = request.router_action.as_deref().unwrap_or("preserve");
        apply_router_primary_policy(&mut cfg, &provider, router_action);
        entry_router_binding = cfg
            .squilla_router
            .as_ref()
            .and_then(|r| r.preset_binding.clone())
            .filter(|b| b == "follow_primary" || b == "custom")
            .unwrap_or_else(|| "legacy".to_string());

        let intent = request.image_generation_intent.as_deref().unwrap_or("preserve");
        capability_changes = apply_image_generation_intent_minimal(&mut cfg, &provider, intent);

        upsert_providers_entry(&mut cfg, &provider, &effective_model, &base_url, &api_key);

        (active, effective_model)
    };
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist LLM profile activation");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }

    let mut entry = serde_json::json!({
        "provider": provider,
        "model": effective_model,
        "previousProvider": previous_provider,
        "active": true,
        "routerBinding": entry_router_binding,
    });
    if let Some(changes) = capability_changes {
        entry["capabilityChanges"] = serde_json::json!({ "imageGeneration": changes });
    }
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

/// `onboarding.llmProfile.active.remove` — atomically replace the primary.
#[tauri::command]
pub async fn onboarding_llm_profile_active_remove(
    state: State<'_, AppState>,
    request: LlmProfileActiveRemoveRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    let replacement = request.replacement_provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    if replacement.is_empty() || replacement == provider {
        return Err(TauriError::bad_request(
            "replacement provider must differ from the active provider",
        ));
    }
    let spec = provider_spec(&replacement);
    let capability_changes;

    let (removed_provider, active_provider, effective_model) = {
        let mut cfg = state.config_mut().await;
        let active = cfg.llm.as_ref().map(|l| l.provider.trim().to_lowercase()).unwrap_or_default();
        if provider != active {
            return Err(TauriError::bad_request(format!(
                "provider {provider:?} is not the active LLM provider (active: {active:?})"
            )));
        }

        let profile = {
            let profiles = cfg.llm_profiles.clone().unwrap_or_default();
            let key = profile_storage_keys(&profiles, &replacement)
                .into_iter()
                .next()
                .map(|k| k.to_string());
            key.as_ref().and_then(|k| profiles.get(k).cloned())
        };
        let Some(profile) = profile else {
            return Err(TauriError::not_found(format!("LLM profile {replacement:?} does not exist")));
        };
        if !profile.api_key_env_pool.is_empty() {
            return Err(TauriError::bad_request(
                "the primary provider does not support api_key_env_pool",
            ));
        }

        let effective_model = request
            .replacement_model
            .as_deref()
            .map(|m| m.trim().to_string())
            .filter(|m| !m.is_empty())
            .or_else(|| profile.model.clone())
            .filter(|m| !m.is_empty())
            .or_else(|| spec.as_ref().map(|s| s.default_model.to_string()))
            .unwrap_or_else(|| LlmConfig::default().model);

        // Promote the replacement; the removed active provider is demoted into
        // a profile only so any remaining route reference stays resolvable,
        // then removed (matching activate + remove_llm_profile).
        let mut profiles = cfg.llm_profiles.clone().unwrap_or_default();
        if let Some(llm) = cfg.llm.as_ref() {
            profiles.insert(
                active.clone(),
                LlmProfile {
                    model: Some(llm.model.clone()),
                    api_key: llm.api_key.clone(),
                    api_key_env: llm.api_key_env.clone(),
                    api_key_env_pool: Vec::new(),
                    base_url: Some(llm.base_url.clone()),
                    proxy: llm.proxy.clone(),
                },
            );
        }
        let variants: Vec<String> = profiles
            .keys()
            .filter(|k| k.eq_ignore_ascii_case(&replacement) || k.eq_ignore_ascii_case(&provider))
            .cloned()
            .collect();
        for v in variants {
            profiles.remove(&v);
        }
        cfg.llm_profiles = Some(profiles);
        if cfg.llm_profiles.as_ref().map(|m| m.is_empty()).unwrap_or(false) {
            cfg.llm_profiles = None;
        }

        let api_key = profile.api_key.clone().unwrap_or_default();
        let api_key_env = profile.api_key_env.clone().unwrap_or_default();
        let base_url = profile
            .base_url
            .clone()
            .filter(|b| !b.is_empty())
            .or_else(|| spec.as_ref().map(|s| s.api_base.to_string()))
            .unwrap_or_else(|| LlmConfig::default().base_url);
        let proxy = profile.proxy.clone().unwrap_or_default();
        set_or_remove(&mut cfg, "llm.provider", &replacement)?;
        set_or_remove(&mut cfg, "llm.model", &effective_model)?;
        set_or_remove(&mut cfg, "llm.api_key", &api_key)?;
        set_or_remove(&mut cfg, "llm.api_key_env", &api_key_env)?;
        set_or_remove(&mut cfg, "llm.base_url", &base_url)?;
        set_or_remove(&mut cfg, "llm.proxy", &proxy)?;

        let router_action = request.router_action.as_deref().unwrap_or("preserve");
        apply_router_primary_policy(&mut cfg, &replacement, router_action);

        let intent = request.image_generation_intent.as_deref().unwrap_or("preserve");
        capability_changes = apply_image_generation_intent_minimal(&mut cfg, &replacement, intent);

        upsert_providers_entry(&mut cfg, &replacement, &effective_model, &base_url, &api_key);

        (active, replacement, effective_model)
    };
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist active provider removal");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }

    let mut entry = serde_json::json!({
        "removedProvider": removed_provider,
        "removed": true,
        "activeProvider": active_provider,
        "activeModel": effective_model,
    });
    if let Some(changes) = capability_changes {
        entry["capabilityChanges"] = serde_json::json!({ "imageGeneration": changes });
    }
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required: false,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// onboarding.capability.reset
// ---------------------------------------------------------------------------

/// `onboarding.capability.reset` — restore one capability to its built-in state.
#[tauri::command]
pub async fn onboarding_capability_reset(
    state: State<'_, AppState>,
    request: CapabilityResetRequest,
) -> TauriResult<OnboardingWriteResponse> {
    let capability = request.capability_id.trim().to_lowercase();
    if capability.is_empty() {
        return Err(TauriError::bad_request("capabilityId is required"));
    }
    let mut restart_required = false;
    {
        let mut cfg = state.config_mut().await;
        match capability.as_str() {
            "search" => {
                cfg.search_provider = Some("duckduckgo".to_string());
                cfg.search_api_key = None;
                cfg.search_api_key_env = None;
                cfg.search_max_results = Some(DEFAULT_SEARCH_MAX_RESULTS);
                cfg.search_proxy = None;
                cfg.search_use_env_proxy = Some(false);
                cfg.search_fallback_policy = Some("off".to_string());
                cfg.search_diagnostics = Some(false);
            }
            "image_generation" => {
                cfg.image_generation = Some(ImageGenerationConfig::default());
            }
            "audio" => {
                cfg.audio = Some(AudioConfig::default());
            }
            "memory_embedding" => {
                let default_embedding = MemoryEmbeddingConfig::default();
                if let Some(memory) = cfg.memory.as_mut() {
                    memory.embedding = default_embedding;
                } else {
                    cfg.memory = Some(MemoryConfig {
                        embedding: default_embedding,
                        ..MemoryConfig::default()
                    });
                }
                restart_required = true;
            }
            other => {
                return Err(TauriError::bad_request(format!("unknown capability: {other:?}")));
            }
        }
    }
    {
        let cfg = state.config().await;
        if let Err(e) = cfg.save() {
            warn!(error = %e, "Failed to persist capability reset");
            return Err(TauriError::internal(format!("Failed to persist config: {e}")));
        }
    }
    let entry = serde_json::json!({
        "capabilityId": capability,
        "reset": true,
    });
    Ok(OnboardingWriteResponse {
        changed: true,
        restart_required,
        config_path: config_path(),
        entry,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configured_llm() -> Config {
        let mut config = Config::default();
        config.llm = Some(LlmConfig {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            api_key: Some("sk-test1234".to_string()),
            api_key_env: None,
            base_url: "https://api.openai.com/v1".to_string(),
            proxy: None,
            max_tokens: 0,
            context_window_tokens: 0,
            temperature: None,
            top_p: None,
            thinking: None,
            provider_request_proof_max_chars: 0,
            provider_routing: HashMap::new(),
        });
        config
    }

    #[test]
    fn redacted_sentinel_only_all_asterisks() {
        assert!(is_redacted_secret_sentinel("***"));
        assert!(is_redacted_secret_sentinel("*****"));
        assert!(!is_redacted_secret_sentinel(""));
        assert!(!is_redacted_secret_sentinel("sk-abc"));
        assert!(!is_redacted_secret_sentinel("***abc"));
    }

    #[test]
    fn provider_env_key_mapping() {
        assert_eq!(provider_env_key("openai"), "OPENAI_API_KEY");
        assert_eq!(provider_env_key("openrouter"), "OPENROUTER_API_KEY");
        assert_eq!(provider_env_key("ollama"), "OLLAMA_API_KEY");
        assert_eq!(provider_env_key("unknown_provider"), "");
    }

    #[test]
    fn provider_type_mapping() {
        assert_eq!(provider_type_for("anthropic"), "anthropic");
        assert_eq!(provider_type_for("ollama"), "ollama");
        assert_eq!(provider_type_for("deepseek"), "openai_compat");
        assert_eq!(provider_type_for("custom"), "openai_compat");
    }

    #[test]
    fn profile_storage_keys_is_case_insensitive() {
        let mut profiles = HashMap::new();
        profiles.insert(
            "OpenAI".to_string(),
            LlmProfile {
                model: Some("gpt-4o".to_string()),
                ..Default::default()
            },
        );
        let keys = profile_storage_keys(&profiles, "openai");
        assert_eq!(keys, vec!["OpenAI"]);
    }

    #[test]
    fn credential_clear_effective_reports_explicit() {
        let config = configured_llm();
        let (available, source, _env) = credential_clear_effective(&config, "openai", true);
        assert!(available);
        assert_eq!(source, "explicit");
    }

    #[test]
    fn credential_clear_effective_none_when_cleared() {
        let mut config = configured_llm();
        config.llm.as_mut().unwrap().api_key = None;
        config.llm.as_mut().unwrap().api_key_env = None;
        let (available, source, env) = credential_clear_effective(&config, "openai", true);
        assert!(!available);
        assert_eq!(source, "none");
        assert!(env.is_empty());
    }

    #[test]
    fn upsert_providers_entry_creates_and_updates() {
        let mut config = Config::default();
        upsert_providers_entry(&mut config, "openai", "gpt-4o", "https://api.openai.com/v1", "sk-1");
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].name, "openai");
        assert_eq!(config.providers[0].default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(config.providers[0].provider_type, "openai_compat");

        upsert_providers_entry(&mut config, "openai", "gpt-4o-mini", "https://api.openai.com/v1", "");
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].models.len(), 2);
        // Empty api_key on update keeps the stored key.
        assert_eq!(config.providers[0].api_key.as_deref(), Some("sk-1"));
    }

    #[test]
    fn upsert_providers_entry_moves_new_primary_to_front() {
        let mut config = Config::default();
        upsert_providers_entry(&mut config, "openai", "gpt-4o", "https://api.openai.com/v1", "sk-1");
        upsert_providers_entry(&mut config, "deepseek", "deepseek-chat", "https://api.deepseek.com/v1", "sk-2");
        // A brand-new primary becomes the default (first) entry (D3).
        assert_eq!(config.providers[0].name, "deepseek");
        assert_eq!(config.providers[0].default_model.as_deref(), Some("deepseek-chat"));
        assert_eq!(config.providers[1].name, "openai");
        // Re-save of the same provider updates in place without duplicating.
        upsert_providers_entry(&mut config, "deepseek", "deepseek-reasoner", "https://api.deepseek.com/v1", "sk-3");
        assert_eq!(config.providers.len(), 2);
        assert_eq!(config.providers[0].name, "deepseek");
        assert_eq!(config.providers[0].models, vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()]);
    }
}
