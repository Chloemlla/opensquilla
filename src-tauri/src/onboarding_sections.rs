//! Onboarding section-configure commands (S4 write path).
//!
//! Implements the six `onboarding.*.configure` RPC methods over the Tauri
//! bridge so the WebUI setup wizard can persist router, ensemble, search,
//! image generation, memory embedding, and audio configuration. Every write
//! follows the same order as the Python gateway: mutate the single
//! source-of-truth [`Config`] held in `AppState`, persist via
//! [`Config::save`], then return the shared response shape (`changed` /
//! `restartRequired` / `configPath` / `entry` / `warnings`).

use crate::error::{TauriError, TauriResult};
use crate::state::AppState;
use opensquilla_core::config::{Config, MemoryEmbeddingConfig};
use serde::Serialize;
use tauri::State;

/// Redaction placeholder echoed in `entry` payloads for stored secrets.
const REDACTED: &str = "***";
const DEFAULT_SEARCH_MAX_RESULTS: u32 = 10;
const MAX_SEARCH_RESULTS: u32 = 20;
const DEFAULT_REMOTE_EMBEDDING_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_OLLAMA_EMBEDDING_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_AUDIO_BASE_URL: &str = "https://api.elevenlabs.io";

/// Shared response shape for every onboarding section-configure write.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnboardingWriteResponse {
    changed: bool,
    restart_required: bool,
    config_path: Option<String>,
    entry: serde_json::Value,
    warnings: Vec<String>,
}

impl OnboardingWriteResponse {
    fn new(restart_required: bool, entry: serde_json::Value) -> Self {
        Self {
            changed: true,
            restart_required,
            config_path: resolved_config_path(),
            entry,
            warnings: Vec::new(),
        }
    }
}

/// The config file path that [`Config::save`] just wrote (or would write).
fn resolved_config_path() -> Option<String> {
    Config::discover_path()
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Apply a typed JSON value at a dotted key, mapping failures to bad_request.
fn set_cfg(cfg: &mut Config, key: &str, value: serde_json::Value) -> TauriResult<()> {
    cfg.set_value(key, &value)
        .map_err(|e| TauriError::bad_request(format!("Failed to apply {key}: {e}")))
}

fn env_var_set(name: &str) -> bool {
    !name.is_empty() && std::env::var_os(name).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Classify a credential like the Python onboarding surfaces (`explicit`,
/// `env`, `missing_env`, or `none`).
fn api_key_source(api_key: &str, api_key_env: &str) -> &'static str {
    if !api_key.is_empty() {
        "explicit"
    } else if api_key_env.is_empty() {
        "none"
    } else if env_var_set(api_key_env) {
        "env"
    } else {
        "missing_env"
    }
}

fn is_http_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

fn is_secret_like_key(key: &str) -> bool {
    let k = key.to_lowercase().replace(['-', ' '], "_");
    matches!(k.as_str(), "key" | "token" | "secret" | "password" | "authorization")
        || k.ends_with("_key")
        || k.ends_with("_token")
        || k.ends_with("_secret")
        || k.ends_with("_password")
}

/// Redact secret-shaped keys in the untyped router `tiers` payload so the RPC
/// response never echoes an operator-pasted credential back to the WebUI.
fn redact_tiers(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if is_secret_like_key(k) && !v.is_null() {
                    out.insert(k.clone(), serde_json::Value::String(REDACTED.to_string()));
                } else if v.is_object() {
                    out.insert(k.clone(), redact_tiers(v));
                } else {
                    out.insert(k.clone(), v.clone());
                }
            }
            serde_json::Value::Object(out)
        }
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// `onboarding.router.configure` — persist the `[squilla_router]` surface.
#[tauri::command]
pub async fn onboarding_router_configure(
    state: State<'_, AppState>,
    mode: Option<String>,
    default_tier: Option<String>,
    tiers: Option<serde_json::Value>,
    cross_provider_tiers: Option<bool>,
    tier_provider_mismatch: Option<String>,
) -> TauriResult<OnboardingWriteResponse> {
    const ROUTER_MODES: &[&str] = &["recommended", "openrouter-mix", "custom", "disabled"];
    let mode = mode.unwrap_or("recommended".to_string());
    if !ROUTER_MODES.contains(&mode.as_str()) {
        return Err(TauriError::bad_request(format!(
            "router mode must be recommended, openrouter-mix, custom, or disabled (got {mode:?})"
        )));
    }
    if let Some(t) = &tiers {
        if !t.is_object() {
            return Err(TauriError::bad_request("tiers must be an object of tier configs"));
        }
    }
    let mismatch = match tier_provider_mismatch {
        Some(v) => {
            let v = v.trim().to_string();
            if !matches!(v.as_str(), "route" | "veto") {
                return Err(TauriError::bad_request(
                    "tierProviderMismatch must be route or veto",
                ));
            }
            Some(v)
        }
        None => None,
    };
    let enabled = mode != "disabled";

    {
        let mut cfg = state.config_mut().await;
        cfg.squilla_router.get_or_insert_default();
        set_cfg(&mut cfg, "squilla_router.enabled", serde_json::json!(enabled))?;
        if let Some(tier) = default_tier {
            if !tier.trim().is_empty() {
                set_cfg(&mut cfg, "squilla_router.default_tier", serde_json::json!(tier))?;
            }
        }
        if let Some(cp) = cross_provider_tiers {
            set_cfg(
                &mut cfg,
                "squilla_router.cross_provider_tiers",
                serde_json::json!(cp),
            )?;
        }
        if let Some(m) = mismatch {
            set_cfg(
                &mut cfg,
                "squilla_router.tier_provider_mismatch",
                serde_json::json!(m),
            )?;
        }
        if let Some(t) = &tiers {
            set_cfg(&mut cfg, "squilla_router.tiers", t.clone())?;
        }
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let router = state.config().await.squilla_router.clone().unwrap_or_default();
    let tiers_entry = tiers
        .as_ref()
        .map(redact_tiers)
        .unwrap_or(serde_json::json!({}));
    let entry = serde_json::json!({
        "mode": mode,
        "enabled": enabled,
        "tier_profile": router.tier_profile,
        "default_tier": router.default_tier,
        "tiers": tiers_entry,
        "cross_provider_tiers": router.cross_provider_tiers,
        "tier_provider_mismatch": router.tier_provider_mismatch,
        "router_binding": router.preset_binding.as_deref().unwrap_or("legacy"),
    });
    Ok(OnboardingWriteResponse::new(false, entry))
}

// ---------------------------------------------------------------------------
// Ensemble
// ---------------------------------------------------------------------------

/// `onboarding.ensemble.configure` — persist the `[llm_ensemble]` surface.
///
/// Partial-payload semantics are pinned: omitted keys keep their current
/// stored value, matching the Python `upsert_llm_ensemble` keep-current merge.
#[tauri::command]
pub async fn onboarding_ensemble_configure(
    state: State<'_, AppState>,
    enabled: Option<bool>,
    selection_mode: Option<String>,
    model_options: Option<Vec<String>>,
    candidates: Option<Vec<serde_json::Value>>,
    min_successful_proposers: Option<u32>,
    all_failed_policy: Option<String>,
) -> TauriResult<OnboardingWriteResponse> {
    const SELECTION_MODES: &[&str] = &[
        "router_dynamic",
        "static_openrouter_b5",
        "static_tokenrhythm_b5",
        "custom_b5",
    ];
    const ALL_FAILED_POLICIES: &[&str] = &["fallback_single", "error"];
    if let Some(m) = &selection_mode {
        if !SELECTION_MODES.contains(&m.trim()) {
            return Err(TauriError::bad_request(format!(
                "selectionMode must be one of: {}",
                SELECTION_MODES.join(", ")
            )));
        }
    }
    if let Some(p) = &all_failed_policy {
        if !ALL_FAILED_POLICIES.contains(&p.trim()) {
            return Err(TauriError::bad_request(format!(
                "allFailedPolicy must be one of: {}",
                ALL_FAILED_POLICIES.join(", ")
            )));
        }
    }
    if let Some(n) = min_successful_proposers {
        if n < 1 {
            return Err(TauriError::bad_request("minSuccessfulProposers must be at least 1"));
        }
    }
    if let Some(list) = &candidates {
        if list.iter().any(|c| !c.is_object()) {
            return Err(TauriError::bad_request(
                "candidates must be a list of candidate objects",
            ));
        }
    }

    {
        let mut cfg = state.config_mut().await;
        cfg.llm_ensemble.get_or_insert_default();
        if let Some(v) = enabled {
            set_cfg(&mut cfg, "llm_ensemble.enabled", serde_json::json!(v))?;
        }
        if let Some(v) = selection_mode {
            set_cfg(&mut cfg, "llm_ensemble.selection_mode", serde_json::json!(v.trim()))?;
        }
        if let Some(v) = model_options {
            set_cfg(&mut cfg, "llm_ensemble.model_options", serde_json::json!(v))?;
        }
        if let Some(v) = candidates {
            set_cfg(&mut cfg, "llm_ensemble.candidates", serde_json::json!(v))?;
        }
        if let Some(v) = min_successful_proposers {
            set_cfg(
                &mut cfg,
                "llm_ensemble.min_successful_proposers",
                serde_json::json!(v),
            )?;
        }
        if let Some(v) = all_failed_policy {
            set_cfg(&mut cfg, "llm_ensemble.all_failed_policy", serde_json::json!(v.trim()))?;
        }
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let ensemble = state.config().await.llm_ensemble.clone().unwrap_or_default();
    let mut entry = serde_json::json!({
        "enabled": ensemble.enabled,
        "selection_mode": ensemble.selection_mode,
        "model_options": ensemble.model_options,
        "min_successful_proposers": ensemble.min_successful_proposers,
        "all_failed_policy": ensemble.all_failed_policy,
    });
    if !ensemble.candidates.is_empty() {
        entry["candidates"] = serde_json::json!(ensemble.candidates);
    }
    Ok(OnboardingWriteResponse::new(false, entry))
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// `onboarding.search.configure` — persist the inline `search_*` surface.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn onboarding_search_configure(
    state: State<'_, AppState>,
    provider_id: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
    max_results: Option<u32>,
    proxy: Option<String>,
    use_env_proxy: Option<bool>,
    fallback_policy: Option<String>,
    diagnostics: Option<bool>,
) -> TauriResult<OnboardingWriteResponse> {
    let provider_id = provider_id.trim().to_string();
    if provider_id.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    if let Some(p) = &fallback_policy {
        if !matches!(p.as_str(), "off" | "network") {
            return Err(TauriError::bad_request("fallbackPolicy must be 'off' or 'network'"));
        }
    }
    let api_key = api_key.unwrap_or_default();
    let api_key_env = api_key_env.unwrap_or_default();
    let max_results = max_results
        .unwrap_or(DEFAULT_SEARCH_MAX_RESULTS)
        .clamp(1, MAX_SEARCH_RESULTS);
    let proxy = proxy.unwrap_or_default();
    let use_env_proxy = use_env_proxy.unwrap_or(false);
    let fallback_policy = fallback_policy.unwrap_or("off".to_string());
    let diagnostics = diagnostics.unwrap_or(false);

    let mut effective_api_key = api_key.trim().to_string();
    let effective_api_key_env = api_key_env.trim().to_string();
    // A blank key keeps the stored key when re-saving the provider that is
    // already active (the same keep-current contract as Python); a genuinely
    // fresh keyed provider fails.
    if effective_api_key.is_empty() && effective_api_key_env.is_empty() && provider_id != "duckduckgo"
    {
        let cfg = state.config().await;
        let keep_stored = cfg.search_provider.as_deref() == Some(provider_id.as_str())
            && cfg
                .search_api_key
                .as_deref()
                .map(|k| !k.is_empty())
                .unwrap_or(false);
        if keep_stored {
            effective_api_key = cfg.search_api_key.clone().unwrap_or_default();
        } else {
            return Err(TauriError::bad_request(format!(
                "search provider {provider_id:?} requires an api_key or api_key_env"
            )));
        }
    }
    let source = api_key_source(&effective_api_key, &effective_api_key_env);

    {
        let mut cfg = state.config_mut().await;
        set_cfg(&mut cfg, "search_provider", serde_json::json!(provider_id))?;
        set_cfg(&mut cfg, "search_api_key", serde_json::json!(effective_api_key))?;
        set_cfg(&mut cfg, "search_api_key_env", serde_json::json!(effective_api_key_env))?;
        set_cfg(&mut cfg, "search_max_results", serde_json::json!(max_results))?;
        set_cfg(&mut cfg, "search_proxy", serde_json::json!(proxy))?;
        set_cfg(&mut cfg, "search_use_env_proxy", serde_json::json!(use_env_proxy))?;
        set_cfg(&mut cfg, "search_fallback_policy", serde_json::json!(fallback_policy))?;
        set_cfg(&mut cfg, "search_diagnostics", serde_json::json!(diagnostics))?;
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let entry_api_key = if effective_api_key.is_empty() {
        ""
    } else {
        REDACTED
    };
    let entry = serde_json::json!({
        "provider": provider_id,
        "api_key": entry_api_key,
        "api_key_env": effective_api_key_env,
        "api_key_source": source,
        "max_results": max_results,
        "proxy": proxy,
        "use_env_proxy": use_env_proxy,
        "fallback_policy": fallback_policy,
        "diagnostics": diagnostics,
    });
    Ok(OnboardingWriteResponse::new(false, entry))
}

// ---------------------------------------------------------------------------
// Image generation
// ---------------------------------------------------------------------------

/// `onboarding.imageGeneration.configure` — persist the `[image_generation]`
/// surface including the per-provider credential block.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn onboarding_image_generation_configure(
    state: State<'_, AppState>,
    provider_id: String,
    primary: Option<String>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    base_url: Option<String>,
    enabled: Option<bool>,
    size: Option<String>,
    output_format: Option<String>,
    fallbacks: Option<Vec<String>>,
    clear_fallbacks: Option<bool>,
    credential_mode: Option<String>,
) -> TauriResult<OnboardingWriteResponse> {
    const VALID_SIZES: &[&str] = &["1024x1024", "1536x1024", "1024x1536"];
    const VALID_FORMATS: &[&str] = &["png", "jpeg", "webp"];
    let provider_id = provider_id.trim().to_string();
    if provider_id.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    if let Some(m) = &credential_mode {
        if !matches!(m.as_str(), "direct" | "env") {
            return Err(TauriError::bad_request("credentialMode must be 'direct' or 'env'"));
        }
    }
    let api_key = api_key.unwrap_or_default();
    let api_key_env = api_key_env.unwrap_or_default();
    if !api_key.is_empty() && !api_key_env.is_empty() {
        return Err(TauriError::bad_request("configure either api_key or api_key_env, not both"));
    }
    let enabled = enabled.unwrap_or(true);
    let clear_fallbacks = clear_fallbacks.unwrap_or(false);
    let size = size.unwrap_or_default();
    let output_format = output_format.unwrap_or_default();
    if !size.is_empty() && !VALID_SIZES.contains(&size.as_str()) {
        return Err(TauriError::bad_request(format!(
            "image size must be one of: {}",
            VALID_SIZES.join(", ")
        )));
    }
    if !output_format.is_empty() && !VALID_FORMATS.contains(&output_format.as_str()) {
        return Err(TauriError::bad_request(format!(
            "image output format must be one of: {}",
            VALID_FORMATS.join(", ")
        )));
    }
    if let Some(b) = &base_url {
        let b = b.trim();
        if !b.is_empty() && !is_http_url(b) {
            return Err(TauriError::bad_request(
                "image base_url must be an absolute http:// or https:// URL",
            ));
        }
    }
    if clear_fallbacks && fallbacks.is_none() {
        return Err(TauriError::bad_request("clearFallbacks requires a fallbacks list"));
    }

    // Normalize the primary reference to "{provider}/{model}".
    let primary_ref = match primary {
        Some(p) => {
            let p = p.trim().to_string();
            if p.is_empty() {
                None
            } else if let Some((prov, model)) = p.split_once('/') {
                if prov != provider_id || model.is_empty() {
                    return Err(TauriError::bad_request(format!(
                        "primary must be a provider/model reference for image generation provider {provider_id:?}"
                    )));
                }
                Some(p)
            } else {
                Some(format!("{provider_id}/{p}"))
            }
        }
        None => None,
    };

    // Clean the fallback chain, honoring the additive clear intent.
    let mut cleaned_fallbacks: Vec<String> = Vec::new();
    if let Some(list) = &fallbacks {
        for f in list {
            let f = f.trim().to_string();
            if f.is_empty() {
                continue;
            }
            let bad_ref = f
                .split_once('/')
                .is_some_and(|(prov, model)| prov.is_empty() || model.is_empty());
            if bad_ref {
                return Err(TauriError::bad_request(format!(
                    "image fallback {f:?} must be a provider/model reference"
                )));
            }
            cleaned_fallbacks.push(f);
        }
    }
    let effective_fallbacks = if clear_fallbacks {
        Some(cleaned_fallbacks)
    } else if !cleaned_fallbacks.is_empty() {
        Some(cleaned_fallbacks)
    } else {
        None // keep current chain
    };

    // Resolve the credential side; credentialMode is authoritative when the
    // client states which control was edited.
    let mut effective_api_key = api_key.trim().to_string();
    let mut effective_api_key_env = api_key_env.trim().to_string();
    if credential_mode.as_deref() == Some("direct") {
        effective_api_key_env.clear();
    } else if credential_mode.as_deref() == Some("env") {
        effective_api_key.clear();
    } else if !effective_api_key.is_empty() {
        effective_api_key_env.clear();
    }

    {
        let mut cfg = state.config_mut().await;
        let image = cfg.image_generation.get_or_insert_default();
        image.enabled = enabled;
        image.binding = "custom".to_string();
        if let Some(p) = &primary_ref {
            image.primary = p.clone();
        }
        if !size.is_empty() {
            image.size = size.clone();
        }
        if !output_format.is_empty() {
            image.output_format = Some(output_format.clone());
        }
        if let Some(fb) = &effective_fallbacks {
            image.fallbacks = fb.clone();
        }
        let provider = image.providers.entry(provider_id.clone()).or_default();
        if !effective_api_key.is_empty() {
            provider.api_key = Some(effective_api_key.clone());
        } else if !effective_api_key_env.is_empty() || credential_mode.as_deref() == Some("env") {
            // Switching to an env reference replaces a stored direct key.
            provider.api_key = None;
        }
        if !effective_api_key_env.is_empty() {
            provider.api_key_env = Some(effective_api_key_env.clone());
        } else if !effective_api_key.is_empty() || credential_mode.as_deref() == Some("direct") {
            provider.api_key_env = None;
        }
        if let Some(b) = base_url {
            let b = b.trim().to_string();
            if !b.is_empty() {
                provider.base_url = Some(b);
            }
        }
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let cfg = state.config().await;
    let image = cfg.image_generation.clone().unwrap_or_default();
    let provider_cfg = image.providers.get(&provider_id);
    let api_key_val = provider_cfg
        .and_then(|p| p.api_key.clone())
        .unwrap_or_default();
    let api_key_env_val = provider_cfg
        .and_then(|p| p.api_key_env.clone())
        .unwrap_or_default();
    let base_url_val = provider_cfg
        .and_then(|p| p.base_url.clone())
        .unwrap_or_default();
    let source = api_key_source(&api_key_val, &api_key_env_val);
    let entry_api_key = if api_key_val.is_empty() { "" } else { REDACTED };
    let entry = serde_json::json!({
        "provider": provider_id,
        "enabled": image.enabled,
        "primary": image.primary,
        "api_key": entry_api_key,
        "api_key_env": api_key_env_val,
        "api_key_source": source,
        "base_url": base_url_val,
        "size": image.size,
        "output_format": image.output_format,
        "fallbacks": image.fallbacks,
    });
    Ok(OnboardingWriteResponse::new(false, entry))
}

// ---------------------------------------------------------------------------
// Memory embedding
// ---------------------------------------------------------------------------

/// `onboarding.memory_embedding.configure` — persist `[memory.embedding]`.
///
/// Restart-gated like the Python path: `restartRequired` is reported true so
/// the WebUI surfaces the pending-restart toast.
#[tauri::command]
pub async fn onboarding_memory_embedding_configure(
    state: State<'_, AppState>,
    provider_id: String,
    model: Option<String>,
    api_key: Option<String>,
    api_key_env: Option<String>,
    base_url: Option<String>,
    onnx_dir: Option<String>,
) -> TauriResult<OnboardingWriteResponse> {
    const REMOTE_PROVIDERS: &[&str] = &["openai", "openai-compatible"];
    const ALL_PROVIDERS: &[&str] = &[
        "auto",
        "none",
        "local",
        "openai",
        "openai-compatible",
        "ollama",
    ];
    let provider = provider_id.trim().to_string();
    if !ALL_PROVIDERS.contains(&provider.as_str()) {
        return Err(TauriError::bad_request(format!(
            "unknown memory embedding provider: {provider:?}"
        )));
    }
    let api_key = api_key.unwrap_or_default();
    let api_key_env = api_key_env.unwrap_or_default();
    if !api_key.is_empty() && !api_key_env.is_empty() {
        return Err(TauriError::bad_request("configure either api_key or api_key_env, not both"));
    }
    let api_key = api_key.trim().to_string();
    let api_key_env = api_key_env.trim().to_string();
    let model = model.map(|m| m.trim().to_string()).filter(|m| !m.is_empty());
    let base_url = base_url
        .map(|b| b.trim().to_string())
        .filter(|b| !b.is_empty());
    let onnx_dir = onnx_dir
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty());

    // Snapshot the stored embedding so a re-save can keep the current
    // credential without holding the read lock across the write.
    let current_emb = {
        let cfg = state.config().await;
        cfg.memory
            .as_ref()
            .map(|m| m.embedding.clone())
            .unwrap_or_default()
    };

    // Remote providers must carry a credential; a stored one satisfies the
    // requirement on re-save, mirroring the Python keep-current merge.
    if REMOTE_PROVIDERS.contains(&provider.as_str())
        && api_key.is_empty()
        && api_key_env.is_empty()
    {
        let has_stored = current_emb
            .remote
            .api_key
            .as_deref()
            .map(|k| !k.is_empty())
            .unwrap_or(false)
            || current_emb
                .remote
                .api_key_env
                .as_deref()
                .map(|e| !e.is_empty())
                .unwrap_or(false);
        if !has_stored {
            return Err(TauriError::bad_request(
                "remote memory embedding provider requires an api_key or api_key_env",
            ));
        }
    }

    {
        let mut cfg = state.config_mut().await;
        let memory = cfg.memory.get_or_insert_default();
        let emb = &mut memory.embedding;
        // Fresh embedding section: switching providers never carries stale
        // credentials over (mirrors Python's `model_validate(payload)` replace).
        *emb = MemoryEmbeddingConfig::default();
        emb.provider = provider.clone();
        match provider.as_str() {
            "local" => {
                if let Some(dir) = &onnx_dir {
                    emb.local.onnx_dir = Some(dir.clone());
                } else if let Some(stored) = current_emb
                    .local
                    .onnx_dir
                    .as_ref()
                    .filter(|d| !d.is_empty())
                {
                    emb.local.onnx_dir = Some(stored.clone());
                }
            }
            "ollama" => {
                let base = base_url
                    .clone()
                    .or(current_emb.ollama.base_url.clone())
                    .filter(|b| !b.is_empty())
                    .unwrap_or(DEFAULT_OLLAMA_EMBEDDING_BASE_URL.to_string());
                emb.ollama.base_url = Some(base);
                if let Some(m) = &model {
                    emb.ollama.model = Some(m.clone());
                } else if let Some(stored) = current_emb
                    .ollama
                    .model
                    .as_ref()
                    .filter(|m| !m.is_empty())
                {
                    emb.ollama.model = Some(stored.clone());
                }
            }
            "auto" | "openai" | "openai-compatible" => {
                let effective_base = base_url
                    .clone()
                    .or(current_emb.remote.base_url.clone().or(current_emb.base_url.clone()))
                    .filter(|b| !b.is_empty());
                if let Some(b) = &effective_base {
                    emb.remote.base_url = Some(b.clone());
                } else if REMOTE_PROVIDERS.contains(&provider.as_str()) {
                    emb.remote.base_url = Some(DEFAULT_REMOTE_EMBEDDING_BASE_URL.to_string());
                }
                // api_key and api_key_env stay mutually exclusive.
                if !api_key.is_empty() {
                    emb.remote.api_key = Some(api_key.clone());
                } else if !api_key_env.is_empty() {
                    emb.remote.api_key_env = Some(api_key_env.clone());
                } else {
                    if let Some(stored) = current_emb
                        .remote
                        .api_key
                        .as_ref()
                        .filter(|k| !k.is_empty())
                    {
                        emb.remote.api_key = Some(stored.clone());
                    }
                    if let Some(stored) = current_emb
                        .remote
                        .api_key_env
                        .as_ref()
                        .filter(|e| !e.is_empty())
                    {
                        emb.remote.api_key_env = Some(stored.clone());
                    }
                }
                if let Some(m) = &model {
                    emb.remote.model = Some(m.clone());
                } else if let Some(stored) = current_emb
                    .remote
                    .model
                    .as_ref()
                    .or(current_emb.model.as_ref())
                    .filter(|m| !m.is_empty())
                {
                    emb.remote.model = Some(stored.clone());
                }
            }
            _ => {}
        }
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let emb = {
        let cfg = state.config().await;
        cfg.memory
            .as_ref()
            .map(|m| m.embedding.clone())
            .unwrap_or_default()
    };
    let mut entry = serde_json::json!({ "provider": provider });
    match emb.provider.as_str() {
        "local" => {
            let mut local = serde_json::Map::new();
            if let Some(dir) = emb.local.onnx_dir.as_deref().filter(|d| !d.is_empty()) {
                local.insert("onnx_dir".to_string(), serde_json::json!(dir));
            }
            if !local.is_empty() {
                entry["local"] = serde_json::Value::Object(local);
            }
        }
        "ollama" => {
            let mut ollama = serde_json::Map::new();
            if let Some(b) = emb.ollama.base_url.as_deref().filter(|b| !b.is_empty()) {
                ollama.insert("base_url".to_string(), serde_json::json!(b));
            }
            if let Some(m) = emb.ollama.model.as_deref().filter(|m| !m.is_empty()) {
                ollama.insert("model".to_string(), serde_json::json!(m));
            }
            if !ollama.is_empty() {
                entry["ollama"] = serde_json::Value::Object(ollama);
            }
        }
        "auto" | "openai" | "openai-compatible" => {
            let mut remote = serde_json::Map::new();
            if let Some(b) = emb.remote.base_url.as_deref().filter(|b| !b.is_empty()) {
                remote.insert("base_url".to_string(), serde_json::json!(b));
            }
            if let Some(k) = emb.remote.api_key.as_deref().filter(|k| !k.is_empty()) {
                remote.insert("api_key".to_string(), serde_json::json!(REDACTED));
            }
            if let Some(e) = emb.remote.api_key_env.as_deref().filter(|e| !e.is_empty()) {
                remote.insert("api_key_env".to_string(), serde_json::json!(e));
            }
            if let Some(m) = emb.remote.model.as_deref().filter(|m| !m.is_empty()) {
                remote.insert("model".to_string(), serde_json::json!(m));
            }
            if !remote.is_empty() {
                entry["remote"] = serde_json::Value::Object(remote);
            }
        }
        _ => {}
    }
    Ok(OnboardingWriteResponse::new(true, entry))
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

/// `onboarding.audio.configure` — persist the `[audio]` surface.
///
/// Configuration implies enablement for new clients (the WebUI omits
/// `enabled`), matching the Python `apply_audio_provider_configuration`.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn onboarding_audio_configure(
    state: State<'_, AppState>,
    provider_id: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
    base_url: Option<String>,
    enabled: Option<bool>,
    tts_voice: Option<String>,
    tts_model: Option<String>,
    language_code: Option<String>,
) -> TauriResult<OnboardingWriteResponse> {
    let provider_id = provider_id.trim().to_string();
    if provider_id.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    if provider_id != "elevenlabs" {
        return Err(TauriError::bad_request(format!(
            "audio provider {provider_id:?} is not supported (only elevenlabs)"
        )));
    }
    let api_key = api_key.unwrap_or_default();
    let api_key_env = api_key_env.unwrap_or_default();
    if !api_key.is_empty() && !api_key_env.is_empty() {
        return Err(TauriError::bad_request("configure either api_key or api_key_env, not both"));
    }
    let enabled = enabled.unwrap_or(true);
    let effective_api_key = api_key.trim().to_string();
    let effective_api_key_env = api_key_env.trim().to_string();
    if let Some(b) = &base_url {
        let b = b.trim();
        if !b.is_empty() && !is_http_url(b) {
            return Err(TauriError::bad_request(
                "audio base_url must be an absolute http:// or https:// URL",
            ));
        }
    }

    // A stored credential satisfies the requirement on re-save; only a fresh
    // enabled provider with no credential fails (Python `requires_api_key`).
    if enabled && effective_api_key.is_empty() && effective_api_key_env.is_empty() {
        let cfg = state.config().await;
        let has_stored = cfg
            .audio
            .as_ref()
            .and_then(|a| a.providers.get(&provider_id))
            .map(|p| {
                p.api_key.as_deref().map(|k| !k.is_empty()).unwrap_or(false)
                    || p.api_key_env
                        .as_deref()
                        .map(|e| !e.is_empty())
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        if !has_stored {
            return Err(TauriError::bad_request(format!(
                "audio provider {provider_id:?} requires an api_key or ELEVENLABS_API_KEY"
            )));
        }
    }

    {
        let mut cfg = state.config_mut().await;
        let audio = cfg.audio.get_or_insert_default();
        audio.enabled = enabled;
        let provider = audio.providers.entry(provider_id.clone()).or_default();
        if !effective_api_key.is_empty() {
            provider.api_key = Some(effective_api_key.clone());
            provider.api_key_env = None;
        } else if !effective_api_key_env.is_empty() {
            provider.api_key = None;
            provider.api_key_env = Some(effective_api_key_env.clone());
        }
        let stored_base = provider.base_url.clone().unwrap_or_default();
        let effective_base_url = match base_url.as_deref().map(str::trim) {
            Some(b) if !b.is_empty() => b.to_string(),
            _ if stored_base.is_empty() => DEFAULT_AUDIO_BASE_URL.to_string(),
            _ => stored_base,
        };
        provider.base_url = Some(effective_base_url);
        if let Some(v) = tts_voice.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            audio.tts.voice = v.to_string();
        }
        if let Some(v) = tts_model.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            audio.tts.model = v.to_string();
        }
        if let Some(v) = language_code.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            audio.tts.language_code = v.to_string();
        }
    }
    {
        let cfg = state.config().await;
        cfg.save()
            .map_err(|e| TauriError::internal(format!("Failed to persist config: {e}")))?;
    }

    let cfg = state.config().await;
    let audio = cfg.audio.clone().unwrap_or_default();
    let provider_cfg = audio.providers.get(&provider_id);
    let api_key_val = provider_cfg
        .and_then(|p| p.api_key.clone())
        .unwrap_or_default();
    let api_key_env_val = provider_cfg
        .and_then(|p| p.api_key_env.clone())
        .unwrap_or_default();
    let base_url_val = provider_cfg
        .and_then(|p| p.base_url.clone())
        .unwrap_or_default();
    let source = api_key_source(&api_key_val, &api_key_env_val);
    let entry_api_key = if api_key_val.is_empty() { "" } else { REDACTED };
    let entry = serde_json::json!({
        "provider": provider_id,
        "enabled": audio.enabled,
        "api_key": entry_api_key,
        "api_key_env": api_key_env_val,
        "api_key_source": source,
        "base_url": base_url_val,
        "tts_voice": audio.tts.voice,
        "tts_model": audio.tts.model,
        "language_code": audio.tts.language_code,
    });
    Ok(OnboardingWriteResponse::new(false, entry))
}
