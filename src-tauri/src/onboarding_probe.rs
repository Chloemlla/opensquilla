//! Onboarding probe/discover commands (S5).
//!
//! Implements `onboarding.provider.probe`, `onboarding.llmProfile.probe` /
//! `.draft.probe`, `onboarding.models.discover`, `onboarding.llmProfile.models.discover` /
//! `.draft.models.discover` and `onboarding.imageGeneration.models.discover` over
//! the Tauri bridge so the WebUI setup wizard can live-test a candidate
//! provider deployment before persisting it.
//!
//! Fail-closed by design: a probe never throws transport noise — reachability
//! and credential problems come back as a structured not-ok result. Real
//! network calls cannot run in CI; unit tests cover the pure helpers
//! (deployment resolution, endpoint-origin identity, wire mapping).

use crate::error::{TauriError, TauriResult};
use crate::onboarding_provider::{
    is_redacted_secret_sentinel, profile_lookup, provider_env_key, provider_spec,
    requires_api_key,
};
use crate::state::AppState;
use opensquilla_core::config::{Config, LlmConfig};
use opensquilla_provider::live_catalog::parse_models_response;
use opensquilla_provider::model_catalog::ModelCapabilities;
use opensquilla_provider::registry::ProviderSpec;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};
use tauri::State;

/// Probe timeout, mirroring Python `_PROBE_TIMEOUT_SECONDS`.
const PROBE_TIMEOUT_SECS: u64 = 15;

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

/// `onboarding.provider.probe` request — candidate deployment overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderProbeRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
}

/// `onboarding.llmProfile.probe` request — stored profile by provider id.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileProbeRequest {
    pub provider_id: String,
    pub model: Option<String>,
}

/// `onboarding.llmProfile.draft.probe` request — editor draft overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileDraftProbeRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
    #[serde(default)]
    pub keep_current_secret: bool,
}

/// `onboarding.models.discover` request — candidate deployment overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsDiscoverRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
    #[serde(default)]
    pub force_refresh: bool,
}

/// `onboarding.llmProfile.models.discover` request — stored profile by id.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileModelsDiscoverRequest {
    pub provider_id: String,
}

/// `onboarding.llmProfile.draft.models.discover` request — draft overrides.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileDraftModelsDiscoverRequest {
    pub provider_id: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub proxy: Option<String>,
    #[serde(default)]
    pub keep_current_secret: bool,
}

/// `onboarding.imageGeneration.models.discover` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationModelsDiscoverRequest {
    pub provider_id: String,
}

/// Probe result envelope (mirrors Python `ProviderProbeResult.to_payload`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResponse {
    pub ok: bool,
    pub provider: String,
    pub model: String,
    pub failure_kind: String,
    pub message: String,
    pub first_response_ms: Option<u64>,
    pub total_ms: Option<u64>,
    pub latency_ms: Option<u64>,
}

/// Model-discovery result envelope (mirrors Python
/// `ProviderModelsDiscoverResult.to_payload`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelsDiscoverResponse {
    pub ok: bool,
    pub provider_id: String,
    pub failure_kind: String,
    pub detail: String,
    pub source: String,
    pub models: Vec<serde_json::Value>,
    pub catalog: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Deployment resolution
// ---------------------------------------------------------------------------

/// Resolved deployment used by both probe and discover. The API key is the
/// *actual* value (env references are resolved here), never a round-tripped
/// mask.
struct ResolvedDeployment {
    provider: String,
    model: String,
    base_url: String,
    api_key: String,
    api_key_env: String,
    proxy: String,
}

/// Stored deployment for a provider: the active `llm` section when it is the
/// same provider, else the stored LLM profile (case-insensitive key lookup).
struct StoredDeployment {
    model: String,
    base_url: String,
    api_key: Option<String>,
    api_key_env: Option<String>,
    proxy: String,
}

/// Candidate overrides from the wire request; empty fields mean "not provided".
#[derive(Default)]
struct DeploymentOverrides {
    model: String,
    api_key: String,
    api_key_env: String,
    base_url: String,
    proxy: String,
    keep_current_secret: bool,
}

/// Request → override projection shared by every probe/discover variant.
fn overrides(
    model: Option<&str>,
    api_key: Option<&str>,
    api_key_env: Option<&str>,
    base_url: Option<&str>,
    proxy: Option<&str>,
    keep_current_secret: bool,
) -> DeploymentOverrides {
    DeploymentOverrides {
        model: model.unwrap_or("").to_string(),
        api_key: api_key.unwrap_or("").to_string(),
        api_key_env: api_key_env.unwrap_or("").to_string(),
        base_url: base_url.unwrap_or("").to_string(),
        proxy: proxy.unwrap_or("").to_string(),
        keep_current_secret,
    }
}

fn stored_deployment(cfg: &Config, provider: &str) -> Option<StoredDeployment> {
    if let Some(llm) = cfg.llm.as_ref() {
        if llm.provider.trim().to_lowercase() == provider {
            return Some(StoredDeployment {
                model: llm.model.clone(),
                base_url: llm.base_url.clone(),
                api_key: llm.api_key.clone(),
                api_key_env: llm.api_key_env.clone(),
                proxy: llm.proxy.clone().unwrap_or_default(),
            });
        }
    }
    profile_lookup(cfg, provider).map(|p| StoredDeployment {
        model: p.model.clone().unwrap_or_default(),
        base_url: p.base_url.clone().unwrap_or_default(),
        api_key: p.api_key.clone(),
        api_key_env: p.api_key_env.clone(),
        proxy: p.proxy.clone().unwrap_or_default(),
    })
}

/// Origin (scheme + host + port, ignoring path) used for same-endpoint
/// identity — mirrors Python `endpoint_identity` same-endpoint probing.
fn endpoint_origin(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    for prefix in ["http://", "https://"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return rest.split('/').next().unwrap_or("").to_string();
        }
    }
    trimmed.split('/').next().unwrap_or("").to_string()
}

fn same_endpoint_origin(a: &str, b: &str) -> bool {
    let (oa, ob) = (endpoint_origin(a), endpoint_origin(b));
    !oa.is_empty() && !ob.is_empty() && oa == ob
}

/// First non-empty value from a candidate chain (defaults last).
fn first_nonempty<'a>(candidates: &[&'a str]) -> &'a str {
    candidates
        .iter()
        .copied()
        .find(|v| !v.trim().is_empty())
        .unwrap_or("")
}

/// Resolve the deployment the probe/discover should exercise.
///
/// - `base_url` / `proxy`: request override, else stored, else registry spec.
/// - `model`: request override, else stored, else spec default.
/// - key: request override (a `***` mask is treated as absent), else the
///   stored value only when the endpoint is same-origin with the stored one
///   (never send a key for an endpoint it was not provisioned for), else the
///   provider's canonical environment variable.
fn resolve_deployment(
    cfg: &Config,
    provider: &str,
    spec: &ProviderSpec,
    o: &DeploymentOverrides,
) -> ResolvedDeployment {
    let stored = stored_deployment(cfg, provider);
    let stored_base = stored.as_ref().map(|s| s.base_url.as_str()).unwrap_or("");

    let base_url = first_nonempty(&[&o.base_url, stored_base, spec.api_base]).to_string();
    let proxy = first_nonempty(&[&o.proxy, stored.as_ref().map(|s| s.proxy.as_str()).unwrap_or("")])
        .to_string();
    let model = first_nonempty(&[
        &o.model,
        stored.as_ref().map(|s| s.model.as_str()).unwrap_or(""),
        spec.default_model,
        LlmConfig::default().model.as_str(),
    ])
    .to_string();

    let mut api_key = o.api_key.clone();
    if is_redacted_secret_sentinel(&api_key) {
        api_key.clear();
    }
    let mut api_key_env = o.api_key_env.clone();
    let reuse_stored = same_endpoint_origin(&base_url, stored_base)
        && api_key.is_empty()
        && api_key_env.is_empty();
    if reuse_stored || (o.keep_current_secret && api_key.is_empty() && api_key_env.is_empty()) {
        if let Some(stored) = stored.as_ref() {
            if api_key.is_empty() {
                api_key = stored.api_key.clone().unwrap_or_default();
            }
            if api_key_env.is_empty() {
                api_key_env = stored.api_key_env.clone().unwrap_or_default();
            }
        }
    }

    // Resolve environment references to their current values at probe time.
    let mut resolved_key = api_key.clone();
    if resolved_key.is_empty() && !api_key_env.is_empty() {
        if let Ok(v) = std::env::var(&api_key_env) {
            resolved_key = v;
        }
    }
    if resolved_key.is_empty() && api_key_env.is_empty() {
        let default_env = provider_env_key(provider);
        if !default_env.is_empty() {
            if let Ok(v) = std::env::var(&default_env) {
                resolved_key = v;
            }
        }
    }

    ResolvedDeployment {
        provider: provider.to_string(),
        model,
        base_url,
        api_key: resolved_key,
        api_key_env,
        proxy,
    }
}

/// Normalize a user-supplied base URL to an absolute URL with a scheme.
fn absolute_base_url(url: &str) -> String {
    let url = url.trim();
    if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        format!("https://{url}")
    }
}

/// Redact a known secret from an error message before it reaches the frontend.
fn redact_error_text(message: &str, secret: &str) -> String {
    if secret.is_empty() {
        message.to_string()
    } else {
        message.replace(secret, "***")
    }
}

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

/// One live probe result. `ok=true` only on a 2xx chat-completions response.
struct ProbeOutcome {
    ok: bool,
    failure_kind: &'static str,
    message: String,
    first_response_ms: Option<u64>,
    total_ms: Option<u64>,
}

fn probe_client(proxy: &str) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(PROBE_TIMEOUT_SECS));
    if !proxy.is_empty() {
        let proxy_url = absolute_base_url(proxy);
        let proxy = reqwest::Proxy::all(&proxy_url).map_err(|e| e.to_string())?;
        builder = builder.proxy(proxy);
    }
    builder.build().map_err(|e| e.to_string())
}

/// Send a one-token chat completion (mirrors Python `probe_llm_provider`:
/// `max_tokens=1`, `messages=[user "ping"]`).
async fn probe_deployment(d: &ResolvedDeployment) -> ProbeOutcome {
    let client = match probe_client(&d.proxy) {
        Ok(c) => c,
        Err(e) => {
            return ProbeOutcome {
                ok: false,
                failure_kind: "bad_request",
                message: e,
                first_response_ms: None,
                total_ms: None,
            }
        }
    };
    let url = format!("{}/chat/completions", absolute_base_url(&d.base_url).trim_end_matches('/'));
    let start = Instant::now();
    let body = json!({
        "model": d.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": false,
    });
    let mut request = client.post(&url).json(&body);
    if !d.api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", d.api_key));
    }
    match request.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let first_response_ms = Some(start.elapsed().as_millis() as u64);
            let _ = resp.text().await;
            let total_ms = Some(start.elapsed().as_millis() as u64);
            if (200..300).contains(&status) {
                ProbeOutcome {
                    ok: true,
                    failure_kind: "",
                    message: String::new(),
                    first_response_ms,
                    total_ms,
                }
            } else if status == 401 || status == 403 {
                ProbeOutcome {
                    ok: false,
                    failure_kind: "auth_invalid",
                    message: format!("Provider rejected the credential (HTTP {status})."),
                    first_response_ms,
                    total_ms,
                }
            } else {
                ProbeOutcome {
                    ok: false,
                    failure_kind: "bad_request",
                    message: format!("Provider returned HTTP {status}."),
                    first_response_ms,
                    total_ms,
                }
            }
        }
        Err(e) => {
            let total_ms = Some(start.elapsed().as_millis() as u64);
            ProbeOutcome {
                ok: false,
                failure_kind: "transport_transient",
                message: redact_error_text(&e.to_string(), &d.api_key),
                first_response_ms: None,
                total_ms,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/// Map one catalog row to the WebUI wire shape consumed by
/// `normalizeDiscoveredModels`.
fn model_to_wire(m: &ModelCapabilities) -> serde_json::Value {
    let pricing = (m.input_price_per_million.is_some() || m.output_price_per_million.is_some())
        .then(|| {
            json!({
                "inputPer1k": m.input_price_per_million.unwrap_or(0.0) / 1000.0,
                "outputPer1k": m.output_price_per_million.unwrap_or(0.0) / 1000.0,
            })
        });
    json!({
        "id": &m.model,
        "name": m.label.clone().unwrap_or_else(|| m.model.clone()),
        "contextWindow": m.context_window,
        "maxOutputTokens": m.max_output_tokens,
        "capabilities": &m.tags,
        "capabilitySource": "live",
        "metadata": serde_json::Value::Null,
        "pricing": pricing,
    })
}

/// Fetch `GET {base_url}/models` and normalize the rows. `source` is `"live"`
/// only when the provider listed at least one model; otherwise `"none"` (still
/// `ok=true`, mirroring the Python discover contract).
async fn discover_deployment_models(d: &ResolvedDeployment) -> ModelsDiscoverResponse {
    let client = match probe_client(&d.proxy) {
        Ok(c) => c,
        Err(e) => {
            return ModelsDiscoverResponse {
                ok: false,
                provider_id: d.provider.clone(),
                failure_kind: "bad_request".to_string(),
                detail: e,
                source: "none".to_string(),
                models: Vec::new(),
                catalog: None,
            }
        }
    };
    let url = format!("{}/models", absolute_base_url(&d.base_url).trim_end_matches('/'));
    let mut request = client.get(&url);
    if !d.api_key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", d.api_key));
    }
    let models = match request.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            if !(200..300).contains(&status) {
                let kind = if status == 401 || status == 403 {
                    "auth_invalid"
                } else {
                    "bad_request"
                };
                return ModelsDiscoverResponse {
                    ok: false,
                    provider_id: d.provider.clone(),
                    failure_kind: kind.to_string(),
                    detail: format!("Provider returned HTTP {status}."),
                    source: "none".to_string(),
                    models: Vec::new(),
                    catalog: None,
                };
            }
            resp.json().await.unwrap_or_else(|_| serde_json::Value::Null)
        }
        Err(e) => {
            return ModelsDiscoverResponse {
                ok: false,
                provider_id: d.provider.clone(),
                failure_kind: "transport_transient".to_string(),
                detail: redact_error_text(&e.to_string(), &d.api_key),
                source: "none".to_string(),
                models: Vec::new(),
                catalog: None,
            }
        }
    };
    let rows: Vec<serde_json::Value> = parse_models_response(&d.provider, &models)
        .iter()
        .map(model_to_wire)
        .collect();
    ModelsDiscoverResponse {
        ok: true,
        provider_id: d.provider.clone(),
        failure_kind: String::new(),
        detail: String::new(),
        source: if rows.is_empty() { "none".to_string() } else { "live".to_string() },
        models: rows,
        catalog: None,
    }
}

/// Curated image-model rows from the provider's static registry models.
fn curated_image_models(spec: &ProviderSpec) -> Vec<serde_json::Value> {
    let row = |id: &str| {
        json!({
            "id": id,
            "name": id,
            "contextWindow": serde_json::Value::Null,
            "maxOutputTokens": serde_json::Value::Null,
            "capabilities": [],
            "capabilitySource": "curated",
            "metadata": serde_json::Value::Null,
            "pricing": serde_json::Value::Null,
        })
    };
    spec.models.iter().map(|m| row(m)).collect()
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

/// `onboarding.provider.probe` — test the candidate (or current) deployment.
#[tauri::command]
pub async fn onboarding_provider_probe(
    state: State<'_, AppState>,
    request: ProviderProbeRequest,
) -> TauriResult<ProbeResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(
            &cfg,
            &provider,
            &spec,
            &overrides(
                request.model.as_deref(),
                request.api_key.as_deref(),
                request.api_key_env.as_deref(),
                request.base_url.as_deref(),
                request.proxy.as_deref(),
                false,
            ),
        )
    };
    if deployment.model.is_empty() {
        return Err(TauriError::bad_request("model is required for a provider probe"));
    }
    if requires_api_key(&spec) && deployment.api_key.is_empty() {
        return Ok(ProbeResponse {
            ok: false,
            provider: provider.clone(),
            model: deployment.model,
            failure_kind: "auth_invalid".to_string(),
            message: format!(
                "No API key available (checked ${} and the stored config).",
                provider_env_key(&provider)
            ),
            first_response_ms: None,
            total_ms: Some(0),
            latency_ms: Some(0),
        });
    }
    let result = probe_deployment(&deployment).await;
    Ok(ProbeResponse {
        ok: result.ok,
        provider: provider.clone(),
        model: deployment.model,
        failure_kind: result.failure_kind.to_string(),
        message: result.message,
        first_response_ms: result.first_response_ms,
        total_ms: result.total_ms,
        latency_ms: result.total_ms,
    })
}

/// `onboarding.llmProfile.probe` — test a stored profile's deployment.
#[tauri::command]
pub async fn onboarding_llm_profile_probe(
    state: State<'_, AppState>,
    request: LlmProfileProbeRequest,
) -> TauriResult<ProbeResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(
            &cfg,
            &provider,
            &spec,
            &overrides(request.model.as_deref(), None, None, None, None, true),
        )
    };
    if deployment.model.is_empty() {
        return Err(TauriError::bad_request("model is required for a profile probe"));
    }
    if requires_api_key(&spec) && deployment.api_key.is_empty() {
        return Ok(ProbeResponse {
            ok: false,
            provider: provider.clone(),
            model: deployment.model,
            failure_kind: "auth_invalid".to_string(),
            message: format!("No API key available (checked ${}).", provider_env_key(&provider)),
            first_response_ms: None,
            total_ms: Some(0),
            latency_ms: Some(0),
        });
    }
    let result = probe_deployment(&deployment).await;
    Ok(ProbeResponse {
        ok: result.ok,
        provider: provider.clone(),
        model: deployment.model,
        failure_kind: result.failure_kind.to_string(),
        message: result.message,
        first_response_ms: result.first_response_ms,
        total_ms: result.total_ms,
        latency_ms: result.total_ms,
    })
}

/// `onboarding.llmProfile.draft.probe` — test the editor's unsaved draft.
#[tauri::command]
pub async fn onboarding_llm_profile_draft_probe(
    state: State<'_, AppState>,
    request: LlmProfileDraftProbeRequest,
) -> TauriResult<ProbeResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(
            &cfg,
            &provider,
            &spec,
            &overrides(
                request.model.as_deref(),
                request.api_key.as_deref(),
                request.api_key_env.as_deref(),
                request.base_url.as_deref(),
                request.proxy.as_deref(),
                request.keep_current_secret,
            ),
        )
    };
    if deployment.model.is_empty() {
        return Err(TauriError::bad_request("model is required for a profile draft probe"));
    }
    if requires_api_key(&spec) && deployment.api_key.is_empty() {
        return Ok(ProbeResponse {
            ok: false,
            provider: provider.clone(),
            model: deployment.model,
            failure_kind: "auth_invalid".to_string(),
            message: format!("No API key available (checked ${}).", provider_env_key(&provider)),
            first_response_ms: None,
            total_ms: Some(0),
            latency_ms: Some(0),
        });
    }
    let result = probe_deployment(&deployment).await;
    Ok(ProbeResponse {
        ok: result.ok,
        provider: provider.clone(),
        model: deployment.model,
        failure_kind: result.failure_kind.to_string(),
        message: result.message,
        first_response_ms: result.first_response_ms,
        total_ms: result.total_ms,
        latency_ms: result.total_ms,
    })
}

/// `onboarding.models.discover` — list models for a candidate deployment.
#[tauri::command]
pub async fn onboarding_models_discover(
    state: State<'_, AppState>,
    request: ModelsDiscoverRequest,
) -> TauriResult<ModelsDiscoverResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(
            &cfg,
            &provider,
            &spec,
            &overrides(
                request.model.as_deref(),
                request.api_key.as_deref(),
                request.api_key_env.as_deref(),
                request.base_url.as_deref(),
                request.proxy.as_deref(),
                false,
            ),
        )
    };
    let _ = request.force_refresh; // live fetch is always fresh; kept for contract parity
    Ok(discover_deployment_models(&deployment).await)
}

/// `onboarding.llmProfile.models.discover` — list a stored profile's models.
#[tauri::command]
pub async fn onboarding_llm_profile_models_discover(
    state: State<'_, AppState>,
    request: LlmProfileModelsDiscoverRequest,
) -> TauriResult<ModelsDiscoverResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(&cfg, &provider, &spec, &overrides(None, None, None, None, None, true))
    };
    Ok(discover_deployment_models(&deployment).await)
}

/// `onboarding.llmProfile.draft.models.discover` — list a draft's models.
#[tauri::command]
pub async fn onboarding_llm_profile_draft_models_discover(
    state: State<'_, AppState>,
    request: LlmProfileDraftModelsDiscoverRequest,
) -> TauriResult<ModelsDiscoverResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let deployment = {
        let cfg = state.config().await;
        resolve_deployment(
            &cfg,
            &provider,
            &spec,
            &overrides(
                request.model.as_deref(),
                request.api_key.as_deref(),
                request.api_key_env.as_deref(),
                request.base_url.as_deref(),
                request.proxy.as_deref(),
                request.keep_current_secret,
            ),
        )
    };
    Ok(discover_deployment_models(&deployment).await)
}

/// `onboarding.imageGeneration.models.discover` — curated image models for a
/// provider (offline-safe; no network).
#[tauri::command]
pub async fn onboarding_image_generation_models_discover(
    state: State<'_, AppState>,
    request: ImageGenerationModelsDiscoverRequest,
) -> TauriResult<ModelsDiscoverResponse> {
    let provider = request.provider_id.trim().to_lowercase();
    if provider.is_empty() {
        return Err(TauriError::bad_request("providerId is required"));
    }
    let _ = state;
    let spec = match provider_spec(&provider) {
        Some(s) => s,
        None => return Err(TauriError::bad_request(format!("unknown provider {provider:?}"))),
    };
    let models = curated_image_models(&spec);
    Ok(ModelsDiscoverResponse {
        ok: true,
        provider_id: provider.clone(),
        failure_kind: String::new(),
        detail: String::new(),
        source: if models.is_empty() { "none".to_string() } else { "curated".to_string() },
        models,
        catalog: None,
    })
}

// ---------------------------------------------------------------------------
// Tests (pure helpers only — no network)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_provider::registry::ProviderSpecTable;

    fn spec(provider: &str) -> ProviderSpec {
        ProviderSpecTable::get(provider).expect("known provider")
    }

    #[test]
    fn endpoint_origin_ignores_path_and_scheme() {
        assert_eq!(endpoint_origin("https://api.openai.com/v1"), "api.openai.com");
        assert_eq!(endpoint_origin("https://api.openai.com/v1/"), "api.openai.com");
        assert_eq!(endpoint_origin("http://localhost:11434"), "localhost:11434");
        assert!(same_endpoint_origin("https://api.openai.com/v1", "https://api.openai.com/v2"));
        assert!(!same_endpoint_origin("https://api.openai.com", "https://api.anthropic.com"));
    }

    #[test]
    fn first_nonempty_skips_blank() {
        assert_eq!(first_nonempty(&["", "  ", "openai", ""]), "openai");
        assert_eq!(first_nonempty(&["", ""]), "");
    }

    #[test]
    fn resolve_deployment_prefers_request_over_stored() {
        let cfg = Config {
            llm: Some(LlmConfig {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                base_url: "https://api.openai.com/v1".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let o = DeploymentOverrides {
            base_url: "https://proxy.example.com".to_string(),
            ..Default::default()
        };
        let d = resolve_deployment(&cfg, "openai", &spec("openai"), &o);
        assert_eq!(d.base_url, "https://proxy.example.com");
        assert_eq!(d.model, "gpt-4.1"); // stored model, no override
    }

    #[test]
    fn resolve_deployment_does_not_reuse_key_for_foreign_endpoint() {
        let cfg = Config {
            llm: Some(LlmConfig {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                api_key: Some("sk-stored".to_string()),
                base_url: "https://api.openai.com/v1".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let o = DeploymentOverrides {
            base_url: "https://other.example.com".to_string(),
            ..Default::default()
        };
        let d = resolve_deployment(&cfg, "openai", &spec("openai"), &o);
        assert!(d.api_key.is_empty());
    }

    #[test]
    fn redact_sentinel_mask_treated_as_absent() {
        let cfg = Config::default();
        let o = DeploymentOverrides {
            api_key: "***".to_string(),
            ..Default::default()
        };
        let d = resolve_deployment(&cfg, "openai", &spec("openai"), &o);
        assert!(d.api_key.is_empty());
    }
}
