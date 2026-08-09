//! Onboarding catalog and status commands (S3 read-only rescue).
//!
//! Provides the `onboarding.catalog` and `onboarding.status` RPC methods over
//! the Tauri bridge so the WebUI setup wizard can open without
//! `METHOD_NOT_FOUND`. Both are strictly read-only:
//!
//! - `onboarding_catalog` derives the provider dropdown from the static
//!   provider registry ([`ProviderSpecTable::all`]).
//! - `onboarding_status` derives configuration state from the single
//!   source-of-truth [`Config`] held in `AppState`.
//!
//! No network requests, no writes, no dependence on the (deprecated)
//! `ConfigStore` overlay.

use crate::error::TauriResult;
use crate::state::AppState;
use opensquilla_core::config::{Config, LlmProfile};
use opensquilla_provider::registry::{AuthScheme, ProviderSpec, ProviderSpecTable};
use serde::Serialize;
use std::collections::HashMap;
use tauri::State;

// ---------------------------------------------------------------------------
// Catalog DTOs
// ---------------------------------------------------------------------------

/// One provider catalog entry, aligned with the WebUI `ProviderSpec` type.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderSpecDto {
    provider_id: String,
    label: String,
    runtime_supported: bool,
    router_supported: bool,
    fields: Vec<FieldSpecDto>,
    what_you_need: Vec<String>,
    env_key: String,
    accepts_api_key: bool,
    requires_api_key: bool,
    default_base_url: String,
    default_direct_model: String,
    default_model: String,
    suggested_models: Vec<String>,
    deployment: String,
    presets: Vec<serde_json::Value>,
}

/// One setup form field, aligned with the WebUI `FieldSpec` type.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldSpecDto {
    name: String,
    label: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    field_type: Option<String>,
    required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    secret: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    choices: Option<Vec<String>>,
}

/// Router tier profile catalog, aligned with the WebUI `routerProfiles` shape.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouterProfilesDto {
    profiles: Vec<RouterProfileDto>,
    default_tier: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouterProfileDto {
    provider_id: String,
    tiers: HashMap<String, serde_json::Value>,
}

/// `onboarding.catalog` response, aligned with the WebUI `OnboardingCatalog`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnboardingCatalogPayload {
    providers: Vec<ProviderSpecDto>,
    router_profiles: RouterProfilesDto,
    search_providers: Vec<serde_json::Value>,
    image_generation_providers: Vec<serde_json::Value>,
    memory_embedding_providers: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Status DTOs
// ---------------------------------------------------------------------------

/// Per-section detail card, aligned with the WebUI `SectionDetail`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionDetailDto {
    status: String,
    blocking: bool,
    action_required: bool,
    required: bool,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    router_binding: Option<String>,
}

/// Primary LLM credential status, aligned with the WebUI `llmCredentialStatus`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCredentialStatusDto {
    provider: String,
    available: bool,
    source: String,
    env_key: String,
    masked: String,
    reveal_allowed: bool,
}

/// One stored profile status row, aligned with the WebUI `llmProfileStatus`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmProfileStatusDto {
    provider: String,
    ready: bool,
    credential_source: String,
    credential_env: String,
    endpoint_source: String,
    proxy_source: String,
    reason: String,
    primary_eligible: bool,
    primary_block_reason: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityConfigDto {
    resettable: bool,
}

/// `onboarding.status` response, aligned with the WebUI `OnboardingStatus`.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OnboardingStatusPayload {
    needs_onboarding: bool,
    has_config: bool,
    llm_configured: bool,
    llm_source: String,
    llm_env_key: String,
    section_details: HashMap<String, SectionDetailDto>,
    env_recovery_commands: Vec<serde_json::Value>,
    config_path: Option<String>,
    channel_count: usize,
    search_configured: bool,
    search_provider: String,
    search_source: String,
    search_env_key: String,
    image_generation_enabled: bool,
    image_generation_configured: bool,
    image_generation_source: String,
    image_generation_env_key: String,
    image_generation_provider: String,
    image_generation_primary: String,
    memory_embedding_configured: bool,
    memory_embedding_source: String,
    memory_embedding_env_key: String,
    memory_embedding_provider: String,
    audio_configured: bool,
    audio_enabled: bool,
    audio_source: String,
    audio_env_key: String,
    capability_configuration: HashMap<String, CapabilityConfigDto>,
    llm_credential_status: LlmCredentialStatusDto,
    llm_profile_status: Vec<LlmProfileStatusDto>,
    ensemble_credential_status: Vec<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Catalog command
// ---------------------------------------------------------------------------

/// `onboarding.catalog` — the provider/setup catalog for the WebUI wizard.
#[tauri::command]
pub async fn onboarding_catalog() -> TauriResult<OnboardingCatalogPayload> {
    Ok(catalog_payload())
}

fn catalog_payload() -> OnboardingCatalogPayload {
    let providers = ProviderSpecTable::all()
        .iter()
        .map(provider_entry_payload)
        .collect();
    OnboardingCatalogPayload {
        providers,
        // The Rust runtime ships no curated router tier profiles yet; an empty
        // catalog keeps the Model Strategy panel in its provider-first state.
        router_profiles: RouterProfilesDto {
            profiles: Vec::new(),
            default_tier: "c1".to_string(),
        },
        search_providers: Vec::new(),
        image_generation_providers: Vec::new(),
        memory_embedding_providers: Vec::new(),
    }
}

const LOCAL_PROVIDER_IDS: &[&str] = &[
    "ollama",
    "localai",
    "llama_cpp",
    "vllm",
    "lm_studio",
    "ovms",
    "local",
    "custom",
];

fn is_local_provider(provider_id: &str) -> bool {
    LOCAL_PROVIDER_IDS.contains(&provider_id)
}

/// Provider-id -> registry environment-variable name. Mirrors the Python
/// `PROVIDER_ENV_KEYS` / `registry.py` env keys for the well-known providers;
/// unknown ids return an empty key (the WebUI falls back to its own label).
fn env_key_for(provider_id: &str) -> String {
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
        "volcengine" | "volcengine_coding_plan" | "byteplus_coding_plan" => {
            "VOLCENGINE_API_KEY"
        }
        _ => "",
    }
    .to_string()
}

fn requires_base_url(spec: &ProviderSpec) -> bool {
    // Providers with a placeholder host (e.g. Azure's <resource>) need an
    // operator-supplied base URL.
    spec.api_base.contains("://<") || spec.api_base.contains("<resource>")
}

fn what_you_need(spec: &ProviderSpec) -> Vec<String> {
    let requires_key = spec.auth != AuthScheme::None;
    let env_key = env_key_for(spec.id);
    let needs_base_url = requires_base_url(spec);
    let local = is_local_provider(spec.id);
    let mut needs = Vec::new();
    if local {
        needs.push("A local model name available from your model server.".to_string());
    } else {
        needs.push("A provider model id.".to_string());
    }
    if requires_key {
        if env_key.is_empty() {
            needs.push("Provider API key.".to_string());
        } else {
            needs.push(format!("API key via {env_key} or a one-time paste."));
        }
    }
    if needs_base_url {
        needs.push("Provider base URL.".to_string());
    }
    if local {
        needs.push("A reachable local model server.".to_string());
    }
    if needs.is_empty() {
        needs.push("No API key required for the default local path.".to_string());
    }
    needs
}

fn fields_for(spec: &ProviderSpec) -> Vec<FieldSpecDto> {
    let requires_key = spec.auth != AuthScheme::None;
    let env_key = env_key_for(spec.id);
    let needs_base_url = requires_base_url(spec);
    let local = is_local_provider(spec.id);
    let mut fields = Vec::new();

    fields.push(FieldSpecDto {
        name: "model".to_string(),
        label: "Model id".to_string(),
        field_type: Some("text".to_string()),
        required: true,
        default: Some(serde_json::Value::String(spec.default_model.to_string())),
        description: Some(if local {
            "Required local model id. Use a model available from your local model server."
                .to_string()
        } else {
            "Required model id for this provider.".to_string()
        }),
        secret: false,
        choices: None,
    });

    if requires_key {
        fields.push(FieldSpecDto {
            name: "api_key".to_string(),
            label: "API key".to_string(),
            field_type: Some("password".to_string()),
            required: true,
            default: None,
            description: Some(if env_key.is_empty() {
                "Saved as plaintext api_key in the config file.".to_string()
            } else {
                format!(
                    "Saved as plaintext api_key in the config file and used ahead \
                     of {env_key}. Leave blank to read {env_key} from the environment instead."
                )
            }),
            secret: true,
            choices: None,
        });
        fields.push(FieldSpecDto {
            name: "api_key_env".to_string(),
            label: "API key env".to_string(),
            field_type: Some("text".to_string()),
            required: false,
            default: Some(serde_json::Value::String(env_key.clone())),
            description: Some(
                "Environment variable name the gateway reads for this key.".to_string(),
            ),
            secret: false,
            choices: None,
        });
    }

    fields.push(FieldSpecDto {
        name: "base_url".to_string(),
        label: "Base URL".to_string(),
        field_type: Some("text".to_string()),
        required: needs_base_url,
        default: Some(serde_json::Value::String(spec.api_base.to_string())),
        description: Some("Override the upstream HTTP base URL.".to_string()),
        secret: false,
        choices: None,
    });

    fields.push(FieldSpecDto {
        name: "proxy".to_string(),
        label: "HTTP proxy".to_string(),
        field_type: Some("text".to_string()),
        required: false,
        default: None,
        description: Some(
            "Optional explicit HTTP proxy URL (e.g. http://127.0.0.1:7890).".to_string(),
        ),
        secret: false,
        choices: None,
    });

    fields
}

fn provider_entry_payload(spec: &ProviderSpec) -> ProviderSpecDto {
    let requires_key = spec.auth != AuthScheme::None;
    let deployment = if is_local_provider(spec.id) {
        "local"
    } else if requires_base_url(spec) {
        "custom"
    } else {
        "cloud"
    };
    let default_model = spec.default_model.to_string();
    ProviderSpecDto {
        provider_id: spec.id.to_string(),
        label: spec.display_name.to_string(),
        runtime_supported: true,
        router_supported: false,
        fields: fields_for(spec),
        what_you_need: what_you_need(spec),
        env_key: env_key_for(spec.id),
        accepts_api_key: requires_key,
        requires_api_key: requires_key,
        default_base_url: spec.api_base.to_string(),
        default_direct_model: default_model.clone(),
        default_model,
        suggested_models: spec.models.iter().map(|m| m.to_string()).collect(),
        deployment: deployment.to_string(),
        presets: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Status command
// ---------------------------------------------------------------------------

/// `onboarding.status` — current configuration state for the WebUI wizard.
#[tauri::command]
pub async fn onboarding_status(state: State<'_, AppState>) -> TauriResult<OnboardingStatusPayload> {
    let config = state.config().await;
    Ok(status_payload(&config))
}

fn status_payload(config: &Config) -> OnboardingStatusPayload {
    let discovered = Config::discover_path().ok();
    let has_config = discovered.is_some();
    let config_path = discovered.map(|p| p.to_string_lossy().to_string());

    let (llm_configured, llm_source, llm_env_key, llm_credential_status) =
        llm_credential_status(config);

    let search_provider = config.search_provider.clone().unwrap_or_default();
    let (search_configured, search_source, search_env_key) =
        search_annotations(config, &search_provider);

    let (image_enabled, image_configured, image_source, image_env_key, image_provider, image_primary) =
        image_annotations(config);

    let (mem_provider, mem_configured, mem_source, mem_env_key) =
        memory_embedding_annotations(config);

    let (audio_enabled, audio_configured, audio_source, audio_env_key) =
        audio_annotations(config);

    let section_details = build_section_details(
        config,
        llm_configured,
        &llm_source,
        &llm_env_key,
        search_configured,
        &search_source,
        &search_env_key,
        image_enabled,
        image_configured,
        &image_source,
        &image_env_key,
        &mem_provider,
        mem_configured,
        &mem_source,
        &mem_env_key,
        audio_enabled,
        audio_configured,
        &audio_source,
        &audio_env_key,
    );

    let mut capability_configuration = HashMap::new();
    for id in ["search", "image_generation", "audio", "memory_embedding"] {
        capability_configuration.insert(
            id.to_string(),
            CapabilityConfigDto { resettable: false },
        );
    }

    OnboardingStatusPayload {
        needs_onboarding: !llm_configured,
        has_config,
        llm_configured,
        llm_source,
        llm_env_key,
        section_details,
        env_recovery_commands: Vec::new(),
        config_path,
        channel_count: config.channels.len(),
        search_configured,
        search_provider,
        search_source,
        search_env_key,
        image_generation_enabled: image_enabled,
        image_generation_configured: image_configured,
        image_generation_source: image_source,
        image_generation_env_key: image_env_key,
        image_generation_provider: image_provider,
        image_generation_primary: image_primary,
        memory_embedding_configured: mem_configured,
        memory_embedding_source: mem_source,
        memory_embedding_env_key: mem_env_key,
        memory_embedding_provider: mem_provider,
        audio_configured,
        audio_enabled,
        audio_source,
        audio_env_key,
        capability_configuration,
        llm_credential_status,
        llm_profile_status: llm_profile_status(config),
        ensemble_credential_status: Vec::new(),
    }
}

fn env_var_set(name: &str) -> bool {
    !name.is_empty() && std::env::var_os(name).map(|v| !v.is_empty()).unwrap_or(false)
}

fn mask_credential(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let n = chars.len();
    if n == 0 {
        return String::new();
    }
    if n <= 4 {
        return "*".repeat(n);
    }
    let tail: String = chars[n - 4..].iter().collect();
    format!("{}{}", "*".repeat(n - 4), tail)
}

fn source_detail(source: &str, env_key: &str) -> String {
    match source {
        "explicit" => "stored key".to_string(),
        "env" => {
            if env_key.is_empty() {
                "env key visible".to_string()
            } else {
                format!("env key visible: {env_key}")
            }
        }
        "missing_env" => {
            if env_key.is_empty() {
                "env key not visible".to_string()
            } else {
                format!("env key not visible: {env_key}")
            }
        }
        "not_required" => "no key required".to_string(),
        "unsupported" => "registered but not runtime-supported".to_string(),
        _ => String::new(),
    }
}

fn llm_credential_status(
    config: &Config,
) -> (bool, String, String, LlmCredentialStatusDto) {
    let empty = || LlmCredentialStatusDto {
        provider: String::new(),
        available: false,
        source: "none".to_string(),
        env_key: String::new(),
        masked: String::new(),
        reveal_allowed: false,
    };
    let Some(llm) = &config.llm else {
        return (false, "none".to_string(), String::new(), empty());
    };
    let provider = llm.provider.trim().to_lowercase();
    if provider.is_empty() || llm.model.trim().is_empty() {
        return (
            false,
            "none".to_string(),
            String::new(),
            LlmCredentialStatusDto {
                provider,
                available: false,
                source: "none".to_string(),
                env_key: String::new(),
                masked: String::new(),
                reveal_allowed: false,
            },
        );
    }
    let spec = ProviderSpecTable::get(&provider);
    let requires_key = match &spec {
        Some(s) => s.auth != AuthScheme::None,
        None => true,
    };
    let registry_env_key = spec.as_ref().map(|s| env_key_for(s.id)).unwrap_or_default();
    if !requires_key {
        return (
            true,
            "not_required".to_string(),
            registry_env_key.clone(),
            LlmCredentialStatusDto {
                provider,
                available: true,
                source: "not_required".to_string(),
                env_key: registry_env_key,
                masked: String::new(),
                reveal_allowed: false,
            },
        );
    }
    if let Some(key) = llm.api_key.as_deref() {
        if !key.is_empty() {
            let env_key = llm
                .api_key_env
                .clone()
                .unwrap_or_else(|| registry_env_key.clone());
            let masked = mask_credential(key);
            return (
                true,
                "explicit".to_string(),
                env_key.clone(),
                LlmCredentialStatusDto {
                    provider,
                    available: true,
                    source: "explicit".to_string(),
                    env_key,
                    masked,
                    reveal_allowed: false,
                },
            );
        }
    }
    if let Some(env_name) = llm.api_key_env.as_deref() {
        if !env_name.is_empty() {
            let source = if env_var_set(env_name) { "env" } else { "missing_env" };
            let available = source == "env";
            return (
                available,
                source.to_string(),
                env_name.to_string(),
                LlmCredentialStatusDto {
                    provider,
                    available,
                    source: source.to_string(),
                    env_key: env_name.to_string(),
                    masked: String::new(),
                    reveal_allowed: false,
                },
            );
        }
    }
    if !registry_env_key.is_empty() {
        if env_var_set(&registry_env_key) {
            return (
                true,
                "env".to_string(),
                registry_env_key.clone(),
                LlmCredentialStatusDto {
                    provider,
                    available: true,
                    source: "env".to_string(),
                    env_key: registry_env_key,
                    masked: String::new(),
                    reveal_allowed: false,
                },
            );
        }
        return (
            false,
            "missing_env".to_string(),
            registry_env_key.clone(),
            LlmCredentialStatusDto {
                provider,
                available: false,
                source: "missing_env".to_string(),
                env_key: registry_env_key,
                masked: String::new(),
                reveal_allowed: false,
            },
        );
    }
    (
        false,
        "none".to_string(),
        String::new(),
        LlmCredentialStatusDto {
            provider,
            available: false,
            source: "none".to_string(),
            env_key: String::new(),
            masked: String::new(),
            reveal_allowed: false,
        },
    )
}

fn search_annotations(config: &Config, provider: &str) -> (bool, String, String) {
    if provider.is_empty() {
        return (false, "none".to_string(), String::new());
    }
    let source = if config
        .search_api_key
        .as_deref()
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        "explicit"
    } else if let Some(env) = config.search_api_key_env.as_deref() {
        if env.is_empty() {
            "none"
        } else if env_var_set(env) {
            "env"
        } else {
            "missing_env"
        }
    } else if provider == "duckduckgo" {
        "not_required"
    } else {
        "none"
    };
    let configured = matches!(source, "explicit" | "env" | "not_required");
    let env_key = config.search_api_key_env.clone().unwrap_or_default();
    (configured, source.to_string(), env_key)
}

fn image_annotations(config: &Config) -> (bool, bool, String, String, String, String) {
    let Some(image) = &config.image_generation else {
        return (false, false, String::new(), String::new(), String::new(), String::new());
    };
    let enabled = image.enabled;
    let configured = !image.providers.is_empty()
        && image
            .providers
            .values()
            .any(|p| p.api_key.is_some() || p.api_key_env.is_some());
    let source = if configured { "explicit" } else { "" }.to_string();
    let env_key = image
        .providers
        .values()
        .find_map(|p| p.api_key_env.clone())
        .unwrap_or_default();
    let provider = image
        .providers
        .keys()
        .next()
        .cloned()
        .unwrap_or_default();
    (enabled, configured, source, env_key, provider, image.primary.clone())
}

fn memory_embedding_annotations(config: &Config) -> (String, bool, String, String) {
    let Some(embedding) = config.memory.as_ref().map(|m| &m.embedding) else {
        return (String::new(), false, "none".to_string(), String::new());
    };
    let provider = if !embedding.provider.trim().is_empty() {
        embedding.provider.clone()
    } else {
        embedding.mode.clone().unwrap_or_default()
    };
    let source = if provider.is_empty() {
        "none"
    } else if matches!(provider.as_str(), "none" | "auto" | "local" | "ollama") {
        "not_required"
    } else if embedding
        .api_key
        .as_deref()
        .map(|k| !k.is_empty())
        .unwrap_or(false)
        || embedding
            .remote
            .api_key
            .as_deref()
            .map(|k| !k.is_empty())
            .unwrap_or(false)
    {
        "explicit"
    } else if let Some(env) = embedding.remote.api_key_env.as_deref() {
        if env.is_empty() {
            "none"
        } else if env_var_set(env) {
            "env"
        } else {
            "missing_env"
        }
    } else {
        "none"
    };
    let configured = matches!(source, "explicit" | "env" | "not_required");
    let env_key = embedding.remote.api_key_env.clone().unwrap_or_default();
    (provider, configured, source.to_string(), env_key)
}

fn audio_annotations(config: &Config) -> (bool, bool, String, String) {
    let Some(audio) = &config.audio else {
        return (false, false, String::new(), String::new());
    };
    let enabled = audio.enabled;
    let configured = audio
        .providers
        .values()
        .any(|p| p.api_key.is_some() || p.api_key_env.is_some());
    let source = if configured { "explicit" } else { "" }.to_string();
    let env_key = audio
        .providers
        .values()
        .find_map(|p| p.api_key_env.clone())
        .unwrap_or_default();
    (enabled, configured, source, env_key)
}

#[allow(clippy::too_many_arguments)]
fn build_section_details(
    config: &Config,
    llm_configured: bool,
    llm_source: &str,
    llm_env_key: &str,
    search_configured: bool,
    search_source: &str,
    search_env_key: &str,
    image_enabled: bool,
    image_configured: bool,
    image_source: &str,
    image_env_key: &str,
    mem_provider: &str,
    mem_configured: bool,
    mem_source: &str,
    mem_env_key: &str,
    audio_enabled: bool,
    audio_configured: bool,
    audio_source: &str,
    audio_env_key: &str,
) -> HashMap<String, SectionDetailDto> {
    let mut details = HashMap::new();

    details.insert(
        "llm".to_string(),
        SectionDetailDto {
            status: if llm_configured { "ok" } else { "missing" }.to_string(),
            blocking: !llm_configured,
            action_required: !llm_configured,
            required: true,
            label: "Provider".to_string(),
            detail: Some(source_detail(llm_source, llm_env_key)),
            router_mode: None,
            router_binding: None,
        },
    );

    let router = config.squilla_router.as_ref();
    let router_enabled = router.map(|r| r.enabled).unwrap_or(false);
    let router_binding = router
        .and_then(|r| r.preset_binding.clone())
        .filter(|b| b == "follow_primary" || b == "custom")
        .unwrap_or_else(|| "legacy".to_string());
    let router_mode = if !router_enabled {
        "disabled"
    } else if router_binding == "follow_primary" {
        "recommended"
    } else if router_binding == "custom" {
        "custom"
    } else {
        "recommended"
    };
    details.insert(
        "router".to_string(),
        SectionDetailDto {
            status: if router_enabled { "ok" } else { "optional" }.to_string(),
            blocking: false,
            action_required: false,
            required: false,
            label: "Router".to_string(),
            detail: Some(if router_enabled {
                "SquillaRouter enabled".to_string()
            } else {
                "disabled".to_string()
            }),
            router_mode: Some(router_mode.to_string()),
            router_binding: Some(router_binding),
        },
    );

    let ensemble_enabled = config
        .llm_ensemble
        .as_ref()
        .map(|e| e.enabled)
        .unwrap_or(false);
    details.insert(
        "ensemble".to_string(),
        SectionDetailDto {
            status: if ensemble_enabled { "ok" } else { "optional" }.to_string(),
            blocking: false,
            action_required: false,
            required: false,
            label: "LLM ensemble".to_string(),
            detail: Some(if ensemble_enabled {
                "enabled".to_string()
            } else {
                "disabled".to_string()
            }),
            router_mode: None,
            router_binding: None,
        },
    );

    details.insert(
        "search".to_string(),
        SectionDetailDto {
            status: if search_configured { "ok" } else { "optional" }.to_string(),
            blocking: false,
            action_required: false,
            required: false,
            label: "Web search".to_string(),
            detail: Some(source_detail(search_source, search_env_key)),
            router_mode: None,
            router_binding: None,
        },
    );

    let image_action = image_enabled && !image_configured;
    details.insert(
        "image_generation".to_string(),
        SectionDetailDto {
            status: if !image_enabled {
                "optional"
            } else if image_configured {
                "ok"
            } else {
                "missing"
            }
            .to_string(),
            blocking: image_action,
            action_required: image_action,
            required: false,
            label: "Image generation".to_string(),
            detail: Some(source_detail(image_source, image_env_key)),
            router_mode: None,
            router_binding: None,
        },
    );

    let mem_action = !mem_provider.is_empty() && !mem_configured;
    details.insert(
        "memory_embedding".to_string(),
        SectionDetailDto {
            status: if mem_configured {
                "ok"
            } else if mem_provider.is_empty() {
                "optional"
            } else {
                "missing"
            }
            .to_string(),
            blocking: mem_action,
            action_required: mem_action,
            required: false,
            label: "Memory embedding".to_string(),
            detail: Some(source_detail(mem_source, mem_env_key)),
            router_mode: None,
            router_binding: None,
        },
    );

    let audio_action = audio_enabled && !audio_configured;
    details.insert(
        "audio".to_string(),
        SectionDetailDto {
            status: if !audio_enabled {
                "optional"
            } else if audio_configured {
                "ok"
            } else {
                "missing"
            }
            .to_string(),
            blocking: audio_action,
            action_required: audio_action,
            required: false,
            label: "Voice audio".to_string(),
            detail: Some(source_detail(audio_source, audio_env_key)),
            router_mode: None,
            router_binding: None,
        },
    );

    details
}

fn profile_has_credential(profile: &LlmProfile) -> bool {
    if profile
        .api_key
        .as_deref()
        .map(|k| !k.is_empty())
        .unwrap_or(false)
    {
        return true;
    }
    if let Some(env) = profile.api_key_env.as_deref() {
        if !env.is_empty() && env_var_set(env) {
            return true;
        }
    }
    false
}

fn llm_profile_status(config: &Config) -> Vec<LlmProfileStatusDto> {
    let Some(profiles) = &config.llm_profiles else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for (provider_id, profile) in profiles {
        let ready = profile_has_credential(profile);
        let credential_source = if profile
            .api_key
            .as_deref()
            .map(|k| !k.is_empty())
            .unwrap_or(false)
        {
            "profile"
        } else if profile
            .api_key_env
            .as_deref()
            .map(|e| !e.is_empty())
            .unwrap_or(false)
        {
            "profile_env"
        } else {
            "none"
        };
        let endpoint_source = if profile
            .base_url
            .as_deref()
            .map(|b| !b.is_empty())
            .unwrap_or(false)
        {
            "explicit"
        } else {
            ""
        };
        let proxy_source = if profile
            .proxy
            .as_deref()
            .map(|p| !p.is_empty())
            .unwrap_or(false)
        {
            "explicit"
        } else {
            ""
        };
        rows.push(LlmProfileStatusDto {
            provider: provider_id.clone(),
            ready,
            credential_source: credential_source.to_string(),
            credential_env: profile.api_key_env.clone().unwrap_or_default(),
            endpoint_source: endpoint_source.to_string(),
            proxy_source: proxy_source.to_string(),
            reason: if ready {
                String::new()
            } else {
                "credential_unavailable".to_string()
            },
            // S3 is read-only: activation/removal RPCs are not implemented yet,
            // so no stored profile is offered as an activation candidate.
            primary_eligible: false,
            primary_block_reason: "profile_status_unavailable".to_string(),
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_credential_masks_tail() {
        assert_eq!(mask_credential(""), "");
        assert_eq!(mask_credential("abcd"), "****");
        assert_eq!(mask_credential("abcdef"), "**cdef");
    }

    #[test]
    fn env_key_mapping() {
        assert_eq!(env_key_for("openai"), "OPENAI_API_KEY");
        assert_eq!(env_key_for("openrouter"), "OPENROUTER_API_KEY");
        assert_eq!(env_key_for("ollama"), "OLLAMA_API_KEY");
        assert_eq!(env_key_for("unknown_provider"), "");
    }

    #[test]
    fn catalog_contains_known_providers() {
        let payload = catalog_payload();
        let ids: Vec<&str> = payload.providers.iter().map(|p| p.provider_id.as_str()).collect();
        assert!(ids.contains(&"openai"));
        assert!(ids.contains(&"openrouter"));
        assert!(ids.contains(&"ollama"));
        assert!(payload.router_profiles.profiles.is_empty());
    }

    #[test]
    fn pristine_status_is_unconfigured() {
        let config = Config::default();
        let payload = status_payload(&config);
        assert!(!payload.llm_configured);
        assert!(payload.needs_onboarding);
        assert_eq!(payload.llm_source, "none");
        assert_eq!(payload.channel_count, 0);
        assert_eq!(payload.capability_configuration.len(), 4);
        assert_eq!(payload.section_details.len(), 7);
    }

    #[test]
    fn explicit_llm_credential_is_configured() {
        let mut config = Config::default();
        config.llm = Some(opensquilla_core::config::LlmConfig {
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
        let payload = status_payload(&config);
        assert!(payload.llm_configured);
        assert_eq!(payload.llm_source, "explicit");
        assert!(!payload.needs_onboarding);
        assert!(payload.llm_credential_status.available);
        assert_eq!(payload.llm_credential_status.source, "explicit");
    }
}
