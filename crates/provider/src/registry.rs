//! Provider registry and declarative provider specs.
//!
//! Two concerns live here:
//!
//! - [`ProviderSpec`]: a declarative description of a provider backend
//!   (id, backend type, API base, auth scheme, default model). The static
//!   table of 50+ specs mirrors the Python `registry.py`.
//! - [`ProviderRegistry`]: a thread-safe runtime registry of *instantiated*
//!   providers (`Arc<dyn Provider>`), backed by `DashMap`.
//!
//! The specs are data; the registry holds live objects. A bootstrap helper
//! ([`ProviderSpecTable::instantiate`]) can turn specs into providers given
//! credentials.

use crate::anthropic::AuthHeaderStyle;
use crate::types::Provider;
use std::sync::Arc;
use tracing::info;

// ---------------------------------------------------------------------------
// Backend types
// ---------------------------------------------------------------------------

/// The five backend types, mirroring the Python adapter layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// OpenAI-compatible chat completions (`/v1/chat/completions`).
    OpenAiCompat,
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Responses API (`/v1/responses`).
    OpenAiResponses,
    /// OpenAI Codex API (`/v1/codex`).
    OpenAiCodex,
    /// Ollama local API.
    Ollama,
    /// Multi-model ensemble (proposer-aggregator).
    Ensemble,
}

impl BackendType {
    /// The string identifier used in specs.
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendType::OpenAiCompat => "openai_compat",
            BackendType::Anthropic => "anthropic",
            BackendType::OpenAiResponses => "openai_responses",
            BackendType::OpenAiCodex => "openai_codex",
            BackendType::Ollama => "ollama",
            BackendType::Ensemble => "ensemble",
        }
    }
}

/// How a provider authenticates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthScheme {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `x-api-key: <key>` (Anthropic).
    HeaderApiKey,
    /// `api-key: <key>` (Azure).
    AzureApiKey,
    /// No authentication (local providers like Ollama).
    None,
}

// ---------------------------------------------------------------------------
// Python-parity capability / metadata types
// ---------------------------------------------------------------------------
//
// The types below mirror the Python `opensquilla.provider` registry metadata
// (`registry.py`, `context_capabilities.py`, `compat_policy.py`). They are
// carried as extra declarative fields on [`ProviderSpec`] so the static table
// can express the same per-provider capabilities the Python registry does.

/// Whether a provider's live model listing is a verified source of
/// user-selectable model ids (Python `SelectableModelCatalog`).
///
/// ``none`` is deliberately the default: several compatibility adapters
/// expose static or protocol-family model rows that do not describe what the
/// configured service actually serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectableModelCatalog {
    /// No live listing is trusted; fall back to the static model rows.
    #[default]
    None,
    /// The provider's live `/models` listing has been verified as a safe
    /// source for user-selectable model ids.
    VerifiedLive,
}

impl SelectableModelCatalog {
    /// The string identifier used in serialized metadata.
    pub fn as_str(&self) -> &'static str {
        match self {
            SelectableModelCatalog::None => "none",
            SelectableModelCatalog::VerifiedLive => "verified_live",
        }
    }
}

/// Prompt-cache support level (Python `PromptCacheSupport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptCacheSupport {
    /// No prompt caching.
    #[default]
    None,
    /// Caching happens implicitly upstream (Gemini).
    Implicit,
    /// Caching is controlled with explicit `cache_control` breakpoints.
    Explicit,
    /// Caching is automatic and cannot be influenced from the client.
    Automatic,
}

/// Native context-compaction support (Python `NativeCompactionSupport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NativeCompactionSupport {
    /// No native compaction.
    #[default]
    None,
    /// The provider can compact the conversation window server-side.
    Standalone,
}

/// Provider-keyed context / prompt-cache capability profile (Python
/// `ProviderContextProfile`).
///
/// Only capabilities keyed purely on the provider identity live here; request
/// host-guarded branches (Gemini's ``generativelanguage`` endpoint, OpenAI's
/// ``api.openai.com`` guard) deliberately stay as code.
#[derive(Debug, Clone)]
pub struct ContextProfile {
    /// Baseline prompt-cache support.
    pub prompt_cache: PromptCacheSupport,
    /// Native context-compaction support.
    pub native_compaction: NativeCompactionSupport,
    /// State-kind string when native compaction produces a resumable state.
    pub native_compaction_state_kind: Option<&'static str>,
    /// Whether explicit `cache_control` breakpoints are honored.
    pub supports_cache_breakpoints: bool,
    /// Whether the provider state can cross to another provider.
    pub state_portable_across_providers: bool,
    /// Minimum tokens before caching starts.
    pub min_cache_tokens: Option<u32>,
    /// Accepted TTL options for cached contexts.
    pub cache_ttl_options: &'static [u32],
    /// Per-model-id-prefix prompt-cache overrides, first match wins.
    pub prompt_cache_model_prefix_table: &'static [(&'static str, PromptCacheSupport)],
    /// Per-model-basename prompt-cache overrides, consulted after the prefix table.
    pub prompt_cache_model_name_prefix_table: &'static [(&'static str, PromptCacheSupport)],
}

impl Default for ContextProfile {
    fn default() -> Self {
        Self {
            prompt_cache: PromptCacheSupport::None,
            native_compaction: NativeCompactionSupport::None,
            native_compaction_state_kind: None,
            supports_cache_breakpoints: false,
            state_portable_across_providers: false,
            min_cache_tokens: None,
            cache_ttl_options: &[],
            prompt_cache_model_prefix_table: &[],
            prompt_cache_model_name_prefix_table: &[],
        }
    }
}

/// Anthropic's provider-keyed context profile (Python `ANTHROPIC_CONTEXT_PROFILE`).
pub const ANTHROPIC_CONTEXT_PROFILE: ContextProfile = ContextProfile {
    prompt_cache: PromptCacheSupport::Explicit,
    native_compaction: NativeCompactionSupport::None,
    native_compaction_state_kind: None,
    supports_cache_breakpoints: true,
    state_portable_across_providers: false,
    min_cache_tokens: None,
    cache_ttl_options: &[],
    prompt_cache_model_prefix_table: &[],
    prompt_cache_model_name_prefix_table: &[],
};

/// OpenAI Responses context profile (Python `OPENAI_RESPONSES_CONTEXT_PROFILE`).
pub const OPENAI_RESPONSES_CONTEXT_PROFILE: ContextProfile = ContextProfile {
    prompt_cache: PromptCacheSupport::Automatic,
    native_compaction: NativeCompactionSupport::Standalone,
    native_compaction_state_kind: Some("openai_responses_compacted_window"),
    supports_cache_breakpoints: false,
    state_portable_across_providers: false,
    min_cache_tokens: None,
    cache_ttl_options: &[],
    prompt_cache_model_prefix_table: &[],
    prompt_cache_model_name_prefix_table: &[],
};

/// OpenRouter context profile (Python `OPENROUTER_CONTEXT_PROFILE`).
pub const OPENROUTER_CONTEXT_PROFILE: ContextProfile = ContextProfile {
    prompt_cache: PromptCacheSupport::Implicit,
    native_compaction: NativeCompactionSupport::None,
    native_compaction_state_kind: None,
    supports_cache_breakpoints: false,
    state_portable_across_providers: false,
    min_cache_tokens: None,
    cache_ttl_options: &[],
    prompt_cache_model_prefix_table: &[
        ("anthropic/", PromptCacheSupport::Explicit),
        ("google/", PromptCacheSupport::Explicit),
        ("deepseek/", PromptCacheSupport::Explicit),
        ("x-ai/", PromptCacheSupport::Explicit),
        ("z-ai/", PromptCacheSupport::Implicit),
    ],
    prompt_cache_model_name_prefix_table: &[
        ("qwen3.6-flash", PromptCacheSupport::Explicit),
        ("qwen3.5-flash", PromptCacheSupport::Explicit),
        ("qwen3-coder", PromptCacheSupport::Explicit),
    ],
};

/// A single model rule authorizing text-tool dialects (Python `TextToolModelRule`).
#[derive(Debug, Clone)]
pub struct TextToolModelRule {
    /// Model-id glob patterns this rule applies to (lowercase fnmatch).
    pub model_patterns: &'static [&'static str],
    /// Text-tool dialect identifiers granted to matching models.
    pub dialects: &'static [&'static str],
}

/// Declarative per-kind dialect policy for OpenAI-compatible providers
/// (Python `OpenAICompatPolicy`).
///
/// Ported from `compat_policy.py`. The `text_tool_profile` nested type is
/// kept as raw dialect/rule lists (see [`TextToolModelRule`]); fnmatch-style
/// resolution is left to the request builder.
#[derive(Debug, Clone)]
pub struct OpenAiCompatPolicy {
    /// Human-readable name used in error messages.
    pub display_name: &'static str,
    /// Host marker gating quirks that only apply to the official endpoint.
    pub official_host: &'static str,
    /// Models taking `max_completion_tokens` instead of `max_tokens`.
    pub max_completion_tokens_model_prefixes: &'static [&'static str],
    /// Models whose sampling is fixed upstream.
    pub fixed_sampling_model_prefixes: &'static [&'static str],
    /// Models rejecting temperature while extended thinking is active.
    pub omit_temperature_when_thinking_model_prefixes: &'static [&'static str],
    /// JSON Schema keywords the upstream rejects in tool definitions.
    pub tool_schema_unsupported_keywords: &'static [&'static str],
    /// Whether the endpoint reliably supports native `response_format.type=json_schema`.
    pub supports_native_json_schema_output: bool,
    /// Whether the endpoint supports `response_format.type=json_object`.
    pub supports_json_object_output: bool,
    /// Whether `usage.cost` from this upstream is authoritative billing data.
    pub trust_billed_cost: bool,
    /// Whether the request should send OpenRouter-family `usage` extras.
    pub sends_usage_include: bool,
    /// Whether the request may pin a specific provider route.
    pub supports_provider_routing_pin: bool,
    /// Whether the endpoint supports explicit prompt-cache control.
    pub supports_explicit_prompt_cache: bool,
    /// Whether cache breakpoints sit at the top level (Anthropic-style).
    pub anthropic_top_level_cache: bool,
    /// Whether a stream-timeout fallback is applied.
    pub stream_timeout_fallback: bool,
    /// Whether an empty stream is retried/fallback.
    pub empty_stream_fallback: bool,
    /// Whether the payload logs the cache shape.
    pub log_payload_cache_shape: bool,
    /// Whether the endpoint may repeat an already-observed terminal choice.
    pub allow_post_terminal_noop_choice: bool,
    /// Narrower opt-in: an empty choice with `usage: null` before the trailer.
    pub allow_post_terminal_null_usage_noop_choice: bool,
    /// Provider-specific top-level metadata keys on the terminal epilogue.
    pub post_terminal_metadata_keys: &'static [&'static str],
    /// Whether the request disables gateway cross-model fallbacks.
    pub sends_disable_fallbacks: bool,
    /// Response headers reporting the deployment that served the request.
    pub attribution_response_headers: &'static [&'static str],
    /// Reasoning-continuity format to replay when capabilities declare it.
    pub replay_reasoning_format: &'static str,
    /// Reasoning format assumed when no capability profile is available.
    pub default_reasoning_format: &'static str,
    /// Exact ids needing an explicit thinking enable/disable payload.
    pub thinking_toggle_model_ids: &'static [&'static str],
    /// Exact ids requiring `reasoning_content` on every assistant message.
    pub require_reasoning_content_model_ids: &'static [&'static str],
    /// Exact ids that stream reasoning by default and need explicit disable.
    pub disable_reasoning_by_default_models: &'static [&'static str],
    /// Model-id prefixes rejecting `enable_thinking=False`.
    pub thinking_required_model_prefixes: &'static [&'static str],
    /// Exact forced-thinking ids for multi-family endpoints.
    pub force_thinking_model_ids: &'static [&'static str],
    /// Exact ids whose requests opt into reasoning continuity.
    pub preserve_thinking_model_ids: &'static [&'static str],
    /// Reasoning models requiring `reasoning_content` only while thinking.
    pub require_reasoning_content_when_thinking_model_ids: &'static [&'static str],
    /// Tool-call subset requiring `reasoning_content` while thinking.
    pub require_tool_call_reasoning_content_when_thinking_model_ids: &'static [&'static str],
    /// Thinking-mode tool choice accepts only auto/none.
    pub thinking_tool_choice_auto_only: bool,
    /// Exact ids that are reasoning-only upstream without a toggle.
    pub implicit_thinking_tool_choice_model_ids: &'static [&'static str],
    /// Preserve a pinned tool selector by disabling thinking instead.
    pub prefer_pinned_tool_choice_over_thinking: bool,
    /// Exact ids requiring `tool_stream=True` whenever tools are present.
    pub tool_stream_model_ids: &'static [&'static str],
    /// Thinking-only models imposing a minimum sampling temperature.
    pub temperature_floor_model_ids: &'static [&'static str],
    /// The minimum sampling temperature for those models.
    pub temperature_floor: f64,
    /// Model ids excluded from a mixed /models listing.
    pub model_listing_excluded_ids: &'static [&'static str],
    /// Omit a framework-default thinking budget.
    pub omit_implicit_thinking_budget: bool,
    /// Provider-wide text-tool dialect identifiers (additive).
    pub text_tool_dialects: &'static [&'static str],
    /// Model-scoped text-tool dialect rules (additive, first match wins).
    pub text_tool_model_rules: &'static [TextToolModelRule],
}

impl OpenAiCompatPolicy {
    /// The default policy used when no kind-specific entry is registered.
    pub const DEFAULT: Self = Self {
        display_name: "Provider",
        official_host: "",
        max_completion_tokens_model_prefixes: &[],
        fixed_sampling_model_prefixes: &[],
        omit_temperature_when_thinking_model_prefixes: &[],
        tool_schema_unsupported_keywords: &[],
        supports_native_json_schema_output: true,
        supports_json_object_output: false,
        trust_billed_cost: false,
        sends_usage_include: false,
        supports_provider_routing_pin: false,
        supports_explicit_prompt_cache: false,
        anthropic_top_level_cache: false,
        stream_timeout_fallback: false,
        empty_stream_fallback: false,
        log_payload_cache_shape: false,
        allow_post_terminal_noop_choice: false,
        allow_post_terminal_null_usage_noop_choice: false,
        post_terminal_metadata_keys: &[],
        sends_disable_fallbacks: false,
        attribution_response_headers: &[],
        replay_reasoning_format: "",
        default_reasoning_format: "",
        thinking_toggle_model_ids: &[],
        require_reasoning_content_model_ids: &[],
        disable_reasoning_by_default_models: &[],
        thinking_required_model_prefixes: &[],
        force_thinking_model_ids: &[],
        preserve_thinking_model_ids: &[],
        require_reasoning_content_when_thinking_model_ids: &[],
        require_tool_call_reasoning_content_when_thinking_model_ids: &[],
        thinking_tool_choice_auto_only: false,
        implicit_thinking_tool_choice_model_ids: &[],
        prefer_pinned_tool_choice_over_thinking: false,
        tool_stream_model_ids: &[],
        temperature_floor_model_ids: &[],
        temperature_floor: 0.0,
        model_listing_excluded_ids: &[],
        omit_implicit_thinking_budget: false,
        text_tool_dialects: &[],
        text_tool_model_rules: &[],
    };
}

impl Default for OpenAiCompatPolicy {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ---------------------------------------------------------------------------
// ProviderSpec
// ---------------------------------------------------------------------------

/// A declarative description of a registered provider.
///
/// Specs are pure data: they describe *how* to instantiate a provider but do
/// not hold credentials or an HTTP client. The static table
/// ([`ProviderSpecTable::all`]) contains 50+ entries.
#[derive(Debug, Clone)]
pub struct ProviderSpec {
    /// The canonical provider id (e.g. "openai", "deepseek").
    pub id: &'static str,
    /// The backend type.
    pub backend: BackendType,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// The default API base URL.
    pub api_base: &'static str,
    /// The authentication scheme.
    pub auth: AuthScheme,
    /// The default model id for this provider.
    pub default_model: &'static str,
    /// A list of well-known model ids for this provider.
    pub models: &'static [&'static str],
    /// Whether the provider is enabled by default.
    pub enabled: bool,
    /// Optional documentation URL.
    pub docs_url: Option<&'static str>,
    // --- Python-parity metadata (Python `registry.py` ProviderSpec) --------
    //
    // The fields below mirror the Python provider registry metadata. They are
    // optional / defaulted so the existing static specs keep constructing with
    // the compact `spec()` helper. ProviderSpec is a static table and is not
    // serialized; if serde derives are added later, `#[serde(default)]` should
    // be restored on these fields.
    /// Provider-keyed context/prompt-cache capability profile
    /// (Python `context_profile`).
    pub context_profile: Option<ContextProfile>,
    /// models.dev provider ids feeding the vendored model catalog snapshot
    /// (Python `catalog_source`).
    pub catalog_source: &'static [&'static str],
    /// Capability flags, e.g. "chat", "coding_plan", "responses"
    /// (Python `capabilities`).
    pub capabilities: &'static [&'static str],
    /// Failure classification family, e.g. "openai_compat", "anthropic",
    /// "ollama" (Python `failure_family`).
    pub failure_family: &'static str,
    /// Auth header shape for the Anthropic backend (Python `auth_header_style`).
    pub auth_header_style: AuthHeaderStyle,
    /// Reasoning wire-format shape, e.g. "none", "deepseek", "gemini", "zai"
    /// (Python `reasoning_shape`).
    pub reasoning_shape: &'static str,
    /// Per-kind OpenAI-compatible dialect policy (Python `compat`).
    pub compat: OpenAiCompatPolicy,
    /// Keyless public model-listing endpoint for boot-time live catalog ingest
    /// (Python `live_catalog_url`).
    pub live_catalog_url: &'static str,
    /// Whether the provider's live listing is trusted for user selection
    /// (Python `selectable_model_catalog`).
    pub selectable_model_catalog: SelectableModelCatalog,
    /// Sibling provider id used to discover account entitlements
    /// (Python `selectable_model_discovery_provider_id`).
    pub selectable_model_discovery_provider_id: &'static str,
    /// Exact model list for transports without a trustworthy `/models`
    /// endpoint (Python `static_model_ids`).
    pub static_model_ids: &'static [&'static str],
}

impl ProviderSpec {
    /// Create a new spec with the given id and backend.
    pub const fn new(id: &'static str, backend: BackendType) -> Self {
        Self {
            id,
            backend,
            display_name: id,
            api_base: "",
            auth: AuthScheme::Bearer,
            default_model: "",
            models: &[],
            enabled: true,
            docs_url: None,
            context_profile: None,
            catalog_source: &[],
            capabilities: &["chat"],
            failure_family: "openai_compat",
            auth_header_style: AuthHeaderStyle::Bearer,
            reasoning_shape: "none",
            compat: OpenAiCompatPolicy::DEFAULT,
            live_catalog_url: "",
            selectable_model_catalog: SelectableModelCatalog::None,
            selectable_model_discovery_provider_id: "",
            static_model_ids: &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Static spec table — 50+ providers
// ---------------------------------------------------------------------------

/// The static table of provider specs.
pub struct ProviderSpecTable;

impl ProviderSpecTable {
    /// Return all registered provider specs.
    pub fn all() -> Vec<ProviderSpec> {
        let mut specs = vec![
            // --- OpenAI compat backend (30+) -------------------------------
            spec(
                "openai",
                BackendType::OpenAiCompat,
                "OpenAI",
                "https://api.openai.com/v1",
                AuthScheme::Bearer,
                "gpt-4o",
                &[
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    "o1",
                    "o3-mini",
                ],
            ),
            spec(
                "deepseek",
                BackendType::OpenAiCompat,
                "DeepSeek",
                "https://api.deepseek.com/v1",
                AuthScheme::Bearer,
                "deepseek-chat",
                &["deepseek-chat", "deepseek-reasoner"],
            ),
            spec(
                "gemini",
                BackendType::OpenAiCompat,
                "Google Gemini",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                AuthScheme::Bearer,
                "gemini-2.0-flash",
                &["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            ),
            spec(
                "dashscope",
                BackendType::OpenAiCompat,
                "Alibaba DashScope",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                AuthScheme::Bearer,
                "qwen-plus",
                &["qwen-max", "qwen-plus", "qwen-turbo"],
            ),
            spec(
                "qwen",
                BackendType::OpenAiCompat,
                "Qwen",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                AuthScheme::Bearer,
                "qwen-plus",
                &["qwen-max", "qwen-plus", "qwen-turbo"],
            ),
            spec(
                "moonshot",
                BackendType::OpenAiCompat,
                "Moonshot (Kimi)",
                "https://api.moonshot.cn/v1",
                AuthScheme::Bearer,
                "moonshot-v1-128k",
                &["moonshot-v1-128k", "kimi-k1.5"],
            ),
            spec(
                "mistral",
                BackendType::OpenAiCompat,
                "Mistral AI",
                "https://api.mistral.ai/v1",
                AuthScheme::Bearer,
                "mistral-large-latest",
                &[
                    "mistral-large-latest",
                    "mistral-small-latest",
                    "mixtral-8x7b",
                ],
            ),
            spec(
                "groq",
                BackendType::OpenAiCompat,
                "Groq",
                "https://api.groq.com/openai/v1",
                AuthScheme::Bearer,
                "llama-3.3-70b-versatile",
                &[
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                ],
            ),
            spec(
                "zhipu",
                BackendType::OpenAiCompat,
                "Zhipu (GLM)",
                "https://open.bigmodel.cn/api/paas/v4",
                AuthScheme::Bearer,
                "glm-4-plus",
                &["glm-4-plus", "glm-4", "glm-4-flash"],
            ),
            spec(
                "siliconflow",
                BackendType::OpenAiCompat,
                "SiliconFlow",
                "https://api.siliconflow.cn/v1",
                AuthScheme::Bearer,
                "Qwen/Qwen2.5-72B-Instruct",
                &["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"],
            ),
            spec(
                "openrouter",
                BackendType::OpenAiCompat,
                "OpenRouter",
                "https://openrouter.ai/api/v1",
                AuthScheme::Bearer,
                "openai/gpt-4o",
                &[
                    "openai/gpt-4o",
                    "anthropic/claude-3.5-sonnet",
                    "google/gemini-2.0-flash",
                ],
            ),
            spec(
                "azure",
                BackendType::OpenAiCompat,
                "Azure OpenAI",
                "https://<resource>.openai.azure.com",
                AuthScheme::AzureApiKey,
                "gpt-4o",
                &["gpt-4o", "gpt-4", "gpt-35-turbo"],
            ),
            spec(
                "together",
                BackendType::OpenAiCompat,
                "Together AI",
                "https://api.together.xyz/v1",
                AuthScheme::Bearer,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                &["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
            ),
            spec(
                "fireworks",
                BackendType::OpenAiCompat,
                "Fireworks AI",
                "https://api.fireworks.ai/inference/v1",
                AuthScheme::Bearer,
                "accounts/fireworks/models/llama-v3p3-70b-instruct",
                &["accounts/fireworks/models/llama-v3p3-70b-instruct"],
            ),
            spec(
                "anyscale",
                BackendType::OpenAiCompat,
                "Anyscale",
                "https://api.endpoints.anyscale.com/v1",
                AuthScheme::Bearer,
                "meta-llama/Llama-3.1-70B-Instruct",
                &["meta-llama/Llama-3.1-70B-Instruct"],
            ),
            spec(
                "lepton",
                BackendType::OpenAiCompat,
                "Lepton AI",
                "https://api.lepton.ai/api/v1",
                AuthScheme::Bearer,
                "llama3-8b",
                &["llama3-8b", "llama3-70b"],
            ),
            spec(
                "perplexity",
                BackendType::OpenAiCompat,
                "Perplexity",
                "https://api.perplexity.ai",
                AuthScheme::Bearer,
                "llama-3.1-sonar-large-128k-online",
                &[
                    "llama-3.1-sonar-large-128k-online",
                    "llama-3.1-sonar-small-128k-online",
                ],
            ),
            spec(
                "novita",
                BackendType::OpenAiCompat,
                "Novita AI",
                "https://api.novita.ai/v3/openai",
                AuthScheme::Bearer,
                "meta-llama/llama-3.1-70b-instruct",
                &["meta-llama/llama-3.1-70b-instruct"],
            ),
            spec(
                "huggingface",
                BackendType::OpenAiCompat,
                "Hugging Face",
                "https://api-inference.huggingface.co/v1",
                AuthScheme::Bearer,
                "meta-llama/Llama-3.3-70B-Instruct",
                &["meta-llama/Llama-3.3-70B-Instruct"],
            ),
            spec(
                "infermatic",
                BackendType::OpenAiCompat,
                "Infermatic",
                "https://api.infermatic.ai/v1",
                AuthScheme::Bearer,
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct"],
            ),
            spec(
                "ai21",
                BackendType::OpenAiCompat,
                "AI21 Labs",
                "https://api.ai21.com/studio/v1",
                AuthScheme::Bearer,
                "jamba-1-5-large",
                &["jamba-1-5-large", "jamba-1-5-mini"],
            ),
            spec(
                "cohere",
                BackendType::OpenAiCompat,
                "Cohere",
                "https://api.cohere.ai/v1",
                AuthScheme::Bearer,
                "command-r-plus",
                &["command-r-plus", "command-r"],
            ),
            spec(
                "xai",
                BackendType::OpenAiCompat,
                "xAI (Grok)",
                "https://api.x.ai/v1",
                AuthScheme::Bearer,
                "grok-2-latest",
                &["grok-2-latest", "grok-beta"],
            ),
            spec(
                "deepinfra",
                BackendType::OpenAiCompat,
                "DeepInfra",
                "https://api.deepinfra.com/v1/openai",
                AuthScheme::Bearer,
                "meta-llama/Llama-3.3-70B-Instruct",
                &["meta-llama/Llama-3.3-70B-Instruct"],
            ),
            spec(
                "modelscope",
                BackendType::OpenAiCompat,
                "ModelScope",
                "https://api-inference.modelscope.cn/v1",
                AuthScheme::Bearer,
                "Qwen/Qwen2.5-72B-Instruct",
                &["Qwen/Qwen2.5-72B-Instruct"],
            ),
            spec(
                "yi",
                BackendType::OpenAiCompat,
                "01.AI (Yi)",
                "https://api.01.ai/v1",
                AuthScheme::Bearer,
                "yi-large",
                &["yi-large", "yi-medium"],
            ),
            spec(
                "baichuan",
                BackendType::OpenAiCompat,
                "Baichuan",
                "https://api.baichuan-ai.com/v1",
                AuthScheme::Bearer,
                "Baichuan4",
                &["Baichuan4", "Baichuan3-Turbo"],
            ),
            spec(
                "minimax_text",
                BackendType::OpenAiCompat,
                "MiniMax Text",
                "https://api.minimax.chat/v1",
                AuthScheme::Bearer,
                "abab6.5-chat",
                &["abab6.5-chat", "abab6.5s-chat"],
            ),
            spec(
                "stepfun",
                BackendType::OpenAiCompat,
                "StepFun",
                "https://api.stepfun.com/v1",
                AuthScheme::Bearer,
                "step-2-16k",
                &["step-2-16k", "step-1-8k"],
            ),
            spec(
                "lingyi",
                BackendType::OpenAiCompat,
                "Lingyi (Yi large)",
                "https://api.lingyiwanwu.com/v1",
                AuthScheme::Bearer,
                "yi-large",
                &["yi-large", "yi-medium"],
            ),
            spec(
                "internlm",
                BackendType::OpenAiCompat,
                "InternLM",
                "https://internlm-chat.intern-ai.org.cn/puyu/api/v1",
                AuthScheme::Bearer,
                "internlm2.5-latest",
                &["internlm2.5-latest"],
            ),
            spec(
                "glm",
                BackendType::OpenAiCompat,
                "GLM",
                "https://open.bigmodel.cn/api/paas/v4",
                AuthScheme::Bearer,
                "glm-4",
                &["glm-4", "glm-4-flash"],
            ),
            spec(
                "hunyuan",
                BackendType::OpenAiCompat,
                "Tencent Hunyuan",
                "https://api.hunyuan.cloud.tencent.com/v1",
                AuthScheme::Bearer,
                "hunyuan-pro",
                &["hunyuan-pro", "hunyuan-standard"],
            ),
            spec(
                "tencent_hunyuan",
                BackendType::OpenAiCompat,
                "Tencent Hunyuan (alt)",
                "https://api.hunyuan.cloud.tencent.com/v1",
                AuthScheme::Bearer,
                "hunyuan-pro",
                &["hunyuan-pro"],
            ),
            spec(
                "baidu_ernie",
                BackendType::OpenAiCompat,
                "Baidu ERNIE",
                "https://qianfan.baidubce.com/v2",
                AuthScheme::Bearer,
                "ernie-4.0-8k-latest",
                &["ernie-4.0-8k-latest", "ernie-3.5-8k"],
            ),
            spec(
                "iflytek_spark",
                BackendType::OpenAiCompat,
                "iFlytek Spark",
                "https://spark-api-open.xf-yun.com/v1",
                AuthScheme::Bearer,
                "4.0Ultra",
                &["4.0Ultra", "generalv3.5"],
            ),
            spec(
                "sensetime",
                BackendType::OpenAiCompat,
                "SenseTime",
                "https://api.sensenova.cn/compatible-mode/v1",
                AuthScheme::Bearer,
                "SenseChat-5",
                &["SenseChat-5"],
            ),
            spec(
                "meituan",
                BackendType::OpenAiCompat,
                "Meituan",
                "https://api.meituan.com/v1",
                AuthScheme::Bearer,
                "mao-1",
                &["mao-1"],
            ),
            spec(
                "volcengine",
                BackendType::OpenAiCompat,
                "Volcengine (Doubao)",
                "https://ark.cn-beijing.volces.com/api/v3",
                AuthScheme::Bearer,
                "doubao-pro-32k",
                &["doubao-pro-32k", "doubao-pro-4k"],
            ),
            spec(
                "lambda",
                BackendType::OpenAiCompat,
                "Lambda Labs",
                "https://api.lambdalabs.com/v1",
                AuthScheme::Bearer,
                "hermes3-405b",
                &["hermes3-405b", "llama3.1-405b-instruct"],
            ),
            spec(
                "hyperbolic",
                BackendType::OpenAiCompat,
                "Hyperbolic",
                "https://api.hyperbolic.xyz/v1",
                AuthScheme::Bearer,
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct"],
            ),
            spec(
                "chutes",
                BackendType::OpenAiCompat,
                "Chutes AI",
                "https://api.chutes.ai/v1",
                AuthScheme::Bearer,
                "chutesai/Llama-3.1-8B-Instruct",
                &["chutesai/Llama-3.1-8B-Instruct"],
            ),
            spec(
                "kluster",
                BackendType::OpenAiCompat,
                "Kluster",
                "https://api.kluster.ai/v1",
                AuthScheme::Bearer,
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct"],
            ),
            spec(
                "inference_net",
                BackendType::OpenAiCompat,
                "Inference.net",
                "https://api.inference.net/v1",
                AuthScheme::Bearer,
                "meta-llama/Llama-3.1-70B-Instruct",
                &["meta-llama/Llama-3.1-70B-Instruct"],
            ),
            spec(
                "not_diamond",
                BackendType::OpenAiCompat,
                "Not Diamond",
                "https://api.not-diamond.com/v1",
                AuthScheme::Bearer,
                "not-diamond-auto",
                &["not-diamond-auto"],
            ),
            spec(
                "localai",
                BackendType::OpenAiCompat,
                "LocalAI (self-hosted)",
                "http://localhost:8080/v1",
                AuthScheme::None,
                "gpt-4",
                &["gpt-4", "gpt-3.5-turbo", "llama3.1"],
            ),
            spec(
                "llama_cpp",
                BackendType::OpenAiCompat,
                "llama.cpp server",
                "http://localhost:8080/v1",
                AuthScheme::None,
                "llama-3.1-8b",
                &["llama-3.1-8b", "llama-3.1-70b", "qwen2.5-7b"],
            ),
            spec(
                "vllm",
                BackendType::OpenAiCompat,
                "vLLM",
                "http://localhost:8000/v1",
                AuthScheme::None,
                "meta-llama/Llama-3.1-8B-Instruct",
                &[
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                ],
            ),
            spec(
                "voyage",
                BackendType::OpenAiCompat,
                "Voyage AI",
                "https://api.voyageai.com/v1",
                AuthScheme::Bearer,
                "voyage-3-large",
                &[
                    "voyage-3-large",
                    "voyage-3",
                    "voyage-3-lite",
                    "voyage-code-3",
                ],
            ),
            spec(
                "elevenlabs",
                BackendType::OpenAiCompat,
                "ElevenLabs",
                "https://api.elevenlabs.io/v1",
                AuthScheme::Bearer,
                "eleven_multilingual_v2",
                &["eleven_multilingual_v2", "eleven_turbo_v2_5"],
            ),
            spec(
                "stability",
                BackendType::OpenAiCompat,
                "Stability AI",
                "https://api.stability.ai/v1",
                AuthScheme::Bearer,
                "stable-image-ultra",
                &[
                    "stable-image-ultra",
                    "stable-image-core",
                    "stable-diffusion-xl-1024-v1-0",
                ],
            ),
            // --- OpenAI Responses backend ----------------------------------
            spec(
                "openai_responses",
                BackendType::OpenAiResponses,
                "OpenAI Responses",
                "https://api.openai.com/v1",
                AuthScheme::Bearer,
                "o3",
                &["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o"],
            ),
            spec(
                "volcengine_coding_plan",
                BackendType::OpenAiResponses,
                "Volcengine Coding Plan",
                "https://ark.cn-beijing.volces.com/api/v3",
                AuthScheme::Bearer,
                "doubao-coding",
                &["doubao-coding"],
            ),
            spec(
                "byteplus_coding_plan",
                BackendType::OpenAiResponses,
                "BytePlus Coding Plan",
                "https://ark.byteplus.com/api/v3",
                AuthScheme::Bearer,
                "doubao-coding",
                &["doubao-coding"],
            ),
            // --- OpenAI Codex backend -------------------------------------
            spec(
                "openai_codex",
                BackendType::OpenAiCodex,
                "OpenAI Codex",
                "https://api.openai.com/v1",
                AuthScheme::Bearer,
                "codex-latest",
                &["codex-latest", "codex-mini-latest", "o4-mini"],
            ),
            // --- Anthropic backend ----------------------------------------
            spec(
                "anthropic",
                BackendType::Anthropic,
                "Anthropic",
                "https://api.anthropic.com/v1",
                AuthScheme::HeaderApiKey,
                "claude-3-5-sonnet-20241022",
                &[
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ],
            ),
            spec(
                "minimax",
                BackendType::Anthropic,
                "MiniMax (Anthropic-style)",
                "https://api.minimax.chat/v1",
                AuthScheme::HeaderApiKey,
                "abab6.5-chat",
                &["abab6.5-chat"],
            ),
            spec(
                "qwen_token_plan_anthropic",
                BackendType::Anthropic,
                "Qwen Token Plan (Anthropic)",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                AuthScheme::HeaderApiKey,
                "qwen-plus",
                &["qwen-plus", "qwen-max"],
            ),
            // --- Ollama backend -------------------------------------------
            spec(
                "ollama",
                BackendType::Ollama,
                "Ollama (local)",
                "http://localhost:11434",
                AuthScheme::None,
                "llama3.1",
                &[
                    "llama3.1",
                    "llama3.2",
                    "qwen2.5",
                    "mistral",
                    "deepseek-coder",
                ],
            ),
        ];
        apply_python_parity(&mut specs);
        specs
    }

    /// Look up a spec by id.
    pub fn get(id: &str) -> Option<ProviderSpec> {
        Self::all().into_iter().find(|s| s.id == id)
    }

    /// Return all specs matching a backend type.
    pub fn for_backend(backend: BackendType) -> Vec<ProviderSpec> {
        Self::all()
            .into_iter()
            .filter(|s| s.backend == backend)
            .collect()
    }

    /// Return all provider ids.
    pub fn ids() -> Vec<&'static str> {
        Self::all().into_iter().map(|s| s.id).collect()
    }

    /// The number of registered specs.
    pub fn count() -> usize {
        Self::all().len()
    }

    /// Instantiate a provider from a spec given an API key.
    ///
    /// For the `ensemble` backend this returns `None` (ensembles require
    /// explicit member configuration).
    pub fn instantiate(spec: &ProviderSpec, api_key: &str) -> Option<Arc<dyn Provider>> {
        use crate::anthropic::AnthropicProvider;
        use crate::ollama::OllamaProvider;
        use crate::openai::OpenAiCompatProvider;
        use crate::openai_codex::OpenAICodexProvider;
        use crate::openai_responses::OpenAIResponsesProvider;

        let provider: Arc<dyn Provider> = match spec.backend {
            BackendType::OpenAiCompat => {
                Arc::new(OpenAiCompatProvider::new(spec.id, spec.api_base, api_key))
            }
            BackendType::Anthropic => {
                Arc::new(AnthropicProvider::new(spec.id, spec.api_base, api_key))
            }
            BackendType::OpenAiResponses => Arc::new(OpenAIResponsesProvider::new(
                spec.id,
                spec.api_base,
                api_key,
            )),
            BackendType::OpenAiCodex => {
                Arc::new(OpenAICodexProvider::new(spec.id, spec.api_base, api_key))
            }
            BackendType::Ollama => Arc::new(OllamaProvider::new(spec.id, spec.api_base)),
            BackendType::Ensemble => return None,
        };
        Some(provider)
    }
}

/// Helper to build a spec concisely.
#[allow(clippy::too_many_arguments)]
fn spec(
    id: &'static str,
    backend: BackendType,
    display_name: &'static str,
    api_base: &'static str,
    auth: AuthScheme,
    default_model: &'static str,
    models: &'static [&'static str],
) -> ProviderSpec {
    ProviderSpec {
        id,
        backend,
        display_name,
        api_base,
        auth,
        default_model,
        models,
        enabled: true,
        docs_url: None,
        context_profile: None,
        catalog_source: &[],
        capabilities: &["chat"],
        failure_family: "openai_compat",
        auth_header_style: AuthHeaderStyle::Bearer,
        reasoning_shape: "none",
        compat: OpenAiCompatPolicy::default(),
        live_catalog_url: "",
        selectable_model_catalog: SelectableModelCatalog::None,
        selectable_model_discovery_provider_id: "",
        static_model_ids: &[],
    }
}

/// Apply Python-registry parity metadata (`registry.py` `_spec(...)` calls)
/// that the compact positional `spec()` helper cannot express.
///
/// The static table above mirrors the Python spec table; the Python registry
/// carries extra metadata (reasoning shape, failure family, auth header style,
/// capabilities, catalog sources, context profiles, ...) that this function
/// patches back onto the matching specs so the Rust table is Python-parity.
fn apply_python_parity(specs: &mut [ProviderSpec]) {
    for spec in specs.iter_mut() {
        match spec.id {
            "openrouter" => {
                spec.context_profile = Some(OPENROUTER_CONTEXT_PROFILE);
                spec.catalog_source = &["openrouter"];
                spec.selectable_model_catalog = SelectableModelCatalog::VerifiedLive;
            }
            "openai" => {
                spec.catalog_source = &["openai"];
            }
            "openai_responses" => {
                spec.capabilities = &["chat", "responses"];
                spec.context_profile = Some(OPENAI_RESPONSES_CONTEXT_PROFILE);
                spec.catalog_source = &["openai"];
            }
            "azure" => {
                spec.catalog_source = &["azure"];
            }
            "anthropic" => {
                spec.failure_family = "anthropic";
                spec.auth_header_style = AuthHeaderStyle::XApiKey;
                spec.context_profile = Some(ANTHROPIC_CONTEXT_PROFILE);
                spec.catalog_source = &["anthropic"];
            }
            "ollama" => {
                spec.failure_family = "ollama";
            }
            "deepseek" => {
                spec.reasoning_shape = "deepseek";
                spec.catalog_source = &["deepseek"];
            }
            "gemini" => {
                spec.reasoning_shape = "gemini";
                spec.catalog_source = &["google"];
            }
            "dashscope" => {
                spec.catalog_source = &["alibaba-cn", "alibaba"];
            }
            "moonshot" => {
                spec.catalog_source = &["moonshotai"];
            }
            "minimax" => {
                spec.failure_family = "anthropic";
                spec.auth_header_style = AuthHeaderStyle::Bearer;
                spec.catalog_source = &["minimax"];
            }
            "mistral" => {
                spec.catalog_source = &["mistral"];
            }
            "groq" => {
                spec.catalog_source = &["groq"];
            }
            "zhipu" => {
                spec.reasoning_shape = "zai";
                spec.catalog_source = &["zhipuai", "zai"];
            }
            "siliconflow" => {
                spec.catalog_source = &["siliconflow"];
            }
            "volcengine" => {
                spec.catalog_source = &["volcengine"];
            }
            "openai_codex" => {
                spec.capabilities = &["chat", "coding_plan"];
            }
            "volcengine_coding_plan" => {
                spec.capabilities = &["chat", "coding_plan", "responses"];
            }
            "byteplus_coding_plan" => {
                spec.capabilities = &["chat", "coding_plan", "responses"];
            }
            "qwen_token_plan_anthropic" => {
                spec.failure_family = "anthropic";
                spec.auth_header_style = AuthHeaderStyle::Bearer;
                spec.capabilities = &["chat", "coding_plan"];
                spec.selectable_model_catalog = SelectableModelCatalog::VerifiedLive;
                spec.selectable_model_discovery_provider_id = "qwen_token_plan";
                spec.static_model_ids = &[
                    "qwen3.8-max-preview",
                    "qwen3.7-max",
                    "qwen3.7-plus",
                    "qwen3.6-plus",
                    "qwen3.6-flash",
                    "deepseek-v4-pro",
                    "deepseek-v4-flash",
                    "deepseek-v3.2",
                    "kimi-k2.7-code",
                    "kimi-k2.6",
                    "kimi-k2.5",
                    "glm-5.2",
                    "glm-5.1",
                    "glm-5",
                    "MiniMax-M2.5",
                ];
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime provider registry
// ---------------------------------------------------------------------------

/// A thread-safe registry of named, instantiated providers.
///
/// Providers are stored as `Arc<dyn Provider>` and looked up by name.
#[derive(Clone)]
pub struct ProviderRegistry {
    providers: Arc<dashmap::DashMap<String, Arc<dyn Provider>>>,
    default: Arc<std::sync::RwLock<Option<String>>>,
}

impl ProviderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            providers: Arc::new(dashmap::DashMap::new()),
            default: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    /// Register a provider. If no default has been set, this provider becomes
    /// the default.
    pub fn register(&self, name: String, provider: Arc<dyn Provider>) {
        self.providers.insert(name.clone(), provider);
        let mut default = self.default.write().unwrap();
        if default.is_none() {
            *default = Some(name.clone());
            info!(target = "provider", "Set default provider to '{name}'");
        }
        info!(target = "provider", "Registered provider '{name}'");
    }

    /// Register a provider spec with the given API key, instantiating the
    /// appropriate backend.
    pub fn register_spec(&self, spec: &ProviderSpec, api_key: &str) -> bool {
        if let Some(provider) = ProviderSpecTable::instantiate(spec, api_key) {
            self.register(spec.id.to_string(), provider);
            true
        } else {
            false
        }
    }

    /// Get a provider by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).map(|r| r.clone())
    }

    /// Get the default provider.
    pub fn default(&self) -> Option<Arc<dyn Provider>> {
        let default = self.default.read().unwrap();
        default.as_ref().and_then(|name| self.get(name))
    }

    /// Set the default provider by name.
    pub fn set_default(&self, name: &str) {
        if self.providers.contains_key(name) {
            let mut default = self.default.write().unwrap();
            *default = Some(name.to_string());
            info!(target = "provider", "Default provider set to '{name}'");
        }
    }

    /// List all registered provider names.
    pub fn list(&self) -> Vec<String> {
        self.providers.iter().map(|r| r.key().clone()).collect()
    }

    /// Return the number of registered providers.
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Return `true` if no providers are registered.
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Remove a provider by name.
    pub fn remove(&self, name: &str) -> Option<Arc<dyn Provider>> {
        let removed = self.providers.remove(name).map(|(_, v)| v);
        if removed.is_some() {
            let mut default = self.default.write().unwrap();
            if default.as_deref() == Some(name) {
                *default = self.providers.iter().next().map(|r| r.key().clone());
            }
        }
        removed
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use async_trait::async_trait;
    use futures::Stream;
    use opensquilla_core::types::*;

    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        fn supported_models(&self) -> Vec<String> {
            vec!["mock-model".into()]
        }
        async fn send_message(
            &self,
            _config: &ChatConfig,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> ProviderResult<ProviderResponse> {
            Ok(ProviderResponse {
                content: vec![],
                usage: Usage::default(),
                model: "mock".into(),
                stop_reason: None,
            })
        }
        async fn stream_chat(
            &self,
            _config: &ChatConfig,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>>
        {
            unimplemented!()
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = ProviderRegistry::new();
        registry.register("mock".into(), Arc::new(MockProvider));
        assert_eq!(registry.len(), 1);
        assert!(registry.get("mock").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_default() {
        let registry = ProviderRegistry::new();
        assert!(registry.default().is_none());
        registry.register("p1".into(), Arc::new(MockProvider));
        assert!(registry.default().is_some());
        registry.register("p2".into(), Arc::new(MockProvider));
        registry.set_default("p2");
        assert_eq!(registry.default().unwrap().name(), "mock");
    }

    #[test]
    fn test_spec_table_has_50_plus() {
        let specs = ProviderSpecTable::all();
        assert!(specs.len() >= 50, "expected 50+ specs, got {}", specs.len());
    }

    #[test]
    fn test_spec_table_backends() {
        assert!(ProviderSpecTable::for_backend(BackendType::OpenAiCompat).len() >= 30);
        assert!(ProviderSpecTable::for_backend(BackendType::Anthropic).len() >= 3);
        assert!(ProviderSpecTable::for_backend(BackendType::OpenAiResponses).len() >= 3);
        assert_eq!(
            ProviderSpecTable::for_backend(BackendType::OpenAiCodex).len(),
            1
        );
        assert_eq!(ProviderSpecTable::for_backend(BackendType::Ollama).len(), 1);
    }

    #[test]
    fn test_spec_get() {
        let s = ProviderSpecTable::get("deepseek").unwrap();
        assert_eq!(s.backend, BackendType::OpenAiCompat);
        assert_eq!(s.auth, AuthScheme::Bearer);
        assert_eq!(s.default_model, "deepseek-chat");
    }

    #[test]
    fn test_spec_instantiate_openai_compat() {
        let s = ProviderSpecTable::get("openai").unwrap();
        let p = ProviderSpecTable::instantiate(&s, "sk-test");
        assert!(p.is_some());
        assert_eq!(p.unwrap().name(), "openai");
    }

    #[test]
    fn test_spec_instantiate_ollama() {
        let s = ProviderSpecTable::get("ollama").unwrap();
        let p = ProviderSpecTable::instantiate(&s, "");
        assert!(p.is_some());
        assert_eq!(p.unwrap().name(), "ollama");
    }

    #[test]
    fn test_registry_register_spec() {
        let registry = ProviderRegistry::new();
        let s = ProviderSpecTable::get("openai").unwrap();
        assert!(registry.register_spec(&s, "sk-test"));
        assert_eq!(registry.len(), 1);
        assert!(registry.get("openai").is_some());
    }

    #[test]
    fn test_required_providers_present() {
        let ids = ProviderSpecTable::ids();
        for required in [
            "openai",
            "deepseek",
            "gemini",
            "dashscope",
            "qwen",
            "moonshot",
            "mistral",
            "groq",
            "zhipu",
            "siliconflow",
            "openrouter",
            "azure",
            "anthropic",
            "minimax",
            "ollama",
            "openai_responses",
            "openai_codex",
        ] {
            assert!(ids.contains(&required), "missing provider {required}");
        }
    }
}
