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
        vec![
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
        ]
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
