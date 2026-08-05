//! OpenAI-compatible provider backend.
//!
//! This is the most important provider adapter in the OpenSquilla engine: it
//! speaks the OpenAI Chat Completions wire protocol (`/v1/chat/completions`)
//! that 50+ hosted providers all expose — OpenAI, DeepSeek, Google Gemini
//! (via its compatibility shim), Alibaba DashScope / Qwen, Moonshot Kimi,
//! Mistral, Groq, Zhipu GLM, SiliconFlow, OpenRouter, Azure OpenAI, Together,
//! Fireworks, Perplexity, xAI, and many more.
//!
//! The architecture mirrors the Python `openai.py` backend:
//!
//! - [`OpenAIProvider`] is a *declarative* enum of every known provider. Each
//!   variant carries metadata (base URL, auth header shape, default model list,
//!   rate limit, capability flags) through [`ProviderInfo`].
//! - [`OpenAiConfig`] is the user-facing configuration used to construct a
//!   runtime client ([`OpenAiCompatProvider`]).
//! - [`OpenAiCompatProvider`] is the runtime client; it implements the crate-wide
//!   [`Provider`] and [`ChatProvider`] traits and exposes a richer
//!   protocol-specific surface (`complete`, `stream_request`, `list_models`).
//! - [`build_chat_request`] / [`build_chat_request_full`] translate canonical
//!   [`ChatMessage`] arrays into the OpenAI wire format, applying the per-provider
//!   compatibility policy (system-prompt handling, tool-format quirks, parameter
//!   renames and caps) from [`crate::compat_policy`].
//! - [`parse_chat_response`] / [`parse_openai_sse_event`] decode non-streaming
//!   and SSE-streamed responses, including DeepSeek `reasoning_content`, tool-call
//!   deltas, and text-embedded tool calls normalized by
//!   [`crate::text_tool_normalizer`].
//! - Error paths funnel through [`map_http_error`] and
//!   [`classify_openai_error`], which reuse [`crate::failures`] for retry /
//!   recovery decisions.
//!
//! Pre-flight budget projection ([`crate::request_proof`]), credential rotation
//! ([`crate::credentials`]), per-provider rate limiting, and live model catalog
//! integration ([`crate::live_catalog`]) are all wired into the runtime client.

use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use opensquilla_core::types::{
    ChatMessage, ContentBlock, MessageRole, ToolCall, ToolDefinition, Usage,
};
use regex::Regex;
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use tracing::{debug, info, warn};

use crate::compat_policy::{
    adapt_parameters, needs_text_normalization, policy_for, CompatPolicy, ReasoningPolicy,
    SystemPromptPolicy, ToolCallFormat,
};
use crate::credentials::CredentialPool;
use crate::failures::classify as classify_error;
use crate::live_catalog::{LiveCatalog, LiveProviderConfig};
use crate::model_catalog::{merge_live, ModelCapabilities, ModelCatalog};
use crate::request_proof::RequestProof;
use crate::stream::SseStream;
use crate::text_tool_normalizer::{ToolCallNormalizer as TextToolCallNormalizer, ToolDialect};
use crate::types::{
    ChatConfig, Provider, ProviderError, ProviderResponse, ProviderResult, StreamEvent,
};
use crate::util::{with_retry, RateLimiter, RetryConfig};

// ===========================================================================
// Provider enumeration
// ===========================================================================

/// A provider that speaks the OpenAI-compatible Chat Completions protocol.
///
/// This is a declarative identifier: each variant maps to a set of connection
/// defaults via [`OpenAIProvider::info`] (base URL, auth shape, model list,
/// rate limit, compatibility quirks). The variants intentionally mirror the
/// ids used in [`crate::registry::ProviderSpecTable`] and
/// [`crate::compat_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::enum_variant_names)]
pub enum OpenAIProvider {
    /// OpenAI (`api.openai.com`).
    #[serde(rename = "openai")]
    OpenAi,
    /// DeepSeek (`api.deepseek.com`).
    #[serde(rename = "deepseek")]
    DeepSeek,
    /// Google Gemini OpenAI-compat shim.
    #[serde(rename = "gemini")]
    Gemini,
    /// Alibaba DashScope (Qwen) compatible mode.
    #[serde(rename = "dashscope")]
    DashScope,
    /// Qwen (alias of DashScope compatible mode).
    #[serde(rename = "qwen")]
    Qwen,
    /// Moonshot (Kimi).
    #[serde(rename = "moonshot")]
    Moonshot,
    /// Mistral AI La Plateforme.
    #[serde(rename = "mistral")]
    Mistral,
    /// Groq LPU inference.
    #[serde(rename = "groq")]
    Groq,
    /// Zhipu GLM.
    #[serde(rename = "zhipu")]
    Zhipu,
    /// SiliconFlow aggregator.
    #[serde(rename = "siliconflow")]
    SiliconFlow,
    /// OpenRouter model aggregator.
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// Azure OpenAI.
    #[serde(rename = "azure")]
    Azure,
    /// Together AI.
    #[serde(rename = "together")]
    Together,
    /// Fireworks AI.
    #[serde(rename = "fireworks")]
    Fireworks,
    /// Anyscale Endpoints.
    #[serde(rename = "anyscale")]
    Anyscale,
    /// Lepton AI.
    #[serde(rename = "lepton")]
    Lepton,
    /// Replicate (OpenAI-compatible endpoint).
    #[serde(rename = "replicate")]
    Replicate,
    /// Perplexity AI.
    #[serde(rename = "perplexity")]
    Perplexity,
    /// Cohere.
    #[serde(rename = "cohere")]
    Cohere,
    /// AI21 Labs.
    #[serde(rename = "ai21")]
    Ai21,
    /// xAI (Grok).
    #[serde(rename = "xai")]
    Xai,
    /// DeepInfra.
    #[serde(rename = "deepinfra")]
    DeepInfra,
    /// Hugging Face Inference.
    #[serde(rename = "huggingface")]
    HuggingFace,
    /// Novita AI.
    #[serde(rename = "novita")]
    Novita,
    /// Infermatic AI.
    #[serde(rename = "infermatic")]
    Infermatic,
    /// ModelScope.
    #[serde(rename = "modelscope")]
    ModelScope,
    /// 01.AI (Yi).
    #[serde(rename = "yi")]
    Yi,
    /// Baichuan.
    #[serde(rename = "baichuan")]
    Baichuan,
    /// MiniMax (text backend).
    #[serde(rename = "minimax_text")]
    MiniMax,
    /// StepFun.
    #[serde(rename = "stepfun")]
    StepFun,
    /// Lingyi (Yi large).
    #[serde(rename = "lingyi")]
    Lingyi,
    /// InternLM.
    #[serde(rename = "internlm")]
    InternLm,
    /// GLM (open.bigmodel.cn).
    #[serde(rename = "glm")]
    Glm,
    /// Tencent Hunyuan.
    #[serde(rename = "hunyuan")]
    Hunyuan,
    /// Tencent Hunyuan (alternate id).
    #[serde(rename = "tencent_hunyuan")]
    TencentHunyuan,
    /// Baidu ERNIE.
    #[serde(rename = "baidu_ernie")]
    BaiduErnie,
    /// iFlytek Spark.
    #[serde(rename = "iflytek_spark")]
    IflytekSpark,
    /// SenseTime.
    #[serde(rename = "sensetime")]
    SenseTime,
    /// Meituan.
    #[serde(rename = "meituan")]
    Meituan,
    /// Volcengine (Doubao).
    #[serde(rename = "volcengine")]
    Volcengine,
    /// Lambda Labs.
    #[serde(rename = "lambda")]
    Lambda,
    /// Hyperbolic.
    #[serde(rename = "hyperbolic")]
    Hyperbolic,
    /// Chutes AI.
    #[serde(rename = "chutes")]
    Chutes,
    /// Kluster.
    #[serde(rename = "kluster")]
    Kluster,
    /// Inference.net.
    #[serde(rename = "inference_net")]
    InferenceNet,
    /// Not Diamond router.
    #[serde(rename = "not_diamond")]
    NotDiamond,
    /// LocalAI self-hosted.
    #[serde(rename = "localai")]
    LocalAi,
    /// llama.cpp server.
    #[serde(rename = "llama_cpp")]
    LlamaCpp,
    /// vLLM server.
    #[serde(rename = "vllm")]
    Vllm,
    /// Voyage AI.
    #[serde(rename = "voyage")]
    Voyage,
    /// ElevenLabs.
    #[serde(rename = "elevenlabs")]
    ElevenLabs,
    /// Stability AI.
    #[serde(rename = "stability")]
    Stability,
    /// Anthropic (kept for enumeration completeness; normally served by the
    /// dedicated [`crate::anthropic`] backend, not this adapter).
    #[serde(rename = "anthropic")]
    Anthropic,
    /// A user-supplied / unknown OpenAI-compatible endpoint.
    #[serde(rename = "custom")]
    Custom,
}

/// The shape of the authentication header a provider expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthHeader {
    /// `Authorization: Bearer <key>` (the default for OpenAI-compatible APIs).
    Bearer,
    /// `x-api-key: <key>` (Anthropic-style).
    HeaderApiKey,
    /// `api-key: <key>` (Azure OpenAI).
    AzureApiKey,
    /// No authentication (local servers such as Ollama, llama.cpp, vLLM).
    None,
}

/// Static connection metadata for a single OpenAI-compatible provider.
///
/// All fields are `&'static` data so an [`OpenAIProvider`] value can hand out
/// its complete configuration without heap allocation. `requests_per_second`
/// is a conservative default that callers may override via
/// [`OpenAiConfig::with_requests_per_second`].
#[derive(Debug, Clone, Copy)]
pub struct ProviderInfo {
    /// The canonical provider id (matches the compat-policy / registry id).
    pub id: &'static str,
    /// Human-readable display name.
    pub display_name: &'static str,
    /// The default API base URL (no trailing slash).
    pub base_url: &'static str,
    /// The default model id.
    pub default_model: &'static str,
    /// Well-known model ids for this provider.
    pub models: &'static [&'static str],
    /// The authentication header shape.
    pub auth: AuthHeader,
    /// Optional header name used to send the organization id (OpenAI).
    pub organization_header: Option<&'static str>,
    /// Optional header name used to send the project id (OpenAI).
    pub project_header: Option<&'static str>,
    /// Whether the provider requires `vendor/model`-prefixed model ids
    /// (OpenRouter).
    pub requires_model_prefix: bool,
    /// Hard ceiling on `max_tokens` if the platform enforces one.
    pub max_tokens_cap: Option<u32>,
    /// The JSON field name a provider uses for reasoning/thinking content
    /// (e.g. `reasoning_content` for DeepSeek).
    pub reasoning_field: Option<&'static str>,
    /// Conservative default sustained requests-per-second for rate limiting.
    pub requests_per_second: f64,
    /// Default HTTP timeout in seconds.
    pub timeout_secs: u64,
    /// Whether SSE streaming is supported.
    pub supports_streaming: bool,
    /// Whether function/tool calling is supported.
    pub supports_tools: bool,
    /// Extra headers to send with every request.
    pub extra_headers: &'static [(&'static str, &'static str)],
    /// Free-form notes for diagnostics.
    pub notes: &'static str,
}

/// Compact macro for building a [`ProviderInfo`] value.
#[allow(clippy::too_many_arguments)]
macro_rules! provider_info {
    ($id:expr, $display:expr, $base:expr, $model:expr, $models:expr, $auth:expr,
     $org:expr, $proj:expr, $prefix:expr, $cap:expr, $reason:expr, $rps:expr,
     $timeout:expr, $stream:expr, $tools:expr, $hdrs:expr, $notes:expr) => {
        ProviderInfo {
            id: $id,
            display_name: $display,
            base_url: $base,
            default_model: $model,
            models: $models,
            auth: $auth,
            organization_header: $org,
            project_header: $proj,
            requires_model_prefix: $prefix,
            max_tokens_cap: $cap,
            reasoning_field: $reason,
            requests_per_second: $rps,
            timeout_secs: $timeout,
            supports_streaming: $stream,
            supports_tools: $tools,
            extra_headers: $hdrs,
            notes: $notes,
        }
    };
}

/// Headers shared by several providers (OpenRouter attribution).
static OPENROUTER_HEADERS: [(&str, &str); 2] =
    [("HTTP-Referer", "https://opensquilla.dev"), ("X-Title", "OpenSquilla")];

impl OpenAIProvider {
    /// The canonical provider id string, matching the compat-policy table.
    pub const fn as_str(&self) -> &'static str {
        match self {
            OpenAIProvider::OpenAi => "openai",
            OpenAIProvider::DeepSeek => "deepseek",
            OpenAIProvider::Gemini => "gemini",
            OpenAIProvider::DashScope => "dashscope",
            OpenAIProvider::Qwen => "qwen",
            OpenAIProvider::Moonshot => "moonshot",
            OpenAIProvider::Mistral => "mistral",
            OpenAIProvider::Groq => "groq",
            OpenAIProvider::Zhipu => "zhipu",
            OpenAIProvider::SiliconFlow => "siliconflow",
            OpenAIProvider::OpenRouter => "openrouter",
            OpenAIProvider::Azure => "azure",
            OpenAIProvider::Together => "together",
            OpenAIProvider::Fireworks => "fireworks",
            OpenAIProvider::Anyscale => "anyscale",
            OpenAIProvider::Lepton => "lepton",
            OpenAIProvider::Replicate => "replicate",
            OpenAIProvider::Perplexity => "perplexity",
            OpenAIProvider::Cohere => "cohere",
            OpenAIProvider::Ai21 => "ai21",
            OpenAIProvider::Xai => "xai",
            OpenAIProvider::DeepInfra => "deepinfra",
            OpenAIProvider::HuggingFace => "huggingface",
            OpenAIProvider::Novita => "novita",
            OpenAIProvider::Infermatic => "infermatic",
            OpenAIProvider::ModelScope => "modelscope",
            OpenAIProvider::Yi => "yi",
            OpenAIProvider::Baichuan => "baichuan",
            OpenAIProvider::MiniMax => "minimax_text",
            OpenAIProvider::StepFun => "stepfun",
            OpenAIProvider::Lingyi => "lingyi",
            OpenAIProvider::InternLm => "internlm",
            OpenAIProvider::Glm => "glm",
            OpenAIProvider::Hunyuan => "hunyuan",
            OpenAIProvider::TencentHunyuan => "tencent_hunyuan",
            OpenAIProvider::BaiduErnie => "baidu_ernie",
            OpenAIProvider::IflytekSpark => "iflytek_spark",
            OpenAIProvider::SenseTime => "sensetime",
            OpenAIProvider::Meituan => "meituan",
            OpenAIProvider::Volcengine => "volcengine",
            OpenAIProvider::Lambda => "lambda",
            OpenAIProvider::Hyperbolic => "hyperbolic",
            OpenAIProvider::Chutes => "chutes",
            OpenAIProvider::Kluster => "kluster",
            OpenAIProvider::InferenceNet => "inference_net",
            OpenAIProvider::NotDiamond => "not_diamond",
            OpenAIProvider::LocalAi => "localai",
            OpenAIProvider::LlamaCpp => "llama_cpp",
            OpenAIProvider::Vllm => "vllm",
            OpenAIProvider::Voyage => "voyage",
            OpenAIProvider::ElevenLabs => "elevenlabs",
            OpenAIProvider::Stability => "stability",
            OpenAIProvider::Anthropic => "anthropic",
            OpenAIProvider::Custom => "custom",
        }
    }

    /// Parse a provider from its canonical id. Returns `None` for unknown ids
    /// (callers typically fall back to [`OpenAIProvider::Custom`]).
    pub fn from_str_id(id: &str) -> Option<Self> {
        match id {
            "openai" => Some(OpenAIProvider::OpenAi),
            "deepseek" => Some(OpenAIProvider::DeepSeek),
            "gemini" => Some(OpenAIProvider::Gemini),
            "dashscope" => Some(OpenAIProvider::DashScope),
            "qwen" => Some(OpenAIProvider::Qwen),
            "moonshot" => Some(OpenAIProvider::Moonshot),
            "mistral" => Some(OpenAIProvider::Mistral),
            "groq" => Some(OpenAIProvider::Groq),
            "zhipu" => Some(OpenAIProvider::Zhipu),
            "siliconflow" => Some(OpenAIProvider::SiliconFlow),
            "openrouter" => Some(OpenAIProvider::OpenRouter),
            "azure" => Some(OpenAIProvider::Azure),
            "together" => Some(OpenAIProvider::Together),
            "fireworks" => Some(OpenAIProvider::Fireworks),
            "anyscale" => Some(OpenAIProvider::Anyscale),
            "lepton" => Some(OpenAIProvider::Lepton),
            "replicate" => Some(OpenAIProvider::Replicate),
            "perplexity" => Some(OpenAIProvider::Perplexity),
            "cohere" => Some(OpenAIProvider::Cohere),
            "ai21" => Some(OpenAIProvider::Ai21),
            "xai" | "x.ai" => Some(OpenAIProvider::Xai),
            "deepinfra" => Some(OpenAIProvider::DeepInfra),
            "huggingface" | "hugging_face" => Some(OpenAIProvider::HuggingFace),
            "novita" => Some(OpenAIProvider::Novita),
            "infermatic" => Some(OpenAIProvider::Infermatic),
            "modelscope" => Some(OpenAIProvider::ModelScope),
            "yi" => Some(OpenAIProvider::Yi),
            "baichuan" => Some(OpenAIProvider::Baichuan),
            "minimax_text" | "minimax" => Some(OpenAIProvider::MiniMax),
            "stepfun" => Some(OpenAIProvider::StepFun),
            "lingyi" => Some(OpenAIProvider::Lingyi),
            "internlm" => Some(OpenAIProvider::InternLm),
            "glm" => Some(OpenAIProvider::Glm),
            "hunyuan" => Some(OpenAIProvider::Hunyuan),
            "tencent_hunyuan" => Some(OpenAIProvider::TencentHunyuan),
            "baidu_ernie" | "ernie" => Some(OpenAIProvider::BaiduErnie),
            "iflytek_spark" | "spark" => Some(OpenAIProvider::IflytekSpark),
            "sensetime" => Some(OpenAIProvider::SenseTime),
            "meituan" => Some(OpenAIProvider::Meituan),
            "volcengine" => Some(OpenAIProvider::Volcengine),
            "lambda" => Some(OpenAIProvider::Lambda),
            "hyperbolic" => Some(OpenAIProvider::Hyperbolic),
            "chutes" => Some(OpenAIProvider::Chutes),
            "kluster" => Some(OpenAIProvider::Kluster),
            "inference_net" | "inference.net" => Some(OpenAIProvider::InferenceNet),
            "not_diamond" | "not-diamond" => Some(OpenAIProvider::NotDiamond),
            "localai" => Some(OpenAIProvider::LocalAi),
            "llama_cpp" | "llamacpp" => Some(OpenAIProvider::LlamaCpp),
            "vllm" => Some(OpenAIProvider::Vllm),
            "voyage" => Some(OpenAIProvider::Voyage),
            "elevenlabs" => Some(OpenAIProvider::ElevenLabs),
            "stability" => Some(OpenAIProvider::Stability),
            "anthropic" => Some(OpenAIProvider::Anthropic),
            "custom" | "openai_compat" => Some(OpenAIProvider::Custom),
            _ => None,
        }
    }

    /// Static connection metadata for this provider variant.
    pub fn info(&self) -> ProviderInfo {
        match self {
            OpenAIProvider::OpenAi => provider_info!(
                "openai", "OpenAI", "https://api.openai.com/v1", "gpt-4o",
                &["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o3-mini", "gpt-4.1", "gpt-4.1-mini"],
                AuthHeader::Bearer, Some("OpenAI-Organization"), Some("OpenAI-Project"),
                false, None, None, 60.0, 120, true, true, &[],
                "OpenAI canonical API"
            ),
            OpenAIProvider::DeepSeek => provider_info!(
                "deepseek", "DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat",
                &["deepseek-chat", "deepseek-reasoner"],
                AuthHeader::Bearer, None, None,
                false, Some(8192), Some("reasoning_content"), 30.0, 120, true, true, &[],
                "DeepSeek emits reasoning via reasoning_content and tool calls as DSML text"
            ),
            OpenAIProvider::Gemini => provider_info!(
                "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash",
                &["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
                AuthHeader::Bearer, None, None,
                false, Some(8192), None, 15.0, 120, true, true, &[],
                "Google Gemini OpenAI-compat shim: system prompt prepended to first user message"
            ),
            OpenAIProvider::DashScope => provider_info!(
                "dashscope", "Alibaba DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus",
                &["qwen-max", "qwen-plus", "qwen-turbo"],
                AuthHeader::Bearer, None, None,
                false, Some(8192), None, 30.0, 120, true, true, &[],
                "Alibaba DashScope (Qwen) OpenAI-compat endpoint"
            ),
            OpenAIProvider::Qwen => provider_info!(
                "qwen", "Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus",
                &["qwen-max", "qwen-plus", "qwen-turbo"],
                AuthHeader::Bearer, None, None,
                false, Some(8192), None, 30.0, 120, true, true, &[],
                "Qwen (alias of DashScope compatible mode)"
            ),
            OpenAIProvider::Moonshot => provider_info!(
                "moonshot", "Moonshot (Kimi)", "https://api.moonshot.cn/v1", "moonshot-v1-128k",
                &["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k", "kimi-k2-turbo-preview", "kimi-k1.5"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Moonshot Kimi OpenAI-compat"
            ),
            OpenAIProvider::Mistral => provider_info!(
                "mistral", "Mistral AI", "https://api.mistral.ai/v1", "mistral-large-latest",
                &["mistral-large-latest", "mistral-small-latest", "mistral-medium-latest", "mixtral-8x7b"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Mistral La Plateforme OpenAI-compat"
            ),
            OpenAIProvider::Groq => provider_info!(
                "groq", "Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile",
                &["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                AuthHeader::Bearer, None, None,
                false, Some(8192), None, 30.0, 120, true, true, &[],
                "Groq: fast inference, capped max_tokens"
            ),
            OpenAIProvider::Zhipu => provider_info!(
                "zhipu", "Zhipu (GLM)", "https://open.bigmodel.cn/api/paas/v4", "glm-4-plus",
                &["glm-4-plus", "glm-4", "glm-4-flash", "glm-4-air"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Zhipu GLM OpenAI-compat"
            ),
            OpenAIProvider::SiliconFlow => provider_info!(
                "siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-72B-Instruct",
                &["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "SiliconFlow OpenAI-compat aggregator"
            ),
            OpenAIProvider::OpenRouter => provider_info!(
                "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4o",
                &["openai/gpt-4o", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash", "deepseek/deepseek-chat"],
                AuthHeader::Bearer, None, None,
                true, None, None, 30.0, 120, true, true, &OPENROUTER_HEADERS,
                "OpenRouter: requires provider-prefixed model ids"
            ),
            OpenAIProvider::Azure => provider_info!(
                "azure", "Azure OpenAI", "https://<resource>.openai.azure.com", "gpt-4o",
                &["gpt-4o", "gpt-4", "gpt-35-turbo"],
                AuthHeader::AzureApiKey, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Azure OpenAI: key in api-key header, deployment in path, api-version query"
            ),
            OpenAIProvider::Together => provider_info!(
                "together", "Together AI", "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                &["meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.1-8B-Instruct-Turbo"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Together AI OpenAI-compat"
            ),
            OpenAIProvider::Fireworks => provider_info!(
                "fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/llama-v3p3-70b-instruct",
                &["accounts/fireworks/models/llama-v3p3-70b-instruct", "accounts/fireworks/models/llama-v3p1-8b-instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Fireworks AI: models use accounts/fireworks/models/ prefix"
            ),
            OpenAIProvider::Anyscale => provider_info!(
                "anyscale", "Anyscale", "https://api.endpoints.anyscale.com/v1", "meta-llama/Llama-3.1-70B-Instruct",
                &["meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Anyscale Endpoints OpenAI-compat"
            ),
            OpenAIProvider::Lepton => provider_info!(
                "lepton", "Lepton AI", "https://api.lepton.ai/api/v1", "llama3-8b",
                &["llama3-8b", "llama3-70b"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "Lepton AI OpenAI-compat"
            ),
            OpenAIProvider::Replicate => provider_info!(
                "replicate", "Replicate", "https://api.replicate.com/v1", "meta/meta-llama-3-70b-instruct",
                &["meta/meta-llama-3-70b-instruct", "meta/meta-llama-3-8b-instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Replicate OpenAI-compatible endpoint"
            ),
            OpenAIProvider::Perplexity => provider_info!(
                "perplexity", "Perplexity", "https://api.perplexity.ai", "llama-3.1-sonar-large-128k-online",
                &["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Perplexity OpenAI-compat"
            ),
            OpenAIProvider::Cohere => provider_info!(
                "cohere", "Cohere", "https://api.cohere.ai/v1", "command-r-plus",
                &["command-r-plus", "command-r"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Cohere OpenAI-compat"
            ),
            OpenAIProvider::Ai21 => provider_info!(
                "ai21", "AI21 Labs", "https://api.ai21.com/studio/v1", "jamba-1-5-large",
                &["jamba-1-5-large", "jamba-1-5-mini"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "AI21 Jamba OpenAI-compat"
            ),
            OpenAIProvider::Xai => provider_info!(
                "xai", "xAI (Grok)", "https://api.x.ai/v1", "grok-2-latest",
                &["grok-2-latest", "grok-beta", "grok-3"],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "xAI Grok OpenAI-compat"
            ),
            OpenAIProvider::DeepInfra => provider_info!(
                "deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", "meta-llama/Llama-3.3-70B-Instruct",
                &["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "DeepInfra OpenAI-compat"
            ),
            OpenAIProvider::HuggingFace => provider_info!(
                "huggingface", "Hugging Face", "https://api-inference.huggingface.co/v1", "meta-llama/Llama-3.3-70B-Instruct",
                &["meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Hugging Face Inference OpenAI-compat"
            ),
            OpenAIProvider::Novita => provider_info!(
                "novita", "Novita AI", "https://api.novita.ai/v3/openai", "meta-llama/llama-3.1-70b-instruct",
                &["meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Novita AI OpenAI-compat"
            ),
            OpenAIProvider::Infermatic => provider_info!(
                "infermatic", "Infermatic", "https://api.infermatic.ai/v1", "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Infermatic AI OpenAI-compat"
            ),
            OpenAIProvider::ModelScope => provider_info!(
                "modelscope", "ModelScope", "https://api-inference.modelscope.cn/v1", "Qwen/Qwen2.5-72B-Instruct",
                &["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "ModelScope OpenAI-compat"
            ),
            OpenAIProvider::Yi => provider_info!(
                "yi", "01.AI (Yi)", "https://api.01.ai/v1", "yi-large",
                &["yi-large", "yi-medium", "yi-lightning"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "01.AI Yi OpenAI-compat"
            ),
            OpenAIProvider::Baichuan => provider_info!(
                "baichuan", "Baichuan", "https://api.baichuan-ai.com/v1", "Baichuan4",
                &["Baichuan4", "Baichuan3-Turbo"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Baichuan OpenAI-compat"
            ),
            OpenAIProvider::MiniMax => provider_info!(
                "minimax_text", "MiniMax Text", "https://api.minimax.chat/v1", "abab6.5-chat",
                &["abab6.5-chat", "abab6.5s-chat", "MiniMax-Text-01"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "MiniMax text backend OpenAI-compat"
            ),
            OpenAIProvider::StepFun => provider_info!(
                "stepfun", "StepFun", "https://api.stepfun.com/v1", "step-2-16k",
                &["step-2-16k", "step-1-8k"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "StepFun OpenAI-compat"
            ),
            OpenAIProvider::Lingyi => provider_info!(
                "lingyi", "Lingyi (Yi large)", "https://api.lingyiwanwu.com/v1", "yi-large",
                &["yi-large", "yi-medium"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Lingyi OpenAI-compat"
            ),
            OpenAIProvider::InternLm => provider_info!(
                "internlm", "InternLM", "https://internlm-chat.intern-ai.org.cn/puyu/api/v1", "internlm2.5-latest",
                &["internlm2.5-latest", "internlm2.5-7b"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "InternLM OpenAI-compat"
            ),
            OpenAIProvider::Glm => provider_info!(
                "glm", "GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4",
                &["glm-4", "glm-4-flash"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "GLM OpenAI-compat"
            ),
            OpenAIProvider::Hunyuan => provider_info!(
                "hunyuan", "Tencent Hunyuan", "https://api.hunyuan.cloud.tencent.com/v1", "hunyuan-pro",
                &["hunyuan-pro", "hunyuan-standard"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Tencent Hunyuan OpenAI-compat"
            ),
            OpenAIProvider::TencentHunyuan => provider_info!(
                "tencent_hunyuan", "Tencent Hunyuan (alt)", "https://api.hunyuan.cloud.tencent.com/v1", "hunyuan-pro",
                &["hunyuan-pro"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Tencent Hunyuan alternate id"
            ),
            OpenAIProvider::BaiduErnie => provider_info!(
                "baidu_ernie", "Baidu ERNIE", "https://qianfan.baidubce.com/v2", "ernie-4.0-8k-latest",
                &["ernie-4.0-8k-latest", "ernie-3.5-8k"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Baidu ERNIE OpenAI-compat"
            ),
            OpenAIProvider::IflytekSpark => provider_info!(
                "iflytek_spark", "iFlytek Spark", "https://spark-api-open.xf-yun.com/v1", "4.0Ultra",
                &["4.0Ultra", "generalv3.5"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "iFlytek Spark OpenAI-compat"
            ),
            OpenAIProvider::SenseTime => provider_info!(
                "sensetime", "SenseTime", "https://api.sensenova.cn/compatible-mode/v1", "SenseChat-5",
                &["SenseChat-5"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "SenseTime OpenAI-compat"
            ),
            OpenAIProvider::Meituan => provider_info!(
                "meituan", "Meituan", "https://api.meituan.com/v1", "mao-1",
                &["mao-1"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Meituan OpenAI-compat"
            ),
            OpenAIProvider::Volcengine => provider_info!(
                "volcengine", "Volcengine (Doubao)", "https://ark.cn-beijing.volces.com/api/v3", "doubao-pro-32k",
                &["doubao-pro-32k", "doubao-pro-4k", "doubao-lite-32k"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Volcengine Ark OpenAI-compat"
            ),
            OpenAIProvider::Lambda => provider_info!(
                "lambda", "Lambda Labs", "https://api.lambdalabs.com/v1", "hermes3-405b",
                &["hermes3-405b", "llama3.1-405b-instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Lambda Labs OpenAI-compat"
            ),
            OpenAIProvider::Hyperbolic => provider_info!(
                "hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1", "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Hyperbolic OpenAI-compat"
            ),
            OpenAIProvider::Chutes => provider_info!(
                "chutes", "Chutes AI", "https://api.chutes.ai/v1", "chutesai/Llama-3.1-8B-Instruct",
                &["chutesai/Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Chutes AI OpenAI-compat"
            ),
            OpenAIProvider::Kluster => provider_info!(
                "kluster", "Kluster", "https://api.kluster.ai/v1", "meta-llama/Meta-Llama-3.1-70B-Instruct",
                &["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Kluster OpenAI-compat"
            ),
            OpenAIProvider::InferenceNet => provider_info!(
                "inference_net", "Inference.net", "https://api.inference.net/v1", "meta-llama/Llama-3.1-70B-Instruct",
                &["meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Inference.net OpenAI-compat"
            ),
            OpenAIProvider::NotDiamond => provider_info!(
                "not_diamond", "Not Diamond", "https://api.not-diamond.com/v1", "not-diamond-auto",
                &["not-diamond-auto"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Not Diamond router OpenAI-compat"
            ),
            OpenAIProvider::LocalAi => provider_info!(
                "localai", "LocalAI (self-hosted)", "http://localhost:8080/v1", "gpt-4",
                &["gpt-4", "gpt-3.5-turbo", "llama3.1"],
                AuthHeader::None, None, None,
                false, None, None, 100.0, 60, true, true, &[],
                "LocalAI self-hosted, no auth"
            ),
            OpenAIProvider::LlamaCpp => provider_info!(
                "llama_cpp", "llama.cpp server", "http://localhost:8080/v1", "llama-3.1-8b",
                &["llama-3.1-8b", "llama-3.1-70b", "qwen2.5-7b"],
                AuthHeader::None, None, None,
                false, None, None, 100.0, 60, true, true, &[],
                "llama.cpp server, no auth"
            ),
            OpenAIProvider::Vllm => provider_info!(
                "vllm", "vLLM", "http://localhost:8000/v1", "meta-llama/Llama-3.1-8B-Instruct",
                &["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"],
                AuthHeader::None, None, None,
                false, None, None, 100.0, 60, true, true, &[],
                "vLLM server, no auth"
            ),
            OpenAIProvider::Voyage => provider_info!(
                "voyage", "Voyage AI", "https://api.voyageai.com/v1", "voyage-3-large",
                &["voyage-3-large", "voyage-3", "voyage-3-lite", "voyage-code-3"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Voyage AI OpenAI-compat"
            ),
            OpenAIProvider::ElevenLabs => provider_info!(
                "elevenlabs", "ElevenLabs", "https://api.elevenlabs.io/v1", "eleven_multilingual_v2",
                &["eleven_multilingual_v2", "eleven_turbo_v2_5"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "ElevenLabs OpenAI-compat"
            ),
            OpenAIProvider::Stability => provider_info!(
                "stability", "Stability AI", "https://api.stability.ai/v1", "stable-image-ultra",
                &["stable-image-ultra", "stable-image-core"],
                AuthHeader::Bearer, None, None,
                false, None, None, 20.0, 120, true, true, &[],
                "Stability AI OpenAI-compat"
            ),
            OpenAIProvider::Anthropic => provider_info!(
                "anthropic", "Anthropic", "https://api.anthropic.com/v1", "claude-3-5-sonnet-20241022",
                &["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                AuthHeader::HeaderApiKey, None, None,
                false, Some(8192), None, 30.0, 180, true, true, &[],
                "Anthropic (usually the dedicated backend; OpenAI-compat entry for completeness)"
            ),
            OpenAIProvider::Custom => provider_info!(
                "custom", "Custom OpenAI-compatible", "", "",
                &[],
                AuthHeader::Bearer, None, None,
                false, None, None, 30.0, 120, true, true, &[],
                "User-supplied OpenAI-compatible endpoint"
            ),
        }
    }

    /// The default base URL for this provider (may be empty for custom).
    pub fn default_base_url(&self) -> &'static str {
        self.info().base_url
    }

    /// The default model id for this provider.
    pub fn default_model(&self) -> &'static str {
        self.info().default_model
    }

    /// The well-known model list for this provider.
    pub fn default_models(&self) -> &'static [&'static str] {
        self.info().models
    }

    /// The auth header shape this provider expects.
    pub fn auth_header(&self) -> AuthHeader {
        self.info().auth
    }

    /// Whether this provider requires `vendor/model`-prefixed model ids.
    pub fn requires_model_prefix(&self) -> bool {
        self.info().requires_model_prefix
    }

    /// Whether this provider is a local, unauthenticated endpoint.
    pub fn is_local(&self) -> bool {
        matches!(self.info().auth, AuthHeader::None)
    }

    /// Whether this provider is Azure (special request-path handling).
    pub fn is_azure(&self) -> bool {
        matches!(self, OpenAIProvider::Azure)
    }
}

impl std::fmt::Display for OpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl FromStr for OpenAIProvider {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_str_id(s).unwrap_or(OpenAIProvider::Custom))
    }
}

/// Parse a provider from a string id with a [`Custom`](OpenAIProvider::Custom)
/// fallback.
pub fn provider_from_id(id: &str) -> OpenAIProvider {
    OpenAIProvider::from_str_id(id).unwrap_or(OpenAIProvider::Custom)
}

// ===========================================================================
// Configuration and request/response types
// ===========================================================================

/// Configuration for constructing an OpenAI-compatible runtime client.
///
/// All fields are optional except the provider variant and API key; defaults
/// are resolved from [`ProviderInfo`] when the struct is consumed by
/// [`OpenAiCompatProvider::from_config`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiConfig {
    /// The provider variant (drives base URL, auth, and compat policy).
    pub provider: OpenAIProvider,
    /// The API key (empty for local providers).
    pub api_key: String,
    /// Base URL override. Empty means "use the provider default".
    #[serde(default)]
    pub base_url: String,
    /// OpenAI organization id (sent in the `OpenAI-Organization` header).
    #[serde(default)]
    pub organization: Option<String>,
    /// OpenAI project id (sent in the `OpenAI-Project` header).
    #[serde(default)]
    pub project: Option<String>,
    /// Azure API version (defaults to `2024-10-21`).
    #[serde(default)]
    pub api_version: Option<String>,
    /// HTTP timeout override in seconds.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Max retries for transient failures (defaults to 3).
    #[serde(default)]
    pub max_retries: Option<u32>,
    /// Sustained requests-per-second rate limit override.
    #[serde(default)]
    pub requests_per_second: Option<f64>,
    /// Extra HTTP headers sent with every request.
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Extra JSON parameters merged into every request body.
    #[serde(default)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl OpenAiConfig {
    /// Create a config for a provider with the given API key, using the
    /// provider's default base URL.
    pub fn new(provider: OpenAIProvider, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            base_url: String::new(),
            organization: None,
            project: None,
            api_version: None,
            timeout_secs: None,
            max_retries: None,
            requests_per_second: None,
            extra_headers: HashMap::new(),
            extra_params: HashMap::new(),
        }
    }

    /// Set the base URL (overrides the provider default).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the OpenAI organization id.
    pub fn with_organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    /// Set the OpenAI project id.
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }

    /// Set the Azure API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = Some(version.into());
        self
    }

    /// Override the HTTP timeout.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Override the max retry count.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Override the sustained requests-per-second rate limit.
    pub fn with_requests_per_second(mut self, rps: f64) -> Self {
        self.requests_per_second = Some(rps);
        self
    }

    /// Add an extra HTTP header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }

    /// Add an extra JSON body parameter.
    pub fn with_param(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_params.insert(name.into(), value);
        self
    }
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self::new(OpenAIProvider::Custom, "")
    }
}

/// A fully-specified chat completion request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// The model id (already mapped through any provider prefix rules).
    pub model: String,
    /// The conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Tool definitions the model may call.
    pub tools: Vec<ToolDefinition>,
    /// Sampling temperature.
    pub temperature: f64,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Top-p nucleus sampling.
    pub top_p: f64,
    /// Stop sequences.
    pub stop: Vec<String>,
    /// Whether to stream.
    pub stream: bool,
    /// Provider-specific extra parameters.
    pub extra: HashMap<String, serde_json::Value>,
}

impl ChatRequest {
    /// Create a request with the given model and messages, using OpenAI
    /// defaults for sampling parameters.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            temperature: 0.7,
            max_tokens: 4096,
            top_p: 0.9,
            stop: Vec::new(),
            stream: false,
            extra: HashMap::new(),
        }
    }

    /// Set the tool definitions.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set top-p.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set stop sequences.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Set whether to stream.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Add an extra parameter.
    pub fn with_param(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(name.into(), value);
        self
    }
}

impl From<(&ChatConfig, &[ChatMessage], &[ToolDefinition])> for ChatRequest {
    fn from(
        (config, messages, tools): (&ChatConfig, &[ChatMessage], &[ToolDefinition]),
    ) -> Self {
        ChatRequest {
            model: config.model.clone(),
            messages: messages.to_vec(),
            tools: tools.to_vec(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: config.top_p,
            stop: config.stop_sequences.clone(),
            stream: config.stream,
            extra: config.extra.clone(),
        }
    }
}

/// A parsed non-streaming chat completion response.
#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    /// The response id (when provided).
    pub id: Option<String>,
    /// The generated text content (tool-call markup stripped when applicable).
    pub content: String,
    /// Reasoning / thinking content, when the provider exposes it.
    pub reasoning_content: Option<String>,
    /// Tool calls requested by the model.
    pub tool_calls: Vec<ToolCall>,
    /// Token usage.
    pub usage: Usage,
    /// The model that produced the response.
    pub model: String,
    /// The finish reason (`stop`, `length`, `tool_calls`, ...).
    pub finish_reason: Option<String>,
}

// ===========================================================================
// Runtime client
// ===========================================================================
//
// The runtime client implements the crate-wide [`Provider`] trait and the
// crate-wide [`crate::types::ChatProvider`] trait (whose `chat` method streams
// events). In addition it exposes a richer, protocol-specific surface via
// inherent methods: [`OpenAiCompatProvider::complete`] (non-streaming),
// [`OpenAiCompatProvider::stream_request`], and
// [`OpenAiCompatProvider::list_models`].

/// Runtime OpenAI-compatible chat client.
///
/// Implements both the crate-wide [`Provider`] trait and the crate-wide
/// [`ChatProvider`] trait, plus a richer inherent API for protocol-specific
/// callers. Construct via [`OpenAiCompatProvider::new`] (backwards compatible
/// with the historical three-argument constructor) or the fuller
/// [`OpenAiCompatProvider::from_config`].
pub struct OpenAiCompatProvider {
    name: String,
    kind: OpenAIProvider,
    api_base: String,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    api_version: Option<String>,
    client: Client,
    policy: CompatPolicy,
    limiter: RateLimiter,
    retry: RetryConfig,
    credentials: Option<CredentialPool>,
    proof: Option<RequestProof>,
    extra_headers: HashMap<String, String>,
    extra_params: HashMap<String, serde_json::Value>,
}

impl OpenAiCompatProvider {
    /// Create a new provider with the historical signature.
    ///
    /// The provider *kind* is inferred from `name` (e.g. `"deepseek"` →
    /// [`OpenAIProvider::DeepSeek`]); unknown names become
    /// [`OpenAIProvider::Custom`].
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let kind = OpenAIProvider::from_str_id(&name).unwrap_or(OpenAIProvider::Custom);
        Self::from_config(OpenAiConfig::new(kind, api_key).with_base_url(api_base.into()))
            .with_name(name)
    }

    /// Create a provider from a registry [`crate::registry::ProviderSpec`] and
    /// API key.
    ///
    /// Returns `None` when the spec's backend is not `OpenAiCompat`. This ties
    /// the backend into [`crate::registry`] so any OpenAI-compatible spec can
    /// be instantiated here without a bespoke match arm.
    pub fn from_spec(spec: &crate::registry::ProviderSpec, api_key: &str) -> Option<Self> {
        if spec.backend != crate::registry::BackendType::OpenAiCompat {
            return None;
        }
        let kind = OpenAIProvider::from_str_id(spec.id).unwrap_or(OpenAIProvider::Custom);
        Some(
            Self::from_config(OpenAiConfig::new(kind, api_key).with_base_url(spec.api_base))
                .with_name(spec.id),
        )
    }

    /// Create a provider from a full [`OpenAiConfig`], resolving defaults from
    /// the provider's static metadata.
    pub fn from_config(config: OpenAiConfig) -> Self {
        let info = config.provider.info();
        let base_url = if config.base_url.trim().is_empty() {
            info.base_url.to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        let api_key = config.api_key.clone();
        let rps = config.requests_per_second.unwrap_or(info.requests_per_second).max(0.1);
        let timeout_secs = config.timeout_secs.unwrap_or(info.timeout_secs).max(1);
        let max_retries = config.max_retries.unwrap_or(3).max(1);

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .expect("Failed to create reqwest Client for OpenAI-compatible provider");

        let limiter = RateLimiter::new(rps);
        let retry = RetryConfig {
            max_attempts: max_retries,
            retry_on_429: true,
            ..RetryConfig::default()
        };
        let policy = policy_for(info.id);

        Self {
            name: info.id.to_string(),
            kind: config.provider,
            api_base: base_url,
            api_key,
            organization: config.organization,
            project: config.project,
            api_version: config.api_version,
            client,
            policy,
            limiter,
            retry,
            credentials: None,
            proof: Some(RequestProof::heuristic()),
            extra_headers: config.extra_headers,
            extra_params: config.extra_params,
        }
    }

    /// Override the runtime display name (used by registry/test callers).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Attach a credential pool for round-robin API-key rotation and 429
    /// cooldown.
    pub fn with_credentials(mut self, pool: CredentialPool) -> Self {
        self.credentials = Some(pool);
        self
    }

    /// Disable pre-flight budget projection.
    pub fn without_proof(mut self) -> Self {
        self.proof = None;
        self
    }

    /// Attach a custom pre-flight proof.
    pub fn with_proof(mut self, proof: RequestProof) -> Self {
        self.proof = Some(proof);
        self
    }

    /// Override the retry configuration.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// The provider variant for this client.
    pub fn kind(&self) -> OpenAIProvider {
        self.kind
    }

    /// The resolved base URL.
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    /// The chat completions endpoint URL for the given model.
    pub fn chat_url(&self, model: &str) -> String {
        let base = self.api_base.trim_end_matches('/');
        match self.kind {
            OpenAIProvider::Azure => {
                let deployment = map_model(self.kind, model);
                let version = self.api_version.as_deref().unwrap_or("2024-10-21");
                format!(
                    "{base}/openai/deployments/{deployment}/chat/completions?api-version={version}"
                )
            }
            OpenAIProvider::Custom => {
                if base.ends_with("/chat/completions") {
                    base.to_string()
                } else {
                    format!("{base}/chat/completions")
                }
            }
            _ => format!("{base}/chat/completions"),
        }
    }

    /// Apply the provider's authentication header to a request builder.
    fn apply_auth(&self, builder: RequestBuilder, key: &str) -> RequestBuilder {
        match self.kind.info().auth {
            AuthHeader::Bearer => builder.header("Authorization", format!("Bearer {key}")),
            AuthHeader::HeaderApiKey => builder.header("x-api-key", key),
            AuthHeader::AzureApiKey => builder.header("api-key", key),
            AuthHeader::None => builder,
        }
    }

    /// Apply organization / project / policy / config headers.
    fn apply_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        let mut b = builder;
        for (name, value) in self.policy.extra_headers {
            if !value.is_empty() {
                b = b.header(*name, *value);
            }
        }
        for (name, value) in &self.extra_headers {
            b = b.header(name, value);
        }
        if let Some(org) = &self.organization {
            if let Some(header) = self.kind.info().organization_header {
                b = b.header(header, org);
            } else {
                b = b.header("OpenAI-Organization", org);
            }
        }
        if let Some(proj) = &self.project {
            if let Some(header) = self.kind.info().project_header {
                b = b.header(header, proj);
            } else {
                b = b.header("OpenAI-Project", proj);
            }
        }
        b
    }

    /// Build the request body for a chat completion call, applying the
    /// provider's compatibility policy and model-mapping rules.
    fn build_request_body(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> serde_json::Value {
        let mut body =
            build_chat_request_with_policy(config, messages, tools, stream, &self.policy);

        if let Some(obj) = body.as_object_mut() {
            let mapped = map_model(self.kind, &config.model);
            obj.insert("model".into(), serde_json::json!(mapped));

            // Reasoning models (o1/o3, deepseek-reasoner) use
            // max_completion_tokens and reject temperature/top_p/stop.
            if is_reasoning_model(&mapped) {
                if let Some(v) = obj.remove("max_tokens") {
                    obj.insert("max_completion_tokens".into(), v);
                }
                obj.remove("temperature");
                obj.remove("top_p");
                obj.remove("stop");
            }

            // Merge provider-construction-time extra parameters.
            for (k, v) in &self.extra_params {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// Select the API key to use for the next request.
    fn select_key(&self) -> ProviderResult<(String, usize)> {
        match &self.credentials {
            Some(pool) => {
                let sel = pool.select().ok_or_else(|| {
                    ProviderError::Config("credential pool exhausted".to_string())
                })?;
                Ok((sel.key.clone(), sel.index))
            }
            None => Ok((self.api_key.clone(), 0)),
        }
    }

    /// Report request outcome back to the credential pool.
    fn report_outcome(&self, index: usize, result: &ProviderResult<ChatResponse>) {
        let Some(pool) = &self.credentials else {
            return;
        };
        match result {
            Ok(_) => pool.mark_success(index),
            Err(ProviderError::Auth(_)) => pool.mark_invalid(index),
            Err(ProviderError::RateLimited(_)) => pool.mark_rate_limited(index, None),
            Err(_) => {}
        }
    }
}

impl std::fmt::Debug for OpenAiCompatProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatProvider")
            .field("name", &self.name)
            .field("kind", &self.kind.as_str())
            .field("api_base", &self.api_base)
            .finish()
    }
}

impl OpenAiCompatProvider {
    /// Execute a non-streaming chat completion, returning a parsed
    /// [`ChatResponse`].
    pub async fn complete(&self, request: ChatRequest) -> ProviderResult<ChatResponse> {
        let cfg = ChatConfig {
            model: request.model.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop_sequences: request.stop.clone(),
            stream: false,
            extra: request.extra.clone(),
        };

        // Pre-flight budget projection: trim messages and clamp max_tokens.
        let (messages, cfg) = match &self.proof {
            Some(proof) => {
                let (msgs, new_cfg, report) =
                    proof.project(self.kind.as_str(), &cfg, &request.messages, &request.tools);
                if report.dropped_messages > 0 {
                    debug!(
                        target = "provider",
                        provider = %self.name,
                        dropped = report.dropped_messages,
                        prompt_tokens = report.prompt_tokens,
                        "Pre-flight projection dropped messages"
                    );
                }
                (msgs, new_cfg)
            }
            None => (request.messages.clone(), cfg),
        };

        // Rate limit the request.
        self.limiter.acquire().await?;

        let (key, key_index) = self.select_key()?;
        let body = self.build_request_body(&cfg, &messages, &request.tools, false);
        let url = self.chat_url(&cfg.model);

        debug!(
            target = "provider",
            provider = %self.name,
            kind = %self.kind.as_str(),
            model = %cfg.model,
            url = %url,
            "Sending non-streaming chat request"
        );

        let result = with_retry(&self.retry, |_attempt| {
            let this = &*self;
            let body = body.clone();
            let url = url.clone();
            let key = key.clone();
            async move {
                let mut req = this
                    .client
                    .post(&url)
                    .header("Content-Type", "application/json");
                req = this.apply_auth(req, &key);
                req = this.apply_headers(req);
                let resp = req.json(&body).send().await.map_err(ProviderError::Network)?;

                let status = resp.status();
                if !status.is_success() {
                    let text = resp.text().await.unwrap_or_default();
                    return Err(map_http_error(status, &text, &this.kind.as_str()));
                }

                let data: serde_json::Value = resp.json().await.map_err(ProviderError::Network)?;
                if data.get("error").is_some() {
                    let msg = extract_error_message(&data.to_string())
                        .unwrap_or_else(|| "provider returned an error object".to_string());
                    return Err(ProviderError::Provider(msg));
                }

                Ok(parse_chat_response(&data, &this.policy))
            }
        })
        .await;

        self.report_outcome(key_index, &result);

        match &result {
            Ok(resp) => info!(
                target = "provider",
                provider = %self.name,
                model = %resp.model,
                input_tokens = resp.usage.input_tokens,
                output_tokens = resp.usage.output_tokens,
                "Non-streaming response received"
            ),
            Err(e) => warn!(
                target = "provider",
                provider = %self.name,
                error = %e,
                "Non-streaming request failed"
            ),
        }

        result
    }

    /// Execute a streaming chat completion, yielding typed stream events.
    pub async fn stream_request(
        &self,
        request: ChatRequest,
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let cfg = ChatConfig {
            model: request.model.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            stop_sequences: request.stop.clone(),
            stream: true,
            extra: request.extra.clone(),
        };

        self.limiter.acquire().await?;

        let (key, key_index) = self.select_key()?;
        let body = self.build_request_body(&cfg, &request.messages, &request.tools, true);
        let url = self.chat_url(&cfg.model);

        debug!(
            target = "provider",
            provider = %self.name,
            kind = %self.kind.as_str(),
            model = %cfg.model,
            url = %url,
            "Sending streaming chat request"
        );

        let mut req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");
        req = self.apply_auth(req, &key);
        req = self.apply_headers(req);
        let resp = req.json(&body).send().await.map_err(ProviderError::Network)?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let err = map_http_error(status, &text, &self.kind.as_str());
            if let Some(pool) = &self.credentials {
                match &err {
                    ProviderError::Auth(_) => pool.mark_invalid(key_index),
                    ProviderError::RateLimited(_) => pool.mark_rate_limited(key_index, None),
                    _ => {}
                }
            }
            return Err(err);
        }

        let policy = self.policy.clone();
        let stream = SseStream::new(resp, move |data: &str| parse_openai_sse_event(data, &policy));
        Ok(Box::new(stream))
    }

    /// Fetch the provider's available models, falling back to the static list
    /// when the live `/models` endpoint is unreachable.
    pub async fn list_models(&self) -> ProviderResult<Vec<ModelCapabilities>> {
        let catalog = ModelCatalog::new();
        let live = LiveCatalog::new(catalog);
        let cfg = LiveProviderConfig::new(self.kind.as_str(), self.api_base.clone())
            .with_key(self.api_key.clone());
        let result = live.fetch_one(&cfg).await;
        if result.ok() && !result.models.is_empty() {
            Ok(result.models)
        } else {
            // Fall back to the static well-known model list.
            let models: Vec<ModelCapabilities> = self
                .kind
                .info()
                .models
                .iter()
                .map(|m| ModelCapabilities {
                    model: m.to_string(),
                    label: Some(m.to_string()),
                    supports_tools: self.policy.supports_tools,
                    supports_streaming: self.policy.supports_streaming,
                    ..Default::default()
                })
                .collect();
            Ok(models)
        }
    }
}

#[async_trait]
impl crate::types::ChatProvider for OpenAiCompatProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let request = ChatRequest::from((config, messages, tools)).with_stream(true);
        self.stream_request(request).await
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        self.kind
            .info()
            .models
            .iter()
            .map(|m| m.to_string())
            .collect()
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let request = ChatRequest::from((config, messages, tools)).with_stream(false);
        let resp = self.complete(request).await?;

        let mut blocks = Vec::new();
        if let Some(r) = &resp.reasoning_content {
            if !r.is_empty() {
                blocks.push(ContentBlock::Reasoning(r.clone()));
            }
        }
        if !resp.content.is_empty() {
            blocks.push(ContentBlock::Text(resp.content.clone()));
        }

        let tool_calls = if resp.tool_calls.is_empty() {
            None
        } else {
            Some(resp.tool_calls.clone())
        };

        let content = vec![ChatMessage {
            role: MessageRole::Assistant,
            content: blocks,
            tool_calls,
            tool_call_id: None,
            tool_result: None,
            name: None,
        }];

        Ok(ProviderResponse {
            content,
            usage: resp.usage,
            model: if resp.model.is_empty() {
                config.model.clone()
            } else {
                resp.model
            },
            stop_reason: resp.finish_reason,
        })
    }

    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let request = ChatRequest::from((config, messages, tools)).with_stream(true);
        self.stream_request(request).await
    }
}

// ===========================================================================
// Request building
// ===========================================================================

/// Build a canonical OpenAI-compatible chat completion request body.
///
/// This is the convenient entry point used by external callers: a
/// non-streaming request with no tools. The full variant is
/// [`build_chat_request_full`].
pub fn build_chat_request(messages: &[ChatMessage], config: &ChatConfig) -> serde_json::Value {
    build_chat_request_with_policy(config, messages, &[], false, &CompatPolicy::default_openai())
}

/// Build a full chat completion request body with tools and streaming.
pub fn build_chat_request_full(
    config: &ChatConfig,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    stream: bool,
) -> serde_json::Value {
    build_chat_request_with_policy(
        config,
        messages,
        tools,
        stream,
        &CompatPolicy::default_openai(),
    )
}

/// Build a request body, applying a provider compatibility policy.
///
/// Handles system-prompt policy (system role vs. separate field vs. prepend
/// to first user), tool-definition serialization, parameter adaptation
/// (removing unsupported fields and clamping `max_tokens`), and merging of
/// provider-specific extra parameters.
fn build_chat_request_with_policy(
    config: &ChatConfig,
    messages: &[ChatMessage],
    tools: &[ToolDefinition],
    stream: bool,
    policy: &CompatPolicy,
) -> serde_json::Value {
    let (converted, separate_system) = convert_messages(messages, policy);

    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::json!(config.model));
    body.insert("messages".into(), serde_json::json!(converted));
    body.insert("stream".into(), serde_json::json!(stream));

    if let Some(sys) = separate_system {
        body.insert("system".into(), serde_json::json!(sys));
    }

    if policy.supports_temperature {
        body.insert("temperature".into(), serde_json::json!(config.temperature));
    }
    if policy.supports_top_p {
        body.insert("top_p".into(), serde_json::json!(config.top_p));
    }
    body.insert("max_tokens".into(), serde_json::json!(config.max_tokens));
    if !config.stop_sequences.is_empty() && policy.supports_stop {
        body.insert("stop".into(), serde_json::json!(config.stop_sequences));
    }
    if policy.supports_tools && !tools.is_empty() {
        body.insert("tools".into(), serde_json::json!(build_tools_array(tools)));
    }

    // Adapt parameters to the policy (cap max_tokens, drop unsupported).
    let adapted = adapt_parameters(body, policy);
    let mut body = serde_json::Value::Object(adapted);

    if let Some(obj) = body.as_object_mut() {
        for (k, v) in &config.extra {
            obj.insert(k.clone(), v.clone());
        }
    }

    body
}

/// Convert canonical messages into the OpenAI wire format.
///
/// Returns the `messages` array plus an optional top-level `system` field
/// (used when the policy is [`SystemPromptPolicy::SeparateField`]).
fn convert_messages(
    messages: &[ChatMessage],
    policy: &CompatPolicy,
) -> (Vec<serde_json::Value>, Option<String>) {
    let system_text: String = messages
        .iter()
        .filter(|m| m.role == MessageRole::System)
        .map(|m| m.text_content())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    let mut separate_system = None;
    if policy.system_prompt == SystemPromptPolicy::SeparateField && !system_text.is_empty() {
        separate_system = Some(system_text.clone());
    }

    let folds_system = matches!(
        policy.system_prompt,
        SystemPromptPolicy::PrependToFirstUser | SystemPromptPolicy::MergeIntoFirstUser
    );

    let mut out: Vec<serde_json::Value> = Vec::new();
    let mut first_user_seen = false;

    for msg in messages {
        if msg.role == MessageRole::System {
            match policy.system_prompt {
                SystemPromptPolicy::SystemRole => {
                    out.push(message_to_openai(msg));
                }
                SystemPromptPolicy::SeparateField => {
                    // Emitted as the top-level `system` field instead.
                }
                SystemPromptPolicy::PrependToFirstUser
                | SystemPromptPolicy::MergeIntoFirstUser => {
                    // Folded into the first user message below.
                }
            }
            continue;
        }

        if msg.role == MessageRole::User {
            if folds_system && !first_user_seen && !system_text.is_empty() {
                first_user_seen = true;
                let mut entry = message_to_openai(msg);
                prepend_system_content(&mut entry, &system_text);
                out.push(entry);
                continue;
            }
            first_user_seen = true;
        }

        out.push(message_to_openai(msg));
    }

    (out, separate_system)
}

/// Serialize a single canonical message into the OpenAI wire format.
fn message_to_openai(msg: &ChatMessage) -> serde_json::Value {
    let role_str = match msg.role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    };

    let mut entry = serde_json::Map::new();
    entry.insert("role".into(), serde_json::json!(role_str));

    match msg.role {
        MessageRole::Tool => {
            let content = msg
                .tool_result
                .as_ref()
                .map(|tr| tr.content.clone())
                .unwrap_or_else(|| msg.text_content());
            let tool_call_id = msg
                .tool_call_id
                .clone()
                .or_else(|| {
                    msg.tool_result
                        .as_ref()
                        .map(|tr| tr.tool_use_id.clone())
                })
                .unwrap_or_default();
            entry.insert("tool_call_id".into(), serde_json::json!(tool_call_id));
            entry.insert("content".into(), serde_json::json!(content));
        }
        _ => {
            entry.insert("content".into(), convert_content_blocks(&msg.content));
        }
    }

    if msg.role == MessageRole::Assistant {
        let mut calls: Vec<serde_json::Value> = Vec::new();
        if let Some(tc) = &msg.tool_calls {
            calls.extend(tc.iter().map(tool_call_to_openai));
        }
        for block in &msg.content {
            if let ContentBlock::ToolUse(tc) = block {
                calls.push(tool_call_to_openai(tc));
            }
        }
        if !calls.is_empty() {
            entry.insert("tool_calls".into(), serde_json::json!(calls));
        }
    }

    if let Some(name) = &msg.name {
        entry.insert("name".into(), serde_json::json!(name));
    }

    serde_json::Value::Object(entry)
}

/// Convert content blocks into a string or a content-block array.
fn convert_content_blocks(blocks: &[ContentBlock]) -> serde_json::Value {
    if blocks.iter().all(|b| {
        matches!(b, ContentBlock::Text(_) | ContentBlock::Reasoning(_))
    }) {
        let text = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                ContentBlock::Reasoning(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return serde_json::json!(text);
    }

    let arr: Vec<serde_json::Value> = blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(serde_json::json!({"type": "text", "text": t})),
            ContentBlock::Reasoning(t) => {
                Some(serde_json::json!({"type": "text", "text": t}))
            }
            // ToolUse / ToolResult blocks belong in tool_calls / role=tool
            // messages, not the content array.
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) => None,
        })
        .collect();

    if arr.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::json!(arr)
    }
}

/// Prepend the system prompt to the content of a user message entry.
fn prepend_system_content(entry: &mut serde_json::Value, system: &str) {
    let Some(obj) = entry.as_object_mut() else {
        return;
    };
    match obj.get_mut("content") {
        Some(serde_json::Value::String(s)) => {
            let combined = if s.is_empty() {
                system.to_string()
            } else {
                format!("{system}\n\n{s}")
            };
            *s = combined;
        }
        Some(serde_json::Value::Array(arr)) => {
            arr.insert(0, serde_json::json!({"type": "text", "text": system}));
        }
        _ => {
            obj.insert("content".into(), serde_json::json!(system));
        }
    }
}

/// Convert a [`ToolCall`] into the OpenAI `tool_calls` wire entry.
fn tool_call_to_openai(call: &ToolCall) -> serde_json::Value {
    let args = serde_json::to_string(&call.input).unwrap_or_else(|_| "{}".to_string());
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": args,
        }
    })
}

/// Serialize tool definitions into the OpenAI `tools` array.
fn build_tools_array(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            })
        })
        .collect()
}

// ===========================================================================
// Response parsing
// ===========================================================================

/// Parse a non-streaming chat completions response into a [`ChatResponse`].
///
/// Handles both string and array `content`, `reasoning_content` (DeepSeek),
/// native `tool_calls`, and text-embedded tool calls that require dialect
/// normalization (DeepSeek DSML, XML, JSON blocks).
pub fn parse_chat_response(data: &serde_json::Value, policy: &CompatPolicy) -> ChatResponse {
    let id = data.get("id").and_then(|v| v.as_str()).map(String::from);
    let model = data
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let choice = &data["choices"][0];
    let message = &choice["message"];

    let mut content = parse_message_content(message);
    let reasoning_content = extract_reasoning_content(message);
    let mut tool_calls = parse_message_tool_calls(message);

    // Text-embedded tool calls (DeepSeek DSML, XML, JSON blocks).
    if tool_calls.is_empty() && !content.is_empty() && needs_text_normalization(policy) {
        let dialect = dialect_for_policy(policy);
        let normalizer = TextToolCallNormalizer::new(dialect);
        let extracted = normalizer.extract_and_normalize(&content);
        if !extracted.is_empty() {
            tool_calls = extracted;
            content = strip_text_tool_calls(&content, dialect);
        }
    }

    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .map(String::from);

    ChatResponse {
        id,
        content,
        reasoning_content,
        tool_calls,
        usage: parse_usage(data),
        model,
        finish_reason,
    }
}

/// Extract the text content from an OpenAI message object.
fn parse_message_content(message: &serde_json::Value) -> String {
    match message.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Some(serde_json::Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Extract native `tool_calls` from an OpenAI message object.
fn parse_message_tool_calls(message: &serde_json::Value) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let Some(arr) = message.get("tool_calls").and_then(|v| v.as_array()) else {
        return calls;
    };
    for tc in arr {
        let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let name = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args_str = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .unwrap_or("{}");
        let args = serde_json::from_str(args_str).unwrap_or_else(|_| serde_json::json!({}));
        calls.push(ToolCall::new(id, name, args));
    }
    calls
}

/// Extract reasoning content from a message object (DeepSeek et al.).
fn extract_reasoning_content(message: &serde_json::Value) -> Option<String> {
    for key in [
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "thinking_content",
        "thinking",
    ] {
        if let Some(v) = message.get(key).and_then(|v| v.as_str()) {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Extract token usage from a response object.
fn parse_usage(data: &serde_json::Value) -> Usage {
    let u = data.get("usage").unwrap_or(&serde_json::Value::Null);
    let input = u
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = u
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    Usage::new(input, output)
}

/// Map a policy's tool format to a text-normalizer dialect.
fn dialect_for_policy(policy: &CompatPolicy) -> ToolDialect {
    match policy.tool_format {
        ToolCallFormat::DeepSeekDsml => ToolDialect::DeepSeekDsml,
        ToolCallFormat::Xml => ToolDialect::Xml,
        ToolCallFormat::JsonBlock => ToolDialect::JsonBlock,
        _ => ToolDialect::OpenAi,
    }
}

/// Strip text-embedded tool-call markup from a response, leaving clean text.
fn strip_text_tool_calls(text: &str, dialect: ToolDialect) -> String {
    match dialect {
        ToolDialect::DeepSeekDsml => DSML_BLOCK_RE
            .replace_all(text, "")
            .trim()
            .to_string(),
        ToolDialect::Xml => XML_TOOL_CALL_RE
            .replace_all(text, "")
            .trim()
            .to_string(),
        ToolDialect::JsonBlock => JSON_BLOCK_RE.replace_all(text, "").trim().to_string(),
        _ => text.trim().to_string(),
    }
}

static DSML_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<tool_call>.*?</tool_call>").expect("invalid DSML strip regex")
});
static XML_TOOL_CALL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<tool_call\b[^>]*/>|<invoke\b[^>]*>.*?</invoke>"#)
        .expect("invalid XML strip regex")
});
static JSON_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?:json)?\s*\n.*?\n?```").expect("invalid JSON block strip regex")
});

// ===========================================================================
// Streaming
// ===========================================================================

/// Parse a single OpenAI-compatible SSE `data:` line into a [`StreamEvent`].
///
/// Handles text deltas, DeepSeek `reasoning_content` deltas, tool-call
/// deltas, the trailing chunk that carries `usage`, and `[DONE]`.
pub fn parse_openai_sse_event(
    data: &str,
    policy: &CompatPolicy,
) -> Option<ProviderResult<StreamEvent>> {
    if data == "[DONE]" {
        return Some(Ok(StreamEvent::Done {
            usage: None,
            stop_reason: None,
        }));
    }

    let value: serde_json::Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(e) => {
            warn!(target = "provider", error = %e, "Failed to parse OpenAI SSE JSON");
            return None;
        }
    };

    // Top-level error object.
    if value.get("error").is_some() {
        let msg = extract_error_message(&value["error"].to_string())
            .unwrap_or_else(|| "provider stream error".to_string());
        return Some(Ok(StreamEvent::Error { message: msg }));
    }

    // The final chunk carries cumulative usage.
    let mut final_usage: Option<Usage> = None;
    if let Some(u) = value.get("usage") {
        if !u.is_null() {
            final_usage = Some(parse_usage(&value));
        }
    }

    let choices = match value.get("choices").and_then(|v| v.as_array()) {
        Some(c) if !c.is_empty() => c,
        _ => {
            if let Some(usage) = final_usage {
                return Some(Ok(StreamEvent::Done {
                    usage: Some(usage),
                    stop_reason: None,
                }));
            }
            return None;
        }
    };

    let choice = &choices[0];
    let delta = choice.get("delta");
    let finish_reason = choice.get("finish_reason").and_then(|v| v.as_str());

    if let Some(delta) = delta {
        // Text content delta.
        if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                return Some(Ok(StreamEvent::Text {
                    text: text.to_string(),
                }));
            }
        }

        // Reasoning content delta (DeepSeek).
        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            if !r.is_empty() {
                return Some(Ok(StreamEvent::Reasoning {
                    reasoning: r.to_string(),
                }));
            }
        }
        if let Some(r) = delta.get("reasoning").and_then(|v| v.as_str()) {
            if !r.is_empty() && policy.reasoning != ReasoningPolicy::None {
                return Some(Ok(StreamEvent::Reasoning {
                    reasoning: r.to_string(),
                }));
            }
        }

        // Tool-call deltas.
        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if !id.is_empty() || !name.is_empty() || !args.is_empty() {
                    return Some(Ok(StreamEvent::ToolCall { id, name, arguments: args }));
                }
            }
        }
    }

    if let Some(reason) = finish_reason {
        if !reason.is_empty() && reason != "null" {
            return Some(Ok(StreamEvent::Done {
                usage: final_usage,
                stop_reason: Some(reason.to_string()),
            }));
        }
    }

    if let Some(usage) = final_usage {
        return Some(Ok(StreamEvent::Done {
            usage: Some(usage),
            stop_reason: None,
        }));
    }

    None
}

// ===========================================================================
// Error handling
// ===========================================================================

/// Map an HTTP error response to a [`ProviderError`].
///
/// Parses provider error bodies in the OpenAI shape
/// (`{"error": {"message": ..., "code": ...}}`) and maps common status codes
/// and error codes to typed errors.
pub fn map_http_error(status: StatusCode, body: &str, _provider: &str) -> ProviderError {
    let code = extract_error_code(body);
    let message = extract_error_message(body).unwrap_or_else(|| body.to_string());

    match status.as_u16() {
        401 | 403 => ProviderError::Auth(message),
        429 => ProviderError::RateLimited(message),
        408 => ProviderError::Timeout(message),
        400 => match code.as_deref() {
            Some("content_policy_violation")
            | Some("content_filter")
            | Some("content_filtered") => {
                ProviderError::Provider(format!("content filter: {message}"))
            }
            Some("context_length_exceeded")
            | Some("context_length")
            | Some("token_limit_exceeded") => {
                ProviderError::Provider(format!("context length exceeded: {message}"))
            }
            Some("invalid_api_key") => ProviderError::Auth(message),
            _ => ProviderError::Provider(format!("bad request: {message}")),
        },
        404 => ProviderError::UnsupportedModel(message),
        413 => ProviderError::Provider(format!("request too large: {message}")),
        500 | 502 | 503 | 504 => ProviderError::Provider(format!("server error: {message}")),
        _ => ProviderError::Provider(format!("HTTP {}: {message}", status.as_u16())),
    }
}

/// Extract the human-readable message from a provider error body.
pub fn extract_error_message(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(err) = v.get("error") {
        for key in ["message", "msg", "detail"] {
            if let Some(msg) = err.get(key).and_then(|m| m.as_str()) {
                return Some(msg.to_string());
            }
        }
        if let Some(msg) = err.as_str() {
            return Some(msg.to_string());
        }
    }
    for key in ["message", "msg", "detail"] {
        if let Some(msg) = v.get(key).and_then(|m| m.as_str()) {
            return Some(msg.to_string());
        }
    }
    v.as_str().map(String::from)
}

/// Extract a provider error code (e.g. `invalid_api_key`, `insufficient_quota`).
pub fn extract_error_code(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(err) = v.get("error") {
        if let Some(code) = err.get("code").and_then(|c| c.as_str()) {
            return Some(code.to_string());
        }
        if let Some(code) = err.get("code").and_then(|c| c.as_u64()) {
            return Some(code.to_string());
        }
        if let Some(t) = err.get("type").and_then(|c| c.as_str()) {
            return Some(t.to_string());
        }
    }
    if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
        return Some(code.to_string());
    }
    None
}

/// Classify an OpenAI-compatible error using the crate-wide failure taxonomy.
pub fn classify_openai_error(
    err: &ProviderError,
    status: Option<u16>,
    body: Option<&str>,
) -> crate::failures::ClassifiedError {
    classify_error(err, status, body)
}

/// Whether an OpenAI-compatible error is worth retrying, per the failure
/// taxonomy (rate limits, server errors, network/timeout, overload).
pub fn should_retry(err: &ProviderError, status: Option<u16>, body: Option<&str>) -> bool {
    classify_openai_error(err, status, body)
        .category
        .is_retryable()
}

// ===========================================================================
// Provider-specific behaviors
// ===========================================================================

/// Map a model id for a provider that requires a specific naming scheme.
///
/// - OpenRouter requires `vendor/model` prefixes; unprefixed ids get the
///   `openai/` prefix.
/// - Fireworks expects `accounts/fireworks/models/...` prefixes.
pub fn map_model(provider: OpenAIProvider, model: &str) -> String {
    match provider {
        OpenAIProvider::OpenRouter => {
            if model.contains('/') {
                model.to_string()
            } else {
                format!("openai/{model}")
            }
        }
        OpenAIProvider::Fireworks => {
            if model.starts_with("accounts/fireworks/models/") {
                model.to_string()
            } else {
                format!("accounts/fireworks/models/{model}")
            }
        }
        _ => model.to_string(),
    }
}

/// Whether a model id is a reasoning model (o1/o3/o4, deepseek-reasoner,
/// *-thinking-*).
pub fn is_reasoning_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.starts_with("o1")
        || m.starts_with("o3")
        || m.starts_with("o4")
        || m.contains("reasoner")
        || m.contains("thinking")
        || m.contains("deepseek-r1")
}

// ===========================================================================
// Catalog / integration helpers
// ===========================================================================

/// Seed a [`ModelCatalog`] with the static model list for an OpenAI-compatible
/// provider variant. Returns the number of entries inserted.
pub fn seed_catalog(catalog: &ModelCatalog, provider: OpenAIProvider) -> usize {
    let info = provider.info();
    let caps: Vec<ModelCapabilities> = info
        .models
        .iter()
        .map(|m| ModelCapabilities {
            model: m.to_string(),
            label: Some(m.to_string()),
            supports_tools: info.supports_tools,
            supports_streaming: info.supports_streaming,
            ..Default::default()
        })
        .collect();
    if caps.is_empty() {
        return 0;
    }
    catalog.upsert_many(info.id, caps);
    caps.len()
}

/// Merge a live-fetched model list into a catalog under this provider's id.
pub fn merge_live_models(catalog: &ModelCatalog, provider: OpenAIProvider, live: Vec<ModelCapabilities>) {
    merge_live(catalog, provider.as_str(), live);
}

/// Convert an OpenAI-compatible SSE `data:` payload into normalized stream
/// deltas suitable for [`crate::stream_assembly::StreamAssembler`].
///
/// This is the bridge between the raw wire format and the format-agnostic
/// [`SseDelta`](crate::stream_assembly::SseDelta) events consumed by the
/// stream assembler. A single chunk may carry multiple parallel tool-call
/// entries, so the result is always a vector.
pub fn sse_deltas(data: &str) -> Vec<crate::stream_assembly::SseDelta> {
    crate::stream_assembly::SseDelta::from_json_many(data)
}

/// Resolve the effective chat endpoint URL for a provider and base URL.
///
/// Handles the Azure deployment path and custom endpoints that already carry
/// the full `/chat/completions` suffix.
pub fn resolve_chat_url(provider: OpenAIProvider, base_url: &str, model: &str, api_version: Option<&str>) -> String {
    let base = base_url.trim_end_matches('/');
    match provider {
        OpenAIProvider::Azure => {
            let deployment = map_model(provider, model);
            let version = api_version.unwrap_or("2024-10-21");
            format!("{base}/openai/deployments/{deployment}/chat/completions?api-version={version}")
        }
        OpenAIProvider::Custom => {
            if base.ends_with("/chat/completions") {
                base.to_string()
            } else {
                format!("{base}/chat/completions")
            }
        }
        _ => format!("{base}/chat/completions"),
    }
}

/// Construct a [`CredentialPool`] from a list of API keys, or `None` when no
/// non-empty keys are provided (local providers).
pub fn build_credential_pool<I, S>(keys: I) -> Option<CredentialPool>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    CredentialPool::new(keys).ok()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{Message, MessageRole};

    fn chat_config(model: &str) -> ChatConfig {
        ChatConfig {
            model: model.to_string(),
            temperature: 0.7,
            max_tokens: 1024,
            top_p: 0.9,
            stop_sequences: vec!["\n".to_string()],
            stream: false,
            extra: HashMap::new(),
        }
    }

    fn msg(role: MessageRole, text: &str) -> ChatMessage {
        Message::text(role, text)
    }

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            msg(MessageRole::System, "You are helpful."),
            msg(MessageRole::User, "Hello"),
            msg(MessageRole::Assistant, "Hi there"),
            msg(MessageRole::User, "What is 2+2?"),
        ]
    }

    // ------------------------------------------------------------------
    // Enum / metadata tests
    // ------------------------------------------------------------------

    #[test]
    fn provider_round_trip_ids() {
        for provider in [
            OpenAIProvider::OpenAi,
            OpenAIProvider::DeepSeek,
            OpenAIProvider::Gemini,
            OpenAIProvider::DashScope,
            OpenAIProvider::Qwen,
            OpenAIProvider::Moonshot,
            OpenAIProvider::Mistral,
            OpenAIProvider::Groq,
            OpenAIProvider::Zhipu,
            OpenAIProvider::SiliconFlow,
            OpenAIProvider::OpenRouter,
            OpenAIProvider::Azure,
            OpenAIProvider::Together,
            OpenAIProvider::Fireworks,
            OpenAIProvider::Perplexity,
            OpenAIProvider::Anyscale,
            OpenAIProvider::Lepton,
            OpenAIProvider::Replicate,
            OpenAIProvider::Cohere,
            OpenAIProvider::Ai21,
            OpenAIProvider::Xai,
            OpenAIProvider::DeepInfra,
            OpenAIProvider::HuggingFace,
            OpenAIProvider::Novita,
            OpenAIProvider::Infermatic,
            OpenAIProvider::ModelScope,
            OpenAIProvider::Yi,
            OpenAIProvider::Baichuan,
            OpenAIProvider::MiniMax,
            OpenAIProvider::StepFun,
            OpenAIProvider::Lingyi,
            OpenAIProvider::InternLm,
            OpenAIProvider::Glm,
            OpenAIProvider::Hunyuan,
            OpenAIProvider::TencentHunyuan,
            OpenAIProvider::BaiduErnie,
            OpenAIProvider::IflytekSpark,
            OpenAIProvider::SenseTime,
            OpenAIProvider::Meituan,
            OpenAIProvider::Volcengine,
            OpenAIProvider::Lambda,
            OpenAIProvider::Hyperbolic,
            OpenAIProvider::Chutes,
            OpenAIProvider::Kluster,
            OpenAIProvider::InferenceNet,
            OpenAIProvider::NotDiamond,
            OpenAIProvider::LocalAi,
            OpenAIProvider::LlamaCpp,
            OpenAIProvider::Vllm,
            OpenAIProvider::Voyage,
            OpenAIProvider::ElevenLabs,
            OpenAIProvider::Stability,
            OpenAIProvider::Anthropic,
            OpenAIProvider::Custom,
        ] {
            let id = provider.as_str();
            assert_eq!(OpenAIProvider::from_str_id(id), Some(provider), "round trip {id}");
        }
    }

    #[test]
    fn provider_from_str_falls_back_to_custom() {
        assert_eq!(provider_from_id("not-a-real-provider"), OpenAIProvider::Custom);
        assert_eq!(provider_from_id("minimax"), OpenAIProvider::MiniMax);
        assert_eq!(provider_from_id("minimax_text"), OpenAIProvider::MiniMax);
    }

    #[test]
    fn provider_base_urls() {
        assert_eq!(OpenAIProvider::OpenAi.default_base_url(), "https://api.openai.com/v1");
        assert_eq!(OpenAIProvider::DeepSeek.default_base_url(), "https://api.deepseek.com/v1");
        assert_eq!(
            OpenAIProvider::Gemini.default_base_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            OpenAIProvider::DashScope.default_base_url(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        );
        assert_eq!(OpenAIProvider::OpenRouter.default_base_url(), "https://openrouter.ai/api/v1");
        assert_eq!(
            OpenAIProvider::SiliconFlow.default_base_url(),
            "https://api.siliconflow.cn/v1"
        );
    }

    #[test]
    fn provider_auth_shapes() {
        assert_eq!(OpenAIProvider::OpenAi.auth_header(), AuthHeader::Bearer);
        assert_eq!(OpenAIProvider::Azure.auth_header(), AuthHeader::AzureApiKey);
        assert_eq!(OpenAIProvider::Anthropic.auth_header(), AuthHeader::HeaderApiKey);
        assert_eq!(OpenAIProvider::LocalAi.auth_header(), AuthHeader::None);
        assert!(OpenAIProvider::LocalAi.is_local());
        assert!(OpenAIProvider::Vllm.is_local());
        assert!(!OpenAIProvider::OpenAi.is_local());
    }

    #[test]
    fn openrouter_requires_prefix() {
        assert!(OpenAIProvider::OpenRouter.requires_model_prefix());
        assert!(!OpenAIProvider::OpenAi.requires_model_prefix());
    }

    #[test]
    fn default_models_are_non_empty_for_key_providers() {
        for p in [
            OpenAIProvider::OpenAi,
            OpenAIProvider::DeepSeek,
            OpenAIProvider::Gemini,
            OpenAIProvider::DashScope,
            OpenAIProvider::Moonshot,
            OpenAIProvider::Mistral,
            OpenAIProvider::Groq,
            OpenAIProvider::Zhipu,
            OpenAIProvider::OpenRouter,
            OpenAIProvider::Together,
            OpenAIProvider::Fireworks,
            OpenAIProvider::Perplexity,
        ] {
            assert!(!p.default_models().is_empty(), "{} has no models", p.as_str());
            assert!(!p.default_model().is_empty(), "{} has no default model", p.as_str());
        }
    }

    // ------------------------------------------------------------------
    // Config tests
    // ------------------------------------------------------------------

    #[test]
    fn config_resolves_default_base_url() {
        let cfg = OpenAiConfig::new(OpenAIProvider::DeepSeek, "sk-test");
        let provider = OpenAiCompatProvider::from_config(cfg);
        assert_eq!(provider.api_base(), "https://api.deepseek.com/v1");
        assert_eq!(provider.kind(), OpenAIProvider::DeepSeek);
    }

    #[test]
    fn config_override_base_url() {
        let cfg = OpenAiConfig::new(OpenAIProvider::OpenAi, "sk-test")
            .with_base_url("https://proxy.example.com/v1");
        let provider = OpenAiCompatProvider::from_config(cfg);
        assert_eq!(provider.api_base(), "https://proxy.example.com/v1");
    }

    #[test]
    fn new_infers_kind_from_name() {
        let provider = OpenAiCompatProvider::new("groq", "https://api.groq.com/openai/v1", "key");
        assert_eq!(provider.kind(), OpenAIProvider::Groq);
        let unknown = OpenAiCompatProvider::new("mystery", "http://x", "key");
        assert_eq!(unknown.kind(), OpenAIProvider::Custom);
    }

    #[test]
    fn chat_url_standard_and_azure() {
        let provider = OpenAiCompatProvider::from_config(
            OpenAiConfig::new(OpenAIProvider::OpenAi, "k").with_base_url("https://api.openai.com/v1"),
        );
        assert_eq!(provider.chat_url("gpt-4o"), "https://api.openai.com/v1/chat/completions");

        let azure = OpenAiCompatProvider::from_config(
            OpenAiConfig::new(OpenAIProvider::Azure, "k")
                .with_base_url("https://myres.openai.azure.com")
                .with_api_version("2024-10-21"),
        );
        assert_eq!(
            azure.chat_url("gpt-4o"),
            "https://myres.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn custom_chat_url_with_suffix() {
        let provider = OpenAiCompatProvider::new(
            "custom",
            "https://myhost.example.com/v1/chat/completions",
            "",
        );
        assert_eq!(
            provider.chat_url("model-x"),
            "https://myhost.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn config_serde_round_trip() {
        let cfg = OpenAiConfig::new(OpenAIProvider::Moonshot, "sk-kimi")
            .with_organization("org-1")
            .with_max_retries(5);
        let json = serde_json::to_string(&cfg).unwrap();
        let back: OpenAiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.provider, OpenAIProvider::Moonshot);
        assert_eq!(back.api_key, "sk-kimi");
        assert_eq!(back.organization.as_deref(), Some("org-1"));
        assert_eq!(back.max_retries, Some(5));
    }

    // ------------------------------------------------------------------
    // Request building tests
    // ------------------------------------------------------------------

    #[test]
    fn build_request_basic() {
        let body = build_chat_request(&sample_messages(), &chat_config("gpt-4o"));
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["stop"], serde_json::json!(["\n"]));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[3]["role"], "user");
    }

    #[test]
    fn build_request_with_tools() {
        let tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather for a city".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }),
        }];
        let body = build_chat_request_full(&chat_config("gpt-4o"), &sample_messages(), &tools, false);
        let arr = body["tools"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "function");
        assert_eq!(arr[0]["function"]["name"], "get_weather");
        assert_eq!(arr[0]["function"]["parameters"]["properties"]["city"]["type"], "string");
    }

    #[test]
    fn build_request_with_assistant_tool_calls() {
        let mut messages = sample_messages();
        messages.push(ChatMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text("calling".into())],
            tool_calls: Some(vec![ToolCall::new("call_1", "get_weather", serde_json::json!({"city": "NYC"}))]),
            tool_call_id: None,
            tool_result: None,
            name: None,
        });
        messages.push(ChatMessage {
            role: MessageRole::Tool,
            content: vec![ContentBlock::Text("72f".into())],
            tool_call_id: Some("call_1".into()),
            tool_result: None,
            name: None,
        });
        let body = build_chat_request(&messages, &chat_config("gpt-4o"));
        let arr = body["messages"].as_array().unwrap();
        let assistant = &arr[4];
        let calls = assistant["tool_calls"].as_array().unwrap();
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"NYC"}"#);
        let tool = &arr[5];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "call_1");
        assert_eq!(tool["content"], "72f");
    }

    #[test]
    fn build_request_content_block_array() {
        let messages = vec![ChatMessage {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text("look".into()),
                ContentBlock::Reasoning("think".into()),
            ],
            tool_calls: None,
            tool_call_id: None,
            tool_result: None,
            name: None,
        }];
        let body = build_chat_request(&messages, &chat_config("gpt-4o"));
        let m = &body["messages"][0];
        // All-text/reasoning blocks collapse into a plain string.
        assert_eq!(m["content"], "look\nthink");
    }

    #[test]
    fn build_request_gemini_prepends_system() {
        let policy = policy_for("gemini");
        let body = build_chat_request_with_policy(
            &chat_config("gemini-2.0-flash"),
            &sample_messages(),
            &[],
            false,
            &policy,
        );
        let messages = body["messages"].as_array().unwrap();
        // System message folded into the first user message.
        assert_eq!(messages[0]["role"], "user");
        let content = messages[0]["content"].as_str().unwrap();
        assert!(content.starts_with("You are helpful."));
        assert!(content.contains("Hello"));
        // No system role remains.
        assert!(messages.iter().all(|m| m["role"] != "system"));
    }

    #[test]
    fn build_request_separate_system_field() {
        let mut policy = policy_for("anthropic");
        policy.system_prompt = SystemPromptPolicy::SeparateField;
        let body = build_chat_request_with_policy(
            &chat_config("claude-3-5-sonnet-20241022"),
            &sample_messages(),
            &[],
            false,
            &policy,
        );
        assert_eq!(body["system"], "You are helpful.");
        let messages = body["messages"].as_array().unwrap();
        assert!(messages.iter().all(|m| m["role"] != "system"));
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn build_request_clamps_max_tokens_for_groq() {
        let policy = policy_for("groq");
        let mut config = chat_config("llama-3.3-70b-versatile");
        config.max_tokens = 20000;
        let body = build_chat_request_with_policy(&config, &sample_messages(), &[], false, &policy);
        assert_eq!(body["max_tokens"], 8192);
    }

    #[test]
    fn build_request_merges_extras() {
        let mut config = chat_config("gpt-4o");
        config.extra.insert("seed".into(), serde_json::json!(42));
        config.extra.insert("user".into(), serde_json::json!("u-1"));
        let body = build_chat_request(&sample_messages(), &config);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["user"], "u-1");
    }

    // ------------------------------------------------------------------
    // Model mapping / reasoning detection
    // ------------------------------------------------------------------

    #[test]
    fn map_openrouter_prefix() {
        assert_eq!(map_model(OpenAIProvider::OpenRouter, "gpt-4o"), "openai/gpt-4o");
        assert_eq!(
            map_model(OpenAIProvider::OpenRouter, "anthropic/claude-3-5-sonnet"),
            "anthropic/claude-3-5-sonnet"
        );
    }

    #[test]
    fn map_fireworks_prefix() {
        assert_eq!(
            map_model(OpenAIProvider::Fireworks, "llama-v3p3-70b-instruct"),
            "accounts/fireworks/models/llama-v3p3-70b-instruct"
        );
        assert_eq!(
            map_model(OpenAIProvider::Fireworks, "accounts/fireworks/models/foo"),
            "accounts/fireworks/models/foo"
        );
    }

    #[test]
    fn reasoning_model_detection() {
        assert!(is_reasoning_model("o1"));
        assert!(is_reasoning_model("o3-mini"));
        assert!(is_reasoning_model("o4-mini"));
        assert!(is_reasoning_model("deepseek-reasoner"));
        assert!(is_reasoning_model("deepseek-r1"));
        assert!(!is_reasoning_model("gpt-4o"));
        assert!(!is_reasoning_model("claude-3-5-sonnet"));
    }

    #[test]
    fn reasoning_model_uses_max_completion_tokens() {
        let provider = OpenAiCompatProvider::new("openai", "https://api.openai.com/v1", "k");
        let config = chat_config("o3-mini");
        let body = provider.build_request_body(&config, &sample_messages(), &[], false);
        assert_eq!(body["max_completion_tokens"], 1024);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    // ------------------------------------------------------------------
    // Response parsing tests
    // ------------------------------------------------------------------

    #[test]
    fn parse_response_basic() {
        let data = serde_json::json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp = parse_chat_response(&data, &CompatPolicy::default_openai());
        assert_eq!(resp.content, "Hello there");
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn parse_response_content_array() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"}
                    ]
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&data, &CompatPolicy::default_openai());
        assert_eq!(resp.content, "Hello world");
    }

    #[test]
    fn parse_response_tool_calls() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}
        });
        let resp = parse_chat_response(&data, &CompatPolicy::default_openai());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_9");
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].input["city"], "SF");
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn parse_response_reasoning_content() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Answer",
                    "reasoning_content": "Thinking deeply..."
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&data, &CompatPolicy::default_openai());
        assert_eq!(resp.content, "Answer");
        assert_eq!(resp.reasoning_content.as_deref(), Some("Thinking deeply..."));
    }

    #[test]
    fn parse_response_deepseek_dsml_normalizes_tool_calls() {
        let policy = policy_for("deepseek");
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<tool_call><tool_name>get_weather</tool_name><parameters><city>NYC</city></parameters></tool_call>"
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&data, &policy);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        // The DSML markup is stripped from content.
        assert!(!resp.content.contains("<tool_call>"));
    }

    #[test]
    fn parse_response_json_block_normalizes() {
        let mut policy = CompatPolicy::default_openai();
        policy.tool_format = ToolCallFormat::JsonBlock;
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "```json\n{\"function\": \"search\", \"parameters\": {\"q\": \"rust\"}}\n```"
                },
                "finish_reason": "stop"
            }]
        });
        let resp = parse_chat_response(&data, &policy);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "search");
        assert_eq!(resp.tool_calls[0].input["q"], "rust");
    }

    // ------------------------------------------------------------------
    // SSE parsing tests
    // ------------------------------------------------------------------

    #[test]
    fn parse_sse_done() {
        let event = parse_openai_sse_event("[DONE]", &CompatPolicy::default_openai());
        assert!(event.is_some());
        match event.unwrap() {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                assert!(usage.is_none());
                assert!(stop_reason.is_none());
            }
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn parse_sse_text_delta() {
        let data = r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"finish_reason":null}]}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::Text { text } => assert_eq!(text, "Hel"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn parse_sse_reasoning_delta() {
        let data = r#"{"choices":[{"index":0,"delta":{"reasoning_content":"step 1"},"finish_reason":null}]}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::Reasoning { reasoning } => assert_eq!(reasoning, "step 1"),
            _ => panic!("expected Reasoning"),
        }
    }

    #[test]
    fn parse_sse_tool_call_delta() {
        let data = r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\":"}}]},"finish_reason":null}]}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::ToolCall { id, name, arguments } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"#);
            }
            _ => panic!("expected ToolCall"),
        }
    }

    #[test]
    fn parse_sse_finish_reason() {
        let data = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::Done { stop_reason, .. } => {
                assert_eq!(stop_reason.as_deref(), Some("stop"));
            }
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn parse_sse_final_usage_chunk() {
        let data = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::Done { usage, .. } => {
                let usage = usage.unwrap();
                assert_eq!(usage.input_tokens, 10);
                assert_eq!(usage.output_tokens, 4);
            }
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn parse_sse_error_object() {
        let data = r#"{"error":{"message":"boom","type":"server_error"}}"#;
        let event = parse_openai_sse_event(data, &CompatPolicy::default_openai()).unwrap().unwrap();
        match event {
            StreamEvent::Error { message } => assert!(message.contains("boom")),
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn parse_sse_malformed_returns_none() {
        assert!(parse_openai_sse_event("not json", &CompatPolicy::default_openai()).is_none());
        assert!(parse_openai_sse_event("{}", &CompatPolicy::default_openai()).is_none());
    }

    // ------------------------------------------------------------------
    // Error handling tests
    // ------------------------------------------------------------------

    #[test]
    fn map_http_error_status_codes() {
        assert!(matches!(
            map_http_error(StatusCode::UNAUTHORIZED, r#"{"error":{"message":"bad key"}}"#, "openai"),
            ProviderError::Auth(_)
        ));
        assert!(matches!(
            map_http_error(StatusCode::TOO_MANY_REQUESTS, r#"{"error":{"message":"slow down"}}"#, "openai"),
            ProviderError::RateLimited(_)
        ));
        assert!(matches!(
            map_http_error(StatusCode::NOT_FOUND, "no model", "openai"),
            ProviderError::UnsupportedModel(_)
        ));
        assert!(matches!(
            map_http_error(StatusCode::INTERNAL_SERVER_ERROR, "kaboom", "openai"),
            ProviderError::Provider(_)
        ));
    }

    #[test]
    fn map_http_error_context_length() {
        let err = map_http_error(
            StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"This model's maximum context length is 128000 tokens","code":"context_length_exceeded"}}"#,
            "openai",
        );
        match err {
            ProviderError::Provider(msg) => {
                assert!(msg.contains("context length exceeded"));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn extract_error_message_shapes() {
        assert_eq!(
            extract_error_message(r#"{"error":{"message":"msg1"}}"#).as_deref(),
            Some("msg1")
        );
        assert_eq!(extract_error_message(r#"{"message":"msg2"}"#).as_deref(), Some("msg2"));
        assert_eq!(extract_error_message(r#"just a string"#).as_deref(), Some("just a string"));
        assert_eq!(extract_error_message("not json at all {"), None);
    }

    #[test]
    fn extract_error_code_shapes() {
        assert_eq!(
            extract_error_code(r#"{"error":{"code":"invalid_api_key"}}"#).as_deref(),
            Some("invalid_api_key")
        );
        assert_eq!(
            extract_error_code(r#"{"error":{"code":429}}"#).as_deref(),
            Some("429")
        );
        assert_eq!(
            extract_error_code(r#"{"error":{"type":"server_error"}}"#).as_deref(),
            Some("server_error")
        );
        assert_eq!(extract_error_code("no code"), None);
    }

    #[test]
    fn classify_and_retry() {
        let err = map_http_error(
            StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":{"message":"rate"}}"#,
            "openai",
        );
        let classified = classify_openai_error(&err, Some(429), None);
        assert!(classified.category.is_retryable());
        assert!(should_retry(&err, Some(429), None));

        let auth_err = ProviderError::Auth("nope".into());
        assert!(!should_retry(&auth_err, Some(401), None));
    }

    // ------------------------------------------------------------------
    // Provider integration tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn supported_models_matches_provider() {
        let provider = OpenAiCompatProvider::new("deepseek", "https://api.deepseek.com/v1", "k");
        let models = provider.supported_models();
        assert!(models.contains(&"deepseek-chat".to_string()));
        assert!(models.contains(&"deepseek-reasoner".to_string()));
        assert_eq!(provider.name(), "deepseek");
    }

    #[tokio::test]
    async fn list_models_falls_back_to_static() {
        // A live fetch against an unreachable host should fail gracefully and
        // fall back to the static model list.
        let provider = OpenAiCompatProvider::from_config(
            OpenAiConfig::new(OpenAIProvider::OpenAi, "k")
                .with_base_url("http://127.0.0.1:1/v1"),
        );
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.model == "gpt-4o"));
    }

    #[tokio::test]
    async fn provider_trait_stream_rejects_unreachable() {
        // Point at a closed localhost port so the request fails fast with a
        // connection error, proving the wiring reaches the transport layer.
        let provider = OpenAiCompatProvider::from_config(
            OpenAiConfig::new(OpenAIProvider::OpenAi, "k")
                .with_base_url("http://127.0.0.1:1/v1"),
        );
        let cfg = chat_config("gpt-4o");
        let stream = provider.stream_chat(&cfg, &sample_messages(), &[]).await;
        assert!(stream.is_err());
    }

    #[test]
    fn seed_catalog_inserts_models() {
        let catalog = ModelCatalog::new();
        let n = seed_catalog(&catalog, OpenAIProvider::DeepSeek);
        assert_eq!(n, 2);
        assert!(catalog.get("deepseek", "deepseek-chat").is_some());
        let list = catalog.list("deepseek");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn resolve_chat_url_helper() {
        assert_eq!(
            resolve_chat_url(OpenAIProvider::OpenAi, "https://api.openai.com/v1", "gpt-4o", None),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            resolve_chat_url(
                OpenAIProvider::Azure,
                "https://res.openai.azure.com",
                "gpt-4o",
                Some("2024-10-21")
            ),
            "https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn credential_pool_single() {
        let pool = build_credential_pool(["sk-1", "sk-2"]);
        assert!(pool.is_some());
        assert_eq!(pool.unwrap().len(), 2);
        assert!(build_credential_pool::<[String; 0], String>([]).is_none());
    }

    #[test]
    fn from_spec_instantiates_compat_backend() {
        let spec = crate::registry::ProviderSpecTable::get("deepseek").unwrap();
        let provider = OpenAiCompatProvider::from_spec(&spec, "k");
        assert!(provider.is_some());
        assert_eq!(provider.unwrap().kind(), OpenAIProvider::DeepSeek);

        // Non-compat backends are rejected.
        let anthropic_spec = crate::registry::ProviderSpecTable::get("anthropic").unwrap();
        assert!(OpenAiCompatProvider::from_spec(&anthropic_spec, "k").is_none());
    }

    #[test]
    fn openrouter_policy_has_attribution_headers() {
        let provider =
            OpenAiCompatProvider::from_config(OpenAiConfig::new(OpenAIProvider::OpenRouter, "k"));
        assert_eq!(provider.policy.extra_headers.len(), 2);
        assert!(provider
            .policy
            .extra_headers
            .iter()
            .any(|(n, _)| *n == "HTTP-Referer"));
        assert!(provider
            .policy
            .extra_headers
            .iter()
            .any(|(n, _)| *n == "X-Title"));
    }

    #[test]
    fn build_request_stream_flag() {
        let body = build_chat_request_full(&chat_config("gpt-4o"), &sample_messages(), &[], true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn build_request_dashscope() {
        let policy = policy_for("dashscope");
        let body = build_chat_request_with_policy(
            &chat_config("qwen-plus"),
            &sample_messages(),
            &[],
            false,
            &policy,
        );
        assert_eq!(body["model"], "qwen-plus");
        // DashScope keeps native tools; max_tokens capped at 8192.
        let mut cfg = chat_config("qwen-max");
        cfg.max_tokens = 20000;
        let body = build_chat_request_with_policy(&cfg, &sample_messages(), &[], false, &policy);
        assert_eq!(body["max_tokens"], 8192);
    }

    #[test]
    fn config_extra_params_and_headers_wired() {
        let provider = OpenAiCompatProvider::from_config(
            OpenAiConfig::new(OpenAIProvider::OpenAi, "k")
                .with_param("seed", serde_json::json!(42))
                .with_header("X-Custom", "yes"),
        );
        let body = provider.build_request_body(&chat_config("gpt-4o"), &sample_messages(), &[], false);
        assert_eq!(body["seed"], 42);
        assert_eq!(
            provider.extra_headers.get("X-Custom").map(String::as_str),
            Some("yes")
        );
    }

    #[test]
    fn required_providers_request_building() {
        let required = [
            ("openai", OpenAIProvider::OpenAi, "gpt-4o"),
            ("deepseek", OpenAIProvider::DeepSeek, "deepseek-chat"),
            ("gemini", OpenAIProvider::Gemini, "gemini-2.0-flash"),
            ("dashscope", OpenAIProvider::DashScope, "qwen-plus"),
            ("qwen", OpenAIProvider::Qwen, "qwen-plus"),
            ("moonshot", OpenAIProvider::Moonshot, "moonshot-v1-128k"),
            ("mistral", OpenAIProvider::Mistral, "mistral-large-latest"),
            ("groq", OpenAIProvider::Groq, "llama-3.3-70b-versatile"),
            ("zhipu", OpenAIProvider::Zhipu, "glm-4-plus"),
            ("siliconflow", OpenAIProvider::SiliconFlow, "Qwen/Qwen2.5-72B-Instruct"),
            ("openrouter", OpenAIProvider::OpenRouter, "openai/gpt-4o"),
            ("azure", OpenAIProvider::Azure, "gpt-4o"),
            ("together", OpenAIProvider::Together, "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
            (
                "fireworks",
                OpenAIProvider::Fireworks,
                "accounts/fireworks/models/llama-v3p3-70b-instruct",
            ),
            ("perplexity", OpenAIProvider::Perplexity, "llama-3.1-sonar-large-128k-online"),
        ];
        for (id, provider, default_model) in required {
            assert_eq!(provider.as_str(), id, "{id} id mismatch");
            assert_eq!(provider.default_model(), default_model, "{id} default model");
            assert!(!provider.default_models().is_empty(), "{id} has no models");

            let cfg = chat_config(default_model);
            let body = build_chat_request(&sample_messages(), &cfg);
            assert_eq!(body["model"], default_model, "{id} body model");
            assert_eq!(
                body["messages"].as_array().map(|a| a.len()),
                Some(4),
                "{id} message count"
            );
        }
    }

    #[test]
    fn sse_delta_bridge() {
        let deltas = sse_deltas(r#"{"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#);
        assert_eq!(deltas.len(), 1);
        assert!(matches!(deltas[0], crate::stream_assembly::SseDelta::Text(_)));

        // A chunk carrying only a role marker yields no deltas.
        let empty = sse_deltas(r#"{"choices":[{"index":0,"delta":{"role":"assistant"}}]}"#);
        assert!(empty.is_empty());
    }

    #[test]
    fn stream_assembler_integration() {
        use crate::stream_assembly::StreamAssembler;
        let mut assembler = StreamAssembler::new();
        for d in sse_deltas(r#"{"choices":[{"delta":{"content":"Hello"}}]}"#) {
            assembler.push_delta(&d);
        }
        for d in sse_deltas(r#"{"choices":[{"delta":{"content":" world"}}]}"#) {
            assembler.push_delta(&d);
        }
        let blocks = assembler.finalize();
        assert!(!blocks.is_empty());
        assert!(matches!(
            &blocks[0],
            opensquilla_core::types::ContentBlock::Text(t) if t == "Hello world"
        ));
    }
}
