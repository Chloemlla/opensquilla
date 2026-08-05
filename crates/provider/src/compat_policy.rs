//! OpenAI compatibility policy dialect data.
//!
//! Although dozens of providers expose an "OpenAI-compatible" `/v1/chat/completions`
//! endpoint, almost every one deviates in some way: system-prompt handling, tool
//! call wire format, parameter name spelling, supported fields, reasoning
//! exposure, and hard limits. This module captures those per-provider quirks as
//! declarative data so the request builder, normalizer, and request-proof layers
//! can adapt a single canonical payload to each dialect without scattering
//! `if provider == "..."` checks across the codebase.
//!
//! This is one of the two load-bearing modules for cross-provider correctness
//! (the other being [`crate::request_proof`]).

use std::collections::HashMap;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Enums describing the dialect axes
// ---------------------------------------------------------------------------

/// How a provider expects the system prompt to be delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SystemPromptPolicy {
    /// Send the system message as a `{"role":"system"}` entry (standard OpenAI).
    #[default]
    SystemRole,
    /// Anthropic-style: a top-level `system` field, no system role in messages.
    SeparateField,
    /// Prepend the system text to the first user message (Gemini OpenAI-compat shim).
    PrependToFirstUser,
    /// Drop system messages entirely and merge into the first user turn.
    MergeIntoFirstUser,
}

/// The wire format a provider uses to emit (and accept) tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallFormat {
    /// Native OpenAI `tool_calls` / `tools` JSON structure.
    #[default]
    Native,
    /// DeepSeek-style structured text (DSML) requiring normalization.
    DeepSeekDsml,
    /// Generic XML `<tool_call .../>` requiring normalization.
    Xml,
    /// Fenced JSON code blocks requiring normalization.
    JsonBlock,
    /// Provider does not support tool/function calling.
    Unsupported,
}

/// Parameter-name overrides for providers that spell things differently.
///
/// Maps the canonical OpenAI parameter name to the provider-specific name.
#[derive(Debug, Clone, Default)]
pub struct ParameterOverrides {
    /// e.g. `max_tokens` -> `max_completion_tokens` for some OpenAI responses APIs.
    pub max_tokens: Option<&'static str>,
    /// e.g. some providers use `top_k` instead of `top_p`.
    pub top_p: Option<&'static str>,
}

/// Reasoning / thinking exposure mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningPolicy {
    /// Reasoning is not exposed.
    #[default]
    None,
    /// Reasoning arrives inline as a `reasoning_content` field (DeepSeek).
    ReasoningContentField,
    /// Reasoning arrives as a dedicated content block type (Anthropic).
    ReasoningBlock,
    /// Reasoning arrives as a separate `thinking` SSE event.
    ThinkingEvent,
}

// ---------------------------------------------------------------------------
// The policy record
// ---------------------------------------------------------------------------

/// A complete compatibility policy for a single provider backend.
///
/// All fields are `Copy`/cheap-clone primitives or `&'static str` so a policy
/// can be handed around freely.
#[derive(Debug, Clone)]
pub struct CompatPolicy {
    /// The canonical provider id this policy applies to (e.g. "deepseek").
    pub provider: &'static str,
    /// How system prompts are delivered.
    pub system_prompt: SystemPromptPolicy,
    /// Tool-call wire format.
    pub tool_format: ToolCallFormat,
    /// Parameter name overrides.
    pub params: ParameterOverrides,
    /// Reasoning exposure mode.
    pub reasoning: ReasoningPolicy,
    /// Whether streaming is supported over SSE.
    pub supports_streaming: bool,
    /// Whether function/tool calling is supported at all.
    pub supports_tools: bool,
    /// Whether `temperature` is honored.
    pub supports_temperature: bool,
    /// Whether `top_p` is honored.
    pub supports_top_p: bool,
    /// Whether `stop` sequences are honored.
    pub supports_stop: bool,
    /// Hard ceiling on `max_tokens` if the provider enforces one.
    pub max_tokens_cap: Option<u32>,
    /// Whether the provider accepts `n > 1` (multiple completions).
    pub supports_n: bool,
    /// Whether `logprobs` is supported.
    pub supports_logprobs: bool,
    /// Whether the provider requires the model id to be prefixed (e.g. openrouter).
    pub requires_model_prefix: bool,
    /// Extra headers to send with every request (header name, value).
    pub extra_headers: &'static [(&'static str, &'static str)],
    /// A free-form notes string for diagnostics.
    pub notes: &'static str,
}

impl CompatPolicy {
    /// The default OpenAI-compatible policy used when a provider has no
    /// specific entry.
    pub const fn default_openai() -> Self {
        Self {
            provider: "default",
            system_prompt: SystemPromptPolicy::SystemRole,
            tool_format: ToolCallFormat::Native,
            params: ParameterOverrides {
                max_tokens: None,
                top_p: None,
            },
            reasoning: ReasoningPolicy::None,
            supports_streaming: true,
            supports_tools: true,
            supports_temperature: true,
            supports_top_p: true,
            supports_stop: true,
            max_tokens_cap: None,
            supports_n: true,
            supports_logprobs: true,
            requires_model_prefix: false,
            extra_headers: &[],
            notes: "Default OpenAI-compatible policy",
        }
    }
}

impl Default for CompatPolicy {
    fn default() -> Self {
        Self::default_openai()
    }
}

// ---------------------------------------------------------------------------
// Static policy table — one entry per registered provider backend
// ---------------------------------------------------------------------------

/// Build the static policy table keyed by provider id.
fn build_table() -> HashMap<&'static str, CompatPolicy> {
    let mut m = HashMap::new();

    // Helper macro to keep the table compact and readable.
    macro_rules! p {
        ($id:expr, $sys:expr, $tool:expr, $reason:expr, $stream:expr,
         $tools:expr, $temp:expr, $topp:expr, $stop:expr, $cap:expr,
         $n:expr, $logp:expr, $prefix:expr, $hdrs:expr, $notes:expr) => {
            m.insert(
                $id,
                CompatPolicy {
                    provider: $id,
                    system_prompt: $sys,
                    tool_format: $tool,
                    params: ParameterOverrides {
                        max_tokens: None,
                        top_p: None,
                    },
                    reasoning: $reason,
                    supports_streaming: $stream,
                    supports_tools: $tools,
                    supports_temperature: $temp,
                    supports_top_p: $topp,
                    supports_stop: $stop,
                    max_tokens_cap: $cap,
                    supports_n: $n,
                    supports_logprobs: $logp,
                    requires_model_prefix: $prefix,
                    extra_headers: $hdrs,
                    notes: $notes,
                },
            );
        };
    }

    // --- OpenAI core -------------------------------------------------------
    p!(
        "openai",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        true,
        true,
        false,
        &[],
        "OpenAI canonical API"
    );
    p!(
        "openai_responses",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::ReasoningBlock,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "OpenAI Responses API (/v1/responses) with reasoning items"
    );
    p!(
        "openai_codex",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::ReasoningContentField,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "OpenAI Codex code API (/v1/codex)"
    );

    // --- DeepSeek: DSML tool text, reasoning_content field -----------------
    p!(
        "deepseek",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::DeepSeekDsml,
        ReasoningPolicy::ReasoningContentField,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "DeepSeek emits tool calls as DSML text and reasoning via reasoning_content"
    );

    // --- Gemini (OpenAI-compat shim) ---------------------------------------
    // Gemini's OpenAI-compatible endpoint folds system prompts into the first
    // user turn and does not support n/logprobs.
    p!(
        "gemini",
        SystemPromptPolicy::PrependToFirstUser,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "Google Gemini OpenAI-compat shim: system prompt prepended to first user message"
    );

    // --- Qwen / DashScope --------------------------------------------------
    // DashScope spells `max_tokens` as `max_tokens` but some plans prefer
    // `max_completion_tokens`; tools are native. Qwen-token-plan-anthropic is
    // handled under the anthropic backend separately.
    p!(
        "dashscope",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "Alibaba DashScope (Qwen) OpenAI-compat endpoint"
    );
    p!(
        "qwen",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "Qwen (alias of dashscope)"
    );

    // --- Moonshot (Kimi) ----------------------------------------------------
    p!(
        "moonshot",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "Moonshot Kimi OpenAI-compat"
    );

    // --- Mistral -----------------------------------------------------------
    p!(
        "mistral",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        true,
        false,
        &[],
        "Mistral La Plateforme OpenAI-compat"
    );

    // --- Groq: very fast, low max_tokens cap -------------------------------
    p!(
        "groq",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "Groq: fast inference, capped max_tokens"
    );

    // --- Zhipu (GLM) -------------------------------------------------------
    p!(
        "zhipu",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "Zhipu GLM OpenAI-compat"
    );

    // --- SiliconFlow -------------------------------------------------------
    p!(
        "siliconflow",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "SiliconFlow OpenAI-compat aggregator"
    );

    // --- OpenRouter: requires "provider/model" prefixed model ids ----------
    p!(
        "openrouter",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        true,
        true,
        true,
        &[
            ("HTTP-Referer", "https://opensquilla.dev"),
            ("X-Title", "OpenSquilla"),
        ],
        "OpenRouter: requires provider-prefixed model ids"
    );

    // --- Azure OpenAI ------------------------------------------------------
    p!(
        "azure",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        true,
        true,
        false,
        &[("api-key", "")],
        "Azure OpenAI: key in api-key header, deployment in path"
    );

    // --- Volcengine / BytePlus coding plans (Responses-style) --------------
    p!(
        "volcengine_coding_plan",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::ReasoningBlock,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "Volcengine coding plan (Responses-style API)"
    );
    p!(
        "byteplus_coding_plan",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::ReasoningBlock,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "BytePlus coding plan (Responses-style API)"
    );

    // --- Together / Fireworks / Anyscale / Lepton (generic compat) ---------
    for id in [
        "together",
        "fireworks",
        "anyscale",
        "lepton",
        "perplexity",
        "novita",
        "huggingface",
        "infermatic",
        "ai21",
        "cohere",
        "xai",
        "deepinfra",
        "modelscope",
        "yi",
        "baichuan",
        "minimax_text",
        "stepfun",
        "lingyi",
        "internlm",
        "glm",
        "hunyuan",
        "tencent_hunyuan",
        "baidu_ernie",
        "iflytek_spark",
        "sensetime",
        "meituan",
        "lambda",
        "hyperbolic",
        "chutes",
        "kluster",
        "inference_net",
        "not_diamond",
        "localai",
        "llama_cpp",
        "vllm",
        "voyage",
        "elevenlabs",
        "stability",
    ] {
        p!(
            id,
            SystemPromptPolicy::SystemRole,
            ToolCallFormat::Native,
            ReasoningPolicy::None,
            true,
            true,
            true,
            true,
            true,
            None,
            false,
            false,
            false,
            &[],
            "Generic OpenAI-compatible provider"
        );
    }

    // --- Anthropic backend -------------------------------------------------
    p!(
        "anthropic",
        SystemPromptPolicy::SeparateField,
        ToolCallFormat::Native,
        ReasoningPolicy::ReasoningBlock,
        true,
        true,
        true,
        true,
        true,
        Some(8192),
        false,
        false,
        false,
        &[],
        "Anthropic Messages API: top-level system field, reasoning blocks"
    );
    p!(
        "minimax",
        SystemPromptPolicy::SeparateField,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "MiniMax (Anthropic-style backend)"
    );
    p!(
        "qwen_token_plan_anthropic",
        SystemPromptPolicy::SeparateField,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "Qwen token plan via Anthropic-style backend"
    );

    // --- Ollama ------------------------------------------------------------
    p!(
        "ollama",
        SystemPromptPolicy::SystemRole,
        ToolCallFormat::Native,
        ReasoningPolicy::None,
        true,
        true,
        true,
        true,
        true,
        None,
        false,
        false,
        false,
        &[],
        "Ollama local API: options bag instead of top-level params"
    );

    m
}

static TABLE: LazyLock<HashMap<&'static str, CompatPolicy>> = LazyLock::new(build_table);

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Thread-safe read-only registry of per-provider compatibility policies.
#[derive(Debug, Clone, Default)]
pub struct CompatPolicyRegistry;

impl CompatPolicyRegistry {
    /// Look up the policy for a provider id.
    ///
    /// Falls back to [`CompatPolicy::default_openai`] when the provider is
    /// unknown, which is the safest assumption for an OpenAI-compatible
    /// endpoint.
    pub fn get(&self, provider: &str) -> CompatPolicy {
        TABLE.get(provider).cloned().unwrap_or_default()
    }

    /// Returns `true` if a provider-specific policy is registered.
    pub fn contains(&self, provider: &str) -> bool {
        TABLE.contains_key(provider)
    }

    /// List all registered provider ids.
    pub fn providers(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = TABLE.keys().copied().collect();
        v.sort();
        v
    }
}

/// Convenience free function: look up a policy for a provider id.
pub fn policy_for(provider: &str) -> CompatPolicy {
    CompatPolicyRegistry.get(provider)
}

// ---------------------------------------------------------------------------
// Payload adaptation helpers
// ---------------------------------------------------------------------------

/// Rewrite a canonical OpenAI parameter map according to a policy.
///
/// Removes unsupported parameters, applies name overrides, and clamps
/// `max_tokens` to the provider cap. Returns the adapted map.
pub fn adapt_parameters(
    mut params: serde_json::Map<String, serde_json::Value>,
    policy: &CompatPolicy,
) -> serde_json::Map<String, serde_json::Value> {
    // max_tokens override + cap
    if let Some(max_val) = params.remove("max_tokens") {
        let cap = policy.max_tokens_cap;
        let capped = match (max_val.as_u64(), cap) {
            (Some(n), Some(c)) => serde_json::json!(n.min(c as u64)),
            _ => max_val,
        };
        let key = policy.params.max_tokens.unwrap_or("max_tokens");
        params.insert(key.to_string(), capped);
    } else if policy.supports_temperature {
        // leave as-is
    }

    if !policy.supports_temperature {
        params.remove("temperature");
    }
    if !policy.supports_top_p {
        params.remove("top_p");
        params.remove("top_p_alt");
    }
    if !policy.supports_stop {
        params.remove("stop");
    }
    if !policy.supports_n {
        params.remove("n");
    }
    if !policy.supports_logprobs {
        params.remove("logprobs");
        params.remove("top_logprobs");
    }

    // top_p name override
    if let Some(tp) = policy.params.top_p {
        if let Some(v) = params.remove("top_p") {
            params.insert(tp.to_string(), v);
        }
    }

    params
}

/// Decide whether text-based tool-call normalization is required for a policy.
pub fn needs_text_normalization(policy: &CompatPolicy) -> bool {
    matches!(
        policy.tool_format,
        ToolCallFormat::DeepSeekDsml | ToolCallFormat::Xml | ToolCallFormat::JsonBlock
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_fallback() {
        let reg = CompatPolicyRegistry;
        let p = reg.get("does-not-exist");
        assert_eq!(p.system_prompt, SystemPromptPolicy::SystemRole);
        assert_eq!(p.tool_format, ToolCallFormat::Native);
    }

    #[test]
    fn test_deepseek_policy() {
        let p = policy_for("deepseek");
        assert_eq!(p.tool_format, ToolCallFormat::DeepSeekDsml);
        assert_eq!(p.reasoning, ReasoningPolicy::ReasoningContentField);
        assert_eq!(p.max_tokens_cap, Some(8192));
        assert!(needs_text_normalization(&p));
    }

    #[test]
    fn test_gemini_system_prompt() {
        let p = policy_for("gemini");
        assert_eq!(p.system_prompt, SystemPromptPolicy::PrependToFirstUser);
    }

    #[test]
    fn test_anthropic_system_prompt() {
        let p = policy_for("anthropic");
        assert_eq!(p.system_prompt, SystemPromptPolicy::SeparateField);
        assert_eq!(p.reasoning, ReasoningPolicy::ReasoningBlock);
    }

    #[test]
    fn test_adapt_parameters_caps_max_tokens() {
        let p = policy_for("groq"); // cap 8192
        let mut m = serde_json::Map::new();
        m.insert("max_tokens".into(), serde_json::json!(20000));
        m.insert("n".into(), serde_json::json!(2)); // unsupported
        let out = adapt_parameters(m, &p);
        assert_eq!(out["max_tokens"], 8192);
        assert!(!out.contains_key("n"));
    }

    #[test]
    fn test_adapt_parameters_renames() {
        let mut p = policy_for("openai");
        p.params.max_tokens = Some("max_completion_tokens");
        let mut m = serde_json::Map::new();
        m.insert("max_tokens".into(), serde_json::json!(1000));
        let out = adapt_parameters(m, &p);
        assert_eq!(out["max_completion_tokens"], 1000);
        assert!(!out.contains_key("max_tokens"));
    }

    #[test]
    fn test_registry_has_many_providers() {
        let reg = CompatPolicyRegistry;
        for id in [
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
        ] {
            assert!(reg.contains(id), "missing {id}");
        }
        assert!(reg.providers().len() >= 40);
    }
}
