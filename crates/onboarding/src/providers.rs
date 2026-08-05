use serde::{Deserialize, Serialize};

/// A provider specification for onboarding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// Internal name (e.g., "openai", "anthropic", "deepseek").
    pub name: String,
    /// Display name for the UI.
    pub display_name: String,
    /// Description of the provider.
    pub description: String,
    /// Whether this provider requires an API key.
    pub requires_api_key: bool,
    /// Whether this provider has a free tier available.
    pub has_free_tier: bool,
    /// The base URL for API requests.
    pub base_url: String,
    /// Available models for this provider.
    pub models: Vec<ModelSpec>,
    /// Documentation URL.
    pub docs_url: Option<String>,
    /// Whether this provider is a proxy/relay.
    pub is_proxy: bool,
}

/// A model specification for onboarding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model identifier (e.g., "gpt-4", "claude-3-opus").
    pub id: String,
    /// Display name.
    pub display_name: String,
    /// Context window size in tokens.
    pub context_window: u64,
    /// Whether this is the recommended model for this provider.
    pub recommended: bool,
    /// Whether this model supports vision/image inputs.
    pub supports_vision: bool,
    /// Whether this model supports function/tool calling.
    pub supports_tools: bool,
    /// Whether this model supports streaming.
    pub supports_streaming: bool,
    /// Approximate cost per 1M input tokens in USD.
    pub cost_per_1m_input: f64,
    /// Approximate cost per 1M output tokens in USD.
    pub cost_per_1m_output: f64,
}

/// Discover available providers.
pub fn discover_providers() -> Vec<ProviderSpec> {
    vec![
        ProviderSpec {
            name: "openai".to_string(),
            display_name: "OpenAI".to_string(),
            description: "GPT-4, GPT-4o, and GPT-3.5 models by OpenAI".to_string(),
            requires_api_key: true,
            has_free_tier: false,
            base_url: "https://api.openai.com/v1".to_string(),
            models: vec![
                ModelSpec {
                    id: "gpt-4o".to_string(),
                    display_name: "GPT-4o".to_string(),
                    context_window: 128000,
                    recommended: true,
                    supports_vision: true,
                    supports_tools: true,
                    supports_streaming: true,
                    cost_per_1m_input: 5.0,
                    cost_per_1m_output: 15.0,
                },
                ModelSpec {
                    id: "gpt-4o-mini".to_string(),
                    display_name: "GPT-4o Mini".to_string(),
                    context_window: 128000,
                    recommended: false,
                    supports_vision: true,
                    supports_tools: true,
                    supports_streaming: true,
                    cost_per_1m_input: 0.15,
                    cost_per_1m_output: 0.60,
                },
                ModelSpec {
                    id: "gpt-4-turbo".to_string(),
                    display_name: "GPT-4 Turbo".to_string(),
                    context_window: 128000,
                    recommended: false,
                    supports_vision: true,
                    supports_tools: true,
                    supports_streaming: true,
                    cost_per_1m_input: 10.0,
                    cost_per_1m_output: 30.0,
                },
            ],
            docs_url: Some("https://platform.openai.com/docs".to_string()),
            is_proxy: false,
        },
        ProviderSpec {
            name: "anthropic".to_string(),
            display_name: "Anthropic".to_string(),
            description: "Claude models by Anthropic".to_string(),
            requires_api_key: true,
            has_free_tier: false,
            base_url: "https://api.anthropic.com/v1".to_string(),
            models: vec![
                ModelSpec {
                    id: "claude-sonnet-4-20250514".to_string(),
                    display_name: "Claude Sonnet 4".to_string(),
                    context_window: 200000,
                    recommended: true,
                    supports_vision: true,
                    supports_tools: true,
                    supports_streaming: true,
                    cost_per_1m_input: 3.0,
                    cost_per_1m_output: 15.0,
                },
                ModelSpec {
                    id: "claude-haiku-3-5-20241022".to_string(),
                    display_name: "Claude Haiku 3.5".to_string(),
                    context_window: 200000,
                    recommended: false,
                    supports_vision: true,
                    supports_tools: true,
                    supports_streaming: true,
                    cost_per_1m_input: 0.80,
                    cost_per_1m_output: 4.0,
                },
            ],
            docs_url: Some("https://docs.anthropic.com".to_string()),
            is_proxy: false,
        },
        ProviderSpec {
            name: "deepseek".to_string(),
            display_name: "DeepSeek".to_string(),
            description: "DeepSeek models (DeepSeek-V2, DeepSeek-Coder)".to_string(),
            requires_api_key: true,
            has_free_tier: false,
            base_url: "https://api.deepseek.com/v1".to_string(),
            models: vec![ModelSpec {
                id: "deepseek-chat".to_string(),
                display_name: "DeepSeek Chat".to_string(),
                context_window: 128000,
                recommended: true,
                supports_vision: false,
                supports_tools: true,
                supports_streaming: true,
                cost_per_1m_input: 0.14,
                cost_per_1m_output: 0.28,
            }],
            docs_url: Some("https://platform.deepseek.com/docs".to_string()),
            is_proxy: false,
        },
        ProviderSpec {
            name: "ollama".to_string(),
            display_name: "Ollama (Local)".to_string(),
            description: "Local LLMs via Ollama".to_string(),
            requires_api_key: false,
            has_free_tier: true,
            base_url: "http://localhost:11434".to_string(),
            models: vec![
                ModelSpec {
                    id: "llama3.2".to_string(),
                    display_name: "Llama 3.2".to_string(),
                    context_window: 128000,
                    recommended: true,
                    supports_vision: false,
                    supports_tools: false,
                    supports_streaming: true,
                    cost_per_1m_input: 0.0,
                    cost_per_1m_output: 0.0,
                },
                ModelSpec {
                    id: "mistral".to_string(),
                    display_name: "Mistral".to_string(),
                    context_window: 32000,
                    recommended: false,
                    supports_vision: false,
                    supports_tools: false,
                    supports_streaming: true,
                    cost_per_1m_input: 0.0,
                    cost_per_1m_output: 0.0,
                },
            ],
            docs_url: Some("https://ollama.ai".to_string()),
            is_proxy: false,
        },
        ProviderSpec {
            name: "openrouter".to_string(),
            display_name: "OpenRouter".to_string(),
            description: "Multi-provider proxy via OpenRouter".to_string(),
            requires_api_key: true,
            has_free_tier: true,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            models: vec![ModelSpec {
                id: "openrouter/auto".to_string(),
                display_name: "Auto (best model)".to_string(),
                context_window: 128000,
                recommended: true,
                supports_vision: true,
                supports_tools: true,
                supports_streaming: true,
                cost_per_1m_input: 0.0,
                cost_per_1m_output: 0.0,
            }],
            docs_url: Some("https://openrouter.ai/docs".to_string()),
            is_proxy: true,
        },
    ]
}

/// Get a specific provider by name.
pub fn get_provider(name: &str) -> Option<ProviderSpec> {
    discover_providers().into_iter().find(|p| p.name == name)
}
