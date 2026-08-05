use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Information about a specific model, including its capabilities and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model identifier (e.g., "gpt-4", "claude-3-opus-20240229").
    pub id: String,
    /// A human-readable display name for the model.
    pub name: String,
    /// The provider that serves this model.
    pub provider: String,
    /// The capabilities this model supports.
    pub capabilities: ModelCapabilities,
    /// The context window size in tokens.
    pub context_window: u64,
    /// Maximum output tokens the model can generate.
    pub max_output_tokens: u64,
    /// Pricing information for this model.
    #[serde(default)]
    pub pricing: ModelPricing,
    /// Additional metadata for this model.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// The capabilities a model supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    /// Whether the model supports streaming responses.
    #[serde(default = "default_true")]
    pub streaming: bool,
    /// Whether the model supports tool/function calling.
    #[serde(default = "default_true")]
    pub tool_calling: bool,
    /// Whether the model supports parallel tool calls.
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// Whether the model supports vision/image inputs.
    #[serde(default)]
    pub vision: bool,
    /// Whether the model supports audio inputs.
    #[serde(default)]
    pub audio: bool,
    /// Whether the model supports system prompts.
    #[serde(default = "default_true")]
    pub system_prompt: bool,
    /// Whether the model supports reasoning/thinking output.
    #[serde(default)]
    pub reasoning: bool,
    /// Whether the model supports JSON mode / structured output.
    #[serde(default)]
    pub json_mode: bool,
    /// Whether the model supports function calling (non-tool-based).
    #[serde(default)]
    pub function_calling: bool,
    /// Whether the model supports image generation.
    #[serde(default)]
    pub image_generation: bool,
    /// Whether the model supports embedding.
    #[serde(default)]
    pub embedding: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            streaming: true,
            tool_calling: true,
            parallel_tool_calls: false,
            vision: false,
            audio: false,
            system_prompt: true,
            reasoning: false,
            json_mode: false,
            function_calling: false,
            image_generation: false,
            embedding: false,
        }
    }
}

/// Pricing information for a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// Price per 1,000 input tokens in USD.
    pub input_price_per_1k: f64,
    /// Price per 1,000 output tokens in USD.
    pub output_price_per_1k: f64,
    /// Currency for pricing.
    #[serde(default = "default_currency")]
    pub currency: String,
}

fn default_currency() -> String {
    "USD".to_string()
}

impl Default for ModelPricing {
    fn default() -> Self {
        Self {
            input_price_per_1k: 0.0,
            output_price_per_1k: 0.0,
            currency: default_currency(),
        }
    }
}

impl ModelPricing {
    /// Calculate the cost for a given number of input and output tokens.
    pub fn cost_for_tokens(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = (input_tokens as f64 / 1000.0) * self.input_price_per_1k;
        let output_cost = (output_tokens as f64 / 1000.0) * self.output_price_per_1k;
        input_cost + output_cost
    }
}

/// Specification for a provider's model offerings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// The provider name (e.g., "openai", "anthropic").
    pub name: String,
    /// The display name for the provider.
    pub display_name: String,
    /// The base URL for API requests.
    pub base_url: String,
    /// The API version header value, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    /// The list of models available from this provider.
    pub models: Vec<ModelInfo>,
    /// Supported authentication methods.
    #[serde(default)]
    pub auth_methods: Vec<AuthMethod>,
    /// Headers to include in all requests to this provider.
    #[serde(default)]
    pub default_headers: HashMap<String, String>,
}

/// Authentication method supported by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// API key sent via header.
    ApiKey,
    /// Bearer token authentication.
    BearerToken,
    /// OAuth 2.0 authentication.
    OAuth2,
    /// Basic authentication (username/password).
    Basic,
    /// Custom authentication method with a description.
    Custom(String),
}

/// A collection of models indexed by provider and model ID.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRegistry {
    /// Models indexed by provider name, then by model ID.
    #[serde(default)]
    pub models: HashMap<String, HashMap<String, ModelInfo>>,
}

impl ModelRegistry {
    /// Register a model in the registry.
    pub fn register(&mut self, model: ModelInfo) {
        self.models
            .entry(model.provider.clone())
            .or_default()
            .insert(model.id.clone(), model);
    }

    /// Look up a model by provider and model ID.
    pub fn get(&self, provider: &str, model_id: &str) -> Option<&ModelInfo> {
        self.models.get(provider)?.get(model_id)
    }

    /// Get all models for a given provider.
    pub fn get_provider_models(&self, provider: &str) -> Option<&HashMap<String, ModelInfo>> {
        self.models.get(provider)
    }

    /// Find a model across all providers by model ID (returns first match).
    pub fn find_model(&self, model_id: &str) -> Option<&ModelInfo> {
        self.models
            .values()
            .flat_map(|models| models.values())
            .find(|m| m.id == model_id)
    }
}

/// Model selection strategy for routing requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Always use the default model.
    Default,
    /// Select the cheapest model that meets requirements.
    Cheapest,
    /// Select the fastest model that meets requirements.
    Fastest,
    /// Select the most capable model.
    MostCapable,
    /// Round-robin selection across available models.
    RoundRobin,
    /// Custom selection strategy.
    Custom(String),
}