//! Live catalog: startup-time fetch from provider `/v1/models` endpoints.
//!
//! At startup (or on demand), [`LiveCatalog`] queries each configured
//! provider's `/v1/models` (or equivalent) endpoint to discover the set of
//! models currently available, then merges the result with the static catalog
//! via [`crate::model_catalog::merge_live`].
//!
//! All HTTP is done with raw `reqwest` (no SDK), matching the Python backend's
//! approach. Failed fetches are logged but never abort startup — the static
//! catalog remains as a fallback.

use crate::compat_policy::policy_for;
use crate::model_catalog::{ModelCapabilities, ModelCatalog, merge_live};
use crate::types::ProviderError;
use reqwest::Client;
use std::time::Duration;
use tracing::{debug, info, warn};

/// A single provider's connection details for a live fetch.
#[derive(Debug, Clone)]
pub struct LiveProviderConfig {
    /// The provider id (must match a `CompatPolicy` / catalog key).
    pub provider: String,
    /// The API base URL (e.g. "https://api.openai.com/v1").
    pub api_base: String,
    /// Optional bearer token.
    pub api_key: Option<String>,
    /// Optional override for the models endpoint path (default `/models`).
    pub models_path: Option<String>,
}

impl LiveProviderConfig {
    /// Create a new live provider config.
    pub fn new(provider: impl Into<String>, api_base: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            api_base: api_base.into(),
            api_key: None,
            models_path: None,
        }
    }

    /// Set the API key.
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the models endpoint path.
    pub fn with_models_path(mut self, path: impl Into<String>) -> Self {
        self.models_path = Some(path.into());
        self
    }

    /// The full URL to the models endpoint.
    pub fn models_url(&self) -> String {
        let path = self.models_path.as_deref().unwrap_or("/models");
        let base = self.api_base.trim_end_matches('/');
        format!("{base}{path}")
    }
}

/// The result of fetching a single provider's live model list.
#[derive(Debug, Clone)]
pub struct LiveFetchResult {
    /// The provider id.
    pub provider: String,
    /// The models discovered (empty on failure).
    pub models: Vec<ModelCapabilities>,
    /// Error message if the fetch failed.
    pub error: Option<String>,
}

impl LiveFetchResult {
    /// Whether the fetch succeeded.
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
}

/// Fetches live model lists from provider `/v1/models` endpoints and merges
/// them into a [`ModelCatalog`].
#[derive(Clone)]
pub struct LiveCatalog {
    client: Client,
    catalog: ModelCatalog,
}

impl LiveCatalog {
    /// Create a new live catalog fetcher that writes into `catalog`.
    pub fn new(catalog: ModelCatalog) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .expect("Failed to create reqwest Client");
        Self { client, catalog }
    }

    /// Create with a custom reqwest client (e.g. for testing or proxy config).
    pub fn with_client(catalog: ModelCatalog, client: Client) -> Self {
        Self { client, catalog }
    }

    /// Fetch the live model list for a single provider.
    ///
    /// Returns the parsed capabilities. Errors are logged and returned.
    pub async fn fetch_one(&self, cfg: &LiveProviderConfig) -> LiveFetchResult {
        let url = cfg.models_url();
        debug!(target = "provider", provider = %cfg.provider, url = %url, "Fetching live models");

        let mut req = self.client.get(&url);
        if let Some(key) = &cfg.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }
        // Apply extra headers from the compatibility policy.
        let policy = policy_for(&cfg.provider);
        for (name, value) in policy.extra_headers {
            if !value.is_empty() {
                req = req.header(*name, *value);
            }
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(target = "provider", provider = %cfg.provider, error = %e, "Live models fetch failed");
                return LiveFetchResult {
                    provider: cfg.provider.clone(),
                    models: Vec::new(),
                    error: Some(e.to_string()),
                };
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!(target = "provider", provider = %cfg.provider, status = %status, "Live models fetch non-2xx");
            return LiveFetchResult {
                provider: cfg.provider.clone(),
                models: Vec::new(),
                error: Some(format!("HTTP {status}: {body}")),
            };
        }

        let data: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => {
                warn!(target = "provider", provider = %cfg.provider, error = %e, "Live models JSON parse failed");
                return LiveFetchResult {
                    provider: cfg.provider.clone(),
                    models: Vec::new(),
                    error: Some(e.to_string()),
                };
            }
        };

        let models = parse_models_response(&cfg.provider, &data);
        info!(
            target = "provider",
            provider = %cfg.provider,
            count = models.len(),
            "Live models fetched"
        );
        LiveFetchResult {
            provider: cfg.provider.clone(),
            models,
            error: None,
        }
    }

    /// Fetch and merge live models for a single provider into the catalog.
    pub async fn refresh_one(&self, cfg: &LiveProviderConfig) -> LiveFetchResult {
        let result = self.fetch_one(cfg).await;
        if result.ok() {
            merge_live(&self.catalog, &cfg.provider, result.models.clone());
            self.catalog.mark_refreshed(&cfg.provider);
        }
        result
    }

    /// Fetch and merge live models for many providers concurrently.
    ///
    /// Each provider is fetched in its own task; failures are isolated and do
    /// not abort the others. Returns one result per provider.
    pub async fn refresh_many(&self, configs: Vec<LiveProviderConfig>) -> Vec<LiveFetchResult> {
        let mut handles = Vec::with_capacity(configs.len());
        for cfg in configs {
            let this = self.clone();
            handles.push(tokio::spawn(async move { this.refresh_one(&cfg).await }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.await {
                Ok(r) => results.push(r),
                Err(e) => results.push(LiveFetchResult {
                    provider: String::new(),
                    models: Vec::new(),
                    error: Some(format!("join error: {e}")),
                }),
            }
        }
        results
    }

    /// Return a reference to the underlying catalog.
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }
}

/// Parse a `/v1/models` JSON response into capabilities.
///
/// Handles the common OpenAI-compatible shape `{"data": [{"id": "...", ...}]}`
/// as well as a bare array `[{"id": "..."}]` and Anthropic's
/// `{"data": [{"id": "...", "display_name": "..."}]}`.
pub fn parse_models_response(provider: &str, data: &serde_json::Value) -> Vec<ModelCapabilities> {
    let arr = data
        .get("data")
        .and_then(|d| d.as_array())
        .or_else(|| data.as_array());

    let Some(arr) = arr else {
        return Vec::new();
    };

    arr.iter()
        .filter_map(|m| parse_one_model(provider, m))
        .collect()
}

fn parse_one_model(provider: &str, m: &serde_json::Value) -> Option<ModelCapabilities> {
    // The model id may be under "id" (OpenAI) or "name".
    let id = m
        .get("id")
        .and_then(|v| v.as_str())
        .or_else(|| m.get("name").and_then(|v| v.as_str()))?;
    if id.is_empty() {
        return None;
    }

    let label = m
        .get("display_name")
        .and_then(|v| v.as_str())
        .or_else(|| m.get("label").and_then(|v| v.as_str()))
        .map(String::from);

    // Context window may be advertised by some providers.
    let context_window = m
        .get("context_length")
        .and_then(|v| v.as_u64())
        .or_else(|| m.get("context_window").and_then(|v| v.as_u64()))
        .map(|n| n as u32);

    let max_output_tokens = m
        .get("max_output_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| m.get("max_tokens").and_then(|v| v.as_u64()))
        .map(|n| n as u32);

    // Capabilities may be advertised as a list of strings or booleans.
    let caps_list = m
        .get("capabilities")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let supports_tools = caps_list.iter().any(|c| {
        let c = c.to_ascii_lowercase();
        c.contains("tool") || c.contains("function")
    }) || m
        .get("supports_tool_calls")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let supports_vision = caps_list.iter().any(|c| c.contains("vision"))
        || m.get("supports_vision")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    let supports_audio = caps_list.iter().any(|c| c.contains("audio"))
        || m.get("supports_audio")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
    let supports_reasoning = caps_list
        .iter()
        .any(|c| c.contains("reasoning") || c.contains("thinking"))
        || m.get("supports_reasoning")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

    let input_price = m
        .get("pricing")
        .and_then(|p| p.get("prompt"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());
    let output_price = m
        .get("pricing")
        .and_then(|p| p.get("completion"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok());

    Some(ModelCapabilities {
        model: id.to_string(),
        label,
        context_window,
        max_output_tokens,
        supports_tools,
        supports_vision,
        supports_audio,
        supports_reasoning,
        supports_streaming: true,
        // OpenRouter returns prices per-token; convert to per-million.
        input_price_per_million: input_price.map(|p| p * 1_000_000.0),
        output_price_per_million: output_price.map(|p| p * 1_000_000.0),
        tags: caps_list,
    })
    .map(|mut c| {
        // Tag with provider for diagnostics.
        c.tags.push(format!("provider:{provider}"));
        c
    })
}

/// Convenience: convert a `reqwest::Error` into a `ProviderError`.
pub fn reqwest_to_provider_error(e: reqwest::Error) -> ProviderError {
    if e.is_timeout() {
        ProviderError::Timeout(e.to_string())
    } else if e.is_connect() {
        ProviderError::Network(e)
    } else {
        ProviderError::Provider(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_shape() {
        let data = serde_json::json!({
            "data": [
                {"id": "gpt-4o", "context_length": 128000},
                {"id": "gpt-4o-mini"}
            ]
        });
        let models = parse_models_response("openai", &data);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].model, "gpt-4o");
        assert_eq!(models[0].context_window, Some(128_000));
    }

    #[test]
    fn test_parse_array_shape() {
        let data = serde_json::json!([
            {"id": "llama3.1"},
            {"name": "qwen2.5"}
        ]);
        let models = parse_models_response("ollama", &data);
        assert_eq!(models.len(), 2);
        assert_eq!(models[1].model, "qwen2.5");
    }

    #[test]
    fn test_parse_capabilities_and_pricing() {
        let data = serde_json::json!({
            "data": [{
                "id": "llama-3.3-70b-versatile",
                "capabilities": ["tools", "vision"],
                "pricing": {"prompt": "0.00000059", "completion": "0.00000079"}
            }]
        });
        let models = parse_models_response("groq", &data);
        assert_eq!(models.len(), 1);
        assert!(models[0].supports_tools);
        assert!(models[0].supports_vision);
        assert!((models[0].input_price_per_million.unwrap() - 0.59).abs() < 0.01);
    }

    #[test]
    fn test_models_url() {
        let cfg = LiveProviderConfig::new("openai", "https://api.openai.com/v1");
        assert_eq!(cfg.models_url(), "https://api.openai.com/v1/models");

        let cfg = LiveProviderConfig::new("azure", "https://x.openai.azure.com")
            .with_models_path("/openai/deployments?api-version=2024-06-01");
        assert_eq!(
            cfg.models_url(),
            "https://x.openai.azure.com/openai/deployments?api-version=2024-06-01"
        );
    }

    #[test]
    fn test_parse_empty() {
        let data = serde_json::json!({});
        assert!(parse_models_response("p", &data).is_empty());
    }
}
