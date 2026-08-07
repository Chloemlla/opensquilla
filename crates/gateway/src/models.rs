//! Models catalog RPC handlers.
//!
//! Provides `rpc_models` for listing the model catalog, backed by the core
//! crate's [`ModelRegistry`] and [`ModelInfo`] types.

use opensquilla_core::error::AppError;
use opensquilla_core::model::{ModelInfo, ModelRegistry};
use parking_lot::RwLock;
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A shared model catalog.
#[derive(Clone, Default)]
pub struct ModelCatalog {
    registry: Arc<RwLock<ModelRegistry>>,
}

impl ModelCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a catalog seeded with a few common models.
    pub fn with_defaults() -> Self {
        let catalog = Self::new();
        let models = vec![
            ModelInfo {
                id: "gpt-4o".to_string(),
                name: "GPT-4o".to_string(),
                provider: "openai".to_string(),
                capabilities: Default::default(),
                context_window: 128000,
                max_output_tokens: 16384,
                pricing: Default::default(),
                metadata: Default::default(),
            },
            ModelInfo {
                id: "claude-sonnet-4-20250514".to_string(),
                name: "Claude Sonnet 4".to_string(),
                provider: "anthropic".to_string(),
                capabilities: Default::default(),
                context_window: 200000,
                max_output_tokens: 8192,
                pricing: Default::default(),
                metadata: Default::default(),
            },
            ModelInfo {
                id: "deepseek-chat".to_string(),
                name: "DeepSeek Chat".to_string(),
                provider: "deepseek".to_string(),
                capabilities: Default::default(),
                context_window: 128000,
                max_output_tokens: 4096,
                pricing: Default::default(),
                metadata: Default::default(),
            },
        ];
        for model in models {
            catalog.register(model);
        }
        catalog
    }

    /// Register a model in the catalog.
    pub fn register(&self, model: ModelInfo) {
        self.registry.write().register(model);
    }

    /// Look up a model by provider and model id.
    pub fn get(&self, provider: &str, model_id: &str) -> Option<ModelInfo> {
        self.registry.read().get(provider, model_id).cloned()
    }

    /// Find a model across all providers by model id.
    pub fn find(&self, model_id: &str) -> Option<ModelInfo> {
        self.registry.read().find_model(model_id).cloned()
    }

    /// List all models.
    pub fn list(&self) -> Vec<ModelInfo> {
        let registry = self.registry.read();
        registry
            .models
            .values()
            .flat_map(|m| m.values().cloned())
            .collect()
    }

    /// List models for a specific provider.
    pub fn list_by_provider(&self, provider: &str) -> Vec<ModelInfo> {
        self.registry
            .read()
            .get_provider_models(provider)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// List all provider names.
    pub fn providers(&self) -> Vec<String> {
        let registry = self.registry.read();
        let mut names: Vec<String> = registry.models.keys().cloned().collect();
        names.sort();
        names
    }
}

/// Register models RPC handlers on the given registry.
pub fn register_models_handlers(registry: &mut RpcRegistry, catalog: ModelCatalog) {
    let catalog = Arc::new(catalog);

    // models.list — list all models in the catalog
    registry.register(rpc_handler("models.list", {
        let catalog = catalog.clone();
        move |_params| {
            let catalog = catalog.clone();
            async move {
                let models = catalog.list();
                Ok(serde_json::json!({
                    "models": models,
                    "count": models.len(),
                }))
            }
        }
    }));

    // models.get — fetch a model by id (searches all providers)
    registry.register(rpc_handler("models.get", {
        let catalog = catalog.clone();
        move |params| {
            let catalog = catalog.clone();
            async move {
                let model_id = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?;
                match catalog.find(model_id) {
                    Some(model) => Ok(serde_json::to_value(model)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Model '{model_id}' not found"))),
                }
            }
        }
    }));

    // models.by_provider — list models for a specific provider
    registry.register(rpc_handler("models.by_provider", {
        let catalog = catalog.clone();
        move |params| {
            let catalog = catalog.clone();
            async move {
                let provider = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'provider' parameter"))?;
                let models = catalog.list_by_provider(provider);
                Ok(serde_json::json!({
                    "provider": provider,
                    "models": models,
                    "count": models.len(),
                }))
            }
        }
    }));

    // models.providers — list all provider names in the catalog
    registry.register(rpc_handler("models.providers", {
        let catalog = catalog.clone();
        move |_params| {
            let catalog = catalog.clone();
            async move {
                let providers = catalog.providers();
                Ok(serde_json::json!({
                    "providers": providers,
                    "count": providers.len(),
                }))
            }
        }
    }));

    // models.register — register a new model in the catalog
    registry.register(rpc_handler("models.register", {
        let catalog = catalog.clone();
        move |params| {
            let catalog = catalog.clone();
            async move {
                let model: ModelInfo = serde_json::from_value(params)
                    .map_err(|e| AppError::bad_request(format!("Invalid model spec: {e}")))?;
                catalog.register(model.clone());
                Ok(serde_json::to_value(model).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_models_list_defaults() {
        let catalog = ModelCatalog::with_defaults();
        let mut registry = RpcRegistry::new();
        register_models_handlers(&mut registry, catalog);

        let r = registry
            .dispatch("models.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_models_get_by_id() {
        let catalog = ModelCatalog::with_defaults();
        let mut registry = RpcRegistry::new();
        register_models_handlers(&mut registry, catalog);

        let r = registry
            .dispatch("models.get", serde_json::json!({"model": "gpt-4o"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["provider"], "openai");
    }

    #[tokio::test]
    async fn test_models_by_provider() {
        let catalog = ModelCatalog::with_defaults();
        let mut registry = RpcRegistry::new();
        register_models_handlers(&mut registry, catalog);

        let r = registry
            .dispatch(
                "models.by_provider",
                serde_json::json!({"provider": "anthropic"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_models_providers_list() {
        let catalog = ModelCatalog::with_defaults();
        let mut registry = RpcRegistry::new();
        register_models_handlers(&mut registry, catalog);

        let r = registry
            .dispatch("models.providers", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() >= 3);
    }
}
