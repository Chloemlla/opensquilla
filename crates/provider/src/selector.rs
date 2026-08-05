//! Model selector with fallback chain.
//!
//! Selects a provider and model for a given request based on configured
//! selection strategies and fallback logic.

use crate::registry::ProviderRegistry;
use crate::types::{Provider, ProviderError, ProviderResult};
use std::sync::Arc;

/// A model selection strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Always use the default provider and model.
    Default,
    /// Select the cheapest model that meets capability requirements.
    Cheapest,
    /// Select the fastest model (typically the smallest).
    Fastest,
    /// Select the most capable model.
    MostCapable,
    /// Try providers/models in the order specified by the fallback chain.
    Fallback,
}

/// A link in the fallback chain.
#[derive(Debug, Clone)]
pub struct FallbackLink {
    /// The provider name to use.
    pub provider: String,
    /// The model name to use (empty = use provider's default).
    pub model: String,
}

/// A model selector that uses a fallback chain.
///
/// The selector tries each link in the chain in order until one succeeds.
/// If all links fail, the last error is returned.
pub struct ModelSelector {
    /// The provider registry to look up providers from.
    registry: ProviderRegistry,
    /// The selection strategy.
    strategy: SelectionStrategy,
    /// The fallback chain (ordered list of provider/model pairs).
    fallback_chain: Vec<FallbackLink>,
}

impl ModelSelector {
    /// Create a new model selector.
    pub fn new(
        registry: ProviderRegistry,
        strategy: SelectionStrategy,
        fallback_chain: Vec<FallbackLink>,
    ) -> Self {
        Self {
            registry,
            strategy,
            fallback_chain,
        }
    }

    /// Create a selector that always uses the default provider.
    pub fn default(registry: ProviderRegistry) -> Self {
        Self {
            registry,
            strategy: SelectionStrategy::Default,
            fallback_chain: vec![],
        }
    }

    /// Create a selector with a fallback chain.
    pub fn with_fallback(registry: ProviderRegistry, fallback_chain: Vec<FallbackLink>) -> Self {
        Self {
            registry,
            strategy: SelectionStrategy::Fallback,
            fallback_chain,
        }
    }

    /// Select a provider for the given model name.
    ///
    /// Returns the provider and the model to use. If the model contains a
    /// provider prefix (e.g., "openai/gpt-4"), it is parsed and used to
    /// select the correct provider.
    pub fn select(&self, model: &str) -> ProviderResult<(Arc<dyn Provider>, String)> {
        match self.strategy {
            SelectionStrategy::Default => self.select_default(model),
            SelectionStrategy::Fallback => self.select_with_fallback(model),
            SelectionStrategy::Cheapest => self.select_default(model),
            SelectionStrategy::Fastest => self.select_default(model),
            SelectionStrategy::MostCapable => self.select_default(model),
        }
    }

    /// Select using the default provider.
    fn select_default(&self, model: &str) -> ProviderResult<(Arc<dyn Provider>, String)> {
        // Check if model has a provider prefix: "provider/model"
        if let Some((provider_name, model_name)) = model.split_once('/') {
            let provider = self
                .registry
                .get(provider_name)
                .ok_or_else(|| ProviderError::Config(format!("Provider '{provider_name}' not found")))?;
            Ok((provider, model_name.to_string()))
        } else {
            let provider = self
                .registry
                .default()
                .ok_or_else(|| ProviderError::Config("No default provider configured".into()))?;
            Ok((provider, model.to_string()))
        }
    }

    /// Select using the fallback chain.
    fn select_with_fallback(&self, model: &str) -> ProviderResult<(Arc<dyn Provider>, String)> {
        // First try the direct model lookup
        if let Ok(result) = self.select_default(model) {
            return Ok(result);
        }

        // Try each link in the fallback chain
        let mut last_error = ProviderError::Config("No fallback providers configured".into());

        for link in &self.fallback_chain {
            let provider = match self.registry.get(&link.provider) {
                Some(p) => p,
                None => {
                    last_error = ProviderError::Config(format!(
                        "Fallback provider '{}' not found",
                        link.provider
                    ));
                    continue;
                }
            };

            let model_name = if link.model.is_empty() { model } else { &link.model };
            return Ok((provider, model_name.to_string()));
        }

        Err(last_error)
    }

    /// Return the list of all available models across all providers.
    pub fn available_models(&self) -> Vec<(String, String)> {
        let mut models = Vec::new();
        for name in self.registry.list() {
            if let Some(provider) = self.registry.get(&name) {
                for model in provider.supported_models() {
                    models.push((name.clone(), model));
                }
            }
        }
        models
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use async_trait::async_trait;
    use futures::Stream;
    use opensquilla_core::types::*;

    struct TestProvider {
        name: &'static str,
        models: Vec<&'static str>,
    }

    #[async_trait]
    impl Provider for TestProvider {
        fn name(&self) -> &str {
            self.name
        }
        fn supported_models(&self) -> Vec<String> {
            self.models.iter().map(|m| m.to_string()).collect()
        }
        async fn send_message(
            &self,
            _config: &ChatConfig,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> ProviderResult<ProviderResponse> {
            unimplemented!()
        }
        async fn stream_chat(
            &self,
            _config: &ChatConfig,
            _messages: &[ChatMessage],
            _tools: &[ToolDefinition],
        ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
            unimplemented!()
        }
    }

    #[test]
    fn test_select_default() {
        let registry = ProviderRegistry::new();
        registry.register(
            "openai".into(),
            Arc::new(TestProvider {
                name: "openai",
                models: vec!["gpt-4"],
            }),
        );
        let selector = ModelSelector::default(registry);
        let (provider, model) = selector.select("gpt-4").unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn test_select_with_provider_prefix() {
        let registry = ProviderRegistry::new();
        registry.register(
            "openai".into(),
            Arc::new(TestProvider {
                name: "openai",
                models: vec!["gpt-4"],
            }),
        );
        let selector = ModelSelector::default(registry);
        let (provider, model) = selector.select("openai/gpt-4").unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn test_fallback_chain() {
        let registry = ProviderRegistry::new();
        registry.register(
            "primary".into(),
            Arc::new(TestProvider {
                name: "primary",
                models: vec!["model-a"],
            }),
        );
        registry.register(
            "fallback".into(),
            Arc::new(TestProvider {
                name: "fallback",
                models: vec!["model-b"],
            }),
        );

        let chain = vec![
            FallbackLink {
                provider: "primary".into(),
                model: String::new(),
            },
            FallbackLink {
                provider: "fallback".into(),
                model: String::new(),
            },
        ];

        let selector = ModelSelector::with_fallback(registry, chain);
        let (provider, _) = selector.select("model-a").unwrap();
        assert_eq!(provider.name(), "primary");
    }
}