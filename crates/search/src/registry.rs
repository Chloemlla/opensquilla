use std::collections::HashMap;

use opensquilla_core::config::Config;
use tracing::{debug, info};

use crate::bocha::BochaSearch;
use crate::brave::BraveSearch;
use crate::duckduckgo::DuckDuckGoSearch;
use crate::exa::ExaSearch;
use crate::iqs::IqsSearch;
use crate::tavily::TavilySearch;
use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse};

/// Registry of all available search providers.
pub struct SearchRegistry {
    providers: HashMap<String, Box<dyn SearchProvider>>,
    default_provider: String,
}

impl SearchRegistry {
    /// Create a new search registry with all built-in providers.
    pub fn new(config: &Config) -> Self {
        let mut providers: HashMap<String, Box<dyn SearchProvider>> = HashMap::new();

        // Register Brave Search
        let brave = BraveSearch::new(config);
        providers.insert(brave.name().to_string(), Box::new(brave));

        // Register DuckDuckGo
        let duckduckgo = DuckDuckGoSearch::new();
        providers.insert(duckduckgo.name().to_string(), Box::new(duckduckgo));

        // Register Tavily
        let tavily = TavilySearch::new(config);
        providers.insert(tavily.name().to_string(), Box::new(tavily));

        // Register Exa
        let exa = ExaSearch::new(config);
        providers.insert(exa.name().to_string(), Box::new(exa));

        // Register Bocha
        let bocha = BochaSearch::new(config);
        providers.insert(bocha.name().to_string(), Box::new(bocha));

        // Register IQS
        let iqs = IqsSearch::new(config);
        providers.insert(iqs.name().to_string(), Box::new(iqs));

        let default_provider = config
            .get("search.default_provider")
            .unwrap_or_else(|| "duckduckgo".to_string());

        info!(
            "Search registry created with {} providers, default: {default_provider}",
            providers.len()
        );

        Self {
            providers,
            default_provider,
        }
    }

    /// Get a search provider by name.
    pub fn get(&self, name: &str) -> Option<&dyn SearchProvider> {
        self.providers.get(name).map(|p| p.as_ref())
    }

    /// Search using the default provider.
    pub async fn search_default(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        self.search(&self.default_provider, request).await
    }

    /// Search using a specific provider.
    pub async fn search(&self, provider_name: &str, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| SearchError::ConfigError(format!("Unknown provider: {provider_name}")))?;

        debug!("Searching with provider: {provider_name}");
        let response = provider.search(request).await?;
        Ok(response)
    }

    /// List all registered provider names.
    pub fn list_providers(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.providers.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// List providers that are ready (configured) to use.
    pub fn list_ready_providers(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .providers
            .iter()
            .filter(|(_, p)| p.is_ready())
            .map(|(name, _)| name.as_str())
            .collect();
        names.sort();
        names
    }

    /// Register a custom search provider.
    pub fn register(&mut self, provider: Box<dyn SearchProvider>) {
        let name = provider.name().to_string();
        self.providers.insert(name.clone(), provider);
        info!("Registered search provider: {name}");
    }

    /// Set the default provider.
    pub fn set_default(&mut self, name: &str) {
        self.default_provider = name.to_string();
    }

    /// Get the default provider name.
    pub fn default_provider(&self) -> &str {
        &self.default_provider
    }
}