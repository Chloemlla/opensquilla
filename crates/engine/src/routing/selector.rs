//! Model selector with fallback chains.
//!
//! Mirrors the Python `provider/selector.py` `ModelSelector`. The selector
//! resolves the active deployment from a primary [`ProviderConfig`] plus an
//! ordered fallback chain, and exposes the chain-walking API the runtime uses
//! for failover. It is also health-aware: [`ModelSelector::select_model`]
//! consults the shared [`ProviderHealthLedger`] so a benched deployment is
//! skipped when an unbenched alternative exists (never-strand rule).
//!
//! The engine crate does not construct wire-protocol provider objects; this
//! selector works against plain config and delegates the actual HTTP call to
//! the provider crate (feature-gated).

use crate::routing::health_ledger::ProviderHealthLedger;
use opensquilla_core::model::{ModelInfo, ModelRegistry};
use std::collections::HashMap;
use std::sync::Arc;

/// Runtime configuration for a single provider.
#[derive(Clone)]
pub struct ProviderConfig {
    /// Provider id (e.g. `"anthropic"`, `"openai"`, `"ollama"`).
    pub provider: String,
    /// Model id.
    pub model: String,
    /// API key (redacted in Debug).
    pub api_key: String,
    /// Base URL override.
    pub base_url: String,
    /// Organization id.
    pub org_id: String,
    /// Explicit HTTP proxy URL.
    pub proxy: String,
    /// Provider-routing overrides.
    pub provider_routing: HashMap<String, String>,
    /// Whether provider-private continuity state may be replayed.
    pub replay_provider_state: bool,
}

impl ProviderConfig {
    /// Create a new provider config.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            api_key: String::new(),
            base_url: String::new(),
            org_id: String::new(),
            proxy: String::new(),
            provider_routing: HashMap::new(),
            replay_provider_state: true,
        }
    }

    /// Set the API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .field("org_id", &self.org_id)
            .field("proxy", &self.proxy)
            .field("provider_routing", &self.provider_routing)
            .field("replay_provider_state", &self.replay_provider_state)
            .finish()
    }
}

/// Full model selection config: primary + ordered fallback chain.
#[derive(Debug, Clone)]
pub struct SelectorConfig {
    /// The primary provider config.
    pub primary: ProviderConfig,
    /// Ordered fallback configs.
    pub fallbacks: Vec<ProviderConfig>,
}

impl SelectorConfig {
    /// Create a new selector config.
    pub fn new(primary: ProviderConfig) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
        }
    }

    /// Append a fallback config.
    pub fn with_fallback(mut self, fallback: ProviderConfig) -> Self {
        self.fallbacks.push(fallback);
        self
    }
}

/// An error produced while resolving a selector deployment.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SelectorError {
    /// The active chain link is not configured yet (missing API key).
    #[error("Provider '{0}' is not configured yet (missing API key)")]
    NotConfigured(String),
    /// No fallback remains in the chain.
    #[error("No provider fallback available")]
    NoFallback,
    /// The chain is empty.
    #[error("Selector has an empty provider chain")]
    EmptyChain,
}

/// One provider's failure while listing models.
#[derive(Debug, Clone)]
pub struct ProviderListError {
    /// The provider id.
    pub provider: String,
    /// The chain link's configured model (operator anchor).
    pub model_hint: String,
    /// The classified failure kind token.
    pub kind: String,
    /// Credential-masked detail safe to surface.
    pub detail: String,
}

/// Aggregated model-listing outcome across the whole selector chain.
#[derive(Debug, Clone, Default)]
pub struct ModelListResult {
    /// The models found across the chain.
    pub models: Vec<ModelInfo>,
    /// Per-provider failures, additive.
    pub errors: Vec<ProviderListError>,
}

/// A fallback chain entry produced by router override.
#[derive(Debug, Clone)]
pub struct FallbackEntry {
    /// Provider id; `None` means "reuse the current provider's credentials".
    pub provider: Option<String>,
    /// The fallback model id.
    pub model: String,
}

/// Resolves a provider from primary config with fallback chain support.
#[derive(Debug, Clone)]
pub struct ModelSelector {
    /// The immutable config (primary + fallbacks).
    config: SelectorConfig,
    /// The active chain (reordered by overrides).
    chain: Vec<ProviderConfig>,
    /// Index of the active chain link.
    index: usize,
    /// Whether provider-private state replay is disabled for the whole chain.
    provider_state_replay_disabled: bool,
    /// Shared health ledger consulted by `select_model`/`eligible`.
    health: Option<Arc<ProviderHealthLedger>>,
}

impl ModelSelector {
    /// Create a new selector from a config.
    pub fn new(config: SelectorConfig) -> Self {
        let mut chain = Vec::with_capacity(1 + config.fallbacks.len());
        chain.push(config.primary.clone());
        chain.extend(config.fallbacks.iter().cloned());
        Self {
            config,
            chain,
            index: 0,
            provider_state_replay_disabled: false,
            health: None,
        }
    }

    /// Attach a shared health ledger so selection is bench-aware.
    pub fn with_health_ledger(mut self, health: Arc<ProviderHealthLedger>) -> Self {
        self.health = Some(health);
        self
    }

    /// True when the active chain link can serve requests.
    ///
    /// A provider id must be set, and an API key must be present unless the
    /// provider is one that never requires one (Ollama, LM Studio, ...).
    pub fn is_configured(&self) -> bool {
        let cfg = self.chain.get(self.index);
        let Some(cfg) = cfg else {
            return false;
        };
        if cfg.provider.trim().is_empty() {
            return false;
        }
        // Providers that never require a key: ollama and local endpoints.
        let requires_key = !matches!(
            cfg.provider.trim().to_lowercase().as_str(),
            "ollama" | "lmstudio" | "local"
        ) && !cfg.base_url.contains("localhost")
            && !cfg.base_url.contains("127.0.0.1");
        !requires_key || !cfg.api_key.is_empty()
    }

    /// The operator-facing provider id of the active chain link.
    pub fn active_provider_id(&self) -> &str {
        self.chain
            .get(self.index)
            .map(|c| c.provider.as_str())
            .unwrap_or("")
    }

    /// The currently-active provider config.
    pub fn active_config(&self) -> &ProviderConfig {
        self.chain
            .get(self.index)
            .expect("selector chain is never empty")
    }

    /// True if there is at least one more fallback available.
    pub fn has_fallback(&self) -> bool {
        self.index < self.chain.len().saturating_sub(1)
    }

    /// Copy of the active chain link plus untried fallbacks, in order.
    ///
    /// Read-only view for callers that need the candidate deployment set —
    /// e.g. the health ledger's never-strand eligibility check.
    pub fn remaining_chain(&self) -> Vec<ProviderConfig> {
        self.chain[self.index..].to_vec()
    }

    /// The full candidate deployment list `(provider, model)`.
    pub fn candidate_deployments(&self) -> Vec<(String, String)> {
        self.chain
            .iter()
            .map(|c| (c.provider.clone(), c.model.clone()))
            .collect()
    }

    /// Advance to the next fallback and return its config.
    pub fn next_fallback(&mut self) -> Result<&ProviderConfig, SelectorError> {
        if !self.has_fallback() {
            return Err(SelectorError::NoFallback);
        }
        self.index += 1;
        Ok(self.active_config())
    }

    /// Reset the selector to the primary provider.
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Replace the active chain head with a full per-turn provider config.
    ///
    /// The previous primary is kept as the first fallback so pre-content
    /// failover still has somewhere to go.
    pub fn override_provider_config(&mut self, cfg: ProviderConfig) {
        let original_primary = self.chain[0].clone();
        let mut deduped: Vec<ProviderConfig> = vec![cfg.clone()];
        for candidate in std::iter::once(original_primary).chain(self.chain[1..].iter().cloned()) {
            if !same_identity(&candidate, &cfg)
                && !deduped.iter().any(|c| same_identity(c, &candidate))
            {
                deduped.push(candidate);
            }
        }
        self.chain = deduped;
        self.index = 0;
    }

    /// Update the model on the primary provider config (for runtime switching).
    pub fn override_model(&mut self, model: &str) {
        if model.is_empty() || model == self.chain[0].model {
            return;
        }
        let original_primary = self.chain[0].clone();
        let mut overridden = original_primary.clone();
        overridden.model = model.to_string();
        let mut deduped: Vec<ProviderConfig> = vec![overridden.clone()];
        for candidate in std::iter::once(original_primary).chain(self.chain[1..].iter().cloned()) {
            if !same_identity(&candidate, &overridden) {
                deduped.push(candidate);
            }
        }
        self.chain = deduped;
        self.index = 0;
    }

    /// Prefer router-provided fallback models, reusing current credentials.
    pub fn override_model_with_fallback_chain(&mut self, model: &str, fallbacks: &[FallbackEntry]) {
        self.override_model(model);
        if fallbacks.is_empty() {
            return;
        }
        let current = self.chain[0].clone();
        let existing_tail = self.chain[1..].to_vec();
        let mut router_fallbacks: Vec<ProviderConfig> = Vec::new();
        for entry in fallbacks {
            if entry.model.trim().is_empty() {
                continue;
            }
            let provider = entry
                .provider
                .clone()
                .unwrap_or_else(|| current.provider.clone());
            let same_provider = provider.to_lowercase() == current.provider.to_lowercase();
            // Cross-provider entries need matching credentials; reuse only
            // same-provider fallbacks to avoid guessing secrets.
            if !same_provider {
                continue;
            }
            let mut cfg = current.clone();
            cfg.model = entry.model.trim().to_string();
            if !router_fallbacks.iter().any(|c| same_identity(c, &cfg)) {
                router_fallbacks.push(cfg);
            }
        }
        let mut deduped: Vec<ProviderConfig> = vec![current.clone()];
        for cfg in router_fallbacks.into_iter().chain(existing_tail) {
            if !deduped.iter().any(|c| same_identity(c, &cfg)) {
                deduped.push(cfg);
            }
        }
        self.chain = deduped;
        self.index = 0;
    }

    /// Replace the primary provider config for future resolves.
    pub fn sync_primary(&mut self, cfg: ProviderConfig) {
        self.config.primary = cfg.clone();
        if let Some(head) = self.chain.first_mut() {
            *head = cfg;
        }
        self.reset();
    }

    /// Disable provider-private history replay across the whole chain.
    pub fn disable_provider_state_replay(&mut self) {
        self.provider_state_replay_disabled = true;
        for cfg in &mut self.chain {
            cfg.replay_provider_state = false;
        }
        self.config.primary.replay_provider_state = false;
        for cfg in &mut self.config.fallbacks {
            cfg.replay_provider_state = false;
        }
    }

    /// Whether the active deployment is eligible per the health ledger.
    ///
    /// With no ledger attached, everything is eligible.
    pub fn is_eligible(&self) -> bool {
        let Some(ledger) = &self.health else {
            return true;
        };
        let active = self.active_config();
        ledger.eligible(
            &active.provider,
            &active.model,
            &self.candidate_deployments(),
            None,
        )
    }

    /// Select the first model in the chain whose deployment is eligible.
    ///
    /// Returns the model id of the first eligible chain link, or `None` when
    /// the whole chain is benched (the caller then surfaces the primary).
    pub fn select_model(&self) -> Option<String> {
        let Some(ledger) = &self.health else {
            return self.active_config().model.clone().into();
        };
        let candidates = self.candidate_deployments();
        for cfg in &self.chain[self.index..] {
            if ledger.eligible(&cfg.provider, &cfg.model, &candidates, None) {
                return Some(cfg.model.clone());
            }
        }
        // Never-strand: fall back to the active link.
        Some(self.active_config().model.clone())
    }

    /// Return an independent copy for concurrent use.
    ///
    /// Deep-copies the chain so the clone starts at index 0 with its own state
    /// and is unaffected by later mutations of the original.
    pub fn clone_independent(&self) -> Self {
        let config = SelectorConfig {
            primary: self.config.primary.clone(),
            fallbacks: self.config.fallbacks.clone(),
        };
        Self {
            config,
            chain: self.chain.clone(),
            index: 0,
            provider_state_replay_disabled: self.provider_state_replay_disabled,
            health: self.health.clone(),
        }
    }

    /// List models across the chain from a static registry.
    ///
    /// When a registry is supplied, models are looked up per chain link; links
    /// without registry entries are skipped. This is the network-free model
    /// listing path (the gateway may substitute a live catalog).
    pub fn list_models(&self, registry: &ModelRegistry) -> ModelListResult {
        let mut result = ModelListResult::default();
        for cfg in &self.chain {
            let found = registry
                .get(&cfg.provider, &cfg.model)
                .or_else(|| registry.find_model(&cfg.model));
            match found {
                Some(info) => result.models.push(info.clone()),
                None => result.errors.push(ProviderListError {
                    provider: cfg.provider.clone(),
                    model_hint: cfg.model.clone(),
                    kind: "model_not_found".to_string(),
                    detail: format!("No catalog entry for '{}' / '{}'", cfg.provider, cfg.model),
                }),
            }
        }
        result
    }
}

/// Compare two configs for dedup purposes (provider + model identity).
fn same_identity(a: &ProviderConfig, b: &ProviderConfig) -> bool {
    a.provider == b.provider && a.model == b.model
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selector() -> ModelSelector {
        ModelSelector::new(
            SelectorConfig::new(ProviderConfig::new("anthropic", "claude-sonnet"))
                .with_fallback(ProviderConfig::new("ollama", "llama3")),
        )
    }

    #[test]
    fn test_primary_is_configured_with_key() {
        let s = ModelSelector::new(SelectorConfig::new(
            ProviderConfig::new("openai", "gpt-4o").with_api_key("sk-test"),
        ));
        assert!(s.is_configured());
        assert_eq!(s.active_provider_id(), "openai");
    }

    #[test]
    fn test_primary_unconfigured_without_key() {
        let s = ModelSelector::new(SelectorConfig::new(ProviderConfig::new("openai", "gpt-4o")));
        assert!(!s.is_configured());
    }

    #[test]
    fn test_ollama_never_requires_key() {
        let s = ModelSelector::new(SelectorConfig::new(ProviderConfig::new("ollama", "llama3")));
        assert!(s.is_configured());
    }

    #[test]
    fn test_fallback_chain_walk() {
        let mut s = selector();
        assert_eq!(s.active_config().provider, "anthropic");
        assert!(s.has_fallback());
        s.next_fallback().unwrap();
        assert_eq!(s.active_config().provider, "ollama");
        assert!(!s.has_fallback());
        assert!(s.next_fallback().is_err());
    }

    #[test]
    fn test_reset_to_primary() {
        let mut s = selector();
        s.next_fallback().unwrap();
        assert_eq!(s.active_config().provider, "ollama");
        s.reset();
        assert_eq!(s.active_config().provider, "anthropic");
    }

    #[test]
    fn test_override_model() {
        let mut s = selector();
        s.override_model("claude-opus");
        assert_eq!(s.active_config().model, "claude-opus");
        assert_eq!(s.active_config().provider, "anthropic");
    }

    #[test]
    fn test_clone_independent() {
        let mut s = selector();
        let clone = s.clone_independent();
        s.next_fallback().unwrap();
        assert_eq!(clone.active_config().provider, "anthropic");
        assert_eq!(s.active_config().provider, "ollama");
    }

    #[test]
    fn test_list_models_from_registry() {
        let mut registry = ModelRegistry::default();
        let info = opensquilla_core::model::ModelInfo {
            id: "gpt-4o".into(),
            name: "GPT-4o".into(),
            provider: "openai".into(),
            capabilities: Default::default(),
            context_window: 128_000,
            max_output_tokens: 4096,
            pricing: Default::default(),
            metadata: Default::default(),
        };
        registry.register(info);
        let s = ModelSelector::new(SelectorConfig::new(ProviderConfig::new("openai", "gpt-4o")));
        let result = s.list_models(&registry);
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "gpt-4o");
    }
}
