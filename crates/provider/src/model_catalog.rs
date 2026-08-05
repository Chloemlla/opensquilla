//! Hierarchical model metadata cache.
//!
//! [`ModelCatalog`] stores per-provider, per-model capability metadata in a
//! three-level hierarchy: provider -> model -> capabilities. Entries are
//! populated lazily and refreshed on a TTL basis so that lookups after the
//! first hit are cheap in-memory map reads.
//!
//! The catalog is thread-safe via [`DashMap`](dashmap::DashMap) and an inner
//! [`RwLock`](std::sync::RwLock) per provider entry.

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Default TTL for cached model metadata.
pub const DEFAULT_TTL: Duration = Duration::from_secs(600);

/// Capabilities and metadata for a single model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelCapabilities {
    /// The model id (e.g. "gpt-4o").
    pub model: String,
    /// Human-readable label.
    pub label: Option<String>,
    /// Context window size in tokens.
    pub context_window: Option<u32>,
    /// Maximum output tokens.
    pub max_output_tokens: Option<u32>,
    /// Whether the model supports tool/function calling.
    pub supports_tools: bool,
    /// Whether the model supports vision/image input.
    pub supports_vision: bool,
    /// Whether the model supports audio input.
    pub supports_audio: bool,
    /// Whether the model is a reasoning model (emits thinking).
    pub supports_reasoning: bool,
    /// Whether streaming is supported.
    pub supports_streaming: bool,
    /// Input price per 1M tokens in USD, if known.
    pub input_price_per_million: Option<f64>,
    /// Output price per 1M tokens in USD, if known.
    pub output_price_per_million: Option<f64>,
    /// Free-form tags (e.g. "chat", "code", "embedding").
    pub tags: Vec<String>,
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self {
            model: String::new(),
            label: None,
            context_window: None,
            max_output_tokens: None,
            supports_tools: false,
            supports_vision: false,
            supports_audio: false,
            supports_reasoning: false,
            supports_streaming: true,
            input_price_per_million: None,
            output_price_per_million: None,
            tags: Vec::new(),
        }
    }
}

/// A provider's worth of cached models, with a last-refresh timestamp.
#[derive(Debug)]
struct ProviderCache {
    models: RwLock<HashMap<String, ModelCapabilities>>,
    refreshed_at: RwLock<Instant>,
}

impl ProviderCache {
    fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            refreshed_at: RwLock::new(Instant::now()),
        }
    }
}

/// A hierarchical, TTL-refreshed model metadata cache.
#[derive(Clone)]
pub struct ModelCatalog {
    providers: Arc<DashMap<String, Arc<ProviderCache>>>,
    ttl: Duration,
}

impl ModelCatalog {
    /// Create a new empty catalog with the default TTL.
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_TTL)
    }

    /// Create a new empty catalog with a custom TTL.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            providers: Arc::new(DashMap::new()),
            ttl,
        }
    }

    fn entry_for(&self, provider: &str) -> Arc<ProviderCache> {
        self.providers
            .entry(provider.to_string())
            .or_insert_with(|| Arc::new(ProviderCache::new()))
            .clone()
    }

    /// Insert or update a single model's metadata for a provider.
    pub fn upsert(&self, provider: &str, caps: ModelCapabilities) {
        let entry = self.entry_for(provider);
        let mut models = entry.models.write().unwrap();
        models.insert(caps.model.clone(), caps);
        debug!(
            target = "provider",
            provider = provider,
            "Upserted model metadata"
        );
    }

    /// Insert or update multiple models for a provider at once, refreshing the
    /// TTL timestamp.
    pub fn upsert_many(&self, provider: &str, caps: Vec<ModelCapabilities>) {
        let entry = self.entry_for(provider);
        let mut models = entry.models.write().unwrap();
        for c in caps {
            models.insert(c.model.clone(), c);
        }
        *entry.refreshed_at.write().unwrap() = Instant::now();
    }

    /// Look up a model's capabilities.
    pub fn get(&self, provider: &str, model: &str) -> Option<ModelCapabilities> {
        let entry = self.entry_for(provider);
        let models = entry.models.read().unwrap();
        models.get(model).cloned()
    }

    /// List all models for a provider.
    pub fn list(&self, provider: &str) -> Vec<ModelCapabilities> {
        let entry = self.entry_for(provider);
        let models = entry.models.read().unwrap();
        let mut v: Vec<ModelCapabilities> = models.values().cloned().collect();
        v.sort_by(|a, b| a.model.cmp(&b.model));
        v
    }

    /// List all provider ids known to the catalog.
    pub fn providers(&self) -> Vec<String> {
        let mut v: Vec<String> = self.providers.iter().map(|r| r.key().clone()).collect();
        v.sort();
        v
    }

    /// Returns `true` if the cached entry for `provider` is stale (older than TTL).
    pub fn is_stale(&self, provider: &str) -> bool {
        if let Some(entry) = self.providers.get(provider) {
            let refreshed = *entry.refreshed_at.read().unwrap();
            Instant::now().duration_since(refreshed) > self.ttl
        } else {
            true
        }
    }

    /// Returns the instant the provider's cache was last refreshed, if present.
    pub fn last_refreshed(&self, provider: &str) -> Option<Instant> {
        self.providers
            .get(provider)
            .map(|e| *e.refreshed_at.read().unwrap())
    }

    /// Force a refresh timestamp for a provider (called after a live fetch).
    pub fn mark_refreshed(&self, provider: &str) {
        let entry = self.entry_for(provider);
        *entry.refreshed_at.write().unwrap() = Instant::now();
    }

    /// Remove all cached entries for a provider.
    pub fn invalidate(&self, provider: &str) {
        self.providers.remove(provider);
    }

    /// Clear the entire catalog.
    pub fn clear(&self) {
        self.providers.clear();
    }

    /// Filter models for a provider by a capability predicate.
    pub fn filter(
        &self,
        provider: &str,
        predicate: impl Fn(&ModelCapabilities) -> bool,
    ) -> Vec<ModelCapabilities> {
        let entry = self.entry_for(provider);
        let models = entry.models.read().unwrap();
        let mut v: Vec<ModelCapabilities> =
            models.values().filter(|c| predicate(c)).cloned().collect();
        v.sort_by(|a, b| a.model.cmp(&b.model));
        v
    }

    /// Find the cheapest model for a provider that supports tools.
    pub fn cheapest_with_tools(&self, provider: &str) -> Option<ModelCapabilities> {
        self.filter(provider, |c| c.supports_tools)
            .into_iter()
            .min_by(|a, b| {
                let pa = a.input_price_per_million.unwrap_or(f64::MAX);
                let pb = b.input_price_per_million.unwrap_or(f64::MAX);
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

impl Default for ModelCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Seed the catalog with a baseline set of well-known models for a provider.
///
/// This is used at startup so the catalog is useful even before a live fetch
/// completes. Returns the number of entries inserted.
pub fn seed_static(catalog: &ModelCatalog, provider: &str) -> usize {
    let models = static_models_for(provider);
    if models.is_empty() {
        return 0;
    }
    let n = models.len();
    catalog.upsert_many(provider, models);
    n
}

/// Return a static baseline list of models for a provider id.
pub fn static_models_for(provider: &str) -> Vec<ModelCapabilities> {
    match provider {
        "openai" | "openai_compat" => vec![
            cap(
                "gpt-4o",
                Some(128_000),
                Some(16_384),
                true,
                true,
                false,
                false,
                2.5,
                10.0,
            ),
            cap(
                "gpt-4o-mini",
                Some(128_000),
                Some(16_384),
                true,
                true,
                false,
                false,
                0.15,
                0.6,
            ),
            cap(
                "gpt-4-turbo",
                Some(128_000),
                Some(4_096),
                true,
                true,
                false,
                false,
                10.0,
                30.0,
            ),
            cap(
                "o1",
                Some(200_000),
                Some(100_000),
                true,
                true,
                false,
                true,
                15.0,
                60.0,
            ),
            cap(
                "o3-mini",
                Some(200_000),
                Some(100_000),
                true,
                false,
                false,
                true,
                1.1,
                4.4,
            ),
        ],
        "anthropic" => vec![
            cap(
                "claude-3-5-sonnet-20241022",
                Some(200_000),
                Some(8_192),
                true,
                true,
                false,
                false,
                3.0,
                15.0,
            ),
            cap(
                "claude-3-5-haiku-20241022",
                Some(200_000),
                Some(8_192),
                true,
                true,
                false,
                false,
                0.8,
                4.0,
            ),
            cap(
                "claude-3-opus-20240229",
                Some(200_000),
                Some(4_096),
                true,
                true,
                false,
                false,
                15.0,
                75.0,
            ),
        ],
        "deepseek" => vec![
            cap(
                "deepseek-chat",
                Some(128_000),
                Some(8_192),
                true,
                false,
                false,
                false,
                0.14,
                0.28,
            ),
            cap(
                "deepseek-reasoner",
                Some(128_000),
                Some(32_768),
                true,
                false,
                false,
                true,
                0.55,
                2.19,
            ),
        ],
        "gemini" => vec![
            cap(
                "gemini-2.0-flash",
                Some(1_000_000),
                Some(8_192),
                true,
                true,
                true,
                false,
                0.1,
                0.4,
            ),
            cap(
                "gemini-1.5-pro",
                Some(2_000_000),
                Some(8_192),
                true,
                true,
                true,
                false,
                1.25,
                5.0,
            ),
            cap(
                "gemini-1.5-flash",
                Some(1_000_000),
                Some(8_192),
                true,
                true,
                true,
                false,
                0.075,
                0.3,
            ),
        ],
        "ollama" => vec![
            cap(
                "llama3.1",
                Some(128_000),
                Some(4_096),
                true,
                false,
                false,
                false,
                0.0,
                0.0,
            ),
            cap(
                "qwen2.5",
                Some(131_072),
                Some(8_192),
                true,
                false,
                false,
                false,
                0.0,
                0.0,
            ),
            cap(
                "mistral",
                Some(32_000),
                Some(4_096),
                true,
                false,
                false,
                false,
                0.0,
                0.0,
            ),
        ],
        "groq" => vec![
            cap(
                "llama-3.3-70b-versatile",
                Some(128_000),
                Some(32_768),
                true,
                false,
                false,
                false,
                0.59,
                0.79,
            ),
            cap(
                "llama-3.1-8b-instant",
                Some(128_000),
                Some(8_192),
                true,
                false,
                false,
                false,
                0.05,
                0.08,
            ),
        ],
        "mistral" => vec![
            cap(
                "mistral-large-latest",
                Some(128_000),
                Some(8_192),
                true,
                false,
                false,
                false,
                2.0,
                6.0,
            ),
            cap(
                "mistral-small-latest",
                Some(32_000),
                Some(8_192),
                true,
                false,
                false,
                false,
                0.2,
                0.6,
            ),
        ],
        "moonshot" => vec![
            cap(
                "moonshot-v1-128k",
                Some(128_000),
                Some(8_192),
                true,
                false,
                false,
                false,
                0.83,
                2.49,
            ),
            cap(
                "kimi-k1.5",
                Some(128_000),
                Some(8_192),
                true,
                false,
                false,
                true,
                0.55,
                2.19,
            ),
        ],
        "qwen" | "dashscope" => vec![
            cap(
                "qwen-max",
                Some(131_072),
                Some(8_192),
                true,
                true,
                false,
                false,
                1.6,
                6.4,
            ),
            cap(
                "qwen-plus",
                Some(131_072),
                Some(8_192),
                true,
                true,
                false,
                false,
                0.4,
                1.2,
            ),
            cap(
                "qwen-turbo",
                Some(131_072),
                Some(8_192),
                true,
                true,
                false,
                false,
                0.05,
                0.2,
            ),
        ],
        _ => Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn cap(
    model: &str,
    ctx: Option<u32>,
    out: Option<u32>,
    tools: bool,
    vision: bool,
    audio: bool,
    reasoning: bool,
    in_price: f64,
    out_price: f64,
) -> ModelCapabilities {
    ModelCapabilities {
        model: model.to_string(),
        label: Some(model.to_string()),
        context_window: ctx,
        max_output_tokens: out,
        supports_tools: tools,
        supports_vision: vision,
        supports_audio: audio,
        supports_reasoning: reasoning,
        supports_streaming: true,
        input_price_per_million: Some(in_price),
        output_price_per_million: Some(out_price),
        tags: Vec::new(),
    }
}

/// Merge live-fetched capabilities into the catalog, preferring live data
/// where it provides fields the static entry lacks.
pub fn merge_live(catalog: &ModelCatalog, provider: &str, live: Vec<ModelCapabilities>) {
    let existing: HashMap<String, ModelCapabilities> = catalog
        .list(provider)
        .into_iter()
        .map(|c| (c.model.clone(), c))
        .collect();
    let mut merged = Vec::with_capacity(live.len());
    for l in live {
        if let Some(s) = existing.get(&l.model) {
            // Prefer live non-None fields, fall back to static.
            let m = ModelCapabilities {
                model: l.model.clone(),
                label: l.label.or(s.label.clone()),
                context_window: l.context_window.or(s.context_window),
                max_output_tokens: l.max_output_tokens.or(s.max_output_tokens),
                supports_tools: l.supports_tools || s.supports_tools,
                supports_vision: l.supports_vision || s.supports_vision,
                supports_audio: l.supports_audio || s.supports_audio,
                supports_reasoning: l.supports_reasoning || s.supports_reasoning,
                supports_streaming: l.supports_streaming || s.supports_streaming,
                input_price_per_million: l.input_price_per_million.or(s.input_price_per_million),
                output_price_per_million: l.output_price_per_million.or(s.output_price_per_million),
                tags: if l.tags.is_empty() {
                    s.tags.clone()
                } else {
                    l.tags
                },
            };
            merged.push(m);
        } else {
            merged.push(l);
        }
    }
    if merged.is_empty() {
        warn!(
            target = "provider",
            provider = provider,
            "Live merge produced no models"
        );
    }
    catalog.upsert_many(provider, merged);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_and_get() {
        let cat = ModelCatalog::new();
        cat.upsert(
            "openai",
            cap(
                "gpt-4o",
                Some(128_000),
                Some(16_384),
                true,
                true,
                false,
                false,
                2.5,
                10.0,
            ),
        );
        let m = cat.get("openai", "gpt-4o").unwrap();
        assert_eq!(m.context_window, Some(128_000));
        assert!(m.supports_tools);
    }

    #[test]
    fn test_list_and_filter() {
        let cat = ModelCatalog::new();
        cat.upsert_many("openai", static_models_for("openai"));
        let all = cat.list("openai");
        assert!(all.len() >= 3);
        let tools = cat.filter("openai", |c| c.supports_tools);
        assert!(tools.iter().all(|c| c.supports_tools));
    }

    #[test]
    fn test_cheapest_with_tools() {
        let cat = ModelCatalog::new();
        cat.upsert_many("openai", static_models_for("openai"));
        let cheapest = cat.cheapest_with_tools("openai").unwrap();
        // gpt-4o-mini at 0.15 is the cheapest.
        assert_eq!(cheapest.model, "gpt-4o-mini");
    }

    #[test]
    fn test_ttl_stale() {
        let cat = ModelCatalog::with_ttl(Duration::from_millis(10));
        cat.upsert(
            "p",
            cap("m", None, None, false, false, false, false, 0.0, 0.0),
        );
        assert!(!cat.is_stale("p"));
        std::thread::sleep(Duration::from_millis(20));
        assert!(cat.is_stale("p"));
    }

    #[test]
    fn test_seed_static() {
        let cat = ModelCatalog::new();
        let n = seed_static(&cat, "anthropic");
        assert!(n >= 3);
        assert!(cat.get("anthropic", "claude-3-5-sonnet-20241022").is_some());
        assert_eq!(seed_static(&cat, "nonexistent"), 0);
    }

    #[test]
    fn test_merge_live_prefers_live() {
        let cat = ModelCatalog::new();
        seed_static(&cat, "openai");
        // Live says gpt-4o has a different (larger) context window.
        let live = vec![ModelCapabilities {
            model: "gpt-4o".into(),
            context_window: Some(200_000),
            ..Default::default()
        }];
        merge_live(&cat, "openai", live);
        let m = cat.get("openai", "gpt-4o").unwrap();
        assert_eq!(m.context_window, Some(200_000));
        // Static-only field preserved.
        assert!(m.supports_tools);
    }
}
