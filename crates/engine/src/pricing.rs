//! Model pricing.
//!
//! The `PricingCache` fetches live model pricing (via OpenRouter's public
//! pricing API, or a configured custom endpoint) with a TTL of 1 hour, caches
//! per-model input/output prices, and computes request cost.
//!
//! When the network fetch fails or the cache is empty, a built-in fallback
//! price table is used so cost accounting never hard-fails.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// The default TTL for cached pricing data (1 hour, as the Python backend).
pub const DEFAULT_TTL: Duration = Duration::from_secs(3600);

/// The default OpenRouter pricing endpoint.
pub const OPENROUTER_PRICING_URL: &str = "https://openrouter.ai/api/v1/models";

/// The default price (USD per 1M tokens) used when no data is available.
pub const FALLBACK_INPUT_PRICE_PER_1M: f64 = 1.0;
pub const FALLBACK_OUTPUT_PRICE_PER_1M: f64 = 3.0;

/// Price information for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrice {
    /// The model identifier.
    pub model: String,
    /// Price in USD per 1,000,000 input tokens.
    pub input_per_1m: f64,
    /// Price in USD per 1,000,000 output tokens.
    pub output_per_1m: f64,
    /// The context window size in tokens, if known.
    pub context_window: Option<u64>,
}

impl ModelPrice {
    /// The input price per 1k tokens.
    pub fn input_per_1k(&self) -> f64 {
        self.input_per_1m / 1000.0
    }

    /// The output price per 1k tokens.
    pub fn output_per_1k(&self) -> f64 {
        self.output_per_1m / 1000.0
    }
}

/// A resolved pricing record for a model, including the provider and currency.
///
/// Prices are expressed in USD per 1,000,000 tokens (`input_price` /
/// `output_price`), matching the Python backend's `PriceEntry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPricing {
    /// The model identifier.
    pub model: String,
    /// The provider the price was resolved for, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Price in USD per 1,000,000 input tokens.
    pub input_price: f64,
    /// Price in USD per 1,000,000 output tokens.
    pub output_price: f64,
    /// The currency of the prices (always `"usd"` today).
    pub currency: String,
    /// The model's context window in tokens, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_window: Option<u64>,
}

impl ModelPricing {
    /// Build a `ModelPricing` from a `ModelPrice`.
    pub fn from_model_price(price: &ModelPrice, provider: Option<String>) -> Self {
        Self {
            model: price.model.clone(),
            provider,
            input_price: price.input_per_1m,
            output_price: price.output_per_1m,
            currency: "usd".to_string(),
            token_window: price.context_window,
        }
    }
}

/// A single cached pricing entry with its fetch timestamp.
#[derive(Debug, Clone)]
struct CachedPrice {
    price: ModelPrice,
    fetched_at: SystemTime,
}

/// The response shape of OpenRouter's `/api/v1/models` endpoint.
#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    #[serde(default)]
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    pricing: OpenRouterPricing,
    #[serde(default)]
    context_length: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
}

/// The result of a pricing cache operation.
#[derive(Debug)]
pub struct PricingResult {
    /// The price for the requested model.
    pub price: ModelPrice,
    /// Whether the price came from a live fetch (`true`) or the fallback
    /// table (`false`).
    pub from_cache: bool,
}

/// A cache of model prices with TTL-based invalidation.
///
/// The cache is shared behind a `parking_lot::RwLock` and a `reqwest::Client`
/// for live fetches. Fetching is debounced: within the TTL the cached price is
/// returned without any network call.
#[derive(Debug, Clone)]
pub struct PricingCache {
    /// The shared pricing data.
    inner: std::sync::Arc<PriceCacheInner>,
}

#[derive(Debug)]
struct PriceCacheInner {
    /// The cached prices keyed by model name.
    prices: parking_lot::RwLock<HashMap<String, CachedPrice>>,
    /// The HTTP client used for live fetches.
    client: reqwest::Client,
    /// The pricing endpoint to fetch from.
    endpoint: String,
    /// The TTL for cached entries.
    ttl: Duration,
    /// Unix timestamp (seconds) of the last successful full refresh, or 0 if
    /// never refreshed.
    last_refresh: AtomicI64,
    /// Fallback prices keyed by model name.
    fallback: HashMap<String, ModelPrice>,
}

impl Default for PricingCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PricingCache {
    /// Create a new pricing cache with a default HTTP client.
    pub fn new() -> Self {
        Self::with_client(reqwest::Client::new())
    }

    /// Create a new pricing cache with the given HTTP client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self::with_endpoint(client, OPENROUTER_PRICING_URL.to_string())
    }

    /// Create a pricing cache pointed at a custom endpoint.
    pub fn with_endpoint(client: reqwest::Client, endpoint: String) -> Self {
        Self {
            inner: Arc::new(PriceCacheInner {
                prices: parking_lot::RwLock::new(HashMap::new()),
                client,
                endpoint,
                ttl: DEFAULT_TTL,
                last_refresh: AtomicI64::new(0),
                fallback: builtin_fallback_prices(),
            }),
        }
    }

    /// Set a custom TTL for cached entries.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.ttl = ttl;
        }
        self
    }

    /// Set the fallback price table used when fetches fail.
    pub fn with_fallback(mut self, prices: Vec<ModelPrice>) -> Self {
        if let Some(inner) = Arc::get_mut(&mut self.inner) {
            inner.fallback = prices.into_iter().map(|p| (p.model.clone(), p)).collect();
        }
        self
    }

    /// Register an explicit price for a model (no network needed).
    ///
    /// Seeded entries are considered fresh, so a subsequent [`PricingCache::get_price`]
    /// will not trigger a network refresh.
    pub fn seed(&self, price: ModelPrice) {
        let mut prices = self.inner.prices.write();
        prices.insert(
            price.model.clone(),
            CachedPrice {
                price: price.clone(),
                fetched_at: SystemTime::now(),
            },
        );
        drop(prices);
        self.inner.last_refresh.store(unix_now(), Ordering::Relaxed);
    }

    /// Get the price for a model, fetching from the network if the cache is
    /// stale or empty.
    pub async fn price_for(&self, model: &str) -> PricingResult {
        if let Some(cached) = self.cached_price(model) {
            return PricingResult {
                price: cached,
                from_cache: true,
            };
        }

        // Attempt a live fetch. On any error, fall back.
        match self.fetch_all().await {
            Ok(_) => {
                if let Some(cached) = self.cached_price(model) {
                    return PricingResult {
                        price: cached,
                        from_cache: true,
                    };
                }
            }
            Err(e) => {
                warn!(model = %model, error = %e, "Failed to fetch live pricing");
            }
        }

        // Fallback table, static prefix table, or default.
        let price = self
            .fallback_price(model)
            .or_else(|| lookup_static_price(model))
            .unwrap_or_else(|| default_price(model));
        PricingResult {
            price,
            from_cache: false,
        }
    }

    /// Compute the cost of a request in USD.
    pub fn cost_for(&self, model: &str, input_tokens: u64, output_tokens: u64) -> f64 {
        let price = match self.inner.prices.read().get(model) {
            Some(cached) => cached.price.clone(),
            None => self
                .fallback_price(model)
                .or_else(|| lookup_static_price(model))
                .unwrap_or_else(|| default_price(model)),
        };
        cost(
            price.input_per_1m,
            price.output_per_1m,
            input_tokens,
            output_tokens,
        )
    }

    /// Check whether a cached price exists and is within its TTL.
    pub fn is_fresh(&self, model: &str) -> bool {
        self.cached_price(model).is_some()
    }

    /// Invalidate a single model's cached price.
    pub fn invalidate(&self, model: &str) {
        self.inner.prices.write().remove(model);
    }

    /// Invalidate the entire cache.
    pub fn clear(&self) {
        self.inner.prices.write().clear();
    }

    /// The number of models currently cached.
    pub fn cached_len(&self) -> usize {
        self.inner.prices.read().len()
    }

    /// The configured TTL.
    pub fn ttl(&self) -> Duration {
        self.inner.ttl
    }

    /// The configured TTL in whole seconds.
    pub fn ttl_seconds(&self) -> u64 {
        self.inner.ttl.as_secs()
    }

    /// Whether the cached pricing data is stale (older than the TTL) or has
    /// never been fetched.
    pub fn is_stale(&self) -> bool {
        let last = self.inner.last_refresh.load(Ordering::Relaxed);
        if last <= 0 {
            return true;
        }
        let now = unix_now();
        now.saturating_sub(last) >= self.inner.ttl.as_secs() as i64
    }

    /// Refresh the pricing cache from the configured endpoint.
    ///
    /// On success the per-model entries and the `last_refresh` timestamp are
    /// updated. On failure the existing cache is left untouched and the error
    /// is returned, so callers can fall back to cached or static prices.
    pub async fn refresh(&self) -> Result<(), PricingError> {
        self.fetch_all().await
    }

    /// Resolve the price for a model, mirroring the layered resolution of the
    /// Python backend:
    ///
    /// 1. Local-free providers (`ollama`, `lm_studio`, `ovms`, `vllm`,
    ///    `local`) short-circuit to a zero price.
    /// 2. If the cache is stale, a live refresh is attempted (fail-open).
    /// 3. Prices are resolved from the cache, the static table, and finally a
    ///    sensible default, in that order.
    ///
    /// `provider` is the configured provider id (e.g. `"openrouter"`,
    /// `"ollama"`). The returned [`ModelPricing`] always carries a price so
    /// cost accounting never hard-fails offline.
    pub async fn get_price(
        &self,
        model: &str,
        provider: &str,
    ) -> Result<ModelPricing, PricingError> {
        let model = model.trim();
        let prov = provider.trim().to_lowercase();

        if is_local_free_provider(&prov) {
            return Ok(ModelPricing {
                model: model.to_string(),
                provider: Some(provider.trim().to_string()),
                input_price: 0.0,
                output_price: 0.0,
                currency: "usd".to_string(),
                token_window: None,
            });
        }

        if self.is_stale() {
            if let Err(e) = self.refresh().await {
                warn!(
                    model = %model,
                    provider = %prov,
                    error = %e,
                    "Pricing refresh failed; using cached/static fallback"
                );
            }
        }

        let qualified = format!("{}/{}", prov, model.to_lowercase());
        let price = self
            .cached_model(&qualified)
            .or_else(|| self.cached_model(model))
            .or_else(|| self.fallback_price(model))
            .or_else(|| lookup_static_price(model))
            .unwrap_or_else(|| default_price(model));

        Ok(ModelPricing::from_model_price(
            &price,
            Some(provider.trim().to_string()),
        ))
    }

    /// Look up a cached price without a per-entry TTL check (callers gate on
    /// [`PricingCache::is_stale`]).
    fn cached_model(&self, model: &str) -> Option<ModelPrice> {
        self.inner.prices.read().get(model).map(|c| c.price.clone())
    }

    /// Look up an explicitly seeded fallback price, trying the exact key then
    /// the lowercased key.
    fn fallback_price(&self, model: &str) -> Option<ModelPrice> {
        self.inner
            .fallback
            .get(model)
            .cloned()
            .or_else(|| self.inner.fallback.get(&model.to_lowercase()).cloned())
    }

    fn cached_price(&self, model: &str) -> Option<ModelPrice> {
        let prices = self.inner.prices.read();
        let entry = prices.get(model)?;
        if entry.fetched_at.elapsed().ok()? > self.inner.ttl {
            return None;
        }
        Some(entry.price.clone())
    }

    /// Fetch the full pricing table from the endpoint and store it.
    async fn fetch_all(&self) -> Result<(), PricingError> {
        let resp = self
            .inner
            .client
            .get(&self.inner.endpoint)
            .header("User-Agent", "opensquilla-engine/0.1")
            .send()
            .await
            .map_err(PricingError::Http)?;

        if !resp.status().is_success() {
            return Err(PricingError::Status(resp.status().as_u16()));
        }

        let body: OpenRouterResponse = resp.json().await.map_err(PricingError::Json)?;
        let now = SystemTime::now();
        let mut prices = self.inner.prices.write();

        for model in body.data {
            let price = ModelPrice {
                model: model.id,
                input_per_1m: parse_price(model.pricing.prompt),
                output_per_1m: parse_price(model.pricing.completion),
                context_window: model.context_length,
            };
            prices.insert(
                price.model.clone(),
                CachedPrice {
                    price,
                    fetched_at: now,
                },
            );
        }

        self.inner.last_refresh.store(unix_now(), Ordering::Relaxed);
        debug!(cached_models = prices.len(), "Pricing cache refreshed");
        Ok(())
    }
}

/// Errors produced by the pricing fetch.
#[derive(Debug, thiserror::Error)]
pub enum PricingError {
    #[error("HTTP request failed: {0}")]
    Http(reqwest::Error),
    #[error("pricing endpoint returned status {0}")]
    Status(u16),
    #[error("failed to parse pricing response: {0}")]
    Json(reqwest::Error),
}

/// Compute request cost in USD from per-1M-token prices.
pub fn cost(input_per_1m: f64, output_per_1m: f64, input_tokens: u64, output_tokens: u64) -> f64 {
    let input_cost = (input_tokens as f64 / 1_000_000.0) * input_per_1m;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output_per_1m;
    input_cost + output_cost
}

/// Parse a price string such as "$0.50" or "0.50" into a float.
fn parse_price(value: Option<String>) -> f64 {
    value
        .and_then(|v| {
            let trimmed = v.trim().trim_start_matches('$');
            trimmed.parse::<f64>().ok()
        })
        .unwrap_or(0.0)
}

/// A reasonable default price for an unknown model.
pub fn default_price(model: &str) -> ModelPrice {
    ModelPrice {
        model: model.to_string(),
        input_per_1m: FALLBACK_INPUT_PRICE_PER_1M,
        output_per_1m: FALLBACK_OUTPUT_PRICE_PER_1M,
        context_window: None,
    }
}

/// A small built-in fallback table for well-known models.
fn builtin_fallback_prices() -> HashMap<String, ModelPrice> {
    let rows = [
        ("gpt-4o", 2.5, 10.0, Some(128_000)),
        ("gpt-4o-mini", 0.15, 0.6, Some(128_000)),
        ("gpt-4.1", 2.0, 8.0, Some(1_000_000)),
        ("claude-3-5-sonnet", 3.0, 15.0, Some(200_000)),
        ("claude-3-7-sonnet", 3.0, 15.0, Some(200_000)),
        ("claude-3-haiku", 0.25, 1.25, Some(200_000)),
        ("deepseek-chat", 0.14, 0.28, Some(64_000)),
        ("deepseek-reasoner", 0.55, 1.19, Some(64_000)),
        ("gemini-1.5-pro", 1.25, 5.0, Some(2_000_000)),
        ("gemini-2.0-flash", 0.10, 0.40, Some(1_000_000)),
        ("llama-3.3-70b", 0.25, 0.75, Some(128_000)),
    ];
    rows.iter()
        .map(|(name, in_p, out_p, ctx)| {
            (
                name.to_string(),
                ModelPrice {
                    model: name.to_string(),
                    input_per_1m: *in_p,
                    output_per_1m: *out_p,
                    context_window: *ctx,
                },
            )
        })
        .collect()
}

/// The current Unix timestamp in whole seconds, or 0 if the system clock is
/// before the Unix epoch.
fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Providers whose runtime is local — inference is free regardless of model id.
fn is_local_free_provider(provider: &str) -> bool {
    matches!(
        provider.trim().to_lowercase().as_str(),
        "ollama" | "lm_studio" | "ovms" | "vllm" | "local"
    )
}

/// Look up a model price in the static prefix table, mirroring the Python
/// backend's `_PRICING_TABLE` prefix matching. Returns `None` on no match.
fn lookup_static_price(model: &str) -> Option<ModelPrice> {
    let lower = model.trim().to_lowercase();
    for (prefix, in_p, out_p, ctx) in static_price_table() {
        if lower.starts_with(prefix) {
            return Some(ModelPrice {
                model: model.trim().to_string(),
                input_per_1m: *in_p,
                output_per_1m: *out_p,
                context_window: *ctx,
            });
        }
    }
    None
}

/// Built-in static pricing rows: `(model_prefix, input_per_1m, output_per_1m,
/// context_window)`. More specific prefixes must appear before broader ones so
/// prefix matching picks the tightest match.
fn static_price_table() -> &'static [(&'static str, f64, f64, Option<u64>)] {
    &[
        // Anthropic (OpenRouter-qualified, then unqualified).
        ("anthropic/claude-opus", 15.0, 75.0, Some(200_000)),
        ("anthropic/claude-sonnet", 3.0, 15.0, Some(200_000)),
        ("anthropic/claude-3-5-haiku", 0.80, 4.0, Some(200_000)),
        ("anthropic/claude-3-haiku", 0.25, 1.25, Some(200_000)),
        ("claude-opus", 15.0, 75.0, Some(200_000)),
        ("claude-sonnet", 3.0, 15.0, Some(200_000)),
        ("claude-3-5-sonnet", 3.0, 15.0, Some(200_000)),
        ("claude-3-5-haiku", 0.80, 4.0, Some(200_000)),
        ("claude-3-haiku", 0.25, 1.25, Some(200_000)),
        // OpenAI — narrow prefixes before broad.
        ("openai/gpt-4.1-nano", 0.10, 0.40, Some(1_000_000)),
        ("openai/gpt-4.1-mini", 0.40, 1.60, Some(1_000_000)),
        ("openai/gpt-4.1", 2.0, 8.0, Some(1_000_000)),
        ("openai/gpt-4o-mini", 0.15, 0.60, Some(128_000)),
        ("openai/gpt-4o", 2.50, 10.0, Some(128_000)),
        ("openai/gpt-4-turbo", 10.0, 30.0, Some(128_000)),
        ("gpt-4.1-nano", 0.10, 0.40, Some(1_000_000)),
        ("gpt-4.1-mini", 0.40, 1.60, Some(1_000_000)),
        ("gpt-4.1", 2.0, 8.0, Some(1_000_000)),
        ("gpt-4o-mini", 0.15, 0.60, Some(128_000)),
        ("gpt-4o", 2.50, 10.0, Some(128_000)),
        ("gpt-4-turbo", 10.0, 30.0, Some(128_000)),
        ("gpt-4", 30.0, 60.0, Some(128_000)),
        ("text-embedding-3-small", 0.02, 0.0, None),
        ("text-embedding-3-large", 0.13, 0.0, None),
        // DeepSeek.
        ("deepseek/deepseek-v4-flash", 0.14, 0.28, Some(64_000)),
        ("deepseek/deepseek-v4-pro", 0.435, 0.87, Some(64_000)),
        ("deepseek/deepseek-chat", 0.14, 0.28, Some(64_000)),
        ("deepseek/deepseek-reasoner", 0.26, 0.38, Some(64_000)),
        ("deepseek-v4-flash", 0.14, 0.28, Some(64_000)),
        ("deepseek-v4-pro", 0.435, 0.87, Some(64_000)),
        ("deepseek-chat", 0.14, 0.28, Some(64_000)),
        ("deepseek-reasoner", 0.26, 0.38, Some(64_000)),
        // Google Gemini.
        ("google/gemini-2.5-pro", 1.25, 10.0, Some(2_000_000)),
        ("google/gemini-2.5-flash", 0.15, 0.60, Some(1_000_000)),
        ("google/gemini-2.0-flash", 0.10, 0.40, Some(1_000_000)),
        ("gemini-2.5-pro", 1.25, 10.0, Some(2_000_000)),
        ("gemini-2.5-flash", 0.15, 0.60, Some(1_000_000)),
        ("gemini-2.0-flash", 0.10, 0.40, Some(1_000_000)),
        // Meta / Mistral / Qwen / xAI / Zhipu / Moonshot.
        ("meta-llama/llama-3.3-70b", 0.25, 0.75, Some(128_000)),
        ("llama-3.3-70b", 0.25, 0.75, Some(128_000)),
        ("mistralai/mistral-large", 0.50, 1.50, Some(128_000)),
        ("mistral-large", 0.50, 1.50, Some(128_000)),
        ("qwen/qwen3", 0.65, 3.25, Some(256_000)),
        ("qwen-max", 0.345, 1.377, Some(32_000)),
        ("qwen-plus", 0.115, 0.287, Some(128_000)),
        ("x-ai/grok", 1.25, 2.5, Some(256_000)),
        ("grok", 1.25, 2.5, Some(256_000)),
        ("z-ai/glm", 0.43, 1.74, Some(128_000)),
        ("glm", 0.43, 1.74, Some(128_000)),
        ("moonshotai/kimi", 0.95, 4.0, Some(128_000)),
        ("kimi", 0.95, 4.0, Some(128_000)),
        // Free / local runtimes.
        ("ollama/", 0.0, 0.0, None),
        ("local/", 0.0, 0.0, None),
        ("baai/", 0.0, 0.0, None),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        // gpt-4o: $2.5/1M input, $10/1M output.
        // 1M input + 0 output = $2.50
        assert!((cost(2.5, 10.0, 1_000_000, 0) - 2.5).abs() < 1e-9);
        // 0 input + 100k output = $1.00
        assert!((cost(2.5, 10.0, 0, 100_000) - 1.0).abs() < 1e-9);
        // 1k input + 1k output = tiny
        assert!((cost(2.5, 10.0, 1_000, 1_000) - 0.0125).abs() < 1e-9);
    }

    #[test]
    fn test_parse_price() {
        assert_eq!(parse_price(Some("$0.50".to_string())), 0.5);
        assert_eq!(parse_price(Some("0.50".to_string())), 0.5);
        assert_eq!(parse_price(Some("$2.5".to_string())), 2.5);
        assert_eq!(parse_price(None), 0.0);
        assert_eq!(parse_price(Some("n/a".to_string())), 0.0);
    }

    #[test]
    fn test_seed_and_fresh() {
        let cache = PricingCache::new();
        cache.seed(ModelPrice {
            model: "m1".to_string(),
            input_per_1m: 1.0,
            output_per_1m: 2.0,
            context_window: None,
        });
        assert!(cache.is_fresh("m1"));
        assert!(!cache.is_fresh("missing"));
        cache.invalidate("m1");
        assert!(!cache.is_fresh("m1"));
    }

    #[tokio::test]
    async fn test_price_for_falls_back() {
        // Endpoint intentionally broken so the fetch fails and we fall back.
        let cache = PricingCache::with_endpoint(
            reqwest::Client::new(),
            "http://127.0.0.1:1/nonexistent".to_string(),
        );
        let result = cache.price_for("gpt-4o").await;
        assert!(!result.from_cache);
        assert_eq!(result.price.input_per_1m, 2.5);
    }

    #[test]
    fn test_cached_len() {
        let cache = PricingCache::new();
        cache.seed(ModelPrice {
            model: "a".to_string(),
            input_per_1m: 1.0,
            output_per_1m: 2.0,
            context_window: None,
        });
        cache.seed(ModelPrice {
            model: "b".to_string(),
            input_per_1m: 1.0,
            output_per_1m: 2.0,
            context_window: None,
        });
        assert_eq!(cache.cached_len(), 2);
        cache.clear();
        assert_eq!(cache.cached_len(), 0);
    }

    #[test]
    fn test_new_cache_is_stale() {
        let cache = PricingCache::new();
        assert!(cache.is_stale(), "a fresh cache with no data is stale");
    }

    #[tokio::test]
    async fn test_get_price_uses_static_table() {
        // Endpoint is intentionally unreachable so resolution falls back to the
        // static table.
        let cache = PricingCache::with_endpoint(
            reqwest::Client::new(),
            "http://127.0.0.1:1/nonexistent".to_string(),
        );
        let price = cache
            .get_price("gpt-4o", "openrouter")
            .await
            .expect("price resolves");
        assert_eq!(price.input_price, 2.5);
        assert_eq!(price.output_price, 10.0);
        assert_eq!(price.currency, "usd");
        assert_eq!(price.token_window, Some(128_000));
    }

    #[tokio::test]
    async fn test_get_price_local_free_provider() {
        let cache = PricingCache::new();
        let price = cache
            .get_price("qwen3:4b", "ollama")
            .await
            .expect("price resolves");
        assert_eq!(price.input_price, 0.0);
        assert_eq!(price.output_price, 0.0);
    }

    #[tokio::test]
    async fn test_get_price_unknown_model_falls_back_to_default() {
        let cache = PricingCache::with_endpoint(
            reqwest::Client::new(),
            "http://127.0.0.1:1/nonexistent".to_string(),
        );
        let price = cache
            .get_price("totally-unknown-model-xyz", "")
            .await
            .expect("price resolves");
        assert_eq!(price.input_price, FALLBACK_INPUT_PRICE_PER_1M);
        assert_eq!(price.output_price, FALLBACK_OUTPUT_PRICE_PER_1M);
    }

    #[test]
    fn test_static_lookup_and_local_providers() {
        assert!(is_local_free_provider("ollama"));
        assert!(is_local_free_provider("LM_STUDIO"));
        assert!(!is_local_free_provider("openrouter"));
        let p = lookup_static_price("deepseek/deepseek-v4-flash").expect("static row");
        assert_eq!(p.input_per_1m, 0.14);
        assert!(lookup_static_price("nope").is_none());
    }
}
