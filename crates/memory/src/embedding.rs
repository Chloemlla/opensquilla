use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};

use crate::store::MemoryStore;

/// Configuration for an [`EmbeddingProvider`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub dimension: usize,
    /// Path to a local ONNX model directory (for the ONNX provider).
    #[serde(default)]
    pub model_path: Option<String>,
    /// Path to a tokenizer file or directory (for the ONNX provider).
    #[serde(default)]
    pub tokenizer_path: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: String::from("openai"),
            model: String::from("text-embedding-3-small"),
            api_key: None,
            base_url: None,
            dimension: 1536,
            model_path: None,
            tokenizer_path: None,
        }
    }
}

/// A provider of dense vector embeddings for text.
///
/// Implementations are expected to be cheap to clone (they typically wrap a
/// `reqwest::Client` or a loaded ONNX session).
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single piece of text.
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>>;

    /// Embed a batch of texts. Default implementation loops over [`embed`],
    /// but providers with a native batch endpoint should override this.
    async fn embed_batch(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t).await?);
        }
        Ok(out)
    }

    /// The dimensionality of vectors produced by this provider.
    fn dimension(&self) -> usize;

    /// The provider's configuration.
    fn config(&self) -> &EmbeddingConfig;

    /// A short identifier for the provider (e.g. `"openai"`, `"ollama"`, `"onnx"`).
    fn name(&self) -> &str {
        &self.config().provider
    }
}

/// Detect the embedding dimension by embedding a probe string. Useful when the
/// configured `dimension` is unknown/zero.
pub async fn detect_dimension(provider: &dyn EmbeddingProvider) -> CoreResult<usize> {
    let probe = provider.embed("dimension probe").await?;
    Ok(probe.len())
}

/// Wrap a provider with an embedding cache backed by a [`MemoryStore`].
///
/// On every [`embed`] call, the cache is consulted first using
/// `(model, text)` as the key; misses fall through to the inner provider and
/// the result is stored before being returned.
#[derive(Clone)]
pub struct CachedEmbeddingProvider {
    inner: Arc<dyn EmbeddingProvider>,
    store: MemoryStore,
}

impl CachedEmbeddingProvider {
    pub fn new(inner: Arc<dyn EmbeddingProvider>, store: MemoryStore) -> Self {
        Self { inner, store }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for CachedEmbeddingProvider {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let model = &self.inner.config().model;
        if let Ok(Some(cached)) = self.store.get_cached_embedding(model, text) {
            debug!(
                "embedding cache hit (model={}, len={} chars)",
                model,
                text.len()
            );
            return Ok(cached);
        }
        let emb = self.inner.embed(text).await?;
        if let Err(e) = self.store.set_cached_embedding(model, text, &emb) {
            warn!("failed to cache embedding: {}", e);
        }
        Ok(emb)
    }

    async fn embed_batch(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let model = &self.inner.config().model;
        let mut results = Vec::with_capacity(texts.len());
        let mut misses: Vec<(usize, String)> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            match self.store.get_cached_embedding(model, text) {
                Ok(Some(cached)) => results.push(cached),
                _ => {
                    results.push(Vec::new());
                    misses.push((i, text.clone()));
                }
            }
        }
        if !misses.is_empty() {
            let miss_texts: Vec<String> = misses.iter().map(|(_, t)| t.clone()).collect();
            let computed = self.inner.embed_batch(&miss_texts).await?;
            for ((i, text), emb) in misses.into_iter().zip(computed) {
                if let Err(e) = self.store.set_cached_embedding(model, &text, &emb) {
                    warn!("failed to cache embedding: {}", e);
                }
                results[i] = emb;
            }
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn config(&self) -> &EmbeddingConfig {
        self.inner.config()
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// --- OpenAI Embedding Provider ---

/// Embedding provider that calls an OpenAI-compatible `/v1/embeddings` endpoint.
pub struct OpenAIEmbeddingProvider {
    config: EmbeddingConfig,
    client: reqwest::Client,
}

impl OpenAIEmbeddingProvider {
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    pub fn with_client(config: EmbeddingConfig, client: reqwest::Client) -> Self {
        Self { config, client }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| CoreError::Provider("No embedding returned".to_string()))
    }

    async fn embed_batch(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| CoreError::Provider("No API key configured".to_string()))?;
        let base_url = self
            .config
            .base_url
            .clone()
            .unwrap_or_else(|| String::from("https://api.openai.com/v1"));

        let body = serde_json::json!({
            "model": self.config.model,
            "input": texts,
        });

        let resp = self
            .client
            .post(format!("{}/embeddings", base_url))
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| CoreError::Provider(format!("Embedding request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CoreError::Provider(format!(
                "Embedding API returned {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CoreError::Provider(format!("Parse failed: {}", e)))?;

        let embeddings: Vec<Vec<f32>> = data["data"]
            .as_array()
            .ok_or_else(|| CoreError::Provider("No data in response".to_string()))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

// --- Ollama Embedding Provider ---

/// Embedding provider that calls an Ollama `/api/embeddings` endpoint.
pub struct OllamaEmbeddingProvider {
    config: EmbeddingConfig,
    client: reqwest::Client,
}

impl OllamaEmbeddingProvider {
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = reqwest::Client::new();
        Self { config, client }
    }

    pub fn with_client(config: EmbeddingConfig, client: reqwest::Client) -> Self {
        Self { config, client }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OllamaEmbeddingProvider {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let base_url = self
            .config
            .base_url
            .clone()
            .unwrap_or_else(|| String::from("http://localhost:11434"));

        let body = serde_json::json!({
            "model": self.config.model,
            "prompt": text,
        });

        let resp = self
            .client
            .post(format!("{}/api/embeddings", base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| CoreError::Provider(format!("Ollama embedding failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(CoreError::Provider(format!(
                "Ollama embeddings API returned {}: {}",
                status, body
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| CoreError::Provider(format!("Parse failed: {}", e)))?;

        let embedding: Vec<f32> = data["embedding"]
            .as_array()
            .ok_or_else(|| CoreError::Provider("No embedding in response".to_string()))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

// --- ONNX Embedding Provider (local inference via the `ort` crate) ---

/// Embedding provider that runs a local ONNX model via the `ort` crate and a
/// HuggingFace `tokenizers` tokenizer.
///
/// Enabled by the `onnx` cargo feature. Construction loads the model session
/// and tokenizer eagerly; both are stored behind an `Arc` so the provider is
/// cheap to share across tasks.
#[cfg(feature = "onnx")]
pub struct OnnxEmbeddingProvider {
    config: EmbeddingConfig,
    session: Option<Arc<std::sync::Mutex<ort::session::Session>>>,
    tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    detected_dimension: usize,
}

#[cfg(feature = "onnx")]
impl OnnxEmbeddingProvider {
    /// Create a new ONNX provider. Loads the model and tokenizer from the
    /// paths in the config; if loading fails the provider still constructs but
    /// every embed call returns an error, so callers can degrade gracefully.
    pub fn new(config: EmbeddingConfig) -> Self {
        let session = Self::load_session(&config);
        let tokenizer = Self::load_tokenizer(&config);
        let detected_dimension = config.dimension;
        Self {
            config,
            session,
            tokenizer,
            detected_dimension,
        }
    }

    fn load_session(
        config: &EmbeddingConfig,
    ) -> Option<Arc<std::sync::Mutex<ort::session::Session>>> {
        let path = config.model_path.as_ref()?;
        let builder = match ort::session::Session::builder() {
            Ok(b) => b,
            Err(e) => {
                warn!("failed to create ONNX session builder: {}", e);
                return None;
            }
        };
        let mut builder = match builder.with_intra_threads(1) {
            Ok(b) => b,
            Err(e) => {
                warn!("failed to configure ONNX intra threads: {}", e);
                return None;
            }
        };
        match builder.commit_from_file(path) {
            Ok(s) => Some(Arc::new(std::sync::Mutex::new(s))),
            Err(e) => {
                warn!("failed to load ONNX model from {}: {}", path, e);
                None
            }
        }
    }

    fn load_tokenizer(config: &EmbeddingConfig) -> Option<Arc<tokenizers::Tokenizer>> {
        let path = config.tokenizer_path.as_ref()?;
        match tokenizers::Tokenizer::from_file(path) {
            Ok(t) => Some(Arc::new(t)),
            Err(e) => {
                warn!("failed to load tokenizer from {}: {}", path, e);
                None
            }
        }
    }

    /// Tokenize a piece of text and return (input_ids, attention_mask) as
    /// flat `i64` tensors suitable for the model.
    fn tokenize(&self, text: &str) -> CoreResult<(Vec<i64>, Vec<i64>)> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| CoreError::Provider("ONNX tokenizer not loaded".to_string()))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| CoreError::Provider(format!("Tokenization failed: {}", e)))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect();
        Ok((ids, mask))
    }
}

#[cfg(feature = "onnx")]
#[async_trait::async_trait]
impl EmbeddingProvider for OnnxEmbeddingProvider {
    async fn embed(&self, text: &str) -> CoreResult<Vec<f32>> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| CoreError::Provider("ONNX model not loaded".to_string()))?;
        let mut session = session
            .lock()
            .map_err(|e| CoreError::Provider(format!("lock ONNX session: {}", e)))?;

        let (input_ids, attention_mask) = self.tokenize(text)?;
        let seq_len = input_ids.len();

        // Build input tensors with shape [1, seq_len]. The `(shape, data)`
        // tuple form is used so we don't depend on a specific ndarray version
        // (the `ort` crate vendors its own ndarray).
        let inputs = ort::value::Tensor::from_array(([1usize, seq_len], input_ids))
            .map_err(|e| CoreError::Provider(format!("ids value: {}", e)))?;
        let mask_value = ort::value::Tensor::from_array(([1usize, seq_len], attention_mask))
            .map_err(|e| CoreError::Provider(format!("mask value: {}", e)))?;

        let outputs = session
            .run(ort::inputs!["input_ids" => inputs, "attention_mask" => mask_value])
            .map_err(|e| CoreError::Provider(format!("ONNX inference failed: {}", e)))?;

        // The first output is the token embeddings / last_hidden_state with
        // shape [1, seq_len, hidden]. Mean-pool across the seq_len axis to get
        // a single sentence embedding.
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Provider(format!("extract array: {}", e)))?;

        // Expect shape [1, seq_len, hidden] or [1, hidden].
        let dims: &[i64] = shape;
        let pooled: Vec<f32> = match dims {
            [1, seq, hidden] if *seq == seq_len as i64 => {
                let seq = *seq as usize;
                let hidden = *hidden as usize;
                let mut acc = vec![0.0_f32; hidden];
                for s in 0..seq {
                    for h in 0..hidden {
                        acc[h] += data[s * hidden + h];
                    }
                }
                for v in acc.iter_mut() {
                    *v /= seq as f32;
                }
                acc
            }
            [1, _] => data.to_vec(),
            other => {
                return Err(CoreError::Provider(format!(
                    "Unexpected ONNX output shape: {:?}",
                    other
                )));
            }
        };

        // L2-normalize the pooled embedding for cosine similarity.
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            pooled.iter().map(|v| v / norm).collect()
        } else {
            pooled
        };

        Ok(normalized)
    }

    fn dimension(&self) -> usize {
        if self.detected_dimension > 0 {
            self.detected_dimension
        } else {
            self.config.dimension
        }
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Embedding provider for local ONNX inference when the `onnx` feature is
/// disabled. Construction succeeds but every embed call returns an error, so
/// callers degrade gracefully.
#[cfg(not(feature = "onnx"))]
pub struct OnnxEmbeddingProvider {
    config: EmbeddingConfig,
}

#[cfg(not(feature = "onnx"))]
impl OnnxEmbeddingProvider {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self { config }
    }
}

#[cfg(not(feature = "onnx"))]
#[async_trait::async_trait]
impl EmbeddingProvider for OnnxEmbeddingProvider {
    async fn embed(&self, _text: &str) -> CoreResult<Vec<f32>> {
        Err(CoreError::Provider(
            "ONNX embedding requires the 'onnx' cargo feature".to_string(),
        ))
    }

    async fn embed_batch(&self, _texts: &[String]) -> CoreResult<Vec<Vec<f32>>> {
        Err(CoreError::Provider(
            "ONNX embedding requires the 'onnx' cargo feature".to_string(),
        ))
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }

    fn config(&self) -> &EmbeddingConfig {
        &self.config
    }
}

/// Construct a provider from its config, selecting on `config.provider`.
pub fn build_provider(config: EmbeddingConfig) -> Arc<dyn EmbeddingProvider> {
    match config.provider.to_lowercase().as_str() {
        "openai" => Arc::new(OpenAIEmbeddingProvider::new(config)),
        "ollama" => Arc::new(OllamaEmbeddingProvider::new(config)),
        "onnx" => Arc::new(OnnxEmbeddingProvider::new(config)),
        other => {
            warn!(
                "unknown embedding provider '{}'; falling back to OpenAI-compatible",
                other
            );
            Arc::new(OpenAIEmbeddingProvider::new(config))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let c = EmbeddingConfig::default();
        assert_eq!(c.provider, "openai");
        assert_eq!(c.dimension, 1536);
    }

    #[test]
    fn test_build_provider_selects() {
        let mut c = EmbeddingConfig::default();
        c.provider = "ollama".to_string();
        let p = build_provider(c);
        assert_eq!(p.name(), "ollama");

        let mut c = EmbeddingConfig::default();
        c.provider = "onnx".to_string();
        let p = build_provider(c);
        assert_eq!(p.name(), "onnx");
    }

    #[test]
    fn test_cache_key_is_stable() {
        let k1 = MemoryStore::embedding_cache_key("model-a", "hello");
        let k2 = MemoryStore::embedding_cache_key("model-a", "hello");
        let k3 = MemoryStore::embedding_cache_key("model-b", "hello");
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }
}
