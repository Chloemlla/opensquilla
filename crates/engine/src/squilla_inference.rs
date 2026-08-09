//! Squilla Phase 3 BGE ONNX embedding engine.
//!
//! The Python-side `squilla_router/models/v4.2_phase3_inference` pipeline is:
//! BGE ONNX encoder (`OnnxBGE`) -> ensemble fusion -> six-layer
//! post-processing -> R0-R3 route class. The full pipeline lives in
//! [`crate::squilla_router`]; this module carries only the portable BGE ONNX
//! encoder used by `squilla_router::predict::Phase3Router`, with a
//! feature-gated ONNX session wrapper (mirroring the `memory` crate's `onnx`
//! feature). The ensemble heads / post-processing have no portable duplicate
//! here; use `squilla_router` for routing.
//!
//! With the `onnx` feature disabled construction succeeds but every embed
//! returns `SquillaInferenceError::FeatureDisabled`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Errors produced by the BGE embedding engine.
#[derive(Debug, thiserror::Error)]
pub enum SquillaInferenceError {
    #[error("BGE ONNX inference requires the `onnx` cargo feature")]
    FeatureDisabled,
    #[error("missing model artifact `{artifact}` in {model_dir:?}")]
    MissingArtifact {
        artifact: &'static str,
        model_dir: PathBuf,
    },
    #[error("ONNX session failure: {0}")]
    Onnx(String),
    #[error("tokenizer failure: {0}")]
    Tokenizer(String),
}

/// BGE ONNX inference engine for Squilla Phase 3 model routing.
///
/// Construction only records the model directory; the ONNX session and
/// tokenizer are loaded eagerly and stored behind an `Arc` (mirroring
/// `memory::embedding::OnnxEmbeddingProvider`). If loading fails the struct
/// still constructs and every embed call returns an error, so callers degrade
/// gracefully.
#[cfg(feature = "onnx")]
pub struct SquillaInference {
    model_dir: PathBuf,
    session: Option<Arc<std::sync::Mutex<ort::session::Session>>>,
    tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

/// BGE ONNX inference engine when the `onnx` feature is disabled.
#[cfg(not(feature = "onnx"))]
pub struct SquillaInference {
    model_dir: PathBuf,
}

#[cfg(feature = "onnx")]
impl SquillaInference {
    /// Create a new inference engine for a model directory expected to contain
    /// `model.onnx` and `tokenizer.json`.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        let model_dir = model_dir.into();
        let session = Self::load_session(&model_dir);
        let tokenizer = Self::load_tokenizer(&model_dir);
        Self {
            model_dir,
            session,
            tokenizer,
        }
    }

    fn load_session(model_dir: &Path) -> Option<Arc<std::sync::Mutex<ort::session::Session>>> {
        let path = model_dir.join("model.onnx");
        if !path.exists() {
            tracing::warn!("missing ONNX model at {}", path.display());
            return None;
        }
        let mut builder = match ort::session::Session::builder() {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("failed to create ONNX session builder: {}", e);
                return None;
            }
        };
        match builder.commit_from_file(&path) {
            Ok(s) => Some(Arc::new(std::sync::Mutex::new(s))),
            Err(e) => {
                tracing::warn!("failed to load ONNX model from {}: {}", path.display(), e);
                None
            }
        }
    }

    fn load_tokenizer(model_dir: &Path) -> Option<Arc<tokenizers::Tokenizer>> {
        let path = model_dir.join("tokenizer.json");
        if !path.exists() {
            tracing::warn!("missing tokenizer at {}", path.display());
            return None;
        }
        match tokenizers::Tokenizer::from_file(&path) {
            Ok(mut t) => {
                // BGE truncates inputs to 510 tokens
                // (`bge_onnx.py::_DEFAULT_MAX_LENGTH`), so long inputs must not
                // exceed the model's max position embeddings.
                let params = tokenizers::TruncationParams {
                    max_length: 510,
                    stride: 0,
                    strategy: tokenizers::TruncationStrategy::LongestFirst,
                    direction: tokenizers::TruncationDirection::Right,
                };
                match t.with_truncation(Some(params)) {
                    Ok(_) => Some(Arc::new(t)),
                    Err(e) => {
                        tracing::warn!(
                            "failed to configure truncation for {}: {}",
                            path.display(),
                            e
                        );
                        None
                    }
                }
            }
            Err(e) => {
                tracing::warn!("failed to load tokenizer from {}: {}", path.display(), e);
                None
            }
        }
    }

    /// Embed a single text with the BGE model.
    ///
    /// Runs tokenize -> ONNX -> CLS pooling -> L2 normalization, matching the
    /// Python `OnnxBGE.encode` (the memory crate's mean-pooled encoder is not
    /// used here). Returns an error when the model/tokenizer failed to load or
    /// the `onnx` feature is disabled.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, SquillaInferenceError> {
        let tokenizer =
            self.tokenizer
                .as_ref()
                .ok_or_else(|| SquillaInferenceError::MissingArtifact {
                    artifact: "tokenizer.json",
                    model_dir: self.model_dir.clone(),
                })?;
        let session =
            self.session
                .as_ref()
                .ok_or_else(|| SquillaInferenceError::MissingArtifact {
                    artifact: "model.onnx",
                    model_dir: self.model_dir.clone(),
                })?;
        let mut session = session
            .lock()
            .map_err(|e| SquillaInferenceError::Onnx(format!("lock ONNX session: {e}")))?;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| SquillaInferenceError::Tokenizer(e.to_string()))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect();
        let types: Vec<i64> = encoding.get_type_ids().iter().map(|&v| v as i64).collect();
        let seq_len = ids.len();

        let inputs = ort::value::Tensor::from_array(([1usize, seq_len], ids))
            .map_err(|e| SquillaInferenceError::Onnx(e.to_string()))?;
        let mask_value = ort::value::Tensor::from_array(([1usize, seq_len], mask))
            .map_err(|e| SquillaInferenceError::Onnx(e.to_string()))?;
        let types_value = ort::value::Tensor::from_array(([1usize, seq_len], types))
            .map_err(|e| SquillaInferenceError::Onnx(e.to_string()))?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => inputs,
                "attention_mask" => mask_value,
                "token_type_ids" => types_value
            ])
            .map_err(|e| SquillaInferenceError::Onnx(e.to_string()))?;

        // BGE uses CLS pooling (`last_hidden[:, 0, :]`), unlike the memory
        // crate's mean pooling.
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SquillaInferenceError::Onnx(e.to_string()))?;
        let dims: &[i64] = shape;
        let cls: Vec<f32> = match dims {
            [1, _seq, hidden] => data[..*hidden as usize].to_vec(),
            [1, _hidden] => data.to_vec(),
            other => {
                return Err(SquillaInferenceError::Onnx(format!(
                    "unexpected BGE output shape: {other:?}"
                )));
            }
        };

        let norm: f32 = cls.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(cls.iter().map(|v| v / norm).collect())
        } else {
            Ok(cls)
        }
    }
}

#[cfg(not(feature = "onnx"))]
impl SquillaInference {
    /// Create a new inference engine; embedding is unavailable until the
    /// `onnx` feature is enabled.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
        }
    }

    /// Embedding is unavailable without the `onnx` cargo feature.
    pub fn embed(&self, _text: &str) -> Result<Vec<f32>, SquillaInferenceError> {
        Err(SquillaInferenceError::FeatureDisabled)
    }
}
