//! Squilla Phase 3 model routing: portable BGE ONNX inference subset.
//!
//! The Python-side `squilla_router/models/v4.2_phase3_inference` pipeline is:
//! BGE ONNX encoder (`OnnxBGE`) -> ensemble fusion (`fuse_probabilities`) ->
//! six-layer post-processing (`apply_post_processing`) -> R0-R3 route class.
//! The ONNX encoder (`model.onnx` + `tokenizer.json`) and the trained ensemble
//! heads (`lgbm_model.bin` / `joblib` / `pkl`) are binary assets loaded at
//! runtime; they cannot be reconstructed in code. This module carries the
//! portable parts: a feature-gated ONNX session wrapper (mirroring the
//! `memory` crate's `onnx` feature) and the pure probability / post-processing
//! logic, which is always compiled.
//!
//! With the `onnx` feature disabled construction succeeds but every embed
//! returns `SquillaInferenceError::FeatureDisabled`; the pure functions are
//! unaffected.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The four Phase 3 route classes, in index order (R0 = easiest, R3 = hardest).
pub const ROUTE_CLASSES: [&str; 4] = ["R0", "R1", "R2", "R3"];

/// Flag summary used by the post-processing overrides
/// (Python `flags.py::compute_flags` subset).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutingFlags {
    pub high_risk: bool,
    pub debug: bool,
    pub long_context: bool,
    pub repo_arch: bool,
    pub strict_format: bool,
}

/// Final routing decision after the full post-processing pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteDecision {
    pub route_class: &'static str,
    pub margin: f64,
    pub flags: RoutingFlags,
}

/// Errors produced by the Squilla inference module.
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
    #[error(
        "ensemble heads (lgbm_model.bin / joblib / pkl) are binary assets not yet wired into Rust: {0}"
    )]
    HeadsMissing(String),
    #[error("invalid probability vector: {0}")]
    InvalidProbabilities(String),
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

impl SquillaInference {
    /// The model directory this engine was constructed with.
    pub fn model_dir(&self) -> &PathBuf {
        &self.model_dir
    }

    /// Route a text to one of R0-R3.
    ///
    /// Placeholder: the full pipeline needs the ensemble heads
    /// (`lgbm_model.bin` + MLP/alpha `joblib`/`pkl`), which are binary assets
    /// loaded at runtime and not yet wired in, so this always returns
    /// [`SquillaInferenceError::HeadsMissing`]. Callers should feed
    /// [`Self::embed`] + [`fuse_probabilities`] + [`apply_post_processing`]
    /// once the heads are available.
    pub fn route(&self, _text: &str) -> Result<RouteDecision, SquillaInferenceError> {
        Err(SquillaInferenceError::HeadsMissing(
            "lgbm_model.bin + ensemble heads must be loaded at runtime".to_string(),
        ))
    }
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
            Ok(t) => Some(Arc::new(t)),
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

/// Fused 4-class probability vector via alpha-weighted mixing, renormalized to
/// sum to one. Mirrors `inference/ensemble.py::fuse_probabilities`.
pub fn fuse_probabilities(
    p_main: [f64; 4],
    p_mlp: [f64; 4],
    alpha: [f64; 4],
) -> Result<[f64; 4], SquillaInferenceError> {
    for (i, (&p, &m)) in p_main.iter().zip(p_mlp.iter()).enumerate() {
        if !p.is_finite() || !m.is_finite() || !alpha[i].is_finite() {
            return Err(SquillaInferenceError::InvalidProbabilities(
                "ensemble inputs must be finite".to_string(),
            ));
        }
    }
    for &a in &alpha {
        if !(0.0..=1.0).contains(&a) {
            return Err(SquillaInferenceError::InvalidProbabilities(
                "alpha must stay within [0, 1]".to_string(),
            ));
        }
    }
    let mixed: [f64; 4] =
        std::array::from_fn(|i| alpha[i] * p_main[i] + (1.0 - alpha[i]) * p_mlp[i]);
    let total: f64 = mixed.iter().sum();
    if total <= 0.0 {
        return Err(SquillaInferenceError::InvalidProbabilities(
            "fused probability mass must be positive".to_string(),
        ));
    }
    Ok(std::array::from_fn(|i| mixed[i] / total))
}

/// Indices of the largest and second-largest entries of a 4-vector.
fn argmax2(probs: &[f64; 4]) -> (usize, usize) {
    let mut best = 0usize;
    let mut second = 1usize;
    for i in 1..4 {
        if probs[i] > probs[best] {
            second = best;
            best = i;
        } else if i != best && probs[i] > probs[second] {
            second = i;
        }
    }
    (best, second)
}

/// Apply the pure post-processing layers over a 4-class probability vector:
/// argmax -> margin upgrade -> R1 rescue -> under-routing safety net -> flag
/// overrides. Mirrors `predictor.py::apply_post_processing` (context rules and
/// KV-cache sticky tier are omitted; they need turn history).
pub fn apply_post_processing(
    probs: [f64; 4],
    flags: RoutingFlags,
) -> Result<RouteDecision, SquillaInferenceError> {
    for &p in &probs {
        if !p.is_finite() {
            return Err(SquillaInferenceError::InvalidProbabilities(
                "probabilities must be finite".to_string(),
            ));
        }
    }
    let (best, second) = argmax2(&probs);
    let margin = probs[best] - probs[second];
    let mut class_idx = best;

    // Margin upgrade: a thin confidence gap bumps one tier up.
    if margin < 0.15 && class_idx < 3 {
        class_idx += 1;
    }
    // R1 rescue: promote R0 -> R1 when R1 is a close second.
    if class_idx == 0 && probs[0] - probs[1] < 0.20 {
        class_idx = 1;
    }
    // Under-routing safety net: heavy R2+R3 mass forces at least R2.
    if class_idx < 2 && probs[2] + probs[3] > 0.45 {
        class_idx = 2;
    }
    // Flag overrides floor-lift the class.
    if flags.high_risk {
        class_idx = class_idx.max(2);
    }
    if flags.debug && flags.long_context {
        class_idx = class_idx.max(2);
    }
    if flags.repo_arch {
        class_idx = class_idx.max(1);
    }

    Ok(RouteDecision {
        route_class: ROUTE_CLASSES[class_idx],
        margin,
        flags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuse_probabilities_renormalizes_mixture() {
        let p_main = [0.8, 0.1, 0.05, 0.05];
        let p_mlp = [0.2, 0.5, 0.2, 0.1];
        let fused = fuse_probabilities(p_main, p_mlp, [0.7; 4]).expect("finite inputs fuse");
        let total: f64 = fused.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "fused vector must be normalized, got {total}"
        );
    }

    #[test]
    fn fuse_probabilities_rejects_out_of_range_alpha() {
        let err = fuse_probabilities([0.5; 4], [0.5; 4], [0.5, 1.5, 0.5, 0.5]).unwrap_err();
        assert!(matches!(
            err,
            SquillaInferenceError::InvalidProbabilities(_)
        ));
    }

    #[test]
    fn apply_post_processing_applies_margin_upgrade() {
        // Thin-margin R0 win: margin upgrade bumps to R1.
        let thin = [0.42, 0.30, 0.16, 0.12];
        let decision = apply_post_processing(thin, RoutingFlags::default()).expect("valid probs");
        assert_eq!(decision.route_class, "R1");
        assert!((decision.margin - 0.12).abs() < 1e-9);
    }

    #[test]
    fn apply_post_processing_applies_safety_net_and_flag_override() {
        // Heavy R2+R3 tail under an R0 win triggers the under-routing safety net.
        let heavy_tail = [0.40, 0.05, 0.30, 0.25];
        let decision =
            apply_post_processing(heavy_tail, RoutingFlags::default()).expect("valid probs");
        assert_eq!(decision.route_class, "R2");

        // high_risk flag floor-lifts a confident R0 to R2.
        let flags = RoutingFlags {
            high_risk: true,
            ..RoutingFlags::default()
        };
        let decision = apply_post_processing([0.5, 0.2, 0.2, 0.1], flags).expect("valid probs");
        assert_eq!(decision.route_class, "R2");
    }

    #[test]
    fn apply_post_processing_rejects_non_finite_probs() {
        let err = apply_post_processing([f64::NAN, 0.5, 0.25, 0.25], RoutingFlags::default())
            .unwrap_err();
        assert!(matches!(
            err,
            SquillaInferenceError::InvalidProbabilities(_)
        ));
    }
}
