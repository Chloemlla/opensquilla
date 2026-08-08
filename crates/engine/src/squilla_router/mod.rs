//! SquillaRouter V4 Phase 3 inference port.
//!
//! Mirrors the Python bundle under
//! `src/opensquilla/squilla_router/models/v4.2_phase3_inference`
//! (`runtime_src/src/router/inference`): 390-dim feature assembly, three
//! heads (LightGBM main/aux + ONNX MLP), alpha fusion, and the config-driven
//! six-layer post-processing pipeline.
//!
//! All trained parameters are loaded from JSON files exported by
//! `scripts/export_router_params.py` (pickle/`.bin` assets are never read by
//! Rust — that is a deliberate security property of the migration). The
//! `onnx` cargo feature gates the BGE/MLP ONNX sessions; every pure function
//! compiles without it.

pub mod config;
pub mod features;
pub mod flags;
pub mod heads;
pub mod postprocess;
pub mod predict;

use std::path::PathBuf;

/// Errors produced by the SquillaRouter Phase 3 inference port.
#[derive(Debug, thiserror::Error)]
pub enum SquillaRouterError {
    #[error("BGE/MLP ONNX inference requires the `onnx` cargo feature")]
    FeatureDisabled,
    #[error("missing model artifact `{artifact}` in {model_dir:?}")]
    MissingArtifact {
        artifact: &'static str,
        model_dir: PathBuf,
    },
    #[error("invalid params `{0}`")]
    InvalidParams(String),
    #[error("feature dimension mismatch: {0}")]
    FeatureDim(String),
    #[error("onnx session failure: {0}")]
    Onnx(String),
    #[error("invalid probability vector: {0}")]
    InvalidProbabilities(String),
    #[error("config error: {0}")]
    Config(String),
}
