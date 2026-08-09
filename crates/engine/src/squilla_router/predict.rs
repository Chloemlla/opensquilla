//! Top-level Phase 3 router: assembles the 390-dim feature bundle, runs the
//! three heads, fuses with the manifest per-class alpha, and applies the
//! config-driven post-processing pipeline.
//!
//! Mirrors `runtime_src/src/router/inference/core.py::route`. All trained
//! parameters are loaded at runtime from the bundle directory: the config
//! (`router.runtime.yaml`), the manifest (`inference_manifest.json`), the
//! exported feature params and LightGBM forests (`params/*.json`), the ONNX
//! MLP head (`mlp/model.onnx`), and the BGE ONNX encoder (`bge_onnx/`). The
//! BGE/MLP steps are gated behind the `onnx` cargo feature; the pure fusion,
//! channel-assembly, and head-decision helpers compile without it.

use crate::squilla_router::SquillaRouterError;
use crate::squilla_router::config::{Manifest, RouterConfig};
use crate::squilla_router::features::assistant::extract_assistant_handcrafted;
use crate::squilla_router::features::context::extract_context_features;
use crate::squilla_router::features::continuation::extract_continuation_features;
use crate::squilla_router::features::handcrafted::extract_handcrafted;
use crate::squilla_router::features::history::extract_hist_features;
use crate::squilla_router::features::matrix::TrainedTransforms;
use crate::squilla_router::features::reasoning::extract_reasoning_features;
use crate::squilla_router::features::{
    ContextMetadata, FEATURE_390_DIM, FeatureChannels, FeatureInput, RAW_BGE_1536_DIM,
};
#[cfg(feature = "onnx")]
use crate::squilla_router::features::{HISTORY_USER_MAX_CHARS, make_history_user_text_with};
use crate::squilla_router::heads::{GbdtScorer, HeadOutputs};
use crate::squilla_router::postprocess::{FinalDecision, apply_postprocess};
use std::path::{Path, PathBuf};

#[cfg(feature = "onnx")]
use crate::squilla_inference::SquillaInference;
#[cfg(feature = "onnx")]
use crate::squilla_router::heads::{MlpHead, run_heads};

/// The full Phase 3 router state: configuration, trained transforms, and the
/// three heads. Constructed by [`Phase3Router::load`]; the BGE/MLP pieces are
/// only present under the `onnx` cargo feature.
pub struct Phase3Router {
    pub model_dir: PathBuf,
    pub config: RouterConfig,
    pub manifest: Manifest,
    pub transforms: TrainedTransforms,
    pub main_head: GbdtScorer,
    pub aux_head: Option<GbdtScorer>,
    #[cfg(feature = "onnx")]
    pub mlp_head: Option<MlpHead>,
    #[cfg(feature = "onnx")]
    pub bge: Option<SquillaInference>,
}

impl Phase3Router {
    /// Load the full router state from a Phase 3 bundle directory.
    ///
    /// The bundle must contain `router.runtime.yaml`, `inference_manifest.json`,
    /// a `params/` dir (as produced by `scripts/export_router_params.py`) and,
    /// under the `onnx` feature, the `mlp/` and `bge_onnx/` directories. Any
    /// missing required artifact maps to `SquillaRouterError::MissingArtifact`.
    pub fn load(bundle_dir: &Path) -> Result<Self, SquillaRouterError> {
        let config = RouterConfig::from_file(&bundle_dir.join("router.runtime.yaml"))?;
        let manifest = Manifest::from_file(&bundle_dir.join("inference_manifest.json"))?;
        let transforms = TrainedTransforms::load(&bundle_dir.join("params"))?;
        let main_head = GbdtScorer::from_json(&bundle_dir.join("params").join("lgbm_main.json"))?;
        let aux_head = if config.v4.aux_head_inference {
            Some(GbdtScorer::from_json(
                &bundle_dir.join("params").join("lgbm_aux.json"),
            )?)
        } else {
            None
        };
        #[cfg(feature = "onnx")]
        let mlp_head = Some(MlpHead::load(&bundle_dir.join("mlp"))?);
        #[cfg(feature = "onnx")]
        let bge = Some(SquillaInference::new(
            bundle_dir.join(&config.v4.bge_onnx_dir),
        ));

        Ok(Self {
            model_dir: bundle_dir.to_path_buf(),
            config,
            manifest,
            transforms,
            main_head,
            aux_head,
            #[cfg(feature = "onnx")]
            mlp_head,
            #[cfg(feature = "onnx")]
            bge,
        })
    }

    /// Route a single turn through the full Phase 3 pipeline: embed the three
    /// BGE segments, assemble the 390-dim bundle, run the heads, fuse, and
    /// post-process. Requires the `onnx` cargo feature.
    #[cfg(feature = "onnx")]
    pub fn route(&self, request: &FeatureInput) -> Result<FinalDecision, SquillaRouterError> {
        let bge = self
            .bge
            .as_ref()
            .ok_or_else(|| SquillaRouterError::MissingArtifact {
                artifact: "bge_onnx/model.onnx",
                model_dir: self.model_dir.clone(),
            })?;
        let mlp = self
            .mlp_head
            .as_ref()
            .ok_or_else(|| SquillaRouterError::MissingArtifact {
                artifact: "mlp/model.onnx",
                model_dir: self.model_dir.clone(),
            })?;
        let bge_vecs = embed_segments(request, bge, self.config.v4.history_user_max_turns)?;
        self.predict_with_bge(request, mlp, &bge_vecs)
    }

    /// Stub for builds without the `onnx` feature: BGE/MLP are unavailable, so
    /// routing always returns `SquillaRouterError::FeatureDisabled`.
    #[cfg(not(feature = "onnx"))]
    pub fn route(&self, _request: &FeatureInput) -> Result<FinalDecision, SquillaRouterError> {
        Err(SquillaRouterError::FeatureDisabled)
    }

    /// Run the head + fusion + post-process chain from pre-computed BGE
    /// vectors (the embed step is factored out so the rest is testable).
    #[cfg(feature = "onnx")]
    fn predict_with_bge(
        &self,
        request: &FeatureInput,
        mlp: &MlpHead,
        bge_vecs: &[Vec<f32>; 3],
    ) -> Result<FinalDecision, SquillaRouterError> {
        let channels = assemble_channels(request, &self.transforms, bge_vecs);
        let features_390 = channels_to_390(&channels);
        let raw_1536 = concat_bge_raw(bge_vecs);
        let outputs = run_heads(
            &features_390,
            &raw_1536,
            &self.main_head,
            self.aux_head.as_ref(),
            mlp,
            &self.transforms,
            self.manifest.temperature,
        )?;
        route_from_heads(&outputs, &self.config, &self.manifest, request)
    }
}

/// Embed the three Phase 3 text segments into 512-dim vectors, in the order
/// `[current_user, history_user, prev_assistant]`. The history segment joins
/// up to `history_user_max_turns` prior user turns (`config.v4.*`), matching
/// the Python `v4_features.make_history_user_text`.
#[cfg(feature = "onnx")]
fn embed_segments(
    request: &FeatureInput,
    bge: &SquillaInference,
    history_user_max_turns: usize,
) -> Result<[Vec<f32>; 3], SquillaRouterError> {
    let history = make_history_user_text_with(
        &request.history_user_texts,
        history_user_max_turns,
        HISTORY_USER_MAX_CHARS,
    );
    let current = bge
        .embed(&request.current_user_text)
        .map_err(|e| SquillaRouterError::Onnx(format!("bge embed current_user: {e}")))?;
    let history_vec = bge
        .embed(&history)
        .map_err(|e| SquillaRouterError::Onnx(format!("bge embed history_user: {e}")))?;
    let prev = bge
        .embed(request.prev_assistant_text.as_deref().unwrap_or(""))
        .map_err(|e| SquillaRouterError::Onnx(format!("bge embed prev_assistant: {e}")))?;
    Ok([current, history_vec, prev])
}

/// Fuse the main and MLP head probabilities with per-class alpha weights,
/// renormalized to sum to one. Mirrors `ensemble.py::fuse_probabilities`.
pub fn fuse_per_class_alpha(
    p_main: [f64; 4],
    p_mlp: [f64; 4],
    alpha: [f64; 4],
) -> Result<[f64; 4], SquillaRouterError> {
    for (i, (&p, &m)) in p_main.iter().zip(p_mlp.iter()).enumerate() {
        if !p.is_finite() || !m.is_finite() || !alpha[i].is_finite() {
            return Err(SquillaRouterError::InvalidProbabilities(format!(
                "fusion inputs must be finite (p_main={p}, p_mlp={m}, alpha={})",
                alpha[i]
            )));
        }
    }
    for &a in &alpha {
        if !(0.0..=1.0).contains(&a) {
            return Err(SquillaRouterError::InvalidProbabilities(format!(
                "alpha must stay within [0, 1], got {a}"
            )));
        }
    }
    let mixed: [f64; 4] =
        std::array::from_fn(|i| alpha[i] * p_main[i] + (1.0 - alpha[i]) * p_mlp[i]);
    let total: f64 = mixed.iter().sum();
    if total <= 0.0 {
        return Err(SquillaRouterError::InvalidProbabilities(
            "fused probability mass must be positive".to_string(),
        ));
    }
    Ok(std::array::from_fn(|i| mixed[i] / total))
}

/// Assemble the eight Phase 3 feature channels from a request and its
/// pre-embedded BGE vectors. Pure: the caller supplies `bge_vecs`.
pub fn assemble_channels(
    request: &FeatureInput,
    transforms: &TrainedTransforms,
    bge_vecs: &[Vec<f32>; 3],
) -> FeatureChannels {
    let hc = extract_handcrafted(&request.current_user_text);
    let tfidf = transforms.tfidf_svd_channel(&request.current_user_text);
    let ctx = extract_context_features(
        request
            .context_metadata
            .as_ref()
            .unwrap_or(&ContextMetadata::default()),
    );
    let hist = extract_hist_features(&request.prev_route_decisions);
    let bge_pca = transforms.bge_pca_channel(bge_vecs);
    // Python `features.py` falsy-coerces an empty assistant text to None before
    // extracting the assistant channel (`prev_assistant_text if ... else None`),
    // so an empty string must not be treated as a real assistant turn.
    let prev_assistant = request
        .prev_assistant_text
        .as_deref()
        .filter(|s| !s.is_empty());
    let asst_hc = extract_assistant_handcrafted(
        prev_assistant,
        request.prev_assistant_usage.as_ref(),
        &request.current_user_text,
    );
    let cont = extract_continuation_features(
        request.prev_assistant_usage.as_ref(),
        &request.current_user_text,
    );
    let reasoning = extract_reasoning_features(
        request.prev_assistant_usage.as_ref(),
        &request.current_user_text,
    );
    FeatureChannels {
        hc,
        tfidf,
        ctx,
        hist,
        bge_pca,
        asst_hc,
        cont,
        reasoning,
    }
}

/// Concatenate the eight channels into the fixed 390-dim feature vector in
/// Phase 3 order. The channel widths sum to 390, so the copy always stays in
/// bounds (no panics).
pub fn channels_to_390(channels: &FeatureChannels) -> [f64; FEATURE_390_DIM] {
    let mut out = [0.0; FEATURE_390_DIM];
    let mut i = 0;
    for src in [
        &channels.hc[..],
        &channels.tfidf[..],
        &channels.ctx[..],
        &channels.hist[..],
        &channels.bge_pca[..],
        &channels.asst_hc[..],
        &channels.cont[..],
        &channels.reasoning[..],
    ] {
        out[i..i + src.len()].copy_from_slice(src);
        i += src.len();
    }
    out
}

/// Concatenate the three 512-dim BGE segment vectors into the raw 1536-dim
/// MLP input, zero-padding any segment shorter than 512.
pub fn concat_bge_raw(bge_vecs: &[Vec<f32>; 3]) -> [f32; RAW_BGE_1536_DIM] {
    let mut out = [0.0f32; RAW_BGE_1536_DIM];
    for (seg, start) in bge_vecs.iter().zip([0usize, 512, 1024]) {
        let n = seg.len().min(512);
        out[start..start + n].copy_from_slice(&seg[..n]);
    }
    out
}

/// Run the post-process pipeline from pre-computed head outputs, fusing the
/// main/MLP heads with the manifest alpha weights.
pub fn route_from_heads(
    outputs: &HeadOutputs,
    config: &RouterConfig,
    manifest: &Manifest,
    request: &FeatureInput,
) -> Result<FinalDecision, SquillaRouterError> {
    let fused = fuse_per_class_alpha(outputs.p_main, outputs.p_mlp, manifest.per_class_alpha)?;
    apply_postprocess(fused, outputs.p_aux, request, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squilla_router::features::params::{
        BgePcaParams, ScalerParams, SvdParams, TfidfParams,
    };
    use std::collections::HashMap;

    fn empty_transforms() -> TrainedTransforms {
        TrainedTransforms {
            tfidf: TfidfParams {
                ngram_range: [2, 4],
                sublinear_tf: true,
                max_features: 10000,
                vocabulary: HashMap::new(),
                idf: Vec::new(),
            },
            svd: SvdParams {
                n_components: 0,
                n_features: 0,
                rows: Vec::new(),
                cols: Vec::new(),
                values: Vec::new(),
            },
            bge_pca: BgePcaParams {
                components: Vec::new(),
                mean: Vec::new(),
            },
            scaler: ScalerParams {
                mean: Vec::new(),
                scale: Vec::new(),
            },
        }
    }

    fn zero_bge_vecs() -> [Vec<f32>; 3] {
        [vec![0.0f32; 512], vec![0.0f32; 512], vec![0.0f32; 512]]
    }

    const MINIMAL_YAML: &str = r#"route_classes: [R0, R1, R2, R3]
tier_mapping: {R0: S, R1: M, R2: L, R3: XL}
tier_registry:
  S: [deepseek/deepseek-v4-flash]
  M: [deepseek/deepseek-v4-pro]
  L: [z-ai/glm-5.2]
  XL: [anthropic/claude-opus-4.8]
thresholds:
  margin_upgrade: 0.10
  high_confidence: 0.7
  r1_rescue: {from_r0_max_gap: 0.10}
  cascade_stage1_threshold: 0.4
  under_routing_safety: 0.45
  kv_cache_aware: true
flag_rules: {}
thinking_mode_rules:
  T0: {max_class: R0, min_margin: 0.5}
  T1: {max_class: R1, min_margin: 0.4}
  T2: {default: true}
  T3: {min_class: R2, flags: [debug, long_context, high_risk]}
prompt_policies:
  P0: {hint_zh: "", hint_en: "", conditions: {max_difficulty: 0.8, min_margin: 0.4, no_flags: [high_risk, strict_format, debug]}}
  P1: {hint_zh: "", hint_en: ""}
  P2: {hint_zh: "", hint_en: "", conditions: {any_flag: [high_risk, long_context, debug, strict_format]}}
context_rules:
  deep_conversation_threshold: 4
  heavy_context_tokens: 2000
  heavy_context_min_class: R1
  deep_conversation_min_class: R1
trajectory:
  delta_threshold: 0.3
  history_max_turns: 5
v4:
  aux_head_inference: false
  bge_model_name: BAAI/bge-small-zh-v1.5
  bge_backend: onnx
  bge_onnx_dir: bge_onnx
  pca_dim: 64
  feature_dim: 390
  history_user_max_turns: 4
  bge_truncate_tokens: 510
  aux_downgrade: {enabled: false, threshold: 0.55}
  sticky_tier: {enabled: false, max_user_len: 200}
"#;

    #[test]
    fn fuse_renormalizes_mixture() {
        let fused = fuse_per_class_alpha(
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        )
        .expect("valid inputs fuse");
        assert!((fused[0] - 0.5).abs() < 1e-9);
        assert!((fused[1] - 0.5).abs() < 1e-9);
        assert_eq!(fused[2], 0.0);
        assert!((fused.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fuse_rejects_out_of_range_alpha_and_non_finite() {
        let err = fuse_per_class_alpha([0.5; 4], [0.5; 4], [0.5, 1.5, 0.5, 0.5]).unwrap_err();
        assert!(matches!(err, SquillaRouterError::InvalidProbabilities(_)));
        let err = fuse_per_class_alpha([f64::NAN; 4], [0.5; 4], [0.5; 4]).unwrap_err();
        assert!(matches!(err, SquillaRouterError::InvalidProbabilities(_)));
    }

    #[test]
    fn channels_to_390_matches_bundle_order() {
        let channels = FeatureChannels {
            hc: [1.0; 51],
            tfidf: [2.0; 102],
            ctx: [3.0; 10],
            hist: [4.0; 16],
            bge_pca: [5.0; 192],
            asst_hc: [6.0; 12],
            cont: [7.0; 2],
            reasoning: [8.0; 5],
        };
        let v = channels_to_390(&channels);
        assert_eq!(v.len(), 390);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[51], 2.0);
        assert_eq!(v[51 + 102], 3.0);
        assert_eq!(v[51 + 102 + 10], 4.0);
        assert_eq!(v[51 + 102 + 10 + 16], 5.0);
        assert_eq!(v[51 + 102 + 10 + 16 + 192], 6.0);
        assert_eq!(v[383], 7.0); // cont
        assert_eq!(v[385], 8.0); // reasoning
    }

    #[test]
    fn concat_bge_raw_pads_short_segments() {
        let vecs = [vec![1.0f32; 512], vec![2.0f32; 512], vec![3.0f32; 512]];
        let raw = concat_bge_raw(&vecs);
        assert_eq!(raw.len(), 1536);
        assert_eq!(raw[0], 1.0);
        assert_eq!(raw[511], 1.0);
        assert_eq!(raw[512], 2.0);
        assert_eq!(raw[1023], 2.0);
        assert_eq!(raw[1024], 3.0);
        assert_eq!(raw[1535], 3.0);

        let short = concat_bge_raw(&[vec![7.0f32; 3], Vec::new(), vec![9.0f32; 1000]]);
        assert_eq!(short[0], 7.0);
        assert_eq!(short[3], 0.0);
        assert_eq!(short[512], 0.0);
        assert_eq!(short[1024], 9.0);
        assert_eq!(short[1535], 9.0); // 1000-len segment fills its 512 slot
    }

    #[test]
    fn assemble_channels_has_fixed_widths() {
        let channels = assemble_channels(
            &FeatureInput::default(),
            &empty_transforms(),
            &zero_bge_vecs(),
        );
        assert_eq!(channels.hc.len(), 51);
        assert_eq!(channels.tfidf.len(), 102);
        assert_eq!(channels.ctx.len(), 10);
        assert_eq!(channels.hist.len(), 16);
        assert_eq!(channels.bge_pca.len(), 192);
        assert_eq!(channels.asst_hc.len(), 12);
        assert_eq!(channels.cont.len(), 2);
        assert_eq!(channels.reasoning.len(), 5);
    }

    #[test]
    fn route_from_heads_derives_full_decision() {
        let config = RouterConfig::from_yaml_str(MINIMAL_YAML).expect("minimal yaml parses");
        let manifest = Manifest {
            temperature: 1.0,
            per_class_alpha: [0.5, 0.05, 0.5, 0.85],
        };
        let outputs = HeadOutputs {
            p_main: [0.9, 0.05, 0.03, 0.02],
            p_aux: None,
            p_mlp: [0.9, 0.05, 0.03, 0.02],
        };
        let d = route_from_heads(&outputs, &config, &manifest, &FeatureInput::default())
            .expect("route succeeds");
        assert_eq!(d.route_class, "R0");
        assert_eq!(d.tier, "S");
        assert_eq!(d.selected_model, "deepseek/deepseek-v4-flash");
        assert_eq!(d.thinking_mode, "T0");
        assert_eq!(d.prompt_policy, "P0");
        assert!((d.margin - 0.85).abs() < 1e-9);
        assert!((d.difficulty_score - 0.17).abs() < 1e-9);
    }
}
