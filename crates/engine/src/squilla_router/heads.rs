//! Phase-3 model heads: the LightGBM GBDT scorers (main/aux) and the ONNX MLP
//! head.
//!
//! Mirrors `runtime_src/src/router/inference/heads.py`. The GBDT forests are
//! deserialized from the flattened pre-order JSON schema exported by
//! `scripts/export_router_params.py` (`lgbm_main.json`, `lgbm_aux.json`); the
//! MLP head runs the raw 1536-dim BGE input through a `model.onnx` session and
//! is gated behind the `onnx` cargo feature.

use crate::squilla_router::SquillaRouterError;
use crate::squilla_router::features::matrix::TrainedTransforms;
use std::path::Path;

#[cfg(feature = "onnx")]
use crate::squilla_router::features::RAW_BGE_1536_DIM;

/// One LightGBM decision tree flattened by pre-order traversal.
///
/// All arrays have length = node count and are indexed by the unified node id
/// space produced by that traversal (visit -> next id -> recurse left then
/// right). Internal nodes carry a `split_feature`/`split_threshold` with
/// `left_child`/`right_child` node ids and a zeroed `leaf_value`; leaf nodes
/// have `split_feature`, `left_child`, `right_child` = -1, `split_threshold` =
/// 0.0, `default_left` = false, and a scalar raw score in `leaf_value`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LgbmTree {
    /// Feature index used at each internal node (-1 marks a leaf).
    split_feature: Vec<i32>,
    /// Split threshold at each internal node (0.0 for leaves).
    split_threshold: Vec<f64>,
    /// Left child node id in the pre-order id space; -1 for leaves.
    left_child: Vec<i32>,
    /// Right child node id in the pre-order id space; -1 for leaves.
    right_child: Vec<i32>,
    /// Missing-value branch hint. Kept for schema parity: the pipeline always
    /// supplies finite features, so it is never consulted during scoring.
    default_left: Vec<bool>,
    /// Per-node scalar raw score. Internal nodes are zeroed; a leaf's value is
    /// added to the class `tree_index % num_class`.
    leaf_value: Vec<f64>,
}

/// A LightGBM multiclass forest (4 classes) as exported to JSON.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LgbmForest {
    /// Number of classes (always 4 in the Phase 3 pipeline).
    num_class: usize,
    /// Total tree count (`trees.len()`); for multiclass this is
    /// `num_class * num_iterations`.
    num_iterations: usize,
    /// The component trees in round-robin class order; tree `i` contributes to
    /// class `i % num_class`.
    trees: Vec<LgbmTree>,
}

/// GBDT scorer over a 4-class LightGBM forest.
pub struct GbdtScorer {
    forest: LgbmForest,
}

impl GbdtScorer {
    /// Load a forest from a JSON file (schema above), mapping I/O and parse
    /// failures to `SquillaRouterError::InvalidParams`.
    pub fn from_json(path: &Path) -> Result<Self, SquillaRouterError> {
        let file = std::fs::File::open(path).map_err(|e| {
            SquillaRouterError::InvalidParams(format!("open {}: {e}", path.display()))
        })?;
        let forest: LgbmForest = serde_json::from_reader(file).map_err(|e| {
            SquillaRouterError::InvalidParams(format!("parse {}: {e}", path.display()))
        })?;
        Ok(Self { forest })
    }

    /// Sum each tree's leaf score into its class slot and softmax the four
    /// class sums into probabilities. Mirrors LightGBM's multiclass
    /// `predict_proba`: `dump_model` emits one scalar tree per class per
    /// iteration in round-robin order, so tree `t` contributes to class
    /// `t % num_class`; the class sums are then softmaxed.
    pub fn predict_proba(&self, features: &[f64]) -> [f64; 4] {
        let num_class = self.forest.num_class.max(1);
        let mut sums = [0.0f64; 4];
        for (t, tree) in self.forest.trees.iter().enumerate() {
            let mut node = 0usize;
            while tree.split_feature[node] != -1 {
                let f = tree.split_feature[node] as usize;
                // The Python reference rounds the assembled 390-dim bundle to
                // float32 before scoring (`features.py::build_feature_bundle`
                // `.astype(np.float32)`); LightGBM stores features as f32 and
                // compares against an f64 threshold. Match that rounding so
                // threshold comparisons see the same values.
                let feature = (features[f] as f32) as f64;
                // LightGBM `NumericalDecision` uses `fval <= threshold` to go
                // left, so a feature exactly equal to the threshold must take
                // the left child.
                let go_left = feature <= tree.split_threshold[node];
                node = if go_left {
                    tree.left_child[node] as usize
                } else {
                    tree.right_child[node] as usize
                };
            }
            let class = t % num_class;
            if class < 4 {
                sums[class] += tree.leaf_value[node];
            }
        }
        softmax_4(&sums)
    }
}

/// Numerically stable softmax over a 4-vector: subtracts the max before
/// exponentiating, matching `heads.py::_softmax`.
fn softmax_4(values: &[f64; 4]) -> [f64; 4] {
    let mut max = values[0];
    for &x in &values[1..] {
        max = max.max(x);
    }
    let exps: [f64; 4] = std::array::from_fn(|i| (values[i] - max).exp());
    let sum: f64 = exps.iter().sum();
    std::array::from_fn(|i| exps[i] / sum)
}

/// Temperature-scaled softmax of the MLP logits.
///
/// Requires `temperature` finite and > 0 (else `SquillaRouterError::InvalidParams`)
/// and exactly 4 logits (else `SquillaRouterError::InvalidProbabilities`).
/// Mirrors the calibration step in `heads.py::run_heads`.
pub fn softmax_calibrate(logits: &[f64], temperature: f64) -> Result<[f64; 4], SquillaRouterError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SquillaRouterError::InvalidParams(
            "temperature must be finite and > 0".to_string(),
        ));
    }
    if logits.len() != 4 {
        return Err(SquillaRouterError::InvalidProbabilities(format!(
            "expected 4 logits, got {}",
            logits.len()
        )));
    }
    let scaled = std::array::from_fn(|i| logits[i] / temperature);
    Ok(softmax_4(&scaled))
}

/// ONNX MLP head over the raw 1536-dim scaled BGE input (feature-gated).
#[cfg(feature = "onnx")]
pub struct MlpHead {
    session: std::sync::Arc<std::sync::Mutex<ort::session::Session>>,
}

#[cfg(feature = "onnx")]
impl MlpHead {
    /// Load `mlp_dir/model.onnx` into an ONNX session. A missing file maps to
    /// `MissingArtifact { artifact: "mlp/model.onnx", model_dir }`; session
    /// construction failures map to `SquillaRouterError::Onnx`.
    pub fn load(mlp_dir: &Path) -> Result<Self, SquillaRouterError> {
        let path = mlp_dir.join("model.onnx");
        if !path.exists() {
            return Err(SquillaRouterError::MissingArtifact {
                artifact: "mlp/model.onnx",
                model_dir: mlp_dir.to_path_buf(),
            });
        }
        let mut builder = ort::session::Session::builder()
            .map_err(|e| SquillaRouterError::Onnx(format!("create ONNX session builder: {e}")))?;
        let session = builder
            .commit_from_file(&path)
            .map_err(|e| SquillaRouterError::Onnx(format!("load {}: {e}", path.display())))?;
        Ok(Self {
            session: std::sync::Arc::new(std::sync::Mutex::new(session)),
        })
    }

    /// Run the MLP on a 1536-dim scaled BGE input and return the first output
    /// row as 4 logits. Rejects inputs that are not exactly 1536 dims.
    pub fn predict_logits(&self, scaled_input: &[f32]) -> Result<Vec<f64>, SquillaRouterError> {
        if scaled_input.len() != RAW_BGE_1536_DIM {
            return Err(SquillaRouterError::InvalidParams(format!(
                "MLP input must be {RAW_BGE_1536_DIM} dims, got {}",
                scaled_input.len()
            )));
        }
        let mut session = self
            .session
            .lock()
            .map_err(|e| SquillaRouterError::Onnx(format!("lock ONNX session: {e}")))?;
        let tensor =
            ort::value::Tensor::from_array(([1usize, RAW_BGE_1536_DIM], scaled_input.to_vec()))
                .map_err(|e| SquillaRouterError::Onnx(e.to_string()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| SquillaRouterError::Onnx(e.to_string()))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SquillaRouterError::Onnx(e.to_string()))?;
        let dims: &[i64] = shape;
        let n: usize = match dims {
            [1, n] | [1, _, n] => *n as usize,
            other => {
                return Err(SquillaRouterError::Onnx(format!(
                    "unexpected MLP output shape: {other:?}"
                )));
            }
        };
        let row: Vec<f64> = data[..n].iter().copied().map(|v| v as f64).collect();
        Ok(row)
    }
}

/// The three head outputs assembled by `run_heads`.
#[derive(Debug, Clone)]
pub struct HeadOutputs {
    /// Main LightGBM probabilities (4 classes).
    pub p_main: [f64; 4],
    /// Auxiliary LightGBM probabilities, when an aux forest is configured.
    pub p_aux: Option<[f64; 4]>,
    /// Temperature-calibrated MLP probabilities (4 classes).
    pub p_mlp: [f64; 4],
}

/// Run the main/aux GBDT scorers and the ONNX MLP head, returning the three
/// probability vectors. Requires the `onnx` cargo feature.
#[cfg(feature = "onnx")]
pub fn run_heads(
    features_390: &[f64; 390],
    raw_1536: &[f32; 1536],
    main: &GbdtScorer,
    aux: Option<&GbdtScorer>,
    mlp: &MlpHead,
    transforms: &TrainedTransforms,
    temperature: f64,
) -> Result<HeadOutputs, SquillaRouterError> {
    let p_main = main.predict_proba(features_390);
    let p_aux = aux.map(|a| a.predict_proba(features_390));
    let scaled = transforms.scaled_raw_bge(raw_1536);
    let logits = mlp.predict_logits(&scaled)?;
    let p_mlp = softmax_calibrate(&logits, temperature)?;
    Ok(HeadOutputs {
        p_main,
        p_aux,
        p_mlp,
    })
}

/// Stub for builds without the `onnx` feature: the MLP head is unavailable, so
/// `run_heads` always returns `SquillaRouterError::FeatureDisabled`.
#[cfg(not(feature = "onnx"))]
pub fn run_heads(
    _features_390: &[f64; 390],
    _raw_1536: &[f32; 1536],
    _main: &GbdtScorer,
    _aux: Option<&GbdtScorer>,
    _mlp: &std::marker::PhantomData<()>,
    _transforms: &TrainedTransforms,
    _temperature: f64,
) -> Result<HeadOutputs, SquillaRouterError> {
    Err(SquillaRouterError::FeatureDisabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_forest() -> LgbmForest {
        LgbmForest {
            num_class: 4,
            num_iterations: 1,
            trees: vec![LgbmTree {
                split_feature: vec![0, -1, -1],
                split_threshold: vec![0.5, 0.0, 0.0],
                left_child: vec![1, -1, -1],
                right_child: vec![2, -1, -1],
                default_left: vec![true, false, false],
                leaf_value: vec![0.0, 0.0, 0.9],
            }],
        }
    }

    fn argmax4(p: &[f64; 4]) -> usize {
        let mut best = 0usize;
        for i in 1..4 {
            if p[i] > p[best] {
                best = i;
            }
        }
        best
    }

    /// A 4-tree forest where tree `c` routes to the right leaf carrying score
    /// `scores[c]`, making the round-robin class assignment directly observable.
    fn forest_with_class_scores(scores: [f64; 4]) -> LgbmForest {
        LgbmForest {
            num_class: 4,
            num_iterations: 4,
            trees: (0..4)
                .map(|c| LgbmTree {
                    split_feature: vec![0, -1, -1],
                    split_threshold: vec![0.5, 0.0, 0.0],
                    left_child: vec![1, -1, -1],
                    right_child: vec![2, -1, -1],
                    default_left: vec![true, false, false],
                    leaf_value: vec![0.0, 0.0, scores[c]],
                })
                .collect(),
        }
    }

    #[test]
    fn lgbm_forest_serde_round_trip() {
        let forest = sample_forest();
        let json = serde_json::to_string(&forest).unwrap();
        let back: LgbmForest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.num_class, 4);
        assert_eq!(back.num_iterations, 1);
        assert_eq!(back.trees.len(), 1);
        assert_eq!(back.trees[0].split_feature, vec![0, -1, -1]);
        assert_eq!(back.trees[0].leaf_value[1], 0.0);
        assert_eq!(back.trees[0].leaf_value[2], 0.9);
        assert_eq!(back.trees[0].default_left, vec![true, false, false]);
    }

    #[test]
    fn predict_proba_walks_tree_and_softmaxes() {
        // A single tree contributes its leaf score to class `0 % 4 == 0`.
        let scorer = GbdtScorer {
            forest: sample_forest(),
        };
        // Feature 0 < 0.5 -> left leaf (score 0.0); >= 0.5 -> right leaf (0.9).
        let left = scorer.predict_proba(&[0.2, 0.0, 0.0, 0.0]);
        let right = scorer.predict_proba(&[0.8, 0.0, 0.0, 0.0]);
        assert!((left.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!((right.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert_eq!(argmax4(&left), 0);
        assert_eq!(argmax4(&right), 0); // all-zero scores -> uniform distribution
    }

    #[test]
    fn predict_proba_accumulates_round_robin_per_class() {
        // Tree 2's score lands in class 2, proving `t % num_class` routing.
        let scorer = GbdtScorer {
            forest: forest_with_class_scores([0.0, 0.0, 10.0, 0.0]),
        };
        let p = scorer.predict_proba(&[0.8, 0.0, 0.0, 0.0]);
        assert_eq!(argmax4(&p), 2);
        // Mixed scores across classes: class 0 and class 2 both accumulate.
        let scorer = GbdtScorer {
            forest: forest_with_class_scores([0.9, 0.0, 0.05, 0.0]),
        };
        let p = scorer.predict_proba(&[0.8, 0.0, 0.0, 0.0]);
        assert_eq!(argmax4(&p), 0);
        // Left branch (all-zero leaves) yields a uniform distribution.
        let p = scorer.predict_proba(&[0.2, 0.0, 0.0, 0.0]);
        assert!((p[0] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn predict_proba_threshold_equality_goes_left() {
        // LightGBM's `NumericalDecision` is `fval <= threshold` -> left, so a
        // feature exactly equal to the split threshold must take the left child
        // (here an all-zero leaf -> uniform distribution).
        let scorer = GbdtScorer {
            forest: forest_with_class_scores([0.0, 0.0, 10.0, 0.0]),
        };
        let p = scorer.predict_proba(&[0.5, 0.0, 0.0, 0.0]);
        assert!((p[2] - 0.25).abs() < 1e-9);
        // One ULP above 0.5 (the next f32) routes right -> class 2 dominates.
        let above = f32::from_bits(0.5f32.to_bits() + 1) as f64;
        let p = scorer.predict_proba(&[above, 0.0, 0.0, 0.0]);
        assert!(p[2] > 0.99);
    }

    #[test]
    fn softmax_calibrate_matches_known_values() {
        let p = softmax_calibrate(&[1.0, 2.0, 3.0, 4.0], 1.0).unwrap();
        let expected = [
            0.032058603280085,
            0.087144317688908,
            0.236882818089911,
            0.643914260941097,
        ];
        for (got, exp) in p.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-9);
        }
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn softmax_calibrate_rejects_bad_temperature() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                softmax_calibrate(&[1.0, 2.0, 3.0, 4.0], bad),
                Err(SquillaRouterError::InvalidParams(_))
            ));
        }
    }

    #[test]
    fn softmax_calibrate_rejects_wrong_length() {
        assert!(matches!(
            softmax_calibrate(&[1.0, 2.0, 3.0], 1.0),
            Err(SquillaRouterError::InvalidProbabilities(_))
        ));
    }

    #[cfg(not(feature = "onnx"))]
    fn empty_transforms() -> TrainedTransforms {
        use crate::squilla_router::features::params::{
            BgePcaParams, ScalerParams, SvdParams, TfidfParams,
        };
        TrainedTransforms {
            tfidf: TfidfParams {
                ngram_range: [2, 4],
                sublinear_tf: true,
                max_features: 10000,
                vocabulary: std::collections::HashMap::new(),
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

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn run_heads_disabled_without_onnx() {
        let scorer = GbdtScorer {
            forest: sample_forest(),
        };
        let err = run_heads(
            &[0.0; 390],
            &[0.0; 1536],
            &scorer,
            None,
            &std::marker::PhantomData,
            &empty_transforms(),
            1.0,
        );
        assert!(matches!(err, Err(SquillaRouterError::FeatureDisabled)));
    }
}
