//! Trained-channel matrix transforms (TruncatedSVD / PCA / StandardScaler)
//! and the `TrainedTransforms` facade that loads all four param files.

use super::params::{
    BgePcaParams, ScalerParams, SvdParams, TfidfParams, load_bge_pca, load_scaler, load_svd,
    load_tfidf,
};
use super::tfidf::tfidf_transform;
use super::{BGE_PCA_DIMS, TFIDF_DIMS};
use crate::squilla_router::SquillaRouterError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// TruncatedSVD.transform: y[k] = sum over (col,val) of val * components[k, col].
/// `components` are the COO `SvdParams`. Simple O(nonzeros * n_components) loop.
pub fn svd_project(sparse: &[(usize, f64)], params: &SvdParams) -> Vec<f64> {
    // Index the COO by feature column so each sparse entry touches only the
    // component rows it actually contributes to (<= n_components).
    let mut by_col: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for ((&row, &col), &val) in params.rows.iter().zip(&params.cols).zip(&params.values) {
        by_col.entry(col).or_default().push((row, val));
    }
    let mut y = vec![0.0; params.n_components];
    for &(col, val) in sparse {
        if let Some(entries) = by_col.get(&col) {
            for &(k, w) in entries {
                y[k] += val * w;
            }
        }
    }
    y
}

/// sklearn PCA.transform: y[k] = sum_j (x[j] - mean[j]) * components[k][j].
/// `x` is a 512-dim BGE embedding (f32). Returns n_components dims.
pub fn pca_project(x: &[f32], params: &BgePcaParams) -> Vec<f64> {
    params
        .components
        .iter()
        .map(|comp| {
            let mut s = 0.0;
            for (j, &m) in params.mean.iter().enumerate() {
                // Missing features are treated as zero-padded (xj = 0.0).
                let xj = x.get(j).copied().unwrap_or(0.0) as f64;
                s += (xj - m) * comp.get(j).copied().unwrap_or(0.0);
            }
            s
        })
        .collect()
}

/// StandardScaler.transform: z[i] = (x[i] - mean[i]) / scale[i]; a zero scale
/// yields 0.0 (stay finite).
pub fn scaler_transform(x: &[f32], params: &ScalerParams) -> Vec<f32> {
    params
        .mean
        .iter()
        .zip(&params.scale)
        .enumerate()
        .map(|(i, (&m, &s))| {
            let xi = x.get(i).copied().unwrap_or(0.0) as f64;
            if s == 0.0 { 0.0 } else { ((xi - m) / s) as f32 }
        })
        .collect()
}

/// Facade that loads all four param files from a params dir and exposes the
/// Phase-3 channel transforms.
pub struct TrainedTransforms {
    pub tfidf: TfidfParams,
    pub svd: SvdParams,
    pub bge_pca: BgePcaParams,
    pub scaler: ScalerParams,
}

impl TrainedTransforms {
    /// Loads params/tfidf.json, params/svd.json, params/bge_pca.json,
    /// params/scaler.json from `params_dir`. Any missing file ->
    /// SquillaRouterError::MissingArtifact { artifact, model_dir }.
    pub fn load(params_dir: &Path) -> Result<Self, SquillaRouterError> {
        let tfidf = load_tfidf(&require_file(params_dir, "tfidf.json")?)?;
        let svd = load_svd(&require_file(params_dir, "svd.json")?)?;
        let bge_pca = load_bge_pca(&require_file(params_dir, "bge_pca.json")?)?;
        let scaler = load_scaler(&require_file(params_dir, "scaler.json")?)?;
        Ok(Self {
            tfidf,
            svd,
            bge_pca,
            scaler,
        })
    }

    /// text -> tfidf_transform -> svd_project -> zero-padded/truncated to [f64; TFIDF_DIMS].
    pub fn tfidf_svd_channel(&self, text: &str) -> [f64; TFIDF_DIMS] {
        let sparse = tfidf_transform(text, &self.tfidf);
        let projected = svd_project(&sparse, &self.svd);
        let mut out = [0.0; TFIDF_DIMS];
        let n = projected.len().min(TFIDF_DIMS);
        out[..n].copy_from_slice(&projected[..n]);
        out
    }

    /// 3 x 512-dim BGE vectors -> pca_project each -> concatenated [f64; BGE_PCA_DIMS].
    /// A BGE vector shorter than expected is zero-padded.
    pub fn bge_pca_channel(&self, bge_vecs: &[Vec<f32>; 3]) -> [f64; BGE_PCA_DIMS] {
        let seg = BGE_PCA_DIMS / 3;
        let mut out = [0.0; BGE_PCA_DIMS];
        for (i, vec) in bge_vecs.iter().enumerate() {
            let projected = pca_project(vec, &self.bge_pca);
            let n = projected.len().min(seg);
            let start = i * seg;
            out[start..start + n].copy_from_slice(&projected[..n]);
        }
        out
    }

    /// 1536-dim raw BGE -> scaler_transform.
    pub fn scaled_raw_bge(&self, raw: &[f32]) -> Vec<f32> {
        scaler_transform(raw, &self.scaler)
    }
}

/// Resolve a single required artifact path, mapping absence to `MissingArtifact`.
fn require_file(dir: &Path, artifact: &'static str) -> Result<PathBuf, SquillaRouterError> {
    let path = dir.join(artifact);
    if path.is_file() {
        Ok(path)
    } else {
        Err(SquillaRouterError::MissingArtifact {
            artifact,
            model_dir: dir.to_path_buf(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained_transforms() -> TrainedTransforms {
        let mut vocabulary = HashMap::new();
        vocabulary.insert("ab".to_string(), 0usize);
        vocabulary.insert("bc".to_string(), 1usize);
        TrainedTransforms {
            tfidf: TfidfParams {
                ngram_range: [2, 2],
                sublinear_tf: true,
                max_features: 10,
                vocabulary,
                idf: vec![1.0, 1.0],
            },
            svd: SvdParams {
                n_components: 2,
                n_features: 2,
                rows: vec![0, 1],
                cols: vec![0, 1],
                values: vec![1.0, 1.0],
            },
            bge_pca: BgePcaParams {
                components: vec![vec![1.0; 512], vec![2.0; 512]],
                mean: vec![0.0; 512],
            },
            scaler: ScalerParams {
                mean: vec![0.0; 1536],
                scale: vec![1.0; 1536],
            },
        }
    }

    #[test]
    fn svd_project_identity_coo() {
        let params = SvdParams {
            n_components: 2,
            n_features: 3,
            rows: vec![0, 1, 0],
            cols: vec![0, 1, 2],
            values: vec![1.0, 1.0, 0.5],
        };
        let y = svd_project(&[(0, 2.0), (2, 4.0)], &params);
        assert_eq!(y, vec![4.0, 0.0]);
    }

    #[test]
    fn pca_project_one_component() {
        let params = BgePcaParams {
            components: vec![vec![1.0, 2.0]],
            mean: vec![1.0, 1.0],
        };
        assert_eq!(pca_project(&[3.0f32, 4.0], &params), vec![8.0]);
    }

    #[test]
    fn scaler_division_and_zero_scale() {
        let params = ScalerParams {
            mean: vec![1.0, 2.0],
            scale: vec![2.0, 0.0],
        };
        assert_eq!(scaler_transform(&[3.0f32, 5.0], &params), vec![1.0, 0.0]);
    }

    #[test]
    fn channels_have_fixed_widths() {
        let tt = trained_transforms();
        let tfidf_channel = tt.tfidf_svd_channel("ab bc");
        assert_eq!(tfidf_channel.len(), 102);
        assert!((tfidf_channel[0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
        assert!((tfidf_channel[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
        assert_eq!(tfidf_channel[2], 0.0);

        let bge = [vec![1.0f32; 512], vec![1.0f32; 512], vec![1.0f32; 512]];
        let bge_channel = tt.bge_pca_channel(&bge);
        assert_eq!(bge_channel.len(), 192);
        assert!((bge_channel[0] - 512.0).abs() < 1e-9);
        assert!((bge_channel[1] - 1024.0).abs() < 1e-9);
        assert!((bge_channel[64] - 512.0).abs() < 1e-9);
        assert!((bge_channel[128] - 512.0).abs() < 1e-9);
        assert_eq!(bge_channel[2], 0.0);
    }
}
