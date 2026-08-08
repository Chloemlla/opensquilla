//! JSON schemas and loaders for the fitted feature-transform parameters.
//!
//! The Python export script (`scripts/export_router_params.py`) dumps the
//! fitted sklearn objects as plain JSON; these structs mirror that schema and
//! the loaders surface any I/O or parse failure as `InvalidParams`.

use crate::squilla_router::SquillaRouterError;
use std::path::Path;

/// Fitted `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
/// max_features=10000, sublinear_tf=True)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TfidfParams {
    /// `[min_n, max_n]` of the char n-grams (`[2, 4]`).
    pub ngram_range: [usize; 2],
    /// Whether `tf` is replaced by `1 + ln(tf)` (`true`).
    pub sublinear_tf: bool,
    /// Vocabulary size cap applied at fit time (`10000`).
    pub max_features: usize,
    /// n-gram token -> column id.
    pub vocabulary: std::collections::HashMap<String, usize>,
    /// Per-column inverse document frequency, `len == vocabulary.len()`.
    pub idf: Vec<f64>,
}

/// Fitted `TruncatedSVD` `components_` in COO form (n_components x n_features).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SvdParams {
    pub n_components: usize,
    pub n_features: usize,
    /// Row index of each COO entry (component axis).
    pub rows: Vec<usize>,
    /// Column index of each COO entry (feature axis).
    pub cols: Vec<usize>,
    /// Value of each COO entry.
    pub values: Vec<f64>,
}

/// Fitted `PCA(n_components=64)` for the 512-dim BGE embeddings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BgePcaParams {
    /// Component rows, `n_components` x 512.
    pub components: Vec<Vec<f64>>,
    /// Feature-wise mean, `len 512`.
    pub mean: Vec<f64>,
}

/// Fitted `StandardScaler` for the raw 1536-dim BGE MLP input.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalerParams {
    /// Feature-wise mean, `len 1536`.
    pub mean: Vec<f64>,
    /// Feature-wise scale (std), `len 1536`.
    pub scale: Vec<f64>,
}

/// Read and deserialize one JSON params file, mapping failures to `InvalidParams`.
fn load<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, SquillaRouterError> {
    let file = std::fs::File::open(path)
        .map_err(|e| SquillaRouterError::InvalidParams(format!("open {}: {e}", path.display())))?;
    serde_json::from_reader(file)
        .map_err(|e| SquillaRouterError::InvalidParams(format!("parse {}: {e}", path.display())))
}

pub fn load_tfidf(path: &Path) -> Result<TfidfParams, SquillaRouterError> {
    load(path)
}

pub fn load_svd(path: &Path) -> Result<SvdParams, SquillaRouterError> {
    load(path)
}

pub fn load_bge_pca(path: &Path) -> Result<BgePcaParams, SquillaRouterError> {
    load(path)
}

pub fn load_scaler(path: &Path) -> Result<ScalerParams, SquillaRouterError> {
    load(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tfidf_params() -> TfidfParams {
        let mut vocabulary = std::collections::HashMap::new();
        vocabulary.insert(" ab".to_string(), 0usize);
        TfidfParams {
            ngram_range: [2, 4],
            sublinear_tf: true,
            max_features: 10000,
            vocabulary,
            idf: vec![1.5],
        }
    }

    #[test]
    fn tfidf_params_serde_round_trip() {
        let json = serde_json::to_string(&tfidf_params()).unwrap();
        let back: TfidfParams = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ngram_range, [2, 4]);
        assert!(back.sublinear_tf);
        assert_eq!(back.max_features, 10000);
        assert_eq!(back.idf, vec![1.5]);
        assert_eq!(back.vocabulary.get(" ab"), Some(&0));
    }

    #[test]
    fn svd_params_serde_round_trip() {
        let p = SvdParams {
            n_components: 2,
            n_features: 3,
            rows: vec![0, 1],
            cols: vec![0, 1],
            values: vec![1.0, 1.0],
        };
        let back: SvdParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back.rows, vec![0, 1]);
        assert_eq!(back.cols, vec![0, 1]);
        assert_eq!(back.values, vec![1.0, 1.0]);
    }

    #[test]
    fn pca_and_scaler_serde_round_trip() {
        let pca = BgePcaParams {
            components: vec![vec![1.0, 0.0]],
            mean: vec![0.5, 0.5],
        };
        let back: BgePcaParams =
            serde_json::from_str(&serde_json::to_string(&pca).unwrap()).unwrap();
        assert_eq!(back.components, vec![vec![1.0, 0.0]]);
        assert_eq!(back.mean, vec![0.5, 0.5]);

        let sc = ScalerParams {
            mean: vec![1.0, 2.0],
            scale: vec![3.0, 4.0],
        };
        let back: ScalerParams =
            serde_json::from_str(&serde_json::to_string(&sc).unwrap()).unwrap();
        assert_eq!(back.mean, vec![1.0, 2.0]);
        assert_eq!(back.scale, vec![3.0, 4.0]);
    }
}
