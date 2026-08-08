//! Phase 3 feature assembly.
//!
//! The hand-crafted channels are pure ports of the Python feature functions
//! (`runtime_src/src/router/features.py`, `runtime_src/src/router/v4_features.py`).
//! The trained channels (`tfidf`, `bge_pca`) load JSON params exported by
//! `scripts/export_router_params.py`; see `matrix` and `params`.

pub mod assistant;
pub mod context;
pub mod continuation;
pub mod handcrafted;
pub mod history;
pub mod matrix;
pub mod params;
pub mod reasoning;
pub mod tfidf;

/// Channel widths in the Phase 3 bundle (`inference_manifest.json`).
pub const HC_DIMS: usize = 51;
pub const TFIDF_DIMS: usize = 102;
pub const CTX_DIMS: usize = 10;
pub const HIST_DIMS: usize = 16;
pub const BGE_PCA_DIMS: usize = 192;
pub const ASST_HC_DIMS: usize = 12;
pub const CONT_DIMS: usize = 2;
pub const REASONING_DIMS: usize = 5;
/// The total Phase 3 feature dimension.
pub const FEATURE_390_DIM: usize = 390;
/// Raw BGE input width for the MLP head (3 segments x 512).
pub const RAW_BGE_1536_DIM: usize = 1536;

/// The four Phase 3 route classes, in index order (R0 = easiest, R3 = hardest).
pub const ROUTE_CLASSES: [&str; 4] = ["R0", "R1", "R2", "R3"];

/// Map a route class label (`R0`..`R3`) to its 0-based index.
pub fn route_class_idx(class: &str) -> Option<usize> {
    ROUTE_CLASSES.iter().position(|&c| c == class)
}

/// Usage counters from the previous assistant turn (mirrors the Python
/// `prev_assistant_usage` dict).
#[derive(Debug, Clone, Default)]
pub struct PrevAssistantUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub reasoning_tokens: u64,
    pub duration_ms: u64,
}

/// A prior routing decision consumed by the history channel.
#[derive(Debug, Clone)]
pub struct PrevRouteDecision {
    /// `R0`..`R3`.
    pub route_class: String,
    pub difficulty: f64,
    pub margin: f64,
}

/// Session and tool context for the context channel (mirrors the Python
/// `ContextMetadata` dataclass).
#[derive(Debug, Clone, Default)]
pub struct ContextMetadata {
    pub turn_index: u32,
    pub context_tokens_est: u32,
    pub n_tools: u32,
    pub tool_result_length: u32,
    pub has_code_block: bool,
    pub has_file_reference: bool,
    pub has_url: bool,
    pub has_tool_results: bool,
}

/// All inputs to the Phase 3 pipeline (mirrors the Python `InferenceRequest`).
#[derive(Debug, Clone, Default)]
pub struct FeatureInput {
    pub current_user_text: String,
    pub history_user_texts: Vec<String>,
    pub prev_assistant_text: Option<String>,
    pub prev_assistant_usage: Option<PrevAssistantUsage>,
    pub prev_route_decisions: Vec<PrevRouteDecision>,
    pub flags_text_override: Option<String>,
    pub context_metadata: Option<ContextMetadata>,
}

/// The assembled feature channels in Phase 3 order (lengths fixed by the
/// `*_DIMS` constants).
#[derive(Debug, Clone)]
pub struct FeatureChannels {
    pub hc: [f64; HC_DIMS],
    pub tfidf: [f64; TFIDF_DIMS],
    pub ctx: [f64; CTX_DIMS],
    pub hist: [f64; HIST_DIMS],
    pub bge_pca: [f64; BGE_PCA_DIMS],
    pub asst_hc: [f64; ASST_HC_DIMS],
    pub cont: [f64; CONT_DIMS],
    pub reasoning: [f64; REASONING_DIMS],
}

impl FeatureChannels {
    /// Concatenate the channels into the 390-dim vector in the exact Phase 3
    /// order (`inference/features.py::build_feature_bundle`):
    /// hc, tfidf, ctx, hist, bge_pca, asst_hc, cont, reasoning.
    pub fn to_vec_390(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(FEATURE_390_DIM);
        out.extend_from_slice(&self.hc);
        out.extend_from_slice(&self.tfidf);
        out.extend_from_slice(&self.ctx);
        out.extend_from_slice(&self.hist);
        out.extend_from_slice(&self.bge_pca);
        out.extend_from_slice(&self.asst_hc);
        out.extend_from_slice(&self.cont);
        out.extend_from_slice(&self.reasoning);
        out
    }
}

/// The separator used to join prior user turns for the BGE history channel.
pub const HISTORY_USER_SEP: &str = "\n[SEP]\n";
/// Default max prior user turns joined into the history channel.
pub const HISTORY_USER_MAX_TURNS: usize = 4;
/// Default max chars of the joined history channel.
pub const HISTORY_USER_MAX_CHARS: usize = 1500;

/// Concatenate up to `max_turns` prior user turns (oldest to newest),
/// `[SEP]`-separated, dropping the oldest turns and hard-truncating from the
/// front when still over `max_chars`. Mirrors `v4_features.make_history_user_text`.
pub fn make_history_user_text(prior_user_turns: &[String]) -> String {
    make_history_user_text_with(
        prior_user_turns,
        HISTORY_USER_MAX_TURNS,
        HISTORY_USER_MAX_CHARS,
    )
}

/// `make_history_user_text` with explicit window and length bounds.
pub fn make_history_user_text_with(
    prior_user_turns: &[String],
    max_turns: usize,
    max_chars: usize,
) -> String {
    if prior_user_turns.is_empty() {
        return String::new();
    }
    let mut selected: Vec<&str> = prior_user_turns
        .iter()
        .skip(prior_user_turns.len().saturating_sub(max_turns))
        .map(|s| s.as_str())
        .collect();
    let mut text = selected.join(HISTORY_USER_SEP);
    while text.chars().count() > max_chars && selected.len() > 1 {
        selected.remove(0);
        text = selected.join(HISTORY_USER_SEP);
    }
    if text.chars().count() > max_chars {
        let start = text.chars().count() - max_chars;
        text = text.chars().skip(start).collect();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_vec_390_matches_phase3_order() {
        let ch = FeatureChannels {
            hc: [1.0; HC_DIMS],
            tfidf: [2.0; TFIDF_DIMS],
            ctx: [3.0; CTX_DIMS],
            hist: [4.0; HIST_DIMS],
            bge_pca: [5.0; BGE_PCA_DIMS],
            asst_hc: [6.0; ASST_HC_DIMS],
            cont: [7.0; CONT_DIMS],
            reasoning: [8.0; REASONING_DIMS],
        };
        let v = ch.to_vec_390();
        assert_eq!(v.len(), FEATURE_390_DIM);
        assert!(v[0] == 1.0 && v[HC_DIMS - 1] == 1.0);
        assert_eq!(v[HC_DIMS], 2.0);
        assert_eq!(v[HC_DIMS + TFIDF_DIMS], 3.0);
        assert_eq!(v[HC_DIMS + TFIDF_DIMS + CTX_DIMS], 4.0);
        assert_eq!(v[HC_DIMS + TFIDF_DIMS + CTX_DIMS + HIST_DIMS], 5.0);
        assert_eq!(
            v[HC_DIMS + TFIDF_DIMS + CTX_DIMS + HIST_DIMS + BGE_PCA_DIMS],
            6.0
        );
        assert_eq!(v[FEATURE_390_DIM - CONT_DIMS - REASONING_DIMS], 7.0);
        assert_eq!(v[FEATURE_390_DIM - REASONING_DIMS], 8.0);
    }

    #[test]
    fn history_user_text_drops_oldest_then_truncates() {
        let turns = vec![
            "aaa".to_string(),
            "bbb".to_string(),
            "ccc".to_string(),
            "ddd".to_string(),
            "eee".to_string(),
        ];
        // max_turns = 2 -> newest two only, oldest->newest order.
        let joined = make_history_user_text_with(&turns, 2, 100);
        assert_eq!(joined, "ddd\n[SEP]\neee");
        // small max_chars -> drop whole turns first; a lone remaining turn
        // under the cap is kept whole (no truncation needed).
        let tiny = make_history_user_text_with(&turns, 4, 5);
        assert_eq!(tiny, "eee");
        // a single turn still over the cap is hard front-truncated.
        let truncated = make_history_user_text_with(&["abcdefghijklmnop".to_string()], 4, 5);
        assert_eq!(truncated, "lmnop");
        assert_eq!(make_history_user_text_with(&[], 4, 1500), "");
    }
}
