//! Deterministic promotion ranking for Dream.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/ranking.py`. Candidates
//! are scored from accumulated evidence (frequency, signal balance, source
//! confidence, source-day consolidation) and sorted deterministically.

use crate::dream::models::{PromotionCandidate, PromotionEvidenceStore};

/// Rank promotion candidates from the evidence store by a deterministic score.
///
/// TODO(parity): implement the log-frequency / signal-balance / source
/// confidence scoring and pure-negative recurrence gating from ranking.py.
pub fn rank_promotion_candidates(
    store: &PromotionEvidenceStore,
    min_score: f64,
    negative_recurrence_threshold: i64,
    min_seen_count: i64,
    limit: Option<usize>,
) -> Vec<PromotionCandidate> {
    let _ = (
        store,
        min_score,
        negative_recurrence_threshold,
        min_seen_count,
        limit,
    );
    Vec::new()
}
