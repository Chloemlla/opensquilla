//! Deterministic promotion ranking for Dream.
//!
//! Parity port of `src/opensquilla/memory/dream/ranking.py`. Candidates are
//! scored from accumulated evidence (log-frequency, signal balance, source
//! confidence, source-day consolidation), gated on pure-negative recurrence,
//! and sorted deterministically by (score desc, signal total desc, id asc).

use crate::dream::models::{PromotionCandidate, PromotionEvidenceEntry, PromotionEvidenceStore};
use std::collections::HashMap;

/// Clamp a score into `[0.0, 1.0]`, mapping non-finite values to `0.0`.
fn clamp_score(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    value.clamp(0.0, 1.0)
}

/// Extract the signal-count map a candidate carries and sorts by.
fn signal_counts(entry: &PromotionEvidenceEntry) -> HashMap<String, i64> {
    HashMap::from([
        ("positive".to_string(), entry.positive_signal_count),
        ("correction".to_string(), entry.correction_signal_count),
        ("failure".to_string(), entry.failure_signal_count),
        ("manual".to_string(), entry.manual_signal_count),
    ])
}

/// True when there are correction/failure signals and no positive/manual ones.
fn is_pure_negative(entry: &PromotionEvidenceEntry) -> bool {
    let negative = entry.correction_signal_count + entry.failure_signal_count;
    let positive = entry.positive_signal_count + entry.manual_signal_count;
    negative > 0 && positive == 0
}

/// Deterministic score in `[0.0, 1.0]` for a single evidence entry.
fn score(entry: &PromotionEvidenceEntry) -> f64 {
    let frequency = clamp_score((entry.seen_count.max(0) as f64 + 1.0).ln() / (6.0_f64 + 1.0).ln());
    let positive_or_manual = entry.positive_signal_count + entry.manual_signal_count;
    let negative = entry.correction_signal_count + entry.failure_signal_count;
    let mut signal_balance = 0.55;
    if positive_or_manual > 0 {
        signal_balance += 0.3;
    }
    if entry.manual_signal_count > 0 {
        signal_balance += 0.1;
    }
    if negative > 0 && positive_or_manual == 0 {
        signal_balance -= 0.25;
        if negative > 1 {
            signal_balance += 0.25;
        }
    }
    let source_confidence = if entry.source_kind == "memory_file" { 0.75 } else { 0.5 };
    let consolidation = clamp_score(entry.source_days.len() as f64 / 3.0);
    clamp_score(
        0.35 * frequency
            + 0.30 * clamp_score(signal_balance)
            + 0.20 * source_confidence
            + 0.15 * consolidation,
    )
}

/// Rank promotion candidates from the evidence store by a deterministic score.
pub fn rank_promotion_candidates(
    store: &PromotionEvidenceStore,
    min_score: f64,
    negative_recurrence_threshold: i64,
    min_seen_count: i64,
    limit: Option<usize>,
) -> Vec<PromotionCandidate> {
    let mut ranked: Vec<PromotionCandidate> = Vec::new();
    for entry in store.entries.values() {
        if entry.status != "candidate" || entry.snippet.trim().is_empty() {
            continue;
        }
        if entry.seen_count < min_seen_count {
            continue;
        }
        let mut reasons: Vec<String> = Vec::new();
        if entry.positive_signal_count + entry.manual_signal_count > 0 {
            reasons.push("positive_or_manual_signal".to_string());
        }
        if is_pure_negative(entry) {
            if entry.seen_count < negative_recurrence_threshold {
                continue;
            }
            reasons.push("negative_recurrence".to_string());
        }
        if entry.seen_count > 1 {
            reasons.push(format!("seen_count={}", entry.seen_count));
        }
        let candidate_score = score(entry);
        if candidate_score < min_score {
            continue;
        }
        ranked.push(PromotionCandidate {
            candidate_id: entry.candidate_id.clone(),
            source_path: entry.source_path.clone(),
            snippet: entry.snippet.clone(),
            snippet_sha256: entry.snippet_sha256.clone(),
            claim_sha256: entry.claim_sha256.clone(),
            score: candidate_score,
            reasons,
            signal_counts: signal_counts(entry),
        });
    }

    ranked.sort_by(|a, b| {
        let a_total: i64 = a.signal_counts.values().sum();
        let b_total: i64 = b.signal_counts.values().sum();
        b.score
            .total_cmp(&a.score)
            .then_with(|| b_total.cmp(&a_total))
            .then_with(|| a.candidate_id.cmp(&b.candidate_id))
    });

    if let Some(limit) = limit {
        ranked.truncate(limit);
    }
    ranked
}
