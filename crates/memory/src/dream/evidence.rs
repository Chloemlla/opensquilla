//! Promotion evidence store for Dream.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/evidence.py`. The
//! evidence store accumulates per-candidate signal counts and persists to
//! `memory/.dream_state/promotion_evidence.json`.

use crate::dream::models::{PromotionEvidenceStore, RawDreamCandidate};

/// Path of the persisted promotion evidence store.
pub fn promotion_evidence_path(workspace: &std::path::Path) -> std::path::PathBuf {
    workspace
        .join("memory")
        .join(".dream_state")
        .join("promotion_evidence.json")
}

/// Load the promotion evidence store, returning a fresh store when missing or
/// corrupt.
///
/// TODO(parity): implement JSON loading / per-entry sanitization from
/// evidence.py.
pub fn load_evidence_store(workspace: &std::path::Path) -> PromotionEvidenceStore {
    let _ = workspace;
    PromotionEvidenceStore::default()
}

/// Persist the promotion evidence store atomically (tmp file + rename).
///
/// TODO(parity): implement the tmp-file + replace write from evidence.py.
pub fn write_evidence_store(
    workspace: &std::path::Path,
    store: &PromotionEvidenceStore,
) -> std::io::Result<()> {
    let _ = (workspace, store);
    Ok(())
}

/// Fold raw candidates into the evidence store, incrementing seen counts and
/// signal counters, and persist when `persist` is true.
///
/// TODO(parity): implement candidate-id derivation, source-day tracking and
/// signal incrementing from evidence.py.
pub fn update_promotion_evidence(
    workspace: &std::path::Path,
    candidates: &[RawDreamCandidate],
    now_iso: &str,
    persist: bool,
) -> PromotionEvidenceStore {
    let _ = (workspace, candidates, now_iso, persist);
    PromotionEvidenceStore::default()
}

/// Mark candidates as promoted in the evidence store.
///
/// TODO(parity): implement status mutation from evidence.py.
pub fn mark_evidence_promoted(
    store: &mut PromotionEvidenceStore,
    candidate_ids: &[String],
    now_iso: &str,
) {
    let _ = (store, candidate_ids, now_iso);
}

/// Record a skip reason for a candidate in the evidence store.
///
/// TODO(parity): implement from evidence.py.
pub fn mark_evidence_skipped(store: &mut PromotionEvidenceStore, candidate_id: &str, reason: &str) {
    let _ = (store, candidate_id, reason);
}

/// Mark candidates as represented (no curated change) in the evidence store.
///
/// TODO(parity): implement from evidence.py.
pub fn mark_evidence_represented(
    store: &mut PromotionEvidenceStore,
    candidate_ids: &[String],
    reason: &str,
) {
    let _ = (store, candidate_ids, reason);
}
