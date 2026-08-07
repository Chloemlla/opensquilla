//! Dream receipt writer.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/receipts.py`. Each dream
//! batch emits a JSON receipt under `memory/.dream_receipts/<artifact>.json`
//! with the evidence + apply summary and a rollback plan.

use crate::dream::models::{ApplyPromotionResult, PromotionCandidate};

/// Write a JSON dream receipt and return its workspace-relative path.
///
/// TODO(parity): implement the receipt payload (schema_version 1, rollback
/// block, ranked/skipped candidates) from receipts.py.
#[allow(clippy::too_many_arguments)]
pub fn write_dream_receipt(
    workspace: &std::path::Path,
    artifact_id: &str,
    agent_id: &str,
    dry_run: bool,
    candidate_paths: &[String],
    evidence_updated: usize,
    ranked_candidates: &[PromotionCandidate],
    skipped_candidates: &[serde_json::Value],
    applied: &ApplyPromotionResult,
    memory_md_backup_path: &str,
    cursor_before: f64,
    cursor_after: f64,
) -> String {
    let _ = (
        workspace,
        artifact_id,
        agent_id,
        dry_run,
        candidate_paths,
        evidence_updated,
        ranked_candidates,
        skipped_candidates,
        applied,
        memory_md_backup_path,
        cursor_before,
        cursor_after,
    );
    String::new()
}
