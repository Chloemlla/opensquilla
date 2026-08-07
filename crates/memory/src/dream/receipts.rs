//! Dream receipt writer.
//!
//! Parity port of `src/opensquilla/memory/dream/receipts.py`. Each dream batch
//! emits a JSON receipt under `memory/.dream_receipts/<artifact>.json` with the
//! evidence + apply summary and a rollback plan. Keys are serialized in sorted
//! order (serde_json's default map) to match Python's `sort_keys=True`.

use crate::dream::models::{ApplyPromotionResult, PromotionCandidate};

/// Write a JSON dream receipt and return its workspace-relative path.
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
    let receipt_dir = workspace.join("memory").join(".dream_receipts");
    let _ = std::fs::create_dir_all(&receipt_dir);
    let receipt_path = receipt_dir.join(format!("{artifact_id}.json"));

    let ranked: Vec<serde_json::Value> = ranked_candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "candidate_id": c.candidate_id,
                "source_path": c.source_path,
                "score": c.score,
                "reasons": c.reasons,
            })
        })
        .collect();

    let payload = serde_json::json!({
        "schema_version": 1,
        "agent_id": agent_id,
        "dry_run": dry_run,
        "candidate_paths": candidate_paths,
        "evidence_updated": evidence_updated,
        "ranked_candidates": ranked,
        "skipped_candidates": skipped_candidates,
        "applied_promotions": applied.applied_operations,
        "memory_md_backup_path": memory_md_backup_path,
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "rollback": {
            "restore_memory_from": memory_md_backup_path,
            "reset_cursor_to": cursor_before,
        },
    });

    let mut text = serde_json::to_string_pretty(&payload).unwrap_or_default();
    text.push('\n');
    let _ = std::fs::write(&receipt_path, text);

    receipt_path
        .strip_prefix(workspace)
        .map(|rel| {
            rel.components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/")
        })
        .unwrap_or_default()
}
