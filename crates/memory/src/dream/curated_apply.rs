//! Curated MEMORY.md writes for Dream.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/curated_apply.py`. The
//! apply step rewrites the workspace `MEMORY.md`, upserting or merging bullets
//! under `## Section` headings, and honors a dry-run flag.

use crate::dream::models::{ApplyPromotionResult, PromotionPatch};

/// Apply a promotion patch to the workspace `MEMORY.md`, honoring `dry_run`.
///
/// TODO(parity): implement section upsert / merge / skip operations from
/// curated_apply.py.
pub fn apply_promotion_patch(
    workspace: &std::path::Path,
    patch: &PromotionPatch,
    dry_run: bool,
) -> ApplyPromotionResult {
    let _ = (workspace, patch, dry_run);
    ApplyPromotionResult::default()
}
