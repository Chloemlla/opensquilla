//! Write-time source rehydration for Dream.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/rehydrate.py`. Before a
//! ranked candidate is promoted, its snippet must still exist verbatim (with a
//! matching sha256) inside the workspace source file.

use crate::dream::models::{PromotionCandidate, RehydrateResult};

/// Verify that a ranked candidate's snippet is still present in its workspace
/// source file and matches the recorded sha256.
///
/// TODO(parity): implement sha256 hashing, the workspace-containment guard and
/// the quarantine check from rehydrate.py.
pub fn rehydrate_candidate(
    workspace: &std::path::Path,
    candidate: &PromotionCandidate,
) -> RehydrateResult {
    let _ = (workspace, candidate);
    RehydrateResult {
        ok: false,
        reason: Some("TODO(parity): rehydrate_candidate not implemented".to_string()),
    }
}
