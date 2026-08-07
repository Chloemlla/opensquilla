//! Write-time source rehydration for Dream.
//!
//! Parity mirroring `src/opensquilla/memory/dream/rehydrate.py`. Before a ranked
//! candidate is promoted, its snippet must still exist verbatim (with a matching
//! sha256) inside the workspace source file.

use crate::dream::models::{PromotionCandidate, RehydrateResult};
use crate::dream::quarantine::is_quarantined_path;
use sha2::{Digest, Sha256};

/// Collapse all whitespace runs in `text` to single spaces.
fn normalize_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Verify that a ranked candidate's snippet is still present in its workspace
/// source file and matches the recorded sha256.
pub fn rehydrate_candidate(
    workspace: &std::path::Path,
    candidate: &PromotionCandidate,
) -> RehydrateResult {
    let source_rel = candidate
        .source_path
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_string();
    if is_quarantined_path(&source_rel) {
        return RehydrateResult {
            ok: false,
            reason: Some("source_quarantined".to_string()),
        };
    }

    // Guard the resolved source path against escaping the workspace root.
    let workspace_root =
        std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
    let source_path = std::fs::canonicalize(workspace.join(&source_rel))
        .unwrap_or_else(|_| workspace.join(&source_rel));
    if source_path.strip_prefix(&workspace_root).is_err() {
        return RehydrateResult {
            ok: false,
            reason: Some("source_outside_workspace".to_string()),
        };
    }

    let raw = match std::fs::read(&source_path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return RehydrateResult {
                ok: false,
                reason: Some("source_missing".to_string()),
            };
        }
        Err(_) => {
            return RehydrateResult {
                ok: false,
                reason: Some("source_unreadable".to_string()),
            };
        }
    };

    let normalized_snippet = normalize_text(&candidate.snippet);
    if !normalize_text(&raw).contains(&normalized_snippet) {
        return RehydrateResult {
            ok: false,
            reason: Some("snippet_missing".to_string()),
        };
    }

    let mut hasher = Sha256::new();
    hasher.update(candidate.snippet.as_bytes());
    if hex::encode(hasher.finalize()) != candidate.snippet_sha256 {
        return RehydrateResult {
            ok: false,
            reason: Some("hash_mismatch".to_string()),
        };
    }

    RehydrateResult {
        ok: true,
        reason: None,
    }
}
