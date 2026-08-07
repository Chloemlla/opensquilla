//! Shared Dream promotion data models.
//!
//! Parity port of `src/opensquilla/memory/dream/models.py`. These types back
//! the *evidence-gated promotion* pipeline (quarantine → evidence → ranking →
//! rehydrate → curated_apply → receipts), which is distinct from the
//! clustering/merging path implemented directly on [`crate::dream::DreamEngine`].
//! Struct definitions match `models.py` field-by-field; the persistence and
//! pipeline logic lives in `evidence`, `runner`, `curated_apply`, etc.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A raw candidate scanned from the workspace memory directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDreamCandidate {
    pub agent_id: String,
    pub source_path: String,
    pub source_kind: String,
    pub source_mtime_ns: i64,
    pub source_size: i64,
    pub snippet: String,
    pub snippet_sha256: String,
    pub claim_sha256: String,
    pub source_day: Option<String>,
    pub signal_kind: String,
}

/// Evidence accumulated for a single promotion candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionEvidenceEntry {
    pub candidate_id: String,
    pub agent_id: String,
    pub source_path: String,
    pub source_kind: String,
    pub source_mtime_ns: i64,
    pub source_size: i64,
    pub snippet: String,
    pub snippet_sha256: String,
    pub claim_sha256: String,
    pub first_seen_at: String,
    pub last_seen_at: String,
    pub seen_count: i64,
    pub positive_signal_count: i64,
    pub correction_signal_count: i64,
    pub failure_signal_count: i64,
    pub manual_signal_count: i64,
    pub source_days: Vec<String>,
    pub status: String,
    pub promoted_at: Option<String>,
    pub rejected_at: Option<String>,
    pub last_skip_reason: Option<String>,
}

/// The persisted evidence store (a map of candidate_id → entry).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionEvidenceStore {
    pub version: i64,
    pub updated_at: String,
    pub entries: HashMap<String, PromotionEvidenceEntry>,
}

impl Default for PromotionEvidenceStore {
    fn default() -> Self {
        Self {
            version: 1,
            updated_at: String::new(),
            entries: HashMap::new(),
        }
    }
}

/// A candidate that has been ranked for promotion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCandidate {
    pub candidate_id: String,
    pub source_path: String,
    pub snippet: String,
    pub snippet_sha256: String,
    pub claim_sha256: String,
    pub score: f64,
    pub reasons: Vec<String>,
    pub signal_counts: HashMap<String, i64>,
}

/// One operation in a curated MEMORY.md patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPatchOperation {
    pub op: String,
    pub candidate_ids: Vec<String>,
    pub section: String,
    pub memory_id: String,
    pub text: String,
    pub replaces_memory_id: Option<String>,
    pub replaces_memory_ids: Vec<String>,
    pub expected_old_text_sha256: Option<String>,
    pub reason: Option<String>,
}

/// A patch of operations against MEMORY.md.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromotionPatch {
    pub operations: Vec<PromotionPatchOperation>,
}

/// Outcome of applying a promotion patch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApplyPromotionResult {
    pub applied: i64,
    pub skipped: i64,
    pub changed: bool,
    pub applied_operations: Vec<serde_json::Value>,
}

/// Outcome of rehydrating a candidate against its source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RehydrateResult {
    pub ok: bool,
    pub reason: Option<String>,
}
