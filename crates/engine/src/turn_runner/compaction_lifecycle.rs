//! Compaction lifecycle: durable receipts and persistence-gated compaction.
//!
//! Mirrors the Python `session/compaction_lifecycle.py` (durability enum,
//! receipt status, event chain, lifecycle result, timeout error) and
//! `engine/compaction_control.py` (the pure post-compaction continuation
//! gate). The gate consumes booleans produced by the receipt/flush checks
//! rather than validating receipts itself, so raw/session substrate ownership
//! stays with the session compaction lifecycle layer.
//!
//! See `compaction.rs` for the compaction stage that consumes these types.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Durability enum
// ---------------------------------------------------------------------------

/// How durable a compaction result is.
///
/// Mirrors the Python `CompactionDurability = Literal["durable",
/// "request_scoped", "none"]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionDurability {
    /// The compaction was persisted to durable storage.
    Durable,
    /// The compaction lives only for the duration of the request.
    RequestScoped,
    /// No durability guarantee.
    None,
}

impl fmt::Display for CompactionDurability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompactionDurability::Durable => f.write_str("durable"),
            CompactionDurability::RequestScoped => f.write_str("request_scoped"),
            CompactionDurability::None => f.write_str("none"),
        }
    }
}

impl Default for CompactionDurability {
    fn default() -> Self {
        CompactionDurability::None
    }
}

// ---------------------------------------------------------------------------
// Flush receipt status
// ---------------------------------------------------------------------------

/// The status of a flush receipt, as classified by
/// [`flush_receipt_status`].
///
/// Mirrors the return values of the Python `flush_receipt_status` function:
/// `"not_requested"`, `"safe"`, `"noop_no_memory"`, `"archive_only"`,
/// `"degraded_forensic"`, `"unsafe"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlushReceiptStatus {
    /// No flush was requested (receipt is absent).
    NotRequested,
    /// The receipt authorizes destructive compaction.
    Safe,
    /// The flush pipeline no-op'd because there was nothing durable to write.
    NoopNoMemory,
    /// Only an archive copy was written (no semantic memory).
    ArchiveOnly,
    /// A degraded forensic archive was written.
    DegradedForensic,
    /// The receipt does not authorize destructive compaction.
    Unsafe,
}

impl fmt::Display for FlushReceiptStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlushReceiptStatus::NotRequested => f.write_str("not_requested"),
            FlushReceiptStatus::Safe => f.write_str("safe"),
            FlushReceiptStatus::NoopNoMemory => f.write_str("noop_no_memory"),
            FlushReceiptStatus::ArchiveOnly => f.write_str("archive_only"),
            FlushReceiptStatus::DegradedForensic => f.write_str("degraded_forensic"),
            FlushReceiptStatus::Unsafe => f.write_str("unsafe"),
        }
    }
}

impl Default for FlushReceiptStatus {
    fn default() -> Self {
        FlushReceiptStatus::NotRequested
    }
}

// ---------------------------------------------------------------------------
// Flush receipt
// ---------------------------------------------------------------------------

/// A flush receipt — the durable proof that a flush pipeline completed.
///
/// The Python `flush_receipt` is opaque (`Any`); this struct captures the
/// fields the Python helpers consult via `_receipt_value`. Unknown fields are
/// kept in `extra` so receipts with additional provider-specific fields round
/// trip through serde without loss.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlushReceipt {
    /// The compaction id this receipt correlates to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_id: Option<String>,
    /// The classified status of the receipt.
    #[serde(default)]
    pub status: FlushReceiptStatus,
    /// The durability guarantee of the receipt.
    #[serde(default)]
    pub durability: CompactionDurability,
    /// The flush mode: `"llm"` authorizes destructive compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Number of indexed chunks written by the flush.
    #[serde(default)]
    pub indexed_chunk_count: i64,
    /// Integrity check status: `"ok"` is required for safety.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrity_status: Option<String>,
    /// Output coverage status: must be `"ok"` for safety.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_coverage_status: Option<String>,
    /// Number of invalid candidates encountered.
    #[serde(default)]
    pub invalid_candidate_count: i64,
    /// Candidate ids that were missing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_missing_ids: Vec<String>,
    /// Number of obligations the flush must satisfy.
    #[serde(default)]
    pub obligation_count: i64,
    /// Obligation ids that were missing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub obligation_missing_ids: Vec<String>,
    /// Obligation status: `"ok"` or `"backfilled"` is required when there are
    /// obligations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obligation_status: Option<String>,
    /// The result status string returned by the flush pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_status: Option<String>,
    /// Receipt scope: `"checkpoint"`, `"flush"`, `"preimage"`, `"repair"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Source/target path written by the flush.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    /// Target path written by the flush.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,
    /// Content hash of the written data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    /// Reason string (used by `repair` scope receipts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Paths flushed by the pipeline.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flushed_paths: Vec<String>,
    /// Additional provider-specific fields not modeled above.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl FlushReceipt {
    /// Create a new receipt with the given compaction id and status.
    pub fn new(compaction_id: impl Into<String>, status: FlushReceiptStatus) -> Self {
        Self {
            compaction_id: Some(compaction_id.into()),
            status,
            ..Default::default()
        }
    }

    /// Whether the receipt authorizes destructive compaction.
    ///
    /// Ports `flush_receipt_allows_destructive_compaction` from
    /// `compaction_lifecycle.py`.
    pub fn allows_destructive_compaction(&self) -> bool {
        if self.mode.as_deref() != Some("llm") {
            return false;
        }
        if self.indexed_chunk_count <= 0 {
            return false;
        }
        let integrity = self.integrity_status.as_deref().unwrap_or("unverified");
        if integrity != "ok" {
            return false;
        }
        let output_coverage = self
            .output_coverage_status
            .as_deref()
            .unwrap_or("unverified");
        if output_coverage != "ok" {
            return false;
        }
        if self.invalid_candidate_count > 0 {
            return false;
        }
        if !self.candidate_missing_ids.is_empty() {
            return false;
        }
        if self.obligation_count <= 0 && self.obligation_missing_ids.is_empty() {
            return true;
        }
        let obligation_status = self.obligation_status.as_deref().unwrap_or("unverified");
        if obligation_status != "ok" && obligation_status != "backfilled" {
            return false;
        }
        self.obligation_missing_ids.is_empty()
    }

    /// Whether the flush pipeline completed without needing retry.
    ///
    /// Ports `flush_receipt_is_successful_flush` from `compaction_lifecycle.py`.
    /// This is intentionally weaker than destructive-compaction safety.
    pub fn is_successful_flush(&self) -> bool {
        if self.allows_destructive_compaction() {
            return true;
        }
        matches!(self.result_status.as_deref(), Some("ok_noop_no_memory"))
    }

    /// Whether a durable receipt (checkpoint/flush/preimage/repair scope)
    /// authorizes destructive compaction.
    ///
    /// Ports `durable_receipt_allows_destructive_compaction` from
    /// `compaction_lifecycle.py`.
    pub fn durable_allows_destructive_compaction(&self) -> bool {
        let scope = self.scope.as_deref().unwrap_or("");
        let status = self.status;
        match scope {
            "checkpoint" => {
                status == FlushReceiptStatus::Safe
                    && self.source_path.as_deref().map_or(false, |s| !s.is_empty())
                    && self
                        .content_hash
                        .as_deref()
                        .map_or(false, |s| !s.is_empty())
            }
            "flush" => {
                self.target_path.as_deref().map_or(false, |s| !s.is_empty())
                    && self
                        .result_status
                        .as_deref()
                        .map_or(false, |s| s == "flush_appended")
            }
            "preimage" => {
                self.target_path
                    .as_deref()
                    .map_or(false, |s| s.starts_with("memory/.raw_fallbacks/"))
                    && self
                        .content_hash
                        .as_deref()
                        .map_or(false, |s| !s.is_empty())
                    && self
                        .result_status
                        .as_deref()
                        .map_or(false, |s| s == "preimage_saved")
            }
            "repair" => {
                let archived_reasons = [
                    "ok_archive_only",
                    "parse_failed_archived",
                    "provider_failed_archived",
                    "apply_failed_archived",
                ];
                self.target_path
                    .as_deref()
                    .map_or(false, |s| s.starts_with("memory/.raw_fallbacks/"))
                    && self
                        .content_hash
                        .as_deref()
                        .map_or(false, |s| !s.is_empty())
                    && self
                        .reason
                        .as_deref()
                        .map_or(false, |r| archived_reasons.contains(&r))
                    && self
                        .result_status
                        .as_deref()
                        .map_or(false, |s| s == "repair_pending")
            }
            _ => self.allows_destructive_compaction(),
        }
    }
}

// ---------------------------------------------------------------------------
// Lifecycle result
// ---------------------------------------------------------------------------

/// Configuration for the persistence gate on [`CompactionStage`].
///
/// When `enabled`, the stage consults [`decide_compaction_continuation`]
/// before applying compaction and skips when the gate rejects.
///
/// [`CompactionStage`]: super::compaction::CompactionStage
#[derive(Debug, Clone, Default)]
pub struct PersistenceGateConfig {
    /// Whether the persistence gate is enabled.
    pub enabled: bool,
    /// The flush receipt used to classify gate safety.
    pub receipt: Option<FlushReceipt>,
}

/// The outcome of a compaction lifecycle run.
///
/// Mirrors the Python `CompactionLifecycleResult` dataclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionLifecycleResult {
    /// Whether compaction was actually applied.
    #[serde(default)]
    pub compacted: bool,
    /// Whether compaction was refused by the lifecycle gate.
    #[serde(default)]
    pub refused: bool,
    /// The reason compaction was refused or skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Estimated tokens before compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_before: Option<u64>,
    /// Estimated tokens after compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_after: Option<u64>,
    /// Remaining budget tokens after compaction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining_budget_tokens: Option<u64>,
    /// Number of messages removed.
    #[serde(default)]
    pub removed_count: usize,
    /// Number of messages kept.
    #[serde(default)]
    pub kept_count: usize,
    /// Length of the generated summary.
    #[serde(default)]
    pub summary_len: usize,
    /// Source of the summary (`"unknown"` when not generated).
    #[serde(default = "default_summary_source")]
    pub summary_source: String,
    /// The flush receipt produced by the pre-compaction flush, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flush_receipt: Option<FlushReceipt>,
    /// The durability guarantee of the compaction result.
    #[serde(default)]
    pub durability: CompactionDurability,
    /// The lifecycle event chain completed by this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<String>,
}

fn default_summary_source() -> String {
    "unknown".to_string()
}

impl CompactionLifecycleResult {
    /// Create a refused result with the given reason.
    pub fn refused(reason: impl Into<String>) -> Self {
        Self {
            compacted: false,
            refused: true,
            reason: Some(reason.into()),
            ..Default::default()
        }
    }

    /// Create a compacted result with the given outcome metrics.
    pub fn compacted(
        tokens_before: u64,
        tokens_after: u64,
        removed: usize,
        kept: usize,
        durability: CompactionDurability,
    ) -> Self {
        Self {
            compacted: true,
            refused: false,
            tokens_before: Some(tokens_before),
            tokens_after: Some(tokens_after),
            removed_count: removed,
            kept_count: kept,
            durability,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Timeout error
// ---------------------------------------------------------------------------

/// A compaction operation exhausted its shared absolute deadline.
///
/// Mirrors the Python `CompactionTimeoutError`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("Compaction timed out during {phase}{detail}")]
pub struct CompactionTimeoutError {
    /// The phase that timed out.
    pub phase: String,
    /// The timeout duration in seconds, if known.
    pub timeout_seconds: Option<f64>,
    detail: String,
}

impl CompactionTimeoutError {
    /// Create a new timeout error for the given phase.
    pub fn new(phase: impl Into<String>, timeout_seconds: Option<f64>) -> Self {
        let phase = phase.into();
        let detail = match timeout_seconds {
            Some(t) if t > 0.0 => format!(" after {t:.1}s"),
            _ => String::new(),
        };
        Self {
            phase: if phase.is_empty() {
                "unknown".to_string()
            } else {
                phase
            },
            timeout_seconds,
            detail,
        }
    }
}

// ---------------------------------------------------------------------------
// Event chain
// ---------------------------------------------------------------------------

/// The lifecycle event name for a compaction trigger.
pub const COMPACTION_TRIGGERED_EVENT: &str = "compaction.triggered";
/// The lifecycle event name for a summarized chunk.
pub const COMPACTION_CHUNK_SUMMARIZED_EVENT: &str = "compaction.chunk_summarized";
/// The lifecycle event name for a verified summary.
pub const COMPACTION_SUMMARY_VERIFIED_EVENT: &str = "compaction.summary_verified";
/// The lifecycle event name for a persisted compaction.
pub const COMPACTION_PERSISTED_EVENT: &str = "compaction.persisted";
/// The lifecycle event name for a replayed compaction.
pub const COMPACTION_REPLAYED_EVENT: &str = "compaction.replayed";
/// The coverage status used before a persist/replay event confirms it.
pub const COMPACTION_COVERAGE_UNKNOWN: &str = "unknown";

/// Return the lifecycle events completed by the given telemetry event.
///
/// Ports `compaction_event_chain` from `compaction_lifecycle.py`.
pub fn compaction_event_chain(event: &str) -> Vec<String> {
    if event == COMPACTION_REPLAYED_EVENT {
        vec![
            COMPACTION_TRIGGERED_EVENT.to_string(),
            COMPACTION_CHUNK_SUMMARIZED_EVENT.to_string(),
            COMPACTION_SUMMARY_VERIFIED_EVENT.to_string(),
            COMPACTION_PERSISTED_EVENT.to_string(),
            COMPACTION_REPLAYED_EVENT.to_string(),
        ]
    } else if event == COMPACTION_PERSISTED_EVENT {
        vec![
            COMPACTION_TRIGGERED_EVENT.to_string(),
            COMPACTION_CHUNK_SUMMARIZED_EVENT.to_string(),
            COMPACTION_SUMMARY_VERIFIED_EVENT.to_string(),
            COMPACTION_PERSISTED_EVENT.to_string(),
        ]
    } else if event == COMPACTION_SUMMARY_VERIFIED_EVENT {
        vec![
            COMPACTION_TRIGGERED_EVENT.to_string(),
            COMPACTION_CHUNK_SUMMARIZED_EVENT.to_string(),
            COMPACTION_SUMMARY_VERIFIED_EVENT.to_string(),
        ]
    } else if event == COMPACTION_CHUNK_SUMMARIZED_EVENT {
        vec![
            COMPACTION_TRIGGERED_EVENT.to_string(),
            COMPACTION_CHUNK_SUMMARIZED_EVENT.to_string(),
        ]
    } else {
        vec![COMPACTION_TRIGGERED_EVENT.to_string()]
    }
}

// ---------------------------------------------------------------------------
// Receipt classification
// ---------------------------------------------------------------------------

/// Result statuses that indicate a no-op flush with no memory written.
pub const NOOP_FLUSH_RESULT_STATUSES: &[&str] = &["ok_noop_no_memory"];
/// Result statuses that indicate only an archive copy was written.
pub const ARCHIVE_ONLY_FLUSH_RESULT_STATUSES: &[&str] = &["ok_archive_only"];
/// Result statuses that indicate a degraded forensic archive was written.
pub const ARCHIVED_DEGRADED_FLUSH_RESULT_STATUSES: &[&str] = &[
    "parse_failed_archived",
    "provider_failed_archived",
    "apply_failed_archived",
];
/// Result statuses that indicate the flush failed.
pub const FAILED_FLUSH_RESULT_STATUSES: &[&str] = &["archive_failed"];

/// Classify a flush receipt into a status.
///
/// Ports `flush_receipt_status` from `compaction_lifecycle.py`.
pub fn flush_receipt_status(receipt: Option<&FlushReceipt>) -> FlushReceiptStatus {
    let Some(receipt) = receipt else {
        return FlushReceiptStatus::NotRequested;
    };
    if receipt.allows_destructive_compaction() {
        return FlushReceiptStatus::Safe;
    }
    let result_status = receipt.result_status.as_deref().unwrap_or("");
    if NOOP_FLUSH_RESULT_STATUSES.contains(&result_status) {
        return FlushReceiptStatus::NoopNoMemory;
    }
    if ARCHIVE_ONLY_FLUSH_RESULT_STATUSES.contains(&result_status) {
        return FlushReceiptStatus::ArchiveOnly;
    }
    if ARCHIVED_DEGRADED_FLUSH_RESULT_STATUSES.contains(&result_status) {
        return FlushReceiptStatus::DegradedForensic;
    }
    FlushReceiptStatus::Unsafe
}

/// Whether the receipt has archive evidence (content hash + a flushed path
/// under `memory/.raw_fallbacks/`).
fn receipt_has_archive_evidence(receipt: &FlushReceipt) -> bool {
    let has_hash = receipt
        .content_hash
        .as_deref()
        .map_or(false, |s| !s.is_empty());
    let has_fallback_path = receipt
        .flushed_paths
        .iter()
        .any(|p| p.starts_with("memory/.raw_fallbacks/"));
    has_hash && has_fallback_path
}

/// Whether compaction safety allows destructive compaction given a receipt.
///
/// Ports `compaction_safety_allows_destructive_compaction` from
/// `compaction_lifecycle.py`.
pub fn compaction_safety_allows_destructive_compaction(
    receipt: Option<&FlushReceipt>,
    deterministic_receipt_safe: bool,
) -> bool {
    if deterministic_receipt_safe {
        return true;
    }
    let Some(receipt) = receipt else {
        return false;
    };
    if receipt.allows_destructive_compaction() {
        return true;
    }
    let result_status = receipt.result_status.as_deref().unwrap_or("");
    let is_archive = ARCHIVE_ONLY_FLUSH_RESULT_STATUSES.contains(&result_status)
        || ARCHIVED_DEGRADED_FLUSH_RESULT_STATUSES.contains(&result_status);
    is_archive && receipt_has_archive_evidence(receipt)
}

// ---------------------------------------------------------------------------
// Continuation gate (ports engine/compaction_control.py)
// ---------------------------------------------------------------------------

/// The action a compaction continuation gate recommends.
///
/// Mirrors the Python `CompactionContinuationAction` Literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionContinuationAction {
    /// Compaction completed; continue the turn.
    ContinueAfterCompaction,
    /// Retry the compaction.
    RetryAfterCompaction,
    /// Continue in a degraded mode after compaction.
    DegradedContinueAfterCompaction,
    /// A partial compaction was applied; finalization still needed.
    PartialAfterCompaction,
    /// Compaction is blocked; the context is not safe to compact.
    BlockedAfterCompaction,
    /// Finalization failed after retries.
    FailedAfterCompaction,
}

impl fmt::Display for CompactionContinuationAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompactionContinuationAction::ContinueAfterCompaction => {
                f.write_str("continue_after_compaction")
            }
            CompactionContinuationAction::RetryAfterCompaction => {
                f.write_str("retry_after_compaction")
            }
            CompactionContinuationAction::DegradedContinueAfterCompaction => {
                f.write_str("degraded_continue_after_compaction")
            }
            CompactionContinuationAction::PartialAfterCompaction => {
                f.write_str("partial_after_compaction")
            }
            CompactionContinuationAction::BlockedAfterCompaction => {
                f.write_str("blocked_after_compaction")
            }
            CompactionContinuationAction::FailedAfterCompaction => {
                f.write_str("failed_after_compaction")
            }
        }
    }
}

/// The decision returned by the compaction continuation gate.
///
/// Mirrors the Python `CompactionContinuationDecision` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionContinuationDecision {
    /// The recommended action.
    pub action: CompactionContinuationAction,
    /// The human-readable reason for the decision.
    pub reason: String,
    /// Structured details about the decision inputs.
    #[serde(default)]
    pub details: BTreeMap<String, serde_json::Value>,
}

impl CompactionContinuationDecision {
    /// Returns `true` when the gate allows compaction to proceed.
    ///
    /// `ContinueAfterCompaction` and `DegradedContinueAfterCompaction` are
    /// considered proceed-able; everything else blocks or retries.
    pub fn may_proceed(&self) -> bool {
        matches!(
            self.action,
            CompactionContinuationAction::ContinueAfterCompaction
                | CompactionContinuationAction::DegradedContinueAfterCompaction
        )
    }

    /// Returns `true` when the gate recommends a retry.
    pub fn is_retry(&self) -> bool {
        matches!(
            self.action,
            CompactionContinuationAction::RetryAfterCompaction
        )
    }

    /// Returns `true` when the gate blocks compaction entirely.
    pub fn is_blocked(&self) -> bool {
        matches!(
            self.action,
            CompactionContinuationAction::BlockedAfterCompaction
        )
    }
}

/// Decide whether compaction should continue, degrade, finalize, or block.
///
/// Ports `decide_compaction_continuation` from `engine/compaction_control.py`
/// (lines 28-90). The helper consumes booleans from existing receipt/flush
/// checks instead of validating receipts itself, so raw/session substrate
/// ownership stays with the session compaction lifecycle layer.
///
/// # Decision table (faithful port)
///
/// | Condition | Action | Reason |
/// |-----------|--------|--------|
/// | `context_unsalvageable` OR NOT `receipt_safe` | `BlockedAfterCompaction` | `"context_unsalvageable"` |
/// | `prompt_changed` AND `semantic_flush_ok` | `ContinueAfterCompaction` | `"receipt_safe_prompt_changed"` |
/// | `retry_count < max_retries` | `RetryAfterCompaction` | `"prompt_not_reduced"` |
/// | `raw_session_durable` AND NOT `semantic_flush_ok` | `DegradedContinueAfterCompaction` | `"semantic_flush_degraded_raw_durable"` |
/// | `finalization_attempted` | `FailedAfterCompaction` | `"finalization_failed_after_retries"` |
/// | (otherwise) | `PartialAfterCompaction` | `"finalization_required_after_retries"` |
pub fn decide_compaction_continuation(
    receipt_safe: bool,
    raw_session_durable: bool,
    context_unsalvageable: bool,
    semantic_flush_ok: bool,
    retry_count: u32,
    max_retries: u32,
    prompt_changed: bool,
    finalization_attempted: bool,
) -> CompactionContinuationDecision {
    let details: BTreeMap<String, serde_json::Value> = [
        ("receipt_safe".to_string(), serde_json::json!(receipt_safe)),
        (
            "raw_session_durable".to_string(),
            serde_json::json!(raw_session_durable),
        ),
        (
            "semantic_flush_ok".to_string(),
            serde_json::json!(semantic_flush_ok),
        ),
        ("retry_count".to_string(), serde_json::json!(retry_count)),
        ("max_retries".to_string(), serde_json::json!(max_retries)),
        (
            "prompt_changed".to_string(),
            serde_json::json!(prompt_changed),
        ),
        (
            "finalization_attempted".to_string(),
            serde_json::json!(finalization_attempted),
        ),
        (
            "context_unsalvageable".to_string(),
            serde_json::json!(context_unsalvageable),
        ),
    ]
    .into_iter()
    .collect();

    if context_unsalvageable || !receipt_safe {
        return CompactionContinuationDecision {
            action: CompactionContinuationAction::BlockedAfterCompaction,
            reason: "context_unsalvageable".to_string(),
            details,
        };
    }
    if prompt_changed && semantic_flush_ok {
        return CompactionContinuationDecision {
            action: CompactionContinuationAction::ContinueAfterCompaction,
            reason: "receipt_safe_prompt_changed".to_string(),
            details,
        };
    }
    if retry_count < max_retries {
        return CompactionContinuationDecision {
            action: CompactionContinuationAction::RetryAfterCompaction,
            reason: "prompt_not_reduced".to_string(),
            details,
        };
    }
    if raw_session_durable && !semantic_flush_ok {
        return CompactionContinuationDecision {
            action: CompactionContinuationAction::DegradedContinueAfterCompaction,
            reason: "semantic_flush_degraded_raw_durable".to_string(),
            details,
        };
    }
    if finalization_attempted {
        return CompactionContinuationDecision {
            action: CompactionContinuationAction::FailedAfterCompaction,
            reason: "finalization_failed_after_retries".to_string(),
            details,
        };
    }
    CompactionContinuationDecision {
        action: CompactionContinuationAction::PartialAfterCompaction,
        reason: "finalization_required_after_retries".to_string(),
        details,
    }
}

/// Generate a new opaque compaction id.
///
/// Mirrors `new_compaction_id` from `compaction_lifecycle.py`.
pub fn new_compaction_id() -> String {
    format!("cmp_{}", uuid::Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_durability_serde_snake_case() {
        let json = serde_json::to_string(&CompactionDurability::RequestScoped).unwrap();
        assert_eq!(json, "\"request_scoped\"");
        let parsed: CompactionDurability = serde_json::from_str("\"durable\"").unwrap();
        assert_eq!(parsed, CompactionDurability::Durable);
    }

    #[test]
    fn test_flush_receipt_status_not_requested() {
        assert_eq!(flush_receipt_status(None), FlushReceiptStatus::NotRequested);
    }

    #[test]
    fn test_flush_receipt_status_safe() {
        let receipt = FlushReceipt {
            mode: Some("llm".to_string()),
            indexed_chunk_count: 5,
            integrity_status: Some("ok".to_string()),
            output_coverage_status: Some("ok".to_string()),
            obligation_count: 0,
            obligation_missing_ids: vec![],
            ..Default::default()
        };
        assert_eq!(
            flush_receipt_status(Some(&receipt)),
            FlushReceiptStatus::Safe
        );
    }

    #[test]
    fn test_flush_receipt_status_noop_no_memory() {
        let receipt = FlushReceipt {
            result_status: Some("ok_noop_no_memory".to_string()),
            ..Default::default()
        };
        assert_eq!(
            flush_receipt_status(Some(&receipt)),
            FlushReceiptStatus::NoopNoMemory
        );
    }

    #[test]
    fn test_flush_receipt_status_archive_only() {
        let receipt = FlushReceipt {
            result_status: Some("ok_archive_only".to_string()),
            ..Default::default()
        };
        assert_eq!(
            flush_receipt_status(Some(&receipt)),
            FlushReceiptStatus::ArchiveOnly
        );
    }

    #[test]
    fn test_flush_receipt_status_degraded_forensic() {
        let receipt = FlushReceipt {
            result_status: Some("parse_failed_archived".to_string()),
            ..Default::default()
        };
        assert_eq!(
            flush_receipt_status(Some(&receipt)),
            FlushReceiptStatus::DegradedForensic
        );
    }

    #[test]
    fn test_flush_receipt_status_unsafe() {
        let receipt = FlushReceipt {
            result_status: Some("archive_failed".to_string()),
            ..Default::default()
        };
        assert_eq!(
            flush_receipt_status(Some(&receipt)),
            FlushReceiptStatus::Unsafe
        );
    }

    #[test]
    fn test_event_chain_replayed() {
        let chain = compaction_event_chain(COMPACTION_REPLAYED_EVENT);
        assert_eq!(chain.len(), 5);
        assert_eq!(chain[0], COMPACTION_TRIGGERED_EVENT);
        assert_eq!(chain[4], COMPACTION_REPLAYED_EVENT);
    }

    #[test]
    fn test_event_chain_persisted() {
        let chain = compaction_event_chain(COMPACTION_PERSISTED_EVENT);
        assert_eq!(chain.len(), 4);
    }

    #[test]
    fn test_event_chain_chunk_summarized() {
        let chain = compaction_event_chain(COMPACTION_CHUNK_SUMMARIZED_EVENT);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_event_chain_unknown_defaults_to_triggered() {
        let chain = compaction_event_chain("unknown.event");
        assert_eq!(chain, vec![COMPACTION_TRIGGERED_EVENT.to_string()]);
    }

    #[test]
    fn test_decide_blocked_when_context_unsalvageable() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            true,  // raw_session_durable
            true,  // context_unsalvageable
            true,  // semantic_flush_ok
            0,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::BlockedAfterCompaction
        );
        assert_eq!(decision.reason, "context_unsalvageable");
    }

    #[test]
    fn test_decide_blocked_when_receipt_not_safe() {
        let decision = decide_compaction_continuation(
            false, // receipt_safe
            true,  // raw_session_durable
            false, // context_unsalvageable
            true,  // semantic_flush_ok
            0,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::BlockedAfterCompaction
        );
        assert_eq!(decision.reason, "context_unsalvageable");
    }

    #[test]
    fn test_decide_continue_when_prompt_changed_and_flush_ok() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            true,  // raw_session_durable
            false, // context_unsalvageable
            true,  // semantic_flush_ok
            0,     // retry_count
            3,     // max_retries
            true,  // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::ContinueAfterCompaction
        );
        assert_eq!(decision.reason, "receipt_safe_prompt_changed");
    }

    #[test]
    fn test_decide_retry_when_under_retry_limit() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            true,  // raw_session_durable
            false, // context_unsalvageable
            false, // semantic_flush_ok
            1,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::RetryAfterCompaction
        );
        assert_eq!(decision.reason, "prompt_not_reduced");
    }

    #[test]
    fn test_decide_degraded_when_raw_durable_and_flush_not_ok() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            true,  // raw_session_durable
            false, // context_unsalvageable
            false, // semantic_flush_ok
            3,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::DegradedContinueAfterCompaction
        );
        assert_eq!(decision.reason, "semantic_flush_degraded_raw_durable");
    }

    #[test]
    fn test_decide_failed_when_finalization_attempted() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            false, // raw_session_durable
            false, // context_unsalvageable
            false, // semantic_flush_ok
            3,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            true,  // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::FailedAfterCompaction
        );
        assert_eq!(decision.reason, "finalization_failed_after_retries");
    }

    #[test]
    fn test_decide_partial_when_nothing_else_matches() {
        let decision = decide_compaction_continuation(
            true,  // receipt_safe
            false, // raw_session_durable
            false, // context_unsalvageable
            false, // semantic_flush_ok
            3,     // retry_count
            3,     // max_retries
            false, // prompt_changed
            false, // finalization_attempted
        );
        assert_eq!(
            decision.action,
            CompactionContinuationAction::PartialAfterCompaction
        );
        assert_eq!(decision.reason, "finalization_required_after_retries");
    }

    #[test]
    fn test_decision_may_proceed() {
        let continue_decision = CompactionContinuationDecision {
            action: CompactionContinuationAction::ContinueAfterCompaction,
            reason: "ok".to_string(),
            details: BTreeMap::new(),
        };
        assert!(continue_decision.may_proceed());
        assert!(!continue_decision.is_retry());
        assert!(!continue_decision.is_blocked());

        let blocked_decision = CompactionContinuationDecision {
            action: CompactionContinuationAction::BlockedAfterCompaction,
            reason: "no".to_string(),
            details: BTreeMap::new(),
        };
        assert!(!blocked_decision.may_proceed());
        assert!(blocked_decision.is_blocked());
    }

    #[test]
    fn test_new_compaction_id_format() {
        let id = new_compaction_id();
        assert!(id.starts_with("cmp_"));
        assert!(id.len() > "cmp_".len());
    }

    #[test]
    fn test_timeout_error_message() {
        let err = CompactionTimeoutError::new("flush", Some(5.0));
        assert!(err.to_string().contains("flush"));
        assert!(err.to_string().contains("5.0s"));
    }

    #[test]
    fn test_timeout_error_no_timeout() {
        let err = CompactionTimeoutError::new("flush", None);
        assert!(err.to_string().contains("flush"));
        assert!(!err.to_string().contains("s"));
    }

    #[test]
    fn test_lifecycle_result_refused() {
        let result = CompactionLifecycleResult::refused("within_budget");
        assert!(result.refused);
        assert!(!result.compacted);
        assert_eq!(result.reason.as_deref(), Some("within_budget"));
    }

    #[test]
    fn test_lifecycle_result_compacted() {
        let result = CompactionLifecycleResult::compacted(
            100_000,
            30_000,
            50,
            10,
            CompactionDurability::Durable,
        );
        assert!(result.compacted);
        assert!(!result.refused);
        assert_eq!(result.tokens_before, Some(100_000));
        assert_eq!(result.tokens_after, Some(30_000));
        assert_eq!(result.removed_count, 50);
        assert_eq!(result.kept_count, 10);
        assert_eq!(result.durability, CompactionDurability::Durable);
    }

    #[test]
    fn test_receipt_allows_destructive_when_safe() {
        let receipt = FlushReceipt {
            mode: Some("llm".to_string()),
            indexed_chunk_count: 3,
            integrity_status: Some("ok".to_string()),
            output_coverage_status: Some("ok".to_string()),
            obligation_count: 0,
            obligation_missing_ids: vec![],
            ..Default::default()
        };
        assert!(receipt.allows_destructive_compaction());
    }

    #[test]
    fn test_receipt_does_not_allow_destructive_when_not_llm_mode() {
        let receipt = FlushReceipt {
            mode: Some("archive".to_string()),
            ..Default::default()
        };
        assert!(!receipt.allows_destructive_compaction());
    }

    #[test]
    fn test_receipt_does_not_allow_destructive_with_invalid_candidates() {
        let receipt = FlushReceipt {
            mode: Some("llm".to_string()),
            indexed_chunk_count: 3,
            integrity_status: Some("ok".to_string()),
            output_coverage_status: Some("ok".to_string()),
            invalid_candidate_count: 1,
            ..Default::default()
        };
        assert!(!receipt.allows_destructive_compaction());
    }
}
