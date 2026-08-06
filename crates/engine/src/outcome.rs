//! Normalized turn outcome taxonomy.
//!
//! Mirrors the Python backend's `engine/outcome.py`. A [`TurnOutcome`] is a
//! coarse, normalized classification of how a turn ended. The turn runner
//! finalizer maps raw error codes into this taxonomy so callers (and
//! observability) can react to `budgetLimited` / `partial` / `interrupted` /
//! `blocked` outcomes uniformly instead of matching individual provider error
//! strings.
//!
//! This module performs NO I/O; it only classifies and renders.

/// The coarse kind of a turn outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnOutcomeKind {
    /// The turn completed normally.
    Completed,
    /// The turn stopped early but produced usable work (e.g. output truncated).
    Partial,
    /// The turn was stopped because a token / cost budget was exhausted.
    BudgetLimited,
    /// The turn is blocked on an external dependency or policy decision.
    Blocked,
    /// The turn failed with an unrecoverable error.
    Failed,
    /// The turn was interrupted (cancelled, timed out, dropped).
    Interrupted,
}

impl TurnOutcomeKind {
    /// The wire-string spelling used by the Python runtime.
    ///
    /// Note that `BudgetLimited` renders in camelCase to match the Python
    /// `Literal` spelling exactly.
    pub fn as_str(self) -> &'static str {
        match self {
            TurnOutcomeKind::Completed => "completed",
            TurnOutcomeKind::Partial => "partial",
            TurnOutcomeKind::BudgetLimited => "budgetLimited",
            TurnOutcomeKind::Blocked => "blocked",
            TurnOutcomeKind::Failed => "failed",
            TurnOutcomeKind::Interrupted => "interrupted",
        }
    }
}

impl std::fmt::Display for TurnOutcomeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A normalized turn outcome.
///
/// Mirrors the Python `engine/outcome.TurnOutcome` frozen dataclass. `reason`
/// is always present (the normalized error code for failures); the optional
/// fields carry diagnostics when available.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnOutcome {
    /// The coarse kind of the outcome.
    pub kind: TurnOutcomeKind,
    /// The normalized reason code (e.g. `provider_request_too_large`).
    pub reason: String,
    /// The provider-side error class when one was reported.
    pub error_class: Option<String>,
    /// The human-readable error message when one was reported.
    pub error_message: Option<String>,
    /// Whether retrying the turn is likely to help.
    pub retryable: bool,
}

impl TurnOutcome {
    /// Render the outcome as a JSON object, omitting `None` fields.
    ///
    /// Mirrors `TurnOutcome.to_dict()`.
    pub fn to_dict(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("kind".into(), serde_json::Value::String(self.kind.as_str().into()));
        obj.insert("reason".into(), serde_json::Value::String(self.reason.clone()));
        if let Some(cls) = &self.error_class {
            obj.insert("error_class".into(), serde_json::Value::String(cls.clone()));
        }
        if let Some(msg) = &self.error_message {
            obj.insert("error_message".into(), serde_json::Value::String(msg.clone()));
        }
        // Python's `asdict(self)` keeps `retryable: False` (only `None` fields
        // are dropped), so the wire dict must always carry the flag — both
        // `true` and `false`.
        obj.insert("retryable".into(), serde_json::Value::Bool(self.retryable));
        serde_json::Value::Object(obj)
    }
}

/// A `completed` outcome with the given reason (defaults to `"done"`).
pub fn completed_outcome(reason: impl Into<String>) -> TurnOutcome {
    TurnOutcome {
        kind: TurnOutcomeKind::Completed,
        reason: reason.into(),
        error_class: None,
        error_message: None,
        retryable: false,
    }
}

/// Normalize a raw error code the way the Python runtime does: trim
/// whitespace, lowercase, and replace `-` with `_`.
pub fn normalize_code(value: Option<&str>) -> String {
    let text = value.unwrap_or("").trim().to_lowercase();
    text.replace('-', "_")
}

/// Classify a provider/tool error into a normalized [`TurnOutcome`].
///
/// Mirrors `engine/outcome.outcome_from_error`. The classification is driven
/// by the normalized code's membership in the budget / partial / interrupted /
/// blocked vocabularies; anything else becomes `Failed`.
pub fn outcome_from_error(
    code: Option<&str>,
    message: Option<&str>,
    error_class: Option<&str>,
) -> TurnOutcome {
    let normalized = normalize_code(code);
    let text = message.filter(|s| !s.is_empty());
    let cls = error_class.filter(|s| !s.is_empty()).map(str::to_string);

    if _BUDGET_CODES.contains(&normalized.as_str()) {
        return TurnOutcome {
            kind: TurnOutcomeKind::BudgetLimited,
            reason: normalized.clone(),
            error_class: cls.or_else(|| Some(normalized.clone())),
            error_message: text.map(str::to_string),
            retryable: true,
        };
    }
    if _PARTIAL_CODES.contains(&normalized.as_str()) {
        return TurnOutcome {
            kind: TurnOutcomeKind::Partial,
            reason: normalized.clone(),
            error_class: cls.or_else(|| Some(normalized.clone())),
            error_message: text.map(str::to_string),
            retryable: normalized == "provider_output_truncated",
        };
    }
    if _INTERRUPTED_CODES.contains(&normalized.as_str()) {
        return TurnOutcome {
            kind: TurnOutcomeKind::Interrupted,
            reason: normalized.clone(),
            error_class: cls.or_else(|| Some(normalized.clone())),
            error_message: text.map(str::to_string),
            retryable: true,
        };
    }
    if _BLOCKED_CODES.contains(&normalized.as_str()) {
        return TurnOutcome {
            kind: TurnOutcomeKind::Blocked,
            reason: normalized.clone(),
            error_class: cls.or_else(|| Some(normalized.clone())),
            error_message: text.map(str::to_string),
            retryable: true,
        };
    }
    TurnOutcome {
        kind: TurnOutcomeKind::Failed,
        reason: if normalized.is_empty() { "error".into() } else { normalized.clone() },
        error_class: cls.or_else(|| {
            if normalized.is_empty() {
                Some("error".into())
            } else {
                Some(normalized.clone())
            }
        }),
        error_message: text.map(str::to_string),
        retryable: false,
    }
}

/// Wrap an outcome in the `{"turn_outcome": ...}` details envelope used by
/// runtime diagnostics. Mirrors `engine/outcome.turn_outcome_details`.
pub fn turn_outcome_details(outcome: &TurnOutcome) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("turn_outcome".into(), outcome.to_dict());
    serde_json::Value::Object(obj)
}

/// Error codes classified as `budgetLimited`.
pub const BUDGET_CODES: &[&str] = &[
    "current_turn_context_exhausted",
    "provider_request_too_large",
    "provider_request_budget_exhausted",
    "provider_output_limit",
    "tool_run_budget_exhausted",
    "llm_budget_exhausted",
    "turn_llm_call_budget_exceeded",
    "turn_input_token_budget_exceeded",
    "turn_output_token_budget_exceeded",
    "turn_billed_cost_budget_exceeded",
];

/// Error codes classified as `partial`.
pub const PARTIAL_CODES: &[&str] = &[
    "max_iterations",
    "output_truncated",
    "provider_output_truncated",
    "turn_tool_error_budget_exceeded",
    "tool_failure_loop_exhausted",
];

/// Error codes classified as `interrupted`.
pub const INTERRUPTED_CODES: &[&str] = &[
    "cancelled",
    "cancelled_before_start",
    "dropped_by_overflow",
    "interrupted",
    "timeout",
    "iteration_timeout",
];

/// Error codes classified as `blocked`.
pub const BLOCKED_CODES: &[&str] = &[
    "human_decision_required",
    "approval_required",
    "external_dependency",
    "provider_unavailable",
    "usage_accounting_busy",
    "usage_accounting_unavailable",
    "sandbox_threshold_exceeded",
    "tool_policy_denied",
    "compaction_refused_flush_timeout",
    "compaction_refused_memory_flush",
    "compaction_refused_empty_summary",
    "context_unsalvageable",
];

const _BUDGET_CODES: &[&str] = BUDGET_CODES;
const _PARTIAL_CODES: &[&str] = PARTIAL_CODES;
const _INTERRUPTED_CODES: &[&str] = INTERRUPTED_CODES;
const _BLOCKED_CODES: &[&str] = BLOCKED_CODES;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completed_outcome_defaults_to_done() {
        let outcome = completed_outcome("done");
        assert_eq!(outcome.kind, TurnOutcomeKind::Completed);
        assert_eq!(outcome.reason, "done");
        assert_eq!(outcome.error_class, None);
        assert_eq!(outcome.retryable, false);
    }

    #[test]
    fn test_normalize_code_lowercases_and_replaces_dashes() {
        assert_eq!(normalize_code(Some("  Provider-Request-Too-Large ")), "provider_request_too_large");
        assert_eq!(normalize_code(None), "");
    }

    #[test]
    fn test_budget_code_is_budget_limited_and_retryable() {
        let outcome = outcome_from_error(Some("provider_request_too_large"), Some("msg"), None);
        assert_eq!(outcome.kind, TurnOutcomeKind::BudgetLimited);
        assert_eq!(outcome.reason, "provider_request_too_large");
        assert!(outcome.retryable);
        assert_eq!(outcome.error_message.as_deref(), Some("msg"));
        assert_eq!(outcome.error_class.as_deref(), Some("provider_request_too_large"));
    }

    #[test]
    fn test_partial_code_retryable_only_for_truncation() {
        let truncated = outcome_from_error(Some("provider_output_truncated"), None, None);
        assert_eq!(truncated.kind, TurnOutcomeKind::Partial);
        assert!(truncated.retryable);

        let max_iter = outcome_from_error(Some("max_iterations"), None, None);
        assert_eq!(max_iter.kind, TurnOutcomeKind::Partial);
        assert!(!max_iter.retryable);
    }

    #[test]
    fn test_interrupted_and_blocked_codes() {
        let interrupted = outcome_from_error(Some("timeout"), None, None);
        assert_eq!(interrupted.kind, TurnOutcomeKind::Interrupted);
        assert!(interrupted.retryable);

        let blocked = outcome_from_error(Some("approval_required"), None, None);
        assert_eq!(blocked.kind, TurnOutcomeKind::Blocked);
        assert!(blocked.retryable);
    }

    #[test]
    fn test_unknown_code_is_failed() {
        let outcome = outcome_from_error(Some("some_unknown_failure"), None, Some("MyError"));
        assert_eq!(outcome.kind, TurnOutcomeKind::Failed);
        assert_eq!(outcome.reason, "some_unknown_failure");
        assert_eq!(outcome.error_class.as_deref(), Some("MyError"));
        assert!(!outcome.retryable);
    }

    #[test]
    fn test_empty_code_is_failed_with_error_reason() {
        let outcome = outcome_from_error(None, None, None);
        assert_eq!(outcome.kind, TurnOutcomeKind::Failed);
        assert_eq!(outcome.reason, "error");
        assert_eq!(outcome.error_class.as_deref(), Some("error"));
    }

    #[test]
    fn test_to_dict_omits_none_fields() {
        let outcome = completed_outcome("done");
        let dict = outcome.to_dict();
        assert_eq!(dict["kind"], "completed");
        assert_eq!(dict["reason"], "done");
        assert!(dict.get("error_class").is_none());
        assert!(dict.get("error_message").is_none());
        // Python's `asdict` keeps `retryable: False`; the wire dict must match.
        assert_eq!(dict["retryable"], false);
    }

    #[test]
    fn test_to_dict_includes_retryable_true() {
        let outcome = outcome_from_error(Some("timeout"), None, None);
        let dict = outcome.to_dict();
        assert_eq!(dict["kind"], "interrupted");
        assert_eq!(dict["retryable"], true);
    }

    #[test]
    fn test_turn_outcome_details_envelope() {
        let outcome = outcome_from_error(Some("cancelled"), None, None);
        let details = turn_outcome_details(&outcome);
        assert_eq!(details["turn_outcome"]["kind"], "interrupted");
    }

    #[test]
    fn test_budget_limited_renders_camel_case() {
        let outcome = outcome_from_error(Some("llm_budget_exhausted"), None, None);
        assert_eq!(outcome.kind.as_str(), "budgetLimited");
    }
}
