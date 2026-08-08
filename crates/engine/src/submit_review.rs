//! Pure, I/O-free state machine for the review-on-submit finalize checkpoint.
//!
//! Mirrors the Python backend's `engine/submit_review.py`. The agent loop owns
//! one `SubmitReviewState` per run, captures the workspace diff, and calls
//! these helpers to decide whether to show the model a review of its own
//! changes before its work is finalized.
//!
//! This module owns no I/O. Two entry points share one state:
//!
//! * **explicit** — the model calls the `submit` tool; `evaluate_explicit_submit`
//!   returns the action to take and mutates the state.
//! * **implicit** — the model stops emitting tool calls while holding a
//!   non-empty workspace diff; `should_fire_implicit` decides whether the
//!   review is injected as a user message before the existing finalize chain
//!   lets the turn end.

/// One checklist and one restatement per run, ever. Both bounds are hard.
pub const SUBMIT_REVIEW_STAGE_LIMIT: usize = 1;
/// Maximum anti-rubber-stamp nudges per run.
pub const SUBMIT_REVIEW_NUDGE_LIMIT: usize = 1;
/// Default cap for the diff body shown to the model.
pub const SUBMIT_REVIEW_DIFF_MAX_CHARS_DEFAULT: usize = 20_000;

/// Runtime strings shown to the model must never contain these tokens.
const FORBIDDEN_SUBSTRINGS: &[&str] = &["minimal", "localized", "not sufficient"];

/// Reserve room for the truncation marker when splitting a long diff head+tail.
const TRUNCATION_MARKER_RESERVE: usize = 160;

/// Outcome of an explicit `submit` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitAction {
    /// The diff is empty; the call is premature/exploratory.
    EmptyDiffNote,
    /// Show the review checklist.
    ShowChecklist,
    /// Remind the model to perform at least one checklist action.
    Nudge,
    /// Confirm the submission.
    Confirm,
}

impl SubmitAction {
    /// The wire-string spelling used by the Python runtime.
    pub fn as_str(self) -> &'static str {
        match self {
            SubmitAction::EmptyDiffNote => "empty_diff_note",
            SubmitAction::ShowChecklist => "show_checklist",
            SubmitAction::Nudge => "nudge",
            SubmitAction::Confirm => "confirm",
        }
    }
}

/// Per-run review progress. Owned by the agent loop; mutated in place.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubmitReviewState {
    /// 0 = unreviewed, 1 = reviewed, 2 = confirmed.
    pub stage: u8,
    /// `"explicit"` or `"implicit"`.
    pub reviewed_via: Option<String>,
    /// Number of anti-rubber-stamp nudges issued.
    pub nudges: usize,
    /// Whether the model executed a real (non-`submit`) tool since review.
    pub acted_since_review: bool,
}

impl SubmitReviewState {
    /// Advance to the reviewed stage exactly once.
    pub fn mark_reviewed(&mut self, via: impl Into<String>) {
        if self.stage < 1 {
            self.stage = 1;
            self.reviewed_via = Some(via.into());
            self.acted_since_review = false;
        }
    }
}

/// Record that the model executed a real (non-`submit`) tool.
///
/// Only meaningful once a review has been shown; it distinguishes a model that
/// kept working after seeing the checklist from one immediately re-submitting.
pub fn observe_tool_activity(state: &mut SubmitReviewState) {
    if state.stage >= 1 {
        state.acted_since_review = true;
    }
}

/// Decide what an explicit `submit` call should do, mutating `state`.
pub fn evaluate_explicit_submit(
    state: &mut SubmitReviewState,
    diff_empty: bool,
    headroom_ok: bool,
) -> SubmitAction {
    if diff_empty {
        // A premature/exploratory submit must not consume the run's one review.
        return SubmitAction::EmptyDiffNote;
    }
    if state.stage == 0 {
        if !headroom_ok {
            // No budget for a follow-up call: never strand the submission.
            state.stage = 2;
            return SubmitAction::Confirm;
        }
        state.mark_reviewed("explicit");
        return SubmitAction::ShowChecklist;
    }
    if state.stage == 1
        && !state.acted_since_review
        && state.nudges < SUBMIT_REVIEW_NUDGE_LIMIT
        && headroom_ok
    {
        state.nudges += 1;
        return SubmitAction::Nudge;
    }
    state.stage = 2;
    SubmitAction::Confirm
}

/// Whether the implicit finalize path should inject the review this attempt.
pub fn should_fire_implicit(
    state: &SubmitReviewState,
    enabled: bool,
    diff_empty: bool,
    headroom_ok: bool,
    other_gate_injected: bool,
    red_detected: bool,
    pending_flags_clear: bool,
) -> bool {
    enabled
        && state.stage == 0
        && !diff_empty
        && headroom_ok
        && !other_gate_injected
        && !red_detected
        && pending_flags_clear
}

/// Whether `build_submit_review_message` will truncate this diff.
pub fn diff_is_truncated(diff_text: &str, max_chars: usize) -> bool {
    max_chars > 0 && diff_text.chars().count() > max_chars
}

/// Head+tail truncation so both the first and last hunks stay visible.
pub fn truncate_diff(diff_text: &str, max_chars: usize) -> (String, bool) {
    let total = diff_text.chars().count();
    if !diff_is_truncated(diff_text, max_chars) {
        return (diff_text.to_string(), false);
    }
    let budget = max_chars.saturating_sub(TRUNCATION_MARKER_RESERVE);
    let head = budget / 2;
    let tail = budget - head;
    let shown = head + tail;
    let marker = format!(
        "[diff truncated: {shown} of {total} characters shown; the file list \
         above is complete — run git_diff for the rest]"
    );
    let head_txt: String = diff_text.chars().take(head).collect();
    let tail_txt: String = if tail > 0 {
        diff_text
            .chars()
            .rev()
            .take(tail)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    } else {
        String::new()
    };
    (format!("{head_txt}\n{marker}\n{tail_txt}"), true)
}

/// Render the review shown to the model (explicit result or injected user message).
pub fn build_submit_review_message(
    file_index: &str,
    diff_text: &str,
    implicit: bool,
    max_chars: usize,
) -> String {
    let (truncated_diff, _) = truncate_diff(diff_text, max_chars);
    let index = if file_index.trim().is_empty() {
        "(no per-file summary available)"
    } else {
        file_index
    };
    let mut lines = vec![
        "[Submit review]".to_string(),
        "Your current changes are shown below. Walk through this checklist before your work is finalized:".to_string(),
        "1. Search for other places in the codebase that use the code you changed — callers of the same function, copies of the same pattern, or related constants. If the same issue exists at another site, fix it there too.".to_string(),
        "2. Review each change in the diff. Keep every change your fix needs. Revert files you only changed to help yourself debug rather than to solve the task (for example: git checkout -- <file>). Do not revert changes that are part of your fix.".to_string(),
        "3. If you edited any code after your last verification run, run your verification command again now and confirm it passes.".to_string(),
        "4. Delete temporary or scratch files you created while working (scratch scripts, log dumps, notes), unless the task asked for them.".to_string(),
        "5. When every item checks out, call submit to confirm. Your workspace changes at that point become your final answer.".to_string(),
    ];
    if implicit {
        lines.push(
            "If you stop without calling submit after this review, your current changes will be submitted as they are.".to_string(),
        );
    }
    lines.push("Per-file summary of your changes:".to_string());
    lines.push(index.to_string());
    lines.push("Full diff:".to_string());
    lines.push("<diff>".to_string());
    lines.push(truncated_diff);
    lines.push("</diff>".to_string());
    lines.join("\n")
}

/// Message for an empty-diff explicit submit.
pub fn empty_diff_note() -> &'static str {
    "You have no workspace changes yet. Continue working and call submit when you have changes to finalize."
}

/// Anti-rubber-stamp nudge message.
pub fn nudge_message() -> &'static str {
    "Before confirming, complete at least one checklist action — for example, run your verification command — or state in your final answer why no item applies. Then call submit again."
}

/// Submission confirmation message.
pub fn confirmation_message() -> &'static str {
    "Submission received; running final checks."
}

/// Every template string shown to the model, with placeholder interpolants.
pub fn all_runtime_strings() -> Vec<String> {
    vec![
        empty_diff_note().to_string(),
        nudge_message().to_string(),
        confirmation_message().to_string(),
        build_submit_review_message("", "", false, SUBMIT_REVIEW_DIFF_MAX_CHARS_DEFAULT),
        build_submit_review_message("", "", true, SUBMIT_REVIEW_DIFF_MAX_CHARS_DEFAULT),
    ]
}

/// Raise an error if any authored runtime string carries a forbidden token.
pub fn assert_runtime_strings_clean() -> Result<(), String> {
    for text in all_runtime_strings() {
        let lowered = text.to_lowercase();
        for token in FORBIDDEN_SUBSTRINGS {
            if lowered.contains(token) {
                return Err(format!(
                    "forbidden token {token:?} present in submit-review runtime string"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_submit_empty_diff_does_not_consume_review() {
        let mut state = SubmitReviewState::default();
        assert_eq!(
            evaluate_explicit_submit(&mut state, true, true),
            SubmitAction::EmptyDiffNote
        );
        assert_eq!(state.stage, 0);
    }

    #[test]
    fn test_explicit_submit_first_time_shows_checklist() {
        let mut state = SubmitReviewState::default();
        assert_eq!(
            evaluate_explicit_submit(&mut state, false, true),
            SubmitAction::ShowChecklist
        );
        assert_eq!(state.stage, 1);
        assert_eq!(state.reviewed_via.as_deref(), Some("explicit"));
    }

    #[test]
    fn test_explicit_submit_no_headroom_confirms_immediately() {
        let mut state = SubmitReviewState::default();
        assert_eq!(
            evaluate_explicit_submit(&mut state, false, false),
            SubmitAction::Confirm
        );
        assert_eq!(state.stage, 2);
    }

    #[test]
    fn test_nudge_only_without_activity_once() {
        let mut state = SubmitReviewState::default();
        state.mark_reviewed("explicit");
        assert_eq!(
            evaluate_explicit_submit(&mut state, false, true),
            SubmitAction::Nudge
        );
        assert_eq!(state.nudges, 1);
        // Second identical submit confirms (nudge limit reached).
        assert_eq!(
            evaluate_explicit_submit(&mut state, false, true),
            SubmitAction::Confirm
        );
        assert_eq!(state.stage, 2);
    }

    #[test]
    fn test_activity_after_review_skips_nudge() {
        let mut state = SubmitReviewState::default();
        state.mark_reviewed("explicit");
        observe_tool_activity(&mut state);
        assert!(state.acted_since_review);
        assert_eq!(
            evaluate_explicit_submit(&mut state, false, true),
            SubmitAction::Confirm
        );
    }

    #[test]
    fn test_should_fire_implicit_gates() {
        let state = SubmitReviewState::default();
        assert!(should_fire_implicit(
            &state, true, false, true, false, false, true
        ));
        assert!(!should_fire_implicit(
            &state, false, false, true, false, false, true
        ));
        assert!(!should_fire_implicit(
            &state, true, true, true, false, false, true
        ));
        assert!(!should_fire_implicit(
            &state, true, false, false, false, false, true
        ));
        assert!(!should_fire_implicit(
            &state, true, false, true, true, false, true
        ));
        assert!(!should_fire_implicit(
            &state, true, false, true, false, true, true
        ));
        assert!(!should_fire_implicit(
            &state, true, false, true, false, false, false
        ));

        let mut reviewed = SubmitReviewState::default();
        reviewed.mark_reviewed("implicit");
        assert!(!should_fire_implicit(
            &reviewed, true, false, true, false, false, true
        ));
    }

    #[test]
    fn test_truncate_diff_keeps_head_and_tail() {
        let long = "a".repeat(1000);
        let (truncated, changed) = truncate_diff(&long, 500);
        assert!(changed);
        assert!(truncated.contains("[diff truncated:"));
        // Head + tail + marker fit under the cap.
        assert!(truncated.chars().count() <= 500);
        assert!(truncated.starts_with(&"a".repeat(170)));
        assert!(truncated.ends_with(&"a".repeat(170)));
    }

    #[test]
    fn test_truncate_diff_returns_unchanged_when_fits() {
        let (out, changed) = truncate_diff("short diff", 10_000);
        assert!(!changed);
        assert_eq!(out, "short diff");
    }

    #[test]
    fn test_truncate_diff_handles_unicode_boundaries() {
        let long = "é".repeat(1000);
        let (truncated, changed) = truncate_diff(&long, 300);
        assert!(changed);
        assert!(truncated.chars().count() <= 300);
    }

    #[test]
    fn test_build_message_structure() {
        let message = build_submit_review_message("a.py (+3)", "diff body", false, 10_000);
        assert!(message.starts_with("[Submit review]\n"));
        assert!(message.contains("a.py (+3)"));
        assert!(message.contains("<diff>\ndiff body\n</diff>"));
        // Explicit path does not include the implicit warning.
        assert!(!message.contains("If you stop without calling submit"));
    }

    #[test]
    fn test_build_message_implicit_appends_warning() {
        let message = build_submit_review_message("", "diff", true, 10_000);
        assert!(message.contains("If you stop without calling submit after this review"));
        assert!(message.contains("(no per-file summary available)"));
    }

    #[test]
    fn test_runtime_strings_are_clean() {
        assert_runtime_strings_clean().unwrap();
    }

    #[test]
    fn test_forbidden_tokens_not_present() {
        for text in all_runtime_strings() {
            let lowered = text.to_lowercase();
            for token in FORBIDDEN_SUBSTRINGS {
                assert!(!lowered.contains(token), "found {token:?} in: {text}");
            }
        }
    }
}
