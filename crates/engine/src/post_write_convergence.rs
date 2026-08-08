//! Post-write convergence tracking for coding-agent turns.
//!
//! Mirrors the Python backend's `engine/post_write_convergence.py`. Detects a
//! stable post-verification workspace diff that keeps consuming turns: once the
//! model has a verified, unchanged diff and keeps running activity, the tracker
//! first warns and then suggests finalizing.

/// The action a [`PostWriteConvergenceTracker`] decision requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostWriteConvergenceAction {
    /// No action; keep observing.
    Observe,
    /// Warn the model that it is re-verifying an unchanged diff.
    Warn,
    /// The diff has been stable past the warning; suggest finalizing.
    Finalize,
    /// The diff fingerprint changed; counters reset.
    Reset,
}

impl PostWriteConvergenceAction {
    /// The wire-string spelling used by the Python runtime.
    pub fn as_str(self) -> &'static str {
        match self {
            PostWriteConvergenceAction::Observe => "observe",
            PostWriteConvergenceAction::Warn => "warn",
            PostWriteConvergenceAction::Finalize => "finalize",
            PostWriteConvergenceAction::Reset => "reset",
        }
    }
}

/// One convergence observation fed into the tracker.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostWriteConvergenceObservation {
    /// The turn iteration index.
    pub iteration: i64,
    /// Number of provider calls so far.
    pub provider_call_count: i64,
    /// Number of workspace writes so far.
    pub workspace_write_count: i64,
    /// Number of changed workspace receipts so far.
    pub changed_receipt_count: i64,
    /// A fingerprint of the current workspace diff.
    pub diff_fingerprint: Option<String>,
    /// Paths present in the current diff.
    pub diff_paths: Vec<String>,
    /// Whether a focused verification run has been observed to succeed.
    pub focused_verification_success_observed: bool,
    /// Whether the model continued activity after the verification run.
    pub continued_activity_after_verification: bool,
}

/// A decision returned by [`PostWriteConvergenceTracker::observe`].
#[derive(Debug, Clone, PartialEq)]
pub struct PostWriteConvergenceDecision {
    /// The requested action.
    pub action: PostWriteConvergenceAction,
    /// A stable machine-readable reason.
    pub reason: String,
    /// Structured detail for diagnostics.
    pub details: serde_json::Value,
}

impl PostWriteConvergenceDecision {
    /// Render the decision as a JSON object.
    pub fn to_dict(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert(
            "action".into(),
            serde_json::Value::String(self.action.as_str().into()),
        );
        obj.insert(
            "reason".into(),
            serde_json::Value::String(self.reason.clone()),
        );
        obj.insert("details".into(), self.details.clone());
        serde_json::Value::Object(obj)
    }
}

/// Detects stable post-verification diffs that keep consuming turns.
#[derive(Debug, Clone)]
pub struct PostWriteConvergenceTracker {
    /// Number of stable observations before the first warning.
    pub warn_threshold: i64,
    /// Extra stable observations after the warning before finalizing.
    pub finalize_after_warning: i64,
    diff_fingerprint: Option<String>,
    stable_count: i64,
    warned_at_count: i64,
    finalized: bool,
}

impl Default for PostWriteConvergenceTracker {
    fn default() -> Self {
        Self {
            warn_threshold: 3,
            finalize_after_warning: 3,
            diff_fingerprint: None,
            stable_count: 0,
            warned_at_count: 0,
            finalized: false,
        }
    }
}

impl PostWriteConvergenceTracker {
    /// Create a tracker with the given thresholds.
    pub fn new(warn_threshold: i64, finalize_after_warning: i64) -> Self {
        Self {
            warn_threshold: warn_threshold.max(0),
            finalize_after_warning: finalize_after_warning.max(0),
            ..Default::default()
        }
    }

    /// Observe one convergence sample and return a decision.
    pub fn observe(
        &mut self,
        observation: &PostWriteConvergenceObservation,
    ) -> PostWriteConvergenceDecision {
        if !self.eligible(observation) {
            self.reset();
            return self.decision(
                PostWriteConvergenceAction::Observe,
                "not_eligible",
                observation,
                &[],
            );
        }

        if let Some(previous) = &self.diff_fingerprint {
            if observation.diff_fingerprint.as_deref() != Some(previous.as_str()) {
                let previous = previous.clone();
                self.diff_fingerprint = observation.diff_fingerprint.clone();
                self.stable_count = 1;
                self.warned_at_count = 0;
                self.finalized = false;
                return self.decision(
                    PostWriteConvergenceAction::Reset,
                    "diff_fingerprint_changed",
                    observation,
                    &[(
                        "previous_diff_fingerprint",
                        serde_json::Value::String(previous),
                    )],
                );
            }
        }

        self.diff_fingerprint = observation.diff_fingerprint.clone();
        self.stable_count += 1;

        if self.should_finalize() {
            self.finalized = true;
            return self.decision(
                PostWriteConvergenceAction::Finalize,
                "stable_verified_workspace_diff_finalization",
                observation,
                &[],
            );
        }
        if self.should_warn() {
            self.warned_at_count = self.stable_count;
            return self.decision(
                PostWriteConvergenceAction::Warn,
                "stable_verified_workspace_diff_continued_activity",
                observation,
                &[],
            );
        }
        self.decision(
            PostWriteConvergenceAction::Observe,
            "stable_verified_workspace_diff",
            observation,
            &[],
        )
    }

    /// Whether this observation qualifies for convergence tracking.
    pub fn eligible(&self, observation: &PostWriteConvergenceObservation) -> bool {
        observation.changed_receipt_count > 0
            && observation.diff_fingerprint.is_some()
            && !observation.diff_paths.is_empty()
            && observation.focused_verification_success_observed
            && observation.continued_activity_after_verification
    }

    fn should_warn(&self) -> bool {
        if self.warn_threshold <= 0 {
            return false;
        }
        if self.stable_count < self.warn_threshold {
            return false;
        }
        self.warned_at_count == 0
    }

    fn should_finalize(&self) -> bool {
        if self.finalized || self.warned_at_count <= 0 {
            return false;
        }
        let threshold = self.warned_at_count + self.finalize_after_warning;
        self.finalize_after_warning > 0 && self.stable_count >= threshold
    }

    fn reset(&mut self) {
        self.diff_fingerprint = None;
        self.stable_count = 0;
        self.warned_at_count = 0;
        self.finalized = false;
    }

    fn decision(
        &self,
        action: PostWriteConvergenceAction,
        reason: &str,
        observation: &PostWriteConvergenceObservation,
        extra: &[(&str, serde_json::Value)],
    ) -> PostWriteConvergenceDecision {
        let mut details = serde_json::Map::new();
        details.insert(
            "iteration".into(),
            serde_json::Value::from(observation.iteration),
        );
        details.insert(
            "provider_call_count".into(),
            serde_json::Value::from(observation.provider_call_count),
        );
        details.insert(
            "workspace_write_count".into(),
            serde_json::Value::from(observation.workspace_write_count),
        );
        details.insert(
            "changed_receipt_count".into(),
            serde_json::Value::from(observation.changed_receipt_count),
        );
        details.insert(
            "diff_fingerprint".into(),
            observation
                .diff_fingerprint
                .as_ref()
                .map(|fp| serde_json::Value::String(fp.clone()))
                .unwrap_or(serde_json::Value::Null),
        );
        details.insert(
            "diff_paths".into(),
            serde_json::Value::Array(
                observation
                    .diff_paths
                    .iter()
                    .map(|p| serde_json::Value::String(p.clone()))
                    .collect(),
            ),
        );
        details.insert(
            "stable_count".into(),
            serde_json::Value::from(self.stable_count),
        );
        details.insert(
            "warn_threshold".into(),
            serde_json::Value::from(self.warn_threshold),
        );
        details.insert(
            "finalize_after_warning".into(),
            serde_json::Value::from(self.finalize_after_warning),
        );
        details.insert(
            "warned_at_count".into(),
            serde_json::Value::from(self.warned_at_count),
        );
        for (key, value) in extra {
            details.insert((*key).to_string(), value.clone());
        }
        PostWriteConvergenceDecision {
            action,
            reason: reason.to_string(),
            details: serde_json::Value::Object(details),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stable_observation(fingerprint: &str) -> PostWriteConvergenceObservation {
        PostWriteConvergenceObservation {
            iteration: 1,
            provider_call_count: 2,
            workspace_write_count: 3,
            changed_receipt_count: 1,
            diff_fingerprint: Some(fingerprint.to_string()),
            diff_paths: vec!["src/lib.rs".to_string()],
            focused_verification_success_observed: true,
            continued_activity_after_verification: true,
        }
    }

    #[test]
    fn test_not_eligible_resets() {
        let mut tracker = PostWriteConvergenceTracker::default();
        let mut observation = stable_observation("fp");
        observation.diff_paths.clear();
        let decision = tracker.observe(&observation);
        assert_eq!(decision.action, PostWriteConvergenceAction::Observe);
        assert_eq!(decision.reason, "not_eligible");
        assert_eq!(tracker.diff_fingerprint, None);
    }

    #[test]
    fn test_warns_after_threshold_then_finalizes() {
        let mut tracker = PostWriteConvergenceTracker::new(3, 2);
        let obs = stable_observation("same");
        for _ in 0..2 {
            let decision = tracker.observe(&obs);
            assert_eq!(decision.action, PostWriteConvergenceAction::Observe);
        }
        let warn = tracker.observe(&obs);
        assert_eq!(warn.action, PostWriteConvergenceAction::Warn);
        assert_eq!(
            warn.reason,
            "stable_verified_workspace_diff_continued_activity"
        );

        // Two more stable observations finalize.
        let observe_again = tracker.observe(&obs);
        assert_eq!(observe_again.action, PostWriteConvergenceAction::Observe);
        let finalize = tracker.observe(&obs);
        assert_eq!(finalize.action, PostWriteConvergenceAction::Finalize);
        assert_eq!(
            finalize.reason,
            "stable_verified_workspace_diff_finalization"
        );
    }

    #[test]
    fn test_fingerprint_change_resets() {
        let mut tracker = PostWriteConvergenceTracker::new(3, 2);
        let decision = tracker.observe(&stable_observation("first"));
        assert_eq!(decision.action, PostWriteConvergenceAction::Observe);
        let reset = tracker.observe(&stable_observation("second"));
        assert_eq!(reset.action, PostWriteConvergenceAction::Reset);
        assert_eq!(reset.details["previous_diff_fingerprint"], "first");
        assert_eq!(reset.reason, "diff_fingerprint_changed");
    }

    #[test]
    fn test_finalize_never_fires_without_prior_warning() {
        // finalize_after_warning = 0 disables finalization.
        let mut tracker = PostWriteConvergenceTracker::new(1, 0);
        let obs = stable_observation("fp");
        let decision = tracker.observe(&obs);
        assert_eq!(decision.action, PostWriteConvergenceAction::Warn);
        let decision = tracker.observe(&obs);
        assert_eq!(decision.action, PostWriteConvergenceAction::Observe);
    }

    #[test]
    fn test_to_dict_shape() {
        let mut tracker = PostWriteConvergenceTracker::new(3, 2);
        let decision = tracker.observe(&stable_observation("fp"));
        let dict = decision.to_dict();
        assert_eq!(dict["action"], "observe");
        assert_eq!(dict["reason"], "stable_verified_workspace_diff");
        assert!(dict["details"].is_object());
    }
}
