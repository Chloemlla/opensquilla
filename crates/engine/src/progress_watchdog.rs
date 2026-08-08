//! Observe-first progress watchdog for agent turns.
//!
//! Mirrors the Python backend's `engine/progress_watchdog.py`. Detects repeated
//! no-progress loops (repeated tool errors, repeated provider failures,
//! repeated failure anchors, source-context exploration without workspace
//! writes, stable post-verification activity, ...) without owning the main turn
//! loop. In the default `observe_only` mode decisions surface as `warn`; with
//! `observe_only = false` they escalate to `block`.

use serde_json::json;

/// The action a [`ProgressWatchdog`] decision requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// Keep observing; no action required.
    Observe,
    /// Surface a warning about the observed pattern.
    Warn,
    /// Block the turn loop because no-progress was detected.
    Block,
}

impl ProgressAction {
    /// The wire-string spelling used by the Python runtime.
    pub fn as_str(self) -> &'static str {
        match self {
            ProgressAction::Observe => "observe",
            ProgressAction::Warn => "warn",
            ProgressAction::Block => "block",
        }
    }
}

/// One observation fed into the watchdog.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProgressObservation {
    /// The turn iteration index.
    pub iteration: i64,
    /// Number of provider calls so far.
    pub provider_call_count: i64,
    /// Whether any tool call returned a successful result.
    pub successful_tool_result: bool,
    /// Whether a source-context tool returned a successful result.
    pub successful_source_context_tool_result: bool,
    /// Whether an execution tool returned a successful result.
    pub successful_execution_tool_result: bool,
    /// Signature identifying the source-context tool result.
    pub source_context_signature: Option<String>,
    /// Whether user-visible output was produced.
    pub user_visible_output: bool,
    /// Whether an artifact was completed.
    pub artifact_completed: bool,
    /// Whether a workspace change is likely required.
    pub workspace_change_likely_required: bool,
    /// Number of workspace writes.
    pub workspace_write_count: i64,
    /// Number of changed mutation receipts.
    pub changed_receipt_count: i64,
    /// Number of noop mutation receipts.
    pub noop_receipt_count: i64,
    /// Number of partial mutation receipts.
    pub partial_receipt_count: i64,
    /// Number of scratch writes.
    pub scratch_write_count: i64,
    /// Whether post-write focused verification was observed.
    pub post_write_focused_verification_observed: bool,
    /// Signature identifying a repeated tool error.
    pub tool_error_signature: Option<String>,
    /// Signature identifying a repeated provider failure.
    pub provider_failure_signature: Option<String>,
    /// Signature identifying a repeated failure anchor.
    pub failure_anchor_signature: Option<String>,
    /// Human-readable summary of the failure anchor.
    pub failure_anchor_summary: Option<String>,
}

/// A decision returned by [`ProgressWatchdog::observe`].
#[derive(Debug, Clone, PartialEq)]
pub struct ProgressDecision {
    /// The requested action.
    pub action: ProgressAction,
    /// A stable machine-readable reason.
    pub reason: String,
    /// Structured detail for diagnostics.
    pub details: serde_json::Value,
}

impl ProgressDecision {
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

/// Detect repeated no-progress loops without owning the main turn loop.
#[derive(Debug, Clone)]
pub struct ProgressWatchdog {
    /// Repeated tool-error signature count before a decision.
    pub repeated_tool_error_threshold: i64,
    /// Repeated provider-failure signature count before a decision.
    pub repeated_provider_failure_threshold: i64,
    /// Repeated failure-anchor signature count before a decision.
    pub repeated_failure_anchor_threshold: i64,
    /// Source-context reads without a workspace write before a decision.
    pub source_context_without_write_threshold: i64,
    /// Source-context explorations without a write before a decision.
    pub source_context_exploration_without_write_threshold: i64,
    /// Source-context reads after a write before a decision.
    pub source_context_after_write_threshold: i64,
    /// Tool activity without a write before a decision.
    pub tool_activity_without_write_threshold: i64,
    /// Verified post-write activity before a decision.
    pub verified_post_write_activity_threshold: i64,
    /// When true, decisions surface as `warn` instead of `block`.
    pub observe_only: bool,

    last_tool_error: Option<String>,
    tool_error_count: i64,
    last_provider_failure: Option<String>,
    provider_failure_count: i64,
    last_failure_anchor: Option<String>,
    failure_anchor_count: i64,
    failure_anchor_warned_at: i64,
    last_workspace_progress_count: i64,
    last_source_context_without_write_signature: Option<String>,
    source_context_without_write_count: i64,
    source_context_without_write_warned_at: i64,
    source_context_exploration_without_write_count: i64,
    source_context_exploration_without_write_warned_at: i64,
    source_context_after_write_count: i64,
    source_context_after_write_warned_at: i64,
    tool_activity_without_write_count: i64,
    tool_activity_without_write_warned_at: i64,
    verified_post_write_activity_count: i64,
    verified_post_write_activity_warned_at: i64,
}

/// Compute the workspace-progress signal for an observation.
pub fn workspace_progress_count(observation: &ProgressObservation) -> i64 {
    let changed_receipts = observation.changed_receipt_count.max(0);
    if workspace_receipt_count(observation) > 0 {
        return changed_receipts;
    }
    observation.workspace_write_count.max(0)
}

/// Total mutation-receipt count (changed + noop + partial).
pub fn workspace_receipt_count(observation: &ProgressObservation) -> i64 {
    observation.changed_receipt_count.max(0)
        + observation.noop_receipt_count.max(0)
        + observation.partial_receipt_count.max(0)
}

impl Default for ProgressWatchdog {
    fn default() -> Self {
        Self {
            repeated_tool_error_threshold: 3,
            repeated_provider_failure_threshold: 2,
            repeated_failure_anchor_threshold: 3,
            source_context_without_write_threshold: 8,
            source_context_exploration_without_write_threshold: 12,
            source_context_after_write_threshold: 8,
            tool_activity_without_write_threshold: 8,
            verified_post_write_activity_threshold: 3,
            observe_only: true,
            last_tool_error: None,
            tool_error_count: 0,
            last_provider_failure: None,
            provider_failure_count: 0,
            last_failure_anchor: None,
            failure_anchor_count: 0,
            failure_anchor_warned_at: 0,
            last_workspace_progress_count: 0,
            last_source_context_without_write_signature: None,
            source_context_without_write_count: 0,
            source_context_without_write_warned_at: 0,
            source_context_exploration_without_write_count: 0,
            source_context_exploration_without_write_warned_at: 0,
            source_context_after_write_count: 0,
            source_context_after_write_warned_at: 0,
            tool_activity_without_write_count: 0,
            tool_activity_without_write_warned_at: 0,
            verified_post_write_activity_count: 0,
            verified_post_write_activity_warned_at: 0,
        }
    }
}

impl ProgressWatchdog {
    /// Create a watchdog with the given threshold configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        repeated_tool_error_threshold: i64,
        repeated_provider_failure_threshold: i64,
        repeated_failure_anchor_threshold: i64,
        source_context_without_write_threshold: i64,
        source_context_exploration_without_write_threshold: i64,
        source_context_after_write_threshold: i64,
        tool_activity_without_write_threshold: i64,
        verified_post_write_activity_threshold: i64,
        observe_only: bool,
    ) -> Self {
        Self {
            repeated_tool_error_threshold,
            repeated_provider_failure_threshold,
            repeated_failure_anchor_threshold,
            source_context_without_write_threshold,
            source_context_exploration_without_write_threshold,
            source_context_after_write_threshold,
            tool_activity_without_write_threshold,
            verified_post_write_activity_threshold,
            observe_only,
            ..Default::default()
        }
    }

    /// Observe one turn iteration and return a decision.
    pub fn observe(&mut self, observation: &ProgressObservation) -> ProgressDecision {
        let workspace_progress_observed =
            self.sync_workspace_progress_count(workspace_progress_count(observation));

        if let Some(decision) = self.record_source_context_without_write(observation) {
            return decision;
        }
        if let Some(decision) = self.record_source_context_exploration_without_write(observation) {
            return decision;
        }
        if let Some(decision) = self.record_source_context_after_write(observation) {
            return decision;
        }
        if let Some(decision) = self.record_repeated_failure_anchor_without_write(observation) {
            return decision;
        }
        if let Some(decision) = self.record_tool_activity_without_write(observation) {
            return decision;
        }
        if let Some(decision) = self.record_verified_post_write_activity(observation) {
            return decision;
        }

        if has_progress(observation, workspace_progress_observed) {
            self.reset_progress_sensitive_counts();
            return self.decision(
                "progress",
                serde_json::Value::Object(serde_json::Map::new()),
                ProgressAction::Observe,
            );
        }

        if let Some(decision) = self.record_repeated_tool_error(observation) {
            return decision;
        }
        if let Some(decision) = self.record_repeated_provider_failure(observation) {
            return decision;
        }

        self.decision(
            "no_signal",
            serde_json::Value::Object(serde_json::Map::new()),
            ProgressAction::Observe,
        )
    }

    fn record_source_context_without_write(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        if observation.artifact_completed {
            self.reset_source_context_without_write_count();
            return None;
        }
        if !observation.successful_source_context_tool_result {
            return None;
        }
        if workspace_progress_count(observation) > 0 {
            return None;
        }

        let signature = observation
            .source_context_signature
            .clone()
            .unwrap_or_else(|| "<unknown>".to_string());
        if Some(signature.as_str()) == self.last_source_context_without_write_signature.as_deref() {
            self.source_context_without_write_count += 1;
        } else {
            self.last_source_context_without_write_signature = Some(signature.clone());
            self.source_context_without_write_count = 1;
            self.source_context_without_write_warned_at = 0;
        }
        let threshold = self.source_context_without_write_threshold.max(0);
        if threshold <= 0 || self.source_context_without_write_count < threshold {
            return None;
        }
        if self.source_context_without_write_warned_at != 0
            && self.source_context_without_write_count % threshold != 0
        {
            return None;
        }
        self.source_context_without_write_warned_at = self.source_context_without_write_count;
        let details = json!({
            "count": self.source_context_without_write_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "source_context_signature": signature,
            "workspace_change_likely_required": observation.workspace_change_likely_required,
        });
        Some(self.decision(
            "source_context_without_workspace_write",
            details,
            ProgressAction::Warn,
        ))
    }

    fn record_source_context_exploration_without_write(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        if observation.artifact_completed {
            self.reset_source_context_exploration_without_write_count();
            return None;
        }
        if !observation.successful_source_context_tool_result {
            return None;
        }
        if workspace_progress_count(observation) > 0 {
            self.reset_source_context_exploration_without_write_count();
            return None;
        }

        self.source_context_exploration_without_write_count += 1;
        let threshold = self
            .source_context_exploration_without_write_threshold
            .max(0);
        if threshold <= 0 || self.source_context_exploration_without_write_count < threshold {
            return None;
        }
        if self.source_context_exploration_without_write_warned_at != 0
            && self.source_context_exploration_without_write_count % threshold != 0
        {
            return None;
        }
        self.source_context_exploration_without_write_warned_at =
            self.source_context_exploration_without_write_count;
        let details = json!({
            "count": self.source_context_exploration_without_write_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "source_context_signature": observation.source_context_signature.clone().unwrap_or_else(|| "<unknown>".to_string()),
            "workspace_change_likely_required": observation.workspace_change_likely_required,
        });
        Some(self.decision(
            "source_context_exploration_without_workspace_write",
            details,
            ProgressAction::Warn,
        ))
    }

    fn record_source_context_after_write(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        if observation.artifact_completed {
            self.reset_source_context_after_write_count();
            return None;
        }
        if workspace_progress_count(observation) <= 0 {
            self.reset_source_context_after_write_count();
            return None;
        }
        if !observation.successful_source_context_tool_result {
            return None;
        }

        self.source_context_after_write_count += 1;
        let threshold = self.source_context_after_write_threshold.max(0);
        if threshold <= 0 || self.source_context_after_write_count < threshold {
            return None;
        }
        if self.source_context_after_write_warned_at != 0
            && self.source_context_after_write_count % threshold != 0
        {
            return None;
        }
        self.source_context_after_write_warned_at = self.source_context_after_write_count;
        let details = json!({
            "count": self.source_context_after_write_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "workspace_write_count": observation.workspace_write_count,
        });
        Some(self.decision(
            "source_context_after_workspace_write",
            details,
            ProgressAction::Warn,
        ))
    }

    fn record_repeated_failure_anchor_without_write(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        let signature = observation.failure_anchor_signature.clone()?;
        if Some(signature.as_str()) == self.last_failure_anchor.as_deref() {
            self.failure_anchor_count += 1;
        } else {
            self.last_failure_anchor = Some(signature.clone());
            self.failure_anchor_count = 1;
            self.failure_anchor_warned_at = 0;
        }

        let threshold = self.repeated_failure_anchor_threshold.max(0);
        if threshold <= 0 || self.failure_anchor_count < threshold {
            return None;
        }
        if self.failure_anchor_warned_at != 0 && self.failure_anchor_count % threshold != 0 {
            return None;
        }
        self.failure_anchor_warned_at = self.failure_anchor_count;
        let details = json!({
            "signature": signature,
            "count": self.failure_anchor_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "workspace_write_count": observation.workspace_write_count,
            "failure_anchor_summary": observation.failure_anchor_summary.clone().unwrap_or_default(),
            "workspace_change_likely_required": observation.workspace_change_likely_required,
        });
        Some(self.decision(
            "repeated_failure_anchor_without_workspace_write",
            details,
            ProgressAction::Warn,
        ))
    }

    fn record_tool_activity_without_write(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        if observation.artifact_completed {
            self.reset_tool_activity_without_write_count();
            return None;
        }
        if workspace_progress_count(observation) > 0 {
            self.reset_tool_activity_without_write_count();
            return None;
        }
        if !observation.successful_tool_result {
            return None;
        }
        if !observation.successful_execution_tool_result && observation.scratch_write_count <= 0 {
            return None;
        }

        self.tool_activity_without_write_count += 1;
        let threshold = self.tool_activity_without_write_threshold.max(0);
        if threshold <= 0 || self.tool_activity_without_write_count < threshold {
            return None;
        }
        if self.tool_activity_without_write_warned_at != 0
            && self.tool_activity_without_write_count % threshold != 0
        {
            return None;
        }
        self.tool_activity_without_write_warned_at = self.tool_activity_without_write_count;
        let details = json!({
            "count": self.tool_activity_without_write_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "scratch_write_count": observation.scratch_write_count,
            "successful_execution_tool_result": observation.successful_execution_tool_result,
            "workspace_change_likely_required": observation.workspace_change_likely_required,
        });
        Some(self.decision(
            "tool_activity_without_workspace_write",
            details,
            ProgressAction::Warn,
        ))
    }

    fn record_verified_post_write_activity(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        if observation.artifact_completed {
            self.reset_verified_post_write_activity_count();
            return None;
        }
        if workspace_progress_count(observation) <= 0 {
            self.reset_verified_post_write_activity_count();
            return None;
        }
        if !observation.post_write_focused_verification_observed {
            self.reset_verified_post_write_activity_count();
            return None;
        }
        if !observation.successful_tool_result {
            return None;
        }
        if !observation.successful_execution_tool_result
            && !observation.successful_source_context_tool_result
        {
            return None;
        }

        self.verified_post_write_activity_count += 1;
        let threshold = self.verified_post_write_activity_threshold.max(0);
        if threshold <= 0 || self.verified_post_write_activity_count < threshold {
            return None;
        }
        if self.verified_post_write_activity_warned_at != 0
            && self.verified_post_write_activity_count % threshold != 0
        {
            return None;
        }
        self.verified_post_write_activity_warned_at = self.verified_post_write_activity_count;
        let details = json!({
            "count": self.verified_post_write_activity_count,
            "threshold": threshold,
            "iteration": observation.iteration,
            "provider_call_count": observation.provider_call_count,
            "workspace_write_count": observation.workspace_write_count,
            "changed_receipt_count": observation.changed_receipt_count,
            "noop_receipt_count": observation.noop_receipt_count,
            "partial_receipt_count": observation.partial_receipt_count,
            "successful_execution_tool_result": observation.successful_execution_tool_result,
            "successful_source_context_tool_result": observation.successful_source_context_tool_result,
        });
        Some(self.decision(
            "verified_workspace_diff_continued_tool_activity",
            details,
            ProgressAction::Warn,
        ))
    }

    fn sync_workspace_progress_count(&mut self, workspace_progress_count: i64) -> bool {
        if workspace_progress_count > self.last_workspace_progress_count {
            self.last_workspace_progress_count = workspace_progress_count;
            self.reset_workspace_dependent_counts();
            return true;
        }
        if workspace_progress_count < self.last_workspace_progress_count {
            self.last_workspace_progress_count = workspace_progress_count;
            self.reset_workspace_dependent_counts();
        }
        false
    }

    fn record_repeated_tool_error(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        let signature = observation.tool_error_signature.clone()?;
        if Some(signature.as_str()) == self.last_tool_error.as_deref() {
            self.tool_error_count += 1;
        } else {
            self.last_tool_error = Some(signature.clone());
            self.tool_error_count = 1;
        }
        if self.tool_error_count < self.repeated_tool_error_threshold {
            return None;
        }
        let details = decision_details(observation, &signature, self.tool_error_count);
        Some(self.decision("repeated_tool_error", details, ProgressAction::Warn))
    }

    fn record_repeated_provider_failure(
        &mut self,
        observation: &ProgressObservation,
    ) -> Option<ProgressDecision> {
        let signature = observation.provider_failure_signature.clone()?;
        if Some(signature.as_str()) == self.last_provider_failure.as_deref() {
            self.provider_failure_count += 1;
        } else {
            self.last_provider_failure = Some(signature.clone());
            self.provider_failure_count = 1;
        }
        if self.provider_failure_count < self.repeated_provider_failure_threshold {
            return None;
        }
        let details = decision_details(observation, &signature, self.provider_failure_count);
        Some(self.decision("repeated_provider_failure", details, ProgressAction::Warn))
    }

    fn decision(
        &self,
        reason: &str,
        details: serde_json::Value,
        action: ProgressAction,
    ) -> ProgressDecision {
        ProgressDecision {
            action: if action == ProgressAction::Observe {
                ProgressAction::Observe
            } else if self.observe_only {
                ProgressAction::Warn
            } else {
                ProgressAction::Block
            },
            reason: reason.to_string(),
            details,
        }
    }

    fn reset_workspace_dependent_counts(&mut self) {
        self.reset_source_context_without_write_count();
        self.reset_source_context_exploration_without_write_count();
        self.reset_source_context_after_write_count();
        self.reset_failure_anchor_count();
        self.reset_tool_activity_without_write_count();
        self.reset_verified_post_write_activity_count();
    }

    fn reset_source_context_without_write_count(&mut self) {
        self.last_source_context_without_write_signature = None;
        self.source_context_without_write_count = 0;
        self.source_context_without_write_warned_at = 0;
    }

    fn reset_source_context_exploration_without_write_count(&mut self) {
        self.source_context_exploration_without_write_count = 0;
        self.source_context_exploration_without_write_warned_at = 0;
    }

    fn reset_source_context_after_write_count(&mut self) {
        self.source_context_after_write_count = 0;
        self.source_context_after_write_warned_at = 0;
    }

    fn reset_failure_anchor_count(&mut self) {
        self.last_failure_anchor = None;
        self.failure_anchor_count = 0;
        self.failure_anchor_warned_at = 0;
    }

    fn reset_tool_activity_without_write_count(&mut self) {
        self.tool_activity_without_write_count = 0;
        self.tool_activity_without_write_warned_at = 0;
    }

    fn reset_verified_post_write_activity_count(&mut self) {
        self.verified_post_write_activity_count = 0;
        self.verified_post_write_activity_warned_at = 0;
    }

    fn reset_progress_sensitive_counts(&mut self) {
        self.last_tool_error = None;
        self.tool_error_count = 0;
        self.last_provider_failure = None;
        self.provider_failure_count = 0;
    }
}

/// Whether an observation shows any progress signal.
pub fn has_progress(observation: &ProgressObservation, workspace_progress_observed: bool) -> bool {
    workspace_progress_observed
        || observation.successful_tool_result
        || observation.user_visible_output
        || observation.artifact_completed
}

fn decision_details(
    observation: &ProgressObservation,
    signature: &str,
    count: i64,
) -> serde_json::Value {
    json!({
        "signature": signature,
        "count": count,
        "iteration": observation.iteration,
        "provider_call_count": observation.provider_call_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_signal_observes() {
        let mut watchdog = ProgressWatchdog::default();
        let decision = watchdog.observe(&ProgressObservation::default());
        assert_eq!(decision.action, ProgressAction::Observe);
        assert_eq!(decision.reason, "no_signal");
    }

    #[test]
    fn test_progress_resets_tool_error_count() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 8, 12, 8, 8, 3, true);
        let observation = ProgressObservation {
            tool_error_signature: Some("boom".to_string()),
            ..Default::default()
        };
        // Two failures, then progress.
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        let progressing = ProgressObservation {
            successful_tool_result: true,
            ..Default::default()
        };
        let decision = watchdog.observe(&progressing);
        assert_eq!(decision.reason, "progress");
        assert_eq!(decision.action, ProgressAction::Observe);
        // Failure count reset: one more failure does not fire the decision.
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "no_signal");
    }

    #[test]
    fn test_repeated_tool_error_fires_at_threshold() {
        let mut watchdog = ProgressWatchdog::default();
        let observation = ProgressObservation {
            tool_error_signature: Some("boom".to_string()),
            ..Default::default()
        };
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "repeated_tool_error");
        assert_eq!(decision.action, ProgressAction::Warn);
        assert_eq!(decision.details["count"], 3);
    }

    #[test]
    fn test_block_mode_when_not_observe_only() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 8, 12, 8, 8, 3, false);
        let observation = ProgressObservation {
            tool_error_signature: Some("boom".to_string()),
            ..Default::default()
        };
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.action, ProgressAction::Block);
    }

    #[test]
    fn test_repeated_provider_failure_fires() {
        let mut watchdog = ProgressWatchdog::default();
        let observation = ProgressObservation {
            provider_failure_signature: Some("429".to_string()),
            ..Default::default()
        };
        watchdog.observe(&observation);
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "repeated_provider_failure");
    }

    #[test]
    fn test_source_context_without_write() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 3, 12, 8, 8, 3, true);
        let observation = ProgressObservation {
            successful_source_context_tool_result: true,
            source_context_signature: Some("grep.rs".to_string()),
            ..Default::default()
        };
        for _ in 0..2 {
            watchdog.observe(&observation);
        }
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "source_context_without_workspace_write");
    }

    #[test]
    fn test_workspace_write_resets_source_context_count() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 3, 12, 8, 8, 3, true);
        let mut observation = ProgressObservation {
            successful_source_context_tool_result: true,
            source_context_signature: Some("grep.rs".to_string()),
            ..Default::default()
        };
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        observation.workspace_write_count = 1;
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "progress");
        // Count was reset: two source-context observations without a write
        // stay under the threshold of 3.
        observation.workspace_write_count = 0;
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "no_signal");
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "no_signal");
        // The count is rebuilt and the threshold fires again.
        let decision = watchdog.observe(&observation);
        assert_eq!(decision.reason, "source_context_without_workspace_write");
    }

    #[test]
    fn test_repeated_failure_anchor() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 8, 12, 8, 8, 3, true);
        let observation = ProgressObservation {
            failure_anchor_signature: Some("anchor-x".to_string()),
            ..Default::default()
        };
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        let decision = watchdog.observe(&observation);
        assert_eq!(
            decision.reason,
            "repeated_failure_anchor_without_workspace_write"
        );
        assert_eq!(decision.action, ProgressAction::Warn);
    }

    #[test]
    fn test_verified_post_write_activity() {
        let mut watchdog = ProgressWatchdog::new(3, 2, 3, 8, 12, 8, 8, 3, true);
        let observation = ProgressObservation {
            workspace_write_count: 2,
            post_write_focused_verification_observed: true,
            successful_tool_result: true,
            successful_execution_tool_result: true,
            ..Default::default()
        };
        watchdog.observe(&observation);
        watchdog.observe(&observation);
        let decision = watchdog.observe(&observation);
        assert_eq!(
            decision.reason,
            "verified_workspace_diff_continued_tool_activity"
        );
    }

    #[test]
    fn test_workspace_progress_count_uses_receipts_when_present() {
        let mut observation = ProgressObservation {
            changed_receipt_count: 4,
            noop_receipt_count: 1,
            workspace_write_count: 0,
            ..Default::default()
        };
        assert_eq!(workspace_progress_count(&observation), 4);

        // Receipts exist, so the write count does not override the receipts.
        observation.noop_receipt_count = 0;
        observation.workspace_write_count = 9;
        assert_eq!(workspace_progress_count(&observation), 4);

        // With no receipts at all, the workspace write count is used.
        observation.changed_receipt_count = 0;
        observation.partial_receipt_count = 0;
        assert_eq!(workspace_progress_count(&observation), 9);
    }
}
