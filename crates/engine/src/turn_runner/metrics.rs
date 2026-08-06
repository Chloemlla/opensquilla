//! Per-stage metrics collection for the turn runner.
//!
//! This module provides a shared, thread-safe metrics collector that the turn
//! stages write into during execution. The finalizer (or the runtime) can read
//! the collected metrics for observability, persistence, or dashboards.
//!
//! Mirrors the Python `engine/turn_runner/harness.py` `_STAGE_METRICS` slot
//! plus the observability timers collected by each stage.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// A single stage's execution record.
#[derive(Debug, Clone, Default)]
pub struct StageMetric {
    /// The stage name.
    pub stage: String,
    /// The turn id.
    pub turn_id: String,
    /// Wall-clock duration of the stage execution in milliseconds.
    pub duration_ms: u64,
    /// The number of messages when the stage started.
    pub messages_in: usize,
    /// The number of messages when the stage finished.
    pub messages_out: usize,
    /// Whether the stage succeeded.
    pub success: bool,
    /// A stage-specific error code, when the stage failed.
    pub error_code: Option<String>,
    /// The wall-clock start timestamp (epoch ms).
    pub started_at_ms: i64,
}

impl StageMetric {
    /// Create a new stage metric for the given stage and turn.
    pub fn new(stage: impl Into<String>, turn_id: impl Into<String>) -> Self {
        Self {
            stage: stage.into(),
            turn_id: turn_id.into(),
            duration_ms: 0,
            messages_in: 0,
            messages_out: 0,
            success: true,
            error_code: None,
            started_at_ms: crate::recovery::crash::current_epoch_ms(),
        }
    }

    /// Record the wall-clock duration.
    pub fn with_duration(mut self, started: Instant) -> Self {
        self.duration_ms = started.elapsed().as_millis() as u64;
        self
    }

    /// Mark the record as failed with an error code.
    pub fn failed(mut self, code: impl Into<String>) -> Self {
        self.success = false;
        self.error_code = Some(code.into());
        self
    }

    /// Whether the stage succeeded.
    pub fn is_success(&self) -> bool {
        self.success
    }
}

/// A thread-safe collector of per-stage execution metrics.
#[derive(Debug, Default)]
pub struct StageMetricsCollector {
    /// The collected records, keyed by turn id then stage name.
    records: Mutex<HashMap<String, HashMap<String, StageMetric>>>,
}

impl StageMetricsCollector {
    /// Create a new empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a stage metric for a turn.
    pub fn record(&self, metric: StageMetric) {
        let mut records = self.records.lock().unwrap_or_else(|e| e.into_inner());
        let turn = records.entry(metric.turn_id.clone()).or_default();
        turn.insert(metric.stage.clone(), metric);
    }

    /// Convenience: record a stage with an explicit start instant.
    pub fn record_stage(
        &self,
        stage: &str,
        turn_id: &str,
        started: Instant,
        messages_in: usize,
        messages_out: usize,
        success: bool,
    ) {
        let mut metric = StageMetric::new(stage, turn_id);
        metric.duration_ms = started.elapsed().as_millis() as u64;
        metric.messages_in = messages_in;
        metric.messages_out = messages_out;
        metric.success = success;
        self.record(metric);
    }

    /// Get the records for a turn.
    pub fn for_turn(&self, turn_id: &str) -> HashMap<String, StageMetric> {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(turn_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get a specific stage's record for a turn.
    pub fn stage_metric(&self, turn_id: &str, stage: &str) -> Option<StageMetric> {
        self.for_turn(turn_id).get(stage).cloned()
    }

    /// Remove all records for a turn.
    pub fn clear_turn(&self, turn_id: &str) {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(turn_id);
    }

    /// The total number of records across all turns.
    pub fn total_records(&self) -> usize {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|m| m.len())
            .sum()
    }

    /// The set of turn ids with recorded metrics.
    pub fn turn_ids(&self) -> Vec<String> {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Compute a per-stage rollup across all turns (average duration).
    pub fn stage_rollups(&self) -> Vec<StageRollup> {
        let records = self.records.lock().unwrap_or_else(|e| e.into_inner());
        let mut totals: HashMap<String, (u64, usize)> = HashMap::new();
        let mut failures: HashMap<String, usize> = HashMap::new();
        for turn in records.values() {
            for metric in turn.values() {
                let entry = totals.entry(metric.stage.clone()).or_default();
                entry.0 += metric.duration_ms;
                entry.1 += 1;
                if !metric.success {
                    *failures.entry(metric.stage.clone()).or_default() += 1;
                }
            }
        }
        let mut rollups: Vec<StageRollup> = totals
            .into_iter()
            .map(|(stage, (total_ms, count))| StageRollup {
                stage: stage.clone(),
                runs: count,
                average_duration_ms: if count > 0 {
                    total_ms / count as u64
                } else {
                    0
                },
                failures: failures.get(&stage).copied().unwrap_or(0),
            })
            .collect();
        rollups.sort_by(|a, b| a.stage.cmp(&b.stage));
        rollups
    }

    /// Whether any recorded stage failed for the given turn.
    pub fn turn_has_failure(&self, turn_id: &str) -> bool {
        self.for_turn(turn_id)
            .values()
            .any(|m| !m.success)
    }
}

/// A per-stage rollup across all recorded turns.
#[derive(Debug, Clone)]
pub struct StageRollup {
    /// The stage name.
    pub stage: String,
    /// The number of recorded runs.
    pub runs: usize,
    /// The average duration in milliseconds.
    pub average_duration_ms: u64,
    /// The number of failed runs.
    pub failures: usize,
}

impl StageRollup {
    /// The success rate for this stage.
    pub fn success_rate(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            (self.runs - self.failures) as f64 / self.runs as f64
        }
    }
}

/// A timer helper for a single stage execution.
#[derive(Debug)]
pub struct StageTimer {
    /// The stage name.
    stage: String,
    /// The turn id.
    turn_id: String,
    /// When the stage started.
    started: Instant,
    /// The message count when the stage started.
    messages_in: usize,
}

impl StageTimer {
    /// Start a new stage timer.
    pub fn start(stage: impl Into<String>, turn_id: impl Into<String>, messages_in: usize) -> Self {
        Self {
            stage: stage.into(),
            turn_id: turn_id.into(),
            started: Instant::now(),
            messages_in,
        }
    }

    /// Finish the stage, producing a [`StageMetric`].
    pub fn finish(&self, messages_out: usize, success: bool) -> StageMetric {
        let mut metric = StageMetric::new(&self.stage, &self.turn_id);
        metric.duration_ms = self.started.elapsed().as_millis() as u64;
        metric.messages_in = self.messages_in;
        metric.messages_out = messages_out;
        metric.success = success;
        metric
    }

    /// The elapsed duration.
    pub fn elapsed(&self) -> std::time::Duration {
        self.started.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_metric_new_and_failed() {
        let metric = StageMetric::new("harness", "t1");
        assert_eq!(metric.stage, "harness");
        assert!(metric.success);
        assert!(metric.error_code.is_none());

        let failed = StageMetric::new("harness", "t1").failed("EMPTY_MESSAGES");
        assert!(!failed.success);
        assert_eq!(failed.error_code.as_deref(), Some("EMPTY_MESSAGES"));
    }

    #[test]
    fn test_collector_record_and_get() {
        let collector = StageMetricsCollector::new();
        collector.record(StageMetric::new("harness", "t1"));
        collector.record(StageMetric::new("provider", "t1"));
        collector.record(StageMetric::new("harness", "t2"));

        assert_eq!(collector.for_turn("t1").len(), 2);
        assert_eq!(collector.total_records(), 3);
        assert_eq!(collector.turn_ids().len(), 2);
    }

    #[test]
    fn test_collector_stage_metric() {
        let collector = StageMetricsCollector::new();
        collector.record(StageMetric::new("harness", "t1"));
        let metric = collector.stage_metric("t1", "harness");
        assert!(metric.is_some());
        assert!(collector.stage_metric("t1", "missing").is_none());
    }

    #[test]
    fn test_collector_clear_turn() {
        let collector = StageMetricsCollector::new();
        collector.record(StageMetric::new("harness", "t1"));
        collector.record(StageMetric::new("harness", "t2"));
        collector.clear_turn("t1");
        assert_eq!(collector.total_records(), 1);
    }

    #[test]
    fn test_stage_rollups() {
        let collector = StageMetricsCollector::new();
        let mut m1 = StageMetric::new("provider", "t1");
        m1.duration_ms = 100;
        collector.record(m1);
        let mut m2 = StageMetric::new("provider", "t2");
        m2.duration_ms = 300;
        collector.record(m2);
        let m3 = StageMetric::new("harness", "t1").failed("X");
        collector.record(m3);

        let rollups = collector.stage_rollups();
        let provider = rollups.iter().find(|r| r.stage == "provider").unwrap();
        assert_eq!(provider.runs, 2);
        assert_eq!(provider.average_duration_ms, 200);
        let harness = rollups.iter().find(|r| r.stage == "harness").unwrap();
        assert_eq!(harness.failures, 1);
        assert_eq!(harness.success_rate(), 0.0);
    }

    #[test]
    fn test_turn_has_failure() {
        let collector = StageMetricsCollector::new();
        collector.record(StageMetric::new("harness", "t1").failed("X"));
        collector.record(StageMetric::new("harness", "t2"));
        assert!(collector.turn_has_failure("t1"));
        assert!(!collector.turn_has_failure("t2"));
    }

    #[test]
    fn test_stage_timer() {
        let timer = StageTimer::start("harness", "t1", 3);
        let metric = timer.finish(5, true);
        assert_eq!(metric.messages_in, 3);
        assert_eq!(metric.messages_out, 5);
        assert!(metric.success);
    }
}
