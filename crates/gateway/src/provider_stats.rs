//! Provider statistics.
//!
//! Mirrors the Python `provider_stats.py` module. Tracks request count,
//! latency, error rate, and token usage per provider, with aggregate stats,
//! a rolling stats history, and a query API.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::debug;

/// The result of a single provider request.
#[derive(Debug, Clone)]
pub struct ProviderRequestOutcome {
    pub provider: String,
    pub model: String,
    pub latency_ms: u64,
    pub success: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Aggregated statistics for a single provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderStats {
    pub provider: String,
    pub request_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub total_latency_ms: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub first_seen: Option<DateTime<Utc>>,
    pub last_seen: Option<DateTime<Utc>>,
}

impl ProviderStats {
    /// The error rate as a fraction in `[0, 1]`.
    pub fn error_rate(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.error_count as f64 / self.request_count as f64
        }
    }

    /// The average latency in milliseconds.
    pub fn average_latency_ms(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.total_latency_ms as f64 / self.request_count as f64
        }
    }

    /// Merge another stats object into this one (used for aggregation).
    pub fn merge(&mut self, other: &ProviderStats) {
        self.request_count += other.request_count;
        self.success_count += other.success_count;
        self.error_count += other.error_count;
        self.total_latency_ms += other.total_latency_ms;
        self.total_input_tokens += other.total_input_tokens;
        self.total_output_tokens += other.total_output_tokens;
        if self.first_seen.is_none() {
            self.first_seen = other.first_seen;
        }
        if other.last_seen > self.last_seen {
            self.last_seen = other.last_seen;
        }
    }
}

/// Per-model breakdown within a provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStats {
    pub model: String,
    pub request_count: u64,
    pub error_count: u64,
    pub total_latency_ms: u64,
}

/// A snapshot of provider statistics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatsSnapshot {
    pub captured_at: DateTime<Utc>,
    pub providers: Vec<ProviderStats>,
}

/// Tracks per-provider and per-model statistics.
///
/// Thread-safe: stats are updated with interior mutability and can be queried
/// concurrently. Clone is cheap.
#[derive(Clone, Default)]
pub struct ProviderStatsTracker {
    providers: std::sync::Arc<RwLock<HashMap<String, ProviderStats>>>,
    models: std::sync::Arc<RwLock<HashMap<String, HashMap<String, ModelStats>>>>,
    history: std::sync::Arc<Mutex<Vec<ProviderStatsSnapshot>>>,
    history_capacity: usize,
}

impl ProviderStatsTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            history_capacity: 128,
            ..Default::default()
        }
    }

    /// Set the rolling history capacity (number of snapshots retained).
    pub fn with_history_capacity(mut self, capacity: usize) -> Self {
        self.history_capacity = capacity;
        self
    }

    /// Record the outcome of a single provider request.
    pub fn record(&self, outcome: ProviderRequestOutcome) {
        let now = Utc::now();

        {
            let mut providers = self.providers.write();
            let stats = providers
                .entry(outcome.provider.clone())
                .or_insert_with(|| ProviderStats {
                    provider: outcome.provider.clone(),
                    ..Default::default()
                });
            stats.request_count += 1;
            if outcome.success {
                stats.success_count += 1;
            } else {
                stats.error_count += 1;
            }
            stats.total_latency_ms += outcome.latency_ms;
            stats.total_input_tokens += outcome.input_tokens;
            stats.total_output_tokens += outcome.output_tokens;
            if stats.first_seen.is_none() {
                stats.first_seen = Some(now);
            }
            stats.last_seen = Some(now);
        }

        {
            let mut models = self.models.write();
            let by_model = models.entry(outcome.provider.clone()).or_default();
            let model_stats = by_model
                .entry(outcome.model.clone())
                .or_insert_with(|| ModelStats {
                    model: outcome.model.clone(),
                    ..Default::default()
                });
            model_stats.request_count += 1;
            if !outcome.success {
                model_stats.error_count += 1;
            }
            model_stats.total_latency_ms += outcome.latency_ms;
        }

        debug!(provider = %outcome.provider, model = %outcome.model, latency_ms = outcome.latency_ms, "Provider request recorded");
    }

    /// Return stats for a single provider.
    pub fn provider(&self, provider: &str) -> Option<ProviderStats> {
        self.providers.read().get(provider).cloned()
    }

    /// Return stats for all providers, sorted by provider name.
    pub fn all_providers(&self) -> Vec<ProviderStats> {
        let mut list: Vec<ProviderStats> = self.providers.read().values().cloned().collect();
        list.sort_by(|a, b| a.provider.cmp(&b.provider));
        list
    }

    /// Return per-model stats for a provider.
    pub fn models_for(&self, provider: &str) -> Vec<ModelStats> {
        let mut list: Vec<ModelStats> = self
            .models
            .read()
            .get(provider)
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default();
        list.sort_by(|a, b| a.model.cmp(&b.model));
        list
    }

    /// Aggregate all provider stats into one combined [`ProviderStats`].
    pub fn aggregate(&self) -> ProviderStats {
        let mut aggregate = ProviderStats::default();
        for stats in self.all_providers() {
            aggregate.merge(&stats);
        }
        aggregate
    }

    /// Capture a snapshot into the rolling history.
    pub fn snapshot(&self) -> ProviderStatsSnapshot {
        ProviderStatsSnapshot {
            captured_at: Utc::now(),
            providers: self.all_providers(),
        }
    }

    /// Push a snapshot to the history and enforce the capacity bound.
    pub fn push_history(&self, snapshot: ProviderStatsSnapshot) {
        let mut history = self.history.lock();
        history.push(snapshot);
        let overflow = history.len().saturating_sub(self.history_capacity);
        if overflow > 0 {
            history.drain(..overflow);
        }
    }

    /// Capture a snapshot and push it to history in one call.
    pub fn capture(&self) -> ProviderStatsSnapshot {
        let snapshot = self.snapshot();
        self.push_history(snapshot.clone());
        snapshot
    }

    /// Return the recorded history snapshots.
    pub fn history(&self) -> Vec<ProviderStatsSnapshot> {
        self.history.lock().clone()
    }

    /// Return the most recent snapshot, if any.
    pub fn latest_snapshot(&self) -> Option<ProviderStatsSnapshot> {
        self.history.lock().last().cloned()
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        self.providers.write().clear();
        self.models.write().clear();
        self.history.lock().clear();
    }
}

/// Convenience: build a stats payload for an RPC response.
///
/// Returns a JSON object with per-provider stats, an aggregate, and the
/// total request count.
pub fn stats_payload(tracker: &ProviderStatsTracker) -> serde_json::Value {
    let providers = tracker.all_providers();
    let aggregate = tracker.aggregate();
    serde_json::json!({
        "providers": providers,
        "aggregate": aggregate,
        "total_requests": aggregate.request_count,
        "total_errors": aggregate.error_count,
        "overall_error_rate": aggregate.error_rate(),
        "captured_at": Utc::now(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(
        provider: &str,
        model: &str,
        success: bool,
        latency_ms: u64,
    ) -> ProviderRequestOutcome {
        ProviderRequestOutcome {
            provider: provider.to_string(),
            model: model.to_string(),
            latency_ms,
            success,
            input_tokens: 10,
            output_tokens: 5,
        }
    }

    #[test]
    fn test_record_and_query() {
        let tracker = ProviderStatsTracker::new();
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        tracker.record(outcome("openai", "gpt-4o", false, 50));
        tracker.record(outcome("anthropic", "claude-sonnet-4", true, 200));

        let openai = tracker.provider("openai").unwrap();
        assert_eq!(openai.request_count, 2);
        assert_eq!(openai.error_count, 1);
        assert_eq!(openai.total_latency_ms, 150);
        assert_eq!(openai.average_latency_ms(), 75.0);
        assert!((openai.error_rate() - 0.5).abs() < 1e-9);

        let anthropic = tracker.provider("anthropic").unwrap();
        assert_eq!(anthropic.request_count, 1);
        assert_eq!(anthropic.success_count, 1);
        assert_eq!(anthropic.error_rate(), 0.0);

        assert_eq!(tracker.all_providers().len(), 2);
    }

    #[test]
    fn test_model_breakdown() {
        let tracker = ProviderStatsTracker::new();
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        tracker.record(outcome("openai", "gpt-4o-mini", true, 50));
        let models = tracker.models_for("openai");
        assert_eq!(models.len(), 2);
        let by_name: HashMap<_, _> = models
            .iter()
            .map(|m| (m.model.clone(), m.clone()))
            .collect();
        assert_eq!(by_name["gpt-4o"].request_count, 1);
        assert_eq!(by_name["gpt-4o-mini"].request_count, 1);
    }

    #[test]
    fn test_aggregate() {
        let tracker = ProviderStatsTracker::new();
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        tracker.record(outcome("anthropic", "claude-sonnet-4", false, 200));
        let aggregate = tracker.aggregate();
        assert_eq!(aggregate.request_count, 2);
        assert_eq!(aggregate.error_count, 1);
        assert_eq!(aggregate.total_latency_ms, 300);
        assert_eq!(aggregate.total_input_tokens, 20);
        assert_eq!(aggregate.total_output_tokens, 10);
    }

    #[test]
    fn test_history_and_capture() {
        let tracker = ProviderStatsTracker::new().with_history_capacity(3);
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        tracker.capture();
        tracker.record(outcome("openai", "gpt-4o", false, 50));
        tracker.capture();
        tracker.record(outcome("anthropic", "claude", true, 10));
        tracker.capture();
        tracker.record(outcome("anthropic", "claude", false, 20));
        tracker.capture();
        // Capacity 3: the first snapshot should be evicted.
        assert_eq!(tracker.history().len(), 3);
        assert!(tracker.latest_snapshot().is_some());
    }

    #[test]
    fn test_reset() {
        let tracker = ProviderStatsTracker::new();
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        tracker.capture();
        tracker.reset();
        assert!(tracker.all_providers().is_empty());
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_merge() {
        let mut a = ProviderStats {
            provider: "openai".into(),
            request_count: 1,
            success_count: 1,
            ..Default::default()
        };
        let b = ProviderStats {
            provider: "openai".into(),
            request_count: 2,
            error_count: 1,
            total_latency_ms: 300,
            total_input_tokens: 30,
            total_output_tokens: 15,
            first_seen: Some(Utc::now()),
            last_seen: Some(Utc::now()),
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.request_count, 3);
        assert_eq!(a.error_count, 1);
        assert_eq!(a.total_input_tokens, 30);
    }

    #[test]
    fn test_stats_payload_shape() {
        let tracker = ProviderStatsTracker::new();
        tracker.record(outcome("openai", "gpt-4o", true, 100));
        let payload = stats_payload(&tracker);
        assert_eq!(payload["total_requests"], 1);
        assert_eq!(payload["providers"].as_array().unwrap().len(), 1);
        assert_eq!(payload["aggregate"]["request_count"], 1);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = ProviderStatsTracker::new();
        assert!(tracker.all_providers().is_empty());
        assert_eq!(tracker.aggregate().request_count, 0);
        assert_eq!(tracker.aggregate().error_rate(), 0.0);
        let payload = stats_payload(&tracker);
        assert_eq!(payload["total_requests"], 0);
    }
}
