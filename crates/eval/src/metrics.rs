use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::benchmark::BenchmarkRun;

/// Evaluation metrics computed from a set of benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvalMetrics {
    /// Number of runs.
    pub total_runs: u64,
    /// Number of successful runs.
    pub successful_runs: u64,
    /// Number of failed runs.
    pub failed_runs: u64,
    /// Accuracy (proportion of runs that passed).
    pub accuracy: f64,
    /// Precision (true positives / (true positives + false positives)).
    pub precision: f64,
    /// Recall (true positives / (true positives + false negatives)).
    pub recall: f64,
    /// F1 score (harmonic mean of precision and recall).
    pub f1: f64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// P50 latency in milliseconds.
    pub p50_latency_ms: f64,
    /// P90 latency in milliseconds.
    pub p90_latency_ms: f64,
    /// P95 latency in milliseconds.
    pub p95_latency_ms: f64,
    /// P99 latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Maximum latency in milliseconds.
    pub max_latency_ms: u64,
    /// Minimum latency in milliseconds.
    pub min_latency_ms: u64,
    /// Total prompt tokens.
    pub total_prompt_tokens: u64,
    /// Total completion tokens.
    pub total_completion_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Metrics broken down by scenario.
    pub per_scenario: HashMap<String, ScenarioMetrics>,
}

/// Metrics for a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScenarioMetrics {
    pub runs: u64,
    pub passed: u64,
    pub failed: u64,
    pub pass_rate: f64,
    pub avg_latency_ms: f64,
}

impl EvalMetrics {
    /// Compute metrics from a list of benchmark runs.
    pub fn compute(runs: &[BenchmarkRun]) -> Self {
        if runs.is_empty() {
            return Self::default();
        }

        let total_runs = runs.len() as u64;
        let successful_runs = runs.iter().filter(|r| r.success).count() as u64;
        let failed_runs = total_runs - successful_runs;

        // Latency statistics
        let mut latencies: Vec<u64> = runs.iter().map(|r| r.duration_ms).collect();
        latencies.sort_unstable();

        let avg_latency_ms = latencies.iter().sum::<u64>() as f64 / total_runs as f64;
        let p = |percentile: f64| -> f64 {
            if latencies.is_empty() {
                return 0.0;
            }
            let idx = ((latencies.len() - 1) as f64 * percentile).round() as usize;
            latencies[idx] as f64
        };

        // Token statistics
        let total_prompt_tokens: u64 = runs.iter().map(|r| r.prompt_tokens).sum();
        let total_completion_tokens: u64 = runs.iter().map(|r| r.completion_tokens).sum();
        let total_tokens = total_prompt_tokens + total_completion_tokens;

        // Accuracy: proportion of runs that succeeded
        let accuracy = successful_runs as f64 / total_runs as f64;

        // Precision/Recall/F1
        // For evaluation, treat "expected keywords found" as positive class.
        // Since BenchmarkRun doesn't track keyword matches directly,
        // we estimate using success as the positive class.
        let true_positives = successful_runs;
        let _true_negatives = 0u64; // not tracked at run level
        let false_positives = 0u64; // not tracked at run level
        let false_negatives = failed_runs;

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // Per-scenario breakdown
        let mut per_scenario: HashMap<String, Vec<&BenchmarkRun>> = HashMap::new();
        for run in runs {
            per_scenario
                .entry(run.scenario.clone())
                .or_default()
                .push(run);
        }

        let per_scenario_metrics: HashMap<String, ScenarioMetrics> = per_scenario
            .into_iter()
            .map(|(name, runs)| {
                let total = runs.len() as u64;
                let passed = runs.iter().filter(|r| r.success).count() as u64;
                let failed = total - passed;
                let avg_latency = if total > 0 {
                    runs.iter().map(|r| r.duration_ms).sum::<u64>() as f64 / total as f64
                } else {
                    0.0
                };
                (
                    name,
                    ScenarioMetrics {
                        runs: total,
                        passed,
                        failed,
                        pass_rate: if total > 0 {
                            passed as f64 / total as f64
                        } else {
                            0.0
                        },
                        avg_latency_ms: avg_latency,
                    },
                )
            })
            .collect();

        Self {
            total_runs,
            successful_runs,
            failed_runs,
            accuracy,
            precision,
            recall,
            f1,
            avg_latency_ms,
            p50_latency_ms: p(0.5),
            p90_latency_ms: p(0.9),
            p95_latency_ms: p(0.95),
            p99_latency_ms: p(0.99),
            max_latency_ms: *latencies.last().unwrap_or(&0),
            min_latency_ms: *latencies.first().unwrap_or(&0),
            total_prompt_tokens,
            total_completion_tokens,
            total_tokens,
            per_scenario: per_scenario_metrics,
        }
    }

    /// Merge two metrics objects by averaging.
    pub fn merge(&self, other: &EvalMetrics) -> EvalMetrics {
        let total_runs = self.total_runs + other.total_runs;
        if total_runs == 0 {
            return Self::default();
        }

        let successful_runs = self.successful_runs + other.successful_runs;
        let failed_runs = self.failed_runs + other.failed_runs;

        let weighted_avg = |a: f64, b: f64| {
            if total_runs == 0 {
                0.0
            } else {
                (a * self.total_runs as f64 + b * other.total_runs as f64) / total_runs as f64
            }
        };

        Self {
            total_runs,
            successful_runs,
            failed_runs,
            accuracy: successful_runs as f64 / total_runs as f64,
            precision: weighted_avg(self.precision, other.precision),
            recall: weighted_avg(self.recall, other.recall),
            f1: weighted_avg(self.f1, other.f1),
            avg_latency_ms: weighted_avg(self.avg_latency_ms, other.avg_latency_ms),
            p50_latency_ms: weighted_avg(self.p50_latency_ms, other.p50_latency_ms),
            p90_latency_ms: weighted_avg(self.p90_latency_ms, other.p90_latency_ms),
            p95_latency_ms: weighted_avg(self.p95_latency_ms, other.p95_latency_ms),
            p99_latency_ms: weighted_avg(self.p99_latency_ms, other.p99_latency_ms),
            max_latency_ms: self.max_latency_ms.max(other.max_latency_ms),
            min_latency_ms: if self.min_latency_ms == 0 {
                other.min_latency_ms
            } else if other.min_latency_ms == 0 {
                self.min_latency_ms
            } else {
                self.min_latency_ms.min(other.min_latency_ms)
            },
            total_prompt_tokens: self.total_prompt_tokens + other.total_prompt_tokens,
            total_completion_tokens: self.total_completion_tokens + other.total_completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            per_scenario: HashMap::new(), // Not merged; too complex for averaging
        }
    }
}

/// Metrics collector that accumulates runs and computes metrics on demand.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    runs: std::sync::Arc<std::sync::Mutex<Vec<BenchmarkRun>>>,
    start_time: DateTime<Utc>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            runs: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            start_time: Utc::now(),
        }
    }

    /// Record a run result.
    pub async fn record_run(&self, run: &BenchmarkRun) {
        if let Ok(mut runs) = self.runs.lock() {
            runs.push(run.clone());
        }
    }

    /// Compute metrics from all recorded runs.
    pub async fn compute_metrics(&self) -> EvalMetrics {
        let runs = self.runs.lock().map(|r| r.clone()).unwrap_or_default();
        EvalMetrics::compute(&runs)
    }

    /// The recorded runs, in insertion order.
    pub async fn runs(&self) -> Vec<BenchmarkRun> {
        self.runs.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Reset the collector.
    pub async fn reset(&self) {
        if let Ok(mut runs) = self.runs.lock() {
            runs.clear();
        }
    }

    /// Get the number of recorded runs.
    pub fn len(&self) -> usize {
        self.runs.lock().map(|r| r.len()).unwrap_or(0)
    }

    /// Check if the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the collection start time.
    pub fn start_time(&self) -> DateTime<Utc> {
        self.start_time
    }
}

/// Calculate accuracy from predictions and ground truth labels.
pub fn accuracy(predictions: &[bool], ground_truth: &[bool]) -> f64 {
    if predictions.is_empty() || predictions.len() != ground_truth.len() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(ground_truth.iter())
        .filter(|(p, g)| p == g)
        .count();
    correct as f64 / predictions.len() as f64
}

/// Calculate precision, recall, and F1 from confusion matrix counts.
pub fn precision_recall_f1(
    true_positives: u64,
    false_positives: u64,
    false_negatives: u64,
) -> (f64, f64, f64) {
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkRun;

    fn make_run(success: bool, duration_ms: u64) -> BenchmarkRun {
        BenchmarkRun {
            scenario: "test".to_string(),
            iteration: 0,
            duration_ms,
            prompt_tokens: 10,
            completion_tokens: 20,
            success,
            error: if success {
                None
            } else {
                Some("fail".to_string())
            },
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_metrics_compute() {
        let runs = vec![
            make_run(true, 100),
            make_run(true, 200),
            make_run(false, 300),
        ];
        let metrics = EvalMetrics::compute(&runs);
        assert_eq!(metrics.total_runs, 3);
        assert_eq!(metrics.successful_runs, 2);
        assert_eq!(metrics.failed_runs, 1);
        assert!(metrics.avg_latency_ms > 0.0);
        assert!(metrics.p50_latency_ms > 0.0);
    }

    #[test]
    fn test_empty_metrics() {
        let metrics = EvalMetrics::compute(&[]);
        assert_eq!(metrics.total_runs, 0);
        assert_eq!(metrics.accuracy, 0.0);
    }

    #[test]
    fn test_precision_recall_f1() {
        let (p, r, f) = precision_recall_f1(8, 2, 1);
        assert!((p - 0.8).abs() < 0.01);
        assert!((r - 0.888).abs() < 0.01);
        assert!((f - 0.842).abs() < 0.01);
    }
}
