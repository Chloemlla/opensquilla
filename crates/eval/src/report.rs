use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::benchmark::BenchmarkResult;
use crate::metrics::EvalMetrics;

/// A generated evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report title.
    pub title: String,
    /// Report generation timestamp.
    pub generated_at: DateTime<Utc>,
    /// Benchmark result this report is based on.
    pub benchmark: BenchmarkResult,
    /// Evaluation metrics.
    pub metrics: EvalMetrics,
    /// Format of the report.
    pub format: ReportFormat,
}

/// Report output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Markdown,
}

impl Report {
    /// Create a new report from a benchmark result.
    pub fn from_benchmark(benchmark: BenchmarkResult) -> Self {
        Self {
            title: format!("Benchmark Report: {}", benchmark.config.name),
            generated_at: Utc::now(),
            metrics: benchmark.metrics.clone(),
            benchmark,
            format: ReportFormat::Json,
        }
    }

    /// Generate the report in JSON format.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generate the report in markdown format.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str(&format!("*Generated at {}*\n\n", self.generated_at.to_rfc3339()));

        // Configuration
        md.push_str("## Configuration\n\n");
        md.push_str(&format!("- **Name**: {}\n", self.benchmark.config.name));
        md.push_str(&format!("- **Provider**: {}\n", self.benchmark.config.provider));
        md.push_str(&format!("- **Model**: {}\n", self.benchmark.config.model));
        md.push_str(&format!("- **Iterations**: {}\n", self.benchmark.config.iterations));
        md.push_str(&format!("- **Timeout**: {}s\n", self.benchmark.config.timeout_seconds));
        md.push_str(&format!("- **Concurrency**: {}\n", self.benchmark.config.concurrency));
        md.push_str("\n");

        // Overall metrics
        md.push_str("## Overall Metrics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total Runs | {} |\n", self.metrics.total_runs));
        md.push_str(&format!("| Successful | {} |\n", self.metrics.successful_runs));
        md.push_str(&format!("| Failed | {} |\n", self.metrics.failed_runs));
        md.push_str(&format!("| Accuracy | {:.2}% |\n", self.metrics.accuracy * 100.0));
        md.push_str(&format!("| Precision | {:.2} |\n", self.metrics.precision));
        md.push_str(&format!("| Recall | {:.2} |\n", self.metrics.recall));
        md.push_str(&format!("| F1 Score | {:.2} |\n", self.metrics.f1));
        md.push_str(&format!("| Avg Latency | {:.2} ms |\n", self.metrics.avg_latency_ms));
        md.push_str(&format!("| P50 Latency | {:.2} ms |\n", self.metrics.p50_latency_ms));
        md.push_str(&format!("| P90 Latency | {:.2} ms |\n", self.metrics.p90_latency_ms));
        md.push_str(&format!("| P95 Latency | {:.2} ms |\n", self.metrics.p95_latency_ms));
        md.push_str(&format!("| Max Latency | {} ms |\n", self.metrics.max_latency_ms));
        md.push_str(&format!("| Min Latency | {} ms |\n", self.metrics.min_latency_ms));
        md.push_str(&format!("| Total Tokens | {} |\n", self.metrics.total_tokens));
        md.push_str("\n");

        // Per-scenario breakdown
        if !self.metrics.per_scenario.is_empty() {
            md.push_str("## Per-Scenario Results\n\n");
            md.push_str("| Scenario | Runs | Passed | Failed | Pass Rate | Avg Latency |\n");
            md.push_str("|----------|------|--------|--------|-----------|-------------|\n");
            let mut scenarios: Vec<_> = self.metrics.per_scenario.iter().collect();
            scenarios.sort_by(|a, b| a.0.cmp(b.0));
            for (name, sm) in scenarios {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1}% | {:.2} ms |\n",
                    name, sm.runs, sm.passed, sm.failed, sm.pass_rate * 100.0, sm.avg_latency_ms
                ));
            }
            md.push_str("\n");
        }

        // Run history
        md.push_str("## Run History\n\n");
        md.push_str("| # | Scenario | Iteration | Duration (ms) | Success | Error |\n");
        md.push_str("|---|----------|-----------|---------------|---------|-------|\n");
        for (i, run) in self.benchmark.runs.iter().enumerate() {
            let success = if run.success { "YES" } else { "NO" };
            let error = run.error.as_deref().unwrap_or("");
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                i + 1, run.scenario, run.iteration, run.duration_ms, success, error
            ));
        }
        md.push_str("\n");

        md.push_str("---\n*Report generated by OpenSquilla eval.*\n");
        md
    }

    /// Generate the report in the specified format.
    pub fn render(&self, format: ReportFormat) -> Result<String, ReportError> {
        match format {
            ReportFormat::Json => self.to_json().map_err(ReportError::Serialization),
            ReportFormat::Markdown => Ok(self.to_markdown()),
        }
    }

    /// Write the report to a file.
    pub fn write_to_file(&self, path: &str, format: ReportFormat) -> Result<(), ReportError> {
        let content = self.render(format)?;
        std::fs::write(path, content).map_err(|e| ReportError::Io(e.to_string()))
    }

    /// Get the report title.
    pub fn title(&self) -> &str { &self.title }

    /// Get the benchmark result.
    pub fn benchmark(&self) -> &BenchmarkResult { &self.benchmark }

    /// Get the evaluation metrics.
    pub fn metrics(&self) -> &EvalMetrics { &self.metrics }
}

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error("Serialization error: {0}")]
    Serialization(serde_json::Error),

    #[error("IO error: {0}")]
    Io(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{BenchmarkConfig, BenchmarkRun};

    fn sample_report() -> Report {
        let config = BenchmarkConfig::default();
        let runs = vec![
            BenchmarkRun {
                scenario: "test_scenario".to_string(),
                iteration: 0,
                duration_ms: 100,
                prompt_tokens: 10,
                completion_tokens: 20,
                success: true,
                error: None,
                timestamp: Utc::now(),
            },
        ];
        let metrics = EvalMetrics::compute(&runs);
        let benchmark = BenchmarkResult {
            config,
            runs,
            metrics,
            start_time: Utc::now(),
            end_time: Utc::now(),
            total_duration_ms: 100,
        };
        Report::from_benchmark(benchmark)
    }

    #[test]
    fn test_report_to_json() {
        let report = sample_report();
        let json = report.to_json().unwrap();
        assert!(json.contains("Benchmark Report"));
    }

    #[test]
    fn test_report_to_markdown() {
        let report = sample_report();
        let md = report.to_markdown();
        assert!(md.contains("# Benchmark Report"));
        assert!(md.contains("Overall Metrics"));
        assert!(md.contains("test_scenario"));
    }

    #[test]
    fn test_report_render() {
        let report = sample_report();
        assert!(report.render(ReportFormat::Json).is_ok());
        assert!(report.render(ReportFormat::Markdown).is_ok());
    }
}
