use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::metrics::{EvalMetrics, MetricsCollector};
use crate::scenarios::{Scenario, ScenarioResult};

/// A benchmark configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Name of the benchmark.
    pub name: String,
    /// Provider to test.
    pub provider: String,
    /// Model to test.
    pub model: String,
    /// Number of iterations per scenario.
    pub iterations: usize,
    /// Maximum time per request.
    pub timeout_seconds: u64,
    /// Concurrent request count.
    pub concurrency: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default_benchmark".to_string(),
            provider: "default".to_string(),
            model: "default".to_string(),
            iterations: 3,
            timeout_seconds: 60,
            concurrency: 1,
        }
    }
}

/// A collection of benchmark scenarios with a shared configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub config: BenchmarkConfig,
    pub scenarios: Vec<Scenario>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite.
    pub fn new(name: impl Into<String>, config: BenchmarkConfig, scenarios: Vec<Scenario>) -> Self {
        Self {
            name: name.into(),
            config,
            scenarios,
        }
    }

    /// Append a scenario to the suite.
    pub fn add(&mut self, scenario: Scenario) {
        self.scenarios.push(scenario);
    }

    /// The number of scenarios in the suite.
    pub fn len(&self) -> usize {
        self.scenarios.len()
    }

    /// Returns true when the suite contains no scenarios.
    pub fn is_empty(&self) -> bool {
        self.scenarios.is_empty()
    }
}

/// A single benchmark run result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    pub scenario: String,
    pub iteration: usize,
    pub duration_ms: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl BenchmarkRun {
    /// Evaluate a run against the expected keywords of its scenario.
    pub fn evaluate_keywords(&self, scenario: &Scenario) -> ScenarioResult {
        let response = self.error.as_deref().unwrap_or("");
        let mut result = scenario.evaluate(response, self.duration_ms);
        // If the run itself failed, mark the scenario as not passed
        if !self.success {
            result.passed = false;
        }
        result
    }
}

/// Overall benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub runs: Vec<BenchmarkRun>,
    pub metrics: EvalMetrics,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_duration_ms: u64,
}

/// A trait for the engine under test.
///
/// Implementations adapt a real agent engine (e.g., opensquilla-engine's
/// TurnRunner or a provider client) to the benchmark harness.
#[async_trait::async_trait]
pub trait BenchmarkEngine: Send + Sync {
    /// Process a single user message with the given system prompt, provider, and model.
    async fn process_message(
        &self,
        system_prompt: &str,
        provider: &str,
        model: &str,
        user_input: &str,
    ) -> Result<String, BenchmarkError>;
}

/// Benchmark runner for evaluating model performance.
#[derive(Clone)]
pub struct BenchmarkRunner {
    engine: Arc<RwLock<Option<Arc<dyn BenchmarkEngine>>>>,
    collector: MetricsCollector,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new() -> Self {
        info!("Benchmark runner initialized");
        Self {
            engine: Arc::new(RwLock::new(None)),
            collector: MetricsCollector::new(),
        }
    }

    /// Initialize the engine under test.
    pub async fn init_engine(&self, engine: Arc<dyn BenchmarkEngine>) {
        let mut lock = self.engine.write().await;
        *lock = Some(engine);
        info!("Engine initialized for benchmarking");
    }

    /// Check whether an engine has been initialized.
    pub async fn has_engine(&self) -> bool {
        self.engine.read().await.is_some()
    }

    /// Run a benchmark with the given configuration and scenarios.
    pub async fn run_benchmark(
        &self,
        config: &BenchmarkConfig,
        scenarios: &[Scenario],
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let start_time = Utc::now();
        let mut runs = Vec::new();

        info!(
            "Starting benchmark '{}' with {} scenarios, {} iterations each",
            config.name,
            scenarios.len(),
            config.iterations
        );

        let engine = {
            let guard = self.engine.read().await;
            guard
                .as_ref()
                .ok_or(BenchmarkError::EngineNotInitialized)?
                .clone()
        };

        let provider = &config.provider;
        let model = &config.model;

        for scenario in scenarios {
            for iteration in 0..config.iterations {
                debug!(
                    "Running scenario '{}' iteration {}/{}",
                    scenario.name,
                    iteration + 1,
                    config.iterations
                );

                let run_start = std::time::Instant::now();

                let result = tokio::time::timeout(
                    Duration::from_secs(config.timeout_seconds),
                    engine.process_message(
                        &scenario.system_prompt,
                        provider,
                        model,
                        &scenario.user_input,
                    ),
                )
                .await;

                let run = match result {
                    Ok(Ok(response)) => {
                        let duration = run_start.elapsed().as_millis() as u64;
                        let prompt_tokens = (scenario.user_input.len() / 4) as u64;
                        let completion_tokens = (response.len() / 4) as u64;

                        info!(
                            "Scenario '{}' iteration {} completed in {}ms",
                            scenario.name,
                            iteration + 1,
                            duration
                        );

                        BenchmarkRun {
                            scenario: scenario.name.clone(),
                            iteration,
                            duration_ms: duration,
                            prompt_tokens,
                            completion_tokens,
                            success: true,
                            error: None,
                            timestamp: Utc::now(),
                        }
                    }
                    Ok(Err(e)) => {
                        let duration = run_start.elapsed().as_millis() as u64;
                        warn!(
                            "Scenario '{}' iteration {} failed: {e}",
                            scenario.name,
                            iteration + 1
                        );
                        BenchmarkRun {
                            scenario: scenario.name.clone(),
                            iteration,
                            duration_ms: duration,
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            success: false,
                            error: Some(e.to_string()),
                            timestamp: Utc::now(),
                        }
                    }
                    Err(_) => {
                        let duration = run_start.elapsed().as_millis() as u64;
                        warn!(
                            "Scenario '{}' iteration {} timed out",
                            scenario.name,
                            iteration + 1
                        );
                        BenchmarkRun {
                            scenario: scenario.name.clone(),
                            iteration,
                            duration_ms: duration,
                            prompt_tokens: 0,
                            completion_tokens: 0,
                            success: false,
                            error: Some("Timeout".to_string()),
                            timestamp: Utc::now(),
                        }
                    }
                };

                self.collector.record_run(&run).await;
                runs.push(run);
            }
        }

        let end_time = Utc::now();
        let total_duration = (end_time - start_time).num_milliseconds() as u64;
        let metrics = self.collector.compute_metrics().await;

        info!(
            "Benchmark '{}' completed: {} runs, {}ms total",
            config.name,
            runs.len(),
            total_duration
        );

        Ok(BenchmarkResult {
            config: config.clone(),
            runs,
            metrics,
            start_time,
            end_time,
            total_duration_ms: total_duration,
        })
    }

    /// Run a quick benchmark with default scenarios.
    pub async fn run_quick_benchmark(
        &self,
        provider: &str,
        model: &str,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let config = BenchmarkConfig {
            name: format!("quick_{provider}_{model}"),
            provider: provider.to_string(),
            model: model.to_string(),
            iterations: 2,
            timeout_seconds: 30,
            concurrency: 1,
        };

        let scenarios = crate::scenarios::default_scenarios();
        self.run_benchmark(&config, &scenarios).await
    }

    /// Get the metrics collector.
    pub fn metrics(&self) -> &MetricsCollector {
        &self.collector
    }

    /// Reset metrics.
    pub async fn reset_metrics(&self) {
        self.collector.reset().await;
    }

    /// Run a full benchmark suite (config + scenarios in one value).
    pub async fn run_benchmark_suite(
        &self,
        suite: &BenchmarkSuite,
    ) -> Result<BenchmarkResult, BenchmarkError> {
        let mut config = suite.config.clone();
        config.name = suite.name.clone();
        self.run_benchmark(&config, &suite.scenarios).await
    }

    /// Run a single scenario once against the engine, recording the run.
    pub async fn run_scenario(
        &self,
        config: &BenchmarkConfig,
        scenario: &Scenario,
    ) -> Result<BenchmarkRun, BenchmarkError> {
        let engine = {
            let guard = self.engine.read().await;
            guard
                .as_ref()
                .ok_or(BenchmarkError::EngineNotInitialized)?
                .clone()
        };

        let run_start = std::time::Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(config.timeout_seconds),
            engine.process_message(
                &scenario.system_prompt,
                &config.provider,
                &config.model,
                &scenario.user_input,
            ),
        )
        .await;

        let run = match result {
            Ok(Ok(response)) => {
                let duration = run_start.elapsed().as_millis() as u64;
                BenchmarkRun {
                    scenario: scenario.name.clone(),
                    iteration: 0,
                    duration_ms: duration,
                    prompt_tokens: (scenario.user_input.len() / 4) as u64,
                    completion_tokens: (response.len() / 4) as u64,
                    success: true,
                    error: None,
                    timestamp: Utc::now(),
                }
            }
            Ok(Err(e)) => {
                let duration = run_start.elapsed().as_millis() as u64;
                BenchmarkRun {
                    scenario: scenario.name.clone(),
                    iteration: 0,
                    duration_ms: duration,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    success: false,
                    error: Some(e.to_string()),
                    timestamp: Utc::now(),
                }
            }
            Err(_) => {
                let duration = run_start.elapsed().as_millis() as u64;
                BenchmarkRun {
                    scenario: scenario.name.clone(),
                    iteration: 0,
                    duration_ms: duration,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    success: false,
                    error: Some("Timeout".to_string()),
                    timestamp: Utc::now(),
                }
            }
        };

        self.collector.record_run(&run).await;
        Ok(run)
    }

    /// Collect all recorded runs from the metrics collector.
    pub async fn collect_results(&self) -> Vec<BenchmarkRun> {
        self.collector.runs().await
    }

    /// Evaluate the recorded runs against their scenarios' expected keywords.
    pub async fn evaluate_suite(&self, suite: &BenchmarkSuite) -> Vec<ScenarioResult> {
        let runs = self.collector.runs().await;
        let by_name: std::collections::HashMap<&str, &Scenario> = suite
            .scenarios
            .iter()
            .map(|s| (s.name.as_str(), s))
            .collect();
        runs.iter()
            .filter_map(|run| {
                by_name
                    .get(run.scenario.as_str())
                    .map(|scenario| run.evaluate_keywords(scenario))
            })
            .collect()
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("Engine not initialized")]
    EngineNotInitialized,

    #[error("Engine error: {0}")]
    EngineError(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEngine;
    #[async_trait::async_trait]
    impl BenchmarkEngine for MockEngine {
        async fn process_message(
            &self,
            _system_prompt: &str,
            _provider: &str,
            _model: &str,
            user_input: &str,
        ) -> Result<String, BenchmarkError> {
            Ok(format!("Echo: {user_input}"))
        }
    }

    #[tokio::test]
    async fn test_benchmark_requires_engine() {
        let runner = BenchmarkRunner::new();
        let scenarios = crate::scenarios::default_scenarios();
        let config = BenchmarkConfig {
            iterations: 1,
            timeout_seconds: 5,
            ..Default::default()
        };
        let result = runner.run_benchmark(&config, &scenarios).await;
        assert!(matches!(result, Err(BenchmarkError::EngineNotInitialized)));
    }

    #[tokio::test]
    async fn test_benchmark_with_mock_engine() {
        let runner = BenchmarkRunner::new();
        runner.init_engine(Arc::new(MockEngine)).await;
        let scenarios = crate::scenarios::default_scenarios();
        let config = BenchmarkConfig {
            iterations: 1,
            timeout_seconds: 5,
            ..Default::default()
        };
        let result = runner.run_benchmark(&config, &scenarios).await.unwrap();
        assert_eq!(result.runs.len(), scenarios.len());
    }

    #[tokio::test]
    async fn test_run_benchmark_suite() {
        let runner = BenchmarkRunner::new();
        runner.init_engine(Arc::new(MockEngine)).await;
        let suite = BenchmarkSuite::new(
            "smoke",
            BenchmarkConfig {
                iterations: 1,
                timeout_seconds: 5,
                ..Default::default()
            },
            crate::scenarios::default_scenarios(),
        );
        let result = runner.run_benchmark_suite(&suite).await.unwrap();
        assert_eq!(result.config.name, "smoke");
        assert_eq!(result.runs.len(), suite.len());
    }

    #[tokio::test]
    async fn test_run_scenario() {
        let runner = BenchmarkRunner::new();
        runner.init_engine(Arc::new(MockEngine)).await;
        let scenario = crate::scenarios::default_scenarios().remove(0);
        let config = BenchmarkConfig {
            timeout_seconds: 5,
            ..Default::default()
        };
        let run = runner.run_scenario(&config, &scenario).await.unwrap();
        assert_eq!(run.scenario, scenario.name);
        assert!(run.success);
    }

    #[tokio::test]
    async fn test_collect_results() {
        let runner = BenchmarkRunner::new();
        runner.init_engine(Arc::new(MockEngine)).await;
        let scenario = crate::scenarios::default_scenarios().remove(0);
        let config = BenchmarkConfig {
            timeout_seconds: 5,
            ..Default::default()
        };
        runner.run_scenario(&config, &scenario).await.unwrap();
        let results = runner.collect_results().await;
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_suite_add_and_len() {
        let mut suite = BenchmarkSuite::new(
            "suite",
            BenchmarkConfig::default(),
            crate::scenarios::default_scenarios(),
        );
        let n = suite.len();
        suite.add(crate::scenarios::default_scenarios().remove(0));
        assert_eq!(suite.len(), n + 1);
        assert!(!suite.is_empty());
    }
}
