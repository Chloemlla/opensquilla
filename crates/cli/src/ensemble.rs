//! Provider ensemble management commands.
//!
//! Implements the `ensemble` subcommand for inspecting and validating the
//! proposer-aggregator ensemble configuration. Mirrors the Python
//! `ensemble_cmd.py`, which benchmarks the ensemble against a single-model
//! baseline.
//!
//! The Rust `opensquilla-provider` crate exposes the full ensemble types
//! (`EnsembleConfig`, `ProposerSpec`, `AggregationSpec`, …). The Python
//! benchmark harness (`run_dry_run_benchmark` / `run_config_benchmark`) is not
//! yet ported to Rust, so `bench` is stubbed with `// TODO:`.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_provider::ensemble::EnsembleConfig;
use opensquilla_provider::ensemble::{AggregationStrategy, ExecutionMode, ScoringStrategy};
use tracing::info;

use crate::table::{self, Color, KeyValue, Style};

/// Ensemble subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum EnsembleAction {
    /// Show the effective ensemble configuration.
    Show,
    /// Validate the ensemble configuration.
    Validate,
    /// List the configured proposers.
    List,
    /// Benchmark the ensemble against a single-model baseline.
    Bench {
        /// JSON array of {id, text, system?}; omit for built-in set.
        #[arg(long)]
        prompts: Option<String>,
        /// Offline mode: scripted synthetic providers, no credentials.
        #[arg(long)]
        dry_run: bool,
        /// Runs per prompt per arm.
        #[arg(long, default_value = "1")]
        repeat: u32,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Run an ensemble subcommand.
pub async fn run_ensemble(action: EnsembleAction) -> Result<()> {
    match action {
        EnsembleAction::Show => ensemble_show().await,
        EnsembleAction::Validate => ensemble_validate().await,
        EnsembleAction::List => ensemble_list().await,
        EnsembleAction::Bench {
            prompts,
            dry_run,
            repeat,
            json,
        } => ensemble_bench(prompts, dry_run, repeat, json).await,
    }
}

/// Resolve the ensemble config from the loaded OpenSquilla config.
///
/// The config file may carry an `ensemble` table; when absent, the default
/// (empty) ensemble is returned.
fn load_ensemble_config() -> Result<EnsembleConfig> {
    let config = Config::load().context("Failed to load configuration")?;
    let raw = config.get("ensemble").unwrap_or_default();
    if raw.is_empty() {
        return Ok(EnsembleConfig::default());
    }
    // Try parsing the raw string as TOML/JSON into an EnsembleConfig.
    toml::from_str::<EnsembleConfig>(&raw)
        .or_else(|_| serde_json::from_str::<EnsembleConfig>(&raw))
        .context("Failed to parse ensemble configuration")
}

/// Show the effective ensemble configuration.
pub async fn ensemble_show() -> Result<()> {
    let ensemble = load_ensemble_config()?;

    println!("{}", "Ensemble Configuration".bold());
    println!("{:-<60}", "");
    KeyValue::new()
        .entry("name", ensemble.name.clone())
        .entry("proposers", ensemble.proposers.len().to_string())
        .entry(
            "aggregator",
            aggregation_strategy_name(&ensemble.aggregator.strategy),
        )
        .entry("scoring", scoring_strategy_name(&ensemble.scoring))
        .entry(
            "execution_mode",
            execution_mode_name(&ensemble.execution_mode),
        )
        .entry(
            "min_successful_proposers",
            ensemble.min_successful_proposers.to_string(),
        )
        .entry(
            "proposer_timeout",
            format!("{}s", ensemble.proposer_timeout.as_secs()),
        )
        .entry(
            "aggregator_timeout",
            format!("{}s", ensemble.aggregator_timeout.as_secs()),
        )
        .entry(
            "quorum_grace",
            format!("{}s", ensemble.quorum_grace.as_secs()),
        )
        .entry(
            "shuffle_candidates",
            ensemble.shuffle_candidates.to_string(),
        )
        .entry("max_total_calls", ensemble.max_total_calls.to_string())
        .entry("proposer_tools", ensemble.proposer_tools.to_string())
        .print();

    if !ensemble.proposers.is_empty() {
        println!();
        println!("{}", "Proposers".bold());
        println!("{:-<60}", "");
        for p in &ensemble.proposers {
            println!(
                "  {} {:<16} model={:<20} role={:?} weight={:.2}",
                table::info(),
                p.label,
                if p.model.is_empty() {
                    "(inherit)"
                } else {
                    &p.model
                },
                p.role,
                p.weight
            );
        }
    }

    if let Some(fb) = &ensemble.fallback {
        println!();
        println!(
            "{} fallback: provider={} model={}",
            table::warn(),
            fb.provider.as_deref().unwrap_or("(default)"),
            if fb.model.is_empty() {
                "(inherit)"
            } else {
                &fb.model
            }
        );
    }
    Ok(())
}

/// Validate the ensemble configuration.
pub async fn ensemble_validate() -> Result<()> {
    let ensemble = load_ensemble_config()?;
    match ensemble.validate() {
        Ok(()) => {
            println!("{} Ensemble configuration is valid", table::ok());
            println!(
                "  {} proposer(s), {} min successful",
                ensemble.proposers.len(),
                ensemble.min_successful_proposers
            );
            Ok(())
        }
        Err(e) => {
            println!("{} Ensemble configuration is invalid", table::fail());
            println!("  {e}");
            anyhow::bail!("ensemble validation failed: {e}");
        }
    }
}

/// List the configured proposers.
pub async fn ensemble_list() -> Result<()> {
    let ensemble = load_ensemble_config()?;
    if ensemble.proposers.is_empty() {
        println!("No proposers configured.");
        return Ok(());
    }

    println!("Proposers ({}):", ensemble.proposers.len());
    println!("{:-<80}", "");
    println!(
        "{:<16} {:<20} {:<12} {:<8} {:<8}",
        "Label", "Model", "Role", "Weight", "Tools"
    );
    println!("{:-<80}", "");
    for p in &ensemble.proposers {
        println!(
            "{:<16} {:<20} {:<12} {:<8.2} {:<8}",
            p.label,
            if p.model.is_empty() {
                "(inherit)"
            } else {
                &p.model
            },
            format!("{:?}", p.role),
            p.weight,
            p.tools_enabled
        );
    }
    Ok(())
}

/// Benchmark the ensemble against a single-model baseline.
pub async fn ensemble_bench(
    prompts: Option<String>,
    dry_run: bool,
    repeat: u32,
    json: bool,
) -> Result<()> {
    if dry_run {
        return ensemble_bench_dry_run(prompts, repeat, json).await;
    }

    // TODO: port the Python `opensquilla.eval.run_config_benchmark` harness to
    // Rust. The Rust `opensquilla-eval` crate exposes a generic
    // `BenchmarkRunner` that requires a `BenchmarkEngine` implementation;
    // wiring the ensemble as a benchmark engine is out of scope for this port.
    info!(repeat, "ensemble live bench requested (not yet implemented)");

    let report = serde_json::json!({
        "status": "not_implemented",
        "dry_run": false,
        "repeat": repeat,
        "message": "Live ensemble benchmark harness is not yet ported to Rust.",
    });

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!(
        "{} Ensemble live benchmark is not yet implemented in Rust.",
        table::warn()
    );
    println!("  Use `--dry-run` for the offline synthetic benchmark, which is ported.");
    Ok(())
}

/// Run the offline, deterministic ensemble benchmark against scripted synthetic
/// providers (no network, no credentials). Mirrors the Python
/// `opensquilla.eval.scenarios.run_dry_run_benchmark`.
pub async fn ensemble_bench_dry_run(
    prompts: Option<String>,
    repeat: u32,
    json: bool,
) -> Result<()> {
    let prompts = match prompts {
        Some(raw) => serde_json::from_str::<Vec<opensquilla_eval::SyntheticPrompt>>(&raw)
            .context("Failed to parse --prompts as a JSON array of {id, text, system?}")?,
        None => opensquilla_eval::default_synthetic_prompts(),
    };

    let report =
        opensquilla_eval::run_dry_run_benchmark(prompts, repeat, true).await;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("{} Ensemble dry-run benchmark (offline, synthetic)", table::ok());
    println!("{:-<72}", "");
    println!(
        "{:<12} {:>6} {:>9} {:>6} {:>8} {:>12}",
        "Arm", "Runs", "Success", "Fail", "Rate", "Mean ms"
    );
    println!("{:-<72}", "");
    for arm in [&report.ensemble, &report.baseline] {
        println!(
            "{:<12} {:>6} {:>9} {:>6} {:>7.1}% {:>12.1}",
            arm.label,
            arm.runs,
            arm.successes,
            arm.failures,
            arm.success_rate * 100.0,
            arm.mean_latency_ms,
        );
    }
    println!("{:-<72}", "");
    println!(
        "  latency delta     : {:+.1} ms",
        report.deltas.latency_delta_ms
    );
    println!(
        "  success rate delta: {:.1}%",
        report.deltas.success_rate_delta * 100.0
    );
    println!(
        "  billed cost delta : {:+.6} USD",
        report.deltas.billed_cost_delta_usd
    );
    if let (Some(mean_succ), Some(mean_total)) = (
        report.ensemble.mean_successful_proposers,
        report.ensemble.mean_total_candidates,
    ) {
        println!(
            "  ensemble proposers: {:.1} successful / {:.1} candidates",
            mean_succ, mean_total
        );
    }
    Ok(())
}

/// Human-readable name for an aggregation strategy.
fn aggregation_strategy_name(s: &AggregationStrategy) -> &'static str {
    match s {
        AggregationStrategy::Aggregator => "aggregator",
        AggregationStrategy::BestOfN => "best_of_n",
        AggregationStrategy::MixtureOfAgents => "mixture_of_agents",
        AggregationStrategy::Debate => "debate",
        AggregationStrategy::Voting => "voting",
        AggregationStrategy::FirstComplete => "first_complete",
    }
}

/// Human-readable name for a scoring strategy.
fn scoring_strategy_name(s: &ScoringStrategy) -> &'static str {
    match s {
        ScoringStrategy::BestOfN => "best_of_n",
        ScoringStrategy::MixtureOfAgents => "mixture_of_agents",
        ScoringStrategy::Debate => "debate",
        ScoringStrategy::Voting => "voting",
    }
}

/// Human-readable name for an execution mode.
fn execution_mode_name(m: &ExecutionMode) -> &'static str {
    match m {
        ExecutionMode::Parallel => "parallel",
        ExecutionMode::Sequential => "sequential",
    }
}

/// Bold helper for headers.
trait BoldStr {
    fn bold(&self) -> String;
}

impl BoldStr for &str {
    fn bold(&self) -> String {
        format!(
            "{}",
            Style::new().bold().fg(Color::BrightBlue).styled(*self)
        )
    }
}

impl BoldStr for String {
    fn bold(&self) -> String {
        format!("{}", Style::new().bold().fg(Color::BrightBlue).styled(self))
    }
}
