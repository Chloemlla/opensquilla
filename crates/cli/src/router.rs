//! # Router commands
//!
//! Implements the `router` subcommand for model routing calibration and
//! inspection. The router selects the best provider/model tier for each turn
//! based on task complexity, cost, and latency. This command lets you:
//!
//! - `router calibrate` — run calibration to score model tiers
//! - `router status` — show the current calibration state
//! - `router tiers` — list configured routing tiers
//! - `router policy` — show the effective routing policy
//! - `router reset` — reset calibration to defaults

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_core::types::Message;
use opensquilla_provider::{ChatConfig, Provider, ProviderSpecTable};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

use crate::table::{self, Alignment, Color, Column, KeyValue, Style, Stylize, Table};
use crate::util;

/// Router subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum RouterAction {
    /// Run calibration to score model tiers.
    Calibrate {
        /// Number of probes per model.
        probes: usize,
        /// Only calibrate a specific tier.
        tier: Option<String>,
    },
    /// Show the current calibration state.
    Status,
    /// List configured routing tiers.
    Tiers,
    /// Show the effective routing policy.
    Policy,
    /// Reset calibration to defaults.
    Reset,
    /// Simulate a routing decision for a task description.
    Test {
        /// A description of the task to route.
        task: String,
        /// Force a specific tier.
        tier: Option<String>,
    },
    /// Show which model would be selected for a task.
    Select {
        task: String,
    },
}

/// A model tier definition for routing.
#[derive(Debug, Clone, Serialize)]
struct Tier {
    name: String,
    models: Vec<String>,
    max_tokens: u64,
    cost_per_1m: f64,
    avg_latency_ms: u64,
    success_rate: f64,
    score: f64,
}

/// Calibration result for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationResult {
    model: String,
    tier: String,
    latency_ms: u64,
    tokens: u64,
    cost_usd: f64,
    success: bool,
    quality_score: f64,
}

/// Run a router subcommand.
pub async fn run_router(action: RouterAction) -> Result<()> {
    match action {
        RouterAction::Calibrate { probes, tier } => calibrate(probes, tier).await,
        RouterAction::Status => router_status().await,
        RouterAction::Tiers => list_tiers().await,
        RouterAction::Policy => show_policy().await,
        RouterAction::Reset => reset_calibration().await,
        RouterAction::Test { task, tier } => simulate_decision(&task, tier).await,
        RouterAction::Select { task } => select_model(&task).await,
    }
}

/// Simulate a routing decision for a task description.
pub async fn simulate_decision(task: &str, tier: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let tiers = build_tiers(&config, &registry);

    println!("Routing Simulation");
    println!();
    KeyValue::new()
        .entry("Task", task.to_string())
        .entry(
            "Routing enabled",
            if is_routing_enabled(&config) { "yes" } else { "no (static selection)" }.to_string(),
        )
        .entry(
            "Forced tier",
            tier.clone().unwrap_or_else(|| "(auto)".to_string()),
        )
        .print();
    println!();

    // Score the task.
    let (complexity, task_type) = classify_task(task);
    let calibration = load_calibration(&config)?;

    println!("Task classification:");
    KeyValue::new()
        .entry("Complexity", format!("{complexity:.2}/1.00"))
        .entry("Type", task_type.to_string())
        .print();
    println!();

    // Pick the tier.
    let selected_tier = match &tier {
        Some(name) => tiers.iter().find(|t| &t.name == name).cloned(),
        None => choose_tier(&tiers, complexity),
    };

    match selected_tier {
        Some(t) => {
            println!("{} Selected tier: {}", "→".green().bold(), t.name);
            println!("  Models: {}", t.models.join(", "));
            println!();

            // Show per-model decision within the tier.
            println!("Model selection within tier:");
            let mut table = Table::from_headers(&["Model", "Latency", "Cost/1M", "Calibrated"])
                .border(table::TableBorder::Header);
            for model in &t.models {
                let cal = calibration.iter().find(|c| &c.model == model);
                let latency = cal.map(|c| format!("{}ms", c.latency_ms)).unwrap_or_else(|| "—".into());
                let cost = cal.map(|c| format!("${:.4}", c.cost_usd)).unwrap_or_else(|| "—".into());
                let calibrated = if cal.is_some() { "yes" } else { "no" };
                table = table.row_owned(vec![
                    model.clone(),
                    latency,
                    cost,
                    calibrated.to_string(),
                ]);
            }
            table.print();

            // Recommendation.
            let recommended = t.models.first().cloned().unwrap_or_default();
            println!();
            println!("{} Recommended: {recommended}", table::ok());
            Ok(())
        }
        None => {
            println!("{} No routing tier available for this task.", table::warn());
            println!("Run 'osq router calibrate' to build tier data.");
            Ok(())
        }
    }
}

/// Show which model would be selected for a task.
pub async fn select_model(task: &str) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let tiers = build_tiers(&config, &registry);
    let (complexity, task_type) = classify_task(task);

    let selected = choose_tier(&tiers, complexity);
    match selected {
        Some(t) => {
            let model = t
                .models
                .first()
                .cloned()
                .unwrap_or_else(|| util::default_model(&config));
            println!("Task:        {task}");
            println!("Type:        {task_type}");
            println!("Complexity:  {complexity:.2}/1.00");
            println!("Tier:        {}", t.name);
            println!("Model:       {model}");
            Ok(())
        }
        None => {
            println!("Task:        {task}");
            println!("Model:       {}", util::default_model(&config));
            println!("(routing disabled — using default model)");
            Ok(())
        }
    }
}

/// Classify a task by complexity and type on a 0..1 scale.
fn classify_task(task: &str) -> (f64, &'static str) {
    let lower = task.to_lowercase();
    let mut score = 0.0f64;

    let complexity_keywords = [
        "complex",
        "difficult",
        "analy",
        "research",
        "multi",
        "architect",
        "design",
        "debug",
        "optimiz",
        "comprehensive",
        "thorough",
        "deep",
        "reasoning",
        "math",
        "algorithm",
    ];
    for kw in &complexity_keywords {
        if lower.contains(kw) {
            score += 0.15;
        }
    }

    let simple_keywords = [
        "hello",
        "hi",
        "thanks",
        "yes",
        "no",
        "simple",
        "quick",
        "short",
        "summarize in one",
        "what is 2",
    ];
    for kw in &simple_keywords {
        if lower.contains(kw) {
            score -= 0.2;
        }
    }

    let task_type = if lower.contains("code") || lower.contains("program") || lower.contains("debug") {
        "code"
    } else if lower.contains("write") || lower.contains("draft") || lower.contains("email") {
        "writing"
    } else if lower.contains("summar") || lower.contains("extract") {
        "summarization"
    } else if lower.contains("math") || lower.contains("equation") {
        "math"
    } else {
        "general"
    };

    (score.clamp(0.0, 1.0), task_type)
}

/// Choose the best tier for a complexity score.
fn choose_tier(tiers: &[Tier], complexity: f64) -> Option<Tier> {
    if tiers.is_empty() {
        return None;
    }
    if complexity >= 0.6 {
        tiers.iter().find(|t| t.name == "powerful").cloned()
    } else if complexity >= 0.3 {
        tiers.iter().find(|t| t.name == "balanced").cloned()
    } else {
        tiers.iter().find(|t| t.name == "fast").cloned()
    }
    .or_else(|| tiers.first().cloned())
}

/// Run calibration probes against configured models.
pub async fn calibrate(probes: usize, tier_filter: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;

    println!("Router Calibration");
    println!();
    KeyValue::new()
        .entry("Probes per model", probes.to_string())
        .entry("Tier filter", tier_filter.clone().unwrap_or_else(|| "all".into()))
        .print();
    println!();

    let tiers = build_tiers(&config, &registry);
    let target_tiers: Vec<&Tier> = match &tier_filter {
        Some(name) => tiers.iter().filter(|t| &t.name == name).collect(),
        None => tiers.iter().collect(),
    };

    if target_tiers.is_empty() {
        println!("No matching tiers found.");
        return Ok(());
    }

    let mut results = Vec::new();

    for tier in &target_tiers {
        println!("{}", format!("Tier: {} ({})", tier.name, tier.models.join(", ")).bold());
        for model_name in &tier.models {
            let mut model_results = Vec::new();
            for probe_idx in 0..probes {
                let prompt = calibration_prompt(probe_idx);
                let result = run_probe(&registry, model_name, &prompt).await;
                model_results.push(result);
            }

            let successes: u64 = model_results.iter().filter(|r| r.success).count() as u64;
            let avg_latency = if model_results.is_empty() {
                0
            } else {
                model_results.iter().map(|r| r.latency_ms).sum::<u64>() / model_results.len() as u64
            };
            let total_tokens: u64 = model_results.iter().map(|r| r.tokens).sum();
            let total_cost: f64 = model_results.iter().map(|r| r.cost_usd).sum();
            let success_rate = if probes > 0 {
                successes as f64 / probes as f64
            } else {
                0.0
            };
            let quality = if !model_results.is_empty() {
                model_results.iter().map(|r| r.quality_score).sum::<f64>()
                    / model_results.len() as f64
            } else {
                0.0
            };

            println!(
                "  {} {:<24} latency={:>5}ms  success={:>5.1}%  quality={:.2}  cost=${:.4}",
                if success_rate > 0.8 { table::ok() } else { table::warn() },
                model_name,
                avg_latency,
                success_rate * 100.0,
                quality,
                total_cost
            );

            results.push(CalibrationResult {
                model: model_name.clone(),
                tier: tier.name.clone(),
                latency_ms: avg_latency,
                tokens: total_tokens,
                cost_usd: total_cost,
                success: success_rate > 0.5,
                quality_score: quality,
            });
        }
        println!();
    }

    // Save calibration state.
    save_calibration(&config, &results)?;

    // Print summary table.
    println!("{}", "Calibration Results".bold());
    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("Model").max_width(28))
        .column(Column::new("Tier"))
        .column(Column::new("Latency").align(Alignment::Right))
        .column(Column::new("Tokens").align(Alignment::Right))
        .column(Column::new("Cost").align(Alignment::Right))
        .column(Column::new("Quality").align(Alignment::Right));

    for r in &results {
        table = table.row_owned(vec![
            r.model.clone(),
            r.tier.clone(),
            format!("{}ms", r.latency_ms),
            r.tokens.to_string(),
            format!("${:.4}", r.cost_usd),
            format!("{:.2}", r.quality_score),
        ]);
    }
    table.print();

    info!("Calibration complete: {} probes", results.len());
    Ok(())
}

/// Show the current calibration state.
pub async fn router_status() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let cal = load_calibration(&config)?;

    println!("Router Status");
    println!();
    KeyValue::new()
        .entry("Routing enabled", is_routing_enabled(&config).to_string())
        .entry("Calibrated models", cal.len().to_string())
        .entry(
            "Last calibration",
            config
                .get("router.last_calibration")
                .unwrap_or_else(|| "(never)".to_string()),
        )
        .print();

    if cal.is_empty() {
        println!();
        println!("Router is not calibrated. Run 'osq router calibrate' to score models.");
        return Ok(());
    }

    println!();
    let mut table = Table::from_headers(&["Model", "Tier", "Latency", "Success", "Quality"])
        .border(table::TableBorder::Header);
    for r in &cal {
        table = table.row_owned(vec![
            r.model.clone(),
            r.tier.clone(),
            format!("{}ms", r.latency_ms),
            if r.success { "✓" } else { "✗" }.to_string(),
            format!("{:.2}", r.quality_score),
        ]);
    }
    table.print();
    Ok(())
}

/// List configured routing tiers.
pub async fn list_tiers() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let tiers = build_tiers(&config, &registry);

    if tiers.is_empty() {
        println!("No routing tiers configured.");
        return Ok(());
    }

    println!("Routing Tiers ({})", tiers.len());
    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("Tier"))
        .column(Column::new("Models").max_width(40))
        .column(Column::new("Max Tokens").align(Alignment::Right))
        .column(Column::new("Cost/1M").align(Alignment::Right))
        .column(Column::new("Avg Latency").align(Alignment::Right))
        .column(Column::new("Score").align(Alignment::Right));

    for t in &tiers {
        table = table.row_owned(vec![
            t.name.clone(),
            t.models.join(", "),
            t.max_tokens.to_string(),
            format!("${:.2}", t.cost_per_1m),
            format!("{}ms", t.avg_latency_ms),
            format!("{:.2}", t.score),
        ]);
    }
    table.print();
    Ok(())
}

/// Show the effective routing policy.
pub async fn show_policy() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;

    println!("Routing Policy");
    println!();
    KeyValue::new()
        .entry("Enabled", is_routing_enabled(&config).to_string())
        .entry(
            "Strategy",
            config
                .get("router.strategy")
                .unwrap_or_else(|| "balanced".to_string()),
        )
        .entry(
            "Fallback model",
            config
                .get("router.fallback_model")
                .unwrap_or_else(|| "(none)".to_string()),
        )
        .entry(
            "Min confidence",
            config
                .get("router.min_confidence")
                .unwrap_or_else(|| "0.5".to_string()),
        )
        .entry(
            "Cost weight",
            config
                .get("router.cost_weight")
                .unwrap_or_else(|| "0.3".to_string()),
        )
        .entry(
            "Latency weight",
            config
                .get("router.latency_weight")
                .unwrap_or_else(|| "0.3".to_string()),
        )
        .entry(
            "Quality weight",
            config
                .get("router.quality_weight")
                .unwrap_or_else(|| "0.4".to_string()),
        )
        .print();
    Ok(())
}

/// Reset calibration to defaults.
pub async fn reset_calibration() -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;
    config.remove("router.last_calibration");
    config.remove("router.calibration_data");
    config.save().context("Failed to save configuration")?;
    println!("{} Calibration reset.", table::ok());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the routing tiers from config and registry.
fn build_tiers(
    config: &Config,
    registry: &opensquilla_provider::ProviderRegistry,
) -> Vec<Tier> {
    let mut tiers = Vec::new();

    // Try to read tiers from config.
    let tier_names = ["fast", "balanced", "powerful"];
    for name in &tier_names {
        let key = format!("router.tiers.{name}.models");
        if let Some(models_str) = config.get(&key) {
            let models: Vec<String> = models_str.split(',').map(|s| s.trim().to_string()).collect();
            if !models.is_empty() {
                tiers.push(Tier {
                    name: name.to_string(),
                    models,
                    max_tokens: config
                        .get(&format!("router.tiers.{name}.max_tokens"))
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4096),
                    cost_per_1m: config
                        .get(&format!("router.tiers.{name}.cost_per_1m"))
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    avg_latency_ms: 0,
                    success_rate: 1.0,
                    score: 0.0,
                });
            }
        }
    }

    // If no tiers configured, derive from registry.
    if tiers.is_empty() {
        let all_models: Vec<String> = registry
            .list()
            .iter()
            .filter_map(|n| registry.get(n).map(|p| p.supported_models()))
            .flatten()
            .collect();

        if !all_models.is_empty() {
            let midpoint = all_models.len() / 3.max(1);
            let chunks: Vec<Vec<String>> = vec![
                all_models[..midpoint.min(all_models.len())].to_vec(),
                all_models[midpoint.min(all_models.len())..(midpoint * 2).min(all_models.len())]
                    .to_vec(),
                all_models[(midpoint * 2).min(all_models.len())..].to_vec(),
            ];
            for (i, models) in chunks.into_iter().enumerate() {
                if !models.is_empty() {
                    tiers.push(Tier {
                        name: tier_names[i].to_string(),
                        models,
                        max_tokens: 4096,
                        cost_per_1m: 0.0,
                        avg_latency_ms: 0,
                        success_rate: 1.0,
                        score: 0.0,
                    });
                }
            }
        }
    }

    // Fallback to the provider spec table.
    if tiers.is_empty() {
        if let Some(spec) = ProviderSpecTable::get("openai") {
            tiers.push(Tier {
                name: "fast".to_string(),
                models: vec![spec.default_model.to_string()],
                max_tokens: 4096,
                cost_per_1m: 0.0,
                avg_latency_ms: 0,
                success_rate: 1.0,
                score: 0.0,
            });
        }
    }

    tiers
}

/// Check if routing is enabled in config.
fn is_routing_enabled(config: &Config) -> bool {
    config
        .get("router.enabled")
        .map(|v| v == "true")
        .unwrap_or(false)
}

/// Get a calibration prompt for a given probe index.
fn calibration_prompt(index: usize) -> String {
    let prompts = [
        "What is 2+2? Reply with just the number.",
        "Name the capital of France. One word answer.",
        "What color is the sky? One word.",
        "Complete this: The opposite of 'hot' is ___.",
        "What is 10 divided by 2? Reply with just the number.",
    ];
    prompts[index % prompts.len()].to_string()
}

/// Run a single calibration probe.
async fn run_probe(
    registry: &opensquilla_provider::ProviderRegistry,
    model: &str,
    prompt: &str,
) -> CalibrationResult {
    // Find a provider that supports this model.
    let provider = registry
        .list()
        .iter()
        .find_map(|name| {
            registry.get(name).and_then(|p| {
                if p.supported_models().iter().any(|m| m == model) {
                    Some(p)
                } else {
                    None
                }
            })
        });

    let Some(provider) = provider else {
        return CalibrationResult {
            model: model.to_string(),
            tier: String::new(),
            latency_ms: 0,
            tokens: 0,
            cost_usd: 0.0,
            success: false,
            quality_score: 0.0,
        };
    };

    let chat_config = ChatConfig {
        model: model.to_string(),
        temperature: 0.0,
        max_tokens: 64,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: false,
        extra: Default::default(),
    };

    let start = Instant::now();
    match provider
        .send_message(&chat_config, &[Message::user(prompt)], &[])
        .await
    {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            let reply = resp
                .content
                .iter()
                .map(|m| m.text_content())
                .collect::<Vec<_>>()
                .join(" ");
            let tokens = resp.usage.total_tokens;
            let cost = estimate_cost(model, tokens);
            let quality = score_quality(prompt, &reply);
            CalibrationResult {
                model: model.to_string(),
                tier: String::new(),
                latency_ms,
                tokens,
                cost_usd: cost,
                success: true,
                quality_score: quality,
            }
        }
        Err(e) => {
            tracing::warn!("Probe failed for {model}: {e}");
            CalibrationResult {
                model: model.to_string(),
                tier: String::new(),
                latency_ms: start.elapsed().as_millis() as u64,
                tokens: 0,
                cost_usd: 0.0,
                success: false,
                quality_score: 0.0,
            }
        }
    }
}

/// Estimate cost for a number of tokens.
fn estimate_cost(model: &str, tokens: u64) -> f64 {
    // Rough pricing per 1M tokens.
    let price_per_1m = if model.contains("gpt-4") {
        30.0
    } else if model.contains("gpt-3.5") || model.contains("gpt-4o-mini") {
        0.5
    } else if model.contains("claude-3") {
        15.0
    } else if model.contains("haiku") {
        0.25
    } else {
        5.0
    };
    tokens as f64 / 1_000_000.0 * price_per_1m
}

/// Score the quality of a response on a 0-1 scale.
fn score_quality(prompt: &str, reply: &str) -> f64 {
    let reply = reply.trim().to_lowercase();
    if reply.is_empty() {
        return 0.0;
    }
    // Simple heuristic: check if the reply contains the expected answer.
    if prompt.contains("2+2") && reply.contains('4') {
        return 1.0;
    }
    if prompt.contains("capital of France") && reply.contains("paris") {
        return 1.0;
    }
    if prompt.contains("color is the sky") && reply.contains("blue") {
        return 1.0;
    }
    if prompt.contains("opposite of 'hot'") && reply.contains("cold") {
        return 1.0;
    }
    if prompt.contains("10 divided by 2") && reply.contains('5') {
        return 1.0;
    }
    // Partial credit for non-empty responses.
    if reply.len() > 2 {
        0.5
    } else {
        0.1
    }
}

/// Save calibration data to config.
fn save_calibration(
    config: &Config,
    results: &[CalibrationResult],
) -> Result<()> {
    let mut config = config.clone();
    let now = chrono::Utc::now().to_rfc3339();
    config.set("router.last_calibration", &now).ok();
    let json = serde_json::to_string(results).unwrap_or_else(|_| "[]".to_string());
    config.set("router.calibration_data", &json).ok();
    config.save().context("Failed to save calibration")?;
    Ok(())
}

/// Load calibration data from config.
fn load_calibration(config: &Config) -> Result<Vec<CalibrationResult>> {
    match config.get("router.calibration_data") {
        Some(json) => {
            serde_json::from_str(&json).context("Failed to parse calibration data")
        }
        None => Ok(Vec::new()),
    }
}

/// Bold helper.
trait BoldStr {
    fn bold(&self) -> String;
}

impl BoldStr for &str {
    fn bold(&self) -> String {
        format!("{}", Style::new().bold().fg(Color::BrightBlue).styled(*self))
    }
}

impl BoldStr for String {
    fn bold(&self) -> String {
        format!("{}", Style::new().bold().fg(Color::BrightBlue).styled(self))
    }
}
