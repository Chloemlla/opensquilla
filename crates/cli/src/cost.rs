//! # Cost and usage commands
//!
//! Implements the `cost` subcommand for querying token usage and spending
//! across sessions, providers, and models. Data comes from the session store
//! (which records per-message token counts and cost estimates) and is
//! optionally enriched via the gateway RPC.
//!
//! Subcommands:
//! - `cost summary` — total spend and token usage over a time range
//! - `cost usage` — per-session or per-provider breakdown
//! - `cost report` — a full report saved to a file
//! - `cost budget` — show or set a monthly budget
//! - `cost rates` — show the pricing table used for estimates

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Duration, Utc};
use opensquilla_core::config::Config;
use opensquilla_provider::model_catalog::{ModelCatalog, seed_static};
use serde::Serialize;
use tracing::info;

use crate::table::{Alignment, Color, Column, KeyValue, Style, Table};
use crate::util;

/// Cost subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum CostAction {
    /// Show a summary of spending over a time range.
    Summary {
        start: Option<String>,
        end: Option<String>,
    },
    /// Show a breakdown by session, provider, or model.
    Usage { group_by: String, limit: usize },
    /// Generate a full report and save it.
    Report {
        output: String,
        start: Option<String>,
        end: Option<String>,
    },
    /// Show or set the monthly budget.
    Budget { amount: Option<f64> },
    /// Show the pricing rates table.
    Rates,
}

/// A row in the cost summary.
#[derive(Serialize, Debug)]
struct CostRow {
    date: String,
    sessions: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cost_usd: f64,
}

/// Full cost report envelope.
#[derive(Serialize)]
struct CostReport {
    generated_at: DateTime<Utc>,
    range_start: Option<String>,
    range_end: Option<String>,
    summary: CostSummary,
    by_session: Vec<SessionCost>,
    by_provider: Vec<ProviderCost>,
    by_model: Vec<ModelCost>,
}

/// Aggregate cost summary.
#[derive(Serialize, Default)]
struct CostSummary {
    total_sessions: u64,
    total_messages: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_tokens: u64,
    total_cost_usd: f64,
    average_cost_per_session: f64,
    average_tokens_per_session: f64,
}

/// Cost broken down by session.
#[derive(Serialize)]
struct SessionCost {
    session_id: String,
    name: String,
    messages: u64,
    tokens: u64,
    cost_usd: f64,
}

/// Cost broken down by provider.
#[derive(Serialize)]
struct ProviderCost {
    provider: String,
    sessions: u64,
    tokens: u64,
    cost_usd: f64,
    pct: f64,
}

/// Cost broken down by model.
#[derive(Serialize)]
struct ModelCost {
    model: String,
    tokens: u64,
    cost_usd: f64,
    pct: f64,
}

/// Run a cost subcommand.
pub async fn run_cost(action: CostAction) -> Result<()> {
    match action {
        CostAction::Summary { start, end } => cost_summary(start, end).await,
        CostAction::Usage { group_by, limit } => cost_usage(&group_by, limit).await,
        CostAction::Report { output, start, end } => cost_report(output, start, end).await,
        CostAction::Budget { amount } => cost_budget(amount).await,
        CostAction::Rates => cost_rates().await,
    }
}

/// Show a summary of token usage and cost over a time range.
pub async fn cost_summary(start: Option<String>, end: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let (start_dt, end_dt) = parse_range(start.as_deref(), end.as_deref())?;

    let sessions = manager
        .list_sessions(&util::default_agent_id(), 1000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    let mut summary = CostSummary::default();
    let mut daily: std::collections::BTreeMap<String, CostRow> = std::collections::BTreeMap::new();

    for session in &sessions {
        if session.created_at < start_dt || session.created_at > end_dt {
            continue;
        }
        summary.total_sessions += 1;
        summary.total_messages += session.message_count;
        summary.total_tokens += session.total_tokens;
        summary.total_cost_usd += session.total_cost_usd;

        let date = session.created_at.format("%Y-%m-%d").to_string();
        let row = daily.entry(date).or_insert_with(|| CostRow {
            date: String::new(),
            sessions: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_usd: 0.0,
        });
        row.date = session.created_at.format("%Y-%m-%d").to_string();
        row.sessions += 1;
        row.total_tokens += session.total_tokens;
        row.cost_usd += session.total_cost_usd;
    }

    summary.average_cost_per_session = if summary.total_sessions > 0 {
        summary.total_cost_usd / summary.total_sessions as f64
    } else {
        0.0
    };
    summary.average_tokens_per_session = if summary.total_sessions > 0 {
        summary.total_tokens as f64 / summary.total_sessions as f64
    } else {
        0.0
    };

    println!("Cost Summary");
    println!(
        "Range: {} to {}",
        start_dt.format("%Y-%m-%d"),
        end_dt.format("%Y-%m-%d")
    );
    println!();

    KeyValue::new()
        .entry("Sessions", summary.total_sessions.to_string())
        .entry("Messages", summary.total_messages.to_string())
        .entry("Input tokens", summary.total_input_tokens.to_string())
        .entry("Output tokens", summary.total_output_tokens.to_string())
        .entry("Total tokens", summary.total_tokens.to_string())
        .entry_styled(
            "Total cost",
            format!("${:.4}", summary.total_cost_usd),
            Style::new().fg(Color::Green).bold(),
        )
        .entry(
            "Avg cost/session",
            format!("${:.4}", summary.average_cost_per_session),
        )
        .entry(
            "Avg tokens/session",
            format!("{:.0}", summary.average_tokens_per_session),
        )
        .print();

    if !daily.is_empty() {
        println!();
        let mut table = Table::from_headers(&["Date", "Sessions", "Tokens", "Cost (USD)"])
            .border(crate::table::TableBorder::Header);
        for row in daily.values() {
            table = table.row_owned(vec![
                row.date.clone(),
                row.sessions.to_string(),
                row.total_tokens.to_string(),
                format!("{:.4}", row.cost_usd),
            ]);
        }
        table.print();
    }

    Ok(())
}

/// Show usage broken down by session, provider, or model.
pub async fn cost_usage(group_by: &str, limit: usize) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;

    let sessions = manager
        .list_sessions(&util::default_agent_id(), 1000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    match group_by {
        "session" | "sessions" => usage_by_session(&sessions, limit),
        "provider" | "providers" => usage_by_provider(&sessions, &config, limit),
        "model" | "models" => usage_by_model(&sessions, &config, limit),
        "day" | "daily" => usage_by_day(&sessions, limit),
        other => anyhow::bail!("Unknown group_by '{other}'. Use: session, provider, model, or day"),
    }
}

/// Group usage by session.
fn usage_by_session(sessions: &[opensquilla_session::Session], limit: usize) -> Result<()> {
    let mut rows: Vec<SessionCost> = sessions
        .iter()
        .map(|s| SessionCost {
            session_id: s.id.to_string(),
            name: s.name.clone(),
            messages: s.message_count,
            tokens: s.total_tokens,
            cost_usd: s.total_cost_usd,
        })
        .collect();
    rows.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);

    println!("Usage by Session (top {limit})");
    let mut table = Table::new()
        .border(crate::table::TableBorder::Header)
        .column(Column::new("Session ID").max_width(36))
        .column(Column::new("Name").max_width(24))
        .column(Column::new("Msgs").align(Alignment::Right))
        .column(Column::new("Tokens").align(Alignment::Right))
        .column(Column::new("Cost (USD)").align(Alignment::Right));

    for r in &rows {
        table = table.row_owned(vec![
            r.session_id.clone(),
            r.name.clone(),
            r.messages.to_string(),
            r.tokens.to_string(),
            format!("{:.4}", r.cost_usd),
        ]);
    }
    table.print();
    Ok(())
}

/// Group usage by provider.
fn usage_by_provider(
    sessions: &[opensquilla_session::Session],
    _config: &Config,
    limit: usize,
) -> Result<()> {
    let mut by_provider: std::collections::HashMap<String, ProviderCost> =
        std::collections::HashMap::new();
    let mut total_cost = 0.0f64;

    for s in sessions {
        // The session's metadata may carry a provider; otherwise attribute to "unknown".
        let provider = s
            .metadata
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let entry = by_provider.entry(provider.clone()).or_insert(ProviderCost {
            provider,
            sessions: 0,
            tokens: 0,
            cost_usd: 0.0,
            pct: 0.0,
        });
        entry.sessions += 1;
        entry.tokens += s.total_tokens;
        entry.cost_usd += s.total_cost_usd;
        total_cost += s.total_cost_usd;
    }

    let mut rows: Vec<ProviderCost> = by_provider.into_values().collect();
    for r in &mut rows {
        r.pct = if total_cost > 0.0 {
            r.cost_usd / total_cost * 100.0
        } else {
            0.0
        };
    }
    rows.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);

    println!("Usage by Provider (top {limit})");
    let mut table = Table::new()
        .border(crate::table::TableBorder::Header)
        .column(Column::new("Provider"))
        .column(Column::new("Sessions").align(Alignment::Right))
        .column(Column::new("Tokens").align(Alignment::Right))
        .column(Column::new("Cost (USD)").align(Alignment::Right))
        .column(Column::new("Share").align(Alignment::Right));

    for r in &rows {
        table = table.row_owned(vec![
            r.provider.clone(),
            r.sessions.to_string(),
            r.tokens.to_string(),
            format!("{:.4}", r.cost_usd),
            format!("{:.1}%", r.pct),
        ]);
    }
    table.print();
    Ok(())
}

/// Group usage by model.
fn usage_by_model(
    sessions: &[opensquilla_session::Session],
    _config: &Config,
    limit: usize,
) -> Result<()> {
    let mut by_model: std::collections::HashMap<String, ModelCost> =
        std::collections::HashMap::new();
    let mut total_cost = 0.0f64;

    for s in sessions {
        let model = s
            .metadata
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entry = by_model.entry(model.clone()).or_insert(ModelCost {
            model,
            tokens: 0,
            cost_usd: 0.0,
            pct: 0.0,
        });
        entry.tokens += s.total_tokens;
        entry.cost_usd += s.total_cost_usd;
        total_cost += s.total_cost_usd;
    }

    let mut rows: Vec<ModelCost> = by_model.into_values().collect();
    for r in &mut rows {
        r.pct = if total_cost > 0.0 {
            r.cost_usd / total_cost * 100.0
        } else {
            0.0
        };
    }
    rows.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);

    println!("Usage by Model (top {limit})");
    let mut table = Table::new()
        .border(crate::table::TableBorder::Header)
        .column(Column::new("Model"))
        .column(Column::new("Tokens").align(Alignment::Right))
        .column(Column::new("Cost (USD)").align(Alignment::Right))
        .column(Column::new("Share").align(Alignment::Right));

    for r in &rows {
        table = table.row_owned(vec![
            r.model.clone(),
            r.tokens.to_string(),
            format!("{:.4}", r.cost_usd),
            format!("{:.1}%", r.pct),
        ]);
    }
    table.print();
    Ok(())
}

/// Group usage by day.
fn usage_by_day(sessions: &[opensquilla_session::Session], limit: usize) -> Result<()> {
    let mut by_day: std::collections::BTreeMap<String, CostRow> = std::collections::BTreeMap::new();

    for s in sessions {
        let date = s.created_at.format("%Y-%m-%d").to_string();
        let row = by_day.entry(date.clone()).or_insert_with(|| CostRow {
            date,
            sessions: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cost_usd: 0.0,
        });
        row.sessions += 1;
        row.total_tokens += s.total_tokens;
        row.cost_usd += s.total_cost_usd;
    }

    println!("Usage by Day (last {limit} days)");
    let mut table = Table::from_headers(&["Date", "Sessions", "Tokens", "Cost (USD)"])
        .border(crate::table::TableBorder::Header);

    let rows: Vec<&CostRow> = by_day.values().rev().take(limit).collect();
    for row in rows {
        table = table.row_owned(vec![
            row.date.clone(),
            row.sessions.to_string(),
            row.total_tokens.to_string(),
            format!("{:.4}", row.cost_usd),
        ]);
    }
    table.print();
    Ok(())
}

/// Generate a full cost report and save it as JSON.
pub async fn cost_report(output: String, start: Option<String>, end: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let (start_dt, end_dt) = parse_range(start.as_deref(), end.as_deref())?;

    let sessions = manager
        .list_sessions(&util::default_agent_id(), 1000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    let mut summary = CostSummary::default();
    let mut session_costs: Vec<SessionCost> = Vec::new();
    let mut provider_map: std::collections::HashMap<String, ProviderCost> =
        std::collections::HashMap::new();
    let mut model_map: std::collections::HashMap<String, ModelCost> =
        std::collections::HashMap::new();
    let mut total_cost = 0.0f64;

    for s in &sessions {
        if s.created_at < start_dt || s.created_at > end_dt {
            continue;
        }
        summary.total_sessions += 1;
        summary.total_messages += s.message_count;
        summary.total_tokens += s.total_tokens;
        summary.total_cost_usd += s.total_cost_usd;
        total_cost += s.total_cost_usd;

        session_costs.push(SessionCost {
            session_id: s.id.to_string(),
            name: s.name.clone(),
            messages: s.message_count,
            tokens: s.total_tokens,
            cost_usd: s.total_cost_usd,
        });

        let provider = s
            .metadata
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let p = provider_map
            .entry(provider.clone())
            .or_insert(ProviderCost {
                provider,
                sessions: 0,
                tokens: 0,
                cost_usd: 0.0,
                pct: 0.0,
            });
        p.sessions += 1;
        p.tokens += s.total_tokens;
        p.cost_usd += s.total_cost_usd;

        let model = s
            .metadata
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let m = model_map.entry(model.clone()).or_insert(ModelCost {
            model,
            tokens: 0,
            cost_usd: 0.0,
            pct: 0.0,
        });
        m.tokens += s.total_tokens;
        m.cost_usd += s.total_cost_usd;
    }

    summary.average_cost_per_session = if summary.total_sessions > 0 {
        summary.total_cost_usd / summary.total_sessions as f64
    } else {
        0.0
    };
    summary.average_tokens_per_session = if summary.total_sessions > 0 {
        summary.total_tokens as f64 / summary.total_sessions as f64
    } else {
        0.0
    };

    let mut providers: Vec<ProviderCost> = provider_map.into_values().collect();
    for p in &mut providers {
        p.pct = if total_cost > 0.0 {
            p.cost_usd / total_cost * 100.0
        } else {
            0.0
        };
    }
    providers.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut models: Vec<ModelCost> = model_map.into_values().collect();
    for m in &mut models {
        m.pct = if total_cost > 0.0 {
            m.cost_usd / total_cost * 100.0
        } else {
            0.0
        };
    }
    models.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    session_costs.sort_by(|a, b| {
        b.cost_usd
            .partial_cmp(&a.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let report = CostReport {
        generated_at: Utc::now(),
        range_start: start,
        range_end: end,
        summary,
        by_session: session_costs,
        by_provider: providers,
        by_model: models,
    };

    let json = serde_json::to_string_pretty(&report).context("Failed to serialize report")?;

    if let Some(dir) = Path::new(&output).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&output, json).with_context(|| format!("Failed to write {output}"))?;
    info!("Cost report saved to {output}");
    println!("Cost report saved to {output}");
    Ok(())
}

/// Show or set the monthly budget.
pub async fn cost_budget(amount: Option<f64>) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;

    match amount {
        Some(a) => {
            config.set("cost.monthly_budget_usd", &a.to_string()).ok();
            config.save().context("Failed to save config")?;
            println!("Monthly budget set to ${a:.2}");
        }
        None => {
            let budget = config
                .get("cost.monthly_budget_usd")
                .and_then(|v| v.parse::<f64>().ok());
            match budget {
                Some(b) => {
                    let now = Utc::now();
                    let month_start = now
                        .date_naive()
                        .with_day(1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc();
                    let manager = util::build_session_manager(&config)?;
                    let sessions = manager
                        .list_sessions(&util::default_agent_id(), 1000, 0)
                        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;
                    let spent: f64 = sessions
                        .iter()
                        .filter(|s| s.created_at >= month_start)
                        .map(|s| s.total_cost_usd)
                        .sum();
                    let remaining = (b - spent).max(0.0);
                    let pct = if b > 0.0 { spent / b * 100.0 } else { 0.0 };

                    println!("Monthly Budget");
                    KeyValue::new()
                        .entry("Budget", format!("${b:.2}"))
                        .entry_styled(
                            "Spent",
                            format!("${spent:.2}"),
                            if pct > 80.0 {
                                Style::new().fg(Color::Red).bold()
                            } else if pct > 50.0 {
                                Style::new().fg(Color::Yellow).bold()
                            } else {
                                Style::new().fg(Color::Green).bold()
                            },
                        )
                        .entry("Remaining", format!("${remaining:.2}"))
                        .entry("Used", format!("{pct:.1}%"))
                        .print();

                    if pct > 100.0 {
                        println!();
                        println!(
                            "{} Budget exceeded by ${:.2}",
                            crate::table::fail(),
                            spent - b
                        );
                    } else if pct > 80.0 {
                        println!();
                        println!("{} Approaching budget limit", crate::table::warn());
                    }
                }
                None => {
                    println!("No monthly budget configured.");
                    println!("Set one with: osq cost budget <amount>");
                }
            }
        }
    }
    Ok(())
}

/// Show the pricing rates table used for cost estimates.
pub async fn cost_rates() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let catalog = ModelCatalog::new();

    println!("Pricing Rates (per 1M tokens)");
    println!();

    let mut table = Table::new()
        .border(crate::table::TableBorder::Header)
        .column(Column::new("Backend"))
        .column(Column::new("Model").max_width(30))
        .column(Column::new("Input $/1M").align(Alignment::Right))
        .column(Column::new("Output $/1M").align(Alignment::Right));

    for name in registry.list() {
        if let Some(p) = registry.get(&name) {
            let backend = p.name();
            seed_static(&catalog, backend);
            for model in p.supported_models() {
                if let Some(caps) = catalog.get(backend, &model) {
                    let input = caps
                        .input_price_per_million
                        .map(|p| format!("${p:.2}"))
                        .unwrap_or_else(|| "—".to_string());
                    let output = caps
                        .output_price_per_million
                        .map(|p| format!("${p:.2}"))
                        .unwrap_or_else(|| "—".to_string());
                    table = table.row_owned(vec![backend.to_string(), model, input, output]);
                }
            }
        }
    }

    if let Some(spec) = opensquilla_provider::ProviderSpecTable::get(
        config
            .providers
            .first()
            .map(|p| p.provider_type.as_str())
            .unwrap_or("openai"),
    ) {
        seed_static(&catalog, spec.id);
        for model in spec.models {
            if let Some(caps) = catalog.get(spec.id, model) {
                let input = caps
                    .input_price_per_million
                    .map(|p| format!("${p:.2}"))
                    .unwrap_or_else(|| "—".to_string());
                let output = caps
                    .output_price_per_million
                    .map(|p| format!("${p:.2}"))
                    .unwrap_or_else(|| "—".to_string());
                table =
                    table.row_owned(vec![spec.id.to_string(), model.to_string(), input, output]);
            }
        }
    }

    table.print();
    Ok(())
}

/// Parse an optional date range, defaulting to the last 30 days.
fn parse_range(start: Option<&str>, end: Option<&str>) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    let end_dt = match end {
        Some(e) => parse_date(e)?,
        None => Utc::now(),
    };
    let start_dt = match start {
        Some(s) => parse_date(s)?,
        None => end_dt - Duration::days(30),
    };
    if start_dt > end_dt {
        anyhow::bail!("Start date must be before end date");
    }
    Ok((start_dt, end_dt))
}

/// Parse a date string in common formats.
fn parse_date(s: &str) -> Result<DateTime<Utc>> {
    // Try RFC3339 first.
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    // Try YYYY-MM-DD.
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    // Try relative offsets like "7d", "30d", "1h".
    if let Some(rest) = s.strip_suffix('d') {
        if let Ok(days) = rest.parse::<i64>() {
            return Ok(Utc::now() - Duration::days(days));
        }
    }
    if let Some(rest) = s.strip_suffix('h') {
        if let Ok(hours) = rest.parse::<i64>() {
            return Ok(Utc::now() - Duration::hours(hours));
        }
    }
    anyhow::bail!("Unrecognized date format: '{s}'. Use YYYY-MM-DD, RFC3339, or Nd/Nh")
}
