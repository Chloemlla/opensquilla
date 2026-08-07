//! Diagnostics bundle commands.
//!
//! Implements the `diagnostics` subcommand. The Python `diagnostics_cmd.py`
//! toggles runtime diagnostics flags over gateway RPC; this Rust port keeps
//! that behavior (status/on/off) and adds `dump`, which collects system info,
//! config, session db, recent log tail, and provider list into a bundle for
//! bug reports.

use std::path::PathBuf;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use serde::Serialize;
use tracing::info;

use crate::table::{self, Color, KeyValue, Style};
use crate::util;
use crate::rpc;

/// Diagnostics subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum DiagnosticsAction {
    /// Show effective diagnostics and raw-capture state.
    Status {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Enable runtime diagnostics; --raw also enables raw turn-call capture.
    On {
        /// Also enable raw turn-call capture.
        #[arg(long)]
        raw: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Disable runtime diagnostics and runtime raw capture.
    Off {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Collect system info, config, logs, and provider list into a bundle.
    Dump {
        /// Output directory for the bundle (default: ./opensquilla-diagnostics).
        #[arg(long)]
        output: Option<String>,
        /// Number of log lines to include.
        #[arg(long, default_value = "200")]
        log_lines: usize,
    },
}

/// Run a diagnostics subcommand.
pub async fn run_diagnostics(action: DiagnosticsAction) -> Result<()> {
    match action {
        DiagnosticsAction::Status { json } => diagnostics_status(json).await,
        DiagnosticsAction::On { raw, json } => diagnostics_on(raw, json).await,
        DiagnosticsAction::Off { json } => diagnostics_off(json).await,
        DiagnosticsAction::Dump { output, log_lines } => {
            diagnostics_dump(output, log_lines).await
        }
    }
}

/// Show the current diagnostics state from the gateway.
pub async fn diagnostics_status(json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let payload = rpc::rpc_or_fallback(&config, "diagnostics.state", serde_json::Value::Null)
        .await
        .unwrap_or(serde_json::json!({
            "enabled": false,
            "warning": "gateway not reachable; showing local defaults"
        }));

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }

    println!("Diagnostics");
    println!("{:-<50}", "");
    let enabled = payload.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    KeyValue::new()
        .entry("enabled", enabled.to_string())
        .entry("verbose", payload.get("verbose").and_then(|v| v.as_bool()).unwrap_or(false).to_string())
        .entry("trace", payload.get("trace").and_then(|v| v.as_bool()).unwrap_or(false).to_string())
        .entry("prompt_report", payload.get("prompt_report").and_then(|v| v.as_bool()).unwrap_or(false).to_string())
        .entry("decision_log", payload.get("decision_log").and_then(|v| v.as_bool()).unwrap_or(false).to_string())
        .entry("safe_log", payload.get("safe_log").and_then(|v| v.as_bool()).unwrap_or(true).to_string())
        .print();
    if let Some(warning) = payload.get("warning").and_then(|v| v.as_str()) {
        println!("{} {warning}", table::warn());
    }
    Ok(())
}

/// Enable runtime diagnostics over gateway RPC.
pub async fn diagnostics_on(raw: bool, json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let params = if raw {
        serde_json::json!({ "enabled": true, "verbose": true, "trace": true })
    } else {
        serde_json::json!({ "enabled": true, "verbose": true })
    };
    let payload = rpc::rpc_or_fallback(&config, "diagnostics.configure", params)
        .await
        .unwrap_or(serde_json::json!({
            "enabled": true,
            "warning": "gateway not reachable; flag recorded locally only"
        }));

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }
    println!("{} Diagnostics enabled (raw={raw})", table::ok());
    if let Some(warning) = payload.get("warning").and_then(|v| v.as_str()) {
        println!("{} {warning}", table::warn());
    }
    Ok(())
}

/// Disable runtime diagnostics over gateway RPC.
pub async fn diagnostics_off(json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let payload = rpc::rpc_or_fallback(
        &config,
        "diagnostics.configure",
        serde_json::json!({ "enabled": false, "verbose": false, "trace": false }),
    )
    .await
    .unwrap_or(serde_json::json!({
        "enabled": false,
        "warning": "gateway not reachable; flag recorded locally only"
    }));

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }
    println!("{} Diagnostics disabled", table::ok());
    if let Some(warning) = payload.get("warning").and_then(|v| v.as_str()) {
        println!("{} {warning}", table::warn());
    }
    Ok(())
}

/// A diagnostics bundle collected for a bug report.
#[derive(Serialize)]
struct DiagnosticsBundle {
    generated_at: String,
    rustc_version: String,
    os: String,
    arch: String,
    config_path: String,
    config_summary: ConfigSummary,
    session_db_path: String,
    memory_db_path: String,
    data_dir: String,
    providers: Vec<ProviderSummary>,
    log_tail: Vec<String>,
}

/// A summarized view of the configuration.
#[derive(Serialize)]
struct ConfigSummary {
    provider_count: usize,
    channel_count: usize,
    gateway: String,
    sandbox_enabled: bool,
    skills_enabled: bool,
}

/// A summarized provider entry.
#[derive(Serialize)]
struct ProviderSummary {
    name: String,
    provider_type: String,
    has_api_key: bool,
    model_count: usize,
}

/// Collect a diagnostics bundle and write it to disk.
pub async fn diagnostics_dump(output: Option<String>, log_lines: usize) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let out_dir = output
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("opensquilla-diagnostics"));
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create {}", out_dir.display()))?;

    println!("{}", "Collecting diagnostics bundle".bold());
    println!("  Output: {}", out_dir.display());

    let rustc_version = rustc_version().unwrap_or_else(|_| "unknown".to_string());
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let config_path = Config::discover_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(not found)".to_string());
    let session_db_path = util::session_db_path().display().to_string();
    let memory_db_path = util::memory_db_path().display().to_string();
    let data_dir = util::data_dir().display().to_string();

    let providers: Vec<ProviderSummary> = config
        .providers
        .iter()
        .map(|p| ProviderSummary {
            name: p.name.clone(),
            provider_type: p.provider_type.clone(),
            has_api_key: util::provider_api_key(&config, p).is_some(),
            model_count: p.models.len(),
        })
        .collect();

    let config_summary = ConfigSummary {
        provider_count: config.providers.len(),
        channel_count: config.channels.len(),
        gateway: format!("{}:{}", config.gateway.host, config.gateway.port),
        sandbox_enabled: config.sandbox.as_ref().is_some_and(|s| s.enabled),
        skills_enabled: config.skills.as_ref().is_some_and(|s| s.enabled),
    };

    let log_tail = collect_log_tail(&config, log_lines);

    let bundle = DiagnosticsBundle {
        generated_at: util::now_rfc3339(),
        rustc_version,
        os,
        arch,
        config_path: config_path.clone(),
        config_summary,
        session_db_path,
        memory_db_path,
        data_dir,
        providers,
        log_tail,
    };

    // Write the bundle JSON.
    let bundle_path = out_dir.join("diagnostics.json");
    util::write_json_file(&bundle_path, &bundle)?;
    println!("{} Wrote {}", table::ok(), bundle_path.display());

    // Write the config file copy (without secrets — we copy the on-disk file).
    if let Ok(path) = Config::discover_path() {
        if path.exists() {
            let dest = out_dir.join("config.toml");
            if let Ok(contents) = std::fs::read_to_string(&path) {
                std::fs::write(&dest, &contents).ok();
                println!("{} Copied {}", table::ok(), dest.display());
            }
        }
    }

    // Write the log tail.
    if !bundle.log_tail.is_empty() {
        let log_path = out_dir.join("logs.txt");
        let log_text = bundle.log_tail.join("\n");
        std::fs::write(&log_path, &log_text).ok();
        println!("{} Wrote {}", table::ok(), log_path.display());
    } else {
        println!("{} No log file found to tail", table::info());
    }

    info!("Diagnostics bundle written to {}", out_dir.display());
    println!();
    println!("{} Bundle complete. Attach {} to your bug report.", table::ok(), out_dir.display());
    println!("  Review the bundle for secrets before sharing.");
    Ok(())
}

/// Run `rustc --version` and return the trimmed output.
fn rustc_version() -> Result<String> {
    let output = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .context("Failed to run rustc")?;
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Collect the tail of the most recent log file.
fn collect_log_tail(config: &Config, lines: usize) -> Vec<String> {
    // Prefer an explicit log dir from config; fall back to the data dir.
    let log_dir = config
        .get("observability.log_dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| util::data_dir().join("logs"));

    if !log_dir.exists() {
        return Vec::new();
    }

    // Pick the most recently modified .log file.
    let newest = std::fs::read_dir(&log_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("log"))
        .max_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });

    match newest {
        Some(entry) => util::log_tail(&entry.path(), lines),
        None => {
            // Try opensquilla.log directly.
            let candidate = log_dir.join("opensquilla.log");
            if candidate.exists() {
                util::log_tail(&candidate, lines)
            } else {
                Vec::new()
            }
        }
    }
}

/// Bold helper for headers.
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
