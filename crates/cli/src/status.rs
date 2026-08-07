//! # System status command
//!
//! Implements the `status` subcommand, a single overview of the whole
//! OpenSquilla installation: configuration health, providers, sessions,
//! memory, skills, channels, scheduler jobs, and gateway state.
//!
//! - `status` — full overview
//! - `status brief` — one-line summary
//! - `status components` — per-subsystem detail

use anyhow::{Context, Result};
use opensquilla_core::config::Config;

use crate::table::{self, Color, KeyValue, Style};
use crate::util;

/// Status subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum StatusAction {
    /// Full system overview.
    Full,
    /// One-line summary.
    Brief,
    /// Per-subsystem component detail.
    Components,
}

/// Run a status subcommand.
pub async fn run_status(action: StatusAction) -> Result<()> {
    match action {
        StatusAction::Full => status_full().await,
        StatusAction::Brief => status_brief().await,
        StatusAction::Components => status_components().await,
    }
}

/// Full system overview.
pub async fn status_full() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;

    println!("{}", "OpenSquilla Status".bold());
    println!();
    table::rule();

    // Configuration summary.
    println!("{}", "Configuration".bold());
    KeyValue::new()
        .entry("Config path", config_path())
        .entry(
            "Gateway",
            format!("{}:{}", config.gateway.host, config.gateway.port),
        )
        .entry("Providers", config.providers.len().to_string())
        .entry("Channels", config.channels.len().to_string())
        .entry(
            "Sandbox",
            config
                .sandbox
                .as_ref()
                .map(|s| s.enabled.to_string())
                .unwrap_or_else(|| "disabled".to_string()),
        )
        .entry(
            "Skills",
            config
                .skills
                .as_ref()
                .map(|s| s.enabled.to_string())
                .unwrap_or_else(|| "disabled".to_string()),
        )
        .print();
    println!();

    // Provider status.
    println!("{}", "Providers".bold());
    match util_build_registry(&config) {
        Ok(registry) => {
            let names = registry.list();
            if names.is_empty() {
                println!("  {} No providers configured", table::warn());
            } else {
                for name in names {
                    if let Some(p) = registry.get(&name) {
                        let model_count = p.supported_models().len();
                        println!(
                            "  {} {:<20} {} model(s) [{}]",
                            table::ok(),
                            name,
                            model_count,
                            p.name()
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!("  {} Provider registry error: {e}", table::fail());
        }
    }
    println!();

    // Session store.
    println!("{}", "Sessions".bold());
    match util::build_session_manager(&config) {
        Ok(manager) => match manager.list_sessions(&util::default_agent_id(), 1000, 0) {
            Ok(sessions) => {
                let total_tokens: u64 = sessions.iter().map(|s| s.total_tokens).sum();
                let total_cost: f64 = sessions.iter().map(|s| s.total_cost_usd).sum();
                let active = sessions
                    .iter()
                    .filter(|s| s.status == opensquilla_session::SessionStatus::Active)
                    .count();
                println!(
                    "  {} {} session(s), {} active, {} tokens, ${:.4}",
                    table::ok(),
                    sessions.len(),
                    active,
                    total_tokens,
                    total_cost
                );
            }
            Err(e) => println!("  {} Failed to list sessions: {e}", table::fail()),
        },
        Err(e) => println!("  {} Session store error: {e}", table::fail()),
    }
    println!();

    // Memory store.
    println!("{}", "Memory".bold());
    match opensquilla_memory::store::MemoryStore::new(&util::memory_db_path().to_string_lossy()) {
        Ok(store) => match store.list_memories(&util::default_agent_id(), None, 1000, 0) {
            Ok(entries) => {
                println!("  {} {} memory entr(ies)", table::ok(), entries.len());
            }
            Err(e) => println!("  {} Failed to read memory: {e}", table::fail()),
        },
        Err(e) => println!("  {} Memory store error: {e}", table::fail()),
    }
    println!();

    // Skills.
    println!("{}", "Skills".bold());
    match build_skill_loader(&config).await {
        Ok(loader) => {
            let skills = loader.get_skills(None).await;
            println!("  {} {} skill(s) loaded", table::ok(), skills.len());
        }
        Err(e) => println!("  {} Skill scan error: {e}", table::fail()),
    }
    println!();

    // Gateway.
    println!("{}", "Gateway".bold());
    let host = config.gateway.host.clone();
    let port = config.gateway.port;
    if tokio::net::TcpStream::connect((host.as_str(), port))
        .await
        .is_ok()
    {
        println!(
            "  {} Running on {}:{}",
            table::ok(),
            config.gateway.host,
            config.gateway.port
        );
    } else {
        println!(
            "  {} Not running on {}:{}",
            table::warn(),
            config.gateway.host,
            config.gateway.port
        );
    }
    println!();

    // Scheduler.
    println!("{}", "Scheduler".bold());
    match build_scheduler_engine() {
        Ok(engine) => match engine.ops().get_stats().await {
            Ok(stats) => {
                println!(
                    "  {} {} active job(s), {} completed, {} failed",
                    table::ok(),
                    stats.active_jobs,
                    stats.completed_jobs,
                    stats.failed_jobs
                );
            }
            Err(e) => println!("  {} Scheduler stats error: {e}", table::fail()),
        },
        Err(e) => println!("  {} Scheduler error: {e}", table::fail()),
    }

    println!();
    table::rule();
    Ok(())
}

/// One-line summary.
pub async fn status_brief() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let providers = config.providers.len();

    let sessions = match util::build_session_manager(&config) {
        Ok(manager) => manager
            .list_sessions(&util::default_agent_id(), 1000, 0)
            .map(|s| s.len())
            .unwrap_or(0),
        Err(_) => 0,
    };

    let gateway_up =
        tokio::net::TcpStream::connect((config.gateway.host.as_str(), config.gateway.port))
            .await
            .is_ok();

    let gateway_str = if gateway_up { "up" } else { "down" };
    println!(
        "OpenSquilla: {} provider(s), {} session(s), gateway {gateway_str}",
        providers, sessions
    );
    Ok(())
}

/// Per-subsystem component detail.
pub async fn status_components() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;

    let mut rows: Vec<(String, &'static str, String)> = Vec::new();

    // Config.
    rows.push((
        "config".to_string(),
        "ok",
        format!(
            "{} provider(s), {} channel(s)",
            config.providers.len(),
            config.channels.len()
        ),
    ));

    // Provider registry.
    match util_build_registry(&config) {
        Ok(registry) => {
            let count = registry.len();
            rows.push((
                "providers".to_string(),
                if count > 0 { "ok" } else { "warn" },
                format!("{count} provider(s) registered"),
            ));
        }
        Err(e) => rows.push(("providers".to_string(), "error", e.to_string())),
    }

    // Session store.
    match util::build_session_manager(&config) {
        Ok(_) => rows.push((
            "sessions".to_string(),
            "ok",
            "store initialized".to_string(),
        )),
        Err(e) => rows.push(("sessions".to_string(), "error", e.to_string())),
    }

    // Memory store.
    let mem_path = util::memory_db_path();
    if mem_path.exists() {
        rows.push(("memory".to_string(), "ok", mem_path.display().to_string()));
    } else {
        rows.push((
            "memory".to_string(),
            "info",
            "not yet initialized".to_string(),
        ));
    }

    // Skills.
    match build_skill_loader(&config).await {
        Ok(loader) => {
            let count = loader.get_skills(None).await.len();
            rows.push(("skills".to_string(), "ok", format!("{count} skill(s)")));
        }
        Err(e) => rows.push(("skills".to_string(), "error", e.to_string())),
    }

    // Scheduler.
    match build_scheduler_engine() {
        Ok(engine) => match engine.ops().get_stats().await {
            Ok(stats) => rows.push((
                "scheduler".to_string(),
                "ok",
                format!("{} active job(s)", stats.active_jobs),
            )),
            Err(e) => rows.push(("scheduler".to_string(), "error", e.to_string())),
        },
        Err(e) => rows.push(("scheduler".to_string(), "error", e.to_string())),
    }

    // Gateway.
    let gateway_up =
        tokio::net::TcpStream::connect((config.gateway.host.as_str(), config.gateway.port))
            .await
            .is_ok();
    rows.push((
        "gateway".to_string(),
        if gateway_up { "ok" } else { "warn" },
        format!("{}:{}", config.gateway.host, config.gateway.port),
    ));

    println!("Component Status");
    println!("{:-<70}", "");
    for (component, status, detail) in &rows {
        let icon = match *status {
            "ok" => table::ok(),
            "warn" => table::warn(),
            "info" => table::info(),
            _ => table::fail(),
        };
        println!("  {icon} {:<12} {:<40}", component, detail);
    }
    println!("{:-<70}", "");
    Ok(())
}

/// Build a provider registry (helper).
fn util_build_registry(config: &Config) -> Result<opensquilla_provider::ProviderRegistry> {
    crate::util::build_provider_registry(config)
}

/// Build a skill loader (helper).
async fn build_skill_loader(config: &Config) -> Result<opensquilla_skills::loader::SkillLoader> {
    use opensquilla_skills::bundled::load_bundled_skills;
    use opensquilla_skills::loader::SkillLoader;
    use opensquilla_skills::types::SkillLayer;

    let loader = SkillLoader::new();
    if let Some(skills_cfg) = config.skills.as_ref() {
        for dir in &skills_cfg.skill_dirs {
            loader.register_layer_dir(SkillLayer::Extra, std::path::Path::new(dir).to_path_buf());
        }
    }
    let managed = crate::util::skills_dir();
    std::fs::create_dir_all(&managed).ok();
    loader.register_layer_dir(SkillLayer::Managed, managed);
    let bundled = load_bundled_skills();
    let _ = loader.register_skills(bundled);
    loader
        .scan_all()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to scan skills: {e}"))?;
    Ok(loader)
}

/// Build a scheduler engine (helper).
fn build_scheduler_engine() -> Result<opensquilla_scheduler::SchedulerEngine> {
    let path = crate::util::data_dir().join("scheduler.db");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    opensquilla_scheduler::SchedulerBuilder::new()
        .with_db_path(path.to_string_lossy().to_string())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build scheduler: {e}"))
}

/// Resolve the config file path.
fn config_path() -> String {
    Config::discover_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "(not found)".to_string())
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
