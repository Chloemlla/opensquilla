//! Migration commands.
//!
//! Implements the `migrate` subcommand for importing config and sessions from
//! OpenClaw, Hermes, or old Python OpenSquilla layouts. Mirrors the Python
//! `migrate_cmd.py`.
//!
//! The Rust gateway exposes `migration.discover`, `migration.preview`,
//! `migration.discover_path`, and `migration.validate` RPC handlers (see
//! `crates/gateway/src/migration.rs`). The full OpenClaw/Hermes/OpenSquilla
//! home migrators (Python `opensquilla.migration.*`) are not yet ported to
//! Rust; those subcommands are stubbed with `// TODO:` until the migrator
//! crates exist.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use serde::Serialize;
use std::path::PathBuf;

use crate::rpc;
use crate::table::{self, Color, KeyValue, Style};
use crate::util;

/// Migration subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum MigrateAction {
    /// Auto-detect importable homes and report what was found.
    Detect {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Discover config file candidates from common locations.
    Discover {
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Preview a config file's parsed contents without applying it.
    Preview {
        /// Config file path to preview.
        path: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Validate a config file's structure.
    Validate {
        /// Config file path to validate.
        path: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Import a supported OpenSquilla CLI/Desktop/Portable profile.
    Opensquilla {
        /// Source profile directory.
        #[arg(long)]
        source: Option<String>,
        /// Source kind: cli-home, windows-portable, or desktop-home.
        #[arg(long, default_value = "cli-home")]
        kind: String,
        /// Apply the migration (default is a dry-run report).
        #[arg(long)]
        apply: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Migrate OpenClaw state into OpenSquilla-native files.
    Openclaw {
        /// OpenClaw home directory.
        #[arg(long)]
        source: Option<String>,
        /// OpenSquilla config path to write or preview.
        #[arg(long)]
        config: Option<String>,
        /// Apply the migration (default is a dry-run report).
        #[arg(long)]
        apply: bool,
        /// Copy recognized secrets.
        #[arg(long)]
        migrate_secrets: bool,
        /// Migration preset: user-data or full.
        #[arg(long, default_value = "full")]
        preset: String,
        /// Skill conflict behavior: skip, overwrite, or rename.
        #[arg(long, default_value = "skip")]
        skill_conflict: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Migrate Hermes Agent state into OpenSquilla-native files.
    Hermes {
        /// Hermes home directory.
        #[arg(long)]
        source: Option<String>,
        /// Hermes profile name under ~/.hermes/profiles.
        #[arg(long)]
        profile: Option<String>,
        /// OpenSquilla config path to write or preview.
        #[arg(long)]
        config: Option<String>,
        /// Apply the migration (default is a dry-run report).
        #[arg(long)]
        apply: bool,
        /// Copy recognized secrets.
        #[arg(long)]
        migrate_secrets: bool,
        /// Migration preset: user-data or full.
        #[arg(long, default_value = "full")]
        preset: String,
        /// Skill conflict behavior: skip, overwrite, or rename.
        #[arg(long, default_value = "skip")]
        skill_conflict: String,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Run a migrate subcommand.
pub async fn run_migrate(action: MigrateAction) -> Result<()> {
    match action {
        MigrateAction::Detect { json } => migrate_detect(json).await,
        MigrateAction::Discover { json } => migrate_discover(json).await,
        MigrateAction::Preview { path, json } => migrate_preview(path, json).await,
        MigrateAction::Validate { path, json } => migrate_validate(path, json).await,
        MigrateAction::Opensquilla { source, kind, apply, json } => {
            migrate_opensquilla(source, kind, apply, json).await
        }
        MigrateAction::Openclaw {
            source,
            config,
            apply,
            migrate_secrets,
            preset,
            skill_conflict,
            json,
        } => {
            migrate_openclaw(source, config, apply, migrate_secrets, preset, skill_conflict, json)
                .await
        }
        MigrateAction::Hermes {
            source,
            profile,
            config,
            apply,
            migrate_secrets,
            preset,
            skill_conflict,
            json,
        } => {
            migrate_hermes(
                source,
                profile,
                config,
                apply,
                migrate_secrets,
                preset,
                skill_conflict,
                json,
            )
            .await
        }
    }
}

/// Auto-detect importable homes under the user's home directory.
pub async fn migrate_detect(json: bool) -> Result<()> {
    let detected = detect_sources();

    if json {
        util::print_json(&detected)?;
        return Ok(());
    }

    if detected.is_empty() {
        println!("No migration source detected.");
        println!("Checked ~/.opensquilla, ~/.openclaw, and ~/.hermes.");
        println!("Use `osq migrate opensquilla --source <path>` (or openclaw/hermes) to point at a non-default home.");
        return Ok(());
    }

    println!("{}", "Detected migration sources".bold());
    println!("{:-<60}", "");
    for source in &detected {
        println!(
            "  {} {:<12} {}",
            table::info(),
            source.name,
            source.path
        );
    }
    println!();
    println!("Re-run with `osq migrate <name> --apply` to import.");
    Ok(())
}

/// Discover config file candidates via the gateway migration RPC.
pub async fn migrate_discover(json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let payload = rpc::rpc_or_fallback(&config, "migration.discover", serde_json::Value::Null)
        .await
        .map_err(|e| anyhow::anyhow!("migration.discover failed: {e}"))?;

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }

    let candidates = payload
        .get("candidates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    println!("{}", "Config candidates".bold());
    println!("{:-<60}", "");
    if candidates.is_empty() {
        println!("  (none found)");
    }
    for c in &candidates {
        let path = c.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let exists = c.get("exists").and_then(|v| v.as_bool()).unwrap_or(false);
        let size = c.get("size_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
        let icon = if exists { table::ok() } else { table::warn() };
        println!("  {icon} {path}  ({}, {} bytes)", if exists { "present" } else { "missing" }, size);
    }
    Ok(())
}

/// Preview a config file via the gateway migration RPC.
pub async fn migrate_preview(path: String, json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let payload = rpc::rpc_or_fallback(
        &config,
        "migration.preview",
        serde_json::json!({ "path": path }),
    )
    .await
    .map_err(|e| anyhow::anyhow!("migration.preview failed: {e}"))?;

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }

    let valid = payload.get("valid").and_then(|v| v.as_bool()).unwrap_or(false);
    let entry_count = payload.get("entry_count").and_then(|v| v.as_u64()).unwrap_or(0);
    println!("{}", format!("Preview: {path}").bold());
    KeyValue::new()
        .entry("valid", valid.to_string())
        .entry("entries", entry_count.to_string())
        .print();
    if let Some(err) = payload.get("error").and_then(|v| v.as_str()) {
        println!("{} {err}", table::fail());
    }
    Ok(())
}

/// Validate a config file via the gateway migration RPC.
pub async fn migrate_validate(path: String, json: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let payload = rpc::rpc_or_fallback(
        &config,
        "migration.validate",
        serde_json::json!({ "path": path }),
    )
    .await
    .map_err(|e| anyhow::anyhow!("migration.validate failed: {e}"))?;

    if json {
        util::print_json(&payload)?;
        return Ok(());
    }

    let valid = payload.get("valid").and_then(|v| v.as_bool()).unwrap_or(false);
    let icon = if valid { table::ok() } else { table::fail() };
    println!("{icon} {path}: {}", if valid { "valid" } else { "invalid" });
    if let Some(err) = payload.get("error").and_then(|v| v.as_str()) {
        println!("  {err}");
    }
    Ok(())
}

/// Import a supported OpenSquilla CLI/Desktop/Portable profile.
pub async fn migrate_opensquilla(
    source: Option<String>,
    kind: String,
    apply: bool,
    json: bool,
) -> Result<()> {
    // TODO: port `opensquilla.migration.opensquilla_home.OpenSquillaHomeMigrator`
    // to a Rust crate. Until then, report the planned action.
    let source_path = source
        .map(PathBuf::from)
        .unwrap_or_else(|| default_opensquilla_home());
    let mode = if apply { "apply" } else { "dry-run" };

    let report = serde_json::json!({
        "source": source_path.display().to_string(),
        "kind": kind,
        "mode": mode,
        "status": "not_implemented",
        "message": "OpenSquilla self-migration is not yet ported to Rust.",
    });

    if json {
        util::print_json(&report)?;
        return Ok(());
    }

    println!("{} OpenSquilla self-migration ({mode})", table::warn());
    println!("  source: {}", source_path.display());
    println!("  kind:   {kind}");
    println!("  status: not implemented in Rust yet");
    Ok(())
}

/// Migrate OpenClaw state into OpenSquilla-native files.
pub async fn migrate_openclaw(
    source: Option<String>,
    _config: Option<String>,
    apply: bool,
    _migrate_secrets: bool,
    preset: String,
    skill_conflict: String,
    json: bool,
) -> Result<()> {
    // TODO: port `opensquilla.migration.openclaw.OpenClawMigrator` to a Rust crate.
    let source_path = source
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".openclaw"));
    let mode = if apply { "apply" } else { "dry-run" };

    let report = serde_json::json!({
        "source": source_path.display().to_string(),
        "preset": preset,
        "skill_conflict": skill_conflict,
        "mode": mode,
        "status": "not_implemented",
        "message": "OpenClaw migration is not yet ported to Rust.",
    });

    if json {
        util::print_json(&report)?;
        return Ok(());
    }

    println!("{} OpenClaw migration ({mode})", table::warn());
    println!("  source:         {}", source_path.display());
    println!("  preset:         {preset}");
    println!("  skill_conflict: {skill_conflict}");
    println!("  status:         not implemented in Rust yet");
    Ok(())
}

/// Migrate Hermes Agent state into OpenSquilla-native files.
pub async fn migrate_hermes(
    source: Option<String>,
    profile: Option<String>,
    _config: Option<String>,
    apply: bool,
    _migrate_secrets: bool,
    preset: String,
    skill_conflict: String,
    json: bool,
) -> Result<()> {
    // TODO: port `opensquilla.migration.hermes.HermesMigrator` to a Rust crate.
    let source_path = source
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".hermes"));
    let mode = if apply { "apply" } else { "dry-run" };

    let report = serde_json::json!({
        "source": source_path.display().to_string(),
        "profile": profile,
        "preset": preset,
        "skill_conflict": skill_conflict,
        "mode": mode,
        "status": "not_implemented",
        "message": "Hermes migration is not yet ported to Rust.",
    });

    if json {
        util::print_json(&report)?;
        return Ok(());
    }

    println!("{} Hermes migration ({mode})", table::warn());
    println!("  source:         {}", source_path.display());
    if let Some(p) = profile {
        println!("  profile:        {p}");
    }
    println!("  preset:         {preset}");
    println!("  skill_conflict: {skill_conflict}");
    println!("  status:         not implemented in Rust yet");
    Ok(())
}

/// A detected migration source.
#[derive(Serialize)]
struct DetectedSource {
    name: String,
    path: String,
}

/// Detect importable homes on disk (opensquilla, openclaw, hermes).
fn detect_sources() -> Vec<DetectedSource> {
    let mut found = Vec::new();
    let home = home_dir();

    // OpenSquilla CLI home (only when it is not the active home).
    let opensquilla = home.join(".opensquilla");
    if opensquilla.exists() {
        let active = default_opensquilla_home();
        if opensquilla != active {
            found.push(DetectedSource {
                name: "opensquilla".to_string(),
                path: opensquilla.display().to_string(),
            });
        }
    }

    // OpenClaw home.
    let openclaw = home.join(".openclaw");
    if openclaw.exists() {
        found.push(DetectedSource {
            name: "openclaw".to_string(),
            path: openclaw.display().to_string(),
        });
    }

    // Hermes home.
    let hermes = home.join(".hermes");
    if hermes.exists() {
        found.push(DetectedSource {
            name: "hermes".to_string(),
            path: hermes.display().to_string(),
        });
    }

    found
}

/// Resolve the user's home directory.
fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Resolve the default OpenSquilla home (active state directory).
fn default_opensquilla_home() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_STATE_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("OPENSQUILLA_DATA_DIR") {
        return PathBuf::from(p);
    }
    home_dir().join(".opensquilla")
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
