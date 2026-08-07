//! Migration commands.
//!
//! Implements the `migrate` subcommand for importing config and sessions from
//! OpenClaw, Hermes, or old Python OpenSquilla layouts. Mirrors the Python
//! `migrate_cmd.py`.
//!
//! The Rust gateway exposes `migration.discover`, `migration.preview`,
//! `migration.discover_path`, and `migration.validate` RPC handlers (see
//! `crates/gateway/src/migration.rs`). OpenClaw and Hermes home migration
//! (Python `opensquilla.migration.openclaw` / `hermes`) is implemented here;
//! OpenSquilla self-migration and the deeper OpenClaw/Hermes item types remain
//! marked with `// TODO(parity):`.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
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
        MigrateAction::Opensquilla {
            source,
            kind,
            apply,
            json,
        } => migrate_opensquilla(source, kind, apply, json).await,
        MigrateAction::Openclaw {
            source,
            config,
            apply,
            migrate_secrets,
            preset,
            skill_conflict,
            json,
        } => {
            migrate_openclaw(
                source,
                config,
                apply,
                migrate_secrets,
                preset,
                skill_conflict,
                json,
            )
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
        println!(
            "Use `osq migrate opensquilla --source <path>` (or openclaw/hermes) to point at a non-default home."
        );
        return Ok(());
    }

    println!("{}", "Detected migration sources".bold());
    println!("{:-<60}", "");
    for source in &detected {
        println!("  {} {:<12} {}", table::info(), source.name, source.path);
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
        println!(
            "  {icon} {path}  ({}, {} bytes)",
            if exists { "present" } else { "missing" },
            size
        );
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

    let valid = payload
        .get("valid")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let entry_count = payload
        .get("entry_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
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

    let valid = payload
        .get("valid")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
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
    config: Option<String>,
    apply: bool,
    migrate_secrets: bool,
    preset: String,
    skill_conflict: String,
    json: bool,
) -> Result<()> {
    let source_path = source
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".openclaw"));
    let config_path = config.map(PathBuf::from);
    let home = default_opensquilla_home();
    let mut items: Vec<MigrationItem> = Vec::new();
    let notes: Vec<String> = Vec::new();

    // Non-capturing recorder so helpers can also push into `items`.
    let record = |items: &mut Vec<MigrationItem>,
                  kind: &str,
                  src: Option<&PathBuf>,
                  dst: Option<&PathBuf>,
                  status: &str,
                  reason: String| {
        items.push(MigrationItem {
            kind: kind.to_string(),
            source: src.map(|p| p.display().to_string()),
            destination: dst.map(|p| p.display().to_string()),
            status: status.to_string(),
            reason,
        });
    };

    // Validation mirrors Python `_validation_error()`.
    if !MIGRATION_PRESETS.iter().any(|(name, _)| *name == preset) {
        record(
            &mut items,
            "options",
            None,
            None,
            "error",
            format!("Unknown migration preset: {preset}"),
        );
    }
    if !SKILL_CONFLICT_MODES.contains(&skill_conflict.as_str()) {
        record(
            &mut items,
            "options",
            None,
            None,
            "error",
            format!("Unknown skill conflict behavior: {skill_conflict}"),
        );
    }
    if !source_path.is_dir() {
        record(
            &mut items,
            "source",
            Some(&source_path),
            None,
            "error",
            "OpenClaw source directory does not exist",
        );
        return finish_migration(
            "openclaw",
            &source_path,
            &home,
            config_path,
            apply,
            migrate_secrets,
            &preset,
            &skill_conflict,
            items,
            notes,
            json,
        );
    }
    if !is_valid_openclaw_home(&source_path) {
        record(
            &mut items,
            "source",
            Some(&source_path),
            None,
            "error",
            "not a valid OpenClaw home",
        );
    }

    let config_data = load_openclaw_config(&source_path);
    let selected = selected_options(&preset);

    let workspace_src = openclaw_workspace(&source_path, &config_data);
    let workspace_dst = home.join("workspace");

    if selected.contains(&"soul") {
        migrate_workspace_file(
            &mut items,
            apply,
            &workspace_src,
            &workspace_dst,
            "SOUL.md",
            "soul",
        );
    }
    if selected.contains(&"workspace-agents") {
        migrate_workspace_file(
            &mut items,
            apply,
            &workspace_src,
            &workspace_dst,
            "AGENTS.md",
            "workspace-agents",
        );
    }
    if selected.contains(&"user-profile") {
        migrate_workspace_file(
            &mut items,
            apply,
            &workspace_src,
            &workspace_dst,
            "USER.md",
            "user-profile",
        );
    }

    if selected.contains(&"provider-keys") {
        migrate_provider_keys(
            &mut items,
            apply,
            migrate_secrets,
            &source_path,
            &home,
            &config_data,
        );
    }

    // TODO(parity): OpenClaw memory, skills, mcp-servers, agent-config,
    // tools-config, channels, tts, command-allowlist and archive config
    // migrations (Python `_migrate_memory/_migrate_skills/_migrate_mcp_servers/...`)
    // are not yet ported. Rollback (`_ApplyRollback`), rebrand text
    // substitution, persona-conflict prompts and report-file writing are
    // also deferred.

    finish_migration(
        "openclaw",
        &source_path,
        &home,
        config_path,
        apply,
        migrate_secrets,
        &preset,
        &skill_conflict,
        items,
        notes,
        json,
    )
}

/// Migrate Hermes Agent state into OpenSquilla-native files.
pub async fn migrate_hermes(
    source: Option<String>,
    profile: Option<String>,
    config: Option<String>,
    apply: bool,
    migrate_secrets: bool,
    preset: String,
    skill_conflict: String,
    json: bool,
) -> Result<()> {
    let home = default_opensquilla_home();
    let config_path = config.map(PathBuf::from);
    let mut items: Vec<MigrationItem> = Vec::new();
    let notes: Vec<String> = Vec::new();

    // Non-capturing recorder so helpers can also push into `items`.
    let record = |items: &mut Vec<MigrationItem>,
                  kind: &str,
                  src: Option<&PathBuf>,
                  dst: Option<&PathBuf>,
                  status: &str,
                  reason: String| {
        items.push(MigrationItem {
            kind: kind.to_string(),
            source: src.map(|p| p.display().to_string()),
            destination: dst.map(|p| p.display().to_string()),
            status: status.to_string(),
            reason,
        });
    };

    if !MIGRATION_PRESETS.iter().any(|(name, _)| *name == preset) {
        record(
            &mut items,
            "options",
            None,
            None,
            "error",
            format!("Unknown migration preset: {preset}"),
        );
    }
    if !SKILL_CONFLICT_MODES.contains(&skill_conflict.as_str()) {
        record(
            &mut items,
            "options",
            None,
            None,
            "error",
            format!("Unknown skill conflict behavior: {skill_conflict}"),
        );
    }

    // Profile names are validated like Python `_HERMES_PROFILE_NAME_RE`.
    let source_path = resolve_hermes_source(source, profile.as_deref());
    if !source_path.is_dir() {
        record(
            &mut items,
            "source",
            Some(&source_path),
            None,
            "error",
            "Hermes source directory does not exist",
        );
        return finish_migration(
            "hermes",
            &source_path,
            &home,
            config_path,
            apply,
            migrate_secrets,
            &preset,
            &skill_conflict,
            items,
            notes,
            json,
        );
    }
    if !is_valid_hermes_home(&source_path) {
        record(
            &mut items,
            "source",
            Some(&source_path),
            None,
            "error",
            "not a valid Hermes home",
        );
    }

    let config_data = load_hermes_config(&source_path);
    let selected = selected_options(&preset);

    let workspace_src = source_path.clone();
    let workspace_dst = home.join("workspace");

    if selected.contains(&"soul") {
        migrate_workspace_file(
            &mut items,
            apply,
            &workspace_src,
            &workspace_dst,
            "SOUL.md",
            "soul",
        );
    }

    if selected.contains(&"provider-keys") {
        migrate_provider_keys(
            &mut items,
            apply,
            migrate_secrets,
            &source_path,
            &home,
            &config_data,
        );
    }

    // TODO(parity): Hermes memories, skills, and MCP-server migrations plus
    // profile metadata (created_at/tool_config/transport) are not yet ported.

    finish_migration(
        "hermes",
        &source_path,
        &home,
        config_path,
        apply,
        migrate_secrets,
        &preset,
        &skill_conflict,
        items,
        notes,
        json,
    )
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

/// A single migration result item (mirrors Python `ItemResult`).
#[derive(Serialize)]
struct MigrationItem {
    kind: String,
    source: Option<String>,
    destination: Option<String>,
    status: String,
    reason: String,
}

impl MigrationItem {
    fn new(
        kind: &str,
        source: Option<&PathBuf>,
        destination: Option<&PathBuf>,
        status: &str,
        reason: String,
    ) -> Self {
        Self {
            kind: kind.to_string(),
            source: source.map(|p| p.display().to_string()),
            destination: destination.map(|p| p.display().to_string()),
            status: status.to_string(),
            reason,
        }
    }
}

/// Migration presets and the option names each selects.
const MIGRATION_PRESETS: &[(&str, &[&str])] = &[
    (
        "full",
        &[
            "config",
            "provider-keys",
            "soul",
            "workspace-agents",
            "user-profile",
            "memory",
            "skills",
            "mcp-servers",
            "agent-config",
            "tools-config",
            "channels",
            "tts",
            "command-allowlist",
            "archive",
        ],
    ),
    (
        "user-data",
        &[
            "provider-keys",
            "soul",
            "workspace-agents",
            "user-profile",
            "memory",
            "skills",
            "channels",
            "tts",
        ],
    ),
];

/// Allowed skill conflict behaviors.
const SKILL_CONFLICT_MODES: &[&str] = &["skip", "overwrite", "rename"];

/// OpenClaw raw config filenames, in priority order (Python `RAW_CONFIG_FILENAMES`).
const OPENCLAW_RAW_CONFIG_FILENAMES: &[&str] = &["openclaw.json", "clawdbot.json", "moltbot.json"];

/// Files that mark an OpenClaw workspace (Python `_OPENCLAW_WORKSPACE_MARKERS`).
const OPENCLAW_WORKSPACE_MARKERS: &[&str] = &[
    "SOUL.md",
    "MEMORY.md",
    "USER.md",
    "AGENTS.md",
    "IDENTITY.md",
];

/// Recognized secret env keys copied from a source `.env`.
const SECRET_ENV_KEYS: &[&str] = &[
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "ZAI_API_KEY",
    "MINIMAX_API_KEY",
    "BRAVE_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
];

/// Provider-id -> env key mapping (Python `PROVIDER_ENV_KEYS`).
const PROVIDER_ENV_KEYS: &[(&str, &str)] = &[
    ("openrouter", "OPENROUTER_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("zai", "ZAI_API_KEY"),
    ("zhipu", "ZAI_API_KEY"),
];

/// Option names selected by a migration preset.
fn selected_options(preset: &str) -> Vec<&'static str> {
    MIGRATION_PRESETS
        .iter()
        .find(|(name, _)| *name == preset)
        .map(|(_, opts)| opts.to_vec())
        .unwrap_or_default()
}

/// True when `path` looks like an OpenClaw home: a raw config file or a
/// workspace containing a marker file.
fn is_valid_openclaw_home(path: &PathBuf) -> bool {
    let has_raw_config = OPENCLAW_RAW_CONFIG_FILENAMES
        .iter()
        .any(|name| path.join(name).is_file());
    if has_raw_config {
        return true;
    }
    let workspace = openclaw_workspace(path, &Value::Null);
    OPENCLAW_WORKSPACE_MARKERS
        .iter()
        .any(|name| workspace.join(name).is_file())
}

/// Load the first existing OpenClaw raw config as JSON (null if none).
fn load_openclaw_config(source: &PathBuf) -> Value {
    for name in OPENCLAW_RAW_CONFIG_FILENAMES {
        if let Ok(contents) = fs::read_to_string(source.join(name)) {
            if let Ok(value) = serde_json::from_str::<Value>(&contents) {
                return value;
            }
        }
    }
    Value::Null
}

/// Resolve the OpenClaw workspace directory (configured path, else `<source>/workspace`).
fn openclaw_workspace(source: &PathBuf, config: &Value) -> PathBuf {
    for pointer in ["/projects/workspace", "/workspace"] {
        if let Some(Value::String(p)) = config.pointer(pointer) {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    source.join("workspace")
}

/// Migrate a single workspace file (SOUL.md / AGENTS.md / USER.md).
fn migrate_workspace_file(
    items: &mut Vec<MigrationItem>,
    apply: bool,
    workspace_src: &PathBuf,
    workspace_dst: &PathBuf,
    filename: &str,
    kind: &str,
) {
    let src = workspace_src.join(filename);
    let dst = workspace_dst.join(filename);
    if !src.is_file() {
        items.push(MigrationItem::new(
            kind,
            Some(&src),
            Some(&dst),
            "skipped",
            "not found in source".to_string(),
        ));
        return;
    }
    if !apply {
        items.push(MigrationItem::new(
            kind,
            Some(&src),
            Some(&dst),
            "planned",
            "dry-run, would copy".to_string(),
        ));
        return;
    }
    match fs::create_dir_all(workspace_dst).and_then(|_| fs::copy(&src, &dst)) {
        Ok(_) => items.push(MigrationItem::new(
            kind,
            Some(&src),
            Some(&dst),
            "migrated",
            "copied".to_string(),
        )),
        Err(e) => items.push(MigrationItem::new(
            kind,
            Some(&src),
            Some(&dst),
            "error",
            e.to_string(),
        )),
    }
}

/// Parse a simple `KEY=VALUE` env file (blank lines, `#` comments, optional quotes).
fn parse_env_file(contents: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        let mut value = value.trim().to_string();
        if (value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\''))
        {
            value = value[1..value.len() - 1].to_string();
        }
        if !key.is_empty() {
            out.push((key, value));
        }
    }
    out
}

/// Normalize a provider id like Python `_normalize_provider_id` (zai -> zhipu).
fn normalize_provider_id(id: &str) -> String {
    let id = id.trim().to_ascii_lowercase();
    if id == "zai" { "zhipu".to_string() } else { id }
}

/// Migrate recognized provider keys from the source `.env` + config into `home/.env`.
fn migrate_provider_keys(
    items: &mut Vec<MigrationItem>,
    apply: bool,
    migrate_secrets: bool,
    source: &PathBuf,
    home: &PathBuf,
    config: &Value,
) {
    let env_file = source.join(".env");
    let dst_env = home.join(".env");
    let mut found: Vec<(String, String)> = Vec::new();

    if let Ok(contents) = fs::read_to_string(&env_file) {
        for (key, value) in parse_env_file(&contents) {
            if SECRET_ENV_KEYS.contains(&key.as_str()) {
                found.push((key, value));
            }
        }
    }

    if let Some(providers) = config
        .get("models")
        .and_then(|m| m.get("providers"))
        .and_then(|p| p.as_object())
    {
        for (pid, pcfg) in providers {
            let normalized = normalize_provider_id(pid);
            let Some(env_key) = PROVIDER_ENV_KEYS
                .iter()
                .find(|(p, _)| *p == normalized)
                .map(|(_, k)| *k)
            else {
                continue;
            };
            for key in ["apiKey", "api_key", "token", "auth_token"] {
                if let Some(v) = pcfg.get(key).and_then(|v| v.as_str()) {
                    found.push((env_key.to_string(), v.to_string()));
                    break;
                }
            }
        }
    }

    if found.is_empty() {
        items.push(MigrationItem::new(
            "provider-keys",
            Some(&env_file),
            Some(&dst_env),
            "skipped",
            "no recognized provider keys found".to_string(),
        ));
        return;
    }
    if !apply {
        for (k, _) in &found {
            items.push(MigrationItem::new(
                "provider-keys",
                Some(&env_file),
                Some(&dst_env),
                "planned",
                format!("would set {k}"),
            ));
        }
        return;
    }
    if !migrate_secrets {
        items.push(MigrationItem::new(
            "provider-keys",
            Some(&env_file),
            Some(&dst_env),
            "skipped",
            "secrets not migrated (pass --migrate-secrets)".to_string(),
        ));
        return;
    }

    let existing = fs::read_to_string(&dst_env).unwrap_or_default();
    let existing_keys: HashSet<String> = parse_env_file(&existing)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let mut append = String::new();
    let mut added = 0;
    for (k, v) in &found {
        if existing_keys.contains(k) {
            continue;
        }
        append.push_str(&format!("{k}={v}\n"));
        added += 1;
    }
    if added == 0 {
        items.push(MigrationItem::new(
            "provider-keys",
            Some(&env_file),
            Some(&dst_env),
            "skipped",
            "all recognized keys already present".to_string(),
        ));
        return;
    }
    let mut next = existing;
    if !next.is_empty() && !next.ends_with('\n') {
        next.push('\n');
    }
    next.push_str(&append);
    match fs::create_dir_all(home).and_then(|_| fs::write(&dst_env, next)) {
        Ok(_) => items.push(MigrationItem::new(
            "provider-keys",
            Some(&env_file),
            Some(&dst_env),
            "migrated",
            format!("{added} key(s) appended"),
        )),
        Err(e) => items.push(MigrationItem::new(
            "provider-keys",
            Some(&env_file),
            Some(&dst_env),
            "error",
            e.to_string(),
        )),
    }
}

/// Resolve the Hermes home: explicit source, else `$HERMES_HOME`, else
/// `~/.hermes`, else `~/.hermes/profiles/<profile>`.
fn resolve_hermes_source(source: Option<String>, profile: Option<&str>) -> PathBuf {
    if let Some(s) = source {
        return PathBuf::from(s);
    }
    let hermes_home = std::env::var("HERMES_HOME")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".hermes"));
    if let Some(profile) = profile {
        if is_valid_hermes_profile_name(profile) {
            return hermes_home.join("profiles").join(profile);
        }
    }
    hermes_home
}

/// True when a profile name matches Python `_HERMES_PROFILE_NAME_RE`.
fn is_valid_hermes_profile_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 64 {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit() => {}
        _ => return false,
    }
    name.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// True when `path` looks like a Hermes home (config.yaml / .env / SOUL.md /
/// memories / skills).
fn is_valid_hermes_home(path: &PathBuf) -> bool {
    let markers = ["config.yaml", ".env", "SOUL.md", "memories", "skills"];
    markers.iter().any(|name| path.join(name).exists())
}

/// Load a Hermes `config.yaml` as JSON (null if absent or unparsable).
fn load_hermes_config(source: &PathBuf) -> Value {
    match fs::read_to_string(source.join("config.yaml")) {
        Ok(contents) => serde_yaml::from_str::<Value>(&contents).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

/// Emit the final migration report (JSON or human-readable table).
fn finish_migration(
    kind: &str,
    source: &PathBuf,
    home: &PathBuf,
    config_path: Option<PathBuf>,
    apply: bool,
    migrate_secrets: bool,
    preset: &str,
    skill_conflict: &str,
    items: Vec<MigrationItem>,
    notes: Vec<String>,
    json: bool,
) -> Result<()> {
    let mode = if apply { "apply" } else { "dry-run" };
    let status = if items.iter().any(|i| i.status == "error") {
        "partial"
    } else {
        "ok"
    };

    let report = serde_json::json!({
        "source": source.display().to_string(),
        "home": home.display().to_string(),
        "config": config_path.map(|p| p.display().to_string()),
        "mode": mode,
        "migrate_secrets": migrate_secrets,
        "preset": preset,
        "skill_conflict": skill_conflict,
        "status": status,
        "notes": notes,
        "items": items,
    });

    if json {
        util::print_json(&report)?;
        return Ok(());
    }

    println!("{} {kind} migration ({mode})", table::info());
    println!("{:-<60}", "");
    KeyValue::new()
        .entry("source", source.display().to_string())
        .entry("home", home.display().to_string())
        .entry("preset", preset.to_string())
        .entry("skill_conflict", skill_conflict.to_string())
        .entry("status", status.to_string())
        .print();
    println!();
    println!("{}", "Items".bold());
    println!("{:-<60}", "");
    for item in &items {
        let icon = match item.status.as_str() {
            "error" => table::fail(),
            "skipped" => table::warn(),
            "migrated" | "ok" => table::ok(),
            _ => table::info(),
        };
        let dst = item.destination.as_deref().unwrap_or("-");
        println!("  {icon} {:<18} {:<9} {dst}", item.kind, item.status);
        if item.status == "error" && !item.reason.is_empty() {
            println!("           {} {item.reason}", table::fail());
        }
    }
    if !notes.is_empty() {
        println!();
        println!("{}", "Notes".bold());
        for n in &notes {
            println!("  - {n}");
        }
    }
    println!();
    println!("Use --apply to write changes; pass --migrate-secrets to copy provider keys.");
    Ok(())
}
