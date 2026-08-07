//! Shared helpers used across the CLI command modules.
//!
//! Provides path resolution, session-store construction, provider registry
//! building, and in-process gateway RPC dispatch. These helpers centralize the
//! wiring so each command module can stay focused on presenting results.

use std::path::PathBuf;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_provider::{OpenAiCompatProvider, ProviderRegistry, ProviderSpecTable};
use serde_json::Value;
use uuid::Uuid;

/// The default agent id used for CLI-created sessions and memories.
///
/// A fixed, well-known id keeps CLI operations deterministic across commands
/// without requiring a persisted identity store.
pub const DEFAULT_AGENT_ID: &str = "00000000-0000-4000-8000-000000000001";

/// Parse the default agent id into a `Uuid`.
pub fn default_agent_id() -> Uuid {
    Uuid::parse_str(DEFAULT_AGENT_ID).expect("valid DEFAULT_AGENT_ID")
}

/// Resolve the top-level data directory for runtime state (databases, skills,
/// lockfiles). Honors `OPENSQUILLA_DATA_DIR`, then places data beside the
/// discovered config file, and finally falls back to the current directory.
pub fn data_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_DATA_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(cfg) = Config::discover_path() {
        if let Some(dir) = cfg.parent() {
            return dir.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Path to the SQLite session database.
pub fn session_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_SESSION_DB") {
        return PathBuf::from(p);
    }
    data_dir().join("sessions.db")
}

/// Path to the SQLite memory database.
pub fn memory_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_MEMORY_DB") {
        return PathBuf::from(p);
    }
    data_dir().join("memory.db")
}

/// Path to the managed skills directory used by the skill hub.
pub fn skills_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_SKILLS_DIR") {
        return PathBuf::from(p);
    }
    data_dir().join("skills")
}

/// Build a `SessionManager` backed by a file-backed SQLite store.
pub fn build_session_manager(_config: &Config) -> Result<opensquilla_session::SessionManager> {
    let path = session_db_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let storage = opensquilla_session::SessionStorage::new(&path.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("Failed to open session store at {}: {e}", path.display()))?;
    Ok(opensquilla_session::SessionManager::new(storage))
}

/// Determine the default provider name from the configuration.
pub fn default_provider(config: &Config) -> String {
    config
        .providers
        .first()
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "openai".to_string())
}

/// Determine the default model name from the configuration.
pub fn default_model(config: &Config) -> String {
    if let Some(models) = config.models.as_ref() {
        if let Some(m) = &models.default_model {
            return m.clone();
        }
    }
    if let Some(p) = config.providers.first() {
        if let Some(m) = &p.default_model {
            return m.clone();
        }
        return p.models.first().cloned().unwrap_or_default();
    }
    if let Some(spec) = ProviderSpecTable::get("openai") {
        return spec.default_model.to_string();
    }
    "gpt-4".to_string()
}

/// Resolve an API key for a provider from its config, falling back to the
/// conventional `<TYPE>_API_KEY` environment variable.
pub fn provider_api_key(
    _config: &Config,
    provider: &opensquilla_core::config::ProviderConfig,
) -> Option<String> {
    if let Some(key) = provider.api_key.clone() {
        if !key.is_empty() {
            return Some(key);
        }
    }
    let env_name = format!("{}_API_KEY", provider.provider_type.to_uppercase());
    std::env::var(&env_name).ok().filter(|k| !k.is_empty())
}

/// Build a provider registry populated from the configured providers.
pub fn build_provider_registry(config: &Config) -> Result<ProviderRegistry> {
    let registry = ProviderRegistry::new();
    for p in &config.providers {
        let api_key = provider_api_key(config, p).unwrap_or_default();
        match ProviderSpecTable::get(&p.provider_type) {
            Some(spec) => {
                if let Some(provider) = ProviderSpecTable::instantiate(&spec, &api_key) {
                    registry.register(p.name.clone(), provider);
                }
            }
            None => {
                // Unknown provider type; fall back to an OpenAI-compatible
                // client if a base URL was supplied.
                if let Some(base_url) = &p.base_url {
                    let provider = OpenAiCompatProvider::new(&p.name, base_url, &api_key);
                    registry.register(p.name.clone(), std::sync::Arc::new(provider));
                }
            }
        }
    }
    Ok(registry)
}

/// Build an in-process gateway for RPC dispatch (Mode A).
pub fn build_gateway(config: &Config) -> opensquilla_gateway::Gateway {
    opensquilla_gateway::Gateway::new(config.gateway.clone())
}

/// Dispatch an RPC call against an in-process gateway registry.
///
/// Mode A commands (sessions, chat, config) resolve through the gateway's RPC
/// registry. A missing handler or a handler error is surfaced as an `anyhow`
/// error for the caller.
pub async fn gateway_rpc(config: &Config, method: &str, params: Value) -> Result<Value> {
    let gateway = build_gateway(config);
    let result = gateway
        .rpc_registry
        .dispatch(method, params)
        .await
        .ok_or_else(|| anyhow::anyhow!("RPC method '{method}' is not registered"))?
        .map_err(|e| anyhow::anyhow!("RPC {method} failed: {} (status {})", e.message, e.status))?;
    Ok(result)
}

/// Serialize a session export to JSON, pretty-printed.
pub fn to_pretty_json(value: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string_pretty(value).context("Failed to serialize JSON")
}

/// Resolve the active session by id, or create a fresh session.
pub async fn resolve_or_create_session(
    manager: &opensquilla_session::SessionManager,
    session_id: Option<&str>,
) -> Result<opensquilla_session::Session> {
    match session_id {
        Some(sid) => {
            let id =
                Uuid::parse_str(sid).map_err(|_| anyhow::anyhow!("Invalid session id: {sid}"))?;
            manager
                .get_session(&id)
                .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
                .ok_or_else(|| anyhow::anyhow!("Session '{sid}' not found"))
        }
        None => manager
            .create_session(
                default_agent_id(),
                "CLI Session".to_string(),
                String::new(),
                opensquilla_session::SessionMode::Chat,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create session: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string (e.g., "1.5 MiB").
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Format a duration in milliseconds as a human-readable string.
pub fn human_duration(ms: u64) -> String {
    if ms < 1_000 {
        format!("{ms} ms")
    } else if ms < 60_000 {
        format!("{:.1} s", ms as f64 / 1_000.0)
    } else if ms < 3_600_000 {
        format!("{:.1} min", ms as f64 / 60_000.0)
    } else {
        format!("{:.1} h", ms as f64 / 3_600_000.0)
    }
}

/// Format a large number with thousands separators.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Truncate a string to `max` display characters, appending `…`.
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Parse a `KEY=VALUE` environment string into a pair.
pub fn parse_env_pair(s: &str) -> Option<(String, String)> {
    s.split_once('=')
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
}

/// Read the current timestamp as an RFC3339 string.
pub fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

// ---------------------------------------------------------------------------
// JSON output helpers
// ---------------------------------------------------------------------------

/// Print a value as pretty JSON to stdout.
pub fn print_json(value: &impl serde::Serialize) -> Result<()> {
    let json = serde_json::to_string_pretty(value).context("Failed to serialize JSON")?;
    println!("{json}");
    Ok(())
}

/// Print a value as compact JSON to stdout.
pub fn print_json_compact(value: &impl serde::Serialize) -> Result<()> {
    let json = serde_json::to_string(value).context("Failed to serialize JSON")?;
    println!("{json}");
    Ok(())
}

/// Write a JSON value to a file atomically.
pub fn write_json_file(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    let json = serde_json::to_string_pretty(value).context("Failed to serialize JSON")?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    // Write to a temp file then rename for atomicity.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &json).with_context(|| format!("Failed to write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

/// Recursively compute the total size of a directory in bytes.
pub fn dir_size(path: &std::path::Path) -> u64 {
    fn walk(dir: &std::path::Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, total);
                } else if let Ok(meta) = entry.metadata() {
                    *total += meta.len();
                }
            }
        }
    }
    let mut total = 0u64;
    walk(path, &mut total);
    total
}

/// Count the number of files in a directory tree.
pub fn file_count(path: &std::path::Path) -> u64 {
    fn walk(dir: &std::path::Path, total: &mut u64) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, total);
                } else {
                    *total += 1;
                }
            }
        }
    }
    let mut total = 0u64;
    walk(path, &mut total);
    total
}

// ---------------------------------------------------------------------------
// Network helpers
// ---------------------------------------------------------------------------

/// Check whether a TCP port is open on a host.
pub async fn port_open(host: &str, port: u16) -> bool {
    tokio::net::TcpStream::connect((host, port)).await.is_ok()
}

/// Get the process id of the running gateway, if any.
pub fn gateway_pid() -> Option<u32> {
    let path = data_dir().join("gateway.pid");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Read the log tail from a file, returning the last `n` lines.
pub fn log_tail(path: &std::path::Path, n: usize) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|c| {
            let lines: Vec<String> = c.lines().map(|l| l.to_string()).collect();
            let start = lines.len().saturating_sub(n);
            lines[start..].to_vec()
        })
        .unwrap_or_default()
}
