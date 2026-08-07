//! Session management commands.
//!
//! Implements the `sessions` subcommand against the session crate's SQLite
//! store. Listing, showing, archiving, deleting, and exporting sessions all go
//! through a [`SessionManager`] built on a file-backed `SessionStorage`.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use opensquilla_session::manager::SessionManager;
use opensquilla_session::{Session, SessionMode, SessionStatus, TranscriptEntry};
use serde::Serialize;
use tracing::info;
use uuid::Uuid;

use crate::util;

/// The response shape for a single session row.
#[derive(Serialize)]
struct SessionRow {
    id: Uuid,
    name: String,
    mode: String,
    status: String,
    message_count: u64,
    total_tokens: u64,
    total_cost_usd: f64,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

/// The export envelope for a session and its transcript.
#[derive(Serialize)]
struct SessionExport {
    session: Session,
    messages: Vec<TranscriptEntry>,
    exported_at: DateTime<Utc>,
}

fn status_str(status: &SessionStatus) -> &'static str {
    match status {
        SessionStatus::Active => "active",
        SessionStatus::Paused => "paused",
        SessionStatus::Archived => "archived",
        SessionStatus::Compacting => "compacting",
        SessionStatus::Killed => "killed",
    }
}

fn mode_str(mode: &SessionMode) -> &'static str {
    match mode {
        SessionMode::Chat => "chat",
        SessionMode::Plan => "plan",
        SessionMode::Agent => "agent",
        SessionMode::Batch => "batch",
    }
}

/// List all sessions for the default agent with status and usage.
pub async fn list_sessions() -> Result<()> {
    list_sessions_filtered(None, 50).await
}

/// List sessions with optional status filter and limit.
pub async fn list_sessions_filtered(status: Option<String>, limit: u64) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let mut sessions = manager
        .list_sessions(&util::default_agent_id(), limit, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    // Apply status filter.
    if let Some(ref status_filter) = status {
        sessions.retain(|s| status_str(&s.status).eq_ignore_ascii_case(status_filter));
    }

    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }

    let rows: Vec<SessionRow> = sessions
        .into_iter()
        .map(|s| SessionRow {
            id: s.id,
            name: s.name,
            mode: mode_str(&s.mode).to_string(),
            status: status_str(&s.status).to_string(),
            message_count: s.message_count,
            total_tokens: s.total_tokens,
            total_cost_usd: s.total_cost_usd,
            created_at: s.created_at,
            updated_at: s.updated_at,
        })
        .collect();

    println!("{:-<100}", "");
    println!(
        "{:<38} {:<24} {:<8} {:<10} {:>5} {:>12}",
        "ID", "Name", "Mode", "Status", "Msgs", "Cost"
    );
    println!("{:-<100}", "");
    for r in &rows {
        println!(
            "{:<38} {:<24} {:<8} {:<10} {:>5} {:>8.4}",
            r.id, r.name, r.mode, r.status, r.message_count, r.total_cost_usd
        );
    }
    println!("{:-<100}", "");
    Ok(())
}

/// Show detailed information about a single session.
pub async fn show_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let session = load_session(&manager, &id)?;

    println!("Session: {}", session.id);
    println!("  Name:        {}", session.name);
    println!("  Mode:        {}", mode_str(&session.mode));
    println!("  Status:      {}", status_str(&session.status));
    println!("  Created:     {}", session.created_at.to_rfc3339());
    println!("  Updated:     {}", session.updated_at.to_rfc3339());
    println!("  Messages:    {}", session.message_count);
    println!("  Tokens:      {}", session.total_tokens);
    println!("  Cost (USD):  {:.4}", session.total_cost_usd);
    if let Some(parent) = session.parent_session_id {
        println!("  Parent:      {parent}");
    }

    let transcript = manager
        .get_transcript(&session.id, 1000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    if !transcript.is_empty() {
        println!();
        println!("  Recent messages:");
        let start = transcript.len().saturating_sub(5);
        for entry in &transcript[start..] {
            let preview: String = entry.content.chars().take(80).collect();
            println!(
                "    [{}] {}: {}",
                entry.created_at.format("%H:%M:%S"),
                entry.role,
                preview
            );
        }
    }
    Ok(())
}

/// Delete a session and its transcript from the store.
pub async fn delete_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    println!("Deleting session {} ({})...", session.id, session.name);
    manager
        .storage()
        .delete_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to delete session: {e}"))?;
    info!("Session {id} deleted");
    println!("Session {id} deleted successfully.");
    Ok(())
}

/// Archive a session by transitioning its status to `Archived`.
pub async fn archive_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let mut session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    session.status = SessionStatus::Archived;
    session.updated_at = Utc::now();
    manager
        .storage()
        .update_session(&session)
        .map_err(|e| anyhow::anyhow!("Failed to archive session: {e}"))?;
    info!("Session {id} archived");
    println!("Session {id} archived.");
    Ok(())
}

/// Export a session and its full transcript to JSON, optionally to a file.
pub async fn export_session(id: String, output: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let session = load_session(&manager, &id)?;

    let messages = manager
        .get_transcript(&session.id, 10000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;

    let export = SessionExport {
        session,
        messages,
        exported_at: Utc::now(),
    };
    let json = util::to_pretty_json(&export)?;

    match output {
        Some(path) => {
            if let Some(dir) = Path::new(&path).parent() {
                std::fs::create_dir_all(dir).ok();
            }
            std::fs::write(&path, json).with_context(|| format!("Failed to write {path}"))?;
            println!("Session exported to {path}");
        }
        None => println!("{json}"),
    }
    Ok(())
}

/// Create a new session with optional name and mode.
pub async fn create_session(name: Option<String>, mode: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let session_mode = match mode.as_deref() {
        Some("plan") => SessionMode::Plan,
        Some("agent") => SessionMode::Agent,
        Some("batch") => SessionMode::Batch,
        _ => SessionMode::Chat,
    };
    let session = manager
        .create_session(
            util::default_agent_id(),
            name.unwrap_or_else(|| "New Session".to_string()),
            String::new(),
            session_mode,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create session: {e}"))?;
    println!("Created session {} ({})", session.id, session.name);
    Ok(())
}

/// Print the transcript for a session with an optional limit.
pub async fn show_messages(id: String, limit: Option<u64>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let session = load_session(&manager, &id)?;

    let max = limit.unwrap_or(10000);
    let messages = manager
        .get_transcript(&session.id, max, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    if messages.is_empty() {
        println!("No messages in session {id}.");
        return Ok(());
    }
    for entry in &messages {
        let ts = entry.created_at.format("%Y-%m-%d %H:%M:%S");
        println!("[{ts}] {:>9}: {}", entry.role, entry.content);
        println!();
    }
    Ok(())
}

/// Fork a session into a new session.
pub async fn fork_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let original = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    let forked = manager
        .fork_session(
            &uid,
            "cli".to_string(),
            Some(format!("Fork of {}", original.name)),
        )
        .map_err(|e| anyhow::anyhow!("Failed to fork session: {e}"))?;
    println!("Forked session {} -> {}", original.id, forked.id);
    println!("  Name: {}", forked.name);
    Ok(())
}

/// Kill a session (force stop).
pub async fn kill_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let mut session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    session.status = SessionStatus::Killed;
    session.updated_at = Utc::now();
    manager
        .storage()
        .update_session(&session)
        .map_err(|e| anyhow::anyhow!("Failed to kill session: {e}"))?;
    println!("Killed session {id}.");
    Ok(())
}

/// Pause a session.
pub async fn pause_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let mut session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    session.status = SessionStatus::Paused;
    session.updated_at = Utc::now();
    manager
        .storage()
        .update_session(&session)
        .map_err(|e| anyhow::anyhow!("Failed to pause session: {e}"))?;
    println!("Paused session {id}.");
    Ok(())
}

/// Resume a paused session.
pub async fn resume_session(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let mut session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    session.status = SessionStatus::Active;
    session.updated_at = Utc::now();
    manager
        .storage()
        .update_session(&session)
        .map_err(|e| anyhow::anyhow!("Failed to resume session: {e}"))?;
    println!("Resumed session {id}.");
    Ok(())
}

/// Compact a session's context window.
pub async fn compact_session_cmd(id: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = parse_id(&id)?;
    let _session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))?;

    let entries = manager
        .get_transcript(&uid, 500, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    let report = manager
        .compact(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to compact session: {e}"))?;
    println!(
        "Compacted session {id}: {} entries, {} -> {} tokens ({})",
        report.entries_compacted,
        report.tokens_before,
        report.tokens_after,
        report.strategy.label(),
    );
    let _ = entries;
    Ok(())
}

/// Search sessions by name or content.
pub async fn search_sessions(query: String, limit: u64) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let sessions = manager
        .list_sessions(&util::default_agent_id(), 1000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    let needle = query.to_lowercase();
    let matches: Vec<_> = sessions
        .into_iter()
        .filter(|s| {
            s.name.to_lowercase().contains(&needle)
                || s.system_prompt.to_lowercase().contains(&needle)
        })
        .take(limit as usize)
        .collect();

    if matches.is_empty() {
        println!("No sessions matching '{query}'.");
        return Ok(());
    }

    println!("Sessions matching '{query}' ({}):", matches.len());
    for s in &matches {
        println!(
            "  {} {} ({} messages, {} tokens)",
            s.id, s.name, s.message_count, s.total_tokens
        );
    }
    Ok(())
}

fn parse_id(id: &str) -> Result<Uuid> {
    Uuid::parse_str(id).map_err(|_| anyhow::anyhow!("Invalid session id: {id}"))
}

fn load_session(manager: &SessionManager, id: &str) -> Result<Session> {
    let uid = parse_id(id)?;
    manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{id}' not found"))
}
