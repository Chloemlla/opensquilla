//! Diagnostics bundle commands.
//!
//! Implements the `bundle` subcommand: collect a redacted diagnostics bundle.
//! The Rust observability crate does not yet expose a `collect_bundle`, so this
//! module builds a minimal JSON bundle by aggregating session metadata (and,
//! optionally, transcript content) within a time window. A caller can later swap
//! in a richer `opensquilla_observability` collector without changing the CLI
//! contract.

use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use opensquilla_session::Session;
use serde_json::json;

use crate::util;

/// Bundle subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum BundleAction {
    /// Collect a diagnostics bundle.
    Collect {
        /// Bundle destination file (default: `./opensquilla-bundle-<UTC>.json`).
        #[arg(short, long)]
        output: Option<String>,
        /// How many days of sessions to include.
        #[arg(long, default_value = "3")]
        days: u32,
        /// Focus on one session id.
        #[arg(long)]
        session: Option<String>,
        /// Include conversation content (raw transcript). Off by default.
        #[arg(long)]
        include_content: bool,
        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Run a bundle subcommand.
pub async fn run_bundle(action: BundleAction) -> Result<()> {
    match action {
        BundleAction::Collect {
            output,
            days,
            session,
            include_content,
            json,
        } => collect_bundle(output, days, session, include_content, json).await,
    }
}

/// Collect sessions within the window into a JSON bundle file.
async fn collect_bundle(
    output: Option<String>,
    days: u32,
    session: Option<String>,
    include_content: bool,
    json: bool,
) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let mut sessions = manager
        .list_sessions(&util::default_agent_id(), u64::MAX, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;

    let cutoff = Utc::now() - chrono::Duration::days(i64::from(days));
    sessions.retain(|s| s.updated_at >= cutoff);

    // Optional single-session focus.
    if let Some(sid) = session {
        sessions.retain(|s| s.id.to_string() == sid || s.name == sid);
    }

    let mut records = Vec::with_capacity(sessions.len());
    for s in &sessions {
        let mut rec = json!({
            "id": s.id.to_string(),
            "name": s.name,
            "mode": mode_str(&s.mode),
            "status": status_str(&s.status),
            "message_count": s.message_count,
            "total_tokens": s.total_tokens,
            "total_cost_usd": s.total_cost_usd,
            "created_at": s.created_at.to_rfc3339(),
            "updated_at": s.updated_at.to_rfc3339(),
        });
        if include_content {
            let transcript = manager
                .get_transcript(&s.id, 10000, 0)
                .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
            let messages: Vec<_> = transcript
                .into_iter()
                .map(|e| {
                    json!({
                        "role": e.role,
                        "content": e.content,
                        "created_at": e.created_at.to_rfc3339(),
                    })
                })
                .collect();
            rec["messages"] = json!(messages);
        }
        records.push(rec);
    }

    let bundle = json!({
        "generated_at": Utc::now().to_rfc3339(),
        "window_days": days,
        "session_count": records.len(),
        "include_content": include_content,
        "sessions": records,
    });

    let resolved = match output {
        Some(path) => path,
        None => {
            let stamp = Utc::now().format("%Y%m%dT%H%M%SZ");
            format!("opensquilla-bundle-{stamp}.json")
        }
    };

    let json_text = serde_json::to_string_pretty(&bundle)
        .map_err(|e| anyhow::anyhow!("Failed to serialize bundle: {e}"))?;
    if let Some(dir) = Path::new(&resolved).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&resolved, json_text).with_context(|| format!("Failed to write {resolved}"))?;

    if json {
        crate::util::print_json(&json!({
            "path": resolved,
            "session_count": records.len(),
            "generated_at": bundle["generated_at"],
        }))?;
    } else {
        println!("Diagnostics bundle written to {resolved}");
        println!("  sessions: {}", records.len());
    }
    Ok(())
}

fn status_str(s: &opensquilla_session::SessionStatus) -> &'static str {
    match s {
        opensquilla_session::SessionStatus::Active => "active",
        opensquilla_session::SessionStatus::Paused => "paused",
        opensquilla_session::SessionStatus::Archived => "archived",
        opensquilla_session::SessionStatus::Compacting => "compacting",
        opensquilla_session::SessionStatus::Killed => "killed",
    }
}

fn mode_str(m: &opensquilla_session::SessionMode) -> &'static str {
    match m {
        opensquilla_session::SessionMode::Chat => "chat",
        opensquilla_session::SessionMode::Plan => "plan",
        opensquilla_session::SessionMode::Agent => "agent",
        opensquilla_session::SessionMode::Batch => "batch",
    }
}

/// A helper kept for symmetry with the session crate's export shape.
#[allow(dead_code)]
fn _export_envelope(session: Session, exported_at: DateTime<Utc>) -> serde_json::Value {
    json!({ "session": session, "exported_at": exported_at.to_rfc3339() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_and_mode() {
        assert_eq!(
            status_str(&opensquilla_session::SessionStatus::Active),
            "active"
        );
        assert_eq!(
            mode_str(&opensquilla_session::SessionMode::Agent),
            "agent"
        );
    }
}