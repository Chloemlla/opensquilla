//! Crash recovery commands.
//!
//! Implements the `recovery` subcommand against the recovery crate's
//! [`CrashRecovery`]. Lists crash snapshots, inspects one, attempts recovery,
//! validates/repairs session state, and clears the snapshot store.
//!
//! The Python `recovery_cmd.py` drives Desktop profile repair (a separate
//! concern handled by `opensquilla.recovery.*` profile tooling); this Rust
//! port focuses on the crash-snapshot and session-integrity surface that the
//! Rust `opensquilla-recovery` crate exposes.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_recovery::crash::{CrashRecovery, RecoveryError, SessionState};
use tracing::info;

use crate::table::{self, KeyValue};

/// Recovery subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum RecoveryAction {
    /// List recoverable crash snapshots.
    List,
    /// Show details of a single crash snapshot.
    Show {
        /// Snapshot id.
        id: String,
    },
    /// Attempt to recover from a single crash snapshot.
    Recover {
        /// Snapshot id.
        id: String,
    },
    /// Attempt to recover from all outstanding crash snapshots.
    RecoverAll,
    /// Delete a single crash snapshot.
    Delete {
        /// Snapshot id.
        id: String,
    },
    /// Clear all crash snapshots.
    Clear,
}

/// Run a recovery subcommand.
pub async fn run_recovery(action: RecoveryAction) -> Result<()> {
    match action {
        RecoveryAction::List => recovery_list().await,
        RecoveryAction::Show { id } => recovery_show(id).await,
        RecoveryAction::Recover { id } => recovery_recover(id).await,
        RecoveryAction::RecoverAll => recovery_recover_all().await,
        RecoveryAction::Delete { id } => recovery_delete(id).await,
        RecoveryAction::Clear => recovery_clear().await,
    }
}

/// Build a crash recovery manager from the loaded config.
fn build_recovery() -> Result<CrashRecovery> {
    let config = Config::load().context("Failed to load configuration")?;
    Ok(CrashRecovery::new(&config))
}

/// List all recoverable crash snapshots.
pub async fn recovery_list() -> Result<()> {
    let recovery = build_recovery()?;
    let snapshots = recovery.list_snapshots().await;

    if snapshots.is_empty() {
        println!("No recoverable crash snapshots.");
        return Ok(());
    }

    println!("Crash snapshots ({}):", snapshots.len());
    println!("{:-<90}", "");
    println!("{:<38} {:<24} {:<24}", "ID", "Timestamp", "Session");
    println!("{:-<90}", "");
    for s in &snapshots {
        println!(
            "{:<38} {:<24} {:<24}",
            s.id,
            s.timestamp.to_rfc3339(),
            s.session_id.as_deref().unwrap_or("-")
        );
    }
    println!("{:-<90}", "");
    Ok(())
}

/// Show the details of a single crash snapshot.
pub async fn recovery_show(id: String) -> Result<()> {
    let recovery = build_recovery()?;
    let snapshot = recovery
        .load_snapshot(&id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {e}"))?;

    println!("Snapshot: {}", snapshot.id);
    println!("  Timestamp:  {}", snapshot.timestamp.to_rfc3339());
    println!("  Error:      {}", snapshot.error_message);
    println!(
        "  Session:    {}",
        snapshot.session_id.as_deref().unwrap_or("(none)")
    );
    println!("  Checksum:   {}", snapshot.checksum);
    if !snapshot.context.is_empty() {
        let ctx = if snapshot.context.len() > 200 {
            format!("{}…", &snapshot.context[..200])
        } else {
            snapshot.context.clone()
        };
        println!("  Context:    {ctx}");
    }
    Ok(())
}

/// Attempt to recover from a single crash snapshot.
pub async fn recovery_recover(id: String) -> Result<()> {
    let recovery = build_recovery()?;
    let snapshot = recovery
        .load_snapshot(&id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load snapshot: {e}"))?;

    let result = recovery
        .recover_from_snapshot(&snapshot)
        .await
        .map_err(|e| anyhow::anyhow!("Recovery failed: {e}"))?;

    let icon = if result.recovered {
        table::ok()
    } else {
        table::warn()
    };
    println!("{icon} {}", result.message);
    println!("  snapshot: {}", result.snapshot_id);
    println!("  recovered: {}", result.recovered);
    println!("  at:        {}", result.timestamp.to_rfc3339());

    // TODO: drive the engine turn replay (`crates/engine/src/recovery/replay.rs`)
    // once the session store is wired here, to actually resume the interrupted
    // turn. `recover_from_snapshot` only verifies the snapshot and classifies
    // the outcome; turn replay is the engine's responsibility.
    info!(snapshot_id = %result.snapshot_id, recovered = result.recovered, "Recovery attempt complete");
    Ok(())
}

/// Attempt to recover from all outstanding crash snapshots.
pub async fn recovery_recover_all() -> Result<()> {
    let recovery = build_recovery()?;
    let report = recovery
        .recover_from_crash()
        .await
        .map_err(|e| anyhow::anyhow!("Recovery failed: {e}"))?;

    println!("Crash recovery report: {}", report.crash_id);
    println!("  Timestamp:          {}", report.timestamp.to_rfc3339());
    if !report.error_message.is_empty() {
        println!("  Last error:         {}", report.error_message);
    }
    println!(
        "  Affected sessions:  {}",
        if report.affected_sessions.is_empty() {
            "(none)".to_string()
        } else {
            report.affected_sessions.join(", ")
        }
    );
    println!("  Recovered:          {}", report.recovered);
    if !report.recovered_session_ids.is_empty() {
        println!(
            "  Recovered sessions: {}",
            report.recovered_session_ids.join(", ")
        );
    }
    println!("  Actions:");
    for action in &report.recovery_actions {
        println!("    - {action}");
    }
    Ok(())
}

/// Delete a single crash snapshot.
pub async fn recovery_delete(id: String) -> Result<()> {
    let recovery = build_recovery()?;
    recovery
        .delete_snapshot(&id)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to delete snapshot: {e}"))?;
    println!("{} Deleted snapshot {id}", table::ok());
    Ok(())
}

/// Clear all crash snapshots.
pub async fn recovery_clear() -> Result<()> {
    let recovery = build_recovery()?;
    recovery
        .clear_snapshots()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to clear snapshots: {e}"))?;
    println!("{} Cleared all crash snapshots", table::ok());
    Ok(())
}

/// Validate a session's integrity and report issues.
#[allow(dead_code)]
pub async fn validate_session(session: SessionState) -> Result<()> {
    let recovery = build_recovery()?;
    let validation = recovery.validate_session_state(&session).await;
    let icon = if validation.valid {
        table::ok()
    } else {
        table::fail()
    };
    println!(
        "{icon} Session {} ({} messages)",
        validation.session_id, validation.message_count
    );
    KeyValue::new()
        .entry("valid", validation.valid.to_string())
        .entry("issues", validation.issues.len().to_string())
        .print();
    for issue in &validation.issues {
        println!("  {} {issue}", table::warn());
    }
    Ok(())
}

/// Repair a corrupted session in place, reporting the fixes applied.
#[allow(dead_code)]
pub async fn repair_session(session: &mut SessionState) -> Result<()> {
    let recovery = build_recovery()?;
    let fixes = recovery
        .fix_corrupted_session(session)
        .await
        .map_err(|e: RecoveryError| anyhow::anyhow!("Repair failed: {e}"))?;
    if fixes.is_empty() {
        println!("{} No fixes needed", table::ok());
    } else {
        println!("{} Applied {} fix(es):", table::ok(), fixes.len());
        for fix in &fixes {
            println!("  - {fix}");
        }
    }
    Ok(())
}
