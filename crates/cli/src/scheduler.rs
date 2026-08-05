//! Scheduled task commands.
//!
//! Implements the `scheduler` subcommand against the scheduler crate's
//! [`SchedulerEngine`]. Jobs are persisted to a SQLite store next to the
//! session database. Listing, creating, pausing, resuming, and removing jobs
//! all operate through the engine's [`JobOps`](opensquilla_scheduler::ops::JobOps).

use anyhow::Result;
use chrono::Utc;
use opensquilla_scheduler::SchedulerBuilder;
use opensquilla_scheduler::SchedulerEngine;
use opensquilla_scheduler::types::{CronJob, ScheduleKind};
use std::collections::HashMap;
use tracing::info;
use uuid::Uuid;

use crate::util;

/// Build a scheduler engine backed by a file-based job store.
fn build_engine() -> Result<SchedulerEngine> {
    let path = util::data_dir().join("scheduler.db");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    SchedulerBuilder::new()
        .with_db_path(path.to_string_lossy().to_string())
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build scheduler: {e}"))
}

/// List all scheduled jobs with their status.
pub async fn list_tasks() -> Result<()> {
    let engine = build_engine()?;
    let jobs = engine
        .ops()
        .list_jobs(None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to list jobs: {e}"))?;

    if jobs.is_empty() {
        println!("No scheduled tasks.");
        return Ok(());
    }

    println!("Scheduled tasks ({}):", jobs.len());
    println!("{:-<90}", "");
    println!(
        "{:<38} {:<22} {:<14} {:<12} {:<10}",
        "ID", "Name", "Schedule", "Handler", "Status"
    );
    println!("{:-<90}", "");
    for job in &jobs {
        println!(
            "{:<38} {:<22} {:<14} {:<12} {:?}",
            job.id,
            job.name,
            schedule_str(&job.kind),
            job.handler,
            job.status
        );
    }
    println!("{:-<90}", "");

    let stats = engine
        .ops()
        .get_stats()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load stats: {e}"))?;
    println!(
        "Totals: {} active, {} paused, {} failed, {} completed, {} executions",
        stats.active_jobs,
        stats.paused_jobs,
        stats.failed_jobs,
        stats.completed_jobs,
        stats.total_executions
    );
    Ok(())
}

/// Show the details of a single scheduled job.
pub async fn show_task(id: String) -> Result<()> {
    let engine = build_engine()?;
    let job = load_job(&engine, &id)?;

    println!("Task: {}", job.id);
    println!("  Name:        {}", job.name);
    println!("  Schedule:    {}", schedule_str(&job.kind));
    println!("  Handler:     {}", job.handler);
    println!("  Status:      {:?}", job.status);
    println!("  Max retries: {}", job.max_retries);
    println!("  Created:     {}", job.created_at.to_rfc3339());
    println!("  Next run:    {}", opt_rfc(job.next_run_at));
    println!("  Last run:    {}", opt_rfc(job.last_run_at));
    if !job.payload.is_null() {
        println!("  Payload:     {}", job.payload);
    }
    Ok(())
}

/// Remove (cancel) a scheduled job.
pub async fn cancel_task(id: String) -> Result<()> {
    let engine = build_engine()?;
    let uid = parse_id(&id)?;
    let job = engine
        .ops()
        .get_job(uid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load job: {e}"))?;
    engine
        .ops()
        .delete_job(uid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to delete job: {e}"))?;
    info!("Task {} removed", job.name);
    println!("Removed task: {} ({})", job.name, job.id);
    Ok(())
}

/// Create a new scheduled job.
pub async fn add_task(name: String, schedule: String, handler: Option<String>) -> Result<()> {
    let engine = build_engine()?;
    let kind = parse_schedule(&schedule)?;
    let handler = handler.unwrap_or_else(|| "heartbeat".to_string());

    let job = engine
        .ops()
        .create_job(
            name.clone(),
            kind,
            handler,
            serde_json::Value::Null,
            HashMap::new(),
            Some(util::default_agent_id()),
            None,
            0,
            0,
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create job: {e}"))?;

    println!("Created task: {} ({})", job.name, job.id);
    println!("  Schedule: {}", schedule_str(&job.kind));
    println!("  Handler:  {}", job.handler);
    Ok(())
}

/// Pause a scheduled job.
pub async fn pause_task(id: String) -> Result<()> {
    let engine = build_engine()?;
    let uid = parse_id(&id)?;
    engine
        .ops()
        .pause_job(uid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to pause job: {e}"))?;
    println!("Paused task: {id}");
    Ok(())
}

/// Resume a paused scheduled job.
pub async fn resume_task(id: String) -> Result<()> {
    let engine = build_engine()?;
    let uid = parse_id(&id)?;
    engine
        .ops()
        .resume_job(uid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to resume job: {e}"))?;
    println!("Resumed task: {id}");
    Ok(())
}

/// Remove a scheduled job (alias for cancel).
pub async fn remove_task(id: String) -> Result<()> {
    cancel_task(id).await
}

fn parse_id(id: &str) -> Result<Uuid> {
    Uuid::parse_str(id).map_err(|_| anyhow::anyhow!("Invalid task id: {id}"))
}

fn load_job(engine: &SchedulerEngine, id: &str) -> Result<CronJob> {
    let uid = parse_id(id)?;
    engine
        .ops()
        .get_job(uid)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load job: {e}"))
}

/// Parse a schedule string into a `ScheduleKind`.
///
/// Supported forms:
/// - `every:<seconds>` — run every N seconds
/// - `at:<rfc3339>` — run once at a timestamp
/// - anything else — treated as a POSIX 5-field cron expression
fn parse_schedule(schedule: &str) -> Result<ScheduleKind> {
    if let Some(secs) = schedule.strip_prefix("every:") {
        let secs: u64 = secs
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid 'every' interval: {secs}"))?;
        return Ok(ScheduleKind::Every(secs));
    }
    if let Some(ts) = schedule.strip_prefix("at:") {
        let dt = chrono::DateTime::parse_from_rfc3339(ts)
            .map_err(|_| anyhow::anyhow!("Invalid 'at' timestamp: {ts}"))?
            .with_timezone(&Utc);
        return Ok(ScheduleKind::At(dt));
    }
    Ok(ScheduleKind::Cron(schedule.to_string()))
}

fn schedule_str(kind: &ScheduleKind) -> String {
    match kind {
        ScheduleKind::Cron(expr) => format!("cron {expr}"),
        ScheduleKind::At(dt) => format!("at {}", dt.to_rfc3339()),
        ScheduleKind::Every(secs) => format!("every {secs}s"),
    }
}

fn opt_rfc(dt: Option<chrono::DateTime<Utc>>) -> String {
    match dt {
        Some(d) => d.to_rfc3339(),
        None => "(never)".to_string(),
    }
}
