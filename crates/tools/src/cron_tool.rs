//! Cron tools: schedule_task, list_tasks, cancel_task.
//!
//! Schedule and manage recurring jobs via `opensquilla-scheduler`'s
//! [`SchedulerEngine`]. The engine is shared (wrapped in `Arc`) between the
//! three tools so a job created by `schedule_task` can be listed by
//! `list_tasks` and removed by `cancel_task`.
//!
//! The engine does not need to be started for these CRUD operations; starting
//! the tick loop is the caller's responsibility.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use opensquilla_scheduler::HandlerRegistry;
use opensquilla_scheduler::JobStore;
use opensquilla_scheduler::engine::SchedulerEngine;
use opensquilla_scheduler::types::{CronJob, JobStatus, ScheduleKind};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Map a scheduler `OpsError` to a tool error.
fn map_ops_error(err: opensquilla_scheduler::ops::OpsError) -> ToolError {
    match err {
        opensquilla_scheduler::ops::OpsError::JobNotFound(id) => {
            ToolError::new("JOB_NOT_FOUND", format!("Scheduled job '{}' not found", id))
        }
        opensquilla_scheduler::ops::OpsError::HandlerNotFound(h) => ToolError::new(
            "HANDLER_NOT_FOUND",
            format!(
                "No cron handler '{}' is registered. Available handlers: {}",
                h,
                HandlerRegistry::with_defaults().handler_names().join(", ")
            ),
        ),
        opensquilla_scheduler::ops::OpsError::InvalidSchedule(s) => {
            ToolError::new("INVALID_SCHEDULE", format!("Invalid schedule: {}", s))
        }
        opensquilla_scheduler::ops::OpsError::Store(e) => ToolError::new(
            "SCHEDULER_STORE_ERROR",
            format!("Scheduler store error: {}", e),
        ),
    }
}

/// Parse an optional UUID parameter.
fn parse_opt_uuid(params: &Value, name: &str) -> Result<Option<Uuid>, ToolError> {
    match params.get(name).and_then(|v| v.as_str()) {
        None | Some("") => Ok(None),
        Some(raw) => Uuid::parse_str(raw).map(Some).map_err(|e| {
            ToolError::invalid_args(format!("Invalid '{}' UUID '{}': {}", name, raw, e))
        }),
    }
}

/// Serialize a `CronJob` to a JSON value for tool output.
fn job_to_json(job: &CronJob) -> Value {
    serde_json::json!({
        "job_id": job.id.to_string(),
        "name": job.name,
        "kind": job.kind,
        "handler": job.handler,
        "status": job.status,
        "payload": job.payload,
        "max_retries": job.max_retries,
        "retry_delay_secs": job.retry_delay_secs,
        "tags": job.tags,
        "agent_id": job.agent_id.map(|id| id.to_string()),
        "session_id": job.session_id.map(|id| id.to_string()),
        "next_run_at": job.next_run_at.map(|t| t.to_rfc3339()),
        "last_run_at": job.last_run_at.map(|t| t.to_rfc3339()),
        "created_at": job.created_at.to_rfc3339(),
    })
}

/// Build a scheduler engine, optionally backed by a SQLite file.
pub fn build_scheduler_engine(db_path: Option<&str>) -> Result<SchedulerEngine, ToolError> {
    match db_path {
        Some(path) => opensquilla_scheduler::SchedulerBuilder::new()
            .with_db_path(path)
            .build()
            .map_err(|e| {
                ToolError::new(
                    "SCHEDULER_ERROR",
                    format!("Failed to build scheduler: {}", e),
                )
            }),
        None => Ok(SchedulerEngine::new(
            JobStore::in_memory().map_err(|e| {
                ToolError::new(
                    "SCHEDULER_ERROR",
                    format!("Failed to open job store: {}", e),
                )
            })?,
            HandlerRegistry::with_defaults(),
        )),
    }
}

/// Tool for creating a scheduled cron job.
pub struct ScheduleTaskTool {
    engine: Arc<SchedulerEngine>,
}

impl ScheduleTaskTool {
    /// Create the tool from a shared scheduler engine.
    pub fn new(engine: SchedulerEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    /// Create the tool from an existing `Arc<SchedulerEngine>`.
    pub fn from_arc(engine: Arc<SchedulerEngine>) -> Self {
        Self { engine }
    }

    /// Build a schedule kind from the `kind`/`cron`/`at`/`every_secs` params.
    fn parse_schedule(&self, params: &Value) -> Result<ScheduleKind, ToolError> {
        match params["kind"]
            .as_str()
            .unwrap_or("cron")
            .to_lowercase()
            .as_str()
        {
            "cron" => {
                let expr = params["cron"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'cron' expression for kind=cron")
                })?;
                Ok(ScheduleKind::Cron(expr.to_string()))
            }
            "at" => {
                let raw = params["at"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'at' timestamp for kind=at"))?;
                let parsed = DateTime::parse_from_rfc3339(raw).map_err(|e| {
                    ToolError::invalid_args(format!("Invalid 'at' timestamp '{}': {}", raw, e))
                })?;
                Ok(ScheduleKind::At(parsed.with_timezone(&Utc)))
            }
            "every" => {
                let secs = params["every_secs"].as_i64().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'every_secs' for kind=every")
                })?;
                if secs <= 0 {
                    return Err(ToolError::invalid_args("'every_secs' must be positive"));
                }
                Ok(ScheduleKind::Every(secs as u64))
            }
            other => Err(ToolError::invalid_args(format!(
                "Unknown schedule kind '{}'. Use cron, at, or every",
                other
            ))),
        }
    }
}

#[async_trait]
impl Tool for ScheduleTaskTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "schedule_task",
                concat!(
                    "Create a scheduled cron job. Supports POSIX cron expressions, ",
                    "one-time execution at a datetime, or recurring execution every N seconds. ",
                    "The handler must be one of the registered cron handlers.",
                ),
                HashMap::from([
                    (
                        "name".to_string(),
                        ParameterDefinition::required_string(
                            "A unique human-readable name for the job",
                        ),
                    ),
                    (
                        "kind".to_string(),
                        ParameterDefinition::string("Schedule kind: cron, at, or every")
                            .enum_values(vec!["cron".into(), "at".into(), "every".into()])
                            .default(serde_json::json!("cron")),
                    ),
                    (
                        "cron".to_string(),
                        ParameterDefinition::string(
                            "POSIX 5-field cron expression, e.g. '0 * * * *' (for kind=cron)",
                        ),
                    ),
                    (
                        "at".to_string(),
                        ParameterDefinition::string(
                            "RFC3339 datetime for one-time execution (for kind=at)",
                        ),
                    ),
                    (
                        "every_secs".to_string(),
                        ParameterDefinition::integer("Repeat interval in seconds (for kind=every)"),
                    ),
                    (
                        "handler".to_string(),
                        ParameterDefinition::string(
                            "Registered cron handler name (default: heartbeat)",
                        )
                        .default(serde_json::json!("heartbeat")),
                    ),
                    (
                        "payload".to_string(),
                        ParameterDefinition::string(
                            "Optional JSON payload string passed to the handler",
                        ),
                    ),
                    (
                        "tags".to_string(),
                        ParameterDefinition::string(
                            "Optional tags as a JSON object string, e.g. {\"env\":\"prod\"}",
                        ),
                    ),
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::string(
                            "Optional agent UUID to associate with the job",
                        ),
                    ),
                    (
                        "session_id".to_string(),
                        ParameterDefinition::string(
                            "Optional session UUID to associate with the job",
                        ),
                    ),
                    (
                        "max_retries".to_string(),
                        ParameterDefinition::integer("Maximum retry attempts on failure")
                            .default(serde_json::json!(3)),
                    ),
                    (
                        "retry_delay_secs".to_string(),
                        ParameterDefinition::integer("Seconds between retries")
                            .default(serde_json::json!(10)),
                    ),
                ]),
            )
            .category("cron")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?
            .to_string();
        let kind = self.parse_schedule(&params)?;
        let handler = params["handler"]
            .as_str()
            .unwrap_or("heartbeat")
            .to_string();
        let payload: Value = params["payload"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null);
        let tags: HashMap<String, String> = params["tags"]
            .as_str()
            .map(|s| serde_json::from_str(s).unwrap_or_default())
            .unwrap_or_default();
        let agent_id = parse_opt_uuid(&params, "agent_id")?;
        let session_id = parse_opt_uuid(&params, "session_id")?;
        let max_retries = params["max_retries"].as_i64().unwrap_or(3).max(0) as u32;
        let retry_delay_secs = params["retry_delay_secs"].as_i64().unwrap_or(10).max(0) as u64;

        let job = self
            .engine
            .ops()
            .create_job(
                name,
                kind,
                handler,
                payload,
                tags,
                agent_id,
                session_id,
                max_retries,
                retry_delay_secs,
            )
            .await
            .map_err(map_ops_error)?;

        let data = serde_json::json!({
            "job": job_to_json(&job),
            "job_id": job.id.to_string(),
        });
        Ok(ToolOutput::success_with_data(
            format!("Scheduled job '{}' ({})", job.name, job.id),
            data,
        ))
    }
}

/// Tool for listing scheduled cron jobs.
pub struct ListTasksTool {
    engine: Arc<SchedulerEngine>,
}

impl ListTasksTool {
    /// Create the tool from a shared scheduler engine.
    pub fn new(engine: SchedulerEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    /// Create the tool from an existing `Arc<SchedulerEngine>`.
    pub fn from_arc(engine: Arc<SchedulerEngine>) -> Self {
        Self { engine }
    }

    /// Parse an optional status filter string into a `JobStatus`.
    fn parse_status(&self, raw: Option<&str>) -> Result<Option<JobStatus>, ToolError> {
        match raw.unwrap_or("").to_lowercase().as_str() {
            "" => Ok(None),
            "active" => Ok(Some(JobStatus::Active)),
            "paused" => Ok(Some(JobStatus::Paused)),
            "disabled" => Ok(Some(JobStatus::Disabled)),
            "completed" => Ok(Some(JobStatus::Completed)),
            "failed" => Ok(Some(JobStatus::Failed)),
            other => Err(ToolError::invalid_args(format!(
                "Unknown status filter '{}'. Use active, paused, disabled, completed, or failed",
                other
            ))),
        }
    }
}

#[async_trait]
impl Tool for ListTasksTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "list_tasks",
                "List scheduled cron jobs, optionally filtered by status.",
                HashMap::from([(
                    "status".to_string(),
                    ParameterDefinition::string(
                        "Optional status filter: active, paused, disabled, completed, failed",
                    ),
                )]),
            )
            .category("cron")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let status = self.parse_status(params["status"].as_str())?;

        let jobs = self
            .engine
            .ops()
            .list_jobs(status)
            .await
            .map_err(map_ops_error)?;

        let items: Vec<Value> = jobs.iter().map(job_to_json).collect();
        let data = serde_json::json!({
            "count": items.len(),
            "jobs": items,
        });

        let content = if items.is_empty() {
            "No scheduled jobs found".to_string()
        } else {
            serde_json::to_string_pretty(&data).unwrap_or_default()
        };
        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for cancelling (deleting) a scheduled cron job.
pub struct CancelTaskTool {
    engine: Arc<SchedulerEngine>,
}

impl CancelTaskTool {
    /// Create the tool from a shared scheduler engine.
    pub fn new(engine: SchedulerEngine) -> Self {
        Self {
            engine: Arc::new(engine),
        }
    }

    /// Create the tool from an existing `Arc<SchedulerEngine>`.
    pub fn from_arc(engine: Arc<SchedulerEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Tool for CancelTaskTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "cancel_task",
                "Cancel and permanently delete a scheduled cron job by its ID.",
                HashMap::from([(
                    "job_id".to_string(),
                    ParameterDefinition::required_string("The UUID of the scheduled job to cancel"),
                )]),
            )
            .category("cron")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let raw = params["job_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'job_id'"))?;
        let job_id = Uuid::parse_str(raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'job_id' UUID '{}': {}", raw, e))
        })?;

        self.engine
            .ops()
            .delete_job(job_id)
            .await
            .map_err(map_ops_error)?;

        let data = serde_json::json!({ "job_id": raw, "cancelled": true });
        Ok(ToolOutput::success_with_data(
            format!("Cancelled scheduled job '{}'", raw),
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_engine() -> SchedulerEngine {
        build_scheduler_engine(None).expect("default engine")
    }

    #[tokio::test]
    async fn test_schedule_and_list_and_cancel() {
        let engine = test_engine();
        let schedule = ScheduleTaskTool::new(engine);
        let list = ListTasksTool::from_arc(schedule.engine.clone());
        let cancel = CancelTaskTool::from_arc(schedule.engine.clone());

        let result = schedule
            .execute(serde_json::json!({
                "name": "test job",
                "kind": "every",
                "every_secs": 60,
                "handler": "heartbeat",
            }))
            .await;
        assert!(result.is_ok(), "schedule failed: {:?}", result.err());
        let job_id = result.unwrap().data.unwrap()["job_id"]
            .as_str()
            .unwrap()
            .to_string();

        let result = list.execute(serde_json::json!({})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.unwrap()["count"].as_u64(), Some(1));

        let result = cancel
            .execute(serde_json::json!({ "job_id": job_id }))
            .await;
        assert!(result.is_ok());

        let result = list.execute(serde_json::json!({})).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.unwrap()["count"].as_u64(), Some(0));
    }

    #[tokio::test]
    async fn test_invalid_schedule_rejected() {
        let engine = test_engine();
        let schedule = ScheduleTaskTool::new(engine);
        let result = schedule
            .execute(serde_json::json!({
                "name": "bad",
                "kind": "cron",
                "cron": "not a cron expression",
                "handler": "heartbeat",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_SCHEDULE");
    }

    #[tokio::test]
    async fn test_cancel_missing_job() {
        let engine = test_engine();
        let cancel = CancelTaskTool::new(engine);
        let result = cancel
            .execute(serde_json::json!({ "job_id": Uuid::new_v4().to_string() }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "JOB_NOT_FOUND");
    }
}
