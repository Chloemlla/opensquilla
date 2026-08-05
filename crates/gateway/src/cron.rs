//! Cron / scheduler RPC handlers.
//!
//! Provides `rpc_cron` for scheduled-task management, backed by the
//! scheduler crate's [`JobOps`] CRUD facade over a SQLite [`JobStore`].

use std::sync::Arc;
use opensquilla_core::error::AppError;
use opensquilla_scheduler::engine::SchedulerEngine;
use opensquilla_scheduler::types::{CronJob, JobExecution, JobStatus, ScheduleKind, SchedulerStats};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::rpc::{rpc_handler, RpcRegistry};

/// A shared scheduler handle. Wraps an [`SchedulerEngine`] behind a Mutex so
/// that concurrent RPC calls serialize on the ops facade.
#[derive(Clone)]
pub struct SchedulerHandle {
    engine: Arc<Mutex<SchedulerEngine>>,
}

impl SchedulerHandle {
    /// Build a handle from an existing engine.
    pub fn new(engine: SchedulerEngine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    /// Build a handle backed by an in-memory store with default handlers.
    pub fn in_memory() -> Result<Self, AppError> {
        let store = opensquilla_scheduler::persistence::JobStore::in_memory()
            .map_err(|e| AppError::internal(format!("Failed to open scheduler store: {e}")))?;
        let handlers = opensquilla_scheduler::handlers::HandlerRegistry::with_defaults();
        let engine = SchedulerEngine::new(store, handlers);
        Ok(Self::new(engine))
    }
}

/// Resolve a [`ScheduleKind`] from RPC params. Accepts `cron`, `every`, or `at`.
fn parse_schedule(params: &serde_json::Value) -> Result<ScheduleKind, AppError> {
    if let Some(expr) = params.get("cron").and_then(|v| v.as_str()) {
        return Ok(ScheduleKind::Cron(expr.to_string()));
    }
    if let Some(secs) = params.get("every").and_then(|v| v.as_u64()) {
        return Ok(ScheduleKind::Every(secs));
    }
    if let Some(at) = params.get("at").and_then(|v| v.as_str()) {
        let dt = chrono::DateTime::parse_from_rfc3339(at)
            .map_err(|e| AppError::bad_request(format!("Invalid 'at' datetime: {e}")))?
            .with_timezone(&chrono::Utc);
        return Ok(ScheduleKind::At(dt));
    }
    Err(AppError::bad_request(
        "Missing schedule: provide one of 'cron', 'every', or 'at'",
    ))
}

/// Parse a `JobStatus` from a string.
fn parse_status(s: &str) -> Result<JobStatus, AppError> {
    match s.to_ascii_lowercase().as_str() {
        "active" => Ok(JobStatus::Active),
        "paused" => Ok(JobStatus::Paused),
        "disabled" => Ok(JobStatus::Disabled),
        "completed" => Ok(JobStatus::Completed),
        "failed" => Ok(JobStatus::Failed),
        other => Err(AppError::bad_request(format!("Unknown status '{other}'"))),
    }
}

/// View of a created/updated job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJobView {
    pub id: String,
    pub name: String,
    pub handler: String,
    pub status: String,
    pub next_run_at: Option<String>,
    pub last_run_at: Option<String>,
    pub max_retries: u32,
    pub retry_delay_secs: u64,
}

impl From<&CronJob> for CronJobView {
    fn from(j: &CronJob) -> Self {
        Self {
            id: j.id.to_string(),
            name: j.name.clone(),
            handler: j.handler.clone(),
            status: format!("{:?}", j.status).to_lowercase(),
            next_run_at: j.next_run_at.map(|dt| dt.to_rfc3339()),
            last_run_at: j.last_run_at.map(|dt| dt.to_rfc3339()),
            max_retries: j.max_retries,
            retry_delay_secs: j.retry_delay_secs,
        }
    }
}

fn ops_err(e: opensquilla_scheduler::ops::OpsError) -> AppError {
    use opensquilla_scheduler::ops::OpsError;
    match e {
        OpsError::JobNotFound(id) => AppError::not_found(format!("Job {id} not found")),
        OpsError::HandlerNotFound(h) => AppError::bad_request(format!("Handler '{h}' not registered")),
        OpsError::InvalidSchedule(msg) => AppError::bad_request(msg),
        OpsError::Store(s) => AppError::internal(format!("Scheduler store error: {s}")),
    }
}

/// Register cron RPC handlers on the given registry.
pub fn register_cron_handlers(registry: &mut RpcRegistry, handle: SchedulerHandle) {
    let handle = Arc::new(handle);

    // cron.create — create a new scheduled job
    registry.register(rpc_handler("cron.create", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?
                    .to_string();
                let handler_name = params
                    .get("handler")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'handler' parameter"))?
                    .to_string();
                let kind = parse_schedule(&params)?;
                let payload = params.get("payload").cloned().unwrap_or(serde_json::Value::Null);
                let tags: std::collections::HashMap<String, String> = params
                    .get("tags")
                    .and_then(|v| v.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();
                let agent_id = params
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| Uuid::parse_str(s).ok());
                let max_retries = params
                    .get("max_retries")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as u32;
                let retry_delay_secs = params
                    .get("retry_delay_secs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10);

                let engine = handle.engine.lock().await;
                let job = engine
                    .ops()
                    .create_job(
                        name, kind, handler_name, payload, tags,
                        agent_id, session_id, max_retries, retry_delay_secs,
                    )
                    .await
                    .map_err(ops_err)?;
                Ok(serde_json::to_value(CronJobView::from(&job))
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // cron.get — fetch a job by id
    registry.register(rpc_handler("cron.get", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let engine = handle.engine.lock().await;
                let job = engine.ops().get_job(id).await.map_err(ops_err)?;
                Ok(serde_json::to_value(CronJobView::from(&job))
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // cron.list — list jobs, optionally filtered by status
    registry.register(rpc_handler("cron.list", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let status_filter = params
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(parse_status)
                    .transpose()?;
                let engine = handle.engine.lock().await;
                let jobs = engine.ops().list_jobs(status_filter).await.map_err(ops_err)?;
                let views: Vec<CronJobView> = jobs.iter().map(CronJobView::from).collect();
                Ok(serde_json::json!({
                    "jobs": views,
                    "count": views.len(),
                }))
            }
        }
    }));

    // cron.update — update an existing job
    registry.register(rpc_handler("cron.update", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let name = params.get("name").and_then(|v| v.as_str()).map(String::from);
                let kind = if params.get("cron").is_some()
                    || params.get("every").is_some()
                    || params.get("at").is_some()
                {
                    Some(parse_schedule(&params)?)
                } else {
                    None
                };
                let handler_name = params.get("handler").and_then(|v| v.as_str()).map(String::from);
                let payload = params.get("payload").cloned();
                let max_retries = params.get("max_retries").and_then(|v| v.as_u64()).map(|n| n as u32);
                let retry_delay_secs = params.get("retry_delay_secs").and_then(|v| v.as_u64());

                let engine = handle.engine.lock().await;
                let job = engine
                    .ops()
                    .update_job(id, name, kind, handler_name, payload, max_retries, retry_delay_secs)
                    .await
                    .map_err(ops_err)?;
                Ok(serde_json::to_value(CronJobView::from(&job))
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // cron.delete — delete a job
    registry.register(rpc_handler("cron.delete", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let engine = handle.engine.lock().await;
                engine.ops().delete_job(id).await.map_err(ops_err)?;
                Ok(serde_json::json!({"deleted": true, "id": id.to_string()}))
            }
        }
    }));

    // cron.pause — pause an active job
    registry.register(rpc_handler("cron.pause", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let engine = handle.engine.lock().await;
                engine.ops().pause_job(id).await.map_err(ops_err)?;
                Ok(serde_json::json!({"paused": true, "id": id.to_string()}))
            }
        }
    }));

    // cron.resume — resume a paused job
    registry.register(rpc_handler("cron.resume", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let engine = handle.engine.lock().await;
                engine.ops().resume_job(id).await.map_err(ops_err)?;
                Ok(serde_json::json!({"resumed": true, "id": id.to_string()}))
            }
        }
    }));

    // cron.disable — permanently disable a job
    registry.register(rpc_handler("cron.disable", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let engine = handle.engine.lock().await;
                engine.ops().disable_job(id).await.map_err(ops_err)?;
                Ok(serde_json::json!({"disabled": true, "id": id.to_string()}))
            }
        }
    }));

    // cron.stats — scheduler statistics
    registry.register(rpc_handler("cron.stats", {
        let handle = handle.clone();
        move |_params| {
            let handle = handle.clone();
            async move {
                let engine = handle.engine.lock().await;
                let stats: SchedulerStats = engine.ops().get_stats().await.map_err(ops_err)?;
                Ok(serde_json::to_value(stats)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // cron.executions — list execution history for a job
    registry.register(rpc_handler("cron.executions", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let id = parse_job_id(&params)?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50);
                let offset = params.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);
                let engine = handle.engine.lock().await;
                let execs: Vec<JobExecution> = engine
                    .ops()
                    .get_executions(id, limit, offset)
                    .await
                    .map_err(ops_err)?;
                Ok(serde_json::json!({
                    "executions": execs,
                    "count": execs.len(),
                }))
            }
        }
    }));

    // cron.due — list jobs currently due for execution
    registry.register(rpc_handler("cron.due", {
        let handle = handle.clone();
        move |_params| {
            let handle = handle.clone();
            async move {
                let engine = handle.engine.lock().await;
                let due = engine.ops().list_due_jobs().await.map_err(ops_err)?;
                let views: Vec<CronJobView> = due.iter().map(CronJobView::from).collect();
                Ok(serde_json::json!({
                    "due": views,
                    "count": views.len(),
                }))
            }
        }
    }));
}

fn parse_job_id(params: &serde_json::Value) -> Result<Uuid, AppError> {
    params
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))
        .and_then(|s| Uuid::parse_str(s).map_err(|_| AppError::bad_request("Invalid job id")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cron_create_get_list() {
        let handle = SchedulerHandle::in_memory().unwrap();
        let mut registry = RpcRegistry::new();
        register_cron_handlers(&mut registry, handle);

        let params = serde_json::json!({
            "name": "heartbeat-job",
            "handler": "heartbeat",
            "every": 60u64,
            "payload": {"k": "v"},
        });
        let r = registry.dispatch("cron.create", params).await;
        let resp = r.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();
        assert_eq!(resp["handler"], "heartbeat");

        let r = registry
            .dispatch("cron.get", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry.dispatch("cron.list", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        let r = registry.dispatch("cron.stats", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["total_jobs"], 1);
    }

    #[tokio::test]
    async fn test_cron_pause_resume() {
        let handle = SchedulerHandle::in_memory().unwrap();
        let mut registry = RpcRegistry::new();
        register_cron_handlers(&mut registry, handle);

        let params = serde_json::json!({
            "name": "j",
            "handler": "heartbeat",
            "every": 30u64,
        });
        let resp = registry.dispatch("cron.create", params).await.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();

        let r = registry.dispatch("cron.pause", serde_json::json!({"id": id})).await;
        assert!(r.unwrap().is_ok());

        let r = registry.dispatch("cron.resume", serde_json::json!({"id": id})).await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_cron_create_unknown_handler() {
        let handle = SchedulerHandle::in_memory().unwrap();
        let mut registry = RpcRegistry::new();
        register_cron_handlers(&mut registry, handle);

        let params = serde_json::json!({
            "name": "bad",
            "handler": "nonexistent",
            "every": 30u64,
        });
        let r = registry.dispatch("cron.create", params).await;
        assert!(r.unwrap().is_err());
    }
}
