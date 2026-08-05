use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use crate::handlers::HandlerRegistry;
use crate::parser::{compute_next_run, validate_schedule_expression};
use crate::persistence::JobStore;
use crate::types::{CronJob, JobStatus, ScheduleKind, SchedulerStats};

/// CRUD operations for managing scheduled jobs.
pub struct JobOps {
    store: Arc<Mutex<JobStore>>,
    handlers: Arc<HandlerRegistry>,
}

impl JobOps {
    pub fn new(store: Arc<Mutex<JobStore>>, handlers: Arc<HandlerRegistry>) -> Self {
        Self { store, handlers }
    }

    /// Create a new cron job.
    pub async fn create_job(
        &self, name: String, kind: ScheduleKind, handler: String,
        payload: serde_json::Value, tags: std::collections::HashMap<String, String>,
        agent_id: Option<Uuid>, session_id: Option<Uuid>,
        max_retries: u32, retry_delay_secs: u64,
    ) -> Result<CronJob, OpsError> {
        if !self.handlers.has_handler(&handler) {
            return Err(OpsError::HandlerNotFound(handler));
        }
        if let Some(error) = validate_schedule_expression(&kind) {
            return Err(OpsError::InvalidSchedule(error));
        }

        let next_run_at = compute_next_run(&kind);
        let job = CronJob {
            id: Uuid::new_v4(), name, kind, handler, payload,
            status: JobStatus::Active, max_retries, retry_delay_secs,
            created_at: chrono::Utc::now(), updated_at: chrono::Utc::now(),
            next_run_at, last_run_at: None, tags, agent_id, session_id,
        };

        let store = self.store.lock().await;
        store.insert_job(&job)?;
        info!("Created job '{}' (id: {}, handler: {}, next_run: {:?})", job.name, job.id, job.handler, job.next_run_at);
        Ok(job)
    }

    /// Update an existing job.
    pub async fn update_job(
        &self, job_id: Uuid, name: Option<String>, kind: Option<ScheduleKind>,
        handler: Option<String>, payload: Option<serde_json::Value>,
        max_retries: Option<u32>, retry_delay_secs: Option<u64>,
    ) -> Result<CronJob, OpsError> {
        let store = self.store.lock().await;
        let mut job = store.get_job(&job_id)?.ok_or(OpsError::JobNotFound(job_id))?;

        if let Some(ref h) = handler {
            if !self.handlers.has_handler(h) { return Err(OpsError::HandlerNotFound(h.clone())); }
            job.handler = h.clone();
        }
        if let Some(ref k) = kind {
            if let Some(e) = validate_schedule_expression(k) { return Err(OpsError::InvalidSchedule(e)); }
            job.kind = k.clone();
            job.next_run_at = compute_next_run(k);
        }
        if let Some(n) = name { job.name = n; }
        if let Some(p) = payload { job.payload = p; }
        if let Some(m) = max_retries { job.max_retries = m; }
        if let Some(r) = retry_delay_secs { job.retry_delay_secs = r; }
        job.updated_at = chrono::Utc::now();
        store.update_job(&job)?;
        info!("Updated job '{}' (id: {})", job.name, job.id);
        Ok(job)
    }

    pub async fn delete_job(&self, job_id: Uuid) -> Result<(), OpsError> {
        let store = self.store.lock().await;
        if store.get_job(&job_id)?.is_none() { return Err(OpsError::JobNotFound(job_id)); }
        store.delete_job(&job_id)?;
        info!("Deleted job (id: {})", job_id);
        Ok(())
    }

    pub async fn list_jobs(&self, status_filter: Option<JobStatus>) -> Result<Vec<CronJob>, OpsError> {
        Ok(self.store.lock().await.list_jobs(status_filter)?)
    }

    pub async fn get_job(&self, job_id: Uuid) -> Result<CronJob, OpsError> {
        self.store.lock().await.get_job(&job_id)?.ok_or(OpsError::JobNotFound(job_id))
    }

    pub async fn pause_job(&self, job_id: Uuid) -> Result<(), OpsError> {
        let store = self.store.lock().await;
        let job = store.get_job(&job_id)?.ok_or(OpsError::JobNotFound(job_id))?;
        if job.status == JobStatus::Active { store.update_job_status(&job_id, JobStatus::Paused)?; }
        Ok(())
    }

    pub async fn resume_job(&self, job_id: Uuid) -> Result<(), OpsError> {
        let store = self.store.lock().await;
        let job = store.get_job(&job_id)?.ok_or(OpsError::JobNotFound(job_id))?;
        if job.status == JobStatus::Paused { store.update_job_status(&job_id, JobStatus::Active)?; }
        Ok(())
    }

    pub async fn disable_job(&self, job_id: Uuid) -> Result<(), OpsError> {
        self.store.lock().await.update_job_status(&job_id, JobStatus::Disabled)?;
        info!("Disabled job (id: {})", job_id);
        Ok(())
    }

    pub async fn get_stats(&self) -> Result<SchedulerStats, OpsError> {
        Ok(self.store.lock().await.stats()?)
    }

    pub async fn list_due_jobs(&self) -> Result<Vec<CronJob>, OpsError> {
        Ok(self.store.lock().await.list_due_jobs()?)
    }

    pub async fn mark_job_run(&self, job_id: Uuid) -> Result<(), OpsError> {
        let store = self.store.lock().await;
        let job = store.get_job(&job_id)?.ok_or(OpsError::JobNotFound(job_id))?;
        let next_run = match &job.kind {
            ScheduleKind::Cron(expr) => crate::parser::CronParser::new(expr).ok().and_then(|p| p.next_after(chrono::Utc::now())),
            ScheduleKind::At(_) => { store.update_job_status(&job_id, JobStatus::Completed)?; None }
            ScheduleKind::Every(secs) => Some(chrono::Utc::now() + chrono::Duration::seconds(*secs as i64)),
        };
        store.mark_job_run(&job_id, next_run)?;
        Ok(())
    }

    pub async fn get_executions(&self, job_id: Uuid, limit: u64, offset: u64) -> Result<Vec<crate::types::JobExecution>, OpsError> {
        Ok(self.store.lock().await.list_executions(&job_id, limit, offset)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpsError {
    #[error("Job not found: {0}")] JobNotFound(Uuid),
    #[error("Handler not found: {0}")] HandlerNotFound(String),
    #[error("Invalid schedule: {0}")] InvalidSchedule(String),
    #[error("Store error: {0}")] Store(#[from] crate::persistence::StoreError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_get_job() {
        let store = JobStore::in_memory().unwrap();
        let store = Arc::new(Mutex::new(store));
        let mut handlers = HandlerRegistry::new();
        handlers.register(Box::new(crate::handlers::HeartbeatHandler::new()));
        let handlers = Arc::new(handlers);
        let ops = JobOps::new(store, handlers);

        let job = ops.create_job("test_job".into(), ScheduleKind::Every(60), "heartbeat".into(),
            serde_json::Value::Null, std::collections::HashMap::new(), None, None, 3, 10).await.unwrap();
        assert_eq!(job.name, "test_job");
        let retrieved = ops.get_job(job.id).await.unwrap();
        assert_eq!(retrieved.id, job.id);
    }

    #[tokio::test]
    async fn test_create_job_missing_handler() {
        let store = JobStore::in_memory().unwrap();
        let store = Arc::new(Mutex::new(store));
        let ops = JobOps::new(store, Arc::new(HandlerRegistry::new()));
        let result = ops.create_job("bad".into(), ScheduleKind::Every(60), "nonexistent".into(),
            serde_json::Value::Null, std::collections::HashMap::new(), None, None, 3, 10).await;
        assert!(matches!(result, Err(OpsError::HandlerNotFound(_))));
    }
}
