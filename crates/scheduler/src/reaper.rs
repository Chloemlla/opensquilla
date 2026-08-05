use chrono::Utc;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::persistence::JobStore;
use crate::types::JobStatus;

/// Expired session reaper.
///
/// Periodically cleans up stale sessions and old execution records.
pub struct SessionReaper {
    store: Arc<Mutex<JobStore>>,
    session_ttl_hours: u64,
    keep_executions: u64,
    completed_job_retention_days: u64,
    cycles_run: std::sync::atomic::AtomicU64,
}

impl SessionReaper {
    pub fn new(store: Arc<Mutex<JobStore>>) -> Self {
        Self {
            store,
            session_ttl_hours: 24,
            keep_executions: 100,
            completed_job_retention_days: 7,
            cycles_run: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn with_session_ttl(mut self, hours: u64) -> Self { self.session_ttl_hours = hours; self }
    pub fn with_keep_executions(mut self, keep: u64) -> Self { self.keep_executions = keep; self }
    pub fn with_completed_job_retention(mut self, days: u64) -> Self { self.completed_job_retention_days = days; self }

    /// Run one reaper cycle.
    pub async fn run_cycle(&self) -> ReaperResult {
        let cycle_start = std::time::Instant::now();
        let store = self.store.lock().await;
        let mut cleaned_executions = 0u64;
        let mut cleaned_jobs = 0u64;

        if let Ok(count) = store.cleanup_executions(self.keep_executions) {
            cleaned_executions = count;
            if count > 0 { debug!("Reaper: cleaned up {} old execution records", count); }
        } else { warn!("Reaper: failed to cleanup executions"); }

        if let Ok(count) = store.cleanup_completed_jobs(self.completed_job_retention_days) {
            cleaned_jobs = count;
            if count > 0 { debug!("Reaper: cleaned up {} completed jobs", count); }
        } else { warn!("Reaper: failed to cleanup completed jobs"); }

        if let Err(e) = self.auto_disable_stale_failed_jobs(&store) {
            warn!("Reaper: failed to auto-disable stale jobs: {}", e);
        }

        let duration_ms = cycle_start.elapsed().as_millis() as u64;
        self.cycles_run.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        if cleaned_executions > 0 || cleaned_jobs > 0 {
            info!("Reaper: {} executions, {} jobs cleaned in {}ms", cleaned_executions, cleaned_jobs, duration_ms);
        }

        ReaperResult { cleaned_executions, cleaned_jobs, disabled_jobs: 0, duration_ms }
    }

    fn auto_disable_stale_failed_jobs(&self, store: &JobStore) -> Result<(), crate::persistence::StoreError> {
        let failed_jobs = store.list_jobs(Some(JobStatus::Failed))?;
        let now = Utc::now();
        for job in failed_jobs {
            if let Some(last_run) = job.last_run_at {
                if (now - last_run).num_hours() as u64 >= self.session_ttl_hours {
                    store.update_job_status(&job.id, JobStatus::Disabled)?;
                    info!("Reaper: auto-disabled job '{}' after {} hours in Failed status", job.name, self.session_ttl_hours);
                }
            }
        }
        Ok(())
    }

    pub fn cycles_run(&self) -> u64 { self.cycles_run.load(std::sync::atomic::Ordering::SeqCst) }
}

#[derive(Debug, Clone)]
pub struct ReaperResult {
    pub cleaned_executions: u64,
    pub cleaned_jobs: u64,
    pub disabled_jobs: u64,
    pub duration_ms: u64,
}
