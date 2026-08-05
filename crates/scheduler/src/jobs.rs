use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::handlers::HandlerRegistry;
use crate::persistence::JobStore;
use crate::types::{CronJob, JobExecution, JobStatus};

/// Manages the lifecycle of job execution.
pub struct JobExecutor {
    store: Arc<Mutex<JobStore>>,
    handlers: Arc<HandlerRegistry>,
}

impl JobExecutor {
    pub fn new(store: Arc<Mutex<JobStore>>, handlers: Arc<HandlerRegistry>) -> Self {
        Self { store, handlers }
    }

    /// Execute a single job, spawning a tokio task.
    pub async fn execute_job(&self, job: CronJob) -> tokio::task::JoinHandle<()> {
        let store = self.store.clone();
        let handlers = self.handlers.clone();
        let job_id = job.id;
        let job_name = job.name.clone();
        let handler_name = job.handler.clone();
        let max_retries = job.max_retries;
        let retry_delay = job.retry_delay_secs;

        tokio::spawn(async move {
            info!("Executing job '{}' (id: {}, handler: {}, max_retries: {})", job_name, job_id, handler_name, max_retries);
            let mut attempt = 0u32;

            loop {
                let handler = match handlers.get(&handler_name) {
                    Some(h) => h,
                    None => { error!("Handler '{}' not found for job '{}'", handler_name, job_name); break; }
                };

                let mut execution = JobExecution::new(job_id, attempt);
                let exec_start = std::time::Instant::now();

                // Execute handler
                let result = handler.execute(&job).await;
                let duration_ms = exec_start.elapsed().as_millis() as u64;
                execution.duration_ms = Some(duration_ms);

                if result.success {
                    execution.complete(result.result.unwrap_or_default());
                    info!("Job '{}' completed in {}ms (attempt {})", job_name, duration_ms, attempt);
                } else {
                    execution.fail(result.error.clone().unwrap_or_else(|| "Unknown error".to_string()));
                    warn!("Job '{}' failed: {:?} (attempt {}/{})", job_name, result.error, attempt + 1, max_retries + 1);
                }

                // Save execution result
                { let s = store.lock().await; let _ = s.insert_execution(&execution); }

                // Mark job as run
                {
                    let s = store.lock().await;
                    let next_run = match &job.kind {
                        crate::types::ScheduleKind::Cron(expr) =>
                            crate::parser::CronParser::new(expr).ok().and_then(|p| p.next_after(chrono::Utc::now())),
                        crate::types::ScheduleKind::At(_) => { let _ = s.update_job_status(&job_id, JobStatus::Completed); None }
                        crate::types::ScheduleKind::Every(secs) =>
                            Some(chrono::Utc::now() + chrono::Duration::seconds(*secs as i64)),
                    };
                    let _ = s.mark_job_run(&job_id, next_run);
                }

                if result.success { break; }

                attempt += 1;
                if attempt > max_retries {
                    error!("Job '{}' failed after {} attempts. Auto-disabling.", job_name, max_retries + 1);
                    let s = store.lock().await;
                    let _ = s.update_job_status(&job_id, JobStatus::Failed);
                    break;
                }

                let backoff_secs = compute_backoff(attempt, retry_delay);
                debug!("Retrying job '{}' in {}s (attempt {}/{})", job_name, backoff_secs, attempt, max_retries);
                tokio::time::sleep(tokio::time::Duration::from_secs(backoff_secs)).await;
            }
        })
    }

    /// Execute multiple due jobs concurrently.
    pub async fn execute_due_jobs(&self, jobs: Vec<CronJob>) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::with_capacity(jobs.len());
        for job in jobs { handles.push(self.execute_job(job).await); }
        handles
    }
}

/// Compute backoff delay with exponential backoff and jitter.
fn compute_backoff(attempt: u32, base_delay_secs: u64) -> u64 {
    let exponential = base_delay_secs.saturating_mul(1u64 << attempt.min(10));
    let capped = exponential.min(3600);
    let jitter = rand::random::<u64>() % 6;
    capped + jitter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::{CronJobHandler, HandlerResult};
    use crate::persistence::JobStore;
    use crate::types::ScheduleKind;
    use async_trait::async_trait;

    struct TestHandler;
    #[async_trait]
    impl CronJobHandler for TestHandler {
        fn name(&self) -> &str { "test" }
        async fn execute(&self, _job: &CronJob) -> HandlerResult { HandlerResult::success("test ok") }
    }

    #[tokio::test]
    async fn test_execute_job() {
        let store = JobStore::in_memory().unwrap();
        let store = Arc::new(Mutex::new(store));
        let mut handlers = HandlerRegistry::new();
        handlers.register(Box::new(TestHandler));
        let handlers = Arc::new(handlers);

        let job = {
            let s = store.lock().await;
            let job = CronJob::new("exec_test", ScheduleKind::Every(60), "test");
            s.insert_job(&job).unwrap();
            job
        };

        let executor = JobExecutor::new(store.clone(), handlers);
        executor.execute_job(job.clone()).await.await.unwrap();

        let s = store.lock().await;
        let executions = s.list_executions(&job.id, 10, 0).unwrap();
        assert!(!executions.is_empty());
        assert!(executions[0].success);
    }

    #[test]
    fn test_backoff() {
        let d = compute_backoff(0, 10);
        assert!(d >= 10 && d <= 15);
        let d = compute_backoff(1, 10);
        assert!(d >= 20 && d <= 25);
        let d = compute_backoff(10, 10);
        assert!(d <= 3605);
    }
}
