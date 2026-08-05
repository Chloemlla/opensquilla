use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::handlers::HandlerRegistry;
use crate::jobs::JobExecutor;
use crate::ops::JobOps;
use crate::persistence::JobStore;
use crate::reaper::{ReaperResult, SessionReaper};
use crate::timer::{TickHandle, TickLoop};
use crate::types::SchedulerStats;

/// The main scheduler engine.
pub struct SchedulerEngine {
    store: Arc<Mutex<JobStore>>,
    handlers: Arc<HandlerRegistry>,
    ops: Arc<JobOps>,
    executor: Arc<JobExecutor>,
    reaper: Arc<SessionReaper>,
    tick_loop: Option<TickLoop>,
    reaper_tick_loop: Option<TickLoop>,
    tick_handle: Option<TickHandle>,
    reaper_tick_handle: Option<TickHandle>,
    check_interval_secs: u64,
    reaper_interval_secs: u64,
    running: bool,
}

impl SchedulerEngine {
    /// Create a new scheduler engine.
    pub fn new(store: JobStore, handlers: HandlerRegistry) -> Self {
        let store = Arc::new(Mutex::new(store));
        let handlers = Arc::new(handlers);
        let ops = Arc::new(JobOps::new(store.clone(), handlers.clone()));
        let executor = Arc::new(JobExecutor::new(store.clone(), handlers.clone()));
        let reaper = Arc::new(SessionReaper::new(store.clone()));

        Self {
            store,
            handlers,
            ops,
            executor,
            reaper,
            tick_loop: None,
            reaper_tick_loop: None,
            tick_handle: None,
            reaper_tick_handle: None,
            check_interval_secs: 10,
            reaper_interval_secs: 3600,
            running: false,
        }
    }

    pub fn with_check_interval(mut self, secs: u64) -> Self {
        self.check_interval_secs = secs;
        self
    }
    pub fn with_reaper_interval(mut self, secs: u64) -> Self {
        self.reaper_interval_secs = secs;
        self
    }

    /// Start the scheduler engine.
    pub fn start(&mut self) {
        if self.running {
            warn!("Scheduler engine is already running");
            return;
        }

        info!(
            "Starting scheduler engine (check: {}s, reaper: {}s)",
            self.check_interval_secs, self.reaper_interval_secs
        );

        let tick_loop = TickLoop::new("scheduler", self.check_interval_secs);
        let executor = self.executor.clone();
        let ops = self.ops.clone();

        let tick_handle = tick_loop.start_async(move |_tick| {
            let executor = executor.clone();
            let ops = ops.clone();
            async move {
                match ops.list_due_jobs().await {
                    Ok(due_jobs) => {
                        if !due_jobs.is_empty() {
                            info!("Scheduler: {} jobs due for execution", due_jobs.len());
                            let handles = executor.execute_due_jobs(due_jobs).await;
                            info!("Scheduler: {} jobs dispatched", handles.len());
                        }
                    }
                    Err(e) => error!("Scheduler: failed to list due jobs: {}", e),
                }
            }
        });

        let reaper_tick_loop = TickLoop::new("reaper", self.reaper_interval_secs);
        let reaper_clone = self.reaper.clone();
        let reaper_tick_handle = reaper_tick_loop.start_async(move |_tick| {
            let reaper = reaper_clone.clone();
            async move {
                let result = reaper.run_cycle().await;
                if result.cleaned_executions > 0 || result.cleaned_jobs > 0 {
                    info!(
                        "Reaper: cleaned {} executions, {} jobs in {}ms",
                        result.cleaned_executions, result.cleaned_jobs, result.duration_ms
                    );
                }
            }
        });

        self.tick_loop = Some(tick_loop);
        self.reaper_tick_loop = Some(reaper_tick_loop);
        self.tick_handle = Some(tick_handle);
        self.reaper_tick_handle = Some(reaper_tick_handle);
        self.running = true;
        info!("Scheduler engine started");
    }

    /// Stop the scheduler engine.
    pub fn stop(&mut self) {
        if !self.running {
            warn!("Scheduler engine is not running");
            return;
        }
        info!("Stopping scheduler engine...");
        if let Some(ref h) = self.tick_handle {
            h.stop();
        }
        if let Some(ref h) = self.reaper_tick_handle {
            h.stop();
        }
        self.running = false;
        info!("Scheduler engine stopped");
    }

    pub fn is_running(&self) -> bool {
        self.running
    }
    pub fn ops(&self) -> &JobOps {
        &self.ops
    }
    pub fn handlers(&self) -> &HandlerRegistry {
        &self.handlers
    }
    pub fn reaper(&self) -> &SessionReaper {
        &self.reaper
    }
    pub fn check_interval_secs(&self) -> u64 {
        self.check_interval_secs
    }

    /// Get the underlying job store (for advanced operations).
    pub fn store(&self) -> &Arc<Mutex<JobStore>> {
        &self.store
    }

    pub async fn get_stats(&self) -> Result<SchedulerStats, crate::ops::OpsError> {
        self.ops.get_stats().await
    }
    pub async fn run_reaper_cycle(&self) -> ReaperResult {
        self.reaper.run_cycle().await
    }
}

/// Builder for constructing a SchedulerEngine with custom configuration.
pub struct SchedulerBuilder {
    store: Option<JobStore>,
    handlers: HandlerRegistry,
    check_interval_secs: u64,
    reaper_interval_secs: u64,
    db_path: Option<String>,
}

impl SchedulerBuilder {
    pub fn new() -> Self {
        Self {
            store: None,
            handlers: HandlerRegistry::with_defaults(),
            check_interval_secs: 10,
            reaper_interval_secs: 3600,
            db_path: None,
        }
    }

    pub fn with_store(mut self, store: JobStore) -> Self {
        self.store = Some(store);
        self
    }
    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }
    pub fn with_handler(mut self, handler: Box<dyn crate::handlers::CronJobHandler>) -> Self {
        self.handlers.register(handler);
        self
    }
    pub fn with_check_interval(mut self, secs: u64) -> Self {
        self.check_interval_secs = secs;
        self
    }
    pub fn with_reaper_interval(mut self, secs: u64) -> Self {
        self.reaper_interval_secs = secs;
        self
    }

    pub fn build(self) -> Result<SchedulerEngine, Box<dyn std::error::Error>> {
        let store = if let Some(store) = self.store {
            store
        } else if let Some(path) = self.db_path {
            JobStore::open(&path)?
        } else {
            JobStore::in_memory()?
        };
        Ok(SchedulerEngine::new(store, self.handlers)
            .with_check_interval(self.check_interval_secs)
            .with_reaper_interval(self.reaper_interval_secs))
    }
}

impl Default for SchedulerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scheduler_start_stop() {
        let store = JobStore::in_memory().unwrap();
        let mut engine =
            SchedulerEngine::new(store, HandlerRegistry::with_defaults()).with_check_interval(60);
        assert!(!engine.is_running());
        engine.start();
        assert!(engine.is_running());
        engine.stop();
        assert!(!engine.is_running());
    }

    #[tokio::test]
    async fn test_scheduler_builder() {
        let engine = SchedulerBuilder::new()
            .with_check_interval(30)
            .with_reaper_interval(7200)
            .build()
            .unwrap();
        assert_eq!(engine.check_interval_secs(), 30);
    }

    #[tokio::test]
    async fn test_scheduler_stats() {
        let store = JobStore::in_memory().unwrap();
        let engine = SchedulerEngine::new(store, HandlerRegistry::with_defaults());
        let stats = engine.get_stats().await.unwrap();
        assert_eq!(stats.total_jobs, 0);
    }
}
