use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// The kind of schedule for a cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleKind {
    /// POSIX 5-field cron expression (e.g., "0 * * * *").
    Cron(String),
    /// One-time execution at a specific UTC datetime.
    At(DateTime<Utc>),
    /// Recurring execution every N seconds.
    Every(u64),
}

/// The current status of a scheduled job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is active and will be scheduled.
    Active,
    /// Job is paused (will not run, but retains schedule).
    Paused,
    /// Job is disabled (will not run, explicitly turned off).
    Disabled,
    /// Job has completed (for one-time AT jobs).
    Completed,
    /// Job has failed beyond the retry limit.
    Failed,
}

impl JobStatus {
    /// Returns true if the job is eligible to run.
    pub fn is_runnable(&self) -> bool {
        matches!(self, JobStatus::Active)
    }
}

/// A scheduled cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    /// Unique job identifier.
    pub id: Uuid,
    /// Human-readable name for this job.
    pub name: String,
    /// The schedule kind (cron, at, every).
    pub kind: ScheduleKind,
    /// The handler name that will execute this job.
    pub handler: String,
    /// JSON payload passed to the handler on execution.
    pub payload: serde_json::Value,
    /// Current status of the job.
    pub status: JobStatus,
    /// Maximum number of retry attempts on failure.
    pub max_retries: u32,
    /// Delay in seconds between retry attempts.
    pub retry_delay_secs: u64,
    /// When this job was created.
    pub created_at: DateTime<Utc>,
    /// When this job was last updated.
    pub updated_at: DateTime<Utc>,
    /// The next scheduled run time.
    pub next_run_at: Option<DateTime<Utc>>,
    /// The last time this job ran.
    pub last_run_at: Option<DateTime<Utc>>,
    /// Arbitrary tags for filtering and organization.
    pub tags: HashMap<String, String>,
    /// Optional agent ID this job is associated with.
    pub agent_id: Option<Uuid>,
    /// Optional session ID this job is associated with.
    pub session_id: Option<Uuid>,
}

impl CronJob {
    /// Create a new cron job with an auto-generated ID.
    pub fn new(
        name: impl Into<String>,
        kind: ScheduleKind,
        handler: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            kind,
            handler: handler.into(),
            payload: serde_json::Value::Null,
            status: JobStatus::Active,
            max_retries: 3,
            retry_delay_secs: 10,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            next_run_at: None,
            last_run_at: None,
            tags: HashMap::new(),
            agent_id: None,
            session_id: None,
        }
    }
}

/// A single execution record for a cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobExecution {
    /// Unique execution identifier.
    pub id: Uuid,
    /// The job ID this execution belongs to.
    pub job_id: Uuid,
    /// When execution started.
    pub started_at: DateTime<Utc>,
    /// When execution finished (None if still running).
    pub finished_at: Option<DateTime<Utc>>,
    /// Whether the execution was successful.
    pub success: bool,
    /// Optional result payload from the handler.
    pub result: Option<String>,
    /// Error message if execution failed.
    pub error: Option<String>,
    /// Which attempt number this is (0-based).
    pub attempt: u32,
    /// Duration of execution in milliseconds.
    pub duration_ms: Option<u64>,
}

impl JobExecution {
    /// Create a new execution record for the given job.
    pub fn new(job_id: Uuid, attempt: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            job_id,
            started_at: Utc::now(),
            finished_at: None,
            success: false,
            result: None,
            error: None,
            attempt,
            duration_ms: None,
        }
    }

    /// Mark this execution as completed successfully.
    pub fn complete(&mut self, result: String) {
        let now = Utc::now();
        self.duration_ms = Some((now - self.started_at).num_milliseconds() as u64);
        self.finished_at = Some(now);
        self.success = true;
        self.result = Some(result);
    }

    /// Mark this execution as failed.
    pub fn fail(&mut self, error: String) {
        let now = Utc::now();
        self.duration_ms = Some((now - self.started_at).num_milliseconds() as u64);
        self.finished_at = Some(now);
        self.success = false;
        self.error = Some(error);
    }
}

/// Summary of a scheduler tick cycle.
#[derive(Debug, Clone)]
pub struct TickSummary {
    /// Timestamp of the tick.
    pub timestamp: DateTime<Utc>,
    /// Number of jobs checked.
    pub jobs_checked: usize,
    /// Number of jobs that were due and executed.
    pub jobs_due: usize,
    /// Number of jobs that failed during execution.
    pub jobs_failed: usize,
}

/// Statistics about the scheduler state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchedulerStats {
    pub total_jobs: u64,
    pub active_jobs: u64,
    pub paused_jobs: u64,
    pub disabled_jobs: u64,
    pub failed_jobs: u64,
    pub completed_jobs: u64,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
}
