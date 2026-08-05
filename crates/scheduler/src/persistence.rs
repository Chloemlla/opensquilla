use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use crate::types::{CronJob, JobExecution, JobStatus, ScheduleKind, SchedulerStats};

/// SQLite-backed persistent store for scheduled jobs and execution records.
pub struct JobStore {
    conn: Mutex<Connection>,
}

impl JobStore {
    /// Open or create a job store at the given database path.
    pub fn open(path: &str) -> Result<Self, StoreError> {
        let conn = Connection::open(path).map_err(|e| StoreError::Open(e.to_string()))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.initialize_tables()?;
        info!("Job store opened at {}", path);
        Ok(store)
    }

    /// Create an in-memory job store (for testing).
    pub fn in_memory() -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory().map_err(|e| StoreError::Open(e.to_string()))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    fn initialize_tables(&self) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                schedule_value TEXT NOT NULL,
                handler TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT 'null',
                status TEXT NOT NULL DEFAULT 'Active',
                max_retries INTEGER NOT NULL DEFAULT 3,
                retry_delay_secs INTEGER NOT NULL DEFAULT 10,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                next_run_at TEXT,
                last_run_at TEXT,
                agent_id TEXT,
                session_id TEXT
            );

            CREATE TABLE IF NOT EXISTS job_executions (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                success INTEGER NOT NULL DEFAULT 0,
                result TEXT,
                error TEXT,
                attempt INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER,
                FOREIGN KEY (job_id) REFERENCES cron_jobs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_job_executions_job_id ON job_executions(job_id);
            CREATE INDEX IF NOT EXISTS idx_job_executions_started_at ON job_executions(started_at);
            CREATE INDEX IF NOT EXISTS idx_cron_jobs_status ON cron_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_cron_jobs_next_run ON cron_jobs(next_run_at);
            CREATE INDEX IF NOT EXISTS idx_cron_jobs_handler ON cron_jobs(handler);"
        ).map_err(|e| StoreError::Initialize(e.to_string()))?;

        Ok(())
    }

    // --- Job CRUD ---

    /// Insert a new cron job.
    pub fn insert_job(&self, job: &CronJob) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let (kind_str, schedule_value) = serialize_schedule_kind(&job.kind);

        conn.execute(
            "INSERT INTO cron_jobs (id, name, kind, schedule_value, handler, payload, status,
             max_retries, retry_delay_secs, created_at, updated_at,
             next_run_at, last_run_at, agent_id, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                job.id.to_string(), job.name, kind_str, schedule_value, job.handler,
                job.payload.to_string(), serde_json::to_string(&job.status).unwrap_or_default(),
                job.max_retries as i64, job.retry_delay_secs as i64,
                job.created_at.to_rfc3339(), job.updated_at.to_rfc3339(),
                job.next_run_at.map(|dt| dt.to_rfc3339()),
                job.last_run_at.map(|dt| dt.to_rfc3339()),
                job.agent_id.map(|id| id.to_string()),
                job.session_id.map(|id| id.to_string())
            ],
        ).map_err(|e| StoreError::Insert(e.to_string()))?;

        Ok(())
    }

    /// Get a job by its ID.
    pub fn get_job(&self, job_id: &Uuid) -> Result<Option<CronJob>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, name, kind, schedule_value, handler, payload, status,
             max_retries, retry_delay_secs, created_at, updated_at,
             next_run_at, last_run_at, agent_id, session_id
             FROM cron_jobs WHERE id = ?1"
        ).map_err(|e| StoreError::Query(e.to_string()))?;

        let mut rows = stmt.query_map(params![job_id.to_string()], |row| job_from_row(row))
            .map_err(|e| StoreError::Query(e.to_string()))?;

        match rows.next() {
            Some(Ok(job)) => Ok(Some(job)),
            Some(Err(e)) => Err(StoreError::Deserialize(e.to_string())),
            None => Ok(None),
        }
    }

    /// Update an existing cron job.
    pub fn update_job(&self, job: &CronJob) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let (kind_str, schedule_value) = serialize_schedule_kind(&job.kind);

        conn.execute(
            "UPDATE cron_jobs SET name=?1, kind=?2, schedule_value=?3, handler=?4,
             payload=?5, status=?6, max_retries=?7, retry_delay_secs=?8,
             updated_at=?9, next_run_at=?10, last_run_at=?11,
             agent_id=?12, session_id=?13 WHERE id=?14",
            params![
                job.name, kind_str, schedule_value, job.handler,
                job.payload.to_string(), serde_json::to_string(&job.status).unwrap_or_default(),
                job.max_retries as i64, job.retry_delay_secs as i64,
                job.updated_at.to_rfc3339(),
                job.next_run_at.map(|dt| dt.to_rfc3339()),
                job.last_run_at.map(|dt| dt.to_rfc3339()),
                job.agent_id.map(|id| id.to_string()),
                job.session_id.map(|id| id.to_string()),
                job.id.to_string()
            ],
        ).map_err(|e| StoreError::Update(e.to_string()))?;

        Ok(())
    }

    /// Delete a job by its ID.
    pub fn delete_job(&self, job_id: &Uuid) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        conn.execute("DELETE FROM cron_jobs WHERE id=?1", params![job_id.to_string()])
            .map_err(|e| StoreError::Delete(e.to_string()))?;
        conn.execute("DELETE FROM job_executions WHERE job_id=?1", params![job_id.to_string()])
            .map_err(|e| StoreError::Delete(e.to_string()))?;
        Ok(())
    }

    /// List all jobs, optionally filtered by status.
    pub fn list_jobs(&self, status_filter: Option<JobStatus>) -> Result<Vec<CronJob>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;

        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(status) = status_filter {
                let status_str = serde_json::to_string(&status).unwrap_or_default();
                ("SELECT id, name, kind, schedule_value, handler, payload, status,
                  max_retries, retry_delay_secs, created_at, updated_at,
                  next_run_at, last_run_at, agent_id, session_id
                  FROM cron_jobs WHERE status=?1 ORDER BY created_at DESC".to_string(),
                 vec![Box::new(status_str)])
            } else {
                ("SELECT id, name, kind, schedule_value, handler, payload, status,
                  max_retries, retry_delay_secs, created_at, updated_at,
                  next_run_at, last_run_at, agent_id, session_id
                  FROM cron_jobs ORDER BY created_at DESC".to_string(),
                 Vec::new())
            };

        let mut stmt = conn.prepare(&sql).map_err(|e| StoreError::Query(e.to_string()))?;
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|b| b.as_ref()).collect();
        let rows = stmt.query_map(params_ref.as_slice(), |row| job_from_row(row))
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let jobs: Vec<CronJob> = rows.filter_map(|r| r.ok()).collect();
        Ok(jobs)
    }

    /// List jobs that are due for execution.
    pub fn list_due_jobs(&self) -> Result<Vec<CronJob>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let active_str = serde_json::to_string(&JobStatus::Active).unwrap_or_default();

        let mut stmt = conn.prepare(
            "SELECT id, name, kind, schedule_value, handler, payload, status,
             max_retries, retry_delay_secs, created_at, updated_at,
             next_run_at, last_run_at, agent_id, session_id
             FROM cron_jobs WHERE status=?1 AND next_run_at IS NOT NULL AND next_run_at<=?2
             ORDER BY next_run_at ASC"
        ).map_err(|e| StoreError::Query(e.to_string()))?;

        let rows = stmt.query_map(params![active_str, now], |row| job_from_row(row))
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let jobs: Vec<CronJob> = rows.filter_map(|r| r.ok()).collect();
        Ok(jobs)
    }

    /// Mark a job's next_run_at and last_run_at.
    pub fn mark_job_run(&self, job_id: &Uuid, next_run: Option<DateTime<Utc>>) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE cron_jobs SET last_run_at=?1, next_run_at=?2, updated_at=?3 WHERE id=?4",
            params![now, next_run.map(|dt| dt.to_rfc3339()), now, job_id.to_string()],
        ).map_err(|e| StoreError::Update(e.to_string()))?;
        Ok(())
    }

    /// Update a job's status.
    pub fn update_job_status(&self, job_id: &Uuid, status: JobStatus) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let status_str = serde_json::to_string(&status).unwrap_or_default();
        conn.execute(
            "UPDATE cron_jobs SET status=?1, updated_at=?2 WHERE id=?3",
            params![status_str, Utc::now().to_rfc3339(), job_id.to_string()],
        ).map_err(|e| StoreError::Update(e.to_string()))?;
        Ok(())
    }

    // --- Execution Records ---

    /// Insert an execution record.
    pub fn insert_execution(&self, execution: &JobExecution) -> Result<(), StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        conn.execute(
            "INSERT INTO job_executions (id, job_id, started_at, finished_at, success, result, error, attempt, duration_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                execution.id.to_string(), execution.job_id.to_string(),
                execution.started_at.to_rfc3339(),
                execution.finished_at.map(|dt| dt.to_rfc3339()),
                execution.success as i64, execution.result, execution.error,
                execution.attempt as i64, execution.duration_ms.map(|d| d as i64)
            ],
        ).map_err(|e| StoreError::Insert(e.to_string()))?;
        Ok(())
    }

    /// List executions for a specific job.
    pub fn list_executions(&self, job_id: &Uuid, limit: u64, offset: u64) -> Result<Vec<JobExecution>, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, job_id, started_at, finished_at, success, result, error, attempt, duration_ms
             FROM job_executions WHERE job_id=?1 ORDER BY started_at DESC LIMIT ?2 OFFSET ?3"
        ).map_err(|e| StoreError::Query(e.to_string()))?;

        let rows = stmt.query_map(params![job_id.to_string(), limit as i64, offset as i64], |row| execution_from_row(row))
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let executions: Vec<JobExecution> = rows.filter_map(|r| r.ok()).collect();
        Ok(executions)
    }

    /// Clean up old execution records, keeping only the most recent N per job.
    pub fn cleanup_executions(&self, keep_per_job: u64) -> Result<u64, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let deleted = conn.execute(
            "DELETE FROM job_executions WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY started_at DESC) AS rn
                    FROM job_executions
                ) WHERE rn > ?1
            )", params![keep_per_job as i64],
        ).map_err(|e| StoreError::Delete(e.to_string()))?;
        if deleted > 0 { info!("Cleaned up {} old execution records", deleted); }
        Ok(deleted as u64)
    }

    /// Clean up old completed jobs (for AT jobs that have completed).
    pub fn cleanup_completed_jobs(&self, older_than_days: u64) -> Result<u64, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let cutoff = (Utc::now() - chrono::Duration::days(older_than_days as i64)).to_rfc3339();
        let completed_str = serde_json::to_string(&JobStatus::Completed).unwrap_or_default();
        let deleted = conn.execute(
            "DELETE FROM cron_jobs WHERE status=?1 AND updated_at<?2",
            params![completed_str, cutoff],
        ).map_err(|e| StoreError::Delete(e.to_string()))?;
        if deleted > 0 { info!("Cleaned up {} completed jobs", deleted); }
        Ok(deleted as u64)
    }

    // --- Stats ---

    /// Compute scheduler statistics.
    pub fn stats(&self) -> Result<SchedulerStats, StoreError> {
        let conn = self.conn.lock().map_err(|e| StoreError::Lock(e.to_string()))?;
        let total_jobs: i64 = conn.query_row("SELECT COUNT(*) FROM cron_jobs", [], |row| row.get(0))
            .map_err(|e| StoreError::Query(e.to_string()))?;

        let count_status = |s: &str| -> Result<i64, StoreError> {
            conn.query_row("SELECT COUNT(*) FROM cron_jobs WHERE status=?1", params![s], |row| row.get(0))
                .map_err(|e| StoreError::Query(e.to_string()))
        };

        let active_s = serde_json::to_string(&JobStatus::Active).unwrap_or_default();
        let paused_s = serde_json::to_string(&JobStatus::Paused).unwrap_or_default();
        let disabled_s = serde_json::to_string(&JobStatus::Disabled).unwrap_or_default();
        let failed_s = serde_json::to_string(&JobStatus::Failed).unwrap_or_default();
        let completed_s = serde_json::to_string(&JobStatus::Completed).unwrap_or_default();

        let total_executions: i64 = conn.query_row("SELECT COUNT(*) FROM job_executions", [], |row| row.get(0))
            .map_err(|e| StoreError::Query(e.to_string()))?;
        let successful_executions: i64 = conn.query_row(
            "SELECT COUNT(*) FROM job_executions WHERE success=1", [], |row| row.get(0)
        ).map_err(|e| StoreError::Query(e.to_string()))?;

        Ok(SchedulerStats {
            total_jobs: total_jobs as u64,
            active_jobs: count_status(&active_s)? as u64,
            paused_jobs: count_status(&paused_s)? as u64,
            disabled_jobs: count_status(&disabled_s)? as u64,
            failed_jobs: count_status(&failed_s)? as u64,
            completed_jobs: count_status(&completed_s)? as u64,
            total_executions: total_executions as u64,
            successful_executions: successful_executions as u64,
            failed_executions: (total_executions - successful_executions) as u64,
        })
    }
}

// --- Helper functions ---

fn serialize_schedule_kind(kind: &ScheduleKind) -> (String, String) {
    match kind {
        ScheduleKind::Cron(expr) => ("cron".to_string(), expr.clone()),
        ScheduleKind::At(dt) => ("at".to_string(), dt.to_rfc3339()),
        ScheduleKind::Every(secs) => ("every".to_string(), secs.to_string()),
    }
}

fn deserialize_schedule_kind(kind_str: &str, value: &str) -> Result<ScheduleKind, StoreError> {
    match kind_str {
        "cron" => Ok(ScheduleKind::Cron(value.to_string())),
        "at" => {
            let dt = DateTime::parse_from_rfc3339(value)
                .map_err(|e| StoreError::Deserialize(format!("Invalid datetime: {}", e)))?
                .with_timezone(&Utc);
            Ok(ScheduleKind::At(dt))
        }
        "every" => {
            let secs: u64 = value.parse()
                .map_err(|e| StoreError::Deserialize(format!("Invalid interval: {}", e)))?;
            Ok(ScheduleKind::Every(secs))
        }
        _ => Err(StoreError::Deserialize(format!("Unknown schedule kind: {}", kind_str))),
    }
}

fn job_from_row(row: &rusqlite::Row) -> rusqlite::Result<CronJob> {
    let kind_str: String = row.get(2)?;
    let schedule_value: String = row.get(3)?;
    let kind = deserialize_schedule_kind(&kind_str, &schedule_value)
        .unwrap_or(ScheduleKind::Cron("0 * * * *".to_string()));
    let status_str: String = row.get(6)?;
    let status: JobStatus = serde_json::from_str(&status_str).unwrap_or(JobStatus::Active);

    Ok(CronJob {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
        name: row.get(1)?,
        kind,
        handler: row.get(4)?,
        payload: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or(serde_json::Value::Null),
        status,
        max_retries: row.get::<_, i64>(7)? as u32,
        retry_delay_secs: row.get::<_, i64>(8)? as u64,
        created_at: row.get::<_, String>(9)?.parse::<DateTime<Utc>>().unwrap_or(Utc::now()),
        updated_at: row.get::<_, String>(10)?.parse::<DateTime<Utc>>().unwrap_or(Utc::now()),
        next_run_at: row.get::<_, Option<String>>(11)?.and_then(|s| s.parse::<DateTime<Utc>>().ok()),
        last_run_at: row.get::<_, Option<String>>(12)?.and_then(|s| s.parse::<DateTime<Utc>>().ok()),
        tags: HashMap::new(),
        agent_id: row.get::<_, Option<String>>(13)?.and_then(|s| Uuid::parse_str(&s).ok()),
        session_id: row.get::<_, Option<String>>(14)?.and_then(|s| Uuid::parse_str(&s).ok()),
    })
}

fn execution_from_row(row: &rusqlite::Row) -> rusqlite::Result<JobExecution> {
    Ok(JobExecution {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
        job_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or(Uuid::nil()),
        started_at: row.get::<_, String>(2)?.parse::<DateTime<Utc>>().unwrap_or(Utc::now()),
        finished_at: row.get::<_, Option<String>>(3)?.and_then(|s| s.parse::<DateTime<Utc>>().ok()),
        success: row.get::<_, i64>(4)? != 0,
        result: row.get(5)?,
        error: row.get(6)?,
        attempt: row.get::<_, i64>(7)? as u32,
        duration_ms: row.get::<_, Option<i64>>(8)?.map(|d| d as u64),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Failed to open database: {0}")]  Open(String),
    #[error("Failed to acquire lock: {0}")]   Lock(String),
    #[error("Failed to initialize tables: {0}")] Initialize(String),
    #[error("Failed to insert record: {0}")]  Insert(String),
    #[error("Failed to update record: {0}")]  Update(String),
    #[error("Failed to delete record: {0}")]  Delete(String),
    #[error("Query failed: {0}")]             Query(String),
    #[error("Failed to deserialize: {0}")]    Deserialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_job() {
        let store = JobStore::in_memory().unwrap();
        let job = CronJob::new("test_job", ScheduleKind::Every(60), "test_handler");
        store.insert_job(&job).unwrap();
        let retrieved = store.get_job(&job.id).unwrap().unwrap();
        assert_eq!(retrieved.id, job.id);
        assert_eq!(retrieved.name, "test_job");
    }

    #[test]
    fn test_list_jobs() {
        let store = JobStore::in_memory().unwrap();
        store.insert_job(&CronJob::new("j1", ScheduleKind::Every(60), "h1")).unwrap();
        store.insert_job(&CronJob::new("j2", ScheduleKind::Every(120), "h2")).unwrap();
        assert_eq!(store.list_jobs(None).unwrap().len(), 2);
    }

    #[test]
    fn test_delete_job() {
        let store = JobStore::in_memory().unwrap();
        let job = CronJob::new("to_delete", ScheduleKind::Every(60), "h");
        store.insert_job(&job).unwrap();
        store.delete_job(&job.id).unwrap();
        assert!(store.get_job(&job.id).unwrap().is_none());
    }

    #[test]
    fn test_insert_execution() {
        let store = JobStore::in_memory().unwrap();
        let job = CronJob::new("exec_test", ScheduleKind::Every(60), "h");
        store.insert_job(&job).unwrap();
        let mut exec = JobExecution::new(job.id, 0);
        exec.complete("ok".to_string());
        store.insert_execution(&exec).unwrap();
        assert_eq!(store.list_executions(&job.id, 10, 0).unwrap().len(), 1);
    }

    #[test]
    fn test_stats() {
        let store = JobStore::in_memory().unwrap();
        store.insert_job(&CronJob::new("s", ScheduleKind::Every(60), "h")).unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(stats.total_jobs, 1);
        assert_eq!(stats.active_jobs, 1);
    }
}
