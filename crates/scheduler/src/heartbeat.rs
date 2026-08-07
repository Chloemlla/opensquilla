//! Heartbeat runner — periodic heartbeat check with SQLite-backed state.
//!
//! The heartbeat runner records a heartbeat every configurable interval
//! (default 30s) into a SQLite `heartbeats` table. A heartbeat is considered
//! *missed* when `now - last_seen` exceeds `interval * timeout_multiplier`
//! (default 3x), which detects a stalled or dead process. When a beat
//! arrives after a missed gap the status transitions through `Recovering` /
//! `Recovered` back to `Healthy`.
//!
//! The runner is modeled on the scheduler's `TickLoop`/`SessionReaper`
//! pattern: the [`HeartbeatStore`] performs synchronous SQLite calls, and the
//! runner serializes access behind a `tokio::sync::Mutex`. `start()` spawns a
//! tokio interval loop; callers that want to drive the cadence themselves can
//! call [`HeartbeatRunner::run_cycle`] directly.
//!
//! This is the Rust port of the Python `scheduler/heartbeat.py` module.

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Default heartbeat interval in seconds.
pub const DEFAULT_HEARTBEAT_INTERVAL_SECS: u64 = 30;
/// Default multiplier applied to the interval to derive the missed threshold.
pub const DEFAULT_TIMEOUT_MULTIPLIER: u32 = 3;

/// Lifecycle status of a heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeartbeatStatus {
    /// No heartbeat record exists yet.
    Unknown,
    /// Beats are arriving within the expected interval.
    Healthy,
    /// A beat was due but `last_seen` is stale (process likely stalled).
    Missed,
    /// A beat arrived after a missed gap but the missed state is being
    /// recovered from.
    Recovering,
    /// A beat arrived after a missed gap and the runner has recovered.
    Recovered,
}

impl HeartbeatStatus {
    /// The canonical string form used for SQLite persistence.
    pub fn as_str(&self) -> &'static str {
        match self {
            HeartbeatStatus::Unknown => "unknown",
            HeartbeatStatus::Healthy => "healthy",
            HeartbeatStatus::Missed => "missed",
            HeartbeatStatus::Recovering => "recovering",
            HeartbeatStatus::Recovered => "recovered",
        }
    }

    /// Parse a status from its string form; unknown strings map to
    /// [`HeartbeatStatus::Unknown`].
    pub fn parse(s: &str) -> Self {
        match s {
            "healthy" => HeartbeatStatus::Healthy,
            "missed" => HeartbeatStatus::Missed,
            "recovering" => HeartbeatStatus::Recovering,
            "recovered" => HeartbeatStatus::Recovered,
            _ => HeartbeatStatus::Unknown,
        }
    }
}

/// A single heartbeat record persisted in SQLite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    /// Unique instance/process identifier this heartbeat belongs to.
    pub id: Uuid,
    /// The last time a beat was recorded.
    pub last_seen: DateTime<Utc>,
    /// The current lifecycle status.
    pub status: HeartbeatStatus,
    /// Schema/runner version for future migrations.
    pub version: u32,
    /// Cumulative number of missed-beat episodes.
    pub missed_count: u32,
    /// The configured beat interval in seconds.
    pub interval_secs: u64,
    /// When the record was last written.
    pub updated_at: DateTime<Utc>,
}

impl Heartbeat {
    /// Create a fresh, healthy heartbeat for the given instance.
    pub fn new(id: Uuid) -> Self {
        let now = Utc::now();
        Self {
            id,
            last_seen: now,
            status: HeartbeatStatus::Healthy,
            version: 1,
            missed_count: 0,
            interval_secs: DEFAULT_HEARTBEAT_INTERVAL_SECS,
            updated_at: now,
        }
    }
}

/// SQLite-backed persistent store for heartbeat records.
pub struct HeartbeatStore {
    conn: std::sync::Mutex<Connection>,
}

/// Errors produced by the heartbeat store.
#[derive(Debug, thiserror::Error)]
pub enum HeartbeatError {
    #[error("Failed to open database: {0}")]
    Open(String),
    #[error("Failed to acquire lock: {0}")]
    Lock(String),
    #[error("Failed to initialize tables: {0}")]
    Initialize(String),
    #[error("Failed to insert record: {0}")]
    Insert(String),
    #[error("Failed to update record: {0}")]
    Update(String),
    #[error("Query failed: {0}")]
    Query(String),
}

const CREATE_HEARTBEATS_TABLE: &str = "
CREATE TABLE IF NOT EXISTS heartbeats (
    id TEXT PRIMARY KEY,
    last_seen TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'healthy',
    version INTEGER NOT NULL DEFAULT 1,
    missed_count INTEGER NOT NULL DEFAULT 0,
    interval_secs INTEGER NOT NULL DEFAULT 30,
    updated_at TEXT NOT NULL
)";

impl HeartbeatStore {
    /// Open (or create) a heartbeat store at the given database path.
    pub fn open(path: &str) -> Result<Self, HeartbeatError> {
        let conn = Connection::open(path).map_err(|e| HeartbeatError::Open(e.to_string()))?;
        let store = Self {
            conn: std::sync::Mutex::new(conn),
        };
        store.initialize_tables()?;
        info!("Heartbeat store opened at {}", path);
        Ok(store)
    }

    /// Create an in-memory heartbeat store (for testing).
    pub fn in_memory() -> Result<Self, HeartbeatError> {
        let conn = Connection::open_in_memory().map_err(|e| HeartbeatError::Open(e.to_string()))?;
        let store = Self {
            conn: std::sync::Mutex::new(conn),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    fn initialize_tables(&self) -> Result<(), HeartbeatError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HeartbeatError::Lock(e.to_string()))?;
        conn.execute_batch(CREATE_HEARTBEATS_TABLE)
            .map_err(|e| HeartbeatError::Initialize(e.to_string()))?;
        Ok(())
    }

    /// Insert or update a heartbeat record (upsert by instance id).
    pub fn upsert(&self, heartbeat: &Heartbeat) -> Result<(), HeartbeatError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HeartbeatError::Lock(e.to_string()))?;
        conn.execute(
            "INSERT INTO heartbeats
                 (id, last_seen, status, version, missed_count, interval_secs, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id) DO UPDATE SET
                 last_seen = excluded.last_seen,
                 status = excluded.status,
                 version = excluded.version,
                 missed_count = excluded.missed_count,
                 interval_secs = excluded.interval_secs,
                 updated_at = excluded.updated_at",
            params![
                heartbeat.id.to_string(),
                heartbeat.last_seen.to_rfc3339(),
                heartbeat.status.as_str(),
                heartbeat.version as i64,
                heartbeat.missed_count as i64,
                heartbeat.interval_secs as i64,
                heartbeat.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HeartbeatError::Update(e.to_string()))?;
        Ok(())
    }

    /// Get a heartbeat record by instance id.
    pub fn get(&self, id: &Uuid) -> Result<Option<Heartbeat>, HeartbeatError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HeartbeatError::Lock(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, last_seen, status, version, missed_count, interval_secs, updated_at
                 FROM heartbeats WHERE id=?1",
            )
            .map_err(|e| HeartbeatError::Query(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![id.to_string()], heartbeat_from_row)
            .map_err(|e| HeartbeatError::Query(e.to_string()))?;

        match rows.next() {
            Some(Ok(hb)) => Ok(Some(hb)),
            Some(Err(e)) => Err(HeartbeatError::Query(e.to_string())),
            None => Ok(None),
        }
    }

    /// List the most recently seen heartbeats.
    pub fn list(&self, limit: u64) -> Result<Vec<Heartbeat>, HeartbeatError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HeartbeatError::Lock(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, last_seen, status, version, missed_count, interval_secs, updated_at
                 FROM heartbeats ORDER BY last_seen DESC LIMIT ?1",
            )
            .map_err(|e| HeartbeatError::Query(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64], heartbeat_from_row)
            .map_err(|e| HeartbeatError::Query(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}

fn heartbeat_from_row(row: &rusqlite::Row) -> rusqlite::Result<Heartbeat> {
    Ok(Heartbeat {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or(Uuid::nil()),
        last_seen: row
            .get::<_, String>(1)?
            .parse::<DateTime<Utc>>()
            .unwrap_or(Utc::now()),
        status: HeartbeatStatus::parse(&row.get::<_, String>(2)?),
        version: row.get::<_, i64>(3)? as u32,
        missed_count: row.get::<_, i64>(4)? as u32,
        interval_secs: row.get::<_, i64>(5)? as u64,
        updated_at: row
            .get::<_, String>(6)?
            .parse::<DateTime<Utc>>()
            .unwrap_or(Utc::now()),
    })
}

/// Snapshot of a heartbeat check, without mutating state.
#[derive(Debug, Clone)]
pub struct HeartbeatCheck {
    /// Whether the heartbeat is currently considered missed.
    pub missed: bool,
    /// The status recorded in the store.
    pub status: HeartbeatStatus,
    /// The last recorded beat time (None if no record exists).
    pub last_seen: Option<DateTime<Utc>>,
    /// Seconds since the last recorded beat.
    pub seconds_since_last_seen: u64,
    /// Cumulative missed-beat episodes.
    pub missed_count: u32,
}

/// Result of one runner cycle (a check followed by a recorded beat).
#[derive(Debug, Clone)]
pub struct HeartbeatCycle {
    /// The pre-record check state.
    pub check: HeartbeatCheck,
    /// The newly recorded heartbeat.
    pub heartbeat: Heartbeat,
}

/// Periodic heartbeat runner.
///
/// Records a beat into the SQLite store every `interval_secs`, detects missed
/// beats (stale `last_seen`), and transitions the status through recovery.
#[derive(Clone)]
pub struct HeartbeatRunner {
    store: Arc<Mutex<HeartbeatStore>>,
    instance_id: Uuid,
    interval_secs: u64,
    timeout_multiplier: u32,
    version: u32,
    running: Arc<AtomicBool>,
}

impl HeartbeatRunner {
    /// Create a runner with the default 30s interval backed by the given store.
    pub fn new(store: HeartbeatStore) -> Self {
        Self {
            store: Arc::new(Mutex::new(store)),
            instance_id: Uuid::new_v4(),
            interval_secs: DEFAULT_HEARTBEAT_INTERVAL_SECS,
            timeout_multiplier: DEFAULT_TIMEOUT_MULTIPLIER,
            version: 1,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the beat interval in seconds (minimum 1).
    pub fn with_interval(mut self, secs: u64) -> Self {
        self.interval_secs = secs.max(1);
        self
    }

    /// Set the instance/process id this runner records under.
    pub fn with_instance_id(mut self, id: Uuid) -> Self {
        self.instance_id = id;
        self
    }

    /// Set the multiplier used to derive the missed threshold
    /// (`interval * multiplier`).
    pub fn with_timeout_multiplier(mut self, multiplier: u32) -> Self {
        self.timeout_multiplier = multiplier.max(1);
        self
    }

    /// Set the schema/runner version stamped onto each beat.
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version.max(1);
        self
    }

    /// The instance id this runner records under.
    pub fn instance_id(&self) -> Uuid {
        self.instance_id
    }

    /// The configured beat interval in seconds.
    pub fn interval_secs(&self) -> u64 {
        self.interval_secs
    }

    /// The staleness threshold (seconds) beyond which a beat is missed.
    pub fn missed_threshold_secs(&self) -> u64 {
        self.interval_secs * self.timeout_multiplier as u64
    }

    /// Record the current heartbeat, detecting missed beats and recovering.
    ///
    /// Returns the persisted record so callers can inspect the resulting
    /// status. This is the core unit of work invoked each tick.
    pub async fn record_heartbeat(&self) -> Result<Heartbeat, HeartbeatError> {
        let now = Utc::now();
        let threshold = self.missed_threshold_secs();
        let store = self.store.lock().await;

        let previous = store.get(&self.instance_id)?;
        let (status, missed_count) = match &previous {
            None => (HeartbeatStatus::Healthy, 0u32),
            Some(prev) => {
                let since = (now - prev.last_seen).num_seconds().max(0) as u64;
                if since > threshold {
                    let missed = prev.missed_count + 1;
                    // A second beat still arriving after a missed gap means we
                    // are actively recovering.
                    let status = if prev.status == HeartbeatStatus::Missed {
                        HeartbeatStatus::Recovering
                    } else {
                        HeartbeatStatus::Missed
                    };
                    (status, missed)
                } else if matches!(
                    prev.status,
                    HeartbeatStatus::Missed
                        | HeartbeatStatus::Recovering
                        | HeartbeatStatus::Recovered
                ) {
                    // The gap has closed: the runner has recovered.
                    (HeartbeatStatus::Recovered, prev.missed_count)
                } else {
                    (HeartbeatStatus::Healthy, prev.missed_count)
                }
            }
        };

        let heartbeat = Heartbeat {
            id: self.instance_id,
            last_seen: now,
            status,
            version: self.version,
            missed_count,
            interval_secs: self.interval_secs,
            updated_at: now,
        };
        store.upsert(&heartbeat)?;

        match status {
            HeartbeatStatus::Missed | HeartbeatStatus::Recovering => {
                warn!(
                    instance = %self.instance_id,
                    missed_count = missed_count,
                    threshold_secs = threshold,
                    "Heartbeat missed; gap exceeded threshold"
                );
            }
            HeartbeatStatus::Recovered => {
                info!(
                    instance = %self.instance_id,
                    missed_count = missed_count,
                    "Heartbeat recovered after missed gap"
                );
            }
            _ => {}
        }

        Ok(heartbeat)
    }

    /// Check whether the heartbeat is currently missed, without recording.
    pub async fn check(&self) -> Result<HeartbeatCheck, HeartbeatError> {
        let now = Utc::now();
        let threshold = self.missed_threshold_secs();
        let store = self.store.lock().await;

        match store.get(&self.instance_id)? {
            Some(hb) => {
                let since = (now - hb.last_seen).num_seconds().max(0) as u64;
                Ok(HeartbeatCheck {
                    missed: since > threshold,
                    status: hb.status,
                    last_seen: Some(hb.last_seen),
                    seconds_since_last_seen: since,
                    missed_count: hb.missed_count,
                })
            }
            None => Ok(HeartbeatCheck {
                missed: false,
                status: HeartbeatStatus::Unknown,
                last_seen: None,
                seconds_since_last_seen: 0,
                missed_count: 0,
            }),
        }
    }

    /// Run a single check + record cycle. Used by tests and the background
    /// loop.
    pub async fn run_cycle(&self) -> Result<HeartbeatCycle, HeartbeatError> {
        let check = self.check().await?;
        let heartbeat = self.record_heartbeat().await?;
        Ok(HeartbeatCycle { check, heartbeat })
    }

    /// Start the periodic heartbeat loop. Returns a handle that can stop it.
    pub fn start(&self) -> HeartbeatHandle {
        self.running.store(true, Ordering::SeqCst);
        let runner = self.clone();
        let running = self.running.clone();
        let handle = HeartbeatHandle {
            running: running.clone(),
        };

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(runner.interval_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(
                instance = %runner.instance_id,
                interval_secs = runner.interval_secs,
                "Heartbeat runner started"
            );

            while running.load(Ordering::SeqCst) {
                ticker.tick().await;
                match runner.run_cycle().await {
                    Ok(cycle) => {
                        debug!(
                            instance = %runner.instance_id,
                            status = cycle.heartbeat.status.as_str(),
                            missed = cycle.check.missed,
                            "Heartbeat cycle completed"
                        );
                    }
                    Err(e) => {
                        warn!(
                            instance = %runner.instance_id,
                            error = %e,
                            "Heartbeat cycle failed"
                        );
                    }
                }
            }

            info!(
                instance = %runner.instance_id,
                "Heartbeat runner stopped"
            );
        });

        handle
    }

    /// Whether the runner loop is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// Handle to control a running [`HeartbeatRunner`] loop.
#[derive(Clone)]
pub struct HeartbeatHandle {
    running: Arc<AtomicBool>,
}

impl HeartbeatHandle {
    /// Request the loop to stop after the current cycle.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Whether the loop is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

/// Builder for [`HeartbeatRunner`] with explicit configuration.
pub struct HeartbeatBuilder {
    store: Option<HeartbeatStore>,
    db_path: Option<String>,
    instance_id: Option<Uuid>,
    interval_secs: u64,
    timeout_multiplier: u32,
    version: u32,
}

impl HeartbeatBuilder {
    /// Create a builder with default configuration (30s interval).
    pub fn new() -> Self {
        Self {
            store: None,
            db_path: None,
            instance_id: None,
            interval_secs: DEFAULT_HEARTBEAT_INTERVAL_SECS,
            timeout_multiplier: DEFAULT_TIMEOUT_MULTIPLIER,
            version: 1,
        }
    }

    /// Use a pre-constructed store.
    pub fn with_store(mut self, store: HeartbeatStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Open (or create) the store at the given database path.
    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    /// Set the instance/process id.
    pub fn with_instance_id(mut self, id: Uuid) -> Self {
        self.instance_id = Some(id);
        self
    }

    /// Set the beat interval in seconds (minimum 1).
    pub fn with_interval(mut self, secs: u64) -> Self {
        self.interval_secs = secs.max(1);
        self
    }

    /// Set the missed-threshold multiplier (minimum 1).
    pub fn with_timeout_multiplier(mut self, multiplier: u32) -> Self {
        self.timeout_multiplier = multiplier.max(1);
        self
    }

    /// Set the schema/runner version.
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version.max(1);
        self
    }

    /// Build the runner.
    pub fn build(self) -> Result<HeartbeatRunner, HeartbeatError> {
        let store = if let Some(store) = self.store {
            store
        } else if let Some(path) = self.db_path {
            HeartbeatStore::open(&path)?
        } else {
            HeartbeatStore::in_memory()?
        };

        let mut runner = HeartbeatRunner::new(store)
            .with_interval(self.interval_secs)
            .with_timeout_multiplier(self.timeout_multiplier)
            .with_version(self.version);
        if let Some(id) = self.instance_id {
            runner = runner.with_instance_id(id);
        }
        Ok(runner)
    }
}

impl Default for HeartbeatBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_heartbeat_is_healthy() {
        let store = HeartbeatStore::in_memory().unwrap();
        let runner = HeartbeatRunner::new(store).with_interval(30);
        let heartbeat = runner.record_heartbeat().await.unwrap();
        assert_eq!(heartbeat.status, HeartbeatStatus::Healthy);
        assert_eq!(heartbeat.missed_count, 0);
        assert_eq!(heartbeat.version, 1);
    }

    #[tokio::test]
    async fn test_missed_detection_and_recovery() {
        let store = HeartbeatStore::in_memory().unwrap();
        let runner = HeartbeatRunner::new(store)
            .with_interval(1)
            .with_timeout_multiplier(2); // missed after 2s

        runner.record_heartbeat().await.unwrap();
        let check = runner.check().await.unwrap();
        assert!(!check.missed);

        // Simulate a stall: backdate the recorded beat beyond the threshold.
        {
            let store = runner.store.lock().await;
            let mut hb = store.get(&runner.instance_id()).unwrap().unwrap();
            hb.last_seen = Utc::now() - chrono::Duration::seconds(10);
            hb.status = HeartbeatStatus::Healthy;
            store.upsert(&hb).unwrap();
        }

        let check = runner.check().await.unwrap();
        assert!(check.missed);
        assert!(check.seconds_since_last_seen >= 10);

        // The next beat records the miss, then a further beat recovers.
        let hb = runner.record_heartbeat().await.unwrap();
        assert_eq!(hb.status, HeartbeatStatus::Missed);
        assert_eq!(hb.missed_count, 1);

        // Now beats arrive on cadence again; the runner recovers.
        runner.record_heartbeat().await.unwrap();
        let hb = runner.record_heartbeat().await.unwrap();
        assert_eq!(hb.status, HeartbeatStatus::Recovered);
    }

    #[tokio::test]
    async fn test_start_stop() {
        let store = HeartbeatStore::in_memory().unwrap();
        let runner = HeartbeatRunner::new(store).with_interval(1);
        let handle = runner.start();
        assert!(handle.is_running());
        tokio::time::sleep(Duration::from_millis(1100)).await;
        let heartbeat = runner
            .store
            .lock()
            .await
            .get(&runner.instance_id())
            .unwrap();
        assert!(heartbeat.is_some());
        handle.stop();
        assert!(!handle.is_running());
    }

    #[tokio::test]
    async fn test_builder_in_memory() {
        let runner = HeartbeatBuilder::new()
            .with_interval(60)
            .with_version(2)
            .build()
            .unwrap();
        assert_eq!(runner.interval_secs(), 60);
        assert_eq!(runner.missed_threshold_secs(), 180);
    }

    #[test]
    fn test_status_roundtrip() {
        for status in [
            HeartbeatStatus::Unknown,
            HeartbeatStatus::Healthy,
            HeartbeatStatus::Missed,
            HeartbeatStatus::Recovering,
            HeartbeatStatus::Recovered,
        ] {
            assert_eq!(HeartbeatStatus::parse(status.as_str()), status);
        }
    }
}
