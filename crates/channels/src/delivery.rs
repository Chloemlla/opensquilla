//! Reliable message delivery store with outbox pattern and fencing tokens.
//!
//! This module provides at-least-once delivery semantics:
//!
//! - **Outbox**: every outbound message is persisted to SQLite before being
//!   dispatched. A crash before dispatch leaves the entry in a claimable
//!   state, so the message is re-sent once the process restarts.
//! - **Fencing token**: a lease row in `delivery_fencing` guards against two
//!   processes delivering the same queue. Tokens expire and can be renewed.
//! - **Leases**: [`DeliveryStore::claim_next`] marks entries `delivering` and
//!   stamps `claimed_at`; stale leases (older than the timeout) are reclaimed.
//! - **Retry**: [`DeliveryStore::mark_failed`] re-queues with exponential
//!   backoff (with jitter) until `max_attempts` is reached, then dead-letters.
//! - **Ledger**: every transition is recorded in `delivery_ledger` for audit.

use chrono::{DateTime, Utc};
use rusqlite::{Connection, Row, params};
use sha2::{Digest, Sha256};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::types::{ChannelType, OutgoingMessage};

/// Default maximum delivery attempts before dead-lettering.
pub const DEFAULT_MAX_ATTEMPTS: u32 = 8;
/// Default base retry delay (doubles per attempt).
pub const DEFAULT_BASE_RETRY_SECS: u64 = 1;
/// Default maximum retry delay.
pub const DEFAULT_MAX_RETRY_SECS: u64 = 3600;
/// Default lease timeout: a `delivering` entry is reclaimed after this long.
pub const DEFAULT_LEASE_SECS: i64 = 120;
/// Default fencing token lifetime.
pub const DEFAULT_FENCING_LIFETIME_SECS: i64 = 3600;

/// Delivery status of an outbox entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryStatus {
    /// Not yet claimed (including scheduled retries).
    Pending,
    /// Claimed by a worker and currently being delivered.
    Delivering,
    /// Successfully delivered.
    Delivered,
    /// Failed permanently after exhausting retries.
    Failed,
}

impl DeliveryStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Delivering => "delivering",
            Self::Delivered => "delivered",
            Self::Failed => "failed",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "delivering" => Self::Delivering,
            "delivered" => Self::Delivered,
            "failed" => Self::Failed,
            _ => Self::Pending,
        }
    }
}

/// A single queued outbound message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutboxEntry {
    pub id: String,
    pub channel_id: String,
    pub channel_type: String,
    /// The serialized outgoing message payload.
    pub payload: serde_json::Value,
    pub status: DeliveryStatus,
    pub created_at: DateTime<Utc>,
    pub delivered_at: Option<DateTime<Utc>>,
    pub attempts: u32,
    pub max_attempts: u32,
    pub next_retry_at: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
    pub dedup_key: Option<String>,
    pub worker_id: Option<String>,
}

impl OutboxEntry {
    /// The message text carried in the payload, when the payload is an
    /// [`OutgoingMessage`] (i.e. has a `text` field).
    pub fn text(&self) -> Option<&str> {
        if let Some(s) = self.payload.as_str() {
            return Some(s);
        }
        self.payload.get("text").and_then(|v| v.as_str())
    }

    /// Deserialize the payload into an [`OutgoingMessage`], if possible.
    pub fn to_outgoing(&self) -> Option<OutgoingMessage> {
        serde_json::from_value(self.payload.clone()).ok()
    }
}

/// A delivery store backed by SQLite with outbox + fencing semantics.
#[derive(Clone)]
pub struct DeliveryStore {
    db: Arc<Mutex<Connection>>,
    fencing_token: String,
    instance_id: String,
    max_attempts: u32,
    lease_secs: i64,
}

impl DeliveryStore {
    /// Open (or create) the delivery store at `path`, acquiring a fencing
    /// token derived from `instance_id`.
    pub fn open(path: impl AsRef<Path>, instance_id: String) -> Result<Self, String> {
        Self::open_with_options(path, instance_id, DEFAULT_MAX_ATTEMPTS, DEFAULT_LEASE_SECS)
    }

    /// Open with custom retry / lease options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        instance_id: String,
        max_attempts: u32,
        lease_secs: i64,
    ) -> Result<Self, String> {
        let db = Connection::open(path).map_err(|e| format!("DB open: {e}"))?;
        let token = hex::encode(Sha256::digest(instance_id.as_bytes()));

        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS delivery_outbox (
                id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                channel_type TEXT NOT NULL DEFAULT 'custom',
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                delivered_at TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 8,
                next_retry_at TEXT,
                last_error TEXT,
                dedup_key TEXT,
                worker_id TEXT,
                claimed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS delivery_fencing (
                token TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS delivery_ledger (
                id TEXT PRIMARY KEY,
                outbox_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                status TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                detail TEXT,
                FOREIGN KEY (outbox_id) REFERENCES delivery_outbox(id)
            );
            CREATE INDEX IF NOT EXISTS idx_outbox_status ON delivery_outbox(status);
            CREATE INDEX IF NOT EXISTS idx_outbox_channel ON delivery_outbox(channel_id);
            CREATE INDEX IF NOT EXISTS idx_outbox_dedup ON delivery_outbox(dedup_key);
            CREATE INDEX IF NOT EXISTS idx_outbox_retry ON delivery_outbox(next_retry_at);
            CREATE INDEX IF NOT EXISTS idx_outbox_claimed ON delivery_outbox(claimed_at);",
        )
        .map_err(|e| format!("Schema init: {e}"))?;

        let now = Utc::now().to_rfc3339();
        let expires =
            (Utc::now() + chrono::Duration::seconds(DEFAULT_FENCING_LIFETIME_SECS)).to_rfc3339();
        db.execute(
            "INSERT OR REPLACE INTO delivery_fencing (token, instance_id, acquired_at, expires_at) VALUES (?1, ?2, ?3, ?4)",
            params![token, instance_id, now, expires],
        )
        .map_err(|e| format!("Fencing token: {e}"))?;

        info!(
            "DeliveryStore opened with fencing token {} (instance {})",
            &token[..8.min(token.len())],
            instance_id
        );

        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            fencing_token: token,
            instance_id,
            max_attempts,
            lease_secs,
        })
    }

    /// The fencing token for this instance.
    pub fn fencing_token(&self) -> &str {
        &self.fencing_token
    }

    /// The instance id this store belongs to.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    // -- fencing ------------------------------------------------------------

    /// Whether this instance's fencing token is present and unexpired.
    pub async fn verify_fencing(&self) -> Result<bool, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare("SELECT expires_at FROM delivery_fencing WHERE token = ?1")
            .map_err(|e| format!("Query: {e}"))?;
        let result: Result<String, _> =
            stmt.query_row(params![self.fencing_token], |row| row.get(0));
        match result {
            Ok(expires) => {
                let expires_at = DateTime::parse_from_rfc3339(&expires)
                    .map_err(|e| format!("Date parse: {e}"))?
                    .with_timezone(&Utc);
                Ok(Utc::now() < expires_at)
            }
            Err(_) => Ok(false),
        }
    }

    /// Try to take the fencing token. Returns `true` if this instance now
    /// holds it (either it already did, or the previous holder's token was
    /// expired). Returns `false` if another live instance holds it.
    pub async fn acquire_fencing(&self) -> Result<bool, String> {
        let db = self.db.lock().await;
        let now = Utc::now();
        let mut stmt = db
            .prepare("SELECT expires_at FROM delivery_fencing ORDER BY acquired_at DESC LIMIT 1")
            .map_err(|e| format!("Query: {e}"))?;
        let existing: Result<String, _> = stmt.query_row([], |row| row.get(0));
        if let Ok(expires) = existing {
            if let Ok(expires_at) = DateTime::parse_from_rfc3339(&expires) {
                if expires_at.with_timezone(&Utc) > now {
                    return Ok(false);
                }
            }
        }
        let expires = (now + chrono::Duration::seconds(DEFAULT_FENCING_LIFETIME_SECS)).to_rfc3339();
        db.execute(
            "INSERT OR REPLACE INTO delivery_fencing (token, instance_id, acquired_at, expires_at) VALUES (?1, ?2, ?3, ?4)",
            params![self.fencing_token, self.instance_id, now.to_rfc3339(), expires],
        )
        .map_err(|e| format!("Acquire fencing: {e}"))?;
        Ok(true)
    }

    /// Renew the fencing token, extending its expiry.
    pub async fn renew_fencing(&self) -> Result<(), String> {
        let expires =
            (Utc::now() + chrono::Duration::seconds(DEFAULT_FENCING_LIFETIME_SECS)).to_rfc3339();
        let db = self.db.lock().await;
        db.execute(
            "UPDATE delivery_fencing SET expires_at = ?1 WHERE token = ?2",
            params![expires, self.fencing_token],
        )
        .map_err(|e| format!("Renew fencing: {e}"))?;
        Ok(())
    }

    /// Release the fencing token (allows another instance to acquire it).
    pub async fn release_fencing(&self) -> Result<(), String> {
        let db = self.db.lock().await;
        db.execute(
            "DELETE FROM delivery_fencing WHERE token = ?1",
            params![self.fencing_token],
        )
        .map_err(|e| format!("Release fencing: {e}"))?;
        Ok(())
    }

    // -- outbox -------------------------------------------------------------

    /// Enqueue a plain-text message (legacy API). Stores the text as a JSON
    /// string payload.
    pub async fn enqueue(&self, channel_id: &str, message: &str) -> Result<String, String> {
        self.enqueue_payload(
            channel_id,
            "custom",
            serde_json::Value::String(message.to_string()),
            None,
        )
        .await
    }

    /// Enqueue an outgoing message for delivery.
    pub async fn enqueue_outgoing(
        &self,
        channel_id: &str,
        message: &OutgoingMessage,
        dedup_key: Option<&str>,
    ) -> Result<String, String> {
        let payload =
            serde_json::to_value(message).map_err(|e| format!("Serialize message: {e}"))?;
        let channel_type = channel_type_str(&message.channel_type);
        self.enqueue_payload(channel_id, &channel_type, payload, dedup_key)
            .await
    }

    /// Enqueue an arbitrary JSON payload.
    pub async fn enqueue_payload(
        &self,
        channel_id: &str,
        channel_type: &str,
        payload: serde_json::Value,
        dedup_key: Option<&str>,
    ) -> Result<String, String> {
        let now = Utc::now().to_rfc3339();
        let db = self.db.lock().await;

        // Dedup: if a non-final entry with the same key exists, return it.
        if let Some(key) = dedup_key {
            let mut stmt = db
                .prepare(
                    "SELECT id FROM delivery_outbox WHERE dedup_key = ?1 AND status IN ('pending','delivering') LIMIT 1",
                )
                .map_err(|e| format!("Dedup query: {e}"))?;
            if let Ok(Some(id)) = stmt
                .query_row(params![key], |row| row.get::<_, String>(0))
                .map(Some)
            {
                return Ok(id);
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let payload_str =
            serde_json::to_string(&payload).map_err(|e| format!("Serialize payload: {e}"))?;
        db.execute(
            "INSERT INTO delivery_outbox (id, channel_id, channel_type, payload, status, created_at, attempts, max_attempts, dedup_key)
             VALUES (?1, ?2, ?3, ?4, 'pending', ?5, 0, ?6, ?7)",
            params![
                id,
                channel_id,
                channel_type,
                payload_str,
                now,
                self.max_attempts,
                dedup_key
            ],
        )
        .map_err(|e| format!("Enqueue: {e}"))?;

        self.log_ledger(
            &db,
            &id,
            channel_id,
            "enqueued",
            "Message enqueued for delivery",
        )?;
        Ok(id)
    }

    // -- claiming -----------------------------------------------------------

    /// Claim up to `batch_size` entries for this worker.
    ///
    /// Claims pending/retryable entries, and reclaims `delivering` entries
    /// whose lease (`claimed_at`) is older than the lease timeout.
    pub async fn claim_next(
        &self,
        worker_id: &str,
        batch_size: usize,
    ) -> Result<Vec<OutboxEntry>, String> {
        let now = Utc::now().to_rfc3339();
        let stale = (Utc::now() - chrono::Duration::seconds(self.lease_secs)).to_rfc3339();
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(
                "SELECT id FROM delivery_outbox
                 WHERE (status = 'pending' AND (next_retry_at IS NULL OR next_retry_at <= ?1))
                    OR (status = 'delivering' AND claimed_at IS NOT NULL AND claimed_at <= ?2)
                 ORDER BY created_at ASC LIMIT ?3",
            )
            .map_err(|e| format!("Claim query: {e}"))?;
        let ids: Vec<String> = stmt
            .query_map(params![now, stale, batch_size as i64], |row| row.get(0))
            .map_err(|e| format!("Claim rows: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Claim collect: {e}"))?;
        drop(stmt);

        let mut entries = Vec::with_capacity(ids.len());
        for id in ids {
            let updated = db
                .execute(
                    "UPDATE delivery_outbox
                     SET status = 'delivering', worker_id = ?1, claimed_at = ?2, next_retry_at = NULL
                     WHERE id = ?3 AND (status = 'pending' OR status = 'delivering')",
                    params![worker_id, now, id],
                )
                .map_err(|e| format!("Claim update: {e}"))?;
            if updated > 0 {
                if let Ok(entry) = Self::load_entry(&db, &id) {
                    self.log_ledger(
                        &db,
                        &id,
                        &entry.channel_id,
                        "claimed",
                        "Entry claimed by worker",
                    )?;
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }

    /// Claim a single entry by id (used for targeted retries).
    pub async fn claim_by_id(
        &self,
        id: &str,
        worker_id: &str,
    ) -> Result<Option<OutboxEntry>, String> {
        let now = Utc::now().to_rfc3339();
        let db = self.db.lock().await;
        let updated = db
            .execute(
                "UPDATE delivery_outbox SET status = 'delivering', worker_id = ?1, claimed_at = ?2
                 WHERE id = ?3 AND status != 'delivered'",
                params![worker_id, now, id],
            )
            .map_err(|e| format!("Claim by id: {e}"))?;
        if updated == 0 {
            return Ok(None);
        }
        let entry = Self::load_entry(&db, id)?;
        self.log_ledger(
            &db,
            id,
            &entry.channel_id,
            "claimed",
            "Entry claimed by worker",
        )?;
        Ok(Some(entry))
    }

    // -- delivery outcomes --------------------------------------------------

    /// Mark an entry as successfully delivered.
    pub async fn mark_delivered(&self, id: &str) -> Result<(), String> {
        let now = Utc::now().to_rfc3339();
        let db = self.db.lock().await;
        let entry = Self::load_entry(&db, id)?;
        db.execute(
            "UPDATE delivery_outbox SET status = 'delivered', delivered_at = ?1, worker_id = NULL, claimed_at = NULL, last_error = NULL WHERE id = ?2",
            params![now, id],
        )
        .map_err(|e| format!("Mark delivered: {e}"))?;
        self.log_ledger(
            &db,
            id,
            &entry.channel_id,
            "delivered",
            "Message delivered successfully",
        )?;
        Ok(())
    }

    /// Mark an entry as failed; re-queues with backoff until `max_attempts`
    /// is exhausted, then dead-letters the entry.
    pub async fn mark_failed(&self, id: &str, error: &str) -> Result<(), String> {
        let db = self.db.lock().await;
        let entry = Self::load_entry(&db, id)?;
        let attempts = entry.attempts.saturating_add(1);
        if attempts >= entry.max_attempts {
            db.execute(
                "UPDATE delivery_outbox SET status = 'failed', attempts = ?1, last_error = ?2, worker_id = NULL, claimed_at = NULL WHERE id = ?3",
                params![attempts, error, id],
            )
            .map_err(|e| format!("Dead-letter: {e}"))?;
            self.log_ledger(&db, id, &entry.channel_id, "failed", error)?;
            warn!("Outbox entry {id} dead-lettered after {attempts} attempts: {error}");
        } else {
            let delay = retry_delay(attempts, DEFAULT_BASE_RETRY_SECS, DEFAULT_MAX_RETRY_SECS);
            let next =
                (Utc::now() + chrono::Duration::from_std(delay).unwrap_or_default()).to_rfc3339();
            db.execute(
                "UPDATE delivery_outbox SET status = 'pending', attempts = ?1, last_error = ?2, next_retry_at = ?3, worker_id = NULL, claimed_at = NULL WHERE id = ?4",
                params![attempts, error, next, id],
            )
            .map_err(|e| format!("Requeue: {e}"))?;
            self.log_ledger(
                &db,
                id,
                &entry.channel_id,
                "retry",
                &format!("{error}; retry in {}s", delay.as_secs()),
            )?;
            debug("requeued with backoff", id, attempts);
        }
        Ok(())
    }

    /// Explicitly dead-letter an entry without further retries.
    pub async fn mark_dead(&self, id: &str, error: &str) -> Result<(), String> {
        let db = self.db.lock().await;
        let entry = Self::load_entry(&db, id)?;
        db.execute(
            "UPDATE delivery_outbox SET status = 'failed', last_error = ?1, worker_id = NULL, claimed_at = NULL WHERE id = ?2",
            params![error, id],
        )
        .map_err(|e| format!("Mark dead: {e}"))?;
        self.log_ledger(&db, id, &entry.channel_id, "failed", error)?;
        Ok(())
    }

    /// Requeue a `delivered` entry for re-sending (idempotency override).
    pub async fn requeue(&self, id: &str, error: &str) -> Result<(), String> {
        let db = self.db.lock().await;
        db.execute(
            "UPDATE delivery_outbox SET status = 'pending', attempts = 0, next_retry_at = NULL, delivered_at = NULL, last_error = ?1 WHERE id = ?2",
            params![error, id],
        )
        .map_err(|e| format!("Requeue: {e}"))?;
        let entry = Self::load_entry(&db, id)?;
        self.log_ledger(&db, id, &entry.channel_id, "requeued", error)?;
        Ok(())
    }

    // -- queries ------------------------------------------------------------

    /// Load a single entry by id.
    pub async fn get(&self, id: &str) -> Result<Option<OutboxEntry>, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(&format!("{} WHERE id = ?1", Self::SELECT_COLS))
            .map_err(|e| format!("Query: {e}"))?;
        let result = stmt
            .query_row(params![id], Self::map_row)
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })
            .map_err(|e| format!("Get entry: {e}"))?;
        Ok(result)
    }

    /// List all entries for a channel.
    pub async fn list_by_channel(&self, channel_id: &str) -> Result<Vec<OutboxEntry>, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(&format!(
                "{} WHERE channel_id = ?1 ORDER BY created_at DESC",
                Self::SELECT_COLS
            ))
            .map_err(|e| format!("Query: {e}"))?;
        stmt.query_map(params![channel_id], Self::map_row)
            .map_err(|e| format!("Query rows: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect rows: {e}"))
    }

    /// List entries by status.
    pub async fn list_by_status(&self, status: DeliveryStatus) -> Result<Vec<OutboxEntry>, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(&format!(
                "{} WHERE status = ?1 ORDER BY created_at DESC",
                Self::SELECT_COLS
            ))
            .map_err(|e| format!("Query: {e}"))?;
        stmt.query_map(params![status.as_str()], Self::map_row)
            .map_err(|e| format!("Query rows: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect rows: {e}"))
    }

    /// Get all entries that are not yet delivered (pending + failed).
    pub async fn get_pending(&self) -> Result<Vec<OutboxEntry>, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(&format!(
                "{} WHERE status IN ('pending', 'failed') ORDER BY created_at ASC",
                Self::SELECT_COLS
            ))
            .map_err(|e| format!("Query: {e}"))?;
        stmt.query_map([], Self::map_row)
            .map_err(|e| format!("Query rows: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect rows: {e}"))
    }

    /// Count of undelivered entries.
    pub async fn pending_count(&self) -> Result<u64, String> {
        let db = self.db.lock().await;
        let count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM delivery_outbox WHERE status != 'delivered'",
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("Count: {e}"))?;
        Ok(count.max(0) as u64)
    }

    /// Per-status counts for observability.
    pub async fn stats(&self) -> Result<serde_json::Value, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare("SELECT status, COUNT(*) FROM delivery_outbox GROUP BY status")
            .map_err(|e| format!("Stats query: {e}"))?;
        let mut counts = serde_json::Map::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(|e| format!("Stats rows: {e}"))?;
        for (status, count) in rows.flatten() {
            counts.insert(status, serde_json::json!(count));
        }
        Ok(serde_json::Value::Object(counts))
    }

    /// The delivery ledger for a given outbox entry.
    pub async fn ledger(&self, outbox_id: &str) -> Result<Vec<serde_json::Value>, String> {
        let db = self.db.lock().await;
        let mut stmt = db
            .prepare(
                "SELECT status, occurred_at, detail FROM delivery_ledger WHERE outbox_id = ?1 ORDER BY occurred_at ASC",
            )
            .map_err(|e| format!("Ledger query: {e}"))?;
        let rows = stmt
            .query_map(params![outbox_id], |row| {
                Ok(serde_json::json!({
                    "status": row.get::<_, String>(0)?,
                    "occurred_at": row.get::<_, String>(1)?,
                    "detail": row.get::<_, Option<String>>(2)?,
                }))
            })
            .map_err(|e| format!("Ledger rows: {e}"))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Ledger collect: {e}"))
    }

    // -- internals ----------------------------------------------------------

    const SELECT_COLS: &'static str = "SELECT id, channel_id, channel_type, payload, status, created_at, delivered_at, \
         attempts, max_attempts, next_retry_at, last_error, dedup_key, worker_id FROM delivery_outbox";

    fn map_row(row: &Row) -> rusqlite::Result<OutboxEntry> {
        let created_at_str: String = row.get(5)?;
        let delivered_at_str: Option<String> = row.get(6)?;
        let next_retry_str: Option<String> = row.get(9)?;
        let payload_str: String = row.get(3)?;
        let payload: serde_json::Value =
            serde_json::from_str(&payload_str).unwrap_or(serde_json::Value::Null);
        Ok(OutboxEntry {
            id: row.get(0)?,
            channel_id: row.get(1)?,
            channel_type: row.get(2)?,
            payload,
            status: DeliveryStatus::from_str(&row.get::<_, String>(4)?),
            created_at: parse_ts(&created_at_str).unwrap_or_else(Utc::now),
            delivered_at: delivered_at_str.as_deref().and_then(parse_ts),
            attempts: row.get(7)?,
            max_attempts: row.get(8)?,
            next_retry_at: next_retry_str.as_deref().and_then(parse_ts),
            last_error: row.get(10)?,
            dedup_key: row.get(11)?,
            worker_id: row.get(12)?,
        })
    }

    fn load_entry(db: &Connection, id: &str) -> Result<OutboxEntry, String> {
        let mut stmt = db
            .prepare(&format!("{} WHERE id = ?1", Self::SELECT_COLS))
            .map_err(|e| format!("Load entry: {e}"))?;
        stmt.query_row(params![id], Self::map_row)
            .map_err(|e| format!("Load entry {id}: {e}"))
    }

    fn log_ledger(
        &self,
        db: &Connection,
        outbox_id: &str,
        channel_id: &str,
        status: &str,
        detail: &str,
    ) -> Result<(), String> {
        db.execute(
            "INSERT INTO delivery_ledger (id, outbox_id, channel_id, status, occurred_at, detail) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                uuid::Uuid::new_v4().to_string(),
                outbox_id,
                channel_id,
                status,
                Utc::now().to_rfc3339(),
                detail
            ],
        )
        .map(|_| ())
        .map_err(|e| format!("Ledger: {e}"))
    }
}

fn parse_ts(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .ok()
}

/// Convert a [`ChannelType`] to its snake_case string form via serde.
fn channel_type_str(ct: &ChannelType) -> String {
    serde_json::to_string(ct)
        .map(|s| s.trim_matches('"').to_string())
        .unwrap_or_else(|_| "custom".to_string())
}

fn debug(msg: &str, id: &str, attempts: u32) {
    warn!("{msg}: id={id} attempts={attempts}");
}

/// Exponential backoff with jitter, capped at `max_secs`.
pub fn retry_delay(attempts: u32, base_secs: u64, max_secs: u64) -> Duration {
    let exp = base_secs.saturating_mul(1u64 << attempts.min(6));
    let secs = exp.min(max_secs.max(1));
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let jitter_ms = nanos % 250;
    Duration::from_secs(secs) + Duration::from_millis(jitter_ms.into())
}

// ---------------------------------------------------------------------------
// Outbox worker
// ---------------------------------------------------------------------------

/// The dispatch function used by an [`OutboxWorker`]: turn a claimed entry
/// into a send future.
pub type SendEntry =
    dyn Fn(&OutboxEntry) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> + Send + Sync;

/// A background task that drains the outbox and dispatches entries through a
/// send function.
///
/// The worker claims entries in batches, awaits `send` for each, and records
/// the outcome in the store. At-least-once semantics follow from claiming
/// before sending and only marking delivered after a successful send.
pub struct OutboxWorker {
    store: Arc<DeliveryStore>,
    send: Arc<SendEntry>,
    worker_id: String,
    poll_interval: Duration,
    max_batch: usize,
    running: Arc<std::sync::Mutex<bool>>,
}

impl OutboxWorker {
    /// Create a worker that dispatches claimed entries via `send`.
    pub fn new(
        store: Arc<DeliveryStore>,
        send: impl Fn(&OutboxEntry) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
        + Send
        + Sync
        + 'static,
        worker_id: impl Into<String>,
    ) -> Self {
        Self::with_options(store, send, worker_id, Duration::from_secs(1), 32)
    }

    /// Create a worker with a custom poll interval and batch size.
    pub fn with_options(
        store: Arc<DeliveryStore>,
        send: impl Fn(&OutboxEntry) -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>>
        + Send
        + Sync
        + 'static,
        worker_id: impl Into<String>,
        poll_interval: Duration,
        max_batch: usize,
    ) -> Self {
        Self {
            store,
            send: Arc::new(send),
            worker_id: worker_id.into(),
            poll_interval,
            max_batch,
            running: Arc::new(std::sync::Mutex::new(false)),
        }
    }

    /// Process one batch of claimed entries (returns the number of entries
    /// handled). Used by the loop and tests.
    pub async fn run_once(&self) -> Result<usize, String> {
        let claimed = self
            .store
            .claim_next(&self.worker_id, self.max_batch)
            .await?;
        let mut handled = 0;
        for entry in &claimed {
            match (self.send)(entry).await {
                Ok(()) => self.store.mark_delivered(&entry.id).await?,
                Err(e) => self.store.mark_failed(&entry.id, &e).await?,
            }
            handled += 1;
        }
        Ok(handled)
    }

    /// Start the worker loop in the background.
    pub fn start(&self) {
        {
            let mut running = self.running.lock().unwrap();
            if *running {
                return;
            }
            *running = true;
        }
        let store = self.store.clone();
        let send = self.send.clone();
        let worker_id = self.worker_id.clone();
        let poll_interval = self.poll_interval;
        let max_batch = self.max_batch;
        let running = self.running.clone();
        tokio::spawn(async move {
            let worker = OutboxWorker {
                store,
                send,
                worker_id,
                poll_interval,
                max_batch,
                running,
            };
            let mut interval = tokio::time::interval(poll_interval);
            interval.tick().await;
            loop {
                if !*worker.running.lock().unwrap() {
                    return;
                }
                if let Err(e) = worker.run_once().await {
                    warn!("Outbox worker error: {e}");
                }
                interval.tick().await;
            }
        });
    }

    /// Stop the worker loop.
    pub async fn stop(&self) {
        *self.running.lock().unwrap() = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChannelType;
    use rusqlite::params;
    use serde_json::json;

    fn temp_store() -> (DeliveryStore, String) {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("delivery_test_{}.sqlite", uuid::Uuid::new_v4()));
        let store = DeliveryStore::open(&path, "test-instance".to_string()).unwrap();
        (store, path.to_string_lossy().to_string())
    }

    fn outgoing(channel_id: &str) -> OutgoingMessage {
        OutgoingMessage::new(
            channel_id.to_string(),
            ChannelType::Slack,
            "hello".to_string(),
        )
    }

    #[test]
    fn test_enqueue_and_get() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id = store.enqueue("c1", "hello").await.unwrap();
            let entry = store.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.channel_id, "c1");
            assert_eq!(entry.payload, json!("hello"));
            assert_eq!(entry.status, DeliveryStatus::Pending);
            assert_eq!(entry.attempts, 0);
        });
    }

    #[test]
    fn test_dedup() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id1 = store
                .enqueue_payload("c1", "slack", json!({"a": 1}), Some("key1"))
                .await
                .unwrap();
            let id2 = store
                .enqueue_payload("c1", "slack", json!({"a": 2}), Some("key1"))
                .await
                .unwrap();
            assert_eq!(id1, id2);
        });
    }

    #[test]
    fn test_claim_and_deliver() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id = store
                .enqueue_outgoing("c1", &outgoing("c1"), None)
                .await
                .unwrap();
            let claimed = store.claim_next("worker1", 10).await.unwrap();
            assert_eq!(claimed.len(), 1);
            assert_eq!(claimed[0].status, DeliveryStatus::Delivering);
            assert_eq!(claimed[0].worker_id.as_deref(), Some("worker1"));
            store.mark_delivered(&id).await.unwrap();
            assert_eq!(store.pending_count().await.unwrap(), 0);
        });
    }

    #[test]
    fn test_failed_requeues_with_backoff() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id = store.enqueue("c1", "hi").await.unwrap();
            let _ = store.claim_next("w", 10).await.unwrap();
            store.mark_failed(&id, "boom").await.unwrap();
            let entry = store.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.status, DeliveryStatus::Pending);
            assert_eq!(entry.attempts, 1);
            assert!(entry.next_retry_at.is_some());
            assert_eq!(entry.last_error.as_deref(), Some("boom"));
        });
    }

    #[test]
    fn test_dead_letter_after_max_attempts() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id = store.enqueue("c1", "hi").await.unwrap();
            // Exhaust attempts.
            for _ in 0..DEFAULT_MAX_ATTEMPTS {
                let _ = store.claim_next("w", 10).await.unwrap();
                store.mark_failed(&id, "boom").await.unwrap();
            }
            let entry = store.get(&id).await.unwrap().unwrap();
            assert_eq!(entry.status, DeliveryStatus::Failed);
            assert_eq!(entry.attempts, DEFAULT_MAX_ATTEMPTS);
        });
    }

    #[test]
    fn test_stale_lease_reclaimed() {
        let (store, path) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let id = store.enqueue("c1", "hi").await.unwrap();
            // Claim, then manually age the claimed_at stamp.
            let _ = store.claim_next("dead-worker", 10).await.unwrap();
            {
                let db = store.db.lock().await;
                let stale = (Utc::now() - chrono::Duration::seconds(9999)).to_rfc3339();
                db.execute(
                    "UPDATE delivery_outbox SET claimed_at = ?1 WHERE id = ?2",
                    params![stale, id],
                )
                .unwrap();
            }
            let claimed = store.claim_next("alive-worker", 10).await.unwrap();
            assert_eq!(claimed.len(), 1);
            assert_eq!(claimed[0].worker_id.as_deref(), Some("alive-worker"));
        });
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_fencing() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            assert!(store.verify_fencing().await.unwrap());
            store.renew_fencing().await.unwrap();
            assert!(store.verify_fencing().await.unwrap());
            store.release_fencing().await.unwrap();
            assert!(!store.verify_fencing().await.unwrap());
        });
    }

    #[test]
    fn test_retry_delay_monotonic() {
        let a = retry_delay(1, 1, 3600).as_secs();
        let b = retry_delay(2, 1, 3600).as_secs();
        let c = retry_delay(3, 1, 3600).as_secs();
        assert!(b >= a);
        assert!(c >= b);
        assert!(retry_delay(100, 1, 3600).as_secs() <= 3600);
    }

    #[test]
    fn test_stats() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            store.enqueue("c1", "a").await.unwrap();
            store.enqueue("c1", "b").await.unwrap();
            let stats = store.stats().await.unwrap();
            assert_eq!(stats["pending"], 2);
        });
    }

    #[test]
    fn test_worker_delivers() {
        let (store, _) = temp_store();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            store.enqueue("c1", "hi").await.unwrap();
            let worker = OutboxWorker::new(
                Arc::new(store.clone()),
                |entry| {
                    let text = entry.text().unwrap_or("").to_string();
                    Box::pin(async move {
                        assert_eq!(text, "hi");
                        Ok(())
                    })
                },
                "worker-1",
            );
            let handled = worker.run_once().await.unwrap();
            assert_eq!(handled, 1);
            assert_eq!(store.pending_count().await.unwrap(), 0);
        });
    }
}
