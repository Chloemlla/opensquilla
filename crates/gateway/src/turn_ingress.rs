//! Inbound turn processing.
//!
//! Mirrors the Python `turn_ingress.py` module. Provides the message ingress
//! pipeline: incoming user messages are validated, deduplicated, queued, and
//! handed to the engine one turn per session at a time (concurrency control).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::session_events::{SessionEvent, SessionEventBroadcaster, SessionEventKind};
use crate::session_services::SessionServices;

/// Maximum length of a single inbound message.
pub const MAX_MESSAGE_BYTES: usize = 256 * 1024;
/// Maximum number of attachments allowed per turn.
pub const MAX_ATTACHMENTS_PER_TURN: usize = 32;
/// Default per-session queue capacity.
pub const DEFAULT_QUEUE_CAPACITY: usize = 64;

/// Status of an inbound turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    /// Received but not yet validated.
    Pending,
    /// Validated and queued for processing.
    Queued,
    /// Currently being processed by the engine.
    Running,
    /// Finished processing.
    Completed,
    /// Failed during validation or processing.
    Failed,
    /// Duplicate of an already-seen turn; ignored.
    Duplicate,
}

impl TurnStatus {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnStatus::Pending => "pending",
            TurnStatus::Queued => "queued",
            TurnStatus::Running => "running",
            TurnStatus::Completed => "completed",
            TurnStatus::Failed => "failed",
            TurnStatus::Duplicate => "duplicate",
        }
    }
}

impl std::fmt::Display for TurnStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An inbound turn enqueued for processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundTurn {
    /// Unique id of this turn (used for deduplication).
    pub turn_id: Uuid,
    /// The session this turn belongs to.
    pub session_id: String,
    /// The raw user message text.
    pub message: String,
    /// Optional attachments referenced by this turn.
    pub attachment_ids: Vec<String>,
    /// When the turn was received.
    pub received_at: DateTime<Utc>,
    /// Current processing status.
    pub status: TurnStatus,
    /// Optional error message if the turn failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl InboundTurn {
    /// Create a new turn. A fresh id is generated unless the caller supplies
    /// a client-generated id for deduplication.
    pub fn new(
        session_id: impl Into<String>,
        message: impl Into<String>,
        id: Option<Uuid>,
    ) -> Self {
        Self {
            turn_id: id.unwrap_or_else(Uuid::new_v4),
            session_id: session_id.into(),
            message: message.into(),
            attachment_ids: Vec::new(),
            received_at: Utc::now(),
            status: TurnStatus::Pending,
            error: None,
        }
    }
}

/// A bounded queue of inbound turns for a single session.
///
/// The sender is used by the ingress to enqueue turns; the receiver is
/// claimed by the session's worker task.
struct SessionQueue {
    tx: mpsc::Sender<InboundTurn>,
    rx: Arc<Mutex<Option<mpsc::Receiver<InboundTurn>>>>,
}

/// The inbound turn pipeline.
///
/// Owns per-session queues plus shared validation and deduplication state.
/// Clone is cheap; all state is shared behind `Arc`s.
#[derive(Clone)]
pub struct TurnIngress {
    services: Arc<SessionServices>,
    broadcaster: SessionEventBroadcaster,
    queues: Arc<Mutex<HashMap<String, SessionQueue>>>,
    seen_turns: Arc<Mutex<HashSet<Uuid>>>,
    active_sessions: Arc<Mutex<HashSet<String>>>,
    queue_capacity: usize,
}

impl TurnIngress {
    /// Create a new ingress pipeline with the given services.
    pub fn new(services: SessionServices, broadcaster: SessionEventBroadcaster) -> Self {
        Self {
            services: Arc::new(services),
            broadcaster,
            queues: Arc::new(Mutex::new(HashMap::new())),
            seen_turns: Arc::new(Mutex::new(HashSet::new())),
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
        }
    }

    /// Create a new ingress pipeline with a custom per-session queue capacity.
    pub fn with_queue_capacity(
        services: SessionServices,
        broadcaster: SessionEventBroadcaster,
        capacity: usize,
    ) -> Self {
        Self {
            services: Arc::new(services),
            broadcaster,
            queues: Arc::new(Mutex::new(HashMap::new())),
            seen_turns: Arc::new(Mutex::new(HashSet::new())),
            active_sessions: Arc::new(Mutex::new(HashSet::new())),
            queue_capacity: capacity,
        }
    }

    /// Validate an inbound message before enqueuing it.
    ///
    /// Checks the message is non-empty and within size limits.
    fn validate(&self, turn: &InboundTurn) -> Result<(), AppError> {
        if turn.message.trim().is_empty() {
            return Err(AppError::bad_request("Message must not be empty"));
        }
        if turn.message.len() > MAX_MESSAGE_BYTES {
            return Err(AppError::bad_request(format!(
                "Message exceeds maximum length of {MAX_MESSAGE_BYTES} bytes"
            )));
        }
        if turn.attachment_ids.len() > MAX_ATTACHMENTS_PER_TURN {
            return Err(AppError::bad_request(format!(
                "Too many attachments: max {MAX_ATTACHMENTS_PER_TURN}"
            )));
        }
        Ok(())
    }

    /// Deduplicate: return `true` if the turn has already been seen.
    ///
    /// Idempotent retries send the same `turn_id`, so a retry that reaches the
    /// ingress after the first turn was recorded is dropped as a duplicate.
    fn deduplicate(&self, turn_id: Uuid) -> bool {
        let mut seen = self.seen_turns.lock();
        if seen.contains(&turn_id) {
            return true;
        }
        seen.insert(turn_id);
        false
    }

    /// Get or create the queue for a session.
    fn queue_for(&self, session_id: &str) -> mpsc::Sender<InboundTurn> {
        let mut queues = self.queues.lock();
        let entry = queues.entry(session_id.to_string()).or_insert_with(|| {
            let (tx, rx) = mpsc::channel(self.queue_capacity);
            SessionQueue {
                tx,
                rx: Arc::new(Mutex::new(Some(rx))),
            }
        });
        entry.tx.clone()
    }

    /// Enqueue an inbound turn for processing.
    ///
    /// Runs validation and deduplication, then forwards the turn to the
    /// session's worker queue. Returns the recorded turn (or a duplicate
    /// marker).
    pub async fn enqueue(&self, mut turn: InboundTurn) -> Result<InboundTurn, AppError> {
        turn.status = TurnStatus::Pending;

        // Deduplicate first: a repeated turn is dropped silently.
        if self.deduplicate(turn.turn_id) {
            turn.status = TurnStatus::Duplicate;
            return Ok(turn);
        }

        // Validate.
        if let Err(e) = self.validate(&turn) {
            turn.status = TurnStatus::Failed;
            turn.error = Some(e.to_string());
            return Err(e);
        }

        // Route to the per-session queue.
        let tx = self.queue_for(&turn.session_id);
        turn.status = TurnStatus::Queued;
        tx.send(turn.clone())
            .await
            .map_err(|_| AppError::internal("Session turn queue is closed"))?;

        self.broadcaster.publish(
            SessionEvent::new(SessionEventKind::TurnStarted, turn.session_id.clone())
                .with_payload(serde_json::json!({"turn_id": turn.turn_id})),
        );
        Ok(turn)
    }

    /// Claim the receiver half of a session's queue, marking the session
    /// active. Returns `None` if the receiver was already claimed.
    fn claim_receiver(&self, session_id: &str) -> Option<mpsc::Receiver<InboundTurn>> {
        let mut queues = self.queues.lock();
        let mut entry = queues.remove(session_id)?;
        let mut rx_guard = entry.rx.lock();
        let rx = rx_guard.take()?;
        self.active_sessions.lock().insert(session_id.to_string());
        Some(rx)
    }

    /// Spawn a worker task that drains a session's queue one turn at a time.
    ///
    /// This enforces one-turn-per-session concurrency: each session has at
    /// most one worker, and the worker processes turns sequentially.
    pub async fn spawn_worker<F, Fut>(&self, session_id: &str, handler: F) -> Result<(), AppError>
    where
        F: FnMut(InboundTurn, Arc<SessionServices>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<(), AppError>> + Send + 'static,
    {
        if self.is_session_active(session_id) {
            return Err(AppError::bad_request(format!(
                "Session '{session_id}' already has an active worker"
            )));
        }
        let rx = self
            .claim_receiver(session_id)
            .ok_or_else(|| AppError::not_found(format!("No queue for session '{session_id}'")))?;
        let services = self.services.clone();
        let broadcaster = self.broadcaster.clone();
        let ingress = self.clone();
        let session_id = session_id.to_string();
        tokio::spawn(async move {
            let mut rx = rx;
            let mut handler = handler;
            while let Some(mut turn) = rx.recv().await {
                turn.status = TurnStatus::Running;
                let result = handler(turn.clone(), services.clone()).await;
                match result {
                    Ok(()) => {
                        broadcaster.publish(
                            SessionEvent::new(
                                SessionEventKind::TurnCompleted,
                                turn.session_id.clone(),
                            )
                            .with_turn(turn.turn_id.to_string()),
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            turn_id = %turn.turn_id,
                            "turn failed"
                        );
                    }
                }
            }
            ingress.active_sessions.lock().remove(&session_id);
        });
        Ok(())
    }

    /// Return `true` if the session currently has an active worker.
    pub fn is_session_active(&self, session_id: &str) -> bool {
        self.active_sessions.lock().contains(session_id)
    }

    /// Return the number of distinct sessions currently queued.
    pub fn session_count(&self) -> usize {
        self.queues.lock().len()
    }

    /// Return the number of distinct turn ids seen so far (for diagnostics).
    pub fn seen_turn_count(&self) -> usize {
        self.seen_turns.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ingress() -> TurnIngress {
        let services = SessionServices::builder().build();
        let broadcaster = SessionEventBroadcaster::new();
        TurnIngress::new(services, broadcaster)
    }

    #[test]
    fn test_validate_rejects_empty() {
        let ingress = ingress();
        let turn = InboundTurn::new("s1", "   ", None);
        assert!(ingress.validate(&turn).is_err());
    }

    #[test]
    fn test_validate_rejects_too_long() {
        let ingress = ingress();
        let long = "x".repeat(MAX_MESSAGE_BYTES + 1);
        let turn = InboundTurn::new("s1", long, None);
        assert!(ingress.validate(&turn).is_err());
    }

    #[test]
    fn test_deduplication() {
        let ingress = ingress();
        let id = Uuid::new_v4();
        assert!(!ingress.deduplicate(id));
        assert!(ingress.deduplicate(id));
        assert_eq!(ingress.seen_turn_count(), 1);
    }

    #[tokio::test]
    async fn test_enqueue_sets_status() {
        let ingress = ingress();
        let turn = InboundTurn::new("s1", "hello", None);
        let recorded = ingress.enqueue(turn).await.unwrap();
        assert_eq!(recorded.status, TurnStatus::Queued);
        assert_eq!(ingress.session_count(), 1);
    }

    #[tokio::test]
    async fn test_enqueue_duplicate_turn() {
        let ingress = ingress();
        let id = Uuid::new_v4();
        let turn1 = InboundTurn::new("s1", "hello", Some(id));
        ingress.enqueue(turn1).await.unwrap();
        let turn2 = InboundTurn::new("s1", "hello", Some(id));
        let recorded = ingress.enqueue(turn2).await.unwrap();
        assert_eq!(recorded.status, TurnStatus::Duplicate);
    }

    #[tokio::test]
    async fn test_worker_processes_turns_sequentially() {
        let ingress = ingress();
        let turn = InboundTurn::new("s1", "first", None);
        ingress.enqueue(turn).await.unwrap();
        let turn = InboundTurn::new("s1", "second", None);
        ingress.enqueue(turn).await.unwrap();

        let processed = Arc::new(Mutex::new(Vec::<String>::new()));
        let processed2 = processed.clone();
        ingress
            .spawn_worker("s1", move |turn, _services| {
                let processed = processed2.clone();
                async move {
                    // Simulate work; ensure sequential ordering.
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    processed.lock().push(turn.message.clone());
                    Ok(())
                }
            })
            .await
            .unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let done = processed.lock().clone();
        assert_eq!(done, vec!["first".to_string(), "second".to_string()]);
        assert!(!ingress.is_session_active("s1"));
    }

    #[tokio::test]
    async fn test_spawn_worker_duplicate_rejected() {
        let ingress = ingress();
        let turn = InboundTurn::new("s1", "hello", None);
        ingress.enqueue(turn).await.unwrap();
        ingress
            .spawn_worker("s1", |_turn, _services| async move { Ok(()) })
            .await
            .unwrap();
        let second = ingress
            .spawn_worker("s1", |_turn, _services| async move { Ok(()) })
            .await;
        assert!(second.is_err());
    }

    #[tokio::test]
    async fn test_spawn_worker_no_queue() {
        let ingress = ingress();
        let result = ingress
            .spawn_worker("missing", |_turn, _services| async move { Ok(()) })
            .await;
        assert!(result.is_err());
    }
}
