//! Session event broadcasting.
//!
//! Mirrors the Python `session_events.py` module. Provides a
//! [`tokio::sync::broadcast`] channel for session lifecycle events so that
//! multiple subscribers (WebSocket clients, the memory subsystem, audit
//! logging) can observe changes to sessions in real time.
//!
//! Producers call [`SessionEventBroadcaster::publish`]; consumers call
//! [`SessionEventBroadcaster::subscribe`] to obtain a receiver. Late
//! subscribers only see events published after they subscribe — the channel
//! does not replay history.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, warn};

/// Default channel capacity for the broadcast bus.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

/// The kind of session lifecycle event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventKind {
    /// A session was created.
    Created,
    /// A session's metadata was updated.
    Updated,
    /// A session was paused.
    Paused,
    /// A session was resumed from the paused state.
    Resumed,
    /// A session was archived.
    Archived,
    /// A session was deleted.
    Deleted,
    /// A message was appended to a session's transcript.
    MessageAdded,
    /// A turn started processing.
    TurnStarted,
    /// A turn finished processing.
    TurnCompleted,
}

impl SessionEventKind {
    /// Stable string identifier used in log/JSON output.
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionEventKind::Created => "created",
            SessionEventKind::Updated => "updated",
            SessionEventKind::Paused => "paused",
            SessionEventKind::Resumed => "resumed",
            SessionEventKind::Archived => "archived",
            SessionEventKind::Deleted => "deleted",
            SessionEventKind::MessageAdded => "message_added",
            SessionEventKind::TurnStarted => "turn_started",
            SessionEventKind::TurnCompleted => "turn_completed",
        }
    }
}

impl std::fmt::Display for SessionEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A session lifecycle event emitted on the broadcast bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Monotonic sequence number assigned by the broadcaster.
    pub seq: u64,
    /// The event kind.
    pub kind: SessionEventKind,
    /// The session the event pertains to.
    pub session_id: String,
    /// Optional turn id for turn-scoped events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    /// Optional event payload (e.g. updated fields, message snippet).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// When the event was emitted.
    pub timestamp: DateTime<Utc>,
}

impl SessionEvent {
    /// Create a new event with the given kind and session id.
    pub fn new(kind: SessionEventKind, session_id: impl Into<String>) -> Self {
        Self {
            seq: 0,
            kind,
            session_id: session_id.into(),
            turn_id: None,
            payload: None,
            timestamp: Utc::now(),
        }
    }

    /// Attach a turn id.
    pub fn with_turn(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    /// Attach a payload.
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }
}

/// A broadcaster for session lifecycle events.
///
/// Wraps a `tokio::sync::broadcast` channel. Cloning is cheap (the inner
/// state is behind an `Arc`); every clone shares the same bus.
#[derive(Clone)]
pub struct SessionEventBroadcaster {
    sender: broadcast::Sender<SessionEvent>,
    /// Monotonic sequence counter so subscribers can detect gaps.
    seq: Arc<Mutex<u64>>,
}

impl SessionEventBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            seq: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new broadcaster with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    /// Subscribe to the event stream.
    ///
    /// The returned receiver will see events published after the call. There
    /// is no replay of historical events.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.sender.subscribe()
    }

    /// Publish an event to all current subscribers.
    ///
    /// The event's sequence number is assigned here before transmission.
    /// Returns the number of subscribers that received the event. If there
    /// are no subscribers the event is silently dropped (this is normal
    /// during startup).
    pub fn publish(&self, mut event: SessionEvent) -> usize {
        let seq = {
            let mut guard = self.seq.lock();
            *guard += 1;
            *guard
        };
        event.seq = seq;
        // No subscribers currently; not an error.
        self.sender.send(event).unwrap_or_default()
    }

    /// Publish a simple event (no payload/turn id) and return the receiver
    /// count.
    pub fn publish_simple(&self, kind: SessionEventKind, session_id: impl Into<String>) -> usize {
        self.publish(SessionEvent::new(kind, session_id))
    }

    /// Return the current receiver count.
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Return the number of events published so far.
    pub fn emitted_count(&self) -> u64 {
        *self.seq.lock()
    }
}

impl Default for SessionEventBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

/// A helper that logs every event it receives. Spawn one of these per
/// process to get a structured audit trail of session lifecycle changes.
pub struct EventLogger;

impl EventLogger {
    /// Spawn a task that drains the given receiver and logs each event at
    /// `DEBUG` level. Returns a `JoinHandle` so the caller can await or
    /// abort the task.
    pub fn spawn(mut rx: broadcast::Receiver<SessionEvent>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        debug!(
                            seq = event.seq,
                            kind = %event.kind,
                            session_id = %event.session_id,
                            turn_id = ?event.turn_id,
                            "session event"
                        );
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(lagged = n, "session event logger lagged; events dropped");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_subscribe_and_publish() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        let mut rx = broadcaster.subscribe();

        let n = broadcaster.publish_simple(SessionEventKind::Created, "s1");
        assert_eq!(n, 1);

        let event = rx.recv().await.unwrap();
        assert_eq!(event.kind, SessionEventKind::Created);
        assert_eq!(event.session_id, "s1");
        assert_eq!(event.seq, 1);
    }

    #[tokio::test]
    async fn test_no_subscribers_silent() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        // Publishing with no subscribers must not panic or error.
        let n = broadcaster.publish_simple(SessionEventKind::Updated, "s1");
        assert_eq!(n, 0);
        assert_eq!(broadcaster.emitted_count(), 1);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        let mut rx1 = broadcaster.subscribe();
        let mut rx2 = broadcaster.subscribe();

        broadcaster.publish_simple(SessionEventKind::MessageAdded, "s1");

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();
        assert_eq!(e1.seq, e2.seq);
        assert_eq!(e1.kind, SessionEventKind::MessageAdded);
    }

    #[tokio::test]
    async fn test_lagged_receiver() {
        let broadcaster = SessionEventBroadcaster::with_capacity(2);
        let mut rx = broadcaster.subscribe();

        // Publish more than the capacity to force a lag.
        for i in 0..5 {
            broadcaster.publish_simple(SessionEventKind::Updated, format!("s{i}"));
        }

        // The receiver should either get a Lagged error or the most recent
        // events; both are acceptable behaviors.
        let mut got_lagged = false;
        let mut got_event = false;
        for _ in 0..3 {
            match rx.try_recv() {
                Ok(_) => got_event = true,
                Err(broadcast::error::TryRecvError::Lagged(_)) => got_lagged = true,
                Err(_) => break,
            }
        }
        assert!(got_lagged || got_event);
    }

    #[test]
    fn test_event_builder() {
        let event = SessionEvent::new(SessionEventKind::TurnStarted, "s1")
            .with_turn("t1")
            .with_payload(serde_json::json!({"model": "gpt-4o"}));
        assert_eq!(event.turn_id, Some("t1".to_string()));
        assert!(event.payload.is_some());
    }

    #[test]
    fn test_event_kind_serde() {
        let json = serde_json::to_string(&SessionEventKind::MessageAdded).unwrap();
        assert_eq!(json, "\"message_added\"");
        let kind: SessionEventKind = serde_json::from_str("\"turn_completed\"").unwrap();
        assert_eq!(kind, SessionEventKind::TurnCompleted);
    }

    #[tokio::test]
    async fn test_event_logger_spawn() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        let rx = broadcaster.subscribe();
        let handle = EventLogger::spawn(rx);
        broadcaster.publish_simple(SessionEventKind::Created, "s1");
        // Give the logger a moment to drain.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        handle.abort();
    }
}
