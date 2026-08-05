//! Session stream management.
//!
//! Mirrors the Python `session_streams.py` module. Provides a real-time
//! update stream for a single session, multiplexed fan-out to multiple
//! subscribers, backpressure handling, and stream cancellation.
//!
//! The stream is built on [`tokio::sync::broadcast`] so that multiple
//! subscribers each receive the same sequence of session updates; slow
//! subscribers are notified of lagged/overflowing buffers rather than
//! blocking the producers. Streams are cancelled by dropping the receiver.

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use chrono::{DateTime, Utc};
use futures::Stream;
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, warn};

use crate::session_events::{SessionEvent, SessionEventBroadcaster, SessionEventKind};

/// Default stream channel capacity.
pub const DEFAULT_STREAM_CAPACITY: usize = 256;

/// A single update in a session stream.
///
/// This is a lightweight wrapper over [`SessionEvent`] that carries the
/// session id so a multiplexed stream can be filtered by session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStreamUpdate {
    /// The session this update belongs to.
    pub session_id: String,
    /// The underlying lifecycle event.
    pub event: SessionEvent,
}

impl SessionStreamUpdate {
    /// Wrap an event for a session.
    pub fn from_event(event: SessionEvent) -> Self {
        let session_id = event.session_id.clone();
        Self { session_id, event }
    }
}

/// An update filter applied to a session stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamFilter {
    /// Only forward events of these kinds. `None` forwards all kinds.
    pub kinds: Option<&'static [SessionEventKind]>,
}

impl StreamFilter {
    /// Return `true` if the given event kind passes the filter.
    pub fn accepts(&self, kind: SessionEventKind) -> bool {
        match self.kinds {
            Some(kinds) => kinds.contains(&kind),
            None => true,
        }
    }
}

/// A broadcast-based session update bus.
///
/// Producers call [`SessionStreamBus::publish`]; consumers call
/// [`SessionStreamBus::subscribe`] and consume the returned receiver as a
/// [`Stream`].
#[derive(Clone)]
pub struct SessionStreamBus {
    sender: broadcast::Sender<SessionStreamUpdate>,
    /// Track subscriber count for diagnostics.
    subscriber_count: Arc<Mutex<usize>>,
}

impl SessionStreamBus {
    /// Create a new bus with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            subscriber_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new bus with the default capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_STREAM_CAPACITY)
    }

    /// Build a bus directly from a [`SessionEventBroadcaster`] by forwarding
    /// its events onto a new stream bus in a background task.
    pub fn from_event_broadcaster(broadcaster: &SessionEventBroadcaster) -> Self {
        let bus = Self::new();
        let mut rx = broadcaster.subscribe();
        let sink = bus.clone();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        sink.publish(SessionStreamUpdate::from_event(event));
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(lagged = n, "stream bus lagged behind event broadcaster");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
        bus
    }

    /// Subscribe to the update stream.
    ///
    /// The returned receiver can be consumed as a [`Stream`] via
    /// [`SessionReceiver::into_stream`], or polled directly with
    /// [`SessionReceiver::recv`].
    pub fn subscribe(&self) -> SessionReceiver {
        {
            let mut count = self.subscriber_count.lock();
            *count += 1;
        }
        SessionReceiver {
            inner: self.sender.subscribe(),
            filter: StreamFilter::default(),
            session_filter: None,
        }
    }

    /// Publish an update to all subscribers.
    ///
    /// Returns the number of receivers that accepted the update.
    pub fn publish(&self, update: SessionStreamUpdate) -> usize {
        match self.sender.send(update) {
            Ok(n) => n,
            Err(_) => 0,
        }
    }

    /// Publish a simple lifecycle event for a session.
    pub fn publish_event(&self, kind: SessionEventKind, session_id: impl Into<String>) -> usize {
        let event = SessionEvent::new(kind, session_id);
        let update = SessionStreamUpdate::from_event(event);
        self.publish(update)
    }

    /// Return the current subscriber count.
    pub fn subscriber_count(&self) -> usize {
        *self.subscriber_count.lock()
    }

    /// Return the total receiver count seen by the underlying channel.
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for SessionStreamBus {
    fn default() -> Self {
        Self::new()
    }
}

/// A stream receiver for session updates.
///
/// Wrap in [`SessionReceiver::into_stream`] to get a `futures::Stream`, or
/// use [`SessionReceiver::recv`] directly.
pub struct SessionReceiver {
    inner: broadcast::Receiver<SessionStreamUpdate>,
    filter: StreamFilter,
    session_filter: Option<String>,
}

impl SessionReceiver {
    /// Apply a filter so only matching event kinds are forwarded.
    pub fn with_filter(mut self, filter: StreamFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Filter to a single session id. Updates for other sessions are skipped.
    pub fn for_session(mut self, session_id: &str) -> Self {
        self.session_filter = Some(session_id.to_string());
        self
    }

    /// Convert this receiver into a [`Stream`] of filtered updates.
    pub fn into_stream(self) -> SessionStream {
        SessionStream { inner: self }
    }

    /// Return `true` if this update passes the configured filters.
    fn accepts(&self, update: &SessionStreamUpdate) -> bool {
        if let Some(ref session_id) = self.session_filter {
            if &update.session_id != session_id {
                return false;
            }
        }
        self.filter.accepts(update.event.kind)
    }

    /// Receive the next update, applying the filters.
    pub async fn recv(&mut self) -> Result<SessionStreamUpdate, StreamError> {
        loop {
            match self.inner.recv().await {
                Ok(update) => {
                    if self.accepts(&update) {
                        return Ok(update);
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    return Err(StreamError::Lagged(n));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(StreamError::Closed);
                }
            }
        }
    }
}

impl std::fmt::Debug for SessionReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionReceiver")
            .field("filter", &self.filter)
            .field("session_filter", &self.session_filter)
            .finish()
    }
}

/// A `futures::Stream` adapter over a [`SessionReceiver`].
pub struct SessionStream {
    inner: SessionReceiver,
}

impl Stream for SessionStream {
    type Item = Result<SessionStreamUpdate, StreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        // Poll the raw broadcast receiver. On `Ready(Ok(update))`, apply the
        // filters; if they reject it, keep polling.
        loop {
            match this.inner.inner.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Ok(update)) => {
                    if this.inner.accepts(&update) {
                        return Poll::Ready(Some(Ok(update)));
                    }
                }
                Poll::Ready(Err(broadcast::error::RecvError::Lagged(n))) => {
                    return Poll::Ready(Some(Err(StreamError::Lagged(n))));
                }
                Poll::Ready(Err(broadcast::error::RecvError::Closed)) => {
                    return Poll::Ready(None);
                }
            }
        }
    }
}

/// Errors that can occur while consuming a session stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// The receiver fell behind and `n` updates were dropped.
    Lagged(u64),
    /// The stream was closed (all senders dropped).
    Closed,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Lagged(n) => write!(f, "stream lagged by {n} updates"),
            StreamError::Closed => write!(f, "stream closed"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Multiplexes one underlying bus into several per-session streams.
///
/// Used by the WebSocket layer to give each client its own view of session
/// updates without each client subscribing separately to the bus.
pub struct SessionStreamMultiplexer {
    bus: SessionStreamBus,
    routes: Arc<Mutex<std::collections::HashMap<String, usize>>>,
}

impl SessionStreamMultiplexer {
    /// Create a new multiplexer over the given bus.
    pub fn new(bus: SessionStreamBus) -> Self {
        Self {
            bus,
            routes: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Return a stream of updates filtered to a single session id.
    pub fn stream_for(&self, session_id: &str) -> SessionStream {
        let receiver = self.bus.subscribe().for_session(session_id);
        {
            let mut routes = self.routes.lock();
            let count = routes.entry(session_id.to_string()).or_insert(0);
            *count += 1;
        }
        receiver.into_stream()
    }

    /// Release one subscription for a session id. Returns the remaining
    /// subscriber count.
    pub fn unsubscribe(&self, session_id: &str) -> usize {
        let mut routes = self.routes.lock();
        let mut remaining = 0;
        if let Some(count) = routes.get_mut(session_id) {
            if *count > 0 {
                *count -= 1;
            }
            remaining = *count;
            if remaining == 0 {
                routes.remove(session_id);
            }
        }
        remaining
    }

    /// Return the number of sessions currently multiplexed.
    pub fn active_sessions(&self) -> usize {
        self.routes.lock().len()
    }
}

/// Backpressure diagnostics for a stream consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamBackpressure {
    pub session_id: String,
    pub last_lag: u64,
    pub total_lagged_updates: u64,
    pub updated_at: DateTime<Utc>,
}

/// Tracks backpressure events for a single stream consumer.
#[derive(Default)]
pub struct BackpressureTracker {
    session_id: String,
    total_lagged: u64,
    last_lag: u64,
    updated_at: Option<DateTime<Utc>>,
}

impl BackpressureTracker {
    /// Create a new tracker for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    /// Record a lagged event.
    pub fn record_lag(&mut self, lag: u64) {
        self.last_lag = lag;
        self.total_lagged = self.total_lagged.saturating_add(lag);
        self.updated_at = Some(Utc::now());
    }

    /// Export the current diagnostics snapshot.
    pub fn snapshot(&self) -> StreamBackpressure {
        StreamBackpressure {
            session_id: self.session_id.clone(),
            last_lag: self.last_lag,
            total_lagged_updates: self.total_lagged,
            updated_at: self.updated_at.unwrap_or_else(Utc::now),
        }
    }
}

/// Helper that converts a [`StreamError`] into an [`AppError`].
pub fn stream_error_to_app_error(err: StreamError) -> AppError {
    AppError::internal(format!("Session stream error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_subscribe_and_receive() {
        let bus = SessionStreamBus::with_capacity(16);
        let mut rx = bus.subscribe();
        bus.publish_event(SessionEventKind::Created, "s1");
        let update = rx.recv().await.unwrap();
        assert_eq!(update.session_id, "s1");
        assert_eq!(update.event.kind, SessionEventKind::Created);
    }

    #[tokio::test]
    async fn test_filter_rejects_kinds() {
        use SessionEventKind as K;
        let bus = SessionStreamBus::with_capacity(16);
        let mut rx = bus
            .subscribe()
            .with_filter(StreamFilter {
                kinds: Some(&[K::TurnStarted, K::TurnCompleted]),
            });
        bus.publish_event(K::Created, "s1");
        bus.publish_event(K::TurnStarted, "s1");
        let update = rx.recv().await.unwrap();
        assert_eq!(update.event.kind, K::TurnStarted);
    }

    #[tokio::test]
    async fn test_for_session_filters() {
        let bus = SessionStreamBus::with_capacity(16);
        let mut rx = bus.subscribe().for_session("s1");
        bus.publish_event(SessionEventKind::Created, "s2");
        bus.publish_event(SessionEventKind::Created, "s1");
        let update = rx.recv().await.unwrap();
        assert_eq!(update.session_id, "s1");
    }

    #[tokio::test]
    async fn test_stream_adapter() {
        let bus = SessionStreamBus::with_capacity(16);
        let mut stream = bus.subscribe().into_stream();
        bus.publish_event(SessionEventKind::Created, "s1");
        let item = futures::StreamExt::next(&mut stream).await.unwrap().unwrap();
        assert_eq!(item.session_id, "s1");
    }

    #[tokio::test]
    async fn test_multiplexer() {
        let bus = SessionStreamBus::with_capacity(16);
        let mux = SessionStreamMultiplexer::new(bus.clone());
        let mut s1_stream = mux.stream_for("s1");
        bus.publish_event(SessionEventKind::TurnCompleted, "s1");
        bus.publish_event(SessionEventKind::TurnCompleted, "s2");
        let item = futures::StreamExt::next(&mut s1_stream)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(item.session_id, "s1");
        assert_eq!(mux.active_sessions(), 1);
        assert_eq!(mux.unsubscribe("s1"), 0);
        assert_eq!(mux.active_sessions(), 0);
    }

    #[test]
    fn test_backpressure_tracker() {
        let mut tracker = BackpressureTracker::new("s1");
        tracker.record_lag(5);
        tracker.record_lag(3);
        let snap = tracker.snapshot();
        assert_eq!(snap.total_lagged_updates, 8);
        assert_eq!(snap.last_lag, 3);
    }

    #[test]
    fn test_stream_error_conversion() {
        let err = stream_error_to_app_error(StreamError::Lagged(2));
        assert!(err.to_string().contains("lagged"));
    }

    #[tokio::test]
    async fn test_from_event_broadcaster_forward() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        let bus = SessionStreamBus::from_event_broadcaster(&broadcaster);
        let mut rx = bus.subscribe();
        broadcaster.publish_simple(SessionEventKind::Created, "s1");
        // The forward task is async; poll with a timeout.
        let timeout = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv());
        let update = timeout.await.unwrap().unwrap();
        assert_eq!(update.event.kind, SessionEventKind::Created);
    }
}
