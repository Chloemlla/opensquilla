//! Session lifecycle state machine.
//!
//! Mirrors the Python `session_lifecycle.py` module. Models the legal
//! transitions a session can make through its lifetime:
//!
//! ```text
//! Created ─▶ Active ─⇄ Paused
//!    │          │
//!    │          ▼
//!    │      Archived ─▶ Deleted
//!    │          ▲
//!    └──────────┘ (archive directly from Created)
//! ```
//!
//! Each transition is validated against the current state, emits a lifecycle
//! hook, and publishes a [`SessionEvent`] on the supplied broadcaster so
//! downstream consumers (memory, audit log, WebSocket clients) react in real
//! time.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::session_events::{SessionEvent, SessionEventBroadcaster, SessionEventKind};

/// The discrete states a session can occupy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    /// Just created, not yet activated.
    Created,
    /// Currently active and able to process turns.
    Active,
    /// Temporarily paused; turns are rejected until resumed.
    Paused,
    /// Archived; read-only, no new turns.
    Archived,
    /// Soft-deleted; eligible for garbage collection.
    Deleted,
}

impl SessionState {
    /// Stable string identifier used in storage and API output.
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionState::Created => "created",
            SessionState::Active => "active",
            SessionState::Paused => "paused",
            SessionState::Archived => "archived",
            SessionState::Deleted => "deleted",
        }
    }

    /// Parse a state from its string identifier.
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s {
            "created" => Ok(SessionState::Created),
            "active" => Ok(SessionState::Active),
            "paused" => Ok(SessionState::Paused),
            "archived" => Ok(SessionState::Archived),
            "deleted" => Ok(SessionState::Deleted),
            other => Err(AppError::bad_request(format!(
                "Unknown session state '{other}'"
            ))),
        }
    }

    /// Return `true` if the session can accept new turns.
    pub fn accepts_turns(&self) -> bool {
        matches!(self, SessionState::Active)
    }

    /// Return `true` if the session is read-only.
    pub fn is_read_only(&self) -> bool {
        matches!(self, SessionState::Archived | SessionState::Deleted)
    }

    /// Return `true` if the state is terminal (no further transitions).
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionState::Deleted)
    }
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The set of lifecycle transitions that a state machine can apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleTransition {
    Activate,
    Pause,
    Resume,
    Archive,
    Delete,
    Restore,
}

impl LifecycleTransition {
    /// The event kind emitted when this transition fires.
    pub fn event_kind(&self) -> SessionEventKind {
        match self {
            LifecycleTransition::Activate => SessionEventKind::Updated,
            LifecycleTransition::Pause => SessionEventKind::Paused,
            LifecycleTransition::Resume => SessionEventKind::Resumed,
            LifecycleTransition::Archive => SessionEventKind::Archived,
            LifecycleTransition::Delete => SessionEventKind::Deleted,
            LifecycleTransition::Restore => SessionEventKind::Updated,
        }
    }
}

/// Validate that a transition is legal from the given state.
///
/// Returns the resulting state on success, or an [`AppError`] describing why
/// the transition is not allowed.
pub fn validate_transition(
    from: SessionState,
    transition: LifecycleTransition,
) -> Result<SessionState, AppError> {
    use {LifecycleTransition as T, SessionState as S};
    let next = match (from, transition) {
        (S::Created, T::Activate) => S::Active,
        (S::Created, T::Archive) => S::Archived,
        (S::Created, T::Delete) => S::Deleted,
        (S::Active, T::Pause) => S::Paused,
        (S::Active, T::Archive) => S::Archived,
        (S::Active, T::Delete) => S::Deleted,
        (S::Paused, T::Resume) => S::Active,
        (S::Paused, T::Archive) => S::Archived,
        (S::Paused, T::Delete) => S::Deleted,
        (S::Archived, T::Restore) => S::Active,
        (S::Archived, T::Delete) => S::Deleted,
        (S::Deleted, _) => {
            return Err(AppError::bad_request(
                "Cannot transition from terminal 'deleted' state",
            ));
        }
        (state, t) => {
            return Err(AppError::bad_request(format!(
                "Illegal transition '{:?}' from state '{}'",
                t, state
            )));
        }
    };
    Ok(next)
}

/// A hook invoked before/after a lifecycle transition. Implementations can
/// reject a transition by returning an error from the `before` callback.
pub trait LifecycleHook: Send + Sync {
    /// Called before the transition is applied. Return an error to abort.
    fn before(
        &self,
        session_id: &str,
        from: SessionState,
        transition: LifecycleTransition,
    ) -> Result<(), AppError> {
        let _ = (session_id, from, transition);
        Ok(())
    }

    /// Called after the transition has been applied.
    fn after(
        &self,
        session_id: &str,
        from: SessionState,
        to: SessionState,
        transition: LifecycleTransition,
    ) {
        let _ = (session_id, from, to, transition);
    }
}

/// A no-op hook for use as a default.
pub struct NoopHook;

impl LifecycleHook for NoopHook {}

/// Tracks the lifecycle state of a single session.
#[derive(Debug, Clone)]
pub struct SessionLifecycle {
    pub session_id: String,
    pub state: SessionState,
    pub entered_at: DateTime<Utc>,
    pub history: Vec<LifecycleRecord>,
}

/// A single recorded transition in a session's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleRecord {
    pub transition: String,
    pub from: String,
    pub to: String,
    pub at: DateTime<Utc>,
}

/// Thread-safe manager for the lifecycle state of many sessions.
#[derive(Clone)]
pub struct SessionLifecycleManager {
    sessions: Arc<Mutex<std::collections::HashMap<String, SessionLifecycle>>>,
    hooks: Arc<Vec<Arc<dyn LifecycleHook>>>,
    broadcaster: SessionEventBroadcaster,
}

impl SessionLifecycleManager {
    /// Create a new manager with the given event broadcaster and no hooks.
    pub fn new(broadcaster: SessionEventBroadcaster) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            hooks: Arc::new(Vec::new()),
            broadcaster,
        }
    }

    /// Create a new manager with explicit lifecycle hooks.
    pub fn with_hooks(
        broadcaster: SessionEventBroadcaster,
        hooks: Vec<Arc<dyn LifecycleHook>>,
    ) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(std::collections::HashMap::new())),
            hooks: Arc::new(hooks),
            broadcaster,
        }
    }

    /// Register a new session in the `Created` state. Returns an error if a
    /// session with the same id is already tracked.
    pub fn register(&self, session_id: impl Into<String>) -> Result<(), AppError> {
        let session_id = session_id.into();
        let mut sessions = self.sessions.lock();
        if sessions.contains_key(&session_id) {
            return Err(AppError::bad_request(format!(
                "Session '{session_id}' is already tracked"
            )));
        }
        let now = Utc::now();
        sessions.insert(
            session_id.clone(),
            SessionLifecycle {
                session_id: session_id.clone(),
                state: SessionState::Created,
                entered_at: now,
                history: Vec::new(),
            },
        );
        drop(sessions);
        self.broadcaster
            .publish_simple(SessionEventKind::Created, session_id);
        Ok(())
    }

    /// Return the current state of a session, if tracked.
    pub fn state(&self, session_id: &str) -> Option<SessionState> {
        self.sessions.lock().get(session_id).map(|s| s.state)
    }

    /// Return a snapshot of a session's lifecycle, if tracked.
    pub fn get(&self, session_id: &str) -> Option<SessionLifecycle> {
        self.sessions.lock().get(session_id).cloned()
    }

    /// Apply a transition to a session.
    ///
    /// Runs all `before` hooks (any of which may abort), validates the
    /// transition, updates the state, records the transition in history,
    /// publishes the lifecycle event, then runs all `after` hooks.
    pub fn transition(
        &self,
        session_id: &str,
        transition: LifecycleTransition,
    ) -> Result<SessionState, AppError> {
        // Run before hooks while holding a snapshot of the current state.
        let from = {
            let sessions = self.sessions.lock();
            let session = sessions.get(session_id).ok_or_else(|| {
                AppError::not_found(format!("Session '{session_id}' not tracked"))
            })?;
            session.state
        };

        for hook in self.hooks.iter() {
            hook.before(session_id, from, transition)?;
        }

        let to = validate_transition(from, transition)?;
        let now = Utc::now();
        let record = LifecycleRecord {
            transition: format!("{transition:?}"),
            from: from.to_string(),
            to: to.to_string(),
            at: now,
        };

        {
            let mut sessions = self.sessions.lock();
            let session = sessions.get_mut(session_id).ok_or_else(|| {
                AppError::not_found(format!("Session '{session_id}' not tracked"))
            })?;
            session.state = to;
            session.entered_at = now;
            session.history.push(record);
        }

        let event = SessionEvent::new(transition.event_kind(), session_id).with_payload(
            serde_json::json!({
                "from": from.to_string(),
                "to": to.to_string(),
            }),
        );
        self.broadcaster.publish(event);

        for hook in self.hooks.iter() {
            hook.after(session_id, from, to, transition);
        }

        Ok(to)
    }

    /// Convenience: activate a session.
    pub fn activate(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Activate)
    }

    /// Convenience: pause a session.
    pub fn pause(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Pause)
    }

    /// Convenience: resume a session.
    pub fn resume(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Resume)
    }

    /// Convenience: archive a session.
    pub fn archive(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Archive)
    }

    /// Convenience: delete a session.
    pub fn delete(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Delete)
    }

    /// Convenience: restore a session from archived to active.
    pub fn restore(&self, session_id: &str) -> Result<SessionState, AppError> {
        self.transition(session_id, LifecycleTransition::Restore)
    }

    /// Return the number of tracked sessions.
    pub fn len(&self) -> usize {
        self.sessions.lock().len()
    }

    /// Return `true` if no sessions are tracked.
    pub fn is_empty(&self) -> bool {
        self.sessions.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> SessionLifecycleManager {
        SessionLifecycleManager::new(SessionEventBroadcaster::new())
    }

    #[test]
    fn test_state_parse_roundtrip() {
        for state in [
            SessionState::Created,
            SessionState::Active,
            SessionState::Paused,
            SessionState::Archived,
            SessionState::Deleted,
        ] {
            let parsed = SessionState::parse(state.as_str()).unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn test_accepts_turns_and_terminal() {
        assert!(SessionState::Active.accepts_turns());
        assert!(!SessionState::Paused.accepts_turns());
        assert!(SessionState::Archived.is_read_only());
        assert!(SessionState::Deleted.is_terminal());
        assert!(!SessionState::Active.is_terminal());
    }

    #[test]
    fn test_validate_legal_transitions() {
        assert_eq!(
            validate_transition(SessionState::Created, LifecycleTransition::Activate).unwrap(),
            SessionState::Active
        );
        assert_eq!(
            validate_transition(SessionState::Active, LifecycleTransition::Pause).unwrap(),
            SessionState::Paused
        );
        assert_eq!(
            validate_transition(SessionState::Paused, LifecycleTransition::Resume).unwrap(),
            SessionState::Active
        );
        assert_eq!(
            validate_transition(SessionState::Active, LifecycleTransition::Archive).unwrap(),
            SessionState::Archived
        );
        assert_eq!(
            validate_transition(SessionState::Archived, LifecycleTransition::Restore).unwrap(),
            SessionState::Active
        );
    }

    #[test]
    fn test_validate_illegal_transitions() {
        assert!(validate_transition(SessionState::Paused, LifecycleTransition::Activate).is_err());
        assert!(validate_transition(SessionState::Deleted, LifecycleTransition::Restore).is_err());
        assert!(validate_transition(SessionState::Active, LifecycleTransition::Resume).is_err());
    }

    #[test]
    fn test_register_and_activate() {
        let mgr = manager();
        mgr.register("s1").unwrap();
        assert_eq!(mgr.state("s1"), Some(SessionState::Created));
        let to = mgr.activate("s1").unwrap();
        assert_eq!(to, SessionState::Active);
        let lifecycle = mgr.get("s1").unwrap();
        assert_eq!(lifecycle.history.len(), 1);
    }

    #[test]
    fn test_duplicate_register_rejected() {
        let mgr = manager();
        mgr.register("s1").unwrap();
        assert!(mgr.register("s1").is_err());
    }

    #[test]
    fn test_full_lifecycle_chain() {
        let mgr = manager();
        mgr.register("s1").unwrap();
        assert_eq!(mgr.activate("s1").unwrap(), SessionState::Active);
        assert_eq!(mgr.pause("s1").unwrap(), SessionState::Paused);
        assert_eq!(mgr.resume("s1").unwrap(), SessionState::Active);
        assert_eq!(mgr.archive("s1").unwrap(), SessionState::Archived);
        assert_eq!(mgr.restore("s1").unwrap(), SessionState::Active);
        assert_eq!(mgr.delete("s1").unwrap(), SessionState::Deleted);
        let lifecycle = mgr.get("s1").unwrap();
        // register does not create a history entry; transitions do.
        assert_eq!(lifecycle.history.len(), 6);
    }

    #[tokio::test]
    async fn test_events_emitted_on_transition() {
        let broadcaster = SessionEventBroadcaster::with_capacity(16);
        let mut rx = broadcaster.subscribe();
        let mgr = SessionLifecycleManager::new(broadcaster);

        mgr.register("s1").unwrap();
        // register publishes a Created event.
        let created = rx.recv().await.unwrap();
        assert_eq!(created.kind, SessionEventKind::Created);

        mgr.pause("s1").map_err(|e| panic!("{e}")).ok();
        // Pause requires Active, so activate first.
        mgr.activate("s1").unwrap();
        let activated = rx.recv().await.unwrap();
        assert_eq!(activated.kind, SessionEventKind::Updated);
    }

    #[test]
    fn test_hook_can_abort_transition() {
        struct RejectPause;
        impl LifecycleHook for RejectPause {
            fn before(
                &self,
                _id: &str,
                _from: SessionState,
                transition: LifecycleTransition,
            ) -> Result<(), AppError> {
                if matches!(transition, LifecycleTransition::Pause) {
                    return Err(AppError::bad_request("Pause not allowed"));
                }
                Ok(())
            }
        }
        let broadcaster = SessionEventBroadcaster::new();
        let mgr = SessionLifecycleManager::with_hooks(broadcaster, vec![Arc::new(RejectPause)]);
        mgr.register("s1").unwrap();
        mgr.activate("s1").unwrap();
        assert!(mgr.pause("s1").is_err());
        assert_eq!(mgr.state("s1"), Some(SessionState::Active));
    }

    #[test]
    fn test_transition_unknown_session() {
        let mgr = manager();
        assert!(mgr.activate("nope").is_err());
    }
}
