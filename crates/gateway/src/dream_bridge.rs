//! Gateway bridge for the dream consolidation engine.
//!
//! Mirrors the Python `dream_bridge.py`: boot installs a reconciler callback
//! that re-reconciles dream cron jobs against the live config, so RPC edits to
//! `memory.dream.*` take effect immediately instead of waiting for a restart.
//! The bridge also holds an [`Arc<DreamEngine>`] so RPC handlers can trigger
//! or query dream cycles directly.
//!
//! See `crates/memory/src/dream.rs` for the engine itself and
//! `crates/scheduler/src/handlers.rs` for the cron-driven `DreamHandler`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::result::CoreResult;
use opensquilla_memory::{DreamEngine, DreamEvent, DreamSummary};
use parking_lot::Mutex;
use uuid::Uuid;

/// An async reconciler callback that re-reconciles dream cron jobs against the
/// live config.
///
/// Boot installs this once the scheduler is ready; the RPC layer calls it after
/// config edits to `memory.dream.*` so the linkage flips immediately.
pub type DreamReconcilerFn =
    Arc<dyn (Fn() -> Pin<Box<dyn Future<Output = ()> + Send + 'static>>) + Send + Sync>;

/// Status snapshot of the dream engine.
#[derive(Debug, Clone)]
pub struct DreamStatus {
    /// Whether a dream cycle is currently due.
    pub is_due: bool,
    /// The next time a dream cycle is due (`None` when no cycle has run yet).
    pub next_due: Option<DateTime<Utc>>,
    /// The most recent dream event, if any.
    pub last_event: Option<DreamEvent>,
}

/// Thin gateway bridge over the [`DreamEngine`].
///
/// Construct with [`DreamBridge::new`] and hold in the gateway's DI container.
/// RPC handlers call [`trigger_cycle`](Self::trigger_cycle) /
/// [`status`](Self::status); boot calls
/// [`register_reconciler`](Self::register_reconciler) once the scheduler is
/// ready.
#[derive(Clone)]
pub struct DreamBridge {
    engine: Arc<DreamEngine>,
    reconciler: Arc<Mutex<Option<DreamReconcilerFn>>>,
}

impl DreamBridge {
    /// Create a new bridge over the given dream engine.
    pub fn new(engine: Arc<DreamEngine>) -> Self {
        Self {
            engine,
            reconciler: Arc::new(Mutex::new(None)),
        }
    }

    /// Trigger a full dream cycle for the given agent.
    pub async fn trigger_cycle(&self, agent_id: &Uuid) -> CoreResult<DreamSummary> {
        self.engine.run_dream_cycle(agent_id).await
    }

    /// Trigger a dream cycle only if one is due. Returns `None` when not due.
    pub async fn trigger_if_due(&self, agent_id: &Uuid) -> CoreResult<Option<DreamSummary>> {
        self.engine.run_dream_if_due(agent_id).await
    }

    /// Query the current status of the dream engine.
    pub fn status(&self) -> DreamStatus {
        DreamStatus {
            is_due: self.engine.is_due(),
            next_due: self.engine.next_dream_due_at(),
            last_event: self.engine.last_event(),
        }
    }

    /// Drain all recorded dream events.
    pub fn drain_events(&self) -> Vec<DreamEvent> {
        self.engine.drain_events()
    }

    /// Boot installs the reconciler once the scheduler is ready.
    /// Pass `None` to clear.
    pub fn register_reconciler(&self, reconciler: Option<DreamReconcilerFn>) {
        *self.reconciler.lock() = reconciler;
    }

    /// RPC + tests read the live reconciler; `None` means restart-gated.
    pub fn get_reconciler(&self) -> Option<DreamReconcilerFn> {
        self.reconciler.lock().clone()
    }

    /// Clear the reconciler (gateway shutdown / tests).
    pub fn reset_reconciler(&self) {
        *self.reconciler.lock() = None;
    }
}
