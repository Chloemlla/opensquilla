//! Crash recovery, state reconstruction, and partial turn replay.
//!
//! This module mirrors the Python backend's `engine/recovery/` package. It
//! provides:
//!
//! * [`CrashRecovery`] — detects interrupted turns and reconstructs the
//!   agent state from a persisted snapshot.
//! * [`StateReconstructor`] — rebuilds the conversation history and agent
//!   state from a flattened transcript.
//! * [`TurnReplay`] — replays a partial turn from the point of interruption.
//! * [`TransactionalUpdate`] — applies state changes transactionally so a
//!   crash never leaves the agent in an inconsistent state.
//!
//! The recovery module is designed to be idempotent: replaying a completed
//! turn or re-applying a committed transaction is a safe no-op.

pub mod crash;
pub mod replay;
pub mod state_reconstruction;
pub mod transactional;

pub use crash::{CrashRecovery, CrashRecoveryConfig, CrashSnapshot, RecoveryStatus};
pub use replay::{
    ReplayCheckpoint, ReplayDecision, ReplayOutcome, TurnReplay,
};
pub use state_reconstruction::{
    ReconstructedState, StateReconstructor,
};
pub use transactional::{
    TransactionalUpdate, TransactionOutcome, TransactionStatus,
};
