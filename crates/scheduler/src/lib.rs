//! # OpenSquilla Scheduler
//!
//! Self-built scheduler with pure Rust cron parsing and tokio tick loop.
//! No external scheduler dependency (no APScheduler/croniter).
//!
//! ## Architecture
//!
//! - `SchedulerEngine` - Main facade that starts/stops the scheduler.
//! - `JobOps` - CRUD operations for managing scheduled jobs.
//! - `JobExecutor` - Job execution lifecycle with retry and backoff.
//! - `JobStore` - SQLite-backed persistent job storage.
//! - `CronParser` - 5-field POSIX cron expression parser.
//! - `TickLoop` - Precise sleep-to-next-due tick loop.
//! - `HandlerRegistry` - Registered handler traits.
//! - `SessionReaper` - Periodic cleanup of stale sessions.
//! - `HeartbeatRunner` - Periodic heartbeat check with SQLite-backed state.
//! - `DeliveryChain` - Result delivery (channel/WebSocket/Webhook).

pub mod engine;
pub mod types;
pub mod persistence;
pub mod parser;
pub mod timer;
pub mod ops;
pub mod jobs;
pub mod handlers;
pub mod delivery;
pub mod reaper;
pub mod heartbeat;

pub use engine::{SchedulerBuilder, SchedulerEngine};
pub use types::{CronJob, JobExecution, JobStatus, ScheduleKind, SchedulerStats, TickSummary};
pub use persistence::JobStore;
pub use parser::CronParser;
pub use timer::TickLoop;
pub use jobs::JobExecutor;
pub use handlers::HandlerRegistry;
pub use delivery::DeliveryChain;
pub use reaper::SessionReaper;
pub use heartbeat::{
    Heartbeat, HeartbeatBuilder, HeartbeatCheck, HeartbeatCycle, HeartbeatError, HeartbeatHandle,
    HeartbeatRunner, HeartbeatStatus, HeartbeatStore,
};