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

pub mod delivery;
pub mod engine;
pub mod handlers;
pub mod heartbeat;
pub mod jobs;
pub mod ops;
pub mod parser;
pub mod persistence;
pub mod reaper;
pub mod timer;
pub mod types;

pub use delivery::{DeliveryChain, WebhookDelivery, validate_webhook_url};
pub use engine::{SchedulerBuilder, SchedulerEngine};
pub use handlers::HandlerRegistry;
pub use heartbeat::{
    Heartbeat, HeartbeatBuilder, HeartbeatCheck, HeartbeatCycle, HeartbeatError, HeartbeatHandle,
    HeartbeatRunner, HeartbeatStatus, HeartbeatStore,
};
pub use jobs::{JobExecutor, TriggerError};
pub use parser::CronParser;
pub use persistence::JobStore;
pub use reaper::SessionReaper;
pub use timer::TickLoop;
pub use types::{CronJob, JobExecution, JobStatus, ScheduleKind, SchedulerStats, TickSummary};
