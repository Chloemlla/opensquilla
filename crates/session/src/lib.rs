//! # OpenSquilla Session Management
//!
//! Session lifecycle management, transcript storage, compaction, and usage
//! tracking backed by rusqlite.

pub mod storage;
pub mod manager;
pub mod models;
pub mod keys;
pub mod compaction;
pub mod naming;
pub mod plans;
pub mod usage_ledger;

pub use storage::SessionStorage;
pub use manager::{
    CreateSessionConfig, ForkConfig, LifecyclePhase, RoutingPrefs, SessionError, SessionFilter,
    SessionManager, SessionTransition,
};
pub use models::{
    AgentTask, CompactedTranscriptEntry, CompactionHistory, PlanRevision, PlanRun, PlanRunStatus,
    PlanStatus, ProjectWorkspace, RoutingDecision, Session, SessionAttachment, SessionContextState,
    SessionFork, SessionLock, SessionMetadata, SessionMode, SessionStatus, SessionSummary,
    SessionTag, TaskStatus, TranscriptEntry, UsageEvent, UsageEventItem, UsageLedgerState,
};
pub use compaction::{
    CompactionEngine, CompactionEstimate, CompactionEvent, CompactionExecutor, CompactionPlan,
    CompactionPlanner, CompactionReport, CompactionStrategy, SessionSummarizer,
};
pub use naming::{NamingEngine, NamingOptions, NamingStrategy, SessionNamer, TitleGenerator};
pub use plans::{PlanRunReport, PlanSnapshot, PlanStateMachine, PlanStep, PlanStepStatus};
pub use usage_ledger::UsageLedger;