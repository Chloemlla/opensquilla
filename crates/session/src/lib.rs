//! # OpenSquilla Session Management
//!
//! Session lifecycle management, transcript storage, compaction, and usage
//! tracking backed by rusqlite.

pub mod branch;
pub mod compaction;
pub mod export;
pub mod keys;
pub mod manager;
pub mod models;
pub mod naming;
pub mod plans;
pub mod storage;
pub mod usage_ledger;

pub use branch::{BranchConfig, BranchDiff, MergeResult, SessionBrancher};
pub use compaction::{
    CompactionEngine, CompactionEstimate, CompactionEvent, CompactionExecutor, CompactionPlan,
    CompactionPlanner, CompactionReport, CompactionStrategy, SessionSummarizer,
};
pub use export::{
    render_markdown, ExportOptions, ImportOptions, ImportResult, SessionExporter,
    SessionExportDocument, SessionImporter, EXPORT_FORMAT_VERSION, export_session_json,
    export_session_markdown, import_session_json,
};
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
pub use naming::{NamingEngine, NamingOptions, NamingStrategy, SessionNamer, TitleGenerator};
pub use plans::{PlanRunReport, PlanSnapshot, PlanStateMachine, PlanStep, PlanStepStatus};
pub use storage::{SessionStorage, TARGET_SCHEMA_VERSION, migrate_schema};
pub use usage_ledger::UsageLedger;
