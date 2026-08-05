//! # OpenSquilla Persistence
//!
//! SQLite schema migrations, process PID locking, and pre-migration snapshot
//! backups for the OpenSquilla gateway.

pub mod backup;
pub mod migration;
pub mod pidlock;
pub mod runner;

pub use backup::{snapshot, Backup};
pub use migration::{AppliedMigration, SchemaMigration};
pub use pidlock::PidLock;
pub use runner::MigrationRunner;

/// Error type for persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Migration {0} already applied")]
    AlreadyApplied(i64),

    #[error("Rollback failed: {0}")]
    RollbackFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Convenience alias for persistence results.
pub type Result<T> = std::result::Result<T, Error>;
