//! # OpenSquilla Uninstall
//!
//! Safe uninstall: inventory checking, plan generation, and safe deletion.

pub mod execute;
pub mod inventory;
pub mod plan;

pub use execute::{dry_run, execute};
pub use inventory::{scan_install, InventoryItem, InventoryItemKind};
pub use plan::{UninstallAction, UninstallActionKind, UninstallPlan};

/// Error type for uninstall operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Uninstall aborted: {0}")]
    Aborted(String),

    #[error("Plan execution failed: {0}")]
    Execution(String),
}

/// Convenience alias for uninstall results.
pub type Result<T> = std::result::Result<T, Error>;
