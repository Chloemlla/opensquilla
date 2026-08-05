//! # OpenSquilla Contrib
//!
//! Community contributions: CodeTask and SWE-bench integration.

pub mod code_task;
pub mod swebench;
pub mod task;

pub use code_task::CodeTask;
pub use swebench::{SweBenchRunner, SweBenchTask};
pub use task::{TaskInput, TaskOutput, TaskRunner};

/// Error type for contrib task operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Task execution failed: {0}")]
    TaskExecution(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Convenience alias for contrib results.
pub type Result<T> = std::result::Result<T, Error>;
