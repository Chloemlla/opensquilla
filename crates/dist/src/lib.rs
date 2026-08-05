//! # OpenSquilla Dist
//!
//! Distribution workspace state and build information.

pub mod build;
pub mod state;

pub use build::BuildInfo;
pub use state::DistState;

/// Error type for distribution state operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: String, found: String },
}

/// Convenience alias for dist results.
pub type Result<T> = std::result::Result<T, Error>;
