//! # OpenSquilla Health
//!
//! Unified health check reporting across all OpenSquilla subsystems (config,
//! database, providers, sandbox, channels).

pub mod builder;
pub mod check;
pub mod checks;
pub mod report;

pub use builder::{build_report, ReportOptions};
pub use check::HealthCheck;
pub use report::{HealthReport, HealthStatus, SubsystemHealth};

/// Error type for health check operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Health check error: {0}")]
    Check(String),
}

/// Convenience alias for health results.
pub type Result<T> = std::result::Result<T, Error>;
