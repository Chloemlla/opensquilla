//! # OpenSquilla Agents
//!
//! Agent registry, per-agent limits, and capability scoping.

pub mod limits;
pub mod registry;
pub mod scope;

pub use limits::AgentLimits;
pub use registry::{AgentDefinition, AgentRegistry};
pub use scope::{AgentScope, ScopeConfig};

/// Error type for agent operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Agent already registered: {0}")]
    AlreadyRegistered(String),

    #[error("Agent not found: {0}")]
    NotFound(String),

    #[error("Invalid scope: {0}")]
    InvalidScope(String),

    #[error("Persistence I/O error: {0}")]
    Persistence(String),
}

/// Convenience alias for agent results.
pub type Result<T> = std::result::Result<T, Error>;
