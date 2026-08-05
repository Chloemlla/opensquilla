//! # OpenSquilla Identity
//!
//! Terminal identity: bootstrapping, prompt templates, workspace discovery,
//! and input parsing for the interactive terminal surface.

pub mod identity;
pub mod parser;
pub mod prompt;
pub mod workspace;

pub use identity::Identity;
pub use parser::ParsedCommand;
pub use prompt::PromptTemplate;
pub use workspace::Workspace;

use std::path::PathBuf;

/// Error type for identity operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Workspace error: {0}")]
    Workspace(String),
}

/// Convenience alias for identity results.
pub type Result<T> = std::result::Result<T, Error>;
