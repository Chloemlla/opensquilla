//! # OpenSquilla Chat
//!
//! Conversation abstraction: message management, history trimming, and source
//! tracking for interactive chat sessions.

pub mod conversation;
pub mod history;
pub mod source;

pub use conversation::Conversation;
pub use history::{trim_history, HistoryTrimOptions};
pub use source::SessionSource;

/// Error type for chat operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),
}

/// Convenience alias for chat results.
pub type Result<T> = std::result::Result<T, Error>;
