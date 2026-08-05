//! # OpenSquilla Plugins
//!
//! The TokenJuice plugin system: plugin traits and a thread-safe registry for
//! loading, unloading, and listing plugins.

pub mod plugin;
pub mod registry;

pub use plugin::{Plugin, PluginInfo};
pub use registry::PluginRegistry;

/// Error type for plugin operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Plugin already loaded: {0}")]
    AlreadyLoaded(String),

    #[error("Plugin not found: {0}")]
    NotFound(String),

    #[error("Plugin registry lock poisoned")]
    Poisoned,
}

/// Convenience alias for plugin results.
pub type Result<T> = std::result::Result<T, Error>;
