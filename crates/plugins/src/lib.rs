//! # OpenSquilla Plugins
//!
//! The TokenJuice plugin system: plugin traits and a thread-safe registry for
//! loading, unloading, and listing plugins, plus the TokenJuice tool-result
//! reducer that compresses verbose tool output before it re-enters the LLM
//! context.

pub mod plugin;
pub mod registry;
pub mod tokenjuice;

pub use plugin::{Plugin, PluginInfo};
pub use registry::PluginRegistry;
pub use tokenjuice::{
    Reduction, Rule, default_rules, format_inline, load_rules, reduce_tool_result,
    reduce_tool_result_with_limit, reduce_with_rule, select_rule, sort_rules,
};

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
