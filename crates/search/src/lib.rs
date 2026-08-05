//! # OpenSquilla Search
//!
//! Search providers for web and news search. Supports Brave, DuckDuckGo,
//! Tavily, Exa, Bocha, and IQS search APIs.

pub mod bocha;
pub mod brave;
pub mod duckduckgo;
pub mod exa;
pub mod iqs;
pub mod registry;
pub mod tavily;
pub mod types;

pub use registry::SearchRegistry;
pub use types::{SearchError, SearchOptions, SearchProvider, SearchRequest, SearchResult, SearchResponse};