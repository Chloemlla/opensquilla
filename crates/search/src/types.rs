use serde::{Deserialize, Serialize};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

/// Options for a search query.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    /// Country code for localized results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Language code for results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Whether to include a safe search filter.
    #[serde(default)]
    pub safe_search: bool,
    /// Time range for results (e.g., "d" for day, "w" for week).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_range: Option<String>,
}

fn default_max_results() -> usize {
    10
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Title of the result.
    pub title: String,
    /// URL of the result.
    pub url: String,
    /// Snippet or description of the result.
    pub snippet: String,
    /// Source/provider name.
    pub source: String,
    /// Relevance score (0.0 - 1.0) if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    /// Published date if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub published_date: Option<String>,
    /// Whether the result is from a news source.
    #[serde(default)]
    pub is_news: bool,
    /// Whether the result is a video.
    #[serde(default)]
    pub is_video: bool,
}

/// A search request to be sent to a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    /// The search query string.
    pub query: String,
    /// Search options.
    #[serde(default)]
    pub options: SearchOptions,
}

/// The response from a search provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// The search results.
    pub results: Vec<SearchResult>,
    /// Total number of results (if available).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_results: Option<u64>,
    /// Whether there are more results available.
    #[serde(default)]
    pub has_more: bool,
    /// The provider that served this response.
    pub provider: String,
    /// How long the search took in milliseconds.
    #[serde(default)]
    pub elapsed_ms: u64,
}

/// Errors that can occur during search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchError {
    /// Invalid configuration (missing API key, etc.)
    ConfigError(String),
    /// Network or HTTP error.
    NetworkError(String),
    /// Rate limited by the provider.
    RateLimited(String),
    /// Authentication error.
    AuthError(String),
    /// The provider returned an unexpected response.
    ParseError(String),
    /// No results found.
    NoResults(String),
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            SearchError::NetworkError(msg) => write!(f, "Network error: {msg}"),
            SearchError::RateLimited(msg) => write!(f, "Rate limited: {msg}"),
            SearchError::AuthError(msg) => write!(f, "Authentication error: {msg}"),
            SearchError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            SearchError::NoResults(msg) => write!(f, "No results: {msg}"),
        }
    }
}

impl std::error::Error for SearchError {}

/// The search provider trait. All search backends must implement this.
pub trait SearchProvider: Send + Sync {
    /// Search the web with the given query.
    fn search<'a>(
        &'a self,
        request: &'a SearchRequest,
    ) -> Pin<Box<dyn Future<Output = Result<SearchResponse, SearchError>> + Send + 'a>>;

    /// Return the name of this provider.
    fn name(&self) -> &str;

    /// Check if this provider is properly configured and ready.
    fn is_ready(&self) -> bool;
}
