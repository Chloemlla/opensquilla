use opensquilla_core::config::Config;
use tracing::{debug, info, warn};

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// Tavily Search API adapter.
pub struct TavilySearch {
    api_key: String,
    base_url: String,
    configured: bool,
}

impl TavilySearch {
    /// Create a new Tavily search adapter.
    pub fn new(config: &Config) -> Self {
        let api_key = config.get("search.tavily.api_key").unwrap_or_default();
        let base_url = config
            .get("search.tavily.base_url")
            .unwrap_or_else(|| "https://api.tavily.com".to_string());
        let configured = !api_key.is_empty();

        if configured {
            info!("Tavily Search configured");
        } else {
            warn!("Tavily Search not configured (no API key)");
        }

        Self {
            api_key,
            base_url,
            configured,
        }
    }

    /// Perform a search via the Tavily Search API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        if !self.configured {
            return Err(SearchError::ConfigError(
                "Tavily API key not configured".to_string(),
            ));
        }

        let client = reqwest::Client::new();
        let url = format!("{}/search", self.base_url);

        let body = serde_json::json!({
            "api_key": self.api_key,
            "query": request.query,
            "max_results": request.options.max_results,
            "include_answer": false,
            "include_raw_content": false,
            "include_images": false,
            "search_depth": "basic",
        });

        let start = std::time::Instant::now();

        let response = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SearchError::NetworkError(format!("HTTP request failed: {e}")))?;

        let elapsed = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return match status.as_u16() {
                401 => Err(SearchError::AuthError("Invalid Tavily API key".to_string())),
                429 => Err(SearchError::RateLimited(
                    "Tavily rate limit exceeded".to_string(),
                )),
                _ => Err(SearchError::NetworkError(format!(
                    "Tavily returned {status}: {body_text}"
                ))),
            };
        }

        let tavily_response: TavilyResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        let results: Vec<SearchResult> = tavily_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.content,
                source: "tavily".to_string(),
                score: Some(r.score),
                published_date: None,
                is_news: false,
                is_video: false,
            })
            .collect();

        let total = results.len() as u64;

        debug!("Tavily Search returned {total} results in {elapsed}ms");

        Ok(SearchResponse {
            results,
            total_results: Some(total),
            has_more: false,
            provider: "tavily".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for TavilySearch {
    fn search<'a>(
        &'a self,
        request: &'a SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + 'a>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "tavily"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

#[derive(serde::Deserialize, Debug)]
struct TavilyResponse {
    #[serde(default)]
    results: Vec<TavilyResult>,
    #[serde(default)]
    answer: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
    score: f64,
}
