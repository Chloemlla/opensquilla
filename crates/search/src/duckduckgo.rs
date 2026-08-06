use tracing::debug;

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// DuckDuckGo search adapter (uses the HTML-based instant answer API).
pub struct DuckDuckGoSearch {
    base_url: String,
    configured: bool,
}

impl DuckDuckGoSearch {
    /// Create a new DuckDuckGo search adapter.
    pub fn new() -> Self {
        Self {
            base_url: "https://api.duckduckgo.com".to_string(),
            configured: true,
        }
    }

    /// Perform a search via the DuckDuckGo Instant Answer API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        let client = reqwest::Client::new();
        let url = format!("{}/", self.base_url);

        let params = [
            ("q", request.query.clone()),
            ("format", "json".to_string()),
            ("no_html", "1".to_string()),
            ("skip_disambig", "1".to_string()),
        ];

        let start = std::time::Instant::now();

        let response = client
            .get(&url)
            .query(&params)
            .send()
            .await
            .map_err(|e| SearchError::NetworkError(format!("HTTP request failed: {e}")))?;

        let elapsed = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            return Err(SearchError::NetworkError(format!(
                "DuckDuckGo returned {status}"
            )));
        }

        let ddg_response: DuckDuckGoResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        let mut results = Vec::new();

        // Add the abstract result if present
        if !ddg_response.abstract_text.is_empty() {
            results.push(SearchResult {
                title: ddg_response.heading.clone(),
                url: ddg_response.abstract_url.clone(),
                snippet: ddg_response.abstract_text,
                source: "duckduckgo".to_string(),
                score: Some(1.0),
                published_date: None,
                is_news: false,
                is_video: false,
            });
        }

        // Add related topics
        for topic in &ddg_response.related_topics {
            if let Some(ref first_url) = topic.first_url {
                results.push(SearchResult {
                    title: topic.text.clone(),
                    url: first_url.clone(),
                    snippet: topic.text.clone(),
                    source: "duckduckgo".to_string(),
                    score: Some(0.5),
                    published_date: None,
                    is_news: false,
                    is_video: false,
                });
            }
        }

        // Limit results to max_results
        if results.len() > request.options.max_results {
            results.truncate(request.options.max_results);
        }

        debug!(
            "DuckDuckGo returned {} results in {elapsed}ms",
            results.len()
        );

        let total_results = Some(results.len() as u64);
        Ok(SearchResponse {
            results,
            total_results,
            has_more: false,
            provider: "duckduckgo".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for DuckDuckGoSearch {
    fn search<'a>(
        &'a self,
        request: &'a SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + 'a>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "duckduckgo"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

impl Default for DuckDuckGoSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct DuckDuckGoResponse {
    #[serde(default)]
    heading: String,
    #[serde(default)]
    abstract_text: String,
    #[serde(default)]
    abstract_url: String,
    #[serde(default)]
    related_topics: Vec<DuckDuckGoTopic>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct DuckDuckGoTopic {
    text: String,
    #[serde(default)]
    first_url: Option<String>,
}
