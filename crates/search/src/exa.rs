use opensquilla_core::config::Config;
use tracing::{debug, info, warn};

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// Exa Search API adapter (formerly Metaphor).
pub struct ExaSearch {
    api_key: String,
    base_url: String,
    configured: bool,
}

impl ExaSearch {
    /// Create a new Exa search adapter.
    pub fn new(config: &Config) -> Self {
        let api_key = config.get("search.exa.api_key").unwrap_or_default();
        let base_url = config
            .get("search.exa.base_url")
            .unwrap_or_else(|| "https://api.exa.ai".to_string());
        let configured = !api_key.is_empty();

        if configured {
            info!("Exa Search configured");
        } else {
            warn!("Exa Search not configured (no API key)");
        }

        Self {
            api_key,
            base_url,
            configured,
        }
    }

    /// Perform a search via the Exa Search API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        if !self.configured {
            return Err(SearchError::ConfigError(
                "Exa API key not configured".to_string(),
            ));
        }

        let client = reqwest::Client::new();
        let url = format!("{}/search", self.base_url);

        let body = serde_json::json!({
            "query": request.query,
            "numResults": request.options.max_results,
            "useAutoprompt": false,
            "type": "keyword",
            "includeDomains": null,
            "excludeDomains": null,
            "startPublishedDate": null,
            "endPublishedDate": null,
        });

        let start = std::time::Instant::now();

        let response = client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| SearchError::NetworkError(format!("HTTP request failed: {e}")))?;

        let elapsed = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return match status.as_u16() {
                401 => Err(SearchError::AuthError("Invalid Exa API key".to_string())),
                429 => Err(SearchError::RateLimited(
                    "Exa rate limit exceeded".to_string(),
                )),
                _ => Err(SearchError::NetworkError(format!(
                    "Exa returned {status}: {body_text}"
                ))),
            };
        }

        let exa_response: ExaResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        let results: Vec<SearchResult> = exa_response
            .results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                snippet: r.text.unwrap_or_default(),
                source: "exa".to_string(),
                score: Some(r.score),
                published_date: r.published_date,
                is_news: false,
                is_video: false,
            })
            .collect();

        let total = results.len() as u64;

        debug!("Exa Search returned {total} results in {elapsed}ms");

        Ok(SearchResponse {
            results,
            total_results: Some(total),
            has_more: false,
            provider: "exa".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for ExaSearch {
    fn search(
        &self,
        request: &SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + '_>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "exa"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

#[derive(serde::Deserialize, Debug)]
struct ExaResponse {
    #[serde(default)]
    results: Vec<ExaResult>,
    #[serde(default)]
    autoprompt_string: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
struct ExaResult {
    title: String,
    url: String,
    #[serde(default)]
    text: Option<String>,
    score: f64,
    #[serde(default)]
    published_date: Option<String>,
    #[serde(default)]
    author: Option<String>,
}
