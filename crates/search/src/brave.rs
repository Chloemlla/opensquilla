use opensquilla_core::config::Config;
use tracing::{debug, info, warn};

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// Brave Search API adapter.
pub struct BraveSearch {
    api_key: String,
    base_url: String,
    configured: bool,
}

impl BraveSearch {
    /// Create a new Brave Search adapter.
    pub fn new(config: &Config) -> Self {
        let api_key = config.get("search.brave.api_key").unwrap_or_default();
        let base_url = config
            .get("search.brave.base_url")
            .unwrap_or_else(|| "https://api.search.brave.com".to_string());
        let configured = !api_key.is_empty();

        if configured {
            info!("Brave Search configured");
        } else {
            warn!("Brave Search not configured (no API key)");
        }

        Self {
            api_key,
            base_url,
            configured,
        }
    }

    /// Perform a web search via the Brave Search API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        if !self.configured {
            return Err(SearchError::ConfigError(
                "Brave Search API key not configured".to_string(),
            ));
        }

        let client = reqwest::Client::new();
        let url = format!("{}/res/v1/web/search", self.base_url);

        let mut params = vec![
            ("q", request.query.clone()),
            ("count", request.options.max_results.to_string()),
        ];

        if let Some(ref country) = request.options.country {
            params.push(("country", country.clone()));
        }

        if let Some(ref language) = request.options.language {
            params.push(("search_lang", language.clone()));
        }

        if request.options.safe_search {
            params.push(("safesearch", "strict".to_string()));
        }

        if let Some(ref time_range) = request.options.time_range {
            params.push(("freshness", time_range.clone()));
        }

        let start = std::time::Instant::now();

        let response = client
            .get(&url)
            .header("Accept", "application/json")
            .header("Accept-Encoding", "gzip")
            .header("X-Subscription-Token", &self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| SearchError::NetworkError(format!("HTTP request failed: {e}")))?;

        let elapsed = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return match status.as_u16() {
                401 => Err(SearchError::AuthError(
                    "Invalid Brave Search API key".to_string(),
                )),
                429 => Err(SearchError::RateLimited(
                    "Brave Search rate limit exceeded".to_string(),
                )),
                _ => Err(SearchError::NetworkError(format!(
                    "Brave Search returned {status}: {body}"
                ))),
            };
        }

        let brave_response: BraveWebResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        let mut results = Vec::new();

        if let Some(web) = brave_response.web {
            for result in web.results {
                results.push(SearchResult {
                    title: result.title,
                    url: result.url,
                    snippet: result.description,
                    source: "brave".to_string(),
                    score: Some(result.age.or(Some(0.0)).unwrap_or(0.0)),
                    published_date: None,
                    is_news: false,
                    is_video: false,
                });
            }
        }

        if let Some(news) = brave_response.news {
            for result in news.results {
                results.push(SearchResult {
                    title: result.title,
                    url: result.url,
                    snippet: result.description,
                    source: "brave".to_string(),
                    score: None,
                    published_date: Some(result.age),
                    is_news: true,
                    is_video: false,
                });
            }
        }

        let total = Some(results.len() as u64);
        let has_more = results.len() >= request.options.max_results;

        debug!(
            "Brave Search returned {} results in {elapsed}ms",
            results.len()
        );

        Ok(SearchResponse {
            results,
            total_results: total,
            has_more,
            provider: "brave".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for BraveSearch {
    fn search(
        &self,
        request: &SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + '_>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "brave"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

// Brave API response types

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveWebResponse {
    #[serde(default)]
    web: Option<BraveWebSection>,
    #[serde(default)]
    news: Option<BraveNewsSection>,
    #[serde(default)]
    videos: Option<BraveVideoSection>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveWebSection {
    results: Vec<BraveWebResult>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveWebResult {
    title: String,
    url: String,
    description: String,
    #[serde(default)]
    age: Option<f64>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveNewsSection {
    results: Vec<BraveNewsResult>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveNewsResult {
    title: String,
    url: String,
    description: String,
    age: String,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveVideoSection {
    results: Vec<BraveVideoResult>,
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BraveVideoResult {
    title: String,
    url: String,
    description: String,
}
