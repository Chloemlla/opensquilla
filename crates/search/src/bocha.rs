use opensquilla_core::config::Config;
use tracing::{debug, info, warn};

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// Bocha Web Search API adapter.
///
/// Bocha is a web search engine that returns inline summaries alongside
/// results. It requires an API key (`BOCHA_SEARCH_API_KEY` / `search.bocha.api_key`)
/// and supports freshness filtering.
pub struct BochaSearch {
    api_key: String,
    base_url: String,
    configured: bool,
}

impl BochaSearch {
    /// Create a new Bocha search adapter.
    pub fn new(config: &Config) -> Self {
        let api_key = config.get("search.bocha.api_key").unwrap_or_default();
        let base_url = config
            .get("search.bocha.base_url")
            .unwrap_or_else(|| "https://api.bochaai.com".to_string());
        let configured = !api_key.is_empty();

        if configured {
            info!("Bocha Search configured");
        } else {
            warn!("Bocha Search not configured (no API key)");
        }

        Self {
            api_key,
            base_url,
            configured,
        }
    }

    /// Perform a search via the Bocha Web Search API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        if !self.configured {
            return Err(SearchError::ConfigError(
                "Bocha API key not configured".to_string(),
            ));
        }

        let client = reqwest::Client::new();
        let url = format!("{}/v1/web-search", self.base_url);

        let mut body = serde_json::json!({
            "query": request.query.clone(),
            "count": request.options.max_results.clamp(1, 20),
            "summary": true,
        });

        if let Some(ref time_range) = request.options.time_range {
            body["freshness"] = serde_json::Value::String(freshness_from_time_range(time_range));
        }

        let start = std::time::Instant::now();

        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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
                401 | 403 => Err(SearchError::AuthError("Invalid Bocha API key".to_string())),
                429 => Err(SearchError::RateLimited(
                    "Bocha rate limit exceeded".to_string(),
                )),
                _ => Err(SearchError::NetworkError(format!(
                    "Bocha returned {status}: {body_text}"
                ))),
            };
        }

        let bocha_response: BochaResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        // Bocha reports API-level errors in the body with a `code`/`msg` pair,
        // even on HTTP 200. A numeric code other than 200 is an error.
        if let Some(ref code) = bocha_response.code {
            match api_code_to_i64(code) {
                Some(200) => {}
                Some(code_int) => {
                    let message = bocha_response
                        .msg
                        .clone()
                        .unwrap_or_else(|| "Bocha search request failed".to_string());
                    return Err(match code_int {
                        401 | 403 => SearchError::AuthError(message),
                        429 => SearchError::RateLimited(message),
                        _ => SearchError::NetworkError(message),
                    });
                }
                None => {
                    // Non-numeric API error code, e.g. "INVALID_PARAMETER".
                    let message = bocha_response
                        .msg
                        .clone()
                        .unwrap_or_else(|| "Bocha search request failed".to_string());
                    return Err(SearchError::NetworkError(message));
                }
            }
        }

        let items = bocha_response
            .data
            .as_ref()
            .map(items_from_data)
            .unwrap_or_default();

        let results: Vec<SearchResult> = items
            .into_iter()
            .map(|item| SearchResult {
                title: item.name.or(item.title).unwrap_or_default(),
                url: item.url.or(item.id).unwrap_or_default(),
                snippet: item.snippet.or(item.description).unwrap_or_default(),
                source: "bocha".to_string(),
                score: None,
                published_date: item.date_published.or(item.published_at),
                is_news: false,
                is_video: false,
            })
            .collect();

        let total = results.len() as u64;

        debug!("Bocha Search returned {total} results in {elapsed}ms");

        Ok(SearchResponse {
            results,
            total_results: Some(total),
            has_more: false,
            provider: "bocha".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for BochaSearch {
    fn search<'a>(
        &'a self,
        request: &'a SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + 'a>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "bocha"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

/// Map a Rust `SearchOptions.time_range` value to Bocha's `freshness` codes.
fn freshness_from_time_range(time_range: &str) -> String {
    match time_range {
        "day" => "oneDay",
        "week" => "oneWeek",
        "month" => "oneMonth",
        "year" => "oneYear",
        other => other,
    }
    .to_string()
}

/// Normalize Bocha's `code` field (number or numeric string) to an i64.
fn api_code_to_i64(code: &serde_json::Value) -> Option<i64> {
    match code {
        serde_json::Value::Number(n) => n.as_i64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Extract result items from the response payload, honoring Bocha's three
/// possible nesting shapes: `data.webPages.value`, `data.results`, `data.value`.
fn items_from_data(data: &BochaData) -> Vec<BochaItem> {
    if let Some(web_pages) = &data.web_pages {
        if let Some(items) = &web_pages.value {
            if !items.is_empty() {
                return items.clone();
            }
        }
    }
    if let Some(items) = &data.results {
        if !items.is_empty() {
            return items.clone();
        }
    }
    data.value.clone().unwrap_or_default()
}

// Bocha API response types

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct BochaResponse {
    #[serde(default)]
    code: Option<serde_json::Value>,
    #[serde(default)]
    msg: Option<String>,
    #[serde(rename = "log_id", default)]
    log_id: Option<String>,
    #[serde(default)]
    data: Option<BochaData>,
}

#[derive(serde::Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
struct BochaData {
    #[serde(default)]
    web_pages: Option<BochaWebPages>,
    #[serde(default)]
    results: Option<Vec<BochaItem>>,
    #[serde(default)]
    value: Option<Vec<BochaItem>>,
}

#[derive(serde::Deserialize, Debug, Default)]
struct BochaWebPages {
    #[serde(default)]
    value: Option<Vec<BochaItem>>,
}

#[derive(serde::Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
struct BochaItem {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    snippet: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    date_published: Option<String>,
    #[serde(rename = "published_at", default)]
    published_at: Option<String>,
}
