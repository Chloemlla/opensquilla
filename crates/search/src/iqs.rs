use opensquilla_core::config::Config;
use tracing::{debug, info, warn};

use crate::types::{SearchError, SearchProvider, SearchRequest, SearchResponse, SearchResult};

/// Alibaba Cloud IQS unified search API adapter.
///
/// IQS is Alibaba Cloud's unified search endpoint. It requires an API key
/// (`IQS_SEARCH_API_KEY` / `search.iqs.api_key`) and supports freshness
/// filtering via the `LiteAdvanced` engine.
pub struct IqsSearch {
    api_key: String,
    base_url: String,
    configured: bool,
}

/// The IQS unified endpoint rejects queries longer than 500 characters outright.
const QUERY_MAX_CHARS: usize = 500;

impl IqsSearch {
    /// Create a new IQS search adapter.
    pub fn new(config: &Config) -> Self {
        let api_key = config.get("search.iqs.api_key").unwrap_or_default();
        let base_url = config
            .get("search.iqs.base_url")
            .unwrap_or_else(|| "https://cloud-iqs.aliyuncs.com".to_string());
        let configured = !api_key.is_empty();

        if configured {
            info!("IQS Search configured");
        } else {
            warn!("IQS Search not configured (no API key)");
        }

        Self {
            api_key,
            base_url,
            configured,
        }
    }

    /// Perform a search via the IQS unified search API.
    pub async fn search_web(&self, request: &SearchRequest) -> Result<SearchResponse, SearchError> {
        if !self.configured {
            return Err(SearchError::ConfigError(
                "IQS API key not configured".to_string(),
            ));
        }

        let client = reqwest::Client::new();
        let url = format!("{}/search/unified", self.base_url);

        // Truncate overlong queries up front; the endpoint rejects them.
        let query: String = request.query.chars().take(QUERY_MAX_CHARS).collect();

        let mut body = serde_json::json!({
            "query": query,
            "engineType": "LiteAdvanced",
            "contents": {
                "mainText": true,
                "rerankScore": true,
            },
            "advancedParams": {
                "numResults": request.options.max_results.clamp(1, 20),
            },
        });

        if let Some(ref time_range) = request.options.time_range {
            body["timeRange"] = serde_json::Value::String(time_range_from_value(time_range));
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
            let detail = error_detail(&body_text);
            let message = format!("IQS search failed with HTTP {status}{detail}");
            return match status.as_u16() {
                // IQS reports bad credentials as 403 (Retrieval.InvalidAPIKey)
                // and, on the SDK-compatible path, as 404.
                401 | 403 | 404 => Err(SearchError::AuthError(message)),
                429 => Err(SearchError::RateLimited(message)),
                _ => Err(SearchError::NetworkError(message)),
            };
        }

        let iqs_response: IqsResponse = response
            .json()
            .await
            .map_err(|e| SearchError::ParseError(format!("Failed to parse response: {e}")))?;

        let results: Vec<SearchResult> = iqs_response
            .page_items
            .into_iter()
            .map(|item| SearchResult {
                title: item.title,
                url: item.link,
                snippet: if item.snippet.is_empty() {
                    item.main_text
                } else {
                    item.snippet
                },
                source: "iqs".to_string(),
                score: item.rerank_score,
                published_date: item.published_time.or(item.published_at),
                is_news: false,
                is_video: false,
            })
            .collect();

        let total = results.len() as u64;

        debug!("IQS Search returned {total} results in {elapsed}ms");

        Ok(SearchResponse {
            results,
            total_results: Some(total),
            has_more: false,
            provider: "iqs".to_string(),
            elapsed_ms: elapsed,
        })
    }
}

impl SearchProvider for IqsSearch {
    fn search<'a>(
        &'a self,
        request: &'a SearchRequest,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<SearchResponse, SearchError>> + Send + 'a>,
    > {
        Box::pin(self.search_web(request))
    }

    fn name(&self) -> &str {
        "iqs"
    }

    fn is_ready(&self) -> bool {
        self.configured
    }
}

/// Map a Rust `SearchOptions.time_range` value to IQS's `timeRange` codes.
fn time_range_from_value(time_range: &str) -> String {
    match time_range {
        "day" => "OneDay",
        "week" => "OneWeek",
        "month" => "OneMonth",
        "year" => "OneYear",
        other => other,
    }
    .to_string()
}

/// Extract a short human-readable detail from an IQS error body.
///
/// Error bodies are usually `{"errorCode", "errorMessage"}` JSON, but some 400s
/// return a bare server stack trace as text.
fn error_detail(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(obj) = value.as_object() {
            let code = obj
                .get("errorCode")
                .or_else(|| obj.get("code"))
                .and_then(|v| v.as_str());
            let message = obj
                .get("errorMessage")
                .or_else(|| obj.get("message"))
                .and_then(|v| v.as_str());
            return match (code, message) {
                (Some(c), Some(m)) => format!(" ({c}: {m})"),
                (Some(c), None) => format!(" ({c})"),
                (None, Some(m)) => format!(" ({m})"),
                _ => String::new(),
            };
        }
    }

    // Non-JSON body: use a truncated snippet.
    let snippet: String = trimmed.chars().take(200).collect();
    format!(" ({snippet})")
}

// IQS API response types

#[derive(serde::Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
struct IqsResponse {
    #[allow(dead_code)]
    #[serde(default)]
    request_id: Option<String>,
    #[serde(default)]
    page_items: Vec<IqsItem>,
}

#[derive(serde::Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
struct IqsItem {
    #[serde(default)]
    title: String,
    #[serde(default)]
    link: String,
    #[serde(default)]
    snippet: String,
    #[serde(default)]
    main_text: String,
    #[serde(default)]
    published_time: Option<String>,
    #[serde(rename = "published_at", default)]
    published_at: Option<String>,
    #[serde(default)]
    rerank_score: Option<f64>,
}
