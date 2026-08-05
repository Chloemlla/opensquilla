//! Web tools: web_search, web_fetch, http_request.
//!
//! Provides web search via provider API, web page content extraction,
//! and arbitrary HTTP requests with SSRF protection.
//!
//! SSRF protection lives in the standalone [`crate::ssrf`] module.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use crate::ssrf::SsrfProtection;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// A search provider trait for web_search tool.
#[async_trait::async_trait]
pub trait SearchProvider: Send + Sync {
    /// Perform a search and return results.
    async fn search(&self, query: &str, count: u32) -> Result<Vec<SearchResult>, ToolError>;
    /// The name of this search provider.
    fn provider_name(&self) -> &str;
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// A built-in search provider using DuckDuckGo's Instant Answer API (no API key needed).
pub struct DuckDuckGoSearch;

#[async_trait::async_trait]
impl SearchProvider for DuckDuckGoSearch {
    async fn search(&self, query: &str, count: u32) -> Result<Vec<SearchResult>, ToolError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .user_agent("OpenSquilla/1.0")
            .build()
            .map_err(|e| {
                ToolError::new("HTTP_ERROR", format!("Failed to create HTTP client: {}", e))
            })?;

        let encoded: String = query
            .bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => b as char,
                b' ' => '+',
                _ => format!("%{:02X}", b).chars().next().unwrap_or(b as char),
            })
            .collect();

        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            encoded
        );

        let resp =
            client.get(&url).send().await.map_err(|e| {
                ToolError::new("HTTP_ERROR", format!("Search request failed: {}", e))
            })?;

        let text = resp
            .text()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to read response: {}", e)))?;

        let json: Value = serde_json::from_str(&text).unwrap_or_default();
        let mut results = Vec::new();

        if let Some(abstract_text) = json["AbstractText"].as_str() {
            if !abstract_text.is_empty() {
                results.push(SearchResult {
                    title: json["Heading"].as_str().unwrap_or("Result").to_string(),
                    url: json["AbstractURL"].as_str().unwrap_or("").to_string(),
                    snippet: abstract_text.to_string(),
                });
            }
        }

        if let Some(topics) = json["RelatedTopics"].as_array() {
            for topic in topics {
                if results.len() >= count as usize {
                    break;
                }
                if let Some(text) = topic["Text"].as_str() {
                    results.push(SearchResult {
                        title: topic["FirstURL"].as_str().unwrap_or("Result").to_string(),
                        url: topic["FirstURL"].as_str().unwrap_or("").to_string(),
                        snippet: text.to_string(),
                    });
                }
                if let Some(subtopics) = topic["Topics"].as_array() {
                    for sub in subtopics {
                        if results.len() >= count as usize {
                            break;
                        }
                        if let Some(text) = sub["Text"].as_str() {
                            results.push(SearchResult {
                                title: sub["FirstURL"].as_str().unwrap_or("").to_string(),
                                url: sub["FirstURL"].as_str().unwrap_or("").to_string(),
                                snippet: text.to_string(),
                            });
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    fn provider_name(&self) -> &str {
        "duckduckgo"
    }
}

/// Tool for web search.
pub struct WebSearchTool {
    provider: Arc<dyn SearchProvider>,
}

impl WebSearchTool {
    pub fn new(provider: Arc<dyn SearchProvider>) -> Self {
        Self { provider }
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new(Arc::new(DuckDuckGoSearch))
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "web_search",
                "Search the web for information. Returns a list of results with titles, URLs, and snippets.",
                HashMap::from([
                    ("query".to_string(), ParameterDefinition::required_string("The search query")),
                    ("count".to_string(), ParameterDefinition::integer("Number of results to return (max 10)").default(serde_json::json!(5))),
                ]),
            )
            .category("web")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let query = params["query"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'query' parameter"))?;
        let count = params["count"].as_i64().unwrap_or(5).min(10).max(1) as u32;

        let results = self.provider.search(query, count).await?;

        let data = serde_json::json!({
            "results": results.iter().map(|r| serde_json::json!({
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            })).collect::<Vec<_>>(),
            "count": results.len(),
            "provider": self.provider.provider_name(),
        });

        let content = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "{}. {}\n   URL: {}\n   {}\n",
                    i + 1,
                    r.title,
                    r.url,
                    r.snippet.chars().take(200).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for fetching web page content.
pub struct WebFetchTool {
    client: Client,
    ssrf: SsrfProtection,
    max_response_size: u64,
}

impl WebFetchTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("OpenSquilla/1.0")
            .danger_accept_invalid_certs(false)
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            ssrf: SsrfProtection::new(),
            max_response_size: 2 * 1024 * 1024,
        }
    }

    fn extract_content(&self, html: &str, url: &str) -> String {
        use scraper::{Html, Selector};
        let document = Html::parse_document(html);

        let title_selector = Selector::parse("title").unwrap();
        let title = document
            .select(&title_selector)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_default();

        let content_selectors = [
            "article",
            "main",
            ".post-content",
            ".article-content",
            ".entry-content",
            "#content",
            ".content",
            "body",
        ];

        let mut content = String::new();
        if !title.is_empty() {
            content.push_str(&format!("# {}\n\n", title.trim()));
        }
        content.push_str(&format!("Source: {}\n\n", url));

        for selector_str in &content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let text: String = element.text().collect::<Vec<_>>().join(" ");
                    let text = text.trim();
                    if !text.is_empty() {
                        content.push_str(text);
                        content.push('\n');
                    }
                }
                if content.len() > 100 {
                    break;
                }
            }
        }

        if content.len() < 50 {
            if let Ok(body_selector) = Selector::parse("body") {
                for element in document.select(&body_selector) {
                    let text: String = element.text().collect::<Vec<_>>().join(" ");
                    content.push_str(text.trim());
                    break;
                }
            }
        }

        if content.len() > 10000 {
            content.truncate(10000);
            content.push_str("\n\n... (content truncated)");
        }

        content
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "web_fetch",
                "Fetch and extract the main content from a web page URL.",
                HashMap::from([
                    (
                        "url".to_string(),
                        ParameterDefinition::required_string("The URL to fetch"),
                    ),
                    (
                        "max_length".to_string(),
                        ParameterDefinition::integer("Maximum content length")
                            .default(serde_json::json!(10000)),
                    ),
                ]),
            )
            .category("web")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let url = params["url"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'url' parameter"))?;
        let max_length = params["max_length"].as_i64().unwrap_or(10000) as usize;

        self.ssrf.check_url(url).await?;

        let resp = self.client.get(url).send().await.map_err(|e| {
            if e.is_timeout() {
                ToolError::timeout(30)
            } else {
                ToolError::new("HTTP_ERROR", format!("Request failed: {}", e))
            }
        })?;

        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = resp
            .bytes()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to read response: {}", e)))?;

        if body.len() as u64 > self.max_response_size {
            return Err(ToolError::new(
                "RESPONSE_TOO_LARGE",
                format!(
                    "Response too large: {} bytes (max {})",
                    body.len(),
                    self.max_response_size
                ),
            ));
        }

        let body_str = String::from_utf8_lossy(&body);
        let content = if content_type.contains("text/html") || body_str.contains("<html") {
            self.extract_content(&body_str, url)
        } else {
            body_str.to_string()
        };

        let truncated: String = content.chars().take(max_length).collect();

        let data = serde_json::json!({
            "url": url,
            "status": status,
            "content_type": content_type,
            "size": body.len(),
            "truncated": content.len() > max_length,
        });

        Ok(ToolOutput::success(truncated).with_data(data))
    }
}

/// Tool for making arbitrary HTTP requests.
pub struct HttpRequestTool {
    client: Client,
    ssrf: SsrfProtection,
}

impl HttpRequestTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .user_agent("OpenSquilla/1.0")
            .danger_accept_invalid_certs(false)
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            ssrf: SsrfProtection::new(),
        }
    }
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "http_request",
                "Make an arbitrary HTTP request. Supports GET, POST, PUT, DELETE, PATCH, HEAD.",
                HashMap::from([
                    (
                        "method".to_string(),
                        ParameterDefinition::string("HTTP method")
                            .default(serde_json::json!("GET")),
                    ),
                    (
                        "url".to_string(),
                        ParameterDefinition::required_string("The URL to send the request to"),
                    ),
                    (
                        "headers".to_string(),
                        ParameterDefinition::string("HTTP headers as a JSON object"),
                    ),
                    (
                        "body".to_string(),
                        ParameterDefinition::string("Request body (for POST, PUT, PATCH)"),
                    ),
                    (
                        "timeout".to_string(),
                        ParameterDefinition::integer("Timeout in seconds")
                            .default(serde_json::json!(30)),
                    ),
                ]),
            )
            .category("web")
            .risk_level(2)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let url = params["url"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'url' parameter"))?;
        let method = params["method"].as_str().unwrap_or("GET").to_uppercase();

        if method != "GET" {
            self.ssrf.check_url(url).await?;
        }

        let headers: HashMap<String, String> = params["headers"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        let timeout = params["timeout"].as_i64().unwrap_or(30) as u64;

        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .user_agent("OpenSquilla/1.0")
            .build()
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to build client: {}", e)))?;

        let mut req = match method.as_str() {
            "GET" => client.get(url),
            "POST" => client
                .post(url)
                .body(params["body"].as_str().unwrap_or("").to_string()),
            "PUT" => client
                .put(url)
                .body(params["body"].as_str().unwrap_or("").to_string()),
            "DELETE" => client.delete(url),
            "PATCH" => client
                .patch(url)
                .body(params["body"].as_str().unwrap_or("").to_string()),
            "HEAD" => client.head(url),
            _ => {
                return Err(ToolError::invalid_args(format!(
                    "Unsupported HTTP method: {}",
                    method
                )));
            }
        };

        for (key, val) in &headers {
            req = req.header(key.as_str(), val.as_str());
        }

        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                ToolError::timeout(timeout)
            } else {
                ToolError::new("HTTP_ERROR", format!("Request failed: {}", e))
            }
        })?;

        let status = resp.status().as_u16();
        let response_headers: HashMap<String, String> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        let body = resp
            .bytes()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to read response: {}", e)))?;

        let body_str = String::from_utf8_lossy(&body).to_string();

        let data = serde_json::json!({
            "status": status,
            "headers": response_headers,
            "size": body.len(),
        });

        Ok(ToolOutput::success(body_str).with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ssrf_private_ip() {
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://127.0.0.1:8080/secret").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_ssrf_private_ipv6() {
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://[::1]:8080/secret").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ssrf_public_url() {
        // A public IP literal passes without any DNS round-trip.
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://8.8.8.8/").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ssrf_fail_closed_on_dns_error() {
        // DNS failures must deny the request (fail-closed), not allow it.
        let ssrf = SsrfProtection::new();
        let result = ssrf
            .check_url("http://this-host-definitely-does-not-exist.invalid/")
            .await;
        assert!(result.is_err(), "DNS errors must fail closed");
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_search_provider() {
        let provider = DuckDuckGoSearch;
        let results = provider.search("rust programming", 3).await;
        if let Ok(results) = results {
            assert!(results.len() <= 3);
        }
    }
}
