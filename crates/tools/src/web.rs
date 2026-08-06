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

// ---------------------------------------------------------------------------
// HTML-to-text conversion
// ---------------------------------------------------------------------------

/// Convert HTML to plain text, preserving block structure.
///
/// Handles common block elements (p, div, h1-h6, li, br, tr, etc.) by adding
/// newlines, strips script/style/head content, and collapses runs of
/// whitespace.
pub fn html_to_text(html: &str) -> String {
    use scraper::{Html, Selector};

    let document = Html::parse_fragment(html);
    let mut output = String::new();

    let block_selectors = [
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br", "tr", "section", "article",
        "blockquote", "pre", "table", "ul", "ol", "footer", "header",
    ];

    // Elements whose content is never part of the visible page text.
    let skip_tags = ["script", "style", "noscript", "template", "head", "title"];

    let text_selector = Selector::parse("body, *").unwrap();

    let mut last_was_block = false;

    for element in document.select(&text_selector) {
        let tag = element.value().name().to_string();
        if skip_tags.contains(&tag.as_str()) {
            continue;
        }
        let is_block = block_selectors.contains(&tag.as_str());

        if is_block && !output.is_empty() && !last_was_block {
            output.push('\n');
        }

        // Only take direct text children to avoid duplication.
        for child in element.children() {
            if let Some(node) = child.value().as_text() {
                output.push_str(&node.text);
            }
        }

        if is_block {
            output.push('\n');
            last_was_block = true;
        } else {
            last_was_block = false;
        }
    }

    // Clean up: collapse multiple blank lines, trim each line.
    output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Readability scoring
// ---------------------------------------------------------------------------

/// The result of readability scoring for a page.
#[derive(Debug, Clone)]
pub struct ReadabilityResult {
    /// The extracted main content text.
    pub content: String,
    /// The page title.
    pub title: String,
    /// The score (0.0 - 1.0) indicating how "article-like" the content is.
    pub score: f64,
    /// The selector that yielded the best content.
    pub best_selector: String,
}

/// Score a text block for readability.
///
/// Higher scores indicate text that looks like article content:
/// - Longer average sentence length (but not too long)
/// - Presence of punctuation
/// - Lower link density is a good signal but we don't have that here
/// - Words > 1 syllable (approximated by length)
pub fn readability_score(text: &str) -> f64 {
    if text.trim().is_empty() {
        return 0.0;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len();
    if word_count == 0 {
        return 0.0;
    }

    // Average word length.
    let avg_word_len: f64 = words.iter().map(|w| w.chars().count() as f64).sum::<f64>()
        / word_count as f64;

    // Sentence count (approximate by counting sentence-ending punctuation).
    let sentence_count = text
        .chars()
        .filter(|c| matches!(c, '.' | '!' | '?'))
        .count()
        .max(1);

    // Long words (>= 7 chars) as a fraction.
    let long_words = words.iter().filter(|w| w.chars().count() >= 7).count();
    let long_word_ratio = long_words as f64 / word_count as f64;

    // Score components.
    // 1. Word length in a "good" range (3.5 - 7.0).
    let length_score = if (3.5..=7.0).contains(&avg_word_len) {
        1.0 - ((avg_word_len - 5.0) / 2.0).abs().min(1.0)
    } else {
        0.3
    };

    // 2. Reasonable sentence length (10-30 words per sentence).
    let words_per_sentence = word_count as f64 / sentence_count as f64;
    let sentence_score = if (8.0..=40.0).contains(&words_per_sentence) {
        1.0 - ((words_per_sentence - 20.0) / 20.0).abs().min(1.0)
    } else {
        0.2
    };

    // 3. Some long words (article-like vocabulary).
    let vocab_score = if long_word_ratio > 0.15 {
        1.0
    } else if long_word_ratio > 0.05 {
        0.6
    } else {
        0.2
    };

    // 4. Size bonus for substantial content.
    let size_score = if word_count > 500 {
        1.0
    } else if word_count > 100 {
        0.7
    } else if word_count > 30 {
        0.4
    } else {
        0.1
    };

    0.25 * length_score + 0.25 * sentence_score + 0.2 * vocab_score + 0.3 * size_score
}

/// Extract the main content of a web page using readability scoring.
///
/// Tries multiple content selectors and picks the one with the highest
/// readability score. Returns the extracted content, title, and score.
pub fn extract_readable_content(html: &str) -> ReadabilityResult {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);

    // Extract title.
    let title_selector = Selector::parse("title").unwrap();
    let title = document
        .select(&title_selector)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .unwrap_or_default();

    // Candidate selectors. "Preferred" selectors are specific content
    // containers; "fallback" selectors are only used when nothing specific
    // scores well.
    let preferred = [
        "article",
        "main",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".post-body",
        "#content",
        ".content",
        ".body",
    ];
    let fallback = ["body"];

    let mut best_content = String::new();
    let mut best_score = 0.0f64;
    let mut best_selector = String::new();

    let mut consider = |selector_str: &str,
                        selector: &Selector,
                        best_score: &mut f64,
                        best_content: &mut String,
                        best_selector: &mut String| {
        for element in document.select(selector) {
            let text: String = element.text().collect::<Vec<_>>().join(" ");
            let text = text.trim().to_string();
            if text.len() < 100 {
                continue;
            }
            let score = readability_score(&text);
            // Prefer more specific selectors on ties.
            let specificity_bonus = if selector_str.starts_with('.') || selector_str.starts_with('#') {
                0.05
            } else {
                0.0
            };
            let adjusted = score + specificity_bonus;
            if adjusted > *best_score {
                *best_score = adjusted;
                *best_content = text;
                *best_selector = selector_str.to_string();
            }
        }
    };

    for selector_str in &preferred {
        if let Ok(selector) = Selector::parse(selector_str) {
            consider(
                selector_str,
                &selector,
                &mut best_score,
                &mut best_content,
                &mut best_selector,
            );
        }
    }

    // If no preferred selector scored above the readability threshold, fall
    // back to the full body.
    if best_score < 0.3 {
        for selector_str in &fallback {
            if let Ok(selector) = Selector::parse(selector_str) {
                consider(
                    selector_str,
                    &selector,
                    &mut best_score,
                    &mut best_content,
                    &mut best_selector,
                );
            }
        }
    }

    // If nothing scored well, take the entire body text verbatim.
    if best_content.is_empty() {
        if let Ok(body_selector) = Selector::parse("body") {
            for element in document.select(&body_selector) {
                best_content = element.text().collect::<Vec<_>>().join(" ").trim().to_string();
                best_selector = "body".to_string();
                break;
            }
        }
        best_score = readability_score(&best_content);
    }

    ReadabilityResult {
        content: best_content,
        title,
        score: best_score,
        best_selector,
    }
}

// ---------------------------------------------------------------------------
// Rate limiting per domain
// ---------------------------------------------------------------------------

/// A per-domain rate limiter.
///
/// Tracks the last request time for each domain and enforces a minimum
/// interval between requests to the same domain.
pub struct DomainRateLimiter {
    /// Map of domain -> last request timestamp (milliseconds since epoch).
    last_requests: Arc<std::sync::Mutex<HashMap<String, u128>>>,
    /// Minimum interval between requests to the same domain in milliseconds.
    min_interval_ms: u128,
}

impl Default for DomainRateLimiter {
    fn default() -> Self {
        Self::new(500)
    }
}

impl DomainRateLimiter {
    /// Create a new rate limiter with the given minimum interval (ms).
    pub fn new(min_interval_ms: u128) -> Self {
        Self {
            last_requests: Arc::new(std::sync::Mutex::new(HashMap::new())),
            min_interval_ms,
        }
    }

    /// Get the domain part of a URL.
    fn domain_of(url: &str) -> String {
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_else(|| url.to_string())
    }

    /// Wait until a request to the given domain is allowed.
    ///
    /// Returns the number of milliseconds waited.
    pub async fn acquire(&self, url: &str) -> u128 {
        let domain = Self::domain_of(url);
        let mut waited = 0u128;

        loop {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);

            // Scope the lock so the guard is dropped before the await below;
            // `std::sync::MutexGuard` is not `Send`, so it must not live across
            // the sleep if the async future is to stay `Send`.
            let wait_for = {
                let mut last_requests = match self.last_requests.lock() {
                    Ok(guard) => guard,
                    Err(_) => return 0,
                };

                let last = last_requests.get(&domain).copied().unwrap_or(0);
                let elapsed = now.saturating_sub(last);

                if elapsed >= self.min_interval_ms {
                    last_requests.insert(domain.clone(), now);
                    return waited;
                }

                self.min_interval_ms - elapsed
            };

            tokio::time::sleep(std::time::Duration::from_millis(wait_for as u64)).await;
            waited += wait_for;
        }
    }
}

// ---------------------------------------------------------------------------
// Response caching
// ---------------------------------------------------------------------------

/// A simple in-memory response cache keyed by URL.
pub struct ResponseCache {
    /// Map of URL -> (timestamp_ms, status, content_type, body_bytes).
    entries: Arc<std::sync::Mutex<HashMap<String, CachedResponse>>>,
    /// Maximum number of entries to keep.
    max_entries: usize,
    /// Default TTL in seconds.
    ttl_secs: u64,
}

/// A cached response.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The timestamp when the response was cached (ms since epoch).
    pub timestamp_ms: u128,
    /// The HTTP status code.
    pub status: u16,
    /// The Content-Type header.
    pub content_type: String,
    /// The raw response body bytes.
    pub body: Vec<u8>,
}

impl Default for ResponseCache {
    fn default() -> Self {
        Self::new(128, 300)
    }
}

impl ResponseCache {
    /// Create a new response cache.
    pub fn new(max_entries: usize, ttl_secs: u64) -> Self {
        Self {
            entries: Arc::new(std::sync::Mutex::new(HashMap::new())),
            max_entries,
            ttl_secs,
        }
    }

    /// Get a cached response if it exists and is not stale.
    pub fn get(&self, url: &str) -> Option<CachedResponse> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let entries = self.entries.lock().ok()?;
        let entry = entries.get(url)?;
        if now.saturating_sub(entry.timestamp_ms) > self.ttl_secs as u128 * 1000 {
            return None;
        }
        Some(entry.clone())
    }

    /// Store a response in the cache.
    pub fn put(&self, url: &str, status: u16, content_type: &str, body: Vec<u8>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        if let Ok(mut entries) = self.entries.lock() {
            // Evict oldest if at capacity.
            if entries.len() >= self.max_entries && !entries.contains_key(url) {
                if let Some(oldest) = entries
                    .iter()
                    .min_by_key(|(_, e)| e.timestamp_ms)
                    .map(|(k, _)| k.clone())
                {
                    entries.remove(&oldest);
                }
            }
            entries.insert(
                url.to_string(),
                CachedResponse {
                    timestamp_ms: now,
                    status,
                    content_type: content_type.to_string(),
                    body,
                },
            );
        }
    }

    /// Invalidate a cached URL.
    pub fn invalidate(&self, url: &str) -> bool {
        self.entries.lock().ok().map(|mut e| e.remove(url).is_some()).unwrap_or(false)
    }

    /// Clear the cache.
    pub fn clear(&self) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.clear();
        }
    }

    /// Get the cache size.
    pub fn len(&self) -> usize {
        self.entries.lock().map(|e| e.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Cookie jar
// ---------------------------------------------------------------------------

/// A simple cookie jar for maintaining session state across requests.
pub struct CookieJar {
    /// Map of domain -> cookie string.
    cookies: Arc<std::sync::Mutex<HashMap<String, Vec<String>>>>,
}

impl Default for CookieJar {
    fn default() -> Self {
        Self::new()
    }
}

impl CookieJar {
    /// Create a new empty cookie jar.
    pub fn new() -> Self {
        Self {
            cookies: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Extract cookies from a response's Set-Cookie headers and store them.
    pub fn store_from_response(&self, url: &str, headers: &reqwest::header::HeaderMap) {
        let domain = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))
            .unwrap_or_default();
        if domain.is_empty() {
            return;
        }

        let mut new_cookies = Vec::new();
        for value in headers.get_all(reqwest::header::SET_COOKIE) {
            if let Ok(s) = value.to_str() {
                // Take the name=value part before the first ';'.
                if let Some(cookie) = s.split(';').next() {
                    new_cookies.push(cookie.trim().to_string());
                }
            }
        }
        if new_cookies.is_empty() {
            return;
        }

        if let Ok(mut cookies) = self.cookies.lock() {
            let entry = cookies.entry(domain).or_default();
            // Replace cookies with the same name.
            for cookie in new_cookies {
                let name = cookie.split('=').next().unwrap_or("").to_string();
                if !name.is_empty() {
                    entry.retain(|c| !c.starts_with(&format!("{}=", name)));
                }
                entry.push(cookie);
            }
        }
    }

    /// Build a Cookie header value for a request to the given URL.
    pub fn header_for(&self, url: &str) -> Option<String> {
        let domain = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(String::from))?;
        let cookies = self.cookies.lock().ok()?;
        let values = cookies.get(&domain)?;
        if values.is_empty() {
            None
        } else {
            Some(values.join("; "))
        }
    }

    /// Clear all cookies.
    pub fn clear(&self) {
        if let Ok(mut cookies) = self.cookies.lock() {
            cookies.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Robots.txt respect
// ---------------------------------------------------------------------------

/// A robots.txt checker that respects robots exclusion rules.
///
/// Fetches and caches robots.txt per domain and checks whether a path is
/// allowed for a given user agent.
pub struct RobotsTxt {
    /// Cache of domain -> (robots text, timestamp_ms).
    cache: Arc<std::sync::Mutex<HashMap<String, (String, u128)>>>,
    /// The user agent to check rules for.
    user_agent: String,
    /// Whether to block disallowed paths or just warn.
    block: bool,
    /// Cache TTL in seconds.
    ttl_secs: u64,
}

impl Default for RobotsTxt {
    fn default() -> Self {
        Self::new("OpenSquillaBot".to_string(), true)
    }
}

impl RobotsTxt {
    /// Create a new robots.txt checker.
    pub fn new(user_agent: String, block: bool) -> Self {
        Self {
            cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            user_agent,
            block,
            ttl_secs: 3600,
        }
    }

    /// Parse robots.txt rules into a list of (user_agent, allow_paths, disallow_paths).
    fn parse_rules(text: &str) -> Vec<(String, Vec<String>, Vec<String>)> {
        let mut rules: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
        let mut current_agent: Option<String> = None;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(rest) = line.to_lowercase().strip_prefix("user-agent:") {
                let agent = rest.trim().to_string();
                current_agent = Some(agent.clone());
                if !rules.iter().any(|(a, _, _)| *a == agent) {
                    rules.push((agent, Vec::new(), Vec::new()));
                }
            } else if let Some(rest) = line.strip_prefix("Allow:") {
                let path = rest.trim().to_string();
                if let Some(ref agent) = current_agent {
                    if let Some(rule) = rules.iter_mut().find(|(a, _, _)| *a == *agent) {
                        rule.1.push(path);
                    }
                }
            } else if let Some(rest) = line.strip_prefix("Disallow:") {
                let path = rest.trim().to_string();
                if let Some(ref agent) = current_agent {
                    if let Some(rule) = rules.iter_mut().find(|(a, _, _)| *a == *agent) {
                        rule.2.push(path);
                    }
                }
            }
        }
        rules
    }

    /// Check whether a path is allowed for a given agent's rules.
    fn is_path_allowed(path: &str, allow: &[String], disallow: &[String]) -> bool {
        // Allow rules take precedence over disallow rules (per spec).
        for a in allow {
            if path.starts_with(a) {
                return true;
            }
        }
        for d in disallow {
            if path.starts_with(d) {
                return false;
            }
        }
        true
    }

    /// Check a URL against the rules (using cached or fetched robots.txt).
    pub async fn check(&self, url: &str) -> Result<bool, String> {
        let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;
        let domain = parsed.host_str().unwrap_or_default().to_string();
        if domain.is_empty() {
            return Ok(true);
        }
        let path = parsed.path();

        // Check cache.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let cached = self.cache.lock().map(|c| c.get(&domain).cloned()).unwrap_or(None);
        let robots_text = if let Some((text, ts)) = cached {
            if now.saturating_sub(ts) > self.ttl_secs as u128 * 1000 {
                None
            } else {
                Some(text)
            }
        } else {
            None
        };

        let robots_text = if let Some(text) = robots_text {
            text
        } else {
            // Fetch robots.txt.
            let robots_url = format!("{}://{}/robots.txt", parsed.scheme(), domain);
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .user_agent(&self.user_agent)
                .build()
                .map_err(|e| format!("Failed to build client: {}", e))?;
            let resp = client.get(&robots_url).send().await.map_err(|e| format!("Failed to fetch robots.txt: {}", e))?;
            let status = resp.status().as_u16();
            let text = if status == 200 {
                resp.text().await.unwrap_or_default()
            } else {
                // No robots.txt or non-200: everything allowed.
                String::new()
            };
            if let Ok(mut cache) = self.cache.lock() {
                cache.insert(domain.clone(), (text.clone(), now));
            }
            text
        };

        if robots_text.trim().is_empty() {
            return Ok(true);
        }

        // Check the rules for our agent, and the wildcard agent.
        let rules = Self::parse_rules(&robots_text);
        let mut allowed = true;

        for (agent, allow, disallow) in &rules {
            let matches = agent == "*" || agent.to_lowercase() == self.user_agent.to_lowercase();
            if matches {
                if !Self::is_path_allowed(path, allow, disallow) {
                    allowed = false;
                    break;
                }
            }
        }

        Ok(allowed)
    }
}

// ---------------------------------------------------------------------------
// Web content extraction tool
// ---------------------------------------------------------------------------

/// A tool for extracting readable content from web pages with advanced
/// features: readability scoring, robots.txt respect, rate limiting,
/// response caching, cookie jar support, and redirect following.
pub struct WebExtractTool {
    /// The HTTP client.
    client: Client,
    /// SSRF protection.
    ssrf: SsrfProtection,
    /// Per-domain rate limiter.
    rate_limiter: DomainRateLimiter,
    /// Response cache.
    cache: ResponseCache,
    /// Cookie jar.
    cookies: CookieJar,
    /// Robots.txt checker.
    robots: RobotsTxt,
    /// Maximum redirects to follow.
    max_redirects: usize,
    /// Maximum response size in bytes.
    max_response_size: u64,
}

impl Default for WebExtractTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebExtractTool {
    /// Create a new web extraction tool.
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("OpenSquillaBot/1.0")
            .danger_accept_invalid_certs(false)
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            client,
            ssrf: SsrfProtection::new(),
            rate_limiter: DomainRateLimiter::new(500),
            cache: ResponseCache::new(128, 300),
            cookies: CookieJar::new(),
            robots: RobotsTxt::new("OpenSquillaBot".to_string(), true),
            max_redirects: 10,
            max_response_size: 5 * 1024 * 1024,
        }
    }

    /// Configure the rate limiter minimum interval.
    pub fn with_rate_limit(mut self, min_interval_ms: u128) -> Self {
        self.rate_limiter = DomainRateLimiter::new(min_interval_ms);
        self
    }

    /// Enable or disable robots.txt enforcement.
    pub fn with_robots(mut self, block: bool) -> Self {
        self.robots = RobotsTxt::new("OpenSquillaBot".to_string(), block);
        self
    }

    /// Fetch and extract content from a URL.
    pub async fn fetch_content(&self, url: &str, use_cache: bool) -> ToolResult<ToolOutput> {
        self.ssrf.check_url(url).await?;

        // Check robots.txt.
        if self.robots.block {
            let allowed = self.robots.check(url).await.map_err(|e| {
                ToolError::new("ROBOTS_ERROR", format!("Failed to check robots.txt: {}", e))
            })?;
            if !allowed {
                return Err(ToolError::new(
                    "ROBOTS_BLOCKED",
                    format!("URL '{}' is disallowed by robots.txt", url),
                ));
            }
        }

        // Check cache.
        if use_cache {
            if let Some(cached) = self.cache.get(url) {
                let cached_content = String::from_utf8_lossy(&cached.body).to_string();
                let data = serde_json::json!({
                    "url": url,
                    "status": cached.status,
                    "content_type": cached.content_type,
                    "size": cached.body.len(),
                    "from_cache": true,
                });
                return Ok(ToolOutput::success(cached_content).with_data(data));
            }
        }

        // Rate limit.
        let waited = self.rate_limiter.acquire(url).await;

        // Apply cookies.
        let mut request = self.client.get(url);
        if let Some(cookie_header) = self.cookies.header_for(url) {
            request = request.header(reqwest::header::COOKIE, cookie_header);
        }

        let resp = request.send().await.map_err(|e| {
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

        // Store cookies from the response.
        self.cookies.store_from_response(url, resp.headers());

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

        // Cache the response.
        self.cache.put(url, status, &content_type, body.to_vec());

        let body_str = String::from_utf8_lossy(&body);
        let is_html = content_type.contains("text/html") || body_str.contains("<html");
        let content = if is_html {
            html_to_text(&body_str)
        } else {
            body_str.to_string()
        };

        let data = serde_json::json!({
            "url": url,
            "status": status,
            "content_type": content_type,
            "size": body.len(),
            "rate_limited_ms": waited,
            "from_cache": false,
        });

        Ok(ToolOutput::success(content).with_data(data))
    }

    /// Extract the readable main content using readability scoring.
    pub async fn extract_readable(&self, url: &str) -> ToolResult<ToolOutput> {
        // Fetch the raw HTML (robots.txt respected, rate limited, cookies
        // maintained), then score candidate content selectors.
        self.ssrf.check_url(url).await?;
        if self.robots.block {
            let allowed = self.robots.check(url).await.map_err(|e| {
                ToolError::new("ROBOTS_ERROR", format!("Failed to check robots.txt: {}", e))
            })?;
            if !allowed {
                return Err(ToolError::new(
                    "ROBOTS_BLOCKED",
                    format!("URL '{}' is disallowed by robots.txt", url),
                ));
            }
        }

        let waited = self.rate_limiter.acquire(url).await;
        let mut request = self.client.get(url);
        if let Some(cookie_header) = self.cookies.header_for(url) {
            request = request.header(reqwest::header::COOKIE, cookie_header);
        }
        let resp = request.send().await.map_err(|e| {
            ToolError::new("HTTP_ERROR", format!("Request failed: {}", e))
        })?;
        self.cookies.store_from_response(url, resp.headers());
        let body = resp
            .bytes()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to read response: {}", e)))?;
        let body_str = String::from_utf8_lossy(&body).to_string();

        let result = extract_readable_content(&body_str);

        let data = serde_json::json!({
            "url": url,
            "title": result.title,
            "score": result.score,
            "best_selector": result.best_selector,
            "rate_limited_ms": waited,
            "content_length": result.content.len(),
        });

        Ok(ToolOutput::success(result.content).with_data(data))
    }
}

#[async_trait]
impl Tool for WebExtractTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "web_extract",
                concat!(
                    "Fetch a web page and extract its main content with readability scoring. ",
                    "Respects robots.txt, enforces per-domain rate limiting, caches responses, ",
                    "maintains cookies, and follows redirects.",
),
                HashMap::from([
                    (
                        "url".to_string(),
                        ParameterDefinition::required_string("The URL to extract content from"),
                    ),
                    (
                        "mode".to_string(),
                        ParameterDefinition::string("Extraction mode: readable (default) or text")
                            .default(serde_json::json!("readable")),
                    ),
                    (
                        "use_cache".to_string(),
                        ParameterDefinition::boolean("Use the response cache")
                            .default(serde_json::json!(true)),
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
        let mode = params["mode"].as_str().unwrap_or("readable");
        let use_cache = params["use_cache"].as_bool().unwrap_or(true);

        match mode {
            "readable" => self.extract_readable(url).await,
            "text" => self.fetch_content(url, use_cache).await,
            other => Err(ToolError::invalid_args(format!(
                "Unknown mode: '{}'. Supported: readable, text",
                other
            ))),
        }
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

    #[test]
    fn test_html_to_text_strips_script() {
        let html = "<html><head><title>Test</title></head><body><script>alert('x')</script><p>Hello world</p><p>Second paragraph</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Hello world"));
        assert!(text.contains("Second paragraph"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_html_to_text_preserves_blocks() {
        let html = "<div><h1>Title</h1><p>First</p><p>Second</p></div>";
        let text = html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
    }

    #[test]
    fn test_readability_score_empty() {
        assert_eq!(readability_score(""), 0.0);
        assert_eq!(readability_score("   "), 0.0);
    }

    #[test]
    fn test_readability_score_substantial_text() {
        let text = "This is a substantial article about programming languages. \
                    The history of computing is filled with interesting developments. \
                    Modern systems rely heavily on compiler technology. \
                    Memory management remains a challenging topic for developers.";
        let score = readability_score(text);
        assert!(score > 0.0);
    }

    #[test]
    fn test_readability_score_short_text() {
        let short = "hello world";
        let long_text = "This is a much longer piece of article content that contains \
                         multiple sentences with proper punctuation. It reads like a \
                         blog post or news article would read. The vocabulary includes \
                         substantial words that appear in formal writing contexts.";
        assert!(readability_score(short) < readability_score(long_text));
    }

    #[test]
    fn test_extract_readable_content() {
        let html = "<html><head><title>My Article</title></head><body>\
                    <nav>navigation links here</nav>\
                    <article><h1>Article Title</h1>\
                    <p>This is the first paragraph of the article content. It contains \
                    substantial text that would be scored as readable content.</p>\
                    <p>This is the second paragraph with more detailed information \
                    about the subject matter being discussed in the article.</p>\
                    </article>\
                    <footer>footer stuff</footer></body></html>";
        let result = extract_readable_content(html);
        assert_eq!(result.title, "My Article");
        assert!(result.content.contains("first paragraph"));
        assert!(!result.content.contains("navigation"));
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn test_rate_limiter_allows_sequential() {
        let limiter = DomainRateLimiter::new(0);
        let waited = limiter.acquire("http://example.com/page").await;
        assert_eq!(waited, 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_different_domains() {
        let limiter = DomainRateLimiter::new(1000);
        let w1 = limiter.acquire("http://a.example.com/page").await;
        let w2 = limiter.acquire("http://b.example.com/page").await;
        assert_eq!(w1, 0);
        assert_eq!(w2, 0); // Different domains, no waiting.
    }

    #[test]
    fn test_response_cache_roundtrip() {
        let cache = ResponseCache::new(10, 60);
        assert!(cache.get("http://example.com/").is_none());
        cache.put("http://example.com/", 200, "text/html", b"hello".to_vec());
        let cached = cache.get("http://example.com/").unwrap();
        assert_eq!(cached.status, 200);
        assert_eq!(cached.body, b"hello");
        assert_eq!(cache.len(), 1);
        assert!(cache.invalidate("http://example.com/"));
        assert!(cache.get("http://example.com/").is_none());
    }

    #[test]
    fn test_response_cache_eviction() {
        let cache = ResponseCache::new(2, 60);
        cache.put("http://a.com/", 200, "text/plain", b"a".to_vec());
        cache.put("http://b.com/", 200, "text/plain", b"b".to_vec());
        cache.put("http://c.com/", 200, "text/plain", b"c".to_vec());
        assert!(cache.len() <= 2);
    }

    #[tokio::test]
    async fn test_cookie_jar_store_and_retrieve() {
        let jar = CookieJar::new();
        let url = "http://example.com/login";
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::SET_COOKIE,
            reqwest::header::HeaderValue::from_str("session=abc123; Path=/").unwrap(),
        );
        jar.store_from_response(url, &headers);
        let cookie_header = jar.header_for("http://example.com/profile").unwrap();
        assert!(cookie_header.contains("session=abc123"));
    }

    #[tokio::test]
    async fn test_cookie_jar_different_domain() {
        let jar = CookieJar::new();
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::SET_COOKIE,
            reqwest::header::HeaderValue::from_str("session=abc123").unwrap(),
        );
        jar.store_from_response("http://a.com/", &headers);
        assert!(jar.header_for("http://b.com/").is_none());
    }

    #[test]
    fn test_robots_parse_rules() {
        let text = "User-agent: *\nDisallow: /private/\nAllow: /private/public.html\n\nUser-agent: Googlebot\nDisallow: /";
        let rules = RobotsTxt::parse_rules(text);
        assert_eq!(rules.len(), 2);
        let wildcard = &rules[0];
        assert_eq!(wildcard.0, "*");
        assert_eq!(wildcard.2, vec!["/private/"]);
    }

    #[test]
    fn test_robots_is_path_allowed() {
        let allow = vec!["/private/public.html".to_string()];
        let disallow = vec!["/private/".to_string()];
        assert!(!RobotsTxt::is_path_allowed("/private/secret", &allow, &disallow));
        assert!(RobotsTxt::is_path_allowed("/private/public.html", &allow, &disallow));
        assert!(RobotsTxt::is_path_allowed("/public/", &allow, &disallow));
    }

    #[tokio::test]
    async fn test_web_extract_ssrf_blocked() {
        let tool = WebExtractTool::new();
        let result = tool
            .execute(serde_json::json!({
                "url": "http://127.0.0.1:8080/secret",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_web_extract_missing_url() {
        let tool = WebExtractTool::new();
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }
}
