//! Session search.
//!
//! Full-text search over session transcripts and metadata. The in-memory
//! index stores messages keyed by session and message id; search results are
//! ranked by a simple term-frequency score.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// A single indexed message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedMessage {
    pub session_id: String,
    pub message_id: String,
    pub role: String,
    pub content: String,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// A search hit with its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSearchHit {
    pub session_id: String,
    pub message_id: String,
    pub role: String,
    pub content: String,
    /// Ranked snippet around the first match.
    pub snippet: String,
    /// Term-frequency score (higher is more relevant).
    pub score: f64,
    pub timestamp: DateTime<Utc>,
}

/// The result of a session search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSearchResult {
    pub query: String,
    pub hits: Vec<SessionSearchHit>,
    pub total: usize,
    pub took_ms: u64,
}

/// Search options.
#[derive(Debug, Clone, Copy, Default)]
pub struct SessionSearchOptions {
    /// Maximum number of hits to return.
    pub limit: usize,
    /// Only search this session, if set.
    pub session_id: Option<&'static str>,
}

/// In-memory full-text index over session messages.
///
/// Thread-safe and clone-friendly. Messages are indexed incrementally via
/// [`SessionSearchIndex::index`]; the search itself is a naive token scan,
/// which is sufficient for the gateway's in-memory session store.
#[derive(Clone, Default)]
pub struct SessionSearchIndex {
    messages: std::sync::Arc<RwLock<Vec<IndexedMessage>>>,
}

impl SessionSearchIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a message.
    pub fn index(&self, message: IndexedMessage) {
        let mut guard = self.messages.write();
        // Replace an existing message with the same id to keep the index
        // consistent on re-indexing.
        if let Some(existing) = guard
            .iter_mut()
            .find(|m| m.message_id == message.message_id)
        {
            *existing = message;
        } else {
            guard.push(message);
        }
    }

    /// Bulk-index messages.
    pub fn index_many(&self, messages: Vec<IndexedMessage>) {
        for message in messages {
            self.index(message);
        }
    }

    /// Remove a message from the index.
    pub fn remove(&self, session_id: &str, message_id: &str) -> bool {
        let mut guard = self.messages.write();
        let len_before = guard.len();
        guard.retain(|m| !(m.session_id == session_id && m.message_id == message_id));
        guard.len() != len_before
    }

    /// Remove all messages for a session.
    pub fn remove_session(&self, session_id: &str) -> usize {
        let mut guard = self.messages.write();
        let len_before = guard.len();
        guard.retain(|m| m.session_id != session_id);
        len_before - guard.len()
    }

    /// Search the index.
    ///
    /// Returns hits ranked by term frequency, optionally restricted to a
    /// single session.
    pub fn search(&self, query: &str, options: SessionSearchOptions) -> SessionSearchResult {
        let started = std::time::Instant::now();
        let terms: Vec<String> = tokenize(query);

        let guard = self.messages.read();
        let mut scored: Vec<(f64, &IndexedMessage)> = Vec::new();
        for message in guard.iter() {
            if let Some(session_id) = options.session_id {
                if message.session_id != session_id {
                    continue;
                }
            }
            let score = score_message(message, &terms);
            if score > 0.0 {
                scored.push((score, message));
            }
        }

        // Rank by score descending, then by recency descending.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.timestamp.cmp(&a.1.timestamp))
        });

        let limit = if options.limit == 0 {
            50
        } else {
            options.limit
        };
        let hits: Vec<SessionSearchHit> = scored
            .into_iter()
            .take(limit)
            .map(|(score, m)| SessionSearchHit {
                session_id: m.session_id.clone(),
                message_id: m.message_id.clone(),
                role: m.role.clone(),
                content: m.content.clone(),
                snippet: build_snippet(&m.content, &terms),
                score,
                timestamp: m.timestamp,
            })
            .collect();

        SessionSearchResult {
            query: query.to_string(),
            total: hits.len(),
            hits,
            took_ms: started.elapsed().as_millis() as u64,
        }
    }

    /// Return the total number of indexed messages.
    pub fn len(&self) -> usize {
        self.messages.read().len()
    }

    /// Return `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.read().is_empty()
    }
}

/// Convert an error-free query string into lowercase tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|tok| !tok.is_empty())
        .map(|tok| tok.to_lowercase())
        .collect()
}

/// Score a message against the query terms using term frequency.
fn score_message(message: &IndexedMessage, terms: &[String]) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let body = message.content.to_lowercase();
    let mut score = 0.0;
    for term in terms {
        // Count occurrences of the term in the content.
        let occurrences = body.match_indices(term).count();
        if occurrences > 0 {
            // Weight by exact-boundary hits more than substring hits.
            score += occurrences as f64;
            if body.starts_with(term.as_str()) || body.contains(&format!(" {term} ")) {
                score += 1.0;
            }
        }
    }
    // Normalize slightly by message length so short messages don't dominate.
    score / (1.0 + (message.content.len() as f64 / 500.0))
}

/// Build a snippet around the first term occurrence.
fn build_snippet(content: &str, terms: &[String]) -> String {
    let lower = content.to_lowercase();
    let first = terms.iter().find_map(|t| lower.find(t)).unwrap_or(0);
    let start = first.saturating_sub(40);
    let end = (first + 120).min(content.len());
    let mut snippet = content[start..end].to_string();
    if start > 0 {
        snippet.insert_str(0, "…");
    }
    if end < content.len() {
        snippet.push('…');
    }
    snippet
}

/// Convenience: search all sessions.
pub fn search_all(index: &SessionSearchIndex, query: &str, limit: usize) -> SessionSearchResult {
    index.search(
        query,
        SessionSearchOptions {
            limit,
            session_id: None,
        },
    )
}

/// Convenience: search within a single session.
pub fn search_session(
    index: &SessionSearchIndex,
    query: &str,
    session_id: &'static str,
    limit: usize,
) -> SessionSearchResult {
    index.search(
        query,
        SessionSearchOptions {
            limit,
            session_id: Some(session_id),
        },
    )
}

/// Map a search error to an [`AppError`]. Search itself is infallible in the
/// current implementation, but this exists for API uniformity.
pub fn search_error(err: AppError) -> AppError {
    err
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_index() -> SessionSearchIndex {
        let index = SessionSearchIndex::new();
        index.index_many(vec![
            IndexedMessage {
                session_id: "s1".into(),
                message_id: "m1".into(),
                role: "user".into(),
                content: "How do I configure the provider API key?".into(),
                model: None,
                timestamp: Utc::now(),
            },
            IndexedMessage {
                session_id: "s1".into(),
                message_id: "m2".into(),
                role: "assistant".into(),
                content: "The API key goes in the config TOML under [providers].".into(),
                model: Some("gpt-4o".into()),
                timestamp: Utc::now(),
            },
            IndexedMessage {
                session_id: "s2".into(),
                message_id: "m3".into(),
                role: "user".into(),
                content: "What is the weather today?".into(),
                model: None,
                timestamp: Utc::now(),
            },
        ]);
        index
    }

    #[test]
    fn test_search_returns_matching_hits() {
        let index = sample_index();
        let result = search_all(&index, "API key", 10);
        assert!(result.total >= 1);
        assert!(result.hits.iter().any(|h| h.session_id == "s1"));
        assert!(
            !result
                .hits
                .iter()
                .any(|h| h.session_id == "s2" && h.role == "user")
        );
    }

    #[test]
    fn test_search_session_scope() {
        let index = sample_index();
        let result = search_session(&index, "config", "s2", 10);
        assert_eq!(result.total, 0);
    }

    #[test]
    fn test_search_empty_query() {
        let index = sample_index();
        let result = search_all(&index, "", 10);
        assert_eq!(result.total, 0);
    }

    #[test]
    fn test_search_ranking() {
        let index = sample_index();
        // "config" appears in both s1 messages; the hit with more occurrences
        // should rank first.
        let result = search_all(&index, "config TOML providers", 10);
        assert!(!result.hits.is_empty());
        let first = &result.hits[0];
        assert!(first.score > 0.0);
        assert!(first.snippet.contains("config"));
    }

    #[test]
    fn test_index_remove() {
        let index = sample_index();
        assert_eq!(index.len(), 3);
        assert!(index.remove("s1", "m1"));
        assert_eq!(index.len(), 2);
        assert_eq!(index.remove_session("s1"), 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_index_replace_same_id() {
        let index = SessionSearchIndex::new();
        index.index(IndexedMessage {
            session_id: "s1".into(),
            message_id: "m1".into(),
            role: "user".into(),
            content: "first".into(),
            model: None,
            timestamp: Utc::now(),
        });
        index.index(IndexedMessage {
            session_id: "s1".into(),
            message_id: "m1".into(),
            role: "user".into(),
            content: "second".into(),
            model: None,
            timestamp: Utc::now(),
        });
        assert_eq!(index.len(), 1);
        let result = search_all(&index, "second", 10);
        assert_eq!(result.total, 1);
    }

    #[test]
    fn test_tokenize_splits_non_alphanumeric() {
        assert_eq!(tokenize("Hello, World!"), vec!["hello", "world"]);
        assert_eq!(tokenize("  spaced  "), vec!["spaced"]);
    }

    #[test]
    fn test_snippet_has_ellipses() {
        let long = "x".repeat(300);
        let snippet = build_snippet(&long, &["x".to_string()]);
        assert!(snippet.starts_with('…'));
        assert!(snippet.ends_with('…'));
    }
}
