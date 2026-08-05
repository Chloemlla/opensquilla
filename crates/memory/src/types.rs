//! Core memory domain types.
//!
//! These are the canonical data structures shared across the memory
//! subsystem: [`MemoryEntry`] (an agent-scoped memory record), [`MemoryChunk`]
//! (a chunked slice of an indexed file with its embedding), [`MemoryQuery`] (a
//! search request carrying an optional embedding and filters), and
//! [`MemorySearchResult`] (a scored hit wrapping a memory entry).
//!
//! `MemoryEntry` historically lived in [`crate::store`]; it is re-exported from
//! there for backwards compatibility, but this module is now the single source
//! of truth.

use chrono::{DateTime, Utc};
use opensquilla_core::types::MemoryId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single memory record belonging to an agent.
///
/// A `MemoryEntry` is the unit of recall: it carries the textual content, an
/// optional dense embedding used for vector search, free-form tags, lifecycle
/// timestamps, an access counter, and an importance score in `[0.0, 1.0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Unique identifier for this memory.
    pub id: MemoryId,
    /// The agent this memory belongs to (per-agent isolation).
    pub agent_id: Uuid,
    /// Human-readable content of the memory.
    pub content: String,
    /// Free-form tags attached to this memory, used for filtering.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Dense vector embedding of `content`, used for similarity search.
    /// `None` when no embedding provider is configured or generation failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// When the memory was first created.
    pub created_at: DateTime<Utc>,
    /// When the memory was last updated.
    pub updated_at: DateTime<Utc>,
    /// When the memory was last accessed (read). `None` until first access.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accessed_at: Option<DateTime<Utc>>,
    /// Provenance: how this memory entered the store
    /// (e.g. `"conversation"`, `"file_sync"`, `"import"`, `"dream"`).
    pub source: String,
    /// Semantic category (e.g. `"episodic"`, `"preference"`, `"document"`).
    pub memory_type: String,
    /// Importance score in `[0.0, 1.0]`; higher means more salient.
    pub importance: f64,
    /// Alias of [`importance`][Self::importance] exposed for API symmetry with
    /// the Python memory system. Kept in sync by the store on write.
    #[serde(default = "default_importance_score")]
    pub importance_score: f64,
    /// Number of times this memory has been retrieved / touched.
    pub access_count: u64,
    /// Arbitrary structured metadata associated with the memory.
    pub metadata: serde_json::Value,
}

fn default_importance_score() -> f64 {
    0.5
}

impl MemoryEntry {
    /// Create a new memory entry with sensible defaults for the optional
    /// fields. The caller is expected to fill in `content`, `source`,
    /// `memory_type`, and `importance`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: MemoryId,
        agent_id: Uuid,
        content: String,
        source: String,
        memory_type: String,
        importance: f64,
        metadata: serde_json::Value,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            agent_id,
            content,
            tags: Vec::new(),
            embedding: None,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source,
            memory_type,
            importance,
            importance_score: importance,
            access_count: 0,
            metadata,
        }
    }

    /// Convenience builder: set the tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Convenience builder: attach an embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Mark the importance score, keeping [`importance`][Self::importance] and
    /// [`importance_score`][Self::importance_score] in sync.
    pub fn set_importance(&mut self, importance: f64) {
        self.importance = importance;
        self.importance_score = importance;
    }
}

/// A chunk of an indexed file, with its own embedding.
///
/// Files imported into the memory store are split into chunks so that
/// retrieval can operate at sub-document granularity. Each chunk records the
/// parent file, its positional index, an approximate token count, and an
/// optional dense embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChunk {
    /// Unique identifier for this chunk.
    pub id: Uuid,
    /// Identifier of the parent indexed file this chunk belongs to.
    pub file_id: Uuid,
    /// The chunk's text content.
    pub content: String,
    /// Dense vector embedding of `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Zero-based positional index of this chunk within the parent file.
    pub chunk_index: u32,
    /// Approximate token count of `content` (used for budgeting).
    pub token_count: u32,
}

impl MemoryChunk {
    /// Create a new chunk with no embedding.
    pub fn new(file_id: Uuid, chunk_index: u32, content: String, token_count: u32) -> Self {
        Self {
            id: Uuid::new_v4(),
            file_id,
            content,
            embedding: None,
            chunk_index,
            token_count,
        }
    }
}

/// A memory search request.
///
/// Carries the query text, an optional precomputed embedding (so callers can
/// reuse a cached query embedding), optional filters, a result limit, and a
/// minimum score threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// The natural-language query text.
    pub query: String,
    /// Optional precomputed embedding of [`query`][Self::query].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Filters to apply to candidate memories.
    #[serde(default)]
    pub filters: MemoryFilters,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Minimum normalized score (`[0.0, 1.0]`) for a result to be included.
    #[serde(default)]
    pub min_score: f64,
}

impl MemoryQuery {
    /// Build a query from text with a given limit and no filters.
    pub fn from_text(query: impl Into<String>, limit: usize) -> Self {
        Self {
            query: query.into(),
            embedding: None,
            filters: MemoryFilters::default(),
            limit,
            min_score: 0.0,
        }
    }

    /// Attach a precomputed embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attach filters.
    pub fn with_filters(mut self, filters: MemoryFilters) -> Self {
        self.filters = filters;
        self
    }

    /// Set the minimum score threshold.
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }
}

/// Filters applied to candidate memories during retrieval.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilters {
    /// Restrict to memories belonging to this agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<Uuid>,
    /// Restrict to memories tagged with at least one of these tags (OR semantics).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Restrict to memories of one of these semantic types.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_types: Vec<String>,
    /// Restrict to memories created at or after this instant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<DateTime<Utc>>,
    /// Restrict to memories created at or before this instant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub until: Option<DateTime<Utc>>,
}

impl MemoryFilters {
    /// Returns `true` if no filter is set.
    pub fn is_empty(&self) -> bool {
        self.agent_id.is_none()
            && self.tags.is_empty()
            && self.memory_types.is_empty()
            && self.since.is_none()
            && self.until.is_none()
    }

    /// Returns `true` if the given entry passes all configured filters.
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        if let Some(agent) = self.agent_id {
            if entry.agent_id != agent {
                return false;
            }
        }
        if !self.tags.is_empty() && !self.tags.iter().any(|t| entry.tags.contains(t)) {
            return false;
        }
        if !self.memory_types.is_empty() && !self.memory_types.contains(&entry.memory_type) {
            return false;
        }
        if let Some(since) = self.since {
            if entry.created_at < since {
                return false;
            }
        }
        if let Some(until) = self.until {
            if entry.created_at > until {
                return false;
            }
        }
        true
    }
}

/// A single scored search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    /// The matching memory entry.
    pub entry: MemoryEntry,
    /// Normalized score in `[0.0, 1.0]` (higher is better).
    pub score: f64,
    /// Which retrieval method produced this hit
    /// (`"vector"`, `"bm25"`, `"hybrid"`).
    pub source: String,
}

impl MemorySearchResult {
    /// Create a new result.
    pub fn new(entry: MemoryEntry, score: f64, source: impl Into<String>) -> Self {
        Self {
            entry,
            score,
            source: source.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> MemoryEntry {
        MemoryEntry::new(
            MemoryId::new(),
            Uuid::new_v4(),
            "hello world".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        )
    }

    #[test]
    fn test_new_sets_defaults() {
        let e = entry();
        assert!(e.tags.is_empty());
        assert!(e.embedding.is_none());
        assert!(e.accessed_at.is_none());
        assert_eq!(e.importance, e.importance_score);
        assert_eq!(e.access_count, 0);
    }

    #[test]
    fn test_filters_match_agent() {
        let e = entry();
        let mut f = MemoryFilters::default();
        assert!(f.is_empty());
        f.agent_id = Some(e.agent_id);
        assert!(f.matches(&e));
        f.agent_id = Some(Uuid::new_v4());
        assert!(!f.matches(&e));
    }

    #[test]
    fn test_filters_match_tags_and_time() {
        let mut e = entry();
        e.tags = vec!["rust".to_string(), "memory".to_string()];
        let f = MemoryFilters {
            tags: vec!["rust".to_string()],
            ..Default::default()
        };
        assert!(f.matches(&e));

        let f = MemoryFilters {
            since: Some(e.created_at + chrono::Duration::hours(1)),
            ..Default::default()
        };
        assert!(!f.matches(&e));
    }

    #[test]
    fn test_set_importance_keeps_in_sync() {
        let mut e = entry();
        e.set_importance(0.9);
        assert_eq!(e.importance, 0.9);
        assert_eq!(e.importance_score, 0.9);
    }

    #[test]
    fn test_chunk_new() {
        let c = MemoryChunk::new(Uuid::new_v4(), 0, "chunk text".to_string(), 4);
        assert_eq!(c.chunk_index, 0);
        assert_eq!(c.token_count, 4);
        assert!(c.embedding.is_none());
    }

    #[test]
    fn test_query_builder() {
        let q = MemoryQuery::from_text("rust async", 10).with_min_score(0.2);
        assert_eq!(q.limit, 10);
        assert_eq!(q.min_score, 0.2);
        assert!(q.embedding.is_none());
    }
}
