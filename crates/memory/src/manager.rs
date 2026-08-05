use chrono::{Duration, Utc};
use dashmap::DashMap;
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::{MemoryId, Message, SessionId};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::embedding::{CachedEmbeddingProvider, EmbeddingProvider};
use crate::retrieval::RetrievalEngine;
use crate::session_source::SessionSource;
use crate::store::MemoryStore;
use crate::turn_capture::TurnCapture;
use crate::types::{MemoryEntry, MemoryFilters, MemoryQuery, MemorySearchResult};

/// Configuration for automatic turn capture.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Whether automatic turn capture is enabled.
    pub enabled: bool,
    /// Whether to store the raw transcript for each turn.
    pub store_transcript: bool,
    /// Default importance for turn-derived memories.
    pub importance: f64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            store_transcript: true,
            importance: 0.5,
        }
    }
}

/// Configuration for memory expiry (TTL).
#[derive(Debug, Clone)]
pub struct ExpiryConfig {
    /// How long a memory may live before being eligible for expiry.
    pub ttl: Duration,
    /// Memories with importance below this threshold are eligible for expiry.
    pub min_importance: f64,
}

impl Default for ExpiryConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::days(180),
            min_importance: 0.3,
        }
    }
}

/// Per-agent memory manager: isolation, capture, query, consolidation, expiry.
pub struct MemoryManager {
    store: MemoryStore,
    retrieval: RetrievalEngine,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    agent_memories: DashMap<Uuid, Vec<MemoryId>>,
    capture_config: CaptureConfig,
    expiry_config: ExpiryConfig,
}

impl MemoryManager {
    pub fn new(store: MemoryStore) -> Self {
        let retrieval = RetrievalEngine::new(store.clone());
        Self {
            store,
            retrieval,
            embedding_provider: None,
            agent_memories: DashMap::new(),
            capture_config: CaptureConfig::default(),
            expiry_config: ExpiryConfig::default(),
        }
    }

    pub fn with_embedding_provider(mut self, provider: Box<dyn EmbeddingProvider>) -> Self {
        // Wrap the provider in a store-backed cache when possible.
        self.embedding_provider = Some(Arc::new(CachedEmbeddingProvider::new(
            Arc::from(provider),
            self.store.clone(),
        )));
        self
    }

    /// Set the embedding provider directly (as an `Arc`).
    pub fn with_embedding_provider_arc(mut self, provider: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(Arc::new(CachedEmbeddingProvider::new(
            provider,
            self.store.clone(),
        )));
        self
    }

    pub fn with_capture_config(mut self, config: CaptureConfig) -> Self {
        self.capture_config = config;
        self
    }

    pub fn with_expiry_config(mut self, config: ExpiryConfig) -> Self {
        self.expiry_config = config;
        self
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn retrieval(&self) -> &RetrievalEngine {
        &self.retrieval
    }

    pub fn embedding_provider(&self) -> Option<&Arc<dyn EmbeddingProvider>> {
        self.embedding_provider.as_ref()
    }

    /// Register a memory id in the per-agent index. Used by the capture helper.
    pub fn note_memory_id(&self, agent_id: Uuid, memory_id: MemoryId) {
        self.agent_memories
            .entry(agent_id)
            .or_default()
            .push(memory_id);
    }

    // --- Add / write path ---

    /// Add a memory entry, generating an embedding when a provider is set.
    pub async fn add_memory(
        &self,
        agent_id: Uuid,
        content: String,
        source: String,
        memory_type: String,
        importance: f64,
        metadata: serde_json::Value,
    ) -> CoreResult<MemoryId> {
        self.add_memory_full(
            agent_id,
            content,
            Vec::new(),
            source,
            memory_type,
            importance,
            metadata,
        )
        .await
    }

    /// Add a memory entry with tags, generating an embedding when available.
    pub async fn add_memory_full(
        &self,
        agent_id: Uuid,
        content: String,
        tags: Vec<String>,
        source: String,
        memory_type: String,
        importance: f64,
        metadata: serde_json::Value,
    ) -> CoreResult<MemoryId> {
        let now = Utc::now();
        let memory_id = MemoryId(Uuid::new_v4());
        let importance = importance.clamp(0.0, 1.0);

        let mut entry = MemoryEntry {
            id: memory_id,
            agent_id,
            content,
            tags,
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
        };

        // Generate embedding if provider is available.
        if let Some(provider) = &self.embedding_provider {
            match provider.embed(&entry.content).await {
                Ok(embedding) => entry.embedding = Some(embedding),
                Err(e) => warn!("Embedding failed for memory {}: {}", memory_id.0, e),
            }
        }

        self.store.insert_memory(&entry)?;
        if let Some(embedding) = &entry.embedding {
            self.store.store_embedding(&entry.id, embedding)?;
        }
        // Persist tags to the legacy tag table as well.
        for tag in &entry.tags {
            self.store.add_tag(&entry.id, tag)?;
        }

        self.note_memory_id(agent_id, memory_id);
        info!("Added memory {} for agent {}", memory_id.0, agent_id);
        Ok(memory_id)
    }

    /// Add a memory entry with a precomputed embedding (skips provider call).
    pub async fn add_memory_with_embedding(
        &self,
        agent_id: Uuid,
        content: String,
        embedding: Vec<f32>,
        tags: Vec<String>,
        source: String,
        memory_type: String,
        importance: f64,
        metadata: serde_json::Value,
    ) -> CoreResult<MemoryId> {
        let now = Utc::now();
        let memory_id = MemoryId(Uuid::new_v4());
        let importance = importance.clamp(0.0, 1.0);
        let entry = MemoryEntry {
            id: memory_id,
            agent_id,
            content,
            tags: tags.clone(),
            embedding: Some(embedding.clone()),
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source,
            memory_type,
            importance,
            importance_score: importance,
            access_count: 0,
            metadata,
        };
        self.store.insert_memory(&entry)?;
        self.store.store_embedding(&entry.id, &embedding)?;
        for tag in &tags {
            self.store.add_tag(&entry.id, tag)?;
        }
        self.note_memory_id(agent_id, memory_id);
        Ok(memory_id)
    }

    // --- Automatic capture ---

    /// Automatically capture a completed turn's messages into memory.
    pub async fn capture_turn(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
        metadata: serde_json::Value,
    ) -> CoreResult<Vec<MemoryId>> {
        if !self.capture_config.enabled {
            return Ok(Vec::new());
        }
        let config = crate::turn_capture::TurnCaptureConfig {
            store_raw_transcript: self.capture_config.store_transcript,
            default_importance: self.capture_config.importance,
            ..Default::default()
        };
        let capture = TurnCapture::with_config(self.store.clone(), config);
        let ids = capture.capture_turn(session_id, agent_id, messages, metadata)?;
        for id in &ids {
            self.note_memory_id(agent_id, *id);
        }
        Ok(ids)
    }

    /// Derive a session memory document from a completed session.
    pub fn derive_session_document(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
        metadata: serde_json::Value,
    ) -> CoreResult<Option<MemoryId>> {
        let source = SessionSource::new(self.store.clone());
        let id = source.derive_and_store(session_id, agent_id, messages, metadata)?;
        if let Some(id) = id {
            self.note_memory_id(agent_id, id);
        }
        Ok(id)
    }

    // --- Query interface ---

    /// Run a query with filters against the hybrid retrieval engine.
    pub async fn query(&self, query: &MemoryQuery) -> CoreResult<Vec<MemorySearchResult>> {
        // Ensure an embedding exists for vector search if we have a provider.
        let embedding = match (&query.embedding, &self.embedding_provider) {
            (Some(e), _) => Some(e.clone()),
            (None, Some(provider)) => Some(provider.embed(&query.query).await?),
            (None, None) => None,
        };

        self.retrieval.search(
            &query.query,
            embedding.as_deref(),
            query.filters.agent_id.as_ref(),
            &query.filters,
            query.limit,
        )
    }

    /// Search memories for an agent (backwards-compatible entry point).
    pub async fn search(
        &self,
        agent_id: &Uuid,
        query_text: &str,
        limit: u64,
    ) -> CoreResult<Vec<MemoryEntry>> {
        let query = MemoryQuery {
            query: query_text.to_string(),
            embedding: None,
            filters: MemoryFilters {
                agent_id: Some(*agent_id),
                ..Default::default()
            },
            limit: limit as usize,
            min_score: 0.0,
        };
        let results = self.query(&query).await?;
        Ok(results.into_iter().map(|r| r.entry).collect())
    }

    /// Search with full filters, returning scored results.
    pub async fn search_filtered(
        &self,
        agent_id: &Uuid,
        query_text: &str,
        filters: MemoryFilters,
        limit: usize,
    ) -> CoreResult<Vec<MemorySearchResult>> {
        let mut filters = filters;
        filters.agent_id = Some(*agent_id);
        let query = MemoryQuery {
            query: query_text.to_string(),
            embedding: None,
            filters,
            limit,
            min_score: 0.0,
        };
        self.query(&query).await
    }

    /// Get a specific memory, bumping its access counter.
    pub fn get_memory(&self, memory_id: &MemoryId) -> CoreResult<Option<MemoryEntry>> {
        let entry = self.store.get_memory(memory_id)?;
        if entry.is_some() {
            self.store.increment_access(memory_id)?;
        }
        Ok(entry)
    }

    /// Delete a memory.
    pub fn delete_memory(&self, memory_id: &MemoryId) -> CoreResult<()> {
        self.store.delete_memory(memory_id)?;
        for mut entry in self.agent_memories.iter_mut() {
            entry.retain(|id| id != memory_id);
        }
        info!("Deleted memory {}", memory_id.0);
        Ok(())
    }

    /// Update memory importance (re-scores both `importance` fields).
    pub fn update_importance(&self, memory_id: &MemoryId, importance: f64) -> CoreResult<()> {
        let mut entry = self
            .store
            .get_memory(memory_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Memory {}", memory_id.0)))?;

        entry.set_importance(importance.clamp(0.0, 1.0));
        entry.updated_at = Utc::now();
        self.store.update_memory(&entry)
    }

    /// List memories for an agent.
    pub fn list_memories(
        &self,
        agent_id: &Uuid,
        memory_type: Option<&str>,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<MemoryEntry>> {
        self.store
            .list_memories(agent_id, memory_type, limit, offset)
    }

    /// Get memory count for an agent (in-memory index).
    pub fn memory_count(&self, agent_id: &Uuid) -> usize {
        self.agent_memories
            .get(agent_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// List all agent ids currently tracked.
    pub fn known_agents(&self) -> Vec<Uuid> {
        self.agent_memories.iter().map(|e| *e.key()).collect()
    }

    // --- Consolidation ---

    /// Merge similar memories for an agent into a single consolidated entry.
    ///
    /// Uses a Jaccard-like content similarity threshold: pairs above
    /// `threshold` are merged, the merged entry keeps the highest importance
    /// and the union of tags, and source memories are deleted.
    pub fn consolidate(&self, agent_id: &Uuid, threshold: f64) -> CoreResult<ConsolidationSummary> {
        let memories = self.store.list_memories(agent_id, None, 1000, 0)?;
        let mut summary = ConsolidationSummary {
            agent_id: *agent_id,
            candidates: memories.len() as u64,
            merged: 0,
            removed: 0,
        };

        if memories.len() < 2 {
            return Ok(summary);
        }

        let mut visited: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
        for i in 0..memories.len() {
            if visited.contains(&memories[i].id) {
                continue;
            }
            let mut cluster = vec![memories[i].clone()];
            visited.insert(memories[i].id);

            for j in (i + 1)..memories.len() {
                if visited.contains(&memories[j].id) {
                    continue;
                }
                let sim = crate::retrieval::content_similarity(
                    &memories[i].content,
                    &memories[j].content,
                );
                if sim >= threshold {
                    cluster.push(memories[j].clone());
                    visited.insert(memories[j].id);
                }
            }

            if cluster.len() > 1 {
                let merged = merge_entries(&cluster);
                // Keep the first entry as the canonical id; store merged content.
                self.store.update_memory(&merged)?;
                for m in cluster.iter().skip(1) {
                    self.store.delete_memory(&m.id)?;
                    summary.removed += 1;
                }
                summary.merged += 1;
            }
        }

        debug!(
            "Consolidation for agent {}: {} merged, {} removed",
            agent_id, summary.merged, summary.removed
        );
        Ok(summary)
    }

    // --- Importance scoring ---

    /// Compute an importance score for a memory based on recency, access
    /// frequency, and content length. Used to refresh stored importance.
    pub fn score_importance(&self, entry: &MemoryEntry, recency_half_life_hours: f64) -> f64 {
        let now = Utc::now();
        let age_hours = (now - entry.created_at).num_hours().max(0) as f64;
        let recency = (-age_hours / recency_half_life_hours).exp();

        let access_factor = (entry.access_count as f64).ln_1p().min(1.0);
        let length_factor = ((entry.content.chars().count() as f64) / 500.0).min(1.0);

        // Blend: base importance, recency, access, and modest length bonus.
        let score =
            0.4 * entry.importance + 0.3 * recency + 0.2 * access_factor + 0.1 * length_factor;
        score.clamp(0.0, 1.0)
    }

    /// Recompute and persist importance scores for all of an agent's memories.
    pub fn rebalance_importances(
        &self,
        agent_id: &Uuid,
        recency_half_life_hours: f64,
    ) -> CoreResult<u64> {
        let memories = self.store.list_memories(agent_id, None, 1000, 0)?;
        let mut updated = 0u64;
        for mut entry in memories {
            let score = self.score_importance(&entry, recency_half_life_hours);
            entry.set_importance(score);
            entry.updated_at = Utc::now();
            self.store.update_memory(&entry)?;
            updated += 1;
        }
        Ok(updated)
    }

    // --- Expiry / TTL ---

    /// Expire old, low-importance memories for all agents.
    pub fn expire_old_memories(&self) -> CoreResult<u64> {
        self.store
            .expire_old_memories(self.expiry_config.ttl, self.expiry_config.min_importance)
    }

    /// Expire old memories with a custom TTL / importance threshold.
    pub fn expire_with(&self, ttl: Duration, min_importance: f64) -> CoreResult<u64> {
        self.store.expire_old_memories(ttl, min_importance)
    }
}

/// Merge a cluster of similar memories into a single canonical entry.
fn merge_entries(cluster: &[MemoryEntry]) -> MemoryEntry {
    let mut merged = cluster[0].clone();
    let mut max_importance = 0.0_f64;
    let mut combined_content = String::new();
    let mut tags: Vec<String> = Vec::new();

    for entry in cluster {
        max_importance = max_importance.max(entry.importance);
        if !combined_content.is_empty() {
            combined_content.push_str("\n---\n");
        }
        combined_content.push_str(&entry.content);
        for tag in &entry.tags {
            if !tags.contains(tag) {
                tags.push(tag.clone());
            }
        }
    }

    merged.content = combined_content;
    merged.tags = tags;
    merged.importance = max_importance;
    merged.importance_score = max_importance;
    merged.updated_at = Utc::now();
    merged.metadata = serde_json::json!({
        "consolidated": true,
        "merged_memories": cluster.len(),
        "original_importance": cluster[0].importance,
    });
    merged
}

/// Summary of a consolidation pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsolidationSummary {
    pub agent_id: Uuid,
    pub candidates: u64,
    pub merged: u64,
    pub removed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;

    #[tokio::test]
    async fn test_add_and_query() {
        let manager = MemoryManager::new(MemoryStore::in_memory().unwrap());
        let agent = Uuid::new_v4();
        manager
            .add_memory(
                agent,
                "The user prefers dark mode in their editor".to_string(),
                "test".to_string(),
                "preference".to_string(),
                0.7,
                serde_json::Value::Null,
            )
            .await
            .unwrap();

        let results = manager.search(&agent, "dark mode", 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(manager.memory_count(&agent), 1);
    }

    #[tokio::test]
    async fn test_per_agent_isolation() {
        let manager = MemoryManager::new(MemoryStore::in_memory().unwrap());
        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();
        manager
            .add_memory(
                agent_a,
                "alpha secret".to_string(),
                "test".to_string(),
                "episodic".to_string(),
                0.5,
                serde_json::Value::Null,
            )
            .await
            .unwrap();

        let results = manager.search(&agent_b, "secret", 10).await.unwrap();
        assert!(results.is_empty());
        let results = manager.search(&agent_a, "secret", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_capture_turn_enabled_disabled() {
        let manager = MemoryManager::new(MemoryStore::in_memory().unwrap());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            Message::user("I prefer concise replies"),
            Message::assistant("Understood."),
        ];
        let ids = manager
            .capture_turn(session, agent, &messages, serde_json::json!({}))
            .await
            .unwrap();
        assert!(!ids.is_empty());

        let disabled = MemoryManager::new(MemoryStore::in_memory().unwrap()).with_capture_config(
            CaptureConfig {
                enabled: false,
                ..Default::default()
            },
        );
        let ids = disabled
            .capture_turn(session, agent, &messages, serde_json::json!({}))
            .await
            .unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_consolidate() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = MemoryManager::new(store.clone());
        let agent = Uuid::new_v4();
        // Insert two near-duplicate memories synchronously through the store.
        let entry = |content: &str| -> MemoryEntry {
            MemoryEntry::new(
                MemoryId::new(),
                agent,
                content.to_string(),
                "test".to_string(),
                "episodic".to_string(),
                0.5,
                serde_json::Value::Null,
            )
        };
        store
            .insert_memory(&entry("the user likes to use rust for async work"))
            .unwrap();
        store
            .insert_memory(&entry("the user likes to use rust for concurrency work"))
            .unwrap();
        store
            .insert_memory(&entry("completely unrelated python topic"))
            .unwrap();

        let summary = manager.consolidate(&agent, 0.3).unwrap();
        assert!(summary.merged >= 1);
        assert!(summary.removed >= 1);
        // The unrelated memory survives.
        assert_eq!(
            manager.list_memories(&agent, None, 100, 0).unwrap().len(),
            2
        );
    }

    #[test]
    fn test_importance_scoring() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = MemoryManager::new(store);
        let mut entry = MemoryEntry::new(
            MemoryId::new(),
            Uuid::new_v4(),
            "important recent fact".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.8,
            serde_json::Value::Null,
        );
        entry.access_count = 10;
        let score = manager.score_importance(&entry, 24.0);
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_expiry() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = MemoryManager::new(store.clone());
        let agent = Uuid::new_v4();
        let mut old = MemoryEntry::new(
            MemoryId::new(),
            agent,
            "old unimportant".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.1,
            serde_json::Value::Null,
        );
        old.created_at = Utc::now() - Duration::days(400);
        store.insert_memory(&old).unwrap();

        let deleted = manager.expire_with(Duration::days(30), 0.5).unwrap();
        assert_eq!(deleted, 1);
    }
}
