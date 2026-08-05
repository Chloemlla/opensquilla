use opensquilla_core::result::CoreResult;
use opensquilla_core::types::MemoryId;
use std::collections::HashSet;
use uuid::Uuid;

use crate::store::MemoryStore;
use crate::types::{MemoryEntry, MemoryFilters, MemorySearchResult};

/// Tuning knobs for hybrid retrieval.
#[derive(Debug, Clone)]
pub struct HybridSearchConfig {
    /// Weight of the vector (cosine) component in `[0,1]`.
    pub vector_weight: f64,
    /// Weight of the BM25 (FTS5 rank) component in `[0,1]`.
    pub bm25_weight: f64,
    /// Weight of the access-frequency boost.
    pub access_boost_weight: f64,
    /// Exponential time-decay half-life in hours.
    pub time_decay_hours: f64,
    /// `lambda` for MMR; `1.0` = pure relevance, `0.0` = pure diversity.
    pub mmr_lambda: f64,
    /// Minimum combined score (`[0,1]`) for a result to be returned.
    pub min_score: f64,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.5,
            bm25_weight: 0.5,
            access_boost_weight: 0.1,
            time_decay_hours: 48.0,
            mmr_lambda: 0.7,
            min_score: 0.0,
        }
    }
}

/// Hybrid search combining vector similarity, BM25 (FTS5), time decay, and MMR.
#[derive(Debug, Clone)]
pub struct RetrievalEngine {
    store: MemoryStore,
    config: HybridSearchConfig,
}

impl RetrievalEngine {
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            config: HybridSearchConfig::default(),
        }
    }

    /// Replace the retrieval tuning knobs.
    pub fn with_config(mut self, config: HybridSearchConfig) -> Self {
        self.config = config;
        self
    }

    pub fn config(&self) -> &HybridSearchConfig {
        &self.config
    }

    /// Search by FTS5 full-text search only.
    pub fn search_fts(
        &self,
        query: &str,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<MemoryEntry>> {
        self.store.search_fts(query, limit, offset)
    }

    /// Search by vector similarity (cosine) only.
    pub fn search_vector(
        &self,
        agent_id: &Uuid,
        query_embedding: &[f32],
        top_k: usize,
    ) -> CoreResult<Vec<(MemoryEntry, f64)>> {
        let embeddings = self.store.get_all_embeddings(agent_id)?;

        let mut scored: Vec<(MemoryEntry, f64)> = embeddings
            .into_iter()
            .filter_map(|(memory_id, emb)| {
                let similarity = cosine_similarity(query_embedding, &emb);
                if similarity > 0.0 {
                    self.store.get_memory(&memory_id).ok().flatten().map(|entry| {
                        (entry, similarity)
                    })
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        Ok(scored)
    }

    /// Hybrid search: combine FTS5 BM25 and vector cosine scores with time decay.
    ///
    /// Retained as a convenience for callers that already have a query
    /// embedding and want a lightweight `(entry, score)` result.
    pub fn search_hybrid(
        &self,
        agent_id: &Uuid,
        query: &str,
        query_embedding: &[f32],
        top_k: usize,
        alpha: f64, // weight for vector score (0.0 = pure FTS5, 1.0 = pure vector)
        time_decay_hours: f64,
    ) -> CoreResult<Vec<(MemoryEntry, f64)>> {
        let fts_results = self.search_fts(query, 100, 0)?;
        let fts_ids: HashSet<MemoryId> = fts_results.iter().map(|e| e.id).collect();

        let vector_results = self.search_vector(agent_id, query_embedding, 100)?;
        let vector_ids: HashSet<MemoryId> = vector_results.iter().map(|(e, _)| e.id).collect();

        let all_ids: HashSet<MemoryId> = fts_ids.union(&vector_ids).copied().collect();

        let now = chrono::Utc::now();
        let mut combined: Vec<(MemoryEntry, f64)> = all_ids
            .iter()
            .filter_map(|id| {
                if let Ok(Some(entry)) = self.store.get_memory(id) {
                    let fts_score = fts_results
                        .iter()
                        .position(|e| e.id == *id)
                        .map(|p| 1.0 - (p as f64 / fts_results.len() as f64))
                        .unwrap_or(0.0);

                    let vec_score = vector_results
                        .iter()
                        .position(|(e, _)| e.id == *id)
                        .map(|p| vector_results[p].1)
                        .unwrap_or(0.0);

                    // Time decay: newer entries get higher score.
                    let age_hours = (now - entry.created_at).num_hours().max(0) as f64;
                    let time_boost = (-age_hours / time_decay_hours).exp();

                    // Access frequency boost.
                    let access_boost = (entry.access_count as f64).ln_1p() * 0.1;

                    let combined_score = (alpha * vec_score + (1.0 - alpha) * fts_score)
                        * time_boost
                        + access_boost;

                    Some((entry, combined_score))
                } else {
                    None
                }
            })
            .collect();

        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(top_k);

        Ok(combined)
    }

    /// Complete hybrid retrieval pipeline:
    ///
    /// 1. Vector similarity (cosine) over stored embeddings.
    /// 2. BM25 via FTS5 `rank` (normalized to `[0,1]`).
    /// 3. Exponential time decay toward recency.
    /// 4. Access-frequency boost.
    /// 5. Optional MMR diversity re-ranking.
    ///
    /// Results are returned as [`MemorySearchResult`] with a normalized score.
    pub fn search(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        agent_id: Option<&Uuid>,
        filters: &MemoryFilters,
        top_k: usize,
    ) -> CoreResult<Vec<MemorySearchResult>> {
        let cfg = &self.config;

        // --- Vector candidates (cosine) ---
        let vector_scored: Vec<(MemoryEntry, f64)> = match (query_embedding, agent_id) {
            (Some(qe), Some(aid)) => self.search_vector(aid, qe, 200)?,
            _ => Vec::new(),
        };

        // --- BM25 candidates (FTS5 rank) ---
        let bm25_scored: Vec<(MemoryEntry, f64)> = match agent_id {
            Some(aid) => self.store.search_fts_scored(aid, query, 200)?,
            None => Vec::new(),
        };

        // --- Merge into a unified map with normalized per-method scores ---
        // key: MemoryId -> (entry, normalized_vector, normalized_bm25)
        let mut merged: std::collections::HashMap<MemoryId, (MemoryEntry, f64, f64)> =
            std::collections::HashMap::new();

        let max_vec = vector_scored
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::MIN, f64::max);
        let max_bm25 = bm25_scored
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::MIN, f64::max);

        for (entry, score) in vector_scored {
            let normalized = if max_vec > 0.0 { score / max_vec } else { 0.0 };
            merged
                .entry(entry.id)
                .or_insert_with(|| (entry, 0.0, 0.0))
                .1 = normalized;
        }

        for (entry, rank) in bm25_scored {
            // FTS5 `rank` is negative; larger (closer to zero) is better.
            // Normalize to [0,1] via min-max over the returned set.
            let normalized = if max_bm25 > 0.0 { rank / max_bm25 } else { 0.0 };
            let entry_id = entry.id;
            let slot = merged.entry(entry_id).or_insert_with(|| (entry, 0.0, 0.0));
            slot.2 = normalized.clamp(0.0, 1.0);
        }

        let now = chrono::Utc::now();
        let mut results: Vec<MemorySearchResult> = Vec::new();
        for (entry, v_score, b_score) in merged.into_values() {
            // Skip entries failing the supplied filters.
            if !filters.matches(&entry) {
                continue;
            }
            if let Some(aid) = agent_id {
                if entry.agent_id != *aid {
                    continue;
                }
            }

            // Time decay: exponential toward recency.
            let age_hours = (now - entry.created_at).num_hours().max(0) as f64;
            let time_boost = (-age_hours / cfg.time_decay_hours).exp();

            // Access-frequency boost.
            let access_boost = (entry.access_count as f64).ln_1p() * cfg.access_boost_weight;

            let combined = (cfg.vector_weight * v_score + cfg.bm25_weight * b_score) * time_boost
                + access_boost;

            // Normalize combined score to [0,1] by saturating at 1.0.
            let combined_norm = combined.clamp(0.0, 1.0);
            if combined_norm < cfg.min_score {
                continue;
            }

            results.push(MemorySearchResult::new(entry, combined_norm, "hybrid"));
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Optional MMR diversity re-ranking.
        if cfg.mmr_lambda < 1.0 {
            let diversified = self.mmr_rerank_scored(results, cfg.mmr_lambda);
            return Ok(diversified.into_iter().take(top_k).collect());
        }

        results.truncate(top_k);
        Ok(results)
    }

    /// Maximal Marginal Relevance (MMR) re-ranking for diversity.
    ///
    /// Operates on `(entry, score)` pairs; preserves the entry's score.
    pub fn mmr_rerank(
        &self,
        results: Vec<(MemoryEntry, f64)>,
        _query_embedding: &[f32],
        top_k: usize,
        lambda: f64, // 0.0 = pure diversity, 1.0 = pure relevance
    ) -> Vec<(MemoryEntry, f64)> {
        if results.is_empty() {
            return results;
        }

        let mut selected: Vec<(MemoryEntry, f64)> = Vec::new();
        let mut remaining = results;

        while selected.len() < top_k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (i, (entry, relevance)) in remaining.iter().enumerate() {
                let relevance_score = *relevance;

                // Diversity penalty: max similarity to already selected.
                let max_sim_to_selected = selected
                    .iter()
                    .map(|(sel_entry, _)| content_similarity(&entry.content, &sel_entry.content))
                    .fold(0.0_f64, f64::max);

                let mmr_score = lambda * relevance_score - (1.0 - lambda) * max_sim_to_selected;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            let (entry, relevance) = remaining.remove(best_idx);
            selected.push((entry, relevance));
        }

        selected
    }

    /// MMR over [`MemorySearchResult`] values (preserves `source`).
    pub fn mmr_rerank_scored(
        &self,
        results: Vec<MemorySearchResult>,
        lambda: f64,
    ) -> Vec<MemorySearchResult> {
        if results.is_empty() {
            return results;
        }

        let mut selected: Vec<MemorySearchResult> = Vec::new();
        let mut remaining = results;

        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (i, result) in remaining.iter().enumerate() {
                let relevance = result.score;

                let max_sim_to_selected = selected
                    .iter()
                    .map(|sel| content_similarity(&result.entry.content, &sel.entry.content))
                    .fold(0.0_f64, f64::max);

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            selected.push(remaining.remove(best_idx));
        }

        selected
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm_a: f64 = a.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| *x as f64 * *x as f64).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Simple content similarity based on word overlap (Jaccard-like).
pub fn content_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::MemoryId;

    fn make_entry(agent_id: Uuid, content: &str) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId::new(),
            agent_id,
            content.to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        )
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_content_similarity() {
        assert_eq!(content_similarity("rust is great", "rust is great"), 1.0);
        assert_eq!(content_similarity("", ""), 0.0);
        assert!(content_similarity("rust lang", "python lang") > 0.0);
    }

    #[test]
    fn test_search_fts_roundtrip() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = RetrievalEngine::new(store.clone());
        let agent_id = Uuid::new_v4();
        store.insert_memory(&make_entry(agent_id, "Rust async runtime")).unwrap();
        store.insert_memory(&make_entry(agent_id, "Python scripting")).unwrap();

        let results = engine.search_fts("rust", 10, 0).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_pipeline_without_embeddings() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = RetrievalEngine::new(store.clone());
        let agent_id = Uuid::new_v4();
        store.insert_memory(&make_entry(agent_id, "memory management in rust")).unwrap();
        store.insert_memory(&make_entry(agent_id, "async tokio runtime")).unwrap();

        let results = engine
            .search("rust memory", None, Some(&agent_id), &MemoryFilters::default(), 10)
            .unwrap();
        // BM25 should find at least one hit.
        assert!(!results.is_empty());
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_filters_agent_id() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = RetrievalEngine::new(store.clone());
        let agent_a = Uuid::new_v4();
        let agent_b = Uuid::new_v4();
        store.insert_memory(&make_entry(agent_a, "tokio async")).unwrap();
        store.insert_memory(&make_entry(agent_b, "tokio async")).unwrap();

        let filters = MemoryFilters {
            agent_id: Some(agent_a),
            ..Default::default()
        };
        let results = engine
            .search("tokio", None, Some(&agent_a), &filters, 10)
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.entry.agent_id == agent_a));
    }

    #[test]
    fn test_mmr_rerank() {
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let agent_id = Uuid::new_v4();
        let results = vec![
            (make_entry(agent_id, "the quick brown fox"), 0.9),
            (make_entry(agent_id, "the quick brown dog"), 0.8),
            (make_entry(agent_id, "completely different topic"), 0.7),
        ];
        let reranked = engine.mmr_rerank(results, &[0.5; 8], 2, 0.5);
        assert_eq!(reranked.len(), 2);
    }
}
