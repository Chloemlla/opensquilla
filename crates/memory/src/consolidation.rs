//! # Memory consolidation
//!
//! Operational consolidation for the memory store, distinct from the
//! LLM-driven dream engine:
//!
//! - **Deduplication** — find and merge memories that say the same thing
//!   (high content similarity, same memory type).
//! - **Clustering** — group memories into semantic clusters by content
//!   similarity and summarize the centroid topic for each.
//! - **Memory aging** — apply a decay policy that fades low-importance,
//!   rarely-accessed memories toward expiry, and optionally expires them.
//!
//! The [`Consolidator`] orchestrates all three passes and reports what
//! happened in a [`ConsolidationReport`].

use chrono::{DateTime, Utc};
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::retrieval::{content_similarity, cosine_similarity};
use crate::store::MemoryStore;
use crate::types::MemoryEntry;

/// Configuration for [`Consolidator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    /// Minimum content similarity for two memories to be considered
    /// duplicates.
    pub dedup_threshold: f64,
    /// Minimum content similarity for two memories to join a cluster.
    pub cluster_threshold: f64,
    /// Minimum cluster size before a centroid is materialized.
    pub min_cluster_size: usize,
    /// Whether to delete duplicate memories (`false` keeps both but tags them).
    pub delete_duplicates: bool,
    /// Whether to run the aging pass during consolidation.
    pub run_aging: bool,
    /// Age (in hours) after which a low-importance memory becomes "aged".
    pub age_hours: f64,
    /// Access-count floor below which a memory is considered cold.
    pub min_access_count: u64,
    /// Importance floor below which a memory is considered low-salience.
    pub min_importance: f64,
    /// Whether to actually delete aged memories (`false` only marks them).
    pub delete_aged: bool,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            dedup_threshold: 0.85,
            cluster_threshold: 0.5,
            min_cluster_size: 3,
            delete_duplicates: true,
            run_aging: true,
            age_hours: 24.0 * 30.0, // ~1 month
            min_access_count: 1,
            min_importance: 0.3,
            delete_aged: false,
        }
    }
}

/// A pair of duplicate memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicatePair {
    /// The memory that is kept.
    pub keep_id: MemoryId,
    /// The memory that is merged/deleted.
    pub duplicate_id: MemoryId,
    /// Content similarity between the two.
    pub similarity: f64,
    /// Whether the duplicate was removed.
    pub removed: bool,
}

/// A semantic cluster of memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCluster {
    /// The agent owning the cluster.
    pub agent_id: Uuid,
    /// Member memory ids.
    pub memory_ids: Vec<MemoryId>,
    /// The dominant memory type.
    pub memory_type: String,
    /// Average pairwise similarity within the cluster.
    pub similarity: f64,
    /// The most frequent terms (a cheap centroid proxy).
    pub common_terms: Vec<String>,
    /// Whether a centroid memory was materialized for this cluster.
    pub centroid_created: bool,
}

/// A single memory's aging assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgingAssessment {
    pub memory_id: MemoryId,
    pub age_hours: f64,
    pub access_count: u64,
    pub importance: f64,
    /// Whether the memory meets the aging criteria.
    pub aged: bool,
    /// Whether the memory was expired.
    pub expired: bool,
}

/// The aggregate result of a consolidation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Duplicate pairs found.
    pub duplicates: Vec<DuplicatePair>,
    /// Clusters formed.
    pub clusters: Vec<MemoryCluster>,
    /// Aging assessments produced.
    pub aging: Vec<AgingAssessment>,
    /// Memories deleted as duplicates.
    pub duplicates_deleted: u64,
    /// Centroid memories created.
    pub centroids_created: u64,
    /// Memories expired by aging.
    pub aged_expired: u64,
    /// When the pass ran.
    pub ran_at: DateTime<Utc>,
}

impl Default for ConsolidationReport {
    fn default() -> Self {
        Self {
            duplicates: Vec::new(),
            clusters: Vec::new(),
            aging: Vec::new(),
            duplicates_deleted: 0,
            centroids_created: 0,
            aged_expired: 0,
            ran_at: Utc::now(),
        }
    }
}

impl ConsolidationReport {
    /// Whether anything changed in the store.
    pub fn changed(&self) -> bool {
        self.duplicates_deleted > 0 || self.centroids_created > 0 || self.aged_expired > 0
    }
}

/// The consolidation engine.
#[derive(Debug, Clone)]
pub struct Consolidator {
    store: MemoryStore,
    config: ConsolidationConfig,
}

impl Consolidator {
    /// Create a consolidator over the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            config: ConsolidationConfig::default(),
        }
    }

    /// Create a consolidator with a custom config.
    pub fn with_config(store: MemoryStore, config: ConsolidationConfig) -> Self {
        Self { store, config }
    }

    /// The active configuration.
    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    /// The underlying store.
    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Run the full consolidation pass for an agent: dedupe, cluster, age.
    pub fn consolidate_agent(&self, agent_id: &Uuid) -> CoreResult<ConsolidationReport> {
        let memories = self.store.list_memories(agent_id, None, u64::MAX, 0)?;
        let mut report = ConsolidationReport {
            ran_at: Utc::now(),
            ..Default::default()
        };

        // Pass 1: deduplicate.
        let duplicates = self.deduplicate(agent_id, &memories)?;
        report.duplicates.extend(duplicates.iter().cloned());
        report.duplicates_deleted = duplicates.iter().filter(|d| d.removed).count() as u64;

        // Reload so later passes ignore removed rows.
        let remaining = self.store.list_memories(agent_id, None, u64::MAX, 0)?;

        // Pass 2: cluster.
        let clusters = self.cluster(agent_id, &remaining)?;
        report.clusters.extend(clusters.iter().cloned());
        report.centroids_created = clusters.iter().filter(|c| c.centroid_created).count() as u64;

        // Pass 3: aging.
        if self.config.run_aging {
            let aging = self.assess_aging(&remaining)?;
            report.aged_expired = aging.iter().filter(|a| a.expired).count() as u64;
            report.aging.extend(aging);
        }

        Ok(report)
    }

    // -----------------------------------------------------------------------
    // Deduplication
    // -----------------------------------------------------------------------

    /// Find near-duplicate memories for an agent. When
    /// [`ConsolidationConfig::delete_duplicates`] is set, the lower-importance
    /// member of each pair is deleted (with provenance recorded in metadata).
    pub fn deduplicate(
        &self,
        agent_id: &Uuid,
        memories: &[MemoryEntry],
    ) -> CoreResult<Vec<DuplicatePair>> {
        let mut pairs: Vec<DuplicatePair> = Vec::new();
        let mut removed: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();

        for i in 0..memories.len() {
            if removed.contains(&memories[i].id) {
                continue;
            }
            for j in (i + 1)..memories.len() {
                if removed.contains(&memories[j].id) {
                    continue;
                }
                let a = &memories[i];
                let b = &memories[j];
                // Only dedupe within the same memory type by default.
                if a.memory_type != b.memory_type {
                    continue;
                }
                let similarity = content_similarity(&a.content, &b.content);
                if similarity >= self.config.dedup_threshold {
                    // Keep the higher-importance, more-recently-updated one.
                    let (keep, duplicate) = if b.importance > a.importance
                        || (b.importance == a.importance && b.updated_at >= a.updated_at)
                    {
                        (b, a)
                    } else {
                        (a, b)
                    };

                    let mut removed_flag = false;
                    if self.config.delete_duplicates {
                        // Record provenance on the kept memory.
                        let mut kept = keep.clone();
                        let mut meta = kept
                            .metadata
                            .as_object()
                            .cloned()
                            .unwrap_or_default();
                        let mut dup_list = meta
                            .get("merged_duplicates")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        dup_list.push(serde_json::json!(duplicate.id.0.to_string()));
                        meta.insert("merged_duplicates".to_string(), serde_json::Value::Array(dup_list));
                        kept.metadata = serde_json::Value::Object(meta);
                        kept.updated_at = Utc::now();
                        self.store.update_memory(&kept)?;

                        self.store.delete_memory(&duplicate.id)?;
                        removed.insert(duplicate.id);
                        removed_flag = true;
                    }

                    pairs.push(DuplicatePair {
                        keep_id: keep.id,
                        duplicate_id: duplicate.id,
                        similarity,
                        removed: removed_flag,
                    });
                }
            }
        }

        let _ = agent_id;
        Ok(pairs)
    }

    // -----------------------------------------------------------------------
    // Clustering
    // -----------------------------------------------------------------------

    /// Group memories into semantic clusters by content similarity.
    pub fn cluster(
        &self,
        agent_id: &Uuid,
        memories: &[MemoryEntry],
    ) -> CoreResult<Vec<MemoryCluster>> {
        let mut visited: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
        let mut clusters: Vec<MemoryCluster> = Vec::new();

        for i in 0..memories.len() {
            if visited.contains(&memories[i].id) {
                continue;
            }
            let mut cluster_members: Vec<MemoryEntry> = vec![memories[i].clone()];
            visited.insert(memories[i].id);

            for j in (i + 1)..memories.len() {
                if visited.contains(&memories[j].id) {
                    continue;
                }
                // A memory joins if it is similar to ANY member (transitive
                // clustering), but capped to keep clusters tight.
                let max_sim = cluster_members
                    .iter()
                    .map(|m| content_similarity(&m.content, &memories[j].content))
                    .fold(0.0_f64, f64::max);
                if max_sim >= self.config.cluster_threshold {
                    cluster_members.push(memories[j].clone());
                    visited.insert(memories[j].id);
                }
            }

            if cluster_members.len() >= self.config.min_cluster_size {
                let cluster = self.build_cluster(agent_id, &cluster_members);
                clusters.push(cluster);
            }
        }

        clusters.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        Ok(clusters)
    }

    /// Materialize a centroid memory for a cluster if it is large enough and
    /// does not already exist. Returns the created memory id.
    pub fn materialize_centroid(&self, cluster: &MemoryCluster) -> CoreResult<Option<MemoryId>> {
        if cluster.centroid_created || cluster.memory_ids.len() < self.config.min_cluster_size {
            return Ok(None);
        }
        let centroid_content = format!(
            "[Cluster centroid] {} memories about: {}",
            cluster.memory_ids.len(),
            cluster.common_terms.join(", ")
        );
        let entry = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            cluster.agent_id,
            centroid_content,
            "consolidation".to_string(),
            format!("cluster_{}", cluster.memory_type),
            // Clusters are moderately salient.
            0.6,
            serde_json::json!({
                "cluster": true,
                "member_ids": cluster.memory_ids.iter().map(|m| m.0.to_string()).collect::<Vec<_>>(),
                "similarity": cluster.similarity,
                "consolidated_at": Utc::now().to_rfc3339(),
            }),
        );
        self.store.insert_memory(&entry)?;
        Ok(Some(entry.id))
    }

    fn build_cluster(&self, agent_id: &Uuid, members: &[MemoryEntry]) -> MemoryCluster {
        let mut total_sim = 0.0;
        let mut pairs = 0usize;
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                total_sim += content_similarity(&members[a].content, &members[b].content);
                pairs += 1;
            }
        }
        let similarity = if pairs > 0 {
            total_sim / pairs as f64
        } else {
            self.config.cluster_threshold
        };

        let memory_type = members[0].memory_type.clone();
        let common_terms = extract_common_terms(members);

        let cluster = MemoryCluster {
            agent_id: *agent_id,
            memory_ids: members.iter().map(|m| m.id).collect(),
            memory_type,
            similarity,
            common_terms,
            centroid_created: false,
        };
        cluster
    }

    // -----------------------------------------------------------------------
    // Memory aging
    // -----------------------------------------------------------------------

    /// Assess which memories have aged past the configured policy.
    pub fn assess_aging(&self, memories: &[MemoryEntry]) -> CoreResult<Vec<AgingAssessment>> {
        let now = Utc::now();
        let mut assessments = Vec::new();

        for memory in memories {
            let age_hours = (now - memory.created_at).num_minutes() as f64 / 60.0;
            let aged = age_hours >= self.config.age_hours
                && memory.access_count <= self.config.min_access_count
                && memory.importance < self.config.min_importance;

            let mut expired = false;
            if aged && self.config.delete_aged {
                // Record the expiry reason before deleting.
                let mut copy = memory.clone();
                if let Some(obj) = copy.metadata.as_object_mut() {
                    obj.insert(
                        "aged_out".to_string(),
                        serde_json::json!({
                            "age_hours": age_hours,
                            "access_count": memory.access_count,
                            "importance": memory.importance,
                            "expired_at": now.to_rfc3339(),
                        }),
                    );
                }
                let _ = self.store.update_memory(&copy);
                self.store.delete_memory(&memory.id)?;
                expired = true;
            }

            assessments.push(AgingAssessment {
                memory_id: memory.id,
                age_hours,
                access_count: memory.access_count,
                importance: memory.importance,
                aged,
                expired,
            });
        }

        Ok(assessments)
    }

    /// Run only the dedup pass across all agents.
    pub fn deduplicate_all(&self) -> CoreResult<u64> {
        let memories = self.store.list_memories_by_agent_all()?;
        let mut by_agent: HashMap<Uuid, Vec<MemoryEntry>> = HashMap::new();
        for m in memories {
            by_agent.entry(m.agent_id).or_default().push(m);
        }
        let mut deleted = 0u64;
        for (agent, entries) in by_agent {
            let pairs = self.deduplicate(&agent, &entries)?;
            deleted += pairs.iter().filter(|p| p.removed).count() as u64;
        }
        Ok(deleted)
    }
}

/// Extract the most frequent meaningful terms from a set of memories.
fn extract_common_terms(members: &[MemoryEntry]) -> Vec<String> {
    let stop_words: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "and", "but", "or", "nor", "not", "so", "yet", "this",
        "that", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "our", "their",
    ]
    .iter()
    .cloned()
    .collect();

    let mut freq: HashMap<String, usize> = HashMap::new();
    for member in members {
        let words: std::collections::HashSet<String> = member
            .content
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 3 && !stop_words.contains(w))
            .map(String::from)
            .collect();
        for word in words {
            *freq.entry(word).or_insert(0) += 1;
        }
    }

    let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .take(6)
        .map(|(word, _)| word)
        .collect()
}

/// Compute the average embedding of a cluster, when any member has one.
/// Returns `None` when no member carries an embedding.
pub fn cluster_centroid_embedding(members: &[MemoryEntry]) -> Option<Vec<f32>> {
    let mut dims: Option<usize> = None;
    let mut sum: Vec<f64> = Vec::new();
    let mut count = 0usize;

    for member in members {
        if let Some(emb) = &member.embedding {
            if dims.is_none() {
                dims = Some(emb.len());
                sum = vec![0.0; emb.len()];
            }
            let d = dims.unwrap();
            if emb.len() != d {
                continue;
            }
            for (s, v) in sum.iter_mut().zip(emb.iter()) {
                *s += *v as f64;
            }
            count += 1;
        }
    }

    if count == 0 {
        None
    } else {
        Some(sum.iter().map(|s| (*s / count as f64) as f32).collect())
    }
}

/// Pairwise cosine similarity between two embeddings, or `None` if either is
/// missing.
pub fn embedding_similarity(a: Option<&[f32]>, b: Option<&[f32]>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(cosine_similarity(a, b)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::MemoryId;

    fn insert(store: &MemoryStore, agent: Uuid, content: &str, mem_type: &str, importance: f64) {
        let entry = MemoryEntry::new(
            MemoryId::new(),
            agent,
            content.to_string(),
            "test".to_string(),
            mem_type.to_string(),
            importance,
            serde_json::Value::Null,
        );
        store.insert_memory(&entry).unwrap();
    }

    #[test]
    fn dedup_removes_near_identical() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            dedup_threshold: 0.7,
            delete_duplicates: true,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let agent = Uuid::new_v4();
        insert(&store, agent, "user prefers dark mode in the editor", "preference", 0.5);
        insert(&store, agent, "user prefers dark mode in the editor", "preference", 0.4);
        insert(&store, agent, "user likes pizza", "preference", 0.5);

        let report = consolidator.consolidate_agent(&agent).unwrap();
        assert_eq!(report.duplicates_deleted, 1);
        let remaining = store.list_memories(&agent, None, 100, 0).unwrap();
        assert_eq!(remaining.len(), 2);
    }

    #[test]
    fn dedup_keeps_both_when_disabled() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            dedup_threshold: 0.8,
            delete_duplicates: false,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let agent = Uuid::new_v4();
        insert(&store, agent, "same exact content here", "episodic", 0.5);
        insert(&store, agent, "same exact content here", "episodic", 0.5);

        let pairs = consolidator.deduplicate(&agent, &store.list_memories(&agent, None, 100, 0).unwrap()).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!(!pairs[0].removed);
        assert_eq!(store.list_memories(&agent, None, 100, 0).unwrap().len(), 2);
    }

    #[test]
    fn cluster_groups_similar_memories() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            cluster_threshold: 0.4,
            min_cluster_size: 3,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let agent = Uuid::new_v4();
        insert(&store, agent, "user works with rust async runtimes", "episodic", 0.5);
        insert(&store, agent, "rust async is fast", "episodic", 0.5);
        insert(&store, agent, "async rust code patterns", "episodic", 0.5);
        insert(&store, agent, "unrelated cooking tip", "episodic", 0.5);

        let clusters = consolidator.cluster(&agent, &store.list_memories(&agent, None, 100, 0).unwrap()).unwrap();
        assert!(!clusters.is_empty());
        assert!(clusters[0].memory_ids.len() >= 3);
        assert!(!clusters[0].common_terms.is_empty());
    }

    #[test]
    fn materialize_centroid_creates_memory() {
        let store = MemoryStore::in_memory().unwrap();
        let consolidator = Consolidator::new(store.clone());
        let agent = Uuid::new_v4();
        let cluster = MemoryCluster {
            agent_id: agent,
            memory_ids: vec![MemoryId::new(), MemoryId::new(), MemoryId::new()],
            memory_type: "episodic".to_string(),
            similarity: 0.6,
            common_terms: vec!["rust".to_string(), "async".to_string()],
            centroid_created: false,
        };
        let id = consolidator.materialize_centroid(&cluster).unwrap();
        assert!(id.is_some());
        let stored = store.get_memory(&id.unwrap()).unwrap().unwrap();
        assert_eq!(stored.source, "consolidation");
    }

    #[test]
    fn aging_marks_old_cold_memories() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            run_aging: true,
            age_hours: 1.0,
            min_access_count: 1,
            min_importance: 0.5,
            delete_aged: false,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let agent = Uuid::new_v4();

        let mut entry = MemoryEntry::new(
            MemoryId::new(),
            agent,
            "old cold memory".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.1,
            serde_json::Value::Null,
        );
        // Backdate it.
        entry.created_at = Utc::now() - chrono::Duration::days(30);
        store.insert_memory(&entry).unwrap();

        let assessments = consolidator
            .assess_aging(&store.list_memories(&agent, None, 100, 0).unwrap())
            .unwrap();
        assert_eq!(assessments.len(), 1);
        assert!(assessments[0].aged);
        assert!(!assessments[0].expired);
        // Still present because delete_aged is false.
        assert_eq!(store.list_memories(&agent, None, 100, 0).unwrap().len(), 1);
    }

    #[test]
    fn aging_deletes_when_configured() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            run_aging: true,
            age_hours: 1.0,
            min_access_count: 1,
            min_importance: 0.5,
            delete_aged: true,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let agent = Uuid::new_v4();
        let mut entry = MemoryEntry::new(
            MemoryId::new(),
            agent,
            "expired memory".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.1,
            serde_json::Value::Null,
        );
        entry.created_at = Utc::now() - chrono::Duration::days(30);
        store.insert_memory(&entry).unwrap();

        let report = consolidator.consolidate_agent(&agent).unwrap();
        assert_eq!(report.aged_expired, 1);
        assert!(store.list_memories(&agent, None, 100, 0).unwrap().is_empty());
    }

    #[test]
    fn deduplicate_all_across_agents() {
        let store = MemoryStore::in_memory().unwrap();
        let config = ConsolidationConfig {
            dedup_threshold: 0.8,
            delete_duplicates: true,
            ..Default::default()
        };
        let consolidator = Consolidator::with_config(store.clone(), config);
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        insert(&store, a, "identical phrase one", "episodic", 0.5);
        insert(&store, a, "identical phrase one", "episodic", 0.5);
        insert(&store, b, "identical phrase one", "episodic", 0.5);
        insert(&store, b, "identical phrase one", "episodic", 0.5);

        let deleted = consolidator.deduplicate_all().unwrap();
        assert_eq!(deleted, 2);
    }

    #[test]
    fn cluster_centroid_embedding_averages() {
        let mut a = MemoryEntry::new(
            MemoryId::new(),
            Uuid::new_v4(),
            "a".to_string(),
            "t".to_string(),
            "e".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        a.embedding = Some(vec![1.0, 2.0]);
        let mut b = MemoryEntry::new(
            MemoryId::new(),
            a.agent_id,
            "b".to_string(),
            "t".to_string(),
            "e".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        b.embedding = Some(vec![3.0, 4.0]);

        let agent_id = a.agent_id;
        let centroid = cluster_centroid_embedding(&[a, b]).unwrap();
        assert_eq!(centroid, vec![2.0, 3.0]);

        let bare = MemoryEntry::new(
            MemoryId::new(),
            agent_id,
            "c".to_string(),
            "t".to_string(),
            "e".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        assert!(cluster_centroid_embedding(&[bare]).is_none());
    }

    #[test]
    fn embedding_similarity_handles_missing() {
        assert!(embedding_similarity(None, Some(&[1.0])).is_none());
        assert!(embedding_similarity(Some(&[1.0]), None).is_none());
        let sim = embedding_similarity(Some(&[1.0, 0.0]), Some(&[1.0, 0.0])).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }
}
