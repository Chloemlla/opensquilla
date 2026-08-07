//! Dream consolidation engine.
//!
//! Periodically consolidates memories, identifies patterns, and creates
//! high-level abstractions — the "sleeping" phase of the memory system. The
//! [`DreamEngine`] runs a full dream cycle ([`DreamEngine::run_dream_cycle`])
//! that:
//!
//! 1. Identifies clusters of similar memories
//!    ([`DreamEngine::identify_consolidation_candidates`]).
//! 2. Generates a consolidated memory for each cluster, either via an
//!    injected LLM-backed [`DreamConsolidator`] or a deterministic heuristic.
//! 3. Applies the consolidated memories to the store
//!    ([`DreamEngine::apply_consolidation`]).
//! 4. Optionally prunes contradictory memories
//!    ([`DreamEngine::prune_contradictions`]).
//!
//! Progress is surfaced through [`DreamEvent`] records that callers can drain
//! for monitoring.

use chrono::{DateTime, Utc};
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::MemoryId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::retrieval::{RetrievalEngine, content_similarity};
use crate::store::MemoryStore;
use crate::types::MemoryEntry;

// ---------------------------------------------------------------------------
// Evidence-gated promotion pipeline (parity stubs)
//
// The Python `src/opensquilla/memory/dream/` package splits the dream feature
// into candidate scanning, quarantine, evidence, ranking, rehydration, curated
// apply, prompts and receipts. The engine below implements the
// clustering/merging core; the submodules below are parity stubs that mirror
// the Python dataclasses and function signatures so the promotion pipeline can
// be ported incrementally. Each stub carries a `TODO(parity)` marker.
// ---------------------------------------------------------------------------

pub mod curated_apply;
pub mod evidence;
pub mod models;
pub mod prompts;
pub mod quarantine;
pub mod ranking;
pub mod receipts;
pub mod rehydrate;
pub mod runner;

/// Configuration for the [`DreamEngine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamConfig {
    /// How often (in hours) a dream cycle is due.
    pub frequency_hours: f64,
    /// Minimum number of memories before consolidation candidates are sought.
    pub min_memories: usize,
    /// Content-similarity threshold for two memories to join a cluster.
    pub consolidation_threshold: f64,
    /// Maximum number of consolidation candidates per cycle.
    pub max_candidates: usize,
    /// Maximum number of contradictions examined per cycle.
    pub max_contradictions: usize,
    /// Minimum content overlap before a pair is considered for contradiction.
    pub contradiction_overlap: f64,
    /// Minimum conflict score for a pair to be treated as a contradiction.
    pub contradiction_threshold: f64,
    /// Whether to prune contradictions during a dream cycle.
    pub prune_contradictions: bool,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            frequency_hours: 24.0,
            min_memories: 5,
            consolidation_threshold: 0.45,
            max_candidates: 20,
            max_contradictions: 10,
            contradiction_overlap: 0.2,
            contradiction_threshold: 0.35,
            prune_contradictions: true,
        }
    }
}

/// A cluster of similar memories that are candidates for consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationCandidate {
    pub agent_id: Uuid,
    pub memory_ids: Vec<MemoryId>,
    /// Average pairwise content similarity within the cluster.
    pub similarity: f64,
    /// The concatenated content of all cluster members.
    pub cluster_content: String,
    /// The dominant memory type of the cluster.
    pub memory_type: String,
}

/// A consolidated memory produced by merging a cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedMemory {
    pub agent_id: Uuid,
    pub content: String,
    pub memory_type: String,
    pub importance: f64,
    pub tags: Vec<String>,
    /// The ids of the source memories this consolidated memory replaces.
    pub source_memory_ids: Vec<MemoryId>,
    pub metadata: serde_json::Value,
}

/// A pair of memories that appear to contradict each other.
#[derive(Debug, Clone)]
pub struct Contradiction {
    pub agent_id: Uuid,
    pub a: MemoryEntry,
    pub b: MemoryEntry,
    /// Content overlap between the two memories.
    pub overlap: f64,
    /// Detected conflict score in `[0, 1]`.
    pub conflict_score: f64,
}

/// Events emitted by the dream engine for monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DreamEvent {
    CycleStarted {
        agent_id: Uuid,
        timestamp: DateTime<Utc>,
    },
    CandidatesFound {
        agent_id: Uuid,
        count: usize,
    },
    ConsolidationCreated {
        agent_id: Uuid,
        memory_id: MemoryId,
        source_count: usize,
    },
    ContradictionResolved {
        agent_id: Uuid,
        kept: MemoryId,
        removed: MemoryId,
    },
    Pruned {
        agent_id: Uuid,
        removed_count: u64,
    },
    CycleComplete {
        agent_id: Uuid,
        summary: DreamSummary,
    },
}

/// LLM-backed consolidation generation.
///
/// The memory crate deliberately does not depend on the provider crate; any
/// caller (gateway, engine) can implement this trait and inject it via
/// [`DreamEngine::with_consolidator`]. A deterministic heuristic fallback is
/// provided so dreams still work without an LLM.
#[async_trait::async_trait]
pub trait DreamConsolidator: Send + Sync {
    /// Generate a consolidated memory for each candidate cluster.
    async fn consolidate(
        &self,
        candidates: &[ConsolidationCandidate],
    ) -> CoreResult<Vec<ConsolidatedMemory>>;

    /// Optionally resolve a contradiction into a merged memory. Returning
    /// `Ok(None)` means "keep the higher-quality memory and drop the other".
    async fn resolve_contradiction(
        &self,
        a: &MemoryEntry,
        b: &MemoryEntry,
    ) -> CoreResult<Option<ConsolidatedMemory>> {
        let _ = (a, b);
        Ok(None)
    }

    /// A short label used in logs.
    fn name(&self) -> &str {
        "dream-consolidator"
    }
}

/// Deterministic, heuristic consolidator used when no LLM is configured.
pub struct HeuristicConsolidator;

#[async_trait::async_trait]
impl DreamConsolidator for HeuristicConsolidator {
    async fn consolidate(
        &self,
        candidates: &[ConsolidationCandidate],
    ) -> CoreResult<Vec<ConsolidatedMemory>> {
        Ok(heuristic_consolidate(candidates))
    }

    async fn resolve_contradiction(
        &self,
        a: &MemoryEntry,
        b: &MemoryEntry,
    ) -> CoreResult<Option<ConsolidatedMemory>> {
        Ok(Some(ConsolidatedMemory {
            agent_id: a.agent_id,
            content: format!(
                "[Dream] Resolved contradiction between two memories:\n{}\n---\n{}",
                a.content, b.content
            ),
            memory_type: "dream_resolved".to_string(),
            importance: a.importance.max(b.importance),
            tags: vec!["dream".to_string(), "resolved".to_string()],
            source_memory_ids: vec![a.id, b.id],
            metadata: serde_json::json!({ "resolved": true }),
        }))
    }

    fn name(&self) -> &str {
        "heuristic"
    }
}

// ---------------------------------------------------------------------------
// LLM-backed consolidator
// ---------------------------------------------------------------------------

/// A chat callback used by [`LlmConsolidator`] to generate consolidation
/// text. The memory crate does not depend on the provider crate; callers
/// inject a thin closure over their LLM client.
#[async_trait::async_trait]
pub trait DreamLlm: Send + Sync {
    /// Complete a single chat turn and return the assistant's text.
    async fn complete(&self, system: &str, user: &str) -> CoreResult<String>;
}

/// An LLM-backed consolidator. Builds structured prompts from consolidation
/// candidates and asks the injected [`DreamLlm`] to produce merged memories.
pub struct LlmConsolidator {
    llm: Arc<dyn DreamLlm>,
    language: String,
}

impl LlmConsolidator {
    pub fn new(llm: Arc<dyn DreamLlm>) -> Self {
        Self {
            llm,
            language: "en".to_string(),
        }
    }

    /// Set the output language for generated consolidations.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    fn build_consolidation_prompt(
        &self,
        candidates: &[ConsolidationCandidate],
    ) -> (String, String) {
        let system = if self.language.starts_with("zh") {
            "你是一个记忆整合助手。把相似的记忆合并成一条简洁、信息密集的摘要。\
             保留姓名、数字、日期等关键细节。为每条候选输出一行 JSON 对象。"
        } else {
            "You are a memory consolidation assistant. Merge similar memories into a single \
             concise, information-dense summary. Preserve names, numbers, dates, and concrete \
             details. Output one JSON object per candidate on a single line."
        };

        let mut user = String::from("Consolidate these memory clusters:\n\n");
        for (i, candidate) in candidates.iter().enumerate() {
            user.push_str(&format!(
                "[Cluster {}] type={} similarity={:.2} members={}\n{}\n\n",
                i,
                candidate.memory_type,
                candidate.similarity,
                candidate.memory_ids.len(),
                candidate.cluster_content
            ));
        }
        user.push_str(
            "Return a JSON array of objects, each with fields: \
             \"content\" (string), \"importance\" (number 0-1), \"tags\" (array of strings).",
        );
        (system.to_string(), user)
    }

    fn parse_response(&self, raw: &str) -> Vec<serde_json::Value> {
        // Extract the first JSON array from the response (tolerant of
        // markdown fences and prose around it).
        if let Some(start) = raw.find('[') {
            if let Some(end) = raw.rfind(']') {
                let slice = &raw[start..=end];
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(slice) {
                    if let serde_json::Value::Array(items) = v {
                        return items;
                    }
                }
            }
        }
        serde_json::from_str(raw)
            .map(|v: serde_json::Value| match v {
                serde_json::Value::Array(items) => items,
                other => vec![other],
            })
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl DreamConsolidator for LlmConsolidator {
    async fn consolidate(
        &self,
        candidates: &[ConsolidationCandidate],
    ) -> CoreResult<Vec<ConsolidatedMemory>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let (system, user) = self.build_consolidation_prompt(candidates);
        let raw = self.llm.complete(&system, &user).await?;
        let parsed = self.parse_response(&raw);

        let mut out = Vec::new();
        for (i, item) in parsed.iter().enumerate() {
            let candidate = &candidates[i.min(candidates.len() - 1)];
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| candidate.cluster_content.clone());
            let importance = item
                .get("importance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.7)
                .clamp(0.0, 1.0);
            let tags: Vec<String> = item
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|t| t.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            out.push(ConsolidatedMemory {
                agent_id: candidate.agent_id,
                content,
                memory_type: format!("dream_{}", candidate.memory_type),
                importance,
                tags,
                source_memory_ids: candidate.memory_ids.clone(),
                metadata: serde_json::json!({
                    "similarity": candidate.similarity,
                    "consolidation_method": "llm",
                    "language": self.language,
                }),
            });
        }

        // If the LLM produced fewer results than candidates, fill the rest
        // deterministically so no candidate is lost.
        if out.len() < candidates.len() {
            let fillers = heuristic_consolidate(&candidates[out.len()..]);
            out.extend(fillers);
        }
        Ok(out)
    }

    async fn resolve_contradiction(
        &self,
        a: &MemoryEntry,
        b: &MemoryEntry,
    ) -> CoreResult<Option<ConsolidatedMemory>> {
        let system = if self.language.starts_with("zh") {
            "你是一个记忆仲裁助手。两个记忆相互矛盾，请生成一条解决矛盾的合并记忆。"
        } else {
            "You are a memory arbitration assistant. Two memories contradict each other. \
             Generate a single merged memory that resolves the conflict."
        };
        let user = format!(
            "Memory A: {}\n\nMemory B: {}\n\nReturn a JSON object with \
             fields: \"content\" (string), \"importance\" (number), \"tags\" (array).",
            a.content, b.content
        );
        let raw = self.llm.complete(system, &user).await?;
        let parsed = self.parse_response(&raw);
        if parsed.is_empty() {
            return Ok(None);
        }
        let item = &parsed[0];
        let content = item
            .get("content")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                format!(
                    "[Dream] Resolved contradiction between two memories:\n{}\n---\n{}",
                    a.content, b.content
                )
            });
        let importance = item
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(a.importance.max(b.importance))
            .clamp(0.0, 1.0);
        let tags: Vec<String> = item
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec!["dream".to_string(), "resolved".to_string()]);

        Ok(Some(ConsolidatedMemory {
            agent_id: a.agent_id,
            content,
            memory_type: "dream_resolved".to_string(),
            importance,
            tags,
            source_memory_ids: vec![a.id, b.id],
            metadata: serde_json::json!({
                "resolved": true,
                "consolidation_method": "llm",
            }),
        }))
    }

    fn name(&self) -> &str {
        "llm"
    }
}

// ---------------------------------------------------------------------------
// Memory merging and importance scoring
// ---------------------------------------------------------------------------

/// Merge two memories into a single [`MemoryEntry`]. The merged entry keeps
/// the higher importance, the union of tags, the more recent timestamps, and
/// a concatenated content body with provenance.
pub fn merge_memories(a: &MemoryEntry, b: &MemoryEntry) -> MemoryEntry {
    let mut content = String::new();
    content.push_str(&a.content);
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content.push_str("---\n");
    content.push_str(&b.content);

    let mut tags = a.tags.clone();
    for t in &b.tags {
        if !tags.contains(t) {
            tags.push(t.clone());
        }
    }

    let importance = a.importance.max(b.importance).clamp(0.0, 1.0);
    let created_at = a.created_at.min(b.created_at);
    let updated_at = chrono::Utc::now();

    let mut metadata = a.metadata.clone();
    if let Some(obj) = metadata.as_object_mut() {
        obj.insert(
            "merged_from".to_string(),
            serde_json::json!([a.id.0.to_string(), b.id.0.to_string()]),
        );
        obj.insert(
            "merged_at".to_string(),
            serde_json::json!(updated_at.to_rfc3339()),
        );
    } else {
        metadata = serde_json::json!({
            "merged_from": [a.id.0.to_string(), b.id.0.to_string()],
            "merged_at": updated_at.to_rfc3339(),
        });
    }

    let mut merged = MemoryEntry::new(
        opensquilla_core::types::MemoryId(uuid::Uuid::new_v4()),
        a.agent_id,
        content,
        "dream".to_string(),
        if a.memory_type == b.memory_type {
            a.memory_type.clone()
        } else {
            format!("merged_{}_{}", a.memory_type, b.memory_type)
        },
        importance,
        metadata,
    );
    merged.tags = tags;
    merged.created_at = created_at;
    merged.updated_at = updated_at;
    merged.importance_score = importance;
    merged
}

/// Heuristic importance scoring for a memory.
///
/// Combines several signals into a score in `[0, 1]`:
/// - recency (newer memories are initially more salient)
/// - length of content (a proxy for information content, with diminishing
///   returns beyond a band)
/// - number of tags
/// - an optional explicit base score.
pub fn score_importance(
    content: &str,
    created_at: chrono::DateTime<chrono::Utc>,
    tag_count: usize,
    base: f64,
) -> f64 {
    let now = chrono::Utc::now();
    let age_days = (now - created_at).num_days().max(0) as f64;
    // Recency: halve the score every 30 days.
    let recency = 0.5f64.powf(age_days / 30.0);

    let chars = content.chars().count() as f64;
    // Information-content proxy: grows to 1.0 around 300 chars then holds.
    let length = (chars / 300.0).min(1.0);

    let tag_boost = (tag_count as f64 * 0.1).min(0.3);

    let combined = base * 0.4 + recency * 0.3 + length * 0.2 + tag_boost;
    combined.clamp(0.0, 1.0)
}

/// Recompute the importance of every memory for an agent using
/// [`score_importance`], updating the store. Returns the number updated.
pub fn reindex_importance(
    store: &crate::store::MemoryStore,
    agent_id: &uuid::Uuid,
) -> CoreResult<usize> {
    let memories = store.list_memories(agent_id, None, u64::MAX, 0)?;
    let mut updated = 0usize;
    for mut memory in memories {
        let new_importance = score_importance(
            &memory.content,
            memory.created_at,
            memory.tags.len(),
            memory.importance,
        );
        if (new_importance - memory.importance).abs() > 0.001 {
            memory.set_importance(new_importance);
            store.update_memory(&memory)?;
            updated += 1;
        }
    }
    Ok(updated)
}

/// Build a merged memory from a set of cluster members.
pub fn merge_cluster_members(members: &[MemoryEntry]) -> Option<MemoryEntry> {
    let first = members.first()?;
    let second = members.get(1)?;
    let mut merged = merge_memories(first, second);
    for member in &members[2..] {
        merged = merge_memories(&merged, member);
    }
    Some(merged)
}

/// The consolidation engine.
pub struct DreamEngine {
    store: MemoryStore,
    retrieval: RetrievalEngine,
    consolidator: Option<Arc<dyn DreamConsolidator>>,
    config: DreamConfig,
    consolidation_interval_hours: f64,
    last_consolidation: std::sync::Mutex<Option<DateTime<Utc>>>,
    events: std::sync::Mutex<Vec<DreamEvent>>,
}

impl DreamEngine {
    pub fn new(store: MemoryStore) -> Self {
        let config = DreamConfig::default();
        let interval = config.frequency_hours;
        Self {
            retrieval: RetrievalEngine::new(store.clone()),
            consolidator: None,
            config,
            consolidation_interval_hours: interval,
            last_consolidation: std::sync::Mutex::new(None),
            events: std::sync::Mutex::new(Vec::new()),
            store,
        }
    }

    pub fn with_interval(mut self, hours: f64) -> Self {
        self.consolidation_interval_hours = hours;
        self.config.frequency_hours = hours;
        self
    }

    pub fn with_config(mut self, config: DreamConfig) -> Self {
        self.consolidation_interval_hours = config.frequency_hours;
        self.config = config;
        self
    }

    pub fn with_consolidator(mut self, consolidator: Arc<dyn DreamConsolidator>) -> Self {
        self.consolidator = Some(consolidator);
        self
    }

    pub fn config(&self) -> &DreamConfig {
        &self.config
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn retrieval(&self) -> &RetrievalEngine {
        &self.retrieval
    }

    /// Check if consolidation is due.
    pub fn is_due(&self) -> bool {
        let last = self
            .last_consolidation
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match *last {
            Some(t) => {
                let elapsed = (Utc::now() - t).num_hours() as f64;
                elapsed >= self.consolidation_interval_hours
            }
            None => true,
        }
    }

    /// The next time a dream cycle is due. Returns `None` when no cycle has
    /// run yet (i.e. a cycle is due immediately).
    pub fn next_dream_due_at(&self) -> Option<DateTime<Utc>> {
        let last = self
            .last_consolidation
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        last.map(|t| {
            t + chrono::Duration::seconds((self.consolidation_interval_hours * 3600.0) as i64)
        })
    }

    /// Run a dream cycle only if one is due. Returns `None` when not due.
    pub async fn run_dream_if_due(&self, agent_id: &Uuid) -> CoreResult<Option<DreamSummary>> {
        if self.is_due() {
            Ok(Some(self.run_dream_cycle(agent_id).await?))
        } else {
            Ok(None)
        }
    }

    /// Merge two memories into one, persist the merged entry, and delete both
    /// originals. Returns the merged memory id.
    pub fn merge_and_store(
        &self,
        a: &MemoryEntry,
        b: &MemoryEntry,
    ) -> CoreResult<opensquilla_core::types::MemoryId> {
        let merged = merge_memories(a, b);
        self.store.insert_memory(&merged)?;
        self.store.delete_memory(&a.id)?;
        self.store.delete_memory(&b.id)?;
        info!(
            "Merged memories {:?} and {:?} into {:?}",
            a.id, b.id, merged.id
        );
        Ok(merged.id)
    }

    /// Recompute importance scores for every memory of an agent.
    pub fn refresh_importance(&self, agent_id: &Uuid) -> CoreResult<usize> {
        reindex_importance(&self.store, agent_id)
    }

    /// Run one full dream cycle for an agent.
    ///
    /// Returns a [`DreamSummary`] describing what happened. Progress is
    /// recorded as [`DreamEvent`]s, retrievable via
    /// [`last_event`][Self::last_event] / [`drain_events`][Self::drain_events].
    pub async fn run_dream_cycle(&self, agent_id: &Uuid) -> CoreResult<DreamSummary> {
        self.record_event(DreamEvent::CycleStarted {
            agent_id: *agent_id,
            timestamp: Utc::now(),
        });

        let total = self.store.list_memories(agent_id, None, 1000, 0)?.len();

        let candidates = self.identify_consolidation_candidates(agent_id)?;
        self.record_event(DreamEvent::CandidatesFound {
            agent_id: *agent_id,
            count: candidates.len(),
        });

        let mut abstractions_created = 0u64;
        if !candidates.is_empty() {
            let consolidated = self.generate_consolidation(&candidates).await?;
            for cm in consolidated {
                let id = self.apply_consolidation(&cm)?;
                abstractions_created += 1;
                self.record_event(DreamEvent::ConsolidationCreated {
                    agent_id: *agent_id,
                    memory_id: id,
                    source_count: cm.source_memory_ids.len(),
                });
            }
        }

        let pruned = if self.config.prune_contradictions {
            self.prune_contradictions(agent_id).await?
        } else {
            0
        };

        if let Ok(mut last) = self.last_consolidation.lock() {
            *last = Some(Utc::now());
        }

        let summary = DreamSummary {
            agent_id: *agent_id,
            total_memories: total as u64,
            consolidated: abstractions_created,
            patterns_found: candidates.len() as u64,
            abstractions_created,
            timestamp: Utc::now(),
        };

        self.record_event(DreamEvent::CycleComplete {
            agent_id: *agent_id,
            summary: summary.clone(),
        });
        if pruned > 0 {
            self.record_event(DreamEvent::Pruned {
                agent_id: *agent_id,
                removed_count: pruned,
            });
        }

        info!(
            "Dream cycle for agent {}: {} candidates, {} abstractions, {} contradictions pruned",
            agent_id,
            candidates.len(),
            abstractions_created,
            pruned
        );
        Ok(summary)
    }

    /// Identify clusters of similar memories that are candidates for
    /// consolidation.
    pub fn identify_consolidation_candidates(
        &self,
        agent_id: &Uuid,
    ) -> CoreResult<Vec<ConsolidationCandidate>> {
        let memories = self.store.list_memories(agent_id, None, 1000, 0)?;
        if memories.len() < self.config.min_memories {
            return Ok(Vec::new());
        }

        let mut visited: std::collections::HashSet<MemoryId> = std::collections::HashSet::new();
        let mut candidates: Vec<ConsolidationCandidate> = Vec::new();

        for i in 0..memories.len() {
            if visited.contains(&memories[i].id) {
                continue;
            }
            let mut cluster: Vec<MemoryEntry> = vec![memories[i].clone()];
            visited.insert(memories[i].id);

            for j in (i + 1)..memories.len() {
                if visited.contains(&memories[j].id) {
                    continue;
                }
                let sim = content_similarity(&memories[i].content, &memories[j].content);
                if sim >= self.config.consolidation_threshold {
                    cluster.push(memories[j].clone());
                    visited.insert(memories[j].id);
                }
            }

            if cluster.len() > 1 {
                candidates.push(build_candidate(
                    *agent_id,
                    &cluster,
                    self.config.consolidation_threshold,
                ));
            }
        }

        candidates.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.max_candidates);
        Ok(candidates)
    }

    /// Generate consolidated memories for a set of candidates.
    ///
    /// Uses the injected [`DreamConsolidator`] when available, otherwise falls
    /// back to the deterministic heuristic.
    pub async fn generate_consolidation(
        &self,
        candidates: &[ConsolidationCandidate],
    ) -> CoreResult<Vec<ConsolidatedMemory>> {
        if let Some(consolidator) = &self.consolidator {
            let out = consolidator.consolidate(candidates).await?;
            debug!(
                "Consolidator '{}' produced {} memories",
                consolidator.name(),
                out.len()
            );
            return Ok(out);
        }
        Ok(heuristic_consolidate(candidates))
    }

    /// Persist a consolidated memory to the store.
    pub fn apply_consolidation(&self, consolidated: &ConsolidatedMemory) -> CoreResult<MemoryId> {
        let id = MemoryId(Uuid::new_v4());
        let mut metadata = consolidated.metadata.clone();
        metadata["source_memory_ids"] = serde_json::json!(
            consolidated
                .source_memory_ids
                .iter()
                .map(|m| m.0.to_string())
                .collect::<Vec<_>>()
        );
        metadata["consolidated_at"] = serde_json::json!(Utc::now().to_rfc3339());

        let mut entry = MemoryEntry::new(
            id,
            consolidated.agent_id,
            consolidated.content.clone(),
            String::from("dream"),
            consolidated.memory_type.clone(),
            consolidated.importance.clamp(0.0, 1.0),
            metadata,
        );
        entry.tags = consolidated.tags.clone();
        entry.metadata["source_count"] = serde_json::json!(consolidated.source_memory_ids.len());

        self.store.insert_memory(&entry)?;
        debug!("Applied dream consolidation {}", id.0);
        Ok(id)
    }

    /// Find pairs of memories that contradict each other.
    pub fn identify_contradictions(&self, agent_id: &Uuid) -> CoreResult<Vec<Contradiction>> {
        let memories = self.store.list_memories(agent_id, None, 1000, 0)?;
        let mut out = Vec::new();
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                let a = &memories[i];
                let b = &memories[j];
                let overlap = content_similarity(&a.content, &b.content);
                if overlap >= self.config.contradiction_overlap {
                    let conflict_score = detect_conflict(&a.content, &b.content);
                    if conflict_score >= self.config.contradiction_threshold {
                        out.push(Contradiction {
                            agent_id: *agent_id,
                            a: a.clone(),
                            b: b.clone(),
                            overlap,
                            conflict_score,
                        });
                    }
                }
                if out.len() >= self.config.max_contradictions {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// Find and resolve contradictory memories.
    ///
    /// For each contradiction, the lower-quality memory (by importance then
    /// recency) is removed. If an LLM-backed consolidator is configured, it is
    /// given a chance to produce a merged resolution first.
    pub async fn prune_contradictions(&self, agent_id: &Uuid) -> CoreResult<u64> {
        let contradictions = self.identify_contradictions(agent_id)?;
        let mut removed = 0u64;

        for contradiction in contradictions {
            let (keep, remove) = if contradiction.a.importance >= contradiction.b.importance
                && contradiction.a.updated_at >= contradiction.b.updated_at
            {
                (contradiction.a.clone(), contradiction.b.clone())
            } else {
                (contradiction.b.clone(), contradiction.a.clone())
            };

            let resolved = if let Some(consolidator) = &self.consolidator {
                consolidator.resolve_contradiction(&keep, &remove).await?
            } else {
                None
            };

            if let Some(merged) = resolved {
                self.apply_consolidation(&merged)?;
            }
            self.store.delete_memory(&remove.id)?;
            removed += 1;

            self.record_event(DreamEvent::ContradictionResolved {
                agent_id: *agent_id,
                kept: keep.id,
                removed: remove.id,
            });
        }

        if removed > 0 {
            info!(
                "Dream pruning removed {} contradictory memories for agent {}",
                removed, agent_id
            );
        }
        Ok(removed)
    }

    /// Run one consolidation cycle for an agent (legacy heuristic).
    pub fn consolidate(&self, agent_id: &Uuid) -> CoreResult<DreamSummary> {
        let memories = self.store.list_memories(agent_id, None, 1000, 0)?;

        if memories.is_empty() {
            return Ok(DreamSummary {
                agent_id: *agent_id,
                total_memories: 0,
                consolidated: 0,
                patterns_found: 0,
                abstractions_created: 0,
                timestamp: Utc::now(),
            });
        }

        // Group memories by type
        let mut by_type: std::collections::HashMap<String, Vec<&crate::types::MemoryEntry>> =
            std::collections::HashMap::new();
        for memory in &memories {
            by_type
                .entry(memory.memory_type.clone())
                .or_default()
                .push(memory);
        }

        let mut patterns_found = 0;
        let mut abstractions_created = 0;

        // Find patterns: frequently occurring words across memories
        for (mem_type, entries) in &by_type {
            if entries.len() < 3 {
                continue;
            }

            // Extract common words (simple pattern detection)
            let patterns = self.extract_patterns(entries);
            patterns_found += patterns.len();

            // Create abstraction if patterns are strong enough
            for pattern in &patterns {
                if pattern.strength > 0.6 {
                    let abstraction_content = format!(
                        "[Dream] Pattern in {}: \"{}\" (appears in {} memories, strength {:.2})",
                        mem_type, pattern.phrase, pattern.frequency, pattern.strength
                    );

                    let mut entry = crate::types::MemoryEntry::new(
                        MemoryId(Uuid::new_v4()),
                        *agent_id,
                        abstraction_content,
                        String::from("dream"),
                        String::from("abstraction"),
                        0.8,
                        serde_json::json!({
                            "consolidated_at": Utc::now().to_rfc3339(),
                            "source_type": mem_type,
                            "pattern_phrase": pattern.phrase,
                            "pattern_frequency": pattern.frequency,
                        }),
                    );
                    entry.metadata["strength"] = serde_json::json!(pattern.strength);

                    self.store.insert_memory(&entry)?;
                    abstractions_created += 1;
                }
            }
        }

        // Update last consolidation time
        if let Ok(mut last) = self.last_consolidation.lock() {
            *last = Some(Utc::now());
        }

        let summary = DreamSummary {
            agent_id: *agent_id,
            total_memories: memories.len() as u64,
            consolidated: memories.len() as u64,
            patterns_found: patterns_found as u64,
            abstractions_created: abstractions_created as u64,
            timestamp: Utc::now(),
        };

        info!(
            "Dream consolidation for agent {}: {} memories, {} patterns, {} abstractions",
            agent_id, summary.total_memories, summary.patterns_found, summary.abstractions_created
        );

        Ok(summary)
    }

    /// Extract common patterns from a set of memories.
    fn extract_patterns(&self, entries: &[&crate::types::MemoryEntry]) -> Vec<Pattern> {
        // Count word frequencies across all entries
        let mut word_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut entry_count = 0usize;

        // Common stop words to filter out
        let stop_words: std::collections::HashSet<&str> = [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "myself",
            "yourself",
            "himself",
            "herself",
            "itself",
            "ourselves",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "because",
            "as",
            "until",
            "while",
            "about",
        ]
        .iter()
        .cloned()
        .collect();

        for entry in entries {
            entry_count += 1;
            let lowered = entry.content.to_lowercase();
            let words: Vec<&str> = lowered
                .split_whitespace()
                .filter(|w| {
                    w.len() > 3 && !stop_words.contains(w) && w.chars().all(|c| c.is_alphabetic())
                })
                .collect();

            let unique_words: std::collections::HashSet<&str> = words.into_iter().collect();
            for word in unique_words {
                *word_freq.entry(word.to_string()).or_default() += 1;
            }
        }

        // Build patterns from frequent words
        let mut patterns: Vec<Pattern> = word_freq
            .into_iter()
            .filter(|(_, count)| *count >= 2 && *count as f64 / entry_count as f64 > 0.2)
            .map(|(word, count)| {
                let strength = count as f64 / entry_count.max(1) as f64;
                Pattern {
                    phrase: word,
                    frequency: count,
                    strength,
                }
            })
            .collect();

        patterns.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        patterns.truncate(10);

        patterns
    }

    /// Record a dream event, bounding the retained history.
    fn record_event(&self, event: DreamEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
            if events.len() > 1000 {
                let excess = events.len() - 1000;
                events.drain(0..excess);
            }
        }
    }

    /// The most recent dream event, if any.
    pub fn last_event(&self) -> Option<DreamEvent> {
        self.events.lock().ok().and_then(|e| e.last().cloned())
    }

    /// Drain all recorded dream events.
    pub fn drain_events(&self) -> Vec<DreamEvent> {
        let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *events)
    }
}

/// Build a consolidation candidate from a cluster of similar memories.
fn build_candidate(
    agent_id: Uuid,
    cluster: &[MemoryEntry],
    fallback_similarity: f64,
) -> ConsolidationCandidate {
    let mut total_sim = 0.0;
    let mut pairs = 0usize;
    for a in 0..cluster.len() {
        for b in (a + 1)..cluster.len() {
            total_sim += content_similarity(&cluster[a].content, &cluster[b].content);
            pairs += 1;
        }
    }
    let similarity = if pairs > 0 {
        total_sim / pairs as f64
    } else {
        fallback_similarity
    };

    let mut ids = Vec::with_capacity(cluster.len());
    let mut cluster_content = String::new();
    for m in cluster {
        ids.push(m.id);
        if !cluster_content.is_empty() {
            cluster_content.push_str("\n---\n");
        }
        cluster_content.push_str(&m.content);
    }

    ConsolidationCandidate {
        agent_id,
        memory_ids: ids,
        similarity,
        cluster_content,
        memory_type: cluster[0].memory_type.clone(),
    }
}

/// Deterministically produce a consolidated memory for each candidate.
fn heuristic_consolidate(candidates: &[ConsolidationCandidate]) -> Vec<ConsolidatedMemory> {
    candidates
        .iter()
        .map(|c| {
            let content = format!(
                "[Dream] Consolidated {} memories ({}):\n{}",
                c.memory_ids.len(),
                c.memory_type,
                c.cluster_content
            );
            ConsolidatedMemory {
                agent_id: c.agent_id,
                content,
                memory_type: format!("dream_{}", c.memory_type),
                importance: 0.8,
                tags: vec!["dream".to_string(), "consolidated".to_string()],
                source_memory_ids: c.memory_ids.clone(),
                metadata: serde_json::json!({
                    "similarity": c.similarity,
                    "consolidation_method": "heuristic",
                }),
            }
        })
        .collect()
}

/// Opposite-signal pairs used to detect contradictory memories.
const OPPOSITE_PAIRS: &[(&str, &str)] = &[
    ("like", "dislike"),
    ("like", "hate"),
    ("love", "hate"),
    ("prefer", "avoid"),
    ("dark", "light"),
    ("on", "off"),
    ("enable", "disable"),
    ("yes", "no"),
    ("true", "false"),
    ("fast", "slow"),
    ("large", "small"),
    ("high", "low"),
    ("accept", "reject"),
    ("include", "exclude"),
    ("start", "stop"),
    ("open", "close"),
    ("increase", "decrease"),
    ("always", "never"),
    ("support", "oppose"),
];

/// Detect how strongly two memory contents contradict each other.
///
/// Returns a score in `[0, 1]` based on how many opposite-signal word pairs
/// appear one in each memory.
fn detect_conflict(a: &str, b: &str) -> f64 {
    let al = a.to_lowercase();
    let bl = b.to_lowercase();
    let mut conflicts = 0.0;
    for (x, y) in OPPOSITE_PAIRS {
        let x_in_a = al.contains(x);
        let y_in_b = bl.contains(y);
        let y_in_a = al.contains(y);
        let x_in_b = bl.contains(x);
        if (x_in_a && y_in_b) || (y_in_a && x_in_b) {
            conflicts += 1.0;
        }
    }
    (conflicts * 0.4_f64).min(1.0_f64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamSummary {
    pub agent_id: Uuid,
    pub total_memories: u64,
    pub consolidated: u64,
    pub patterns_found: u64,
    pub abstractions_created: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct Pattern {
    phrase: String,
    frequency: usize,
    strength: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_memories(store: &MemoryStore, agent: Uuid, contents: &[&str]) -> Vec<MemoryEntry> {
        let mut entries = Vec::new();
        for content in contents {
            let mut entry = MemoryEntry::new(
                MemoryId(Uuid::new_v4()),
                agent,
                content.to_string(),
                "test".to_string(),
                "episodic".to_string(),
                0.5,
                serde_json::Value::Null,
            );
            entry.tags = vec!["dream_test".to_string()];
            store.insert_memory(&entry).unwrap();
            entries.push(entry);
        }
        entries
    }

    #[tokio::test]
    async fn test_run_dream_cycle_creates_abstractions() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone()).with_config(DreamConfig {
            min_memories: 2,
            consolidation_threshold: 0.2,
            prune_contradictions: false,
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        insert_memories(
            &store,
            agent,
            &[
                "The user prefers dark mode in their editor",
                "The user likes dark mode for terminals",
                "The user uses dark themes across tools",
            ],
        );
        let summary = engine.run_dream_cycle(&agent).await.unwrap();
        assert!(summary.abstractions_created >= 1);

        let dreams = store
            .list_memories(&agent, Some("dream_episodic"), 100, 0)
            .unwrap();
        assert!(!dreams.is_empty());

        // Events should have been recorded.
        let events = engine.drain_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DreamEvent::CycleStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, DreamEvent::CycleComplete { .. }))
        );
    }

    #[test]
    fn test_identify_candidates_groups_similar() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone()).with_config(DreamConfig {
            min_memories: 2,
            consolidation_threshold: 0.5,
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        insert_memories(
            &store,
            agent,
            &[
                "prefers dark mode everywhere",
                "prefers dark mode in vim",
                "likes pizza for lunch",
                "likes sushi for dinner",
            ],
        );
        let candidates = engine.identify_consolidation_candidates(&agent).unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates[0].memory_ids.len() >= 2);
        assert!(candidates[0].similarity >= 0.5);
    }

    #[test]
    fn test_no_candidates_below_min_memories() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone()).with_config(DreamConfig {
            min_memories: 10,
            consolidation_threshold: 0.5,
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        insert_memories(&store, agent, &["one memory", "two memory"]);
        let candidates = engine.identify_consolidation_candidates(&agent).unwrap();
        assert!(candidates.is_empty());
    }

    #[tokio::test]
    async fn test_generate_and_apply_consolidation() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone());
        let agent = Uuid::new_v4();
        let entries = insert_memories(&store, agent, &["alpha memory one", "alpha memory two"]);
        let candidate = ConsolidationCandidate {
            agent_id: agent,
            memory_ids: entries.iter().map(|e| e.id).collect(),
            similarity: 0.8,
            cluster_content: "alpha memory one\n---\nalpha memory two".to_string(),
            memory_type: "episodic".to_string(),
        };
        let consolidated = engine.generate_consolidation(&[candidate]).await.unwrap();
        assert_eq!(consolidated.len(), 1);

        let id = engine.apply_consolidation(&consolidated[0]).unwrap();
        let stored = store.get_memory(&id).unwrap().unwrap();
        assert_eq!(stored.source, "dream");
        assert_eq!(stored.memory_type, "dream_episodic");
        assert_eq!(
            stored.metadata["source_memory_ids"]
                .as_array()
                .map(|a| a.len()),
            Some(2)
        );
    }

    #[test]
    fn test_identify_contradictions_finds_conflicts() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone()).with_config(DreamConfig {
            contradiction_overlap: 0.2,
            contradiction_threshold: 0.3,
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        insert_memories(
            &store,
            agent,
            &[
                "The user prefers dark mode for all interfaces",
                "The user prefers light mode for all interfaces",
                "unrelated memory about cooking pasta",
            ],
        );
        let contradictions = engine.identify_contradictions(&agent).unwrap();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].conflict_score, 0.4);
    }

    #[tokio::test]
    async fn test_prune_contradictions() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone()).with_config(DreamConfig {
            contradiction_overlap: 0.2,
            contradiction_threshold: 0.3,
            ..Default::default()
        });
        let agent = Uuid::new_v4();
        insert_memories(
            &store,
            agent,
            &[
                "The user prefers dark mode for all interfaces",
                "The user prefers light mode for all interfaces",
            ],
        );
        let removed = engine.prune_contradictions(&agent).await.unwrap();
        assert_eq!(removed, 1);

        let remaining = store.list_memories(&agent, None, 100, 0).unwrap();
        assert_eq!(remaining.len(), 1);
    }

    #[test]
    fn test_is_due() {
        let engine = DreamEngine::new(MemoryStore::in_memory().unwrap());
        assert!(engine.is_due());
        engine
            .last_consolidation
            .lock()
            .unwrap()
            .replace(Utc::now());
        assert!(!engine.is_due());
    }

    #[test]
    fn test_extract_patterns_returns_patterns() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone());
        let agent = Uuid::new_v4();
        let entries = insert_memories(
            &store,
            agent,
            &[
                "rust memory consolidation patterns",
                "rust memory retrieval patterns",
                "rust memory storage patterns",
            ],
        );
        let refs: Vec<&MemoryEntry> = entries.iter().collect();
        let patterns = engine.extract_patterns(&refs);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_detect_conflict() {
        assert!((detect_conflict("prefers dark mode", "prefers light mode") - 0.4).abs() < 1e-9);
        assert_eq!(
            detect_conflict("prefers dark mode", "prefers dark mode"),
            0.0
        );
        assert!((detect_conflict("always use vim", "never use vim") - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_legacy_consolidate_still_works() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone());
        let agent = Uuid::new_v4();
        insert_memories(
            &store,
            agent,
            &[
                "memory retrieval patterns",
                "memory retrieval patterns",
                "memory retrieval patterns",
            ],
        );
        let summary = engine.consolidate(&agent).unwrap();
        assert_eq!(summary.total_memories, 3);
    }

    #[test]
    fn test_merge_memories_combines() {
        let mut a = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            Uuid::new_v4(),
            "user prefers dark mode".to_string(),
            "test".to_string(),
            "preference".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        a.tags = vec!["ui".to_string()];
        let mut b = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            a.agent_id,
            "user uses vim keybindings".to_string(),
            "test".to_string(),
            "preference".to_string(),
            0.9,
            serde_json::Value::Null,
        );
        b.tags = vec!["editor".to_string()];

        let merged = merge_memories(&a, &b);
        assert!(merged.content.contains("dark mode"));
        assert!(merged.content.contains("vim"));
        assert!(merged.tags.contains(&"ui".to_string()));
        assert!(merged.tags.contains(&"editor".to_string()));
        assert_eq!(merged.importance, 0.9);
        assert_eq!(merged.memory_type, "preference");
    }

    #[test]
    fn test_score_importance_recency_and_length() {
        let now = Utc::now();
        let fresh = score_importance("x".repeat(300).as_str(), now, 1, 0.5);
        let old = score_importance(
            "x".repeat(300).as_str(),
            now - chrono::Duration::days(365),
            1,
            0.5,
        );
        assert!(fresh > old);
        assert!((0.0..=1.0).contains(&fresh));
        assert!((0.0..=1.0).contains(&old));
    }

    #[test]
    fn test_reindex_importance_updates() {
        let store = MemoryStore::in_memory().unwrap();
        let agent = Uuid::new_v4();
        let entry = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "content".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.0,
            serde_json::Value::Null,
        );
        store.insert_memory(&entry).unwrap();

        let updated = reindex_importance(&store, &agent).unwrap();
        assert!(updated >= 1);
        let stored = store.list_memories(&agent, None, 10, 0).unwrap()[0].clone();
        assert!(stored.importance > 0.0);
    }

    #[test]
    fn test_merge_and_store_replaces_two() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = DreamEngine::new(store.clone());
        let agent = Uuid::new_v4();
        let a = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "first fact".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        let b = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "second fact".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        store.insert_memory(&a).unwrap();
        store.insert_memory(&b).unwrap();

        let merged_id = engine.merge_and_store(&a, &b).unwrap();
        assert!(store.get_memory(&merged_id).unwrap().is_some());
        assert!(store.get_memory(&a.id).unwrap().is_none());
        assert!(store.get_memory(&b.id).unwrap().is_none());
    }

    #[test]
    fn test_next_dream_due_at() {
        let engine = DreamEngine::new(MemoryStore::in_memory().unwrap());
        assert!(engine.next_dream_due_at().is_none());
        engine
            .last_consolidation
            .lock()
            .unwrap()
            .replace(Utc::now());
        let next = engine.next_dream_due_at().unwrap();
        assert!(next > Utc::now());
    }

    #[test]
    fn test_merge_cluster_members_anchors() {
        let agent = Uuid::new_v4();
        let a = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "fact one".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        let b = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "fact two".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.6,
            serde_json::Value::Null,
        );
        let c = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "fact three".to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.7,
            serde_json::Value::Null,
        );
        let merged = merge_cluster_members(&[a, b, c]).unwrap();
        assert!(merged.content.contains("fact one"));
        assert!(merged.content.contains("fact three"));
        assert_eq!(merged.importance, 0.7);
    }

    struct FakeLlm {
        response: String,
    }

    #[async_trait::async_trait]
    impl DreamLlm for FakeLlm {
        async fn complete(&self, _system: &str, _user: &str) -> CoreResult<String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_llm_consolidator_parses_json() {
        let llm = Arc::new(FakeLlm {
            response: r#"[
                {"content": "user prefers dark mode in all tools", "importance": 0.8, "tags": ["ui"]}
            ]"#
            .to_string(),
        });
        let consolidator = LlmConsolidator::new(llm);
        let agent = Uuid::new_v4();
        let candidates = vec![ConsolidationCandidate {
            agent_id: agent,
            memory_ids: vec![MemoryId(Uuid::new_v4()), MemoryId(Uuid::new_v4())],
            similarity: 0.7,
            cluster_content: "dark mode\n---\ndark themes".to_string(),
            memory_type: "preference".to_string(),
        }];
        let consolidated = consolidator.consolidate(&candidates).await.unwrap();
        assert_eq!(consolidated.len(), 1);
        assert!(consolidated[0].content.contains("dark mode"));
        assert_eq!(consolidated[0].importance, 0.8);
        assert!(consolidated[0].tags.contains(&"ui".to_string()));
        assert_eq!(consolidated[0].memory_type, "dream_preference");
    }

    #[tokio::test]
    async fn test_llm_consolidator_falls_back_to_heuristic() {
        // An LLM response with no parseable JSON array should fall back to
        // the heuristic fillers so no candidate is lost.
        let llm = Arc::new(FakeLlm {
            response: "I can't help with that.".to_string(),
        });
        let consolidator = LlmConsolidator::new(llm);
        let agent = Uuid::new_v4();
        let candidates = vec![
            ConsolidationCandidate {
                agent_id: agent,
                memory_ids: vec![MemoryId(Uuid::new_v4())],
                similarity: 0.6,
                cluster_content: "first cluster content".to_string(),
                memory_type: "episodic".to_string(),
            },
            ConsolidationCandidate {
                agent_id: agent,
                memory_ids: vec![MemoryId(Uuid::new_v4())],
                similarity: 0.6,
                cluster_content: "second cluster content".to_string(),
                memory_type: "episodic".to_string(),
            },
        ];
        let consolidated = consolidator.consolidate(&candidates).await.unwrap();
        assert_eq!(consolidated.len(), 2);
        assert_eq!(
            consolidated[0].metadata["consolidation_method"],
            "heuristic"
        );
    }

    #[tokio::test]
    async fn test_llm_consolidator_resolves_contradiction() {
        let llm = Arc::new(FakeLlm {
            response: r#"{"content": "user likes dark mode but only at night", "importance": 0.7, "tags": ["ui", "resolved"]}"#
                .to_string(),
        });
        let consolidator = LlmConsolidator::new(llm);
        let agent = Uuid::new_v4();
        let a = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "user prefers dark mode".to_string(),
            "test".to_string(),
            "preference".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        let b = MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent,
            "user prefers light mode".to_string(),
            "test".to_string(),
            "preference".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        let resolved = consolidator.resolve_contradiction(&a, &b).await.unwrap();
        assert!(resolved.is_some());
        let merged = resolved.unwrap();
        assert!(merged.content.contains("dark mode"));
        assert_eq!(merged.importance, 0.7);
    }
}
