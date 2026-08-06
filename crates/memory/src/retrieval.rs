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
    pub fn search_fts(&self, query: &str, limit: u64, offset: u64) -> CoreResult<Vec<MemoryEntry>> {
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
                    self.store
                        .get_memory(&memory_id)
                        .ok()
                        .flatten()
                        .map(|entry| (entry, similarity))
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

                    let combined_score =
                        (alpha * vec_score + (1.0 - alpha) * fts_score) * time_boost + access_boost;

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
        let max_bm25 = bm25_scored.iter().map(|(_, s)| *s).fold(f64::MIN, f64::max);

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

    // -----------------------------------------------------------------------
    // Relevance scoring
    // -----------------------------------------------------------------------

    /// Score a single memory against a query using lightweight lexical
    /// relevance with per-field boosts. Purely deterministic and independent
    /// of the store, so it can be used for re-ranking any candidate set.
    pub fn score_relevance(&self, entry: &MemoryEntry, query: &str) -> f64 {
        score_memory_relevance(entry, query)
    }

    /// Re-rank a set of results by lexical relevance to the query, blending
    /// the existing score with the lexical score.
    ///
    /// `lexical_weight` in `[0,1]` controls how much the lexical score
    /// influences the final ranking (`0` = keep original order).
    pub fn rerank_by_relevance(
        &self,
        query: &str,
        results: Vec<MemorySearchResult>,
        lexical_weight: f64,
    ) -> Vec<MemorySearchResult> {
        let mut scored: Vec<(MemorySearchResult, f64)> = results
            .into_iter()
            .map(|r| {
                let lexical = score_memory_relevance(&r.entry, query);
                let blended = r.score * (1.0 - lexical_weight) + lexical * lexical_weight;
                (r, blended)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.into_iter().map(|(r, _)| r).collect()
    }

    /// Re-rank results by recency: newer memories float up, weighted by
    /// `recency_weight`. Uses the entry's `created_at`.
    pub fn rerank_by_recency(
        &self,
        results: Vec<MemorySearchResult>,
        recency_weight: f64,
    ) -> Vec<MemorySearchResult> {
        let now = chrono::Utc::now();
        let mut scored: Vec<(MemorySearchResult, f64)> = results
            .into_iter()
            .map(|r| {
                let age_hours = (now - r.entry.created_at).num_hours().max(0) as f64;
                let recency = (-age_hours / self.config.time_decay_hours).exp();
                let blended = r.score * (1.0 - recency_weight) + recency * recency_weight;
                (r, blended)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.into_iter().map(|(r, _)| r).collect()
    }

    /// Re-rank results by importance: higher-importance memories float up.
    pub fn rerank_by_importance(
        &self,
        results: Vec<MemorySearchResult>,
        importance_weight: f64,
    ) -> Vec<MemorySearchResult> {
        let mut scored: Vec<(MemorySearchResult, f64)> = results
            .into_iter()
            .map(|r| {
                let importance = r.entry.importance.clamp(0.0, 1.0);
                let blended = r.score * (1.0 - importance_weight) + importance * importance_weight;
                (r, blended)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.into_iter().map(|(r, _)| r).collect()
    }

    /// Run a multi-signal re-rank: combine lexical relevance, importance, and
    /// recency into one blended score per result.
    pub fn rerank_multi(
        &self,
        query: &str,
        results: Vec<MemorySearchResult>,
        weights: &RerankWeights,
    ) -> Vec<MemorySearchResult> {
        let now = chrono::Utc::now();
        let mut scored: Vec<(MemorySearchResult, f64)> = results
            .into_iter()
            .map(|r| {
                let lexical = score_memory_relevance(&r.entry, query);
                let importance = r.entry.importance.clamp(0.0, 1.0);
                let age_hours = (now - r.entry.created_at).num_hours().max(0) as f64;
                let recency = (-age_hours / self.config.time_decay_hours).exp();

                let total = r.score * weights.base
                    + lexical * weights.lexical
                    + importance * weights.importance
                    + recency * weights.recency;
                (r, total)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.into_iter().map(|(r, _)| r).collect()
    }

    /// Filter a set of results using a filter expression.
    pub fn apply_filter_expression(
        &self,
        results: Vec<MemorySearchResult>,
        expression: &FilterExpression,
    ) -> Vec<MemorySearchResult> {
        results
            .into_iter()
            .filter(|r| expression.matches(&r.entry))
            .collect()
    }

    /// Search with a filter expression applied after scoring.
    pub fn search_filtered(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        agent_id: Option<&Uuid>,
        filters: &MemoryFilters,
        expression: &FilterExpression,
        top_k: usize,
    ) -> CoreResult<Vec<MemorySearchResult>> {
        let results = self.search(query, query_embedding, agent_id, filters, top_k * 3)?;
        Ok(self.apply_filter_expression(results, expression).into_iter().take(top_k).collect())
    }
}

/// Weights for the multi-signal re-ranker.
#[derive(Debug, Clone, Copy)]
pub struct RerankWeights {
    /// Weight of the original retrieval score.
    pub base: f64,
    /// Weight of lexical relevance to the query.
    pub lexical: f64,
    /// Weight of the memory's importance score.
    pub importance: f64,
    /// Weight of recency.
    pub recency: f64,
}

impl Default for RerankWeights {
    fn default() -> Self {
        Self {
            base: 0.5,
            lexical: 0.3,
            importance: 0.1,
            recency: 0.1,
        }
    }
}

/// Compute a deterministic lexical relevance score for a memory against a
/// query. Boosts id/title-ish fields (memory_type, tags) above raw content.
pub fn score_memory_relevance(entry: &MemoryEntry, query: &str) -> f64 {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return 0.0;
    }
    let terms: Vec<&str> = query.split_whitespace().filter(|t| t.len() >= 2).collect();
    if terms.is_empty() {
        return 0.0;
    }

    let content_lower = entry.content.to_lowercase();
    let type_lower = entry.memory_type.to_lowercase();
    let tag_hits: usize = entry.tags.iter().filter(|t| query.contains(&t.to_lowercase())).count();

    let mut score = 0.0;
    for term in &terms {
        if content_lower.contains(term) {
            score += 0.6;
        }
        if type_lower.contains(term) {
            score += 0.8;
        }
        for tag in &entry.tags {
            if tag.to_lowercase().contains(term) {
                score += 0.5;
            }
        }
    }
    // Tag-field bonus.
    score += tag_hits as f64 * 0.4;

    // Normalize to [0,1] by saturating.
    (score / (terms.len() as f64 * 2.0)).clamp(0.0, 1.0)
}

/// A declarative filter expression evaluated against memory entries.
///
/// Grammar (whitespace tolerant):
/// - `field == value`, `field != value`
/// - `field >= value`, `field <= value`, `field > value`, `field < value`
/// - `field contains "text"`, `field not contains "text"`
/// - `field in ["a","b"]`, `field not in ["a","b"]`
/// - `AND` / `OR` / `!` combinators, with parentheses.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterExpression {
    /// A single comparison.
    Comparison(FieldOp),
    /// Logical AND.
    And(Box<FilterExpression>, Box<FilterExpression>),
    /// Logical OR.
    Or(Box<FilterExpression>, Box<FilterExpression>),
    /// Negation.
    Not(Box<FilterExpression>),
    /// Always matches.
    Always,
}

/// The fields a filter expression can reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterField {
    Content,
    MemoryType,
    Source,
    Importance,
    AccessCount,
    AgeDays,
    Tag,
}

/// A single field-operator comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldOp {
    Eq(FilterField, String),
    Ne(FilterField, String),
    Gt(FilterField, f64),
    Ge(FilterField, f64),
    Lt(FilterField, f64),
    Le(FilterField, f64),
    Contains(FilterField, String),
    NotContains(FilterField, String),
    In(FilterField, Vec<String>),
    NotIn(FilterField, Vec<String>),
}

impl FilterExpression {
    /// Evaluate the expression against a memory entry.
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        match self {
            FilterExpression::Comparison(op) => op.matches(entry),
            FilterExpression::And(a, b) => a.matches(entry) && b.matches(entry),
            FilterExpression::Or(a, b) => a.matches(entry) || b.matches(entry),
            FilterExpression::Not(inner) => !inner.matches(entry),
            FilterExpression::Always => true,
        }
    }
}

impl FieldOp {
    /// Evaluate a single comparison against a memory entry.
    pub fn matches(&self, entry: &MemoryEntry) -> bool {
        match self {
            FieldOp::Eq(field, value) => field_value(entry, *field).eq_ignore_ascii_case(value),
            FieldOp::Ne(field, value) => !field_value(entry, *field).eq_ignore_ascii_case(value),
            FieldOp::Gt(field, rhs) => field_number(entry, *field).map(|v| v > *rhs).unwrap_or(false),
            FieldOp::Ge(field, rhs) => field_number(entry, *field).map(|v| v >= *rhs).unwrap_or(false),
            FieldOp::Lt(field, rhs) => field_number(entry, *field).map(|v| v < *rhs).unwrap_or(false),
            FieldOp::Le(field, rhs) => field_number(entry, *field).map(|v| v <= *rhs).unwrap_or(false),
            FieldOp::Contains(field, needle) => field_value(entry, *field)
                .to_lowercase()
                .contains(&needle.to_lowercase()),
            FieldOp::NotContains(field, needle) => !field_value(entry, *field)
                .to_lowercase()
                .contains(&needle.to_lowercase()),
            FieldOp::In(field, values) => values
                .iter()
                .any(|v| field_value(entry, *field).eq_ignore_ascii_case(v)),
            FieldOp::NotIn(field, values) => !values
                .iter()
                .any(|v| field_value(entry, *field).eq_ignore_ascii_case(v)),
        }
    }
}

/// Extract a string value for a filter field.
fn field_value(entry: &MemoryEntry, field: FilterField) -> String {
    match field {
        FilterField::Content => entry.content.clone(),
        FilterField::MemoryType => entry.memory_type.clone(),
        FilterField::Source => entry.source.clone(),
        FilterField::Importance => entry.importance.to_string(),
        FilterField::AccessCount => entry.access_count.to_string(),
        FilterField::AgeDays => {
            let age = (chrono::Utc::now() - entry.created_at).num_days();
            age.to_string()
        }
        FilterField::Tag => entry.tags.join(","),
    }
}

/// Extract a numeric value for a filter field.
fn field_number(entry: &MemoryEntry, field: FilterField) -> Option<f64> {
    match field {
        FilterField::Importance => Some(entry.importance),
        FilterField::AccessCount => Some(entry.access_count as f64),
        FilterField::AgeDays => Some((chrono::Utc::now() - entry.created_at).num_days() as f64),
        _ => field_value(entry, field).parse::<f64>().ok(),
    }
}

/// Parse a filter expression string into a [`FilterExpression`]. Returns
/// `Err` on malformed input.
///
/// Supported grammar:
/// - comparisons: `memory_type == "episodic"`, `importance >= 0.5`,
///   `access_count > 2`, `content contains "rust"`,
///   `tags in ["a","b"]`, `source != "conversation"`
/// - combinators: `AND`, `OR`, `!`, and parentheses.
pub fn parse_filter_expression(input: &str) -> Result<FilterExpression, String> {
    let tokens = tokenize_expression(input)?;
    let mut parser = ExpressionParser { tokens };
    let expr = parser.parse_or()?;
    if !parser.tokens.is_empty() {
        return Err(format!(
            "unexpected trailing tokens: {:?}",
            parser.tokens
        ));
    }
    Ok(expr)
}

/// Tokenize a filter expression into logical tokens.
fn tokenize_expression(input: &str) -> Result<Vec<String>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;
    let mut current = String::new();

    let flush = |current: &mut String, tokens: &mut Vec<String>| {
        let t = current.trim();
        if !t.is_empty() {
            tokens.push(t.to_string());
        }
        current.clear();
    };

    while i < chars.len() {
        let c = chars[i];
        match c {
            '(' | ')' => {
                flush(&mut current, &mut tokens);
                tokens.push(c.to_string());
            }
            '[' => {
                // A bracketed list literal: consume until the matching close.
                flush(&mut current, &mut tokens);
                let mut depth = 0usize;
                let mut lit = String::new();
                while i < chars.len() {
                    if chars[i] == '[' {
                        depth += 1;
                    } else if chars[i] == ']' {
                        depth -= 1;
                        if depth == 0 {
                            lit.push(chars[i]);
                            i += 1;
                            break;
                        }
                    }
                    lit.push(chars[i]);
                    i += 1;
                }
                if depth != 0 {
                    return Err("unterminated list literal".to_string());
                }
                tokens.push(lit);
                continue;
            }
            '"' | '\'' => {
                // A quoted string literal.
                flush(&mut current, &mut tokens);
                let quote = c;
                i += 1;
                let mut lit = String::new();
                let mut closed = false;
                while i < chars.len() {
                    if chars[i] == quote {
                        closed = true;
                        i += 1;
                        break;
                    }
                    lit.push(chars[i]);
                    i += 1;
                }
                if !closed {
                    return Err("unterminated string literal".to_string());
                }
                tokens.push(lit);
                continue;
            }
            c if c.is_whitespace() => {
                flush(&mut current, &mut tokens);
            }
            _ => {
                current.push(c);
            }
        }
        i += 1;
    }
    flush(&mut current, &mut tokens);
    Ok(tokens)
}

/// A simple recursive-descent parser for filter expressions.
struct ExpressionParser {
    tokens: Vec<String>,
}

impl ExpressionParser {
    fn peek(&self) -> Option<&str> {
        self.tokens.first().map(|s| s.as_str())
    }

    fn next(&mut self) -> Option<String> {
        if self.tokens.is_empty() {
            None
        } else {
            Some(self.tokens.remove(0))
        }
    }

    fn parse_or(&mut self) -> Result<FilterExpression, String> {
        let mut left = self.parse_and()?;
        while self.peek() == Some("OR") || self.peek() == Some("or") {
            self.next();
            let right = self.parse_and()?;
            left = FilterExpression::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<FilterExpression, String> {
        let mut left = self.parse_unary()?;
        while self.peek() == Some("AND") || self.peek() == Some("and") {
            self.next();
            let right = self.parse_unary()?;
            left = FilterExpression::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<FilterExpression, String> {
        if self.peek() == Some("!") || self.peek() == Some("NOT") || self.peek() == Some("not") {
            self.next();
            let inner = self.parse_unary()?;
            return Ok(FilterExpression::Not(Box::new(inner)));
        }
        if self.peek() == Some("(") {
            self.next();
            let inner = self.parse_or()?;
            if self.next().as_deref() != Some(")") {
                return Err("expected closing ')'".to_string());
            }
            return Ok(inner);
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<FilterExpression, String> {
        let field_tok = self.next().ok_or_else(|| "expected field".to_string())?;
        let field = parse_filter_field(&field_tok)?;
        let op = self.next().ok_or_else(|| "expected operator".to_string())?;

        // Handle compound operators (`not in`, `not contains`, `!=`, ...).
        let lower_op = op.to_lowercase();
        if lower_op == "not" {
            let op2 = self.next().ok_or_else(|| "expected operator after 'not'".to_string())?;
            let value = self.next().ok_or_else(|| "expected value".to_string())?;
            let values = parse_value_list(&value)?;
            return Ok(match op2.to_lowercase().as_str() {
                "in" => FilterExpression::Comparison(FieldOp::NotIn(field, values)),
                "contains" => FilterExpression::Comparison(FieldOp::NotContains(field, values.first().cloned().unwrap_or_default())),
                _ => return Err(format!("unsupported operator 'not {}'", op2)),
            });
        }

        let value = self.next().ok_or_else(|| "expected value".to_string())?;
        let values = parse_value_list(&value)?;

        Ok(match lower_op.as_str() {
            "==" | "=" => FilterExpression::Comparison(FieldOp::Eq(field, values.first().cloned().unwrap_or_default())),
            "!=" => FilterExpression::Comparison(FieldOp::Ne(field, values.first().cloned().unwrap_or_default())),
            ">" => FilterExpression::Comparison(FieldOp::Gt(field, parse_number(&values[0])?)),
            ">=" => FilterExpression::Comparison(FieldOp::Ge(field, parse_number(&values[0])?)),
            "<" => FilterExpression::Comparison(FieldOp::Lt(field, parse_number(&values[0])?)),
            "<=" => FilterExpression::Comparison(FieldOp::Le(field, parse_number(&values[0])?)),
            "contains" => FilterExpression::Comparison(FieldOp::Contains(field, values.first().cloned().unwrap_or_default())),
            "in" => FilterExpression::Comparison(FieldOp::In(field, values)),
            _ => return Err(format!("unsupported operator '{}'", op)),
        })
    }
}

fn parse_filter_field(s: &str) -> Result<FilterField, String> {
    match s.to_lowercase().as_str() {
        "content" | "text" | "body" => Ok(FilterField::Content),
        "memory_type" | "type" => Ok(FilterField::MemoryType),
        "source" | "origin" => Ok(FilterField::Source),
        "importance" | "score" => Ok(FilterField::Importance),
        "access_count" | "accesses" | "count" => Ok(FilterField::AccessCount),
        "age_days" | "age" => Ok(FilterField::AgeDays),
        "tag" | "tags" => Ok(FilterField::Tag),
        other => Err(format!("unknown filter field '{}'", other)),
    }
}

/// Parse a value that may be a single token or a bracketed list.
fn parse_value_list(value: &str) -> Result<Vec<String>, String> {
    let value = value.trim();
    if value.starts_with('[') && value.ends_with(']') {
        let inner = &value[1..value.len() - 1];
        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }
        Ok(inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
            .collect())
    } else if value.starts_with('(') && value.ends_with(')') {
        let inner = &value[1..value.len() - 1];
        Ok(inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
            .collect())
    } else {
        Ok(vec![value.trim_matches('"').trim_matches('\'').to_string()])
    }
}

fn parse_number(s: &str) -> Result<f64, String> {
    s.trim_matches('"')
        .trim_matches('\'')
        .parse::<f64>()
        .map_err(|_| format!("expected number, got '{}'", s))
}

/// Convenience constructors for common filter expressions.
impl FilterExpression {
    /// `memory_type == value`
    pub fn memory_type_eq(value: &str) -> Self {
        FilterExpression::Comparison(FieldOp::Eq(FilterField::MemoryType, value.to_string()))
    }

    /// `source == value`
    pub fn source_eq(value: &str) -> Self {
        FilterExpression::Comparison(FieldOp::Eq(FilterField::Source, value.to_string()))
    }

    /// `importance >= value`
    pub fn importance_ge(value: f64) -> Self {
        FilterExpression::Comparison(FieldOp::Ge(FilterField::Importance, value))
    }

    /// `content contains value`
    pub fn content_contains(value: &str) -> Self {
        FilterExpression::Comparison(FieldOp::Contains(FilterField::Content, value.to_string()))
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
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
        store
            .insert_memory(&make_entry(agent_id, "Rust async runtime"))
            .unwrap();
        store
            .insert_memory(&make_entry(agent_id, "Python scripting"))
            .unwrap();

        let results = engine.search_fts("rust", 10, 0).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_hybrid_pipeline_without_embeddings() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = RetrievalEngine::new(store.clone());
        let agent_id = Uuid::new_v4();
        store
            .insert_memory(&make_entry(agent_id, "memory management in rust"))
            .unwrap();
        store
            .insert_memory(&make_entry(agent_id, "async tokio runtime"))
            .unwrap();

        let results = engine
            .search(
                "rust memory",
                None,
                Some(&agent_id),
                &MemoryFilters::default(),
                10,
            )
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
        store
            .insert_memory(&make_entry(agent_a, "tokio async"))
            .unwrap();
        store
            .insert_memory(&make_entry(agent_b, "tokio async"))
            .unwrap();

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

    #[test]
    fn test_score_relevance_boosts_known_fields() {
        let mut entry = make_entry(Uuid::new_v4(), "rust async runtime memory");
        entry.tags = vec!["rust".to_string()];
        entry.memory_type = "document".to_string();
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let content_score = engine.score_relevance(&entry, "rust");
        let type_score = engine.score_relevance(&entry, "document");
        assert!(content_score > 0.0);
        assert!(type_score > content_score);
    }

    #[test]
    fn test_rerank_by_relevance_moves_lexical_hit_up() {
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let agent_id = Uuid::new_v4();
        let mut fuzzy = make_entry(agent_id, "completely unrelated topic");
        fuzzy.importance = 0.9;
        let mut lexical = make_entry(agent_id, "rust async runtime design");
        lexical.importance = 0.1;

        let results = vec![
            MemorySearchResult::new(fuzzy.clone(), 0.9, "hybrid"),
            MemorySearchResult::new(lexical.clone(), 0.1, "hybrid"),
        ];
        let reranked = engine.rerank_by_relevance("rust async", results, 1.0);
        assert_eq!(reranked[0].entry.id, lexical.id);
    }

    #[test]
    fn test_rerank_by_importance_floats_high() {
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let agent_id = Uuid::new_v4();
        let mut low = make_entry(agent_id, "low importance memory");
        low.importance = 0.1;
        let mut high = make_entry(agent_id, "high importance memory");
        high.importance = 0.9;

        let results = vec![
            MemorySearchResult::new(low.clone(), 0.8, "hybrid"),
            MemorySearchResult::new(high.clone(), 0.2, "hybrid"),
        ];
        let reranked = engine.rerank_by_importance(results, 1.0);
        assert_eq!(reranked[0].entry.id, high.id);
    }

    #[test]
    fn test_rerank_multi_combines_signals() {
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let agent_id = Uuid::new_v4();
        let mut a = make_entry(agent_id, "rust async patterns");
        a.importance = 0.8;
        let mut b = make_entry(agent_id, "unrelated cooking");
        b.importance = 0.2;

        let results = vec![
            MemorySearchResult::new(b.clone(), 0.5, "hybrid"),
            MemorySearchResult::new(a.clone(), 0.5, "hybrid"),
        ];
        let reranked = engine.rerank_multi("rust async", results, &RerankWeights {
            base: 0.2,
            lexical: 0.6,
            importance: 0.1,
            recency: 0.1,
        });
        assert_eq!(reranked[0].entry.id, a.id);
    }

    #[test]
    fn test_rerank_by_recency_prefers_newer() {
        let engine = RetrievalEngine::new(MemoryStore::in_memory().unwrap());
        let agent_id = Uuid::new_v4();
        let mut old = make_entry(agent_id, "old memory");
        old.created_at = chrono::Utc::now() - chrono::Duration::days(100);
        let mut fresh = make_entry(agent_id, "fresh memory");
        fresh.created_at = chrono::Utc::now();

        let results = vec![
            MemorySearchResult::new(old.clone(), 0.9, "hybrid"),
            MemorySearchResult::new(fresh.clone(), 0.1, "hybrid"),
        ];
        let reranked = engine.rerank_by_recency(results, 1.0);
        assert_eq!(reranked[0].entry.id, fresh.id);
    }

    #[test]
    fn test_filter_expression_parser_basic() {
        let expr = parse_filter_expression(r#"memory_type == "episodic""#).unwrap();
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.memory_type = "episodic".to_string();
        assert!(expr.matches(&entry));
        entry.memory_type = "document".to_string();
        assert!(!expr.matches(&entry));
    }

    #[test]
    fn test_filter_expression_parser_contains() {
        let expr = parse_filter_expression(r#"content contains "rust""#).unwrap();
        let entry = make_entry(Uuid::new_v4(), "rust async runtime");
        assert!(expr.matches(&entry));
        let entry2 = make_entry(Uuid::new_v4(), "python scripting");
        assert!(!expr.matches(&entry2));
    }

    #[test]
    fn test_filter_expression_parser_in_list() {
        let expr = parse_filter_expression(r#"tags in ["rust", "memory"]"#).unwrap();
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.tags = vec!["rust".to_string()];
        assert!(expr.matches(&entry));
        let mut other = make_entry(Uuid::new_v4(), "content");
        other.tags = vec!["python".to_string()];
        assert!(!expr.matches(&other));
    }

    #[test]
    fn test_filter_expression_parser_and_not() {
        let expr = parse_filter_expression(r#"memory_type == "episodic" AND importance >= 0.5"#).unwrap();
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.memory_type = "episodic".to_string();
        entry.importance = 0.7;
        assert!(expr.matches(&entry));
        entry.importance = 0.3;
        assert!(!expr.matches(&entry));

        let neg = parse_filter_expression(r#"! (source == "test")"#).unwrap();
        let mut other = make_entry(Uuid::new_v4(), "content");
        other.source = "import".to_string();
        assert!(neg.matches(&other));
        other.source = "test".to_string();
        assert!(!neg.matches(&other));
    }

    #[test]
    fn test_filter_expression_parser_not_in() {
        let expr = parse_filter_expression(r#"tags not in ["rust"]"#).unwrap();
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.tags = vec!["python".to_string()];
        assert!(expr.matches(&entry));
        entry.tags = vec!["rust".to_string()];
        assert!(!expr.matches(&entry));
    }

    #[test]
    fn test_filter_expression_parser_importance_ge() {
        let expr = parse_filter_expression(r#"importance >= 0.6"#).unwrap();
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.importance = 0.9;
        assert!(expr.matches(&entry));
        entry.importance = 0.2;
        assert!(!expr.matches(&entry));
    }

    #[test]
    fn test_filter_expression_parser_errors() {
        assert!(parse_filter_expression(r#"unknown_field == "x""#).is_err());
        assert!(parse_filter_expression(r#"importance >= "#).is_err());
        assert!(parse_filter_expression(r#"content contains "unterminated"#).is_err());
    }

    #[test]
    fn test_filter_expression_convenience_constructors() {
        let expr = FilterExpression::memory_type_eq("episodic");
        let mut entry = make_entry(Uuid::new_v4(), "content");
        entry.memory_type = "episodic".to_string();
        assert!(expr.matches(&entry));

        let expr2 = FilterExpression::importance_ge(0.5);
        entry.importance = 0.8;
        assert!(expr2.matches(&entry));
    }

    #[test]
    fn test_search_filtered_applies_expression() {
        let store = MemoryStore::in_memory().unwrap();
        let engine = RetrievalEngine::new(store.clone());
        let agent_id = Uuid::new_v4();
        let mut episodic = make_entry(agent_id, "rust async memory");
        episodic.memory_type = "episodic".to_string();
        store.insert_memory(&episodic).unwrap();
        let mut doc = make_entry(agent_id, "rust async document");
        doc.memory_type = "document".to_string();
        store.insert_memory(&doc).unwrap();

        let expr = FilterExpression::memory_type_eq("document");
        let results = engine
            .search_filtered("rust async", None, Some(&agent_id), &MemoryFilters::default(), &expr, 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entry.memory_type, "document");
    }
}
