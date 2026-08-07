//! # FTS5 index management and statistics
//!
//! The memory store uses SQLite FTS5 for full-text search over memory entries.
//! This module adds operational tooling on top of the raw store:
//!
//! - [`FtsIndexManager`] — inspect and rebuild the FTS index, prune orphaned
//!   rows, and produce [`IndexStats`].
//! - [`IndexStats`] — entry/chunk counts, index size, token volume, and
//!   per-agent breakdowns.
//! - Reindex orchestration that walks the store's memories and re-inserts
//!   their text into the FTS table.
//!
//! The manager operates through a [`MemoryStore`], so it inherits the store's
//! connection handling and never opens its own database handle.

use chrono::{DateTime, Utc};
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::store::MemoryStore;

/// A snapshot of the FTS index's health and composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of FTS rows.
    pub fts_rows: u64,
    /// Number of memories in the store.
    pub memory_count: u64,
    /// Number of indexed files.
    pub file_count: u64,
    /// Number of file chunks.
    pub chunk_count: u64,
    /// Approximate total bytes stored in the FTS index.
    pub fts_size_bytes: u64,
    /// Approximate total token volume across all indexed entries.
    pub total_tokens: u64,
    /// Number of FTS rows with no matching memory (orphans).
    pub orphan_rows: u64,
    /// Per-agent memory counts.
    pub by_agent: Vec<AgentCount>,
    /// When the stats were collected.
    pub collected_at: DateTime<Utc>,
}

impl Default for IndexStats {
    fn default() -> Self {
        Self {
            fts_rows: 0,
            memory_count: 0,
            file_count: 0,
            chunk_count: 0,
            fts_size_bytes: 0,
            total_tokens: 0,
            orphan_rows: 0,
            by_agent: Vec::new(),
            collected_at: Utc::now(),
        }
    }
}

/// Memory counts for a single agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCount {
    pub agent_id: Uuid,
    pub memory_count: u64,
}

impl IndexStats {
    /// The ratio of orphans to total FTS rows, in `[0, 1]`.
    pub fn orphan_ratio(&self) -> f64 {
        if self.fts_rows == 0 {
            0.0
        } else {
            self.orphan_rows as f64 / self.fts_rows as f64
        }
    }

    /// Whether the index is in a healthy state (no orphans, memories present).
    pub fn is_healthy(&self) -> bool {
        self.orphan_rows == 0
    }

    /// A one-line human summary of the index health.
    pub fn summary(&self) -> String {
        format!(
            "fts_rows={} memories={} files={} chunks={} orphans={} size_bytes={}",
            self.fts_rows,
            self.memory_count,
            self.file_count,
            self.chunk_count,
            self.orphan_rows,
            self.fts_size_bytes
        )
    }
}

/// The result of a reindex operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReindexReport {
    /// Number of memories (re)indexed.
    pub reindexed: u64,
    /// Number of FTS rows removed as orphans.
    pub orphans_removed: u64,
    /// Number of file chunks (re)indexed.
    pub chunks_reindexed: u64,
    /// Duration of the operation in milliseconds.
    pub duration_ms: u64,
    /// The final index stats.
    pub stats: IndexStats,
}

/// Operational manager for the memory FTS index.
#[derive(Debug, Clone)]
pub struct FtsIndexManager {
    store: MemoryStore,
}

impl FtsIndexManager {
    /// Create an index manager over the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }

    /// The underlying store.
    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Collect a snapshot of index statistics.
    pub fn stats(&self) -> CoreResult<IndexStats> {
        let mut stats = IndexStats {
            collected_at: Utc::now(),
            ..Default::default()
        };

        // Memory count and per-agent breakdown.
        let memories = self.store.list_memories_by_agent_all()?;
        stats.memory_count = memories.len() as u64;
        let mut by_agent: std::collections::HashMap<Uuid, u64> = std::collections::HashMap::new();
        for entry in &memories {
            *by_agent.entry(entry.agent_id).or_insert(0) += 1;
            stats.total_tokens += estimate_tokens(&entry.content);
        }
        let mut agent_counts: Vec<AgentCount> = by_agent
            .into_iter()
            .map(|(agent_id, memory_count)| AgentCount {
                agent_id,
                memory_count,
            })
            .collect();
        agent_counts.sort_by_key(|b| std::cmp::Reverse(b.memory_count));
        stats.by_agent = agent_counts;

        // FTS table stats (best-effort; the table name is fixed).
        if let Ok(row) = self.store.query_fts_stats() {
            stats.fts_rows = row.rows;
            stats.fts_size_bytes = row.size_bytes;
            stats.orphan_rows = row.orphans;
        }

        // Files and chunks (best-effort).
        stats.file_count = self.store.count_files()?.unwrap_or(0);
        stats.chunk_count = self.store.count_chunks()?.unwrap_or(0);

        Ok(stats)
    }

    /// Rebuild the FTS index from the store's current memory contents.
    ///
    /// Returns a [`ReindexReport`] describing how many rows were (re)inserted
    /// and how many orphans were removed.
    pub fn reindex(&self) -> CoreResult<ReindexReport> {
        let start = std::time::Instant::now();
        let memories = self.store.list_memories_by_agent_all()?;

        // Re-insert every memory's text into the FTS table.
        let mut reindexed = 0u64;
        for entry in &memories {
            self.store.reindex_memory(entry)?;
            reindexed += 1;
        }

        // Remove FTS rows whose memory no longer exists.
        let orphans_removed = self.store.prune_fts_orphans()?;

        // Re-index file chunks.
        let chunks_reindexed = self.store.reindex_chunks()?;

        let stats = self.stats()?;
        let duration_ms = start.elapsed().as_millis() as u64;
        Ok(ReindexReport {
            reindexed,
            orphans_removed,
            chunks_reindexed,
            duration_ms,
            stats,
        })
    }

    /// Remove orphaned FTS rows (rows referencing deleted memories).
    pub fn prune_orphans(&self) -> CoreResult<u64> {
        self.store.prune_fts_orphans()
    }

    /// The number of FTS rows that reference missing memories.
    pub fn orphan_count(&self) -> CoreResult<u64> {
        Ok(self.store.query_fts_stats()?.orphans)
    }

    /// Whether the index is healthy (no orphans).
    pub fn is_healthy(&self) -> CoreResult<bool> {
        Ok(self.stats()?.is_healthy())
    }

    /// Estimate the total token volume across all memories.
    pub fn total_tokens(&self) -> CoreResult<u64> {
        let memories = self.store.list_memories_by_agent_all()?;
        Ok(memories.iter().map(|e| estimate_tokens(&e.content)).sum())
    }
}

/// Estimate tokens as `chars / 4`, matching the session crate's estimator.
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryEntry;
    use opensquilla_core::types::MemoryId;

    fn insert(store: &MemoryStore, agent: Uuid, content: &str, tags: &[&str]) {
        let mut entry = MemoryEntry::new(
            MemoryId::new(),
            agent,
            content.to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        );
        entry.tags = tags.iter().map(|s| s.to_string()).collect();
        store.insert_memory(&entry).unwrap();
    }

    #[test]
    fn stats_aggregate_counts() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "rust async runtime memory", &["rust"]);
        insert(&store, agent, "tokio event loop", &["rust"]);

        let stats = manager.stats().unwrap();
        assert_eq!(stats.memory_count, 2);
        assert!(stats.fts_rows > 0);
        assert_eq!(stats.by_agent.len(), 1);
        assert_eq!(stats.by_agent[0].memory_count, 2);
    }

    #[test]
    fn reindex_rebuilds_rows() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "distributed systems design", &[]);
        insert(&store, agent, "consensus algorithms raft", &[]);

        let report = manager.reindex().unwrap();
        assert!(report.reindexed >= 2);
        assert!(report.stats.fts_rows >= 2);
        assert_eq!(report.orphans_removed, 0);
    }

    #[test]
    fn prune_removes_orphans() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "some content here", &[]);

        // Inject a raw FTS row for a rowid that has no memory, simulating
        // external-content-table drift.
        store
            .inject_fts_row(9_999_999, "orphaned index entry", "episodic")
            .unwrap();

        assert!(manager.orphan_count().unwrap() >= 1);
        let removed = manager.prune_orphans().unwrap();
        assert!(removed >= 1);
        assert!(manager.is_healthy().unwrap());
    }

    #[test]
    fn prune_is_noop_on_healthy_store() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "healthy content here", &[]);
        assert_eq!(manager.orphan_count().unwrap(), 0);
        assert_eq!(manager.prune_orphans().unwrap(), 0);
        assert!(manager.is_healthy().unwrap());
    }

    #[test]
    fn total_tokens_estimates() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "12345678", &[]); // 8 chars -> 2 tokens
        assert_eq!(manager.total_tokens().unwrap(), 2);
    }

    #[test]
    fn summary_line_is_readable() {
        let store = MemoryStore::in_memory().unwrap();
        let manager = FtsIndexManager::new(store.clone());
        let agent = Uuid::new_v4();
        insert(&store, agent, "hello world", &[]);
        let stats = manager.stats().unwrap();
        assert!(stats.summary().contains("fts_rows="));
    }

    #[test]
    fn orphan_ratio_math() {
        let stats = IndexStats {
            fts_rows: 100,
            orphan_rows: 25,
            ..Default::default()
        };
        assert!((stats.orphan_ratio() - 0.25).abs() < 1e-9);
        assert!(!stats.is_healthy());

        let clean = IndexStats {
            fts_rows: 100,
            orphan_rows: 0,
            ..Default::default()
        };
        assert!(clean.is_healthy());
    }
}
