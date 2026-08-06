use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::MemoryId;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};
use uuid::Uuid;

use crate::types::MemoryChunk;

// Re-export the canonical types from `types` so existing callers using
// `opensquilla_memory::store::{MemoryEntry, ...}` keep compiling.
pub use crate::types::{MemoryEntry, MemoryFilters, MemoryQuery, MemorySearchResult};

/// A row in the `memory_tags` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTag {
    pub id: Uuid,
    pub memory_id: MemoryId,
    pub tag: String,
}

/// Tracked indexed file metadata (the `files` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedFile {
    pub id: Uuid,
    pub path: String,
    pub checksum: String,
    pub size: u64,
    pub modified_at: DateTime<Utc>,
    pub indexed_at: DateTime<Utc>,
    pub chunk_count: u32,
}

/// In-memory representation of a stored embedding cache row.
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    pub key: String,
    pub embedding: Vec<f32>,
    pub model: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
}

impl MemoryStore {
    pub fn new(path: &str) -> CoreResult<Self> {
        let conn = Connection::open(path).map_err(|e| CoreError::Storage(e.to_string()))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    pub fn in_memory() -> CoreResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| CoreError::Storage(e.to_string()))?;
        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
        };
        store.initialize_tables()?;
        Ok(store)
    }

    fn initialize_tables(&self) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;

        // --- Agent-scoped memory corpus (existing schema, kept for compat) ---
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT,
                source TEXT NOT NULL DEFAULT 'conversation',
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                importance REAL NOT NULL DEFAULT 0.0,
                importance_score REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content, memory_type,
                content='memories',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content, memory_type)
                VALUES (new.rowid, new.content, new.memory_type);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
                VALUES ('delete', old.rowid, old.content, old.memory_type);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
                VALUES ('delete', old.rowid, old.content, old.memory_type);
                INSERT INTO memories_fts(rowid, content, memory_type)
                VALUES (new.rowid, new.content, new.memory_type);
            END;

            CREATE TABLE IF NOT EXISTS memory_tags (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );

            CREATE INDEX IF NOT EXISTS idx_memory_tags_memory
                ON memory_tags(memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_tags_tag
                ON memory_tags(tag);

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            ",
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // --- File-indexing + chunk schema (the 6 memory DB tables) ---
        //
        //   files          - indexed file tracking (path, checksum, mtime)
        //   chunks         - text chunks + blob embeddings
        //   chunks_fts     - FTS5 virtual table over chunk content
        //   chunks_vec     - sqlite-vec virtual table for vector search
        //   embedding_cache- content-addressed embedding cache
        //   meta           - generic key/value metadata store
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                checksum TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_at TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                dimension INTEGER,
                chunk_index INTEGER NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(file_id, chunk_index);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            ",
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // sqlite-vec is an optional C extension. Attempt to create the
        // chunks_vec and memories_vec virtual tables; if the extension is not
        // loaded, fall back to in-process cosine similarity (handled in
        // retrieval.rs) and keep operating without the virtual tables.
        match conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[1536]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id TEXT PRIMARY KEY,
                embedding float[1536]
            );",
        ) {
            Ok(()) => {
                info!("Memory store initialized: FTS5 + sqlite-vec virtual tables ready");
            }
            Err(e) => {
                warn!(
                    "sqlite-vec extension unavailable ({}); vector search will use in-process cosine similarity",
                    e
                );
            }
        }

        info!(
            "Memory store tables initialized (memories, files, chunks, chunks_fts, embedding_cache, meta)"
        );
        Ok(())
    }

    // --- CRUD ---

    pub fn insert_memory(&self, entry: &MemoryEntry) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let tags_json =
            serde_json::to_string(&entry.tags).map_err(|e| CoreError::Serialization(e))?;
        conn.execute(
            "INSERT INTO memories (id, agent_id, content, tags, created_at, updated_at,
             accessed_at, source, memory_type, importance, importance_score, access_count,
             metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                entry.id.0.to_string(),
                entry.agent_id.to_string(),
                entry.content,
                tags_json,
                entry.created_at.to_rfc3339(),
                entry.updated_at.to_rfc3339(),
                entry.accessed_at.map(|t| t.to_rfc3339()),
                entry.source,
                entry.memory_type,
                entry.importance,
                entry.importance_score,
                entry.access_count as i64,
                entry.metadata.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_memory(&self, memory_id: &MemoryId) -> CoreResult<Option<MemoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, agent_id, content, tags, created_at, updated_at, accessed_at,
                 source, memory_type, importance, importance_score, access_count, metadata
                 FROM memories WHERE id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![memory_id.0.to_string()], |row| memory_from_row(row))
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(m)) => Ok(Some(m)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn update_memory(&self, entry: &MemoryEntry) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let tags_json =
            serde_json::to_string(&entry.tags).map_err(|e| CoreError::Serialization(e))?;
        conn.execute(
            "UPDATE memories SET content = ?1, tags = ?2, updated_at = ?3, accessed_at = ?4,
             importance = ?5, importance_score = ?6, access_count = ?7, metadata = ?8
             WHERE id = ?9",
            params![
                entry.content,
                tags_json,
                entry.updated_at.to_rfc3339(),
                entry.accessed_at.map(|t| t.to_rfc3339()),
                entry.importance,
                entry.importance_score,
                entry.access_count as i64,
                entry.metadata.to_string(),
                entry.id.0.to_string(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn delete_memory(&self, memory_id: &MemoryId) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![memory_id.0.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute(
            "DELETE FROM memory_embeddings WHERE memory_id = ?1",
            params![memory_id.0.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?1",
            params![memory_id.0.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn list_memories(
        &self,
        agent_id: &Uuid,
        memory_type: Option<&str>,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<MemoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;

        let (sql, type_filter) = if memory_type.is_some() {
            (
                "SELECT id, agent_id, content, tags, created_at, updated_at, accessed_at,
                 source, memory_type, importance, importance_score, access_count, metadata
                 FROM memories WHERE agent_id = ?1 AND memory_type = ?2
                 ORDER BY importance DESC, created_at DESC LIMIT ?3 OFFSET ?4",
                true,
            )
        } else {
            (
                "SELECT id, agent_id, content, tags, created_at, updated_at, accessed_at,
                 source, memory_type, importance, importance_score, access_count, metadata
                 FROM memories WHERE agent_id = ?1
                 ORDER BY importance DESC, created_at DESC LIMIT ?2 OFFSET ?3",
                false,
            )
        };

        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let rows = if type_filter {
            stmt.query_map(
                params![
                    agent_id.to_string(),
                    memory_type.unwrap(),
                    limit as i64,
                    offset as i64,
                ],
                memory_from_row,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
        } else {
            stmt.query_map(
                params![agent_id.to_string(), limit as i64, offset as i64],
                memory_from_row,
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?
        };

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// List memories for an agent filtered by [`MemoryFilters`].
    ///
    /// Filtering by agent and time range happens in SQL; tags and memory_types
    /// (stored as JSON arrays) are enforced in memory after the row is loaded.
    pub fn list_memories_filtered(
        &self,
        filters: &MemoryFilters,
        limit: u64,
        offset: u64,
    ) -> CoreResult<Vec<MemoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut sql = String::from(
            "SELECT id, agent_id, content, tags, created_at, updated_at, accessed_at,
             source, memory_type, importance, importance_score, access_count, metadata
             FROM memories",
        );
        let mut clauses: Vec<String> = Vec::new();
        if filters.agent_id.is_some() {
            clauses.push("agent_id = ?".to_string());
        }
        if filters.since.is_some() {
            clauses.push("created_at >= ?".to_string());
        }
        if filters.until.is_some() {
            clauses.push("created_at <= ?".to_string());
        }
        if !clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }
        sql.push_str(" ORDER BY importance DESC, created_at DESC LIMIT ? OFFSET ?");

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut bindings: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(a) = filters.agent_id {
            bindings.push(Box::new(a.to_string()));
        }
        if let Some(t) = filters.since {
            bindings.push(Box::new(t.to_rfc3339()));
        }
        if let Some(t) = filters.until {
            bindings.push(Box::new(t.to_rfc3339()));
        }
        bindings.push(Box::new(limit as i64));
        bindings.push(Box::new(offset as i64));

        let refs: Vec<&dyn rusqlite::ToSql> = bindings.iter().map(|b| b.as_ref()).collect();
        let rows = stmt
            .query_map(refs.as_slice(), |row| memory_from_row(row))
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut all: Vec<MemoryEntry> = Vec::new();
        for row in rows {
            if let Ok(entry) = row {
                if !filters.tags.is_empty() && !filters.tags.iter().any(|t| entry.tags.contains(t))
                {
                    continue;
                }
                if !filters.memory_types.is_empty()
                    && !filters.memory_types.contains(&entry.memory_type)
                {
                    continue;
                }
                all.push(entry);
            }
        }
        Ok(all)
    }

    // --- FTS5 Search ---

    pub fn search_fts(&self, query: &str, limit: u64, offset: u64) -> CoreResult<Vec<MemoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.agent_id, m.content, m.tags, m.created_at, m.updated_at,
                 m.accessed_at, m.source, m.memory_type, m.importance, m.importance_score,
                 m.access_count, m.metadata
                 FROM memories m
                 INNER JOIN memories_fts fts ON m.rowid = fts.rowid
                 WHERE memories_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2 OFFSET ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![query, limit as i64, offset as i64], |row| {
                memory_from_row(row)
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// FTS5 search restricted to a single agent, returning ranked BM25 scores.
    pub fn search_fts_scored(
        &self,
        agent_id: &Uuid,
        query: &str,
        limit: u64,
    ) -> CoreResult<Vec<(MemoryEntry, f64)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT m.id, m.agent_id, m.content, m.tags, m.created_at, m.updated_at,
                 m.accessed_at, m.source, m.memory_type, m.importance, m.importance_score,
                 m.access_count, m.metadata, rank
                 FROM memories m
                 INNER JOIN memories_fts fts ON m.rowid = fts.rowid
                 WHERE memories_fts MATCH ?1 AND m.agent_id = ?2
                 ORDER BY rank
                 LIMIT ?3",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![query, agent_id.to_string(), limit as i64], |row| {
                let entry = memory_from_row(row)?;
                // FTS5 `rank` is negative; larger (closer to zero) is a
                // better match. Negate so higher = better for scoring.
                let raw_rank: f64 = row.get::<_, f64>(13).unwrap_or(0.0);
                Ok((entry, -raw_rank))
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    // --- Embedding storage (per-memory) ---

    pub fn store_embedding(&self, memory_id: &MemoryId, embedding: &[f32]) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let dimension = embedding.len() as i64;

        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, dimension)
             VALUES (?1, ?2, ?3)",
            params![memory_id.0.to_string(), blob, dimension],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Best-effort mirror into the memories_vec virtual table (sqlite-vec).
        // If the extension is unavailable this is a no-op error we swallow.
        let _ = conn.execute(
            "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?1, ?2)",
            params![memory_id.0.to_string(), blob],
        );
        Ok(())
    }

    pub fn get_embedding(&self, memory_id: &MemoryId) -> CoreResult<Option<Vec<f32>>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT embedding, dimension FROM memory_embeddings WHERE memory_id = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![memory_id.0.to_string()], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let _dimension: usize = row.get::<_, i64>(1)? as usize;
                let embedding: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok(embedding)
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn get_all_embeddings(&self, agent_id: &Uuid) -> CoreResult<Vec<(MemoryId, Vec<f32>)>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT m.id, me.embedding, me.dimension
                 FROM memories m
                 INNER JOIN memory_embeddings me ON m.id = me.memory_id
                 WHERE m.agent_id = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let results = stmt
            .query_map(params![agent_id.to_string()], |row| {
                let id = MemoryId(Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default());
                let blob: Vec<u8> = row.get(1)?;
                let _dimension: usize = row.get::<_, i64>(2)? as usize;
                let embedding: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok((id, embedding))
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(results.filter_map(|r| r.ok()).collect())
    }

    // --- Embedding cache (content-addressed) ---

    /// Compute the cache key for a piece of text under a given model.
    pub fn embedding_cache_key(model: &str, text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model.as_bytes());
        hasher.update(b":");
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Look up a cached embedding by `(model, text)` key.
    pub fn get_cached_embedding(&self, model: &str, text: &str) -> CoreResult<Option<Vec<f32>>> {
        let key = Self::embedding_cache_key(model, text);
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT embedding FROM embedding_cache WHERE key = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![key], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>())
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(e)) => Ok(Some(e)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    /// Store a cached embedding keyed by `(model, text)`.
    pub fn set_cached_embedding(
        &self,
        model: &str,
        text: &str,
        embedding: &[f32],
    ) -> CoreResult<()> {
        let key = Self::embedding_cache_key(model, text);
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        let dim = embedding.len() as i64;
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (key, embedding, dimension, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![key, blob, dim, model, Utc::now().to_rfc3339()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// List cached embedding rows (newest first), for inspection / eviction.
    pub fn list_cached_embeddings(&self, limit: u64) -> CoreResult<Vec<CachedEmbedding>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT key, embedding, model, created_at FROM embedding_cache
                 ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], |row| {
                let key: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                let model: String = row.get(2)?;
                let created_at: String = row.get(3)?;
                let embedding: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let created_at = DateTime::parse_from_rfc3339(&created_at)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                Ok(CachedEmbedding {
                    key,
                    embedding,
                    model,
                    created_at,
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    // --- File indexing ---

    /// Index (or re-index) a file: compute its SHA-256 checksum, record a row
    /// in `files`, and return the [`IndexedFile`] metadata. Does not chunk the
    /// content — call [`insert_chunk`][Self::insert_chunk] for that.
    pub fn index_file(&self, path: &str, content: &[u8]) -> CoreResult<IndexedFile> {
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(content);
            hex::encode(hasher.finalize())
        };
        let size = content.len() as u64;

        let fs_meta = std::fs::metadata(path).ok();
        let modified_at = fs_meta
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| {
                DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos())
                    .unwrap_or_else(Utc::now)
            })
            .unwrap_or_else(Utc::now);

        let id = Uuid::new_v4();
        let now = Utc::now();
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO files (id, path, checksum, size, modified_at, indexed_at, chunk_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)
             ON CONFLICT(path) DO UPDATE SET
                checksum = excluded.checksum,
                size = excluded.size,
                modified_at = excluded.modified_at,
                indexed_at = excluded.indexed_at",
            params![
                id.to_string(),
                path,
                checksum,
                size as i64,
                modified_at.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Resolve the canonical id (may differ from the newly generated one on
        // conflict-update).
        let stored: IndexedFile = conn
            .query_row(
                "SELECT id, path, checksum, size, modified_at, indexed_at, chunk_count
                 FROM files WHERE path = ?1",
                params![path],
                |row| {
                    Ok(IndexedFile {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                        path: row.get(1)?,
                        checksum: row.get(2)?,
                        size: row.get::<_, i64>(3)? as u64,
                        modified_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        indexed_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                            .map(|dt| dt.with_timezone(&Utc))
                            .unwrap_or_else(|_| Utc::now()),
                        chunk_count: row.get::<_, i64>(6)? as u32,
                    })
                },
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        Ok(stored)
    }

    /// Look up an indexed file by path.
    pub fn get_file_by_path(&self, path: &str) -> CoreResult<Option<IndexedFile>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, path, checksum, size, modified_at, indexed_at, chunk_count
                 FROM files WHERE path = ?1",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![path], |row| {
                Ok(IndexedFile {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    path: row.get(1)?,
                    checksum: row.get(2)?,
                    size: row.get::<_, i64>(3)? as u64,
                    modified_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    indexed_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_else(|_| Utc::now()),
                    chunk_count: row.get::<_, i64>(6)? as u32,
                })
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(f)) => Ok(Some(f)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    /// Returns `true` if the file at `path` has already been indexed with the
    /// same checksum (i.e. it is unchanged and need not be re-chunked).
    pub fn file_is_unchanged(&self, path: &str, content: &[u8]) -> CoreResult<bool> {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let checksum = hex::encode(hasher.finalize());
        Ok(match self.get_file_by_path(path)? {
            Some(f) => f.checksum == checksum,
            None => false,
        })
    }

    /// Delete an indexed file and all of its chunks (cascade).
    pub fn delete_file(&self, file_id: &Uuid) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM chunks WHERE file_id = ?1",
            params![file_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute(
            "DELETE FROM files WHERE id = ?1",
            params![file_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Chunk CRUD ---

    /// Insert a chunk, returning the persisted chunk (with its generated id).
    pub fn insert_chunk(&self, chunk: &MemoryChunk, file_id: &Uuid) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let (blob, dim): (Option<Vec<u8>>, Option<i64>) = match &chunk.embedding {
            Some(emb) => {
                let b: Vec<u8> = emb.iter().flat_map(|f| f.to_le_bytes()).collect();
                (Some(b), Some(emb.len() as i64))
            }
            None => (None, None),
        };
        conn.execute(
            "INSERT INTO chunks (id, file_id, content, embedding, dimension, chunk_index,
             token_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                chunk.id.to_string(),
                file_id.to_string(),
                chunk.content,
                blob,
                dim,
                chunk.chunk_index as i64,
                chunk.token_count as i64,
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;

        // Bump the parent file's chunk_count.
        conn.execute(
            "UPDATE files SET chunk_count = (
                SELECT COUNT(*) FROM chunks WHERE file_id = ?1
             ) WHERE id = ?1",
            params![file_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Retrieve all chunks belonging to a file, ordered by chunk index.
    pub fn get_chunks(&self, file_id: &Uuid) -> CoreResult<Vec<MemoryChunk>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, file_id, content, embedding, chunk_index, token_count
                 FROM chunks WHERE file_id = ?1 ORDER BY chunk_index ASC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![file_id.to_string()], chunk_from_row)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// FTS5 search across chunk content.
    pub fn search_chunks_fts(&self, query: &str, limit: u64) -> CoreResult<Vec<MemoryChunk>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT c.id, c.file_id, c.content, c.embedding, c.chunk_index, c.token_count
                 FROM chunks c
                 INNER JOIN chunks_fts fts ON c.rowid = fts.rowid
                 WHERE chunks_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![query, limit as i64], chunk_from_row)
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Delete a single chunk by id.
    pub fn delete_chunk(&self, chunk_id: &Uuid) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "DELETE FROM chunks WHERE id = ?1",
            params![chunk_id.to_string()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // --- Tags ---

    pub fn add_tag(&self, memory_id: &MemoryId, tag: &str) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO memory_tags (id, memory_id, tag) VALUES (?1, ?2, ?3)",
            params![Uuid::new_v4().to_string(), memory_id.0.to_string(), tag],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        // Also mirror into the JSON tags column on the memory row.
        let tags_json: String = conn
            .query_row(
                "SELECT tags FROM memories WHERE id = ?1",
                params![memory_id.0.to_string()],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| "[]".to_string());
        let mut tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
        if !tags.iter().any(|t| t == tag) {
            tags.push(tag.to_string());
            let new_json = serde_json::to_string(&tags).map_err(|e| CoreError::Serialization(e))?;
            conn.execute(
                "UPDATE memories SET tags = ?1 WHERE id = ?2",
                params![new_json, memory_id.0.to_string()],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    pub fn get_tags(&self, memory_id: &MemoryId) -> CoreResult<Vec<String>> {
        // Prefer the JSON column (single source of truth); fall back to the
        // memory_tags table for legacy rows.
        if let Some(entry) = self.get_memory(memory_id)? {
            if !entry.tags.is_empty() {
                return Ok(entry.tags);
            }
        }
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;

        let tags = stmt
            .query_map(params![memory_id.0.to_string()], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|e| CoreError::Storage(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(tags)
    }

    pub fn increment_access(&self, memory_id: &MemoryId) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "UPDATE memories
             SET access_count = access_count + 1, accessed_at = ?2
             WHERE id = ?1",
            params![memory_id.0.to_string(), Utc::now().to_rfc3339()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Delete memories older than `ttl` that have an importance below `min_importance`.
    /// Returns the number of deleted rows.
    pub fn expire_old_memories(
        &self,
        ttl: chrono::Duration,
        min_importance: f64,
    ) -> CoreResult<u64> {
        let cutoff = (Utc::now() - ttl).to_rfc3339();
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let deleted = conn
            .execute(
                "DELETE FROM memories
                 WHERE created_at < ?1 AND importance < ?2",
                params![cutoff, min_importance],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(deleted as u64)
    }

    // --- Meta key/value store ---

    pub fn set_meta(&self, key: &str, value: &str) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value, updated_at) VALUES (?1, ?2, ?3)",
            params![key, value, Utc::now().to_rfc3339()],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> CoreResult<Option<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT value FROM meta WHERE key = ?1")
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let mut rows = stmt
            .query_map(params![key], |row| row.get::<_, String>(0))
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        match rows.next() {
            Some(Ok(v)) => Ok(Some(v)),
            Some(Err(e)) => Err(CoreError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    pub fn delete_meta(&self, key: &str) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute("DELETE FROM meta WHERE key = ?1", params![key])
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // FTS index management
    // -----------------------------------------------------------------------

    /// List every memory across all agents (used by the index manager for
    /// reindexing and statistics). Ordered by agent then importance.
    pub fn list_memories_by_agent_all(&self) -> CoreResult<Vec<MemoryEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, agent_id, content, tags, created_at, updated_at, accessed_at,
                 source, memory_type, importance, importance_score, access_count, metadata
                 FROM memories ORDER BY agent_id, importance DESC",
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| memory_from_row(row))
            .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// A row of FTS index statistics.
    pub fn query_fts_stats(&self) -> CoreResult<FtsStatsRow> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let rows: i64 = conn
            .query_row("SELECT count(*) FROM memories_fts", [], |row| row.get(0))
            .unwrap_or(0);
        let size_bytes: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(length(content)), 0) FROM memories_fts",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        // Orphans: FTS rows whose rowid has no matching memory row.
        let orphans: i64 = conn
            .query_row(
                "SELECT count(*) FROM memories_fts f
                 WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.rowid = f.rowid)",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        Ok(FtsStatsRow {
            rows: rows as u64,
            size_bytes: size_bytes as u64,
            orphans: orphans as u64,
        })
    }

    /// Count tracked indexed files.
    pub fn count_files(&self) -> CoreResult<Option<u64>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        match conn.query_row("SELECT count(*) FROM files", [], |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(n) => Ok(Some(n as u64)),
            Err(_) => Ok(None),
        }
    }

    /// Count indexed file chunks.
    pub fn count_chunks(&self) -> CoreResult<Option<u64>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        match conn.query_row("SELECT count(*) FROM chunks", [], |row| {
            row.get::<_, i64>(0)
        }) {
            Ok(n) => Ok(Some(n as u64)),
            Err(_) => Ok(None),
        }
    }

    /// Re-insert a single memory's text into the FTS table.
    ///
    /// This is used by the index manager to repair a missing or stale FTS row
    /// after a partial failure or manual table surgery. It removes any
    /// existing index entry for the rowid first (the FTS5 special delete),
    /// then inserts a fresh one.
    pub fn reindex_memory(&self, entry: &MemoryEntry) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let rowid: Option<i64> = conn
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![entry.id.0.to_string()],
                |row| row.get(0),
            )
            .ok();
        let Some(rowid) = rowid else {
            return Ok(());
        };
        // FTS5 external-content tables cannot use UPSERT; delete-then-insert.
        conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
             VALUES ('delete', ?1, '', '')",
            params![rowid],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO memories_fts(rowid, content, memory_type) VALUES (?1, ?2, ?3)",
            params![rowid, entry.content, entry.memory_type],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Remove FTS rows whose memory no longer exists. Returns the number of
    /// rows deleted.
    pub fn prune_fts_orphans(&self) -> CoreResult<u64> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let orphan_rowids: Vec<i64> = {
            let mut stmt = conn
                .prepare(
                    "SELECT f.rowid FROM memories_fts f
                     WHERE NOT EXISTS (SELECT 1 FROM memories m WHERE m.rowid = f.rowid)",
                )
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, i64>(0))
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let mut removed = 0u64;
        for rowid in orphan_rowids {
            conn.execute(
                "INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
                 VALUES ('delete', ?1, '', '')",
                params![rowid],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
            removed += 1;
        }
        Ok(removed)
    }

    /// Low-level FTS repair: inject an index row for a raw rowid. Used to
    /// simulate or repair external-content-table drift where a rowid exists in
    /// the FTS index but not in the `memories` table (or vice versa). The
    /// caller must be careful: this bypasses the normal triggers.
    pub fn inject_fts_row(&self, rowid: i64, content: &str, memory_type: &str) -> CoreResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        conn.execute(
            "INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
             VALUES ('delete', ?1, '', '')",
            params![rowid],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO memories_fts(rowid, content, memory_type) VALUES (?1, ?2, ?3)",
            params![rowid, content, memory_type],
        )
        .map_err(|e| CoreError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Re-insert every chunk's content into the chunks_fts table. Returns the
    /// number of chunks re-indexed.
    pub fn reindex_chunks(&self) -> CoreResult<u64> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| CoreError::Internal(e.to_string()))?;
        let chunk_rows: Vec<(i64, String)> = {
            let mut stmt = conn
                .prepare("SELECT rowid, content FROM chunks")
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| CoreError::Storage(e.to_string()))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        let mut reindexed = 0u64;
        for (rowid, content) in chunk_rows {
            conn.execute(
                "INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', ?1, '')",
                params![rowid],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?1, ?2)",
                params![rowid, content],
            )
            .map_err(|e| CoreError::Storage(e.to_string()))?;
            reindexed += 1;
        }
        Ok(reindexed)
    }
}

/// Row-level statistics about the FTS index.
#[derive(Debug, Clone, Default)]
pub struct FtsStatsRow {
    pub rows: u64,
    pub size_bytes: u64,
    pub orphans: u64,
}

fn memory_from_row(row: &rusqlite::Row) -> rusqlite::Result<MemoryEntry> {
    let tags_str: String = row.get::<_, String>(3).unwrap_or_else(|_| "[]".to_string());
    let tags: Vec<String> = serde_json::from_str(&tags_str).unwrap_or_default();
    let accessed_str: Option<String> = row.get(6).ok();
    let accessed_at = accessed_str
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Utc));
    let importance: f64 = row.get(9).unwrap_or(0.0);
    let importance_score: f64 = row.get::<_, f64>(10).unwrap_or(importance);
    Ok(MemoryEntry {
        id: MemoryId(Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default()),
        agent_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        content: row.get(2)?,
        tags,
        embedding: None,
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        accessed_at,
        source: row.get(7)?,
        memory_type: row.get(8)?,
        importance,
        importance_score,
        access_count: row.get::<_, i64>(11)? as u64,
        metadata: serde_json::from_str(&row.get::<_, String>(12)?)
            .unwrap_or(serde_json::Value::Null),
    })
}

fn chunk_from_row(row: &rusqlite::Row) -> rusqlite::Result<MemoryChunk> {
    let id_str: String = row.get(0)?;
    let file_id_str: String = row.get(1)?;
    let content: String = row.get(2)?;
    let blob: Option<Vec<u8>> = row.get(3).ok();
    let embedding = blob.map(|b| {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    });
    let chunk_index: i64 = row.get(4)?;
    let token_count: i64 = row.get(5)?;
    Ok(MemoryChunk {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        file_id: Uuid::parse_str(&file_id_str).unwrap_or_default(),
        content,
        embedding,
        chunk_index: chunk_index as u32,
        token_count: token_count as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(agent_id: Uuid, content: &str) -> MemoryEntry {
        MemoryEntry::new(
            MemoryId(Uuid::new_v4()),
            agent_id,
            content.to_string(),
            "test".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        )
    }

    #[test]
    fn test_insert_and_get_memory() {
        let store = MemoryStore::in_memory().unwrap();
        let agent_id = Uuid::new_v4();
        let entry = make_entry(agent_id, "hello world");
        store.insert_memory(&entry).unwrap();

        let retrieved = store.get_memory(&entry.id).unwrap().unwrap();
        assert_eq!(retrieved.content, "hello world");
        assert_eq!(retrieved.agent_id, agent_id);
        assert!(retrieved.tags.is_empty());
        assert!(retrieved.accessed_at.is_none());
    }

    #[test]
    fn test_delete_memory() {
        let store = MemoryStore::in_memory().unwrap();
        let entry = make_entry(Uuid::new_v4(), "to delete");
        store.insert_memory(&entry).unwrap();
        store.delete_memory(&entry.id).unwrap();
        assert!(store.get_memory(&entry.id).unwrap().is_none());
    }

    #[test]
    fn test_list_memories() {
        let store = MemoryStore::in_memory().unwrap();
        let agent_id = Uuid::new_v4();
        store.insert_memory(&make_entry(agent_id, "first")).unwrap();
        store
            .insert_memory(&make_entry(agent_id, "second"))
            .unwrap();
        let memories = store.list_memories(&agent_id, None, 10, 0).unwrap();
        assert_eq!(memories.len(), 2);
    }

    #[test]
    fn test_search_fts() {
        let store = MemoryStore::in_memory().unwrap();
        let agent_id = Uuid::new_v4();
        store
            .insert_memory(&make_entry(agent_id, "Rust programming language"))
            .unwrap();
        store
            .insert_memory(&make_entry(agent_id, "Python scripting"))
            .unwrap();

        let results = store.search_fts("rust", 10, 0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "Rust programming language");
    }

    #[test]
    fn test_store_and_get_embedding() {
        let store = MemoryStore::in_memory().unwrap();
        let entry = make_entry(Uuid::new_v4(), "embedded content");
        store.insert_memory(&entry).unwrap();

        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        store.store_embedding(&entry.id, &embedding).unwrap();

        let retrieved = store.get_embedding(&entry.id).unwrap().unwrap();
        assert_eq!(retrieved.len(), 4);
        assert!((retrieved[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_increment_access_sets_accessed_at() {
        let store = MemoryStore::in_memory().unwrap();
        let entry = make_entry(Uuid::new_v4(), "access test");
        store.insert_memory(&entry).unwrap();

        store.increment_access(&entry.id).unwrap();
        store.increment_access(&entry.id).unwrap();

        let retrieved = store.get_memory(&entry.id).unwrap().unwrap();
        assert_eq!(retrieved.access_count, 2);
        assert!(retrieved.accessed_at.is_some());
    }

    #[test]
    fn test_tags_round_trip() {
        let store = MemoryStore::in_memory().unwrap();
        let entry = make_entry(Uuid::new_v4(), "tagged content");
        store.insert_memory(&entry).unwrap();

        store.add_tag(&entry.id, "important").unwrap();
        store.add_tag(&entry.id, "reference").unwrap();

        let tags = store.get_tags(&entry.id).unwrap();
        assert_eq!(tags.len(), 2);
        assert!(tags.contains(&"important".to_string()));
        assert!(tags.contains(&"reference".to_string()));

        // The JSON column should now carry the tags too.
        let retrieved = store.get_memory(&entry.id).unwrap().unwrap();
        assert!(retrieved.tags.contains(&"important".to_string()));
    }

    #[test]
    fn test_index_file_and_chunks() {
        let store = MemoryStore::in_memory().unwrap();
        let content = b"chunk one\nchunk two\nchunk three";
        let file = store.index_file("/tmp/test.txt", content).unwrap();
        assert_eq!(file.checksum.len(), 64);
        assert!(store.file_is_unchanged("/tmp/test.txt", content).unwrap());

        let chunk = MemoryChunk::new(file.id, 0, "chunk one".to_string(), 3);
        store.insert_chunk(&chunk, &file.id).unwrap();
        let chunks = store.get_chunks(&file.id).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "chunk one");

        // Chunk count on the file row should be updated.
        let file2 = store.get_file_by_path("/tmp/test.txt").unwrap().unwrap();
        assert_eq!(file2.chunk_count, 1);

        // FTS search over chunks.
        let hits = store.search_chunks_fts("chunk", 10).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn test_embedding_cache() {
        let store = MemoryStore::in_memory().unwrap();
        assert!(
            store
                .get_cached_embedding("text-embedding-3-small", "hello")
                .unwrap()
                .is_none()
        );
        let emb = vec![0.1, 0.2, 0.3];
        store
            .set_cached_embedding("text-embedding-3-small", "hello", &emb)
            .unwrap();
        let cached = store
            .get_cached_embedding("text-embedding-3-small", "hello")
            .unwrap()
            .unwrap();
        assert_eq!(cached.len(), 3);
    }

    #[test]
    fn test_meta_kv() {
        let store = MemoryStore::in_memory().unwrap();
        assert!(store.get_meta("missing").unwrap().is_none());
        store.set_meta("schema_version", "2").unwrap();
        assert_eq!(
            store.get_meta("schema_version").unwrap().as_deref(),
            Some("2")
        );
        store.delete_meta("schema_version").unwrap();
        assert!(store.get_meta("schema_version").unwrap().is_none());
    }

    #[test]
    fn test_expire_old_memories() {
        let store = MemoryStore::in_memory().unwrap();
        let mut old = make_entry(Uuid::new_v4(), "old and unimportant");
        old.importance = 0.1;
        old.created_at = Utc::now() - chrono::Duration::days(365);
        old.updated_at = old.created_at;
        store.insert_memory(&old).unwrap();

        let deleted = store
            .expire_old_memories(chrono::Duration::days(30), 0.5)
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(store.get_memory(&old.id).unwrap().is_none());
    }
}
