//! Memory RPC handlers.
//!
//! Provides `rpc_memory` (memory check, search, repair, refresh) and
//! `rpc_memory_import` (AI-assisted configuration import into the memory
//! store). Backed by [`opensquilla_memory::MemoryStore`] which uses SQLite
//! FTS5 for full-text search and blob-backed embeddings.

use opensquilla_core::error::AppError;
use opensquilla_core::types::MemoryId;
use opensquilla_memory::store::{MemoryEntry, MemoryStore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A lightweight handle around a memory store that can be cheaply cloned
/// across RPC handlers.
#[derive(Clone)]
pub struct MemoryHandle {
    store: Arc<MemoryStore>,
}

impl MemoryHandle {
    /// Create a new handle wrapping an in-memory SQLite store.
    pub fn in_memory() -> Result<Self, AppError> {
        let store = MemoryStore::in_memory()
            .map_err(|e| AppError::internal(format!("Failed to open memory store: {e}")))?;
        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Create a new handle wrapping a file-backed SQLite store.
    pub fn open(path: &str) -> Result<Self, AppError> {
        let store = MemoryStore::new(path)
            .map_err(|e| AppError::internal(format!("Failed to open memory store: {e}")))?;
        Ok(Self {
            store: Arc::new(store),
        })
    }

    /// Access the underlying store.
    pub fn store(&self) -> &MemoryStore {
        &self.store
    }
}

/// Response payload for `memory.check`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCheckResponse {
    pub agent_id: String,
    pub total_memories: u64,
    pub with_embeddings: u64,
    pub orphans: u64,
    pub healthy: bool,
}

/// Response payload for `memory.search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResponse {
    pub query: String,
    pub results: Vec<MemoryEntry>,
    pub count: usize,
}

/// Register memory RPC handlers on the given registry.
pub fn register_memory_handlers(registry: &mut RpcRegistry, handle: MemoryHandle) {
    let handle = Arc::new(handle);

    // memory.check — health/consistency check for an agent's memory corpus
    registry.register(rpc_handler("memory.check", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let agent_id = parse_agent_id(&params)?;
                let total = handle
                    .store()
                    .list_memories(&agent_id, None, u64::MAX, 0)
                    .map_err(store_err)?
                    .len() as u64;

                let with_embeddings = handle
                    .store()
                    .get_all_embeddings(&agent_id)
                    .map_err(store_err)?
                    .len() as u64;

                // Orphaned embeddings point to memories that no longer exist.
                let embeddings = handle
                    .store()
                    .get_all_embeddings(&agent_id)
                    .map_err(store_err)?;
                let mut orphans = 0u64;
                for (mid, _) in &embeddings {
                    if handle.store().get_memory(mid).map_err(store_err)?.is_none() {
                        orphans += 1;
                    }
                }

                let healthy = orphans == 0;
                serde_json::to_value(MemoryCheckResponse {
                    agent_id: agent_id.to_string(),
                    total_memories: total,
                    with_embeddings,
                    orphans,
                    healthy,
                })
                .map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // memory.search — FTS5 full-text search across memories
    registry.register(rpc_handler("memory.search", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'query' parameter"))?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20);
                let offset = params.get("offset").and_then(|v| v.as_u64()).unwrap_or(0);

                let results = handle
                    .store()
                    .search_fts(query, limit, offset)
                    .map_err(store_err)?;
                let count = results.len();
                serde_json::to_value(MemorySearchResponse {
                    query: query.to_string(),
                    results,
                    count,
                })
                .map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // memory.repair — remove orphaned embeddings for deleted memories
    registry.register(rpc_handler("memory.repair", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let agent_id = parse_agent_id(&params)?;
                let embeddings = handle
                    .store()
                    .get_all_embeddings(&agent_id)
                    .map_err(store_err)?;
                let mut repaired = 0u64;
                for (mid, _) in &embeddings {
                    if handle.store().get_memory(mid).map_err(store_err)?.is_none() {
                        // delete_memory also removes the embedding row.
                        handle.store().delete_memory(mid).map_err(store_err)?;
                        repaired += 1;
                    }
                }
                Ok(serde_json::json!({
                    "agent_id": agent_id.to_string(),
                    "repaired": repaired,
                    "status": "ok",
                }))
            }
        }
    }));

    // memory.refresh — bump access counters for a set of memory ids (touch)
    registry.register(rpc_handler("memory.refresh", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let ids = params
                    .get("memory_ids")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| AppError::bad_request("Missing 'memory_ids' array"))?;
                let mut touched = 0u64;
                let mut missing = 0u64;
                for raw in ids {
                    let id_str = raw
                        .as_str()
                        .ok_or_else(|| AppError::bad_request("memory_ids must be strings"))?;
                    let mid = MemoryId(
                        Uuid::parse_str(id_str)
                            .map_err(|_| AppError::bad_request("Invalid memory id"))?,
                    );
                    if handle
                        .store()
                        .get_memory(&mid)
                        .map_err(store_err)?
                        .is_some()
                    {
                        handle.store().increment_access(&mid).map_err(store_err)?;
                        touched += 1;
                    } else {
                        missing += 1;
                    }
                }
                Ok(serde_json::json!({
                    "touched": touched,
                    "missing": missing,
                }))
            }
        }
    }));

    // memory.import — AI-assisted configuration import into the memory store.
    //
    // Accepts a structured document (text + optional metadata) and inserts it
    // as a new memory entry tagged with `source: "import"`. The caller is
    // expected to have already run the LLM extraction step; this handler
    // performs the durable write.
    registry.register(rpc_handler("memory.import", {
        let handle = handle.clone();
        move |params| {
            let handle = handle.clone();
            async move {
                let agent_id = parse_agent_id(&params)?;
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'content' parameter"))?;
                let memory_type = params
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("imported");
                let importance = params
                    .get("importance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                let metadata = params
                    .get("metadata")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let now = chrono::Utc::now();
                let mut entry = MemoryEntry::new(
                    MemoryId::new(),
                    agent_id,
                    content.to_string(),
                    "import".to_string(),
                    memory_type.to_string(),
                    importance,
                    metadata,
                );
                entry.created_at = now;
                entry.updated_at = now;

                handle.store().insert_memory(&entry).map_err(store_err)?;

                // Persist tags if supplied.
                if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
                    for tag in tags {
                        if let Some(t) = tag.as_str() {
                            handle.store().add_tag(&entry.id, t).map_err(store_err)?;
                        }
                    }
                }

                serde_json::to_value(&entry).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));
}

fn parse_agent_id(params: &serde_json::Value) -> Result<Uuid, AppError> {
    params
        .get("agent_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'agent_id' parameter"))
        .and_then(|s| Uuid::parse_str(s).map_err(|_| AppError::bad_request("Invalid agent_id")))
}

fn store_err(e: opensquilla_core::error::CoreError) -> AppError {
    AppError::internal(format!("Memory store error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_import_and_search() {
        let handle = MemoryHandle::in_memory().unwrap();
        let mut registry = RpcRegistry::new();
        register_memory_handlers(&mut registry, handle);

        let agent = Uuid::new_v4();
        let params = serde_json::json!({
            "agent_id": agent.to_string(),
            "content": "The user prefers concise answers without filler text.",
            "memory_type": "preference",
            "importance": 0.9,
            "tags": ["style", "communication"],
        });
        let result = registry.dispatch("memory.import", params).await;
        assert!(result.is_some());
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["source"], "import");

        // Search should find the imported memory.
        let params = serde_json::json!({"query": "concise answers"});
        let result = registry.dispatch("memory.search", params).await;
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        // Check reports a healthy corpus.
        let params = serde_json::json!({"agent_id": agent.to_string()});
        let result = registry.dispatch("memory.check", params).await;
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["total_memories"], 1);
        assert_eq!(resp["healthy"], true);
    }
}
