//! Memory tools: memory_save, memory_search, memory_delete, memory_list.
//!
//! Provide persistent agent memory backed by `opensquilla-memory`'s
//! [`MemoryStore`] (SQLite + FTS5). The store is a `Clone` handle sharing one
//! underlying connection, so construct it once and pass clones to each tool:
//!
//! ```ignore
//! let store = MemoryStore::new(&path)?;
//! let save = MemorySaveTool::new(store.clone());
//! let search = MemorySearchTool::new(store.clone());
//! ```
//!
//! All synchronous SQLite calls are wrapped in `tokio::task::spawn_blocking`
//! so they never stall the async executor.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_core::types::MemoryId;
use opensquilla_memory::MemoryStore;
use opensquilla_memory::store::MemoryEntry;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// A helper for parsing common memory tool arguments.
fn parse_agent_id(params: &Value) -> Result<Uuid, ToolError> {
    let raw = params["agent_id"]
        .as_str()
        .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'agent_id'"))?;
    Uuid::parse_str(raw)
        .map_err(|e| ToolError::invalid_args(format!("Invalid 'agent_id' UUID '{}': {}", raw, e)))
}

/// A helper for mapping memory-store errors to tool errors.
fn map_store_error(op: &str, err: impl std::fmt::Display) -> ToolError {
    ToolError::new("MEMORY_ERROR", format!("Memory {} failed: {}", op, err))
}

/// Tool for storing a memory with optional tags.
pub struct MemorySaveTool {
    store: MemoryStore,
}

impl MemorySaveTool {
    /// Create a new memory_save tool backed by the given store.
    ///
    /// The store is `Clone` and shares one underlying connection, so pass
    /// clones of the same store to all four memory tools.
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }
}

impl Default for MemorySaveTool {
    fn default() -> Self {
        Self::new(MemoryStore::in_memory().expect("in-memory memory store"))
    }
}

#[async_trait]
impl Tool for MemorySaveTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "memory_save",
                concat!(
                    "Store a memory for an agent. Memories are persistent and searchable ",
                    "via memory_search. Optional tags can be attached for classification.",
),
                HashMap::from([
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The memory content to store"),
                    ),
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::required_string(
                            "The agent UUID this memory belongs to",
                        ),
                    ),
                    (
                        "source".to_string(),
                        ParameterDefinition::string(
                            "Where the memory came from (e.g. conversation, user, tool)",
                        )
                        .default(serde_json::json!("tool")),
                    ),
                    (
                        "memory_type".to_string(),
                        ParameterDefinition::string(
                            "Type of memory (e.g. general, preference, fact)",
                        )
                        .default(serde_json::json!("general")),
                    ),
                    (
                        "importance".to_string(),
                        ParameterDefinition::integer("Importance score 0-100")
                            .default(serde_json::json!(50)),
                    ),
                    (
                        "tags".to_string(),
                        ParameterDefinition::array(
                            "Optional tags to attach",
                            ParameterDefinition::string("tag"),
                        ),
                    ),
                    (
                        "metadata".to_string(),
                        ParameterDefinition::string("Optional metadata as a JSON object string"),
                    ),
                ]),
            )
            .category("memory")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'content'"))?
            .to_string();
        if content.trim().is_empty() {
            return Err(ToolError::invalid_args("'content' must not be empty"));
        }

        let agent_id = parse_agent_id(&params)?;
        let source = params["source"].as_str().unwrap_or("tool").to_string();
        let memory_type = params["memory_type"]
            .as_str()
            .unwrap_or("general")
            .to_string();
        let importance =
            (params["importance"].as_i64().unwrap_or(50) as f64 / 100.0).clamp(0.0, 1.0);
        let tags: Vec<String> = params["tags"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let metadata: Value = params["metadata"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null);

        let entry = MemoryEntry::new(
            MemoryId::new(),
            agent_id,
            content,
            source,
            memory_type,
            importance,
            metadata,
        )
        .with_tags(tags.clone());

        let store = self.store.clone();
        let memory_id = entry.id;
        let tags_for_insert = tags.clone();

        // SQLite I/O is synchronous; run it off the async executor.
        tokio::task::spawn_blocking(move || -> Result<(), ToolError> {
            store
                .insert_memory(&entry)
                .map_err(|e| map_store_error("save", e))?;
            for tag in &tags_for_insert {
                store
                    .add_tag(&entry.id, tag)
                    .map_err(|e| map_store_error("tag", e))?;
            }
            Ok(())
        })
        .await
        .map_err(|e| ToolError::new("MEMORY_ERROR", format!("Memory save task failed: {}", e)))??;

        let data = serde_json::json!({
            "memory_id": memory_id.0.to_string(),
            "agent_id": agent_id.to_string(),
            "tags": tags,
            "importance": importance,
        });

        Ok(ToolOutput::success_with_data(
            format!("Saved memory {} for agent {}", memory_id.0, agent_id),
            data,
        ))
    }
}

/// Tool for full-text searching stored memories.
pub struct MemorySearchTool {
    store: MemoryStore,
}

impl MemorySearchTool {
    /// Create a new memory_search tool backed by the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }
}

impl Default for MemorySearchTool {
    fn default() -> Self {
        Self::new(MemoryStore::in_memory().expect("in-memory memory store"))
    }
}

#[async_trait]
impl Tool for MemorySearchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "memory_search",
                concat!(
                    "Full-text search over stored memories using SQLite FTS5. ",
                    "Returns matching memories ordered by relevance.",
),
                HashMap::from([
                    (
                        "query".to_string(),
                        ParameterDefinition::required_string("The search query text"),
                    ),
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::string("Optional agent UUID to restrict results to"),
                    ),
                    (
                        "limit".to_string(),
                        ParameterDefinition::integer("Maximum number of results")
                            .default(serde_json::json!(10)),
                    ),
                    (
                        "offset".to_string(),
                        ParameterDefinition::integer("Result offset for pagination")
                            .default(serde_json::json!(0)),
                    ),
                ]),
            )
            .category("memory")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let query = params["query"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'query'"))?
            .to_string();
        let limit = params["limit"].as_i64().unwrap_or(10).max(1).min(100) as u64;
        let offset = params["offset"].as_i64().unwrap_or(0).max(0) as u64;
        let agent_filter = params["agent_id"]
            .as_str()
            .map(Uuid::parse_str)
            .transpose()
            .map_err(|e| ToolError::invalid_args(format!("Invalid 'agent_id' UUID: {}", e)))?;

        let store = self.store.clone();
        let query_for_search = query.clone();
        let results =
            tokio::task::spawn_blocking(move || -> Result<Vec<MemoryEntry>, ToolError> {
                store
                    .search_fts(&query_for_search, limit, offset)
                    .map_err(|e| map_store_error("search", e))
            })
            .await
            .map_err(|e| {
                ToolError::new("MEMORY_ERROR", format!("Memory search task failed: {}", e))
            })??;

        // The store-level FTS query is global; narrow to the requested agent
        // if one was given.
        let results: Vec<MemoryEntry> = results
            .into_iter()
            .filter(|m| agent_filter.map_or(true, |id| m.agent_id == id))
            .collect();

        let items: Vec<Value> = results
            .iter()
            .map(|m| {
                serde_json::json!({
                    "memory_id": m.id.0.to_string(),
                    "agent_id": m.agent_id.to_string(),
                    "content": m.content,
                    "source": m.source,
                    "memory_type": m.memory_type,
                    "importance": m.importance,
                    "created_at": m.created_at.to_rfc3339(),
                    "updated_at": m.updated_at.to_rfc3339(),
                    "metadata": m.metadata,
                })
            })
            .collect();

        let data = serde_json::json!({
            "query": query,
            "count": items.len(),
            "results": items,
        });

        let content = if items.is_empty() {
            format!("No memories found for query '{}'", query)
        } else {
            serde_json::to_string_pretty(&data).unwrap_or_default()
        };

        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for deleting a stored memory.
pub struct MemoryDeleteTool {
    store: MemoryStore,
}

impl MemoryDeleteTool {
    /// Create a new memory_delete tool backed by the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }
}

impl Default for MemoryDeleteTool {
    fn default() -> Self {
        Self::new(MemoryStore::in_memory().expect("in-memory memory store"))
    }
}

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "memory_delete",
                "Delete a memory by its ID.",
                HashMap::from([(
                    "memory_id".to_string(),
                    ParameterDefinition::required_string("The UUID of the memory to delete"),
                )]),
            )
            .category("memory")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let raw = params["memory_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'memory_id'"))?;
        let id = Uuid::parse_str(raw).map(MemoryId).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'memory_id' UUID '{}': {}", raw, e))
        })?;

        let store = self.store.clone();
        tokio::task::spawn_blocking(move || -> Result<(), ToolError> {
            store
                .delete_memory(&id)
                .map_err(|e| map_store_error("delete", e))
        })
        .await
        .map_err(|e| {
            ToolError::new("MEMORY_ERROR", format!("Memory delete task failed: {}", e))
        })??;

        let data = serde_json::json!({ "memory_id": raw });
        Ok(ToolOutput::success(format!("Deleted memory {}", raw)).with_data(data))
    }
}

/// Tool for listing an agent's memories.
pub struct MemoryListTool {
    store: MemoryStore,
}

impl MemoryListTool {
    /// Create a new memory_list tool backed by the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self { store }
    }
}

impl Default for MemoryListTool {
    fn default() -> Self {
        Self::new(MemoryStore::in_memory().expect("in-memory memory store"))
    }
}

#[async_trait]
impl Tool for MemoryListTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "memory_list",
                "List an agent's stored memories, optionally filtered by type, with pagination.",
                HashMap::from([
                    (
                        "agent_id".to_string(),
                        ParameterDefinition::required_string("The agent UUID to list memories for"),
                    ),
                    (
                        "memory_type".to_string(),
                        ParameterDefinition::string("Optional memory type filter"),
                    ),
                    (
                        "limit".to_string(),
                        ParameterDefinition::integer("Maximum number of results")
                            .default(serde_json::json!(50)),
                    ),
                    (
                        "offset".to_string(),
                        ParameterDefinition::integer("Result offset for pagination")
                            .default(serde_json::json!(0)),
                    ),
                ]),
            )
            .category("memory")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let agent_id = parse_agent_id(&params)?;
        // Own the filter so it can be moved into the 'static blocking closure.
        let memory_type = params["memory_type"].as_str().map(String::from);
        let limit = params["limit"].as_i64().unwrap_or(50).max(1).min(500) as u64;
        let offset = params["offset"].as_i64().unwrap_or(0).max(0) as u64;

        let store = self.store.clone();
        let results =
            tokio::task::spawn_blocking(move || -> Result<Vec<MemoryEntry>, ToolError> {
                store
                    .list_memories(&agent_id, memory_type.as_deref(), limit, offset)
                    .map_err(|e| map_store_error("list", e))
            })
            .await
            .map_err(|e| {
                ToolError::new("MEMORY_ERROR", format!("Memory list task failed: {}", e))
            })??;

        let items: Vec<Value> = results
            .iter()
            .map(|m| {
                serde_json::json!({
                    "memory_id": m.id.0.to_string(),
                    "agent_id": m.agent_id.to_string(),
                    "content": m.content,
                    "source": m.source,
                    "memory_type": m.memory_type,
                    "importance": m.importance,
                    "access_count": m.access_count,
                    "created_at": m.created_at.to_rfc3339(),
                    "updated_at": m.updated_at.to_rfc3339(),
                    "metadata": m.metadata,
                })
            })
            .collect();

        let data = serde_json::json!({
            "agent_id": agent_id.to_string(),
            "count": items.len(),
            "limit": limit,
            "offset": offset,
            "results": items,
        });

        let content = serde_json::to_string_pretty(&data).unwrap_or_default();
        Ok(ToolOutput::success(content).with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> MemoryStore {
        MemoryStore::in_memory().expect("in-memory store")
    }

    #[tokio::test]
    async fn test_memory_save_and_search_roundtrip() {
        let store = test_store();
        let save = MemorySaveTool::new(store.clone());
        let search = MemorySearchTool::new(store.clone());
        let agent = Uuid::new_v4();

        let result = save
            .execute(serde_json::json!({
                "content": "The capital of France is Paris",
                "agent_id": agent.to_string(),
                "tags": ["geography"],
            }))
            .await;
        assert!(result.is_ok(), "save failed: {:?}", result.err());
        let output = result.unwrap();
        let memory_id = output.data.as_ref().unwrap()["memory_id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!memory_id.is_empty());

        let result = search
            .execute(serde_json::json!({
                "query": "France",
                "agent_id": agent.to_string(),
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        let data = output.data.unwrap();
        assert_eq!(data["count"].as_u64(), Some(1));
        assert!(
            data["results"][0]["content"]
                .as_str()
                .unwrap()
                .contains("France")
        );
    }

    #[tokio::test]
    async fn test_memory_list_and_delete() {
        let store = test_store();
        let save = MemorySaveTool::new(store.clone());
        let list = MemoryListTool::new(store.clone());
        let delete = MemoryDeleteTool::new(store.clone());
        let agent = Uuid::new_v4();

        save.execute(serde_json::json!({
            "content": "Remember this fact",
            "agent_id": agent.to_string(),
        }))
        .await
        .unwrap();

        let result = list
            .execute(serde_json::json!({ "agent_id": agent.to_string() }))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().data.unwrap()["count"].as_u64(), Some(1));

        let result = delete
            .execute(serde_json::json!({
                "memory_id": save.execute(serde_json::json!({
                    "content": "second",
                    "agent_id": agent.to_string(),
                })).await.unwrap().data.unwrap()["memory_id"].as_str().unwrap().to_string(),
            }))
            .await;
        assert!(result.is_ok());
    }
}
