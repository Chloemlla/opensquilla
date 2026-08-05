//! Agents RPC handlers.
//!
//! Provides `rpc_agents` for agent CRUD and workspace file checks. Agent
//! records are held in memory; workspace checks verify the existence of
//! expected project files on disk.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use opensquilla_core::types::AgentId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::rpc::{rpc_handler, RpcRegistry};

/// An agent record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub id: String,
    pub name: String,
    pub model: String,
    pub provider: String,
    pub workspace: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Result of a workspace file check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceCheckResult {
    pub workspace: String,
    pub exists: bool,
    pub files: Vec<FileCheck>,
    pub all_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCheck {
    pub path: String,
    pub exists: bool,
    pub size_bytes: u64,
}

/// In-memory agent store.
#[derive(Clone, Default)]
pub struct AgentStore {
    agents: Arc<Mutex<HashMap<String, AgentRecord>>>,
}

impl AgentStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace an agent.
    pub fn upsert(&self, agent: AgentRecord) {
        self.agents.lock().insert(agent.id.clone(), agent);
    }

    /// Get an agent by id.
    pub fn get(&self, id: &str) -> Option<AgentRecord> {
        self.agents.lock().get(id).cloned()
    }

    /// Remove an agent.
    pub fn remove(&self, id: &str) -> Option<AgentRecord> {
        self.agents.lock().remove(id)
    }

    /// List all agents.
    pub fn list(&self) -> Vec<AgentRecord> {
        let mut agents: Vec<AgentRecord> = self.agents.lock().values().cloned().collect();
        agents.sort_by(|a, b| a.name.cmp(&b.name));
        agents
    }
}

/// Register agents RPC handlers on the given registry.
pub fn register_agents_handlers(registry: &mut RpcRegistry, store: AgentStore) {
    let store = Arc::new(store);

    // agents.create — create a new agent
    registry.register(rpc_handler("agents.create", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?;
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let provider = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let workspace = params
                    .get("workspace")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let metadata: HashMap<String, String> = params
                    .get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();

                let now = Utc::now();
                let agent = AgentRecord {
                    id: AgentId::new().to_string(),
                    name: name.to_string(),
                    model,
                    provider,
                    workspace,
                    created_at: now,
                    updated_at: now,
                    metadata,
                };
                store.upsert(agent.clone());
                Ok(serde_json::to_value(agent)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // agents.get — fetch an agent by id
    registry.register(rpc_handler("agents.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                match store.get(id) {
                    Some(agent) => Ok(serde_json::to_value(agent)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Agent '{id}' not found"))),
                }
            }
        }
    }));

    // agents.list — list all agents
    registry.register(rpc_handler("agents.list", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let agents = store.list();
                Ok(serde_json::json!({
                    "agents": agents,
                    "count": agents.len(),
                }))
            }
        }
    }));

    // agents.update — update an agent's mutable fields
    registry.register(rpc_handler("agents.update", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let mut agent = store.get(id).ok_or_else(|| {
                    AppError::not_found(format!("Agent '{id}' not found"))
                })?;
                if let Some(name) = params.get("name").and_then(|v| v.as_str()) {
                    agent.name = name.to_string();
                }
                if let Some(model) = params.get("model").and_then(|v| v.as_str()) {
                    agent.model = model.to_string();
                }
                if let Some(provider) = params.get("provider").and_then(|v| v.as_str()) {
                    agent.provider = provider.to_string();
                }
                if let Some(workspace) = params.get("workspace").and_then(|v| v.as_str()) {
                    agent.workspace = Some(workspace.to_string());
                }
                agent.updated_at = Utc::now();
                store.upsert(agent.clone());
                Ok(serde_json::to_value(agent)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // agents.delete — delete an agent
    registry.register(rpc_handler("agents.delete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                match store.remove(id) {
                    Some(_) => Ok(serde_json::json!({"deleted": true, "id": id})),
                    None => Err(AppError::not_found(format!("Agent '{id}' not found"))),
                }
            }
        }
    }));

    // agents.check_workspace — verify expected files exist in an agent's workspace
    registry.register(rpc_handler("agents.check_workspace", {
        move |params| {
            let workspace = params
                .get("workspace")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::bad_request("Missing 'workspace' parameter"))?;
            let required_files: Vec<String> = params
                .get("files")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        "AGENTS.md".to_string(),
                        "SOUL.md".to_string(),
                        ".agents".to_string(),
                    ]
                });

            let ws_path = PathBuf::from(workspace);
            let ws_exists = ws_path.exists();
            let mut checks = Vec::new();
            let mut all_present = true;

            for file in &required_files {
                let path = ws_path.join(file);
                let (exists, size) = if path.exists() {
                    (true, std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0))
                } else {
                    (false, 0)
                };
                if !exists {
                    all_present = false;
                }
                checks.push(FileCheck {
                    path: file.clone(),
                    exists,
                    size_bytes: size,
                });
            }

            let result = WorkspaceCheckResult {
                workspace: workspace.to_string(),
                exists: ws_exists,
                files: checks,
                all_present,
            };
            Ok(serde_json::to_value(result)
                .map_err(|e| AppError::internal(e.to_string()))?)
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agents_crud() {
        let store = AgentStore::new();
        let mut registry = RpcRegistry::new();
        register_agents_handlers(&mut registry, store);

        let params = serde_json::json!({
            "name": "Code Agent",
            "model": "claude-sonnet-4",
            "provider": "anthropic",
        });
        let r = registry.dispatch("agents.create", params).await;
        let resp = r.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();
        assert_eq!(resp["name"], "Code Agent");

        let r = registry
            .dispatch("agents.get", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry.dispatch("agents.list", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        let r = registry
            .dispatch(
                "agents.update",
                serde_json::json!({"id": id, "model": "gpt-4o"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["model"], "gpt-4o");

        let r = registry
            .dispatch("agents.delete", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_agents_check_workspace_missing() {
        let store = AgentStore::new();
        let mut registry = RpcRegistry::new();
        register_agents_handlers(&mut registry, store);

        let r = registry
            .dispatch(
                "agents.check_workspace",
                serde_json::json!({"workspace": "/nonexistent/xyz"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["exists"], false);
        assert_eq!(resp["all_present"], false);
    }
}
