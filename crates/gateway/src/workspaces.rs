//! Workspaces RPC handlers.
//!
//! Provides `rpc_workspaces` for project workspace lifecycle: create, list,
//! open, close, and delete workspace records. Each workspace tracks a project
//! directory and its open/closed state.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A project workspace record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub path: String,
    pub is_open: bool,
    pub created_at: DateTime<Utc>,
    pub last_opened_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

/// In-memory workspace store.
#[derive(Clone, Default)]
pub struct WorkspaceStore {
    workspaces: Arc<Mutex<HashMap<String, Workspace>>>,
}

impl WorkspaceStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a workspace.
    pub fn upsert(&self, workspace: Workspace) {
        self.workspaces
            .lock()
            .insert(workspace.id.clone(), workspace);
    }

    /// Get a workspace by id.
    pub fn get(&self, id: &str) -> Option<Workspace> {
        self.workspaces.lock().get(id).cloned()
    }

    /// Remove a workspace.
    pub fn remove(&self, id: &str) -> Option<Workspace> {
        self.workspaces.lock().remove(id)
    }

    /// List all workspaces.
    pub fn list(&self) -> Vec<Workspace> {
        let mut workspaces: Vec<Workspace> = self.workspaces.lock().values().cloned().collect();
        workspaces.sort_by(|a, b| a.name.cmp(&b.name));
        workspaces
    }

    /// Update a workspace in place.
    pub fn update<F>(&self, id: &str, f: F) -> Option<Workspace>
    where
        F: FnOnce(&mut Workspace),
    {
        let mut workspaces = self.workspaces.lock();
        if let Some(ws) = workspaces.get_mut(id) {
            f(ws);
            return Some(ws.clone());
        }
        None
    }
}

/// Register workspaces RPC handlers on the given registry.
pub fn register_workspaces_handlers(registry: &mut RpcRegistry, store: WorkspaceStore) {
    let store = Arc::new(store);

    // workspaces.create — register a new project workspace
    registry.register(rpc_handler("workspaces.create", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?;
                let path = params
                    .get("path")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'path' parameter"))?;

                // Verify the path exists on disk.
                let path_buf = PathBuf::from(path);
                if !path_buf.exists() {
                    return Err(AppError::bad_request(format!(
                        "Workspace path does not exist: {path}"
                    )));
                }

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
                let workspace = Workspace {
                    id: Uuid::new_v4().to_string(),
                    name: name.to_string(),
                    path: path.to_string(),
                    is_open: false,
                    created_at: now,
                    last_opened_at: None,
                    metadata,
                };
                store.upsert(workspace.clone());
                Ok(serde_json::to_value(workspace)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // workspaces.get — fetch a workspace by id
    registry.register(rpc_handler("workspaces.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                match store.get(id) {
                    Some(ws) => {
                        Ok(serde_json::to_value(ws)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Workspace '{id}' not found"))),
                }
            }
        }
    }));

    // workspaces.list — list all workspaces
    registry.register(rpc_handler("workspaces.list", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let workspaces = store.list();
                Ok(serde_json::json!({
                    "workspaces": workspaces,
                    "count": workspaces.len(),
                }))
            }
        }
    }));

    // workspaces.open — mark a workspace as open
    registry.register(rpc_handler("workspaces.open", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let updated = store.update(id, |ws| {
                    ws.is_open = true;
                    ws.last_opened_at = Some(Utc::now());
                });
                match updated {
                    Some(ws) => {
                        Ok(serde_json::to_value(ws)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Workspace '{id}' not found"))),
                }
            }
        }
    }));

    // workspaces.close — mark a workspace as closed
    registry.register(rpc_handler("workspaces.close", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let updated = store.update(id, |ws| {
                    ws.is_open = false;
                });
                match updated {
                    Some(ws) => {
                        Ok(serde_json::to_value(ws)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Workspace '{id}' not found"))),
                }
            }
        }
    }));

    // workspaces.update — update a workspace's name/path/metadata
    registry.register(rpc_handler("workspaces.update", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let name = params.get("name").and_then(|v| v.as_str());
                let path = params.get("path").and_then(|v| v.as_str());

                // Validate path if provided.
                if let Some(p) = path {
                    let path_buf = PathBuf::from(p);
                    if !path_buf.exists() {
                        return Err(AppError::bad_request(format!(
                            "Workspace path does not exist: {p}"
                        )));
                    }
                }

                let updated = store.update(id, |ws| {
                    if let Some(n) = name {
                        ws.name = n.to_string();
                    }
                    if let Some(p) = path {
                        ws.path = p.to_string();
                    }
                    if let Some(meta) = params.get("metadata").and_then(|v| v.as_object()) {
                        for (k, v) in meta {
                            if let Some(s) = v.as_str() {
                                ws.metadata.insert(k.clone(), s.to_string());
                            }
                        }
                    }
                });
                match updated {
                    Some(ws) => {
                        Ok(serde_json::to_value(ws)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Workspace '{id}' not found"))),
                }
            }
        }
    }));

    // workspaces.delete — delete a workspace record
    registry.register(rpc_handler("workspaces.delete", {
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
                    None => Err(AppError::not_found(format!("Workspace '{id}' not found"))),
                }
            }
        }
    }));

    // workspaces.open_list — list only open workspaces
    registry.register(rpc_handler("workspaces.open_list", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let open: Vec<Workspace> = store.list().into_iter().filter(|w| w.is_open).collect();
                Ok(serde_json::json!({
                    "workspaces": open,
                    "count": open.len(),
                }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_workspaces_crud_with_real_dir() {
        let store = WorkspaceStore::new();
        let mut registry = RpcRegistry::new();
        register_workspaces_handlers(&mut registry, store);

        let temp_dir = std::env::temp_dir().join(format!("osq-ws-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();

        let params = serde_json::json!({
            "name": "Test Project",
            "path": temp_dir.to_string_lossy(),
        });
        let r = registry.dispatch("workspaces.create", params).await;
        let resp = r.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();
        assert_eq!(resp["is_open"], false);

        let r = registry
            .dispatch("workspaces.open", serde_json::json!({"id": id}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["is_open"], true);

        let r = registry
            .dispatch("workspaces.open_list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        let r = registry
            .dispatch("workspaces.close", serde_json::json!({"id": id}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["is_open"], false);

        let r = registry
            .dispatch("workspaces.delete", serde_json::json!({"id": id}))
            .await;
        assert!(r.unwrap().is_ok());

        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[tokio::test]
    async fn test_workspaces_create_missing_path() {
        let store = WorkspaceStore::new();
        let mut registry = RpcRegistry::new();
        register_workspaces_handlers(&mut registry, store);

        let params = serde_json::json!({
            "name": "Bad",
            "path": "/nonexistent/xyz/path",
        });
        let r = registry.dispatch("workspaces.create", params).await;
        assert!(r.unwrap().is_err());
    }
}
