//! Skills RPC handlers.
//!
//! Provides `rpc_skills` for skill directory scanning, install, and
//! enable/disable operations, backed by [`opensquilla_skills::SkillLoader`].

use opensquilla_core::error::AppError;
use opensquilla_skills::loader::SkillLoader;
use opensquilla_skills::types::{SkillLayer, SkillSpec};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A shared skills service wrapping a [`SkillLoader`] plus a disable list.
#[derive(Clone)]
pub struct SkillsService {
    loader: Arc<SkillLoader>,
    disabled: Arc<Mutex<std::collections::HashSet<String>>>,
}

impl SkillsService {
    /// Create a new skills service with an empty loader.
    pub fn new() -> Self {
        Self {
            loader: Arc::new(SkillLoader::new()),
            disabled: Arc::new(Mutex::new(std::collections::HashSet::new())),
        }
    }

    /// Register a directory to scan for a given layer.
    pub fn register_layer_dir(&self, layer: SkillLayer, dir: PathBuf) {
        self.loader.register_layer_dir(layer, dir);
    }

    /// Scan all registered directories.
    pub async fn scan(&self) -> Result<usize, String> {
        self.loader.scan_all().await
    }

    /// Register skill specs directly (used by tests and programmatic installs).
    pub async fn register_skills(&self, specs: Vec<SkillSpec>) -> usize {
        self.loader.register_skills(specs).await
    }

    /// Enable a previously disabled skill.
    pub fn enable(&self, skill_id: &str) -> bool {
        self.disabled.lock().remove(skill_id)
    }

    /// Disable a skill by id.
    pub fn disable(&self, skill_id: &str) -> bool {
        self.disabled.lock().insert(skill_id.to_string())
    }

    /// Whether a skill is disabled.
    pub fn is_disabled(&self, skill_id: &str) -> bool {
        self.disabled.lock().contains(skill_id)
    }
}

impl Default for SkillsService {
    fn default() -> Self {
        Self::new()
    }
}

/// View of a skill for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillView {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub description: String,
    pub layer: String,
    pub version: Option<String>,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub source_path: Option<String>,
    pub is_meta: bool,
    pub disabled: bool,
}

fn to_view(spec: &SkillSpec, disabled: bool) -> SkillView {
    SkillView {
        id: spec.id.clone(),
        name: spec.name.clone(),
        kind: format!("{:?}", spec.kind).to_lowercase(),
        description: spec.description.clone(),
        layer: spec.layer.to_string(),
        version: spec.version.clone(),
        author: spec.author.clone(),
        tags: spec.tags.clone(),
        source_path: spec.source_path.clone(),
        is_meta: spec.is_meta(),
        disabled,
    }
}

/// Resolve a [`SkillLayer`] from a string parameter.
fn parse_layer(s: &str) -> Result<SkillLayer, AppError> {
    match s.to_ascii_uppercase().as_str() {
        "EXTRA" => Ok(SkillLayer::Extra),
        "BUNDLED" => Ok(SkillLayer::Bundled),
        "MANAGED" => Ok(SkillLayer::Managed),
        "PERSONAL" => Ok(SkillLayer::Personal),
        "PROJECT" => Ok(SkillLayer::Project),
        "WORKSPACE" => Ok(SkillLayer::Workspace),
        other => Err(AppError::bad_request(format!(
            "Unknown skill layer '{other}'"
        ))),
    }
}

/// Register skills RPC handlers on the given registry.
pub fn register_skills_handlers(registry: &mut RpcRegistry, service: SkillsService) {
    let service = Arc::new(service);

    // skills.register_dir — register a directory to scan for a layer
    registry.register(rpc_handler("skills.register_dir", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let layer_str = params
                    .get("layer")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'layer' parameter"))?;
                let layer = parse_layer(layer_str)?;
                let dir = params
                    .get("dir")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'dir' parameter"))?;
                let path = PathBuf::from(dir);
                if !path.exists() {
                    return Err(AppError::bad_request(format!(
                        "Directory does not exist: {dir}"
                    )));
                }
                service.register_layer_dir(layer, path);
                Ok(serde_json::json!({
                    "registered": true,
                    "layer": layer_str.to_uppercase(),
                    "dir": dir,
                }))
            }
        }
    }));

    // skills.scan — scan all registered directories for SKILL.md files
    registry.register(rpc_handler("skills.scan", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let count = service
                    .scan()
                    .await
                    .map_err(|e| AppError::internal(format!("Scan failed: {e}")))?;
                Ok(serde_json::json!({"scanned": count}))
            }
        }
    }));

    // skills.list — list all skills, optionally filtered by layer
    registry.register(rpc_handler("skills.list", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let layer_filter = params
                    .get("layer")
                    .and_then(|v| v.as_str())
                    .map(parse_layer)
                    .transpose()?;
                let skills = service.loader.get_skills(layer_filter).await;
                let views: Vec<SkillView> = skills
                    .iter()
                    .map(|s| to_view(s, service.is_disabled(&s.id)))
                    .collect();
                Ok(serde_json::json!({
                    "skills": views,
                    "count": views.len(),
                }))
            }
        }
    }));

    // skills.get — fetch a single skill by id
    registry.register(rpc_handler("skills.get", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                match service.loader.get_skill(id).await {
                    Some(spec) => Ok(
                        serde_json::to_value(to_view(&spec, service.is_disabled(id)))
                            .map_err(|e| AppError::internal(e.to_string()))?,
                    ),
                    None => Err(AppError::not_found(format!("Skill '{id}' not found"))),
                }
            }
        }
    }));

    // skills.meta — list only meta-skills
    registry.register(rpc_handler("skills.meta", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let skills = service.loader.get_meta_skills().await;
                let views: Vec<SkillView> = skills
                    .iter()
                    .map(|s| to_view(s, service.is_disabled(&s.id)))
                    .collect();
                Ok(serde_json::json!({
                    "meta_skills": views,
                    "count": views.len(),
                }))
            }
        }
    }));

    // skills.enable — re-enable a disabled skill
    registry.register(rpc_handler("skills.enable", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let was_disabled = service.enable(id);
                Ok(serde_json::json!({
                    "id": id,
                    "enabled": true,
                    "was_disabled": was_disabled,
                }))
            }
        }
    }));

    // skills.disable — disable a skill so it is excluded from injection
    registry.register(rpc_handler("skills.disable", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let newly = service.disable(id);
                Ok(serde_json::json!({
                    "id": id,
                    "disabled": true,
                    "newly_disabled": newly,
                }))
            }
        }
    }));

    // skills.count — total number of loaded skills
    registry.register(rpc_handler("skills.count", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let count = service.loader.count().await;
                let disabled = service.disabled.lock().len();
                Ok(serde_json::json!({
                    "total": count,
                    "disabled": disabled,
                    "enabled": count.saturating_sub(disabled),
                }))
            }
        }
    }));

    // skills.search — search skill metadata (id, name, description, tags)
    registry.register(rpc_handler("skills.search", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let layer_filter = params
                    .get("layer")
                    .and_then(|v| v.as_str())
                    .map(parse_layer)
                    .transpose()?;
                let skills = service.loader.get_skills(layer_filter).await;
                let views: Vec<SkillView> = skills
                    .iter()
                    .filter(|s| {
                        query.is_empty()
                            || s.id.to_lowercase().contains(&query)
                            || s.name.to_lowercase().contains(&query)
                            || s.description.to_lowercase().contains(&query)
                            || s.tags.iter().any(|t| t.to_lowercase().contains(&query))
                    })
                    .map(|s| to_view(s, service.is_disabled(&s.id)))
                    .collect();
                Ok(serde_json::json!({
                    "skills": views,
                    "count": views.len(),
                }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_skills_disable_enable() {
        let service = SkillsService::new();
        let mut registry = RpcRegistry::new();
        register_skills_handlers(&mut registry, service);

        let r = registry
            .dispatch("skills.disable", serde_json::json!({"id": "my-skill"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["disabled"], true);
        assert_eq!(resp["newly_disabled"], true);

        let r = registry
            .dispatch("skills.enable", serde_json::json!({"id": "my-skill"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["enabled"], true);
        assert_eq!(resp["was_disabled"], true);
    }

    #[tokio::test]
    async fn test_skills_list_empty() {
        let service = SkillsService::new();
        let mut registry = RpcRegistry::new();
        register_skills_handlers(&mut registry, service);

        let r = registry
            .dispatch("skills.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 0);
    }

    #[tokio::test]
    async fn test_skills_register_dir_missing() {
        let service = SkillsService::new();
        let mut registry = RpcRegistry::new();
        register_skills_handlers(&mut registry, service);

        let r = registry
            .dispatch(
                "skills.register_dir",
                serde_json::json!({"layer": "EXTRA", "dir": "/nonexistent/path/xyz"}),
            )
            .await;
        assert!(r.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_skills_search() {
        let service = SkillsService::new();
        service
            .register_skills(vec![
                SkillSpec::new(
                    "test-skill".to_string(),
                    "Test Skill".to_string(),
                    "Does testing things".to_string(),
                    SkillLayer::Bundled,
                ),
                SkillSpec::new(
                    "other-skill".to_string(),
                    "Other".to_string(),
                    "Unrelated skill".to_string(),
                    SkillLayer::Bundled,
                ),
            ])
            .await;
        let mut registry = RpcRegistry::new();
        register_skills_handlers(&mut registry, service);

        let r = registry
            .dispatch("skills.search", serde_json::json!({"query": "testing"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
        assert_eq!(resp["skills"][0]["id"], "test-skill");

        // No matches.
        let r = registry
            .dispatch("skills.search", serde_json::json!({"query": "nonexistent"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 0);
    }
}
