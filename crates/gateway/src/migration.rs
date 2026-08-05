//! Migration RPC handlers.
//!
//! Provides `rpc_migration` for configuration discovery and preview: locating
//! existing config files on disk and previewing their contents without
//! applying them.

use std::path::PathBuf;
use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};

use crate::rpc::{rpc_handler, RpcRegistry};

/// A discovered configuration file candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredConfig {
    pub path: String,
    pub format: String,
    pub exists: bool,
    pub size_bytes: u64,
}

/// A preview of a configuration file's parsed contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigPreview {
    pub path: String,
    pub entries: std::collections::HashMap<String, String>,
    pub entry_count: usize,
    pub valid: bool,
    pub error: Option<String>,
}

/// Discover configuration candidates from common locations.
fn discover_candidates() -> Vec<DiscoveredConfig> {
    let mut candidates = Vec::new();

    // OPENSQUILLA_CONFIG env var
    if let Ok(path) = std::env::var("OPENSQUILLA_CONFIG") {
        let p = PathBuf::from(&path);
        candidates.push(DiscoveredConfig {
            path,
            format: format_from_path(&p),
            exists: p.exists(),
            size_bytes: file_size(&p),
        });
    }

    // Current directory
    let local = PathBuf::from("opensquilla.toml");
    candidates.push(DiscoveredConfig {
        path: local.display().to_string(),
        format: format_from_path(&local),
        exists: local.exists(),
        size_bytes: file_size(&local),
    });

    // Platform config directory
    if let Some(config_dir) = dirs::config_dir() {
        let p = config_dir.join("opensquilla").join("opensquilla.toml");
        candidates.push(DiscoveredConfig {
            path: p.display().to_string(),
            format: format_from_path(&p),
            exists: p.exists(),
            size_bytes: file_size(&p),
        });
    }

    // Home directory dotfile
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".opensquilla").join("opensquilla.toml");
        candidates.push(DiscoveredConfig {
            path: p.display().to_string(),
            format: format_from_path(&p),
            exists: p.exists(),
            size_bytes: file_size(&p),
        });
    }

    candidates
}

fn format_from_path(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => "toml".to_string(),
        Some("yaml" | "yml") => "yaml".to_string(),
        Some("json") => "json".to_string(),
        _ => "toml".to_string(),
    }
}

fn file_size(path: &PathBuf) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Register migration RPC handlers on the given registry.
pub fn register_migration_handlers(registry: &mut RpcRegistry) {
    // migration.discover — discover config file candidates
    registry.register(rpc_handler("migration.discover", {
        move |_params| {
            let candidates = discover_candidates();
            Ok(serde_json::json!({
                "candidates": candidates,
                "count": candidates.len(),
            }))
        }
    }));

    // migration.preview — parse and preview a config file without applying it
    registry.register(rpc_handler("migration.preview", {
        move |params| {
            let path_str = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::bad_request("Missing 'path' parameter"))?;
            let path = PathBuf::from(path_str);
            if !path.exists() {
                return Err(AppError::not_found(format!(
                    "Config file '{path_str}' does not exist"
                )));
            }

            let format = format_from_path(&path);
            let parse_result = match format.as_str() {
                "yaml" | "yml" => Config::from_yaml_file(&path),
                _ => Config::from_file(&path),
            };

            let preview = match parse_result {
                Ok(config) => {
                    let entries = config.list();
                    let entry_count = entries.len();
                    ConfigPreview {
                        path: path_str.to_string(),
                        entries,
                        entry_count,
                        valid: true,
                        error: None,
                    }
                }
                Err(e) => ConfigPreview {
                    path: path_str.to_string(),
                    entries: std::collections::HashMap::new(),
                    entry_count: 0,
                    valid: false,
                    error: Some(e.to_string()),
                },
            };

            Ok(serde_json::to_value(preview)
                .map_err(|e| AppError::internal(e.to_string()))?)
        }
    }));

    // migration.discover_path — resolve the active config path per discovery rules
    registry.register(rpc_handler("migration.discover_path", {
        move |_params| {
            match Config::discover_path() {
                Ok(path) => Ok(serde_json::json!({
                    "path": path.display().to_string(),
                    "found": true,
                })),
                Err(e) => Ok(serde_json::json!({
                    "path": null,
                    "found": false,
                    "error": e.to_string(),
                })),
            }
        }
    }));

    // migration.validate — validate a config file's structure
    registry.register(rpc_handler("migration.validate", {
        move |params| {
            let path_str = params
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::bad_request("Missing 'path' parameter"))?;
            let path = PathBuf::from(path_str);
            if !path.exists() {
                return Err(AppError::not_found(format!(
                    "Config file '{path_str}' does not exist"
                )));
            }

            let format = format_from_path(&path);
            let result = match format.as_str() {
                "yaml" | "yml" => Config::from_yaml_file(&path),
                _ => Config::from_file(&path),
            };

            match result {
                Ok(config) => {
                    let providers = config.providers.len();
                    let channels = config.channels.len();
                    Ok(serde_json::json!({
                        "valid": true,
                        "path": path_str,
                        "providers": providers,
                        "channels": channels,
                    }))
                }
                Err(e) => Ok(serde_json::json!({
                    "valid": false,
                    "path": path_str,
                    "error": e.to_string(),
                })),
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_migration_discover() {
        let mut registry = RpcRegistry::new();
        register_migration_handlers(&mut registry);

        let r = registry.dispatch("migration.discover", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert!(resp["candidates"].is_array());
        assert!(resp["count"].as_u64().is_some());
    }

    #[tokio::test]
    async fn test_migration_preview_missing() {
        let mut registry = RpcRegistry::new();
        register_migration_handlers(&mut registry);

        let r = registry
            .dispatch(
                "migration.preview",
                serde_json::json!({"path": "/nonexistent/xyz.toml"}),
            )
            .await;
        assert!(r.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_migration_discover_path() {
        let mut registry = RpcRegistry::new();
        register_migration_handlers(&mut registry);

        let r = registry
            .dispatch("migration.discover_path", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["found"].is_boolean());
    }
}
