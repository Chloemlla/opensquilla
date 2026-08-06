//! Tools directory RPC handlers.
//!
//! Provides `rpc_tools` for listing the tool directory, searching tools, and
//! checking search provider status. Backed by the tools crate's
//! [`ToolRegistry`] and the search crate's [`SearchRegistry`].

use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use opensquilla_core::types::ToolCall;
use opensquilla_search::registry::SearchRegistry;
use opensquilla_search::types::{SearchOptions, SearchRequest};
use opensquilla_tools::dispatch::{DispatchContext, DispatchEngine};
use opensquilla_tools::registry::{ToolDefinition, ToolRegistry};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A tools service combining the tool registry and search registry.
#[derive(Clone)]
pub struct ToolsService {
    tools: Arc<RwLock<ToolRegistry>>,
    search: Arc<RwLock<Option<Arc<SearchRegistry>>>>,
}

impl ToolsService {
    /// Create a new tools service with an empty tool registry and no search
    /// registry.
    pub fn new() -> Self {
        Self {
            tools: Arc::new(RwLock::new(ToolRegistry::new())),
            search: Arc::new(RwLock::new(None)),
        }
    }

    /// Attach a search registry built from the given config.
    pub fn with_search(config: &Config) -> Self {
        let search = SearchRegistry::new(config);
        Self {
            tools: Arc::new(RwLock::new(ToolRegistry::new())),
            search: Arc::new(RwLock::new(Some(Arc::new(search)))),
        }
    }

    /// Access the tool registry for registration.
    pub fn tools(&self) -> &Arc<RwLock<ToolRegistry>> {
        &self.tools
    }
}

impl Default for ToolsService {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable tool view for API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolView {
    pub name: String,
    pub description: String,
    pub category: Option<String>,
    pub requires_confirmation: bool,
    pub risk_level: u8,
    pub parameters: serde_json::Value,
}

impl From<&ToolDefinition> for ToolView {
    fn from(def: &ToolDefinition) -> Self {
        Self {
            name: def.name.clone(),
            description: def.description.clone(),
            category: def.category.clone(),
            requires_confirmation: def.requires_confirmation,
            risk_level: def.risk_level,
            parameters: def.to_json_schema(),
        }
    }
}

/// Register tools RPC handlers on the given registry.
pub fn register_tools_handlers(registry: &mut RpcRegistry, service: ToolsService) {
    let service = Arc::new(service);

    // tools.list — list all registered tools
    registry.register(rpc_handler("tools.list", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let category = params.get("category").and_then(|v| v.as_str());
                let tools = service.tools.read();
                let views: Vec<ToolView> = if let Some(cat) = category {
                    tools
                        .get_category(cat)
                        .iter()
                        .map(|t| ToolView::from(t.definition()))
                        .collect()
                } else {
                    tools
                        .iter()
                        .map(|(_, t)| ToolView::from(t.definition()))
                        .collect()
                };
                Ok(serde_json::json!({
                    "tools": views,
                    "count": views.len(),
                }))
            }
        }
    }));

    // tools.get — fetch a tool definition by name
    registry.register(rpc_handler("tools.get", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?;
                let tools = service.tools.read();
                match tools.get(name) {
                    Some(tool) => Ok(serde_json::to_value(ToolView::from(tool.definition()))
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Tool '{name}' not found"))),
                }
            }
        }
    }));

    // tools.categories — list tool categories and counts
    registry.register(rpc_handler("tools.categories", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let tools = service.tools.read();
                let by_cat = tools.by_category();
                let categories: Vec<serde_json::Value> = by_cat
                    .into_iter()
                    .map(|(cat, list)| {
                        serde_json::json!({
                            "category": cat,
                            "count": list.len(),
                        })
                    })
                    .collect();
                Ok(serde_json::json!({"categories": categories}))
            }
        }
    }));

    // tools.search — run a web search via the search registry
    registry.register(rpc_handler("tools.search", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'query' parameter"))?;
                let max_results = params
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as usize;
                let provider = params.get("provider").and_then(|v| v.as_str());

                // Clone the Arc out so the parking_lot read guard is dropped
                // before the `.await` below (the guard is `!Send`).
                let search = {
                    let search_guard = service.search.read();
                    search_guard
                        .as_ref()
                        .cloned()
                        .ok_or_else(|| AppError::bad_request("Search registry not configured"))?
                };

                let request = SearchRequest {
                    query: query.to_string(),
                    options: SearchOptions {
                        max_results,
                        ..Default::default()
                    },
                };

                let response = if let Some(name) = provider {
                    search.search(name, &request).await
                } else {
                    search.search_default(&request).await
                };

                let response = response.map_err(|e| AppError::internal(e.to_string()))?;
                Ok(
                    serde_json::to_value(response)
                        .map_err(|e| AppError::internal(e.to_string()))?,
                )
            }
        }
    }));

    // tools.providers — list search provider status
    registry.register(rpc_handler("tools.providers", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let search_guard = service.search.read();
                match search_guard.as_ref() {
                    Some(search) => {
                        let all = search.list_providers();
                        let ready = search.list_ready_providers();
                        let default = search.default_provider().to_string();
                        let providers: Vec<serde_json::Value> = all
                            .into_iter()
                            .map(|name| {
                                let is_ready = ready.contains(&name);
                                serde_json::json!({
                                    "name": name,
                                    "ready": is_ready,
                                    "is_default": name == default,
                                })
                            })
                            .collect();
                        Ok(serde_json::json!({
                            "providers": providers,
                            "count": providers.len(),
                            "default": default,
                        }))
                    }
                    None => Ok(serde_json::json!({
                        "providers": [],
                        "count": 0,
                        "configured": false,
                    })),
                }
            }
        }
    }));

    // tools.count — total number of registered tools
    registry.register(rpc_handler("tools.count", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let count = service.tools.read().len();
                Ok(serde_json::json!({"count": count}))
            }
        }
    }));

    // tools.search_tools — search the tool directory by name/description
    registry.register(rpc_handler("tools.search_tools", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let query = params
                    .get("query")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_lowercase();
                let category = params.get("category").and_then(|v| v.as_str());
                let tools = service.tools.read();
                let mut views: Vec<ToolView> = if let Some(cat) = category {
                    tools
                        .get_category(cat)
                        .iter()
                        .map(|t| ToolView::from(t.definition()))
                        .collect()
                } else {
                    tools
                        .iter()
                        .map(|(_, t)| ToolView::from(t.definition()))
                        .collect()
                };
                if !query.is_empty() {
                    views.retain(|v| {
                        v.name.to_lowercase().contains(&query)
                            || v.description.to_lowercase().contains(&query)
                            || v.category
                                .as_deref()
                                .unwrap_or("")
                                .to_lowercase()
                                .contains(&query)
                    });
                }
                Ok(serde_json::json!({
                    "tools": views,
                    "count": views.len(),
                }))
            }
        }
    }));

    // tools.execute — execute an arbitrary tool call through the dispatch engine
    registry.register(rpc_handler("tools.execute", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let tool_name = params
                    .get("tool")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'tool' parameter"))?;
                let arguments = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let timeout_secs = params
                    .get("timeout_secs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(60);
                let sandbox = params
                    .get("sandbox")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // Build a snapshot registry sharing the same tool handles, then
                // run the call through the dispatch engine (injection guard +
                // policy chain + timeout).
                let snapshot = {
                    let mut tools = service.tools.write();
                    let mut registry = ToolRegistry::new();
                    let names: Vec<String> = tools.tool_names();
                    for name in &names {
                        if let Some(tool) = tools.get(name) {
                            let _ = registry.register_arc(tool);
                        }
                    }
                    registry
                };
                let engine = DispatchEngine::new_with_defaults(Arc::new(snapshot));
                let call = ToolCall::new(Uuid::new_v4().to_string(), tool_name, arguments);
                let ctx = DispatchContext::new(session_id)
                    .with_timeout(timeout_secs)
                    .with_sandbox(sandbox);
                let output = engine.dispatch(call, &ctx).await?;
                Ok(serde_json::to_value(output).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_tools::registry::{ParameterDefinition, Tool, ToolOutput, ToolResult};

    /// A minimal echo tool used to exercise the directory and dispatch handlers.
    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> &ToolDefinition {
            static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
                ToolDefinition::new(
                    "echo",
                    "Echo back text",
                    std::collections::HashMap::from([(
                        "text".to_string(),
                        ParameterDefinition::required_string("Text to echo"),
                    )]),
                )
            });
            &DEF
        }

        async fn execute(&self, args: serde_json::Value) -> ToolResult {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Ok(ToolOutput::success(text))
        }
    }

    #[tokio::test]
    async fn test_tools_list_empty() {
        let service = ToolsService::new();
        let mut registry = RpcRegistry::new();
        register_tools_handlers(&mut registry, service);

        let r = registry
            .dispatch("tools.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 0);
    }

    #[tokio::test]
    async fn test_tools_providers_not_configured() {
        let service = ToolsService::new();
        let mut registry = RpcRegistry::new();
        register_tools_handlers(&mut registry, service);

        let r = registry
            .dispatch("tools.providers", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["configured"], false);
    }

    #[tokio::test]
    async fn test_tools_count() {
        let service = ToolsService::new();
        let mut registry = RpcRegistry::new();
        register_tools_handlers(&mut registry, service);

        let r = registry
            .dispatch("tools.count", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 0);
    }

    #[tokio::test]
    async fn test_tools_execute_and_search() {
        let service = ToolsService::new();
        service.tools().write().register(EchoTool).unwrap();
        let mut registry = RpcRegistry::new();
        register_tools_handlers(&mut registry, service.clone());

        // Directory search finds the tool by name.
        let r = registry
            .dispatch("tools.search_tools", serde_json::json!({"query": "echo"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        // Execution runs the tool and returns its output.
        let r = registry
            .dispatch(
                "tools.execute",
                serde_json::json!({"tool": "echo", "arguments": {"text": "hello"}}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["content"], "hello");
        assert_eq!(resp["is_error"], false);

        // Unknown tool is an error.
        let r = registry
            .dispatch(
                "tools.execute",
                serde_json::json!({"tool": "missing-tool", "arguments": {}}),
            )
            .await;
        assert!(r.unwrap().is_err());
    }
}
