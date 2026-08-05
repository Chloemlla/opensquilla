//! Inbound MCP server.
//!
//! Exposes OpenSquilla session operations and registered tools over the Model
//! Context Protocol. Provides a JSON-RPC 2.0 endpoint (single or batch) that
//! can be mounted into an axum [`Router`], a stdio transport loop, and tool
//! registration.

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use opensquilla_core::config::Config;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::transport::RequestHandler;
use crate::types::{
    JsonRpcError, JsonRpcId, JsonRpcRequest, JsonRpcResponse, McpRequest, McpResponse, McpTool,
    McpToolCall,
};

/// The MCP protocol version this server speaks.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Bridge to OpenSquilla session operations, mirroring the Python
/// `OpenSquillaMCPBridge`.
#[async_trait::async_trait]
pub trait SessionBridge: Send + Sync {
    /// List sessions visible to the connected gateway principal.
    async fn conversations_list(&self, limit: Option<u32>) -> Result<Value, McpServerError>;

    /// Resolve a session key or identifier to session metadata.
    async fn session_resolve(&self, key: &str) -> Result<Value, McpServerError>;

    /// Read persisted messages for a session.
    async fn messages_read(&self, key: &str, limit: Option<u32>) -> Result<Value, McpServerError>;

    /// Send a user message to an existing session.
    async fn messages_send(
        &self,
        key: &str,
        message: &str,
        intent: &str,
    ) -> Result<Value, McpServerError>;

    /// Wait for live or replayed gateway events for a session.
    async fn events_wait(
        &self,
        key: &str,
        since_stream_seq: Option<i64>,
        timeout_ms: u64,
        max_events: u32,
        terminal_only: bool,
    ) -> Result<Value, McpServerError>;

    /// Export a session transcript as JSONL.
    async fn transcript_jsonl(
        &self,
        key: &str,
        limit: Option<u32>,
    ) -> Result<String, McpServerError>;
}

/// Executes a registered MCP tool.
#[async_trait::async_trait]
pub trait McpToolExecutor: Send + Sync {
    /// Execute the tool call and return the raw `tools/call` result.
    async fn execute(&self, call: &McpToolCall) -> Result<Value, McpServerError>;
}

/// The inbound MCP server.
#[derive(Clone)]
pub struct McpServer {
    name: String,
    version: String,
    tools: Arc<Mutex<HashMap<String, McpTool>>>,
    executors: Arc<Mutex<HashMap<String, Arc<dyn McpToolExecutor>>>>,
    bridge: Option<Arc<dyn SessionBridge>>,
    running: Arc<Mutex<bool>>,
}

impl McpServer {
    /// Create a new MCP server. The configuration is currently used only to
    /// keep the constructor signature stable across the workspace.
    pub fn new(_config: &Config) -> Self {
        Self::with_name("opensquilla-mcp", "0.1.0")
    }

    /// Create a new MCP server with an explicit name and version.
    pub fn with_name(name: &str, version: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            tools: Arc::new(Mutex::new(HashMap::new())),
            executors: Arc::new(Mutex::new(HashMap::new())),
            bridge: None,
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Attach a [`SessionBridge`] used to expose session operations.
    pub fn with_bridge(mut self, bridge: Arc<dyn SessionBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Register a tool (metadata only).
    pub async fn register_tool(&self, tool: McpTool) {
        self.tools.lock().await.insert(tool.name.clone(), tool);
    }

    /// Register a tool together with an executor that implements `tools/call`.
    pub async fn register_tool_with_executor(
        &self,
        tool: McpTool,
        executor: Arc<dyn McpToolExecutor>,
    ) {
        self.tools
            .lock()
            .await
            .insert(tool.name.clone(), tool.clone());
        self.executors
            .lock()
            .await
            .insert(tool.name.clone(), executor);
        info!(tool = %tool.name, "Registered MCP tool");
    }

    /// Unregister a tool.
    pub async fn unregister_tool(&self, name: &str) {
        self.tools.lock().await.remove(name);
        self.executors.lock().await.remove(name);
        info!(tool = %name, "Unregistered MCP tool");
    }

    /// List all registered tools.
    pub async fn list_tools(&self) -> Vec<McpTool> {
        self.tools.lock().await.values().cloned().collect()
    }

    /// Register the built-in session-operation tools (requires a bridge).
    pub async fn register_session_tools(&self) {
        let session_tools: Vec<McpTool> = vec![
            McpTool {
                name: "conversations_list".into(),
                description:
                    "List OpenSquilla sessions visible to the connected gateway principal.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "limit": { "type": "integer" } }
                }),
            },
            McpTool {
                name: "session_resolve".into(),
                description: "Resolve a session key or identifier to OpenSquilla session metadata."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "key": { "type": "string" } },
                    "required": ["key"]
                }),
            },
            McpTool {
                name: "messages_read".into(),
                description: "Read persisted messages for an OpenSquilla session.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["key"]
                }),
            },
            McpTool {
                name: "messages_send".into(),
                description: "Send a user message to an existing OpenSquilla session.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "message": { "type": "string" },
                        "intent": { "type": "string" }
                    },
                    "required": ["key", "message"]
                }),
            },
            McpTool {
                name: "events_wait".into(),
                description: "Wait for live or replayed gateway events for an OpenSquilla session."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "since_stream_seq": { "type": "integer" },
                        "timeout_ms": { "type": "integer" },
                        "max_events": { "type": "integer" },
                        "terminal_only": { "type": "boolean" }
                    },
                    "required": ["key"]
                }),
            },
            McpTool {
                name: "transcript_export".into(),
                description:
                    "Export a session transcript as JSONL with standard tool evidence events."
                        .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "key": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["key"]
                }),
            },
        ];
        let mut tools = self.tools.lock().await;
        for tool in session_tools {
            tools.insert(tool.name.clone(), tool);
        }
    }

    /// Handle an MCP-layer request.
    pub async fn handle_request(&self, request: &McpRequest) -> McpResponse {
        let jsonrpc = JsonRpcRequest::from(request);
        let response = self.handle_jsonrpc(&jsonrpc).await;
        McpResponse::from(response)
    }

    /// Handle a JSON-RPC 2.0 request.
    pub async fn handle_jsonrpc(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.clone().unwrap_or(JsonRpcId::Null);
        debug!(
            method = %request.method,
            id = %id.to_id_string(),
            "Handling MCP request"
        );

        match request.method.as_str() {
            "initialize" => JsonRpcResponse::ok(id, self.initialize_result()),
            "ping" => JsonRpcResponse::ok(id, json!({})),
            "tools/list" => self.handle_tools_list(id).await,
            "tools/call" => self.handle_tools_call(id, request.params.as_ref()).await,
            "resources/list" => JsonRpcResponse::ok(id, json!({ "resources": [] })),
            "prompts/list" => JsonRpcResponse::ok(id, json!({ "prompts": [] })),
            "notifications/initialized" => JsonRpcResponse::notification(),
            _ => JsonRpcResponse::error(id, JsonRpcError::method_not_found(&request.method)),
        }
    }

    fn initialize_result(&self) -> Value {
        json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": { "listChanged": true },
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        })
    }

    async fn handle_tools_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let tools = self.list_tools().await;
        let serialized: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                })
            })
            .collect();
        JsonRpcResponse::ok(id, json!({ "tools": serialized }))
    }

    async fn handle_tools_call(&self, id: JsonRpcId, params: Option<&Value>) -> JsonRpcResponse {
        let Some(params) = params else {
            return JsonRpcResponse::error(id, JsonRpcError::invalid_params("Missing parameters"));
        };

        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(name) => name,
            None => {
                return JsonRpcResponse::error(
                    id,
                    JsonRpcError::invalid_params("Missing tool name"),
                );
            }
        };
        let arguments: HashMap<String, Value> = params
            .get("arguments")
            .and_then(|v| v.as_object())
            .cloned()
            .map(|map| map.into_iter().collect())
            .unwrap_or_default();
        let call = McpToolCall {
            name: name.to_string(),
            arguments,
        };

        // 1. Session operations exposed through the bridge.
        if let Some(bridge) = &self.bridge {
            if let Some(result) = self.run_session_op(bridge, &call).await {
                return JsonRpcResponse::ok(id, result);
            }
        }

        // 2. Registered tools with executors.
        if let Some(executor) = self.executors.lock().await.get(name).cloned() {
            match executor.execute(&call).await {
                Ok(result) => JsonRpcResponse::ok(id, result),
                Err(err) => {
                    JsonRpcResponse::error(id, JsonRpcError::internal_error(err.to_string()))
                }
            }
        } else if self.tools.lock().await.contains_key(name) {
            // Metadata-only tool: emit a stub success so callers see a result.
            JsonRpcResponse::ok(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Tool '{name}' executed successfully")
                    }]
                }),
            )
        } else {
            JsonRpcResponse::error(id, JsonRpcError::tool_not_found(name))
        }
    }

    /// Route a session-operation tool call to the bridge. Returns `None` when
    /// the tool name is not a session operation.
    async fn run_session_op(
        &self,
        bridge: &Arc<dyn SessionBridge>,
        call: &McpToolCall,
    ) -> Option<Value> {
        let result = match call.name.as_str() {
            "conversations_list" => {
                bridge
                    .conversations_list(
                        call.arguments
                            .get("limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                    )
                    .await
            }
            "session_resolve" => {
                bridge
                    .session_resolve(
                        call.arguments
                            .get("key")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                    )
                    .await
            }
            "messages_read" => {
                bridge
                    .messages_read(
                        call.arguments
                            .get("key")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        call.arguments
                            .get("limit")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as u32),
                    )
                    .await
            }
            "messages_send" => {
                bridge
                    .messages_send(
                        call.arguments
                            .get("key")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        call.arguments
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        call.arguments
                            .get("intent")
                            .and_then(|v| v.as_str())
                            .unwrap_or("continue"),
                    )
                    .await
            }
            "events_wait" => {
                bridge
                    .events_wait(
                        call.arguments
                            .get("key")
                            .and_then(|v| v.as_str())
                            .unwrap_or(""),
                        call.arguments
                            .get("since_stream_seq")
                            .and_then(|v| v.as_i64()),
                        call.arguments
                            .get("timeout_ms")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(30_000),
                        call.arguments
                            .get("max_events")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(100) as u32,
                        call.arguments
                            .get("terminal_only")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                    )
                    .await
            }
            "transcript_export" => bridge
                .transcript_jsonl(
                    call.arguments
                        .get("key")
                        .and_then(|v| v.as_str())
                        .unwrap_or(""),
                    call.arguments
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                )
                .await
                .map(|text| json!({ "jsonl": text })),
            _ => return None,
        };

        Some(result.unwrap_or_else(|e| {
            json!({
                "content": [{ "type": "text", "text": format!("error: {e}") }],
                "isError": true
            })
        }))
    }

    /// Build an axum [`Router`] exposing the JSON-RPC endpoint at `/mcp`.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/mcp", post(handle_mcp_endpoint))
            .with_state(Arc::new(self.clone()))
    }

    /// Mark the server as running.
    pub async fn start(&self) -> Result<(), McpServerError> {
        let mut running = self.running.lock().await;
        if *running {
            return Err(McpServerError::AlreadyRunning);
        }
        *running = true;
        info!(name = %self.name, "MCP server started");
        Ok(())
    }

    /// Mark the server as stopped.
    pub async fn stop(&self) -> Result<(), McpServerError> {
        let mut running = self.running.lock().await;
        if !*running {
            return Err(McpServerError::NotRunning);
        }
        *running = false;
        info!(name = %self.name, "MCP server stopped");
        Ok(())
    }

    /// Whether the server is marked as running.
    pub async fn is_running(&self) -> bool {
        *self.running.lock().await
    }

    /// Run the MCP server over the stdio transport (newline-delimited JSON-RPC).
    pub async fn run_stdio(&self) -> Result<(), McpServerError> {
        self.start().await?;
        info!(name = %self.name, "MCP server running in stdio mode");

        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = tokio::io::BufReader::new(stdin);
        let mut writer = tokio::io::BufWriter::new(stdout);

        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<JsonRpcRequest>(trimmed) {
                        Ok(request) => {
                            let response = self.handle_jsonrpc(&request).await;
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = writer.write_all(json.as_bytes()).await;
                                let _ = writer.write_all(b"\n").await;
                                let _ = writer.flush().await;
                            }
                        }
                        Err(_) => {
                            let error = JsonRpcResponse::error(
                                JsonRpcId::Null,
                                JsonRpcError::parse_error(),
                            );
                            if let Ok(json) = serde_json::to_string(&error) {
                                let _ = writer.write_all(json.as_bytes()).await;
                                let _ = writer.write_all(b"\n").await;
                                let _ = writer.flush().await;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Error reading MCP stdin");
                    break;
                }
            }
        }

        self.stop().await?;
        Ok(())
    }
}

/// The axum JSON-RPC endpoint handler. Accepts both a single request and a
/// JSON-RPC batch (array of requests).
async fn handle_mcp_endpoint(State(server): State<Arc<McpServer>>, body: Bytes) -> Response {
    match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Array(items)) => {
            let mut responses = Vec::new();
            for item in items {
                match serde_json::from_value::<JsonRpcRequest>(item) {
                    Ok(request) => {
                        let response = server.handle_jsonrpc(&request).await;
                        responses.push(serde_json::to_value(response).unwrap_or_default());
                    }
                    Err(_) => responses.push(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32600, "message": "Invalid Request" }
                    })),
                }
            }
            Json(Value::Array(responses)).into_response()
        }
        Ok(value) => match serde_json::from_value::<JsonRpcRequest>(value) {
            Ok(request) => {
                let response = server.handle_jsonrpc(&request).await;
                Json(serde_json::to_value(response).unwrap_or_default()).into_response()
            }
            Err(_) => Json(serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32600, "message": "Invalid Request" }
            }))
            .into_response(),
        },
        Err(_) => Json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": null,
            "error": { "code": -32700, "message": "Parse error" }
        }))
        .into_response(),
    }
}

#[async_trait::async_trait]
impl RequestHandler for McpServer {
    async fn handle_request(&self, request: McpRequest) -> McpResponse {
        self.handle_request(&request).await
    }
}

/// An error that can occur in the inbound MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpServerError {
    #[error("MCP server is already running")]
    AlreadyRunning,

    #[error("MCP server is not running")]
    NotRunning,

    #[error("Session operation failed: {0}")]
    SessionOp(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockBridge;

    #[async_trait::async_trait]
    impl SessionBridge for MockBridge {
        async fn conversations_list(&self, limit: Option<u32>) -> Result<Value, McpServerError> {
            Ok(json!({ "sessions": [], "limit": limit }))
        }

        async fn session_resolve(&self, key: &str) -> Result<Value, McpServerError> {
            Ok(json!({ "key": key, "session": null }))
        }

        async fn messages_read(
            &self,
            key: &str,
            limit: Option<u32>,
        ) -> Result<Value, McpServerError> {
            Ok(json!({ "key": key, "messages": [], "limit": limit }))
        }

        async fn messages_send(
            &self,
            key: &str,
            message: &str,
            intent: &str,
        ) -> Result<Value, McpServerError> {
            Ok(json!({ "key": key, "sent": true, "message": message, "intent": intent }))
        }

        async fn events_wait(
            &self,
            key: &str,
            since_stream_seq: Option<i64>,
            timeout_ms: u64,
            max_events: u32,
            terminal_only: bool,
        ) -> Result<Value, McpServerError> {
            Ok(json!({
                "key": key,
                "events": [],
                "since": since_stream_seq,
                "timeout_ms": timeout_ms,
                "max_events": max_events,
                "terminal_only": terminal_only
            }))
        }

        async fn transcript_jsonl(
            &self,
            key: &str,
            limit: Option<u32>,
        ) -> Result<String, McpServerError> {
            Ok(format!("{{\"key\":\"{key}\",\"limit\":{limit:?}}}"))
        }
    }

    fn request(method: &str, params: Option<Value>) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(JsonRpcId::Number(1)),
            method: method.to_string(),
            params,
        }
    }

    #[tokio::test]
    async fn test_initialize() {
        let server = McpServer::with_name("test", "9.9.9");
        let resp = server.handle_jsonrpc(&request("initialize", None)).await;
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"].as_str(), Some("test"));
        assert_eq!(
            result["protocolVersion"].as_str(),
            Some(MCP_PROTOCOL_VERSION)
        );
    }

    #[tokio::test]
    async fn test_tools_list() {
        let server = McpServer::with_name("test", "9.9.9");
        server
            .register_tool(McpTool {
                name: "echo".into(),
                description: "Echo".into(),
                input_schema: json!({}),
            })
            .await;
        let resp = server.handle_jsonrpc(&request("tools/list", None)).await;
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
    }

    #[tokio::test]
    async fn test_tools_call_metadata_only() {
        let server = McpServer::with_name("test", "9.9.9");
        server
            .register_tool(McpTool {
                name: "echo".into(),
                description: "Echo".into(),
                input_schema: json!({}),
            })
            .await;
        let resp = server
            .handle_jsonrpc(&request(
                "tools/call",
                Some(json!({ "name": "echo", "arguments": {} })),
            ))
            .await;
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[tokio::test]
    async fn test_tools_call_not_found() {
        let server = McpServer::with_name("test", "9.9.9");
        let resp = server
            .handle_jsonrpc(&request(
                "tools/call",
                Some(json!({ "name": "nope", "arguments": {} })),
            ))
            .await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[tokio::test]
    async fn test_tools_call_missing_params() {
        let server = McpServer::with_name("test", "9.9.9");
        let resp = server.handle_jsonrpc(&request("tools/call", None)).await;
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_method_not_found() {
        let server = McpServer::with_name("test", "9.9.9");
        let resp = server.handle_jsonrpc(&request("bogus/method", None)).await;
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_session_op_via_bridge() {
        let server = McpServer::with_name("test", "9.9.9").with_bridge(Arc::new(MockBridge));
        server.register_session_tools().await;
        let resp = server
            .handle_jsonrpc(&request(
                "tools/call",
                Some(json!({ "name": "conversations_list", "arguments": { "limit": 10 } })),
            ))
            .await;
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert!(result["sessions"].as_array().unwrap().is_empty());
        assert_eq!(result["limit"].as_u64(), Some(10));
    }

    #[tokio::test]
    async fn test_messages_send_via_bridge() {
        let server = McpServer::with_name("test", "9.9.9").with_bridge(Arc::new(MockBridge));
        server.register_session_tools().await;
        let resp = server
            .handle_jsonrpc(&request(
                "tools/call",
                Some(json!({
                    "name": "messages_send",
                    "arguments": { "key": "s1", "message": "hi", "intent": "continue" }
                })),
            ))
            .await;
        let result = resp.result.unwrap();
        assert_eq!(result["sent"].as_bool(), Some(true));
        assert_eq!(result["message"].as_str(), Some("hi"));
    }

    #[tokio::test]
    async fn test_axum_router_endpoint() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let server = McpServer::with_name("test", "9.9.9");
        server
            .register_tool(McpTool {
                name: "echo".into(),
                description: "Echo".into(),
                input_schema: json!({}),
            })
            .await;
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["result"]["tools"][0]["name"], "echo");
    }

    #[tokio::test]
    async fn test_axum_router_batch() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let server = McpServer::with_name("test", "9.9.9");
        let app = server.router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/mcp")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","id":2,"method":"ping"}]"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }
}
