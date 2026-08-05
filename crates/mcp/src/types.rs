use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;

/// An MCP request from a client to a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    pub id: String,
}

/// An MCP response from a server to a client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub id: String,
    #[serde(default)]
    pub result: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

/// A tool exposed by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: Value,
}

/// A request to invoke a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: HashMap<String, Value>,
}

/// A resource exposed by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A prompt template exposed by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

/// An argument for a prompt template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub required: bool,
}

/// Transport protocol for MCP communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransportProtocol {
    #[serde(rename = "stdio")]
    Stdio,
    #[serde(rename = "sse")]
    Sse,
    #[serde(rename = "streamable-http")]
    StreamableHttp,
}

/// Capabilities advertised by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServerCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptCapabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    #[serde(default)]
    pub subscribe: bool,
    #[serde(default)]
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCapabilities {
    #[serde(default)]
    pub list_changed: bool,
}

/// Initialization request parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    pub client_info: ClientInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClientCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roots: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
}

/// A JSON-RPC 2.0 request id. The MCP wire protocol allows numeric, string, or
/// null ids.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum JsonRpcId {
    Number(u64),
    String(String),
    Null,
}

impl JsonRpcId {
    /// Render the id as a string (for logging / MCP-layer conversion).
    pub fn to_id_string(&self) -> String {
        match self {
            JsonRpcId::Number(n) => n.to_string(),
            JsonRpcId::String(s) => s.clone(),
            JsonRpcId::Null => String::new(),
        }
    }
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    /// Create a new JSON-RPC error.
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// JSON-RPC parse error (-32700).
    pub fn parse_error() -> Self {
        Self::new(-32700, "Parse error")
    }

    /// JSON-RPC invalid request (-32600).
    pub fn invalid_request() -> Self {
        Self::new(-32600, "Invalid Request")
    }

    /// JSON-RPC method not found (-32601).
    pub fn method_not_found(method: &str) -> Self {
        Self::new(-32601, format!("Method not found: {method}"))
    }

    /// JSON-RPC invalid params (-32602).
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self::new(-32602, message)
    }

    /// JSON-RPC internal error (-32603).
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::new(-32603, message)
    }

    /// MCP tool-not-found error.
    pub fn tool_not_found(name: &str) -> Self {
        Self::new(-32602, format!("Tool not found: {name}"))
    }
}

/// A JSON-RPC 2.0 request as it appears on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<JsonRpcId>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

/// A JSON-RPC 2.0 response as it appears on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<JsonRpcId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Build a successful response.
    pub fn ok(id: JsonRpcId, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    /// Build an error response.
    pub fn error(id: JsonRpcId, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(id),
            result: None,
            error: Some(error),
        }
    }

    /// Build a response for a notification (no id, no result).
    pub fn notification() -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: None,
            result: None,
            error: None,
        }
    }
}

impl From<&McpRequest> for JsonRpcRequest {
    fn from(req: &McpRequest) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(JsonRpcId::String(req.id.clone())),
            method: req.method.clone(),
            params: req.params.clone(),
        }
    }
}

impl From<JsonRpcResponse> for McpResponse {
    fn from(resp: JsonRpcResponse) -> Self {
        Self {
            id: resp.id.map(|id| id.to_id_string()).unwrap_or_default(),
            result: resp.result.unwrap_or(Value::Null),
            error: resp
                .error
                .map(|e| json!({ "code": e.code, "message": e.message })),
        }
    }
}

impl From<&JsonRpcResponse> for McpResponse {
    fn from(resp: &JsonRpcResponse) -> Self {
        Self {
            id: resp
                .id
                .clone()
                .map(|id| id.to_id_string())
                .unwrap_or_default(),
            result: resp.result.clone().unwrap_or(Value::Null),
            error: resp
                .error
                .clone()
                .map(|e| json!({ "code": e.code, "message": e.message })),
        }
    }
}

impl From<McpResponse> for JsonRpcResponse {
    fn from(resp: McpResponse) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(JsonRpcId::String(resp.id.clone())),
            result: Some(resp.result),
            error: resp.error.map(|e| {
                JsonRpcError::new(
                    e.get("code").and_then(|c| c.as_i64()).unwrap_or(-32603),
                    e.get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error"),
                )
            }),
        }
    }
}

impl From<&McpResponse> for JsonRpcResponse {
    fn from(resp: &McpResponse) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id: Some(JsonRpcId::String(resp.id.clone())),
            result: Some(resp.result.clone()),
            error: resp.error.clone().map(|e| {
                JsonRpcError::new(
                    e.get("code").and_then(|c| c.as_i64()).unwrap_or(-32603),
                    e.get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown error"),
                )
            }),
        }
    }
}

/// The structured result of a tool execution, normalized from the MCP
/// `content` array.
#[derive(Debug, Clone)]
pub struct McpToolResult {
    /// The concatenated text content of the tool result.
    pub content: String,
    /// Whether the server flagged the execution as an error (`isError`).
    pub is_error: bool,
    /// Non-text content items (e.g. images, embedded resources).
    pub structured_content: Vec<Value>,
    /// The raw MCP `tools/call` result.
    pub raw: Value,
}

impl McpToolResult {
    /// Build a result from a raw `tools/call` result value.
    pub fn from_raw(result: Value) -> Self {
        let is_error = result
            .get("isError")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let content_items = result
            .get("content")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut text_parts = Vec::new();
        let mut structured = Vec::new();
        for item in content_items {
            if item.get("type").and_then(|v| v.as_str()) == Some("text") {
                text_parts.push(
                    item.get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                );
            } else {
                structured.push(item);
            }
        }

        Self {
            content: text_parts.join("\n"),
            is_error,
            structured_content: structured,
            raw: result,
        }
    }

    /// Build a successful result from text content.
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            structured_content: Vec::new(),
            raw: json!({}),
        }
    }

    /// Build an error result.
    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            structured_content: Vec::new(),
            raw: json!({}),
        }
    }
}
