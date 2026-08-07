//! Outbound MCP client.
//!
//! Connects to external MCP servers over the stdio transport: spawns the server
//! as a subprocess, speaks JSON-RPC 2.0 over its stdin/stdout using
//! newline-delimited frames, performs the MCP `initialize` handshake, and
//! exposes tool discovery and execution.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use opensquilla_core::config::Config;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStderr, ChildStdin, ChildStdout, Command as TokioCommand};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::types::{
    JsonRpcId, JsonRpcRequest, JsonRpcResponse, McpPrompt, McpRequest, McpResource, McpResponse,
    McpTool, McpToolCall, McpToolResult,
};

/// The MCP protocol version this client speaks.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// Configuration for connecting to an external MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// The logical name of the server (used as the connection key).
    pub name: String,
    /// The command to spawn.
    pub command: String,
    /// Arguments passed to the command.
    pub args: Vec<String>,
    /// Extra environment variables for the subprocess.
    pub env: HashMap<String, String>,
    /// Tool call timeout.
    pub tool_timeout: Duration,
}

impl McpServerConfig {
    /// Create a new stdio server config.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            env: HashMap::new(),
            tool_timeout: Duration::from_secs(30),
        }
    }

    /// Set the subprocess arguments.
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set extra environment variables.
    pub fn env(mut self, env: HashMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Set the tool call timeout.
    pub fn tool_timeout(mut self, timeout: Duration) -> Self {
        self.tool_timeout = timeout;
        self
    }
}

/// A running MCP stdio subprocess.
struct McpProcess {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    child: tokio::process::Child,
    stderr_task: tokio::task::JoinHandle<()>,
}

/// Drain a subprocess stderr pipe into `tracing` warnings.
async fn drain_stderr(server_name: String, stderr: ChildStderr) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        warn!(server = %server_name, stderr = %trimmed, "MCP server stderr");
                    }
                }
                Err(_) => break,
            }
        }
    })
}

/// Outbound MCP client.
///
/// Each server connection is keyed by its configured name. Requests to a given
/// server are serialized (the stdio transport requires one outstanding request
/// at a time).
pub struct McpClient {
    processes: Arc<Mutex<HashMap<String, McpProcess>>>,
    next_id: AtomicU64,
}

impl Default for McpClient {
    /// Create a new client with no configuration.
    fn default() -> Self {
        Self {
            processes: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }
}

impl McpClient {
    /// Create a new client. The configuration is currently unused by the stdio
    /// client (server configs are supplied per-connection).
    pub fn new(_config: &Config) -> Self {
        Self::default()
    }

    /// Connect to an MCP server over stdio and perform the `initialize`
    /// handshake.
    pub async fn connect(&self, config: &McpServerConfig) -> Result<(), McpError> {
        {
            let processes = self.processes.lock().await;
            if processes.contains_key(&config.name) {
                return Err(McpError::AlreadyConnected(config.name.clone()));
            }
        }

        info!(
            server = %config.name,
            command = %config.command,
            "Connecting to MCP server via stdio"
        );

        let mut cmd = TokioCommand::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if !config.env.is_empty() {
            cmd.envs(&config.env);
        }

        let mut child = cmd.spawn().map_err(|e| {
            McpError::ConnectionFailed(format!("Failed to spawn {}: {e}", config.command))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::ConnectionFailed("stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::ConnectionFailed("stdout unavailable".into()))?;
        let stderr = child.stderr.take();

        let server_name = config.name.clone();
        let stderr_task = match stderr {
            Some(pipe) => drain_stderr(server_name.clone(), pipe).await,
            None => tokio::spawn(async {}),
        };

        let process = McpProcess {
            stdin,
            stdout: BufReader::new(stdout),
            child,
            stderr_task,
        };
        {
            let mut processes = self.processes.lock().await;
            processes.insert(server_name.clone(), process);
        }

        // MCP initialize handshake; roll back on failure. The lock is released
        // above so the handshake (which re-acquires it) cannot deadlock.
        if let Err(e) = self.initialize(&server_name).await {
            self.disconnect(&server_name).await.ok();
            return Err(e);
        }

        debug!(server = %server_name, "MCP server connected");
        Ok(())
    }

    /// Backwards-compatible convenience wrapper for `connect`.
    pub async fn connect_stdio(
        &self,
        server_name: &str,
        command: &str,
        args: &[String],
    ) -> Result<(), McpError> {
        let config = McpServerConfig::new(server_name, command).args(args.to_vec());
        self.connect(&config).await
    }

    /// Perform the MCP `initialize` handshake followed by the
    /// `notifications/initialized` notification.
    pub async fn initialize(&self, server_name: &str) -> Result<Value, McpError> {
        let params = json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "opensquilla",
                "version": env!("CARGO_PKG_VERSION"),
            },
        });
        let result = self
            .request(server_name, "initialize", Some(params))
            .await?;
        self.notify(server_name, "notifications/initialized")
            .await?;
        Ok(result)
    }

    /// Disconnect from a server, killing its subprocess.
    pub async fn disconnect(&self, server_name: &str) -> Result<(), McpError> {
        let mut processes = self.processes.lock().await;
        let mut process = processes
            .remove(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        info!(server = %server_name, "Disconnecting MCP server");
        process.child.kill().await.ok();
        process.child.wait().await.ok();
        process.stderr_task.abort();
        Ok(())
    }

    /// Send a JSON-RPC request and await the matching response result.
    pub async fn request(
        &self,
        server_name: &str,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, McpError> {
        let id = JsonRpcId::Number(self.next_id.fetch_add(1, Ordering::SeqCst));
        let wire = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(id.clone()),
            method: method.to_string(),
            params,
        };
        let json = serde_json::to_string(&wire)
            .map_err(|e| McpError::SerializationFailed(e.to_string()))?;

        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        debug!(server = %server_name, method = %method, "Sending MCP request");
        process
            .stdin
            .write_all(format!("{json}\n").as_bytes())
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("write error: {e}")))?;
        process
            .stdin
            .flush()
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("flush error: {e}")))?;

        let mut line = String::new();
        loop {
            line.clear();
            let n = process
                .stdout
                .read_line(&mut line)
                .await
                .map_err(|e| McpError::ConnectionFailed(format!("read error: {e}")))?;
            if n == 0 {
                return Err(McpError::ConnectionFailed(format!(
                    "{server_name} closed the connection"
                )));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(trimmed) else {
                debug!(server = %server_name, "Skipping non-JSON line from MCP server");
                continue;
            };
            // Skip server-initiated notifications and responses to other ids.
            if resp.id.as_ref() != Some(&id) {
                debug!(server = %server_name, "Skipping MCP message with non-matching id");
                continue;
            }
            if let Some(err) = resp.error {
                return Err(McpError::RpcError {
                    code: err.code,
                    message: err.message,
                });
            }
            return Ok(resp.result.unwrap_or(Value::Null));
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(&self, server_name: &str, method: &str) -> Result<(), McpError> {
        let wire = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: None,
            method: method.to_string(),
            params: None,
        };
        let json = serde_json::to_string(&wire)
            .map_err(|e| McpError::SerializationFailed(e.to_string()))?;

        let mut processes = self.processes.lock().await;
        let process = processes
            .get_mut(server_name)
            .ok_or_else(|| McpError::NotConnected(server_name.to_string()))?;

        process
            .stdin
            .write_all(format!("{json}\n").as_bytes())
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("write error: {e}")))?;
        process
            .stdin
            .flush()
            .await
            .map_err(|e| McpError::ConnectionFailed(format!("flush error: {e}")))?;
        Ok(())
    }

    /// Send an MCP-layer request (backwards-compatible wrapper).
    pub async fn send_request(
        &self,
        server_name: &str,
        request: &McpRequest,
    ) -> Result<McpResponse, McpError> {
        let result = self
            .request(server_name, &request.method, request.params.clone())
            .await?;
        Ok(McpResponse {
            id: request.id.clone(),
            result,
            error: None,
        })
    }

    /// Discover the tools exposed by an MCP server.
    pub async fn list_tools(&self, server_name: &str) -> Result<Vec<McpTool>, McpError> {
        let result = self.request(server_name, "tools/list", None).await?;
        let tools = result.get("tools").cloned().unwrap_or(Value::Null);
        serde_json::from_value(tools).map_err(|e| McpError::DeserializationFailed(e.to_string()))
    }

    /// Invoke a tool on an MCP server.
    pub async fn call_tool(
        &self,
        server_name: &str,
        call: &McpToolCall,
    ) -> Result<McpToolResult, McpError> {
        let params = json!({
            "name": call.name,
            "arguments": call.arguments,
        });
        let result = self
            .request(server_name, "tools/call", Some(params))
            .await?;
        Ok(McpToolResult::from_raw(result))
    }

    /// Invoke a tool by name with a JSON argument map.
    pub async fn call_tool_named(
        &self,
        server_name: &str,
        name: &str,
        arguments: HashMap<String, Value>,
    ) -> Result<McpToolResult, McpError> {
        let call = McpToolCall {
            name: name.to_string(),
            arguments,
        };
        self.call_tool(server_name, &call).await
    }

    /// List resources exposed by an MCP server.
    pub async fn list_resources(&self, server_name: &str) -> Result<Vec<McpResource>, McpError> {
        let result = self.request(server_name, "resources/list", None).await?;
        let resources = result.get("resources").cloned().unwrap_or(Value::Null);
        serde_json::from_value(resources)
            .map_err(|e| McpError::DeserializationFailed(e.to_string()))
    }

    /// List prompt templates exposed by an MCP server.
    pub async fn list_prompts(&self, server_name: &str) -> Result<Vec<McpPrompt>, McpError> {
        let result = self.request(server_name, "prompts/list", None).await?;
        let prompts = result.get("prompts").cloned().unwrap_or(Value::Null);
        serde_json::from_value(prompts).map_err(|e| McpError::DeserializationFailed(e.to_string()))
    }

    /// List the names of all connected servers.
    pub async fn connected_servers(&self) -> Vec<String> {
        self.processes.lock().await.keys().cloned().collect()
    }

    /// Disconnect from all connected servers.
    pub async fn disconnect_all(&self) -> Result<(), McpError> {
        let servers = self.connected_servers().await;
        for server in &servers {
            if let Err(e) = self.disconnect(server).await {
                warn!(server = %server, error = %e, "Error disconnecting MCP server");
            }
        }
        Ok(())
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let processes = self.processes.clone();
        tokio::spawn(async move {
            let mut procs = processes.lock().await;
            for (name, mut process) in procs.drain() {
                debug!(server = %name, "Cleaning up MCP server process");
                process.child.kill().await.ok();
                process.child.wait().await.ok();
                process.stderr_task.abort();
            }
        });
    }
}

/// An error that can occur while interacting with an MCP server.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Already connected to server: {0}")]
    AlreadyConnected(String),

    #[error("Not connected to server: {0}")]
    NotConnected(String),

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Serialization failed: {0}")]
    SerializationFailed(String),

    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    #[error("MCP server error (code {code}): {message}")]
    RpcError { code: i64, message: String },

    #[error("Tool call failed: {0}")]
    ToolError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::JsonRpcRequest;

    #[test]
    fn test_request_wire_format() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(JsonRpcId::Number(1)),
            method: "tools/list".into(),
            params: None,
        };
        let json = serde_json::to_string(&request).unwrap();
        // The MCP stdio transport frames each message as one LF-terminated line,
        // so the serialized form must not contain a literal newline.
        assert!(!json.contains('\n'));
        let reparsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed.method, "tools/list");
        assert_eq!(reparsed.id, Some(JsonRpcId::Number(1)));
    }

    #[test]
    fn test_mcp_tool_result_from_raw() {
        let raw = json!({
            "content": [
                { "type": "text", "text": "line one" },
                { "type": "text", "text": "line two" },
                { "type": "image", "data": "base64", "mimeType": "image/png" }
            ],
            "isError": false
        });
        let result = McpToolResult::from_raw(raw);
        assert_eq!(result.content, "line one\nline two");
        assert!(!result.is_error);
        assert_eq!(result.structured_content.len(), 1);
    }

    #[test]
    fn test_mcp_tool_result_error_flag() {
        let raw = json!({
            "content": [{ "type": "text", "text": "boom" }],
            "isError": true
        });
        let result = McpToolResult::from_raw(raw);
        assert!(result.is_error);
        assert_eq!(result.content, "boom");
    }

    #[test]
    fn test_mcp_server_config_builder() {
        let mut env = HashMap::new();
        env.insert("FOO".to_string(), "bar".to_string());
        let config = McpServerConfig::new("srv", "node")
            .args(vec!["server.js".to_string()])
            .env(env)
            .tool_timeout(Duration::from_secs(5));
        assert_eq!(config.name, "srv");
        assert_eq!(config.args, vec!["server.js".to_string()]);
        assert_eq!(config.tool_timeout, Duration::from_secs(5));
    }
}
