//! # Gateway RPC client
//!
//! An HTTP/JSON-RPC client for talking to a running OpenSquilla gateway
//! instance. The CLI uses this for all "Mode A" commands that need to interact
//! with live sessions, models, skills, channels, and cost data without
//! spinning up an in-process gateway.
//!
//! The client speaks the JSON-RPC 2.0 protocol over HTTP POST. It falls back
//! to the in-process gateway dispatcher when the gateway is not reachable,
//! so commands work in both connected and standalone modes.

use std::time::Duration;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, warn};

use crate::util;

/// Default request timeout for RPC calls.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RPC error {}: {}", self.code, self.message)?;
        if let Some(data) = &self.data {
            write!(f, " (data: {data})")?;
        }
        Ok(())
    }
}

impl std::error::Error for RpcError {}

/// JSON-RPC 2.0 response envelope.
#[derive(Debug, Clone, Deserialize)]
struct RpcResponse {
    #[allow(dead_code)]
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    error: Option<RpcError>,
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<Value>,
}

/// JSON-RPC 2.0 request envelope.
#[derive(Debug, Clone, Serialize)]
struct RpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: Value,
    id: u64,
}

/// An HTTP-based JSON-RPC client for the OpenSquilla gateway.
pub struct RpcClient {
    base_url: String,
    client: reqwest::Client,
    next_id: std::sync::atomic::AtomicU64,
}

impl RpcClient {
    /// Create a new RPC client targeting the given base URL.
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to build HTTP client");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            next_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create a client from the gateway configuration.
    pub fn from_config(config: &Config) -> Self {
        let url = format!("http://{}:{}", config.gateway.host, config.gateway.port);
        Self::new(&url)
    }

    /// Build the RPC endpoint URL.
    fn endpoint(&self) -> String {
        format!("{}/rpc", self.base_url)
    }

    /// Allocate the next request id.
    fn next_request_id(&self) -> u64 {
        self.next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Send a JSON-RPC call and return the result.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_request_id();
        let request = RpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id,
        };

        debug!(method = %method, id = id, "Sending RPC call");

        let response = self
            .client
            .post(self.endpoint())
            .json(&request)
            .send()
            .await
            .with_context(|| format!("Failed to send RPC '{method}'"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("RPC '{method}' returned HTTP {status}: {body}");
        }

        let rpc_response: RpcResponse = response
            .json()
            .await
            .with_context(|| format!("Failed to parse RPC '{method}' response"))?;

        if let Some(err) = rpc_response.error {
            return Err(anyhow::Error::new(err).context(format!("RPC '{method}' failed")));
        }

        rpc_response
            .result
            .ok_or_else(|| anyhow::anyhow!("RPC '{method}' returned no result"))
    }

    /// Send a notification (no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let id = self.next_request_id();
        let request = RpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id,
        };

        debug!(method = %method, "Sending RPC notification");

        let response = self
            .client
            .post(self.endpoint())
            .json(&request)
            .send()
            .await
            .with_context(|| format!("Failed to send notification '{method}'"))?;

        if !response.status().is_success() {
            warn!(method = %method, status = %response.status(), "Notification returned non-200");
        }
        Ok(())
    }

    /// Ping the gateway to check connectivity.
    pub async fn ping(&self) -> Result<Duration> {
        let start = std::time::Instant::now();
        self.call("system.ping", Value::Null).await?;
        Ok(start.elapsed())
    }

    /// Check if the gateway is reachable.
    pub async fn is_reachable(&self) -> bool {
        self.ping().await.is_ok()
    }
}

// ---------------------------------------------------------------------------
// Typed convenience methods
// ---------------------------------------------------------------------------

impl RpcClient {
    /// List all sessions for the default agent.
    pub async fn list_sessions(&self, agent_id: &str, limit: u64, offset: u64) -> Result<Value> {
        self.call(
            "sessions.list",
            serde_json::json!({
                "agent_id": agent_id,
                "limit": limit,
                "offset": offset,
            }),
        )
        .await
    }

    /// Get a session by id.
    pub async fn get_session(&self, session_id: &str) -> Result<Value> {
        self.call(
            "sessions.get",
            serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Create a new session.
    pub async fn create_session(&self, agent_id: &str, name: &str, mode: &str) -> Result<Value> {
        self.call(
            "sessions.create",
            serde_json::json!({
                "agent_id": agent_id,
                "name": name,
                "mode": mode,
            }),
        )
        .await
    }

    /// Delete a session.
    pub async fn delete_session(&self, session_id: &str) -> Result<Value> {
        self.call(
            "sessions.delete",
            serde_json::json!({ "session_id": session_id }),
        )
        .await
    }

    /// Get a session transcript.
    pub async fn get_transcript(&self, session_id: &str, limit: u64, offset: u64) -> Result<Value> {
        self.call(
            "sessions.transcript",
            serde_json::json!({
                "session_id": session_id,
                "limit": limit,
                "offset": offset,
            }),
        )
        .await
    }

    /// Send a chat message to a session.
    pub async fn send_chat(
        &self,
        session_id: &str,
        message: &str,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<Value> {
        self.call(
            "chat.send",
            serde_json::json!({
                "session_id": session_id,
                "message": message,
                "provider": provider,
                "model": model,
            }),
        )
        .await
    }

    /// List configured providers.
    pub async fn list_providers(&self) -> Result<Value> {
        self.call("providers.list", Value::Null).await
    }

    /// Get provider status.
    pub async fn provider_status(&self, name: Option<&str>) -> Result<Value> {
        let params = match name {
            Some(n) => serde_json::json!({ "name": n }),
            None => Value::Null,
        };
        self.call("providers.status", params).await
    }

    /// Test a provider connection.
    pub async fn test_provider(&self, name: Option<&str>) -> Result<Value> {
        let params = match name {
            Some(n) => serde_json::json!({ "name": n }),
            None => Value::Null,
        };
        self.call("providers.test", params).await
    }

    /// List models, optionally filtered by provider.
    pub async fn list_models(&self, provider: Option<&str>) -> Result<Value> {
        let params = match provider {
            Some(p) => serde_json::json!({ "provider": p }),
            None => Value::Null,
        };
        self.call("models.list", params).await
    }

    /// Get model details.
    pub async fn get_model(&self, name: &str) -> Result<Value> {
        self.call("models.get", serde_json::json!({ "name": name }))
            .await
    }

    /// List installed skills.
    pub async fn list_skills(&self) -> Result<Value> {
        self.call("skills.list", Value::Null).await
    }

    /// Get skill details.
    pub async fn get_skill(&self, name: &str) -> Result<Value> {
        self.call("skills.get", serde_json::json!({ "name": name }))
            .await
    }

    /// Install a skill.
    pub async fn install_skill(&self, source: &str) -> Result<Value> {
        self.call("skills.install", serde_json::json!({ "source": source }))
            .await
    }

    /// Uninstall a skill.
    pub async fn uninstall_skill(&self, name: &str) -> Result<Value> {
        self.call("skills.uninstall", serde_json::json!({ "name": name }))
            .await
    }

    /// Search the skill hub.
    pub async fn search_skills(&self, query: &str) -> Result<Value> {
        self.call("skills.search", serde_json::json!({ "query": query }))
            .await
    }

    /// Enable a skill.
    pub async fn enable_skill(&self, name: &str) -> Result<Value> {
        self.call("skills.enable", serde_json::json!({ "name": name }))
            .await
    }

    /// Disable a skill.
    pub async fn disable_skill(&self, name: &str) -> Result<Value> {
        self.call("skills.disable", serde_json::json!({ "name": name }))
            .await
    }

    /// List channels.
    pub async fn list_channels(&self) -> Result<Value> {
        self.call("channels.list", Value::Null).await
    }

    /// Get channel status.
    pub async fn channel_status(&self, name: Option<&str>) -> Result<Value> {
        let params = match name {
            Some(n) => serde_json::json!({ "name": n }),
            None => Value::Null,
        };
        self.call("channels.status", params).await
    }

    /// Get cost/usage summary.
    pub async fn cost_summary(&self, start: Option<&str>, end: Option<&str>) -> Result<Value> {
        let params = match (start, end) {
            (Some(s), Some(e)) => serde_json::json!({ "start": s, "end": e }),
            (Some(s), None) => serde_json::json!({ "start": s }),
            _ => Value::Null,
        };
        self.call("cost.summary", params).await
    }

    /// Get usage breakdown.
    pub async fn usage_breakdown(&self, group_by: &str) -> Result<Value> {
        self.call("cost.usage", serde_json::json!({ "group_by": group_by }))
            .await
    }

    /// Get gateway info.
    pub async fn gateway_info(&self) -> Result<Value> {
        self.call("gateway.info", Value::Null).await
    }

    /// Get gateway metrics.
    pub async fn gateway_metrics(&self) -> Result<Value> {
        self.call("gateway.metrics", Value::Null).await
    }

    /// List scheduled tasks.
    pub async fn list_tasks(&self) -> Result<Value> {
        self.call("scheduler.list", Value::Null).await
    }

    /// Get memory entries.
    pub async fn list_memory(&self, agent_id: &str, limit: u64) -> Result<Value> {
        self.call(
            "memory.list",
            serde_json::json!({ "agent_id": agent_id, "limit": limit }),
        )
        .await
    }

    /// Search memory.
    pub async fn search_memory(&self, query: &str, limit: u64) -> Result<Value> {
        self.call(
            "memory.search",
            serde_json::json!({ "query": query, "limit": limit }),
        )
        .await
    }

    /// Trigger memory dream.
    pub async fn memory_dream(&self, agent_id: &str) -> Result<Value> {
        self.call("memory.dream", serde_json::json!({ "agent_id": agent_id }))
            .await
    }

    /// Run a health check.
    pub async fn health_check(&self) -> Result<Value> {
        self.call("health.check", Value::Null).await
    }
}

// ---------------------------------------------------------------------------
// Fallback dispatcher
// ---------------------------------------------------------------------------

/// Try an RPC call against the live gateway; on connection failure, fall back
/// to the in-process gateway dispatcher.
///
/// This lets every command module attempt the network path first (so the CLI
/// talks to a long-running gateway when one is up) and silently degrade to
/// spinning up an in-process gateway for a single call otherwise.
pub async fn rpc_or_fallback(config: &Config, method: &str, params: Value) -> Result<Value> {
    let client = RpcClient::from_config(config);
    if client.is_reachable().await {
        return client.call(method, params).await;
    }
    debug!("Gateway not reachable, falling back to in-process RPC for '{method}'");
    util::gateway_rpc(config, method, params).await
}

/// Always use the in-process gateway (no HTTP).
pub async fn in_process_rpc(config: &Config, method: &str, params: Value) -> Result<Value> {
    util::gateway_rpc(config, method, params).await
}

/// Build an RPC client, preferring the gateway if reachable.
pub async fn connect(config: &Config) -> RpcClient {
    let client = RpcClient::from_config(config);
    if !client.is_reachable().await {
        warn!(
            "Gateway at {}:{} is not reachable; commands will use in-process fallback",
            config.gateway.host, config.gateway.port
        );
    }
    client
}

// ---------------------------------------------------------------------------
// Streaming support
// ---------------------------------------------------------------------------

/// A handle to a streaming RPC response.
pub struct StreamHandle {
    pub session_id: String,
    pub stream_id: String,
}

/// Subscribe to a streaming chat session.
pub async fn subscribe_stream(client: &RpcClient, session_id: &str) -> Result<StreamHandle> {
    let result = client
        .call(
            "chat.stream",
            serde_json::json!({ "session_id": session_id }),
        )
        .await?;
    let stream_id = result
        .get("stream_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("Missing stream_id in response"))?
        .to_string();
    Ok(StreamHandle {
        session_id: session_id.to_string(),
        stream_id,
    })
}

/// Poll a stream for the next delta. Returns `None` when the stream is done.
pub async fn poll_stream(client: &RpcClient, stream_id: &str) -> Result<Option<Value>> {
    let result = client
        .call("chat.poll", serde_json::json!({ "stream_id": stream_id }))
        .await?;
    if result.is_null() {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// Cancel an active stream.
pub async fn cancel_stream(client: &RpcClient, stream_id: &str) -> Result<()> {
    client
        .call("chat.cancel", serde_json::json!({ "stream_id": stream_id }))
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_error_display() {
        let err = RpcError {
            code: -32601,
            message: "Method not found".to_string(),
            data: None,
        };
        assert!(format!("{err}").contains("-32601"));
        assert!(format!("{err}").contains("Method not found"));
    }

    #[test]
    fn test_rpc_error_with_data() {
        let err = RpcError {
            code: -32602,
            message: "Invalid params".to_string(),
            data: Some(serde_json::json!({ "field": "session_id" })),
        };
        let s = format!("{err}");
        assert!(s.contains("session_id"));
    }

    #[test]
    fn test_rpc_request_serialization() {
        let req = RpcRequest {
            jsonrpc: "2.0",
            method: "ping".to_string(),
            params: Value::Null,
            id: 1,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"ping\""));
    }

    #[test]
    fn test_rpc_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","result":{"ok":true},"id":1}"#;
        let resp: RpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_rpc_response_error() {
        let json = r#"{"jsonrpc":"2.0","error":{"code":-32601,"message":"not found"},"id":1}"#;
        let resp: RpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_none());
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_client_url_construction() {
        let client = RpcClient::new("http://localhost:8080/");
        assert_eq!(client.base_url, "http://localhost:8080");
        assert_eq!(client.endpoint(), "http://localhost:8080/rpc");
    }
}
