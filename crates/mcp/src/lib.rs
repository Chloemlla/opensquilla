//! # OpenSquilla MCP
//!
//! Model Context Protocol (MCP) client and server implementations.
//! Supports stdio, SSE, and Streamable HTTP transport.
//!
//! - `client` — Outbound MCP client: connects to external MCP servers via a
//!   stdio subprocess, JSON-RPC 2.0 over stdin/stdout, tool discovery and
//!   execution.
//! - `server` — Inbound MCP server: axum JSON-RPC endpoint, session operation
//!   bridge, and tool registration.
//! - `types` — MCP and JSON-RPC data types.
//! - `transport` — Transport layer (stdio, SSE, Streamable HTTP).

pub mod client;
pub mod server;
pub mod types;
pub mod transport;

pub use client::{McpClient, McpError, McpServerConfig};
pub use server::{McpServer, McpServerError, McpToolExecutor, SessionBridge, MCP_PROTOCOL_VERSION};
pub use types::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError, JsonRpcId, JsonRpcRequest,
    JsonRpcResponse, McpPrompt, McpPromptArgument, McpRequest, McpResource, McpResponse, McpTool,
    McpToolCall, McpToolResult, PromptCapabilities, ResourceCapabilities, ServerCapabilities,
    ToolCapabilities, TransportProtocol,
};
pub use transport::Transport;