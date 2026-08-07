//! MCP server commands.
//!
//! Implements the `mcp-server` subcommand, which runs OpenSquilla as an
//! inbound Model Context Protocol server over stdio. Mirrors the Python
//! `mcp_server_cmd.py`, which bridges session workflows to a gateway.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_mcp::McpServer;
use tracing::info;

/// MCP server subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum McpServerAction {
    /// Run a stdio MCP server exposing OpenSquilla session workflows.
    Run {
        /// OpenSquilla gateway URL to bridge to.
        #[arg(long, default_value = "ws://localhost:18791/ws")]
        gateway: String,
    },
}

/// Run an MCP server subcommand.
pub async fn run_mcp_server(action: McpServerAction) -> Result<()> {
    match action {
        McpServerAction::Run { gateway } => mcp_server_run(gateway).await,
    }
}

/// Run the inbound MCP server over stdio.
///
/// Builds an [`McpServer`], registers the built-in session tools, and drives
/// the newline-delimited JSON-RPC loop on stdin/stdout. A gateway bridge is
/// required for the session-operation tools to resolve real sessions; until the
/// Rust gateway WebSocket bridge is wired, those tools return an error result.
pub async fn mcp_server_run(gateway: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let server = McpServer::new(&config);

    // TODO: construct a SessionBridge backed by the gateway WebSocket URL
    // (`gateway`) and attach it with `server.with_bridge(...)`. The Rust
    // gateway bridge (`OpenSquillaMCPBridge` equivalent) is not yet
    // implemented; until then session-operation tools surface an error.
    info!(gateway = %gateway, "MCP server starting in stdio mode (bridge not yet wired)");

    server.register_session_tools().await;

    server
        .run_stdio()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
    Ok(())
}
