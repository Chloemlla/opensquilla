use std::sync::Arc;

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use tracing::{info, warn};

use crate::types::{McpRequest, McpResponse};

/// Transport layer for MCP communication.
pub enum Transport {
    /// Standard I/O transport (stdio).
    Stdio,
    /// Server-Sent Events transport (SSE).
    Sse {
        endpoint: String,
        port: u16,
    },
    /// Streamable HTTP transport.
    StreamableHttp {
        endpoint: String,
        port: u16,
    },
}

impl Transport {
    pub fn stdio() -> Self {
        Transport::Stdio
    }

    pub fn sse(endpoint: String, port: u16) -> Self {
        Transport::Sse { endpoint, port }
    }

    pub fn streamable_http(endpoint: String, port: u16) -> Self {
        Transport::StreamableHttp { endpoint, port }
    }

    pub fn name(&self) -> &str {
        match self {
            Transport::Stdio => "stdio",
            Transport::Sse { .. } => "sse",
            Transport::StreamableHttp { .. } => "streamable-http",
        }
    }

    pub async fn serve(
        &self,
        handler: Arc<dyn RequestHandler + Send + Sync>,
    ) -> Result<(), TransportError> {
        match self {
            Transport::Stdio => {
                info!("Starting MCP transport: stdio");
                run_stdio(handler).await
            }
            Transport::Sse { endpoint, port } => {
                info!("Starting MCP transport: SSE on port {port}");
                run_sse(handler, endpoint, *port).await
            }
            Transport::StreamableHttp { endpoint, port } => {
                info!("Starting MCP transport: Streamable HTTP on port {port}");
                run_streamable_http(handler, endpoint, *port).await
            }
        }
    }
}

/// Handler for incoming MCP requests.
#[async_trait::async_trait]
pub trait RequestHandler: Send + Sync {
    async fn handle_request(&self, request: McpRequest) -> McpResponse;
}

async fn run_stdio(handler: Arc<dyn RequestHandler + Send + Sync>) -> Result<(), TransportError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
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
                match serde_json::from_str::<McpRequest>(trimmed) {
                    Ok(request) => {
                        let response = handler.handle_request(request).await;
                        let json = serde_json::to_string(&response).unwrap_or_default();
                        writer.write_all(json.as_bytes()).await.ok();
                        writer.write_all(b"\n").await.ok();
                        writer.flush().await.ok();
                    }
                    Err(e) => {
                        warn!("Invalid MCP request: {e}");
                    }
                }
            }
            Err(e) => {
                warn!("Error reading stdin: {e}");
                break;
            }
        }
    }
    Ok(())
}

async fn run_sse(
    handler: Arc<dyn RequestHandler + Send + Sync>,
    endpoint: &str,
    port: u16,
) -> Result<(), TransportError> {
    let state = AppState { handler };

    let app = Router::new()
        .route("/sse", get(sse_handler))
        .route(endpoint, post(message_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("SSE transport listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| TransportError::BindFailed(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| TransportError::ServeError(e.to_string()))?;
    Ok(())
}

async fn run_streamable_http(
    handler: Arc<dyn RequestHandler + Send + Sync>,
    endpoint: &str,
    port: u16,
) -> Result<(), TransportError> {
    let state = AppState { handler };

    let app = Router::new()
        .route(endpoint, post(message_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    info!("Streamable HTTP transport listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| TransportError::BindFailed(e.to_string()))?;
    axum::serve(listener, app)
        .await
        .map_err(|e| TransportError::ServeError(e.to_string()))?;
    Ok(())
}

#[derive(Clone)]
struct AppState {
    handler: Arc<dyn RequestHandler + Send + Sync>,
}

async fn sse_handler(
    State(_state): State<AppState>,
) -> axum::response::Sse<impl futures::stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>
{
    use axum::response::sse::Event;
    use futures::stream::{self, StreamExt};
    use tokio_stream::wrappers::IntervalStream;
    use std::time::Duration;

    let stream = stream::once(async {
        Ok(Event::default().data("connected"))
    });

    let interval = tokio::time::interval(Duration::from_secs(30));
    let keepalive = IntervalStream::new(interval).map(|_| {
        Ok(Event::default().data("ping").event("ping"))
    });

    let combined = stream.chain(keepalive);
    axum::response::Sse::new(combined).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("ping"),
    )
}

async fn message_handler(
    State(state): State<AppState>,
    Json(request): Json<McpRequest>,
) -> Json<McpResponse> {
    let response = state.handler.handle_request(request).await;
    Json(response)
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("Failed to bind to address: {0}")]
    BindFailed(String),

    #[error("Server error: {0}")]
    ServeError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}