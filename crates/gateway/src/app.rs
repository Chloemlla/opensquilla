//! Gateway application builder and router setup.
//!
//! Provides the top-level `Gateway` struct that wires together the axum HTTP
//! server, middleware layers, RPC handlers, and WebSocket endpoint, plus a
//! [`GatewayBuilder`] for configuring auth, CORS, and custom handlers before
//! construction.

use axum::{Extension, Router, routing::get};
use opensquilla_core::config::GatewayConfig;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::auth::{AuthConfig, AuthMode};
use crate::chat::{ChatStore, register_chat_handlers};
use crate::config::{ConfigStore, register_config_handlers};
use crate::middleware::{
    RateLimiter, SlidingWindowRateLimiter, catch_panic_middleware, cors_layer,
    request_logging_middleware, security_headers_middleware, sliding_window_rate_limit_middleware,
    token_auth_middleware, unsafe_origin_guard_middleware,
};
use crate::rpc::{RpcHandler, RpcRegistry};
use crate::sessions::{SessionStore, register_session_handlers};
use crate::websocket::{ConnectionRegistry, SubscriptionManager, ws_handler};

/// The OpenSquilla gateway server.
///
/// Owns the RPC registry, middleware state, and configuration. Call
/// [`Gateway::new`] to create an instance, then [`Gateway::router`] to obtain
/// the axum `Router` for serving.
pub struct Gateway {
    /// Server binding configuration.
    pub config: GatewayConfig,
    /// RPC handler registry.
    pub rpc_registry: RpcRegistry,
    /// Session store.
    pub session_store: SessionStore,
    /// Chat store.
    pub chat_store: ChatStore,
    /// Config store.
    pub config_store: ConfigStore,
    /// Rate limiter state (token bucket).
    pub rate_limiter: Arc<RateLimiter>,
    /// Authentication configuration shared with middleware and WebSocket.
    pub auth_config: Arc<AuthConfig>,
    /// Registry of active WebSocket connections.
    pub connection_registry: Arc<ConnectionRegistry>,
    /// Subscription manager for WebSocket event fan-out.
    pub subscription_manager: Arc<SubscriptionManager>,
}

impl Gateway {
    /// Start building a gateway from a configuration.
    pub fn builder(config: GatewayConfig) -> GatewayBuilder {
        GatewayBuilder::new(config)
    }

    /// Create a new gateway with the given configuration.
    ///
    /// Registers the default set of RPC handlers (sessions, chat, config).
    pub fn new(config: GatewayConfig) -> Self {
        let mut rpc_registry = RpcRegistry::new();
        let session_store = SessionStore::new();
        let chat_store = ChatStore::new();
        let config_store = ConfigStore::new();

        // Register default handlers
        register_session_handlers(&mut rpc_registry, session_store.clone());
        register_chat_handlers(&mut rpc_registry, chat_store.clone());
        register_config_handlers(&mut rpc_registry, config_store.clone());

        let rate_limiter = RateLimiter::new(
            config.max_connections as u64 * 10,
            config.request_timeout_secs,
        );

        Self {
            config,
            rpc_registry,
            session_store,
            chat_store,
            config_store,
            rate_limiter,
            auth_config: Arc::new(AuthConfig::default()),
            connection_registry: Arc::new(ConnectionRegistry::new()),
            subscription_manager: Arc::new(SubscriptionManager::new()),
        }
    }

    /// Register an additional RPC handler on this gateway.
    pub fn register_handler(&mut self, handler: Arc<dyn RpcHandler>) {
        self.rpc_registry.register(handler);
    }

    /// Register multiple additional RPC handlers on this gateway.
    pub fn register_handlers<I>(&mut self, handlers: I)
    where
        I: IntoIterator<Item = Arc<dyn RpcHandler>>,
    {
        for handler in handlers {
            self.rpc_registry.register(handler);
        }
    }

    /// Set the authentication configuration for this gateway.
    pub fn with_auth_config(mut self, auth: AuthConfig) -> Self {
        self.auth_config = Arc::new(auth);
        self
    }

    /// Build the axum `Router` with all middleware and the WebSocket endpoint.
    pub fn router(self) -> Router {
        let rpc_registry = Arc::new(self.rpc_registry);
        let auth_config = self.auth_config;
        let subscription_manager = self.subscription_manager;
        let connection_registry = self.connection_registry;

        // Auth-derived middleware configuration.
        let expected_token = auth_config.token.clone();
        let auth_mode = auth_config.mode.clone();
        let cors_origins = self.config.cors_origins.clone();

        // Build the middleware stack from the inside out:
        // 1. Catch panics      (outermost)
        // 2. Security headers
        // 3. CORS
        // 4. Request logging
        // 5. Token-bucket rate limit
        // 6. Sliding-window rate limit
        // 7. Origin guard      (unsafe cross-origin mutations)
        // 8. Token auth        (with WS-upgrade exemption)
        // 9. RPC registry + connection extensions (innermost, before handlers)

        let cors: CorsLayer = cors_layer(&cors_origins);
        let rate_limiter = self.rate_limiter.clone();
        let sliding_limiter = SlidingWindowRateLimiter::new(
            self.config.max_connections as u64 * 20,
            self.config.request_timeout_secs,
        );
        let origin_origins = cors_origins.clone();

        Router::new()
            .route("/health", get(Self::health_check))
            .route("/ws", get(ws_handler))
            .layer(Extension(rpc_registry))
            .layer(Extension(auth_config))
            .layer(Extension(subscription_manager))
            .layer(Extension(connection_registry))
            .layer(axum::middleware::from_fn(move |req, next| {
                let token = expected_token.clone();
                let mode = auth_mode.clone();
                async move {
                    // In open mode the token middleware lets everything through.
                    if mode == AuthMode::Open {
                        return Ok(next.run(req).await);
                    }
                    token_auth_middleware(req, next, token).await
                }
            }))
            .layer(axum::middleware::from_fn(move |req, next| {
                let origins = origin_origins.clone();
                async move { unsafe_origin_guard_middleware(req, next, origins).await }
            }))
            .layer(axum::middleware::from_fn(move |req, next| {
                let limiter = sliding_limiter.clone();
                async move { sliding_window_rate_limit_middleware(req, next, limiter).await }
            }))
            .layer(axum::middleware::from_fn(move |req, next| {
                let limiter = rate_limiter.clone();
                async move { crate::middleware::rate_limit_middleware(req, next, limiter).await }
            }))
            .layer(axum::middleware::from_fn(request_logging_middleware))
            .layer(cors)
            .layer(axum::middleware::from_fn(security_headers_middleware))
            .layer(axum::middleware::from_fn(catch_panic_middleware))
    }

    /// Bind to the configured address and start serving.
    ///
    /// This is a convenience method that builds the router and starts the
    /// server. It runs until the process receives a shutdown signal.
    pub async fn serve(self) -> Result<(), Box<dyn std::error::Error>> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port).parse()?;
        let router = self.router();

        info!("Gateway listening on {addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await?;

        Ok(())
    }

    /// Return a health-check endpoint handler.
    pub async fn health_check() -> &'static str {
        "OK"
    }
}

/// Wait for a shutdown signal (Ctrl-C).
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("Shutdown signal received; stopping gateway");
}

/// A builder for configuring a [`Gateway`] before construction.
///
/// Use [`Gateway::builder`] to create one, then chain configuration methods
/// and finish with [`GatewayBuilder::build`].
pub struct GatewayBuilder {
    config: GatewayConfig,
    auth_config: AuthConfig,
    cors_origins: Option<Vec<String>>,
    extra_handlers: Vec<Arc<dyn RpcHandler>>,
}

impl GatewayBuilder {
    /// Start a builder from a gateway configuration.
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            config,
            auth_config: AuthConfig::default(),
            cors_origins: None,
            extra_handlers: Vec::new(),
        }
    }

    /// Set the authentication configuration.
    pub fn auth(mut self, auth: AuthConfig) -> Self {
        self.auth_config = auth;
        self
    }

    /// Override the CORS origins allowed by the gateway.
    pub fn cors(mut self, origins: Vec<String>) -> Self {
        self.cors_origins = Some(origins);
        self
    }

    /// Register an additional RPC handler.
    pub fn handler(mut self, handler: Arc<dyn RpcHandler>) -> Self {
        self.extra_handlers.push(handler);
        self
    }

    /// Register additional RPC handlers.
    pub fn handlers<I>(mut self, handlers: I) -> Self
    where
        I: IntoIterator<Item = Arc<dyn RpcHandler>>,
    {
        self.extra_handlers.extend(handlers);
        self
    }

    /// Build the gateway.
    pub fn build(self) -> Gateway {
        let mut gateway = Gateway::new(self.config);
        gateway.auth_config = Arc::new(self.auth_config);
        if let Some(origins) = self.cors_origins {
            gateway.config.cors_origins = origins;
        }
        for handler in self.extra_handlers {
            gateway.register_handler(handler);
        }
        gateway
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_new() {
        let config = GatewayConfig::default();
        let gateway = Gateway::new(config);
        // The default gateway registers the expanded session/chat/config
        // handler sets (lifecycle, turns, compaction, attachments, routing…).
        assert!(gateway.rpc_registry.len() >= 40);
        for method in [
            "sessions.create",
            "sessions.activate",
            "sessions.compaction.trigger",
            "chat.send",
            "chat.message.update",
            "config.get",
            "config.routing.override",
        ] {
            assert!(
                gateway.rpc_registry.contains(method),
                "expected handler '{method}' to be registered"
            );
        }
    }

    #[test]
    fn test_gateway_builder_defaults() {
        let gateway = Gateway::builder(GatewayConfig::default()).build();
        assert!(gateway.rpc_registry.contains("sessions.create"));
        assert!(gateway.rpc_registry.contains("chat.send"));
        assert!(gateway.rpc_registry.contains("config.get"));
        assert_eq!(gateway.auth_config.mode, AuthMode::Open);
    }

    #[test]
    fn test_gateway_builder_registers_extra_handler() {
        let gateway = Gateway::builder(GatewayConfig::default())
            .handler(crate::rpc::rpc_handler("ping", |_| async {
                Ok(serde_json::json!({"pong": true}))
            }))
            .build();
        assert!(gateway.rpc_registry.contains("ping"));
    }

    #[test]
    fn test_gateway_builder_sets_auth_and_cors() {
        let auth = AuthConfig {
            mode: AuthMode::Token,
            token: Some("secret".into()),
            ..Default::default()
        };
        let gateway = Gateway::builder(GatewayConfig::default())
            .auth(auth)
            .cors(vec!["http://good.example".to_string()])
            .build();
        assert_eq!(gateway.auth_config.mode, AuthMode::Token);
        assert_eq!(gateway.auth_config.token.as_deref(), Some("secret"));
        assert_eq!(
            gateway.config.cors_origins,
            vec!["http://good.example".to_string()]
        );
    }
}
