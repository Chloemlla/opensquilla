//! RPC handler registry and dispatch system.
//!
//! Provides a registry of named RPC handlers that can be dispatched to based
//! on the method name received in a WebSocket request frame. The registry
//! supports standard JSON-RPC-style error codes, method discovery, and an
//! optional [`RpcContext`] passed to handlers that opt into context-aware
//! execution.

use async_trait::async_trait;
use opensquilla_core::error::{AppError, AppResult};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The error type returned by RPC handlers.
pub type RpcError = AppError;

/// A type-erased result for RPC handler responses.
pub type RpcResult = Result<Value, RpcError>;

/// Standard JSON-RPC-style error codes used by the gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcErrorCode {
    /// Invalid JSON was received by the server.
    ParseError = -32700,
    /// The JSON sent is not a valid request object.
    InvalidRequest = -32600,
    /// The method does not exist / is not available.
    MethodNotFound = -32601,
    /// Invalid method parameter(s).
    InvalidParams = -32602,
    /// Internal JSON-RPC error.
    InternalError = -32603,
    /// The caller is not authorized.
    Unauthorized = -32001,
    /// The request was rate limited.
    RateLimited = -32029,
    /// The requested resource was not found.
    NotFound = -32004,
}

impl RpcErrorCode {
    /// The numeric error code.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// A stable string identifier for the error code.
    pub fn as_str(self) -> &'static str {
        match self {
            RpcErrorCode::ParseError => "PARSE_ERROR",
            RpcErrorCode::InvalidRequest => "INVALID_REQUEST",
            RpcErrorCode::MethodNotFound => "METHOD_NOT_FOUND",
            RpcErrorCode::InvalidParams => "INVALID_PARAMS",
            RpcErrorCode::InternalError => "INTERNAL_ERROR",
            RpcErrorCode::Unauthorized => "UNAUTHORIZED",
            RpcErrorCode::RateLimited => "RATE_LIMITED",
            RpcErrorCode::NotFound => "NOT_FOUND",
        }
    }
}

/// The per-request context available to context-aware handlers.
///
/// Mirrors the Python `RpcContext` dataclass: connection identity, the
/// authenticated principal, and the request envelope.
#[derive(Debug, Clone)]
pub struct RpcContext {
    /// The WebSocket connection identifier handling this request.
    pub conn_id: String,
    /// The authenticated principal for the connection.
    pub principal: crate::auth::AuthPrincipal,
    /// The RPC method being invoked.
    pub method: String,
    /// The request identifier echoed back in the response.
    pub request_id: String,
    /// When the request was received.
    pub received_at: chrono::DateTime<chrono::Utc>,
}

impl RpcContext {
    /// Create a new RPC context.
    pub fn new(
        conn_id: impl Into<String>,
        principal: crate::auth::AuthPrincipal,
        method: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Self {
        Self {
            conn_id: conn_id.into(),
            principal,
            method: method.into(),
            request_id: request_id.into(),
            received_at: chrono::Utc::now(),
        }
    }
}

/// A registered RPC handler.
///
/// Implementations receive the deserialized parameter map and return a JSON
/// value or an error. Handlers that need the full request context (connection
/// identity, principal) may override [`RpcHandler::handle_with_ctx`]; the
/// default implementation delegates to [`RpcHandler::handle`].
#[async_trait]
pub trait RpcHandler: Send + Sync {
    /// Return the canonical method name this handler is registered under.
    fn name(&self) -> &str;

    /// Handle an incoming RPC call and produce a response.
    async fn handle(&self, params: Value) -> RpcResult;

    /// Handle an incoming RPC call with the full request context.
    ///
    /// The default implementation ignores the context and calls
    /// [`RpcHandler::handle`].
    async fn handle_with_ctx(&self, params: Value, _ctx: &RpcContext) -> RpcResult {
        self.handle(params).await
    }
}

/// A registry of named RPC handlers.
///
/// Handlers are stored as `Arc<dyn RpcHandler>` and looked up by method name
/// at dispatch time.
#[derive(Clone, Default)]
pub struct RpcRegistry {
    handlers: Arc<HashMap<String, Arc<dyn RpcHandler>>>,
}

impl RpcRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(HashMap::new()),
        }
    }

    /// Register a handler. Panics if a handler with the same name is already
    /// registered (duplicate registrations are a programming error).
    pub fn register(&mut self, handler: Arc<dyn RpcHandler>) {
        let name = handler.name().to_string();
        let handlers = Arc::make_mut(&mut self.handlers);
        assert!(
            !handlers.contains_key(&name),
            "RPC handler '{}' is already registered",
            name
        );
        handlers.insert(name, handler);
    }

    /// Register a handler, returning an error if the name is already taken
    /// rather than panicking.
    pub fn try_register(&mut self, handler: Arc<dyn RpcHandler>) -> AppResult<()> {
        let name = handler.name().to_string();
        let handlers = Arc::make_mut(&mut self.handlers);
        if handlers.contains_key(&name) {
            return Err(AppError::new(
                "DUPLICATE_HANDLER",
                format!("RPC handler '{name}' is already registered"),
            )
            .with_status(500));
        }
        handlers.insert(name, handler);
        Ok(())
    }

    /// Look up a handler by method name.
    pub fn get(&self, method: &str) -> Option<Arc<dyn RpcHandler>> {
        self.handlers.get(method).cloned()
    }

    /// Return `true` if a handler exists for the given method.
    pub fn contains(&self, method: &str) -> bool {
        self.handlers.contains_key(method)
    }

    /// Return the list of all registered method names.
    pub fn methods(&self) -> Vec<String> {
        self.handlers.keys().cloned().collect()
    }

    /// Return a description of all registered methods (name and target),
    /// sorted alphabetically for stable client-side discovery.
    pub fn describe(&self) -> Vec<MethodInfo> {
        let mut methods = self
            .handlers
            .iter()
            .map(|(name, handler)| MethodInfo {
                name: name.clone(),
                target: handler.name().to_string(),
            })
            .collect::<Vec<_>>();
        methods.sort_by(|a, b| a.name.cmp(&b.name));
        methods
    }

    /// Dispatch a method call. Returns `None` if no handler is registered.
    pub async fn dispatch(&self, method: &str, params: Value) -> Option<RpcResult> {
        self.dispatch_impl(method, params, None).await
    }

    /// Dispatch a method call with the request context.
    ///
    /// Context-aware handlers receive `ctx`; the rest fall back to
    /// [`RpcHandler::handle`]. Returns `None` if no handler is registered.
    pub async fn dispatch_with_ctx(
        &self,
        method: &str,
        params: Value,
        ctx: &RpcContext,
    ) -> Option<RpcResult> {
        self.dispatch_impl(method, params, Some(ctx)).await
    }

    /// Dispatch a method call, optionally providing context. Returns a
    /// [`RpcErrorCode::MethodNotFound`] error when the method is missing and
    /// the caller wants a structured error instead of `None`.
    pub async fn dispatch_or_error(&self, method: &str, params: Value) -> RpcResult {
        match self.dispatch(method, params).await {
            Some(result) => result,
            None => Err(AppError::new(
                RpcErrorCode::MethodNotFound.as_str(),
                format!("Method not found: {method}"),
            )
            .with_status(404)),
        }
    }

    async fn dispatch_impl(
        &self,
        method: &str,
        params: Value,
        ctx: Option<&RpcContext>,
    ) -> Option<RpcResult> {
        let handler = self.get(method)?;
        let result = match ctx {
            Some(ctx) => handler.handle_with_ctx(params, ctx).await,
            None => handler.handle(params).await,
        };
        Some(result)
    }

    /// Return the number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Return `true` if no handlers are registered.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

/// A description of a registered RPC method, used for method discovery.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MethodInfo {
    /// The registered method name.
    pub name: String,
    /// The handler's canonical name.
    pub target: String,
}

// ---------------------------------------------------------------------------
// Helper: wrap a closure as an RPC handler
// ---------------------------------------------------------------------------

/// Create an `RpcHandler` from a name and an async function.
///
/// The function receives `(Value,)` and returns `RpcResult`.
pub fn rpc_handler<F, Fut>(name: &'static str, f: F) -> Arc<dyn RpcHandler>
where
    F: Fn(Value) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = RpcResult> + Send + 'static,
{
    struct Handler<F> {
        name: &'static str,
        f: F,
    }

    // Safety: F is Send + Sync, and the closure f returns a boxed future.
    // We use a separate dispatch function that boxes the future to avoid
    // the PhantomData<Fut> Sync issue.
    unsafe impl<F: Send + Sync> Sync for Handler<F> {}

    #[async_trait]
    impl<F, Fut> RpcHandler for Handler<F>
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = RpcResult> + Send + 'static,
    {
        fn name(&self) -> &str {
            self.name
        }

        async fn handle(&self, params: Value) -> RpcResult {
            (self.f)(params).await
        }
    }

    Arc::new(Handler { name, f })
}

/// Create an `RpcHandler` from a name and an async function that also
/// receives the request [`RpcContext`].
pub fn rpc_handler_with_ctx<F, Fut>(name: &'static str, f: F) -> Arc<dyn RpcHandler>
where
    F: Fn(Value, &RpcContext) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = RpcResult> + Send + 'static,
{
    struct Handler<F> {
        name: &'static str,
        f: F,
    }

    unsafe impl<F: Send + Sync> Sync for Handler<F> {}

    #[async_trait]
    impl<F, Fut> RpcHandler for Handler<F>
    where
        F: Fn(Value, &RpcContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = RpcResult> + Send + 'static,
    {
        fn name(&self) -> &str {
            self.name
        }

        async fn handle(&self, params: Value) -> RpcResult {
            // Without a context, use a minimal anonymous context.
            let ctx = RpcContext::new(
                "",
                crate::auth::AuthPrincipal::new(
                    "operator",
                    crate::auth::CLI_DEFAULT_OPERATOR_SCOPES,
                    false,
                    false,
                ),
                "",
                "",
            );
            (self.f)(params, &ctx).await
        }

        async fn handle_with_ctx(&self, params: Value, ctx: &RpcContext) -> RpcResult {
            (self.f)(params, ctx).await
        }
    }

    Arc::new(Handler { name, f })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_register_and_dispatch() {
        let mut registry = RpcRegistry::new();
        registry.register(rpc_handler("ping", |_params| async {
            Ok(serde_json::json!({"pong": true}))
        }));

        assert!(registry.contains("ping"));
        assert_eq!(registry.len(), 1);

        let result = registry.dispatch("ping", Value::Null).await;
        assert!(result.is_some());
        let response = result.unwrap().unwrap();
        assert_eq!(response, serde_json::json!({"pong": true}));
    }

    #[tokio::test]
    async fn test_dispatch_missing_handler() {
        let registry = RpcRegistry::new();
        let result = registry.dispatch("unknown", Value::Null).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_dispatch_or_error_returns_method_not_found() {
        let registry = RpcRegistry::new();
        let result = registry.dispatch_or_error("nope", Value::Null).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, RpcErrorCode::MethodNotFound.as_str());
        assert_eq!(err.status, 404);
    }

    #[tokio::test]
    async fn test_dispatch_with_ctx_reaches_ctx_handler() {
        let mut registry = RpcRegistry::new();
        registry.register(rpc_handler_with_ctx("echo", |params, ctx| async move {
            Ok(serde_json::json!({
                "conn_id": ctx.conn_id,
                "method": ctx.method,
                "params": params,
            }))
        }));

        let ctx = RpcContext::new(
            "conn-1",
            crate::auth::AuthPrincipal::new(
                "operator",
                crate::auth::CLI_DEFAULT_OPERATOR_SCOPES,
                true,
                false,
            ),
            "echo",
            "r1",
        );
        let result = registry
            .dispatch_with_ctx("echo", serde_json::json!({"x": 1}), &ctx)
            .await;
        let response = result.unwrap().unwrap();
        assert_eq!(response["conn_id"], "conn-1");
        assert_eq!(response["method"], "echo");
        assert_eq!(response["params"]["x"], 1);
    }

    #[tokio::test]
    async fn test_plain_handler_falls_back_without_ctx() {
        let mut registry = RpcRegistry::new();
        registry.register(rpc_handler("hello", |_| async {
            Ok(serde_json::json!({"greeting": "hi"}))
        }));

        let ctx = RpcContext::new(
            "conn-1",
            crate::auth::AuthPrincipal::new(
                "operator",
                crate::auth::CLI_DEFAULT_OPERATOR_SCOPES,
                true,
                false,
            ),
            "hello",
            "r1",
        );
        let result = registry.dispatch_with_ctx("hello", Value::Null, &ctx).await;
        assert!(result.is_some());
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn test_try_register_duplicate() {
        let mut registry = RpcRegistry::new();
        registry.register(rpc_handler("dup", |_| async { Ok(Value::Null) }));
        let second = registry.try_register(rpc_handler("dup", |_| async { Ok(Value::Null) }));
        assert!(second.is_err());
    }

    #[test]
    fn test_describe_sorts_methods() {
        let mut registry = RpcRegistry::new();
        registry.register(rpc_handler("zeta", |_| async { Ok(Value::Null) }));
        registry.register(rpc_handler("alpha", |_| async { Ok(Value::Null) }));
        let methods = registry.describe();
        assert_eq!(methods.len(), 2);
        assert_eq!(methods[0].name, "alpha");
        assert_eq!(methods[1].name, "zeta");
    }

    #[test]
    fn test_error_code_mapping() {
        assert_eq!(RpcErrorCode::ParseError.code(), -32700);
        assert_eq!(RpcErrorCode::MethodNotFound.code(), -32601);
        assert_eq!(RpcErrorCode::MethodNotFound.as_str(), "METHOD_NOT_FOUND");
        assert_eq!(RpcErrorCode::Unauthorized.as_str(), "UNAUTHORIZED");
    }
}
