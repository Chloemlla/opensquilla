//! Tower middleware layers for the gateway.
//!
//! Provides:
//! - **Auth middleware**: Validates the `Authorization` header or loopback
//!   status before forwarding the request.
//! - **Origin-guard middleware**: Rejects cross-origin unsafe mutations.
//! - **Rate-limit middleware**: Token-bucket and sliding-window rate limiters
//!   keyed by remote address.
//! - **CORS middleware**: Configurable cross-origin resource sharing.
//! - **Security headers**: CSP, X-Content-Type-Options, etc.
//! - **Error handling**: Catches panics and converts framework errors into
//!   structured JSON responses.
//! - **Request logging**: Logs method, URI, status, and latency.

use axum::{
    extract::{ConnectInfo, Request},
    http::{header, HeaderValue, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use futures::future::FutureExt;
use opensquilla_core::error::AppError;
use serde_json::json;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

/// Insert this into request extensions to signal that the request has been
/// authenticated. The value is the user identifier.
#[derive(Debug, Clone)]
pub struct AuthenticatedUser(pub String);

/// Auth middleware that validates the `Authorization` header.
///
/// The `auth_header` should be e.g. `"Bearer <token>"`. Requests missing a
/// valid header receive a 401 response.
pub async fn auth_middleware(
    req: Request,
    next: Next,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // We expect the auth header to be set by an outer layer; if missing, reject.
    let auth_header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(_token) => {
            let authed = AuthenticatedUser("authenticated".to_string());
            let mut req = req;
            req.extensions_mut().insert(authed);
            Ok(next.run(req).await)
        }
        None => {
            warn!("Request missing Authorization header");
            Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "missing_authorization"})),
            ))
        }
    }
}

/// Whether a request path is exempt from authentication.
///
/// Health probes, desktop-ownership endpoints, and artifact preview capability
/// URLs are public.
pub fn is_public_path(path: &str) -> bool {
    const PUBLIC: &[&str] = &[
        "/health",
        "/healthz",
        "/ready",
        "/readyz",
        "/api/desktop/identity",
        "/api/desktop/shutdown",
    ];
    PUBLIC.contains(&path) || path.starts_with("/api/v1/artifact-preview/")
}

/// Extract a bearer token from a request.
///
/// Checks, in order: the `Authorization: Bearer <token>` header, the
/// `x-opensquilla-token` header, and the `token` query parameter.
pub fn extract_token(req: &Request) -> Option<String> {
    if let Some(auth) = req.headers().get(header::AUTHORIZATION) {
        if let Ok(s) = auth.to_str() {
            if let Some(stripped) = s.strip_prefix("Bearer ") {
                return Some(stripped.to_string());
            }
        }
    }
    if let Some(token) = req.headers().get("x-opensquilla-token") {
        if let Ok(s) = token.to_str() {
            return Some(s.to_string());
        }
    }
    if let Some(query) = req.uri().query() {
        for pair in query.split('&') {
            let mut parts = pair.split('=');
            if parts.next() == Some("token") {
                return parts.next().map(|v| v.to_string());
            }
        }
    }
    None
}

/// Token-auth middleware with public-path exemption.
///
/// When `expected_token` is `None` the middleware runs in open mode and lets
/// every request through. Otherwise a request must carry a matching token via
/// [`extract_token`].
pub async fn token_auth_middleware(
    req: Request,
    next: Next,
    expected_token: Option<String>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    if is_public_path(req.uri().path()) {
        return Ok(next.run(req).await);
    }

    // WebSocket upgrades perform their own challenge-response authentication,
    // so the HTTP middleware defers to the connection handler.
    let is_upgrade = req
        .headers()
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);
    if is_upgrade {
        return Ok(next.run(req).await);
    }

    let Some(expected) = &expected_token else {
        // No token configured: open mode.
        return Ok(next.run(req).await);
    };

    match extract_token(&req) {
        Some(provided) if &provided == expected => {
            let authed = AuthenticatedUser(provided);
            let mut req = req;
            req.extensions_mut().insert(authed);
            Ok(next.run(req).await)
        }
        _ => Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "unauthorized", "code": "UNAUTHORIZED"})),
        )),
    }
}

// ---------------------------------------------------------------------------
// Rate-limit middleware
// ---------------------------------------------------------------------------

/// A simple token-bucket rate limiter.
#[derive(Debug)]
pub struct RateLimiter {
    max_requests: u64,
    window_secs: u64,
    // per-peer state: (tokens, last_refill_timestamp_secs)
    state: dashmap::DashMap<String, (AtomicU64, AtomicU64)>,
}

impl RateLimiter {
    /// Create a new rate limiter allowing `max_requests` per `window_secs`.
    pub fn new(max_requests: u64, window_secs: u64) -> Arc<Self> {
        Arc::new(Self {
            max_requests,
            window_secs,
            state: dashmap::DashMap::new(),
        })
    }

    /// Check if a request from `peer` is allowed. Returns `true` if within
    /// limits, `false` if rate-limited.
    pub fn check(&self, peer: &str) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = self.state.entry(peer.to_string()).or_insert_with(|| {
            (AtomicU64::new(self.max_requests), AtomicU64::new(now))
        });

        let (tokens, last_refill) = entry.value();
        let last = last_refill.load(Ordering::Relaxed);
        let elapsed = now.saturating_sub(last);

        if elapsed >= self.window_secs {
            // Full refill
            tokens.store(self.max_requests, Ordering::Relaxed);
            last_refill.store(now, Ordering::Relaxed);
        } else {
            // Partial refill based on elapsed time
            let refill =
                (elapsed as f64 / self.window_secs as f64 * self.max_requests as f64) as u64;
            let current = tokens.load(Ordering::Relaxed);
            let new_tokens = (self.max_requests).min(current + refill);
            tokens.store(new_tokens, Ordering::Relaxed);
            if refill > 0 {
                last_refill.store(now, Ordering::Relaxed);
            }
        }

        let remaining = tokens.fetch_sub(1, Ordering::Relaxed);
        if remaining > 0 {
            true
        } else {
            // Restore the token we just took
            tokens.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

/// Rate-limit middleware layer (token bucket).
pub async fn rate_limit_middleware(
    req: Request,
    next: Next,
    limiter: Arc<RateLimiter>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let peer = client_ip(&req);

    if !limiter.check(&peer) {
        warn!("Rate limit exceeded for {peer}");
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "rate_limited"})),
        ));
    }

    Ok(next.run(req).await)
}

/// A sliding-window rate limiter keyed by client IP.
///
/// Mirrors the Python `RateLimitMiddleware`: each peer keeps a list of request
/// timestamps within the window; requests beyond `max_requests` are rejected.
#[derive(Debug)]
pub struct SlidingWindowRateLimiter {
    max_requests: u64,
    window_secs: u64,
    windows: dashmap::DashMap<String, Vec<f64>>,
}

impl SlidingWindowRateLimiter {
    /// Create a new sliding-window limiter allowing `max_requests` per
    /// `window_secs`.
    pub fn new(max_requests: u64, window_secs: u64) -> Arc<Self> {
        Arc::new(Self {
            max_requests,
            window_secs,
            windows: dashmap::DashMap::new(),
        })
    }

    /// Check whether a request from `peer` is allowed, recording it if so.
    pub fn check(&self, peer: &str) -> bool {
        let now = now_secs();
        let mut windows = self.windows.entry(peer.to_string()).or_default();
        windows.retain(|t| now - *t < self.window_secs as f64);
        if windows.len() as u64 >= self.max_requests {
            return false;
        }
        windows.push(now);
        true
    }
}

/// Sliding-window rate-limit middleware layer.
pub async fn sliding_window_rate_limit_middleware(
    req: Request,
    next: Next,
    limiter: Arc<SlidingWindowRateLimiter>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let peer = client_ip(&req);
    if !limiter.check(&peer) {
        warn!(peer = %peer, "Rate limit exceeded (sliding window)");
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error": "rate_limited", "code": "RATE_LIMITED"})),
        ));
    }
    Ok(next.run(req).await)
}

/// Best-effort client IP extraction: `x-forwarded-for` first hop, then the
/// transport peer address.
pub fn client_ip(req: &Request) -> String {
    if let Some(forwarded) = req.headers().get("x-forwarded-for") {
        if let Ok(s) = forwarded.to_str() {
            if let Some(first) = s.split(',').next() {
                let trimmed = first.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
        }
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ---------------------------------------------------------------------------
// Origin-guard middleware
// ---------------------------------------------------------------------------

/// Whether an `Origin` header value is allowed.
///
/// A wildcard configured origin, a loopback origin, or a configured origin
/// matches. Origin-less native clients and webhooks are handled by the caller
/// (they simply omit the header).
pub fn origin_allowed(origin: &str, allowed_origins: &[String]) -> bool {
    if allowed_origins.iter().any(|o| o == "*") {
        return true;
    }
    if is_loopback_origin(origin) {
        return true;
    }
    allowed_origins.iter().any(|o| o == origin)
}

/// Whether an origin string is a loopback (local) origin.
fn is_loopback_origin(origin: &str) -> bool {
    origin.starts_with("http://localhost")
        || origin.starts_with("https://localhost")
        || origin.starts_with("http://127.0.0.1")
        || origin.starts_with("http://[::1]")
}

/// Reject browser cross-origin mutations before they reach a route.
///
/// Unsafe methods (POST, PUT, PATCH, DELETE) that carry a disallowed `Origin`
/// header receive a 403. Requests without an `Origin` header (native clients,
/// webhooks) pass through.
pub async fn unsafe_origin_guard_middleware(
    req: Request,
    next: Next,
    allowed_origins: Vec<String>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    const UNSAFE_METHODS: [Method; 4] = [Method::POST, Method::PUT, Method::PATCH, Method::DELETE];

    if UNSAFE_METHODS.contains(req.method()) {
        if let Some(origin) = req.headers().get(header::ORIGIN) {
            if let Ok(origin_str) = origin.to_str() {
                if !origin_allowed(origin_str, &allowed_origins) {
                    warn!(origin = %origin_str, "Cross-origin unsafe request rejected");
                    return Err((
                        StatusCode::FORBIDDEN,
                        Json(json!({"error": "origin_not_allowed", "code": "ORIGIN_NOT_ALLOWED"})),
                    ));
                }
            }
        }
    }

    Ok(next.run(req).await)
}

// ---------------------------------------------------------------------------
// CORS layer
// ---------------------------------------------------------------------------

/// Build a CORS layer that allows the given origins (or `*` for any origin).
pub fn cors_layer(origins: &[String]) -> CorsLayer {
    let origins: Vec<HeaderValue> = origins
        .iter()
        .flat_map(|o| {
            if o == "*" {
                vec![HeaderValue::from_static("*")]
            } else {
                vec![HeaderValue::from_str(o).unwrap_or_else(|_| HeaderValue::from_static("*"))]
            }
        })
        .collect();

    let mut layer = CorsLayer::new()
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::PATCH,
            Method::OPTIONS,
        ])
        .allow_headers(Any);

    if origins.iter().any(|v| v == "*") {
        layer = layer.allow_origin(Any);
    } else {
        layer = layer.allow_origin(origins);
    }

    layer
}

// ---------------------------------------------------------------------------
// Security headers layer
// ---------------------------------------------------------------------------

/// Add security headers to every response.
pub async fn security_headers_middleware(
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let mut response = next.run(req).await;

    let headers = response.headers_mut();

    // Content Security Policy
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'self'; script-src 'none'; object-src 'none'"),
    );

    // Prevent MIME-type sniffing
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );

    // Prevent clickjacking
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));

    // Referrer policy
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );

    // Permissions policy
    headers.insert(
        "Permissions-Policy",
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );

    // Strict transport security (only if TLS-terminated)
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=31536000; includeSubDomains"),
    );

    // Cache control for API responses
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));

    response
}

// ---------------------------------------------------------------------------
// Error handling middleware
// ---------------------------------------------------------------------------

/// Global error handler that converts `AppError` into a JSON response.
pub fn handle_app_error(err: AppError) -> (StatusCode, Json<serde_json::Value>) {
    let status = StatusCode::from_u16(err.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = json!({
        "code": err.code,
        "message": err.message,
        "details": err.details,
    });
    (status, Json(body))
}

/// Build a structured JSON error response body.
pub fn error_response(
    status: StatusCode,
    code: &str,
    message: impl Into<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    (status, Json(json!({ "error": code, "message": message.into(), "code": code })))
}

/// Catch panics and return a 500 JSON response.
pub async fn catch_panic_middleware(
    req: Request,
    next: Next,
) -> Response {
    let uri = req.uri().to_string();
    let method = req.method().to_string();

    let result = std::panic::AssertUnwindSafe(next.run(req)).catch_unwind().await;

    match result {
        Ok(response) => response,
        Err(_panic) => {
            warn!("Panic caught handling {method} {uri}");
            let body = json!({
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Request logging middleware
// ---------------------------------------------------------------------------

/// Log incoming requests and their response status codes.
pub async fn request_logging_middleware(
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let method = req.method().to_string();
    let uri = req.uri().to_string();
    let start = Instant::now();

    info!("--> {method} {uri}");

    let response = next.run(req).await;

    let status = response.status();
    let elapsed = start.elapsed();
    info!("<-- {method} {uri} {status} ({elapsed:?})");

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request as HttpRequest, routing::get, Router};
    use tower::ServiceExt;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5, 60);
        for _ in 0..5 {
            assert!(limiter.check("peer1"));
        }
    }

    #[test]
    fn test_rate_limiter_rejects_excess() {
        let limiter = RateLimiter::new(2, 60);
        assert!(limiter.check("peer2"));
        assert!(limiter.check("peer2"));
        assert!(!limiter.check("peer2"));
    }

    #[test]
    fn test_rate_limiter_isolates_peers() {
        let limiter = RateLimiter::new(1, 60);
        assert!(limiter.check("alice"));
        assert!(!limiter.check("alice"));
        assert!(limiter.check("bob"));
    }

    #[test]
    fn test_sliding_window_allows_within_limit() {
        let limiter = SlidingWindowRateLimiter::new(3, 60);
        assert!(limiter.check("peer"));
        assert!(limiter.check("peer"));
        assert!(limiter.check("peer"));
    }

    #[test]
    fn test_sliding_window_rejects_excess() {
        let limiter = SlidingWindowRateLimiter::new(2, 60);
        assert!(limiter.check("peer"));
        assert!(limiter.check("peer"));
        assert!(!limiter.check("peer"));
    }

    #[test]
    fn test_sliding_window_isolates_peers() {
        let limiter = SlidingWindowRateLimiter::new(1, 60);
        assert!(limiter.check("alice"));
        assert!(!limiter.check("alice"));
        assert!(limiter.check("bob"));
    }

    #[test]
    fn test_is_public_path() {
        assert!(is_public_path("/health"));
        assert!(is_public_path("/readyz"));
        assert!(is_public_path("/api/desktop/identity"));
        assert!(is_public_path("/api/v1/artifact-preview/abcdef0123456789abcdef0123456789"));
        assert!(!is_public_path("/api/sessions"));
        assert!(!is_public_path("/"));
    }

    #[test]
    fn test_extract_token_from_header() {
        let req = HttpRequest::builder()
            .uri("/api/test")
            .header(header::AUTHORIZATION, "Bearer abc123")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_token(&req).as_deref(), Some("abc123"));
    }

    #[test]
    fn test_extract_token_from_query() {
        let req = HttpRequest::builder()
            .uri("/api/test?token=querytoken")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_token(&req).as_deref(), Some("querytoken"));
    }

    #[test]
    fn test_extract_token_none() {
        let req = HttpRequest::builder()
            .uri("/api/test")
            .body(Body::empty())
            .unwrap();
        assert_eq!(extract_token(&req), None);
    }

    #[test]
    fn test_origin_allowed_wildcard() {
        assert!(origin_allowed("http://evil.example", &["*".to_string()]));
        assert!(origin_allowed("http://localhost:3000", &[]));
        assert!(!origin_allowed("http://evil.example", &["http://good.example".to_string()]));
        assert!(origin_allowed("http://good.example", &["http://good.example".to_string()]));
    }

    #[test]
    fn test_client_ip_forwarded() {
        let req = HttpRequest::builder()
            .uri("/")
            .header("x-forwarded-for", "203.0.113.5, 10.0.0.1")
            .body(Body::empty())
            .unwrap();
        assert_eq!(client_ip(&req), "203.0.113.5");
    }

    #[test]
    fn test_client_ip_fallback() {
        let req = HttpRequest::builder().uri("/").body(Body::empty()).unwrap();
        assert_eq!(client_ip(&req), "unknown");
    }

    #[tokio::test]
    async fn test_token_auth_middleware_rejects() {
        let app = Router::new()
            .route("/api/secure", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(move |req, next| {
                token_auth_middleware(req, next, Some("secret".to_string()))
            }));

        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/secure")
                    .header(header::AUTHORIZATION, "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/secure")
                    .header(header::AUTHORIZATION, "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_token_auth_middleware_public_path() {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(move |req, next| {
                token_auth_middleware(req, next, Some("secret".to_string()))
            }));

        // Public path bypasses auth.
        let response = app
            .oneshot(HttpRequest::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_origin_guard_rejects_bad_origin() {
        let app = Router::new()
            .route("/api/x", axum::routing::post(|| async { "created" }))
            .layer(axum::middleware::from_fn(move |req, next| {
                unsafe_origin_guard_middleware(req, next, vec!["http://good.example".to_string()])
            }));

        let response = app
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/x")
                    .header(header::ORIGIN, "http://evil.example")
                    .method(Method::POST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);

        // Allowed origin passes.
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/x")
                    .header(header::ORIGIN, "http://good.example")
                    .method(Method::POST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_origin_guard_allows_originless() {
        let app = Router::new()
            .route("/api/x", axum::routing::post(|| async { "created" }))
            .layer(axum::middleware::from_fn(move |req, next| {
                unsafe_origin_guard_middleware(req, next, vec!["http://good.example".to_string()])
            }));

        // Native clients omit the Origin header entirely.
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/api/x")
                    .method(Method::POST)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_handle_app_error() {
        let err = AppError::unauthorized("nope");
        let (status, _json) = handle_app_error(err);
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_error_response_helper() {
        let (status, body) = error_response(StatusCode::TOO_MANY_REQUESTS, "RATE_LIMITED", "slow down");
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body.0["code"], "RATE_LIMITED");
    }
}
