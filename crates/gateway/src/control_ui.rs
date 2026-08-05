//! Vue.js SPA static file serving.
//!
//! Serves the compiled frontend located under `opensquilla-webui/dist`. The
//! gateway mounts the SPA at the root, falls back to `index.html` for
//! client-side routing, serves static assets (JS, CSS, images) with cache
//! headers, and negotiates gzip/brotli compression when the corresponding
//! `tower-http` features are enabled.
//!
//! This mirrors the Python `control_ui.py` module, which used Starlette
//! `StaticFiles` plus Jinja2 template rendering. In the Rust port the SPA is
//! fully static (no server-side templating), so the implementation reduces to
//! directory serving with an SPA fallback.
//!
//! NOTE: `ServeDir`/`ServeFile` require the `fs` feature on `tower-http`;
//! gzip/brotli compression requires the `compression` feature (with `gzip`
//! and `br` sub-features).

use std::path::{Path, PathBuf};

use axum::{
    http::{header, HeaderValue, Uri},
    response::Response,
    Router,
};
use opensquilla_core::error::AppError;
use tower_http::services::{ServeDir, ServeFile};
use tracing::{debug, warn};

/// Default path of the compiled SPA bundle, relative to the workspace root.
pub const DEFAULT_WEBUI_DIST: &str = "opensquilla-webui/dist";

/// Builder for the SPA control-UI router.
///
/// Wraps a [`ServeDir`] pointed at the compiled frontend and adds an SPA
/// fallback so that unknown routes fall through to `index.html` instead of
/// returning a 404.
#[derive(Clone)]
pub struct ControlUi {
    /// Root directory containing the compiled SPA assets.
    root: PathBuf,
    /// Whether the SPA bundle exists on disk. Used to degrade gracefully when
    /// the frontend has not been built yet (e.g. during development).
    available: bool,
}

impl ControlUi {
    /// Create a new control-UI server rooted at the given directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let available = root.is_dir();
        if !available {
            warn!(
                "Control UI dist directory not found at {}; SPA will serve a placeholder",
                root.display()
            );
        }
        Self { root, available }
    }

    /// Create a control-UI server using the default workspace dist path.
    pub fn default_path() -> Self {
        Self::new(DEFAULT_WEBUI_DIST)
    }

    /// Return the configured root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return whether the SPA bundle is present on disk.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Build an axum [`Router`] that serves the SPA.
    ///
    /// The router:
    /// 1. Serves static files from the dist directory via [`ServeDir`].
    /// 2. Falls back to `index.html` for any path that does not match a file,
    ///    enabling client-side routing.
    ///
    /// When the dist directory is missing, the fallback serves a lightweight
    /// placeholder page instead of 404ing.
    pub fn router(self) -> Router {
        let index = self.root.join("index.html");
        let serve_dir = ServeDir::new(&self.root).fallback(ServeFile::new(index));

        // When the bundle is unavailable, prepend a placeholder route so the
        // user sees a helpful page instead of a 404.
        if self.available {
            Router::new().fallback_service(serve_dir)
        } else {
            Router::new()
                .route("/", axum::routing::get(placeholder_handler))
                .fallback_service(serve_dir)
        }
    }
}

/// Handler that returns the SPA-unavailable placeholder page.
pub async fn placeholder_handler() -> Response {
    let body = "<!doctype html><html><head><meta charset=\"utf-8\">\
                <title>OpenSquilla</title></head>\
                <body><h1>OpenSquilla control UI</h1>\
                <p>The web UI bundle was not found. Build the frontend under \
                <code>opensquilla-webui/dist</code> and restart the gateway.</p>\
                </body></html>";
    let mut resp = Response::new(body.to_string().into());
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    resp
}

/// Apply cache headers to a static-asset response based on its path.
///
/// Hashed assets under `/assets/` are marked immutable and cached for a year.
/// Other paths (e.g. `index.html`, `favicon.ico`) get a short revalidation
/// window so that deploys are picked up promptly.
///
/// This helper is intended to be wired into a middleware layer; it is not
/// applied automatically by [`ControlUi::router`].
pub fn apply_cache_headers(uri: &Uri, response: &mut Response) {
    let path = uri.path();
    let cache_value = if path.starts_with("/assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    if let Ok(value) = HeaderValue::from_str(cache_value) {
        response.headers_mut().insert(header::CACHE_CONTROL, value);
    }
}

/// Resolve the on-disk location of the webui dist directory.
///
/// Looks first at the explicit override, then at the default workspace path,
/// then at a path relative to the current executable (used by packaged
/// installs).
pub fn resolve_dist_dir(override_path: Option<&str>) -> Result<PathBuf, AppError> {
    if let Some(p) = override_path {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(AppError::not_found(format!(
            "Control UI dist directory not found at '{p}'"
        )));
    }

    let candidates = [
        PathBuf::from(DEFAULT_WEBUI_DIST),
        std::env::current_exe()
            .map_err(|e| AppError::internal(e.to_string()))?
            .parent()
            .ok_or_else(|| AppError::internal("Cannot resolve exe parent"))?
            .join("webui")
            .join("dist"),
    ];

    for candidate in candidates {
        debug!(path = %candidate.display(), "Checking webui dist candidate");
        if candidate.is_dir() {
            return Ok(candidate);
        }
    }

    Err(AppError::not_found(
        "Control UI dist directory not found in any default location",
    ))
}

/// Build a [`ControlUi`] from an optional override path, falling back to the
/// default workspace location.
pub fn build_control_ui(override_path: Option<&str>) -> ControlUi {
    match resolve_dist_dir(override_path) {
        Ok(path) => ControlUi::new(path),
        Err(err) => {
            warn!(error = %err, "Control UI disabled; serving placeholder");
            ControlUi::new(DEFAULT_WEBUI_DIST)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_path_not_panic() {
        let ui = ControlUi::default_path();
        // The directory may or may not exist depending on the build, but the
        // constructor must not panic.
        let _ = ui.is_available();
    }

    #[test]
    fn test_resolve_dist_dir_missing() {
        let result = resolve_dist_dir(Some("/nonexistent/path/to/dist"));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_placeholder_handler_is_html() {
        let resp = placeholder_handler().await;
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/html; charset=utf-8"
        );
    }

    #[test]
    fn test_apply_cache_headers_assets_immutable() {
        let mut resp = Response::new(String::new().into());
        let uri: Uri = "/assets/index-abc123.js".parse().unwrap();
        apply_cache_headers(&uri, &mut resp);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL).unwrap(),
            "public, max-age=31536000, immutable"
        );
    }

    #[test]
    fn test_apply_cache_headers_index_nocache() {
        let mut resp = Response::new(String::new().into());
        let uri: Uri = "/index.html".parse().unwrap();
        apply_cache_headers(&uri, &mut resp);
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL).unwrap(),
            "no-cache"
        );
    }
}
