//! Deep-link (`opensquilla://`) handling: protocol registration, URL parsing,
//! and frontend dispatch.
//!
//! Replaces `desktop-deep-link.ts` and extends it with the session/import
//! actions the Tauri shell exposes. The recognized URL shapes are:
//!
//! - `opensquilla://open` — surface the main window.
//! - `opensquilla://session/{id}` — resume a session by id.
//! - `opensquilla://import/{data}` — import a session/artifact payload.
//!
//! Every URL is validated against the same contract as the Electron shell:
//! the scheme must be `opensquilla`, and credentials, ports, query strings,
//! hashes, and non-empty paths (other than the action's own path) are
//! rejected. This keeps the link surface intentionally tiny and predictable.
//!
//! Protocol registration is performed by `tauri-plugin-deep-link`; this
//! module owns the parsing and the event payload the frontend receives.

use serde::{Deserialize, Serialize};
use url::Url;

/// The deep-link URL scheme this app registers.
pub const DEEP_LINK_SCHEME: &str = "opensquilla";

/// Recognized deep-link actions (the `host` portion of the URL, plus the
/// `session`/`import` path-bearing actions).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DeepLinkAction {
    /// `opensquilla://open` — surface the main window.
    Open,
    /// `opensquilla://session/{id}` — resume a session.
    Session { id: String },
    /// `opensquilla://import/{data}` — import a payload.
    Import { data: String },
}

/// A payload emitted to the frontend when a deep link is dispatched.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepLinkPayload {
    /// The original URL.
    pub url: String,
    /// The parsed action.
    pub action: DeepLinkAction,
}

/// Errors produced while parsing a deep link.
#[derive(Debug, thiserror::Error)]
pub enum DeepLinkError {
    #[error("empty or non-string deep link")]
    NotAString,
    #[error("invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),
    #[error("unsupported scheme: expected {expected}, got {actual}")]
    WrongScheme { expected: String, actual: String },
    #[error("deep link must not contain credentials, port, query, or fragment")]
    ForbiddenComponent,
    #[error("deep link path is invalid for action {action}: {path}")]
    WrongPath { action: String, path: String },
    #[error("unsupported deep-link action: {0}")]
    UnknownAction(String),
    #[error("deep-link session id is empty")]
    EmptySessionId,
    #[error("deep-link import data is empty")]
    EmptyImportData,
}

/// Parse a raw deep-link URL string into a [`DeepLinkAction`].
///
/// # Examples
///
/// ```
/// use opensquilla_desktop_lib::deep_link::{parse_deep_link, DeepLinkAction};
/// assert_eq!(parse_deep_link("opensquilla://open"), Ok(DeepLinkAction::Open));
/// assert!(matches!(
///     parse_deep_link("opensquilla://session/abc-123"),
///     Ok(DeepLinkAction::Session { .. })
/// ));
/// assert!(parse_deep_link("https://open").is_err());
/// assert!(parse_deep_link("opensquilla://open?x=1").is_err());
/// ```
pub fn parse_deep_link(raw: &str) -> Result<DeepLinkAction, DeepLinkError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(DeepLinkError::NotAString);
    }

    let parsed = Url::parse(trimmed)?;

    let scheme = parsed.scheme();
    if scheme != DEEP_LINK_SCHEME {
        return Err(DeepLinkError::WrongScheme {
            expected: DEEP_LINK_SCHEME.to_string(),
            actual: scheme.to_string(),
        });
    }

    // Reject anything with credentials, port, query, or fragment.
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(DeepLinkError::ForbiddenComponent);
    }

    let host = parsed.host_str().unwrap_or("").to_lowercase();
    let path = parsed.path();

    match host.as_str() {
        "open" => {
            // `open` takes no path.
            if !path.is_empty() && path != "/" {
                return Err(DeepLinkError::WrongPath {
                    action: "open".into(),
                    path: path.to_string(),
                });
            }
            Ok(DeepLinkAction::Open)
        }
        "session" => {
            // `session/{id}`: path must be `/{id}` with a non-empty id.
            let id = path.strip_prefix('/').unwrap_or(path).trim();
            if id.is_empty() {
                return Err(DeepLinkError::EmptySessionId);
            }
            // Reject path traversal / extra segments.
            if id.contains('/') {
                return Err(DeepLinkError::WrongPath {
                    action: "session".into(),
                    path: path.to_string(),
                });
            }
            Ok(DeepLinkAction::Session { id: id.to_string() })
        }
        "import" => {
            // `import/{data}`: path must be `/{data}` with non-empty data.
            let data = path.strip_prefix('/').unwrap_or(path).trim();
            if data.is_empty() {
                return Err(DeepLinkError::EmptyImportData);
            }
            if data.contains('/') {
                return Err(DeepLinkError::WrongPath {
                    action: "import".into(),
                    path: path.to_string(),
                });
            }
            Ok(DeepLinkAction::Import {
                data: data.to_string(),
            })
        }
        other => Err(DeepLinkError::UnknownAction(other.to_string())),
    }
}

/// Filter command-line arguments, returning those that look like deep links.
///
/// Equivalent to `desktopDeepLinkArguments` in the Electron shell. Used by the
/// single-instance plugin to forward deep links from a second launch to the
/// running instance.
pub fn deep_link_arguments(argv: &[String]) -> Vec<String> {
    let prefix = format!("{DEEP_LINK_SCHEME}:");
    argv.iter()
        .filter_map(|v| {
            let t = v.trim();
            if t.to_lowercase().starts_with(&prefix) {
                Some(t.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Dispatch a parsed deep-link action to the frontend via a Tauri event.
///
/// Emits `deep-link://action` carrying a [`DeepLinkPayload`]. The main window
/// is activated first so the user sees the result of the link even if the
/// frontend is slow to mount.
pub fn dispatch<R: tauri::Runtime>(app: &tauri::AppHandle<R>, raw_url: &str, action: &DeepLinkAction) {
    use tauri::Emitter;
    crate::window::activate_main_window(app);
    let payload = DeepLinkPayload {
        url: raw_url.to_string(),
        action: action.clone(),
    };
    if let Err(e) = app.emit("deep-link://action", &payload) {
        tracing::warn!(error = %e, "could not emit deep-link action event");
    }
}

/// Parse and dispatch a raw deep-link URL in one call. Returns the parsed
/// action on success so the caller can log it.
pub fn parse_and_dispatch<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    raw_url: &str,
) -> Result<DeepLinkAction, DeepLinkError> {
    let action = parse_deep_link(raw_url)?;
    dispatch(app, raw_url, &action);
    Ok(action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_open_action() {
        assert_eq!(parse_deep_link("opensquilla://open"), Ok(DeepLinkAction::Open));
        assert_eq!(parse_deep_link("opensquilla://OPEN"), Ok(DeepLinkAction::Open));
    }

    #[test]
    fn parses_session_action() {
        let action = parse_deep_link("opensquilla://session/abc-123").unwrap();
        assert_eq!(action, DeepLinkAction::Session { id: "abc-123".into() });
    }

    #[test]
    fn parses_import_action() {
        let action = parse_deep_link("opensquilla://import/Zm9vYmFy").unwrap();
        assert_eq!(action, DeepLinkAction::Import { data: "Zm9vYmFy".into() });
    }

    #[test]
    fn rejects_other_schemes() {
        assert!(parse_deep_link("https://open").is_err());
        assert!(parse_deep_link("opensquilla2://open").is_err());
    }

    #[test]
    fn rejects_forbidden_components() {
        assert!(parse_deep_link("opensquilla://open?x=1").is_err());
        assert!(parse_deep_link("opensquilla://open#frag").is_err());
        assert!(parse_deep_link("opensquilla://user:pass@open").is_err());
        assert!(parse_deep_link("opensquilla://open:8080").is_err());
    }

    #[test]
    fn rejects_open_with_path() {
        assert!(parse_deep_link("opensquilla://open/foo").is_err());
    }

    #[test]
    fn rejects_empty_session_id() {
        assert!(parse_deep_link("opensquilla://session/").is_err());
        assert!(parse_deep_link("opensquilla://session").is_err());
    }

    #[test]
    fn rejects_multi_segment_session_path() {
        assert!(parse_deep_link("opensquilla://session/abc/def").is_err());
    }

    #[test]
    fn rejects_unknown_actions() {
        assert!(parse_deep_link("opensquilla://delete").is_err());
    }

    #[test]
    fn filters_argv_for_deep_links() {
        let argv = vec![
            "/usr/bin/opensquilla-desktop".to_string(),
            "opensquilla://open".to_string(),
            "--foo".to_string(),
            "OPENsquilla://session/abc".to_string(),
        ];
        let filtered = deep_link_arguments(&argv);
        assert_eq!(filtered.len(), 2);
    }
}
