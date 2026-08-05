//! Tauri error handling.
//!
//! Converts `opensquilla_core::Error` (and related error types) into a
//! serializable `TauriError` that can be returned from `#[tauri::command]`
//! functions. Tauri requires command error types to implement `serde::Serialize`,
//! so we wrap the error in a struct that carries a machine-readable code, a
//! human-readable message, optional details, and an HTTP-like status code mapped
//! from the error category.

use opensquilla_core::error::Error as CoreError;
use opensquilla_provider::ProviderError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A serializable error returned from Tauri command handlers.
///
/// The frontend receives this as the rejection payload of a failed
/// `invoke()` call. Field names are camelCase to match Vue/TypeScript
/// conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauriError {
    /// Machine-readable error code (e.g. `"CONFIG_ERROR"`, `"NOT_FOUND"`).
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// Optional structured details about the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// HTTP-like status code mapped from the error category.
    pub status: u16,
}

impl TauriError {
    /// Create a new `TauriError` with the given code, message, and status.
    pub fn new(code: impl Into<String>, message: impl Into<String>, status: u16) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
            status,
        }
    }

    /// Attach structured details to the error.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Create a 400 Bad Request error.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new("BAD_REQUEST", message, 400)
    }

    /// Create a 401 Unauthorized error.
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new("UNAUTHORIZED", message, 401)
    }

    /// Create a 403 Forbidden error.
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::new("FORBIDDEN", message, 403)
    }

    /// Create a 404 Not Found error.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new("NOT_FOUND", message, 404)
    }

    /// Create a 429 Too Many Requests error.
    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::new("RATE_LIMITED", format!("Rate limited, retry after {retry_after_secs}s"), 429)
    }

    /// Create a 500 Internal Server Error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("INTERNAL_ERROR", message, 500)
    }

    /// Create a 503 Service Unavailable error.
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new("SERVICE_UNAVAILABLE", message, 503)
    }
}

impl fmt::Display for TauriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({})", self.code, self.message, self.status)
    }
}

impl std::error::Error for TauriError {}

/// Map a `CoreError` to a `TauriError`, assigning HTTP-like status codes
/// based on the error category.
impl From<CoreError> for TauriError {
    fn from(err: CoreError) -> Self {
        let (code, status) = match &err {
            CoreError::Config(_) => ("CONFIG_ERROR", 400),
            CoreError::InvalidInput(_) => ("INVALID_INPUT", 400),
            CoreError::NotFound(_) => ("NOT_FOUND", 404),
            CoreError::Auth(_) => ("AUTH_ERROR", 401),
            CoreError::RateLimited(secs) => {
                return TauriError::rate_limited(*secs);
            }
            CoreError::Provider(_) => ("PROVIDER_ERROR", 502),
            CoreError::ToolExecution(_) => ("TOOL_ERROR", 500),
            CoreError::Session(_) => ("SESSION_ERROR", 400),
            CoreError::Channel(_) => ("CHANNEL_ERROR", 500),
            CoreError::Sandbox(_) => ("SANDBOX_ERROR", 500),
            CoreError::Io(_) => ("IO_ERROR", 500),
            CoreError::Serialization(_) => ("SERIALIZATION_ERROR", 400),
            CoreError::Database(_) => ("DATABASE_ERROR", 500),
            CoreError::Storage(_) => ("STORAGE_ERROR", 500),
            CoreError::Internal(_) => ("INTERNAL_ERROR", 500),
        };
        TauriError::new(code, err.to_string(), status)
    }
}

/// Map a `ProviderError` to a `TauriError`.
impl From<ProviderError> for TauriError {
    fn from(err: ProviderError) -> Self {
        let (code, status, message) = match &err {
            ProviderError::Auth(msg) => ("AUTH_ERROR", 401, msg.clone()),
            ProviderError::RateLimited(msg) => ("RATE_LIMITED", 429, msg.clone()),
            ProviderError::Timeout(msg) => ("TIMEOUT", 504, msg.clone()),
            ProviderError::Network(_) => ("NETWORK_ERROR", 502, err.to_string()),
            ProviderError::UnsupportedModel(model) => {
                ("UNSUPPORTED_MODEL", 400, format!("Unsupported model: {model}"))
            }
            ProviderError::Config(msg) => ("CONFIG_ERROR", 400, msg.clone()),
            ProviderError::Provider(msg) => ("PROVIDER_ERROR", 502, msg.clone()),
            ProviderError::Serialization(_) => ("SERIALIZATION_ERROR", 400, err.to_string()),
            ProviderError::Internal(msg) => ("INTERNAL_ERROR", 500, msg.clone()),
        };
        TauriError::new(code, message, status)
    }
}

/// Map a generic `std::io::Error` to a `TauriError`.
impl From<std::io::Error> for TauriError {
    fn from(err: std::io::Error) -> Self {
        TauriError::internal(format!("IO error: {err}"))
    }
}

/// Map `serde_json::Error` to a `TauriError`.
impl From<serde_json::Error> for TauriError {
    fn from(err: serde_json::Error) -> Self {
        TauriError::bad_request(format!("Serialization error: {err}"))
    }
}

/// Map `anyhow::Error` to a `TauriError`.
impl From<anyhow::Error> for TauriError {
    fn from(err: anyhow::Error) -> Self {
        TauriError::internal(err.to_string())
    }
}

/// A type alias for results returned from Tauri command handlers.
pub type TauriResult<T> = Result<T, TauriError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_error_mapping() {
        let err = CoreError::NotFound("session 123".to_string());
        let tauri_err: TauriError = err.into();
        assert_eq!(tauri_err.code, "NOT_FOUND");
        assert_eq!(tauri_err.status, 404);
    }

    #[test]
    fn test_config_error_maps_to_400() {
        let err = CoreError::Config("bad config".to_string());
        let tauri_err: TauriError = err.into();
        assert_eq!(tauri_err.code, "CONFIG_ERROR");
        assert_eq!(tauri_err.status, 400);
    }

    #[test]
    fn test_rate_limited_mapping() {
        let err = CoreError::RateLimited(30);
        let tauri_err: TauriError = err.into();
        assert_eq!(tauri_err.code, "RATE_LIMITED");
        assert_eq!(tauri_err.status, 429);
        assert!(tauri_err.message.contains("30"));
    }

    #[test]
    fn test_provider_error_mapping() {
        let err = ProviderError::Auth("invalid key".to_string());
        let tauri_err: TauriError = err.into();
        assert_eq!(tauri_err.code, "AUTH_ERROR");
        assert_eq!(tauri_err.status, 401);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let err = TauriError::not_found("test").with_details(serde_json::json!({"id": "abc"}));
        let json = serde_json::to_string(&err).unwrap();
        let parsed: TauriError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.code, "NOT_FOUND");
        assert_eq!(parsed.status, 404);
        assert!(parsed.details.is_some());
    }
}
