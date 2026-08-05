//! Error classification, recovery actions, and circuit breaker.
//!
//! Provider errors come back in many shapes: HTTP status codes, provider JSON
//! error bodies, network timeouts, and stream mid-flight failures. This module
//! classifies any error into a [`ErrorCategory`] with an associated
//! [`RecoveryAction`], and tracks per-provider failure rates via a circuit
//! breaker so a flaky provider can be tripped open to fail fast instead of
//! queuing more doomed requests.
//!
//! The classification is table-driven (pattern matching on status code and
//! error-body substrings) so new provider error shapes can be added without
//! touching call sites.

use crate::types::ProviderError;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Error categories
// ---------------------------------------------------------------------------

/// High-level classification of a provider error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Invalid API key / expired token / forbidden (401, 403).
    Authentication,
    /// Rate limit exceeded (429).
    RateLimit,
    /// Bad request — malformed payload or unsupported params (400).
    BadRequest,
    /// Request too large (413).
    RequestTooLarge,
    /// Model not found / not supported (404).
    ModelNotFound,
    /// Provider server error (500, 502, 503, 504).
    ServerError,
    /// Network / connection / DNS failure.
    Network,
    /// Request timed out.
    Timeout,
    /// Overloaded — provider explicitly signals capacity exhaustion.
    Overloaded,
    /// Content filtered by provider safety policy.
    ContentFilter,
    /// Quota / billing exhausted.
    QuotaExhausted,
    /// Anything else.
    Unknown,
}

impl ErrorCategory {
    /// Whether an error of this category is worth retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorCategory::RateLimit
                | ErrorCategory::ServerError
                | ErrorCategory::Network
                | ErrorCategory::Timeout
                | ErrorCategory::Overloaded
        )
    }

    /// Whether this category indicates a configuration problem (no retry).
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            ErrorCategory::Authentication
                | ErrorCategory::BadRequest
                | ErrorCategory::ModelNotFound
                | ErrorCategory::QuotaExhausted
        )
    }
}

// ---------------------------------------------------------------------------
// Recovery actions
// ---------------------------------------------------------------------------

/// The action a caller should take for a classified error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the request (with optional backoff).
    Retry {
        /// Suggested backoff before the next attempt.
        backoff: BackoffStrategy,
        /// Maximum number of retries to attempt.
        max_retries: u32,
    },
    /// Retry after switching to a different API key (e.g. on 429).
    RetryWithNewCredential,
    /// Fail fast; do not retry.
    FailFast,
    /// Retry with a reduced payload (e.g. on 413 / context-too-long).
    RetryWithSmallerPayload,
    /// Retry against a fallback provider/model.
    Fallback,
}

/// Backoff strategy for retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// No delay before retry.
    Immediate,
    /// Fixed delay.
    Fixed(Duration),
    /// Exponential backoff with a base delay.
    Exponential {
        /// Base delay for the first retry.
        base: Duration,
        /// Maximum delay cap.
        max: Duration,
    },
}

impl BackoffStrategy {
    /// Compute the delay for a given retry attempt number (0-based).
    pub fn delay_for(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Immediate => Duration::ZERO,
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Exponential { base, max } => {
                let mut d = *base;
                for _ in 0..attempt {
                    d = d.saturating_mul(2);
                    if d >= *max {
                        return *max;
                    }
                }
                d.min(*max)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// The full result of classifying an error.
#[derive(Debug, Clone)]
pub struct ClassifiedError {
    /// The high-level category.
    pub category: ErrorCategory,
    /// The recommended recovery action.
    pub action: RecoveryAction,
    /// A human-readable description.
    pub message: String,
}

/// Classify a [`ProviderError`] plus optional HTTP status and response body.
///
/// `status` and `body` should be `Some` when the error originated from an HTTP
/// response; they may be `None` for network/timeout errors.
pub fn classify(
    err: &ProviderError,
    status: Option<u16>,
    body: Option<&str>,
) -> ClassifiedError {
    let category = categorize(err, status, body);

    let action = recovery_action(category, status, body);
    let message = err.to_string();

    ClassifiedError {
        category,
        action,
        message,
    }
}

/// Table-driven categorization.
fn categorize(err: &ProviderError, status: Option<u16>, body: Option<&str>) -> ErrorCategory {
    // First, inspect HTTP status if present.
    if let Some(code) = status {
        match code {
            401 | 403 => return ErrorCategory::Authentication,
            429 => return ErrorCategory::RateLimit,
            400 => {
                if let Some(b) = body {
                    let lower = b.to_ascii_lowercase();
                    if lower.contains("quota") || lower.contains("billing") {
                        return ErrorCategory::QuotaExhausted;
                    }
                    if lower.contains("content filter") || lower.contains("safety") {
                        return ErrorCategory::ContentFilter;
                    }
                }
                return ErrorCategory::BadRequest;
            }
            404 => return ErrorCategory::ModelNotFound,
            413 => return ErrorCategory::RequestTooLarge,
            500 | 502 | 503 | 504 => {
                if let Some(b) = body {
                    let lower = b.to_ascii_lowercase();
                    if lower.contains("overload") || lower.contains("capacity") {
                        return ErrorCategory::Overloaded;
                    }
                }
                return ErrorCategory::ServerError;
            }
            _ => {}
        }
    }

    // Fall back to the error variant.
    match err {
        ProviderError::Auth(_) => ErrorCategory::Authentication,
        ProviderError::RateLimited(_) => ErrorCategory::RateLimit,
        ProviderError::Timeout(_) => ErrorCategory::Timeout,
        ProviderError::Network(_) => ErrorCategory::Network,
        ProviderError::UnsupportedModel(_) => ErrorCategory::ModelNotFound,
        ProviderError::Config(_) => ErrorCategory::BadRequest,
        ProviderError::Provider(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("overload") || lower.contains("capacity") {
                ErrorCategory::Overloaded
            } else if lower.contains("quota") || lower.contains("billing") {
                ErrorCategory::QuotaExhausted
            } else if lower.contains("content filter") || lower.contains("safety") {
                ErrorCategory::ContentFilter
            } else if lower.contains("too large") || lower.contains("context length") {
                ErrorCategory::RequestTooLarge
            } else {
                ErrorCategory::Unknown
            }
        }
        ProviderError::Serialization(_) => ErrorCategory::BadRequest,
        ProviderError::Internal(_) => ErrorCategory::Unknown,
    }
}

/// Map a category to a recovery action.
fn recovery_action(
    category: ErrorCategory,
    _status: Option<u16>,
    body: Option<&str>,
) -> RecoveryAction {
    match category {
        ErrorCategory::RateLimit => {
            // If the body mentions a retry-after or a specific reset time,
            // prefer a fixed backoff; otherwise exponential.
            let backoff = if body
                .map(|b| b.to_ascii_lowercase().contains("retry-after"))
                .unwrap_or(false)
            {
                BackoffStrategy::Fixed(Duration::from_secs(20))
            } else {
                BackoffStrategy::Exponential {
                    base: Duration::from_secs(2),
                    max: Duration::from_secs(60),
                }
            };
            RecoveryAction::Retry {
                backoff,
                max_retries: 3,
            }
        }
        ErrorCategory::Overloaded => RecoveryAction::Retry {
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(5),
                max: Duration::from_secs(120),
            },
            max_retries: 2,
        },
        ErrorCategory::ServerError => RecoveryAction::Retry {
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(30),
            },
            max_retries: 3,
        },
        ErrorCategory::Network => RecoveryAction::Retry {
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_millis(500),
                max: Duration::from_secs(10),
            },
            max_retries: 3,
        },
        ErrorCategory::Timeout => RecoveryAction::Retry {
            backoff: BackoffStrategy::Fixed(Duration::from_secs(2)),
            max_retries: 2,
        },
        ErrorCategory::RequestTooLarge => RecoveryAction::RetryWithSmallerPayload,
        ErrorCategory::Authentication => RecoveryAction::FailFast,
        ErrorCategory::BadRequest => RecoveryAction::FailFast,
        ErrorCategory::ModelNotFound => RecoveryAction::Fallback,
        ErrorCategory::QuotaExhausted => RecoveryAction::FailFast,
        ErrorCategory::ContentFilter => RecoveryAction::FailFast,
        ErrorCategory::Unknown => RecoveryAction::Retry {
            backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            max_retries: 1,
        },
    }
}

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

/// State of a circuit breaker for a single provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Closed: requests flow normally.
    Closed,
    /// Open: requests fail fast.
    Open,
    /// Half-open: a limited probe request is allowed through.
    HalfOpen,
}

/// Configuration for a circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    /// Number of consecutive failures that trips the breaker open.
    pub failure_threshold: u32,
    /// Duration after which an open breaker transitions to half-open.
    pub reset_timeout: Duration,
    /// Number of successes in half-open required to close the breaker.
    pub half_open_successes: u32,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            half_open_successes: 1,
        }
    }
}

#[derive(Debug)]
struct CircuitEntry {
    state: CircuitState,
    consecutive_failures: u32,
    half_open_successes: u32,
    opened_at: Option<Instant>,
}

impl CircuitEntry {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            half_open_successes: 0,
            opened_at: None,
        }
    }
}

/// A per-provider circuit breaker registry.
///
/// Tracks failure rates for each provider and trips open when the consecutive
/// failure threshold is exceeded, failing fast until the reset timeout elapses.
#[derive(Clone)]
pub struct CircuitBreaker {
    circuits: Arc<DashMap<String, CircuitEntry>>,
    config: CircuitConfig,
}

impl CircuitBreaker {
    /// Create a new circuit breaker registry with the given config.
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            circuits: Arc::new(DashMap::new()),
            config,
        }
    }

    /// Create a breaker with default configuration.
    pub fn default_config() -> Self {
        Self::new(CircuitConfig::default())
    }

    /// Returns the current state for a provider, transitioning open -> half-open
    /// if the reset timeout has elapsed.
    pub fn state(&self, provider: &str) -> CircuitState {
        let now = Instant::now();
        let mut entry = self
            .circuits
            .entry(provider.to_string())
            .or_insert_with(CircuitEntry::new);

        match entry.state {
            CircuitState::Open => {
                if let Some(opened_at) = entry.opened_at {
                    if now.duration_since(opened_at) >= self.config.reset_timeout {
                        entry.state = CircuitState::HalfOpen;
                        entry.half_open_successes = 0;
                        debug!(target = "provider", provider = provider, "Circuit half-open");
                    }
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }
        entry.state
    }

    /// Returns `true` if a request to `provider` should be allowed through.
    pub fn allow(&self, provider: &str) -> bool {
        match self.state(provider) {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                warn!(target = "provider", provider = provider, "Circuit open; failing fast");
                false
            }
        }
    }

    /// Record a successful request to `provider`.
    pub fn record_success(&self, provider: &str) {
        let mut entry = self
            .circuits
            .entry(provider.to_string())
            .or_insert_with(CircuitEntry::new);
        entry.consecutive_failures = 0;
        match entry.state {
            CircuitState::HalfOpen => {
                entry.half_open_successes += 1;
                if entry.half_open_successes >= self.config.half_open_successes {
                    entry.state = CircuitState::Closed;
                    entry.opened_at = None;
                    debug!(target = "provider", provider = provider, "Circuit closed");
                }
            }
            CircuitState::Closed | CircuitState::Open => {}
        }
    }

    /// Record a failed request to `provider`.
    pub fn record_failure(&self, provider: &str) {
        let mut entry = self
            .circuits
            .entry(provider.to_string())
            .or_insert_with(CircuitEntry::new);
        entry.consecutive_failures += 1;
        match entry.state {
            CircuitState::HalfOpen => {
                // A failure in half-open reopens the circuit.
                entry.state = CircuitState::Open;
                entry.opened_at = Some(Instant::now());
                warn!(target = "provider", provider = provider, "Circuit reopened from half-open");
            }
            CircuitState::Closed => {
                if entry.consecutive_failures >= self.config.failure_threshold {
                    entry.state = CircuitState::Open;
                    entry.opened_at = Some(Instant::now());
                    warn!(
                        target = "provider",
                        provider = provider,
                        failures = entry.consecutive_failures,
                        "Circuit tripped open"
                    );
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the breaker for a provider (e.g. after manual intervention).
    pub fn reset(&self, provider: &str) {
        if let Some(mut entry) = self.circuits.get_mut(provider) {
            entry.state = CircuitState::Closed;
            entry.consecutive_failures = 0;
            entry.half_open_successes = 0;
            entry.opened_at = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Retry driver helper
// ---------------------------------------------------------------------------

/// Decide whether a classified error should abort a retry loop early.
pub fn should_abort(category: ErrorCategory) -> bool {
    !category.is_retryable()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_err(msg: &str) -> ProviderError {
        ProviderError::Provider(msg.to_string())
    }

    #[test]
    fn test_classify_429() {
        let err = provider_err("rate limited");
        let c = classify(&err, Some(429), Some("Too many requests"));
        assert_eq!(c.category, ErrorCategory::RateLimit);
        assert!(matches!(c.action, RecoveryAction::Retry { .. }));
        assert!(c.category.is_retryable());
    }

    #[test]
    fn test_classify_401() {
        let err = ProviderError::Auth("bad key".into());
        let c = classify(&err, Some(401), None);
        assert_eq!(c.category, ErrorCategory::Authentication);
        assert_eq!(c.action, RecoveryAction::FailFast);
        assert!(should_abort(c.category));
    }

    #[test]
    fn test_classify_500_retryable() {
        let err = provider_err("internal");
        let c = classify(&err, Some(500), None);
        assert_eq!(c.category, ErrorCategory::ServerError);
        assert!(matches!(c.action, RecoveryAction::Retry { .. }));
    }

    #[test]
    fn test_classify_overloaded_body() {
        let err = provider_err("overloaded");
        let c = classify(&err, Some(503), Some("service overloaded capacity"));
        assert_eq!(c.category, ErrorCategory::Overloaded);
    }

    #[test]
    fn test_classify_quota() {
        let err = provider_err("billing quota exceeded");
        let c = classify(&err, Some(400), Some("quota exceeded"));
        assert_eq!(c.category, ErrorCategory::QuotaExhausted);
        assert_eq!(c.action, RecoveryAction::FailFast);
    }

    #[test]
    fn test_classify_413_smaller_payload() {
        let err = provider_err("too large");
        let c = classify(&err, Some(413), None);
        assert_eq!(c.category, ErrorCategory::RequestTooLarge);
        assert_eq!(c.action, RecoveryAction::RetryWithSmallerPayload);
    }

    #[test]
    fn test_backoff_exponential() {
        let b = BackoffStrategy::Exponential {
            base: Duration::from_secs(1),
            max: Duration::from_secs(8),
        };
        assert_eq!(b.delay_for(0), Duration::from_secs(1));
        assert_eq!(b.delay_for(1), Duration::from_secs(2));
        assert_eq!(b.delay_for(2), Duration::from_secs(4));
        assert_eq!(b.delay_for(10), Duration::from_secs(8));
    }

    #[test]
    fn test_circuit_breaker_trips_and_resets() {
        let cb = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_millis(50),
            half_open_successes: 1,
        });
        assert!(cb.allow("openai"));
        cb.record_failure("openai");
        cb.record_failure("openai");
        assert!(cb.allow("openai"));
        cb.record_failure("openai");
        // Tripped open.
        assert!(!cb.allow("openai"));
        assert_eq!(cb.state("openai"), CircuitState::Open);
        // Wait for reset timeout.
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(cb.state("openai"), CircuitState::HalfOpen);
        cb.record_success("openai");
        assert_eq!(cb.state("openai"), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(CircuitConfig {
            failure_threshold: 1,
            reset_timeout: Duration::from_millis(20),
            half_open_successes: 1,
        });
        cb.record_failure("p");
        assert!(!cb.allow("p"));
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(cb.state("p"), CircuitState::HalfOpen);
        cb.record_failure("p");
        assert_eq!(cb.state("p"), CircuitState::Open);
    }
}
