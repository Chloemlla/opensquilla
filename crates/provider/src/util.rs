//! Shared provider utilities: retry with exponential backoff, a token-bucket
//! rate limiter, and HTTP status -> [`ProviderError`] mapping.
//!
//! Both the image-generation and audio providers drive every outbound HTTP
//! request through [`with_retry`], gate them through an optional
//! [`RateLimiter`], and normalize non-success statuses with [`check_status`].
//! Keeping these in one place avoids duplicating backoff and 429 handling
//! across the two modules.

use crate::types::{ProviderError, ProviderResult};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::warn;

/// Default HTTP timeout for provider requests.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Retry policy for transient provider failures.
///
/// Exponential backoff with jitter is applied between attempts. Only transient
/// failures (network errors, timeouts, server errors, and optionally 429s) are
/// retried; authentication, configuration, and client errors fail immediately.
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Base delay in milliseconds for the first retry.
    pub base_delay_ms: u64,
    /// Maximum delay in milliseconds between retries.
    pub max_delay_ms: u64,
    /// Whether HTTP 429 responses should be retried.
    pub retry_on_429: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 500,
            max_delay_ms: 8_000,
            retry_on_429: true,
        }
    }
}

impl RetryConfig {
    /// A config that performs no retries.
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            base_delay_ms: 0,
            max_delay_ms: 0,
            retry_on_429: false,
        }
    }

    /// Exponential backoff with jitter for the given 1-based attempt number.
    ///
    /// Attempt 1 returns the base delay, attempt 2 doubles it, and so on up to
    /// `max_delay_ms`, with a small random component added to desynchronize
    /// concurrent retriers.
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1).min(6);
        let base = self
            .base_delay_ms
            .saturating_mul(1u64 << exponent)
            .min(self.max_delay_ms.max(self.base_delay_ms));
        let jitter = rand::thread_rng().gen_range(0..=base.max(1) / 4);
        Duration::from_millis(base + jitter)
    }

    /// Whether a retry should be attempted after the given (0-based) attempt
    /// index failed with `err`.
    pub fn should_retry(&self, attempt: u32, err: &ProviderError) -> bool {
        if attempt.saturating_add(1) >= self.max_attempts {
            return false;
        }
        if !is_retryable_error(err) {
            return false;
        }
        if matches!(err, ProviderError::RateLimited(_)) && !self.retry_on_429 {
            return false;
        }
        true
    }
}

/// Whether an error is transient and worth retrying.
pub fn is_retryable_error(err: &ProviderError) -> bool {
    match err {
        ProviderError::Network(_) | ProviderError::Timeout(_) => true,
        ProviderError::RateLimited(_) => true,
        // Provider(String) carries "HTTP <status>: ..." bodies from
        // `check_status`; server errors and explicit overloads are retryable.
        ProviderError::Provider(msg) => {
            msg.contains("HTTP 5")
                || msg.contains(" 500")
                || msg.contains(" 502")
                || msg.contains(" 503")
                || msg.contains(" 504")
        }
        _ => false,
    }
}

/// Run an operation, retrying transient failures with exponential backoff.
///
/// The caller-supplied closure receives the 0-based attempt index and returns
/// a future. Retries happen only for errors [`RetryConfig::should_retry`]
/// considers transient, so auth/config failures surface immediately.
pub async fn with_retry<T, F, Fut>(config: &RetryConfig, mut op: F) -> ProviderResult<T>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = ProviderResult<T>>,
{
    let mut attempt: u32 = 0;
    loop {
        match op(attempt).await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !config.should_retry(attempt, &err) {
                    return Err(err);
                }
                let delay = config.delay_for_attempt(attempt + 1);
                warn!(
                    target = "provider",
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    error = %err,
                    "Retrying provider request"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
        }
    }
}

/// Map a non-success HTTP status to a [`ProviderError`].
///
/// 401/403 become [`ProviderError::Auth`], 429 becomes
/// [`ProviderError::RateLimited`], and everything else becomes
/// [`ProviderError::Provider`] with the HTTP status and a truncated body.
pub fn check_status(status: reqwest::StatusCode, text: &str, action: &str) -> ProviderResult<()> {
    if status.is_success() {
        return Ok(());
    }
    let body = truncate(text, 1000);
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(ProviderError::Auth(format!("{action} auth failed: {body}")));
    }
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(ProviderError::RateLimited(format!(
            "{action} rate limited: {body}"
        )));
    }
    Err(ProviderError::Provider(format!("HTTP {status}: {body}")))
}

/// Truncate a string to at most `max_chars` characters, appending `...`.
pub fn truncate(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    let mut out: String = chars[..max_chars].iter().collect();
    out.push_str("...");
    out
}

/// A token-bucket rate limiter for pacing outbound provider requests.
///
/// The bucket refills at `requests_per_second` and holds up to `capacity`
/// tokens, so short bursts are allowed while the sustained rate stays bounded.
/// All state lives behind a plain `std::sync::Mutex` because the lock is never
/// held across an `.await` point.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<RateLimiterState>>,
}

#[derive(Debug)]
struct RateLimiterState {
    tokens: f64,
    capacity: f64,
    refill_per_second: f64,
    last_refill: Instant,
}

impl RateLimiterState {
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_per_second).min(self.capacity);
            self.last_refill = now;
        }
    }
}

impl RateLimiter {
    /// Create a rate limiter allowing `requests_per_second` requests sustained.
    pub fn new(requests_per_second: f64) -> Self {
        Self::with_capacity(requests_per_second, requests_per_second.max(1.0))
    }

    /// Create a rate limiter with a custom burst capacity.
    pub fn with_capacity(requests_per_second: f64, capacity: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RateLimiterState {
                tokens: capacity.max(1.0),
                capacity: capacity.max(1.0),
                refill_per_second: requests_per_second.max(0.0),
                last_refill: Instant::now(),
            })),
        }
    }

    /// Non-blocking token acquisition.
    ///
    /// Returns `true` and consumes a token when one is available, `false`
    /// otherwise. Useful for callers that would rather drop a request than wait.
    pub fn try_acquire(&self) -> bool {
        let mut state = self.inner.lock().expect("rate limiter mutex poisoned");
        state.refill();
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Acquire a token, sleeping until one becomes available.
    pub async fn acquire(&self) -> Result<(), ProviderError> {
        loop {
            let wait = {
                let mut state = self.inner.lock().expect("rate limiter mutex poisoned");
                state.refill();
                if state.tokens >= 1.0 {
                    state.tokens -= 1.0;
                    None
                } else {
                    let missing = 1.0 - state.tokens;
                    let secs = missing / state.refill_per_second.max(1e-9);
                    Some(Duration::from_secs_f64(secs.min(60.0)))
                }
            };
            match wait {
                None => return Ok(()),
                Some(delay) => tokio::time::sleep(delay).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_default_config() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_attempts, 3);
        assert!(cfg.retry_on_429);
    }

    #[test]
    fn no_retry_never_retries() {
        let cfg = RetryConfig::no_retry();
        assert_eq!(cfg.max_attempts, 1);
        assert!(!cfg.should_retry(0, &ProviderError::Timeout("boom".into())));
    }

    #[test]
    fn retry_delay_monotonic_increases() {
        let cfg = RetryConfig {
            base_delay_ms: 100,
            max_delay_ms: 5_000,
            retry_on_429: true,
            max_attempts: 5,
        };
        // With jitter, ensure the multiplier grows: attempt 4 >= attempt 2.
        let d2 = cfg.delay_for_attempt(2).as_millis() as u64;
        let d4 = cfg.delay_for_attempt(4).as_millis() as u64;
        assert!(
            d4 > d2,
            "expected attempt 4 delay {d4} > attempt 2 delay {d2}"
        );
    }

    #[test]
    fn should_retry_respects_max_attempts() {
        let cfg = RetryConfig::default();
        let net_err = ProviderError::Timeout("connection reset".into());
        assert!(cfg.should_retry(0, &net_err));
        assert!(cfg.should_retry(1, &net_err));
        // attempt index 2 is the third attempt; max_attempts = 3 => stop.
        assert!(!cfg.should_retry(2, &net_err));
    }

    #[test]
    fn retryable_classification() {
        assert!(is_retryable_error(&ProviderError::Timeout("slow".into())));
        assert!(is_retryable_error(&ProviderError::RateLimited(
            "429".into()
        )));
        assert!(is_retryable_error(&ProviderError::Provider(
            "HTTP 503: busy".into()
        )));
        assert!(!is_retryable_error(&ProviderError::Auth("bad key".into())));
        assert!(!is_retryable_error(&ProviderError::Config(
            "misconfig".into()
        )));
        assert!(!is_retryable_error(&ProviderError::Provider(
            "HTTP 400: bad".into()
        )));
    }

    #[test]
    fn check_status_maps_codes() {
        assert!(check_status(reqwest::StatusCode::OK, "", "test").is_ok());
        assert!(matches!(
            check_status(reqwest::StatusCode::UNAUTHORIZED, "nope", "img"),
            ProviderError::Auth(_)
        ));
        assert!(matches!(
            check_status(reqwest::StatusCode::TOO_MANY_REQUESTS, "slow down", "img"),
            ProviderError::RateLimited(_)
        ));
        assert!(matches!(
            check_status(reqwest::StatusCode::BAD_GATEWAY, "oops", "img"),
            ProviderError::Provider(_)
        ));
    }

    #[test]
    fn truncate_limits_chars() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
        // Multi-byte chars are truncated on char boundaries, not bytes.
        let out = truncate("héllo", 3);
        assert_eq!(out, "hé...");
    }

    #[test]
    fn rate_limiter_burst() {
        let limiter = RateLimiter::with_capacity(10.0, 3.0);
        // Burst of 3 should all pass immediately.
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        // Fourth is beyond capacity.
        assert!(!limiter.try_acquire());
    }

    #[tokio::test]
    async fn with_retry_succeeds_after_transient_failures() {
        let cfg = RetryConfig {
            base_delay_ms: 1,
            max_delay_ms: 5,
            max_attempts: 3,
            retry_on_429: true,
        };
        let mut calls = 0u32;
        let result = with_retry(&cfg, |_| async {
            calls += 1;
            if calls < 3 {
                Err(ProviderError::Timeout("transient".into()))
            } else {
                Ok(42u32)
            }
        })
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn with_retry_gives_up_after_max_attempts() {
        let cfg = RetryConfig {
            base_delay_ms: 1,
            max_delay_ms: 5,
            max_attempts: 2,
            retry_on_429: true,
        };
        let mut calls = 0u32;
        let result = with_retry(&cfg, |_| async {
            calls += 1;
            Err(ProviderError::Timeout("always".into()))
        })
        .await;
        assert!(matches!(result, Err(ProviderError::Timeout(_))));
        assert_eq!(calls, 2);
    }

    #[tokio::test]
    async fn real_network_error_is_retryable() {
        // An invalid IPv6 URL fails during URL parsing, yielding a real
        // `reqwest::Error` without touching the network.
        let client = reqwest::Client::new();
        let err = client.get("http://[::1").send().await.unwrap_err();
        let provider_err = ProviderError::Network(err);
        assert!(is_retryable_error(&provider_err));
    }
}
