//! Provider deployment health ledger: temporary benching on classified
//! failures.
//!
//! Mirrors the Python `engine/routing/health.py`. The ledger answers the
//! question *is this (provider, model) deployment temporarily benched?* across
//! calls and turns.
//!
//! Bench rules (pinned):
//!
//! * a deployment is benched after `failure_threshold` (default 3) recorded
//!   benchable failures, for `cooldown_s` (default 30) seconds;
//! * `RateLimited` (HTTP 429) benches immediately; the cooldown is the
//!   provider's `Retry-After` when present, else the default;
//! * on 5xx-shaped failures (`ProviderOverloaded` / gateway-transient),
//!   `Retry-After` is honored for the cooldown when the bench triggers;
//! * the ledger NEVER reports the only viable deployment for a tier as benched:
//!   [`ProviderHealthLedger::eligible`] refuses to strand routing when every
//!   alternative is also benched.

use std::collections::HashMap;
use std::sync::Mutex;

/// Default strike threshold before a deployment is benched.
pub const DEFAULT_FAILURE_THRESHOLD: usize = 3;
/// Default cooldown in seconds.
pub const DEFAULT_COOLDOWN_S: f64 = 30.0;
/// Defensive ceiling for Retry-After-driven cooldowns (seconds).
pub const DEFAULT_MAX_COOLDOWN_S: f64 = 900.0;

/// Classified provider failure kinds, mirroring `provider/failures.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderFailureKind {
    RateLimited,
    ProviderOverloaded,
    AuthInvalid,
    ContextOverflow,
    UnsupportedFeature,
    InsufficientCredits,
    ModelNotFound,
    TransportTransient,
    PolicyRefusal,
    EmptyResponse,
    MalformedResponse,
    BadRequest,
    Unknown,
}

impl ProviderFailureKind {
    /// The wire/telemetry token for this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderFailureKind::RateLimited => "rate_limited",
            ProviderFailureKind::ProviderOverloaded => "provider_overloaded",
            ProviderFailureKind::AuthInvalid => "auth_invalid",
            ProviderFailureKind::ContextOverflow => "context_overflow",
            ProviderFailureKind::UnsupportedFeature => "unsupported_feature",
            ProviderFailureKind::InsufficientCredits => "insufficient_credits",
            ProviderFailureKind::ModelNotFound => "model_not_found",
            ProviderFailureKind::TransportTransient => "transport_transient",
            ProviderFailureKind::PolicyRefusal => "policy_refusal",
            ProviderFailureKind::EmptyResponse => "empty_response",
            ProviderFailureKind::MalformedResponse => "malformed_response",
            ProviderFailureKind::BadRequest => "bad_request",
            ProviderFailureKind::Unknown => "unknown",
        }
    }

    /// Parse a wire token back into a kind (unknown tokens map to `Unknown`).
    pub fn from_str(value: &str) -> Self {
        match value.trim().to_lowercase().as_str() {
            "rate_limited" => ProviderFailureKind::RateLimited,
            "provider_overloaded" => ProviderFailureKind::ProviderOverloaded,
            "auth_invalid" => ProviderFailureKind::AuthInvalid,
            "context_overflow" => ProviderFailureKind::ContextOverflow,
            "unsupported_feature" => ProviderFailureKind::UnsupportedFeature,
            "insufficient_credits" => ProviderFailureKind::InsufficientCredits,
            "model_not_found" => ProviderFailureKind::ModelNotFound,
            "transport_transient" => ProviderFailureKind::TransportTransient,
            "policy_refusal" => ProviderFailureKind::PolicyRefusal,
            "empty_response" => ProviderFailureKind::EmptyResponse,
            "malformed_response" => ProviderFailureKind::MalformedResponse,
            "bad_request" => ProviderFailureKind::BadRequest,
            _ => ProviderFailureKind::Unknown,
        }
    }
}

/// Kinds that signal *deployment* unhealth. Request-shaped kinds
/// (`ContextOverflow`, `BadRequest`, `PolicyRefusal`, `UnsupportedFeature`)
/// follow the request, not the deployment; deterministic config kinds
/// (`AuthInvalid`, `InsufficientCredits`, `ModelNotFound`) and `Unknown` are
/// excluded.
pub const BENCHABLE_FAILURE_KINDS: &[ProviderFailureKind] = &[
    ProviderFailureKind::RateLimited,
    ProviderFailureKind::ProviderOverloaded,
    ProviderFailureKind::TransportTransient,
    ProviderFailureKind::EmptyResponse,
    ProviderFailureKind::MalformedResponse,
];

/// A (provider, model) deployment key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DeploymentKey {
    provider: String,
    model: String,
}

fn deployment_key(provider: &str, model: &str) -> DeploymentKey {
    DeploymentKey {
        provider: provider.trim().to_lowercase(),
        model: model.trim().to_string(),
    }
}

/// A monotonic clock returning seconds. Reuses the routing module's
/// process-relative monotonic clock.
pub type Clock = dyn Fn() -> f64 + Send + Sync;

/// Strike counter + cooldown bench for (provider, model) deployments.
///
/// Feed it classified failures via [`ProviderHealthLedger::record_failure`]
/// and clear strikes via [`ProviderHealthLedger::record_success`]; query it
/// via [`ProviderHealthLedger::eligible`] (routing paths — enforces the
/// never-strand exemption) or [`ProviderHealthLedger::is_benched`] (raw
/// state). All methods are thread-safe.
pub struct ProviderHealthLedger {
    failure_threshold: usize,
    cooldown_s: f64,
    max_cooldown_s: f64,
    clock: Box<Clock>,
    strikes: Mutex<HashMap<DeploymentKey, usize>>,
    benched_until: Mutex<HashMap<DeploymentKey, f64>>,
}

impl std::fmt::Debug for ProviderHealthLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderHealthLedger")
            .field("failure_threshold", &self.failure_threshold)
            .field("cooldown_s", &self.cooldown_s)
            .field("max_cooldown_s", &self.max_cooldown_s)
            .field(
                "tracked_deployments",
                &self.strikes.lock().map(|s| s.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl Default for ProviderHealthLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderHealthLedger {
    /// Create a ledger with the pinned default parameters.
    pub fn new() -> Self {
        Self::with_clock(Box::new(crate::routing::monotonic_now))
    }

    /// Create a ledger with an injectable monotonic clock (seconds).
    pub fn with_clock(clock: Box<Clock>) -> Self {
        Self {
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            cooldown_s: DEFAULT_COOLDOWN_S,
            max_cooldown_s: DEFAULT_MAX_COOLDOWN_S,
            clock,
            strikes: Mutex::new(HashMap::new()),
            benched_until: Mutex::new(HashMap::new()),
        }
    }

    /// Configure the strike threshold (must be >= 1).
    pub fn with_failure_threshold(mut self, threshold: usize) -> Self {
        self.failure_threshold = threshold.max(1);
        self
    }

    /// Configure the default cooldown in seconds (must be positive).
    pub fn with_cooldown(mut self, cooldown_s: f64) -> Self {
        self.cooldown_s = cooldown_s.max(1e-9);
        self
    }

    /// Configure the Retry-After cooldown ceiling (seconds).
    pub fn with_max_cooldown(mut self, max_cooldown_s: f64) -> Self {
        self.max_cooldown_s = max_cooldown_s.max(self.cooldown_s);
        self
    }

    /// Record one classified failure; returns whether the deployment is now
    /// benched.
    ///
    /// Non-benchable kinds neither count a strike nor bench. `RateLimited`
    /// benches immediately; other benchable kinds bench once the strike
    /// threshold is reached. The cooldown is `retry_after_s` when provided
    /// (clamped to `max_cooldown_s`), else the default. `now` overrides the
    /// clock reading and must be in the same monotonic domain.
    pub fn record_failure(
        &self,
        provider: &str,
        model: &str,
        kind: ProviderFailureKind,
        retry_after_s: Option<f64>,
        now: Option<f64>,
    ) -> bool {
        let key = deployment_key(provider, model);
        let ts = now.unwrap_or_else(|| (self.clock)());

        {
            let mut benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
            self.expire_locked(&key, ts, &mut benched);
        }

        if !BENCHABLE_FAILURE_KINDS.contains(&kind) {
            return self
                .benched_until
                .lock()
                .map(|b| b.contains_key(&key))
                .unwrap_or(false);
        }

        let immediate = kind == ProviderFailureKind::RateLimited;
        let current = {
            let mut strikes = self.strikes.lock().unwrap_or_else(|e| e.into_inner());
            let entry = strikes.entry(key.clone()).or_insert(0);
            *entry += 1;
            *entry
        };

        if !immediate && current < self.failure_threshold {
            return self
                .benched_until
                .lock()
                .map(|b| b.contains_key(&key))
                .unwrap_or(false);
        }

        let cooldown = self.cooldown_for(retry_after_s);
        let benched_until_ts = ts + cooldown;
        // Strikes are consumed by the bench: after the cooldown the deployment
        // starts from a clean slate.
        self.strikes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&key);

        let mut benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
        if benched.get(&key).copied().unwrap_or(f64::NEG_INFINITY) >= benched_until_ts {
            return true;
        }
        benched.insert(key.clone(), benched_until_ts);
        tracing::warn!(
            provider = %key.provider,
            model = %key.model,
            kind = %kind.as_str(),
            cooldown_s = cooldown,
            strikes = current,
            immediate = immediate,
            "provider_health.benched"
        );
        true
    }

    /// A good call clears the strike count (and any active bench).
    pub fn record_success(&self, provider: &str, model: &str) {
        let key = deployment_key(provider, model);
        let was_benched = {
            self.strikes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&key);
            self.benched_until
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(&key)
                .is_some()
        };
        if was_benched {
            tracing::info!(
                provider = %key.provider,
                model = %key.model,
                reason = "success",
                "provider_health.unbenched"
            );
        }
    }

    /// Raw bench state, without the single-deployment exemption.
    ///
    /// Routing paths should prefer [`ProviderHealthLedger::eligible`].
    pub fn is_benched(&self, provider: &str, model: &str, now: Option<f64>) -> bool {
        let key = deployment_key(provider, model);
        let ts = now.unwrap_or_else(|| (self.clock)());
        let mut benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
        self.expire_locked(&key, ts, &mut benched);
        benched.contains_key(&key)
    }

    /// Whether routing may use this deployment, given the tier's candidates.
    ///
    /// `candidate_deployments` is every (provider, model) pair that could
    /// serve the need (it may include the queried pair). A benched deployment
    /// is reported eligible anyway when no alternative candidate is unbenched:
    /// a bench that strands routing is worse than one more failed attempt.
    pub fn eligible(
        &self,
        provider: &str,
        model: &str,
        candidate_deployments: &[(String, String)],
        now: Option<f64>,
    ) -> bool {
        let key = deployment_key(provider, model);
        let ts = now.unwrap_or_else(|| (self.clock)());
        let mut benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
        self.expire_locked(&key, ts, &mut benched);
        if !benched.contains_key(&key) {
            return true;
        }

        let alternatives: Vec<DeploymentKey> = candidate_deployments
            .iter()
            .map(|(p, m)| deployment_key(p, m))
            .filter(|k| *k != key)
            .collect();
        for alt in &alternatives {
            self.expire_locked(alt, ts, &mut benched);
            if !benched.contains_key(alt) {
                return false;
            }
        }
        tracing::info!(
            provider = %key.provider,
            model = %key.model,
            candidates = alternatives.len() + 1,
            "provider_health.bench_exempted_only_deployment"
        );
        true
    }

    /// The number of deployments currently benched.
    pub fn benched_count(&self) -> usize {
        self.benched_until.lock().map(|b| b.len()).unwrap_or(0)
    }

    fn cooldown_for(&self, retry_after_s: Option<f64>) -> f64 {
        match retry_after_s {
            None => self.cooldown_s,
            Some(ra) => ra.max(0.0).min(self.max_cooldown_s),
        }
    }

    fn expire_locked(
        &self,
        key: &DeploymentKey,
        ts: f64,
        benched: &mut HashMap<DeploymentKey, f64>,
    ) {
        if let Some(until) = benched.get(key) {
            if *until <= ts {
                benched.remove(key);
                tracing::info!(
                    provider = %key.provider,
                    model = %key.model,
                    reason = "cooldown_expired",
                    "provider_health.unbenched"
                );
            }
        }
    }

    /// Snapshot the current bench state for persistence.
    ///
    /// Returns the (provider, model) pairs currently benched along with the
    /// timestamp until which they are benched (in the ledger's monotonic
    /// clock domain).
    pub fn snapshot(&self) -> Vec<BenchEntry> {
        let mut entries = Vec::new();
        let now = (self.clock)();
        let benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
        for (key, until) in benched.iter() {
            if *until > now {
                entries.push(BenchEntry {
                    provider: key.provider.clone(),
                    model: key.model.clone(),
                    benched_until_s: *until,
                });
            }
        }
        entries
    }

    /// Restore bench state from a persisted snapshot.
    ///
    /// Entries whose cooldown has already expired are dropped.
    pub fn restore(&self, entries: &[BenchEntry]) -> usize {
        let now = (self.clock)();
        let mut benched = self.benched_until.lock().unwrap_or_else(|e| e.into_inner());
        let mut restored = 0usize;
        for entry in entries {
            if entry.benched_until_s <= now {
                continue;
            }
            let key = deployment_key(&entry.provider, &entry.model);
            benched.insert(key, entry.benched_until_s);
            restored += 1;
        }
        restored
    }

    /// The number of deployments currently tracked (strikes + benched).
    pub fn tracked_count(&self) -> usize {
        let strikes = self.strikes.lock().map(|s| s.len()).unwrap_or(0);
        let benched = self.benched_until.lock().map(|b| b.len()).unwrap_or(0);
        strikes + benched
    }

    /// Clear all strikes and benches.
    pub fn clear(&self) {
        self.strikes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.benched_until
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// A persisted bench entry for the health ledger.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchEntry {
    /// The provider id.
    pub provider: String,
    /// The model id.
    pub model: String,
    /// The timestamp until which the deployment is benched (in the ledger's
    /// monotonic clock domain).
    pub benched_until_s: f64,
}

/// Persist the health ledger's bench state to a JSON file.
///
/// The write is atomic (temp file + rename). A missing directory is created.
pub fn save_health_ledger(
    ledger: &ProviderHealthLedger,
    path: &std::path::Path,
) -> Result<(), opensquilla_core::error::Error> {
    let entries = ledger.snapshot();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_string_pretty(&entries)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &payload)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load a health ledger's bench state from a JSON file.
///
/// A missing or corrupt file is treated as an empty state.
pub fn load_health_ledger(ledger: &ProviderHealthLedger, path: &std::path::Path) -> usize {
    let Ok(raw) = std::fs::read_to_string(path) else {
        return 0;
    };
    let Ok(entries) = serde_json::from_str::<Vec<BenchEntry>>(&raw) else {
        return 0;
    };
    ledger.restore(&entries)
}

/// A process-wide shared ledger, constructed lazily with the pinned defaults.
///
/// Deployment health is global, not per-turn, so opt-in consumers should share
/// this instance. Nothing on the default path calls it.
pub fn get_provider_health_ledger() -> &'static ProviderHealthLedger {
    static LEDGER: std::sync::OnceLock<ProviderHealthLedger> = std::sync::OnceLock::new();
    LEDGER.get_or_init(ProviderHealthLedger::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> ProviderHealthLedger {
        ProviderHealthLedger::with_clock(Box::new(|| 0.0))
    }

    #[test]
    fn test_rate_limited_benches_immediately() {
        let ledger = ledger();
        let benched = ledger.record_failure(
            "openai",
            "gpt-4o",
            ProviderFailureKind::RateLimited,
            None,
            Some(0.0),
        );
        assert!(benched);
        assert!(ledger.is_benched("openai", "gpt-4o", Some(0.0)));
    }

    #[test]
    fn test_threshold_not_met() {
        let ledger = ledger().with_failure_threshold(3);
        for i in 1..=2 {
            let benched = ledger.record_failure(
                "openai",
                "gpt-4o",
                ProviderFailureKind::TransportTransient,
                None,
                Some(i as f64),
            );
            assert!(!benched, "strike {i} should not bench yet");
        }
        let benched = ledger.record_failure(
            "openai",
            "gpt-4o",
            ProviderFailureKind::TransportTransient,
            None,
            Some(3.0),
        );
        assert!(benched);
    }

    #[test]
    fn test_cooldown_expires() {
        let ledger = ledger().with_cooldown(10.0);
        ledger.record_failure(
            "openai",
            "gpt-4o",
            ProviderFailureKind::RateLimited,
            None,
            Some(0.0),
        );
        assert!(ledger.is_benched("openai", "gpt-4o", Some(5.0)));
        assert!(!ledger.is_benched("openai", "gpt-4o", Some(11.0)));
    }

    #[test]
    fn test_success_clears_strikes() {
        let ledger = ledger().with_failure_threshold(3);
        ledger.record_failure(
            "a",
            "m",
            ProviderFailureKind::TransportTransient,
            None,
            Some(1.0),
        );
        ledger.record_failure(
            "a",
            "m",
            ProviderFailureKind::TransportTransient,
            None,
            Some(2.0),
        );
        ledger.record_success("a", "m");
        // Strikes cleared: a third failure should not bench.
        let benched = ledger.record_failure(
            "a",
            "m",
            ProviderFailureKind::TransportTransient,
            None,
            Some(3.0),
        );
        assert!(!benched);
    }

    #[test]
    fn test_never_strand_single_deployment() {
        let ledger = ledger();
        ledger.record_failure("a", "m", ProviderFailureKind::RateLimited, None, Some(0.0));
        // The queried deployment is the only candidate: it stays eligible.
        let eligible = ledger.eligible("a", "m", &[("a".into(), "m".into())], Some(0.0));
        assert!(eligible);
        // With a healthy alternative, the benched one is ineligible.
        let eligible = ledger.eligible(
            "a",
            "m",
            &[("a".into(), "m".into()), ("b".into(), "m".into())],
            Some(0.0),
        );
        assert!(!eligible);
    }

    #[test]
    fn test_non_benchable_kind() {
        let ledger = ledger();
        let benched =
            ledger.record_failure("a", "m", ProviderFailureKind::BadRequest, None, Some(0.0));
        assert!(!benched);
        assert!(!ledger.is_benched("a", "m", Some(0.0)));
    }
}
