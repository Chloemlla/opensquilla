//! Token and cost budget enforcement across turns.
//!
//! This module provides budget enforcement for the agent loop:
//!
//! * **Token budget**: per-turn and per-session token limits, with soft
//!   warnings and hard caps.
//! * **Cost budget**: per-session and per-agent USD spend limits.
//! * **Rate limiting**: per-model request rate limiting with token-bucket.
//! * **Circuit breaker**: automatic failover when a model/provider fails
//!   repeatedly.
//!
//! It mirrors the Python backend's `engine/budget.py` and the budget
//! enforcement hooks in `engine/agent.py`.

use crate::agent::UsageEvent;
use crate::pricing::{cost, PricingCache};
use opensquilla_core::types::Usage;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// The result of a budget check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetCheckResult {
    /// The request is within budget.
    Ok,
    /// The request is approaching the budget limit (soft warning).
    Warning {
        /// The current usage.
        current: u64,
        /// The limit.
        limit: u64,
        /// The fraction of the limit consumed.
        fraction: f64,
    },
    /// The request exceeds the budget limit (hard cap).
    Exceeded {
        /// The current usage.
        current: u64,
        /// The limit.
        limit: u64,
        /// The fraction of the limit consumed.
        fraction: f64,
    },
}

impl BudgetCheckResult {
    /// Whether the budget was exceeded.
    pub fn is_exceeded(&self) -> bool {
        matches!(self, BudgetCheckResult::Exceeded { .. })
    }

    /// Whether the budget is in warning territory.
    pub fn is_warning(&self) -> bool {
        matches!(self, BudgetCheckResult::Warning { .. })
    }

    /// Whether the request is within budget.
    pub fn is_ok(&self) -> bool {
        matches!(self, BudgetCheckResult::Ok)
    }
}

/// A token budget for a turn or session.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// The maximum input tokens allowed.
    pub max_input_tokens: u64,
    /// The maximum output tokens allowed.
    pub max_output_tokens: u64,
    /// The maximum total tokens allowed.
    pub max_total_tokens: u64,
    /// The fraction at which a warning is triggered (default 0.8).
    pub warning_fraction: f64,
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_input_tokens: 1_000_000,
            max_output_tokens: 200_000,
            max_total_tokens: 1_200_000,
            warning_fraction: 0.8,
        }
    }
}

impl TokenBudget {
    /// Create a new token budget with the given limits.
    pub fn new(max_total: u64) -> Self {
        Self {
            max_total_tokens: max_total,
            max_input_tokens: (max_total as f64 * 0.8) as u64,
            max_output_tokens: (max_total as f64 * 0.2) as u64,
            ..Default::default()
        }
    }

    /// Set the input token limit.
    pub fn with_input_limit(mut self, max: u64) -> Self {
        self.max_input_tokens = max;
        self
    }

    /// Set the output token limit.
    pub fn with_output_limit(mut self, max: u64) -> Self {
        self.max_output_tokens = max;
        self
    }

    /// Set the warning fraction.
    pub fn with_warning_fraction(mut self, frac: f64) -> Self {
        self.warning_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Check a usage snapshot against this budget.
    pub fn check(&self, usage: &Usage) -> BudgetCheckResult {
        let total = usage.total_tokens;
        if self.max_total_tokens > 0 && total >= self.max_total_tokens {
            return BudgetCheckResult::Exceeded {
                current: total,
                limit: self.max_total_tokens,
                fraction: total as f64 / self.max_total_tokens as f64,
            };
        }
        if self.max_input_tokens > 0 && usage.input_tokens >= self.max_input_tokens {
            return BudgetCheckResult::Exceeded {
                current: usage.input_tokens,
                limit: self.max_input_tokens,
                fraction: usage.input_tokens as f64 / self.max_input_tokens as f64,
            };
        }
        if self.max_output_tokens > 0 && usage.output_tokens >= self.max_output_tokens {
            return BudgetCheckResult::Exceeded {
                current: usage.output_tokens,
                limit: self.max_output_tokens,
                fraction: usage.output_tokens as f64 / self.max_output_tokens as f64,
            };
        }
        let warning_threshold = (self.max_total_tokens as f64 * self.warning_fraction) as u64;
        if total >= warning_threshold {
            return BudgetCheckResult::Warning {
                current: total,
                limit: self.max_total_tokens,
                fraction: total as f64 / self.max_total_tokens as f64,
            };
        }
        BudgetCheckResult::Ok
    }
}

/// A cost budget in USD.
#[derive(Debug, Clone)]
pub struct CostBudget {
    /// The maximum spend in USD for a session.
    pub max_session_usd: f64,
    /// The maximum spend in USD for a single turn.
    pub max_turn_usd: f64,
    /// The spend at which a warning is triggered (fraction of max).
    pub warning_fraction: f64,
}

impl Default for CostBudget {
    fn default() -> Self {
        Self {
            max_session_usd: 10.0,
            max_turn_usd: 1.0,
            warning_fraction: 0.8,
        }
    }
}

impl CostBudget {
    /// Create a new cost budget with the given session limit.
    pub fn new(max_session_usd: f64) -> Self {
        Self {
            max_session_usd,
            ..Default::default()
        }
    }

    /// Set the per-turn cost limit.
    pub fn with_turn_limit(mut self, max: f64) -> Self {
        self.max_turn_usd = max;
        self
    }

    /// Set the warning fraction.
    pub fn with_warning_fraction(mut self, frac: f64) -> Self {
        self.warning_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Check a session spend against this budget.
    pub fn check_session(&self, spend_usd: f64) -> BudgetCheckResult {
        if spend_usd >= self.max_session_usd {
            return BudgetCheckResult::Exceeded {
                current: spend_usd as u64,
                limit: self.max_session_usd as u64,
                fraction: if self.max_session_usd > 0.0 {
                    spend_usd / self.max_session_usd
                } else {
                    1.0
                },
            };
        }
        let warning_threshold = self.max_session_usd * self.warning_fraction;
        if spend_usd >= warning_threshold {
            return BudgetCheckResult::Warning {
                current: spend_usd as u64,
                limit: self.max_session_usd as u64,
                fraction: if self.max_session_usd > 0.0 {
                    spend_usd / self.max_session_usd
                } else {
                    1.0
                },
            };
        }
        BudgetCheckResult::Ok
    }

    /// Check a per-turn spend against this budget.
    pub fn check_turn(&self, turn_cost: f64) -> BudgetCheckResult {
        if turn_cost >= self.max_turn_usd {
            return BudgetCheckResult::Exceeded {
                current: turn_cost as u64,
                limit: self.max_turn_usd as u64,
                fraction: if self.max_turn_usd > 0.0 {
                    turn_cost / self.max_turn_usd
                } else {
                    1.0
                },
            };
        }
        let warning_threshold = self.max_turn_usd * self.warning_fraction;
        if turn_cost >= warning_threshold {
            return BudgetCheckResult::Warning {
                current: turn_cost as u64,
                limit: self.max_turn_usd as u64,
                fraction: if self.max_turn_usd > 0.0 {
                    turn_cost / self.max_turn_usd
                } else {
                    1.0
                },
            };
        }
        BudgetCheckResult::Ok
    }
}

/// A combined budget configuration.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// The token budget.
    pub token: TokenBudget,
    /// The cost budget.
    pub cost: CostBudget,
    /// Whether the budget is enforced (hard cap) or advisory (warning only).
    pub enforce: bool,
    /// Whether to record spend into the pricing cache for cost calculation.
    pub track_cost: bool,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            token: TokenBudget::default(),
            cost: CostBudget::default(),
            enforce: true,
            track_cost: true,
        }
    }
}

impl BudgetConfig {
    /// Create a new budget config with the given token and cost limits.
    pub fn new(max_tokens: u64, max_cost_usd: f64) -> Self {
        Self {
            token: TokenBudget::new(max_tokens),
            cost: CostBudget::new(max_cost_usd),
            ..Default::default()
        }
    }

    /// Set whether the budget is enforced (hard cap).
    pub fn with_enforce(mut self, enforce: bool) -> Self {
        self.enforce = enforce;
        self
    }

    /// Set whether to track cost via the pricing cache.
    pub fn with_cost_tracking(mut self, track: bool) -> Self {
        self.track_cost = track;
        self
    }
}

/// A per-session budget tracker that accumulates token and cost usage.
#[derive(Debug)]
pub struct SessionBudgetTracker {
    /// The session ID.
    session_id: String,
    /// The budget configuration.
    config: BudgetConfig,
    /// Accumulated input tokens.
    input_tokens: AtomicU64,
    /// Accumulated output tokens.
    output_tokens: AtomicU64,
    /// Accumulated total tokens.
    total_tokens: AtomicU64,
    /// Accumulated cost in USD (stored as micro-USD to avoid float atomic).
    cost_micro_usd: AtomicU64,
    /// The number of turns recorded.
    turn_count: AtomicU64,
    /// The pricing cache (optional, for live cost calculation).
    pricing: Option<Arc<PricingCache>>,
    /// Per-model usage breakdown.
    per_model: Mutex<HashMap<String, ModelUsageEntry>>,
    /// The most recent budget check result.
    last_check: Mutex<Option<BudgetCheckResult>>,
    /// The starting time for this session.
    started_at: Instant,
}

/// Per-model usage entry for the budget tracker.
#[derive(Debug, Clone, Default)]
pub struct ModelUsageEntry {
    /// Input tokens for this model.
    pub input_tokens: u64,
    /// Output tokens for this model.
    pub output_tokens: u64,
    /// Total tokens for this model.
    pub total_tokens: u64,
    /// Cost in USD for this model.
    pub cost_usd: f64,
    /// Number of calls to this model.
    pub calls: u64,
}

impl SessionBudgetTracker {
    /// Create a new session budget tracker.
    pub fn new(session_id: impl Into<String>, config: BudgetConfig) -> Self {
        Self {
            session_id: session_id.into(),
            config,
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            cost_micro_usd: AtomicU64::new(0),
            turn_count: AtomicU64::new(0),
            pricing: None,
            per_model: Mutex::new(HashMap::new()),
            last_check: Mutex::new(None),
            started_at: Instant::now(),
        }
    }

    /// Attach a pricing cache for live cost calculation.
    pub fn with_pricing(mut self, pricing: Arc<PricingCache>) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// The session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The budget configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }

    /// Get the accumulated input tokens.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens.load(Ordering::SeqCst)
    }

    /// Get the accumulated output tokens.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens.load(Ordering::SeqCst)
    }

    /// Get the accumulated total tokens.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::SeqCst)
    }

    /// Get the accumulated cost in USD.
    pub fn cost_usd(&self) -> f64 {
        self.cost_micro_usd.load(Ordering::SeqCst) as f64 / 1_000_000.0
    }

    /// Get the number of turns recorded.
    pub fn turn_count(&self) -> u64 {
        self.turn_count.load(Ordering::SeqCst)
    }

    /// Get the session duration.
    pub fn duration(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Record a usage event against this budget.
    pub fn record(&self, event: &UsageEvent) {
        self.input_tokens
            .fetch_add(event.input_tokens, Ordering::SeqCst);
        self.output_tokens
            .fetch_add(event.output_tokens, Ordering::SeqCst);
        self.total_tokens
            .fetch_add(event.total(), Ordering::SeqCst);
        self.turn_count.fetch_add(1, Ordering::SeqCst);

        // Calculate cost.
        let turn_cost = if let Some(pricing) = &self.pricing {
            pricing.cost_for(
                &event.model,
                event.input_tokens,
                event.output_tokens,
            )
        } else {
            // Use a default price of $1/1M input, $3/1M output.
            cost(1.0, 3.0, event.input_tokens, event.output_tokens)
        };
        let cost_micro = (turn_cost * 1_000_000.0) as u64;
        self.cost_micro_usd.fetch_add(cost_micro, Ordering::SeqCst);

        // Update per-model breakdown.
        if let Ok(mut per_model) = self.per_model.lock() {
            let entry = per_model.entry(event.model.clone()).or_default();
            entry.input_tokens += event.input_tokens;
            entry.output_tokens += event.output_tokens;
            entry.total_tokens += event.total();
            entry.cost_usd += turn_cost;
            entry.calls += 1;
        }

        debug!(
            session = %self.session_id,
            model = %event.model,
            input = event.input_tokens,
            output = event.output_tokens,
            turn_cost_usd = turn_cost,
            session_total_tokens = self.total_tokens(),
            session_cost_usd = self.cost_usd(),
            "usage recorded"
        );
    }

    /// Record a usage snapshot (convenience for `Usage` without model info).
    pub fn record_usage(&self, usage: &Usage, model: &str, provider: &str) {
        let event = UsageEvent::new(model, provider, usage.input_tokens, usage.output_tokens);
        self.record(&event);
    }

    /// Check the current token budget.
    pub fn check_token_budget(&self) -> BudgetCheckResult {
        let usage = Usage::new(self.input_tokens(), self.output_tokens());
        let result = self.config.token.check(&usage);
        *self.last_check.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
        result
    }

    /// Check the current cost budget.
    pub fn check_cost_budget(&self) -> BudgetCheckResult {
        let result = self.config.cost.check_session(self.cost_usd());
        *self.last_check.lock().unwrap_or_else(|e| e.into_inner()) = Some(result.clone());
        result
    }

    /// Check both token and cost budgets, returning the more severe result.
    pub fn check(&self) -> BudgetCheckResult {
        let token_check = self.check_token_budget();
        let cost_check = self.check_cost_budget();
        if token_check.is_exceeded() || cost_check.is_exceeded() {
            return BudgetCheckResult::Exceeded {
                current: self.total_tokens(),
                limit: self.config.token.max_total_tokens,
                fraction: self.cost_usd() / self.config.cost.max_session_usd.max(1e-9),
            };
        }
        if token_check.is_warning() || cost_check.is_warning() {
            return BudgetCheckResult::Warning {
                current: self.total_tokens(),
                limit: self.config.token.max_total_tokens,
                fraction: self.cost_usd() / self.config.cost.max_session_usd.max(1e-9),
            };
        }
        BudgetCheckResult::Ok
    }

    /// Whether the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.check().is_exceeded()
    }

    /// Get the per-model usage breakdown.
    pub fn per_model_usage(&self) -> HashMap<String, ModelUsageEntry> {
        self.per_model
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get a usage snapshot.
    pub fn usage(&self) -> Usage {
        Usage::new(self.input_tokens(), self.output_tokens())
    }

    /// Build a structured report of the session budget state.
    pub fn report(&self) -> SessionBudgetReport {
        let check = self.check();
        SessionBudgetReport {
            session_id: self.session_id.clone(),
            input_tokens: self.input_tokens(),
            output_tokens: self.output_tokens(),
            total_tokens: self.total_tokens(),
            cost_usd: self.cost_usd(),
            turn_count: self.turn_count(),
            token_limit: self.config.token.max_total_tokens,
            cost_limit_usd: self.config.cost.max_session_usd,
            budget_status: match check {
                BudgetCheckResult::Ok => "ok".to_string(),
                BudgetCheckResult::Warning { .. } => "warning".to_string(),
                BudgetCheckResult::Exceeded { .. } => "exceeded".to_string(),
            },
            utilization_fraction: if self.config.cost.max_session_usd > 0.0 {
                self.cost_usd() / self.config.cost.max_session_usd
            } else {
                0.0
            },
            per_model: self.per_model_usage(),
        }
    }

    /// Reset all counters to zero (for a new session).
    pub fn reset(&self) {
        self.input_tokens.store(0, Ordering::SeqCst);
        self.output_tokens.store(0, Ordering::SeqCst);
        self.total_tokens.store(0, Ordering::SeqCst);
        self.cost_micro_usd.store(0, Ordering::SeqCst);
        self.turn_count.store(0, Ordering::SeqCst);
        if let Ok(mut per_model) = self.per_model.lock() {
            per_model.clear();
        }
        debug!(session = %self.session_id, "budget tracker reset");
    }
}

/// A structured report of a session's budget state.
#[derive(Debug, Clone)]
pub struct SessionBudgetReport {
    /// The session id.
    pub session_id: String,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Total cost in USD.
    pub cost_usd: f64,
    /// The number of turns recorded.
    pub turn_count: u64,
    /// The token limit.
    pub token_limit: u64,
    /// The cost limit in USD.
    pub cost_limit_usd: f64,
    /// The budget status token (`ok` | `warning` | `exceeded`).
    pub budget_status: String,
    /// The fraction of the cost limit consumed.
    pub utilization_fraction: f64,
    /// Per-model usage.
    pub per_model: HashMap<String, ModelUsageEntry>,
}

impl SessionBudgetReport {
    /// Whether the budget is exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.budget_status == "exceeded"
    }

    /// Whether the budget is in warning territory.
    pub fn is_warning(&self) -> bool {
        self.budget_status == "warning"
    }
}

/// A per-model rate limiter using a token-bucket algorithm.
#[derive(Debug)]
pub struct ModelRateLimiter {
    /// The model being rate-limited.
    model: String,
    /// The maximum requests per minute.
    max_rpm: u32,
    /// The maximum tokens per minute.
    max_tpm: u64,
    /// The token-bucket state.
    state: Mutex<RateLimiterState>,
}

#[derive(Debug)]
struct RateLimiterState {
    /// Available request tokens.
    request_tokens: f64,
    /// Available token-budget tokens.
    token_budget: f64,
    /// The last refill time.
    last_refill: Instant,
}

impl ModelRateLimiter {
    /// Create a new rate limiter for a model.
    pub fn new(model: impl Into<String>, max_rpm: u32, max_tpm: u64) -> Self {
        Self {
            model: model.into(),
            max_rpm,
            max_tpm,
            state: Mutex::new(RateLimiterState {
                request_tokens: max_rpm as f64,
                token_budget: max_tpm as f64,
                last_refill: Instant::now(),
            }),
        }
    }

    /// The model being rate-limited.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The maximum requests per minute.
    pub fn max_rpm(&self) -> u32 {
        self.max_rpm
    }

    /// The maximum tokens per minute.
    pub fn max_tpm(&self) -> u64 {
        self.max_tpm
    }

    /// Refill the buckets based on elapsed time.
    fn refill(&self, state: &mut RateLimiterState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let refill_fraction = elapsed / 60.0;
        state.request_tokens =
            (state.request_tokens + refill_fraction * self.max_rpm as f64).min(self.max_rpm as f64);
        state.token_budget =
            (state.token_budget + refill_fraction * self.max_tpm as f64).min(self.max_tpm as f64);
        state.last_refill = now;
    }

    /// Try to acquire a request slot and token budget. Returns `true` if
    /// the request is allowed, `false` if it should be rate-limited.
    pub fn try_acquire(&self, tokens: u64) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.refill(&mut state);
        if state.request_tokens < 1.0 {
            return false;
        }
        if state.token_budget < tokens as f64 {
            return false;
        }
        state.request_tokens -= 1.0;
        state.token_budget -= tokens as f64;
        true
    }

    /// Wait until a request slot and token budget are available, then
    /// acquire them.
    pub async fn acquire(&self, tokens: u64) {
        loop {
            {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                self.refill(&mut state);
                if state.request_tokens >= 1.0 && state.token_budget >= tokens as f64 {
                    state.request_tokens -= 1.0;
                    state.token_budget -= tokens as f64;
                    return;
                }
            }
            // Wait a short time before retrying.
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// The current available request slots.
    pub fn available_requests(&self) -> f64 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.refill(&mut state);
        state.request_tokens
    }

    /// The current available token budget.
    pub fn available_tokens(&self) -> f64 {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        self.refill(&mut state);
        state.token_budget
    }
}

/// A circuit breaker that tracks provider failures and opens after a
/// threshold.
#[derive(Debug)]
pub struct CircuitBreaker {
    /// The provider/model being protected.
    target: String,
    /// The failure threshold for opening the circuit.
    failure_threshold: usize,
    /// The cooldown duration before the circuit half-opens.
    cooldown: Duration,
    /// The current failure count.
    failures: AtomicUsize,
    /// Whether the circuit is open.
    open: std::sync::atomic::AtomicBool,
    /// When the circuit opened.
    opened_at: Mutex<Option<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(target: impl Into<String>, failure_threshold: usize, cooldown: Duration) -> Self {
        Self {
            target: target.into(),
            failure_threshold: failure_threshold.max(1),
            cooldown,
            failures: AtomicUsize::new(0),
            open: std::sync::atomic::AtomicBool::new(false),
            opened_at: Mutex::new(None),
        }
    }

    /// The target being protected.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Whether the circuit is open (requests should be blocked).
    pub fn is_open(&self) -> bool {
        if self.open.load(Ordering::SeqCst) {
            // Check if the cooldown has passed.
            if let Ok(opened_at) = self.opened_at.lock() {
                if let Some(opened) = *opened_at {
                    if opened.elapsed() >= self.cooldown {
                        // Half-open: allow a test request.
                        self.open.store(false, Ordering::SeqCst);
                        return false;
                    }
                }
            }
            return true;
        }
        false
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        self.failures.store(0, Ordering::SeqCst);
        self.open.store(false, Ordering::SeqCst);
        if let Ok(mut opened_at) = self.opened_at.lock() {
            *opened_at = None;
        }
    }

    /// Record a failed request. Returns `true` if the circuit opened.
    pub fn record_failure(&self) -> bool {
        let count = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.failure_threshold {
            self.open.store(true, Ordering::SeqCst);
            if let Ok(mut opened_at) = self.opened_at.lock() {
                *opened_at = Some(Instant::now());
            }
            warn!(
                target = %self.target,
                failures = count,
                threshold = self.failure_threshold,
                "circuit breaker opened"
            );
            return true;
        }
        false
    }

    /// Reset the circuit breaker.
    pub fn reset(&self) {
        self.failures.store(0, Ordering::SeqCst);
        self.open.store(false, Ordering::SeqCst);
        if let Ok(mut opened_at) = self.opened_at.lock() {
            *opened_at = None;
        }
    }

    /// The current failure count.
    pub fn failure_count(&self) -> usize {
        self.failures.load(Ordering::SeqCst)
    }
}

/// A combined budget manager that coordinates token, cost, rate-limit, and
/// circuit-breaker enforcement for a session.
#[derive(Debug)]
pub struct BudgetManager {
    /// The session budget tracker.
    tracker: SessionBudgetTracker,
    /// Per-model rate limiters.
    rate_limiters: Mutex<HashMap<String, ModelRateLimiter>>,
    /// Per-target circuit breakers.
    circuit_breakers: Mutex<HashMap<String, CircuitBreaker>>,
}

impl BudgetManager {
    /// Create a new budget manager.
    pub fn new(session_id: impl Into<String>, config: BudgetConfig) -> Self {
        Self {
            tracker: SessionBudgetTracker::new(session_id, config),
            rate_limiters: Mutex::new(HashMap::new()),
            circuit_breakers: Mutex::new(HashMap::new()),
        }
    }

    /// Attach a pricing cache.
    pub fn with_pricing(self, pricing: Arc<PricingCache>) -> Self {
        // Rebuild the tracker with pricing.
        let tracker = SessionBudgetTracker::new(self.tracker.session_id(), self.tracker.config().clone())
            .with_pricing(pricing);
        Self {
            tracker,
            rate_limiters: self.rate_limiters,
            circuit_breakers: self.circuit_breakers,
        }
    }

    /// Register a rate limiter for a model.
    pub fn register_rate_limiter(&self, model: &str, max_rpm: u32, max_tpm: u64) {
        let limiter = ModelRateLimiter::new(model, max_rpm, max_tpm);
        self.rate_limiters
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(model.to_string(), limiter);
    }

    /// Register a circuit breaker for a target (provider/model).
    pub fn register_circuit_breaker(
        &self,
        target: &str,
        failure_threshold: usize,
        cooldown: Duration,
    ) {
        let breaker = CircuitBreaker::new(target, failure_threshold, cooldown);
        self.circuit_breakers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(target.to_string(), breaker);
    }

    /// Check whether a request to the given model is allowed by the rate
    /// limiter and circuit breaker.
    pub fn check_request(&self, model: &str, estimated_tokens: u64) -> RequestAdmission {
        // Check circuit breaker.
        let breakers = self.circuit_breakers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(breaker) = breakers.get(model) {
            if breaker.is_open() {
                return RequestAdmission::CircuitOpen(model.to_string());
            }
        }
        drop(breakers);

        // Check rate limiter.
        let limiters = self.rate_limiters.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(limiter) = limiters.get(model) {
            if !limiter.try_acquire(estimated_tokens) {
                return RequestAdmission::RateLimited(model.to_string());
            }
        }
        drop(limiters);

        // Check budget.
        let budget_check = self.tracker.check();
        if budget_check.is_exceeded() {
            return RequestAdmission::BudgetExceeded(model.to_string());
        }

        RequestAdmission::Allowed
    }

    /// Record a successful request.
    pub fn record_success(&self, model: &str) {
        let breakers = self.circuit_breakers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(breaker) = breakers.get(model) {
            breaker.record_success();
        }
    }

    /// Record a failed request.
    pub fn record_failure(&self, model: &str) -> bool {
        let breakers = self.circuit_breakers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(breaker) = breakers.get(model) {
            return breaker.record_failure();
        }
        false
    }

    /// Record a usage event.
    pub fn record_usage(&self, event: &UsageEvent) {
        self.tracker.record(event);
    }

    /// The session budget tracker.
    pub fn tracker(&self) -> &SessionBudgetTracker {
        &self.tracker
    }

    /// Check the overall budget.
    pub fn check_budget(&self) -> BudgetCheckResult {
        self.tracker.check()
    }
}

/// The admission decision for a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequestAdmission {
    /// The request is allowed.
    Allowed,
    /// The circuit breaker is open for this model.
    CircuitOpen(String),
    /// The rate limit has been exceeded for this model.
    RateLimited(String),
    /// The budget has been exceeded.
    BudgetExceeded(String),
}

impl RequestAdmission {
    /// Whether the request is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, RequestAdmission::Allowed)
    }

    /// Whether the request is blocked.
    pub fn is_blocked(&self) -> bool {
        !self.is_allowed()
    }

    /// A human-readable reason for the decision.
    pub fn reason(&self) -> String {
        match self {
            RequestAdmission::Allowed => "allowed".to_string(),
            RequestAdmission::CircuitOpen(m) => format!("circuit breaker open for '{m}'"),
            RequestAdmission::RateLimited(m) => format!("rate limited for '{m}'"),
            RequestAdmission::BudgetExceeded(m) => format!("budget exceeded for '{m}'"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_ok() {
        let budget = TokenBudget::new(100_000);
        let usage = Usage::new(10_000, 5_000);
        assert_eq!(budget.check(&usage), BudgetCheckResult::Ok);
    }

    #[test]
    fn test_token_budget_warning() {
        let budget = TokenBudget::new(100_000).with_warning_fraction(0.8);
        let usage = Usage::new(40_000, 40_001); // total > 80k (80%)
        assert!(budget.check(&usage).is_warning());
    }

    #[test]
    fn test_token_budget_exceeded() {
        let budget = TokenBudget::new(100_000);
        let usage = Usage::new(60_000, 50_000); // total > 100k
        assert!(budget.check(&usage).is_exceeded());
    }

    #[test]
    fn test_cost_budget_ok() {
        let budget = CostBudget::new(10.0);
        assert_eq!(budget.check_session(5.0), BudgetCheckResult::Ok);
    }

    #[test]
    fn test_cost_budget_warning() {
        let budget = CostBudget::new(10.0).with_warning_fraction(0.8);
        assert!(budget.check_session(8.5).is_warning());
    }

    #[test]
    fn test_cost_budget_exceeded() {
        let budget = CostBudget::new(10.0);
        assert!(budget.check_session(10.0).is_exceeded());
    }

    #[tokio::test]
    async fn test_session_budget_tracker() {
        let tracker = SessionBudgetTracker::new("s1", BudgetConfig::new(100_000, 10.0));
        tracker.record(&UsageEvent::new("model-a", "prov", 10_000, 5_000));
        tracker.record(&UsageEvent::new("model-a", "prov", 20_000, 10_000));

        assert_eq!(tracker.input_tokens(), 30_000);
        assert_eq!(tracker.output_tokens(), 15_000);
        assert_eq!(tracker.total_tokens(), 45_000);
        assert_eq!(tracker.turn_count(), 2);
        assert!(tracker.cost_usd() > 0.0);

        let per_model = tracker.per_model_usage();
        let entry = per_model.get("model-a").unwrap();
        assert_eq!(entry.calls, 2);
        assert_eq!(entry.input_tokens, 30_000);
    }

    #[tokio::test]
    async fn test_session_budget_tracker_reset() {
        let tracker = SessionBudgetTracker::new("s1", BudgetConfig::default());
        tracker.record(&UsageEvent::new("m", "p", 100, 50));
        assert_eq!(tracker.total_tokens(), 150);
        tracker.reset();
        assert_eq!(tracker.total_tokens(), 0);
    }

    #[test]
    fn test_model_rate_limiter() {
        let limiter = ModelRateLimiter::new("gpt-4o", 10, 10_000);
        // Should allow 10 requests.
        for _ in 0..10 {
            assert!(limiter.try_acquire(100));
        }
        // The 11th should be rate-limited.
        assert!(!limiter.try_acquire(100));
    }

    #[test]
    fn test_model_rate_limiter_token_budget() {
        let limiter = ModelRateLimiter::new("gpt-4o", 100, 1_000);
        // 10 requests of 100 tokens each = 1000 tokens.
        for _ in 0..10 {
            assert!(limiter.try_acquire(100));
        }
        // Next request exceeds the token budget.
        assert!(!limiter.try_acquire(100));
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let breaker = CircuitBreaker::new("gpt-4o", 3, Duration::from_secs(1));
        assert!(!breaker.is_open());
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_open());
        let opened = breaker.record_failure();
        assert!(opened);
        assert!(breaker.is_open());
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let breaker = CircuitBreaker::new("gpt-4o", 3, Duration::from_secs(1));
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_success();
        assert_eq!(breaker.failure_count(), 0);
        assert!(!breaker.is_open());
    }

    #[test]
    fn test_circuit_breaker_cooldown() {
        let breaker = CircuitBreaker::new("gpt-4o", 1, Duration::from_millis(1));
        breaker.record_failure();
        assert!(breaker.is_open());
        // After cooldown, the circuit should half-open.
        std::thread::sleep(Duration::from_millis(10));
        assert!(!breaker.is_open());
    }

    #[test]
    fn test_budget_manager_check_request_allowed() {
        let manager = BudgetManager::new("s1", BudgetConfig::new(100_000, 10.0));
        let admission = manager.check_request("gpt-4o", 1_000);
        assert!(admission.is_allowed());
    }

    #[test]
    fn test_budget_manager_circuit_open() {
        let manager = BudgetManager::new("s1", BudgetConfig::default());
        manager.register_circuit_breaker("gpt-4o", 1, Duration::from_secs(60));
        manager.record_failure("gpt-4o");
        let admission = manager.check_request("gpt-4o", 100);
        assert!(matches!(admission, RequestAdmission::CircuitOpen(_)));
    }

    #[test]
    fn test_budget_manager_rate_limited() {
        let manager = BudgetManager::new("s1", BudgetConfig::new(1_000_000, 100.0));
        manager.register_rate_limiter("gpt-4o", 2, 1_000_000);
        assert!(manager.check_request("gpt-4o", 100).is_allowed());
        assert!(manager.check_request("gpt-4o", 100).is_allowed());
        // Third request should be rate-limited.
        let admission = manager.check_request("gpt-4o", 100);
        assert!(matches!(admission, RequestAdmission::RateLimited(_)));
    }

    #[test]
    fn test_request_admission_reason() {
        assert_eq!(RequestAdmission::Allowed.reason(), "allowed");
        assert!(RequestAdmission::CircuitOpen("m".to_string()).reason().contains("circuit"));
        assert!(RequestAdmission::RateLimited("m".to_string()).reason().contains("rate"));
        assert!(RequestAdmission::BudgetExceeded("m".to_string()).reason().contains("budget"));
    }

    #[test]
    fn test_budget_config_builder() {
        let config = BudgetConfig::new(100_000, 10.0)
            .with_enforce(false)
            .with_cost_tracking(false);
        assert!(!config.enforce);
        assert!(!config.track_cost);
    }
}
