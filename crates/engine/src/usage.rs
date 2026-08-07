use crate::agent::{ModelUsage, UsageEvent};
use opensquilla_core::types::Usage;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;

/// Thread-safe token usage tracker that accumulates usage across turns.
///
/// UsageTracker uses atomic counters for efficient concurrent access,
/// making it suitable for tracking usage across multiple agents and
/// turns running in parallel.
#[derive(Debug, Clone)]
pub struct UsageTracker {
    /// Total input tokens across all recorded turns.
    total_input_tokens: Arc<AtomicU64>,
    /// Total output tokens across all recorded turns.
    total_output_tokens: Arc<AtomicU64>,
    /// Total tokens across all recorded turns.
    total_tokens: Arc<AtomicU64>,
    /// The number of turns recorded.
    turn_count: Arc<AtomicU64>,
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageTracker {
    /// Create a new, empty UsageTracker.
    pub fn new() -> Self {
        Self {
            total_input_tokens: Arc::new(AtomicU64::new(0)),
            total_output_tokens: Arc::new(AtomicU64::new(0)),
            total_tokens: Arc::new(AtomicU64::new(0)),
            turn_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a usage snapshot from a completed turn.
    pub fn record(&self, usage: &Usage) {
        self.total_input_tokens
            .fetch_add(usage.input_tokens, Ordering::SeqCst);
        self.total_output_tokens
            .fetch_add(usage.output_tokens, Ordering::SeqCst);
        self.total_tokens
            .fetch_add(usage.total_tokens, Ordering::SeqCst);
        self.turn_count.fetch_add(1, Ordering::SeqCst);

        debug!(
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            total_tokens = usage.total_tokens,
            total_turns = self.turn_count(),
            "Usage recorded"
        );
    }

    /// Record a usage event (model/provider scoped) against this tracker.
    pub fn record_event(&self, event: &UsageEvent) {
        self.total_input_tokens
            .fetch_add(event.input_tokens, Ordering::SeqCst);
        self.total_output_tokens
            .fetch_add(event.output_tokens, Ordering::SeqCst);
        self.total_tokens.fetch_add(event.total(), Ordering::SeqCst);
        self.turn_count.fetch_add(1, Ordering::SeqCst);
        debug!(
            model = %event.model,
            provider = %event.provider,
            input_tokens = event.input_tokens,
            output_tokens = event.output_tokens,
            "Usage event recorded"
        );
    }

    /// Record a batch of usage events.
    pub fn record_events(&self, events: &[UsageEvent]) {
        for event in events {
            self.record_event(event);
        }
    }

    /// Get the total input tokens across all recorded turns.
    pub fn total_input_tokens(&self) -> u64 {
        self.total_input_tokens.load(Ordering::SeqCst)
    }

    /// Get the total output tokens across all recorded turns.
    pub fn total_output_tokens(&self) -> u64 {
        self.total_output_tokens.load(Ordering::SeqCst)
    }

    /// Get the total tokens across all recorded turns.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::SeqCst)
    }

    /// Get the number of turns recorded.
    pub fn turn_count(&self) -> u64 {
        self.turn_count.load(Ordering::SeqCst)
    }

    /// Get the current usage as a Usage struct.
    pub fn current(&self) -> Usage {
        Usage {
            input_tokens: self.total_input_tokens(),
            output_tokens: self.total_output_tokens(),
            total_tokens: self.total_tokens(),
        }
    }

    /// Get the average usage per turn.
    pub fn average_per_turn(&self) -> Option<Usage> {
        let count = self.turn_count();
        if count == 0 {
            return None;
        }
        Some(Usage {
            input_tokens: self.total_input_tokens() / count,
            output_tokens: self.total_output_tokens() / count,
            total_tokens: self.total_tokens() / count,
        })
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.total_input_tokens.store(0, Ordering::SeqCst);
        self.total_output_tokens.store(0, Ordering::SeqCst);
        self.total_tokens.store(0, Ordering::SeqCst);
        self.turn_count.store(0, Ordering::SeqCst);
        debug!("Usage tracker reset");
    }

    /// Merge another UsageTracker's counts into this one.
    pub fn merge(&self, other: &UsageTracker) {
        self.total_input_tokens
            .fetch_add(other.total_input_tokens(), Ordering::SeqCst);
        self.total_output_tokens
            .fetch_add(other.total_output_tokens(), Ordering::SeqCst);
        self.total_tokens
            .fetch_add(other.total_tokens(), Ordering::SeqCst);
        self.turn_count
            .fetch_add(other.turn_count(), Ordering::SeqCst);
    }
}

/// The usage-event sink protocol, mirroring the Python
/// `usage_accounting.UsageEventSink`.
///
/// Anything that can consume usage events (trackers, ledgers, loggers)
/// implements this trait. The engine binds a sink into the per-task scope so
/// provider calls made anywhere on the task record into the correct account.
pub trait UsageEventSink: Send + Sync + fmt::Debug {
    /// Record a single usage event.
    fn record(&self, event: &UsageEvent);

    /// Total input tokens observed by this sink.
    fn total_input_tokens(&self) -> u64;

    /// Total output tokens observed by this sink.
    fn total_output_tokens(&self) -> u64;

    /// Total tokens observed by this sink.
    fn total_tokens(&self) -> u64;
}

impl UsageEventSink for UsageTracker {
    fn record(&self, event: &UsageEvent) {
        self.record_event(event);
    }

    fn total_input_tokens(&self) -> u64 {
        self.total_input_tokens()
    }

    fn total_output_tokens(&self) -> u64 {
        self.total_output_tokens()
    }

    fn total_tokens(&self) -> u64 {
        self.total_tokens()
    }
}

/// A sink that discards every event. Used as the default scope binding when no
/// accounting is configured.
#[derive(Debug, Clone, Default)]
pub struct NoopUsageSink;

impl UsageEventSink for NoopUsageSink {
    fn record(&self, _event: &UsageEvent) {}
    fn total_input_tokens(&self) -> u64 {
        0
    }
    fn total_output_tokens(&self) -> u64 {
        0
    }
    fn total_tokens(&self) -> u64 {
        0
    }
}

/// A composite sink that fans events out to multiple downstream sinks.
#[derive(Debug, Clone)]
pub struct FanOutUsageSink {
    sinks: Vec<Arc<dyn UsageEventSink>>,
}

impl FanOutUsageSink {
    /// Create a fan-out sink over the given downstream sinks.
    pub fn new(sinks: Vec<Arc<dyn UsageEventSink>>) -> Self {
        Self { sinks }
    }

    /// The number of downstream sinks.
    pub fn len(&self) -> usize {
        self.sinks.len()
    }

    /// True when no downstream sinks are attached.
    pub fn is_empty(&self) -> bool {
        self.sinks.is_empty()
    }
}

impl UsageEventSink for FanOutUsageSink {
    fn record(&self, event: &UsageEvent) {
        for sink in &self.sinks {
            sink.record(event);
        }
    }

    fn total_input_tokens(&self) -> u64 {
        self.sinks.iter().map(|s| s.total_input_tokens()).sum()
    }

    fn total_output_tokens(&self) -> u64 {
        self.sinks.iter().map(|s| s.total_output_tokens()).sum()
    }

    fn total_tokens(&self) -> u64 {
        self.sinks.iter().map(|s| s.total_tokens()).sum()
    }
}

// Per-task usage sink binding.
//
// Mirrors the Python `usage.py` `ContextVar` scope binding with
// `tokio::task_local!`: a sink bound inside a task scope is visible to every
// descendant task/await point until the scope exits.
tokio::task_local! {
    static CURRENT_USAGE_SINK: Arc<dyn UsageEventSink>;
}

/// Run a future with the given usage sink bound for the whole task scope.
///
/// All usage recorded via [`record_in_current_scope`] while the future runs
/// (including in spawned descendant tasks that inherit the binding) flows into
/// `sink`.
pub async fn with_usage_scope<F, Fut, T>(sink: Arc<dyn UsageEventSink>, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    CURRENT_USAGE_SINK.scope(sink, f()).await
}

/// Record a usage snapshot into the currently bound usage sink.
///
/// When no sink is bound (outside a [`with_usage_scope`]) the record is a no-op.
pub fn record_in_current_scope(usage: &Usage) {
    let _ = CURRENT_USAGE_SINK.try_with(|sink| {
        sink.record(&UsageEvent::new(
            "unknown".to_string(),
            "unknown".to_string(),
            usage.input_tokens,
            usage.output_tokens,
        ))
    });
}

/// The sink bound to the current task scope, if any.
pub fn current_usage_sink() -> Option<Arc<dyn UsageEventSink>> {
    CURRENT_USAGE_SINK.try_with(|s| s.clone()).ok()
}

/// Per-provider-call accounting.
///
/// Records each provider call's token usage into a bound sink, tagging the
/// event with the model and provider that served the call.
#[derive(Debug, Clone)]
pub struct ProviderCallAccountant {
    sink: Arc<dyn UsageEventSink>,
    model: String,
    provider: String,
}

impl ProviderCallAccountant {
    /// Create an accountant that records into `sink` tagged with `model` and
    /// `provider`.
    pub fn new(
        sink: Arc<dyn UsageEventSink>,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            sink,
            model: model.into(),
            provider: provider.into(),
        }
    }

    /// Record a single provider call.
    pub fn record_call(&self, input_tokens: u64, output_tokens: u64) {
        self.sink.record(&UsageEvent::new(
            self.model.clone(),
            self.provider.clone(),
            input_tokens,
            output_tokens,
        ));
    }

    /// Record a call from a `Usage` snapshot.
    pub fn record_usage(&self, usage: &Usage) {
        self.record_call(usage.input_tokens, usage.output_tokens);
    }

    /// The model this accountant tags events with.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider this accountant tags events with.
    pub fn provider(&self) -> &str {
        &self.provider
    }
}

/// Per-model usage aggregation, tracking input/output tokens and call counts
/// broken down by model id.
#[derive(Debug, Default)]
pub struct PerModelUsageTracker {
    inner: std::sync::Mutex<HashMap<String, ModelUsage>>,
}

impl PerModelUsageTracker {
    /// Create a new empty per-model tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a usage event under its model.
    pub fn record(&self, event: &UsageEvent) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let entry = inner.entry(event.model.clone()).or_default();
        entry.input_tokens += event.input_tokens;
        entry.output_tokens += event.output_tokens;
        entry.calls += 1;
    }

    /// Get the usage recorded for a model, if any.
    pub fn get(&self, model: &str) -> Option<ModelUsage> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
            .cloned()
    }

    /// Aggregate the per-model counts into a single `Usage`.
    pub fn aggregate(&self) -> Usage {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut input = 0u64;
        let mut output = 0u64;
        for usage in inner.values() {
            input += usage.input_tokens;
            output += usage.output_tokens;
        }
        Usage::new(input, output)
    }

    /// All model ids with recorded usage.
    pub fn models(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// The number of models tracked.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// True when no models are tracked.
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

/// A usage tracker that also tracks cost based on model pricing.
#[derive(Debug, Clone)]
pub struct CostTrackingUsageTracker {
    /// The underlying usage tracker.
    inner: UsageTracker,
    /// Price per 1,000 input tokens in USD.
    input_price_per_1k: f64,
    /// Price per 1,000 output tokens in USD.
    output_price_per_1k: f64,
}

impl CostTrackingUsageTracker {
    /// Create a new cost-tracking usage tracker with the given pricing.
    pub fn new(input_price_per_1k: f64, output_price_per_1k: f64) -> Self {
        Self {
            inner: UsageTracker::new(),
            input_price_per_1k,
            output_price_per_1k,
        }
    }

    /// Record a usage snapshot and track its cost.
    pub fn record(&self, usage: &Usage) {
        self.inner.record(usage);
    }

    /// Record a usage event and track its cost.
    pub fn record_event(&self, event: &UsageEvent) {
        self.inner.record_event(event);
    }

    /// Calculate the total cost in USD for all recorded usage.
    pub fn total_cost_usd(&self) -> f64 {
        let input_cost =
            (self.inner.total_input_tokens() as f64 / 1000.0) * self.input_price_per_1k;
        let output_cost =
            (self.inner.total_output_tokens() as f64 / 1000.0) * self.output_price_per_1k;
        input_cost + output_cost
    }

    /// Get the cost for a specific usage snapshot.
    pub fn cost_for(&self, usage: &Usage) -> f64 {
        let input_cost = (usage.input_tokens as f64 / 1000.0) * self.input_price_per_1k;
        let output_cost = (usage.output_tokens as f64 / 1000.0) * self.output_price_per_1k;
        input_cost + output_cost
    }

    /// Get a reference to the underlying usage tracker.
    pub fn inner(&self) -> &UsageTracker {
        &self.inner
    }
}

impl UsageEventSink for CostTrackingUsageTracker {
    fn record(&self, event: &UsageEvent) {
        self.record_event(event);
    }

    fn total_input_tokens(&self) -> u64 {
        self.inner.total_input_tokens()
    }

    fn total_output_tokens(&self) -> u64 {
        self.inner.total_output_tokens()
    }

    fn total_tokens(&self) -> u64 {
        self.inner.total_tokens()
    }
}

// ---------------------------------------------------------------------------
// Per-session / per-agent rollups and cost aggregation
// ---------------------------------------------------------------------------

/// A usage rollup scoped to a single session.
///
/// Mirrors the Python `usage_accounting.SessionUsageRollup`: aggregates token
/// usage, estimated cost, and per-model breakdown for one session.
#[derive(Debug, Clone, Default)]
pub struct SessionUsageRollup {
    /// The session id.
    pub session_id: String,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Estimated cost in USD.
    pub cost_usd: f64,
    /// The number of turns recorded.
    pub turn_count: u64,
    /// Per-model usage.
    pub per_model: HashMap<String, crate::agent::ModelUsage>,
}

impl SessionUsageRollup {
    /// Create a new empty rollup for a session.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    /// Record a usage event into this rollup.
    pub fn record(&mut self, event: &UsageEvent) {
        self.input_tokens += event.input_tokens;
        self.output_tokens += event.output_tokens;
        self.total_tokens += event.total();
        self.turn_count += 1;
        let entry = self.per_model.entry(event.model.clone()).or_default();
        entry.input_tokens += event.input_tokens;
        entry.output_tokens += event.output_tokens;
        entry.calls += 1;
    }

    /// Set the cost in USD.
    pub fn with_cost(mut self, cost_usd: f64) -> Self {
        self.cost_usd = cost_usd;
        self
    }

    /// Convert to a core [`Usage`] snapshot.
    pub fn to_usage(&self) -> Usage {
        Usage::new(self.input_tokens, self.output_tokens)
    }
}

/// A usage rollup scoped to a single agent across sessions.
#[derive(Debug, Clone, Default)]
pub struct AgentUsageRollup {
    /// The agent id.
    pub agent_id: String,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
    /// Estimated cost in USD.
    pub cost_usd: f64,
    /// The number of turns recorded.
    pub turn_count: u64,
    /// Per-model usage.
    pub per_model: HashMap<String, crate::agent::ModelUsage>,
    /// Per-session rollups.
    pub sessions: HashMap<String, SessionUsageRollup>,
}

impl AgentUsageRollup {
    /// Create a new empty rollup for an agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            ..Default::default()
        }
    }

    /// Record a usage event against the agent and (optionally) a session.
    pub fn record(&mut self, event: &UsageEvent, session_id: Option<&str>) {
        self.input_tokens += event.input_tokens;
        self.output_tokens += event.output_tokens;
        self.total_tokens += event.total();
        self.turn_count += 1;
        let entry = self.per_model.entry(event.model.clone()).or_default();
        entry.input_tokens += event.input_tokens;
        entry.output_tokens += event.output_tokens;
        entry.calls += 1;

        if let Some(session_id) = session_id {
            let session = self
                .sessions
                .entry(session_id.to_string())
                .or_insert_with(|| SessionUsageRollup::new(session_id));
            session.record(event);
        }
    }

    /// Merge another agent rollup into this one.
    pub fn merge(&mut self, other: &AgentUsageRollup) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
        self.cost_usd += other.cost_usd;
        self.turn_count += other.turn_count;
        for (model, usage) in &other.per_model {
            let entry = self.per_model.entry(model.clone()).or_default();
            entry.input_tokens += usage.input_tokens;
            entry.output_tokens += usage.output_tokens;
            entry.calls += usage.calls;
        }
        for (session_id, rollup) in &other.sessions {
            let session = self
                .sessions
                .entry(session_id.clone())
                .or_insert_with(|| SessionUsageRollup::new(session_id));
            session.input_tokens += rollup.input_tokens;
            session.output_tokens += rollup.output_tokens;
            session.total_tokens += rollup.total_tokens;
            session.cost_usd += rollup.cost_usd;
            session.turn_count += rollup.turn_count;
        }
    }

    /// Convert to a core [`Usage`] snapshot.
    pub fn to_usage(&self) -> Usage {
        Usage::new(self.input_tokens, self.output_tokens)
    }
}

/// A thread-safe registry of session usage rollups.
#[derive(Debug, Clone, Default)]
pub struct SessionUsageRegistry {
    /// The rollups keyed by session id.
    inner: std::sync::Arc<std::sync::Mutex<HashMap<String, SessionUsageRollup>>>,
}

impl SessionUsageRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an event against a session.
    pub fn record(&self, session_id: &str, event: &UsageEvent) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let rollup = inner
            .entry(session_id.to_string())
            .or_insert_with(|| SessionUsageRollup::new(session_id));
        rollup.record(event);
    }

    /// Get the rollup for a session.
    pub fn get(&self, session_id: &str) -> Option<SessionUsageRollup> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(session_id)
            .cloned()
    }

    /// List all tracked session ids.
    pub fn session_ids(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// Aggregate all sessions into a single rollup.
    pub fn aggregate(&self) -> SessionUsageRollup {
        let mut total = SessionUsageRollup::new("all");
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for rollup in inner.values() {
            total.input_tokens += rollup.input_tokens;
            total.output_tokens += rollup.output_tokens;
            total.total_tokens += rollup.total_tokens;
            total.cost_usd += rollup.cost_usd;
            total.turn_count += rollup.turn_count;
            for (model, usage) in &rollup.per_model {
                let entry = total.per_model.entry(model.clone()).or_default();
                entry.input_tokens += usage.input_tokens;
                entry.output_tokens += usage.output_tokens;
                entry.calls += usage.calls;
            }
        }
        total
    }

    /// Remove a session rollup.
    pub fn remove(&self, session_id: &str) -> Option<SessionUsageRollup> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(session_id)
    }

    /// The number of tracked sessions.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// True when no sessions are tracked.
    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

/// A usage scope that ties a task-local sink to an optional session id.
///
/// When a session id is bound, events recorded in the scope also fan into
/// that session's rollup in the registry.
#[derive(Debug, Clone)]
pub struct UsageScope {
    /// The underlying sink.
    sink: Arc<dyn UsageEventSink>,
    /// The session id, if any.
    session_id: Option<String>,
}

impl UsageScope {
    /// Create a new usage scope.
    pub fn new(sink: Arc<dyn UsageEventSink>, session_id: Option<String>) -> Self {
        Self { sink, session_id }
    }

    /// The underlying sink.
    pub fn sink(&self) -> &Arc<dyn UsageEventSink> {
        &self.sink
    }

    /// The session id, if bound.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Record an event through this scope.
    pub fn record(&self, event: &UsageEvent) {
        self.sink.record(event);
    }
}

/// A usage scope that also fans events into a session registry.
#[derive(Debug, Clone)]
pub struct SessionScopedSink {
    /// The downstream sink.
    inner: Arc<dyn UsageEventSink>,
    /// The session registry.
    registry: SessionUsageRegistry,
    /// The session id.
    session_id: String,
}

impl SessionScopedSink {
    /// Create a new session-scoped sink.
    pub fn new(
        inner: Arc<dyn UsageEventSink>,
        registry: SessionUsageRegistry,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            registry,
            session_id: session_id.into(),
        }
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl UsageEventSink for SessionScopedSink {
    fn record(&self, event: &UsageEvent) {
        self.inner.record(event);
        self.registry.record(&self.session_id, event);
    }

    fn total_input_tokens(&self) -> u64 {
        self.inner.total_input_tokens()
    }

    fn total_output_tokens(&self) -> u64 {
        self.inner.total_output_tokens()
    }

    fn total_tokens(&self) -> u64 {
        self.inner.total_tokens()
    }
}

/// A cost-aggregating sink that computes USD cost using the pricing cache.
#[derive(Debug, Clone)]
pub struct CostAggregatingSink {
    /// The downstream sink.
    inner: Arc<dyn UsageEventSink>,
    /// The pricing cache used for cost calculation.
    pricing: Arc<crate::pricing::PricingCache>,
    /// The accumulated cost in micro-USD (atomic).
    cost_micro_usd: Arc<std::sync::atomic::AtomicU64>,
}

impl CostAggregatingSink {
    /// Create a new cost-aggregating sink.
    pub fn new(inner: Arc<dyn UsageEventSink>, pricing: Arc<crate::pricing::PricingCache>) -> Self {
        Self {
            inner,
            pricing,
            cost_micro_usd: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// The accumulated cost in USD.
    pub fn total_cost_usd(&self) -> f64 {
        self.cost_micro_usd
            .load(std::sync::atomic::Ordering::SeqCst) as f64
            / 1_000_000.0
    }
}

impl UsageEventSink for CostAggregatingSink {
    fn record(&self, event: &UsageEvent) {
        let cost = self
            .pricing
            .cost_for(&event.model, event.input_tokens, event.output_tokens);
        self.cost_micro_usd.fetch_add(
            (cost * 1_000_000.0) as u64,
            std::sync::atomic::Ordering::SeqCst,
        );
        self.inner.record(event);
    }

    fn total_input_tokens(&self) -> u64 {
        self.inner.total_input_tokens()
    }

    fn total_output_tokens(&self) -> u64 {
        self.inner.total_output_tokens()
    }

    fn total_tokens(&self) -> u64 {
        self.inner.total_tokens()
    }
}

/// Record a usage event into the currently bound scope with model/provider
/// tagging, mirroring `usage_accounting.record_event`.
pub fn record_event_in_scope(event: &UsageEvent) {
    let _ = CURRENT_USAGE_SINK.try_with(|sink| sink.record(event));
}

/// Record a provider call (input/output tokens) into the current scope with
/// model and provider tags.
pub fn record_provider_call(model: &str, provider: &str, input_tokens: u64, output_tokens: u64) {
    let event = UsageEvent::new(model, provider, input_tokens, output_tokens);
    record_event_in_scope(&event);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(model: &str, input: u64, output: u64) -> UsageEvent {
        UsageEvent::new(model, "provider-x", input, output)
    }

    #[test]
    fn test_usage_tracker_new() {
        let tracker = UsageTracker::new();
        assert_eq!(tracker.total_input_tokens(), 0);
        assert_eq!(tracker.total_output_tokens(), 0);
        assert_eq!(tracker.total_tokens(), 0);
        assert_eq!(tracker.turn_count(), 0);
    }

    #[test]
    fn test_usage_tracker_record() {
        let tracker = UsageTracker::new();
        let usage = Usage::new(100, 50);
        tracker.record(&usage);

        assert_eq!(tracker.total_input_tokens(), 100);
        assert_eq!(tracker.total_output_tokens(), 50);
        assert_eq!(tracker.total_tokens(), 150);
        assert_eq!(tracker.turn_count(), 1);
    }

    #[test]
    fn test_usage_tracker_multiple_records() {
        let tracker = UsageTracker::new();
        tracker.record(&Usage::new(100, 50));
        tracker.record(&Usage::new(200, 100));

        assert_eq!(tracker.total_input_tokens(), 300);
        assert_eq!(tracker.total_output_tokens(), 150);
        assert_eq!(tracker.total_tokens(), 450);
        assert_eq!(tracker.turn_count(), 2);
    }

    #[test]
    fn test_usage_tracker_current() {
        let tracker = UsageTracker::new();
        tracker.record(&Usage::new(100, 50));

        let current = tracker.current();
        assert_eq!(current.input_tokens, 100);
        assert_eq!(current.output_tokens, 50);
        assert_eq!(current.total_tokens, 150);
    }

    #[test]
    fn test_usage_tracker_average() {
        let tracker = UsageTracker::new();
        tracker.record(&Usage::new(100, 50));
        tracker.record(&Usage::new(200, 100));

        let avg = tracker.average_per_turn().unwrap();
        assert_eq!(avg.input_tokens, 150);
        assert_eq!(avg.output_tokens, 75);
        assert_eq!(avg.total_tokens, 225);
    }

    #[test]
    fn test_usage_tracker_average_empty() {
        let tracker = UsageTracker::new();
        assert!(tracker.average_per_turn().is_none());
    }

    #[test]
    fn test_usage_tracker_reset() {
        let tracker = UsageTracker::new();
        tracker.record(&Usage::new(100, 50));
        tracker.reset();

        assert_eq!(tracker.total_input_tokens(), 0);
        assert_eq!(tracker.turn_count(), 0);
    }

    #[test]
    fn test_usage_tracker_merge() {
        let tracker1 = UsageTracker::new();
        tracker1.record(&Usage::new(100, 50));

        let tracker2 = UsageTracker::new();
        tracker2.record(&Usage::new(200, 100));

        tracker1.merge(&tracker2);
        assert_eq!(tracker1.total_input_tokens(), 300);
        assert_eq!(tracker1.total_output_tokens(), 150);
        assert_eq!(tracker1.turn_count(), 2);
    }

    #[test]
    fn test_record_event_and_sink() {
        let tracker = UsageTracker::new();
        tracker.record_events(&[event("m1", 10, 5), event("m2", 20, 10)]);
        assert_eq!(tracker.total_input_tokens(), 30);
        assert_eq!(tracker.total_output_tokens(), 15);
        assert_eq!(tracker.turn_count(), 2);

        let sink: Arc<dyn UsageEventSink> = Arc::new(tracker);
        assert_eq!(sink.total_tokens(), 45);
    }

    #[test]
    fn test_noop_sink() {
        let sink = NoopUsageSink;
        sink.record(&event("m", 100, 100));
        assert_eq!(sink.total_tokens(), 0);
    }

    #[test]
    fn test_fan_out_sink() {
        let t1 = Arc::new(UsageTracker::new());
        let t2 = Arc::new(UsageTracker::new());
        let fan = FanOutUsageSink::new(vec![t1.clone(), t2.clone()]);
        fan.record(&event("m", 50, 25));
        assert_eq!(t1.total_tokens(), 75);
        assert_eq!(t2.total_tokens(), 75);
        assert_eq!(fan.total_tokens(), 150);
    }

    #[tokio::test]
    async fn test_usage_scope_binds_and_records() {
        let tracker = Arc::new(UsageTracker::new());
        let sink: Arc<dyn UsageEventSink> = tracker.clone();
        let total = with_usage_scope(sink, || async {
            record_in_current_scope(&Usage::new(100, 50));
            current_usage_sink().map(|s| s.total_tokens()).unwrap_or(0)
        })
        .await;
        assert_eq!(total, 150);
        assert_eq!(tracker.total_input_tokens(), 100);
        assert_eq!(tracker.total_output_tokens(), 50);
    }

    #[tokio::test]
    async fn test_usage_scope_unbound_noop() {
        record_in_current_scope(&Usage::new(100, 50));
        assert!(current_usage_sink().is_none());
    }

    #[test]
    fn test_provider_call_accountant() {
        let tracker = Arc::new(UsageTracker::new());
        let accountant = ProviderCallAccountant::new(tracker.clone(), "model-a", "provider-a");
        accountant.record_call(10, 20);
        assert_eq!(tracker.total_input_tokens(), 10);
        assert_eq!(tracker.total_output_tokens(), 20);
        assert_eq!(accountant.model(), "model-a");
        assert_eq!(accountant.provider(), "provider-a");
    }

    #[test]
    fn test_per_model_tracker() {
        let per = PerModelUsageTracker::new();
        per.record(&event("m1", 10, 5));
        per.record(&event("m1", 20, 10));
        per.record(&event("m2", 30, 15));
        assert_eq!(per.len(), 2);
        let m1 = per.get("m1").unwrap();
        assert_eq!(m1.input_tokens, 30);
        assert_eq!(m1.output_tokens, 15);
        assert_eq!(m1.calls, 2);
        let agg = per.aggregate();
        assert_eq!(agg.input_tokens, 60);
        assert_eq!(agg.output_tokens, 30);
    }

    #[test]
    fn test_cost_tracking() {
        let tracker = CostTrackingUsageTracker::new(0.01, 0.03);
        tracker.record(&Usage::new(1000, 500));

        // 1000 input tokens at $0.01/1k = $0.01
        // 500 output tokens at $0.03/1k = $0.015
        // Total = $0.025
        let cost = tracker.total_cost_usd();
        assert!((cost - 0.025).abs() < 1e-10);
    }
}
