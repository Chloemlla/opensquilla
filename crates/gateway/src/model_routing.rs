//! Model routing.
//!
//! Mirrors the Python `model_routing.py` module. Routes a request to the
//! appropriate model based on routing rules, supports per-session routing
//! holds, logs every routing decision, and follows a fallback chain when the
//! preferred model is unavailable.
//!
//! Decisions and holds are persisted through the gateway's existing
//! [`crate::routing::RoutingStore`], which also powers the `routing.hold` and
//! `router.record` RPC handlers.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::routing::{RoutingDecision, RoutingHold, RoutingStore};

/// The strategy used to make a routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    /// The session has an active hold; pinned to the held model.
    Hold,
    /// The client explicitly requested a model.
    Explicit,
    /// The first available model in the fallback chain.
    FallbackChain,
    /// The configured default model.
    Default,
}

/// A routing rule that can steer requests to a model/provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Rule name (for diagnostics).
    pub name: String,
    /// Optional glob pattern to match the session id (e.g. `"project-*"`).
    pub session_pattern: Option<String>,
    /// Optional substring match on the requested model.
    pub requested_model_contains: Option<String>,
    /// The model to route to when the rule matches.
    pub model: String,
    /// The provider to use, if the caller wants to pin one.
    pub provider: Option<String>,
    /// Rule priority; higher wins when multiple rules match.
    pub priority: i32,
}

impl RoutingRule {
    /// Return `true` if this rule applies to the given request.
    pub fn matches(&self, session_id: &str, requested_model: Option<&str>) -> bool {
        if let Some(pattern) = &self.session_pattern {
            if !glob_match(pattern, session_id) {
                return false;
            }
        }
        if let Some(needle) = &self.requested_model_contains {
            let model = requested_model.unwrap_or("");
            if !model.contains(needle.as_str()) {
                return false;
            }
        }
        true
    }
}

/// Configuration for the model router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRouterConfig {
    /// The default model used when nothing else applies.
    pub default_model: String,
    /// Ordered fallback chain tried when the selected model is unavailable.
    pub fallback_chain: Vec<String>,
    /// Routing rules evaluated in priority order.
    pub rules: Vec<RoutingRule>,
}

impl Default for ModelRouterConfig {
    fn default() -> Self {
        Self {
            default_model: "gpt-4o".to_string(),
            fallback_chain: vec!["gpt-4o-mini".to_string()],
            rules: Vec::new(),
        }
    }
}

/// An inbound request to route.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    pub session_id: String,
    pub requested_model: Option<String>,
    /// What the model will be used for (e.g. "chat", "tool", "compaction").
    pub purpose: String,
}

/// The full routing decision for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteOutcome {
    pub session_id: String,
    pub requested_model: Option<String>,
    pub selected_model: String,
    pub selected_provider: Option<String>,
    pub strategy: RoutingStrategy,
    pub rule_used: Option<String>,
    pub fallback_used: bool,
    pub reason: String,
    pub decided_at: DateTime<Utc>,
}

/// The model router.
///
/// Thread-safe and clone-friendly. Holds a shared `RoutingStore` (also used
/// by the RPC layer) plus an in-memory list of routing rules.
#[derive(Clone)]
pub struct ModelRouter {
    config: Arc<RwLock<ModelRouterConfig>>,
    store: RoutingStore,
    /// Ordered list of provider capability callbacks used to test whether a
    /// model is available. The fallback chain consults this before selecting
    /// a fallback model.
    availability: Arc<Vec<Arc<dyn ModelAvailability>>>,
}

/// Determines whether a model is currently available to serve a request.
pub trait ModelAvailability: Send + Sync {
    /// Return `true` if the model is available.
    fn is_available(&self, model: &str) -> bool;
}

/// An availability check that accepts everything.
pub struct AlwaysAvailable;

impl ModelAvailability for AlwaysAvailable {
    fn is_available(&self, _model: &str) -> bool {
        true
    }
}

impl ModelRouter {
    /// Create a new router with the given config and an empty store.
    pub fn new(config: ModelRouterConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            store: RoutingStore::new(),
            availability: Arc::new(Vec::new()),
        }
    }

    /// Attach an availability checker (for fallback-chain testing).
    pub fn with_availability(mut self, checker: Arc<dyn ModelAvailability>) -> Self {
        Arc::make_mut(&mut self.availability).push(checker);
        self
    }

    /// Replace the routing configuration.
    pub fn set_config(&self, config: ModelRouterConfig) {
        *self.config.write() = config;
    }

    /// Return a snapshot of the current configuration.
    pub fn config(&self) -> ModelRouterConfig {
        self.config.read().clone()
    }

    /// Route a request to a model.
    ///
    /// Resolution order:
    /// 1. Active per-session hold (from `RoutingStore`).
    /// 2. Highest-priority matching rule.
    /// 3. Explicitly requested model (if available).
    /// 4. First available model in the fallback chain.
    /// 5. The configured default model.
    ///
    /// Every decision is recorded in the `RoutingStore`.
    pub fn route(&self, request: &RouteRequest) -> RouteOutcome {
        let config = self.config.read();

        // 1. Per-session hold.
        if let Some(hold) = self.store.get_hold(&request.session_id) {
            let expired = hold
                .expires_at
                .map(|exp| Utc::now() > exp)
                .unwrap_or(false);
            if !expired {
                let outcome = RouteOutcome {
                    session_id: request.session_id.clone(),
                    requested_model: request.requested_model.clone(),
                    selected_model: hold.model.clone(),
                    selected_provider: hold.provider.clone(),
                    strategy: RoutingStrategy::Hold,
                    rule_used: None,
                    fallback_used: false,
                    reason: format!("Active routing hold: {}", hold.reason),
                    decided_at: Utc::now(),
                };
                self.record(&outcome);
                return outcome;
            } else {
                // Expired hold: release it.
                self.store.release_hold(&request.session_id);
            }
        }

        // 2. Routing rules (highest priority first).
        if let Some((rule, _)) = self.best_rule(request) {
            let outcome = RouteOutcome {
                session_id: request.session_id.clone(),
                requested_model: request.requested_model.clone(),
                selected_model: rule.model.clone(),
                selected_provider: rule.provider.clone(),
                strategy: RoutingStrategy::Explicit,
                rule_used: Some(rule.name.clone()),
                fallback_used: false,
                reason: format!(
                    "Matched routing rule '{}' (priority {})",
                    rule.name, rule.priority
                ),
                decided_at: Utc::now(),
            };
            self.record(&outcome);
            return outcome;
        }

        // 3. Explicit request (if the model is available).
        if let Some(model) = &request.requested_model {
            if self.is_available(model) {
                let outcome = RouteOutcome {
                    session_id: request.session_id.clone(),
                    requested_model: request.requested_model.clone(),
                    selected_model: model.clone(),
                    selected_provider: None,
                    strategy: RoutingStrategy::Explicit,
                    rule_used: None,
                    fallback_used: false,
                    reason: "Client requested model and it is available".to_string(),
                    decided_at: Utc::now(),
                };
                self.record(&outcome);
                return outcome;
            }
        }

        // 4. Fallback chain.
        for fallback in &config.fallback_chain {
            if self.is_available(fallback) {
                let outcome = RouteOutcome {
                    session_id: request.session_id.clone(),
                    requested_model: request.requested_model.clone(),
                    selected_model: fallback.clone(),
                    selected_provider: None,
                    strategy: RoutingStrategy::FallbackChain,
                    rule_used: None,
                    fallback_used: true,
                    reason: format!("Primary unavailable; fell back to '{fallback}'"),
                    decided_at: Utc::now(),
                };
                self.record(&outcome);
                return outcome;
            }
        }

        // 5. Default model.
        let default = config.default_model.clone();
        let outcome = RouteOutcome {
            session_id: request.session_id.clone(),
            requested_model: request.requested_model.clone(),
            selected_model: default.clone(),
            selected_provider: None,
            strategy: RoutingStrategy::Default,
            rule_used: None,
            fallback_used: false,
            reason: "Used configured default model".to_string(),
            decided_at: Utc::now(),
        };
        self.record(&outcome);
        outcome
    }

    /// Resolve the highest-priority matching rule, if any.
    fn best_rule(
        &self,
        request: &RouteRequest,
    ) -> Option<(&RoutingRule, usize)> {
        let config = self.config.read();
        let mut best: Option<(&RoutingRule, usize)> = None;
        for (idx, rule) in config.rules.iter().enumerate() {
            if rule.matches(&request.session_id, request.requested_model.as_deref()) {
                match &best {
                    Some((current, _)) if current.priority >= rule.priority => {}
                    _ => best = Some((rule, idx)),
                }
            }
        }
        best
    }

    /// Place a routing hold on a session (pins the model).
    pub fn hold_session(
        &self,
        session_id: &str,
        model: &str,
        provider: Option<String>,
        reason: &str,
        expires_at: Option<DateTime<Utc>>,
    ) {
        let hold = RoutingHold {
            session_id: session_id.to_string(),
            model: model.to_string(),
            provider,
            reason: reason.to_string(),
            held_at: Utc::now(),
            expires_at,
        };
        self.store.set_hold(hold);
        info!(session_id = %session_id, model = %model, "Routing hold set");
    }

    /// Release a routing hold on a session.
    pub fn release_hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.store.release_hold(session_id)
    }

    /// Record a decision in the shared store and emit a debug log.
    fn record(&self, outcome: &RouteOutcome) {
        let decision = RoutingDecision {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: outcome.session_id.clone(),
            requested_model: outcome.requested_model.clone(),
            selected_model: outcome.selected_model.clone(),
            selected_provider: outcome
                .selected_provider
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            strategy: format!("{:?}", outcome.strategy),
            reason: outcome.reason.clone(),
            fallback_used: outcome.fallback_used,
            decided_at: outcome.decided_at,
        };
        self.store.record_decision(decision);
        debug!(
            session_id = %outcome.session_id,
            selected_model = %outcome.selected_model,
            strategy = ?outcome.strategy,
            "Routing decision recorded"
        );
    }

    /// Check model availability through all registered checkers.
    fn is_available(&self, model: &str) -> bool {
        let checkers = self.availability.clone();
        if checkers.is_empty() {
            return true;
        }
        checkers.iter().all(|c| c.is_available(model))
    }

    /// Return recent routing decisions (optionally for a session).
    pub fn decisions(&self, session_id: Option<&str>, limit: usize) -> Vec<RoutingDecision> {
        self.store.list_decisions(session_id, limit)
    }

    /// Return the active hold for a session, if any.
    pub fn hold(&self, session_id: &str) -> Option<RoutingHold> {
        self.store.get_hold(session_id)
    }
}

/// Simple glob matching supporting `*` (any run of chars) and `?` (one char).
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0, 0);
    let (mut star_p, mut star_t) = (None, 0);

    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star_p = Some(p);
            star_t = t;
            p += 1;
        } else if let Some(sp) = star_p {
            star_t += 1;
            p = sp + 1;
            t = star_t;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// Convert an error from the routing layer into an [`AppError`].
pub fn routing_err(e: AppError) -> AppError {
    warn!(error = %e, "Model routing error");
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_router() -> ModelRouter {
        ModelRouter::new(ModelRouterConfig::default())
    }

    #[test]
    fn test_route_default() {
        let router = default_router();
        let request = RouteRequest {
            session_id: "s1".into(),
            requested_model: None,
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "gpt-4o");
        assert_eq!(outcome.strategy, RoutingStrategy::Default);
    }

    #[test]
    fn test_route_explicit_model() {
        let router = default_router();
        let request = RouteRequest {
            session_id: "s1".into(),
            requested_model: Some("gpt-4o-mini".into()),
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "gpt-4o-mini");
        assert_eq!(outcome.strategy, RoutingStrategy::Explicit);
    }

    #[test]
    fn test_route_hold_overrides() {
        let router = default_router();
        router.hold_session("s1", "claude-sonnet-4", Some("anthropic".into()), "test", None);
        let request = RouteRequest {
            session_id: "s1".into(),
            requested_model: Some("gpt-4o".into()),
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "claude-sonnet-4");
        assert_eq!(outcome.selected_provider.as_deref(), Some("anthropic"));
        assert_eq!(outcome.strategy, RoutingStrategy::Hold);
    }

    #[test]
    fn test_route_expired_hold_released() {
        let router = default_router();
        let expired = Utc::now() - chrono::Duration::seconds(1);
        router.hold_session("s1", "old-model", None, "expired", Some(expired));
        let request = RouteRequest {
            session_id: "s1".into(),
            requested_model: None,
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "gpt-4o");
        assert!(router.hold("s1").is_none());
    }

    #[test]
    fn test_route_rule_matching() {
        let config = ModelRouterConfig {
            rules: vec![RoutingRule {
                name: "research".into(),
                session_pattern: Some("research-*".into()),
                requested_model_contains: None,
                model: "claude-opus-4".into(),
                provider: Some("anthropic".into()),
                priority: 10,
            }],
            ..Default::default()
        };
        let router = ModelRouter::new(config);
        let request = RouteRequest {
            session_id: "research-1".into(),
            requested_model: None,
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "claude-opus-4");
        assert_eq!(outcome.rule_used.as_deref(), Some("research"));

        let other = RouteRequest {
            session_id: "personal-1".into(),
            requested_model: None,
            purpose: "chat".into(),
        };
        let outcome2 = router.route(&other);
        assert_eq!(outcome2.selected_model, "gpt-4o");
    }

    #[test]
    fn test_route_fallback_chain() {
        struct OnlyMini;
        impl ModelAvailability for OnlyMini {
            fn is_available(&self, model: &str) -> bool {
                model == "gpt-4o-mini"
            }
        }
        let router = default_router().with_availability(Arc::new(OnlyMini));
        let request = RouteRequest {
            session_id: "s1".into(),
            requested_model: Some("gpt-4o".into()),
            purpose: "chat".into(),
        };
        let outcome = router.route(&request);
        assert_eq!(outcome.selected_model, "gpt-4o-mini");
        assert!(outcome.fallback_used);
        assert_eq!(outcome.strategy, RoutingStrategy::FallbackChain);
    }

    #[test]
    fn test_decisions_recorded() {
        let router = default_router();
        for i in 0..3 {
            router.route(&RouteRequest {
                session_id: format!("s{i}"),
                requested_model: None,
                purpose: "chat".into(),
            });
        }
        assert_eq!(router.decisions(None, 10).len(), 3);
        assert_eq!(router.decisions(Some("s1"), 10).len(), 1);
    }

    #[test]
    fn test_release_hold() {
        let router = default_router();
        router.hold_session("s1", "gpt-4o", None, "test", None);
        assert!(router.release_hold("s1").is_some());
        assert!(router.release_hold("s1").is_none());
    }

    #[test]
    fn test_glob_match() {
        assert!(glob_match("research-*", "research-1"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(!glob_match("research-*", "personal-1"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn test_set_config_replaces() {
        let router = default_router();
        router.set_config(ModelRouterConfig {
            default_model: "custom-default".into(),
            ..Default::default()
        });
        assert_eq!(router.config().default_model, "custom-default");
    }
}
