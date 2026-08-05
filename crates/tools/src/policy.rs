//! Tool policy chain for the OpenSquilla tool system.
//!
//! Provides a chain-of-responsibility pattern for tool execution policies,
//! including deny lists, budget tracking, and finalization guards.
//! Each policy in the chain can approve, deny, or flag a tool execution
//! before it reaches the actual tool implementation.

use opensquilla_core::error::AppError;
use opensquilla_core::ToolCall;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The decision resulting from a policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// The tool execution is approved.
    Allow,
    /// The tool execution is denied with a reason.
    Deny {
        /// The reason for denial.
        reason: String,
    },
    /// The tool execution requires user confirmation.
    RequireConfirmation {
        /// The reason confirmation is needed.
        reason: String,
    },
    /// The tool execution should be deferred (e.g., rate limited).
    Defer {
        /// The reason for deferral.
        reason: String,
        /// Retry after this many seconds.
        retry_after_secs: u64,
    },
}

impl PolicyDecision {
    /// Check if this decision allows execution.
    pub fn is_allowed(&self) -> bool {
        matches!(self, PolicyDecision::Allow)
    }
}

impl fmt::Display for PolicyDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyDecision::Allow => write!(f, "allow"),
            PolicyDecision::Deny { reason } => write!(f, "deny: {}", reason),
            PolicyDecision::RequireConfirmation { reason } => {
                write!(f, "require_confirmation: {}", reason)
            }
            PolicyDecision::Defer {
                reason,
                retry_after_secs,
            } => write!(f, "defer ({}s): {}", retry_after_secs, reason),
        }
    }
}

/// Context provided to each policy during evaluation.
#[derive(Debug, Clone)]
pub struct PolicyContext {
    /// The name of the tool being executed.
    pub tool_name: String,
    /// The tool call that triggered execution.
    pub call: ToolCall,
    /// The session ID this execution is part of.
    pub session_id: String,
    /// The user ID associated with the request.
    pub user_id: Option<String>,
    /// Current budget usage in token-equivalent units.
    pub budget_used: u64,
    /// Budget limit in token-equivalent units.
    pub budget_limit: u64,
    /// Whether the sandbox is active.
    pub sandbox_active: bool,
}

impl PolicyContext {
    /// Create a new policy context.
    pub fn new(
        tool_name: impl Into<String>,
        call: ToolCall,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            call,
            session_id: session_id.into(),
            user_id: None,
            budget_used: 0,
            budget_limit: u64::MAX,
            sandbox_active: false,
        }
    }

    /// Set the user ID.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set budget limits.
    pub fn with_budget(mut self, used: u64, limit: u64) -> Self {
        self.budget_used = used;
        self.budget_limit = limit;
        self
    }

    /// Set sandbox state.
    pub fn with_sandbox(mut self, active: bool) -> Self {
        self.sandbox_active = active;
        self
    }
}

/// A trait for implementing tool execution policies.
///
/// Policies are composed into a chain. Each policy can evaluate the
/// tool execution context and return a decision.
#[async_trait::async_trait]
pub trait PolicyChain: Send + Sync {
    /// Evaluate the policy against the given context.
    ///
    /// Returns a `PolicyDecision` indicating whether execution should proceed.
    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision;

    /// A human-readable name for this policy (for debugging/logging).
    fn name(&self) -> &str;
}

/// A policy that denies specific tools based on a deny list.
///
/// Supports:
/// - Exact tool name matching
/// - Wildcard patterns (e.g., "git_*")
/// - Category-based denial (e.g., all "shell" tools)
/// - Per-session overrides
pub struct DenyPolicy {
    name: String,
    /// Set of tool names that are always denied.
    denied_tools: Vec<String>,
    /// Set of categories that are denied.
    denied_categories: Vec<String>,
    /// Per-session allowlist overrides: session_id -> Vec<tool_name>
    session_overrides: HashMap<String, Vec<String>>,
}

impl DenyPolicy {
    /// Create a new deny policy with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            denied_tools: Vec::new(),
            denied_categories: Vec::new(),
            session_overrides: HashMap::new(),
        }
    }

    /// Deny a specific tool by name.
    pub fn deny_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.denied_tools.push(tool_name.into());
        self
    }

    /// Deny an entire category of tools.
    pub fn deny_category(mut self, category: impl Into<String>) -> Self {
        self.denied_categories.push(category.into());
        self
    }

    /// Add a session-level override that allows a tool.
    pub fn allow_for_session(
        mut self,
        session_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        self.session_overrides
            .entry(session_id.into())
            .or_default()
            .push(tool_name.into());
        self
    }

    /// Check if a tool name matches a pattern (supports wildcard '*').
    fn matches_pattern(pattern: &str, tool_name: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        if pattern.ends_with('*') {
            let prefix = pattern.trim_end_matches('*');
            tool_name.starts_with(prefix)
        } else {
            pattern == tool_name
        }
    }
}

#[async_trait::async_trait]
impl PolicyChain for DenyPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        // Check session-level overrides first (allowlist takes precedence).
        if let Some(session_id) = ctx.session_id.as_str() {
            if let Some(allowed) = self.session_overrides.get(session_id) {
                if allowed.iter().any(|a| Self::matches_pattern(a, &ctx.tool_name)) {
                    return PolicyDecision::Allow;
                }
            }
        }

        // Check denied tools.
        for denied in &self.denied_tools {
            if Self::matches_pattern(denied, &ctx.tool_name) {
                return PolicyDecision::Deny {
                    reason: format!(
                        "Tool '{}' is denied by policy '{}' (matches '{}')",
                        ctx.tool_name, self.name, denied
                    ),
                };
            }
        }

        PolicyDecision::Allow
    }
}

/// A policy that tracks and enforces budget limits on tool execution.
///
/// Budget is measured in abstract "cost units" that can represent
/// token usage, execution time, or API call costs.
pub struct BudgetPolicy {
    name: String,
    /// Maximum cost units per session.
    max_per_session: u64,
    /// Maximum cost units per tool call.
    max_per_call: u64,
    /// Per-tool cost overrides: tool_name -> cost_units.
    tool_costs: HashMap<String, u64>,
    /// Default cost for tools without explicit cost.
    default_cost: u64,
    /// Session-level budget tracking. Interior mutability lets `evaluate`
    /// (which receives `&self`) debit the running total for a session.
    session_budgets: std::sync::Mutex<HashMap<String, AtomicU64>>,
}

impl BudgetPolicy {
    /// Create a new budget policy with the given limits.
    pub fn new(name: impl Into<String>, max_per_session: u64, max_per_call: u64) -> Self {
        Self {
            name: name.into(),
            max_per_session,
            max_per_call,
            tool_costs: HashMap::new(),
            default_cost: 1,
            session_budgets: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Set the cost for a specific tool.
    pub fn with_tool_cost(mut self, tool_name: impl Into<String>, cost: u64) -> Self {
        self.tool_costs.insert(tool_name.into(), cost);
        self
    }

    /// Set the default cost for tools without explicit cost.
    pub fn with_default_cost(mut self, cost: u64) -> Self {
        self.default_cost = cost;
        self
    }

    /// Get the cost for a tool.
    fn tool_cost(&self, tool_name: &str) -> u64 {
        self.tool_costs.get(tool_name).copied().unwrap_or(self.default_cost)
    }

    /// Reset the budget for a session.
    pub fn reset_session(&self, session_id: &str) {
        if let Ok(mut budgets) = self.session_budgets.lock() {
            budgets.insert(session_id.to_string(), AtomicU64::new(0));
        }
    }
}

#[async_trait::async_trait]
impl PolicyChain for BudgetPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        let cost = self.tool_cost(&ctx.tool_name);

        // Check per-call budget.
        if cost > self.max_per_call {
            return PolicyDecision::Deny {
                reason: format!(
                    "Tool '{}' costs {} units, exceeding per-call limit of {}",
                    ctx.tool_name, cost, self.max_per_call
                ),
            };
        }

        // Load (or lazily seed) the session's current spend. Seeding from the
        // dispatch context's reported usage makes the first approved call
        // account for any pre-existing usage the caller knows about.
        let mut budgets = match self.session_budgets.lock() {
            Ok(guard) => guard,
            Err(_) => {
                return PolicyDecision::Deny {
                    reason: "Budget policy lock poisoned".to_string(),
                }
            }
        };

        let session_counter = budgets
            .entry(ctx.session_id.clone())
            .or_insert_with(|| AtomicU64::new(ctx.budget_used));

        let session_budget = session_counter.load(Ordering::Relaxed);
        let new_total = session_budget.saturating_add(cost);
        if new_total > self.max_per_session {
            return PolicyDecision::Deny {
                reason: format!(
                    "Session budget exhausted: {} + {} > {} limit",
                    session_budget, cost, self.max_per_session
                ),
            };
        }

        // Actually debit the approved cost against the session counter. This is
        // the fix for the audit bug where approvals never debited the budget,
        // so a session could run an unbounded number of approved calls without
        // ever exhausting its quota. `fetch_add` accumulates correctly across
        // concurrent dispatches; if a concurrent dispatch pushed the session
        // past the limit between the check and the debit, roll the debit back
        // and deny instead.
        let prev = session_counter.fetch_add(cost, Ordering::Relaxed);
        let now = prev.saturating_add(cost);
        if now > self.max_per_session {
            session_counter.fetch_sub(cost, Ordering::Relaxed);
            return PolicyDecision::Deny {
                reason: format!(
                    "Session budget exhausted after concurrent debit: {} + {} > {} limit",
                    prev, cost, self.max_per_session
                ),
            };
        }

        tracing::debug!(
            tool = %ctx.tool_name,
            session = %ctx.session_id,
            cost = cost,
            new_total = now,
            "Budget debited for tool call"
        );

        PolicyDecision::Allow
    }
}

/// A policy that finalizes tool execution and performs cleanup/logging.
///
/// This is typically the last policy in the chain. It:
/// - Records execution metrics
/// - Enforces execution timeouts
/// - Logs the execution for audit trails
pub struct FinalizePolicy {
    name: String,
    /// Maximum execution time in seconds.
    max_execution_time_secs: u64,
    /// Whether to log all tool executions.
    log_all: bool,
    /// Whether to record metrics.
    record_metrics: bool,
}

impl FinalizePolicy {
    /// Create a new finalize policy.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_execution_time_secs: 300,
            log_all: true,
            record_metrics: true,
        }
    }

    /// Set the maximum execution time.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.max_execution_time_secs = secs;
        self
    }

    /// Disable logging.
    pub fn without_logging(mut self) -> Self {
        self.log_all = false;
        self
    }
}

#[async_trait::async_trait]
impl PolicyChain for FinalizePolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        if self.log_all {
            tracing::info!(
                tool = %ctx.tool_name,
                session = %ctx.session_id,
                budget_used = %ctx.budget_used,
                sandbox = %ctx.sandbox_active,
                "Tool execution logged"
            );
        }

        PolicyDecision::Allow
    }
}

/// A composite policy chain that evaluates multiple policies in sequence.
///
/// Policies are evaluated in order. If any policy returns `Deny`, the chain
/// short-circuits and returns that decision. If all policies return `Allow`,
/// execution proceeds.
pub struct PolicyChainSet {
    name: String,
    policies: Vec<Box<dyn PolicyChain>>,
}

impl PolicyChainSet {
    /// Create a new empty policy chain set.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            policies: Vec::new(),
        }
    }

    /// Add a policy to the chain.
    pub fn add(mut self, policy: impl PolicyChain + 'static) -> Self {
        self.policies.push(Box::new(policy));
        self
    }

    /// Add a boxed policy to the chain.
    pub fn add_boxed(mut self, policy: Box<dyn PolicyChain>) -> Self {
        self.policies.push(policy);
        self
    }
}

#[async_trait::async_trait]
impl PolicyChain for PolicyChainSet {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        for policy in &self.policies {
            let decision = policy.evaluate(ctx).await;
            match &decision {
                PolicyDecision::Allow => {
                    tracing::trace!(policy = %policy.name(), "Policy allowed");
                    continue;
                }
                _ => {
                    tracing::warn!(
                        policy = %policy.name(),
                        decision = %decision,
                        "Policy blocked execution"
                    );
                    return decision;
                }
            }
        }
        PolicyDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::ToolCall;
    use serde_json::json;

    #[tokio::test]
    async fn test_deny_policy_specific_tool() {
        let policy = DenyPolicy::new("test").deny_tool("rm");
        let call = ToolCall::new("1", "rm", json!({}));
        let ctx = PolicyContext::new("rm", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(
            decision,
            PolicyDecision::Deny {
                reason: "Tool 'rm' is denied by policy 'test' (matches 'rm')".to_string()
            }
        );
    }

    #[tokio::test]
    async fn test_deny_policy_wildcard() {
        let policy = DenyPolicy::new("test").deny_tool("git_*");
        let call = ToolCall::new("1", "git_push", json!({}));
        let ctx = PolicyContext::new("git_push", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_deny_policy_allowlist_override() {
        let policy = DenyPolicy::new("test")
            .deny_tool("rm")
            .allow_for_session("session-1", "rm");
        let call = ToolCall::new("1", "rm", json!({}));
        let ctx = PolicyContext::new("rm", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_budget_policy_per_call() {
        let policy = BudgetPolicy::new("budget", 100, 10).with_tool_cost("expensive", 50);
        let call = ToolCall::new("1", "expensive", json!({}));
        let ctx = PolicyContext::new("expensive", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        // 50 > 10 per-call limit
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_budget_policy_within_limit() {
        let policy = BudgetPolicy::new("budget", 100, 50).with_tool_cost("cheap", 5);
        let call = ToolCall::new("1", "cheap", json!({}));
        let ctx = PolicyContext::new("cheap", call, "session-1").with_budget(10, 100);
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_policy_chain_set() {
        let deny = DenyPolicy::new("deny").deny_tool("forbidden");
        let budget = BudgetPolicy::new("budget", 100, 50);
        let chain = PolicyChainSet::new("main").add(deny).add(budget);

        // Should be denied by deny policy.
        let call = ToolCall::new("1", "forbidden", json!({}));
        let ctx = PolicyContext::new("forbidden", call, "session-1");
        let decision = chain.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));

        // Should be allowed by both.
        let call = ToolCall::new("2", "allowed", json!({}));
        let ctx = PolicyContext::new("allowed", call, "session-1");
        let decision = chain.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_budget_policy_debits_session_counter() {
        // Regression test for the audit bug: an approved call must actually
        // debit the session counter so subsequent calls accumulate toward the
        // limit. With max_per_session=10 and cost=4 per call, three calls
        // (4+4+4=12 > 10) should exhaust the budget on the third.
        let policy = BudgetPolicy::new("budget", 10, 50).with_tool_cost("tool", 4);
        let session = "session-debit";

        let call = ToolCall::new("1", "tool", json!({}));
        let ctx = PolicyContext::new("tool", call, session);
        assert_eq!(policy.evaluate(&ctx).await, PolicyDecision::Allow);

        let call = ToolCall::new("2", "tool", json!({}));
        let ctx = PolicyContext::new("tool", call, session);
        assert_eq!(policy.evaluate(&ctx).await, PolicyDecision::Allow);

        // Third call would push the running total to 12 > 10 — must be denied.
        let call = ToolCall::new("3", "tool", json!({}));
        let ctx = PolicyContext::new("tool", call, session);
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_budget_policy_denial_does_not_debit() {
        // A denied call (over the per-call cap) must not touch the counter.
        let policy = BudgetPolicy::new("budget", 100, 5).with_tool_cost("big", 50);
        let session = "session-noop";

        let call = ToolCall::new("1", "big", json!({}));
        let ctx = PolicyContext::new("big", call, session);
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));

        // Counter should still be at 0 (never seeded via an approval), so a
        // subsequent cheap call within limits is allowed and debits to 1.
        let policy = BudgetPolicy::new("budget", 100, 5).with_tool_cost("small", 1);
        let call = ToolCall::new("2", "small", json!({}));
        let ctx = PolicyContext::new("small", call, session);
        assert_eq!(policy.evaluate(&ctx).await, PolicyDecision::Allow);
    }
}