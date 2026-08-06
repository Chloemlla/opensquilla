//! Tool policy chain for the OpenSquilla tool system.
//!
//! Provides a chain-of-responsibility pattern for tool execution policies,
//! including deny lists, budget tracking, and finalization guards.
//! Each policy in the chain can approve, deny, or flag a tool execution
//! before it reaches the actual tool implementation.

use opensquilla_core::ToolCall;
use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// The tool's declared risk level (0-3, matching `ToolDefinition::risk_level`).
    pub risk_level: u8,
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
            risk_level: 2,
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

    /// Set the tool's risk level.
    pub fn with_risk_level(mut self, level: u8) -> Self {
        self.risk_level = level;
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
                if allowed
                    .iter()
                    .any(|a| Self::matches_pattern(a, &ctx.tool_name))
                {
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
        self.tool_costs
            .get(tool_name)
            .copied()
            .unwrap_or(self.default_cost)
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
                };
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

// ---------------------------------------------------------------------------
// Risk classification
// ---------------------------------------------------------------------------

/// The risk classification of a tool.
///
/// This determines whether a tool call needs user confirmation, an admin
/// override, or can run automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RiskLevel {
    /// Safe tools that can always run without confirmation.
    Safe,
    /// Tools that modify state and should be confirmed before running.
    Confirm,
    /// Tools that require an admin override (cannot be confirmed by an
    /// ordinary user).
    AdminOnly,
}

impl RiskLevel {
    /// Convert a numeric risk level (0-3, matching `ToolDefinition::risk_level`)
    /// to a `RiskLevel` classification.
    pub fn from_risk_level(level: u8) -> Self {
        match level {
            0 | 1 => RiskLevel::Safe,
            2 => RiskLevel::Confirm,
            _ => RiskLevel::AdminOnly,
        }
    }

    /// Whether this level requires user confirmation.
    pub fn requires_confirmation(&self) -> bool {
        matches!(self, RiskLevel::Confirm)
    }

    /// Whether this level requires admin privileges.
    pub fn requires_admin(&self) -> bool {
        matches!(self, RiskLevel::AdminOnly)
    }
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::Safe => write!(f, "SAFE"),
            RiskLevel::Confirm => write!(f, "CONFIRM"),
            RiskLevel::AdminOnly => write!(f, "ADMIN_ONLY"),
        }
    }
}

/// A policy that classifies tools by risk and requires confirmation for
/// risky operations.
///
/// The risk classification comes from the tool definition's `risk_level`
/// (0-3), unless a per-tool override is registered. A tool classified as
/// `Confirm` requires user confirmation before execution, unless the caller
/// is an admin or the tool is on the session allowlist. A tool classified as
/// `AdminOnly` always requires an admin override.
pub struct RiskPolicy {
    name: String,
    /// Per-tool risk overrides: tool_name -> RiskLevel.
    overrides: HashMap<String, RiskLevel>,
    /// Whether the current user is an admin (per-session).
    admin_sessions: std::sync::Mutex<HashMap<String, bool>>,
    /// Whether to auto-confirm tools whose definition marks them as
    /// `requires_confirmation`.
    honor_definition_confirmation: bool,
}

impl RiskPolicy {
    /// Create a new risk policy.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            overrides: HashMap::new(),
            admin_sessions: std::sync::Mutex::new(HashMap::new()),
            honor_definition_confirmation: true,
        }
    }

    /// Override the risk classification of a tool.
    pub fn with_override(mut self, tool_name: impl Into<String>, level: RiskLevel) -> Self {
        self.overrides.insert(tool_name.into(), level);
        self
    }

    /// Mark a session as an admin session (allows Confirm tools without
    /// confirmation, and AdminOnly tools).
    pub fn with_admin_session(mut self, session_id: impl Into<String>) -> Self {
        if let Ok(mut sessions) = self.admin_sessions.lock() {
            sessions.insert(session_id.into(), true);
        }
        self
    }

    /// Disable honoring the tool definition's `requires_confirmation` flag.
    pub fn without_definition_confirmation(mut self) -> Self {
        self.honor_definition_confirmation = false;
        self
    }

    /// Classify a tool by name.
    fn classify(&self, ctx: &PolicyContext) -> RiskLevel {
        if let Some(level) = self.overrides.get(&ctx.tool_name) {
            return *level;
        }
        // Fall back to the tool definition's risk level, propagated through
        // the policy context.
        RiskLevel::from_risk_level(ctx.risk_level)
    }

    /// Check if the session is an admin session.
    fn is_admin(&self, session_id: &str) -> bool {
        self.admin_sessions
            .lock()
            .ok()
            .and_then(|s| s.get(session_id).copied())
            .unwrap_or(false)
    }
}

#[async_trait]
impl PolicyChain for RiskPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        let risk = self.classify(ctx);
        let is_admin = self.is_admin(&ctx.session_id);

        match risk {
            RiskLevel::Safe => PolicyDecision::Allow,
            RiskLevel::Confirm => {
                // An admin session can run Confirm tools directly.
                if is_admin {
                    PolicyDecision::Allow
                } else {
                    PolicyDecision::RequireConfirmation {
                        reason: format!(
                            "Tool '{}' is classified as {} and requires user confirmation",
                            ctx.tool_name, risk
                        ),
                    }
                }
            }
            RiskLevel::AdminOnly => {
                if is_admin {
                    PolicyDecision::Allow
                } else {
                    PolicyDecision::Deny {
                        reason: format!(
                            "Tool '{}' is classified as {} and requires an admin override",
                            ctx.tool_name, risk
                        ),
                    }
                }
            }
        }
    }
}

/// A policy that enforces per-tool rules such as argument constraints and
/// call quotas.
///
/// Each tool can have:
/// - A maximum call count per session.
/// - Required argument values (e.g., `working_dir` must be within a base).
/// - Argument value denylists.
pub struct ToolPolicy {
    name: String,
    /// Per-tool rules.
    rules: HashMap<String, ToolRule>,
    /// Per-session call counts.
    call_counts: std::sync::Mutex<HashMap<(String, String), usize>>,
}

/// Rules for a single tool.
#[derive(Debug, Clone)]
pub struct ToolRule {
    /// Maximum calls per session for this tool.
    pub max_calls_per_session: Option<usize>,
    /// Argument names whose values are denied (substring match).
    pub denied_arg_values: Vec<String>,
    /// Required argument names that must be present.
    pub required_args: Vec<String>,
}

impl Default for ToolRule {
    fn default() -> Self {
        Self {
            max_calls_per_session: None,
            denied_arg_values: Vec::new(),
            required_args: Vec::new(),
        }
    }
}

impl ToolRule {
    /// Create a new empty tool rule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Limit the number of calls per session.
    pub fn with_max_calls(mut self, max: usize) -> Self {
        self.max_calls_per_session = Some(max);
        self
    }

    /// Add a denied argument value pattern.
    pub fn deny_arg_value(mut self, pattern: impl Into<String>) -> Self {
        self.denied_arg_values.push(pattern.into());
        self
    }

    /// Require an argument.
    pub fn require_arg(mut self, name: impl Into<String>) -> Self {
        self.required_args.push(name.into());
        self
    }
}

impl ToolPolicy {
    /// Create a new tool policy.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: HashMap::new(),
            call_counts: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Register a rule for a tool.
    pub fn with_rule(mut self, tool_name: impl Into<String>, rule: ToolRule) -> Self {
        self.rules.insert(tool_name.into(), rule);
        self
    }

    /// Check the argument constraints for a tool.
    fn check_args(&self, ctx: &PolicyContext) -> Result<(), String> {
        let rule = match self.rules.get(&ctx.tool_name) {
            Some(rule) => rule,
            None => return Ok(()),
        };

        // Required arguments.
        for arg in &rule.required_args {
            if !ctx.call.input.get(arg).is_some_and(|v| !v.is_null()) {
                return Err(format!(
                    "Tool '{}' requires argument '{}'",
                    ctx.tool_name, arg
                ));
            }
        }

        // Denied argument values.
        if let Some(obj) = ctx.call.input.as_object() {
            for (key, value) in obj {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                };
                for pattern in &rule.denied_arg_values {
                    if value_str.contains(pattern) {
                        return Err(format!(
                            "Tool '{}' argument '{}' contains denied value pattern '{}'",
                            ctx.tool_name, key, pattern
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl PolicyChain for ToolPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        // Check call counts.
        if let Some(rule) = self.rules.get(&ctx.tool_name) {
            if let Some(max_calls) = rule.max_calls_per_session {
                let key = (ctx.tool_name.clone(), ctx.session_id.clone());
                let mut counts = match self.call_counts.lock() {
                    Ok(c) => c,
                    Err(_) => {
                        return PolicyDecision::Deny {
                            reason: "Tool policy lock poisoned".to_string(),
                        }
                    }
                };
                let count = counts.entry(key).or_insert(0);
                if *count >= max_calls {
                    return PolicyDecision::Deny {
                        reason: format!(
                            "Tool '{}' exceeded its per-session call limit of {}",
                            ctx.tool_name, max_calls
                        ),
                    };
                }
                *count += 1;
            }
        }

        // Check argument constraints.
        if let Err(reason) = self.check_args(ctx) {
            return PolicyDecision::Deny { reason };
        }

        PolicyDecision::Allow
    }
}

/// A policy that implements the confirmation flow for the dispatch engine.
///
/// When a tool requires confirmation, this policy tracks whether the user has
/// already confirmed the specific tool call. Confirmed calls are allowed;
/// unconfirmed calls return `RequireConfirmation`.
pub struct ConfirmationPolicy {
    name: String,
    /// Set of confirmed call signatures: (session_id, tool_name, args_hash).
    confirmed: std::sync::Mutex<HashMap<String, Vec<String>>>,
    /// Tools that never require confirmation.
    auto_allow: Vec<String>,
}

impl ConfirmationPolicy {
    /// Create a new confirmation policy.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            confirmed: std::sync::Mutex::new(HashMap::new()),
            auto_allow: Vec::new(),
        }
    }

    /// Add a tool that never requires confirmation.
    pub fn auto_allow(mut self, tool_name: impl Into<String>) -> Self {
        self.auto_allow.push(tool_name.into());
        self
    }

    /// Mark a specific tool call as confirmed.
    ///
    /// The signature is `(session_id, tool_name, args_hash)`.
    pub fn confirm(&self, session_id: &str, tool_name: &str, args_hash: &str) {
        if let Ok(mut confirmed) = self.confirmed.lock() {
            confirmed
                .entry(session_id.to_string())
                .or_default()
                .push(format!("{}:{}", tool_name, args_hash));
        }
    }

    /// Check whether a specific tool call has been confirmed.
    pub fn is_confirmed(&self, session_id: &str, tool_name: &str, args_hash: &str) -> bool {
        self.confirmed
            .lock()
            .ok()
            .and_then(|c| c.get(session_id).cloned())
            .map(|list| {
                list.iter()
                    .any(|sig| sig == &format!("{}:{}", tool_name, args_hash))
            })
            .unwrap_or(false)
    }

    /// Compute a simple hash of the arguments for confirmation matching.
    pub fn hash_args(args: &serde_json::Value) -> String {
        use sha2::{Digest, Sha256};
        let serialized = serde_json::to_string(args).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        let digest = hasher.finalize();
        digest.iter().take(8).map(|b| format!("{:02x}", b)).collect()
    }
}

#[async_trait]
impl PolicyChain for ConfirmationPolicy {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&self, ctx: &PolicyContext) -> PolicyDecision {
        // Auto-allowed tools never need confirmation.
        if self.auto_allow.iter().any(|t| t == &ctx.tool_name) {
            return PolicyDecision::Allow;
        }

        let args_hash = Self::hash_args(&ctx.call.input);
        if self.is_confirmed(&ctx.session_id, &ctx.tool_name, &args_hash) {
            return PolicyDecision::Allow;
        }

        PolicyDecision::RequireConfirmation {
            reason: format!(
                "Tool '{}' requires confirmation before execution",
                ctx.tool_name
            ),
        }
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

    #[test]
    fn test_risk_level_classification() {
        assert_eq!(RiskLevel::from_risk_level(0), RiskLevel::Safe);
        assert_eq!(RiskLevel::from_risk_level(1), RiskLevel::Safe);
        assert_eq!(RiskLevel::from_risk_level(2), RiskLevel::Confirm);
        assert_eq!(RiskLevel::from_risk_level(3), RiskLevel::AdminOnly);
    }

    #[test]
    fn test_risk_level_properties() {
        assert!(!RiskLevel::Safe.requires_confirmation());
        assert!(RiskLevel::Confirm.requires_confirmation());
        assert!(!RiskLevel::AdminOnly.requires_confirmation());
        assert!(RiskLevel::AdminOnly.requires_admin());
        assert_eq!(RiskLevel::Confirm.to_string(), "CONFIRM");
        assert_eq!(RiskLevel::AdminOnly.to_string(), "ADMIN_ONLY");
        assert_eq!(RiskLevel::Safe.to_string(), "SAFE");
    }

    #[tokio::test]
    async fn test_risk_policy_confirm_requires_confirmation() {
        let policy = RiskPolicy::new("risk").with_override("git_push", RiskLevel::Confirm);
        let call = ToolCall::new("1", "git_push", json!({}));
        let ctx = PolicyContext::new("git_push", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::RequireConfirmation { .. }));
    }

    #[tokio::test]
    async fn test_risk_policy_admin_session_allows_confirm() {
        let policy = RiskPolicy::new("risk")
            .with_override("git_push", RiskLevel::Confirm)
            .with_admin_session("admin-session");
        let call = ToolCall::new("1", "git_push", json!({}));
        let ctx = PolicyContext::new("git_push", call, "admin-session");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_risk_policy_admin_only_denied_for_user() {
        let policy = RiskPolicy::new("risk").with_override("shell_exec", RiskLevel::AdminOnly);
        let call = ToolCall::new("1", "shell_exec", json!({}));
        let ctx = PolicyContext::new("shell_exec", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_risk_policy_admin_only_allowed_for_admin() {
        let policy = RiskPolicy::new("risk")
            .with_override("shell_exec", RiskLevel::AdminOnly)
            .with_admin_session("admin-session");
        let call = ToolCall::new("1", "shell_exec", json!({}));
        let ctx = PolicyContext::new("shell_exec", call, "admin-session");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_tool_policy_max_calls() {
        let policy = ToolPolicy::new("toolpolicy")
            .with_rule("limited", ToolRule::new().with_max_calls(2));
        let call = ToolCall::new("1", "limited", json!({}));
        let ctx = PolicyContext::new("limited", call, "session-1");

        assert_eq!(policy.evaluate(&ctx).await, PolicyDecision::Allow);
        assert_eq!(policy.evaluate(&ctx).await, PolicyDecision::Allow);
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_tool_policy_required_args() {
        let policy = ToolPolicy::new("toolpolicy")
            .with_rule("needs_args", ToolRule::new().require_arg("path"));
        let call = ToolCall::new("1", "needs_args", json!({}));
        let ctx = PolicyContext::new("needs_args", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_tool_policy_denied_arg_values() {
        let policy = ToolPolicy::new("toolpolicy")
            .with_rule("danger", ToolRule::new().deny_arg_value("/etc/passwd"));
        let call = ToolCall::new("1", "danger", json!({"path": "/etc/passwd"}));
        let ctx = PolicyContext::new("danger", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn test_confirmation_policy_requires_confirmation() {
        let policy = ConfirmationPolicy::new("confirm");
        let call = ToolCall::new("1", "some_tool", json!({"a": 1}));
        let ctx = PolicyContext::new("some_tool", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert!(matches!(decision, PolicyDecision::RequireConfirmation { .. }));
    }

    #[tokio::test]
    async fn test_confirmation_policy_confirmed_call_allowed() {
        let policy = ConfirmationPolicy::new("confirm");
        let args = json!({"a": 1});
        let hash = ConfirmationPolicy::hash_args(&args);
        policy.confirm("session-1", "some_tool", &hash);

        let call = ToolCall::new("1", "some_tool", args);
        let ctx = PolicyContext::new("some_tool", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[tokio::test]
    async fn test_confirmation_policy_auto_allow() {
        let policy = ConfirmationPolicy::new("confirm").auto_allow("safe_tool");
        let call = ToolCall::new("1", "safe_tool", json!({}));
        let ctx = PolicyContext::new("safe_tool", call, "session-1");
        let decision = policy.evaluate(&ctx).await;
        assert_eq!(decision, PolicyDecision::Allow);
    }

    #[test]
    fn test_confirmation_hash_is_stable() {
        let h1 = ConfirmationPolicy::hash_args(&json!({"a": 1, "b": "x"}));
        let h2 = ConfirmationPolicy::hash_args(&json!({"a": 1, "b": "x"}));
        let h3 = ConfirmationPolicy::hash_args(&json!({"a": 1, "b": "y"}));
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
