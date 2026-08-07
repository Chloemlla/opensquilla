//! Concrete policy-chain checks for the dispatch pipeline.
//!
//! Mirrors the Python `opensquilla.tools.policy.checks` module: ordered
//! authorization checks that run against the effective `ToolContext` before a
//! tool executes. Each check is a pure decision over a [`DispatchPolicyInput`];
//! the orchestrator (dispatch) builds the failure envelope and log event.
//!
//! Chain order matches Python: owner_only, denied_tools, private-memory scope,
//! allowlist, profile, permission matrix. First denial wins.

use crate::context::{CallerKind, ToolContext};
use crate::policy_runtime::private_memory_read_tool_denied;
use crate::visibility::{ToolVisibilitySpec, profile_allows_tool, resolve_profile};

/// Inputs available to every policy check.
#[derive(Debug, Clone)]
pub struct DispatchPolicyInput<'a> {
    /// The tool being invoked.
    pub tool_name: &'a str,
    /// The effective request context (`None` always allows).
    pub ctx: Option<&'a ToolContext>,
    /// The registered tool's visibility spec.
    pub spec: ToolVisibilitySpec,
    /// Channel kind for channel callers.
    pub channel_kind: Option<&'a str>,
    /// Source kind (e.g. "webui").
    pub source_kind: Option<&'a str>,
}

impl<'a> DispatchPolicyInput<'a> {
    /// Build a minimal input with a conservative spec.
    ///
    /// The spec is not owner-only: Rust `ToolDefinition`s carry no owner-only
    /// flag, so ownership is expressed through allow/deny lists and the
    /// permission matrix.
    pub fn new(tool_name: &'a str, ctx: Option<&'a ToolContext>) -> Self {
        let spec = ToolVisibilitySpec {
            owner_only: false,
            ..ToolVisibilitySpec::new(tool_name)
        };
        Self {
            tool_name,
            ctx,
            spec,
            channel_kind: ctx.and_then(|c| c.channel_kind.as_deref()),
            source_kind: ctx.and_then(|c| c.source_kind.as_deref()),
        }
    }
}

/// A single policy-check evaluation outcome.
#[derive(Debug, Clone)]
pub struct PolicyCheckDecision {
    /// Whether the check passes.
    pub allowed: bool,
    /// Machine reason code for logging (denials only).
    pub reason: Option<&'static str>,
    /// `error_class` for the failure envelope (denials only).
    pub error_class: &'static str,
    /// User-facing denial message (denials only).
    pub user_message: Option<String>,
    /// Structured log event name (denials only).
    pub event: Option<&'static str>,
}

impl PolicyCheckDecision {
    /// An allowing decision.
    pub fn allow() -> Self {
        Self {
            allowed: true,
            reason: None,
            error_class: "PolicyDenied",
            user_message: None,
            event: None,
        }
    }

    fn deny(
        reason: &'static str,
        error_class: &'static str,
        user_message: String,
        event: &'static str,
    ) -> Self {
        Self {
            allowed: false,
            reason: Some(reason),
            error_class,
            user_message: Some(user_message),
            event: Some(event),
        }
    }
}

/// The `DispatchInput` shape passed through the chain mirrors the Python
/// `opensquilla.tools.policy.types.DispatchInput`.
impl<'a> DispatchPolicyInput<'a> {
    /// Convenience: the registered tool's visibility spec.
    pub fn registered_spec(&self) -> &ToolVisibilitySpec {
        &self.spec
    }
}

/// A single policy-check implementation.
pub trait PolicyCheck: Send + Sync {
    /// Stable check name for logging.
    fn name(&self) -> &'static str;
    /// Evaluate the check against the dispatch input.
    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision;
}

/// Reject `owner_only` tools when `ctx.is_owner` is false.
pub struct OwnerOnlyPolicy;

impl PolicyCheck for OwnerOnlyPolicy {
    fn name(&self) -> &'static str {
        "owner_only"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        let Some(ctx) = input.ctx else {
            return PolicyCheckDecision::allow();
        };
        if input.spec.owner_only && !ctx.is_owner {
            return PolicyCheckDecision::deny(
                "owner_only",
                "OwnerOnly",
                format!("Tool '{}' restricted to owner.", input.tool_name),
                "dispatch.defense_in_depth_block",
            );
        }
        PolicyCheckDecision::allow()
    }
}

/// Reject tools that appear in `ctx.denied_tools`.
pub struct DenyListPolicy;

impl PolicyCheck for DenyListPolicy {
    fn name(&self) -> &'static str {
        "denied"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        let Some(ctx) = input.ctx else {
            return PolicyCheckDecision::allow();
        };
        if ctx.denied_tools.contains(input.tool_name) {
            return PolicyCheckDecision::deny(
                "denied",
                "PolicyDenied",
                format!("Tool '{}' not available in this context.", input.tool_name),
                "dispatch.defense_in_depth_block",
            );
        }
        PolicyCheckDecision::allow()
    }
}

/// Reject private-memory reads from contexts that may not access them.
pub struct PrivateMemoryScopePolicy;

impl PolicyCheck for PrivateMemoryScopePolicy {
    fn name(&self) -> &'static str {
        "private_memory_scope"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        if input
            .ctx
            .map(|ctx| private_memory_read_tool_denied(ctx, input.tool_name))
            .unwrap_or(false)
        {
            return PolicyCheckDecision::deny(
                "private_memory_scope",
                "PolicyDenied",
                format!("Tool '{}' not available in this context.", input.tool_name),
                "dispatch.defense_in_depth_block",
            );
        }
        PolicyCheckDecision::allow()
    }
}

/// Reject tools absent from a non-`None` `ctx.allowed_tools`.
pub struct AllowListPolicy;

impl PolicyCheck for AllowListPolicy {
    fn name(&self) -> &'static str {
        "allowlist"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        let Some(ctx) = input.ctx else {
            return PolicyCheckDecision::allow();
        };
        if let Some(allowed) = &ctx.allowed_tools {
            if !allowed.contains(input.tool_name) {
                return PolicyCheckDecision::deny(
                    "not_allowed",
                    "PolicyDenied",
                    format!("Tool '{}' not available in this context.", input.tool_name),
                    "dispatch.defense_in_depth_block",
                );
            }
        }
        PolicyCheckDecision::allow()
    }
}

/// Reject tools the resolved tool profile does not allow.
pub struct ProfilePolicy;

impl PolicyCheck for ProfilePolicy {
    fn name(&self) -> &'static str {
        "profile"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        let Some(ctx) = input.ctx else {
            return PolicyCheckDecision::allow();
        };
        let profile = resolve_profile(Some(ctx));
        if !profile_allows_tool(input.tool_name, profile, ctx.allowed_tools.as_ref()) {
            return PolicyCheckDecision::deny(
                "profile_denied",
                "PolicyDenied",
                format!("Tool '{}' not available in this context.", input.tool_name),
                "dispatch.profile_block",
            );
        }
        PolicyCheckDecision::allow()
    }
}

/// Channel tool surface classification.
enum ChannelSurfaceKind<'a> {
    WebUi,
    Channel(&'a str),
}

fn channel_surface<'a>(input: &DispatchPolicyInput<'a>) -> ChannelSurfaceKind<'a> {
    if input
        .source_kind
        .map(|s| s.trim().to_ascii_lowercase() == "webui")
        .unwrap_or(false)
    {
        ChannelSurfaceKind::WebUi
    } else {
        ChannelSurfaceKind::Channel(input.channel_kind.unwrap_or("dm"))
    }
}

/// Whether a tool is allowed for a channel principal using the declarative
/// channel groups from `policy_config` plus the hard non-owner deny list.
///
/// This is the Rust stand-in for the Python `safety.permission_matrix`
/// module, which lives outside the tools crate.
fn channel_tool_allowed(
    tool_name: &str,
    surface: &ChannelSurfaceKind<'_>,
    is_operator: bool,
    explicitly_allowed: Option<&std::collections::BTreeSet<String>>,
) -> bool {
    use crate::policy_config::tool_group;
    use crate::visibility::{CHANNEL_DEFAULT_ALLOW, CHANNEL_HARD_DENY_NON_OWNER};

    if CHANNEL_DEFAULT_ALLOW.contains(&tool_name) {
        return true;
    }
    if CHANNEL_HARD_DENY_NON_OWNER.contains(&tool_name) {
        return false;
    }
    if is_operator {
        // A verified channel administrator is the same principal as the
        // WebUI owner; operator-only tools stay reachable.
        return true;
    }
    let group_name = match surface {
        ChannelSurfaceKind::WebUi => "channel:chat",
        ChannelSurfaceKind::Channel(kind) => match *kind {
            "media" => "channel:media",
            "doc" => "channel:doc",
            "wiki" => "channel:wiki",
            "drive" => "channel:drive",
            "web" => "channel:chat",
            "webui" => "channel:chat",
            _ => "channel:chat",
        },
    };
    if let Some(group) = tool_group(group_name) {
        if group.contains(&tool_name) {
            return true;
        }
    }
    explicitly_allowed
        .map(|allowed| allowed.contains(tool_name))
        .unwrap_or(false)
}

/// Run the channel permission matrix when `caller_kind == CHANNEL`.
pub struct PermissionMatrixPolicy;

impl PolicyCheck for PermissionMatrixPolicy {
    fn name(&self) -> &'static str {
        "permission_matrix"
    }

    fn evaluate(&self, input: &DispatchPolicyInput<'_>) -> PolicyCheckDecision {
        let Some(ctx) = input.ctx else {
            return PolicyCheckDecision::allow();
        };
        if ctx.caller_kind != CallerKind::Channel {
            return PolicyCheckDecision::allow();
        }
        let surface = channel_surface(input);
        let is_operator = ctx.channel_admin_verified;
        if !channel_tool_allowed(
            input.tool_name,
            &surface,
            is_operator,
            ctx.allowed_tools.as_ref(),
        ) {
            return PolicyCheckDecision::deny(
                "permission_matrix",
                "UnsupportedSurface",
                format!(
                    "Tool '{}' denied for this channel surface.",
                    input.tool_name
                ),
                "dispatch.permission_matrix_block",
            );
        }
        PolicyCheckDecision::allow()
    }
}

/// The ordered policy chain (mirrors `opensquilla.tools.policy.chain`).
pub const POLICY_CHAIN: &[&dyn PolicyCheck] = &[
    &OwnerOnlyPolicy,
    &DenyListPolicy,
    &PrivateMemoryScopePolicy,
    &AllowListPolicy,
    &ProfilePolicy,
    &PermissionMatrixPolicy,
];

/// Run the chain in order; return the first denial or `None` (allow).
pub fn run_policy_chain(input: &DispatchPolicyInput<'_>) -> Option<PolicyCheckDecision> {
    for check in POLICY_CHAIN {
        let decision = check.evaluate(input);
        if !decision.allowed {
            return Some(decision);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{CallerKind, InteractionMode};
    use std::collections::BTreeSet;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[allow(dead_code)]
    fn owner_spec() -> ToolVisibilitySpec {
        ToolVisibilitySpec {
            owner_only: false,
            exposed_by_default: true,
            ..ToolVisibilitySpec::new("tool")
        }
    }

    #[test]
    fn none_context_always_allows() {
        let input = DispatchPolicyInput::new("exec_command", None);
        assert!(run_policy_chain(&input).is_none());
    }

    #[test]
    fn owner_only_denied_for_non_owner() {
        let ctx = ToolContext {
            is_owner: false,
            ..Default::default()
        };
        let spec = ToolVisibilitySpec {
            owner_only: true,
            ..ToolVisibilitySpec::new("admin_tool")
        };
        let input = DispatchPolicyInput {
            tool_name: "admin_tool",
            ctx: Some(&ctx),
            spec: spec.clone(),
            channel_kind: None,
            source_kind: None,
        };
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("owner_only"));
        assert_eq!(decision.error_class, "OwnerOnly");

        let owner = ToolContext::owner();
        let input = DispatchPolicyInput {
            tool_name: "admin_tool",
            ctx: Some(&owner),
            spec: spec.clone(),
            channel_kind: None,
            source_kind: None,
        };
        assert!(run_policy_chain(&input).is_none());
    }

    #[test]
    fn deny_list_blocks() {
        let ctx = ToolContext {
            denied_tools: set(&["exec_command"]),
            ..Default::default()
        };
        let input = DispatchPolicyInput::new("exec_command", Some(&ctx));
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("denied"));
    }

    #[test]
    fn allow_list_blocks_absent() {
        let ctx = ToolContext {
            allowed_tools: Some(set(&["read_file"])),
            ..Default::default()
        };
        let input = DispatchPolicyInput::new("exec_command", Some(&ctx));
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("not_allowed"));
    }

    #[test]
    fn private_memory_scope_for_subagent() {
        let mut ctx = ToolContext::owner();
        ctx.caller_kind = CallerKind::Subagent;
        let input = DispatchPolicyInput::new("memory_search", Some(&ctx));
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("private_memory_scope"));
        // Non-private tool passes even for a subagent.
        let input = DispatchPolicyInput::new("read_file", Some(&ctx));
        assert!(run_policy_chain(&input).is_none());
    }

    #[test]
    fn profile_denies_missing_channel_tool() {
        let mut ctx = ToolContext {
            is_owner: false,
            ..Default::default()
        };
        ctx.caller_kind = CallerKind::Channel;
        let input = DispatchPolicyInput::new("write_file", Some(&ctx));
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("profile_denied"));
    }

    #[test]
    fn permission_matrix_denies_hard_deny_for_channel() {
        // Owner channel caller: the owner-full profile passes, but the
        // channel permission matrix still hard-denies host execution tools.
        let mut ctx = ToolContext::owner();
        ctx.caller_kind = CallerKind::Channel;
        let input = DispatchPolicyInput::new("exec_command", Some(&ctx));
        let decision = run_policy_chain(&input).expect("denial");
        assert_eq!(decision.reason, Some("permission_matrix"));
        assert_eq!(decision.error_class, "UnsupportedSurface");
    }

    #[test]
    fn permission_matrix_operator_keeps_operator_tools() {
        let mut ctx = ToolContext::owner();
        ctx.caller_kind = CallerKind::Channel;
        ctx.interaction_mode = InteractionMode::Unattended;
        ctx.channel_admin_verified = true;
        let input = DispatchPolicyInput::new("agents_list", Some(&ctx));
        assert!(run_policy_chain(&input).is_none());
    }

    #[test]
    fn first_denial_wins_owner_only() {
        let mut ctx = ToolContext::owner();
        ctx.is_owner = false;
        ctx.denied_tools = set(&["admin_tool"]);
        let spec = ToolVisibilitySpec {
            owner_only: true,
            ..ToolVisibilitySpec::new("admin_tool")
        };
        let input = DispatchPolicyInput {
            tool_name: "admin_tool",
            ctx: Some(&ctx),
            spec: spec,
            channel_kind: None,
            source_kind: None,
        };
        let decision = run_policy_chain(&input).expect("denial");
        // OwnerOnly runs before the deny list in the chain.
        assert_eq!(decision.reason, Some("owner_only"));
    }
}
