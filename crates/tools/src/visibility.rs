//! Tool profile, context, and visibility policy helpers.
//!
//! Mirrors the Python `opensquilla.tools.visibility` module: resolves the
//! effective tool profile for a context, filters tool surfaces by profile,
//! and decides which registered tools are visible.

use crate::context::{CallerKind, InteractionMode, PlanAccess, ToolContext};
use crate::policy_runtime::{ToolSurfaceCapabilities, resolve_runtime_tool_surface};
use crate::registry::ToolDefinition;
use std::collections::BTreeSet;

/// Tool visibility profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    OwnerFull,
    ChannelDefault,
}

impl ToolProfile {
    /// Parse a profile from a string, or `None`.
    pub fn from_str_opt(value: &str) -> Option<ToolProfile> {
        match value.trim().to_ascii_lowercase().as_str() {
            "owner_full" => Some(ToolProfile::OwnerFull),
            "channel_default" => Some(ToolProfile::ChannelDefault),
            _ => None,
        }
    }
}

/// Tools a non-owner channel caller is allowed to invoke by default.
pub const CHANNEL_DEFAULT_ALLOW: &[&str] = &[
    "cron",
    "git_diff",
    "git_log",
    "git_status",
    "glob_search",
    "grep_search",
    "image",
    "image_generate",
    "audio_provider_capabilities",
    "dubbing_download",
    "dubbing_generate",
    "dubbing_status",
    "list_dir",
    "memory_get",
    "memory_search",
    "music_generate",
    "pdf",
    "publish_artifact",
    "song_generate",
    "create_csv",
    "create_pdf_report",
    "create_pptx",
    "create_xlsx",
    "read_file",
    "session_status",
    "sessions_history",
    "sessions_list",
    "tts",
    "voice_clone",
    "voice_convert",
    "voice_search",
    "web_discover",
    "web_fetch",
    "web_search",
];

/// Tools a non-owner channel caller is hard-denied.
pub const CHANNEL_HARD_DENY_NON_OWNER: &[&str] = &[
    "apply_patch",
    "background_process",
    "edit_file",
    "exec_command",
    "execute_code",
    "git_commit",
    "write_file",
];

/// Env override that forces a tool profile.
const TOOL_PROFILE_ENV: &str = "OPENSQUILLA_TOOL_PROFILE";

/// Whether a tool passes the profile capability boundary.
pub fn profile_allows_tool(
    tool_name: &str,
    profile: ToolProfile,
    explicitly_allowed: Option<&BTreeSet<String>>,
) -> bool {
    match profile {
        ToolProfile::OwnerFull => true,
        ToolProfile::ChannelDefault => {
            if CHANNEL_DEFAULT_ALLOW.contains(&tool_name) {
                return true;
            }
            if CHANNEL_HARD_DENY_NON_OWNER.contains(&tool_name) {
                return false;
            }
            explicitly_allowed
                .map(|allowed| allowed.contains(tool_name))
                .unwrap_or(false)
        }
    }
}

/// Filter a list of tool definitions by profile.
pub fn filter_by_profile(
    tools: &[ToolDefinition],
    profile: ToolProfile,
    ctx: Option<&ToolContext>,
) -> Vec<ToolDefinition> {
    match profile {
        ToolProfile::OwnerFull => tools.to_vec(),
        ToolProfile::ChannelDefault => {
            let explicit = ctx.and_then(|c| c.allowed_tools.as_ref());
            tools
                .iter()
                .filter(|tool| profile_allows_tool(&tool.name, profile, explicit))
                .cloned()
                .collect()
        }
    }
}

/// Resolve the effective tool profile for a context (env override wins).
pub fn resolve_profile(ctx: Option<&ToolContext>) -> ToolProfile {
    if let Ok(override_value) = std::env::var(TOOL_PROFILE_ENV) {
        let override_value = override_value.trim().to_string();
        if !override_value.is_empty() {
            if let Some(profile) = ToolProfile::from_str_opt(&override_value) {
                return profile;
            }
        }
    }
    if let Some(ctx) = ctx {
        if ctx.caller_kind == CallerKind::Channel && !ctx.is_owner {
            return ToolProfile::ChannelDefault;
        }
    }
    ToolProfile::OwnerFull
}

/// A default owner context for the agent caller.
pub fn default_tool_context() -> ToolContext {
    ToolContext::owner()
}

/// Build a context for a named entry-point profile.
pub fn tool_context_for_profile(profile: &str) -> ToolContext {
    match profile {
        "subagent" => ToolContext {
            is_owner: true,
            caller_kind: CallerKind::Subagent,
            interaction_mode: InteractionMode::Unattended,
            denied_tools: crate::context::SUBAGENT_TOOL_DENY
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        },
        "cron" => ToolContext {
            is_owner: false,
            caller_kind: CallerKind::Cron,
            interaction_mode: InteractionMode::Unattended,
            allowed_tools: Some(
                crate::context::CRON_AGENT_ALLOW
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            denied_tools: crate::context::CRON_AGENT_DENY
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        },
        _ => default_tool_context(),
    }
}

/// Parse an interaction mode string, or `None`.
pub fn parse_interaction_mode(value: Option<&str>) -> Option<InteractionMode> {
    value.and_then(InteractionMode::from_str_opt)
}

/// Resolve the effective tool context for a request, applying the runtime
/// tool-surface denylist.
#[allow(clippy::too_many_arguments)]
pub fn effective_tool_context(
    session_key: Option<&str>,
    agent_id: Option<&str>,
    caller_kind: Option<&str>,
    interaction_mode: Option<&str>,
    capabilities: Option<&ToolSurfaceCapabilities>,
    is_owner: bool,
) -> ToolContext {
    let explicit_kind = caller_kind.and_then(CallerKind::from_str_opt);
    let explicit_interaction = parse_interaction_mode(interaction_mode);
    let caps = capabilities.copied().unwrap_or_default();
    let agent_id = agent_id.unwrap_or("main").to_string();
    let session_key_owned = session_key.map(|s| s.to_string());

    let is_subagent = explicit_kind == Some(CallerKind::Subagent)
        || session_key
            .map(|key| key.starts_with("subagent:"))
            .unwrap_or(false);
    if is_subagent {
        let ctx = ToolContext {
            is_owner,
            caller_kind: CallerKind::Subagent,
            interaction_mode: explicit_interaction.unwrap_or(InteractionMode::Unattended),
            agent_id,
            denied_tools: crate::context::SUBAGENT_TOOL_DENY
                .iter()
                .map(|s| s.to_string())
                .collect(),
            session_key: session_key_owned,
            ..Default::default()
        };
        return resolve_runtime_tool_surface(&ctx, &caps);
    }

    let is_cron = explicit_kind == Some(CallerKind::Cron)
        || session_key
            .map(|key| key.starts_with("cron:"))
            .unwrap_or(false);
    if is_cron {
        let mode = explicit_interaction.unwrap_or(InteractionMode::Unattended);
        let mut ctx = ToolContext {
            is_owner,
            caller_kind: CallerKind::Cron,
            interaction_mode: mode,
            agent_id,
            session_key: session_key_owned,
            ..Default::default()
        };
        if !is_owner {
            ctx.allowed_tools = Some(
                crate::context::CRON_AGENT_ALLOW
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            );
            ctx.denied_tools = crate::context::CRON_AGENT_DENY
                .iter()
                .map(|s| s.to_string())
                .collect();
        }
        return resolve_runtime_tool_surface(&ctx, &caps);
    }

    if explicit_kind == Some(CallerKind::Channel) {
        let ctx = ToolContext {
            is_owner,
            caller_kind: CallerKind::Channel,
            interaction_mode: explicit_interaction.unwrap_or(InteractionMode::Interactive),
            agent_id,
            allowed_tools: if is_owner {
                None
            } else {
                Some(CHANNEL_DEFAULT_ALLOW.iter().map(|s| s.to_string()).collect())
            },
            session_key: session_key_owned,
            ..Default::default()
        };
        return resolve_runtime_tool_surface(&ctx, &caps);
    }

    let ctx = ToolContext {
        is_owner,
        caller_kind: CallerKind::Agent,
        interaction_mode: explicit_interaction.unwrap_or(InteractionMode::Interactive),
        agent_id,
        session_key: session_key_owned,
        ..Default::default()
    };
    resolve_runtime_tool_surface(&ctx, &caps)
}

/// The spec fields visibility decisions consult.
#[derive(Debug, Clone)]
pub struct ToolVisibilitySpec {
    /// Registered tool name.
    pub name: String,
    /// Whether the tool is exposed by default.
    pub exposed_by_default: bool,
    /// Whether the tool is owner-only.
    pub owner_only: bool,
    /// Plan collaboration-mode access level.
    pub plan_access: PlanAccess,
}

impl ToolVisibilitySpec {
    /// Build a spec with conservative defaults (not exposed, owner-only,
    /// plan-denied).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            exposed_by_default: false,
            owner_only: true,
            plan_access: PlanAccess::Deny,
        }
    }
}

/// Decide whether a registered tool is visible to a context.
pub fn is_tool_visible(spec: &ToolVisibilitySpec, ctx: Option<&ToolContext>) -> bool {
    // Plan collaboration-mode boundary.
    if let Some(ctx) = ctx {
        if ctx.is_plan_mode() && spec.plan_access == PlanAccess::Deny {
            return false;
        }
    }
    let explicitly_allowed = ctx
        .and_then(|c| c.allowed_tools.as_ref())
        .map(|allowed| allowed.contains(&spec.name))
        .unwrap_or(false);
    let surfaced = ctx
        .and_then(|c| c.surfaced_tools.as_ref())
        .map(|surfaced| surfaced.contains(&spec.name))
        .unwrap_or(false);
    let channel_profile_visible = ctx
        .map(|c| {
            c.caller_kind == CallerKind::Channel
                && !c.is_owner
                && profile_allows_tool(&spec.name, ToolProfile::ChannelDefault, c.allowed_tools.as_ref())
        })
        .unwrap_or(false);

    if !spec.exposed_by_default && !explicitly_allowed && !surfaced && !channel_profile_visible {
        return false;
    }
    if let Some(ctx) = ctx {
        if spec.owner_only && !ctx.is_owner {
            return false;
        }
        if let Some(allowed) = &ctx.allowed_tools {
            if !allowed.contains(&spec.name) {
                return false;
            }
        }
        if ctx.denied_tools.contains(&spec.name) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn channel_default_profile_gating() {
        assert!(profile_allows_tool(
            "web_search",
            ToolProfile::ChannelDefault,
            None
        ));
        assert!(!profile_allows_tool(
            "write_file",
            ToolProfile::ChannelDefault,
            None
        ));
        assert!(profile_allows_tool(
            "custom_tool",
            ToolProfile::ChannelDefault,
            Some(&set(&["custom_tool"]))
        ));
        assert!(!profile_allows_tool(
            "custom_tool",
            ToolProfile::ChannelDefault,
            Some(&set(&["other"]))
        ));
        assert!(profile_allows_tool("anything", ToolProfile::OwnerFull, None));
    }

    #[test]
    fn resolve_profile_channel_non_owner() {
        let mut ctx = ToolContext::owner();
        ctx.caller_kind = CallerKind::Channel;
        ctx.is_owner = false;
        assert_eq!(resolve_profile(Some(&ctx)), ToolProfile::ChannelDefault);

        ctx.is_owner = true;
        assert_eq!(resolve_profile(Some(&ctx)), ToolProfile::OwnerFull);
    }

    #[test]
    fn tool_context_for_profiles() {
        let subagent = tool_context_for_profile("subagent");
        assert_eq!(subagent.caller_kind, CallerKind::Subagent);
        assert_eq!(subagent.interaction_mode, InteractionMode::Unattended);
        assert!(subagent.denied_tools.contains("memory_search"));

        let cron = tool_context_for_profile("cron");
        assert_eq!(cron.caller_kind, CallerKind::Cron);
        assert!(!cron.is_owner);
        assert!(cron.allowed_tools.as_ref().unwrap().contains("web_search"));
        assert!(cron.denied_tools.contains("write_file"));

        assert_eq!(
            tool_context_for_profile("other").caller_kind,
            CallerKind::Agent
        );
    }

    #[test]
    fn effective_context_subagent() {
        let ctx = effective_tool_context(
            Some("subagent:1"),
            Some("main"),
            None,
            None,
            None,
            true,
        );
        assert_eq!(ctx.caller_kind, CallerKind::Subagent);
        assert_eq!(ctx.interaction_mode, InteractionMode::Unattended);
        assert!(ctx.denied_tools.contains("cron"));
    }

    #[test]
    fn effective_context_cron_non_owner() {
        let ctx = effective_tool_context(
            Some("cron:job1"),
            Some("main"),
            None,
            None,
            None,
            false,
        );
        assert_eq!(ctx.caller_kind, CallerKind::Cron);
        assert!(ctx.allowed_tools.as_ref().unwrap().contains("web_search"));
        assert!(ctx.denied_tools.contains("exec_command"));
    }

    #[test]
    fn effective_context_channel_allowlist() {
        let ctx = effective_tool_context(None, Some("main"), Some("channel"), None, None, false);
        assert_eq!(ctx.caller_kind, CallerKind::Channel);
        assert!(ctx.allowed_tools.as_ref().unwrap().contains("web_search"));
        assert!(!ctx.allowed_tools.as_ref().unwrap().contains("write_file"));
    }

    #[test]
    fn is_tool_visible_owner_only_and_denied() {
        let ctx = ToolContext::owner();
        let spec = ToolVisibilitySpec {
            owner_only: true,
            exposed_by_default: true,
            ..ToolVisibilitySpec::new("admin_tool")
        };
        // Owner sees exposed-by-default tools.
        let exposed = ToolVisibilitySpec {
            exposed_by_default: true,
            owner_only: false,
            ..ToolVisibilitySpec::new("read_file")
        };
        assert!(is_tool_visible(&exposed, Some(&ctx)));
        // Owner-only tool not visible to a non-owner.
        let non_owner = ToolContext {
            is_owner: false,
            ..Default::default()
        };
        assert!(!is_tool_visible(&spec, Some(&non_owner)));
        assert!(is_tool_visible(&spec, Some(&ctx)));
        // A non-exposed tool is not visible unless allowed/surfaced.
        let hidden = ToolVisibilitySpec::new("hidden");
        assert!(!is_tool_visible(&hidden, Some(&ctx)));
        // Denied tool is not visible even when exposed.
        let denied_ctx = ToolContext {
            denied_tools: set(&["read_file"]),
            ..Default::default()
        };
        assert!(!is_tool_visible(&exposed, Some(&denied_ctx)));
    }

    #[test]
    fn is_tool_visible_plan_mode_denies() {
        let mut plan_ctx = ToolContext::owner();
        plan_ctx.collaboration_mode = "plan".to_string();
        let spec = ToolVisibilitySpec::new("exec_command");
        assert!(!is_tool_visible(&spec, Some(&plan_ctx)));

        let read_only_spec = ToolVisibilitySpec {
            plan_access: PlanAccess::ReadOnly,
            exposed_by_default: true,
            owner_only: false,
            ..ToolVisibilitySpec::new("read_file")
        };
        assert!(is_tool_visible(&read_only_spec, Some(&plan_ctx)));
    }

    #[test]
    fn is_tool_visible_surfaced() {
        let ctx = ToolContext {
            surfaced_tools: Some(set(&["hidden_tool"])),
            ..Default::default()
        };
        let spec = ToolVisibilitySpec {
            owner_only: false,
            ..ToolVisibilitySpec::new("hidden_tool")
        };
        assert!(is_tool_visible(&spec, Some(&ctx)));
    }
}
