//! Runtime tool-surface capability detection and denylist resolution.
//!
//! Mirrors the Python `opensquilla.tools.policy_runtime` module: determines
//! which registered tools can actually work given the injected runtime
//! dependencies, and folds that into a context's allowed/denied tool sets.

use crate::context::{CallerKind, InteractionMode, ToolContext};
use std::collections::BTreeSet;

const PRIVATE_MEMORY_READ_TOOL_NAMES: &[&str] = &["memory_get", "memory_search", "session_search"];
const IMAGE_GENERATION_TOOL_NAMES: &[&str] = &["image_generate"];
const SESSION_READ_TOOL_NAMES: &[&str] = &["session_status", "sessions_history", "sessions_list"];
const SESSION_RUNTIME_TOOL_NAMES: &[&str] = &["sessions_send", "sessions_spawn", "sessions_yield"];
const CHANNEL_RUNTIME_TOOL_NAMES: &[&str] = &["message"];
const ADMIN_RUNTIME_TOOL_NAMES: &[&str] = &["agents_list", "subagents"];
const GATEWAY_RUNTIME_TOOL_NAMES: &[&str] = &["gateway"];
const SCHEDULER_RUNTIME_TOOL_NAMES: &[&str] = &["cron"];

/// Runtime dependencies that determine whether registered tools can work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolSurfaceCapabilities {
    pub session_manager: bool,
    pub task_runtime: bool,
    pub scheduler: bool,
    pub gateway_config: bool,
    pub channel_backing: bool,
    pub image_generation: bool,
}

impl Default for ToolSurfaceCapabilities {
    fn default() -> Self {
        Self {
            session_manager: false,
            task_runtime: false,
            scheduler: false,
            gateway_config: false,
            channel_backing: false,
            image_generation: true,
        }
    }
}

/// Return `true` when this context must not read private memory sources.
///
/// The Python implementation additionally consults the session-key prompt
/// injection gate; that check lives in the session crate and is out of scope
/// for the tools crate, so a session-keyed agent context defaults to allowed.
pub fn private_memory_read_tools_blocked(ctx: &ToolContext) -> bool {
    match ctx.caller_kind {
        CallerKind::Subagent => true,
        CallerKind::Cron => !ctx.is_owner,
        CallerKind::Channel => ctx.session_key.is_none(),
        _ => false,
    }
}

/// Return `true` when a specific tool call would read blocked private memory.
pub fn private_memory_read_tool_denied(ctx: &ToolContext, tool_name: &str) -> bool {
    PRIVATE_MEMORY_READ_TOOL_NAMES.contains(&tool_name) && private_memory_read_tools_blocked(ctx)
}

/// Resolve runtime-capability tool visibility into a context's denylist.
pub fn resolve_runtime_tool_surface(
    ctx: &ToolContext,
    capabilities: &ToolSurfaceCapabilities,
) -> ToolContext {
    let mut resolved = ctx.clone();
    let mut denied_tools: BTreeSet<String> = ctx.denied_tools.clone();
    let mut allowed_tools: Option<BTreeSet<String>> = ctx.allowed_tools.clone();

    if !capabilities.image_generation {
        denied_tools.extend(IMAGE_GENERATION_TOOL_NAMES.iter().map(|s| s.to_string()));
    }
    if !capabilities.session_manager {
        denied_tools.extend(SESSION_READ_TOOL_NAMES.iter().map(|s| s.to_string()));
        denied_tools.extend(SESSION_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
    }
    if !capabilities.task_runtime {
        denied_tools.extend(SESSION_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
    }
    if !capabilities.scheduler {
        denied_tools.extend(SCHEDULER_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
    }
    if !capabilities.gateway_config {
        denied_tools.extend(GATEWAY_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
    }

    if ctx.interaction_mode == InteractionMode::Unattended {
        if !capabilities.channel_backing {
            denied_tools.extend(CHANNEL_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
        }
        // A Channel turn is unattended at the process level, but an
        // authenticated channel administrator is the same principal as the
        // WebUI owner. Keep operator-only tools available for that verified
        // identity.
        let verified_channel_admin =
            ctx.caller_kind == CallerKind::Channel && ctx.channel_admin_verified;
        if !verified_channel_admin {
            denied_tools.extend(ADMIN_RUNTIME_TOOL_NAMES.iter().map(|s| s.to_string()));
        }
    }
    if private_memory_read_tools_blocked(ctx) {
        denied_tools.extend(PRIVATE_MEMORY_READ_TOOL_NAMES.iter().map(|s| s.to_string()));
    }

    if let Some(allowed) = &mut allowed_tools {
        allowed.retain(|tool| !denied_tools.contains(tool));
    }
    resolved.allowed_tools = allowed_tools;
    resolved.denied_tools = denied_tools;
    resolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{CallerKind, InteractionMode};

    #[test]
    fn private_memory_blocked_by_caller_kind() {
        let mut ctx = ToolContext::owner();
        ctx.caller_kind = CallerKind::Subagent;
        assert!(private_memory_read_tools_blocked(&ctx));
        assert!(private_memory_read_tool_denied(&ctx, "memory_get"));
        assert!(!private_memory_read_tool_denied(&ctx, "read_file"));

        let mut cron = ToolContext::owner();
        cron.caller_kind = CallerKind::Cron;
        cron.is_owner = false;
        assert!(private_memory_read_tools_blocked(&cron));

        let mut cron_owner = ToolContext::owner();
        cron_owner.caller_kind = CallerKind::Cron;
        cron_owner.is_owner = true;
        assert!(!private_memory_read_tools_blocked(&cron_owner));

        let mut channel = ToolContext::owner();
        channel.caller_kind = CallerKind::Channel;
        channel.session_key = None;
        assert!(private_memory_read_tools_blocked(&channel));
        channel.session_key = Some("s1".to_string());
        assert!(!private_memory_read_tools_blocked(&channel));
    }

    #[test]
    fn resolve_surface_denies_without_runtime() {
        let ctx = ToolContext::owner();
        let resolved = resolve_runtime_tool_surface(&ctx, &ToolSurfaceCapabilities::default());
        assert!(resolved.denied_tools.contains("cron"));
        assert!(resolved.denied_tools.contains("sessions_list"));
        // image_generation defaults to true, so image_generate is allowed.
        assert!(!resolved.denied_tools.contains("image_generate"));
        // Interactive mode does not deny channel/admin tools.
        assert!(!resolved.denied_tools.contains("message"));
        assert!(!resolved.denied_tools.contains("agents_list"));
    }

    #[test]
    fn resolve_surface_unattended_denies_admin_tools() {
        let mut ctx = ToolContext::owner();
        ctx.interaction_mode = InteractionMode::Unattended;
        let resolved = resolve_runtime_tool_surface(&ctx, &ToolSurfaceCapabilities::default());
        assert!(resolved.denied_tools.contains("agents_list"));
        assert!(resolved.denied_tools.contains("subagents"));
    }

    #[test]
    fn resolve_surface_verified_channel_admin_keeps_admin_tools() {
        let mut ctx = ToolContext::owner();
        ctx.interaction_mode = InteractionMode::Unattended;
        ctx.caller_kind = CallerKind::Channel;
        ctx.channel_admin_verified = true;
        let resolved = resolve_runtime_tool_surface(&ctx, &ToolSurfaceCapabilities::default());
        assert!(!resolved.denied_tools.contains("agents_list"));
        assert!(resolved.denied_tools.contains("message")); // no channel backing
    }

    #[test]
    fn resolve_surface_prunes_allowlist() {
        let ctx = ToolContext {
            allowed_tools: Some(["cron", "web_search"].iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        };
        let resolved = resolve_runtime_tool_surface(&ctx, &ToolSurfaceCapabilities::default());
        let allowed = resolved.allowed_tools.unwrap();
        assert!(!allowed.contains("cron"));
        assert!(allowed.contains("web_search"));
    }
}
