//! Declarative tool policy config and selector helpers.
//!
//! Mirrors the Python `opensquilla.tools.policy_config` module: declarative
//! `ToolPolicy` layers built from profile + allow/deny selectors, resolved
//! against a set of available tool names. Selectors can be exact tool names,
//! `group:*` names, `*`, or fnmatch-style patterns.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Type alias for the declarative policy config layer.
///
/// Kept distinct from `crate::policy::ToolPolicy` (the policy-chain rule
/// type) so the crate-root re-export names stay unambiguous.
pub type DeclarativeToolPolicy = ToolPolicy;

/// Error produced by invalid policy configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyConfigError {
    pub message: String,
}

impl std::fmt::Display for PolicyConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PolicyConfigError {}

fn policy_error(message: impl Into<String>) -> PolicyConfigError {
    PolicyConfigError {
        message: message.into(),
    }
}

/// Declarative tool policy layer.
///
/// `profile` sets the base allowlist; `allow` and `also_allow` add selectors;
/// `deny` removes selectors. `workspace_write_deny_globs` can further block
/// writes to matching workspace-relative paths.
#[derive(Debug, Clone, Default)]
pub struct ToolPolicy {
    /// Named profile that establishes the base allowlist.
    pub profile: Option<String>,
    /// Exact tool names / group names / patterns to allow.
    pub allow: BTreeSet<String>,
    /// Exact tool names / group names / patterns to deny.
    pub deny: BTreeSet<String>,
    /// Additional selectors allowed (union semantics).
    pub also_allow: BTreeSet<String>,
    /// Workspace-relative glob patterns whose writes are denied.
    pub workspace_write_deny_globs: BTreeSet<String>,
    /// Require a fresh full-file read before editing.
    pub file_edit_requires_fresh_read: Option<bool>,
    /// Allow flexible recovery for file edits.
    pub file_edit_flexible_recovery: Option<bool>,
    /// Sender-scoped sub-policies keyed by sender selector.
    pub by_sender: BTreeMap<String, ToolPolicy>,
}

impl ToolPolicy {
    fn is_empty(&self) -> bool {
        self.profile.is_none()
            && self.allow.is_empty()
            && self.deny.is_empty()
            && self.also_allow.is_empty()
            && self.workspace_write_deny_globs.is_empty()
            && self.file_edit_requires_fresh_read.is_none()
            && self.file_edit_flexible_recovery.is_none()
            && self.by_sender.is_empty()
    }
}

/// Coding mode (operator toggle): the in-session write tools that let the
/// agent hand-edit a repository. When coding mode is ON these are denied so
/// every code change is forced through the code-task plugin instead. Shell is
/// intentionally kept so the agent can still LAUNCH code-task.
pub const CODING_MODE_DENIED_TOOLS: &[&str] = &[
    "write_file",
    "edit_file",
    "apply_patch",
    "execute_code",
    "git_commit",
    "create_csv",
    "create_pdf_report",
    "create_pptx",
    "create_xlsx",
];

/// Tools to deny while the coding-mode toggle is on (empty when off).
pub fn coding_mode_denied_tools(coding_mode: bool) -> BTreeSet<String> {
    if coding_mode {
        CODING_MODE_DENIED_TOOLS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        BTreeSet::new()
    }
}

/// Named tool groups (a single member expansion).
pub(crate) fn tool_group(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "group:runtime" => &["exec_command", "background_process"],
        "group:fs" => &[
            "read_file",
            "write_file",
            "edit_file",
            "apply_patch",
            "list_dir",
            "glob_search",
            "grep_search",
        ],
        "group:sessions" => &[
            "sessions_list",
            "sessions_history",
            "sessions_send",
            "sessions_spawn",
            "session_status",
        ],
        "group:memory" => &["memory_search", "memory_get"],
        "group:web" => &["web_search", "web_discover", "web_fetch", "http_request"],
        "group:messaging" => &["message"],
        "channel:chat" => &[
            "message",
            "sessions_list",
            "sessions_history",
            "sessions_send",
            "session_status",
        ],
        "channel:media" => &[
            "create_csv",
            "create_pdf_report",
            "create_pptx",
            "create_xlsx",
            "image",
            "image_generate",
            "audio_provider_capabilities",
            "dubbing_download",
            "dubbing_generate",
            "dubbing_status",
            "music_generate",
            "pdf",
            "publish_artifact",
            "song_generate",
            "tts",
            "voice_clone",
            "voice_convert",
            "voice_search",
        ],
        "channel:doc" => &[
            "create_pdf_report",
            "web_discover",
            "web_fetch",
            "web_search",
        ],
        "channel:wiki" => &["web_discover", "web_fetch", "web_search"],
        "channel:drive" => &[
            "create_csv",
            "create_pdf_report",
            "create_pptx",
            "create_xlsx",
        ],
        "channel:scopes" => &[],
        "channel:perm" => &[],
        // Trusted host/gateway tools intentionally do not imply OS sandbox
        // execution.
        "group:trusted_host" => &[
            "install_skill_deps",
            "skill_install_community",
            "skill_create",
            "skill_edit",
            "skill_delete",
        ],
        _ => return None,
    })
}

const REPO_CODING_SOURCE_EDIT_TOOLS: &[&str] = &[
    "read_source",
    "edit_source",
    "read_file",
    "grep_search",
    "glob_search",
    "list_dir",
    "git_status",
    "git_diff",
    "retrieve_tool_result",
    "exec_command",
];

const REPO_CODING_SOURCE_EDIT_STRICT_TOOLS: &[&str] = &[
    "read_source",
    "edit_source",
    "grep_search",
    "glob_search",
    "git_status",
    "git_diff",
    "retrieve_tool_result",
    "exec_command",
];

const REPO_CODING_SOURCE_EDIT_V2_TOOLS: &[&str] = &[
    "read_source",
    "edit_source",
    "source_symbols",
    "grep_search",
    "glob_search",
    "git_status",
    "git_diff",
    "retrieve_tool_result",
    "exec_command",
];

const REPO_CODING_SOURCE_EDIT_BALANCED_TOOLS: &[&str] = &[
    "read_source",
    "edit_source",
    "create_source",
    "write_scratch",
    "source_symbols",
    "read_file",
    "grep_search",
    "glob_search",
    "list_dir",
    "git_status",
    "git_diff",
    "retrieve_tool_result",
    "exec_command",
];

const REPO_CODING_SCAFFOLD_EDIT_TOOLS: &[&str] = &[
    "exec_command",
    "read_file",
    "edit_file",
    "write_file",
    "glob_search",
    "grep_search",
    "list_dir",
    "git_status",
    "git_diff",
    "retrieve_tool_result",
];

/// Named tool profiles. `None` (the `full` profile) means "no allowlist
/// restriction".
fn profile_tools(profile: &str) -> Option<Option<&'static [&'static str]>> {
    let tools: Option<&'static [&'static str]> = match profile {
        "full" => None,
        "minimal" => Some(&["session_status"]),
        "memory_only" => tool_group("group:memory"),
        "coding" => return Some(Some(&*CODING_PROFILE)),
        "repo_coding_source_edit" => Some(REPO_CODING_SOURCE_EDIT_TOOLS),
        "repo_coding_source_edit_strict" => Some(REPO_CODING_SOURCE_EDIT_STRICT_TOOLS),
        "repo_coding_source_edit_v2" => Some(REPO_CODING_SOURCE_EDIT_V2_TOOLS),
        "repo_coding_source_edit_balanced" => Some(REPO_CODING_SOURCE_EDIT_BALANCED_TOOLS),
        "repo_coding_source_edit_patch_fallback" => {
            return Some(Some(&*SOURCE_EDIT_PATCH_FALLBACK_PROFILE));
        }
        "repo_coding_scaffold_edit" => Some(REPO_CODING_SCAFFOLD_EDIT_TOOLS),
        "repo_coding_scaffold_patch" => return Some(Some(&*SCAFFOLD_PATCH_PROFILE)),
        "messaging" => return Some(Some(&*MESSAGING_PROFILE)),
        _ => return None,
    };
    Some(tools)
}

fn merged_profile(groups: &[&'static str], extra: &[&'static str]) -> Vec<&'static str> {
    let mut set: BTreeSet<&'static str> = BTreeSet::new();
    for group in groups {
        set.extend(tool_group(group).unwrap_or(&[]).iter().copied());
    }
    set.extend(extra.iter().copied());
    set.into_iter().collect()
}

fn union_profile(base: &[&'static str], extra: &[&'static str]) -> Vec<&'static str> {
    let mut set: BTreeSet<&'static str> = base.iter().copied().collect();
    set.extend(extra.iter().copied());
    set.into_iter().collect()
}

static CODING_PROFILE: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    merged_profile(
        &[
            "group:fs",
            "group:runtime",
            "group:sessions",
            "group:memory",
        ],
        &[],
    )
});

static SOURCE_EDIT_PATCH_FALLBACK_PROFILE: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| {
        union_profile(REPO_CODING_SOURCE_EDIT_BALANCED_TOOLS, &["apply_patch"])
    });

static SCAFFOLD_PATCH_PROFILE: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| union_profile(REPO_CODING_SCAFFOLD_EDIT_TOOLS, &["apply_patch"]));

static MESSAGING_PROFILE: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    union_profile(
        tool_group("group:messaging").unwrap_or(&[]),
        &[
            "sessions_list",
            "sessions_history",
            "sessions_send",
            "session_status",
        ],
    )
});

/// Sender-scoped tool groups (selector layers never grant these to a channel).
const SENDER_SCOPED_TOOL_GROUPS: &[&str] = &["channel:perm"];

/// Minimal fnmatch-style matcher supporting `*`, `?`, and `[...]` classes.
pub(crate) fn fnmatchcase(pattern: &str, name: &str) -> bool {
    fn match_here(p: &[char], n: &[char]) -> bool {
        if p.is_empty() {
            return n.is_empty();
        }
        match p[0] {
            '*' => {
                // Collapse consecutive stars.
                let mut idx = 0;
                while idx < p.len() && p[idx] == '*' {
                    idx += 1;
                }
                if idx == p.len() {
                    return true;
                }
                let mut rest = n;
                loop {
                    if match_here(&p[idx..], rest) {
                        return true;
                    }
                    if rest.is_empty() {
                        return false;
                    }
                    rest = &rest[1..];
                }
            }
            '?' => !n.is_empty() && match_here(&p[1..], &n[1..]),
            '[' => {
                if n.is_empty() {
                    return false;
                }
                let (matched, consumed) = match_class(&p[1..], n[0]);
                if !matched {
                    return false;
                }
                match_here(&p[1 + consumed..], &n[1..])
            }
            c => !n.is_empty() && n[0] == c && match_here(&p[1..], &n[1..]),
        }
    }
    fn match_class(p: &[char], c: char) -> (bool, usize) {
        let negate = !p.is_empty() && (p[0] == '!' || p[0] == '^');
        let mut idx = if negate { 1 } else { 0 };
        let mut matched = false;
        while idx < p.len() {
            if p[idx] == ']' && idx > (if negate { 1 } else { 0 }) {
                break;
            }
            if idx + 2 < p.len() && p[idx + 1] == '-' && p[idx + 2] != ']' {
                if p[idx] <= c && c <= p[idx + 2] {
                    matched = true;
                }
                idx += 3;
            } else {
                if p[idx] == c {
                    matched = true;
                }
                idx += 1;
            }
        }
        // Find closing bracket for consumed count.
        let mut close = idx;
        while close < p.len() && p[close] != ']' {
            close += 1;
        }
        let consumed = if close < p.len() { close + 1 } else { p.len() };
        (if negate { !matched } else { matched }, consumed)
    }
    match_here(
        &pattern.chars().collect::<Vec<_>>(),
        &name.chars().collect::<Vec<_>>(),
    )
}

/// Expand a set of selectors (exact names, group names, `*`, fnmatch
/// patterns) against the available tool names.
pub fn expand_selectors(
    selectors: &BTreeSet<String>,
    available_tools: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut expanded: BTreeSet<String> = BTreeSet::new();
    for selector in selectors {
        let item = selector.trim();
        if item.is_empty() {
            continue;
        }
        if item == "*" {
            expanded.extend(available_tools.iter().cloned());
            continue;
        }
        if let Some(group) = tool_group(item) {
            for tool in group {
                if available_tools.contains(*tool) {
                    expanded.insert(tool.to_string());
                }
            }
            continue;
        }
        if item.contains(['*', '?', '[', ']']) {
            for tool in available_tools {
                if fnmatchcase(item, tool) {
                    expanded.insert(tool.clone());
                }
            }
            continue;
        }
        if available_tools.contains(item) {
            expanded.insert(item.to_string());
        }
    }
    expanded
}

/// Resolve a named profile to its allowlist (intersected with available).
///
/// Returns `Ok(None)` for the `full` profile (no restriction) and for an
/// absent/empty profile. Unknown profiles are an error.
pub fn profile_allowlist(
    profile: &str,
    available_tools: &BTreeSet<String>,
) -> Result<Option<BTreeSet<String>>, PolicyConfigError> {
    if profile.is_empty() {
        return Ok(None);
    }
    let key = profile.trim().to_ascii_lowercase();
    let expanded = profile_tools(&key)
        .ok_or_else(|| policy_error(format!("unknown tool profile: {profile}")))?;
    match expanded {
        None => Ok(None),
        Some(tools) => Ok(Some(
            tools
                .iter()
                .filter(|tool| available_tools.contains(**tool))
                .map(|tool| tool.to_string())
                .collect(),
        )),
    }
}

fn add_allowed(
    allowed_tools: Option<BTreeSet<String>>,
    additions: &BTreeSet<String>,
) -> Option<BTreeSet<String>> {
    match allowed_tools {
        None => None,
        Some(mut allowed) => {
            allowed.extend(additions.iter().cloned());
            Some(allowed)
        }
    }
}

/// Apply a base policy layer (global or agent) to allow/deny sets.
pub fn apply_base_policy(
    allowed_tools: Option<BTreeSet<String>>,
    denied_tools: &BTreeSet<String>,
    policy: &ToolPolicy,
    available_tools: &BTreeSet<String>,
    profile_overrides: bool,
) -> Result<(Option<BTreeSet<String>>, BTreeSet<String>), PolicyConfigError> {
    let mut allowed_tools = allowed_tools;
    let mut denied_tools = denied_tools.clone();

    if let Some(profile) = &policy.profile {
        let profile_allowed = profile_allowlist(profile, available_tools)?;
        if profile_allowed.is_some() || (profile_overrides && profile.eq_ignore_ascii_case("full"))
        {
            allowed_tools = profile_allowed;
        }
    }

    let mut allow_selectors: BTreeSet<String> = policy.allow.clone();
    allow_selectors.extend(policy.also_allow.iter().cloned());
    allowed_tools = add_allowed(
        allowed_tools,
        &expand_selectors(&allow_selectors, available_tools),
    );
    let denied = expand_selectors(&policy.deny, available_tools);
    denied_tools.extend(denied);
    if let Some(allowed) = &mut allowed_tools {
        allowed.retain(|tool| !denied_tools.contains(tool));
    }
    Ok((allowed_tools, denied_tools))
}

/// Remove denied tools from an allowlist.
pub fn remove_denied_from_allowed(
    allowed_tools: Option<BTreeSet<String>>,
    denied_tools: &BTreeSet<String>,
) -> Option<BTreeSet<String>> {
    allowed_tools.map(|mut allowed| {
        allowed.retain(|tool| !denied_tools.contains(tool));
        allowed
    })
}

/// Whether a sender selector matches a sender id.
pub fn matches_sender(selector: &str, sender_id: Option<&str>) -> bool {
    let normalized = selector.trim();
    if normalized == "*" {
        return true;
    }
    let sender_id = match sender_id {
        Some(id) if !id.is_empty() => id,
        _ => return false,
    };
    if let Some(colon) = normalized.find(':') {
        let (key, value) = normalized.split_at(colon);
        let value = &value[1..];
        if key.trim().eq_ignore_ascii_case("id") {
            return value == sender_id;
        }
        return false;
    }
    normalized == sender_id
}

/// Read a named field from a JSON object value.
fn get_field<'a>(value: &'a Value, name: &str) -> Option<&'a Value> {
    value.get(name)
}

/// Parse a value into a set of non-empty strings.
fn string_set(value: &Value) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    match value {
        Value::Null => {}
        Value::String(s) => {
            if !s.trim().is_empty() {
                result.insert(s.clone());
            }
        }
        Value::Array(items) => {
            for item in items {
                if let Some(s) = item.as_str() {
                    if !s.trim().is_empty() {
                        result.insert(s.to_string());
                    }
                }
            }
        }
        _ => {}
    }
    result
}

/// Parse a value into an optional boolean (supports string bools).
fn optional_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(*b),
        Value::String(s) => {
            let normalized = s.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => Some(!s.is_empty()),
            }
        }
        _ => Some(!value.is_null()),
    }
}

fn file_edit_requires_fresh_read_from_config(value: &Value) -> Option<bool> {
    get_field(value, "fileEditRequiresFreshRead")
        .or_else(|| get_field(value, "file_edit_requires_fresh_read"))
        .and_then(optional_bool)
}

fn file_edit_flexible_recovery_from_config(value: &Value) -> Option<bool> {
    get_field(value, "fileEditFlexibleRecovery")
        .or_else(|| get_field(value, "file_edit_flexible_recovery"))
        .and_then(optional_bool)
}

/// Parse a config-shaped value into a `ToolPolicy` (or `None`).
pub fn policy_from_config(value: &Value) -> Option<ToolPolicy> {
    if value.is_null() {
        return None;
    }
    if let Some(policy) = value.as_object().map(|_| policy_from_object(value)) {
        return policy;
    }
    None
}

fn policy_from_object(value: &Value) -> Option<ToolPolicy> {
    let tools_value = get_field(value, "tools");
    let sender_value =
        get_field(value, "toolsBySender").or_else(|| get_field(value, "tools_by_sender"));
    if tools_value.is_some() {
        let base = policy_from_config(tools_value.unwrap()).unwrap_or_default();
        let wrapper_by_sender = sender_policies_from_config(sender_value);
        let mut by_sender = base.by_sender.clone();
        for (selector, policy) in wrapper_by_sender {
            by_sender.insert(selector, policy);
        }
        let workspace_globs = get_field(value, "workspaceWriteDenyGlobs")
            .or_else(|| get_field(value, "workspace_write_deny_globs"))
            .map(string_set)
            .unwrap_or_default();
        let mut merged_globs = base.workspace_write_deny_globs.clone();
        merged_globs.extend(workspace_globs);
        let fresh_read = file_edit_requires_fresh_read_from_config(value);
        let flexible = file_edit_flexible_recovery_from_config(value);
        let policy = ToolPolicy {
            profile: base.profile.clone(),
            allow: base.allow.clone(),
            deny: base.deny.clone(),
            also_allow: base.also_allow.clone(),
            workspace_write_deny_globs: merged_globs,
            file_edit_requires_fresh_read: fresh_read.or(base.file_edit_requires_fresh_read),
            file_edit_flexible_recovery: flexible.or(base.file_edit_flexible_recovery),
            by_sender,
        };
        return if policy.is_empty() {
            None
        } else {
            Some(policy)
        };
    }

    let profile = get_field(value, "profile").and_then(|v| v.as_str());
    let policy = ToolPolicy {
        profile: profile.map(|s| s.to_string()),
        allow: get_field(value, "allow")
            .map(string_set)
            .unwrap_or_default(),
        deny: get_field(value, "deny").map(string_set).unwrap_or_default(),
        also_allow: get_field(value, "alsoAllow")
            .or_else(|| get_field(value, "also_allow"))
            .map(string_set)
            .unwrap_or_default(),
        workspace_write_deny_globs: get_field(value, "workspaceWriteDenyGlobs")
            .or_else(|| get_field(value, "workspace_write_deny_globs"))
            .map(string_set)
            .unwrap_or_default(),
        file_edit_requires_fresh_read: file_edit_requires_fresh_read_from_config(value),
        file_edit_flexible_recovery: file_edit_flexible_recovery_from_config(value),
        by_sender: sender_policies_from_config(
            sender_value
                .or_else(|| get_field(value, "by_sender").or_else(|| get_field(value, "bySender"))),
        ),
    };
    if policy.is_empty() {
        None
    } else {
        Some(policy)
    }
}

/// Parse a sender-selector -> policy mapping.
pub fn sender_policies_from_config(value: Option<&Value>) -> BTreeMap<String, ToolPolicy> {
    let mut policies = BTreeMap::new();
    let Some(value) = value else { return policies };
    let Some(map) = value.as_object() else {
        return policies;
    };
    for (selector, policy_value) in map {
        if let Some(policy) = policy_from_config(policy_value) {
            policies.insert(selector.clone(), policy);
        }
    }
    policies
}

/// Select the sender-scoped policy for a sender id, if any.
pub fn sender_policy(policy: &ToolPolicy, sender_id: Option<&str>) -> Option<ToolPolicy> {
    for (selector, candidate) in &policy.by_sender {
        if matches_sender(selector, sender_id) {
            return Some(candidate.clone());
        }
    }
    None
}

/// Resolve the agent policy from a config-shaped value.
pub fn agent_policy_from_config(config: &Value, agent_id: &str) -> Option<ToolPolicy> {
    let agents = get_field(config, "agents")?;
    if let Some(map) = agents.as_object() {
        let entry = map.get(agent_id)?;
        return policy_from_config(get_field(entry, "tools")?);
    }
    let entries: &Value = if let Some(list) = agents.as_array() {
        if list.is_empty() {
            return None;
        }
        agents
    } else {
        get_field(&agents, "list")?
    };
    let Some(list) = entries.as_array() else {
        return None;
    };
    for entry in list {
        if get_field(entry, "id").and_then(|v| v.as_str()) == Some(agent_id) {
            return policy_from_config(get_field(entry, "tools")?);
        }
    }
    None
}

/// Resolve the channel default + specific policies from a config-shaped value.
///
/// Returns `(default_policy, specific_policy)` keyed by the context's channel
/// kind and channel id.
pub fn channel_entry_policy_from_config(
    config: &Value,
    channel_kind: Option<&str>,
    channel_id: Option<&str>,
) -> (Option<ToolPolicy>, Option<ToolPolicy>) {
    let Some(channel_kind) = channel_kind else {
        return (None, None);
    };
    let Some(channels) = get_field(config, "channels") else {
        return (None, None);
    };
    let Some(channel_cfg) = channels.get(channel_kind) else {
        return (None, None);
    };
    let mut entries: Option<&Value> = None;
    for field in ["groups", "channels", "rooms"] {
        if let Some(value) = get_field(channel_cfg, field) {
            if value.is_object() {
                entries = Some(value);
                break;
            }
        }
    }
    let Some(entries) = entries else {
        return (None, None);
    };
    let default_policy = entries.get("*").and_then(policy_from_config);
    let specific_policy = entries
        .get(channel_id.unwrap_or(""))
        .and_then(policy_from_config);
    (default_policy, specific_policy)
}

/// Apply a channel policy layer to allow/deny sets.
pub fn apply_channel_layer(
    allowed_tools: Option<BTreeSet<String>>,
    channel_denied: &BTreeSet<String>,
    policy: &ToolPolicy,
    available_tools: &BTreeSet<String>,
) -> Result<(Option<BTreeSet<String>>, BTreeSet<String>), PolicyConfigError> {
    let mut allowed_tools = allowed_tools;
    let mut channel_denied = channel_denied.clone();
    if let Some(profile) = &policy.profile {
        if let Some(profile_allowed) = profile_allowlist(profile, available_tools)? {
            allowed_tools = Some(profile_allowed);
        }
    }
    let mut channel_selectors: BTreeSet<String> = policy.allow.clone();
    channel_selectors.extend(policy.also_allow.iter().cloned());
    for group in SENDER_SCOPED_TOOL_GROUPS {
        channel_selectors.remove(*group);
    }
    allowed_tools = add_allowed(
        allowed_tools,
        &expand_selectors(&channel_selectors, available_tools),
    );
    channel_denied.extend(expand_selectors(&policy.deny, available_tools));
    Ok((allowed_tools, channel_denied))
}

/// Apply a sender policy layer to allow/deny sets.
pub fn apply_sender_layer(
    allowed_tools: Option<BTreeSet<String>>,
    channel_denied: &BTreeSet<String>,
    policy: &ToolPolicy,
    available_tools: &BTreeSet<String>,
) -> (Option<BTreeSet<String>>, BTreeSet<String>) {
    let mut allowed_tools = allowed_tools;
    let mut channel_denied = channel_denied.clone();
    let also_allowed = expand_selectors(&policy.also_allow, available_tools);
    for tool in &also_allowed {
        channel_denied.remove(tool);
    }
    allowed_tools = add_allowed(
        allowed_tools,
        &expand_selectors(&policy.allow, available_tools),
    );
    allowed_tools = add_allowed(allowed_tools, &also_allowed);
    channel_denied.extend(expand_selectors(&policy.deny, available_tools));
    (allowed_tools, channel_denied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn set(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    fn available() -> BTreeSet<String> {
        set(&[
            "read_file",
            "write_file",
            "edit_file",
            "exec_command",
            "background_process",
            "memory_search",
            "web_search",
            "git_commit",
            "apply_patch",
        ])
    }

    #[test]
    fn expand_exact_and_group_selectors() {
        let expanded = expand_selectors(&set(&["read_file", "group:fs"]), &available());
        assert!(expanded.contains("read_file"));
        assert!(expanded.contains("write_file"));
        assert!(expanded.contains("edit_file"));
        // apply_patch is in group:fs and in the available set.
        assert!(expanded.contains("apply_patch"));
        assert!(!expanded.contains("exec_command"));
    }

    #[test]
    fn expand_star_and_patterns() {
        let expanded = expand_selectors(&set(&["*"]), &available());
        assert_eq!(expanded, available());

        let expanded = expand_selectors(&set(&["git_*"]), &available());
        assert!(expanded.contains("git_commit"));
        assert_eq!(expanded.len(), 1);

        let expanded = expand_selectors(&set(&["mem*_*"]), &available());
        assert!(expanded.contains("memory_search"));
    }

    #[test]
    fn fnmatch_semantics() {
        assert!(fnmatchcase("git_*", "git_commit"));
        assert!(fnmatchcase("git_?", "git_c"));
        assert!(!fnmatchcase("git_?", "git_commit"));
        assert!(fnmatchcase("*", "anything"));
        assert!(fnmatchcase("create_*", "create_pdf_report"));
        assert!(fnmatchcase("mem[eo]*", "memory_search"));
        assert!(!fnmatchcase("mem[a-d]*", "nope"));
    }

    #[test]
    fn profile_allowlist_resolves_groups() {
        let tools = available();
        let allowed = profile_allowlist("memory_only", &tools).unwrap();
        assert_eq!(allowed.unwrap(), set(&["memory_search"]));

        // Full profile means no restriction.
        assert!(profile_allowlist("full", &tools).unwrap().is_none());

        // Unknown profile is an error.
        assert!(profile_allowlist("bogus", &tools).is_err());
    }

    #[test]
    fn coding_profile_merges_groups() {
        let tools = available();
        let allowed = profile_allowlist("coding", &tools).unwrap().unwrap();
        assert!(allowed.contains("read_file"));
        assert!(allowed.contains("exec_command"));
        assert!(allowed.contains("memory_search"));
        assert!(!allowed.contains("web_search"));
    }

    #[test]
    fn apply_base_policy_with_deny() {
        let tools = available();
        let policy = ToolPolicy {
            profile: Some("coding".to_string()),
            deny: set(&["write_file"]),
            ..Default::default()
        };
        let (allowed, denied) =
            apply_base_policy(None, &BTreeSet::new(), &policy, &tools, true).unwrap();
        let allowed = allowed.unwrap();
        assert!(allowed.contains("read_file"));
        assert!(!allowed.contains("write_file"));
        assert!(denied.contains("write_file"));
    }

    #[test]
    fn profile_overrides_existing_allowlist() {
        let tools = available();
        let policy = ToolPolicy {
            profile: Some("memory_only".to_string()),
            ..Default::default()
        };
        let existing = Some(set(&["read_file", "exec_command"]));
        let (allowed, _) =
            apply_base_policy(existing, &BTreeSet::new(), &policy, &tools, true).unwrap();
        // The profile replaces the base allowlist.
        assert_eq!(allowed.unwrap(), set(&["memory_search"]));
    }

    #[test]
    fn sender_matching() {
        assert!(matches_sender("*", Some("alice")));
        assert!(matches_sender("alice", Some("alice")));
        assert!(!matches_sender("alice", Some("bob")));
        assert!(!matches_sender("alice", None));
        assert!(matches_sender("id:123", Some("123")));
        assert!(!matches_sender("id:123", Some("456")));
        assert!(!matches_sender("role:admin", Some("admin")));
    }

    #[test]
    fn policy_from_config_plain() {
        let config = json!({
            "profile": "memory_only",
            "allow": ["memory_search", "web_search"],
            "deny": ["memory_get"],
            "fileEditRequiresFreshRead": true,
        });
        let policy = policy_from_config(&config).expect("policy");
        assert_eq!(policy.profile.as_deref(), Some("memory_only"));
        assert!(policy.allow.contains("memory_search"));
        assert!(policy.deny.contains("memory_get"));
        assert_eq!(policy.file_edit_requires_fresh_read, Some(true));
    }

    #[test]
    fn policy_from_config_wrapper() {
        let config = json!({
            "tools": {
                "allow": ["read_file"],
                "workspaceWriteDenyGlobs": ["**/*.lock"],
            },
            "toolsBySender": {
                "id:42": {"allow": ["web_search"]}
            }
        });
        let policy = policy_from_config(&config).expect("policy");
        assert!(policy.allow.contains("read_file"));
        assert!(policy.workspace_write_deny_globs.contains("**/*.lock"));
        assert!(policy.by_sender.contains_key("id:42"));
    }

    #[test]
    fn agent_policy_from_config_map_and_list() {
        let config = json!({
            "agents": {
                "main": {"tools": {"profile": "memory_only"}}
            }
        });
        let policy = agent_policy_from_config(&config, "main").expect("policy");
        assert_eq!(policy.profile.as_deref(), Some("memory_only"));

        let config = json!({
            "agents": [
                {"id": "worker-1", "tools": {"allow": ["exec_command"]}}
            ]
        });
        let policy = agent_policy_from_config(&config, "worker-1").expect("policy");
        assert!(policy.allow.contains("exec_command"));
    }

    #[test]
    fn channel_entry_policy_from_config_resolution() {
        let config = json!({
            "channels": {
                "telegram": {
                    "groups": {
                        "*": {"allow": ["web_search"]},
                        "room-1": {"deny": ["web_search"]}
                    }
                }
            }
        });
        let (default_policy, specific_policy) =
            channel_entry_policy_from_config(&config, Some("telegram"), Some("room-1"));
        assert!(default_policy.unwrap().allow.contains("web_search"));
        assert!(specific_policy.unwrap().deny.contains("web_search"));
    }

    #[test]
    fn channel_and_sender_layers() {
        let tools = available();
        let channel_policy = ToolPolicy {
            allow: set(&["group:web"]),
            deny: set(&["exec_command"]),
            ..Default::default()
        };
        let (allowed, denied) = apply_channel_layer(
            Some(set(&["web_search"])),
            &BTreeSet::new(),
            &channel_policy,
            &tools,
        )
        .unwrap();
        assert!(allowed.unwrap().contains("web_search"));
        assert!(denied.contains("exec_command"));

        let sender_policy = ToolPolicy {
            also_allow: set(&["exec_command"]),
            ..Default::default()
        };
        let (allowed, denied) = apply_sender_layer(
            Some(set(&["web_search"])),
            &set(&["exec_command"]),
            &sender_policy,
            &tools,
        );
        assert!(allowed.unwrap().contains("exec_command"));
        // also_allow removes from channel_denied.
        assert!(!denied.contains("exec_command"));
    }

    #[test]
    fn test_coding_mode_denied_tools() {
        assert!(coding_mode_denied_tools(true).contains("write_file"));
        assert!(coding_mode_denied_tools(false).is_empty());
    }
}
