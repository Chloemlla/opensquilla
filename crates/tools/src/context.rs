//! Request-scoped tool context and caller metadata.
//!
//! Mirrors the Python `opensquilla.tools.types` module: a `ToolContext`
//! constructed at the entry point (gateway, CLI, cron, channel) that flows
//! through to tool-list building and dispatch. Carries the caller kind,
//! interaction mode, workspace/scratch roots, allowed/denied tool sets, and
//! the request-scoped ledger of file reads/writes and mutation receipts.
//!
//! The type is intentionally additive with respect to the rest of the crate:
//! existing dispatch/registry types are unchanged.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Entry-point caller type — used in `ToolContext` for filtering decisions.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum CallerKind {
    #[default]
    Agent,
    Subagent,
    Cron,
    Channel,
    Cli,
    Web,
}

impl CallerKind {
    /// Parse a caller kind from a string (case-insensitive), or `None`.
    pub fn from_str_opt(value: &str) -> Option<CallerKind> {
        match value.trim().to_ascii_lowercase().as_str() {
            "agent" => Some(CallerKind::Agent),
            "subagent" => Some(CallerKind::Subagent),
            "cron" => Some(CallerKind::Cron),
            "channel" => Some(CallerKind::Channel),
            "cli" => Some(CallerKind::Cli),
            "web" => Some(CallerKind::Web),
            _ => None,
        }
    }
}

/// Whether the entry point has a live operator available for tool approvals.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum InteractionMode {
    #[default]
    Interactive,
    Unattended,
}

impl InteractionMode {
    /// Parse an interaction mode from a string, or `None`.
    pub fn from_str_opt(value: &str) -> Option<InteractionMode> {
        match value.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(InteractionMode::Interactive),
            "unattended" => Some(InteractionMode::Unattended),
            _ => None,
        }
    }
}

/// Whether a tool may be exposed or dispatched while planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanAccess {
    Deny,
    ReadOnly,
    Control,
}

impl PlanAccess {
    /// Parse a plan-access value; unknown values fall back to `Deny`.
    pub fn from_str_opt(value: &str) -> PlanAccess {
        match value.trim().to_ascii_lowercase().as_str() {
            "read_only" => PlanAccess::ReadOnly,
            "control" => PlanAccess::Control,
            _ => PlanAccess::Deny,
        }
    }
}

/// Request-scoped sandbox run mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    Standard,
    Trusted,
    Full,
}

impl RunMode {
    /// Parse a run mode from a string, or `None`.
    pub fn from_str_opt(value: &str) -> Option<RunMode> {
        match value.trim().to_ascii_lowercase().as_str() {
            "standard" => Some(RunMode::Standard),
            "trusted" => Some(RunMode::Trusted),
            "full" => Some(RunMode::Full),
            _ => None,
        }
    }
}

/// Tool deny-list constants — exact registered tool names that subagents
/// may not invoke.
pub const SUBAGENT_TOOL_DENY: &[&str] = &[
    "cron",
    "gateway",
    "agents_list",
    "subagents",
    "memory_get",
    "memory_search",
    "session_search",
    "message",
    "publish_artifact",
];

/// Tools a non-owner cron agent is allowed to invoke.
pub const CRON_AGENT_ALLOW: &[&str] = &[
    "git_diff",
    "git_log",
    "git_status",
    "glob_search",
    "grep_search",
    "list_dir",
    "pdf",
    "read_file",
    "session_status",
    "sessions_history",
    "sessions_list",
    "web_discover",
    "web_fetch",
    "web_search",
];

/// Tools that are never exposed to a non-owner cron agent.
pub const CRON_AGENT_DENY: &[&str] = &[
    "cron",
    "agents_list",
    "subagents",
    "message",
    "exec_command",
    "background_process",
    "write_file",
    "edit_file",
    "apply_patch",
    "execute_code",
    "git_commit",
];

/// Request-scoped context that flows through to tool-list building and
/// dispatch.
///
/// Mirrors the Python `ToolContext` dataclass for the fields the Rust tool
/// infrastructure needs. Ledger-style fields (file reads/writes, mutation
/// records/receipts, source-diff candidates) store JSON values so downstream
/// consumers get the same shape the Python side produces.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    /// Whether the caller owns the workspace/operator surface.
    pub is_owner: bool,
    /// The entry-point caller type.
    pub caller_kind: CallerKind,
    /// Whether a live approval surface is available.
    pub interaction_mode: InteractionMode,
    /// Depth of subagent nesting (0 = top-level agent).
    pub subagent_depth: u32,
    /// The agent id the call runs under.
    pub agent_id: String,
    /// Active workspace directory (filesystem tools are rooted here).
    pub workspace_dir: Option<PathBuf>,
    /// Memory source directory.
    pub memory_source_dir: Option<PathBuf>,
    /// Scratch directory for temporary artifacts.
    pub scratch_dir: Option<PathBuf>,
    /// Workspace write-deny glob patterns (workspace-relative).
    pub workspace_write_deny_globs: Vec<String>,
    /// Request-scoped sandbox run mode (`standard` / `trusted` / `full`).
    pub run_mode: Option<RunMode>,
    /// Legacy elevated-mode compatibility ("full" / "on" / "bypass").
    pub elevated: Option<String>,
    /// Session key for the call.
    pub session_key: Option<String>,
    /// Channel kind for channel calls.
    pub channel_kind: Option<String>,
    /// Channel id for channel calls.
    pub channel_id: Option<String>,
    /// Sender id for channel calls.
    pub sender_id: Option<String>,
    /// Source kind (e.g. "webui").
    pub source_kind: Option<String>,
    /// Explicit allowlist; `None` means "no allowlist restriction".
    pub allowed_tools: Option<BTreeSet<String>>,
    /// Explicit denylist.
    pub denied_tools: BTreeSet<String>,
    /// Additive per-call surfaced tools (made visible even when not exposed
    /// by default). Does NOT relax the strict denylist.
    pub surfaced_tools: Option<BTreeSet<String>>,
    /// Authenticated channel admin verified at the ingress boundary.
    pub channel_admin_verified: bool,
    /// Collaboration mode frozen for this turn (e.g. "plan", "default").
    pub collaboration_mode: String,
    /// When set, edits require a prior fresh full-file read.
    pub file_edit_requires_fresh_read: bool,
    /// When set, edit recovery is flexible.
    pub file_edit_flexible_recovery: bool,
    /// Source-diff candidate capture mode: off / log / warn_model.
    pub source_diff_candidate_mode: String,
    /// Ledger of captured source-diff candidates.
    pub source_diff_candidates: Vec<serde_json::Value>,
    /// Monotonic counter used to mint candidate ids.
    pub source_diff_candidate_counter: u32,
    /// Workspace epoch — increments on each changed semantic mutation.
    pub workspace_epoch: u32,
    /// Ledger of workspace file reads.
    pub workspace_file_reads: Vec<serde_json::Value>,
    /// Read-state index keyed by resolved path.
    pub workspace_file_read_state: std::collections::HashMap<String, serde_json::Value>,
    /// Ledger of workspace file writes.
    pub workspace_file_writes: Vec<serde_json::Value>,
    /// Ledger of observed/effect-enforcement mutation records.
    pub workspace_mutation_records: Vec<serde_json::Value>,
    /// Ledger of semantic mutation receipts.
    pub workspace_mutation_receipts: Vec<serde_json::Value>,
    /// Ledger of scratch file writes.
    pub scratch_file_writes: Vec<serde_json::Value>,
    /// Whether the scratch verify-mirror lever is active.
    pub scratch_verify_mirror_active: bool,
    /// Tool description overrides (tool or "tool.param" key -> verbatim text).
    pub tool_description_overrides: Option<std::collections::HashMap<String, String>>,
    /// Where the description overrides came from ("config" | "env_file").
    pub tool_description_overrides_source: Option<String>,
    /// Whether the endgame git freeze is active.
    pub endgame_git_freeze_active: bool,
    /// Whether the endgame git freeze instrumentation exemption applies.
    pub endgame_git_freeze_instrumentation_exempt: bool,
}

impl ToolContext {
    /// Create a default owner context for the agent caller.
    pub fn owner() -> Self {
        Self {
            is_owner: true,
            caller_kind: CallerKind::Agent,
            interaction_mode: InteractionMode::Interactive,
            agent_id: "main".to_string(),
            ..Default::default()
        }
    }

    /// Return whether the turn's frozen collaboration mode is Plan.
    pub fn is_plan_mode(&self) -> bool {
        self.collaboration_mode.trim().eq_ignore_ascii_case("plan")
    }

    /// Return the resolved run mode for this context, if any.
    pub fn current_run_mode(&self) -> Option<RunMode> {
        if let Some(mode) = self.run_mode {
            return Some(mode);
        }
        match self.elevated.as_deref() {
            Some("full") => Some(RunMode::Full),
            Some("on") | Some("bypass") => Some(RunMode::Trusted),
            _ => None,
        }
    }

    /// Whether Full Host Access semantics are active for this context.
    pub fn full_host_access_active(&self) -> bool {
        self.current_run_mode() == Some(RunMode::Full)
    }

    /// Whether Managed Execution (trusted, sandboxed) is active.
    pub fn trusted_sandbox_active(&self) -> bool {
        !self.full_host_access_active() && self.current_run_mode() == Some(RunMode::Trusted)
    }
}

// ---------------------------------------------------------------------------
// Request-scoped current-context access (mirrors `current_tool_context`).
// ---------------------------------------------------------------------------
//
// The `Tool` trait is intentionally context-free, so the dispatch engine
// scopes the effective `ToolContext` into a task-local slot during execution.
// Tool implementations that need write-tracking, write-policy gating, or
// run-mode classification consult the slot. When nothing is scoped (direct
// tool invocation in tests, or a dispatch without a context), every accessor
// returns `None`/no-op so the pure-tool behaviour is unchanged.

use std::cell::RefCell;
use std::future::Future;

tokio::task_local! {
    static CURRENT_TOOL_CONTEXT: RefCell<Option<ToolContext>>;
}

/// Run `f` with `ctx` scoped into the current-task tool-context slot.
///
/// `None` scopes an empty slot so tools observe "no context" and take their
/// default (pure) paths. Always resets the slot when the future completes.
pub async fn run_with_tool_context<T>(ctx: Option<ToolContext>, f: impl Future<Output = T>) -> T {
    CURRENT_TOOL_CONTEXT.scope(RefCell::new(ctx), f).await
}

/// Clone the scoped tool context, or `None` when nothing is scoped.
pub fn current_tool_context() -> Option<ToolContext> {
    CURRENT_TOOL_CONTEXT
        .try_with(|slot| slot.borrow().clone())
        .ok()
        .flatten()
}

/// Borrow the scoped context for a read-only closure, if any.
pub fn with_current_tool_context<R>(f: impl FnOnce(&ToolContext) -> R) -> Option<R> {
    CURRENT_TOOL_CONTEXT
        .try_with(|slot| slot.borrow().as_ref().map(f))
        .ok()
        .flatten()
}

/// Mutate the scoped context ledger (reads/writes/receipts) in place.
pub fn mutate_current_tool_context<R>(f: impl FnOnce(&mut ToolContext) -> R) -> Option<R> {
    CURRENT_TOOL_CONTEXT
        .try_with(|slot| slot.borrow_mut().as_mut().map(f))
        .ok()
        .flatten()
}

/// Whether a tool context is currently scoped for this task.
pub fn is_tool_context_active() -> bool {
    CURRENT_TOOL_CONTEXT
        .try_with(|slot| slot.borrow().is_some())
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tool description overrides (mirrors `opensquilla.tools.description_overrides`).
// ---------------------------------------------------------------------------

/// Env gate that selects the tool-description override source.
pub const TOOL_DESCRIPTION_OVERRIDES_ENV: &str = "OPENSQUILLA_TOOL_DESCRIPTION_OVERRIDES";

const DESCRIPTION_OVERRIDES_OFF: &[&str] = &["off", "0", "false", "no"];
const DESCRIPTION_OVERRIDES_CONFIG: &[&str] = &["config", "on"];

/// Resolved tool-description override table and its provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptionOverrides {
    /// Tool or `tool.param` key -> verbatim replacement description.
    pub overrides: std::collections::BTreeMap<String, String>,
    /// Where the overrides came from: `config` or `env_file`.
    pub source: String,
}

/// Resolve the tool-description override table for the current turn.
///
/// `config` is the gateway config object (or `None`); the table lives at
/// `config.tools.description_overrides`. The mechanism only activates when
/// `OPENSQUILLA_TOOL_DESCRIPTION_OVERRIDES` names `config`/`on`, an `off`
/// value, or a `.toml`/`.json` override file path. Unrecognized env values and
/// unreadable or malformed override files raise so a run manifest cannot
/// record an override the run did not actually apply.
pub fn resolve_tool_description_overrides(
    config: Option<&serde_json::Value>,
) -> Result<Option<DescriptionOverrides>, String> {
    let env_value = std::env::var(TOOL_DESCRIPTION_OVERRIDES_ENV)
        .unwrap_or_default()
        .trim()
        .to_string();
    if env_value.is_empty()
        || DESCRIPTION_OVERRIDES_OFF.contains(&env_value.to_ascii_lowercase().as_str())
    {
        return Ok(None);
    }

    if DESCRIPTION_OVERRIDES_CONFIG.contains(&env_value.to_ascii_lowercase().as_str()) {
        let table = config
            .and_then(|c| c.get("tools"))
            .and_then(|tools| tools.get("description_overrides"));
        let overrides = normalize_overrides(table, &env_value)?;
        if overrides.is_empty() {
            return Ok(None);
        }
        return Ok(Some(DescriptionOverrides {
            overrides,
            source: "config".to_string(),
        }));
    }

    if env_value.to_ascii_lowercase().ends_with(".toml")
        || env_value.to_ascii_lowercase().ends_with(".json")
    {
        let data = load_override_file(&env_value)?;
        let overrides = normalize_overrides(Some(&data), &env_value)?;
        if overrides.is_empty() {
            return Ok(None);
        }
        return Ok(Some(DescriptionOverrides {
            overrides,
            source: "env_file".to_string(),
        }));
    }

    Err(format!(
        "{} must be one of: {}, or a .toml/.json override file path",
        TOOL_DESCRIPTION_OVERRIDES_ENV,
        DESCRIPTION_OVERRIDES_CONFIG
            .iter()
            .chain(DESCRIPTION_OVERRIDES_OFF.iter())
            .copied()
            .collect::<Vec<&str>>()
            .join(", ")
    ))
}

fn load_override_file(path_value: &str) -> Result<serde_json::Value, String> {
    let is_toml = path_value.to_ascii_lowercase().ends_with(".toml");
    let data = if is_toml {
        let text = std::fs::read_to_string(path_value).map_err(|e| {
            format!("{TOOL_DESCRIPTION_OVERRIDES_ENV} file {path_value:?} could not be loaded: {e}")
        })?;
        toml::from_str::<serde_json::Value>(&text).map_err(|e| {
            format!("{TOOL_DESCRIPTION_OVERRIDES_ENV} file {path_value:?} could not be loaded: {e}")
        })?
    } else {
        let text = std::fs::read_to_string(path_value).map_err(|e| {
            format!("{TOOL_DESCRIPTION_OVERRIDES_ENV} file {path_value:?} could not be loaded: {e}")
        })?;
        serde_json::from_str(&text).map_err(|e| {
            format!("{TOOL_DESCRIPTION_OVERRIDES_ENV} file {path_value:?} could not be loaded: {e}")
        })?
    };
    // Accept both a top-level override table and the gateway config shape
    // ([tools.description_overrides]) so an arm can point the env at its
    // config.toml copy directly.
    if let Some(tools) = data.get("tools").and_then(|t| t.as_object()) {
        if let Some(nested) = tools
            .get("description_overrides")
            .and_then(|v| v.as_object())
        {
            return Ok(serde_json::Value::Object(nested.clone()));
        }
    }
    Ok(data)
}

fn normalize_overrides(
    table: Option<&serde_json::Value>,
    source_label: &str,
) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut overrides = std::collections::BTreeMap::new();
    let Some(value) = table else {
        return Ok(overrides);
    };
    let Some(map) = value.as_object() else {
        return Err(format!(
            "{TOOL_DESCRIPTION_OVERRIDES_ENV} overrides from {source_label:?} must be a table of strings"
        ));
    };
    for (raw_key, raw_value) in map {
        let key = raw_key.clone();
        if let Some(nested) = raw_value.as_object() {
            // TOML parses an unquoted dotted key ("exec_command.command") as a
            // nested table; flatten one level back into dotted parameter keys.
            for (param_key, param_value) in nested {
                add_override(
                    &mut overrides,
                    &format!("{key}.{param_key}"),
                    param_value,
                    source_label,
                )?;
            }
            continue;
        }
        add_override(&mut overrides, &key, raw_value, source_label)?;
    }
    Ok(overrides)
}

fn add_override(
    overrides: &mut std::collections::BTreeMap<String, String>,
    key: &str,
    value: &serde_json::Value,
    source_label: &str,
) -> Result<(), String> {
    match value.as_str() {
        Some(text) if !text.trim().is_empty() => {
            overrides.insert(key.to_string(), text.to_string());
            Ok(())
        }
        _ => Err(format!(
            "{TOOL_DESCRIPTION_OVERRIDES_ENV} override {key:?} from {source_label:?} must be a non-empty string"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_kind_parsing() {
        assert_eq!(CallerKind::from_str_opt("agent"), Some(CallerKind::Agent));
        assert_eq!(
            CallerKind::from_str_opt("SUBAGENT"),
            Some(CallerKind::Subagent)
        );
        assert_eq!(
            CallerKind::from_str_opt("channel"),
            Some(CallerKind::Channel)
        );
        assert_eq!(CallerKind::from_str_opt("bogus"), None);
    }

    #[test]
    fn plan_access_falls_back_to_deny() {
        assert_eq!(PlanAccess::from_str_opt("control"), PlanAccess::Control);
        assert_eq!(PlanAccess::from_str_opt("read_only"), PlanAccess::ReadOnly);
        assert_eq!(PlanAccess::from_str_opt("anything_else"), PlanAccess::Deny);
    }

    #[test]
    fn run_mode_resolution() {
        let mut ctx = ToolContext::owner();
        assert_eq!(ctx.current_run_mode(), None);
        assert!(!ctx.full_host_access_active());

        ctx.run_mode = Some(RunMode::Full);
        assert!(ctx.full_host_access_active());
        assert!(!ctx.trusted_sandbox_active());

        ctx.run_mode = Some(RunMode::Trusted);
        assert!(!ctx.full_host_access_active());
        assert!(ctx.trusted_sandbox_active());

        // Elevated full implies full host access even without run_mode.
        let mut elevated = ToolContext::owner();
        elevated.elevated = Some("full".to_string());
        assert!(elevated.full_host_access_active());

        // Elevated "on"/"bypass" implies trusted.
        let mut trusted = ToolContext::owner();
        trusted.elevated = Some("on".to_string());
        assert!(trusted.trusted_sandbox_active());
    }

    #[test]
    fn plan_mode_detection() {
        let ctx = ToolContext::owner();
        assert!(!ctx.is_plan_mode());
        let mut plan = ToolContext::owner();
        plan.collaboration_mode = "plan".to_string();
        assert!(plan.is_plan_mode());
    }

    #[tokio::test]
    async fn scoped_context_is_visible_inside_and_cleared_after() {
        assert!(!is_tool_context_active());
        let observed = run_with_tool_context(Some(ToolContext::owner()), async {
            is_tool_context_active() && current_tool_context().is_some()
        })
        .await;
        assert!(observed);
        assert!(!is_tool_context_active());

        let scoped = run_with_tool_context(Some(ToolContext::owner()), async {
            mutate_current_tool_context(|ctx| {
                ctx.agent_id = "scoped".to_string();
            });
            current_tool_context().map(|c| c.agent_id)
        })
        .await;
        assert_eq!(scoped.as_deref(), Some("scoped"));
        assert!(!is_tool_context_active());
    }

    #[tokio::test]
    async fn scoped_none_context_is_noop() {
        let result = run_with_tool_context(None, async {
            (is_tool_context_active(), current_tool_context())
        })
        .await;
        assert_eq!(result.0, false);
        assert!(result.1.is_none());
    }

    #[test]
    fn description_overrides_off_env_returns_none() {
        // "off" and empty values both keep the mechanism off.
        with_overrides_env("off", || {
            let result = resolve_tool_description_overrides(None).expect("off result");
            assert!(result.is_none());
        });
    }

    /// Set the overrides env gate for the duration of `f` (serialized so
    /// parallel tests do not race on process env).
    fn with_overrides_env(value: &str, f: impl FnOnce()) {
        static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var(TOOL_DESCRIPTION_OVERRIDES_ENV, value);
        }
        f();
        unsafe {
            std::env::remove_var(TOOL_DESCRIPTION_OVERRIDES_ENV);
        }
    }

    #[test]
    fn description_overrides_config_source() {
        let config = serde_json::json!({
            "tools": {
                "description_overrides": {
                    "exec_command": "Run a shell command.",
                    "edit_file": {"old_text": "Text to replace."}
                }
            }
        });
        with_overrides_env("config", || {
            let result = resolve_tool_description_overrides(Some(&config)).expect("resolve");
            let overrides = result.expect("overrides");
            assert_eq!(overrides.source, "config");
            assert_eq!(
                overrides.overrides.get("exec_command").map(|s| s.as_str()),
                Some("Run a shell command.")
            );
            // TOML-style nested dotted key is flattened.
            assert_eq!(
                overrides
                    .overrides
                    .get("edit_file.old_text")
                    .map(|s| s.as_str()),
                Some("Text to replace.")
            );
        });
    }

    #[test]
    fn description_overrides_bad_env_value_raises() {
        with_overrides_env("bogus", || {
            assert!(resolve_tool_description_overrides(None).is_err());
        });
    }

    #[test]
    fn description_overrides_json_file_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("overrides.json");
        std::fs::write(&file, r#"{"exec_command": "Run a shell command."}"#)
            .expect("write overrides file");
        with_overrides_env(&file.to_string_lossy(), || {
            let result = resolve_tool_description_overrides(None).expect("resolve");
            let overrides = result.expect("overrides");
            assert_eq!(overrides.source, "env_file");
            assert_eq!(
                overrides.overrides.get("exec_command").map(|s| s.as_str()),
                Some("Run a shell command.")
            );
        });
    }
}
