//! # OpenSquilla Tools
//!
//! Built-in tool system for the OpenSquilla agent runtime.
//! Provides 40+ built-in tools for shell execution, file system operations,
//! web access, git, media processing, code execution, patching, diffing,
//! archiving, process monitoring, memory, sessions, messaging, cron
//! scheduling, and more.
//!
//! ## Architecture
//!
//! - `registry` — Tool trait, tool registration, parameter schema
//! - `dispatch` — Central dispatch: injection guard, policy chain, budget, sandbox, rate limiting, post-processing
//! - `policy` — Policy chain: DenyPolicy, BudgetPolicy, RiskPolicy, ToolPolicy, ConfirmationPolicy, FinalizePolicy
//! - `shell` — Shell command execution (exec_command, background_process, signal handling, env filtering)
//! - `filesystem` — File read/write/edit with path traversal protection, tree listing, copy/move/delete
//! - `web` — Web search, web_fetch, http_request, content extraction with readability scoring
//! - `ssrf` — Standalone SSRF protection (DNS + IP range checks, fail-closed)
//! - `schema_validation` — JSON Schema parameter validation
//! - `git` — Git operations via subprocess (clone, branch, merge, rebase, stash, tag, blame, remote)
//! - `media` — Image, PDF, TTS, transcription processing
//! - `code_exec` — Code execution in subprocess with language detection
//! - `patch` — Unified diff parsing and application, 3-way merge, patch reversal
//! - `diff` — File and directory diffing with multiple output formats
//! - `archive` — Zip and tar.gz archive creation/extraction
//! - `process_monitor` — System process monitoring (ps equivalent)
//! - `artifacts` — Artifact generation (docx, xlsx, pdf)
//! - `file_authoring` — File generation tools (generate_pdf, generate_xlsx,
//!   generate_csv, generate_json, generate_markdown, generate_html, read_xlsx, read_csv)
//! - `memory_tools` — Memory save/search/delete/list via opensquilla-memory
//! - `session_tools` — Session create/list/get/switch/export/delete via opensquilla-session
//! - `messaging` — Channel messaging via opensquilla-channels
//! - `cron_tool` — Schedule/list/cancel cron jobs via opensquilla-scheduler
//! - `skill_tools` — Skill list/view/search/install/create/edit/delete via opensquilla-skills
//! - `plan_control` — Plan submission, user-input requests, plan-run checkpoints via opensquilla-session
//! - `router_control` — Runtime router tier hold/clear via an in-tool hold store

pub mod archive;
pub mod argument_normalization;
pub mod artifacts;
pub mod candidate_patch_checkpoint;
pub mod code_exec;
pub mod context;
pub mod cron_tool;
pub mod diff;
pub mod dispatch;
pub mod envelope;
pub mod file_authoring;
pub mod filesystem;
pub mod git;
pub mod media;
pub mod memory_tools;
pub mod messaging;
pub mod patch;
pub mod path_policy;
pub mod patch_classification;
pub mod plan_control;
pub mod policy;
pub mod policy_checks;
pub mod policy_config;
pub mod policy_runtime;
pub mod process_monitor;
pub mod projected_arguments;
pub mod registry;
pub mod router_control;
pub mod run_mode;
pub mod schema_validation;
pub mod session_rpc_tools;
pub mod session_tools;
pub mod shell;
pub mod skill_tools;
pub mod source_diff_candidates;
pub mod source_edit_contract;
pub mod ssrf;
pub mod visibility;
pub mod web;
pub mod write_policy;
pub mod write_tracking;

// Re-export primary types at the crate root.
pub use archive::{ArchiveFormat, ArchiveTool, ZipEntry, ZipReader, ZipWriter};
pub use argument_normalization::{
    AppliedAlias, ToolArgumentAliasConflict, ToolArgumentNormalizationResult,
    canonicalize_tool_arguments, format_alias_conflicts,
};
pub use candidate_patch_checkpoint::{
    CandidatePatchCheckpoint, CandidatePatchFileSnapshot, create_candidate_patch_checkpoint,
    restore_candidate_patch_checkpoint,
};
pub use context::{
    CallerKind, DescriptionOverrides, InteractionMode, PlanAccess, RunMode, ToolContext,
    TOOL_DESCRIPTION_OVERRIDES_ENV, CRON_AGENT_ALLOW, CRON_AGENT_DENY, SUBAGENT_TOOL_DENY,
    current_tool_context, is_tool_context_active, mutate_current_tool_context,
    resolve_tool_description_overrides, run_with_tool_context, with_current_tool_context,
};
pub use cron_tool::{CancelTaskTool, ListTasksTool, ScheduleTaskTool, build_scheduler_engine};
pub use diff::{DiffTool, DirDiffTool};
pub use dispatch::{
    DispatchContext, DispatchEngine, DispatchError, DispatchRateLimiter, InjectionGuard,
    NoopSandbox, SandboxHandle, SandboxResult, redact_sensitive,
};
pub use envelope::{
    EnvelopeOptions, build_denial_envelope, build_tool_failure_envelope, is_denial_payload,
    is_retriable,
};
pub use file_authoring::{
    GenerateCsvTool, GenerateHtmlTool, GenerateJsonTool, GenerateMarkdownTool, GeneratePdfTool,
    GenerateXlsxTool, ReadCsvTool, ReadXlsxTool,
};
pub use git::GitTool;
pub use media::{ImageTool, MediaTool, PdfTool, TranscriptionTool, TtsTool};
pub use memory_tools::{MemoryDeleteTool, MemoryListTool, MemorySaveTool, MemorySearchTool};
pub use messaging::SendMessageTool;
pub use patch::{ApplyPatchTool, ResolveConflictsTool, ReversePatchTool, ThreeWayMergeTool};
pub use path_policy::{foreign_host_path_error, is_foreign_host_path, reject_foreign_host_path};
pub use patch_classification::{
    is_instrumentation_line, is_instrumentation_only_patch, iter_patch_line_changes,
};
pub use plan_control::{PlanRunCheckpointTool, RequestUserInputTool, SubmitPlanTool};
pub use policy::{
    BudgetPolicy, ConfirmationPolicy, DenyPolicy, FinalizePolicy, PolicyChain, PolicyChainSet,
    PolicyContext, PolicyDecision, RiskLevel, RiskPolicy, ToolPolicy, ToolRule,
};
pub use policy_checks::{
    AllowListPolicy, DenyListPolicy, DispatchPolicyInput, OwnerOnlyPolicy, PermissionMatrixPolicy,
    PolicyCheck, PolicyCheckDecision, PrivateMemoryScopePolicy, ProfilePolicy,
    run_policy_chain,
};
pub use policy_config::{
    DeclarativeToolPolicy, PolicyConfigError, apply_base_policy, apply_channel_layer,
    apply_sender_layer, coding_mode_denied_tools, expand_selectors, profile_allowlist,
    remove_denied_from_allowed, sender_policy,
};
pub use policy_runtime::{
    ToolSurfaceCapabilities, private_memory_read_tool_denied, private_memory_read_tools_blocked,
    resolve_runtime_tool_surface,
};
pub use process_monitor::ProcessMonitorTool;
pub use projected_arguments::{
    ProjectedToolArgumentMatch, find_projected_tool_argument, is_provider_context_marker_value,
};
pub use registry::{
    Tool, ToolDefinition, ToolError, ToolInput, ToolOutput, ToolRegistry, ToolResult,
};
pub use router_control::{RoutingHold, RoutingHoldStore, RouterControlTool, normalize_text_tier};
pub use run_mode::{full_host_access_for_context, sandbox_disabled_full_host_fallback};
pub use schema_validation::{SchemaValidator, validate_tool_args};
pub use session_rpc_tools::{
    SessionsHistoryTool, SessionsSendTool, SessionsSpawnTool, SessionsYieldTool,
};
pub use session_tools::{
    SessionCreateTool, SessionDeleteTool, SessionExportTool, SessionGetTool, SessionListTool,
    SessionSwitchTool,
};
pub use shell::{
    BackgroundProcessTool, EnhancedExecTool, EnvFilter, ExecCommandTool, OutputCapture,
    ProcessRegistry, ProcessSupervisor, Signal, SignalProcessTool, StreamOutputTool,
};
pub use skill_tools::{
    InstallSkillDepsTool, SkillCreateTool, SkillDeleteTool, SkillEditTool,
    SkillInstallCommunityTool, SkillListTool, SkillSearchCommunityTool, SkillViewTool,
};
pub use source_diff_candidates::{
    MAX_CANDIDATES, MAX_PATCH_CHARS, capture_source_diff_candidate,
    latest_recoverable_source_candidate, mark_source_diff_candidates_lost,
    recoverable_lost_source_candidate_ids,
};
pub use source_edit_contract::{
    DEFAULT_SOURCE_READ_LINES, SourceEditContractError, apply_line_edits, build_diff_summary,
    build_line_receipt, source_revision_for_path,
};
pub use ssrf::{SsrfConfig, SsrfConfigBuilder, SsrfProtection};
pub use visibility::{
    CHANNEL_DEFAULT_ALLOW, CHANNEL_HARD_DENY_NON_OWNER, ToolProfile, ToolVisibilitySpec,
    default_tool_context, effective_tool_context, filter_by_profile, is_tool_visible,
    profile_allows_tool, resolve_profile, tool_context_for_profile,
};
pub use web::{
    CachedResponse, DomainRateLimiter, DuckDuckGoSearch, HttpRequestTool, ReadabilityResult,
    ResponseCache, RobotsTxt, SearchProvider, SearchResult, WebExtractTool, WebFetchTool,
    WebSearchTool, html_to_text, readability_score,
};
pub use write_policy::{
    WorkspaceScratchArtifactMatch, WorkspaceWriteDenyMatch, gate_workspace_scratch_artifact,
    gate_workspace_write_deny, match_workspace_scratch_artifact, match_workspace_write_deny,
    validate_workspace_write_deny_env, verify_mirror_path, workspace_scratch_artifact_block,
    workspace_write_deny_block,
};
pub use write_tracking::{
    FileFingerprint, classify_workspace_path, diff_workspace_mutations, fingerprint_file,
    mutation_ledger_text_hash, normalize_relative_path, parse_git_status_z,
    record_scratch_file_write, record_workspace_file_read, record_workspace_file_write,
    require_fresh_workspace_file_read, scratch_only_progress_note, snapshot_workspace_mutations,
    summarize_patch_hygiene_warning, summarize_workspace_write_notes, workspace_write_note,
    workspace_write_progress_note,
};
