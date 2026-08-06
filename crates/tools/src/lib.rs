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

pub mod archive;
pub mod artifacts;
pub mod code_exec;
pub mod cron_tool;
pub mod diff;
pub mod dispatch;
pub mod file_authoring;
pub mod filesystem;
pub mod git;
pub mod media;
pub mod memory_tools;
pub mod messaging;
pub mod patch;
pub mod policy;
pub mod process_monitor;
pub mod registry;
pub mod schema_validation;
pub mod session_tools;
pub mod shell;
pub mod ssrf;
pub mod web;

// Re-export primary types at the crate root.
pub use archive::{ArchiveFormat, ArchiveTool, ZipEntry, ZipReader, ZipWriter};
pub use cron_tool::{CancelTaskTool, ListTasksTool, ScheduleTaskTool, build_scheduler_engine};
pub use diff::{DiffTool, DirDiffTool};
pub use dispatch::{
    DispatchContext, DispatchEngine, DispatchError, DispatchRateLimiter, InjectionGuard,
    NoopSandbox, SandboxHandle, SandboxResult, redact_sensitive,
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
pub use policy::{
    BudgetPolicy, ConfirmationPolicy, DenyPolicy, FinalizePolicy, PolicyChain, PolicyChainSet,
    PolicyContext, PolicyDecision, RiskLevel, RiskPolicy, ToolPolicy, ToolRule,
};
pub use process_monitor::ProcessMonitorTool;
pub use registry::{
    Tool, ToolDefinition, ToolError, ToolInput, ToolOutput, ToolRegistry, ToolResult,
};
pub use schema_validation::{SchemaValidator, validate_tool_args};
pub use session_tools::{
    SessionCreateTool, SessionDeleteTool, SessionExportTool, SessionGetTool, SessionListTool,
    SessionSwitchTool,
};
pub use shell::{
    BackgroundProcessTool, EnhancedExecTool, EnvFilter, ExecCommandTool, OutputCapture,
    ProcessRegistry, ProcessSupervisor, Signal, SignalProcessTool, StreamOutputTool,
};
pub use ssrf::{SsrfConfig, SsrfConfigBuilder, SsrfProtection};
pub use web::{
    CachedResponse, DomainRateLimiter, DuckDuckGoSearch, HttpRequestTool, ReadabilityResult,
    ResponseCache, RobotsTxt, SearchProvider, SearchResult, WebExtractTool, WebFetchTool,
    WebSearchTool, html_to_text, readability_score,
};
