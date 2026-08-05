//! # OpenSquilla Tools
//!
//! Built-in tool system for the OpenSquilla agent runtime.
//! Provides 22+ built-in tools for shell execution, file system operations,
//! web access, git, media processing, code execution, patching, memory,
//! sessions, messaging, cron scheduling, and more.
//!
//! ## Architecture
//!
//! - `registry` — Tool trait, tool registration, parameter schema
//! - `dispatch` — Central dispatch: injection guard, policy chain, budget, sandbox
//! - `policy` — Policy chain: DenyPolicy, BudgetPolicy, FinalizePolicy
//! - `shell` — Shell command execution (exec_command, background_process)
//! - `filesystem` — File read/write/edit with path traversal protection
//! - `web` — Web search, web_fetch, http_request
//! - `ssrf` — Standalone SSRF protection (DNS + IP range checks, fail-closed)
//! - `schema_validation` — JSON Schema parameter validation
//! - `git` — Git operations via subprocess
//! - `media` — Image, PDF, TTS processing
//! - `code_exec` — Code execution in subprocess with language detection
//! - `patch` — Unified diff parsing and application
//! - `artifacts` — Artifact generation (docx, xlsx, pdf)
//! - `file_authoring` — File generation tools (generate_pdf, generate_xlsx,
//!   generate_csv, generate_json, generate_markdown)
//! - `memory_tools` — Memory save/search/delete/list via opensquilla-memory
//! - `session_tools` — Session create/list/get/switch/export/delete via opensquilla-session
//! - `messaging` — Channel messaging via opensquilla-channels
//! - `cron_tool` — Schedule/list/cancel cron jobs via opensquilla-scheduler

pub mod registry;
pub mod dispatch;
pub mod policy;
pub mod shell;
pub mod filesystem;
pub mod web;
pub mod ssrf;
pub mod schema_validation;
pub mod git;
pub mod media;
pub mod code_exec;
pub mod patch;
pub mod artifacts;
pub mod file_authoring;
pub mod memory_tools;
pub mod session_tools;
pub mod messaging;
pub mod cron_tool;

// Re-export primary types at the crate root.
pub use registry::{
    Tool, ToolDefinition, ToolError, ToolInput, ToolOutput, ToolRegistry, ToolResult,
};
pub use dispatch::{DispatchContext, DispatchEngine, InjectionGuard, SandboxHandle, SandboxResult};
pub use policy::{BudgetPolicy, DenyPolicy, FinalizePolicy, PolicyChain, PolicyDecision};
pub use ssrf::{SsrfConfig, SsrfConfigBuilder, SsrfProtection};
pub use schema_validation::{validate_tool_args, SchemaValidator};
pub use memory_tools::{MemoryDeleteTool, MemoryListTool, MemorySaveTool, MemorySearchTool};
pub use session_tools::{
    SessionCreateTool, SessionDeleteTool, SessionExportTool, SessionGetTool, SessionListTool,
    SessionSwitchTool,
};
pub use messaging::SendMessageTool;
pub use cron_tool::{CancelTaskTool, ListTasksTool, ScheduleTaskTool, build_scheduler_engine};
pub use file_authoring::{GenerateCsvTool, GeneratePdfTool, GenerateXlsxTool};