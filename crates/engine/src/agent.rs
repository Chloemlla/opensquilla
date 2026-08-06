//! Agent state machine and the `TurnGenerator` trait for producing responses.
//!
//! This module mirrors the Python backend's `engine/agent.py` — the explicit
//! state-machine agent with a tool-call loop. A single turn follows the
//! lifecycle:
//!
//! ```text
//! IDLE → THINKING → [WAITING_FOR_TOOL → THINKING → …] → COMPLETED
//!     any step may transition to ERROR.
//! ```
//!
//! The module provides:
//!
//! - [`AgentState`] — the coarse lifecycle state of an agent.
//! - [`AgentConfig`] — tunable limits for the agent loop.
//! - [`AgentError`] / [`RecoveryAction`] — error classification and recovery.
//! - [`TurnContext`] — a per-turn snapshot of the loop configuration.
//! - [`TurnOutcome`] — the result of a single turn.
//! - [`TurnGenerator`] — the trait any provider-backed generator must satisfy.
//! - [`Agent`] — the concrete state machine, including tool execution,
//!   subprocess management, git operations, and usage tracking.
//! - [`AgentRegistry`] — an in-memory collection of agents.

use async_trait::async_trait;
use opensquilla_core::error::{Error, Result};
use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall, ToolResult, Usage};
use std::fmt;
use std::path::PathBuf;
use std::process::Stdio;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, instrument, warn};

/// The current state of an agent in its lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AgentState {
    /// The agent is initializing and not yet ready to process turns.
    Initializing,
    /// The agent is idle and ready to accept a turn.
    Idle,
    /// The agent is currently processing a turn.
    Processing,
    /// The agent is actively reasoning / generating a response.
    Thinking,
    /// The agent is waiting for a tool call result.
    WaitingForTool,
    /// The agent is waiting for user input to continue.
    WaitingForUser,
    /// The agent has been paused and can be resumed.
    Paused,
    /// The turn has completed successfully.
    Completed,
    /// The agent has encountered an error and cannot continue.
    Error(String),
    /// The agent has been stopped and is no longer active.
    Stopped,
    /// The agent's turn was interrupted (crash, shutdown, network) and needs
    /// recovery before it can continue.
    Interrupted,
    /// The agent is compacting its conversation history.
    Compacting,
    /// The agent is awaiting a retry after a transient failure.
    Retrying { attempt: u32 },
}

impl fmt::Display for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentState::Initializing => write!(f, "initializing"),
            AgentState::Idle => write!(f, "idle"),
            AgentState::Processing => write!(f, "processing"),
            AgentState::Thinking => write!(f, "thinking"),
            AgentState::WaitingForTool => write!(f, "waiting_for_tool"),
            AgentState::WaitingForUser => write!(f, "waiting_for_user"),
            AgentState::Paused => write!(f, "paused"),
            AgentState::Completed => write!(f, "completed"),
            AgentState::Error(e) => write!(f, "error({})", e),
            AgentState::Stopped => write!(f, "stopped"),
            AgentState::Interrupted => write!(f, "interrupted"),
            AgentState::Compacting => write!(f, "compacting"),
            AgentState::Retrying { attempt } => write!(f, "retrying(attempt={})", attempt),
        }
    }
}

/// Return `true` if the transition `from -> to` is legal for the state
/// machine. Illegal transitions are logged by [`Agent::transition`] and
/// rejected (the state is left unchanged).
pub fn is_valid_transition(from: &AgentState, to: &AgentState) -> bool {
    use AgentState::*;
    match (from, to) {
        (Initializing, _) => matches!(to, Idle | Stopped | Error(_)),
        (Idle, _) => matches!(
            to,
            Thinking | Processing | Paused | WaitingForUser | Stopped | Error(_) | Interrupted
        ),
        (Thinking, _) => matches!(
            to,
            WaitingForTool | WaitingForUser | Completed | Idle | Stopped | Error(_) | Compacting
                | Retrying { .. } | Interrupted
        ),
        (WaitingForTool, _) => {
            matches!(
                to,
                Thinking | Completed | WaitingForUser | Stopped | Error(_) | Interrupted
                    | Retrying { .. }
            )
        }
        (WaitingForUser, _) => matches!(to, Idle | Thinking | Stopped | Error(_) | Interrupted),
        (Processing, _) => matches!(
            to,
            Idle | Paused | Completed | Stopped | Error(_) | Interrupted
        ),
        (Paused, _) => matches!(to, Idle | Stopped | Error(_)),
        (Completed, _) => matches!(to, Idle | Thinking | Stopped | Error(_)),
        (Error(_), _) => matches!(to, Idle | Stopped | Initializing),
        (Stopped, _) => matches!(to, Idle | Initializing),
        (Interrupted, _) => matches!(to, Idle | Thinking | Stopped | Error(_) | Completed),
        (Compacting, _) => matches!(to, Thinking | WaitingForTool | Completed | Stopped | Error(_)),
        (Retrying { .. }, _) => matches!(
            to,
            Thinking | WaitingForTool | Completed | Error(_) | Stopped | Idle
        ),
    }
}

/// The outcome of a single conversation turn.
#[derive(Debug, Clone)]
pub enum TurnOutcome {
    /// The turn completed successfully with generated messages.
    Complete {
        /// The messages produced during this turn.
        messages: Vec<Message>,
        /// Token usage for this turn.
        usage: Usage,
        /// Duration of the turn in milliseconds.
        duration_ms: u64,
    },
    /// The turn was halted by a pipeline step.
    Halted {
        /// The reason the turn was halted.
        reason: String,
        /// The messages accumulated before halting.
        messages: Vec<Message>,
        /// Token usage before halting.
        usage: Usage,
    },
    /// The turn encountered an error.
    Error {
        /// The error message.
        message: String,
        /// The messages accumulated before the error.
        messages: Vec<Message>,
        /// Token usage before the error.
        usage: Usage,
    },
}

impl TurnOutcome {
    /// Get the messages from this outcome, if available.
    pub fn messages(&self) -> &[Message] {
        match self {
            TurnOutcome::Complete { messages, .. } => messages,
            TurnOutcome::Halted { messages, .. } => messages,
            TurnOutcome::Error { messages, .. } => messages,
        }
    }

    /// Get the usage from this outcome.
    pub fn usage(&self) -> Usage {
        match self {
            TurnOutcome::Complete { usage, .. } => *usage,
            TurnOutcome::Halted { usage, .. } => *usage,
            TurnOutcome::Error { usage, .. } => *usage,
        }
    }

    /// Returns true if this outcome represents a successful completion.
    pub fn is_success(&self) -> bool {
        matches!(self, TurnOutcome::Complete { .. })
    }

    /// Returns true if this outcome represents an error.
    pub fn is_error(&self) -> bool {
        matches!(self, TurnOutcome::Error { .. })
    }

    /// Returns true if this outcome was halted by a pipeline step.
    pub fn is_halted(&self) -> bool {
        matches!(self, TurnOutcome::Halted { .. })
    }
}

/// The trait that all turn generators must implement.
///
/// A TurnGenerator is responsible for producing model responses given
/// a list of input messages. Implementations may call an LLM provider,
/// use a local model, or delegate to another service.
#[async_trait]
pub trait TurnGenerator: Send + Sync + fmt::Debug {
    /// Generate a response for the given messages, returning the response
    /// messages and token usage.
    async fn generate(&self, messages: &[Message]) -> Result<Vec<Message>>;

    /// Get the name of the model this generator is using.
    fn model_name(&self) -> &str;

    /// Get the name of the provider this generator is using.
    fn provider_name(&self) -> &str;
}

/// Configuration for an agent's execution loop.
///
/// Mirrors the Python `AgentConfig` dataclass: tunable limits for the number
/// of turns, tool calls per turn, timeouts, and whether tool / subprocess
/// execution is permitted.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// The maximum number of turns this agent may execute before it must be
    /// reset. `0` means no limit.
    pub max_turns: u32,
    /// The maximum number of tool calls executed per round. Additional calls
    /// returned by the model beyond this limit are reported as errors.
    pub max_tool_calls_per_turn: u32,
    /// Timeout in seconds for external commands and git operations.
    pub timeout_seconds: u64,
    /// Whether tool execution is permitted at all. When `false` every tool
    /// call is answered with a `PERMISSION_DENIED` result.
    pub allow_tool_execution: bool,
    /// Whether subprocess execution (shell, code execution, git) is allowed.
    pub allow_subprocess: bool,
    /// The workspace directory commands run in. `None` means the process
    /// working directory.
    pub workspace_dir: Option<PathBuf>,
    /// The default model to use, if any.
    pub default_model: String,
    /// The default provider to use, if any.
    pub default_provider: String,
    /// The system prompt injected into the conversation, if any.
    pub system_prompt: String,
    /// The model's context window in tokens, used for compaction and history
    /// trimming decisions.
    pub context_window_tokens: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tool_calls_per_turn: 8,
            timeout_seconds: 60,
            allow_tool_execution: true,
            allow_subprocess: true,
            workspace_dir: None,
            default_model: String::new(),
            default_provider: String::new(),
            system_prompt: String::new(),
            context_window_tokens: 128_000,
        }
    }
}

/// An error raised by the agent state machine.
#[derive(Debug, Clone, thiserror::Error, serde::Serialize, serde::Deserialize)]
pub enum AgentError {
    /// An error surfaced by the underlying provider/generator.
    #[error("Provider error: {0}")]
    Provider(String),

    /// An error while executing a named tool.
    #[error("Tool '{tool}' failed: {message}")]
    Tool { tool: String, message: String },

    /// A command or provider call exceeded the configured timeout.
    #[error("Operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    /// An illegal state-machine transition was attempted.
    #[error("Invalid state transition from '{from}' to '{to}'")]
    State { from: String, to: String },

    /// The agent exhausted its configured turn budget.
    #[error("Maximum turns reached")]
    MaxTurnsReached,

    /// The agent exhausted its per-round tool-call budget.
    #[error("Maximum tool calls per turn reached")]
    MaxToolCallsReached,

    /// An I/O error (file, network, or process).
    #[error("I/O error: {0}")]
    Io(String),

    /// A serialization / deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// The input was invalid.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// The agent loop was asked to stop.
    #[error("Stop requested")]
    StopRequested,
}

impl From<Error> for AgentError {
    fn from(e: Error) -> Self {
        match e {
            Error::Provider(s) => AgentError::Provider(s),
            Error::ToolExecution(s) => AgentError::Tool {
                tool: "tool".to_string(),
                message: s,
            },
            Error::Io(e) => AgentError::Io(e.to_string()),
            Error::Serialization(e) => AgentError::Serialization(e.to_string()),
            Error::InvalidInput(s) => AgentError::InvalidInput(s),
            other => AgentError::Provider(other.to_string()),
        }
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::Serialization(e.to_string())
    }
}

/// The recovery action the runtime should take after an agent error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry after the given delay in milliseconds.
    Retry { delay_ms: u64 },
    /// Fail over to an alternative provider / model.
    FailOver { message: String },
    /// Stop the agent loop and surface the message.
    Stop { message: String },
    /// Continue the loop (e.g. a recoverable tool error).
    Continue,
}

/// A single token-usage event recorded against the agent.
#[derive(Debug, Clone)]
pub struct UsageEvent {
    /// The model that generated the tokens.
    pub model: String,
    /// The provider that served the request.
    pub provider: String,
    /// Input (prompt) tokens consumed.
    pub input_tokens: u64,
    /// Output (completion) tokens produced.
    pub output_tokens: u64,
    /// When the usage was recorded.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl UsageEvent {
    /// Create a new usage event.
    pub fn new(
        model: impl Into<String>,
        provider: impl Into<String>,
        input_tokens: u64,
        output_tokens: u64,
    ) -> Self {
        Self {
            model: model.into(),
            provider: provider.into(),
            input_tokens,
            output_tokens,
            timestamp: chrono::Utc::now(),
        }
    }

    /// The total tokens consumed by this event.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Per-model usage accumulated by the agent.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelUsage {
    /// Input tokens consumed through this model.
    pub input_tokens: u64,
    /// Output tokens produced through this model.
    pub output_tokens: u64,
    /// The number of calls recorded against this model.
    pub calls: u64,
}

/// Accumulated token usage across all turns of an agent.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct UsageStats {
    /// Total input tokens across all turns.
    pub total_input_tokens: u64,
    /// Total output tokens across all turns.
    pub total_output_tokens: u64,
    /// Total tokens (input + output) across all turns.
    pub total_tokens: u64,
    /// The number of turns recorded.
    pub turn_count: u64,
    /// Usage broken down per model.
    pub per_model: std::collections::HashMap<String, ModelUsage>,
}

impl UsageStats {
    /// Create a new empty usage accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a usage event into this accumulator.
    pub fn record(&mut self, event: &UsageEvent) {
        self.total_input_tokens += event.input_tokens;
        self.total_output_tokens += event.output_tokens;
        self.total_tokens += event.total();
        self.turn_count += 1;
        let entry = self.per_model.entry(event.model.clone()).or_default();
        entry.input_tokens += event.input_tokens;
        entry.output_tokens += event.output_tokens;
        entry.calls += 1;
    }

    /// Merge another accumulator's counts into this one.
    pub fn merge(&mut self, other: &UsageStats) {
        self.total_input_tokens += other.total_input_tokens;
        self.total_output_tokens += other.total_output_tokens;
        self.total_tokens += other.total_tokens;
        self.turn_count += other.turn_count;
        for (model, usage) in &other.per_model {
            let entry = self.per_model.entry(model.clone()).or_default();
            entry.input_tokens += usage.input_tokens;
            entry.output_tokens += usage.output_tokens;
            entry.calls += usage.calls;
        }
    }

    /// Convert the accumulated totals into a [`Usage`] struct.
    pub fn to_usage(&self) -> Usage {
        Usage::new(self.total_input_tokens, self.total_output_tokens)
    }
}

/// The per-turn context threaded through the agent loop.
///
/// A `TurnContext` snapshots the loop configuration for one turn so that the
/// tool-call loop, subprocess helpers, and state transitions all agree on the
/// same budget.
#[derive(Debug, Clone)]
pub struct TurnContext {
    /// Unique identifier for this turn.
    pub turn_id: String,
    /// The messages that make up the input for this turn.
    pub messages: Vec<Message>,
    /// The model being used for this turn.
    pub model: String,
    /// The provider serving this turn.
    pub provider: String,
    /// The current tool round (0-based).
    pub tool_round: u32,
    /// The maximum number of tool rounds allowed.
    pub max_tool_rounds: u32,
    /// The maximum number of tool calls executed per round.
    pub max_tool_calls_per_turn: u32,
    /// Whether tool execution is permitted this turn.
    pub allow_tool_execution: bool,
    /// When the turn started.
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl TurnContext {
    /// Create a new turn context.
    pub fn new(
        turn_id: impl Into<String>,
        messages: Vec<Message>,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            messages,
            model: model.into(),
            provider: provider.into(),
            tool_round: 0,
            max_tool_rounds: 10,
            max_tool_calls_per_turn: 8,
            allow_tool_execution: true,
            started_at: chrono::Utc::now(),
        }
    }

    /// Whether this turn has exhausted its tool-round budget.
    pub fn rounds_exhausted(&self) -> bool {
        self.tool_round >= self.max_tool_rounds
    }
}

/// The result of a synchronously executed command.
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// The command line that was executed.
    pub command: String,
    /// The process exit code, if the process exited normally.
    pub exit_code: Option<i32>,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Wall-clock duration of the execution in milliseconds.
    pub duration_ms: u64,
    /// Whether the process exited successfully (exit code 0).
    pub success: bool,
}

impl CommandResult {
    /// Return the primary output text: stdout, or stderr when stdout is empty.
    pub fn output(&self) -> String {
        if self.stdout.trim().is_empty() {
            self.stderr.clone()
        } else {
            self.stdout.clone()
        }
    }
}

/// The result of a git operation.
#[derive(Debug, Clone)]
pub struct GitResult {
    /// The operation that was run (e.g. `status`, `diff`).
    pub operation: String,
    /// The git process exit code, if it exited normally.
    pub exit_code: Option<i32>,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
    /// The first line of stdout for head-producing operations (log/status).
    pub head: Option<String>,
}

impl GitResult {
    /// Whether the git operation completed successfully.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Return the primary output text.
    pub fn output(&self) -> String {
        if self.stdout.trim().is_empty() {
            self.stderr.clone()
        } else {
            self.stdout.clone()
        }
    }
}

/// A handle to a background process.
///
/// Mirrors the Python `background_process` tool. The child process is started
/// detached; [`BackgroundProcess::wait`] reaps it and collects its output,
/// while [`BackgroundProcess::stop`] terminates it.
pub struct BackgroundProcess {
    /// Unique identifier for this background process handle.
    id: String,
    /// The command that was started.
    command: String,
    /// The child process, consumed by `wait`/`stop`.
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    /// When the process was started.
    started_at: Instant,
}

impl fmt::Debug for BackgroundProcess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackgroundProcess")
            .field("id", &self.id)
            .field("command", &self.command)
            .field("started_at", &self.started_at)
            .finish_non_exhaustive()
    }
}

impl BackgroundProcess {
    /// Create a new background process handle from a spawned child.
    pub fn new(command: impl Into<String>, child: tokio::process::Child) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            command: command.into(),
            child: tokio::sync::Mutex::new(Some(child)),
            started_at: Instant::now(),
        }
    }

    /// The unique identifier of this handle.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The command that was started.
    pub fn command(&self) -> &str {
        &self.command
    }

    /// The OS process ID, if the child is still alive.
    pub async fn pid(&self) -> Option<u32> {
        self.child.lock().await.as_ref().and_then(|c| c.id())
    }

    /// Whether the child process is still being tracked.
    pub async fn is_running(&self) -> bool {
        self.child.lock().await.is_some()
    }

    /// Wait for the process to exit and collect its output.
    pub async fn wait(&self) -> Result<CommandResult> {
        let mut guard = self.child.lock().await;
        let child = guard
            .take()
            .ok_or_else(|| Error::Internal("background process already reaped".to_string()))?;
        let output = child.wait_with_output().await.map_err(Error::Io)?;
        Ok(CommandResult {
            command: self.command.clone(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            success: output.status.success(),
        })
    }

    /// Terminate the process and collect any output produced so far.
    pub async fn stop(&self) -> Result<CommandResult> {
        let mut guard = self.child.lock().await;
        let mut child = guard
            .take()
            .ok_or_else(|| Error::Internal("background process already reaped".to_string()))?;
        let _ = child.kill().await;
        let output = child.wait_with_output().await.map_err(Error::Io)?;
        Ok(CommandResult {
            command: self.command.clone(),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration_ms: self.started_at.elapsed().as_millis() as u64,
            success: output.status.success(),
        })
    }
}

/// A git operation supported by [`Agent::git_operation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitOperation {
    /// `git status --short`
    Status,
    /// `git diff`
    Diff,
    /// `git add`
    Add,
    /// `git commit`
    Commit,
    /// `git push`
    Push,
    /// `git pull`
    Pull,
    /// `git log --oneline -n 20`
    Log,
    /// `git clone`
    Clone,
    /// `git checkout`
    Checkout,
    /// `git branch`
    Branch,
    /// `git remote -v`
    Remote,
    /// `git init`
    Init,
    /// `git reset`
    Reset,
    /// `git fetch`
    Fetch,
    /// `git stash`
    Stash,
    /// `git tag`
    Tag,
    /// `git show`
    Show,
    /// `git merge`
    Merge,
}

impl GitOperation {
    /// The stable string identifier of the operation.
    pub fn as_str(&self) -> &'static str {
        match self {
            GitOperation::Status => "status",
            GitOperation::Diff => "diff",
            GitOperation::Add => "add",
            GitOperation::Commit => "commit",
            GitOperation::Push => "push",
            GitOperation::Pull => "pull",
            GitOperation::Log => "log",
            GitOperation::Clone => "clone",
            GitOperation::Checkout => "checkout",
            GitOperation::Branch => "branch",
            GitOperation::Remote => "remote",
            GitOperation::Init => "init",
            GitOperation::Reset => "reset",
            GitOperation::Fetch => "fetch",
            GitOperation::Stash => "stash",
            GitOperation::Tag => "tag",
            GitOperation::Show => "show",
            GitOperation::Merge => "merge",
        }
    }

    /// Whether the operation emits a HEAD line (rev/status summary) on stdout.
    fn is_head_producing(&self) -> bool {
        matches!(
            self,
            GitOperation::Log
                | GitOperation::Status
                | GitOperation::Show
                | GitOperation::Branch
                | GitOperation::Tag
        )
    }
}

impl fmt::Display for GitOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GitOperation {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "status" => Ok(GitOperation::Status),
            "diff" => Ok(GitOperation::Diff),
            "add" => Ok(GitOperation::Add),
            "commit" => Ok(GitOperation::Commit),
            "push" => Ok(GitOperation::Push),
            "pull" => Ok(GitOperation::Pull),
            "log" => Ok(GitOperation::Log),
            "clone" => Ok(GitOperation::Clone),
            "checkout" => Ok(GitOperation::Checkout),
            "branch" => Ok(GitOperation::Branch),
            "remote" => Ok(GitOperation::Remote),
            "init" => Ok(GitOperation::Init),
            "reset" => Ok(GitOperation::Reset),
            "fetch" => Ok(GitOperation::Fetch),
            "stash" => Ok(GitOperation::Stash),
            "tag" => Ok(GitOperation::Tag),
            "show" => Ok(GitOperation::Show),
            "merge" => Ok(GitOperation::Merge),
            other => Err(Error::InvalidInput(format!(
                "Unknown git operation '{other}'"
            ))),
        }
    }
}

/// Build the default git argument vector for an operation.
///
/// Extra arguments supplied by the caller are appended after these.
pub fn git_args_for(op: GitOperation) -> Vec<&'static str> {
    match op {
        GitOperation::Status => vec!["status", "--short"],
        GitOperation::Diff => vec!["diff"],
        GitOperation::Add => vec!["add"],
        GitOperation::Commit => vec!["commit"],
        GitOperation::Push => vec!["push"],
        GitOperation::Pull => vec!["pull"],
        GitOperation::Log => vec!["log", "--oneline", "-n", "20"],
        GitOperation::Clone => vec!["clone"],
        GitOperation::Checkout => vec!["checkout"],
        GitOperation::Branch => vec!["branch"],
        GitOperation::Remote => vec!["remote", "-v"],
        GitOperation::Init => vec!["init"],
        GitOperation::Reset => vec!["reset"],
        GitOperation::Fetch => vec!["fetch"],
        GitOperation::Stash => vec!["stash"],
        GitOperation::Tag => vec!["tag"],
        GitOperation::Show => vec!["show"],
        GitOperation::Merge => vec!["merge"],
    }
}

/// Tool names that require subprocess execution.
const SUBPROCESS_TOOLS: &[&str] = &[
    "exec_command",
    "background_process",
    "execute_code",
    "apply_patch",
    "git_clone",
    "git_status",
    "git_diff",
    "git_add",
    "git_commit",
    "git_push",
    "git_log",
];

/// Return `true` if a tool name is a subprocess-backed tool.
///
/// Git tools are matched both by their explicit names above and by the
/// `git_` prefix so future git tools are covered automatically.
pub fn is_subprocess_tool(name: &str) -> bool {
    SUBPROCESS_TOOLS.contains(&name) || name.starts_with("git_")
}

/// Estimate the token footprint of a message list.
///
/// A cheap deterministic heuristic: roughly one token per four content
/// characters, matching the provider stage's estimate.
fn estimate_message_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|m| m.text_content().chars().count() as u64 / 4)
        .sum()
}

/// A simple agent that holds a generator and manages its lifecycle state.
pub struct Agent {
    /// The unique identifier for this agent.
    id: String,
    /// A human-readable display name for this agent.
    name: String,
    /// The current state of the agent.
    state: AgentState,
    /// Tunable configuration for the agent loop.
    config: AgentConfig,
    /// The turn generator used to produce responses.
    generator: Box<dyn TurnGenerator>,
    /// The conversation history for this agent.
    conversation: Vec<Message>,
    /// The shared tool executor used to run tool calls.
    tool_executor: Option<Arc<dyn crate::runtime::ToolExecutor>>,
    /// Accumulated token usage across all turns.
    usage: UsageStats,
    /// When the agent was created.
    created_at: chrono::DateTime<chrono::Utc>,
    /// The number of turns executed.
    turn_count: u64,
    /// The most recent error, if any.
    last_error: Option<AgentError>,
    /// Background processes spawned by this agent.
    bg_processes: BackgroundProcessManager,
}

impl Agent {
    /// Create a new agent with the given ID and generator, using the default
    /// [`AgentConfig`]. The agent starts in the [`AgentState::Initializing`]
    /// state; call [`Agent::initialize`] to move it to `Idle`.
    pub fn new(id: impl Into<String>, generator: Box<dyn TurnGenerator>) -> Self {
        let id = id.into();
        let mut agent = Self::with_config(id.clone(), generator, AgentConfig::default());
        agent.name = id.clone();
        agent
    }

    /// Create a new agent with the given ID, generator, and configuration.
    pub fn with_config(
        id: impl Into<String>,
        generator: Box<dyn TurnGenerator>,
        config: AgentConfig,
    ) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            state: AgentState::Initializing,
            config,
            generator,
            conversation: Vec::new(),
            tool_executor: None,
            usage: UsageStats::new(),
            created_at: chrono::Utc::now(),
            turn_count: 0,
            last_error: None,
            bg_processes: BackgroundProcessManager::new(),
        }
    }

    /// Set the display name and return the agent (builder style).
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the agent's unique identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the agent's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the agent's display name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Get the agent's configuration.
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get the current state of the agent.
    pub fn state(&self) -> &AgentState {
        &self.state
    }

    /// Get the current state of the agent (alias of [`Agent::state`]).
    pub fn get_state(&self) -> &AgentState {
        &self.state
    }

    /// Get the conversation history for this agent.
    pub fn conversation(&self) -> &[Message] {
        &self.conversation
    }

    /// Get the accumulated usage statistics for this agent.
    pub fn get_usage_stats(&self) -> &UsageStats {
        &self.usage
    }

    /// Get the number of turns this agent has executed.
    pub fn turn_count(&self) -> u64 {
        self.turn_count
    }

    /// Get the most recent error, if any.
    pub fn last_error(&self) -> Option<&AgentError> {
        self.last_error.as_ref()
    }

    /// Get the model name reported by the underlying generator.
    pub fn model_name(&self) -> &str {
        self.generator.model_name()
    }

    /// Get the provider name reported by the underlying generator.
    pub fn provider_name(&self) -> &str {
        self.generator.provider_name()
    }

    /// Access the underlying generator as a trait object.
    pub fn generator(&self) -> &dyn TurnGenerator {
        self.generator.as_ref()
    }

    /// Set the tool executor used to run tool calls.
    pub fn set_tool_executor(&mut self, executor: Arc<dyn crate::runtime::ToolExecutor>) {
        self.tool_executor = Some(executor);
    }

    /// Get the configured tool executor, if any.
    pub fn tool_executor(&self) -> Option<Arc<dyn crate::runtime::ToolExecutor>> {
        self.tool_executor.clone()
    }

    /// Transition the agent to the ready state.
    pub fn initialize(&mut self) {
        self.state = AgentState::Idle;
    }

    /// Transition the agent to a new state, validating the transition.
    ///
    /// Returns an error (and leaves the state unchanged) for illegal
    /// transitions such as `Idle -> WaitingForTool`.
    pub fn transition(&mut self, new_state: AgentState) -> Result<()> {
        let from = self.state.clone();
        if !is_valid_transition(&from, &new_state) {
            warn!(
                agent_id = %self.id,
                from = %from,
                to = %new_state,
                "Invalid agent state transition"
            );
            return Err(Error::InvalidInput(format!(
                "Invalid state transition from {from} to {new_state}"
            )));
        }
        debug!(
            agent_id = %self.id,
            from = %from,
            to = %new_state,
            "Agent state transition"
        );
        self.state = new_state;
        Ok(())
    }

    /// Reset the agent to its idle state, clearing conversation history and
    /// turn counters. Usage statistics are preserved across resets.
    pub fn reset(&mut self) {
        self.conversation.clear();
        self.state = AgentState::Idle;
        self.turn_count = 0;
        self.last_error = None;
        debug!(agent_id = %self.id, "Agent reset");
    }

    /// Add a message to the agent's conversation history.
    pub fn add_message(&mut self, message: Message) {
        self.conversation.push(message);
    }

    /// Record a token-usage event against this agent.
    pub fn track_usage(&mut self, event: UsageEvent) {
        self.usage.record(&event);
    }

    /// Classify an [`AgentError`] into a [`RecoveryAction`].
    ///
    /// This is the pure recovery decision; the runtime applies the returned
    /// action (retry, fail over, stop, or continue). The error is also
    /// recorded as the agent's [`Agent::last_error`].
    pub fn handle_error(&mut self, error: AgentError) -> RecoveryAction {
        self.last_error = Some(error.clone());
        match error {
            AgentError::Provider(ref msg) => {
                let lower = msg.to_ascii_lowercase();
                if lower.contains("rate") || lower.contains("429") {
                    RecoveryAction::Retry { delay_ms: 1000 }
                } else if lower.contains("timeout") || lower.contains("timed out") {
                    RecoveryAction::Retry { delay_ms: 250 }
                } else {
                    RecoveryAction::FailOver {
                        message: msg.clone(),
                    }
                }
            }
            AgentError::Timeout { .. } => RecoveryAction::Retry { delay_ms: 500 },
            AgentError::Tool { .. } => RecoveryAction::Continue,
            AgentError::MaxTurnsReached => RecoveryAction::Stop {
                message: "Maximum turns reached".to_string(),
            },
            AgentError::MaxToolCallsReached => RecoveryAction::Continue,
            AgentError::State { .. } => RecoveryAction::Stop {
                message: "Invalid state transition".to_string(),
            },
            AgentError::StopRequested => RecoveryAction::Stop {
                message: "Stop requested".to_string(),
            },
            AgentError::Io(_) | AgentError::Serialization(_) | AgentError::InvalidInput(_) => {
                RecoveryAction::FailOver {
                    message: error.to_string(),
                }
            }
        }
    }

    /// Create a [`TurnContext`] from a batch of messages using this agent's
    /// configuration and generator.
    pub fn create_turn_context(&self, messages: Vec<Message>) -> TurnContext {
        TurnContext {
            turn_id: uuid::Uuid::new_v4().to_string(),
            messages,
            model: self.generator.model_name().to_string(),
            provider: self.generator.provider_name().to_string(),
            tool_round: 0,
            max_tool_rounds: self.config.max_turns.max(1),
            max_tool_calls_per_turn: self.config.max_tool_calls_per_turn,
            allow_tool_execution: self.config.allow_tool_execution,
            started_at: chrono::Utc::now(),
        }
    }

    /// Execute one turn of generation.
    ///
    /// The agent transitions to `Thinking`, calls the generator once, and then
    /// — if the response contains tool calls — executes them through the
    /// configured tool executor and appends the results. Full multi-round
    /// looping is orchestrated by [`Agent::run_turn_with_runner`]; this method
    /// performs a single generation round plus its immediate tool batch.
    #[instrument(skip(self), fields(agent_id = %self.id, turn_id = %context.turn_id))]
    pub async fn generate_turn(&mut self, context: &TurnContext) -> TurnOutcome {
        let start = Instant::now();
        // Entering a turn always moves to Thinking regardless of the prior
        // state (unless stopped), so set the state directly rather than going
        // through the transition table.
        self.state = AgentState::Thinking;

        let generation = self.generator.generate(&context.messages).await;
        let messages = match generation {
            Ok(msgs) => msgs,
            Err(e) => {
                let message = format!("Provider error: {e}");
                self.state = AgentState::Error(e.to_string());
                return TurnOutcome::Error {
                    message,
                    messages: context.messages.clone(),
                    usage: self.usage.to_usage(),
                };
            }
        };

        let calls = crate::turn_control::pending_tool_calls(&messages);

        let (final_messages, final_state) = if calls.is_empty() {
            (messages, AgentState::Completed)
        } else {
            let _ = self.transition(AgentState::WaitingForTool);
            let results = self.tool_call_loop(context, &calls).await;
            let mut all = messages;
            all.push(self.process_tool_results(results));
            // In a full runner the loop would bounce back to Thinking; here we
            // report the tool round as completed with the results appended.
            (all, AgentState::Completed)
        };

        self.state = final_state;
        self.turn_count += 1;
        let duration_ms = start.elapsed().as_millis() as u64;
        debug!(
            agent_id = %self.id,
            turn_id = %context.turn_id,
            duration_ms,
            "Turn generation complete"
        );
        TurnOutcome::Complete {
            messages: final_messages,
            usage: self.usage.to_usage(),
            duration_ms,
        }
    }

    /// Execute a batch of tool calls in sequence, returning one result each.
    ///
    /// The loop respects the per-round tool-call budget in `context` and the
    /// agent's `allow_tool_execution` / `allow_subprocess` configuration.
    #[instrument(skip(self, context), fields(agent_id = %self.id, calls = calls.len()))]
    pub async fn tool_call_loop(
        &self,
        context: &TurnContext,
        calls: &[ToolCall],
    ) -> Vec<ToolResult> {
        let mut results = Vec::with_capacity(calls.len());
        for (i, call) in calls.iter().enumerate() {
            if i >= context.max_tool_calls_per_turn as usize {
                results.push(ToolResult::error(
                    &call.id,
                    format!(
                        "Max tool calls per turn ({}) exceeded",
                        context.max_tool_calls_per_turn
                    ),
                ));
                continue;
            }
            results.push(self.execute_tool_call(call).await);
        }
        results
    }

    /// Execute a single tool call, mapping all failures to a `ToolResult`.
    ///
    /// Tool execution is skipped (with a `PERMISSION_DENIED` result) when tool
    /// execution or subprocess execution is disabled by configuration.
    #[instrument(skip(self), fields(agent_id = %self.id, tool = %call.name))]
    pub async fn execute_tool_call(&self, call: &ToolCall) -> ToolResult {
        if !self.config.allow_tool_execution {
            return ToolResult::error(
                &call.id,
                format!("Tool execution is disabled for '{}'", call.name),
            );
        }
        if !self.config.allow_subprocess && is_subprocess_tool(&call.name) {
            return ToolResult::error(
                &call.id,
                format!("Subprocess tool '{}' is disabled", call.name),
            );
        }
        match self.tool_executor.as_ref() {
            Some(executor) => match executor.execute(call).await {
                Ok(result) => result,
                Err(e) => self.handle_tool_error(call, e),
            },
            None => ToolResult::error(
                &call.id,
                format!("No tool executor configured for '{}'", call.name),
            ),
        }
    }

    /// Execute a batch of tool calls concurrently and collect their results.
    ///
    /// Each call is executed independently; a failing call does not abort the
    /// others. The per-round budget is still enforced.
    pub async fn execute_tool_calls_parallel(
        &self,
        context: &TurnContext,
        calls: &[ToolCall],
    ) -> Vec<ToolResult> {
        let max = context.max_tool_calls_per_turn as usize;

        // Execute the in-budget calls concurrently.
        let in_budget: Vec<ToolCall> = calls.iter().take(max).cloned().collect();
        let futures = in_budget
            .iter()
            .map(|call| self.execute_tool_call(call))
            .collect::<Vec<_>>();
        let executed = futures::future::join_all(futures).await;

        // Assemble the result vector in call order: executed results first,
        // then budget errors for any calls that were cut off.
        let mut results = Vec::with_capacity(calls.len());
        results.extend(executed);
        for call in calls.iter().skip(max) {
            results.push(ToolResult::error(
                &call.id,
                format!(
                    "Max tool calls per turn ({}) exceeded",
                    context.max_tool_calls_per_turn
                ),
            ));
        }
        results
    }

    /// Convert a tool-execution error into a `ToolResult` error frame.
    pub fn handle_tool_error(&self, call: &ToolCall, error: Error) -> ToolResult {
        debug!(
            agent_id = %self.id,
            tool = %call.name,
            error = %error,
            "Tool execution failed"
        );
        ToolResult::error(&call.id, format!("[{}] {}", call.name, error))
    }

    /// Collapse a batch of tool results into a single tool-role message.
    ///
    /// Each result becomes a `ContentBlock::ToolResult` so the provider can
    /// correlate them by `tool_use_id`.
    pub fn process_tool_results(&self, results: Vec<ToolResult>) -> Message {
        let content = results
            .into_iter()
            .map(ContentBlock::ToolResult)
            .collect::<Vec<_>>();
        Message {
            role: MessageRole::Tool,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    /// Run a command synchronously and capture its output.
    ///
    /// The command runs in the agent's workspace directory (when configured)
    /// and is subject to the configured timeout. Returns an error if
    /// subprocess execution is disabled.
    #[instrument(skip(self, args), fields(agent_id = %self.id, command = %command))]
    pub async fn run_command(&self, command: &str, args: &[String]) -> Result<CommandResult> {
        if !self.config.allow_subprocess {
            return Err(Error::InvalidInput(
                "Subprocess execution is disabled".to_string(),
            ));
        }
        let start = Instant::now();
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        cmd.kill_on_drop(true);
        if let Some(dir) = &self.config.workspace_dir {
            cmd.current_dir(dir);
        }
        let output = tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds),
            cmd.output(),
        )
        .await
        .map_err(|_| {
            Error::Provider(format!(
                "Command '{command}' timed out after {}s",
                self.config.timeout_seconds
            ))
        })?
        .map_err(Error::Io)?;

        Ok(CommandResult {
            command: format!("{command} {}", args.join(" ")),
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
            success: output.status.success(),
        })
    }

    /// Start a command in the background, returning a [`BackgroundProcess`]
    /// handle.
    ///
    /// The process's stdout and stderr are piped so `wait`/`stop` can collect
    /// them. The process is killed if the handle is dropped.
    #[instrument(skip(self, args), fields(agent_id = %self.id, command = %command))]
    pub async fn run_background_process(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<BackgroundProcess> {
        if !self.config.allow_subprocess {
            return Err(Error::InvalidInput(
                "Subprocess execution is disabled".to_string(),
            ));
        }
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args);
        cmd.kill_on_drop(true);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        if let Some(dir) = &self.config.workspace_dir {
            cmd.current_dir(dir);
        }
        let child = cmd.spawn().map_err(Error::Io)?;
        Ok(BackgroundProcess::new(command, child))
    }

    /// Execute a git operation through the `git` binary.
    ///
    /// Extra arguments are appended after the operation's default argument
    /// vector (see [`git_args_for`]).
    #[instrument(skip(self, args), fields(agent_id = %self.id, op = %op.as_str()))]
    pub async fn git_operation(&self, op: GitOperation, args: &[String]) -> Result<GitResult> {
        if !self.config.allow_subprocess {
            return Err(Error::InvalidInput(
                "Git operations require subprocess execution".to_string(),
            ));
        }
        let mut git_args: Vec<&str> = git_args_for(op);
        git_args.extend(args.iter().map(|s| s.as_str()));

        let start = Instant::now();
        let mut cmd = tokio::process::Command::new("git");
        cmd.args(&git_args);
        cmd.kill_on_drop(true);
        if let Some(dir) = &self.config.workspace_dir {
            cmd.current_dir(dir);
        }
        let output = tokio::time::timeout(
            Duration::from_secs(self.config.timeout_seconds),
            cmd.output(),
        )
        .await
        .map_err(|_| {
            Error::Provider(format!(
                "git {} timed out after {}s",
                op.as_str(),
                self.config.timeout_seconds
            ))
        })?
        .map_err(Error::Io)?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let head = if output.status.success() && op.is_head_producing() {
            stdout.lines().next().map(|s| s.trim().to_string())
        } else {
            None
        };

        Ok(GitResult {
            operation: op.as_str().to_string(),
            exit_code: output.status.code(),
            stdout,
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            success: output.status.success(),
            duration_ms: start.elapsed().as_millis() as u64,
            head,
        })
    }

    /// Execute a single turn: generate a response for the given messages.
    #[instrument(skip(self), fields(agent_id = %self.id))]
    pub async fn respond(&mut self, messages: &[Message]) -> Result<Vec<Message>> {
        self.state = AgentState::Processing;

        // Add incoming messages to the conversation.
        for msg in messages {
            self.conversation.push(msg.clone());
        }

        let result = self.generator.generate(&self.conversation).await;

        match result {
            Ok(response_messages) => {
                // Add generated messages to the conversation.
                for msg in &response_messages {
                    self.conversation.push(msg.clone());
                }
                self.state = AgentState::Idle;
                Ok(response_messages)
            }
            Err(e) => {
                self.state = AgentState::Error(e.to_string());
                Err(e)
            }
        }
    }

    /// Run a full turn through the provided [`crate::runtime::TurnRunner`].
    ///
    /// This drives the entire 8-stage pipeline (harness, bootstrap, compaction,
    /// attachment, input, provider, stream, finalizer) and the multi-round
    /// tool-call loop, then records the resulting usage against the agent.
    #[instrument(skip(self, runner), fields(agent_id = %self.id))]
    pub async fn run_turn_with_runner(
        &mut self,
        messages: Vec<Message>,
        runner: &crate::runtime::TurnRunner,
    ) -> Result<TurnOutcome> {
        self.state = AgentState::Thinking;
        let outcome = runner.run_turn(messages, self.generator.as_ref()).await?;

        match &outcome {
            TurnOutcome::Complete { usage, .. } => {
                let model = self.generator.model_name().to_string();
                let provider = self.generator.provider_name().to_string();
                self.track_usage(UsageEvent::new(
                    model,
                    provider,
                    usage.input_tokens,
                    usage.output_tokens,
                ));
                self.turn_count += 1;
            }
            TurnOutcome::Error { .. } => {
                self.state = AgentState::Error("turn failed".to_string());
            }
            _ => {}
        }
        if outcome.is_success() {
            self.state = AgentState::Completed;
        }
        Ok(outcome)
    }

    /// Run a full multi-round turn: the explicit state machine loop.
    ///
    /// Mirrors the Python `Agent._turn_generator`:
    /// `IDLE → THINKING → [WAITING_FOR_TOOL → THINKING → …] → COMPLETED`,
    /// with a context-window check before each generation round and usage
    /// accounting per round. Tool calls are executed through the configured
    /// tool executor and their results fed back into the conversation until
    /// the model returns final text or the round budget is exhausted.
    #[instrument(skip(self), fields(agent_id = %self.id))]
    pub async fn run_turn(&mut self, messages: Vec<Message>) -> TurnOutcome {
        let start = Instant::now();
        self.state = AgentState::Thinking;

        let mut working = self.conversation.clone();
        working.extend(messages);

        let max_rounds = self.config.max_turns.max(1);
        let mut round = 0u32;

        loop {
            // Context-window management: compact before generating when the
            // conversation is over the compaction threshold.
            if self.should_compact() {
                debug!(agent_id = %self.id, round = round, "compacting history before generation");
                if self.compact_history().is_some() {
                    working = self.conversation.clone();
                }
            }

            // Generate the next response round.
            let response = match self.generator.generate(&working).await {
                Ok(response) => response,
                Err(e) => {
                    let message = format!("Provider error: {e}");
                    self.state = AgentState::Error(e.to_string());
                    return TurnOutcome::Error {
                        message,
                        messages: working.clone(),
                        usage: self.usage.to_usage(),
                    };
                }
            };

            // Record usage for this round.
            let input_tokens = estimate_message_tokens(&working);
            let output_tokens = estimate_message_tokens(&response);
            let model = self.generator.model_name().to_string();
            let provider = self.generator.provider_name().to_string();
            self.track_usage(UsageEvent::new(
                model,
                provider,
                input_tokens,
                output_tokens,
            ));

            let calls = crate::turn_control::pending_tool_calls(&response);

            // Final text surface: no tool calls, complete the turn.
            if calls.is_empty() {
                working.extend(response);
                self.conversation = working;
                self.state = AgentState::Completed;
                self.turn_count += 1;
                let duration_ms = start.elapsed().as_millis() as u64;
                debug!(
                    agent_id = %self.id,
                    rounds = round + 1,
                    duration_ms,
                    "turn completed"
                );
                return TurnOutcome::Complete {
                    messages: self.conversation.clone(),
                    usage: self.usage.to_usage(),
                    duration_ms,
                };
            }

            // Execute the tool calls for this round.
            self.state = AgentState::WaitingForTool;
            let turn_ctx = self.create_turn_context(working.clone());
            let results = self.tool_call_loop(&turn_ctx, &calls).await;
            working.extend(response);
            working.push(self.process_tool_results(results));
            // Clone rather than move: the loop continues to use `working` on the
            // next round.
            self.conversation = working.clone();

            round += 1;
            if round >= max_rounds {
                warn!(
                    agent_id = %self.id,
                    round = round,
                    max_rounds = max_rounds,
                    "max tool rounds reached"
                );
                self.state = AgentState::Completed;
                self.turn_count += 1;
                let duration_ms = start.elapsed().as_millis() as u64;
                return TurnOutcome::Complete {
                    messages: self.conversation.clone(),
                    usage: self.usage.to_usage(),
                    duration_ms,
                };
            }

            self.state = AgentState::Thinking;
        }
    }

    /// The model's context window in tokens.
    pub fn context_window_tokens(&self) -> u64 {
        self.config.context_window_tokens.max(1)
    }

    /// Estimate the current token footprint of the conversation history.
    pub fn estimate_context_tokens(&self) -> u64 {
        estimate_message_tokens(&self.conversation)
    }

    /// The fraction of the context window currently consumed.
    pub fn context_utilization(&self) -> f64 {
        self.estimate_context_tokens() as f64 / self.context_window_tokens() as f64
    }

    /// Whether the conversation is within the context window.
    pub fn within_context_window(&self) -> bool {
        self.estimate_context_tokens() < self.context_window_tokens()
    }

    /// Trim the conversation history to a token budget.
    ///
    /// System messages are always preserved; the most recent messages are
    /// retained backwards until the budget is exhausted. Returns the number of
    /// messages dropped.
    pub fn trim_history_to_budget(&mut self, max_tokens: u64) -> usize {
        let before = self.conversation.len();
        self.conversation =
            crate::history::truncate_to_budget(&self.conversation, max_tokens, 0.25);
        before.saturating_sub(self.conversation.len())
    }

    /// Repair the tool-call pairing in the history.
    ///
    /// Ensures every `tool_use` has a matching `tool_result` and drops
    /// orphaned results (e.g. after a truncated or interrupted turn).
    pub fn repair_history(&mut self) -> crate::history::RepairOutcome {
        let outcome = crate::history::repair_tool_pairs(&self.conversation);
        self.conversation = outcome.messages.clone();
        outcome
    }

    /// Deduplicate repeated messages in the history.
    ///
    /// Returns the number of duplicate messages removed. Tool-result messages
    /// are always preserved.
    pub fn deduplicate_history(&mut self) -> usize {
        let before = self.conversation.len();
        self.conversation = crate::history::deduplicate(&self.conversation);
        before.saturating_sub(self.conversation.len())
    }

    /// Whether the conversation has outgrown the compaction threshold.
    pub fn should_compact(&self) -> bool {
        let threshold = (self.context_window_tokens() * 3) / 4;
        self.estimate_context_tokens() >= threshold
    }

    /// Compact the conversation history in place.
    ///
    /// Drops the oldest non-system messages beyond a truncated budget and
    /// inserts a summary placeholder. Returns the strategy that was applied,
    /// or `None` when no compaction was needed.
    pub fn compact_history(&mut self) -> Option<crate::compaction_control::CompactionStrategy> {
        use crate::compaction_control::{CompactionDecision, CompactionInput, decide_compaction};
        let input = CompactionInput {
            token_count: self.estimate_context_tokens(),
            message_count: self.conversation.len(),
            context_window_tokens: self.context_window_tokens(),
            ..Default::default()
        };
        let strategy = match decide_compaction(&input) {
            CompactionDecision::NoCompaction => return None,
            CompactionDecision::Compact(s) => s,
            CompactionDecision::UrgentCompaction(s) => s,
        };

        let budget = crate::compaction_control::truncation_budget(50);
        let system: Vec<Message> = self
            .conversation
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .cloned()
            .collect();
        let rest: Vec<Message> = self
            .conversation
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .cloned()
            .collect();
        let tail_start = rest.len().saturating_sub(budget);
        let mut compacted = system;
        if tail_start > 0 {
            compacted.push(Message::system(format!(
                "[{} earlier messages summarized]",
                tail_start
            )));
        }
        compacted.extend(rest.into_iter().skip(tail_start));
        self.conversation = compacted;

        // Repair any tool pairs broken by the surgery.
        let _ = self.repair_history();
        Some(strategy)
    }

    /// Extract all reasoning/thinking content from the conversation.
    pub fn extract_thinking(&self) -> Vec<String> {
        self.conversation
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::Reasoning(r) if !r.is_empty() => Some(r.clone()),
                _ => None,
            })
            .collect()
    }

    /// Strip reasoning blocks from the history for a provider that does not
    /// support them. Returns the number of blocks stripped.
    pub fn strip_reasoning(&mut self, support: crate::thinking::ReasoningSupport) -> usize {
        use crate::thinking::{count_reasoning_blocks, sanitize_for_provider};
        if matches!(
            support,
            crate::thinking::ReasoningSupport::Supported
                | crate::thinking::ReasoningSupport::Streaming
        ) {
            return 0;
        }
        let before = count_reasoning_blocks(&self.conversation);
        self.conversation = sanitize_for_provider(&self.conversation, support);
        before.saturating_sub(count_reasoning_blocks(&self.conversation))
    }

    /// Resolve a workspace-relative path against the agent's workspace.
    fn resolve_workspace_path(&self, path: &str) -> PathBuf {
        match &self.config.workspace_dir {
            Some(dir) => dir.join(path),
            None => PathBuf::from(path),
        }
    }

    /// Read a workspace file as UTF-8 text.
    pub async fn read_workspace_file(&self, path: &str) -> Result<String> {
        let resolved = self.resolve_workspace_path(path);
        tokio::fs::read_to_string(&resolved)
            .await
            .map_err(Error::Io)
    }

    /// Write UTF-8 text to a workspace file, creating parents as needed.
    pub async fn write_workspace_file(&self, path: &str, content: &str) -> Result<()> {
        if !self.config.allow_subprocess {
            return Err(Error::InvalidInput("File writes are disabled".to_string()));
        }
        let resolved = self.resolve_workspace_path(path);
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(Error::Io)?;
        }
        tokio::fs::write(&resolved, content)
            .await
            .map_err(Error::Io)
    }

    /// List the entries in a workspace directory.
    pub async fn list_workspace_dir(&self, path: &str) -> Result<Vec<String>> {
        let resolved = self.resolve_workspace_path(path);
        let mut entries = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&resolved).await.map_err(Error::Io)?;
        while let Some(entry) = read_dir.next_entry().await.map_err(Error::Io)? {
            entries.push(entry.file_name().to_string_lossy().to_string());
        }
        Ok(entries)
    }

    /// Whether a workspace path exists.
    pub async fn path_exists(&self, path: &str) -> Result<bool> {
        let resolved = self.resolve_workspace_path(path);
        Ok(tokio::fs::metadata(&resolved).await.is_ok())
    }

    /// Create a workspace directory (and any missing parents).
    pub async fn create_workspace_dir(&self, path: &str) -> Result<()> {
        if !self.config.allow_subprocess {
            return Err(Error::InvalidInput(
                "Filesystem operations are disabled".to_string(),
            ));
        }
        let resolved = self.resolve_workspace_path(path);
        tokio::fs::create_dir_all(&resolved)
            .await
            .map_err(Error::Io)
    }

    /// Initialize a git repository in the workspace.
    pub async fn git_init(&self) -> Result<GitResult> {
        self.git_operation(GitOperation::Init, &[]).await
    }

    /// Reset the git index/working tree. When `hard` is true, discards local
    /// changes (`git reset --hard`).
    pub async fn git_reset(&self, hard: bool) -> Result<GitResult> {
        let args = if hard {
            vec!["--hard".to_string()]
        } else {
            Vec::new()
        };
        self.git_operation(GitOperation::Reset, &args).await
    }

    /// Fetch refs from the configured remote.
    pub async fn git_fetch(&self) -> Result<GitResult> {
        self.git_operation(GitOperation::Fetch, &[]).await
    }

    /// Stash local changes.
    pub async fn git_stash(&self) -> Result<GitResult> {
        self.git_operation(GitOperation::Stash, &[]).await
    }

    /// Pause the agent, preserving its state for later resumption.
    pub fn pause(&mut self) {
        if self.state == AgentState::Idle
            || self.state == AgentState::Processing
            || self.state == AgentState::Thinking
        {
            self.state = AgentState::Paused;
        }
    }

    /// Resume the agent from a paused state.
    pub fn resume(&mut self) {
        if self.state == AgentState::Paused {
            self.state = AgentState::Idle;
        }
    }

    /// Stop the agent and clear its state.
    pub fn stop(&mut self) {
        self.state = AgentState::Stopped;
        self.conversation.clear();
    }
}

/// The phase of a turn the agent is currently in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TurnPhase {
    /// The turn has not started.
    NotStarted,
    /// The pre-turn pipeline is running.
    Pipeline,
    /// The bootstrap stage is running.
    Bootstrap,
    /// Context compaction is running.
    Compaction,
    /// The provider is generating a response.
    Generating,
    /// Tool calls are being executed.
    ToolExecution,
    /// The stream is being consumed.
    Streaming,
    /// Finalization is running.
    Finalizing,
    /// The turn completed.
    Completed,
    /// The turn failed.
    Failed,
}

impl fmt::Display for TurnPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TurnPhase::NotStarted => write!(f, "not_started"),
            TurnPhase::Pipeline => write!(f, "pipeline"),
            TurnPhase::Bootstrap => write!(f, "bootstrap"),
            TurnPhase::Compaction => write!(f, "compaction"),
            TurnPhase::Generating => write!(f, "generating"),
            TurnPhase::ToolExecution => write!(f, "tool_execution"),
            TurnPhase::Streaming => write!(f, "streaming"),
            TurnPhase::Finalizing => write!(f, "finalizing"),
            TurnPhase::Completed => write!(f, "completed"),
            TurnPhase::Failed => write!(f, "failed"),
        }
    }
}

/// The decision for how to proceed after a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallDecision {
    /// Continue the loop: append the result and call the provider again.
    Continue,
    /// Retry the same tool call after a delay.
    Retry {
        /// The delay before retrying, in milliseconds.
        delay_ms: u64,
        /// The number of retries already attempted.
        attempt: u32,
    },
    /// Abort the entire turn with an error.
    Abort {
        /// The error message.
        message: String,
    },
    /// Stop the tool loop but continue the turn (e.g. budget exhausted).
    StopLoop {
        /// The reason the loop stopped.
        reason: String,
    },
}

impl ToolCallDecision {
    /// Whether the loop should continue.
    pub fn should_continue(&self) -> bool {
        matches!(self, ToolCallDecision::Continue)
    }

    /// Whether the tool call should be retried.
    pub fn should_retry(&self) -> bool {
        matches!(self, ToolCallDecision::Retry { .. })
    }

    /// Whether the turn should be aborted.
    pub fn should_abort(&self) -> bool {
        matches!(self, ToolCallDecision::Abort { .. })
    }
}

/// A per-turn budget for tool calls.
#[derive(Debug, Clone)]
pub struct ToolCallBudget {
    /// The maximum number of tool rounds.
    pub max_rounds: u32,
    /// The maximum number of tool calls per round.
    pub max_calls_per_round: u32,
    /// The maximum total number of tool calls per turn.
    pub max_total_calls: u32,
}

impl Default for ToolCallBudget {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            max_calls_per_round: 8,
            max_total_calls: 32,
        }
    }
}

impl ToolCallBudget {
    /// Create a new tool call budget.
    pub fn new(max_rounds: u32, max_calls_per_round: u32, max_total_calls: u32) -> Self {
        Self {
            max_rounds: max_rounds.max(1),
            max_calls_per_round: max_calls_per_round.max(1),
            max_total_calls: max_total_calls.max(1),
        }
    }

    /// Whether the round budget is exhausted.
    pub fn round_exhausted(&self, round: u32) -> bool {
        round >= self.max_rounds
    }

    /// Whether the total-call budget is exhausted.
    pub fn total_exhausted(&self, total_calls: u32) -> bool {
        total_calls >= self.max_total_calls
    }
}

/// A manager for background processes spawned by the agent.
///
/// Mirrors the Python `background_process` tool's process registry. Tracks
/// every spawned child so the agent can wait on, stop, or list them.
#[derive(Debug, Default)]
pub struct BackgroundProcessManager {
    /// The tracked processes, keyed by their id.
    processes: std::sync::Mutex<std::collections::HashMap<String, Arc<BackgroundProcess>>>,
}

impl BackgroundProcessManager {
    /// Create a new empty process manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a background process.
    pub fn register(&self, process: Arc<BackgroundProcess>) {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(process.id().to_string(), process);
    }

    /// Get a process by id.
    pub fn get(&self, id: &str) -> Option<Arc<BackgroundProcess>> {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
    }

    /// Remove a process by id, returning it if present.
    pub fn remove(&self, id: &str) -> Option<Arc<BackgroundProcess>> {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(id)
    }

    /// List all tracked process ids.
    pub fn list_ids(&self) -> Vec<String> {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    /// The number of tracked processes.
    pub fn len(&self) -> usize {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// True when no processes are tracked.
    pub fn is_empty(&self) -> bool {
        self.processes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Stop all tracked processes, returning the number stopped.
    pub async fn stop_all(&self) -> usize {
        let ids = self.list_ids();
        let mut stopped = 0usize;
        for id in ids {
            if let Some(process) = self.remove(&id) {
                let _ = process.stop().await;
                stopped += 1;
            }
        }
        stopped
    }

    /// Prune processes that are no longer running.
    pub async fn prune_finished(&self) -> usize {
        let ids = self.list_ids();
        let mut pruned = 0usize;
        for id in ids {
            if let Some(process) = self.get(&id) {
                if !process.is_running().await {
                    self.remove(&id);
                    pruned += 1;
                }
            }
        }
        pruned
    }
}

/// The result of a git operation with extra metadata for the 13 subprocess
/// git operations from the Python backend.
#[derive(Debug, Clone)]
pub struct GitOpResult {
    /// The result of the git operation.
    pub result: GitResult,
    /// The git operation that was run.
    pub operation: GitOperation,
    /// The extra arguments passed.
    pub args: Vec<String>,
}

impl GitOpResult {
    /// The primary output text.
    pub fn output(&self) -> String {
        self.result.output()
    }

    /// Whether the operation succeeded.
    pub fn success(&self) -> bool {
        self.result.success
    }
}

/// The error classification outcome for an agent error.
#[derive(Debug, Clone)]
pub struct ErrorClassification {
    /// The classified category.
    pub category: ErrorCategory,
    /// Whether the error is transient (worth a retry).
    pub transient: bool,
    /// Whether the error is a provider-side failure.
    pub provider_side: bool,
    /// The suggested recovery action.
    pub recovery: RecoveryAction,
    /// A normalized error code for telemetry.
    pub code: String,
}

/// The category of an agent error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// A rate-limit or quota error.
    RateLimit,
    /// A timeout error.
    Timeout,
    /// An authentication / authorization error.
    Auth,
    /// A provider-side error (5xx, overloaded, transport).
    Provider,
    /// A tool execution error.
    Tool,
    /// A context-overflow error.
    ContextOverflow,
    /// An input validation error.
    InvalidInput,
    /// An internal engine error.
    Internal,
    /// A state-machine error.
    State,
    /// An unknown error.
    Unknown,
}

/// Classify an [`AgentError`] into a structured [`ErrorClassification`].
///
/// This extends [`Agent::handle_error`] with a deterministic category,
/// transient flag, provider-side flag, and telemetry code.
pub fn classify_error(error: &AgentError) -> ErrorClassification {
    use ErrorCategory::*;
    match error {
        AgentError::Provider(msg) => {
            let lower = msg.to_ascii_lowercase();
            if lower.contains("rate") || lower.contains("429") {
                ErrorClassification {
                    category: RateLimit,
                    transient: true,
                    provider_side: true,
                    recovery: RecoveryAction::Retry { delay_ms: 1000 },
                    code: "RATE_LIMITED".to_string(),
                }
            } else if lower.contains("auth") || lower.contains("401") || lower.contains("403") {
                ErrorClassification {
                    category: Auth,
                    transient: false,
                    provider_side: true,
                    recovery: RecoveryAction::FailOver {
                        message: msg.clone(),
                    },
                    code: "AUTH_FAILED".to_string(),
                }
            } else if lower.contains("timeout") || lower.contains("timed out") {
                ErrorClassification {
                    category: Timeout,
                    transient: true,
                    provider_side: true,
                    recovery: RecoveryAction::Retry { delay_ms: 250 },
                    code: "TIMEOUT".to_string(),
                }
            } else if lower.contains("context") || lower.contains("token")
                || lower.contains("too long") || lower.contains("window")
            {
                ErrorClassification {
                    category: ContextOverflow,
                    transient: false,
                    provider_side: true,
                    recovery: RecoveryAction::FailOver {
                        message: msg.clone(),
                    },
                    code: "CONTEXT_OVERFLOW".to_string(),
                }
            } else if lower.contains("overloaded") || lower.contains("503")
                || lower.contains("502") || lower.contains("network")
            {
                ErrorClassification {
                    category: Provider,
                    transient: true,
                    provider_side: true,
                    recovery: RecoveryAction::Retry { delay_ms: 500 },
                    code: "PROVIDER_TRANSIENT".to_string(),
                }
            } else {
                ErrorClassification {
                    category: Provider,
                    transient: false,
                    provider_side: true,
                    recovery: RecoveryAction::FailOver {
                        message: msg.clone(),
                    },
                    code: "PROVIDER_ERROR".to_string(),
                }
            }
        }
        AgentError::Timeout { seconds: _ } => ErrorClassification {
            category: Timeout,
            transient: true,
            provider_side: false,
            recovery: RecoveryAction::Retry { delay_ms: 500 },
            code: "TIMEOUT".to_string(),
        },
        AgentError::Tool { .. } => ErrorClassification {
            category: Tool,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::Continue,
            code: "TOOL_ERROR".to_string(),
        },
        AgentError::State { .. } => ErrorClassification {
            category: State,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::Stop {
                message: "Invalid state transition".to_string(),
            },
            code: "STATE_ERROR".to_string(),
        },
        AgentError::MaxTurnsReached => ErrorClassification {
            category: Internal,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::Stop {
                message: "Maximum turns reached".to_string(),
            },
            code: "MAX_TURNS".to_string(),
        },
        AgentError::MaxToolCallsReached => ErrorClassification {
            category: Internal,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::Continue,
            code: "MAX_TOOL_CALLS".to_string(),
        },
        AgentError::Io(_) => ErrorClassification {
            category: Internal,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::FailOver {
                message: error.to_string(),
            },
            code: "IO_ERROR".to_string(),
        },
        AgentError::Serialization(_) => ErrorClassification {
            category: Internal,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::FailOver {
                message: error.to_string(),
            },
            code: "SERIALIZATION".to_string(),
        },
        AgentError::InvalidInput(_) => ErrorClassification {
            category: InvalidInput,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::FailOver {
                message: error.to_string(),
            },
            code: "INVALID_INPUT".to_string(),
        },
        AgentError::StopRequested => ErrorClassification {
            category: Unknown,
            transient: false,
            provider_side: false,
            recovery: RecoveryAction::Stop {
                message: "Stop requested".to_string(),
            },
            code: "STOP_REQUESTED".to_string(),
        },
    }
}

/// A structured result for a tool-call execution within the agent loop.
#[derive(Debug, Clone)]
pub struct ToolRoundResult {
    /// The decision after this round.
    pub decision: ToolCallDecision,
    /// The tool calls executed in this round.
    pub calls: Vec<ToolCall>,
    /// The results produced.
    pub results: Vec<ToolResult>,
    /// The tool round number.
    pub round: u32,
    /// The total number of tool calls across all rounds so far.
    pub total_calls: u32,
}

impl Agent {
    /// Run the explicit tool-call loop with retry/abort/continue decisions.
    ///
    /// This is the full state-machine loop mirroring the Python
    /// `Agent._turn_generator`, with per-round tool-call decisions:
    ///
    /// * `Continue` — the result is appended and the provider is called again.
    /// * `Retry` — a transient tool failure is retried with backoff.
    /// * `Abort` — an unrecoverable tool error aborts the turn.
    /// * `StopLoop` — the budget is exhausted and the turn ends.
    ///
    /// Returns the final messages and the resulting state.
    #[instrument(skip(self), fields(agent_id = %self.id))]
    pub async fn run_turn_loop(&mut self, messages: Vec<Message>) -> TurnOutcome {
        let start = Instant::now();
        self.state = AgentState::Thinking;

        let mut working = self.conversation.clone();
        working.extend(messages);

        let budget = ToolCallBudget::new(
            self.config.max_turns.max(1),
            self.config.max_tool_calls_per_turn,
            (self.config.max_turns * self.config.max_tool_calls_per_turn).max(1),
        );
        let mut round = 0u32;
        let mut total_calls = 0u32;

        loop {
            // Context-window management before generation.
            if self.should_compact() {
                self.state = AgentState::Compacting;
                if self.compact_history().is_some() {
                    working = self.conversation.clone();
                }
            }

            // Generate the next response.
            self.state = AgentState::Thinking;
            let response = match self.generator.generate(&working).await {
                Ok(response) => response,
                Err(e) => {
                    let message = format!("Provider error: {e}");
                    self.state = AgentState::Error(e.to_string());
                    return TurnOutcome::Error {
                        message,
                        messages: working.clone(),
                        usage: self.usage.to_usage(),
                    };
                }
            };

            // Track usage for this round.
            let input_tokens = estimate_message_tokens(&working);
            let output_tokens = estimate_message_tokens(&response);
            let model = self.generator.model_name().to_string();
            let provider = self.generator.provider_name().to_string();
            self.track_usage(UsageEvent::new(model, provider, input_tokens, output_tokens));

            let calls = crate::turn_control::pending_tool_calls(&response);

            // Final text surface: no tool calls.
            if calls.is_empty() {
                working.extend(response);
                self.conversation = working;
                self.state = AgentState::Completed;
                self.turn_count += 1;
                let duration_ms = start.elapsed().as_millis() as u64;
                return TurnOutcome::Complete {
                    messages: self.conversation.clone(),
                    usage: self.usage.to_usage(),
                    duration_ms,
                };
            }

            // Round budget check.
            if budget.round_exhausted(round) {
                warn!(
                    agent_id = %self.id,
                    round = round,
                    max_rounds = budget.max_rounds,
                    "tool round budget exhausted"
                );
                working.extend(response);
                self.conversation = working;
                self.state = AgentState::Completed;
                self.turn_count += 1;
                let duration_ms = start.elapsed().as_millis() as u64;
                return TurnOutcome::Complete {
                    messages: self.conversation.clone(),
                    usage: self.usage.to_usage(),
                    duration_ms,
                };
            }

            // Execute the tool calls for this round with decision handling.
            self.state = AgentState::WaitingForTool;
            let turn_ctx = self.create_turn_context(working.clone());
            let round_result = self
                .execute_tool_round(&turn_ctx, &calls, round, total_calls, &budget)
                .await;

            match round_result.decision {
                ToolCallDecision::Abort { message } => {
                    self.state = AgentState::Error(message.clone());
                    return TurnOutcome::Error {
                        message,
                        messages: working.clone(),
                        usage: self.usage.to_usage(),
                    };
                }
                ToolCallDecision::StopLoop { .. } => {
                    working.extend(response);
                    working.push(self.process_tool_results(round_result.results));
                    self.conversation = working;
                    self.state = AgentState::Completed;
                    self.turn_count += 1;
                    let duration_ms = start.elapsed().as_millis() as u64;
                    return TurnOutcome::Complete {
                        messages: self.conversation.clone(),
                        usage: self.usage.to_usage(),
                        duration_ms,
                    };
                }
                ToolCallDecision::Continue | ToolCallDecision::Retry { .. } => {
                    working.extend(response);
                    working.push(self.process_tool_results(round_result.results));
                    self.conversation = working.clone();
                }
            }

            round += 1;
            total_calls += round_result.calls.len() as u32;
        }
    }

    /// Execute a single round of tool calls with decision handling.
    ///
    /// Transient failures (timeouts, rate limits) are retried up to the
    /// configured limit; unrecoverable failures abort the round.
    pub async fn execute_tool_round(
        &self,
        context: &TurnContext,
        calls: &[ToolCall],
        round: u32,
        total_calls: u32,
        budget: &ToolCallBudget,
    ) -> ToolRoundResult {
        if calls.is_empty() {
            return ToolRoundResult {
                decision: ToolCallDecision::StopLoop {
                    reason: "no tool calls".to_string(),
                },
                calls: Vec::new(),
                results: Vec::new(),
                round,
                total_calls,
            };
        }

        // Total-call budget check.
        if budget.total_exhausted(total_calls + calls.len() as u32) {
            return ToolRoundResult {
                decision: ToolCallDecision::StopLoop {
                    reason: "total tool call budget exhausted".to_string(),
                },
                calls: calls.to_vec(),
                results: Vec::new(),
                round,
                total_calls,
            };
        }

        let max_calls = context.max_tool_calls_per_turn as usize;
        let mut results: Vec<ToolResult> = Vec::with_capacity(calls.len());
        let mut abort: Option<String> = None;

        for (i, call) in calls.iter().enumerate() {
            if i >= max_calls {
                results.push(ToolResult::error(
                    &call.id,
                    format!(
                        "Max tool calls per round ({}) exceeded",
                        context.max_tool_calls_per_turn
                    ),
                ));
                continue;
            }

            // Execute with retry for transient failures.
            let result = self.execute_tool_call_with_retry(call).await;
            if result.is_error {
                // Classify the error message for a decision.
                let lower = result.content.to_ascii_lowercase();
                if lower.contains("timeout") || lower.contains("rate limit") {
                    // Transient: already retried; surface the error result.
                    results.push(result);
                } else if lower.contains("permission") || lower.contains("disabled") {
                    // Unrecoverable within this tool.
                    results.push(result);
                } else if lower.contains("no tool executor") {
                    abort = Some(result.content.clone());
                    results.push(result);
                    break;
                } else {
                    results.push(result);
                }
            } else {
                results.push(result);
            }
        }

        let decision = if let Some(message) = abort {
            ToolCallDecision::Abort { message }
        } else {
            ToolCallDecision::Continue
        };

        ToolRoundResult {
            decision,
            calls: calls.to_vec(),
            results,
            round,
            total_calls,
        }
    }

    /// Execute a tool call with automatic retry for transient failures.
    ///
    /// The retry budget is one immediate retry for timeouts and rate limits,
    /// matching the Python loop's transient-error handling.
    pub async fn execute_tool_call_with_retry(&self, call: &ToolCall) -> ToolResult {
        let first = self.execute_tool_call(call).await;
        if !first.is_error {
            return first;
        }
        let lower = first.content.to_ascii_lowercase();
        if lower.contains("timeout") || lower.contains("timed out") || lower.contains("rate limit") {
            tokio::time::sleep(Duration::from_millis(250)).await;
            let retry = self.execute_tool_call(call).await;
            if !retry.is_error {
                return retry;
            }
            // Return the retry error, annotated with the retry count.
            return ToolResult::error(&call.id, format!("{} (after 1 retry)", retry.content));
        }
        first
    }

    /// Git operation helpers — the 13 subprocess git operations from the
    /// Python backend, plus their higher-level wrappers.
    ///
    /// These wrap [`Agent::git_operation`] with the exact argument shapes the
    /// Python tools use, so the agent exposes the same surface.

    /// `git status --short` in the workspace.
    pub async fn git_status(&self) -> Result<GitOpResult> {
        self.git_op(GitOperation::Status, &[]).await
    }

    /// `git diff` (optionally limited to a path).
    pub async fn git_diff(&self, path: Option<&str>) -> Result<GitOpResult> {
        let args = path.map(|p| vec![p.to_string()]).unwrap_or_default();
        self.git_op(GitOperation::Diff, &args).await
    }

    /// `git add <paths...>`.
    pub async fn git_add(&self, paths: &[String]) -> Result<GitOpResult> {
        self.git_op(GitOperation::Add, paths).await
    }

    /// `git commit -m <message>`.
    pub async fn git_commit(&self, message: &str) -> Result<GitOpResult> {
        self.git_op(GitOperation::Commit, &["-m".to_string(), message.to_string()])
            .await
    }

    /// `git push` (optionally with remote and branch).
    pub async fn git_push(&self, remote: Option<&str>, branch: Option<&str>) -> Result<GitOpResult> {
        let mut args = Vec::new();
        if let Some(r) = remote {
            args.push(r.to_string());
        }
        if let Some(b) = branch {
            args.push(b.to_string());
        }
        self.git_op(GitOperation::Push, &args).await
    }

    /// `git pull` (optionally with remote and branch).
    pub async fn git_pull(&self, remote: Option<&str>, branch: Option<&str>) -> Result<GitOpResult> {
        let mut args = Vec::new();
        if let Some(r) = remote {
            args.push(r.to_string());
        }
        if let Some(b) = branch {
            args.push(b.to_string());
        }
        self.git_op(GitOperation::Pull, &args).await
    }

    /// `git log --oneline -n <limit>`.
    pub async fn git_log(&self, limit: Option<u32>) -> Result<GitOpResult> {
        let args = limit.map(|n| vec!["-n".to_string(), n.to_string()]).unwrap_or_default();
        self.git_op(GitOperation::Log, &args).await
    }

    /// `git clone <url> [dir]`.
    pub async fn git_clone(&self, url: &str, dir: Option<&str>) -> Result<GitOpResult> {
        let mut args = vec![url.to_string()];
        if let Some(d) = dir {
            args.push(d.to_string());
        }
        self.git_op(GitOperation::Clone, &args).await
    }

    /// `git checkout <branch-or-commit>`.
    pub async fn git_checkout(&self, target: &str) -> Result<GitOpResult> {
        self.git_op(GitOperation::Checkout, &[target.to_string()]).await
    }

    /// `git branch` listing.
    pub async fn git_branch(&self) -> Result<GitOpResult> {
        self.git_op(GitOperation::Branch, &[]).await
    }

    /// `git remote -v` listing.
    pub async fn git_remote(&self) -> Result<GitOpResult> {
        self.git_op(GitOperation::Remote, &[]).await
    }

    /// `git tag` listing.
    pub async fn git_tag(&self) -> Result<GitOpResult> {
        self.git_op(GitOperation::Tag, &[]).await
    }

    /// `git show <ref>`.
    pub async fn git_show(&self, reference: &str) -> Result<GitOpResult> {
        self.git_op(GitOperation::Show, &[reference.to_string()]).await
    }

    /// `git merge <branch>`.
    pub async fn git_merge(&self, branch: &str) -> Result<GitOpResult> {
        self.git_op(GitOperation::Merge, &[branch.to_string()]).await
    }

    /// Execute a git operation and wrap it with the operation metadata.
    pub async fn git_op(&self, op: GitOperation, args: &[String]) -> Result<GitOpResult> {
        let result = self.git_operation(op, args).await?;
        Ok(GitOpResult {
            result,
            operation: op,
            args: args.to_vec(),
        })
    }

    /// Restore the agent from an interrupted state.
    ///
    /// Resets the conversation to the last known-good checkpoint and moves
    /// the agent back to `Idle` so a new turn can begin.
    pub fn recover_from_interruption(&mut self) {
        if self.state == AgentState::Interrupted {
            let _ = self.repair_history();
            self.state = AgentState::Idle;
            debug!(agent_id = %self.id, "agent recovered from interruption");
        }
    }

    /// Snapshot the agent's current state for crash recovery.
    ///
    /// The snapshot captures the conversation, state, usage, and turn count.
    /// [`Agent::restore`] can reconstruct the agent from it after a crash.
    pub fn snapshot(&self) -> AgentSnapshot {
        AgentSnapshot {
            id: self.id.clone(),
            name: self.name.clone(),
            state: self.state.clone(),
            conversation: self.conversation.clone(),
            turn_count: self.turn_count,
            usage: self.usage.clone(),
            created_at: self.created_at,
            last_error: self.last_error.clone(),
        }
    }

    /// Restore agent state from a snapshot.
    pub fn restore(&mut self, snapshot: AgentSnapshot) {
        self.id = snapshot.id;
        self.name = snapshot.name;
        self.state = snapshot.state;
        self.conversation = snapshot.conversation;
        self.turn_count = snapshot.turn_count;
        self.usage = snapshot.usage;
        self.created_at = snapshot.created_at;
        self.last_error = snapshot.last_error;
        debug!(agent_id = %self.id, "agent state restored from snapshot");
    }

    /// Advance the agent one step through the state machine, returning the
    /// new state.
    ///
    /// This is a convenience for orchestrators that drive the machine
    /// manually: it applies the transition table and records the change.
    pub fn step(&mut self, next: AgentState) -> Result<AgentState> {
        self.transition(next)?;
        Ok(self.state.clone())
    }

    /// The phase of the current turn (derived from the agent state).
    pub fn turn_phase(&self) -> TurnPhase {
        match &self.state {
            AgentState::Initializing | AgentState::Idle => TurnPhase::NotStarted,
            AgentState::Processing => TurnPhase::Pipeline,
            AgentState::Thinking => TurnPhase::Generating,
            AgentState::WaitingForTool => TurnPhase::ToolExecution,
            AgentState::Compacting => TurnPhase::Compaction,
            AgentState::Completed => TurnPhase::Completed,
            AgentState::Error(_) => TurnPhase::Failed,
            AgentState::Stopped => TurnPhase::Failed,
            AgentState::Paused | AgentState::WaitingForUser => TurnPhase::NotStarted,
            AgentState::Interrupted | AgentState::Retrying { .. } => TurnPhase::Generating,
        }
    }

    /// Register a background process with this agent's process manager.
    pub fn register_background_process(&mut self, process: Arc<BackgroundProcess>) {
        self.bg_processes.register(process);
    }

    /// Get the agent's background process manager.
    pub fn background_processes(&self) -> &BackgroundProcessManager {
        &self.bg_processes
    }

    /// Stop all background processes spawned by this agent.
    pub async fn stop_background_processes(&mut self) -> usize {
        self.bg_processes.stop_all().await
    }
}

/// A serializable snapshot of an agent's state, for crash recovery.
///
/// The snapshot is in-memory only; use [`AgentSnapshot::to_crash_snapshot`]
/// for a persistence-ready [`crate::recovery::CrashSnapshot`].
#[derive(Debug, Clone)]
pub struct AgentSnapshot {
    /// The agent id.
    pub id: String,
    /// The agent name.
    pub name: String,
    /// The agent's current state.
    pub state: AgentState,
    /// The conversation history.
    pub conversation: Vec<Message>,
    /// The number of turns executed.
    pub turn_count: u64,
    /// The accumulated usage.
    pub usage: UsageStats,
    /// When the agent was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// The last error, if any.
    pub last_error: Option<AgentError>,
}

impl AgentSnapshot {
    /// Convert to a `CrashSnapshot` for persistence.
    pub fn to_crash_snapshot(
        &self,
        session_id: &str,
        tool_round: u32,
        max_tool_rounds: u32,
    ) -> crate::recovery::CrashSnapshot {
        let messages_json =
            serde_json::to_string(&self.conversation).unwrap_or_else(|_| "[]".to_string());
        crate::recovery::CrashSnapshot {
            turn_id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            agent_id: self.id.clone(),
            agent_state: self.state.to_string(),
            model: String::new(),
            provider: String::new(),
            tool_round,
            max_tool_rounds,
            messages_json,
            usage: self.usage.to_usage(),
            created_at_ms: crate::recovery::crash::current_epoch_ms(),
            updated_at_ms: crate::recovery::crash::current_epoch_ms(),
            finalized: false,
        }
    }
}

/// A `ToolExecutor` backed by the tools crate's dispatch engine.
///
/// Bridges [`crate::runtime::ToolExecutor`] to
/// [`opensquilla_tools::DispatchEngine`] so the agent loop can drive the
/// centralized tool system (injection guard, policy chain, sandbox, timeout).
///
/// Only available when the `tools` feature is enabled.
#[cfg(feature = "tools")]
#[derive(Debug)]
pub struct ToolDispatchExecutor {
    engine: Arc<opensquilla_tools::DispatchEngine>,
}

#[cfg(feature = "tools")]
impl ToolDispatchExecutor {
    /// Create an executor around a dispatch engine.
    pub fn new(engine: Arc<opensquilla_tools::DispatchEngine>) -> Self {
        Self { engine }
    }

    /// Build an executor with the built-in tool registry and default policy
    /// chain, rooted at the current working directory.
    pub fn with_default_registry() -> Result<Self> {
        let registry = Arc::new(
            opensquilla_tools::ToolRegistry::with_builtins()
                .map_err(|e| Error::ToolExecution(e.to_string()))?,
        );
        let engine = Arc::new(opensquilla_tools::DispatchEngine::new_with_defaults(
            registry,
        ));
        Ok(Self::new(engine))
    }

    /// Build an executor rooted at a specific working directory.
    pub fn with_default_registry_in(working_dir: PathBuf) -> Result<Self> {
        let registry = Arc::new(
            opensquilla_tools::ToolRegistry::with_builtins_in(working_dir)
                .map_err(|e| Error::ToolExecution(e.to_string()))?,
        );
        let engine = Arc::new(opensquilla_tools::DispatchEngine::new_with_defaults(
            registry,
        ));
        Ok(Self::new(engine))
    }

    /// The underlying dispatch engine.
    pub fn engine(&self) -> &opensquilla_tools::DispatchEngine {
        &self.engine
    }
}

#[cfg(feature = "tools")]
#[async_trait]
impl crate::runtime::ToolExecutor for ToolDispatchExecutor {
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
        let ctx = opensquilla_tools::DispatchContext::new(call.id.clone()).with_timeout(60);
        match self.engine.dispatch(call.clone(), &ctx).await {
            Ok(output) => {
                if output.is_error {
                    Ok(ToolResult::error(&call.id, output.content))
                } else {
                    Ok(ToolResult::success(&call.id, output.content))
                }
            }
            Err(e) => Ok(ToolResult::error(&call.id, e.to_string())),
        }
    }

    fn name(&self) -> &str {
        "tools::dispatch"
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("state", &self.state)
            .field("config", &self.config)
            .field("conversation", &self.conversation)
            .field("turn_count", &self.turn_count)
            .field("usage", &self.usage)
            .finish()
    }
}

/// A collection of agents managed by the runtime.
#[derive(Debug, Default)]
pub struct AgentRegistry {
    /// The agents in this registry, indexed by ID.
    agents: Vec<Agent>,
}

impl AgentRegistry {
    /// Create a new empty agent registry.
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    /// Register a new agent in the registry.
    pub fn register(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    /// Find an agent by its ID.
    pub fn get(&self, id: &str) -> Option<&Agent> {
        self.agents.iter().find(|a| a.id() == id)
    }

    /// Find an agent by its ID, mutably.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Agent> {
        self.agents.iter_mut().find(|a| a.id() == id)
    }

    /// Remove an agent by its ID.
    pub fn remove(&mut self, id: &str) -> Option<Agent> {
        if let Some(pos) = self.agents.iter().position(|a| a.id() == id) {
            Some(self.agents.remove(pos))
        } else {
            None
        }
    }

    /// List all agent IDs in the registry.
    pub fn list_ids(&self) -> Vec<String> {
        self.agents.iter().map(|a| a.id().to_string()).collect()
    }

    /// Get the number of agents in the registry.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Returns true if the registry contains no agents.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Iterate over all agents.
    pub fn iter(&self) -> impl Iterator<Item = &Agent> {
        self.agents.iter()
    }

    /// Find an agent by display name.
    pub fn find_by_name(&self, name: &str) -> Option<&Agent> {
        self.agents.iter().find(|a| a.name() == name)
    }

    /// Take a snapshot of every registered agent.
    pub fn snapshots(&self) -> Vec<AgentSnapshot> {
        self.agents.iter().map(|a| a.snapshot()).collect()
    }
}

/// A placeholder generator that always errors.
///
/// Used by [`AgentRegistry`] recovery paths and tests where an agent is
/// constructed from a snapshot and its generator is replaced later.
#[derive(Debug, Default)]
pub struct PlaceholderGenerator {
    /// The placeholder model name.
    pub model: String,
    /// The placeholder provider name.
    pub provider: String,
}

impl PlaceholderGenerator {
    /// Create a new placeholder generator.
    pub fn new() -> Self {
        Self {
            model: "placeholder".to_string(),
            provider: "placeholder".to_string(),
        }
    }
}

#[async_trait]
impl TurnGenerator for PlaceholderGenerator {
    async fn generate(&self, _messages: &[Message]) -> Result<Vec<Message>> {
        Err(Error::InvalidInput(
            "placeholder generator cannot generate; the agent must be bound to a real generator"
                .to_string(),
        ))
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn provider_name(&self) -> &str {
        &self.provider
    }
}

/// A builder for constructing an [`Agent`] with full configuration.
#[derive(Debug)]
pub struct AgentBuilder {
    /// The agent id.
    id: String,
    /// The display name.
    name: Option<String>,
    /// The configuration.
    config: AgentConfig,
    /// The tool executor.
    tool_executor: Option<Arc<dyn crate::runtime::ToolExecutor>>,
    /// Initial conversation messages.
    conversation: Vec<Message>,
}

impl AgentBuilder {
    /// Create a new builder for the given agent id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            config: AgentConfig::default(),
            tool_executor: None,
            conversation: Vec::new(),
        }
    }

    /// Set the display name.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the full configuration.
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the maximum number of turns.
    pub fn max_turns(mut self, max: u32) -> Self {
        self.config.max_turns = max;
        self
    }

    /// Set the tool-call limit per turn.
    pub fn max_tool_calls(mut self, max: u32) -> Self {
        self.config.max_tool_calls_per_turn = max;
        self
    }

    /// Set the command timeout in seconds.
    pub fn timeout_seconds(mut self, seconds: u64) -> Self {
        self.config.timeout_seconds = seconds;
        self
    }

    /// Enable or disable tool execution.
    pub fn allow_tools(mut self, allowed: bool) -> Self {
        self.config.allow_tool_execution = allowed;
        self
    }

    /// Enable or disable subprocess execution.
    pub fn allow_subprocess(mut self, allowed: bool) -> Self {
        self.config.allow_subprocess = allowed;
        self
    }

    /// Set the workspace directory.
    pub fn workspace(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.workspace_dir = Some(dir.into());
        self
    }

    /// Set the default model.
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Set the default provider.
    pub fn default_provider(mut self, provider: impl Into<String>) -> Self {
        self.config.default_provider = provider.into();
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }

    /// Set the context window size.
    pub fn context_window(mut self, tokens: u64) -> Self {
        self.config.context_window_tokens = tokens;
        self
    }

    /// Attach a tool executor.
    pub fn tool_executor(mut self, executor: Arc<dyn crate::runtime::ToolExecutor>) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    /// Seed the initial conversation.
    pub fn with_conversation(mut self, messages: Vec<Message>) -> Self {
        self.conversation = messages;
        self
    }

    /// Build the agent.
    pub fn build(self, generator: Box<dyn TurnGenerator>) -> Agent {
        let mut agent = Agent::with_config(self.id, generator, self.config);
        if let Some(name) = self.name {
            agent.set_name(name);
        }
        if let Some(executor) = self.tool_executor {
            agent.set_tool_executor(executor);
        }
        agent.conversation = self.conversation;
        agent.initialize();
        agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thinking::ReasoningSupport;
    use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall};
    use serde_json::json;

    /// A generator that always returns a fixed response.
    #[derive(Debug)]
    struct MockGenerator {
        model: String,
        provider: String,
        response: Vec<Message>,
    }

    impl MockGenerator {
        fn text(text: &str) -> Self {
            Self {
                model: "mock-model".into(),
                provider: "mock-provider".into(),
                response: vec![Message::assistant(text)],
            }
        }

        fn tool_call() -> Self {
            Self {
                model: "mock-model".into(),
                provider: "mock-provider".into(),
                response: vec![Message {
                    role: MessageRole::Assistant,
                    content: vec![ContentBlock::ToolUse(ToolCall::new(
                        "tc_1",
                        "read_file",
                        json!({"path": "/tmp"}),
                    ))],
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    tool_result: None,
                }],
            }
        }
    }

    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _messages: &[Message]) -> Result<Vec<Message>> {
            Ok(self.response.clone())
        }

        fn model_name(&self) -> &str {
            &self.model
        }

        fn provider_name(&self) -> &str {
            &self.provider
        }
    }

    /// An executor that records the calls it was asked to run.
    #[derive(Debug, Default)]
    struct MockToolExecutor {
        calls: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl crate::runtime::ToolExecutor for MockToolExecutor {
        async fn execute(&self, call: &ToolCall) -> Result<ToolResult> {
            self.calls.lock().unwrap().push(call.name.clone());
            Ok(ToolResult::success(
                &call.id,
                format!("result of {}", call.name),
            ))
        }
    }

    #[test]
    fn test_agent_new_defaults() {
        let agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        assert_eq!(agent.id(), "a1");
        assert_eq!(agent.name(), "a1");
        assert_eq!(agent.state(), &AgentState::Initializing);
        assert_eq!(agent.config().max_tool_calls_per_turn, 8);
        assert_eq!(agent.turn_count(), 0);
        assert!(agent.get_usage_stats().total_tokens == 0);
    }

    #[test]
    fn test_agent_named() {
        let agent = Agent::new("a1", Box::new(MockGenerator::text("hi"))).named("Assistant");
        assert_eq!(agent.name(), "Assistant");
    }

    #[test]
    fn test_initialize_transitions_to_idle() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.initialize();
        assert_eq!(agent.get_state(), &AgentState::Idle);
    }

    #[test]
    fn test_valid_state_transition() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.initialize();
        assert!(agent.transition(AgentState::Thinking).is_ok());
        assert_eq!(agent.get_state(), &AgentState::Thinking);
    }

    #[test]
    fn test_invalid_state_transition_rejected() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.initialize();
        // Idle -> WaitingForTool is not a legal transition.
        let result = agent.transition(AgentState::WaitingForTool);
        assert!(result.is_err());
        assert_eq!(agent.get_state(), &AgentState::Idle);
    }

    #[test]
    fn test_is_valid_transition_table() {
        use AgentState::*;
        assert!(is_valid_transition(&Idle, &Thinking));
        assert!(is_valid_transition(&Thinking, &WaitingForTool));
        assert!(is_valid_transition(&WaitingForTool, &Thinking));
        assert!(is_valid_transition(&Thinking, &Completed));
        assert!(is_valid_transition(&Idle, &Stopped));
        assert!(!is_valid_transition(&Stopped, &Thinking));
        assert!(!is_valid_transition(&Idle, &WaitingForTool));
    }

    #[test]
    fn test_agent_reset_clears_conversation() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.initialize();
        agent.add_message(Message::user("hello"));
        agent.turn_count = 3;
        assert!(!agent.conversation().is_empty());

        agent.reset();
        assert!(agent.conversation().is_empty());
        assert_eq!(agent.get_state(), &AgentState::Idle);
        assert_eq!(agent.turn_count(), 0);
    }

    #[test]
    fn test_track_usage_accumulates() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.track_usage(UsageEvent::new("model-a", "provider-a", 100, 50));
        agent.track_usage(UsageEvent::new("model-a", "provider-a", 200, 100));

        let stats = agent.get_usage_stats();
        assert_eq!(stats.total_input_tokens, 300);
        assert_eq!(stats.total_output_tokens, 150);
        assert_eq!(stats.total_tokens, 450);
        assert_eq!(stats.turn_count, 2);
        let model_usage = stats.per_model.get("model-a").unwrap();
        assert_eq!(model_usage.calls, 2);
        assert_eq!(model_usage.input_tokens, 300);
    }

    #[test]
    fn test_handle_error_classification() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));

        let retry = agent.handle_error(AgentError::Provider("rate limited".into()));
        assert_eq!(retry, RecoveryAction::Retry { delay_ms: 1000 });

        let timeout = agent.handle_error(AgentError::Timeout { seconds: 30 });
        assert_eq!(timeout, RecoveryAction::Retry { delay_ms: 500 });

        let tool_err = agent.handle_error(AgentError::Tool {
            tool: "shell".into(),
            message: "boom".into(),
        });
        assert_eq!(tool_err, RecoveryAction::Continue);

        let max = agent.handle_error(AgentError::MaxTurnsReached);
        assert!(matches!(max, RecoveryAction::Stop { .. }));

        // The most recent error is recorded on the agent.
        assert!(matches!(
            agent.last_error(),
            Some(AgentError::MaxTurnsReached)
        ));
    }

    #[test]
    fn test_generate_turn_completes_with_text() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("done")));
        agent.initialize();
        let ctx = agent.create_turn_context(vec![Message::user("hello")]);
        let outcome = futures::executor::block_on(agent.generate_turn(&ctx));

        assert!(outcome.is_success());
        assert!(
            outcome
                .messages()
                .iter()
                .any(|m| m.text_content() == "done")
        );
        assert_eq!(agent.turn_count(), 1);
        assert_eq!(agent.get_state(), &AgentState::Completed);
    }

    #[test]
    fn test_generate_turn_executes_tool_calls() {
        let executor = Arc::new(MockToolExecutor::default());
        let mut agent = Agent::new("a1", Box::new(MockGenerator::tool_call()));
        agent.initialize();
        agent.set_tool_executor(executor.clone());

        let ctx = agent.create_turn_context(vec![Message::user("hello")]);
        let outcome = futures::executor::block_on(agent.generate_turn(&ctx));

        assert!(outcome.is_success());
        // The tool result message should be appended after the assistant's
        // tool-call message.
        assert!(
            outcome
                .messages()
                .iter()
                .any(|m| m.role == MessageRole::Tool)
        );
        // The executor ran the tool call.
        assert_eq!(executor.calls.lock().unwrap().as_slice(), ["read_file"]);
    }

    #[test]
    fn test_generate_turn_tool_execution_disabled() {
        let executor = Arc::new(MockToolExecutor::default());
        let mut config = AgentConfig::default();
        config.allow_tool_execution = false;
        let mut agent = Agent::with_config("a1", Box::new(MockGenerator::tool_call()), config);
        agent.initialize();
        agent.set_tool_executor(executor.clone());

        let ctx = agent.create_turn_context(vec![Message::user("hello")]);
        let outcome = futures::executor::block_on(agent.generate_turn(&ctx));

        assert!(outcome.is_success());
        // The tool result message carries a permission-denied error.
        let tool_msg = outcome
            .messages()
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .expect("tool result message present");
        let has_error = tool_msg.content.iter().any(|block| match block {
            ContentBlock::ToolResult(r) => r.is_error && r.content.contains("disabled"),
            _ => false,
        });
        assert!(has_error);
        // The executor was never invoked.
        assert!(executor.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn test_tool_call_loop_respects_budget() {
        let mut ctx = TurnContext::new("t1", Vec::new(), "m", "p");
        ctx.max_tool_calls_per_turn = 2;

        let calls = vec![
            ToolCall::new("c1", "read_file", json!({})),
            ToolCall::new("c2", "write_file", json!({})),
            ToolCall::new("c3", "shell", json!({})),
        ];
        // With no executor configured every call becomes a "no executor"
        // error, but the third call should be the budget error because it
        // exceeds the per-round limit.
        let agent = Agent::with_config(
            "a2",
            Box::new(MockGenerator::text("hi")),
            AgentConfig::default(),
        );
        let results = futures::executor::block_on(agent.tool_call_loop(&ctx, &calls));
        assert_eq!(results.len(), 3);
        assert!(results[2].is_error);
        assert!(results[2].content.contains("exceeded"));
    }

    #[test]
    fn test_execute_tool_call_requires_executor() {
        let agent = Agent::with_config(
            "a1",
            Box::new(MockGenerator::text("hi")),
            AgentConfig::default(),
        );
        let call = ToolCall::new("c1", "read_file", json!({}));
        let result = futures::executor::block_on(agent.execute_tool_call(&call));
        assert!(result.is_error);
        assert!(result.content.contains("No tool executor"));
    }

    #[test]
    fn test_execute_tool_call_subprocess_gated() {
        let executor = Arc::new(MockToolExecutor::default());
        let mut config = AgentConfig::default();
        config.allow_subprocess = false;
        let mut agent = Agent::with_config("a1", Box::new(MockGenerator::text("hi")), config);
        agent.set_tool_executor(executor.clone());

        let call = ToolCall::new("c1", "exec_command", json!({"cmd": "ls"}));
        let result = futures::executor::block_on(agent.execute_tool_call(&call));
        assert!(result.is_error);
        assert!(result.content.contains("disabled"));
        assert!(executor.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn test_process_tool_results_builds_message() {
        let agent = Agent::with_config(
            "a1",
            Box::new(MockGenerator::text("hi")),
            AgentConfig::default(),
        );
        let results = vec![
            ToolResult::success("c1", "out 1"),
            ToolResult::error("c2", "bad thing"),
        ];
        let msg = agent.process_tool_results(results);
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.content.len(), 2);
    }

    #[test]
    fn test_is_subprocess_tool() {
        assert!(is_subprocess_tool("exec_command"));
        assert!(is_subprocess_tool("background_process"));
        assert!(is_subprocess_tool("git_status"));
        assert!(is_subprocess_tool("git_future_tool"));
        assert!(!is_subprocess_tool("read_file"));
        assert!(!is_subprocess_tool("web_search"));
    }

    #[test]
    fn test_git_args_for_operations() {
        assert_eq!(git_args_for(GitOperation::Status), ["status", "--short"]);
        assert_eq!(
            git_args_for(GitOperation::Log),
            ["log", "--oneline", "-n", "20"]
        );
        assert_eq!(git_args_for(GitOperation::Commit), ["commit"]);
        assert_eq!(git_args_for(GitOperation::Remote), ["remote", "-v"]);
    }

    #[test]
    fn test_git_operation_from_str() {
        assert_eq!(
            "status".parse::<GitOperation>().unwrap(),
            GitOperation::Status
        );
        assert_eq!("log".parse::<GitOperation>().unwrap(), GitOperation::Log);
        assert!("bogus".parse::<GitOperation>().is_err());
    }

    #[test]
    fn test_command_result_output_fallback() {
        let ok = CommandResult {
            command: "cmd".into(),
            exit_code: Some(0),
            stdout: "hello".into(),
            stderr: String::new(),
            duration_ms: 1,
            success: true,
        };
        assert_eq!(ok.output(), "hello");

        let err = CommandResult {
            command: "cmd".into(),
            exit_code: Some(1),
            stdout: String::new(),
            stderr: "boom".into(),
            duration_ms: 1,
            success: false,
        };
        assert_eq!(err.output(), "boom");
    }

    #[test]
    fn test_usage_stats_merge() {
        let mut a = UsageStats::new();
        a.record(&UsageEvent::new("m", "p", 10, 5));
        let mut b = UsageStats::new();
        b.record(&UsageEvent::new("m", "p", 20, 10));
        a.merge(&b);
        assert_eq!(a.total_input_tokens, 30);
        assert_eq!(a.total_output_tokens, 15);
        assert_eq!(a.turn_count, 2);
    }

    #[test]
    fn test_agent_registry_crud() {
        let mut registry = AgentRegistry::new();
        registry.register(Agent::new("a1", Box::new(MockGenerator::text("hi"))));
        registry.register(Agent::new("a2", Box::new(MockGenerator::text("hi"))));
        assert_eq!(registry.len(), 2);
        assert!(registry.get("a1").is_some());
        assert!(registry.get("missing").is_none());

        let removed = registry.remove("a1");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.list_ids(), vec!["a2".to_string()]);
    }

    #[test]
    fn test_agent_registry_mut() {
        let mut registry = AgentRegistry::new();
        registry.register(Agent::new("a1", Box::new(MockGenerator::text("hi"))));
        if let Some(agent) = registry.get_mut("a1") {
            agent.set_name("Renamed");
        }
        assert_eq!(registry.get("a1").unwrap().name(), "Renamed");
    }

    #[test]
    fn test_run_turn_completes_with_text() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("done")));
        agent.initialize();
        let outcome = futures::executor::block_on(agent.run_turn(vec![Message::user("hello")]));
        assert!(outcome.is_success());
        assert_eq!(agent.turn_count(), 1);
        assert_eq!(agent.get_state(), &AgentState::Completed);
        assert!(
            agent
                .conversation()
                .iter()
                .any(|m| m.text_content() == "done")
        );
    }

    #[test]
    fn test_run_turn_executes_tool_rounds() {
        let executor = Arc::new(MockToolExecutor::default());
        let mut agent = Agent::new("a1", Box::new(MockGenerator::tool_call()));
        agent.initialize();
        agent.set_tool_executor(executor.clone());
        let outcome = futures::executor::block_on(agent.run_turn(vec![Message::user("hello")]));
        assert!(outcome.is_success());
        assert!(
            agent
                .conversation()
                .iter()
                .any(|m| m.role == MessageRole::Tool)
        );
        let calls = executor.calls.lock().unwrap();
        assert!(!calls.is_empty());
        assert_eq!(calls[0], "read_file");
    }

    #[test]
    fn test_estimate_context_tokens() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message::user("01234567890123456789")); // 20 chars -> 5 tokens
        assert_eq!(agent.estimate_context_tokens(), 5);
        assert_eq!(agent.context_window_tokens(), 128_000);
        assert!(agent.within_context_window());
    }

    #[test]
    fn test_should_compact_threshold() {
        let mut config = AgentConfig::default();
        config.context_window_tokens = 100;
        let mut agent = Agent::with_config("a1", Box::new(MockGenerator::text("hi")), config);
        for i in 0..10 {
            agent.add_message(Message::user(format!(
                "a fairly long message body number {i}"
            )));
        }
        assert!(agent.should_compact());
    }

    #[test]
    fn test_compact_history_preserves_system() {
        let mut config = AgentConfig::default();
        config.context_window_tokens = 200;
        let mut agent = Agent::with_config("a1", Box::new(MockGenerator::text("hi")), config);
        agent.add_message(Message::system("instructions"));
        for i in 0..200 {
            agent.add_message(Message::user(format!("message {i}")));
        }
        assert!(agent.should_compact());
        let before = agent.conversation().len();
        let strategy = agent.compact_history();
        assert!(strategy.is_some());
        assert!(agent.conversation().len() < before);
        assert_eq!(agent.conversation()[0].role, MessageRole::System);
    }

    #[test]
    fn test_trim_history_to_budget() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message::system("instructions"));
        for i in 0..50 {
            agent.add_message(Message::user(format!("body {i}")));
        }
        let dropped = agent.trim_history_to_budget(50);
        assert!(dropped > 0);
        // The system message is preserved.
        assert!(
            agent
                .conversation()
                .iter()
                .any(|m| m.role == MessageRole::System)
        );
    }

    #[test]
    fn test_extract_thinking() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message {
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Reasoning("think step by step".into()),
                ContentBlock::Text("answer".into()),
            ],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        });
        let thinking = agent.extract_thinking();
        assert_eq!(thinking, vec!["think step by step".to_string()]);
    }

    #[test]
    fn test_strip_reasoning_unsupported() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message {
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Reasoning("think".into()),
                ContentBlock::Text("answer".into()),
            ],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        });
        let stripped = agent.strip_reasoning(ReasoningSupport::Unsupported);
        assert_eq!(stripped, 1);
        assert_eq!(agent.extract_thinking().len(), 0);
    }

    #[test]
    fn test_deduplicate_history() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message::user("hello"));
        agent.add_message(Message::user("hello"));
        agent.add_message(Message::assistant("world"));
        let removed = agent.deduplicate_history();
        assert_eq!(removed, 1);
        assert_eq!(agent.conversation().len(), 2);
    }

    #[test]
    fn test_repair_history_drops_orphan() {
        let mut agent = Agent::new("a1", Box::new(MockGenerator::text("hi")));
        agent.add_message(Message {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult::success("gone", "ok"))],
            name: Some("shell".into()),
            tool_call_id: Some("gone".into()),
            tool_calls: None,
            tool_result: None,
        });
        let outcome = agent.repair_history();
        assert_eq!(outcome.removed_results, 1);
        assert!(agent.conversation().is_empty());
    }

    #[test]
    fn test_git_args_for_new_operations() {
        assert_eq!(git_args_for(GitOperation::Init), ["init"]);
        assert_eq!(git_args_for(GitOperation::Reset), ["reset"]);
        assert_eq!(git_args_for(GitOperation::Fetch), ["fetch"]);
        assert_eq!(git_args_for(GitOperation::Stash), ["stash"]);
        assert_eq!(git_args_for(GitOperation::Tag), ["tag"]);
    }

    #[test]
    fn test_git_operation_from_str_new() {
        assert_eq!("init".parse::<GitOperation>().unwrap(), GitOperation::Init);
        assert_eq!(
            "merge".parse::<GitOperation>().unwrap(),
            GitOperation::Merge
        );
        assert_eq!("show".parse::<GitOperation>().unwrap(), GitOperation::Show);
    }
}
