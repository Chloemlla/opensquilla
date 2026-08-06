//! Shell command execution tools: exec_command, background_process.
//!
//! Executes shell commands via `tokio::process::Command` with configurable
//! timeout, environment variables, working directory, and output capture.
//! The `exec_command` tool runs a command synchronously and returns its output,
//! while `background_process` starts a long-running process and manages its lifecycle.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::time::Duration;

/// A handle to a background process managed by the background_process tool.
///
/// The process's stdout and stderr are drained into shared buffers by a
/// dedicated tokio task so that `get_output` returns the actual process
/// output rather than an empty string (the audit bug). A completion handle
/// records the exit status once the process and its drain task finish.
pub struct BackgroundProcess {
    /// The command that was started.
    command: String,
    /// The process ID on the system.
    pub pid: Option<u32>,
    /// The accumulated stdout so far (shared with the drain task).
    pub stdout: Arc<Mutex<String>>,
    /// The accumulated stderr so far (shared with the drain task).
    pub stderr: Arc<Mutex<String>>,
    /// Whether the process has completed.
    pub completed: bool,
    /// The exit code, if completed.
    pub exit_code: Option<i32>,
    /// The timestamp when the process was started.
    started_at: Instant,
    /// Handle to the drain task that reads stdout/stderr to EOF and records
    /// the exit status. Resolving it finalizes `completed`/`exit_code`.
    drain_handle: Option<JoinHandle<()>>,
    /// The child process handle, kept so `stop` can kill it.
    pub child: Option<Child>,
}

impl BackgroundProcess {
    fn new(command: String, child: Child, pid: u32) -> Self {
        Self {
            command,
            pid: Some(pid),
            stdout: Arc::new(Mutex::new(String::new())),
            stderr: Arc::new(Mutex::new(String::new())),
            completed: false,
            exit_code: None,
            started_at: Instant::now(),
            drain_handle: None,
            child: Some(child),
        }
    }
}

/// Tool for executing shell commands synchronously.
pub struct ExecCommandTool {
    /// Default timeout in seconds.
    timeout_secs: u64,
    /// Denied command patterns.
    denied_commands: Vec<String>,
}

impl ExecCommandTool {
    /// Create a new exec_command tool.
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            denied_commands: vec![
                "sudo".to_string(),
                "mkfs".to_string(),
                "dd if=".to_string(),
                ":(){ :|:& };:".to_string(),
                "> /dev/sda".to_string(),
            ],
        }
    }

    /// Check if a command is denied.
    fn is_command_allowed(&self, command: &str) -> bool {
        !self.denied_commands.iter().any(|d| command.contains(d))
    }

    /// Execute a command and return its output.
    async fn run_command(
        &self,
        command: &str,
        args: &[String],
        env_vars: &HashMap<String, String>,
        working_dir: &str,
        timeout: Option<u64>,
    ) -> ToolResult<ToolOutput> {
        if !self.is_command_allowed(command) {
            return Err(ToolError::permission_denied(format!(
                "Command '{}' is not allowed by security policy",
                command
            )));
        }

        let timeout_secs = timeout.unwrap_or(self.timeout_secs);
        let start = Instant::now();

        let mut cmd = Command::new(if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        });

        if cfg!(target_os = "windows") {
            cmd.arg("/C").arg(command).args(args);
        } else {
            let full = if args.is_empty() {
                command.to_string()
            } else {
                format!("{} {}", command, args.join(" "))
            };
            cmd.arg("-c").arg(&full);
        }

        // Set environment variables.
        for (key, val) in env_vars {
            cmd.env(key, val);
        }

        // Set working directory.
        if !working_dir.is_empty() {
            cmd.current_dir(working_dir);
        }

        cmd.kill_on_drop(true);

        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .map_err(|_| ToolError::timeout(timeout_secs))?;

        let output = result
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to execute command: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        let duration_ms = start.elapsed().as_millis() as u64;

        let data = serde_json::json!({
            "exit_code": exit_code,
            "stdout_length": stdout.len(),
            "stderr_length": stderr.len(),
            "duration_ms": duration_ms,
            "command": command,
        });

        let content = if output.status.success() {
            if stdout.is_empty() {
                format!("Command completed with exit code 0 in {}ms", duration_ms)
            } else {
                stdout
            }
        } else {
            format!(
                "Command failed (exit code {}):\n{}\n{}",
                exit_code, stderr, stdout
            )
        };

        tracing::info!(
            target = "tools",
            command = %command,
            exit_code = exit_code,
            duration_ms = duration_ms,
            "Shell command completed"
        );

        Ok(ToolOutput::success(content)
            .with_data(data)
            .with_mime_type("text/plain"))
    }
}

impl Default for ExecCommandTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait]
impl Tool for ExecCommandTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "exec_command",
                "Execute a shell command and return its output. "
                    + "The command runs synchronously with a configurable timeout. "
                    + "Environment variables and working directory can be specified.",
                HashMap::from([
                    (
                        "command".to_string(),
                        ParameterDefinition::required_string("The shell command to execute"),
                    ),
                    (
                        "args".to_string(),
                        ParameterDefinition::array(
                            "Additional arguments for the command",
                            ParameterDefinition::string("Argument value"),
                        ),
                    ),
                    (
                        "timeout".to_string(),
                        ParameterDefinition::integer("Timeout in seconds (default: 30)")
                            .default(serde_json::json!(30)),
                    ),
                    (
                        "working_dir".to_string(),
                        ParameterDefinition::string("Working directory for the command"),
                    ),
                    (
                        "env_vars".to_string(),
                        ParameterDefinition::string(
                            "Environment variables as JSON object (key-value pairs)",
                        ),
                    ),
                ]),
            )
            .category("shell")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let command = params["command"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'command' parameter"))?;

        let args: Vec<String> = params["args"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let env_vars: HashMap<String, String> = params["env_vars"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        let working_dir = params["working_dir"].as_str().unwrap_or(".");
        let timeout = params["timeout"].as_i64().map(|t| t as u64);

        self.run_command(command, &args, &env_vars, working_dir, timeout)
            .await
    }
}

/// Tool for managing background processes.
pub struct BackgroundProcessTool {
    /// Map of process IDs to background process handles.
    processes: Arc<Mutex<HashMap<String, BackgroundProcess>>>,
}

impl BackgroundProcessTool {
    /// Create a new background_process tool.
    pub fn new() -> Self {
        Self {
            processes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start a background process.
    async fn start_process(
        &self,
        id: &str,
        command: &str,
        args: &[String],
        env_vars: &HashMap<String, String>,
        working_dir: &str,
    ) -> ToolResult<ToolOutput> {
        let mut cmd = Command::new(if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        });

        if cfg!(target_os = "windows") {
            cmd.arg("/C").arg(command).args(args);
        } else {
            let full = if args.is_empty() {
                command.to_string()
            } else {
                format!("{} {}", command, args.join(" "))
            };
            cmd.arg("-c").arg(&full);
        }

        for (key, val) in env_vars {
            cmd.env(key, val);
        }
        if !working_dir.is_empty() {
            cmd.current_dir(working_dir);
        }

        // Pipe stdout/stderr so we can capture the process's real output.
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        // Spawn without waiting for the process to finish.
        let mut child = cmd
            .spawn()
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to spawn process: {}", e)))?;

        let pid = child
            .id()
            .ok_or_else(|| ToolError::new("IO_ERROR", "Failed to get process ID".to_string()))?;

        let mut process = BackgroundProcess::new(command.to_string(), child, pid);

        // Take the pipes out of the child and spawn a drain task that reads
        // them to EOF and records the exit status. This is the fix for the
        // audit bug where `stdout`/`stderr` were never populated.
        let stdout_buf = process.stdout.clone();
        let stderr_buf = process.stderr.clone();
        let mut stdout = process
            .child
            .as_mut()
            .and_then(|c| c.stdout.take())
            .ok_or_else(|| ToolError::new("IO_ERROR", "Failed to capture stdout".to_string()))?;
        let mut stderr = process
            .child
            .as_mut()
            .and_then(|c| c.stderr.take())
            .ok_or_else(|| ToolError::new("IO_ERROR", "Failed to capture stderr".to_string()))?;

        let drain_handle = tokio::spawn(async move {
            // Drain stdout and stderr concurrently so a pipe filling up on one
            // stream cannot deadlock the other.
            let mut stdout_buf_guard = stdout_buf.clone();
            let mut stderr_buf_guard = stderr_buf.clone();
            let out_task = tokio::spawn(async move {
                let mut data = Vec::new();
                if stdout.read_to_end(&mut data).await.is_ok() {
                    if let Ok(mut guard) = stdout_buf_guard.lock().await {
                        guard.push_str(&String::from_utf8_lossy(&data));
                    }
                }
            });
            let err_task = tokio::spawn(async move {
                let mut data = Vec::new();
                if stderr.read_to_end(&mut data).await.is_ok() {
                    if let Ok(mut guard) = stderr_buf_guard.lock().await {
                        guard.push_str(&String::from_utf8_lossy(&data));
                    }
                }
            });
            let _ = tokio::join!(out_task, err_task);
        });

        process.drain_handle = Some(drain_handle);

        let mut processes = self.processes.lock().await;
        processes.insert(id.to_string(), process);

        let data = serde_json::json!({
            "pid": pid,
            "process_id": id,
            "command": command,
        });

        Ok(ToolOutput::success(format!(
            "Started background process '{}' with PID {}",
            id, pid
        ))
        .with_data(data))
    }

    /// Check the status of a background process.
    async fn check_process(&self, id: &str) -> ToolResult<ToolOutput> {
        let mut processes = self.processes.lock().await;
        let process = processes.get_mut(id).ok_or_else(|| {
            ToolError::not_found(format!("Background process '{}' not found", id))
        })?;

        // Try to reap the process if it has exited.
        if !process.completed {
            if let Some(ref mut child) = process.child {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        process.exit_code = status.code();
                        process.child = None;
                        // Give the drain task a moment to flush the remaining
                        // buffered output after the process exit; pipes are at
                        // EOF so this should return quickly.
                        if let Some(handle) = process.drain_handle.take() {
                            let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
                        }
                        process.completed = true;
                    }
                    Ok(None) => {} // Still running.
                    Err(e) => {
                        tracing::warn!("Error checking process status: {}", e);
                    }
                }
            } else if let Some(handle) = process.drain_handle.take() {
                // The child was already reaped but the drain task has not been
                // joined yet.
                let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
                process.completed = true;
            }
        }

        let elapsed = process.started_at.elapsed().as_secs();

        let data = serde_json::json!({
            "process_id": id,
            "pid": process.pid,
            "command": process.command,
            "running": !process.completed,
            "completed": process.completed,
            "exit_code": process.exit_code,
            "elapsed_secs": elapsed,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }

    /// Get the output of a background process.
    ///
    /// Returns the accumulated stdout/stderr that the drain task has collected
    /// so far. For a still-running process, this returns whatever output has
    /// been produced up to now; for a completed process, the full output.
    async fn get_output(&self, id: &str) -> ToolResult<ToolOutput> {
        let mut processes = self.processes.lock().await;
        let process = processes.get_mut(id).ok_or_else(|| {
            ToolError::not_found(format!("Background process '{}' not found", id))
        })?;

        // Reap first so a process that just exited is finalized before we read.
        if !process.completed {
            if let Some(ref mut child) = process.child {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        process.exit_code = status.code();
                        process.child = None;
                        if let Some(handle) = process.drain_handle.take() {
                            let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
                        }
                        process.completed = true;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        tracing::warn!("Error checking process status: {}", e);
                    }
                }
            }
        }

        let stdout = process.stdout.lock().await.clone();
        let stderr = process.stderr.lock().await.clone();

        let data = serde_json::json!({
            "process_id": id,
            "exit_code": process.exit_code,
            "completed": process.completed,
            "stdout": stdout,
            "stderr": stderr,
        });

        Ok(ToolOutput::success(stdout).with_data(data))
    }

    /// Stop a background process.
    async fn stop_process(&self, id: &str) -> ToolResult<ToolOutput> {
        let mut processes = self.processes.lock().await;
        let process = processes.get_mut(id).ok_or_else(|| {
            ToolError::not_found(format!("Background process '{}' not found", id))
        })?;

        if let Some(ref mut child) = process.child {
            let _ = child.kill().await;
            process.child = None;
        }

        // Wait for the drain task to finish flushing after the kill.
        if let Some(handle) = process.drain_handle.take() {
            let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
        }
        process.completed = true;
        if process.exit_code.is_none() {
            process.exit_code = Some(-1);
        }

        let data = serde_json::json!({
            "process_id": id,
            "stopped": true,
        });

        Ok(ToolOutput::success(format!("Stopped background process '{}'", id)).with_data(data))
    }

    /// List all background processes.
    async fn list_processes(&self) -> ToolResult<ToolOutput> {
        let processes = self.processes.lock().await;
        let list: Vec<serde_json::Value> = processes
            .iter()
            .map(|(id, p)| {
                serde_json::json!({
                    "process_id": id,
                    "pid": p.pid,
                    "command": p.command,
                    "running": !p.completed,
                    "completed": p.completed,
                    "exit_code": p.exit_code,
                    "elapsed_secs": p.started_at.elapsed().as_secs(),
                })
            })
            .collect();

        let data = serde_json::json!({
            "processes": list,
            "count": list.len(),
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }
}

impl Default for BackgroundProcessTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BackgroundProcessTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "background_process",
                "Manage long-running background processes. Supports start, status, output, stop, and list operations.",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The operation to perform")
                            .enum_values(vec![
                                "start".to_string(),
                                "status".to_string(),
                                "output".to_string(),
                                "stop".to_string(),
                                "list".to_string(),
                            ]),
                    ),
                    (
                        "process_id".to_string(),
                        ParameterDefinition::string("Unique identifier for the process (required for start, status, output, stop)"),
                    ),
                    (
                        "command".to_string(),
                        ParameterDefinition::string("The command to run (required for start)"),
                    ),
                    (
                        "args".to_string(),
                        ParameterDefinition::array("Additional arguments", ParameterDefinition::string("arg")),
                    ),
                    (
                        "working_dir".to_string(),
                        ParameterDefinition::string("Working directory for the command"),
                    ),
                ]),
            )
            .category("shell")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?;

        match operation {
            "start" => {
                let id = params["process_id"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'process_id' for start operation")
                })?;
                let command = params["command"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'command' for start operation")
                })?;
                let args: Vec<String> = params["args"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let env_vars = HashMap::new();
                let working_dir = params["working_dir"].as_str().unwrap_or(".");
                self.start_process(id, command, &args, &env_vars, working_dir)
                    .await
            }
            "status" => {
                let id = params["process_id"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'process_id' for status operation")
                })?;
                self.check_process(id).await
            }
            "output" => {
                let id = params["process_id"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'process_id' for output operation")
                })?;
                self.get_output(id).await
            }
            "stop" => {
                let id = params["process_id"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'process_id' for stop operation")
                })?;
                self.stop_process(id).await
            }
            "list" => self.list_processes().await,
            other => Err(ToolError::invalid_args(format!(
                "Unknown operation: {}",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------------

/// A signal that can be sent to a process.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Signal {
    /// Terminate (SIGTERM / taskkill without /F).
    Term,
    /// Force kill (SIGKILL / taskkill /F).
    Kill,
    /// Interrupt (SIGINT / Ctrl+C).
    Interrupt,
    /// Hang up (SIGHUP).
    Hangup,
    /// User-defined signal 1 (SIGUSR1, Unix only).
    User1,
    /// User-defined signal 2 (SIGUSR2, Unix only).
    User2,
}

impl Signal {
    /// Get the Unix signal number for this signal.
    pub fn unix_number(&self) -> Option<i32> {
        match self {
            Signal::Term => Some(15),
            Signal::Kill => Some(9),
            Signal::Interrupt => Some(2),
            Signal::Hangup => Some(1),
            Signal::User1 => Some(10),
            Signal::User2 => Some(12),
        }
    }

    /// Whether this signal is supported on the current platform.
    pub fn is_supported(&self) -> bool {
        match self {
            Signal::Term | Signal::Kill | Signal::Interrupt => true,
            Signal::Hangup | Signal::User1 | Signal::User2 => !cfg!(target_os = "windows"),
        }
    }
}

/// A global process registry that tracks all background processes across
/// multiple tool instances. This enables process management across sessions.
pub struct ProcessRegistry {
    /// Map of process IDs to process handles.
    processes: Arc<Mutex<HashMap<String, RegisteredProcess>>>,
}

/// A process registered in the global registry.
#[derive(Debug)]
struct RegisteredProcess {
    /// The process ID on the system.
    pid: u32,
    /// The command that was started.
    command: String,
    /// When the process was started (epoch seconds).
    started_at: u64,
    /// The working directory.
    working_dir: String,
    /// Tags associated with the process.
    tags: Vec<String>,
    /// Accumulated stdout.
    stdout: Arc<Mutex<String>>,
    /// Accumulated stderr.
    stderr: Arc<Mutex<String>>,
    /// Whether the process has completed.
    completed: bool,
    /// The exit code, if completed.
    exit_code: Option<i32>,
    /// The child handle (for killing).
    child: Option<Child>,
    /// Timeout in seconds (0 = no timeout).
    timeout_secs: u64,
}

impl Default for ProcessRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessRegistry {
    /// Create a new empty process registry.
    pub fn new() -> Self {
        Self {
            processes: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a process in the registry.
    pub async fn register(
        &self,
        id: String,
        pid: u32,
        command: String,
        working_dir: String,
        tags: Vec<String>,
        timeout_secs: u64,
        child: Child,
        stdout: Arc<Mutex<String>>,
        stderr: Arc<Mutex<String>>,
    ) {
        let started_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let process = RegisteredProcess {
            pid,
            command,
            started_at,
            working_dir,
            tags,
            stdout,
            stderr,
            completed: false,
            exit_code: None,
            child: Some(child),
            timeout_secs,
        };

        self.processes.lock().await.insert(id, process);
    }

    /// List all registered processes.
    pub async fn list(&self) -> Vec<RegisteredProcessInfo> {
        let processes = self.processes.lock().await;
        processes
            .iter()
            .map(|(id, p)| RegisteredProcessInfo {
                id: id.clone(),
                pid: p.pid,
                command: p.command.clone(),
                started_at: p.started_at,
                working_dir: p.working_dir.clone(),
                tags: p.tags.clone(),
                completed: p.completed,
                exit_code: p.exit_code,
                timeout_secs: p.timeout_secs,
            })
            .collect()
    }

    /// Remove a process from the registry.
    pub async fn remove(&self, id: &str) -> bool {
        self.processes.lock().await.remove(id).is_some()
    }

    /// Send a signal to a registered process.
    pub async fn signal(&self, id: &str, signal: Signal) -> ToolResult<()> {
        let mut processes = self.processes.lock().await;
        let process = processes.get_mut(id).ok_or_else(|| {
            ToolError::not_found(format!("Process '{}' not found in registry", id))
        })?;

        if process.completed {
            return Err(ToolError::new(
                "PROCESS_COMPLETED",
                format!("Process '{}' has already completed", id),
            ));
        }

        if let Some(ref mut child) = process.child {
            if signal == Signal::Kill {
                let _ = child.kill().await;
            } else if cfg!(target_os = "windows") {
                // On Windows, only Kill is meaningful; Term uses taskkill.
                if signal == Signal::Term {
                    let pid = process.pid.to_string();
                    let _ = tokio::process::Command::new("taskkill")
                        .arg("/PID")
                        .arg(&pid)
                        .output()
                        .await;
                } else {
                    return Err(ToolError::new(
                        "UNSUPPORTED_SIGNAL",
                        format!(
                            "Signal {:?} is not supported on Windows",
                            signal
                        ),
                    ));
                }
            } else if let Some(signum) = signal.unix_number() {
                let pid = process.pid.to_string();
                let flag = format!("-{}", signum);
                let _ = tokio::process::Command::new("kill")
                    .args([flag, pid])
                    .output()
                    .await;
            }
        }
        Ok(())
    }

    /// Get the output of a registered process.
    pub async fn get_output(&self, id: &str) -> ToolResult<(String, String)> {
        let processes = self.processes.lock().await;
        let process = processes.get(id).ok_or_else(|| {
            ToolError::not_found(format!("Process '{}' not found in registry", id))
        })?;
        let stdout = process.stdout.lock().await.clone();
        let stderr = process.stderr.lock().await.clone();
        Ok((stdout, stderr))
    }

    /// Clean up completed processes older than the given age.
    pub async fn cleanup(&self, max_age_secs: u64) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut processes = self.processes.lock().await;
        let to_remove: Vec<String> = processes
            .iter()
            .filter(|(_, p)| {
                p.completed && now.saturating_sub(p.started_at) > max_age_secs
            })
            .map(|(id, _)| id.clone())
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            processes.remove(&id);
        }
        count
    }
}

/// Information about a registered process (for listing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredProcessInfo {
    /// The registry ID.
    pub id: String,
    /// The system PID.
    pub pid: u32,
    /// The command.
    pub command: String,
    /// When the process started (epoch seconds).
    pub started_at: u64,
    /// The working directory.
    pub working_dir: String,
    /// Tags on the process.
    pub tags: Vec<String>,
    /// Whether the process completed.
    pub completed: bool,
    /// The exit code, if completed.
    pub exit_code: Option<i32>,
    /// The timeout in seconds.
    pub timeout_secs: u64,
}

// ---------------------------------------------------------------------------
// Environment variable filtering
// ---------------------------------------------------------------------------

/// Filter environment variables, removing sensitive ones and optionally
/// allowing only a specific set.
#[derive(Debug, Clone)]
pub struct EnvFilter {
    /// If non-empty, only these variables are allowed.
    allowlist: Option<Vec<String>>,
    /// Variables that are always removed.
    denylist: Vec<String>,
    /// Prefixes of variables to remove (e.g., "SECRET_", "API_KEY_").
    deny_prefixes: Vec<String>,
}

impl Default for EnvFilter {
    fn default() -> Self {
        Self {
            allowlist: None,
            denylist: vec![
                "PATH".to_string(),      // PATH is inherited separately
                "PS1".to_string(),
                "LS_COLORS".to_string(),
            ],
            deny_prefixes: vec![
                "SECRET_".to_string(),
                "API_KEY_".to_string(),
                "PRIVATE_KEY".to_string(),
                "TOKEN_".to_string(),
                "PASSWORD_".to_string(),
                "CREDENTIAL_".to_string(),
                "AWS_".to_string(),
                "GITHUB_TOKEN".to_string(),
                "DATABASE_URL".to_string(),
            ],
        }
    }
}

impl EnvFilter {
    /// Create a new environment filter with default deny rules.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an allowlist — only these variables pass through.
    pub fn allow_only(mut self, vars: Vec<String>) -> Self {
        self.allowlist = Some(vars);
        self
    }

    /// Add a variable to the denylist.
    pub fn deny(mut self, var: impl Into<String>) -> Self {
        self.denylist.push(var.into());
        self
    }

    /// Add a prefix to the denylist.
    pub fn deny_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.deny_prefixes.push(prefix.into());
        self
    }

    /// Filter a map of environment variables.
    pub fn filter(&self, env: &HashMap<String, String>) -> HashMap<String, String> {
        env.iter()
            .filter(|(key, _)| self.is_allowed(key))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if a single variable is allowed.
    pub fn is_allowed(&self, key: &str) -> bool {
        // Denylist takes precedence.
        if self.denylist.iter().any(|d| d == key) {
            return false;
        }
        // Check deny prefixes.
        if self.deny_prefixes.iter().any(|p| key.starts_with(p)) {
            return false;
        }
        // Check allowlist if set.
        if let Some(ref allow) = self.allowlist {
            if !allow.iter().any(|a| a == key) {
                return false;
            }
        }
        true
    }

    /// Filter the current process's environment.
    pub fn filter_current(&self) -> HashMap<String, String> {
        let env: HashMap<String, String> = std::env::vars().collect();
        self.filter(&env)
    }
}

// ---------------------------------------------------------------------------
// Process supervisor with timeout enforcement and kill-on-drop
// ---------------------------------------------------------------------------

/// A supervisor that watches a background process and enforces a timeout.
///
/// If the process exceeds its timeout, the supervisor invokes the kill
/// closure. When the supervisor is dropped with `kill_on_drop` enabled, it
/// also invokes the kill closure unless the timeout already fired. The
/// supervisor runs as a separate tokio task and can be cancelled.
pub struct ProcessSupervisor {
    /// The process ID being supervised.
    process_id: String,
    /// The timeout in seconds.
    timeout_secs: u64,
    /// Whether to kill the process when the supervisor is dropped.
    kill_on_drop: bool,
    /// Handle to the supervisor task.
    handle: Option<JoinHandle<()>>,
    /// The kill closure, shared between the timeout task and the Drop impl.
    kill_fn: Option<Arc<std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>>>,
}

impl ProcessSupervisor {
    /// Start supervising a process.
    ///
    /// The `kill_fn` is called if the timeout expires or (when
    /// `kill_on_drop` is true) when the supervisor is dropped.
    pub fn start<F>(process_id: String, timeout_secs: u64, kill_on_drop: bool, kill_fn: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        type SharedKill = std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>;
        let kill_fn: Option<Arc<SharedKill>> = Some(Arc::new(std::sync::Mutex::new(Some(Box::new(kill_fn)))));
        let kill_for_task = kill_fn.clone();

        let pid = process_id.clone();
        let handle = tokio::spawn(async move {
            if timeout_secs > 0 {
                let _ = tokio::time::timeout(
                    Duration::from_secs(timeout_secs),
                    std::future::pending::<()>(),
                )
                .await;
                tracing::warn!(
                    process_id = %pid,
                    timeout_secs = timeout_secs,
                    "Process timed out, killing"
                );
                if let Some(mutex) = kill_for_task.as_ref() {
                    if let Ok(mut guard) = mutex.lock() {
                        if let Some(f) = guard.take() {
                            f();
                        }
                    }
                }
            } else {
                // No timeout; the supervisor just waits forever (until cancelled).
                std::future::pending::<()>().await;
            }
        });

        Self {
            process_id,
            timeout_secs,
            kill_on_drop,
            handle: Some(handle),
            kill_fn,
        }
    }

    /// Cancel the supervisor (stops the timeout enforcement and kill-on-drop).
    pub fn cancel(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        // Release the kill closure so it is never invoked.
        self.kill_fn = None;
    }

    /// Get the process ID being supervised.
    pub fn process_id(&self) -> &str {
        &self.process_id
    }

    /// Get the timeout in seconds.
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }
}

impl Drop for ProcessSupervisor {
    fn drop(&mut self) {
        // Abort the timeout task so it cannot race the kill-on-drop below.
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        // If kill-on-drop is enabled and the kill closure is still present
        // (the timeout never fired), invoke it now.
        if self.kill_on_drop {
            if let Some(mutex) = self.kill_fn.take() {
                if let Ok(mut guard) = mutex.lock() {
                    if let Some(f) = guard.take() {
                        tracing::debug!(
                            process_id = %self.process_id,
                            "Killing supervised process on drop"
                        );
                        f();
                    }
                }
            }
        }
        self.kill_fn = None;
    }
}

// ---------------------------------------------------------------------------
// Enhanced exec tool with env filtering and output capture modes
// ---------------------------------------------------------------------------

/// How to capture command output.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputCapture {
    /// Capture stdout and stderr separately.
    Separate,
    /// Merge stdout and stderr into a single stream.
    Merged,
    /// Capture only stdout.
    StdoutOnly,
    /// Capture only stderr.
    StderrOnly,
    /// Discard all output.
    Discard,
}

impl Default for OutputCapture {
    fn default() -> Self {
        OutputCapture::Separate
    }
}

/// An enhanced shell execution tool with env filtering, output capture modes,
/// and working directory management.
pub struct EnhancedExecTool {
    /// Default timeout in seconds.
    timeout_secs: u64,
    /// Environment filter.
    env_filter: EnvFilter,
    /// Denied command patterns.
    denied_commands: Vec<String>,
    /// Denied command regex patterns.
    denied_patterns: Vec<regex::Regex>,
    /// Maximum output size in bytes.
    max_output_size: u64,
}

impl EnhancedExecTool {
    /// Create a new enhanced exec tool.
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            env_filter: EnvFilter::new(),
            denied_commands: vec![
                "sudo".to_string(),
                "mkfs".to_string(),
                "dd if=".to_string(),
                ":(){ :|:& };:".to_string(),
                "> /dev/sda".to_string(),
                "rm -rf /".to_string(),
                "chmod -R 777 /".to_string(),
            ],
            denied_patterns: vec![
                regex::Regex::new(r"(?i)\bchmod\s+[-R]+\s*777\s+/").unwrap(),
                regex::Regex::new(r"(?i)\brm\s+-rf\s+/(?:\s|$)").unwrap(),
            ],
            max_output_size: 10 * 1024 * 1024, // 10 MB
        }
    }

    /// Set a custom environment filter.
    pub fn with_env_filter(mut self, filter: EnvFilter) -> Self {
        self.env_filter = filter;
        self
    }

    /// Add a denied command pattern.
    pub fn deny_command(mut self, command: impl Into<String>) -> Self {
        self.denied_commands.push(command.into());
        self
    }

    /// Add a denied command regex pattern.
    pub fn deny_pattern(mut self, pattern: &str) -> Self {
        if let Ok(re) = regex::Regex::new(pattern) {
            self.denied_patterns.push(re);
        }
        self
    }

    /// Check if a command is allowed.
    fn is_command_allowed(&self, command: &str) -> bool {
        if self.denied_commands.iter().any(|d| command.contains(d)) {
            return false;
        }
        if self.denied_patterns.iter().any(|p| p.is_match(command)) {
            return false;
        }
        true
    }

    /// Execute a command with enhanced options.
    async fn execute_enhanced(
        &self,
        command: &str,
        args: &[String],
        env_vars: &HashMap<String, String>,
        working_dir: &str,
        timeout: Option<u64>,
        capture: OutputCapture,
    ) -> ToolResult<ToolOutput> {
        if !self.is_command_allowed(command) {
            return Err(ToolError::permission_denied(format!(
                "Command '{}' is not allowed by security policy",
                command
            )));
        }

        // Filter environment variables.
        let filtered_env = self.env_filter.filter(env_vars);

        let timeout_secs = timeout.unwrap_or(self.timeout_secs);
        let start = Instant::now();

        let mut cmd = Command::new(if cfg!(target_os = "windows") {
            "cmd"
        } else {
            "sh"
        });

        if cfg!(target_os = "windows") {
            cmd.arg("/C").arg(command).args(args);
        } else {
            let full = if args.is_empty() {
                command.to_string()
            } else {
                format!("{} {}", command, args.join(" "))
            };
            cmd.arg("-c").arg(&full);
        }

        for (key, val) in &filtered_env {
            cmd.env(key, val);
        }

        if !working_dir.is_empty() {
            cmd.current_dir(working_dir);
        }

        cmd.kill_on_drop(true);

        // Configure output capture.
        match capture {
            OutputCapture::Separate => {
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::piped());
            }
            OutputCapture::Merged => {
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::piped());
            }
            OutputCapture::StdoutOnly => {
                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::null());
            }
            OutputCapture::StderrOnly => {
                cmd.stdout(Stdio::null());
                cmd.stderr(Stdio::piped());
            }
            OutputCapture::Discard => {
                cmd.stdout(Stdio::null());
                cmd.stderr(Stdio::null());
            }
        }

        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output())
            .await
            .map_err(|_| ToolError::timeout(timeout_secs))?
            .map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to execute command: {}", e))
            })?;

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let exit_code = result.status.code().unwrap_or(-1);
        let duration_ms = start.elapsed().as_millis() as u64;

        // Enforce max output size.
        let stdout_truncated = stdout.len() as u64 > self.max_output_size;
        let stderr_truncated = stderr.len() as u64 > self.max_output_size;

        let stdout_final = if stdout_truncated {
            stdout.chars().take(self.max_output_size as usize).collect::<String>()
        } else {
            stdout
        };
        let stderr_final = if stderr_truncated {
            stderr.chars().take(self.max_output_size as usize).collect::<String>()
        } else {
            stderr
        };

        let content = match capture {
            OutputCapture::Separate => {
                if result.status.success() {
                    if stdout_final.is_empty() {
                        format!("Command completed with exit code 0 in {}ms", duration_ms)
                    } else {
                        stdout_final
                    }
                } else {
                    format!(
                        "Command failed (exit code {}):\n{}\n{}",
                        exit_code, stderr_final, stdout_final
                    )
                }
            }
            OutputCapture::Merged => {
                let mut combined = stdout_final;
                combined.push_str(&stderr_final);
                combined
            }
            OutputCapture::StdoutOnly => stdout_final,
            OutputCapture::StderrOnly => stderr_final,
            OutputCapture::Discard => format!("Command completed (exit code {}) in {}ms", exit_code, duration_ms),
        };

        let data = serde_json::json!({
            "exit_code": exit_code,
            "stdout_length": result.stdout.len(),
            "stderr_length": result.stderr.len(),
            "duration_ms": duration_ms,
            "command": command,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "env_vars_filtered": env_vars.len() - filtered_env.len(),
            "capture_mode": match capture {
                OutputCapture::Separate => "separate",
                OutputCapture::Merged => "merged",
                OutputCapture::StdoutOnly => "stdout_only",
                OutputCapture::StderrOnly => "stderr_only",
                OutputCapture::Discard => "discard",
            },
        });

        Ok(ToolOutput::success(content)
            .with_data(data)
            .with_mime_type("text/plain"))
    }
}

impl Default for EnhancedExecTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait]
impl Tool for EnhancedExecTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "exec_command_enhanced",
                "Execute a shell command with enhanced options: environment variable filtering, "
                    + "output capture modes (separate, merged, stdout-only, stderr-only, discard), "
                    + "timeout enforcement, and working directory management. "
                    + "Sensitive environment variables (containing SECRET_, TOKEN_, PASSWORD_, etc.) "
                    + "are automatically filtered out.",
                HashMap::from([
                    (
                        "command".to_string(),
                        ParameterDefinition::required_string("The shell command to execute"),
                    ),
                    (
                        "args".to_string(),
                        ParameterDefinition::array("Additional arguments", ParameterDefinition::string("arg")),
                    ),
                    (
                        "timeout".to_string(),
                        ParameterDefinition::integer("Timeout in seconds")
                            .default(serde_json::json!(30)),
                    ),
                    (
                        "working_dir".to_string(),
                        ParameterDefinition::string("Working directory"),
                    ),
                    (
                        "env_vars".to_string(),
                        ParameterDefinition::string("Environment variables as JSON object"),
                    ),
                    (
                        "capture".to_string(),
                        ParameterDefinition::string("Output capture mode: separate, merged, stdout_only, stderr_only, discard")
                            .default(serde_json::json!("separate")),
                    ),
                ]),
            )
            .category("shell")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let command = params["command"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'command' parameter"))?;

        let args: Vec<String> = params["args"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let env_vars: HashMap<String, String> = params["env_vars"]
            .as_str()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        let working_dir = params["working_dir"].as_str().unwrap_or("");
        let timeout = params["timeout"].as_i64().map(|t| t as u64);

        let capture = match params["capture"].as_str() {
            Some("separate") | None => OutputCapture::Separate,
            Some("merged") => OutputCapture::Merged,
            Some("stdout_only") => OutputCapture::StdoutOnly,
            Some("stderr_only") => OutputCapture::StderrOnly,
            Some("discard") => OutputCapture::Discard,
            Some(other) => {
                return Err(ToolError::invalid_args(format!(
                    "Unknown capture mode: '{}'",
                    other
                )))
            }
        };

        self.execute_enhanced(command, &args, &env_vars, working_dir, timeout, capture)
            .await
    }
}

// ---------------------------------------------------------------------------
// Stream output tool — tail a running process's output
// ---------------------------------------------------------------------------

/// Tool for streaming output from a background process.
///
/// Returns the most recent N lines of stdout/stderr from a process managed by
/// the `background_process` tool. This is useful for monitoring long-running
/// processes without blocking.
pub struct StreamOutputTool {
    /// Reference to the shared process registry (from BackgroundProcessTool).
    /// In practice this is a separate Arc<Mutex<HashMap>> that the tool shares.
    processes: Arc<Mutex<HashMap<String, BackgroundProcess>>>,
}

impl StreamOutputTool {
    /// Create a new stream output tool that shares the process map with a
    /// background process tool.
    pub fn new(processes: Arc<Mutex<HashMap<String, BackgroundProcess>>>) -> Self {
        Self { processes }
    }
}

#[async_trait]
impl Tool for StreamOutputTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "stream_output",
                "Stream or tail the output of a background process. "
                    + "Returns the most recent lines of stdout and stderr.",
                HashMap::from([
                    (
                        "process_id".to_string(),
                        ParameterDefinition::required_string("The background process ID"),
                    ),
                    (
                        "lines".to_string(),
                        ParameterDefinition::integer("Number of recent lines to return (default 50)")
                            .default(serde_json::json!(50)),
                    ),
                    (
                        "stream".to_string(),
                        ParameterDefinition::string("Which stream: stdout, stderr, both")
                            .default(serde_json::json!("both")),
                    ),
                ]),
            )
            .category("shell")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let id = params["process_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'process_id' parameter"))?;
        let lines = params["lines"].as_i64().unwrap_or(50) as usize;
        let stream = params["stream"].as_str().unwrap_or("both");

        let processes = self.processes.lock().await;
        let process = processes.get(id).ok_or_else(|| {
            ToolError::not_found(format!("Background process '{}' not found", id))
        })?;

        let stdout = process.stdout.lock().await.clone();
        let stderr = process.stderr.lock().await.clone();

        let stdout_tail: Vec<&str> = stdout.lines().rev().take(lines).collect::<Vec<_>>().into_iter().rev().collect();
        let stderr_tail: Vec<&str> = stderr.lines().rev().take(lines).collect::<Vec<_>>().into_iter().rev().collect();

        let content = match stream {
            "stdout" => stdout_tail.join("\n"),
            "stderr" => stderr_tail.join("\n"),
            "both" => {
                let mut out = String::new();
                if !stdout_tail.is_empty() {
                    out.push_str("--- stdout ---\n");
                    out.push_str(&stdout_tail.join("\n"));
                    out.push('\n');
                }
                if !stderr_tail.is_empty() {
                    out.push_str("--- stderr ---\n");
                    out.push_str(&stderr_tail.join("\n"));
                }
                out
            }
            other => return Err(ToolError::invalid_args(format!("Unknown stream: {}", other))),
        };

        let data = serde_json::json!({
            "process_id": id,
            "stdout_lines": stdout.lines().count(),
            "stderr_lines": stderr.lines().count(),
            "returned_lines": lines,
            "stream": stream,
            "completed": process.completed,
            "exit_code": process.exit_code,
        });

        Ok(ToolOutput::success(content).with_data(data))
    }
}

// ---------------------------------------------------------------------------
// Signal tool — send signals to background processes
// ---------------------------------------------------------------------------

/// Tool for sending signals to background processes.
pub struct SignalProcessTool {
    processes: Arc<Mutex<HashMap<String, BackgroundProcess>>>,
}

impl SignalProcessTool {
    /// Create a new signal tool.
    pub fn new(processes: Arc<Mutex<HashMap<String, BackgroundProcess>>>) -> Self {
        Self { processes }
    }
}

#[async_trait]
impl Tool for SignalProcessTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "signal_process",
                "Send a signal to a background process (TERM, KILL, INTERRUPT, HANGUP, USER1, USER2). "
                    + "On Windows, only TERM and KILL are supported.",
                HashMap::from([
                    (
                        "process_id".to_string(),
                        ParameterDefinition::required_string("The background process ID"),
                    ),
                    (
                        "signal".to_string(),
                        ParameterDefinition::required_string("The signal to send")
                            .enum_values(vec![
                                "TERM".to_string(),
                                "KILL".to_string(),
                                "INTERRUPT".to_string(),
                                "HANGUP".to_string(),
                                "USER1".to_string(),
                                "USER2".to_string(),
                            ]),
                    ),
                ]),
            )
            .category("shell")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let id = params["process_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'process_id' parameter"))?;
        let signal_str = params["signal"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'signal' parameter"))?;

        let signal = match signal_str {
            "TERM" => Signal::Term,
            "KILL" => Signal::Kill,
            "INTERRUPT" => Signal::Interrupt,
            "HANGUP" => Signal::Hangup,
            "USER1" => Signal::User1,
            "USER2" => Signal::User2,
            other => return Err(ToolError::invalid_args(format!("Unknown signal: {}", other))),
        };

        if !signal.is_supported() {
            return Err(ToolError::new(
                "UNSUPPORTED_SIGNAL",
                format!("Signal {:?} is not supported on this platform", signal),
            ));
        }

        let mut processes = self.processes.lock().await;
        let process = processes.get_mut(id).ok_or_else(|| {
            ToolError::not_found(format!("Background process '{}' not found", id))
        })?;

        if process.completed {
            return Err(ToolError::new(
                "PROCESS_COMPLETED",
                format!("Process '{}' has already completed", id),
            ));
        }

        if let Some(ref mut child) = process.child {
            if signal == Signal::Kill {
                let _ = child.kill().await;
            } else if cfg!(target_os = "windows") {
                // On Windows, TERM uses taskkill.
                if let Some(pid) = process.pid {
                    let pid_str = pid.to_string();
                    let _ = tokio::process::Command::new("taskkill")
                        .arg("/PID")
                        .arg(&pid_str)
                        .output()
                        .await;
                }
            } else if let Some(pid) = process.pid {
                if let Some(signum) = signal.unix_number() {
                    let signum_str = format!("-{}", signum);
                    let pid_str = pid.to_string();
                    let _ = tokio::process::Command::new("kill")
                        .arg(&signum_str)
                        .arg(&pid_str)
                        .output()
                        .await;
                }
            }
        }

        let data = serde_json::json!({
            "process_id": id,
            "signal": signal_str,
            "sent": true,
        });

        Ok(ToolOutput::success(format!(
            "Sent {} signal to process '{}'",
            signal_str, id
        ))
        .with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exec_command_echo() {
        let tool = ExecCommandTool::new(10);
        let result = tool
            .execute(serde_json::json!({"command": "echo", "args": ["hello world"]}))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("hello world"));
    }

    #[tokio::test]
    async fn test_exec_command_timeout() {
        let tool = ExecCommandTool::new(1);
        let result = tool
            .execute(serde_json::json!({"command": "sleep", "args": ["10"]}))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "TIMEOUT");
    }

    #[tokio::test]
    async fn test_exec_command_denied() {
        let tool = ExecCommandTool::new(10);
        let result = tool
            .execute(serde_json::json!({"command": "sudo rm -rf /"}))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PERMISSION_DENIED");
    }

    #[tokio::test]
    async fn test_background_process_lifecycle() {
        let tool = BackgroundProcessTool::new();

        let start = tool
            .execute(serde_json::json!({
                "operation": "start",
                "process_id": "test1",
                "command": "echo",
                "args": ["hello from bg"],
            }))
            .await;
        assert!(start.is_ok());

        let status = tool
            .execute(serde_json::json!({
                "operation": "status",
                "process_id": "test1",
            }))
            .await;
        assert!(status.is_ok());

        let list = tool.execute(serde_json::json!({"operation": "list"})).await;
        assert!(list.is_ok());
        assert!(list.unwrap().content.contains("test1"));
    }

    #[test]
    fn test_env_filter_blocks_sensitive_vars() {
        let filter = EnvFilter::new();
        assert!(!filter.is_allowed("SECRET_KEY"));
        assert!(!filter.is_allowed("API_KEY_OPENAI"));
        assert!(!filter.is_allowed("TOKEN_GITHUB"));
        assert!(!filter.is_allowed("PASSWORD_DB"));
        assert!(!filter.is_allowed("AWS_SECRET_ACCESS_KEY"));
        assert!(!filter.is_allowed("DATABASE_URL"));
        assert!(filter.is_allowed("HOME"));
        assert!(filter.is_allowed("MY_CUSTOM_VAR"));
    }

    #[test]
    fn test_env_filter_allowlist() {
        let filter = EnvFilter::new().allow_only(vec!["HOME".to_string(), "PATH".to_string()]);
        assert!(filter.is_allowed("HOME"));
        assert!(!filter.is_allowed("MY_VAR"));
    }

    #[test]
    fn test_env_filter_custom_deny() {
        let filter = EnvFilter::new().deny("MY_SECRET").deny_prefix("PRIVATE_");
        assert!(!filter.is_allowed("MY_SECRET"));
        assert!(!filter.is_allowed("PRIVATE_DATA"));
        assert!(filter.is_allowed("HOME"));
    }

    #[test]
    fn test_env_filter_filter_map() {
        let mut env = HashMap::new();
        env.insert("HOME".to_string(), "/home/user".to_string());
        env.insert("SECRET_KEY".to_string(), "abc123".to_string());
        env.insert("MY_VAR".to_string(), "value".to_string());

        let filter = EnvFilter::new();
        let filtered = filter.filter(&env);
        assert!(filtered.contains_key("HOME"));
        assert!(filtered.contains_key("MY_VAR"));
        assert!(!filtered.contains_key("SECRET_KEY"));
    }

    #[test]
    fn test_signal_unix_numbers() {
        assert_eq!(Signal::Term.unix_number(), Some(15));
        assert_eq!(Signal::Kill.unix_number(), Some(9));
        assert_eq!(Signal::Interrupt.unix_number(), Some(2));
    }

    #[test]
    fn test_signal_is_supported() {
        assert!(Signal::Term.is_supported());
        assert!(Signal::Kill.is_supported());
    }

    #[tokio::test]
    async fn test_enhanced_exec_echo() {
        let tool = EnhancedExecTool::new(10);
        let result = tool
            .execute(serde_json::json!({
                "command": "echo",
                "args": ["hello"],
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_enhanced_exec_denied_command() {
        let tool = EnhancedExecTool::new(10);
        let result = tool
            .execute(serde_json::json!({
                "command": "sudo rm -rf /",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PERMISSION_DENIED");
    }

    #[tokio::test]
    async fn test_enhanced_exec_filters_env_vars() {
        let tool = EnhancedExecTool::new(10);
        let env_vars = serde_json::json!({
            "SECRET_KEY": "should_be_filtered",
            "MY_VAR": "should_pass",
        })
        .to_string();
        let result = tool
            .execute(serde_json::json!({
                "command": "echo",
                "args": ["test"],
                "env_vars": env_vars,
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        let data = output.data.unwrap();
        assert_eq!(data["env_vars_filtered"], 1);
    }

    #[tokio::test]
    async fn test_enhanced_exec_discard_capture() {
        let tool = EnhancedExecTool::new(10);
        let result = tool
            .execute(serde_json::json!({
                "command": "echo",
                "args": ["hello"],
                "capture": "discard",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.content.contains("hello"));
    }

    #[tokio::test]
    async fn test_process_registry() {
        let registry = ProcessRegistry::new();
        let list = registry.list().await;
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_process_supervisor_cancel() {
        let mut supervisor = ProcessSupervisor::start(
            "test".to_string(),
            60,
            false,
            || {},
        );
        supervisor.cancel();
    }

    #[tokio::test]
    async fn test_signal_process_tool_not_found() {
        let processes = Arc::new(Mutex::new(HashMap::new()));
        let tool = SignalProcessTool::new(processes);
        let result = tool
            .execute(serde_json::json!({
                "process_id": "nonexistent",
                "signal": "TERM",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "TOOL_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_stream_output_tool_not_found() {
        let processes = Arc::new(Mutex::new(HashMap::new()));
        let tool = StreamOutputTool::new(processes);
        let result = tool
            .execute(serde_json::json!({
                "process_id": "nonexistent",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "TOOL_NOT_FOUND");
    }
}
