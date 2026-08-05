//! Shell command execution tools: exec_command, background_process.
//!
//! Executes shell commands via `tokio::process::Command` with configurable
//! timeout, environment variables, working directory, and output capture.
//! The `exec_command` tool runs a command synchronously and returns its output,
//! while `background_process` starts a long-running process and manages its lifecycle.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
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
#[derive(Debug)]
struct BackgroundProcess {
    /// The command that was started.
    command: String,
    /// The process ID on the system.
    pid: Option<u32>,
    /// The accumulated stdout so far (shared with the drain task).
    stdout: Arc<Mutex<String>>,
    /// The accumulated stderr so far (shared with the drain task).
    stderr: Arc<Mutex<String>>,
    /// Whether the process has completed.
    completed: bool,
    /// The exit code, if completed.
    exit_code: Option<i32>,
    /// The timestamp when the process was started.
    started_at: Instant,
    /// Handle to the drain task that reads stdout/stderr to EOF and records
    /// the exit status. Resolving it finalizes `completed`/`exit_code`.
    drain_handle: Option<JoinHandle<()>>,
    /// The child process handle, kept so `stop` can kill it.
    child: Option<Child>,
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

        let output = result.map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to execute command: {}", e))
        })?;

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
                        ParameterDefinition::array("Additional arguments for the command", ParameterDefinition::string("Argument value")),
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
                        ParameterDefinition::string("Environment variables as JSON object (key-value pairs)"),
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
        let mut child = cmd.spawn().map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to spawn process: {}", e))
        })?;

        let pid = child.id().ok_or_else(|| {
            ToolError::new("IO_ERROR", "Failed to get process ID".to_string())
        })?;

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

        Ok(ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
            .with_data(data))
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

        let stdout = process
            .stdout
            .lock()
            .await
            .clone();
        let stderr = process
            .stderr
            .lock()
            .await
            .clone();

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

        Ok(ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
            .with_data(data))
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
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let env_vars = HashMap::new();
                let working_dir = params["working_dir"].as_str().unwrap_or(".");
                self.start_process(id, command, &args, &env_vars, working_dir).await
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
            other => Err(ToolError::invalid_args(format!("Unknown operation: {}", other))),
        }
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

        let list = tool
            .execute(serde_json::json!({"operation": "list"}))
            .await;
        assert!(list.is_ok());
        assert!(list.unwrap().content.contains("test1"));
    }
}