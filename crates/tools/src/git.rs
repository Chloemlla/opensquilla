//! Git tools: git_clone, git_status, git_diff, git_add, git_commit, git_push, git_log.
//!
//! Git operations via `tokio::process::Command`. All operations are
//! scoped to a specific working directory for security.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use tokio::process::Command;

/// Tool for Git operations.
pub struct GitTool {
    /// Allowed working directory for Git operations.
    allowed_base: PathBuf,
    /// Default timeout in seconds.
    timeout_secs: u64,
}

impl GitTool {
    /// Create a new Git tool with the given base directory.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            timeout_secs: 60,
        }
    }

    /// Run a git command with the given arguments in the working directory.
    async fn run_git(&self, args: &[String], working_dir: &str) -> ToolResult<ToolOutput> {
        let start = Instant::now();

        let work_path = if working_dir.is_empty() || working_dir == "." {
            self.allowed_base.clone()
        } else {
            let p = if PathBuf::from(working_dir).is_relative() {
                self.allowed_base.join(working_dir)
            } else {
                PathBuf::from(working_dir)
            };
            p.canonicalize().map_err(|e| {
                ToolError::new(
                    "PATH_INVALID",
                    format!("Cannot access working directory '{}': {}", working_dir, e),
                )
            })?
        };

        // Security: ensure working directory is within allowed base.
        if !work_path.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!(
                    "Working directory '{}' is outside the allowed base",
                    working_dir
                ),
            ));
        }

        let mut cmd = Command::new("git");
        cmd.args(args);
        cmd.current_dir(&work_path);
        cmd.kill_on_drop(true);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(self.timeout_secs),
            cmd.output(),
        )
        .await
        .map_err(|_| ToolError::timeout(self.timeout_secs))?
        .map_err(|e| ToolError::new("IO_ERROR", format!("Git command failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let exit_code = result.status.code().unwrap_or(-1);
        let duration_ms = start.elapsed().as_millis() as u64;

        let data = serde_json::json!({
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "args": args,
        });

        if result.status.success() {
            Ok(ToolOutput::success(if stdout.is_empty() { stderr } else { stdout })
                .with_data(data))
        } else {
            let msg = if stderr.is_empty() {
                format!("Git command failed with exit code {}", exit_code)
            } else {
                format!("Git command failed: {}", stderr.trim())
            };
            Err(ToolError::new("GIT_ERROR", msg))
        }
    }
}

#[async_trait]
impl Tool for GitTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "git",
                "Execute Git operations including clone, status, diff, add, commit, push, and log. "
                    + "All operations are scoped to the allowed working directory.",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The git operation to perform")
                            .enum_values(vec![
                                "clone".to_string(),
                                "status".to_string(),
                                "diff".to_string(),
                                "add".to_string(),
                                "commit".to_string(),
                                "push".to_string(),
                                "log".to_string(),
                                "pull".to_string(),
                                "branch".to_string(),
                                "checkout".to_string(),
                            ]),
                    ),
                    (
                        "repo_url".to_string(),
                        ParameterDefinition::string("Repository URL (required for clone)"),
                    ),
                    (
                        "target_dir".to_string(),
                        ParameterDefinition::string("Target directory name (for clone) or working directory"),
                    ),
                    (
                        "message".to_string(),
                        ParameterDefinition::string("Commit message (required for commit)"),
                    ),
                    (
                        "files".to_string(),
                        ParameterDefinition::array("Files to add (for add operation)", ParameterDefinition::string("file path")),
                    ),
                    (
                        "branch".to_string(),
                        ParameterDefinition::string("Branch name (for branch/checkout operations)"),
                    ),
                    (
                        "max_count".to_string(),
                        ParameterDefinition::integer("Maximum number of log entries (for log operation)")
                            .default(serde_json::json!(10)),
                    ),
                    (
                        "working_dir".to_string(),
                        ParameterDefinition::string("Working directory for the operation"),
                    ),
                ]),
            )
            .category("git")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?;

        let working_dir = params["working_dir"].as_str().unwrap_or("");

        match operation {
            "clone" => {
                let repo_url = params["repo_url"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'repo_url' for clone"))?;
                let target_dir = params["target_dir"].as_str().unwrap_or("");
                let mut args = vec!["clone".to_string(), repo_url.to_string()];
                if !target_dir.is_empty() {
                    args.push(target_dir.to_string());
                }
                self.run_git(&args, working_dir).await
            }
            "status" => {
                let args = vec!["status".to_string()];
                self.run_git(&args, working_dir).await
            }
            "diff" => {
                let args = vec!["diff".to_string()];
                self.run_git(&args, working_dir).await
            }
            "add" => {
                let files: Vec<String> = params["files"]
                    .as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                if files.is_empty() {
                    let args = vec!["add".to_string(), ".".to_string()];
                    self.run_git(&args, working_dir).await
                } else {
                    let mut args = vec!["add".to_string()];
                    args.extend(files);
                    self.run_git(&args, working_dir).await
                }
            }
            "commit" => {
                let message = params["message"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'message' for commit"))?;
                let args = vec![
                    "commit".to_string(),
                    "-m".to_string(),
                    message.to_string(),
                ];
                self.run_git(&args, working_dir).await
            }
            "push" => {
                let args = vec!["push".to_string()];
                self.run_git(&args, working_dir).await
            }
            "pull" => {
                let args = vec!["pull".to_string()];
                self.run_git(&args, working_dir).await
            }
            "log" => {
                let max_count = params["max_count"].as_i64().unwrap_or(10);
                let args = vec![
                    "log".to_string(),
                    format!("--max-count={}", max_count),
                    "--oneline".to_string(),
                ];
                self.run_git(&args, working_dir).await
            }
            "branch" => {
                let branch = params["branch"].as_str();
                if let Some(name) = branch {
                    let args = vec!["branch".to_string(), name.to_string()];
                    self.run_git(&args, working_dir).await
                } else {
                    let args = vec!["branch".to_string()];
                    self.run_git(&args, working_dir).await
                }
            }
            "checkout" => {
                let branch = params["branch"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'branch' for checkout"))?;
                let args = vec!["checkout".to_string(), branch.to_string()];
                self.run_git(&args, working_dir).await
            }
            other => Err(ToolError::invalid_args(format!("Unknown git operation: {}", other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[tokio::test]
    async fn test_git_init_and_status() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("test_repo");
        fs::create_dir(&repo_path).unwrap();

        // Initialize a git repo.
        let init = Command::new("git")
            .arg("init")
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
        assert!(init.status.success());

        // Configure git for the test.
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();

        // Create a test file.
        fs::write(repo_path.join("test.txt"), "hello").unwrap();

        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "status",
                "working_dir": "test_repo",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("test.txt") || output.content.contains("untracked"));
    }

    #[tokio::test]
    async fn test_git_invalid_operation() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({"operation": "nonexistent"}))
            .await;
        assert!(result.is_err());
    }
}