//! Git tools: git_clone, git_status, git_diff, git_add, git_commit, git_push, git_log.
//!
//! Git operations via `tokio::process::Command`. All operations are
//! scoped to a specific working directory for security.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
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
            Ok(
                ToolOutput::success(if stdout.is_empty() { stderr } else { stdout })
                    .with_data(data),
            )
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
                concat!(
                    "Execute Git operations including clone, status, diff, add, commit, push, log, ",
                    "branch, checkout, init, merge, rebase, stash, tag, blame, remote, fetch, ",
                    "reset, revert, cherry-pick, show, and config. ",
                    "All operations are scoped to the allowed working directory.",
),
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
                                "init".to_string(),
                                "merge".to_string(),
                                "rebase".to_string(),
                                "stash".to_string(),
                                "tag".to_string(),
                                "blame".to_string(),
                                "remote".to_string(),
                                "fetch".to_string(),
                                "reset".to_string(),
                                "revert".to_string(),
                                "cherry-pick".to_string(),
                                "show".to_string(),
                                "config".to_string(),
                            ]),
                    ),
                    (
                        "repo_url".to_string(),
                        ParameterDefinition::string("Repository URL (required for clone)"),
                    ),
                    (
                        "target_dir".to_string(),
                        ParameterDefinition::string(
                            "Target directory name (for clone) or working directory",
                        ),
                    ),
                    (
                        "message".to_string(),
                        ParameterDefinition::string("Commit message (required for commit)"),
                    ),
                    (
                        "files".to_string(),
                        ParameterDefinition::array(
                            "Files to add (for add operation)",
                            ParameterDefinition::string("file path"),
                        ),
                    ),
                    (
                        "branch".to_string(),
                        ParameterDefinition::string("Branch name (for branch/checkout operations)"),
                    ),
                    (
                        "max_count".to_string(),
                        ParameterDefinition::integer(
                            "Maximum number of log entries (for log operation)",
                        )
                        .default(serde_json::json!(10)),
                    ),
                    (
                        "working_dir".to_string(),
                        ParameterDefinition::string("Working directory for the operation"),
                    ),
                    (
                        "depth".to_string(),
                        ParameterDefinition::integer(
                            "Shallow clone depth (for clone/fetch, creates a shallow history)",
                        ),
                    ),
                    (
                        "bare".to_string(),
                        ParameterDefinition::boolean("Create a bare repository (for init)"),
                    ),
                    (
                        "no_ff".to_string(),
                        ParameterDefinition::boolean("Force no fast-forward merge (for merge)"),
                    ),
                    (
                        "stash_op".to_string(),
                        ParameterDefinition::string("Stash operation: push, pop, list, drop")
                            .default(serde_json::json!("push")),
                    ),
                    (
                        "stash_index".to_string(),
                        ParameterDefinition::integer("Stash index (for stash drop)"),
                    ),
                    (
                        "tag".to_string(),
                        ParameterDefinition::string("Tag name (for tag operation)"),
                    ),
                    (
                        "file".to_string(),
                        ParameterDefinition::string("File path (for blame)"),
                    ),
                    (
                        "remote_op".to_string(),
                        ParameterDefinition::string("Remote operation: list, add, remove, set-url")
                            .default(serde_json::json!("list")),
                    ),
                    (
                        "remote".to_string(),
                        ParameterDefinition::string("Remote name (for fetch)"),
                    ),
                    (
                        "name".to_string(),
                        ParameterDefinition::string("Remote name (for remote add/remove/set-url)"),
                    ),
                    (
                        "target".to_string(),
                        ParameterDefinition::string("Reset target (commit/branch, for reset)"),
                    ),
                    (
                        "mode".to_string(),
                        ParameterDefinition::string("Reset mode: soft, mixed, hard")
                            .default(serde_json::json!("mixed")),
                    ),
                    (
                        "commit".to_string(),
                        ParameterDefinition::string("Commit hash (for revert, cherry-pick, show)"),
                    ),
                    (
                        "no_commit".to_string(),
                        ParameterDefinition::boolean("No commit (for revert)"),
                    ),
                    (
                        "key".to_string(),
                        ParameterDefinition::string("Config key (for config)"),
                    ),
                    (
                        "value".to_string(),
                        ParameterDefinition::string("Config value (for config)"),
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
                let depth = params["depth"].as_i64();
                let mut args = vec!["clone".to_string(), repo_url.to_string()];
                if let Some(d) = depth {
                    args.push(format!("--depth={}", d));
                }
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
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
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
                let args = vec!["commit".to_string(), "-m".to_string(), message.to_string()];
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
            "init" => {
                let bare = params["bare"].as_bool().unwrap_or(false);
                let mut args = vec!["init".to_string()];
                if bare {
                    args.push("--bare".to_string());
                }
                self.run_git(&args, working_dir).await
            }
            "merge" => {
                let branch = params["branch"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'branch' for merge"))?;
                let no_ff = params["no_ff"].as_bool().unwrap_or(false);
                let mut args = vec!["merge".to_string()];
                if no_ff {
                    args.push("--no-ff".to_string());
                }
                args.push(branch.to_string());
                self.run_git(&args, working_dir).await
            }
            "rebase" => {
                let branch = params["branch"].as_str();
                let mut args = vec!["rebase".to_string()];
                if let Some(b) = branch {
                    args.push(b.to_string());
                }
                self.run_git(&args, working_dir).await
            }
            "stash" => {
                let stash_op = params["stash_op"].as_str().unwrap_or("push");
                let mut args = vec!["stash".to_string()];
                match stash_op {
                    "push" => {
                        if let Some(msg) = params["message"].as_str() {
                            args.push("push".to_string());
                            args.push("-m".to_string());
                            args.push(msg.to_string());
                        }
                    }
                    "pop" => args.push("pop".to_string()),
                    "list" => args.push("list".to_string()),
                    "drop" => {
                        args.push("drop".to_string());
                        if let Some(idx) = params["stash_index"].as_i64() {
                            args.push(format!("stash@{{{}}}", idx));
                        }
                    }
                    other => {
                        return Err(ToolError::invalid_args(format!(
                            "Unknown stash operation: {}",
                            other
                        )));
                    }
                }
                self.run_git(&args, working_dir).await
            }
            "tag" => {
                let tag_name = params["tag"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'tag' for tag operation"))?;
                let message = params["message"].as_str();
                let mut args = vec!["tag".to_string()];
                if let Some(msg) = message {
                    args.push("-a".to_string());
                    args.push(tag_name.to_string());
                    args.push("-m".to_string());
                    args.push(msg.to_string());
                } else {
                    args.push(tag_name.to_string());
                }
                self.run_git(&args, working_dir).await
            }
            "blame" => {
                let file = params["file"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'file' for blame"))?;
                let args = vec!["blame".to_string(), file.to_string()];
                self.run_git(&args, working_dir).await
            }
            "remote" => {
                let remote_op = params["remote_op"].as_str().unwrap_or("list");
                let mut args = vec!["remote".to_string()];
                match remote_op {
                    "list" => {}
                    "add" => {
                        let name = params["name"].as_str().ok_or_else(|| {
                            ToolError::invalid_args("Missing 'name' for remote add")
                        })?;
                        let url = params["repo_url"].as_str().ok_or_else(|| {
                            ToolError::invalid_args("Missing 'repo_url' for remote add")
                        })?;
                        args.push("add".to_string());
                        args.push(name.to_string());
                        args.push(url.to_string());
                    }
                    "remove" => {
                        let name = params["name"].as_str().ok_or_else(|| {
                            ToolError::invalid_args("Missing 'name' for remote remove")
                        })?;
                        args.push("remove".to_string());
                        args.push(name.to_string());
                    }
                    "set-url" => {
                        let name = params["name"].as_str().ok_or_else(|| {
                            ToolError::invalid_args("Missing 'name' for remote set-url")
                        })?;
                        let url = params["repo_url"].as_str().ok_or_else(|| {
                            ToolError::invalid_args("Missing 'repo_url' for remote set-url")
                        })?;
                        args.push("set-url".to_string());
                        args.push(name.to_string());
                        args.push(url.to_string());
                    }
                    other => {
                        return Err(ToolError::invalid_args(format!(
                            "Unknown remote operation: {}",
                            other
                        )));
                    }
                }
                self.run_git(&args, working_dir).await
            }
            "fetch" => {
                let remote = params["remote"].as_str().unwrap_or("origin");
                let mut args = vec!["fetch".to_string(), remote.to_string()];
                if let Some(branch) = params["branch"].as_str() {
                    args.push(branch.to_string());
                }
                if let Some(depth) = params["depth"].as_i64() {
                    args.push(format!("--depth={}", depth));
                }
                self.run_git(&args, working_dir).await
            }
            "reset" => {
                let target = params["target"].as_str().unwrap_or("HEAD");
                let mode = params["mode"].as_str().unwrap_or("mixed");
                let mut args = vec!["reset".to_string()];
                match mode {
                    "soft" => args.push("--soft".to_string()),
                    "mixed" => args.push("--mixed".to_string()),
                    "hard" => args.push("--hard".to_string()),
                    _ => {}
                }
                args.push(target.to_string());
                self.run_git(&args, working_dir).await
            }
            "revert" => {
                let commit = params["commit"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'commit' for revert"))?;
                let no_commit = params["no_commit"].as_bool().unwrap_or(false);
                let mut args = vec!["revert".to_string()];
                if no_commit {
                    args.push("--no-commit".to_string());
                }
                args.push(commit.to_string());
                self.run_git(&args, working_dir).await
            }
            "cherry-pick" => {
                let commit = params["commit"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'commit' for cherry-pick"))?;
                let args = vec!["cherry-pick".to_string(), commit.to_string()];
                self.run_git(&args, working_dir).await
            }
            "show" => {
                let commit = params["commit"].as_str().unwrap_or("HEAD");
                let args = vec!["show".to_string(), commit.to_string()];
                self.run_git(&args, working_dir).await
            }
            "config" => {
                let key = params["key"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'key' for config"))?;
                let value = params["value"].as_str();
                let mut args = vec!["config".to_string()];
                if let Some(v) = value {
                    args.push(key.to_string());
                    args.push(v.to_string());
                } else {
                    args.push("--get".to_string());
                    args.push(key.to_string());
                }
                self.run_git(&args, working_dir).await
            }
            other => Err(ToolError::invalid_args(format!(
                "Unknown git operation: {}",
                other
            ))),
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

    #[tokio::test]
    async fn test_git_init_operation() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("init_repo");
        fs::create_dir(&repo_path).unwrap();

        let tool = GitTool::new(dir.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "operation": "init",
                "working_dir": "init_repo",
            }))
            .await;
        assert!(result.is_ok());
        assert!(repo_path.join(".git").exists());
    }

    #[tokio::test]
    async fn test_git_branch_create_and_checkout() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("branch_repo");
        fs::create_dir(&repo_path).unwrap();

        Command::new("git")
            .arg("init")
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
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

        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "branch",
                "branch": "feature",
                "working_dir": "branch_repo",
            }))
            .await;
        assert!(result.is_ok());

        let result = tool
            .execute(serde_json::json!({
                "operation": "checkout",
                "branch": "feature",
                "working_dir": "branch_repo",
            }))
            .await;
        assert!(result.is_ok());

        let result = tool
            .execute(serde_json::json!({
                "operation": "branch",
                "working_dir": "branch_repo",
            }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("feature"));
    }

    #[tokio::test]
    async fn test_git_log_formatted() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("log_repo");
        fs::create_dir(&repo_path).unwrap();

        Command::new("git")
            .arg("init")
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
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
        fs::write(repo_path.join("file.txt"), "content").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "initial commit"])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();

        let tool = GitTool::new(dir.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "operation": "log",
                "working_dir": "log_repo",
                "max_count": 5,
            }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("initial commit"));
    }

    #[tokio::test]
    async fn test_git_stash_operations() {
        let dir = tempfile::tempdir().unwrap();
        let repo_path = dir.path().join("stash_repo");
        fs::create_dir(&repo_path).unwrap();

        Command::new("git")
            .arg("init")
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
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
        fs::write(repo_path.join("file.txt"), "v1").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(&repo_path)
            .output()
            .await
            .unwrap();
        fs::write(repo_path.join("file.txt"), "v2").unwrap();

        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "stash",
                "stash_op": "push",
                "working_dir": "stash_repo",
            }))
            .await;
        assert!(result.is_ok());

        let result = tool
            .execute(serde_json::json!({
                "operation": "stash",
                "stash_op": "list",
                "working_dir": "stash_repo",
            }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("stash"));
    }

    #[tokio::test]
    async fn test_git_missing_message_for_commit() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "commit",
                "working_dir": ".",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_git_reset_modes() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GitTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "reset",
                "target": "HEAD",
                "mode": "hard",
                "working_dir": ".",
            }))
            .await;
        // Either succeeds or fails with GIT_ERROR (not a repo), but must not
        // be INVALID_ARGS.
        if let Err(e) = result {
            assert_ne!(e.code, "INVALID_ARGS");
        }
    }
}
