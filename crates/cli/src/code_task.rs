//! Coding-task commands.
//!
//! Implements the `code-task` subcommand against the contrib crate's
//! [`CodeTask`] / [`SweBenchRunner`] task runners. Also provides the
//! `stage-task-file` helper used by automation to hand a task description to a
//! subprocess without passing it on the command line.
//!
//! The contrib `CodeTask::run` currently returns placeholder artifacts (it
//! records the target files and validation commands rather than executing a
//! real agent loop). The CLI wires the same runner interface so the command is
//! present and runs to a `TaskOutput`; a full clone-verify orchestration is a
//! follow-up (see TODOs below).

use std::io::Read;
use std::path::PathBuf;

use anyhow::{Context, Result};
use opensquilla_contrib::swebench::{SweBenchRunner, SweBenchTask};
use opensquilla_contrib::task::{TaskInput, TaskRunner};
use opensquilla_contrib::CodeTask;
use opensquilla_core::config::Config;
use serde_json::json;

/// Code-task subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum CodeTaskAction {
    /// Solve a coding task against a repository (or scratch) with an agent.
    Solve {
        /// Repo to work on (git URL or local path).
        #[arg(long)]
        repo: Option<String>,
        /// Free-form task / feature-request text.
        #[arg(long)]
        task: Option<String>,
        /// Path to a file holding the task description.
        #[arg(long)]
        task_file: Option<String>,
        /// Base ref to start from (default: HEAD).
        #[arg(long)]
        base: Option<String>,
        /// Model override; empty lets config decide.
        #[arg(long)]
        model: Option<String>,
        /// Agent timeout in seconds.
        #[arg(long, default_value = "5400")]
        timeout: u64,
        /// How to verify: red-green / build / scratch.
        #[arg(long, default_value = "red-green")]
        verification_mode: String,
        /// Print the result as JSON.
        #[arg(long)]
        json: bool,
        /// Skip the trusted-host confirmation prompt.
        #[arg(short, long)]
        yes: bool,
    },
    /// Read task text from stdin, write a private temp file, print a quoted path.
    StageTaskFile,
    /// Run a SWE-bench instance through the code-task execution flow.
    Swebench {
        /// SWE-bench instance id, e.g. django__django-16429.
        instance_id: String,
        /// Dataset: verified / multilingual / HuggingFace name.
        #[arg(long, default_value = "verified")]
        dataset: String,
        /// Model override; empty lets config decide.
        #[arg(long)]
        model: Option<String>,
        /// Agent timeout in seconds.
        #[arg(long, default_value = "1200")]
        timeout: u64,
        /// Print the result as JSON.
        #[arg(long)]
        json: bool,
    },
}

/// Run a code-task subcommand.
pub async fn run_code_task(action: CodeTaskAction) -> Result<()> {
    match action {
        CodeTaskAction::Solve {
            repo,
            task,
            task_file,
            base,
            model,
            timeout,
            verification_mode,
            json,
            yes,
        } => {
            let _ = timeout;
            let _ = yes;
            code_task_solve(repo, task, task_file, base, model, &verification_mode, json).await
        }
        CodeTaskAction::StageTaskFile => stage_task_file(),
        CodeTaskAction::Swebench {
            instance_id,
            dataset,
            model,
            timeout,
            json,
        } => swebench_solve(instance_id, dataset, model, timeout, json).await,
    }
}

/// Run a coding task through the contrib `TaskRunner` interface.
async fn code_task_solve(
    repo: Option<String>,
    task: Option<String>,
    task_file: Option<String>,
    base: Option<String>,
    model: Option<String>,
    verification_mode: &str,
    json: bool,
) -> Result<()> {
    let given = [task.is_some(), task_file.is_some()].iter().filter(|b| **b).count();
    if given != 1 {
        anyhow::bail!("Pass exactly one of --task or --task-file.");
    }
    if !matches!(verification_mode, "red-green" | "build" | "scratch") {
        anyhow::bail!("Unknown --verification-mode '{verification_mode}'; expected red-green, build, or scratch.");
    }

    let description = match (task, task_file) {
        (Some(t), _) => t,
        (_, Some(path)) => std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read task file {path}"))?,
        _ => unreachable!(),
    };

    // Build a CodeTask carrying the task description and any repo context.
    let mut code_task = CodeTask::new("code-task", description);
    if let Some(repo) = repo {
        code_task = code_task.with_file(repo);
    }
    if let Some(base) = base {
        code_task = code_task.with_validation(format!("git checkout {base}"));
    }
    if let Some(model) = model {
        let _ = model;
    }

    let input = TaskInput::new(formatted_input(&code_task));
    let output = code_task.run(&input).await.map_err(|e| anyhow::anyhow!("{e}"))?;

    if json {
        crate::util::print_json(&json!({
            "task_id": output.task_id,
            "success": output.success,
            "summary": output.summary,
            "artifacts": output.artifacts,
        }))?;
    } else {
        println!("{}", output.summary);
        println!("  task_id:   {}", output.task_id);
        println!("  success:   {}", output.success);
        for artifact in &output.artifacts {
            println!("  artifact:  {artifact}");
        }
    }
    Ok(())
}

/// Build the human-readable description handed to the task runner.
fn formatted_input(task: &CodeTask) -> String {
    let mut s = format!("{}: {}", task.title, task.instructions);
    if !task.files.is_empty() {
        s.push_str(&format!("\nrepo: {}", task.files.join(" ")));
    }
    s
}

/// Write stdin to a private temp file and print a shell-quoted path.
fn stage_task_file() -> Result<()> {
    let mut payload = Vec::new();
    std::io::stdin()
        .read_to_end(&mut payload)
        .context("Failed to read task text from stdin")?;

    let dir = std::env::temp_dir();
    std::fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("codetask-task-{}.txt", uuid::Uuid::new_v4()));
    std::fs::write(&path, &payload).with_context(|| format!("Failed to write {}", path.display()))?;

    println!("{}", shell_quote(&path.to_string_lossy()));
    Ok(())
}

/// Quote a path for the current shell (best-effort on Windows).
fn shell_quote(path: &str) -> String {
    shell_quote_for(path, cfg!(windows))
}

/// Quote a path for a shell, given the target platform's quoting style.
fn shell_quote_for(path: &str, windows: bool) -> String {
    if windows {
        format!("\"{}\"", path.replace('"', "\\\""))
    } else {
        format!("'{}'", path.replace('\'', "'\\''"))
    }
}

/// Run a SWE-bench instance through the code-task execution flow.
async fn swebench_solve(
    instance_id: String,
    dataset: String,
    model: Option<String>,
    timeout: u64,
    json: bool,
) -> Result<()> {
    let _ = model;
    let _ = timeout;
    let _ = dataset;

    // Build a SweBenchTask from the requested instance. The problem statement
    // is filled from the instance id because fetching the official dataset is a
    // Docker/data dependency outside the CLI's reach; `to_code_task` derives a
    // runnable CodeTask from it.
    let swe_task = SweBenchTask {
        instance_id: instance_id.clone(),
        problem_statement: format!("SWE-bench instance {instance_id}"),
        repo: String::new(),
        base_commit: String::new(),
        patch: String::new(),
        test_patch: String::new(),
        fail_to_pass: Vec::new(),
        pass_to_pass: Vec::new(),
    };
    let code_task = swe_task.to_code_task();

    // Workdir: honor OPENSQUILLA_SWE_WORKDIR, else a temp dir.
    let workdir = std::env::var("OPENSQUILLA_SWE_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let runner = SweBenchRunner::new(workdir);

    let input = TaskInput::new(formatted_input(&code_task));
    let output = runner
        .run(&input)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if json {
        crate::util::print_json(&json!({
            "instance_id": instance_id,
            "task_id": output.task_id,
            "state": if output.success { "patch_collected" } else { "failed" },
            "success": output.success,
            "summary": output.summary,
        }))?;
    } else {
        println!("[{}] {}", if output.success { "patch_collected" } else { "failed" }, instance_id);
        println!("  {}", output.summary);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_task_file_quotes_windows() {
        let q = shell_quote_for(r"C:\tmp\a b.txt", true);
        assert!(q.starts_with('"'));
        assert!(q.ends_with('"'));
    }

    #[test]
    fn test_formatted_input_includes_repo() {
        let t = CodeTask::new("t", "desc").with_file("github.com/foo/bar");
        let s = formatted_input(&t);
        assert!(s.contains("github.com/foo/bar"));
    }

    #[test]
    fn test_swe_to_code_task_has_validation() {
        let task = SweBenchTask {
            instance_id: "django__django-16429".to_string(),
            problem_statement: "p".to_string(),
            repo: String::new(),
            base_commit: String::new(),
            patch: String::new(),
            test_patch: String::new(),
            fail_to_pass: Vec::new(),
            pass_to_pass: Vec::new(),
        };
        let ct = task.to_code_task();
        assert!(ct.validation_commands.iter().any(|c| c.contains("pytest")));
    }

    #[tokio::test]
    async fn test_code_task_runner_runs() {
        let task = CodeTask::new("t", "hello");
        let input = TaskInput::new("hello");
        let output = task.run(&input).await.unwrap();
        assert!(output.success);
    }
}

/// Keep `Config` referenced so future solve orchestration can read it.
#[allow(dead_code)]
fn _load_config() -> Result<Config> {
    Config::load().context("Failed to load configuration")
}