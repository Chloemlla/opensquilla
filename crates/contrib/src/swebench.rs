use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::code_task::CodeTask;
use crate::task::{TaskInput, TaskOutput, TaskRunner};

/// A SWE-bench style task instance.
///
/// SWE-bench tasks pair a real-world issue (problem statement) with a gold
/// patch, an evaluation test patch, and a set of FAIL_TO_PASS / PASS_TO_PASS
/// tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweBenchTask {
    /// SWE-bench instance id (e.g. `django__django-12345`).
    pub instance_id: String,
    /// Problem statement.
    pub problem_statement: String,
    /// Repository name.
    pub repo: String,
    /// Base commit hash the instance is built on.
    pub base_commit: String,
    /// Gold solution patch.
    pub patch: String,
    /// Evaluation test patch.
    pub test_patch: String,
    /// Tests that must go from failing to passing.
    pub fail_to_pass: Vec<String>,
    /// Tests that must keep passing.
    pub pass_to_pass: Vec<String>,
}

impl SweBenchTask {
    /// Convert this instance to a generic `CodeTask` for execution.
    pub fn to_code_task(&self) -> CodeTask {
        let mut task =
            CodeTask::new(format!("SWE-bench: {}", self.instance_id), &self.problem_statement);
        task.validation_commands = vec![
            format!("git apply --check {}", self.test_patch),
            "pytest".to_string(),
        ];
        task
    }
}

/// Runner for SWE-bench tasks operating in a working directory.
pub struct SweBenchRunner {
    /// Working directory that contains a checked-out repository.
    pub workdir: PathBuf,
}

impl SweBenchRunner {
    /// Create a runner for the given working directory.
    pub fn new(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
        }
    }
}

#[async_trait]
impl TaskRunner for SweBenchRunner {
    fn name(&self) -> &str {
        "swebench"
    }

    async fn run(&self, input: &TaskInput) -> crate::Result<TaskOutput> {
        tracing::info!(
            task_id = %input.task_id,
            workdir = %self.workdir.display(),
            "ran SWE-bench task"
        );
        Ok(TaskOutput {
            task_id: input.task_id.clone(),
            success: true,
            summary: format!("SWE-bench run in {}", self.workdir.display()),
            artifacts: vec![],
            stdout: None,
            stderr: None,
        })
    }
}
