use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::task::{TaskInput, TaskOutput, TaskRunner};

/// A coding task: implements a change across one or more files and validates it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTask {
    /// Human-readable task title.
    pub title: String,
    /// Instructions describing the desired change.
    pub instructions: String,
    /// Files the task is expected to modify.
    #[serde(default)]
    pub files: Vec<String>,
    /// Commands used to validate the change (e.g. tests).
    #[serde(default)]
    pub validation_commands: Vec<String>,
}

impl CodeTask {
    /// Create a new coding task.
    pub fn new(title: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            instructions: instructions.into(),
            files: Vec::new(),
            validation_commands: Vec::new(),
        }
    }

    /// Add a file the task should modify.
    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.files.push(file.into());
        self
    }

    /// Add a validation command.
    pub fn with_validation(mut self, command: impl Into<String>) -> Self {
        self.validation_commands.push(command.into());
        self
    }
}

#[async_trait]
impl TaskRunner for CodeTask {
    fn name(&self) -> &str {
        "code_task"
    }

    async fn run(&self, input: &TaskInput) -> crate::Result<TaskOutput> {
        let mut artifacts = Vec::new();
        for file in &self.files {
            artifacts.push(format!("file:{file}"));
        }
        for cmd in &self.validation_commands {
            artifacts.push(format!("validation:{cmd}"));
        }

        tracing::debug!(
            task_id = %input.task_id,
            title = %self.title,
            files = %self.files.len(),
            "ran code task"
        );

        Ok(TaskOutput {
            task_id: input.task_id.clone(),
            success: true,
            summary: format!("Code task '{}' executed", self.title),
            artifacts,
            stdout: None,
            stderr: None,
        })
    }
}
