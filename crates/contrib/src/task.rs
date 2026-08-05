use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A task input to be executed by a `TaskRunner`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInput {
    /// Unique task id.
    pub task_id: String,
    /// Human-readable task description.
    pub description: String,
    /// Files relevant to the task.
    #[serde(default)]
    pub files: Vec<String>,
    /// Arbitrary metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TaskInput {
    /// Create a new task input with a generated task id.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            task_id: opensquilla_core::id::new_id().to_string(),
            description: description.into(),
            files: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// The result of running a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutput {
    /// The id of the task that was run.
    pub task_id: String,
    /// Whether the task completed successfully.
    pub success: bool,
    /// A human-readable summary of the outcome.
    pub summary: String,
    /// Paths or references to produced artifacts.
    #[serde(default)]
    pub artifacts: Vec<String>,
    /// Captured stdout, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdout: Option<String>,
    /// Captured stderr, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
}

/// The runner interface implemented by all contrib tasks.
#[async_trait]
pub trait TaskRunner: Send + Sync {
    /// The runner's stable name.
    fn name(&self) -> &str;

    /// Execute the task.
    async fn run(&self, input: &TaskInput) -> crate::Result<TaskOutput>;
}
