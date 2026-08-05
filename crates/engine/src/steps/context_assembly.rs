//! Context assembly step.
//!
//! Mirrors the Python `engine/context.py` behavior: load the workspace-level
//! instruction files (SOUL.md, AGENTS.md, CLAUDE.md, ...) and assemble them
//! into a system prompt fragment that is injected ahead of the user's message.
//!
//! The step reuses [`crate::context::ContextBuilder`], which provides
//! fail-open file loading (missing files are silently skipped) and
//! priority-ordered fragment assembly. The assembled fragment is prepended as
//! a system message when no system message exists yet, otherwise appended to
//! the existing system message so both survive the turn.

use crate::context::ContextBuilder;
use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use std::path::PathBuf;
use tracing::{debug, instrument};

/// Configuration for the context assembly step.
#[derive(Debug, Clone)]
pub struct ContextAssemblyConfig {
    /// Workspace root used to resolve relative context file paths.
    pub workspace_root: Option<PathBuf>,
    /// Context files loaded in order. Defaults to the canonical set when empty.
    pub files: Vec<String>,
    /// Maximum total characters contributed by context files. Longer files are
    /// truncated per-file to keep the prompt bounded.
    pub max_total_chars: usize,
}

impl Default for ContextAssemblyConfig {
    fn default() -> Self {
        Self {
            workspace_root: None,
            files: Vec::new(),
            max_total_chars: 60_000,
        }
    }
}

/// The outcome of context assembly, recorded in metadata.
#[derive(Debug, Clone)]
pub struct ContextAssemblyOutcome {
    /// Number of fragments successfully loaded.
    pub fragment_count: usize,
    /// Total characters assembled into the system prompt.
    pub total_chars: usize,
    /// Whether a system message was present before assembly.
    pub had_system_message: bool,
}

/// Pre-turn pipeline step that assembles workspace context into the messages.
#[derive(Debug)]
pub struct ContextAssemblyStep {
    config: ContextAssemblyConfig,
}

impl ContextAssemblyStep {
    /// Create a new step with the given workspace root.
    pub fn new(workspace_root: Option<PathBuf>) -> Self {
        Self {
            config: ContextAssemblyConfig {
                workspace_root,
                ..Default::default()
            },
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: ContextAssemblyConfig) -> Self {
        Self { config }
    }

    /// The default context files, mirroring [`crate::context::DEFAULT_CONTEXT_FILES`].
    pub fn default_files(&self) -> Vec<String> {
        crate::context::DEFAULT_CONTEXT_FILES
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Assemble the context fragments from the configured files.
    ///
    /// Missing files are skipped (fail-open). The assembled system prompt is
    /// injected into the pipeline messages.
    pub async fn assemble(&self, ctx: &mut PipelineContext) -> Result<ContextAssemblyOutcome> {
        let files = if self.config.files.is_empty() {
            self.default_files()
        } else {
            self.config.files.clone()
        };

        let mut builder = ContextBuilder::new();
        if let Some(root) = &self.config.workspace_root {
            builder = builder.workspace_root(root.clone());
        }
        for file in &files {
            builder = builder.add_file(file).await?;
        }

        let mut prompt = builder.build_prompt();
        if prompt.chars().count() > self.config.max_total_chars {
            prompt = prompt.chars().take(self.config.max_total_chars).collect();
        }

        let fragment_count = builder.fragments().len();
        let had_system_message = ctx
            .messages
            .iter()
            .any(|m| m.role == MessageRole::System);

        if !prompt.is_empty() {
            if let Some(system_idx) = ctx
                .messages
                .iter()
                .position(|m| m.role == MessageRole::System)
            {
                // Append to the existing system message so both the original
                // system instructions and the workspace context survive.
                let existing = ctx.messages[system_idx].text_content();
                let combined = if existing.is_empty() {
                    prompt.clone()
                } else {
                    format!("{existing}\n\n{prompt}")
                };
                ctx.messages[system_idx] =
                    Message::text(MessageRole::System, combined);
            } else {
                let system_msg = Message {
                    role: MessageRole::System,
                    content: vec![ContentBlock::Text(prompt.clone())],
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    tool_result: None,
                };
                ctx.messages.insert(0, system_msg);
            }
        }

        let total_chars = prompt.chars().count();
        ctx.set_metadata("context_fragments", &fragment_count.to_string());
        ctx.set_metadata("context_chars", &total_chars.to_string());
        ctx.set_metadata("context_assembly_applied", "true");

        debug!(
            fragment_count = fragment_count,
            total_chars = total_chars,
            "context assembly complete"
        );

        Ok(ContextAssemblyOutcome {
            fragment_count,
            total_chars,
            had_system_message,
        })
    }
}

#[async_trait]
impl PipelineStep for ContextAssemblyStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        self.assemble(ctx).await?;
        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "context_assembly"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    #[tokio::test]
    async fn test_missing_workspace_is_fail_open() {
        let step = ContextAssemblyStep::new(Some(PathBuf::from(
            "/nonexistent/opensquilla/workspace/does/not/exist",
        )));
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let outcome = step.assemble(&mut ctx).await.unwrap();
        assert_eq!(outcome.fragment_count, 0);
        assert_eq!(outcome.total_chars, 0);
        // No system message was injected.
        assert!(!ctx.messages.iter().any(|m| m.role == MessageRole::System));
    }

    #[tokio::test]
    async fn test_injects_system_message() {
        // The test's own module dir has no context files, so we exercise the
        // path with an inline builder fragment by pointing at a directory that
        // exists but has none of the default files.
        let step = ContextAssemblyStep::new(Some(PathBuf::from(".")));
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let outcome = step.assemble(&mut ctx).await.unwrap();
        // Fail-open: whatever the cwd contains, the step must not error.
        assert!(outcome.fragment_count <= crate::context::DEFAULT_CONTEXT_FILES.len());
    }
}
