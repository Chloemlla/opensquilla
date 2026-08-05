//! Context assembly: load SOUL.md, AGENTS.md, and workspace files.
//!
//! The `ContextBuilder` assembles a system prompt from multiple sources:
//! workspace-level instruction files (SOUL.md, AGENTS.md), project memory,
//! and ad-hoc fragments. All file I/O is performed via `tokio::fs` so the
//! builder is fully async and non-blocking.

use opensquilla_core::error::{Error, Result};
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// A fragment of context that can be assembled into a system prompt.
#[derive(Debug, Clone)]
pub struct ContextFragment {
    /// A human-readable label for the fragment (e.g. "SOUL.md", "AGENTS.md").
    pub label: String,
    /// The text content of the fragment.
    pub content: String,
    /// The priority of this fragment. Higher-priority fragments appear earlier
    /// in the assembled system prompt.
    pub priority: i32,
}

impl ContextFragment {
    /// Create a new context fragment with default priority 0.
    pub fn new(label: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            content: content.into(),
            priority: 0,
        }
    }

    /// Set the priority of this fragment.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// The set of workspace instruction files loaded by default, in load order.
///
/// These mirror the Python backend's `context.py` behavior: SOUL.md provides
/// the core persona, AGENTS.md provides per-agent instructions, and
/// CLAUDE.md / .cursorrules provide editor-injected context.
pub const DEFAULT_CONTEXT_FILES: &[&str] = &[
    "SOUL.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursorrules",
    "GEMINI.md",
];

/// Builder that assembles a system prompt from multiple context sources.
///
/// Fragments are collected from inline strings and files, then sorted by
/// descending priority before being joined into the final system message.
#[derive(Debug, Default)]
pub struct ContextBuilder {
    /// The workspace root used to resolve relative context file paths.
    workspace_root: Option<PathBuf>,
    /// The collected context fragments.
    fragments: Vec<ContextFragment>,
}

impl ContextBuilder {
    /// Create a new empty context builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the workspace root used to resolve relative context file paths.
    pub fn workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// Add an inline context fragment.
    pub fn add_fragment(mut self, fragment: ContextFragment) -> Self {
        self.fragments.push(fragment);
        self
    }

    /// Add an inline text fragment with a label and default priority.
    pub fn add_text(self, label: impl Into<String>, content: impl Into<String>) -> Self {
        self.add_fragment(ContextFragment::new(label, content))
    }

    /// Load a single context file relative to the workspace root.
    ///
    /// Missing files are silently skipped (returning `Ok`), matching the
    /// Python backend's fail-open semantics for optional instruction files.
    /// Files that exist but cannot be read produce an error.
    pub async fn add_file(self, relative_path: impl AsRef<Path>) -> Result<Self> {
        let path = relative_path.as_ref();
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();

        self.add_file_as(path, label).await
    }

    /// Load a single context file with an explicit label.
    pub async fn add_file_as(
        mut self,
        relative_path: impl AsRef<Path>,
        label: impl Into<String>,
    ) -> Result<Self> {
        let path = relative_path.as_ref();
        let resolved = match &self.workspace_root {
            Some(root) => root.join(path),
            None => path.to_path_buf(),
        };

        match tokio::fs::read_to_string(&resolved).await {
            Ok(content) => {
                debug!(
                    file = %resolved.display(),
                    bytes = content.len(),
                    "Loaded context file"
                );
                self.fragments
                    .push(ContextFragment::new(label, content));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!(file = %resolved.display(), "Context file not found, skipping");
            }
            Err(e) => {
                warn!(file = %resolved.display(), error = %e, "Failed to read context file");
                return Err(Error::Io(e));
            }
        }

        Ok(self)
    }

    /// Load all default workspace instruction files (SOUL.md, AGENTS.md, ...).
    ///
    /// Each file is loaded with fail-open semantics: missing files are skipped,
    /// but a genuine I/O error short-circuits the whole load.
    pub async fn add_default_files(self) -> Result<Self> {
        let mut builder = self;
        for &file in DEFAULT_CONTEXT_FILES {
            builder = builder.add_file(file).await?;
        }
        Ok(builder)
    }

    /// Assemble the collected fragments into a single system prompt string.
    ///
    /// Fragments are sorted by descending priority, then joined with a
    /// section header derived from each fragment's label.
    pub fn build_prompt(&self) -> String {
        let mut sorted = self.fragments.clone();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut out = String::new();
        for fragment in &sorted {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            if !fragment.label.is_empty() {
                out.push_str("# ");
                out.push_str(&fragment.label);
                out.push_str("\n");
            }
            out.push_str(&fragment.content);
        }
        out
    }

    /// Assemble the collected fragments into a system `Message`.
    pub fn build_message(&self) -> Message {
        Message {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(self.build_prompt())],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    /// Get an immutable view of the collected fragments.
    pub fn fragments(&self) -> &[ContextFragment] {
        &self.fragments
    }

    /// Returns true if no fragments have been collected.
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragment_priority_sorting() {
        let builder = ContextBuilder::new()
            .add_fragment(ContextFragment::new("low", "low content").with_priority(1))
            .add_fragment(ContextFragment::new("high", "high content").with_priority(10))
            .add_fragment(ContextFragment::new("mid", "mid content").with_priority(5));

        let prompt = builder.build_prompt();
        let high_idx = prompt.find("high content").unwrap();
        let mid_idx = prompt.find("mid content").unwrap();
        let low_idx = prompt.find("low content").unwrap();
        assert!(high_idx < mid_idx);
        assert!(mid_idx < low_idx);
    }

    #[test]
    fn test_build_message_is_system() {
        let builder = ContextBuilder::new().add_text("label", "body");
        let msg = builder.build_message();
        assert_eq!(msg.role, MessageRole::System);
        assert_eq!(msg.text_content(), "# label\nbody");
    }

    #[tokio::test]
    async fn test_missing_file_is_skipped() {
        let builder = ContextBuilder::new()
            .workspace_root("/nonexistent/opensquilla/path/that/does/not/exist")
            .add_file("SOUL.md")
            .await
            .unwrap();
        assert!(builder.is_empty());
    }
}
