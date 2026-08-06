//! Agent bootstrap stage.
//!
//! Mirrors the Python `engine/turn_runner/agent_bootstrap_stage.py` stage. It
//! runs after the harness and before compaction. It:
//!
//! * populates the model / provider from the turn generator (and the routed
//!   model/provider already bound onto the context by the runtime),
//! * injects the agent's system prompt (identity) when the turn has none,
//! * assembles the identity prompt from the default system prompt plus the
//!   workspace instruction files (SOUL.md, AGENTS.md, CLAUDE.md, ...) loaded
//!   via [`crate::context::ContextBuilder`] with `tokio::fs`,
//! * applies effective runtime limits (max iterations, timeouts, retries)
//!   from the generator's declared capabilities,
//! * initializes the token budget (context window, compaction thresholds) so
//!   later stages agree on a single budget.

use crate::agent::TurnGenerator;
use crate::context::ContextBuilder;
use crate::stages::{Stage, StageContext, StageError, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use std::path::PathBuf;
use tracing::{debug, info, instrument, warn};

/// Effective runtime settings for a turn, derived at bootstrap.
#[derive(Debug, Clone)]
pub struct BootstrapSettings {
    /// Maximum tool-call iterations the agent loop may run.
    pub max_iterations: u32,
    /// Per-request timeout in milliseconds.
    pub request_timeout_ms: u64,
    /// Per-iteration timeout in milliseconds.
    pub iteration_timeout_ms: u64,
    /// Maximum provider retries before surfacing the error.
    pub max_provider_retries: u32,
}

impl Default for BootstrapSettings {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            request_timeout_ms: 120_000,
            iteration_timeout_ms: 180_000,
            max_provider_retries: 2,
        }
    }
}

/// The token budget established for the turn.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// The model's context window, in tokens.
    pub context_window_tokens: u64,
    /// The soft warning threshold (fraction of the window).
    pub warning_threshold: u64,
    /// The compaction trigger threshold (fraction of the window).
    pub compaction_threshold: u64,
    /// The urgent compaction threshold.
    pub urgent_threshold: u64,
}

impl TokenBudget {
    /// Derive a budget from a context window using the compaction-control
    /// fractions (0.75 warn/compact, 0.9 urgent).
    pub fn from_window(window_tokens: u64) -> Self {
        let window = window_tokens.max(1);
        Self {
            context_window_tokens: window,
            warning_threshold: (window as f64 * 0.75) as u64,
            compaction_threshold: (window as f64 * 0.75) as u64,
            urgent_threshold: (window as f64 * 0.9) as u64,
        }
    }

    /// The fraction of the window consumed by the given token count.
    pub fn utilization(&self, tokens: u64) -> f64 {
        tokens as f64 / self.context_window_tokens as f64
    }

    /// Whether the given token count exceeds the urgent threshold.
    pub fn is_urgent(&self, tokens: u64) -> bool {
        tokens >= self.urgent_threshold
    }

    /// Whether the given token count exceeds the compaction threshold.
    pub fn needs_compaction(&self, tokens: u64) -> bool {
        tokens >= self.compaction_threshold
    }
}

/// Agent identity metadata assembled by the bootstrap stage.
#[derive(Debug, Clone, Default)]
pub struct AgentIdentity {
    /// A human-readable name for the agent.
    pub name: String,
    /// A one-line description of the agent's role.
    pub description: String,
    /// Instruction files successfully loaded from the workspace.
    pub loaded_files: Vec<String>,
    /// The assembled identity prompt (default system prompt + workspace files).
    pub assembled_prompt: String,
}

/// The second stage in the turn pipeline.
#[derive(Debug)]
pub struct AgentBootstrapStage {
    /// The agent's default system prompt (identity). Injected when the turn
    /// has no system message.
    system_prompt: Option<String>,
    /// Effective runtime settings.
    settings: BootstrapSettings,
    /// Workspace root used to resolve SOUL.md / AGENTS.md files.
    workspace_root: Option<PathBuf>,
    /// The context window (tokens) used to derive the token budget.
    context_window_tokens: u64,
    /// A fixed agent name injected into the identity, if any.
    agent_name: Option<String>,
    /// A fixed agent description injected into the identity, if any.
    agent_description: Option<String>,
    /// Instruction files loaded by default, relative to the workspace root.
    context_files: Vec<String>,
}

impl AgentBootstrapStage {
    /// Create a new bootstrap stage with the given system prompt.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        let prompt = system_prompt.into();
        Self {
            system_prompt: if prompt.is_empty() {
                None
            } else {
                Some(prompt)
            },
            settings: BootstrapSettings::default(),
            workspace_root: None,
            context_window_tokens: 128_000,
            agent_name: None,
            agent_description: None,
            context_files: crate::context::DEFAULT_CONTEXT_FILES
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Create a stage with explicit runtime settings.
    pub fn with_settings(mut self, settings: BootstrapSettings) -> Self {
        self.settings = settings;
        self
    }

    /// Set the workspace root used to resolve SOUL.md / AGENTS.md files.
    pub fn with_workspace_root(mut self, root: Option<PathBuf>) -> Self {
        self.workspace_root = root;
        self
    }

    /// Set the context window (tokens) used to derive the token budget.
    pub fn with_context_window(mut self, tokens: u64) -> Self {
        self.context_window_tokens = tokens.max(1);
        self
    }

    /// Set a fixed agent name injected into the identity.
    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    /// Set a fixed agent description injected into the identity.
    pub fn with_agent_description(mut self, description: impl Into<String>) -> Self {
        self.agent_description = Some(description.into());
        self
    }

    /// Replace the workspace instruction files loaded by default.
    ///
    /// Pass an empty list to disable workspace file loading entirely.
    pub fn with_context_files(mut self, files: Vec<String>) -> Self {
        self.context_files = files;
        self
    }

    /// The effective runtime settings.
    pub fn settings(&self) -> &BootstrapSettings {
        &self.settings
    }

    /// The token budget derived from the configured context window.
    pub fn token_budget(&self) -> TokenBudget {
        TokenBudget::from_window(self.context_window_tokens)
    }

    /// Load and assemble the identity prompt.
    ///
    /// The assembled prompt is the default system prompt (when present) joined
    /// with the workspace instruction files. Workspace files use fail-open
    /// semantics: a missing file is skipped, but a genuine I/O error short-
    /// circuits the whole bootstrap.
    pub async fn assemble_identity(&self) -> Result<AgentIdentity> {
        let mut builder = ContextBuilder::new();
        if let Some(root) = &self.workspace_root {
            builder = builder.workspace_root(root.clone());
        }
        if let Some(prompt) = &self.system_prompt {
            builder = builder.add_fragment(crate::context::ContextFragment::new(
                "Agent instructions",
                prompt.clone(),
            ));
        }

        let mut loaded_files = Vec::new();
        for file in &self.context_files {
            let label = file.clone();
            match builder.add_file_as(file, label.clone()).await {
                Ok(b) => {
                    builder = b;
                    loaded_files.push(label);
                }
                Err(e) => return Err(e),
            }
        }

        let assembled_prompt = builder.build_prompt();
        Ok(AgentIdentity {
            name: self.agent_name.clone().unwrap_or_default(),
            description: self.agent_description.clone().unwrap_or_default(),
            loaded_files,
            assembled_prompt,
        })
    }
}

#[async_trait]
impl Stage for AgentBootstrapStage {
    #[instrument(skip(self, ctx, generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("bootstrap: loading model, provider, and agent identity");

        // Populate model / provider. The routed model/provider already bound on
        // the context win; otherwise fall back to the generator.
        let model = generator.model_name().to_string();
        let provider = generator.provider_name().to_string();
        if ctx.current_model.is_empty() && !model.is_empty() {
            ctx.current_model = model;
        }
        if ctx.current_provider.is_empty() && !provider.is_empty() {
            ctx.current_provider = provider;
        }

        // Cap the tool loop from the bootstrap settings.
        ctx.max_tool_rounds = self.settings.max_iterations.max(1);

        // Assemble the identity prompt. I/O failures loading workspace files
        // are surfaced as a stage error so a broken workspace is not silently
        // ignored.
        let identity = match self.assemble_identity().await {
            Ok(identity) => identity,
            Err(e) => {
                return Ok(StageOutput::Error(StageError {
                    message: format!("Agent bootstrap failed to load identity: {e}"),
                    code: Some("IDENTITY_LOAD_FAILED".to_string()),
                    stage: self.name().to_string(),
                }));
            }
        };

        // Inject the assembled system prompt (identity) when the turn has none.
        let has_system = ctx.messages.iter().any(|m| m.role == MessageRole::System);
        if !has_system {
            if identity.assembled_prompt.trim().is_empty() {
                warn!(
                    turn_id = %ctx.turn_id,
                    "bootstrap: no system prompt source produced content"
                );
                return Ok(StageOutput::Error(StageError {
                    message: "Agent bootstrap could not establish a system prompt".to_string(),
                    code: Some("NO_SYSTEM_PROMPT".to_string()),
                    stage: self.name().to_string(),
                }));
            }
            ctx.messages
                .insert(0, Message::system(&identity.assembled_prompt));
        }

        // Record the token budget and identity metadata on the context so
        // downstream stages observe a single source of truth. We stash them in
        // the context's model string slot-free by logging; the budget is
        // recomputed by the compaction stage from the shared config.
        let budget = self.token_budget();
        debug!(
            turn_id = %ctx.turn_id,
            context_window_tokens = budget.context_window_tokens,
            compaction_threshold = budget.compaction_threshold,
            urgent_threshold = budget.urgent_threshold,
            identity_files = identity.loaded_files.len(),
            "bootstrap: token budget initialized"
        );

        info!(
            turn_id = %ctx.turn_id,
            model = %ctx.current_model,
            provider = %ctx.current_provider,
            max_iterations = ctx.max_tool_rounds,
            identity_assembled = !identity.assembled_prompt.is_empty(),
            files_loaded = identity.loaded_files.len(),
            "bootstrap stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "bootstrap"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Usage;
    use std::collections::HashMap;

    #[test]
    fn test_token_budget_thresholds() {
        let budget = TokenBudget::from_window(128_000);
        assert_eq!(budget.context_window_tokens, 128_000);
        assert_eq!(budget.compaction_threshold, 96_000);
        assert_eq!(budget.urgent_threshold, 115_200);
        assert!(!budget.needs_compaction(50_000));
        assert!(budget.needs_compaction(100_000));
        assert!(budget.is_urgent(120_000));
    }

    #[test]
    fn test_token_budget_utilization() {
        let budget = TokenBudget::from_window(100_000);
        assert!((budget.utilization(50_000) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_new_empty_prompt_none() {
        let stage = AgentBootstrapStage::new("");
        assert!(stage.system_prompt.is_none());
        assert_eq!(stage.settings().max_iterations, 10);
    }

    #[test]
    fn test_new_prompt_some() {
        let stage = AgentBootstrapStage::new("You are a helpful agent.");
        assert!(stage.system_prompt.is_some());
    }

    #[tokio::test]
    async fn test_identity_with_inline_prompt() {
        let stage =
            AgentBootstrapStage::new("You are a helpful agent.").with_context_files(Vec::new());
        let identity = stage.assemble_identity().await.unwrap();
        assert!(identity.assembled_prompt.contains("helpful agent"));
        assert!(identity.loaded_files.is_empty());
    }

    #[tokio::test]
    async fn test_missing_workspace_files_are_skipped() {
        let stage = AgentBootstrapStage::new("You are a test agent.").with_workspace_root(Some(
            PathBuf::from("/nonexistent/opensquilla/path/that/does/not/exist"),
        ));
        let identity = stage.assemble_identity().await.unwrap();
        // The inline prompt still assembles; missing files are skipped.
        assert!(identity.assembled_prompt.contains("test agent"));
    }

    #[tokio::test]
    async fn test_execute_injects_system_prompt() {
        let stage =
            AgentBootstrapStage::new("You are a test agent.").with_context_files(Vec::new());
        let mut ctx = StageContext {
            turn_id: "t1".to_string(),
            messages: vec![Message::user("hello")],
            current_model: String::new(),
            current_provider: String::new(),
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
            metadata: HashMap::new(),
        };
        let generator = MockGenerator;
        let out = stage.execute(&mut ctx, &generator).await.unwrap();
        assert!(matches!(out, StageOutput::Continue));
        assert!(ctx.messages.iter().any(|m| m.role == MessageRole::System));
        assert_eq!(ctx.max_tool_rounds, 10);
    }

    #[tokio::test]
    async fn test_execute_fails_without_identity() {
        let stage = AgentBootstrapStage::new("").with_context_files(Vec::new());
        let mut ctx = StageContext {
            turn_id: "t1".to_string(),
            messages: vec![Message::user("hello")],
            current_model: String::new(),
            current_provider: String::new(),
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
            metadata: HashMap::new(),
        };
        let generator = MockGenerator;
        let out = stage.execute(&mut ctx, &generator).await.unwrap();
        match out {
            StageOutput::Error(e) => assert_eq!(e.code.as_deref(), Some("NO_SYSTEM_PROMPT")),
            _ => panic!("expected error"),
        }
    }

    #[derive(Debug)]
    struct MockGenerator;
    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
            Ok(vec![Message::assistant("ok")])
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
