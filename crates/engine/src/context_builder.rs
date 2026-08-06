//! Context builder: system prompt assembly, context window management,
//! and token budget allocation.
//!
//! This module extends [`crate::context::ContextBuilder`] with structured
//! system-prompt assembly, token-budget allocation across system/user/assistant
//! /tools, and context-window management. It mirrors the Python backend's
//! `engine/context_builder.py` — the layer that turns a set of instruction
//! fragments into a bounded, prioritized system prompt and reserves token
//! budget for each conversation role.

use crate::context::{ContextBuilder, ContextFragment, DEFAULT_CONTEXT_FILES};
use opensquilla_core::error::Result;
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// A labeled section of the system prompt.
///
/// Sections are ordered by [`SystemPromptSection::priority`] (descending)
/// within the assembled prompt. Each section carries a stable key so the
/// assembler can detect duplicate injections (e.g. the same skills block
/// injected by two pipeline steps).
#[derive(Debug, Clone)]
pub struct SystemPromptSection {
    /// A stable, unique key for this section (e.g. `"identity"`, `"skills"`,
    /// `"tools"`, `"workspace_context"`).
    pub key: String,
    /// A human-readable label rendered as a markdown heading.
    pub label: String,
    /// The section body text.
    pub body: String,
    /// Higher-priority sections appear earlier in the assembled prompt.
    pub priority: i32,
    /// Whether the section is required (never truncated even when over budget).
    pub required: bool,
    /// Maximum characters this section may contribute. `0` means unbounded.
    pub max_chars: usize,
}

impl SystemPromptSection {
    /// Create a new section with default priority 0 and no char cap.
    pub fn new(key: impl Into<String>, label: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            body: body.into(),
            priority: 0,
            required: false,
            max_chars: 0,
        }
    }

    /// Set the priority (higher appears earlier).
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Mark this section as required (never truncated).
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set the maximum characters this section may contribute.
    pub fn with_max_chars(mut self, max: usize) -> Self {
        self.max_chars = max;
        self
    }

    /// The effective body, truncated to `max_chars` when set.
    pub fn effective_body(&self) -> &str {
        if self.max_chars == 0 {
            return &self.body;
        }
        match self.body.char_indices().nth(self.max_chars) {
            Some((idx, _)) => &self.body[..idx],
            None => &self.body,
        }
    }

    /// The character count of the effective body.
    pub fn effective_chars(&self) -> usize {
        self.effective_body().chars().count()
    }
}

/// Token budget allocation across conversation roles.
///
/// The budget is derived from the model's context window and allocates
/// fractions to system prompt, conversation history, tool definitions, and
/// output reservation. The fractions are configurable but default to the
/// Python backend's allocation.
#[derive(Debug, Clone)]
pub struct TokenBudgetAllocation {
    /// The total context window in tokens.
    pub context_window: u64,
    /// Fraction reserved for the system prompt (default 0.15).
    pub system_fraction: f64,
    /// Fraction reserved for conversation history (default 0.50).
    pub history_fraction: f64,
    /// Fraction reserved for tool definitions (default 0.10).
    pub tools_fraction: f64,
    /// Fraction reserved for output generation (default 0.25).
    pub output_fraction: f64,
}

impl Default for TokenBudgetAllocation {
    fn default() -> Self {
        Self {
            context_window: 128_000,
            system_fraction: 0.15,
            history_fraction: 0.50,
            tools_fraction: 0.10,
            output_fraction: 0.25,
        }
    }
}

impl TokenBudgetAllocation {
    /// Create a new allocation for the given context window.
    pub fn new(context_window: u64) -> Self {
        Self {
            context_window: context_window.max(1),
            ..Default::default()
        }
    }

    /// Set the system prompt fraction.
    pub fn with_system_fraction(mut self, frac: f64) -> Self {
        self.system_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Set the history fraction.
    pub fn with_history_fraction(mut self, frac: f64) -> Self {
        self.history_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Set the tools fraction.
    pub fn with_tools_fraction(mut self, frac: f64) -> Self {
        self.tools_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Set the output fraction.
    pub fn with_output_fraction(mut self, frac: f64) -> Self {
        self.output_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// The token budget for the system prompt.
    pub fn system_tokens(&self) -> u64 {
        ((self.context_window as f64) * self.system_fraction) as u64
    }

    /// The token budget for conversation history.
    pub fn history_tokens(&self) -> u64 {
        ((self.context_window as f64) * self.history_fraction) as u64
    }

    /// The token budget for tool definitions.
    pub fn tools_tokens(&self) -> u64 {
        ((self.context_window as f64) * self.tools_fraction) as u64
    }

    /// The token budget reserved for output generation.
    pub fn output_tokens(&self) -> u64 {
        ((self.context_window as f64) * self.output_fraction) as u64
    }

    /// The total allocated tokens (should not exceed the context window).
    pub fn total_allocated(&self) -> u64 {
        self.system_tokens() + self.history_tokens() + self.tools_tokens() + self.output_tokens()
    }

    /// The remaining unallocated tokens.
    pub fn remaining(&self) -> u64 {
        self.context_window.saturating_sub(self.total_allocated())
    }

    /// Validate that the fractions sum to at most 1.0.
    pub fn validate(&self) -> Result<()> {
        let total = self.system_fraction + self.history_fraction + self.tools_fraction + self.output_fraction;
        if total > 1.0 + 1e-9 {
            return Err(opensquilla_core::error::Error::InvalidInput(format!(
                "Token budget fractions sum to {total:.3}, which exceeds 1.0"
            )));
        }
        Ok(())
    }

    /// Normalize fractions so they sum to exactly 1.0, scaling proportionally.
    pub fn normalized(mut self) -> Self {
        let total = self.system_fraction + self.history_fraction + self.tools_fraction + self.output_fraction;
        if total <= 0.0 {
            return Self::default();
        }
        if (total - 1.0).abs() < 1e-9 {
            return self;
        }
        let scale = 1.0 / total;
        self.system_fraction *= scale;
        self.history_fraction *= scale;
        self.tools_fraction *= scale;
        self.output_fraction *= scale;
        self
    }

    /// Create an allocation from a context window with explicit role ratios.
    ///
    /// The ratios are normalized so they sum to at most 1.0.
    pub fn from_window_with_ratios(
        context_window: u64,
        system: f64,
        history: f64,
        tools: f64,
        output: f64,
    ) -> Self {
        let total = system + history + tools + output;
        let (s, h, t, o) = if total > 1.0 {
            (system / total, history / total, tools / total, output / total)
        } else {
            (system, history, tools, output)
        };
        Self {
            context_window: context_window.max(1),
            system_fraction: s.clamp(0.0, 1.0),
            history_fraction: h.clamp(0.0, 1.0),
            tools_fraction: t.clamp(0.0, 1.0),
            output_fraction: o.clamp(0.0, 1.0),
        }
    }

    /// The recommended maximum output tokens for a request, derived from the
    /// output reservation.
    pub fn recommended_max_output_tokens(&self) -> u32 {
        self.output_tokens().min(u32::MAX as u64) as u32
    }
}

/// A token estimator trait, so the context builder can use an exact tokenizer
/// when available and fall back to a character-based heuristic otherwise.
pub trait TokenEstimator: Send + Sync + std::fmt::Debug {
    /// Estimate the token count of a text string.
    fn estimate_text(&self, text: &str) -> u64;

    /// Estimate the token count of a message (sum of all content blocks).
    fn estimate_message(&self, message: &Message) -> u64 {
        let mut total = 0u64;
        for block in &message.content {
            match block {
                ContentBlock::Text(t) => total += self.estimate_text(t),
                ContentBlock::Reasoning(r) => total += self.estimate_text(r),
                ContentBlock::ToolUse(c) => {
                    total += self.estimate_text(&c.name);
                    total += self.estimate_text(&c.input.to_string());
                }
                ContentBlock::ToolResult(r) => total += self.estimate_text(&r.content),
            }
        }
        total
    }

    /// Estimate the total token count across a message slice.
    fn estimate_messages(&self, messages: &[Message]) -> u64 {
        messages.iter().map(|m| self.estimate_message(m)).sum()
    }
}

/// A character-based token estimator: roughly 1 token per 4 characters.
#[derive(Debug, Clone, Copy)]
pub struct CharTokenEstimator {
    /// The number of characters per token (default 4).
    pub chars_per_token: f64,
}

impl CharTokenEstimator {
    /// Create a new estimator with the default 4 chars/token ratio.
    pub fn new() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }

    /// Create an estimator with a custom chars/token ratio.
    pub fn with_ratio(chars_per_token: f64) -> Self {
        Self {
            chars_per_token: chars_per_token.max(0.1),
        }
    }
}

impl Default for CharTokenEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenEstimator for CharTokenEstimator {
    fn estimate_text(&self, text: &str) -> u64 {
        (text.chars().count() as f64 / self.chars_per_token).ceil() as u64
    }
}

/// A context-window manager that tracks token usage across roles and enforces
/// the budget by truncating history when needed.
#[derive(Debug, Clone)]
pub struct ContextWindowManager {
    /// The token budget allocation.
    budget: TokenBudgetAllocation,
    /// The token estimator.
    estimator: CharTokenEstimator,
    /// Current system prompt token count (recomputed on each assembly).
    system_tokens: u64,
    /// Current history token count.
    history_tokens: u64,
    /// Current tools token count.
    tools_tokens: u64,
    /// Reserved output tokens.
    output_tokens: u64,
}

impl ContextWindowManager {
    /// Create a new context-window manager with the given context window.
    pub fn new(context_window: u64) -> Self {
        Self {
            budget: TokenBudgetAllocation::new(context_window),
            estimator: CharTokenEstimator::new(),
            system_tokens: 0,
            history_tokens: 0,
            tools_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Set the token budget allocation.
    pub fn with_budget(mut self, budget: TokenBudgetAllocation) -> Self {
        self.budget = budget;
        self
    }

    /// Set the token estimator ratio.
    pub fn with_estimator_ratio(mut self, chars_per_token: f64) -> Self {
        self.estimator = CharTokenEstimator::with_ratio(chars_per_token);
        self
    }

    /// The configured token budget.
    pub fn budget(&self) -> &TokenBudgetAllocation {
        &self.budget
    }

    /// The current system prompt token count.
    pub fn system_tokens(&self) -> u64 {
        self.system_tokens
    }

    /// The current history token count.
    pub fn history_tokens(&self) -> u64 {
        self.history_tokens
    }

    /// The current tools token count.
    pub fn tools_tokens(&self) -> u64 {
        self.tools_tokens
    }

    /// The reserved output tokens.
    pub fn output_tokens(&self) -> u64 {
        self.output_tokens
    }

    /// Record the system prompt token count.
    pub fn set_system_tokens(&mut self, tokens: u64) {
        self.system_tokens = tokens;
    }

    /// Record the tools token count.
    pub fn set_tools_tokens(&mut self, tokens: u64) {
        self.tools_tokens = tokens;
    }

    /// Record the output token reservation.
    pub fn set_output_tokens(&mut self, tokens: u64) {
        self.output_tokens = tokens;
    }

    /// The total tokens currently consumed (system + history + tools + output).
    pub fn total_consumed(&self) -> u64 {
        self.system_tokens + self.history_tokens + self.tools_tokens + self.output_tokens
    }

    /// The remaining tokens in the context window.
    pub fn remaining(&self) -> u64 {
        self.budget.context_window.saturating_sub(self.total_consumed())
    }

    /// The available tokens for history, given the current system/tools/output.
    pub fn available_history_tokens(&self) -> u64 {
        let reserved = self.system_tokens + self.tools_tokens + self.output_tokens;
        self.budget.context_window.saturating_sub(reserved)
    }

    /// Update the history token count from a message slice.
    pub fn update_history(&mut self, messages: &[Message]) -> u64 {
        self.history_tokens = self.estimator.estimate_messages(messages);
        self.history_tokens
    }

    /// Whether the history exceeds its allocated budget.
    pub fn history_over_budget(&self) -> bool {
        let available = self.available_history_tokens();
        self.history_tokens > available
    }

    /// The number of tokens that must be reclaimed from the history to fit
    /// the budget.
    pub fn history_overage(&self) -> u64 {
        let available = self.available_history_tokens();
        self.history_tokens.saturating_sub(available)
    }

    /// Truncate a message list to fit within the available history budget.
    ///
    /// System messages are always preserved. The most recent non-system
    /// messages are retained until the budget is exhausted.
    pub fn truncate_history(&self, messages: &[Message]) -> Vec<Message> {
        let available = self.available_history_tokens();
        if available == 0 {
            return messages
                .iter()
                .filter(|m| m.role == MessageRole::System)
                .cloned()
                .collect();
        }

        let mut system: Vec<Message> = Vec::new();
        let mut tail: Vec<Message> = Vec::new();
        for msg in messages {
            if matches!(msg.role, MessageRole::System) {
                system.push(msg.clone());
            } else {
                tail.push(msg.clone());
            }
        }

        let system_tokens = self.estimator.estimate_messages(&system);
        let mut budget = available.saturating_sub(system_tokens);

        let mut kept: Vec<Message> = Vec::new();
        for msg in tail.into_iter().rev() {
            let cost = self.estimator.estimate_message(&msg);
            if cost > budget {
                if kept.is_empty() {
                    kept.push(msg);
                }
                break;
            }
            budget -= cost;
            kept.push(msg);
        }
        kept.reverse();

        system.extend(kept);
        system
    }
}

/// The system prompt assembler.
///
/// Collects labeled sections, sorts them by priority, deduplicates by key,
/// and renders a bounded system prompt. When the assembled prompt exceeds
/// the system token budget, non-required sections are truncated (lowest
/// priority first) until it fits.
#[derive(Debug, Clone)]
pub struct SystemPromptAssembler {
    /// The collected sections, keyed by their stable key.
    sections: HashMap<String, SystemPromptSection>,
    /// The token budget for the system prompt.
    max_tokens: u64,
    /// The token estimator.
    estimator: CharTokenEstimator,
    /// The separator between sections.
    separator: String,
}

impl SystemPromptAssembler {
    /// Create a new assembler with the given token budget.
    pub fn new(max_tokens: u64) -> Self {
        Self {
            sections: HashMap::new(),
            max_tokens: max_tokens.max(1),
            estimator: CharTokenEstimator::new(),
            separator: "\n\n".to_string(),
        }
    }

    /// Create a new assembler from a budget allocation (uses system_tokens).
    pub fn from_budget(budget: &TokenBudgetAllocation) -> Self {
        Self::new(budget.system_tokens())
    }

    /// Add or replace a section.
    pub fn add_section(&mut self, section: SystemPromptSection) {
        self.sections.insert(section.key.clone(), section);
    }

    /// Remove a section by key.
    pub fn remove_section(&mut self, key: &str) -> Option<SystemPromptSection> {
        self.sections.remove(key)
    }

    /// Get a section by key.
    pub fn get_section(&self, key: &str) -> Option<&SystemPromptSection> {
        self.sections.get(key)
    }

    /// The number of sections registered.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Set the separator between sections.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Set the token estimator ratio.
    pub fn with_estimator_ratio(mut self, chars_per_token: f64) -> Self {
        self.estimator = CharTokenEstimator::with_ratio(chars_per_token);
        self
    }

    /// Assemble the system prompt from the registered sections.
    ///
    /// Sections are sorted by descending priority. Required sections are
    /// always included; non-required sections are truncated (lowest priority
    /// first) until the assembled prompt fits within the token budget.
    pub fn assemble(&self) -> String {
        let mut sorted: Vec<&SystemPromptSection> = self.sections.values().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));

        let required: Vec<&SystemPromptSection> = sorted.iter().copied().filter(|s| s.required).collect();
        let optional: Vec<&SystemPromptSection> = sorted.iter().copied().filter(|s| !s.required).collect();

        let required_tokens: u64 = required.iter().map(|s| self.estimator.estimate_text(s.effective_body())).sum();
        let mut remaining_budget = self.max_tokens.saturating_sub(required_tokens);

        // Truncate optional sections (lowest priority first) until they fit.
        let mut truncated: Vec<SystemPromptSection> = Vec::new();
        for section in optional.iter().rev() {
            let body = section.effective_body();
            let cost = self.estimator.estimate_text(body);
            if cost <= remaining_budget {
                remaining_budget -= cost;
                truncated.push((*section).clone());
            } else {
                // Try to fit a truncated version of this section.
                let affordable_tokens = remaining_budget;
                if affordable_tokens > 10 {
                    let affordable_chars = (affordable_tokens as f64 * self.estimator.chars_per_token) as usize;
                    let truncated_body: String = body.chars().take(affordable_chars).collect();
                    if !truncated_body.is_empty() {
                        let mut truncated_section = (*section).clone();
                        truncated_section.body = format!("{truncated_body}\n[...truncated]");
                        truncated.push(truncated_section);
                    }
                }
                break;
            }
        }
        truncated.reverse();

        let mut all_sections: Vec<&SystemPromptSection> = required.clone();
        all_sections.extend(truncated.iter());

        let mut out = String::new();
        for section in &all_sections {
            if !out.is_empty() {
                out.push_str(&self.separator);
            }
            if !section.label.is_empty() {
                out.push_str("# ");
                out.push_str(&section.label);
                out.push('\n');
            }
            out.push_str(section.effective_body());
        }
        out
    }

    /// Assemble the system prompt and return it as a system [`Message`].
    pub fn assemble_message(&self) -> Message {
        Message {
            role: MessageRole::System,
            content: vec![ContentBlock::Text(self.assemble())],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    /// The estimated token count of the assembled prompt.
    pub fn assembled_tokens(&self) -> u64 {
        self.estimator.estimate_text(&self.assemble())
    }
}

/// A high-level context builder that ties together system-prompt assembly,
/// workspace file loading, and token-budget management.
///
/// This is the main entry point for the `context_assembly` pipeline step and
/// the `agent_bootstrap` stage. It wraps [`crate::context::ContextBuilder`]
/// with structured section management and token-aware truncation.
#[derive(Debug, Clone)]
pub struct ContextPromptBuilder {
    /// The system prompt assembler.
    assembler: SystemPromptAssembler,
    /// The context-window manager.
    window_manager: ContextWindowManager,
    /// The workspace root for resolving instruction files.
    workspace_root: Option<PathBuf>,
    /// Instruction files loaded by default.
    context_files: Vec<String>,
}

impl ContextPromptBuilder {
    /// Create a new context prompt builder for the given context window.
    pub fn new(context_window: u64) -> Self {
        let budget = TokenBudgetAllocation::new(context_window);
        let assembler = SystemPromptAssembler::from_budget(&budget);
        let window_manager = ContextWindowManager::new(context_window)
            .with_budget(budget.clone());
        Self {
            assembler,
            window_manager,
            workspace_root: None,
            context_files: DEFAULT_CONTEXT_FILES.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Set the workspace root.
    pub fn with_workspace_root(mut self, root: PathBuf) -> Self {
        self.workspace_root = Some(root);
        self
    }

    /// Set the context files to load.
    pub fn with_context_files(mut self, files: Vec<String>) -> Self {
        self.context_files = files;
        self
    }

    /// Set the system prompt fraction of the token budget.
    pub fn with_system_fraction(mut self, frac: f64) -> Self {
        let budget = self.window_manager.budget.clone().with_system_fraction(frac);
        self.window_manager = self.window_manager.with_budget(budget.clone());
        self.assembler = SystemPromptAssembler::from_budget(&budget);
        self
    }

    /// Add a section to the system prompt assembler.
    pub fn add_section(&mut self, section: SystemPromptSection) {
        self.assembler.add_section(section);
    }

    /// Load workspace instruction files as sections.
    ///
    /// Each file becomes a section keyed by its filename. Missing files are
    /// silently skipped (fail-open).
    pub async fn load_workspace_files(&mut self) -> Result<()> {
        for file in &self.context_files {
            let path = match &self.workspace_root {
                Some(root) => root.join(file),
                None => PathBuf::from(file),
            };
            match tokio::fs::read_to_string(&path).await {
                Ok(content) => {
                    let label = Path::new(file)
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or(file)
                        .to_string();
                    let section = SystemPromptSection::new(file.clone(), label, content)
                        .with_priority(10);
                    self.assembler.add_section(section);
                    debug!(file = %file, "loaded workspace instruction file");
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    debug!(file = %file, "workspace instruction file not found, skipping");
                }
                Err(e) => {
                    warn!(file = %file, error = %e, "failed to read workspace instruction file");
                    return Err(opensquilla_core::error::Error::Io(e));
                }
            }
        }
        Ok(())
    }

    /// Assemble the final system prompt.
    pub fn assemble_prompt(&self) -> String {
        self.assembler.assemble()
    }

    /// Assemble the system prompt as a [`Message`].
    pub fn assemble_message(&self) -> Message {
        self.assembler.assemble_message()
    }

    /// The estimated token count of the assembled system prompt.
    pub fn system_prompt_tokens(&self) -> u64 {
        self.assembler.assembled_tokens()
    }

    /// The context-window manager.
    pub fn window_manager(&self) -> &ContextWindowManager {
        &self.window_manager
    }

    /// Get a mutable reference to the context-window manager.
    pub fn window_manager_mut(&mut self) -> &mut ContextWindowManager {
        &mut self.window_manager
    }

    /// The system-prompt assembler.
    pub fn assembler(&self) -> &SystemPromptAssembler {
        &self.assembler
    }

    /// Get a mutable reference to the assembler.
    pub fn assembler_mut(&mut self) -> &mut SystemPromptAssembler {
        &mut self.assembler
    }

    /// Build a complete context: system prompt + truncated history, ready
    /// for the provider.
    pub fn build_context(&mut self, messages: &[Message]) -> (Message, Vec<Message>) {
        let system_prompt = self.assemble_prompt();
        let system_tokens = self.window_manager.estimator.estimate_text(&system_prompt);
        self.window_manager.set_system_tokens(system_tokens);

        // Separate system messages from the rest.
        let mut history: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .cloned()
            .collect();

        // Update the history token count.
        self.window_manager.update_history(&history);

        // Truncate if over budget.
        if self.window_manager.history_over_budget() {
            info!(
                overage = self.window_manager.history_overage(),
                "history over budget, truncating"
            );
            history = self.window_manager.truncate_history(messages);
            self.window_manager.update_history(&history);
        }

        let system_message = Message::system(system_prompt);
        (system_message, history)
    }

    /// Build the complete context as a single message list.
    pub fn build_messages(&mut self, messages: &[Message]) -> Vec<Message> {
        let (system, history) = self.build_context(messages);
        let mut out = vec![system];
        out.extend(history);
        out
    }
}

/// A builder for creating context fragments from various sources.
#[derive(Debug, Default)]
pub struct FragmentBuilder {
    /// The collected fragments.
    fragments: Vec<ContextFragment>,
}

impl FragmentBuilder {
    /// Create a new empty fragment builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a text fragment.
    pub fn add_text(mut self, label: impl Into<String>, content: impl Into<String>) -> Self {
        self.fragments.push(ContextFragment::new(label, content));
        self
    }

    /// Add a text fragment with priority.
    pub fn add_text_with_priority(
        mut self,
        label: impl Into<String>,
        content: impl Into<String>,
        priority: i32,
    ) -> Self {
        self.fragments
            .push(ContextFragment::new(label, content).with_priority(priority));
        self
    }

    /// Load a file as a fragment.
    pub async fn add_file(mut self, path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file")
            .to_string();
        match tokio::fs::read_to_string(path).await {
            Ok(content) => {
                self.fragments.push(ContextFragment::new(label, content));
                Ok(self)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(self),
            Err(e) => Err(opensquilla_core::error::Error::Io(e)),
        }
    }

    /// Build the assembled prompt from the collected fragments.
    pub fn build(self) -> String {
        let builder = self.fragments.into_iter().fold(ContextBuilder::new(), |b, f| b.add_fragment(f));
        builder.build_prompt()
    }

    /// Build the assembled prompt as a system [`Message`].
    pub fn build_message(self) -> Message {
        let builder = self.fragments.into_iter().fold(ContextBuilder::new(), |b, f| b.add_fragment(f));
        builder.build_message()
    }

    /// The number of fragments collected.
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// True when no fragments have been collected.
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_allocation() {
        let budget = TokenBudgetAllocation::new(128_000);
        assert_eq!(budget.system_tokens(), 19_200); // 15%
        assert_eq!(budget.history_tokens(), 64_000); // 50%
        assert_eq!(budget.tools_tokens(), 12_800); // 10%
        assert_eq!(budget.output_tokens(), 32_000); // 25%
        assert!(budget.validate().is_ok());
    }

    #[test]
    fn test_token_budget_normalization() {
        let budget = TokenBudgetAllocation::new(100_000)
            .with_system_fraction(0.3)
            .with_history_fraction(0.3)
            .with_tools_fraction(0.3)
            .with_output_fraction(0.3);
        // Total is 1.2; normalization brings it to 1.0.
        let normalized = budget.normalized();
        let total = normalized.system_fraction
            + normalized.history_fraction
            + normalized.tools_fraction
            + normalized.output_fraction;
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_char_token_estimator() {
        let est = CharTokenEstimator::new();
        assert_eq!(est.estimate_text("hello world"), 3); // 11 chars / 4 = 2.75 -> 3
        assert_eq!(est.estimate_text(""), 0);
    }

    #[test]
    fn test_system_prompt_assembler_priority_order() {
        let mut assembler = SystemPromptAssembler::new(100_000);
        assembler.add_section(
            SystemPromptSection::new("low", "Low", "low priority body")
                .with_priority(1),
        );
        assembler.add_section(
            SystemPromptSection::new("high", "High", "high priority body")
                .with_priority(10),
        );
        assembler.add_section(
            SystemPromptSection::new("mid", "Mid", "mid priority body")
                .with_priority(5),
        );
        let prompt = assembler.assemble();
        let high_idx = prompt.find("high priority body").unwrap();
        let mid_idx = prompt.find("mid priority body").unwrap();
        let low_idx = prompt.find("low priority body").unwrap();
        assert!(high_idx < mid_idx);
        assert!(mid_idx < low_idx);
    }

    #[test]
    fn test_assembler_truncates_non_required() {
        let mut assembler = SystemPromptAssembler::new(50); // very small budget
        assembler.add_section(
            SystemPromptSection::new("required", "Required", "must keep this")
                .with_priority(100)
                .required(),
        );
        assembler.add_section(
            SystemPromptSection::new("optional", "Optional", "this is a very long optional section that will not fit")
                .with_priority(1),
        );
        let prompt = assembler.assemble();
        assert!(prompt.contains("must keep this"));
    }

    #[test]
    fn test_context_window_manager_truncate_history() {
        let manager = ContextWindowManager::new(1000)
            .with_estimator_ratio(1.0); // 1 char = 1 token for testing
        let messages: Vec<Message> = (0..100)
            .map(|i| Message::user(format!("message {i} that is fairly long")))
            .collect();
        let truncated = manager.truncate_history(&messages);
        assert!(truncated.len() < messages.len());
    }

    #[test]
    fn test_context_prompt_builder() {
        let mut builder = ContextPromptBuilder::new(128_000);
        builder.add_section(
            SystemPromptSection::new("identity", "Identity", "You are a helpful assistant.")
                .with_priority(100)
                .required(),
        );
        let prompt = builder.assemble_prompt();
        assert!(prompt.contains("helpful assistant"));
        assert!(prompt.contains("Identity"));
    }

    #[tokio::test]
    async fn test_load_workspace_files_fail_open() {
        let mut builder = ContextPromptBuilder::new(128_000)
            .with_workspace_root(PathBuf::from("/nonexistent/opensquilla/path"));
        builder.load_workspace_files().await.unwrap();
        // No sections loaded (all files missing), prompt is empty.
        assert_eq!(builder.assembler().section_count(), 0);
    }

    #[test]
    fn test_fragment_builder() {
        let builder = FragmentBuilder::new()
            .add_text("first", "first content")
            .add_text_with_priority("second", "second content", 10);
        assert_eq!(builder.len(), 2);
        let prompt = builder.build();
        assert!(prompt.contains("first content"));
        assert!(prompt.contains("second content"));
    }

    #[test]
    fn test_window_manager_available_history() {
        let mut manager = ContextWindowManager::new(10_000);
        manager.set_system_tokens(2_000);
        manager.set_tools_tokens(500);
        manager.set_output_tokens(1_000);
        // Available = 10_000 - 2_000 - 500 - 1_000 = 6_500
        assert_eq!(manager.available_history_tokens(), 6_500);
        assert_eq!(manager.total_consumed(), 3_500);
        assert_eq!(manager.remaining(), 6_500);
    }

    #[test]
    fn test_window_manager_history_over_budget() {
        let mut manager = ContextWindowManager::new(1_000)
            .with_estimator_ratio(1.0);
        manager.set_system_tokens(800);
        manager.set_tools_tokens(0);
        manager.set_output_tokens(0);
        // Available history = 1_000 - 800 = 200
        let messages: Vec<Message> = (0..50)
            .map(|i| Message::user(format!("msg{i}")))
            .collect();
        manager.update_history(&messages);
        assert!(manager.history_over_budget());
        assert!(manager.history_overage() > 0);
    }

    #[test]
    fn test_build_context_preserves_system() {
        let mut builder = ContextPromptBuilder::new(128_000);
        builder.add_section(
            SystemPromptSection::new("identity", "Identity", "You are helpful.")
                .required(),
        );
        let messages = vec![
            Message::system("existing system"),
            Message::user("hello"),
            Message::assistant("hi"),
        ];
        let (system, history) = builder.build_context(&messages);
        assert_eq!(system.role, MessageRole::System);
        assert!(!history.is_empty());
    }
}
