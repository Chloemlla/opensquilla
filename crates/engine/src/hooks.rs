use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, ToolCall, ToolResult};
use std::fmt;
use tracing::instrument;

/// A hook that is called before and after each turn.
///
/// Turn hooks allow observers to monitor or modify turn execution.
/// They can be used for logging, metrics, auditing, or other cross-cutting
/// concerns.
#[async_trait]
pub trait TurnHook: Send + Sync + fmt::Debug {
    /// Called before a turn begins, with the current messages.
    ///
    /// The returned messages will replace the original messages, allowing
    /// the hook to modify them.
    async fn before_turn(&self, turn_id: &str, messages: &[Message]) -> Result<Vec<Message>>;

    /// Called after a turn completes, with the final messages and result.
    async fn after_turn(
        &self,
        turn_id: &str,
        messages: &[Message],
        result: &Result<()>,
    ) -> Result<()>;
}

/// A hook that is called when context compaction is triggered.
///
/// Compaction hooks allow custom logic for reducing the context window,
/// such as summarization or selective message retention.
#[async_trait]
pub trait CompactionHook: Send + Sync + fmt::Debug {
    /// Called before compaction occurs, allowing inspection of the messages
    /// that will be compacted.
    async fn before_compaction(&self, turn_id: &str, message_count: usize) -> Result<()>;

    /// Called after compaction completes, with the new message count.
    async fn after_compaction(
        &self,
        turn_id: &str,
        before_count: usize,
        after_count: usize,
    ) -> Result<()>;

    /// Custom compaction logic that returns a compacted version of the
    /// messages. If this returns None, the default compaction strategy is used.
    async fn compact(
        &self,
        turn_id: &str,
        messages: &[Message],
        max_messages: usize,
    ) -> Result<Option<Vec<Message>>>;
}

/// A hook that is called before and after tool execution.
///
/// Tool hooks allow monitoring, validation, or modification of tool calls
/// and their results.
#[async_trait]
pub trait ToolHook: Send + Sync + fmt::Debug {
    /// Called before a tool is executed, with the tool call details.
    ///
    /// The returned ToolCall will replace the original, allowing the hook
    /// to modify the call parameters.
    async fn before_tool_call(&self, turn_id: &str, call: &ToolCall) -> Result<ToolCall>;

    /// Called after a tool completes, with the result.
    ///
    /// The returned ToolResult will replace the original, allowing the hook
    /// to modify the result.
    async fn after_tool_call(
        &self,
        turn_id: &str,
        call: &ToolCall,
        result: &ToolResult,
    ) -> Result<ToolResult>;

    /// Called when a tool call fails with an error.
    async fn on_tool_error(&self, turn_id: &str, call: &ToolCall, error: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Composite Hook Implementations
// ---------------------------------------------------------------------------

/// A composite hook that runs multiple hooks in sequence.
///
/// This allows hook chains where each hook is called in order, and the
/// output of one hook becomes the input to the next.
pub struct HookChain<T: fmt::Debug> {
    /// The ordered list of hooks in this chain.
    hooks: Vec<Box<dyn Fn(&T) -> T + Send + Sync>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: fmt::Debug> fmt::Debug for HookChain<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HookChain")
            .field("hook_count", &self.hooks.len())
            .finish()
    }
}

impl<T: fmt::Debug> Default for HookChain<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> HookChain<T> {
    /// Create a new empty hook chain.
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Add a hook to the chain.
    pub fn add(&mut self, hook: Box<dyn Fn(&T) -> T + Send + Sync>) {
        self.hooks.push(hook);
    }

    /// Execute all hooks in the chain.
    pub fn execute(&self, input: &T) -> T
    where
        T: Clone,
    {
        let mut result = input.clone();
        for hook in &self.hooks {
            result = hook(&result);
        }
        result
    }
}

/// A simple logging hook that records turn events using tracing.
#[derive(Debug)]
pub struct LoggingHook {
    /// The name of this logger for identification.
    name: String,
}

impl LoggingHook {
    /// Create a new logging hook with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl TurnHook for LoggingHook {
    #[instrument(skip(self, messages), fields(hook = %self.name, turn_id))]
    async fn before_turn(&self, turn_id: &str, messages: &[Message]) -> Result<Vec<Message>> {
        tracing::info!(
            turn_id = turn_id,
            message_count = messages.len(),
            "Turn hook: before turn"
        );
        Ok(messages.to_vec())
    }

    #[instrument(skip(self), fields(hook = %self.name, turn_id))]
    async fn after_turn(
        &self,
        turn_id: &str,
        _messages: &[Message],
        result: &Result<()>,
    ) -> Result<()> {
        tracing::info!(
            turn_id = turn_id,
            success = result.is_ok(),
            "Turn hook: after turn"
        );
        Ok(())
    }
}

#[async_trait]
impl CompactionHook for LoggingHook {
    #[instrument(skip(self), fields(hook = %self.name, turn_id))]
    async fn before_compaction(&self, turn_id: &str, message_count: usize) -> Result<()> {
        tracing::info!(
            turn_id = turn_id,
            message_count = message_count,
            "Compaction hook: before compaction"
        );
        Ok(())
    }

    #[instrument(skip(self), fields(hook = %self.name, turn_id))]
    async fn after_compaction(
        &self,
        turn_id: &str,
        before_count: usize,
        after_count: usize,
    ) -> Result<()> {
        tracing::info!(
            turn_id = turn_id,
            before = before_count,
            after = after_count,
            "Compaction hook: after compaction"
        );
        Ok(())
    }

    #[instrument(skip(self), fields(hook = %self.name, turn_id))]
    async fn compact(
        &self,
        _turn_id: &str,
        _messages: &[Message],
        _max_messages: usize,
    ) -> Result<Option<Vec<Message>>> {
        // Return None to use the default compaction strategy.
        Ok(None)
    }
}

#[async_trait]
impl ToolHook for LoggingHook {
    #[instrument(skip(self, call), fields(hook = %self.name, turn_id, tool = %call.name))]
    async fn before_tool_call(&self, _turn_id: &str, call: &ToolCall) -> Result<ToolCall> {
        tracing::info!(
            tool_name = %call.name,
            "Tool hook: before tool call"
        );
        Ok(call.clone())
    }

    #[instrument(skip(self, call, result), fields(hook = %self.name, turn_id, tool = %call.name))]
    async fn after_tool_call(
        &self,
        _turn_id: &str,
        call: &ToolCall,
        result: &ToolResult,
    ) -> Result<ToolResult> {
        tracing::info!(
            tool_name = %call.name,
            is_error = result.is_error,
            "Tool hook: after tool call"
        );
        Ok(result.clone())
    }

    #[instrument(skip(self, call), fields(hook = %self.name, turn_id, tool = %call.name))]
    async fn on_tool_error(&self, _turn_id: &str, call: &ToolCall, error: &str) -> Result<()> {
        tracing::warn!(
            tool_name = %call.name,
            error = error,
            "Tool hook: tool error"
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Extended hook traits
// ---------------------------------------------------------------------------

/// The phase of the turn lifecycle a hook observes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookPhase {
    /// Pre-turn (before the pipeline starts).
    PreTurn,
    /// Post-turn (after finalization).
    PostTurn,
    /// Pre-tool (before a tool call executes).
    PreTool,
    /// Post-tool (after a tool call completes).
    PostTool,
    /// On-error (when a turn or tool fails).
    OnError,
    /// On-compaction (around context compaction).
    OnCompaction,
}

/// A pre-turn hook that runs before the pipeline begins.
#[async_trait]
pub trait PreTurnHook: Send + Sync + fmt::Debug {
    /// Called before the turn begins. May rewrite the messages.
    async fn pre_turn(&self, turn_id: &str, messages: Vec<Message>) -> Result<Vec<Message>>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

/// A post-turn hook that runs after the turn completes.
#[async_trait]
pub trait PostTurnHook: Send + Sync + fmt::Debug {
    /// Called after the turn completes with the final messages and outcome.
    async fn post_turn(&self, turn_id: &str, messages: &[Message], success: bool) -> Result<()>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

/// A pre-tool hook that runs before a tool call executes.
#[async_trait]
pub trait PreToolHook: Send + Sync + fmt::Debug {
    /// Called before the tool call. May modify the call.
    async fn pre_tool(&self, turn_id: &str, call: ToolCall) -> Result<ToolCall>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

/// A post-tool hook that runs after a tool call completes.
#[async_trait]
pub trait PostToolHook: Send + Sync + fmt::Debug {
    /// Called after the tool call. May modify the result.
    async fn post_tool(
        &self,
        turn_id: &str,
        call: &ToolCall,
        result: ToolResult,
    ) -> Result<ToolResult>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

/// An error hook that observes turn/tool errors.
#[async_trait]
pub trait OnErrorHook: Send + Sync + fmt::Debug {
    /// Called when a turn or tool fails.
    async fn on_error(&self, turn_id: &str, context: &str, error: &str) -> Result<()>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

/// A compaction hook with priority ordering (extension of [`CompactionHook`]).
#[async_trait]
pub trait OnCompactionHook: Send + Sync + fmt::Debug {
    /// Called before compaction.
    async fn before_compact(&self, turn_id: &str, message_count: usize) -> Result<()>;

    /// Called after compaction.
    async fn after_compact(&self, turn_id: &str, before: usize, after: usize) -> Result<()>;

    /// The hook's priority (higher runs earlier).
    fn priority(&self) -> i32 {
        0
    }
}

// ---------------------------------------------------------------------------
// Hook registry with priority ordering
// ---------------------------------------------------------------------------

/// An entry in the hook registry.
#[derive(Debug)]
enum HookEntry {
    /// A pre-turn hook.
    PreTurn(Box<dyn PreTurnHook>),
    /// A post-turn hook.
    PostTurn(Box<dyn PostTurnHook>),
    /// A pre-tool hook.
    PreTool(Box<dyn PreToolHook>),
    /// A post-tool hook.
    PostTool(Box<dyn PostToolHook>),
    /// An error hook.
    OnError(Box<dyn OnErrorHook>),
    /// A compaction hook.
    OnCompaction(Box<dyn OnCompactionHook>),
}

impl HookEntry {
    /// The hook's priority.
    fn priority(&self) -> i32 {
        match self {
            HookEntry::PreTurn(h) => h.priority(),
            HookEntry::PostTurn(h) => h.priority(),
            HookEntry::PreTool(h) => h.priority(),
            HookEntry::PostTool(h) => h.priority(),
            HookEntry::OnError(h) => h.priority(),
            HookEntry::OnCompaction(h) => h.priority(),
        }
    }
}

/// A registry of hooks with priority ordering.
///
/// Hooks are stored per phase and executed in descending priority order.
#[derive(Debug, Default)]
pub struct HookRegistry {
    /// The registered hooks.
    hooks: Vec<HookEntry>,
}

impl HookRegistry {
    /// Create a new empty hook registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a pre-turn hook.
    pub fn register_pre_turn(&mut self, hook: impl PreTurnHook + 'static) {
        self.hooks.push(HookEntry::PreTurn(Box::new(hook)));
        self.sort();
    }

    /// Register a post-turn hook.
    pub fn register_post_turn(&mut self, hook: impl PostTurnHook + 'static) {
        self.hooks.push(HookEntry::PostTurn(Box::new(hook)));
        self.sort();
    }

    /// Register a pre-tool hook.
    pub fn register_pre_tool(&mut self, hook: impl PreToolHook + 'static) {
        self.hooks.push(HookEntry::PreTool(Box::new(hook)));
        self.sort();
    }

    /// Register a post-tool hook.
    pub fn register_post_tool(&mut self, hook: impl PostToolHook + 'static) {
        self.hooks.push(HookEntry::PostTool(Box::new(hook)));
        self.sort();
    }

    /// Register an error hook.
    pub fn register_on_error(&mut self, hook: impl OnErrorHook + 'static) {
        self.hooks.push(HookEntry::OnError(Box::new(hook)));
        self.sort();
    }

    /// Register a compaction hook.
    pub fn register_on_compaction(&mut self, hook: impl OnCompactionHook + 'static) {
        self.hooks.push(HookEntry::OnCompaction(Box::new(hook)));
        self.sort();
    }

    /// The number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// True when no hooks are registered.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Sort the hooks by descending priority.
    fn sort(&mut self) {
        self.hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Run all pre-turn hooks, threading the messages through each hook.
    pub async fn run_pre_turn(&self, turn_id: &str, messages: Vec<Message>) -> Vec<Message> {
        let mut current = messages;
        for entry in &self.hooks {
            if let HookEntry::PreTurn(hook) = entry {
                match hook.pre_turn(turn_id, current.clone()).await {
                    Ok(next) => current = next,
                    Err(e) => {
                        tracing::warn!(turn_id = %turn_id, error = %e, "pre_turn hook failed");
                    }
                }
            }
        }
        current
    }

    /// Run all post-turn hooks.
    pub async fn run_post_turn(&self, turn_id: &str, messages: &[Message], success: bool) {
        for entry in &self.hooks {
            if let HookEntry::PostTurn(hook) = entry {
                if let Err(e) = hook.post_turn(turn_id, messages, success).await {
                    tracing::warn!(turn_id = %turn_id, error = %e, "post_turn hook failed");
                }
            }
        }
    }

    /// Run all pre-tool hooks, threading the tool call through each hook.
    pub async fn run_pre_tool(&self, turn_id: &str, call: ToolCall) -> ToolCall {
        let mut current = call;
        for entry in &self.hooks {
            if let HookEntry::PreTool(hook) = entry {
                match hook.pre_tool(turn_id, current.clone()).await {
                    Ok(next) => current = next,
                    Err(e) => {
                        tracing::warn!(turn_id = %turn_id, error = %e, "pre_tool hook failed");
                    }
                }
            }
        }
        current
    }

    /// Run all post-tool hooks, threading the result through each hook.
    pub async fn run_post_tool(
        &self,
        turn_id: &str,
        call: &ToolCall,
        result: ToolResult,
    ) -> ToolResult {
        let mut current = result;
        for entry in &self.hooks {
            if let HookEntry::PostTool(hook) = entry {
                match hook.post_tool(turn_id, call, current.clone()).await {
                    Ok(next) => current = next,
                    Err(e) => {
                        tracing::warn!(turn_id = %turn_id, error = %e, "post_tool hook failed");
                    }
                }
            }
        }
        current
    }

    /// Run all error hooks.
    pub async fn run_on_error(&self, turn_id: &str, context: &str, error: &str) {
        for entry in &self.hooks {
            if let HookEntry::OnError(hook) = entry {
                if let Err(e) = hook.on_error(turn_id, context, error).await {
                    tracing::warn!(turn_id = %turn_id, error = %e, "on_error hook failed");
                }
            }
        }
    }

    /// Run all compaction hooks.
    pub async fn run_on_compaction(&self, turn_id: &str, before: usize, after: usize) {
        for entry in &self.hooks {
            if let HookEntry::OnCompaction(hook) = entry {
                let _ = hook.before_compact(turn_id, before).await;
                let _ = hook.after_compact(turn_id, before, after).await;
            }
        }
    }
}

/// A composite pre-turn hook that runs multiple hooks in order.
#[derive(Debug)]
pub struct CompositePreTurnHook {
    /// The underlying hooks.
    hooks: Vec<Box<dyn PreTurnHook>>,
    /// The priority of this composite.
    priority: i32,
}

impl CompositePreTurnHook {
    /// Create a new composite hook.
    pub fn new(priority: i32) -> Self {
        Self {
            hooks: Vec::new(),
            priority,
        }
    }

    /// Add a hook to the composite.
    pub fn add(&mut self, hook: Box<dyn PreTurnHook>) {
        self.hooks.push(hook);
    }
}

#[async_trait]
impl PreTurnHook for CompositePreTurnHook {
    async fn pre_turn(&self, turn_id: &str, messages: Vec<Message>) -> Result<Vec<Message>> {
        let mut current = messages;
        for hook in &self.hooks {
            current = hook.pre_turn(turn_id, current).await?;
        }
        Ok(current)
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

/// A simple pre-turn hook that injects a system message when missing.
#[derive(Debug)]
pub struct SystemPromptHook {
    /// The system prompt to inject.
    prompt: String,
    /// The hook's priority.
    priority: i32,
}

impl SystemPromptHook {
    /// Create a new system-prompt hook.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            priority: 100,
        }
    }

    /// Set the hook's priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

#[async_trait]
impl PreTurnHook for SystemPromptHook {
    async fn pre_turn(&self, _turn_id: &str, mut messages: Vec<Message>) -> Result<Vec<Message>> {
        let has_system = messages
            .iter()
            .any(|m| m.role == opensquilla_core::types::MessageRole::System);
        if !has_system && !self.prompt.is_empty() {
            messages.insert(0, Message::system(&self.prompt));
        }
        Ok(messages)
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

/// A logging error hook that records errors via tracing.
#[derive(Debug)]
pub struct ErrorLoggingHook {
    /// The hook's priority.
    priority: i32,
}

impl ErrorLoggingHook {
    /// Create a new error logging hook.
    pub fn new() -> Self {
        Self { priority: -100 }
    }
}

impl Default for ErrorLoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OnErrorHook for ErrorLoggingHook {
    async fn on_error(&self, turn_id: &str, context: &str, error: &str) -> Result<()> {
        tracing::error!(
            turn_id = %turn_id,
            context = %context,
            error = %error,
            "error hook: turn failed"
        );
        Ok(())
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}

/// A simple post-tool hook that limits result content length.
#[derive(Debug)]
pub struct MaxResultLengthHook {
    /// The maximum result content length.
    max_chars: usize,
    /// The hook's priority.
    priority: i32,
}

impl MaxResultLengthHook {
    /// Create a new result-length hook.
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            priority: 50,
        }
    }
}

#[async_trait]
impl PostToolHook for MaxResultLengthHook {
    async fn post_tool(
        &self,
        _turn_id: &str,
        _call: &ToolCall,
        result: ToolResult,
    ) -> Result<ToolResult> {
        if result.content.chars().count() > self.max_chars {
            let truncated: String = result.content.chars().take(self.max_chars).collect();
            Ok(ToolResult {
                content: format!("{truncated}\n[...truncated]"),
                ..result
            })
        } else {
            Ok(result)
        }
    }

    fn priority(&self) -> i32 {
        self.priority
    }
}
