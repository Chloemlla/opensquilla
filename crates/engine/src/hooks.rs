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
    async fn before_turn(
        &self,
        turn_id: &str,
        messages: &[Message],
    ) -> Result<Vec<Message>>;

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
    async fn before_compaction(
        &self,
        turn_id: &str,
        message_count: usize,
    ) -> Result<()>;

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
    async fn before_tool_call(
        &self,
        turn_id: &str,
        call: &ToolCall,
    ) -> Result<ToolCall>;

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
    async fn on_tool_error(
        &self,
        turn_id: &str,
        call: &ToolCall,
        error: &str,
    ) -> Result<()>;
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
        Self {
            name: name.into(),
        }
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