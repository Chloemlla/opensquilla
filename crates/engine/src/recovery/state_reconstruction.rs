//! State reconstruction: rebuild conversation history and agent state
//! from a flattened transcript.
//!
//! Mirrors the Python `engine/recovery/state_reconstruction.py`. When the
//! process restarts after a crash, the reconstructor reads the persisted
//! transcript rows and rebuilds the in-memory message list, tool-call
//! pairing, and agent state so the agent can resume cleanly.

use crate::agent::AgentState;
use crate::history::{deduplicate, reconstruct_from_row, repair_tool_pairs, TranscriptRow};
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole, Usage};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// The reconstructed agent state.
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// The session ID.
    pub session_id: String,
    /// The agent ID.
    pub agent_id: String,
    /// The reconstructed message history.
    pub messages: Vec<Message>,
    /// The agent state.
    pub agent_state: AgentState,
    /// The accumulated token usage.
    pub usage: Usage,
    /// The number of transcript rows processed.
    pub rows_processed: usize,
    /// The number of messages reconstructed.
    pub message_count: usize,
    /// The number of tool pairs repaired.
    pub repaired_pairs: usize,
    /// The number of duplicates removed.
    pub duplicates_removed: usize,
    /// Whether the reconstruction was complete.
    pub complete: bool,
}

impl ReconstructedState {
    /// Whether the reconstruction produced a valid message history.
    pub fn is_valid(&self) -> bool {
        !self.messages.is_empty()
    }
}

/// The state reconstructor.
#[derive(Debug, Clone)]
pub struct StateReconstructor {
    /// The session ID.
    session_id: String,
    /// The agent ID.
    agent_id: String,
    /// Whether to deduplicate messages.
    deduplicate: bool,
    /// Whether to repair tool pairs.
    repair_tool_pairs: bool,
    /// The initial agent state.
    initial_state: AgentState,
}

impl StateReconstructor {
    /// Create a new state reconstructor.
    pub fn new(session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            deduplicate: true,
            repair_tool_pairs: true,
            initial_state: AgentState::Idle,
        }
    }

    /// Set whether to deduplicate messages.
    pub fn with_deduplication(mut self, enabled: bool) -> Self {
        self.deduplicate = enabled;
        self
    }

    /// Set whether to repair tool pairs.
    pub fn with_tool_pair_repair(mut self, enabled: bool) -> Self {
        self.repair_tool_pairs = enabled;
        self
    }

    /// Set the initial agent state.
    pub fn with_initial_state(mut self, state: AgentState) -> Self {
        self.initial_state = state;
        self
    }

    /// Reconstruct the state from a list of transcript rows.
    pub fn reconstruct(&self, rows: &[TranscriptRow]) -> Result<ReconstructedState> {
        if rows.is_empty() {
            return Ok(ReconstructedState {
                session_id: self.session_id.clone(),
                agent_id: self.agent_id.clone(),
                messages: Vec::new(),
                agent_state: self.initial_state.clone(),
                usage: Usage::default(),
                rows_processed: 0,
                message_count: 0,
                repaired_pairs: 0,
                duplicates_removed: 0,
                complete: true,
            });
        }

        let mut messages: Vec<Message> = Vec::with_capacity(rows.len());
        let mut errors: Vec<String> = Vec::new();
        let mut total_input_tokens = 0u64;
        let mut total_output_tokens = 0u64;

        for (i, row) in rows.iter().enumerate() {
            match reconstruct_from_row(row) {
                Ok(message) => {
                    // Accumulate token estimates from text content length.
                    let text = message.text_content();
                    let tokens = (text.chars().count() as u64) / 4;
                    match message.role {
                        MessageRole::User | MessageRole::System => {
                            total_input_tokens += tokens;
                        }
                        MessageRole::Assistant => {
                            total_output_tokens += tokens;
                        }
                        MessageRole::Tool => {
                            total_input_tokens += tokens;
                        }
                    }
                    messages.push(message);
                }
                Err(e) => {
                    warn!(row_index = i, error = %e, "failed to reconstruct row");
                    errors.push(format!("row {i}: {e}"));
                }
            }
        }

        let rows_processed = rows.len();
        let mut duplicates_removed = 0usize;
        let mut repaired_pairs = 0usize;

        // Deduplicate messages.
        if self.deduplicate {
            let before = messages.len();
            messages = deduplicate(&messages);
            duplicates_removed = before.saturating_sub(messages.len());
            if duplicates_removed > 0 {
                debug!(removed = duplicates_removed, "deduplicated messages");
            }
        }

        // Repair tool-call pairing.
        if self.repair_tool_pairs {
            let outcome = repair_tool_pairs(&messages);
            repaired_pairs = outcome.removed_results + outcome.unpaired_calls;
            messages = outcome.messages;
            if repaired_pairs > 0 {
                debug!(repaired = repaired_pairs, "repaired tool pairs");
            }
        }

        // Determine the final agent state from the last message.
        let agent_state = self.infer_agent_state(&messages);

        let message_count = messages.len();
        let complete = errors.is_empty();

        info!(
            session_id = %self.session_id,
            rows = rows_processed,
            messages = message_count,
            duplicates_removed = duplicates_removed,
            repaired_pairs = repaired_pairs,
            complete = complete,
            "state reconstruction complete"
        );

        Ok(ReconstructedState {
            session_id: self.session_id.clone(),
            agent_id: self.agent_id.clone(),
            messages,
            agent_state,
            usage: Usage::new(total_input_tokens, total_output_tokens),
            rows_processed,
            message_count,
            repaired_pairs,
            duplicates_removed,
            complete,
        })
    }

    /// Infer the agent state from the last few messages.
    fn infer_agent_state(&self, messages: &[Message]) -> AgentState {
        if messages.is_empty() {
            return self.initial_state.clone();
        }

        let last = messages.last().unwrap();
        let has_pending_tool_calls = messages.iter().rev().any(|m| {
            m.role == MessageRole::Assistant
                && m.content.iter().any(|b| {
                    matches!(b, opensquilla_core::types::ContentBlock::ToolUse(_))
                })
        });

        let last_is_tool_result = last.role == MessageRole::Tool;

        if has_pending_tool_calls && !last_is_tool_result {
            // An assistant message with tool calls but no following tool result
            // means the turn was interrupted mid-execution.
            return AgentState::WaitingForTool;
        }

        if last.role == MessageRole::Assistant {
            return AgentState::Completed;
        }

        if last.role == MessageRole::User {
            return AgentState::Idle;
        }

        self.initial_state.clone()
    }

    /// The session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The agent ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

/// Reconstruct a conversation from a list of rows, returning just the
/// messages (convenience function).
pub fn reconstruct_messages(rows: &[TranscriptRow]) -> Result<Vec<Message>> {
    let mut messages = Vec::with_capacity(rows.len());
    for row in rows {
        match reconstruct_from_row(row) {
            Ok(msg) => messages.push(msg),
            Err(e) => {
                warn!(error = %e, "skipping unparseable row");
            }
        }
    }
    Ok(messages)
}

/// Estimate the token usage from a reconstructed conversation.
pub fn estimate_usage(messages: &[Message]) -> Usage {
    let mut input = 0u64;
    let mut output = 0u64;
    for msg in messages {
        let tokens = (msg.text_content().chars().count() as u64) / 4;
        match msg.role {
            MessageRole::User | MessageRole::System | MessageRole::Tool => {
                input += tokens;
            }
            MessageRole::Assistant => {
                output += tokens;
            }
        }
    }
    Usage::new(input, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::ContentBlock;

    fn row(role: &str, text: &str) -> TranscriptRow {
        TranscriptRow {
            role: role.to_string(),
            text: Some(text.to_string()),
            name: None,
            tool_call_id: None,
            tool_calls_json: None,
            tool_result_json: None,
            reasoning: None,
        }
    }

    #[test]
    fn test_reconstruct_empty() {
        let reconstructor = StateReconstructor::new("s1", "a1");
        let state = reconstructor.reconstruct(&[]).unwrap();
        assert!(state.messages.is_empty());
        assert_eq!(state.rows_processed, 0);
        assert!(state.complete);
    }

    #[test]
    fn test_reconstruct_simple_conversation() {
        let reconstructor = StateReconstructor::new("s1", "a1");
        let rows = vec![
            row("user", "hello"),
            row("assistant", "hi there"),
        ];
        let state = reconstructor.reconstruct(&rows).unwrap();
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.message_count, 2);
        assert!(state.complete);
        assert_eq!(state.agent_state, AgentState::Completed);
    }

    #[test]
    fn test_reconstruct_deduplicates() {
        let reconstructor = StateReconstructor::new("s1", "a1");
        let rows = vec![
            row("user", "hello"),
            row("user", "hello"),
            row("assistant", "hi"),
        ];
        let state = reconstructor.reconstruct(&rows).unwrap();
        assert!(state.duplicates_removed > 0);
    }

    #[test]
    fn test_infer_state_waiting_for_tool() {
        use opensquilla_core::types::{ContentBlock, ToolCall};
        use serde_json::json;
        let reconstructor = StateReconstructor::new("s1", "a1");
        // An assistant message with a tool call but no following tool result.
        let messages = vec![
            Message::user("run the tool"),
            Message {
                role: MessageRole::Assistant,
                content: vec![ContentBlock::ToolUse(ToolCall::new(
                    "c1",
                    "shell",
                    json!({"cmd": "ls"}),
                ))],
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            },
        ];
        let state = reconstructor.infer_agent_state(&messages);
        assert_eq!(state, AgentState::WaitingForTool);
    }

    #[test]
    fn test_infer_state_completed() {
        let reconstructor = StateReconstructor::new("s1", "a1");
        let messages = vec![Message::user("hi"), Message::assistant("bye")];
        let state = reconstructor.infer_agent_state(&messages);
        assert_eq!(state, AgentState::Completed);
    }

    #[test]
    fn test_infer_state_idle() {
        let reconstructor = StateReconstructor::new("s1", "a1");
        let messages = vec![Message::assistant("hi"), Message::user("again")];
        let state = reconstructor.infer_agent_state(&messages);
        assert_eq!(state, AgentState::Idle);
    }

    #[test]
    fn test_reconstruct_with_tool_repair() {
        use opensquilla_core::types::{ContentBlock, ToolCall, ToolResult};
        use serde_json::json;
        let reconstructor = StateReconstructor::new("s1", "a1");
        // A tool result without a matching tool_use.
        let rows = vec![
            row("user", "run tool"),
            TranscriptRow {
                role: "tool".to_string(),
                text: Some("result".to_string()),
                name: Some("shell".to_string()),
                tool_call_id: Some("orphan".to_string()),
                tool_calls_json: None,
                tool_result_json: None,
                reasoning: None,
            },
        ];
        let state = reconstructor.reconstruct(&rows).unwrap();
        assert!(state.repaired_pairs > 0);
    }

    #[test]
    fn test_estimate_usage() {
        let messages = vec![
            Message::user("hello world this is a test"), // 25 chars -> 6 tokens (input)
            Message::assistant("hi there"),               // 8 chars -> 2 tokens (output)
        ];
        let usage = estimate_usage(&messages);
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
    }

    #[test]
    fn test_reconstruct_messages_convenience() {
        let rows = vec![row("user", "hello"), row("assistant", "hi")];
        let messages = reconstruct_messages(&rows).unwrap();
        assert_eq!(messages.len(), 2);
    }
}
