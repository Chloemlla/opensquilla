//! Partial turn replay: replay an interrupted turn from the point of
//! interruption.
//!
//! Mirrors the Python `engine/recovery/replay.py`. When a turn is interrupted
//! (crash, network failure, shutdown), the replay module determines whether the
//! partial turn can be safely replayed and, if so, reconstructs the messages
//! from the last checkpoint so the agent can continue.

use crate::agent::AgentState;
use crate::history::repair_tool_pairs;
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use tracing::{debug, info, warn};

/// A checkpoint of a turn at a replayable boundary.
#[derive(Debug, Clone)]
pub struct ReplayCheckpoint {
    /// The turn ID.
    pub turn_id: String,
    /// The tool round at the checkpoint.
    pub tool_round: u32,
    /// The messages at the checkpoint.
    pub messages: Vec<Message>,
    /// The agent state at the checkpoint.
    pub agent_state: AgentState,
    /// Whether the checkpoint is at a safe replay boundary.
    pub safe_boundary: bool,
}

impl ReplayCheckpoint {
    /// Create a new checkpoint.
    pub fn new(turn_id: impl Into<String>, tool_round: u32, messages: Vec<Message>) -> Self {
        let safe_boundary = is_safe_replay_boundary(&messages);
        Self {
            turn_id: turn_id.into(),
            tool_round,
            messages,
            agent_state: AgentState::Thinking,
            safe_boundary,
        }
    }

    /// The message count at the checkpoint.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Whether the checkpoint is safe to replay from.
    pub fn is_safe(&self) -> bool {
        self.safe_boundary
    }
}

/// The decision for a replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayDecision {
    /// Replay is safe; resume from the checkpoint.
    Safe {
        /// The tool round to resume from.
        tool_round: u32,
        /// The number of messages at the checkpoint.
        message_count: usize,
    },
    /// Replay is not safe; restart the turn from scratch.
    Restart {
        /// The reason restart is required.
        reason: String,
    },
    /// The turn is already complete; no replay needed.
    AlreadyComplete {
        /// The number of messages in the completed turn.
        message_count: usize,
    },
}

impl ReplayDecision {
    /// Whether replay is safe.
    pub fn is_safe(&self) -> bool {
        matches!(self, ReplayDecision::Safe { .. })
    }

    /// Whether a restart is required.
    pub fn needs_restart(&self) -> bool {
        matches!(self, ReplayDecision::Restart { .. })
    }
}

/// The outcome of a replay.
#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    /// The decision that was made.
    pub decision: ReplayDecision,
    /// The messages to resume with (or the restarted messages).
    pub messages: Vec<Message>,
    /// The tool round to resume from.
    pub tool_round: u32,
    /// The number of messages dropped during replay.
    pub dropped_messages: usize,
}

/// The turn replay manager.
#[derive(Debug, Clone)]
pub struct TurnReplay {
    /// The turn ID.
    turn_id: String,
    /// The maximum tool rounds allowed.
    max_tool_rounds: u32,
}

impl TurnReplay {
    /// Create a new turn replay manager.
    pub fn new(turn_id: impl Into<String>, max_tool_rounds: u32) -> Self {
        Self {
            turn_id: turn_id.into(),
            max_tool_rounds: max_tool_rounds.max(1),
        }
    }

    /// The turn ID.
    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    /// The maximum tool rounds.
    pub fn max_tool_rounds(&self) -> u32 {
        self.max_tool_rounds
    }

    /// Determine whether the given messages are safe to replay.
    ///
    /// A replay is safe when:
    /// * the last message is a user message (no provider response yet), or
    /// * the messages end on a complete tool round (tool result present for
    ///   every tool call), or
    /// * the last message is a completed assistant response.
    pub fn decide(&self, messages: &[Message], agent_state: &AgentState) -> ReplayDecision {
        if messages.is_empty() {
            return ReplayDecision::Restart {
                reason: "no messages to replay".to_string(),
            };
        }

        if matches!(agent_state, AgentState::Completed) {
            return ReplayDecision::AlreadyComplete {
                message_count: messages.len(),
            };
        }

        let last = messages.last().unwrap();
        match last.role {
            MessageRole::User => {
                // No provider response yet: replay from the user message.
                ReplayDecision::Safe {
                    tool_round: 0,
                    message_count: messages.len(),
                }
            }
            MessageRole::Tool => {
                // Ends on a tool result: safe to replay the tool round.
                ReplayDecision::Safe {
                    tool_round: 1,
                    message_count: messages.len(),
                }
            }
            MessageRole::Assistant => {
                // Check if the assistant message has pending tool calls.
                let has_pending = has_pending_tool_calls(messages);
                if has_pending {
                    ReplayDecision::Restart {
                        reason: "interrupted mid-tool-round".to_string(),
                    }
                } else {
                    ReplayDecision::Safe {
                        tool_round: 0,
                        message_count: messages.len(),
                    }
                }
            }
            MessageRole::System => {
                // Only system messages: restart is not needed, but there's
                // nothing to replay either.
                ReplayDecision::Safe {
                    tool_round: 0,
                    message_count: messages.len(),
                }
            }
        }
    }

    /// Replay a partial turn from the given messages.
    ///
    /// Returns the replay outcome with the messages to resume with.
    pub fn replay(&self, messages: Vec<Message>, agent_state: &AgentState) -> ReplayOutcome {
        let decision = self.decide(&messages, agent_state);

        match &decision {
            ReplayDecision::Safe { tool_round, .. } => {
                // Repair tool pairs to ensure a clean surface.
                let outcome = repair_tool_pairs(&messages);
                let dropped = outcome.removed_results + outcome.unpaired_calls;
                if dropped > 0 {
                    debug!(
                        turn_id = %self.turn_id,
                        dropped = dropped,
                        "repaired tool pairs during replay"
                    );
                }
                ReplayOutcome {
                    decision: decision.clone(),
                    messages: outcome.messages,
                    tool_round: *tool_round,
                    dropped_messages: dropped,
                }
            }
            ReplayDecision::Restart { reason } => {
                info!(
                    turn_id = %self.turn_id,
                    reason = %reason,
                    "restarting turn from scratch"
                );
                // Restart: keep only system messages and the latest user message.
                let system: Vec<Message> = messages
                    .iter()
                    .filter(|m| m.role == MessageRole::System)
                    .cloned()
                    .collect();
                let last_user = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == MessageRole::User)
                    .cloned();
                let mut restart_messages = system;
                if let Some(user) = last_user {
                    restart_messages.push(user);
                }
                let dropped = messages.len().saturating_sub(restart_messages.len());
                ReplayOutcome {
                    decision: decision.clone(),
                    messages: restart_messages,
                    tool_round: 0,
                    dropped_messages: dropped,
                }
            }
            ReplayDecision::AlreadyComplete { .. } => ReplayOutcome {
                decision: decision.clone(),
                messages,
                tool_round: self.max_tool_rounds,
                dropped_messages: 0,
            },
        }
    }

    /// Create a replay checkpoint from the current messages.
    pub fn checkpoint(&self, messages: Vec<Message>) -> ReplayCheckpoint {
        ReplayCheckpoint::new(&self.turn_id, 0, messages)
    }

    /// Resume a turn from a checkpoint.
    pub fn resume_from_checkpoint(&self, checkpoint: &ReplayCheckpoint) -> Vec<Message> {
        if !checkpoint.is_safe() {
            warn!(
                turn_id = %self.turn_id,
                "checkpoint is not at a safe boundary, repairing tool pairs"
            );
        }
        let outcome = repair_tool_pairs(&checkpoint.messages);
        outcome.messages
    }
}

/// Determine whether a message list ends with pending (unanswered) tool calls.
fn has_pending_tool_calls(messages: &[Message]) -> bool {
    // Collect every tool-use id that appears in the history.
    let mut call_ids: Vec<String> = Vec::new();
    for msg in messages {
        if !matches!(msg.role, MessageRole::Assistant) {
            continue;
        }
        for block in &msg.content {
            if let ContentBlock::ToolUse(call) = block {
                call_ids.push(call.id.clone());
            }
        }
        if let Some(calls) = &msg.tool_calls {
            for call in calls {
                call_ids.push(call.id.clone());
            }
        }
    }
    if call_ids.is_empty() {
        return false;
    }

    // Collect every tool-result id that has been answered.
    let mut answered: Vec<String> = Vec::new();
    for msg in messages {
        if !matches!(msg.role, MessageRole::Tool) {
            continue;
        }
        for block in &msg.content {
            if let ContentBlock::ToolResult(result) = block {
                answered.push(result.tool_use_id.clone());
            }
        }
        if let Some(result) = &msg.tool_result {
            answered.push(result.tool_use_id.clone());
        }
    }

    // A call is pending when its id appears after the last result that
    // answers it, or no result ever answers it. Since messages are ordered,
    // a call is answered if its id is in the answered set.
    call_ids.iter().any(|id| !answered.contains(id))
}

/// Whether the message list is at a safe replay boundary.
pub fn is_safe_replay_boundary(messages: &[Message]) -> bool {
    if messages.is_empty() {
        return true;
    }
    let last = messages.last().unwrap();
    match last.role {
        MessageRole::User | MessageRole::System => true,
        MessageRole::Tool => true,
        MessageRole::Assistant => {
            // Safe if no pending tool calls.
            !has_pending_tool_calls(messages)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, ToolCall, ToolResult};
    use serde_json::json;

    fn user_msg(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_msg(text: &str) -> Message {
        Message::assistant(text)
    }

    fn tool_call_msg(id: &str) -> Message {
        Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall::new(
                id,
                "shell",
                json!({"cmd": "ls"}),
            ))],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    fn tool_result_msg(id: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult::success(id, "ok"))],
            name: Some("shell".into()),
            tool_call_id: Some(id.into()),
            tool_calls: None,
            tool_result: None,
        }
    }

    #[test]
    fn test_decide_safe_on_user_message() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let decision = replay.decide(&messages, &AgentState::Idle);
        assert!(decision.is_safe());
    }

    #[test]
    fn test_decide_restart_on_interrupted_tool_round() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![user_msg("run"), tool_call_msg("c1")];
        let decision = replay.decide(&messages, &AgentState::WaitingForTool);
        assert!(decision.needs_restart());
    }

    #[test]
    fn test_decide_already_complete() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![user_msg("hi"), assistant_msg("bye")];
        let decision = replay.decide(&messages, &AgentState::Completed);
        assert!(matches!(decision, ReplayDecision::AlreadyComplete { .. }));
    }

    #[test]
    fn test_replay_safe_keeps_messages() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![user_msg("hello"), assistant_msg("hi")];
        let outcome = replay.replay(messages.clone(), &AgentState::Idle);
        assert!(outcome.decision.is_safe());
        assert_eq!(outcome.messages.len(), 2);
    }

    #[test]
    fn test_replay_restart_drops_partial_round() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![
            user_msg("run"),
            tool_call_msg("c1"),
            tool_result_msg("c1"),
            tool_call_msg("c2"),
        ];
        let outcome = replay.replay(messages, &AgentState::WaitingForTool);
        assert!(outcome.decision.needs_restart());
        // The restart keeps system + latest user; drops the partial tool round.
        assert!(outcome.messages.iter().any(|m| m.role == MessageRole::User));
        assert!(outcome.dropped_messages > 0);
    }

    #[test]
    fn test_replay_repairs_tool_pairs() {
        let replay = TurnReplay::new("t1", 10);
        let messages = vec![
            user_msg("run"),
            tool_result_msg("orphan"),
        ];
        let outcome = replay.replay(messages, &AgentState::Idle);
        assert!(outcome.dropped_messages > 0);
    }

    #[test]
    fn test_checkpoint_safe_boundary() {
        let replay = TurnReplay::new("t1", 10);
        let checkpoint = replay.checkpoint(vec![user_msg("hello")]);
        assert!(checkpoint.is_safe());
        assert_eq!(checkpoint.message_count(), 1);
    }

    #[test]
    fn test_resume_from_checkpoint() {
        let replay = TurnReplay::new("t1", 10);
        let checkpoint = replay.checkpoint(vec![user_msg("hello"), assistant_msg("hi")]);
        let resumed = replay.resume_from_checkpoint(&checkpoint);
        assert_eq!(resumed.len(), 2);
    }
}
