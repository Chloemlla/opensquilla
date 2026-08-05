//! Turn control decisions: pure functions that classify stop surfaces.
//!
//! This module mirrors the Python backend's `turn_control.py`. It contains
//! NO I/O: every function is a pure transformation over the turn state.
//! The runtime consults these decisions to decide whether the agent loop
//! should continue (tool calls pending), stop (end_turn), or surface an
//! error.

use opensquilla_core::types::{ContentBlock, Message, MessageRole};

/// The high-level action the agent loop should take after a provider
/// response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnControl {
    /// The model requested one or more tool calls; the loop should execute
    /// them and continue.
    Continue,
    /// The model signaled completion (end_turn, final text answer, or no
    /// tool calls); the loop should stop and proceed to finalization.
    Stop(StopReason),
    /// The model produced an unrecoverable stop surface (refusal, empty
    /// output, malformed tool calls).
    Error(TurnControlError),
}

/// The reason the agent loop decided to stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The provider explicitly returned an `end_turn` stop reason.
    EndTurn,
    /// The assistant produced a final text response with no tool calls.
    FinalText,
    /// The maximum number of tool rounds was reached.
    MaxToolRoundsReached,
    /// The turn was halted by a pipeline step or external signal.
    Halted,
}

/// A recoverable or non-recoverable error detected by turn control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnControlError {
    /// The model returned a refusal stop reason.
    Refusal,
    /// The assistant message was entirely empty (no text, no tool calls).
    EmptyOutput,
    /// A tool call referenced an unknown tool.
    UnknownTool(String),
    /// A tool call was malformed (missing id, name, or input).
    MalformedToolCall(String),
}

/// Configuration for turn control decisions.
#[derive(Debug, Clone)]
pub struct TurnControlInput<'a> {
    /// The most recent stop reason reported by the provider, if any.
    pub stop_reason: Option<&'a str>,
    /// The response messages produced by the provider in this round.
    pub response_messages: &'a [Message],
    /// The current tool round (0-based).
    pub tool_round: u32,
    /// The maximum number of tool rounds allowed.
    pub max_tool_rounds: u32,
}

/// Decide the turn control action for a single provider round.
///
/// This is the core pure decision function. It inspects the stop reason,
/// the response messages, and the tool-round counter, and returns whether
/// the loop should continue, stop, or error.
///
/// # Decision order
///
/// 1. If the stop reason is a refusal, return `Error(Refusal)`.
/// 2. If the stop reason is `end_turn` AND there are no pending tool calls,
///    return `Stop(EndTurn)`.
/// 3. If there are pending tool calls and we have not exhausted the round
///    budget, return `Continue`.
/// 4. If there are pending tool calls but `tool_round >= max_tool_rounds`,
///    return `Stop(MaxToolRoundsReached)`.
/// 5. If the response has no tool calls and no text content, return
///    `Error(EmptyOutput)`.
/// 6. Otherwise (final text, no tool calls), return `Stop(FinalText)`.
pub fn decide_turn_control(input: &TurnControlInput<'_>) -> TurnControl {
    let pending = pending_tool_calls(input.response_messages);

    // 1. Refusal short-circuits everything.
    if is_refusal(input.stop_reason) {
        return TurnControl::Error(TurnControlError::Refusal);
    }

    let has_tool_calls = !pending.is_empty();

    // 2. Explicit end_turn with no pending tool calls => stop.
    if is_end_turn(input.stop_reason) && !has_tool_calls {
        return TurnControl::Stop(StopReason::EndTurn);
    }

    // 3 & 4. Tool calls present: respect the round budget.
    if has_tool_calls {
        if input.tool_round + 1 >= input.max_tool_rounds {
            return TurnControl::Stop(StopReason::MaxToolRoundsReached);
        }
        return TurnControl::Continue;
    }

    // 5. Empty output (no text and no tool calls).
    if !has_text_content(input.response_messages) {
        return TurnControl::Error(TurnControlError::EmptyOutput);
    }

    // 6. Final text answer.
    TurnControl::Stop(StopReason::FinalText)
}

/// Extract all pending tool calls from a slice of response messages.
///
/// Tool calls are surfaced either as `ContentBlock::ToolUse` blocks or via
/// the legacy `Message::tool_calls` field. This function normalizes both.
pub fn pending_tool_calls(messages: &[Message]) -> Vec<opensquilla_core::types::ToolCall> {
    let mut calls = Vec::new();
    for msg in messages {
        if !matches!(msg.role, MessageRole::Assistant) {
            continue;
        }
        // Content-block form (preferred).
        for block in &msg.content {
            if let ContentBlock::ToolUse(call) = block {
                calls.push(call.clone());
            }
        }
        // Legacy field form.
        if let Some(extra) = &msg.tool_calls {
            calls.extend(extra.iter().cloned());
        }
    }
    calls
}

/// Validate a slice of tool calls, returning the first error if any.
pub fn validate_tool_calls(
    calls: &[opensquilla_core::types::ToolCall],
) -> Result<(), TurnControlError> {
    for call in calls {
        if call.id.trim().is_empty() {
            return Err(TurnControlError::MalformedToolCall(
                "tool call missing id".to_string(),
            ));
        }
        if call.name.trim().is_empty() {
            return Err(TurnControlError::MalformedToolCall(format!(
                "tool call {} missing name",
                call.id
            )));
        }
        if call.input.is_null() {
            return Err(TurnControlError::MalformedToolCall(format!(
                "tool call {} ({}) has null input",
                call.id, call.name
            )));
        }
    }
    Ok(())
}

/// Returns true if the stop reason indicates the model refused to respond.
pub fn is_refusal(stop_reason: Option<&str>) -> bool {
    matches!(stop_reason, Some(reason) if reason.eq_ignore_ascii_case("refusal"))
}

/// Returns true if the stop reason indicates an explicit end of turn.
pub fn is_end_turn(stop_reason: Option<&str>) -> bool {
    matches!(
        stop_reason,
        Some(reason) if reason.eq_ignore_ascii_case("end_turn")
            || reason.eq_ignore_ascii_case("stop")
            || reason.eq_ignore_ascii_case("tool_use")
            || reason.eq_ignore_ascii_case("max_tokens")
    )
}

/// Returns true if any response message contains non-empty text content.
fn has_text_content(messages: &[Message]) -> bool {
    messages
        .iter()
        .filter(|m| matches!(m.role, MessageRole::Assistant))
        .flat_map(|m| m.content.iter())
        .any(|block| {
            matches!(block, ContentBlock::Text(t) if !t.trim().is_empty())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, Message, MessageRole, ToolCall};
    use serde_json::json;

    fn assistant_with_tool_call() -> Message {
        Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::ToolUse(ToolCall::new(
                "tc_1",
                "read_file",
                json!({"path": "/tmp"}),
            ))],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    fn assistant_with_text() -> Message {
        Message::assistant("done")
    }

    #[test]
    fn test_continue_on_tool_calls() {
        let msgs = vec![assistant_with_tool_call()];
        let input = TurnControlInput {
            stop_reason: Some("tool_use"),
            response_messages: &msgs,
            tool_round: 0,
            max_tool_rounds: 10,
        };
        assert_eq!(decide_turn_control(&input), TurnControl::Continue);
    }

    #[test]
    fn test_stop_on_max_rounds() {
        let msgs = vec![assistant_with_tool_call()];
        let input = TurnControlInput {
            stop_reason: Some("tool_use"),
            response_messages: &msgs,
            tool_round: 9,
            max_tool_rounds: 10,
        };
        assert_eq!(
            decide_turn_control(&input),
            TurnControl::Stop(StopReason::MaxToolRoundsReached)
        );
    }

    #[test]
    fn test_end_turn_no_tool_calls() {
        let msgs = vec![assistant_with_text()];
        let input = TurnControlInput {
            stop_reason: Some("end_turn"),
            response_messages: &msgs,
            tool_round: 0,
            max_tool_rounds: 10,
        };
        assert_eq!(
            decide_turn_control(&input),
            TurnControl::Stop(StopReason::EndTurn)
        );
    }

    #[test]
    fn test_final_text_no_stop_reason() {
        let msgs = vec![assistant_with_text()];
        let input = TurnControlInput {
            stop_reason: None,
            response_messages: &msgs,
            tool_round: 0,
            max_tool_rounds: 10,
        };
        assert_eq!(
            decide_turn_control(&input),
            TurnControl::Stop(StopReason::FinalText)
        );
    }

    #[test]
    fn test_refusal() {
        let input = TurnControlInput {
            stop_reason: Some("refusal"),
            response_messages: &[],
            tool_round: 0,
            max_tool_rounds: 10,
        };
        assert_eq!(
            decide_turn_control(&input),
            TurnControl::Error(TurnControlError::Refusal)
        );
    }

    #[test]
    fn test_empty_output() {
        let input = TurnControlInput {
            stop_reason: None,
            response_messages: &[],
            tool_round: 0,
            max_tool_rounds: 10,
        };
        assert_eq!(
            decide_turn_control(&input),
            TurnControl::Error(TurnControlError::EmptyOutput)
        );
    }

    #[test]
    fn test_pending_tool_calls_legacy_field() {
        let mut msg = assistant_with_text();
        msg.tool_calls = Some(vec![ToolCall::new("tc_2", "shell", json!({}))]);
        let calls = pending_tool_calls(&[msg]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "tc_2");
    }

    #[test]
    fn test_validate_tool_calls_rejects_empty_id() {
        let bad = vec![ToolCall::new("", "shell", json!({}))];
        assert!(matches!(
            validate_tool_calls(&bad),
            Err(TurnControlError::MalformedToolCall(_))
        ));
    }
}
