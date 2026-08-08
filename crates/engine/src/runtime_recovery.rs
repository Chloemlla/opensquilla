//! Empty / no-progress turn recovery.
//!
//! Pure decision functions that detect when a turn produces no output (or
//! repeatedly makes no progress) and decide the recovery action: retry with a
//! prompt variant, rephrase the request, or abort the turn.
//!
//! This module performs NO I/O. It consumes snapshots of turn state and
//! returns a `RecoveryAction`.

use opensquilla_core::types::{ContentBlock, Message, MessageRole};

/// The action to take after detecting a stalled or empty turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the same request up to `max_retries` times.
    Retry,
    /// Rephrase the request (e.g. append a clarifying instruction) and retry.
    Rephrase,
    /// Abort the turn and surface the failure to the caller.
    Abort,
}

/// The reason a turn was flagged as requiring recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallReason {
    /// The model produced no assistant message at all.
    NoAssistantMessage,
    /// The assistant message contained no text and no tool calls.
    EmptyOutput,
    /// The assistant repeated the same text verbatim across rounds.
    NoProgress,
}

/// A classification of a single turn round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundStatus {
    /// The round produced useful output.
    Produced,
    /// The round produced no output.
    Empty,
    /// The round repeated prior output without progress.
    Stalled,
}

/// Inputs for the stall/recovery decision.
#[derive(Debug, Clone)]
pub struct RecoveryInput<'a> {
    /// The message history up to and including the current round.
    pub messages: &'a [Message],
    /// The response messages from the current round only.
    pub round_messages: &'a [Message],
    /// The number of consecutive empty/stalled rounds so far.
    pub consecutive_empty_rounds: u32,
    /// The maximum number of consecutive empty rounds tolerated.
    pub max_empty_rounds: u32,
    /// The total number of retries already attempted.
    pub retries_attempted: u32,
    /// The maximum number of retries allowed.
    pub max_retries: u32,
}

/// Classify whether the current round produced output.
pub fn classify_round(messages: &[Message]) -> RoundStatus {
    let assistant: Vec<&Message> = messages
        .iter()
        .filter(|m| matches!(m.role, MessageRole::Assistant))
        .collect();

    if assistant.is_empty() {
        return RoundStatus::Empty;
    }

    let last = assistant.last().unwrap();
    let last_text = last.text_content();
    let has_tool_calls = last
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse(_)))
        || last
            .tool_calls
            .as_ref()
            .map(|c| !c.is_empty())
            .unwrap_or(false);

    if last_text.trim().is_empty() && !has_tool_calls {
        return RoundStatus::Empty;
    }

    // Detect verbatim repetition of the previous assistant message.
    if assistant.len() >= 2 {
        let prev = assistant[assistant.len() - 2];
        if !prev.text_content().trim().is_empty()
            && prev.text_content() == last_text
            && prev.content.len() == last.content.len()
        {
            return RoundStatus::Stalled;
        }
    }

    RoundStatus::Produced
}

/// Detect whether a turn produced no output at all.
pub fn is_empty_turn(messages: &[Message]) -> bool {
    messages
        .iter()
        .filter(|m| matches!(m.role, MessageRole::Assistant))
        .all(|m| {
            let has_tool_calls = m
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse(_)))
                || m.tool_calls
                    .as_ref()
                    .map(|c| !c.is_empty())
                    .unwrap_or(false);
            m.text_content().trim().is_empty() && !has_tool_calls
        })
}

/// Decide the recovery action for a stalled or empty turn.
///
/// # Decision order
///
/// 1. If the current round produced output, no recovery is needed (`None`).
/// 2. If the consecutive-empty count exceeds `max_empty_rounds`, abort.
/// 3. If retries are exhausted, abort.
/// 4. Otherwise, rephrase when the stall looks like an empty model response
///    (a rephrase is more likely to elicit output than a plain retry), and
///    retry otherwise.
pub fn decide_recovery(input: &RecoveryInput<'_>) -> Option<RecoveryAction> {
    // Classify against the full history so stall detection (repetition of the
    // previous assistant message) is visible across rounds.
    let status = classify_round(input.messages);

    // The round produced useful output — no recovery needed.
    if status == RoundStatus::Produced {
        return None;
    }

    let empty_rounds = input
        .consecutive_empty_rounds
        .saturating_add(if status == RoundStatus::Empty { 1 } else { 0 });

    // Too many consecutive empty rounds.
    if empty_rounds >= input.max_empty_rounds.max(1) {
        return Some(RecoveryAction::Abort);
    }

    // Retry budget exhausted.
    if input.retries_attempted >= input.max_retries {
        return Some(RecoveryAction::Abort);
    }

    // A rephrase is preferred when the last round was completely empty; a
    // plain retry is preferred for stalled (repetitive) output.
    match status {
        RoundStatus::Empty => Some(RecoveryAction::Rephrase),
        RoundStatus::Stalled => Some(RecoveryAction::Retry),
        RoundStatus::Produced => None,
    }
}

/// Compute the stall reason for logging / diagnostics.
pub fn stall_reason(messages: &[Message]) -> Option<StallReason> {
    let status = classify_round(messages);
    match status {
        RoundStatus::Empty => {
            let has_assistant = messages
                .iter()
                .any(|m| matches!(m.role, MessageRole::Assistant));
            if has_assistant {
                Some(StallReason::EmptyOutput)
            } else {
                Some(StallReason::NoAssistantMessage)
            }
        }
        RoundStatus::Stalled => Some(StallReason::NoProgress),
        RoundStatus::Produced => None,
    }
}

/// A rephrasing strategy, used to vary prompts on retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RephraseStyle {
    /// Append a "please be thorough" instruction.
    Thorough,
    /// Ask for a direct answer with no preamble.
    Direct,
    /// Ask for a step-by-step breakdown.
    StepByStep,
}

/// Build a rephrased version of a user prompt.
pub fn rephrase_prompt(prompt: &str, style: RephraseStyle) -> String {
    match style {
        RephraseStyle::Thorough => {
            format!("{prompt}\n\n(Please answer thoroughly and completely.)")
        }
        RephraseStyle::Direct => {
            format!("{prompt}\n\n(Please give a direct answer without preamble.)")
        }
        RephraseStyle::StepByStep => {
            format!("{prompt}\n\n(Please explain your reasoning step by step.)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    #[test]
    fn test_classify_produced() {
        assert_eq!(classify_round(&[assistant("hello")]), RoundStatus::Produced);
    }

    #[test]
    fn test_classify_empty() {
        let empty = Message {
            role: MessageRole::Assistant,
            content: vec![],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        assert_eq!(classify_round(&[empty]), RoundStatus::Empty);
    }

    #[test]
    fn test_classify_stalled() {
        let msgs = vec![assistant("same"), assistant("same")];
        assert_eq!(classify_round(&msgs), RoundStatus::Stalled);
    }

    #[test]
    fn test_no_recovery_when_produced() {
        let input = RecoveryInput {
            messages: &[assistant("done")],
            round_messages: &[assistant("done")],
            consecutive_empty_rounds: 0,
            max_empty_rounds: 3,
            retries_attempted: 0,
            max_retries: 3,
        };
        assert_eq!(decide_recovery(&input), None);
    }

    #[test]
    fn test_rephrase_on_empty() {
        let empty = Message {
            role: MessageRole::Assistant,
            content: vec![],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let input = RecoveryInput {
            messages: std::slice::from_ref(&empty),
            round_messages: &[empty],
            consecutive_empty_rounds: 1,
            max_empty_rounds: 3,
            retries_attempted: 0,
            max_retries: 3,
        };
        assert_eq!(decide_recovery(&input), Some(RecoveryAction::Rephrase));
    }

    #[test]
    fn test_abort_on_max_empty() {
        let empty = Message {
            role: MessageRole::Assistant,
            content: vec![],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        let input = RecoveryInput {
            messages: std::slice::from_ref(&empty),
            round_messages: &[empty],
            consecutive_empty_rounds: 3,
            max_empty_rounds: 3,
            retries_attempted: 0,
            max_retries: 3,
        };
        assert_eq!(decide_recovery(&input), Some(RecoveryAction::Abort));
    }

    #[test]
    fn test_rephrase_prompt() {
        assert!(rephrase_prompt("hi", RephraseStyle::Thorough).contains("thoroughly"));
        assert!(rephrase_prompt("hi", RephraseStyle::Direct).contains("direct"));
        assert!(rephrase_prompt("hi", RephraseStyle::StepByStep).contains("step by step"));
    }
}
