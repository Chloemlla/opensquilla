//! Reasoning cleanup for cross-provider compatibility.
//!
//! Some providers (e.g. Anthropic) emit `reasoning` content blocks, while
//! others (e.g. OpenAI-compatible endpoints) reject messages that contain
//! reasoning blocks. This module strips or preserves reasoning content
//! depending on the target provider's capabilities.

use opensquilla_core::types::{ContentBlock, Message};

/// Describes whether a provider accepts reasoning content in messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSupport {
    /// The provider accepts and may emit reasoning blocks.
    Supported,
    /// The provider rejects reasoning blocks; strip them before sending.
    Unsupported,
    /// The provider is streaming reasoning deltas; keep reasoning but mark it.
    Streaming,
}

/// Strip all reasoning blocks from a single message.
///
/// Returns the cleaned message. If the message becomes empty after stripping
/// (no text, no tool calls, no tool results), `None` is returned so the caller
/// can drop it entirely.
pub fn drop_reasoning_from_message(message: &Message) -> Option<Message> {
    let mut clone = message.clone();
    clone
        .content
        .retain(|block| !matches!(block, ContentBlock::Reasoning { .. }));
    if clone.content.is_empty() {
        None
    } else {
        Some(clone)
    }
}

/// Strip reasoning blocks from a sequence of messages.
///
/// Messages that become empty after stripping are removed. Returns the number
/// of blocks stripped alongside the cleaned sequence.
pub fn drop_reasoning(messages: &[Message]) -> (Vec<Message>, usize) {
    let mut stripped = 0usize;
    let cleaned: Vec<Message> = messages
        .iter()
        .filter_map(|msg| {
            let mut clone = msg.clone();
            let before = clone.content.len();
            clone
                .content
                .retain(|block| !matches!(block, ContentBlock::Reasoning { .. }));
            stripped += before - clone.content.len();
            if clone.content.is_empty() {
                None
            } else {
                Some(clone)
            }
        })
        .collect();
    (cleaned, stripped)
}

/// Conditionally strip reasoning based on the target provider's support.
///
/// This is the primary entry point used by the pipeline when preparing
/// messages for a provider request.
pub fn sanitize_for_provider(messages: &[Message], support: ReasoningSupport) -> Vec<Message> {
    match support {
        ReasoningSupport::Supported | ReasoningSupport::Streaming => messages.to_vec(),
        ReasoningSupport::Unsupported => drop_reasoning(messages).0,
    }
}

/// Preserve reasoning but relocate it out of the visible content stream.
///
/// Some providers require reasoning to be reported separately (e.g. via a
/// side channel) rather than inline in the message content. This function
/// extracts all reasoning blocks from assistant messages into a parallel
/// list, returning the visible messages and the extracted reasoning.
pub fn extract_reasoning(messages: &[Message]) -> (Vec<Message>, Vec<String>) {
    let mut reasoning = Vec::new();
    let mut visible: Vec<Message> = Vec::new();
    for msg in messages {
        let mut clone = msg.clone();
        clone.content.retain(|block| match block {
            ContentBlock::Reasoning { reasoning: ref r } => {
                reasoning.push(r.clone());
                false
            }
            _ => true,
        });
        visible.push(clone);
    }
    (visible, reasoning)
}

/// Returns the number of reasoning blocks present in a message sequence.
pub fn count_reasoning_blocks(messages: &[Message]) -> usize {
    messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter(|b| matches!(b, ContentBlock::Reasoning { .. }))
        .count()
}

/// Determine whether a provider name is known to support reasoning content.
///
/// This is a conservative heuristic used when no explicit configuration is
/// available. Providers that emit native "thinking" blocks are treated as
/// supported; everything else defaults to unsupported (safe to strip).
pub fn default_reasoning_support(provider_name: &str) -> ReasoningSupport {
    let name = provider_name.to_ascii_lowercase();
    if name.contains("anthropic")
        || name.contains("claude")
        || name.contains("deepseek")
        || name.contains("gemini")
        || name.contains("kimi")
    {
        ReasoningSupport::Supported
    } else {
        ReasoningSupport::Unsupported
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::MessageRole;

    fn msg_with_reasoning() -> Message {
        Message {
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Reasoning { reasoning: "think step by step".to_string() },
                ContentBlock::Text { text: "final answer".to_string() },
            ],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        }
    }

    #[test]
    fn test_drop_reasoning() {
        let (cleaned, stripped) = drop_reasoning(&[msg_with_reasoning()]);
        assert_eq!(stripped, 1);
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].text_content(), "final answer");
    }

    #[test]
    fn test_sanitize_unsupported() {
        let cleaned = sanitize_for_provider(&[msg_with_reasoning()], ReasoningSupport::Unsupported);
        assert_eq!(count_reasoning_blocks(&cleaned), 0);
    }

    #[test]
    fn test_sanitize_supported() {
        let kept = sanitize_for_provider(&[msg_with_reasoning()], ReasoningSupport::Supported);
        assert_eq!(count_reasoning_blocks(&kept), 1);
    }

    #[test]
    fn test_extract_reasoning() {
        let (visible, reasoning) = extract_reasoning(&[msg_with_reasoning()]);
        assert_eq!(reasoning, vec!["think step by step".to_string()]);
        assert_eq!(visible[0].text_content(), "final answer");
    }

    #[test]
    fn test_default_support_heuristic() {
        assert_eq!(
            default_reasoning_support("anthropic"),
            ReasoningSupport::Supported
        );
        assert_eq!(
            default_reasoning_support("openai"),
            ReasoningSupport::Unsupported
        );
    }
}
