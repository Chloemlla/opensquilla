//! Conversation history management.
//!
//! Pure data transformations over message sequences:
//! - Truncation to a context-window budget.
//! - Tool call pairing repair (ensure every `tool_use` has a matching
//!   `tool_result`, and drop orphaned results).
//! - Persisted-entry reconstruction (recover a `Message` from a flattened
//!   transcript row).
//! - Message deduplication.

use opensquilla_core::types::{
    ContentBlock, Message, MessageRole, ToolCall, ToolResult,
};

/// An immutable identifier for a message, used for deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MessageFingerprint {
    role: MessageRole,
    text: String,
    tool_use_ids: Vec<String>,
}

impl MessageFingerprint {
    /// Compute a fingerprint for a message.
    ///
    /// Two messages with the same role, text content, and tool call IDs are
    /// considered duplicates regardless of tool-result content, which may
    /// legitimately vary between retries.
    pub fn of(message: &Message) -> Self {
        let mut tool_use_ids = Vec::new();
        for block in &message.content {
            if let ContentBlock::ToolUse(call) = block {
                tool_use_ids.push(call.id.clone());
            }
        }
        if let Some(calls) = &message.tool_calls {
            tool_use_ids.extend(calls.iter().map(|c| c.id.clone()));
        }
        Self {
            role: message.role.clone(),
            text: message.text_content(),
            tool_use_ids,
        }
    }
}

/// The result of repairing a message sequence.
#[derive(Debug, Clone)]
pub struct RepairOutcome {
    /// The repaired message sequence.
    pub messages: Vec<Message>,
    /// The number of orphaned `tool_use` blocks that had no matching result.
    pub unpaired_calls: usize,
    /// The number of orphaned `tool_result` blocks removed.
    pub removed_results: usize,
}

/// Truncate a message sequence to fit within a token budget.
///
/// The system message (and any leading system messages) are always preserved.
/// Everything else is retained from the end (most recent) backwards until the
/// budget is exhausted. This is a pure token *estimate* based on character
/// length; the caller may pass an exact tokenizer if available.
pub fn truncate_to_budget(
    messages: &[Message],
    max_tokens: u64,
    tokens_per_char: f64,
) -> Vec<Message> {
    if max_tokens == 0 {
        return Vec::new();
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

    let system_tokens: u64 = system.iter().map(|m| estimate_tokens(m, tokens_per_char)).sum();
    let mut budget = max_tokens.saturating_sub(system_tokens);
    if budget == 0 && !system.is_empty() {
        // Extremely tight budget: keep only the system context.
        return system;
    }

    let mut kept: Vec<Message> = Vec::new();
    for msg in tail.into_iter().rev() {
        let cost = estimate_tokens(&msg, tokens_per_char);
        if cost > budget {
            // A single message overflows the remaining budget. Always preserve
            // the most recent message (the current request) even if it does
            // not fit; stop including anything older.
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

/// Estimate the token count of a message using character length.
pub fn estimate_tokens(message: &Message, tokens_per_char: f64) -> u64 {
    let chars = message
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => t.len(),
            ContentBlock::Reasoning(r) => r.len(),
            ContentBlock::ToolUse(c) => c.name.len() + c.input.to_string().len(),
            ContentBlock::ToolResult(r) => r.content.len(),
        })
        .sum::<usize>();
    (chars as f64 * tokens_per_char).ceil() as u64
}

/// Repair a message sequence so that every `tool_use` has a matching
/// `tool_result` and no `tool_result` is left without its `tool_use`.
///
/// This is necessary because providers may be interrupted mid-loop, persisted
/// transcripts may be truncated, or a tool call may have been executed but its
/// result never appended.
pub fn repair_tool_pairs(messages: &[Message]) -> RepairOutcome {
    let mut repaired: Vec<Message> = Vec::new();
    let mut unpaired_calls = 0usize;
    let mut removed_results = 0usize;
    // Tool call ids that have a result in the same window.
    let mut satisfied: std::collections::HashSet<String> = std::collections::HashSet::new();

    // First pass: collect all tool call ids so we can determine which results
    // are orphaned.
    let mut all_call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in messages {
        collect_tool_call_ids(msg, &mut all_call_ids);
    }

    for msg in messages {
        let mut clone = msg.clone();

        // Strip tool_result blocks from the message and decide whether to keep
        // the message at all. An assistant message whose content becomes empty
        // after stripping orphaned results is dropped.
        let mut has_live_content = false;
        clone.content.retain(|block| match block {
            ContentBlock::ToolResult(result) => {
                if all_call_ids.contains(&result.tool_use_id) {
                    satisfied.insert(result.tool_use_id.clone());
                    has_live_content = true;
                    true
                } else {
                    removed_results += 1;
                    false
                }
            }
            other => {
                has_live_content = true;
                true
            }
        });

        // The legacy `tool_result` field.
        if let Some(result) = &clone.tool_result {
            if all_call_ids.contains(&result.tool_use_id) {
                satisfied.insert(result.tool_use_id.clone());
            } else {
                removed_results += 1;
                clone.tool_result = None;
            }
        }

        // Tool messages with no result content are dropped entirely.
        let is_empty_tool_result = matches!(clone.role, MessageRole::Tool)
            && clone.content.is_empty()
            && clone.tool_result.is_none();

        if !is_empty_tool_result && (has_live_content || !clone.content.is_empty()) {
            repaired.push(clone);
        }
    }

    // Count tool calls that never got a result.
    for id in all_call_ids {
        if !satisfied.contains(&id) {
            unpaired_calls += 1;
        }
    }

    RepairOutcome {
        messages: repaired,
        unpaired_calls,
        removed_results,
    }
}

/// Remove orphaned `tool_result` blocks whose `tool_use` is missing.
///
/// Returns the cleaned sequence and the number of blocks removed.
pub fn drop_orphaned_results(messages: &[Message]) -> (Vec<Message>, usize) {
    let mut call_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for msg in messages {
        collect_tool_call_ids(msg, &mut call_ids);
    }

    let mut removed = 0usize;
    let cleaned: Vec<Message> = messages
        .iter()
        .map(|msg| {
            let mut clone = msg.clone();
            clone.content.retain(|block| match block {
                ContentBlock::ToolResult(result) if !call_ids.contains(&result.tool_use_id) => {
                    removed += 1;
                    false
                }
                _ => true,
            });
            clone
        })
        .collect();

    (cleaned, removed)
}

/// Deduplicate adjacent and repeated messages.
///
/// Consecutive messages with the same role and identical content are merged
/// (only the first is kept). Additionally, any duplicate message (by
/// `MessageFingerprint`) that appears after the first occurrence is removed.
/// Tool result messages are always preserved.
pub fn deduplicate(messages: &[Message]) -> Vec<Message> {
    let mut seen: std::collections::HashSet<MessageFingerprint> =
        std::collections::HashSet::new();
    let mut out: Vec<Message> = Vec::new();

    for msg in messages {
        if matches!(msg.role, MessageRole::Tool) {
            out.push(msg.clone());
            continue;
        }
        let fp = MessageFingerprint::of(msg);
        if seen.insert(fp) {
            out.push(msg.clone());
        }
    }
    out
}

/// A flattened transcript row, as persisted in the session database.
#[derive(Debug, Clone)]
pub struct TranscriptRow {
    /// The message role as stored (e.g. "system", "user", "assistant", "tool").
    pub role: String,
    /// The text content of the message, if any.
    pub text: Option<String>,
    /// The name of the sender (tool name, function name), if any.
    pub name: Option<String>,
    /// The tool call this row responds to (for tool results), if any.
    pub tool_call_id: Option<String>,
    /// Serialized tool calls, if any.
    pub tool_calls_json: Option<String>,
    /// The tool result payload, if any.
    pub tool_result_json: Option<String>,
    /// The reasoning/thinking content, if any.
    pub reasoning: Option<String>,
}

/// Reconstruct a `Message` from a flattened transcript row.
pub fn reconstruct_from_row(row: &TranscriptRow) -> Result<Message, String> {
    let role = parse_role(&row.role)?;

    let mut content: Vec<ContentBlock> = Vec::new();
    if let Some(reasoning) = &row.reasoning {
        if !reasoning.is_empty() {
            content.push(ContentBlock::Reasoning(reasoning.clone()));
        }
    }
    if let Some(text) = &row.text {
        if !text.is_empty() {
            content.push(ContentBlock::Text(text.clone()));
        }
    }

    // Tool calls from JSON.
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(json) = &row.tool_calls_json {
        let calls: Vec<ToolCall> = serde_json::from_str(json).map_err(|e| {
            format!("failed to parse tool_calls for row: {e}")
        })?;
        tool_calls.extend(calls);
    }
    for call in &tool_calls {
        content.push(ContentBlock::ToolUse(call.clone()));
    }

    // Tool result.
    let tool_result: Option<ToolResult> = match &row.tool_result_json {
        Some(json) if !json.trim().is_empty() => {
            let result: ToolResult = serde_json::from_str(json)
                .map_err(|e| format!("failed to parse tool_result for row: {e}"))?;
            content.push(ContentBlock::ToolResult(result.clone()));
            Some(result)
        }
        _ => None,
    };

    Ok(Message {
        role,
        content,
        name: row.name.clone(),
        tool_call_id: row.tool_call_id.clone(),
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_result,
    })
}

/// Parse a persisted role string into a `MessageRole`.
pub fn parse_role(role: &str) -> Result<MessageRole, String> {
    match role.to_ascii_lowercase().as_str() {
        "system" => Ok(MessageRole::System),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" | "function" => Ok(MessageRole::Tool),
        other => Err(format!("unknown message role: {other}")),
    }
}

fn collect_tool_call_ids(message: &Message, ids: &mut std::collections::HashSet<String>) {
    for block in &message.content {
        if let ContentBlock::ToolUse(call) = block {
            ids.insert(call.id.clone());
        }
    }
    if let Some(calls) = &message.tool_calls {
        for call in calls {
            ids.insert(call.id.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_use_message(id: &str) -> Message {
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

    fn tool_result_message(id: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: vec![ContentBlock::ToolResult(ToolResult::success(id, "ok"))],
            name: Some("shell".to_string()),
            tool_call_id: Some(id.to_string()),
            tool_calls: None,
            tool_result: None,
        }
    }

    #[test]
    fn test_truncate_preserves_system() {
        let msgs = vec![
            Message::system("instructions"),
            Message::user("hello"),
            Message::assistant("hi"),
        ];
        let truncated = truncate_to_budget(&msgs, 10, 1.0);
        assert_eq!(truncated[0].role, MessageRole::System);
    }

    #[test]
    fn test_repair_removes_orphan_result() {
        let msgs = vec![
            tool_use_message("a"),
            tool_result_message("a"),
            tool_result_message("missing"),
        ];
        let outcome = repair_tool_pairs(&msgs);
        assert_eq!(outcome.unpaired_calls, 0);
        assert_eq!(outcome.removed_results, 1);
        assert_eq!(outcome.messages.len(), 2);
    }

    #[test]
    fn test_repair_counts_unpaired_calls() {
        let msgs = vec![tool_use_message("orphan")];
        let outcome = repair_tool_pairs(&msgs);
        assert_eq!(outcome.unpaired_calls, 1);
        assert_eq!(outcome.messages.len(), 1);
    }

    #[test]
    fn test_deduplicate() {
        let msgs = vec![
            Message::user("hello"),
            Message::user("hello"),
            Message::assistant("world"),
        ];
        let deduped = deduplicate(&msgs);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_reconstruct_row() {
        let row = TranscriptRow {
            role: "assistant".to_string(),
            text: Some("hello".to_string()),
            name: None,
            tool_call_id: None,
            tool_calls_json: None,
            tool_result_json: None,
            reasoning: Some("thinking".to_string()),
        };
        let msg = reconstruct_from_row(&row).unwrap();
        assert_eq!(msg.role, MessageRole::Assistant);
        assert_eq!(msg.text_content(), "hello");
        assert!(matches!(msg.content[0], ContentBlock::Reasoning(_)));
    }
}
