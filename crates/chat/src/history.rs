use opensquilla_core::types::{Message, MessageRole};

/// Options for trimming conversation history.
#[derive(Debug, Clone)]
pub struct HistoryTrimOptions {
    /// Maximum number of messages to keep.
    pub max_messages: usize,
    /// Always keep at least this many trailing messages.
    pub keep_trailing: usize,
    /// Whether to preserve system messages while trimming.
    pub preserve_system: bool,
}

impl Default for HistoryTrimOptions {
    fn default() -> Self {
        Self {
            max_messages: 100,
            keep_trailing: 20,
            preserve_system: true,
        }
    }
}

/// Trim a conversation history to fit within a budget.
///
/// The trailing `keep_trailing` messages are always kept, system messages are
/// preserved when `preserve_system` is set, and the remaining budget is filled
/// from the front of the history. Order is preserved.
pub fn trim_history(messages: Vec<Message>, options: &HistoryTrimOptions) -> Vec<Message> {
    if messages.len() <= options.max_messages {
        return messages;
    }

    let keep_trailing = options.keep_trailing.min(options.max_messages);
    let trailing_start = messages.len() - keep_trailing;
    let front_budget = options.max_messages - keep_trailing;

    let mut out: Vec<Message> = Vec::with_capacity(options.max_messages);
    let mut kept = 0usize;
    for (idx, msg) in messages.into_iter().enumerate() {
        let is_trailing = idx >= trailing_start;
        let is_system = options.preserve_system && matches!(msg.role, MessageRole::System);
        if is_trailing || is_system || kept < front_budget {
            out.push(msg);
            kept += 1;
        }
        // otherwise: drop the message
    }
    out
}
