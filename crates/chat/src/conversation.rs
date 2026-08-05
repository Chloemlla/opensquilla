use std::collections::HashMap;

use opensquilla_core::types::{Message, MessageRole, SessionId};
use serde::{Deserialize, Serialize};

use crate::history::{HistoryTrimOptions, trim_history};
use crate::source::SessionSource;

/// A conversation wrapper with metadata and source tracking.
///
/// Builds on `opensquilla_core::types::Conversation`'s message model while
/// adding session identity, source provenance, and history trimming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Stable conversation identifier.
    pub id: String,
    /// The owning session.
    pub session_id: SessionId,
    /// Messages in chronological order.
    pub messages: Vec<Message>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
    /// Where the conversation came from.
    pub source: SessionSource,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modification timestamp.
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl Conversation {
    /// Create a new empty conversation.
    pub fn new(id: impl Into<String>, session_id: SessionId, source: SessionSource) -> Self {
        let now = opensquilla_core::time::now();
        Self {
            id: id.into(),
            session_id,
            messages: Vec::new(),
            metadata: HashMap::new(),
            source,
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a message to the conversation.
    pub fn add(&mut self, message: Message) {
        self.updated_at = opensquilla_core::time::now();
        self.messages.push(message);
    }

    /// Convenience: add a text-only message.
    pub fn add_text(&mut self, role: MessageRole, text: impl Into<String>) {
        self.add(Message::text(role, text));
    }

    /// The number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the conversation has no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Trim the history using the given options.
    pub fn trim(&mut self, options: &HistoryTrimOptions) {
        self.messages = trim_history(std::mem::take(&mut self.messages), options);
    }

    /// The last message, if any.
    pub fn last(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Set a metadata key/value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// The full message text content joined by newlines.
    pub fn text(&self) -> String {
        self.messages
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new(
            opensquilla_core::id::new_id().to_string(),
            SessionId::new(),
            SessionSource::default(),
        )
    }
}
