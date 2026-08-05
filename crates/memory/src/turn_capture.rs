//! Turn incremental persistence.
//!
//! Captures each completed turn's messages, extracts key information (user
//! intent, assistant actions, tool results), and persists the result as memory
//! entries. The [`TurnCapture`] component is the "write path" of the memory
//! system: it runs after a turn finishes and turns the raw transcript into
//! durable, searchable memories.
//!
//! The module also provides batch capture ([`TurnCapture::capture_batch`]),
//! key-point extraction ([`TurnCapture::extract_key_points`]) and an index
//! touch hook ([`TurnCapture::update_memory_index`]) that keeps the
//! per-session capture ledger up to date.

use chrono::Utc;
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::{MemoryId, Message, MessageRole, SessionId};
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use crate::types::MemoryEntry;
use crate::{MemoryManager, MemoryStore};

/// Configuration controlling how turns are captured into memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCaptureConfig {
    /// Maximum content length for a captured memory (truncated beyond this).
    pub max_content_len: usize,
    /// Whether to store the raw transcript as a memory entry.
    pub store_raw_transcript: bool,
    /// Whether to extract and store user intent summaries.
    pub store_user_intent: bool,
    /// Whether to extract and store assistant actions.
    pub store_assistant_actions: bool,
    /// Whether to extract and store tool results.
    pub store_tool_results: bool,
    /// Importance assigned to a turn-derived memory entry.
    pub default_importance: f64,
}

impl Default for TurnCaptureConfig {
    fn default() -> Self {
        Self {
            max_content_len: 4_096,
            store_raw_transcript: true,
            store_user_intent: true,
            store_assistant_actions: true,
            store_tool_results: true,
            default_importance: 0.5,
        }
    }
}

/// A structured view of a single turn's content for memory extraction.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurnSignals {
    /// The session this turn belongs to.
    pub session_id: Option<SessionId>,
    /// The user's primary request / intent for this turn.
    pub user_intent: Option<String>,
    /// Concatenated assistant text output for this turn.
    pub assistant_actions: Vec<String>,
    /// Tool calls invoked during the turn (`name: input` summaries).
    pub tool_calls: Vec<String>,
    /// Tool results returned during the turn.
    pub tool_results: Vec<String>,
    /// The raw transcript (all messages joined), if requested.
    pub raw_transcript: Option<String>,
}

impl TurnSignals {
    /// Returns `true` if nothing meaningful was extracted.
    pub fn is_empty(&self) -> bool {
        self.user_intent.is_none()
            && self.assistant_actions.is_empty()
            && self.tool_calls.is_empty()
            && self.tool_results.is_empty()
            && self.raw_transcript.is_none()
    }
}

/// A fully-deserialized turn ready for capture. Used by
/// [`TurnCapture::capture_batch`] to process many turns at once.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnData {
    /// An optional opaque turn identifier (used for the index ledger).
    pub turn_id: Option<String>,
    /// The session this turn belongs to.
    pub session_id: SessionId,
    /// The agent this turn belongs to.
    pub agent_id: Uuid,
    /// The messages that make up the turn.
    pub messages: Vec<Message>,
    /// Arbitrary caller-supplied metadata attached to the captured memories.
    pub metadata: serde_json::Value,
}

impl TurnData {
    /// Create a new turn payload.
    pub fn new(session_id: SessionId, agent_id: Uuid, messages: Vec<Message>) -> Self {
        Self {
            turn_id: None,
            session_id,
            agent_id,
            messages,
            metadata: serde_json::Value::Null,
        }
    }

    /// Attach a turn identifier.
    pub fn with_turn_id(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    /// Attach metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// The category of a [`KeyPoint`] extracted from a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyPointCategory {
    /// What the user asked for.
    UserIntent,
    /// A decision the user or assistant made.
    Decision,
    /// An action the assistant performed.
    Action,
    /// A tool invocation.
    ToolCall,
    /// A tool result.
    ToolResult,
    /// An explicit or implicit user preference.
    Preference,
    /// A factual statement surfaced during the turn.
    Fact,
}

impl KeyPointCategory {
    /// A short, stable string label for the category.
    pub fn name(&self) -> &'static str {
        match self {
            KeyPointCategory::UserIntent => "user_intent",
            KeyPointCategory::Decision => "decision",
            KeyPointCategory::Action => "action",
            KeyPointCategory::ToolCall => "tool_call",
            KeyPointCategory::ToolResult => "tool_result",
            KeyPointCategory::Preference => "preference",
            KeyPointCategory::Fact => "fact",
        }
    }
}

/// A single extracted key point from a turn, with a category and importance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPoint {
    pub category: KeyPointCategory,
    pub text: String,
    pub importance: f64,
}

/// Aggregate statistics from a batch capture.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TurnCaptureStats {
    /// Number of turns processed.
    pub turns_captured: u64,
    /// Number of memory entries created.
    pub memories_created: u64,
    /// Number of key points extracted.
    pub key_points: u64,
}

/// Captures turn messages and persists them as memory entries.
#[derive(Clone)]
pub struct TurnCapture {
    store: MemoryStore,
    config: TurnCaptureConfig,
}

impl TurnCapture {
    /// Create a new capture component backed by the given store.
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            config: TurnCaptureConfig::default(),
        }
    }

    /// Create a capture component with a custom config.
    pub fn with_config(store: MemoryStore, config: TurnCaptureConfig) -> Self {
        Self { store, config }
    }

    /// Access the store.
    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn config(&self) -> &TurnCaptureConfig {
        &self.config
    }

    /// Capture a completed turn: extract signals and write them as memories.
    ///
    /// Returns the ids of the memories that were created.
    pub fn capture_turn(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
        metadata: serde_json::Value,
    ) -> CoreResult<Vec<MemoryId>> {
        let signals = self.extract_signals(session_id, messages);

        let mut created = Vec::new();

        if self.config.store_user_intent {
            if let Some(intent) = &signals.user_intent {
                let id = self.store_signal(
                    agent_id,
                    session_id,
                    "user_intent",
                    intent,
                    self.config.default_importance,
                    serde_json::json!({
                        "session_id": session_id.to_string(),
                        "kind": "user_intent",
                    }),
                )?;
                created.push(id);
            }
        }

        if self.config.store_assistant_actions {
            for action in &signals.assistant_actions {
                let id = self.store_signal(
                    agent_id,
                    session_id,
                    "assistant_action",
                    action,
                    self.config.default_importance,
                    serde_json::json!({
                        "session_id": session_id.to_string(),
                        "kind": "assistant_action",
                    }),
                )?;
                created.push(id);
            }
        }

        if self.config.store_tool_results {
            for result in &signals.tool_results {
                let id = self.store_signal(
                    agent_id,
                    session_id,
                    "tool_result",
                    result,
                    self.config.default_importance,
                    serde_json::json!({
                        "session_id": session_id.to_string(),
                        "kind": "tool_result",
                    }),
                )?;
                created.push(id);
            }
        }

        if self.config.store_raw_transcript {
            if let Some(transcript) = &signals.raw_transcript {
                let id = self.store_signal(
                    agent_id,
                    session_id,
                    "transcript",
                    transcript,
                    self.config.default_importance,
                    serde_json::json!({
                        "session_id": session_id.to_string(),
                        "kind": "transcript",
                    }),
                )?;
                created.push(id);
            }
        }

        // Preserve original metadata if provided.
        if !metadata.is_null() {
            if let Some(last) = created.last() {
                if let Some(mut entry) = self.store.get_memory(last)? {
                    entry.metadata = metadata;
                    self.store.update_memory(&entry)?;
                }
            }
        }

        // Record the created memories under the session's capture ledger so
        // `update_memory_index` can re-touch them later.
        if !created.is_empty() {
            self.record_capture_index(session_id, &created)?;
        }

        debug!(
            "Captured turn for session {}: {} memories created",
            session_id,
            created.len()
        );
        Ok(created)
    }

    /// Capture a batch of turns at once.
    ///
    /// Each turn is captured independently (a failure in one does not stop the
    /// others), and the aggregate statistics are returned.
    pub fn capture_batch(&self, turns: Vec<TurnData>) -> CoreResult<TurnCaptureStats> {
        let mut stats = TurnCaptureStats::default();
        for turn in turns {
            match self.capture_turn(
                turn.session_id,
                turn.agent_id,
                &turn.messages,
                turn.metadata,
            ) {
                Ok(ids) => {
                    stats.turns_captured += 1;
                    stats.memories_created += ids.len() as u64;
                }
                Err(e) => {
                    debug!(
                        "Failed to capture turn for session {}: {}",
                        turn.session_id, e
                    );
                }
            }
        }
        debug!(
            "Captured batch: {} turns, {} memories",
            stats.turns_captured, stats.memories_created
        );
        Ok(stats)
    }

    /// Extract categorized key points from a turn's messages.
    ///
    /// Combines the structured [`TurnSignals`] view with lighter-weight
    /// preference and fact detection over the raw text.
    pub fn extract_key_points(&self, session_id: SessionId, messages: &[Message]) -> Vec<KeyPoint> {
        let mut points = Vec::new();
        let signals = self.extract_signals(session_id, messages);

        if let Some(intent) = &signals.user_intent {
            points.push(KeyPoint {
                category: KeyPointCategory::UserIntent,
                text: truncate(intent, self.config.max_content_len),
                importance: 0.6,
            });
        }
        for action in &signals.assistant_actions {
            points.push(KeyPoint {
                category: KeyPointCategory::Action,
                text: truncate(action, self.config.max_content_len),
                importance: 0.4,
            });
        }
        for call in &signals.tool_calls {
            points.push(KeyPoint {
                category: KeyPointCategory::ToolCall,
                text: truncate(call, self.config.max_content_len),
                importance: 0.5,
            });
        }
        for result in &signals.tool_results {
            points.push(KeyPoint {
                category: KeyPointCategory::ToolResult,
                text: truncate(result, self.config.max_content_len),
                importance: 0.5,
            });
        }

        points.extend(extract_preference_points(messages));
        points.extend(extract_fact_points(messages));
        points
    }

    /// Capture a turn and additionally persist its preference/fact key points.
    ///
    /// This is a richer variant of [`capture_turn`][Self::capture_turn] for
    /// callers that want the distilled key points stored as first-class
    /// memories as well as the raw signals.
    pub fn capture_turn_detailed(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
        metadata: serde_json::Value,
    ) -> CoreResult<Vec<MemoryId>> {
        let mut ids = self.capture_turn(session_id, agent_id, messages, metadata)?;
        let points = self.extract_key_points(session_id, messages);
        for point in points {
            if point.category == KeyPointCategory::Preference
                || point.category == KeyPointCategory::Fact
            {
                let id = self.store_signal(
                    agent_id,
                    session_id,
                    "key_point",
                    &point.text,
                    point.importance,
                    serde_json::json!({
                        "session_id": session_id.to_string(),
                        "kind": "key_point",
                        "category": point.category.name(),
                    }),
                )?;
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// Re-touch the memories captured for a given turn/session.
    ///
    /// FTS5 keeps the search index current via triggers, so this is mostly an
    /// API hook for future re-indexing work. It reads the capture ledger and
    /// bumps the access counters of every memory captured in that turn.
    pub fn update_memory_index(&self, turn_id: &str) -> CoreResult<u64> {
        let key = capture_ledger_key(turn_id);
        let Some(raw) = self.store.get_meta(&key)? else {
            return Ok(0);
        };
        let ids: Vec<MemoryId> = serde_json::from_str(&raw).unwrap_or_default();
        let mut touched = 0u64;
        for id in &ids {
            if let Some(mut entry) = self.store.get_memory(id)? {
                entry.accessed_at = Some(Utc::now());
                entry.access_count += 1;
                self.store.update_memory(&entry)?;
                touched += 1;
            }
        }
        debug!(
            "Updated memory index for {}: {} memories touched",
            turn_id, touched
        );
        Ok(touched)
    }

    /// Record the ids created for a turn in the capture ledger.
    fn record_capture_index(&self, session_id: SessionId, created: &[MemoryId]) -> CoreResult<()> {
        let key = capture_ledger_key(&session_id.to_string());
        let existing: Vec<String> = self
            .store
            .get_meta(&key)?
            .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
            .unwrap_or_default();
        let mut merged = existing;
        for id in created {
            let s = id.0.to_string();
            if !merged.contains(&s) {
                merged.push(s);
            }
        }
        let payload = serde_json::to_string(&merged).unwrap_or_else(|_| "[]".to_string());
        self.store.set_meta(&key, &payload)
    }

    /// Extract structured signals from a turn's messages.
    pub fn extract_signals(&self, session_id: SessionId, messages: &[Message]) -> TurnSignals {
        let mut signals = TurnSignals {
            session_id: Some(session_id),
            ..Default::default()
        };

        let mut user_text: Vec<String> = Vec::new();
        let mut assistant_text: Vec<String> = Vec::new();
        let mut transcript_parts: Vec<String> = Vec::new();

        for msg in messages {
            let text = msg.text_content();
            if !text.is_empty() {
                transcript_parts.push(format!("{}: {}", role_label(&msg.role), text));
            }

            match msg.role {
                MessageRole::User => {
                    if !text.is_empty() {
                        user_text.push(text);
                    }
                }
                MessageRole::Assistant => {
                    if !text.is_empty() {
                        assistant_text.push(text);
                    }
                    if let Some(calls) = &msg.tool_calls {
                        for call in calls {
                            signals
                                .tool_calls
                                .push(format!("{} ({})", call.name, call.input));
                        }
                    }
                }
                MessageRole::Tool => {
                    if let Some(result) = &msg.tool_result {
                        let status = if result.is_error { "error" } else { "ok" };
                        signals.tool_results.push(format!(
                            "[{}] {}",
                            status,
                            truncate(&result.content, self.config.max_content_len)
                        ));
                    } else if !text.is_empty() {
                        signals
                            .tool_results
                            .push(truncate(&text, self.config.max_content_len));
                    }
                }
                MessageRole::System => {
                    // System prompts are not generally memory-worthy, but keep
                    // them in the transcript.
                }
            }
        }

        if !user_text.is_empty() {
            signals.user_intent = Some(user_text.join("\n"));
        }
        if !assistant_text.is_empty() {
            signals.assistant_actions = vec![assistant_text.join("\n")];
        }

        if !transcript_parts.is_empty() {
            let transcript = transcript_parts.join("\n");
            signals.raw_transcript = Some(truncate(&transcript, self.config.max_content_len));
        }

        signals
    }

    /// Persist one extracted signal as a memory entry.
    fn store_signal(
        &self,
        agent_id: Uuid,
        session_id: SessionId,
        memory_type: &str,
        content: &str,
        importance: f64,
        metadata: serde_json::Value,
    ) -> CoreResult<MemoryId> {
        let now = Utc::now();
        let id = MemoryId(Uuid::new_v4());
        let mut entry = MemoryEntry {
            id,
            agent_id,
            content: truncate(content, self.config.max_content_len),
            tags: vec![memory_type.to_string(), "turn".to_string()],
            embedding: None,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source: "turn_capture".to_string(),
            memory_type: memory_type.to_string(),
            importance,
            importance_score: importance,
            access_count: 0,
            metadata: serde_json::json!({
                "session_id": session_id.to_string(),
                "capture": metadata,
            }),
        };
        // Apply the caller's metadata (may override session_id).
        entry.metadata["session_id"] = serde_json::json!(session_id.to_string());
        self.store.insert_memory(&entry)?;
        Ok(id)
    }
}

fn role_label(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max_len).collect();
        out.push('…');
        out
    }
}

/// The meta-table key that stores the capture ledger for a turn/session.
fn capture_ledger_key(turn_id: &str) -> String {
    format!("turn_capture:{}", turn_id)
}

/// Extract user preference key points from user messages.
fn extract_preference_points(messages: &[Message]) -> Vec<KeyPoint> {
    const MARKERS: &[&str] = &[
        "i prefer",
        "i like",
        "i love",
        "i want",
        "i need",
        "i use",
        "i avoid",
        "prefer",
        "favorite",
        "favourite",
        "please use",
    ];
    let mut points = Vec::new();
    for msg in messages {
        if msg.role != MessageRole::User {
            continue;
        }
        let text = msg.text_content();
        let lower = text.to_lowercase();
        for marker in MARKERS {
            if let Some(idx) = lower.find(marker) {
                let start = (idx + marker.len()).min(text.len());
                let Some(rest_slice) = text.get(start..) else {
                    continue;
                };
                let rest = rest_slice.trim();
                let end = rest
                    .find(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
                    .unwrap_or(rest.len().min(200));
                let candidate = &rest[..end];
                if !candidate.trim().is_empty() {
                    points.push(KeyPoint {
                        category: KeyPointCategory::Preference,
                        text: format!("{} {}", marker.trim(), candidate.trim()),
                        importance: 0.7,
                    });
                }
                break;
            }
        }
    }
    points
}

/// Extract factual key points from user and assistant messages.
fn extract_fact_points(messages: &[Message]) -> Vec<KeyPoint> {
    const MARKERS: &[&str] = &[
        "is a",
        "is an",
        "works at",
        "works on",
        "located in",
        "based in",
        "built with",
        "written in",
        "requires",
        "depends on",
        "uses",
        "runs on",
        "was created",
        "made with",
        "is used for",
    ];
    let mut points = Vec::new();
    for msg in messages {
        let text = msg.text_content();
        for sentence in split_sentences(&text) {
            let sl = sentence.to_lowercase();
            if MARKERS.iter().any(|m| sl.contains(m)) && sentence.chars().count() <= 240 {
                points.push(KeyPoint {
                    category: KeyPointCategory::Fact,
                    text: sentence,
                    importance: 0.45,
                });
            }
        }
    }
    points
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Convenience: capture a turn through a [`MemoryManager`], using its
/// embedding provider automatically.
pub async fn capture_turn_via_manager(
    manager: &MemoryManager,
    session_id: SessionId,
    agent_id: Uuid,
    messages: &[Message],
    metadata: serde_json::Value,
) -> CoreResult<Vec<MemoryId>> {
    let capture = TurnCapture::new(manager.store().clone());
    let ids = capture.capture_turn(session_id, agent_id, messages, metadata)?;

    // Ensure the manager's in-memory agent index knows about the new memories.
    for id in &ids {
        manager.note_memory_id(agent_id, *id);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, ToolCall, ToolResult};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    fn tool_message(name: &str, content: &str) -> Message {
        let mut msg = Message {
            role: MessageRole::Tool,
            content: vec![],
            name: Some(name.to_string()),
            tool_call_id: Some("call_1".to_string()),
            tool_calls: None,
            tool_result: Some(ToolResult::success("call_1", content)),
        };
        msg.content.push(ContentBlock::Text(content.to_string()));
        msg
    }

    fn assistant_with_tool_call(name: &str) -> Message {
        let call = ToolCall::new("call_1", name, serde_json::json!({"arg": 1}));
        Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text("running tool".to_string())],
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![call]),
            tool_result: None,
        }
    }

    #[test]
    fn test_extract_signals() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store);
        let session = SessionId::new();
        let messages = vec![
            user("Please compute fibonacci of 10"),
            assistant_with_tool_call("exec_command"),
            tool_message("exec_command", "55"),
            assistant("The 10th fibonacci number is 55."),
        ];
        let signals = capture.extract_signals(session, &messages);
        assert!(signals.user_intent.is_some());
        assert!(!signals.tool_calls.is_empty());
        assert!(!signals.tool_results.is_empty());
        assert!(!signals.assistant_actions.is_empty());
        assert!(signals.raw_transcript.is_some());
    }

    #[test]
    fn test_capture_turn_persists() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            user("Remember that I prefer dark mode"),
            assistant("Got it, I'll remember dark mode preference."),
        ];
        let ids = capture
            .capture_turn(session, agent, &messages, serde_json::json!({}))
            .unwrap();
        assert!(!ids.is_empty());

        let all = store.list_memories(&agent, None, 100, 0).unwrap();
        assert!(!all.is_empty());
        // Everything should have the turn_capture source.
        assert!(all.iter().all(|m| m.source == "turn_capture"));
    }

    #[test]
    fn test_capture_batch() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let turns = vec![
            TurnData::new(
                session,
                agent,
                vec![user("Turn one: remember I like rust"), assistant("Noted.")],
            ),
            TurnData::new(
                session,
                agent,
                vec![
                    user("Turn two: remember I prefer tokio"),
                    assistant("Noted."),
                ],
            ),
        ];
        let stats = capture.capture_batch(turns).unwrap();
        assert_eq!(stats.turns_captured, 2);
        assert!(stats.memories_created >= 4);
        assert_eq!(
            store.list_memories(&agent, None, 100, 0).unwrap().len(),
            stats.memories_created as usize
        );
    }

    #[test]
    fn test_extract_key_points() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store);
        let session = SessionId::new();
        let messages = vec![
            user("I prefer dark mode. Please use vim."),
            assistant_with_tool_call("search_files"),
            tool_message("search_files", "found 3 files"),
        ];
        let points = capture.extract_key_points(session, &messages);
        assert!(
            points
                .iter()
                .any(|p| p.category == KeyPointCategory::UserIntent)
        );
        assert!(
            points
                .iter()
                .any(|p| p.category == KeyPointCategory::ToolCall)
        );
        assert!(
            points
                .iter()
                .any(|p| p.category == KeyPointCategory::Preference)
        );
        assert!(
            points
                .iter()
                .any(|p| p.category == KeyPointCategory::ToolResult)
        );
    }

    #[test]
    fn test_capture_turn_detailed() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![user("I prefer dark mode"), assistant("Got it.")];
        let ids = capture
            .capture_turn_detailed(session, agent, &messages, serde_json::json!({}))
            .unwrap();
        // Base capture (intent + assistant action + transcript) plus the
        // preference key point.
        assert!(ids.len() >= 4);
    }

    #[test]
    fn test_update_memory_index_touches_ledger() {
        let store = MemoryStore::in_memory().unwrap();
        let capture = TurnCapture::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![user("remember something"), assistant("ok")];
        capture
            .capture_turn(session, agent, &messages, serde_json::json!({}))
            .unwrap();
        let touched = capture.update_memory_index(&session.to_string()).unwrap();
        assert!(touched >= 1);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 100), "hello");
        let long = "x".repeat(100);
        let t = truncate(&long, 10);
        assert_eq!(t.chars().count(), 11); // 10 chars + ellipsis
    }

    #[test]
    fn test_split_sentences() {
        let sentences = split_sentences("Hello world. Second sentence! Third?");
        assert_eq!(sentences.len(), 3);
    }
}
