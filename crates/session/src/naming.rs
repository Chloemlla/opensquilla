use chrono::Utc;
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::models::{Session, TranscriptEntry};
use crate::storage::SessionStorage;

/// Threshold of message count after which a default-named session is
/// eligible for name regeneration.
pub const REGENERATE_AFTER_MESSAGES: u64 = 10;

/// Default maximum name length.
pub const DEFAULT_MAX_NAME_CHARS: usize = 60;

/// Default maximum title length.
pub const DEFAULT_MAX_TITLE_CHARS: usize = 40;

// ---------------------------------------------------------------------------
// Strategy + options
// ---------------------------------------------------------------------------

/// How a session name is derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NamingStrategy {
    /// Use the first user message (or first entry) verbatim, truncated.
    FirstMessage,
    /// Ask an LLM to generate a concise title.
    LLMGenerated,
    /// Fill a template with session attributes.
    Template,
    /// Use a caller-provided literal name (stored in `template`).
    Custom,
}

/// Tuning knobs for the namer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingOptions {
    pub strategy: NamingStrategy,
    pub max_chars: usize,
    pub max_title_chars: usize,
    pub template: Option<String>,
    pub language: Option<String>,
}

impl Default for NamingOptions {
    fn default() -> Self {
        Self {
            strategy: NamingStrategy::FirstMessage,
            max_chars: DEFAULT_MAX_NAME_CHARS,
            max_title_chars: DEFAULT_MAX_TITLE_CHARS,
            template: None,
            language: None,
        }
    }
}

// ---------------------------------------------------------------------------
// LLM title generator protocol
// ---------------------------------------------------------------------------

/// Generates short titles/summaries for sessions. The gateway wires a
/// provider-backed implementation; when absent, the namer falls back to
/// deterministic extraction.
#[async_trait::async_trait]
pub trait TitleGenerator: Send + Sync {
    async fn generate_title(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
        max_chars: usize,
    ) -> Result<String, String>;

    async fn generate_summary(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
        max_chars: usize,
    ) -> Result<String, String>;
}

// ---------------------------------------------------------------------------
// SessionNamer
// ---------------------------------------------------------------------------

/// Auto-names sessions from their transcript, with LLM-backed generation when
/// a [`TitleGenerator`] is wired.
pub struct SessionNamer {
    storage: SessionStorage,
    options: NamingOptions,
    title_generator: Option<Arc<dyn TitleGenerator>>,
}

impl SessionNamer {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage,
            options: NamingOptions::default(),
            title_generator: None,
        }
    }

    pub fn with_strategy(mut self, strategy: NamingStrategy) -> Self {
        self.options.strategy = strategy;
        self
    }

    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.options.max_chars = max_chars;
        self
    }

    pub fn with_max_title_chars(mut self, max_title_chars: usize) -> Self {
        self.options.max_title_chars = max_title_chars;
        self
    }

    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.options.template = Some(template.into());
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.options.language = Some(language.into());
        self
    }

    pub fn with_title_generator(mut self, generator: Arc<dyn TitleGenerator>) -> Self {
        self.title_generator = Some(generator);
        self
    }

    pub fn options(&self) -> &NamingOptions {
        &self.options
    }

    // --- Name derivation ---

    /// Generate a name for a session based on its transcript and strategy.
    pub fn generate_name(&self, session_id: &Uuid) -> CoreResult<String> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.get_transcript_entries(session_id, 50, 0)?;
        Ok(self.name_for(&session, &entries))
    }

    /// Derive a name for an already-loaded session. Synchronous — the LLM
    /// strategy falls back to extraction here; use [`auto_name_async`] for
    /// the LLM path.
    pub fn name_for(&self, session: &Session, entries: &[TranscriptEntry]) -> String {
        match self.options.strategy {
            NamingStrategy::FirstMessage | NamingStrategy::LLMGenerated => {
                first_message_name(entries, self.options.max_chars)
            }
            NamingStrategy::Template => self.render_template(session, entries),
            NamingStrategy::Custom => self
                .options
                .template
                .clone()
                .unwrap_or_else(|| first_message_name(entries, self.options.max_chars)),
        }
    }

    /// Generate a short title for a session (used for UI list rows).
    pub fn generate_title(&self, session_id: &Uuid) -> CoreResult<String> {
        let entries = self.storage.get_transcript_entries(session_id, 50, 0)?;
        Ok(truncate(
            &first_message_text(&entries),
            self.options.max_title_chars,
        ))
    }

    /// Generate a summary of the conversation (deterministic fallback).
    pub fn generate_summary(&self, session_id: &Uuid) -> CoreResult<String> {
        let entries = self.storage.get_transcript_entries(session_id, 200, 0)?;
        Ok(crate::compaction::extractive_summary(&entries, 200))
    }

    /// Regenerate a session's name once the conversation has progressed past
    /// [`REGENERATE_AFTER_MESSAGES`] messages and the current name still looks
    /// like a placeholder.
    pub fn regenerate_name(&self, session_id: &Uuid) -> CoreResult<String> {
        let session = self.require_session(session_id)?;
        let current = session.name.clone();
        let looks_default = current.is_empty()
            || current.starts_with("Session ")
            || current.starts_with("New Session")
            || current.starts_with("CLI Session");
        if session.message_count >= REGENERATE_AFTER_MESSAGES && looks_default {
            let name = self.generate_name(session_id)?;
            self.update_session_name(session_id, &name)?;
            info!("Regenerated session {} name to '{}'", session_id, name);
            Ok(name)
        } else {
            Ok(current)
        }
    }

    // --- Persistence ---

    pub fn update_session_name(&self, session_id: &Uuid, name: &str) -> CoreResult<()> {
        let mut session = self.require_session(session_id)?;
        session.name = name.to_string();
        session.updated_at = Utc::now();
        self.storage.update_session(&session)?;
        info!("Updated session {} name to '{}'", session_id, name);
        Ok(())
    }

    /// Generate and persist a name synchronously.
    pub fn auto_name(&self, session_id: &Uuid) -> CoreResult<String> {
        let name = self.generate_name(session_id)?;
        self.update_session_name(session_id, &name)?;
        Ok(name)
    }

    /// Generate and persist a name, using the LLM title generator when wired.
    pub async fn auto_name_async(&self, session_id: &Uuid) -> CoreResult<String> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.get_transcript_entries(session_id, 50, 0)?;

        if let Some(generator) = &self.title_generator {
            let title = generator
                .generate_title(&session, &entries, self.options.max_chars)
                .await
                .map_err(CoreError::Provider)?;
            self.update_session_name(session_id, &title)?;
            return Ok(title);
        }

        let name = self.name_for(&session, &entries);
        self.update_session_name(session_id, &name)?;
        Ok(name)
    }

    /// Generate an LLM summary when a generator is wired, falling back to the
    /// deterministic extractive summary.
    pub async fn summarize_async(&self, session_id: &Uuid) -> CoreResult<String> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.get_transcript_entries(session_id, 200, 0)?;
        if let Some(generator) = &self.title_generator {
            let summary = generator
                .generate_summary(&session, &entries, 300)
                .await
                .map_err(CoreError::Provider)?;
            return Ok(summary);
        }
        Ok(crate::compaction::extractive_summary(&entries, 300))
    }

    fn require_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.storage
            .get_session(session_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Session {}", session_id)))
    }

    fn render_template(&self, session: &Session, entries: &[TranscriptEntry]) -> String {
        let template = self
            .options
            .template
            .clone()
            .unwrap_or_else(|| "Session {short_id}".to_string());
        let first = first_message_text(entries);
        template
            .replace("{session_id}", &session.id.to_string())
            .replace(
                "{short_id}",
                &session.id.to_string()[..8.min(session.id.to_string().len())],
            )
            .replace("{agent_id}", &session.agent_id.to_string())
            .replace(
                "{created_at}",
                &session.created_at.format("%Y-%m-%d %H:%M").to_string(),
            )
            .replace("{first_message}", &first)
            .replace("{message_count}", &session.message_count.to_string())
            .replace("{mode}", &format!("{:?}", session.mode).to_lowercase())
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn first_message_text(entries: &[TranscriptEntry]) -> String {
    for entry in entries {
        if entry.role == "user" {
            let first_line = entry.content.lines().next().unwrap_or(&entry.content);
            return first_line.trim().to_string();
        }
    }
    entries
        .first()
        .map(|e| {
            e.content
                .lines()
                .next()
                .unwrap_or(&e.content)
                .trim()
                .to_string()
        })
        .unwrap_or_default()
}

fn first_message_name(entries: &[TranscriptEntry], max_chars: usize) -> String {
    let text = first_message_text(entries);
    if text.is_empty() {
        return "New Session".to_string();
    }
    truncate(&text, max_chars)
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(max_chars.saturating_sub(3)).collect();
        out.push_str("...");
        out
    }
}

// ---------------------------------------------------------------------------
// Legacy compatibility engine
// ---------------------------------------------------------------------------

/// Auto-naming engine. Kept for API compatibility; delegates to a
/// [`SessionNamer`] with default options.
pub struct NamingEngine {
    inner: SessionNamer,
}

impl NamingEngine {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            inner: SessionNamer::new(storage),
        }
    }

    /// Access the underlying namer for customization.
    pub fn session_namer(&self) -> &SessionNamer {
        &self.inner
    }

    /// Generate a session name from the first few messages.
    pub fn generate_name(&self, session_id: &Uuid) -> CoreResult<String> {
        self.inner.generate_name(session_id)
    }

    /// Generate a short title from the first few messages.
    pub fn generate_title(&self, session_id: &Uuid) -> CoreResult<String> {
        self.inner.generate_title(session_id)
    }

    /// Update the session name in storage.
    pub fn update_session_name(&self, session_id: &Uuid, name: &str) -> CoreResult<()> {
        self.inner.update_session_name(session_id, name)
    }

    /// Auto-name a session: generate and store the name.
    pub fn auto_name(&self, session_id: &Uuid) -> CoreResult<String> {
        self.inner.auto_name(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{SessionMode, SessionStatus};

    fn namer() -> SessionNamer {
        SessionNamer::new(SessionStorage::in_memory().unwrap())
    }

    fn seed_session(namer: &SessionNamer, name: &str, messages: &[&str]) -> Uuid {
        let id = Uuid::new_v4();
        namer
            .storage
            .create_session(&Session {
                id,
                agent_id: Uuid::new_v4(),
                name: name.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                last_active_at: Utc::now(),
                status: SessionStatus::Active,
                mode: SessionMode::Chat,
                system_prompt: String::new(),
                total_tokens: 0,
                total_cost_usd: 0.0,
                message_count: messages.len() as u64,
                parent_session_id: None,
                fork_event: None,
                metadata: serde_json::Value::Null,
            })
            .unwrap();
        for (i, msg) in messages.iter().enumerate() {
            let entry = TranscriptEntry {
                id: Uuid::new_v4(),
                session_id: id,
                role: if i == 0 { "user" } else { "assistant" }.into(),
                content: msg.to_string(),
                created_at: Utc::now() + chrono::Duration::seconds(i as i64),
                token_count: 10,
                metadata: serde_json::Value::Null,
                compacted: false,
            };
            namer.storage.insert_transcript_entry(&entry).unwrap();
        }
        id
    }

    #[test]
    fn first_message_strategy_uses_user_message() {
        let n = namer();
        let id = seed_session(&n, "Session abc", &["How do I deploy Rust?", "Let me help"]);
        let name = n.generate_name(&id).unwrap();
        assert_eq!(name, "How do I deploy Rust?");
    }

    #[test]
    fn truncates_long_first_message() {
        let n = namer().with_max_chars(20);
        let long = "This is an extremely long user message that should be truncated down";
        let id = seed_session(&n, "Session abc", &[long]);
        let name = n.generate_name(&id).unwrap();
        assert!(name.ends_with("..."));
        assert!(name.chars().count() <= 20);
    }

    #[test]
    fn empty_transcript_falls_back() {
        let n = namer();
        let id = seed_session(&n, "Session abc", &[]);
        let name = n.generate_name(&id).unwrap();
        assert_eq!(name, "New Session");
    }

    #[test]
    fn template_strategy_renders_placeholders() {
        let n = namer()
            .with_strategy(NamingStrategy::Template)
            .with_template("{first_message} — {message_count} msgs");
        let id = seed_session(&n, "Session abc", &["hello", "hi"]);
        let name = n.generate_name(&id).unwrap();
        assert_eq!(name, "hello — 2 msgs");
    }

    #[test]
    fn custom_strategy_uses_literal_name() {
        let n = namer()
            .with_strategy(NamingStrategy::Custom)
            .with_template("My Project");
        let id = seed_session(&n, "Session abc", &["hello"]);
        let name = n.generate_name(&id).unwrap();
        assert_eq!(name, "My Project");
    }

    #[test]
    fn generate_title_is_short() {
        let n = namer().with_max_title_chars(10);
        let id = seed_session(&n, "Session abc", &["A very long first message here"]);
        let title = n.generate_title(&id).unwrap();
        assert!(title.chars().count() <= 10);
    }

    #[test]
    fn auto_name_persists_to_session() {
        let n = namer();
        let id = seed_session(&n, "Session abc", &["Fix the flaky test"]);
        n.auto_name(&id).unwrap();
        let session = n.storage.get_session(&id).unwrap().unwrap();
        assert_eq!(session.name, "Fix the flaky test");
    }

    #[test]
    fn regenerate_only_when_default_and_mature() {
        let n = namer();
        // Immature session: not regenerated.
        let young = seed_session(&n, "Session abc", &["alpha"]);
        let name = n.regenerate_name(&young).unwrap();
        assert_eq!(name, "Session abc");

        // Mature session with a placeholder name: regenerated.
        let msgs: Vec<String> = (0..12).map(|i| format!("message {}", i)).collect();
        let refs: Vec<&str> = msgs.iter().map(|s| s.as_str()).collect();
        let id = seed_session(&n, "Session def", &refs);
        let name = n.regenerate_name(&id).unwrap();
        assert_eq!(name, "message 0");

        // Mature session with a real name: left alone.
        let custom = seed_session(&n, "Real Name", &refs);
        let name = n.regenerate_name(&custom).unwrap();
        assert_eq!(name, "Real Name");
    }

    #[test]
    fn legacy_engine_delegates() {
        let engine = NamingEngine::new(SessionStorage::in_memory().unwrap());
        let id = seed_session(engine.session_namer(), "Session xyz", &["legacy path"]);
        assert_eq!(engine.generate_name(&id).unwrap(), "legacy path");
        engine.auto_name(&id).unwrap();
        let session = engine
            .session_namer()
            .storage
            .get_session(&id)
            .unwrap()
            .unwrap();
        assert_eq!(session.name, "legacy path");
    }
}
