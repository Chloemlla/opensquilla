//! Session-derived memory documents.
//!
//! Generates memory documents from completed sessions, extracting patterns and
//! preferences from the conversation and linking each document back to the
//! source session. The [`SessionSource`] component is the "consolidation" layer
//! for episodic memory: raw turn captures are aggregated into a distilled
//! session-level document that is easier to retrieve and cite.
//!
//! The richer [`SessionMemorySource`] adds typed extractions (knowledge,
//! preferences, skills) with per-fact confidence scores and a structured
//! [`MemoryDocument`] that can be stored or used directly by callers.

use chrono::{DateTime, Utc};
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::{MemoryId, Message, MessageRole, SessionId};
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use crate::types::MemoryEntry;
use crate::MemoryStore;

/// Configuration for session-derived memory generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSourceConfig {
    /// Maximum length of a generated session document.
    pub max_doc_len: usize,
    /// Whether to extract and store user preferences.
    pub extract_preferences: bool,
    /// Whether to extract and store domain patterns.
    pub extract_patterns: bool,
    /// Whether to store a full transcript summary.
    pub store_summary: bool,
    /// Importance assigned to session-derived documents.
    pub default_importance: f64,
}

impl Default for SessionSourceConfig {
    fn default() -> Self {
        Self {
            max_doc_len: 8_192,
            extract_preferences: true,
            extract_patterns: true,
            store_summary: true,
            default_importance: 0.6,
        }
    }
}

/// A distilled memory document derived from one session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMemoryDoc {
    /// The session this document was derived from.
    pub session_id: SessionId,
    /// The agent the session belonged to.
    pub agent_id: Uuid,
    /// A concise summary of what the session accomplished.
    pub summary: String,
    /// Explicit or implicit user preferences observed in the session.
    pub preferences: Vec<String>,
    /// Recurring topics / patterns observed in the session.
    pub patterns: Vec<String>,
    /// Tags attached to the generated document.
    pub tags: Vec<String>,
    /// Importance score.
    pub importance: f64,
}

impl SessionMemoryDoc {
    /// Render the document into a single memory content string.
    pub fn to_content(&self, include_sections: bool) -> String {
        let mut out = String::new();
        if include_sections {
            out.push_str(&format!("## Session Summary\n{}\n", self.summary));
            if !self.preferences.is_empty() {
                out.push_str("\n## Preferences\n");
                for p in &self.preferences {
                    out.push_str(&format!("- {}\n", p));
                }
            }
            if !self.patterns.is_empty() {
                out.push_str("\n## Patterns\n");
                for p in &self.patterns {
                    out.push_str(&format!("- {}\n", p));
                }
            }
        } else {
            out.push_str(&self.summary);
        }
        out
    }
}

/// Generates memory documents from completed sessions.
#[derive(Clone)]
pub struct SessionSource {
    store: MemoryStore,
    config: SessionSourceConfig,
}

impl SessionSource {
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            config: SessionSourceConfig::default(),
        }
    }

    pub fn with_config(store: MemoryStore, config: SessionSourceConfig) -> Self {
        Self { store, config }
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    pub fn config(&self) -> &SessionSourceConfig {
        &self.config
    }

    /// Derive a session memory document from a session's messages and persist
    /// it as a memory entry linked back to the source session.
    pub fn derive_and_store(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
        metadata: serde_json::Value,
    ) -> CoreResult<Option<MemoryId>> {
        let doc = self.derive(session_id, agent_id, messages)?;

        if doc.summary.is_empty() && doc.preferences.is_empty() && doc.patterns.is_empty() {
            debug!("session {} produced no derivable content", session_id);
            return Ok(None);
        }

        let content = doc.to_content(true);
        let now = Utc::now();
        let id = MemoryId(Uuid::new_v4());
        let entry = MemoryEntry {
            id,
            agent_id,
            content,
            tags: doc.tags.clone(),
            embedding: None,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source: "session_source".to_string(),
            memory_type: "session_document".to_string(),
            importance: doc.importance,
            importance_score: doc.importance,
            access_count: 0,
            metadata: serde_json::json!({
                "session_id": session_id.to_string(),
                "summary": doc.summary,
                "preferences": doc.preferences,
                "patterns": doc.patterns,
                "derived": metadata,
            }),
        };

        self.store.insert_memory(&entry)?;
        debug!(
            "Stored session-derived memory {} for session {}",
            id.0, session_id
        );
        Ok(Some(id))
    }

    /// Build a [`SessionMemoryDoc`] from messages without persisting it.
    pub fn derive(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
    ) -> CoreResult<SessionMemoryDoc> {
        let mut user_texts: Vec<String> = Vec::new();
        let mut assistant_texts: Vec<String> = Vec::new();

        for msg in messages {
            let text = msg.text_content();
            if text.trim().is_empty() {
                continue;
            }
            match msg.role {
                MessageRole::User => user_texts.push(text),
                MessageRole::Assistant => assistant_texts.push(text),
                _ => {}
            }
        }

        // Heuristic summary: first user request + first assistant answer.
        let summary = summarize_session(&user_texts, &assistant_texts, self.config.max_doc_len);

        // Heuristic preferences: repeated comparative phrasing.
        let preferences = if self.config.extract_preferences {
            extract_preferences(&user_texts)
        } else {
            Vec::new()
        };

        // Heuristic patterns: frequently occurring meaningful words.
        let patterns = if self.config.extract_patterns {
            extract_patterns(&user_texts, &assistant_texts)
        } else {
            Vec::new()
        };

        let mut tags = vec![
            "session".to_string(),
            "session_document".to_string(),
        ];
        if !preferences.is_empty() {
            tags.push("preference".to_string());
        }
        if !patterns.is_empty() {
            tags.push("pattern".to_string());
        }

        Ok(SessionMemoryDoc {
            session_id,
            agent_id,
            summary,
            preferences,
            patterns,
            tags,
            importance: self.config.default_importance,
        })
    }
}

// ---------------------------------------------------------------------------
// Rich extraction: SessionMemorySource / MemoryDocument
// ---------------------------------------------------------------------------

/// A confidence score in `[0, 1]` attached to an extracted fact.
///
/// Confidence is a measure of how reliably a fact could be inferred from the
/// raw session, combining signal strength and the message role that produced
/// it.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SourceConfidence {
    pub score: f64,
}

impl SourceConfidence {
    pub fn new(score: f64) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
        }
    }

    pub fn high() -> Self {
        Self::new(0.9)
    }

    pub fn medium() -> Self {
        Self::new(0.6)
    }

    pub fn low() -> Self {
        Self::new(0.3)
    }

    pub fn value(&self) -> f64 {
        self.score
    }
}

impl Default for SourceConfidence {
    fn default() -> Self {
        Self::medium()
    }
}

/// The semantic kind of an extracted fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactType {
    Knowledge,
    Preference,
    Skill,
    Pattern,
}

impl FactType {
    pub fn label(&self) -> &'static str {
        match self {
            FactType::Knowledge => "knowledge",
            FactType::Preference => "preference",
            FactType::Skill => "skill",
            FactType::Pattern => "pattern",
        }
    }
}

/// A single fact extracted from a session, with provenance and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub content: String,
    pub fact_type: FactType,
    pub confidence: SourceConfidence,
    /// Indices into the message list that produced this fact.
    pub source_message_indices: Vec<usize>,
}

impl ExtractedFact {
    pub fn new(content: impl Into<String>, fact_type: FactType, confidence: SourceConfidence) -> Self {
        Self {
            content: content.into(),
            fact_type,
            confidence,
            source_message_indices: Vec::new(),
        }
    }
}

/// A structured memory document derived from one session, with typed facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryDocument {
    pub session_id: SessionId,
    pub agent_id: Uuid,
    pub title: String,
    pub summary: String,
    pub facts: Vec<ExtractedFact>,
    pub source_turns: u64,
    pub created_at: DateTime<Utc>,
}

impl MemoryDocument {
    /// Render the document into a memory content string.
    pub fn to_content(&self) -> String {
        let mut out = String::new();
        if !self.title.is_empty() {
            out.push_str(&format!("# {}\n\n", self.title));
        }
        if !self.summary.is_empty() {
            out.push_str(&format!("## Summary\n{}\n\n", self.summary));
        }
        let mut groups: Vec<(&str, Vec<&ExtractedFact>)> = Vec::new();
        for ft in [FactType::Knowledge, FactType::Preference, FactType::Skill, FactType::Pattern] {
            let facts: Vec<&ExtractedFact> = self.facts.iter().filter(|f| f.fact_type == ft).collect();
            if !facts.is_empty() {
                groups.push((ft.label(), facts));
            }
        }
        for (label, facts) in groups {
            out.push_str(&format!("## {}\n", capitalize_first(label)));
            for fact in facts {
                out.push_str(&format!(
                    "- [{}] {}\n",
                    format_confidence(fact.confidence.score),
                    fact.content
                ));
            }
            out.push('\n');
        }
        out
    }

    /// The mean confidence across all facts.
    pub fn average_confidence(&self) -> f64 {
        if self.facts.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.facts.iter().map(|f| f.confidence.score).sum();
        sum / self.facts.len() as f64
    }

    /// The number of facts of a given type.
    pub fn fact_count(&self, fact_type: FactType) -> usize {
        self.facts.iter().filter(|f| f.fact_type == fact_type).count()
    }
}

fn format_confidence(score: f64) -> String {
    if score >= 0.8 {
        "high".to_string()
    } else if score >= 0.5 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

/// Capitalize the first character of a string.
fn capitalize_first(s: &str) -> String {
    let mut it = s.chars();
    match it.next() {
        Some(c) => c.to_uppercase().collect::<String>() + it.as_str(),
        None => String::new(),
    }
}

/// Rich session memory derivation: typed extraction with confidence scoring.
#[derive(Clone)]
pub struct SessionMemorySource {
    store: MemoryStore,
    config: SessionSourceConfig,
}

impl SessionMemorySource {
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            config: SessionSourceConfig::default(),
        }
    }

    pub fn with_config(store: MemoryStore, config: SessionSourceConfig) -> Self {
        Self { store, config }
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Extract factual knowledge from a session.
    pub fn extract_knowledge(&self, session: &[Message]) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        for (idx, msg) in session.iter().enumerate() {
            let text = msg.text_content();
            for sentence in split_sentences(&text) {
                if let Some(fact) = extract_fact_sentence(&sentence) {
                    let confidence = estimate_sentence_confidence(&sentence, msg.role);
                    facts.push(ExtractedFact {
                        content: fact,
                        fact_type: FactType::Knowledge,
                        confidence,
                        source_message_indices: vec![idx],
                    });
                }
            }
        }
        facts
    }

    /// Extract user preferences from a session.
    pub fn extract_preferences(&self, session: &[Message]) -> Vec<ExtractedFact> {
        let mut facts = Vec::new();
        for (idx, msg) in session.iter().enumerate() {
            if msg.role != MessageRole::User {
                continue;
            }
            let text = msg.text_content();
            for sentence in split_sentences(&text) {
                if let Some(pref) = extract_preference_sentence(&sentence) {
                    let confidence = estimate_sentence_confidence(&sentence, msg.role);
                    facts.push(ExtractedFact {
                        content: pref,
                        fact_type: FactType::Preference,
                        confidence,
                        source_message_indices: vec![idx],
                    });
                }
            }
        }
        facts
    }

    /// Extract skill usage patterns from a session (tool call aggregation).
    pub fn extract_skills(&self, session: &[Message]) -> Vec<ExtractedFact> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for msg in session {
            if msg.role == MessageRole::Assistant {
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        *counts.entry(call.name.clone()).or_default() += 1;
                    }
                }
            }
            // Tool results that name the tool also count toward skill usage.
            if msg.role == MessageRole::Tool {
                if let Some(name) = &msg.name {
                    if !name.is_empty() {
                        *counts.entry(name.clone()).or_default() += 1;
                    }
                }
            }
        }

        let mut facts = Vec::new();
        for (tool, count) in counts {
            let confidence = skill_confidence(&tool, count);
            facts.push(ExtractedFact {
                content: format!("Uses the {} tool ({} time{} this session)", tool, count, if count == 1 { "" } else { "s" }),
                fact_type: FactType::Skill,
                confidence,
                source_message_indices: Vec::new(),
            });
        }
        facts.sort_by(|a, b| b.confidence.score.partial_cmp(&a.confidence.score).unwrap_or(std::cmp::Ordering::Equal));
        facts
    }

    /// Build a structured [`MemoryDocument`] from a session.
    pub fn build_memory_document(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
    ) -> MemoryDocument {
        let mut facts = Vec::new();
        if self.config.extract_preferences {
            facts.extend(self.extract_preferences(messages));
        }
        facts.extend(self.extract_knowledge(messages));
        facts.extend(self.extract_skills(messages));
        if self.config.extract_patterns {
            facts.extend(extract_pattern_facts(messages));
        }
        let facts = self.deduplicate(facts);

        MemoryDocument {
            session_id,
            agent_id,
            title: summarize_title(messages),
            summary: summarize_document(messages),
            facts,
            source_turns: messages.len() as u64,
            created_at: Utc::now(),
        }
    }

    /// Derive a memory document and persist it as a memory entry.
    pub fn derive_and_store(
        &self,
        session_id: SessionId,
        agent_id: Uuid,
        messages: &[Message],
    ) -> CoreResult<Option<MemoryId>> {
        let doc = self.build_memory_document(session_id, agent_id, messages);
        if doc.facts.is_empty() && doc.summary.is_empty() {
            return Ok(None);
        }

        let content = doc.to_content();
        let now = Utc::now();
        let id = MemoryId(Uuid::new_v4());
        let tags = vec![
            "session".to_string(),
            "session_document".to_string(),
            "rich".to_string(),
        ];
        let fact_types: Vec<String> = doc
            .facts
            .iter()
            .map(|f| f.fact_type.label().to_string())
            .collect();
        let entry = MemoryEntry {
            id,
            agent_id,
            content,
            tags,
            embedding: None,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source: "session_source".to_string(),
            memory_type: "session_document".to_string(),
            importance: self.config.default_importance,
            importance_score: self.config.default_importance,
            access_count: 0,
            metadata: serde_json::json!({
                "session_id": session_id.to_string(),
                "title": doc.title,
                "summary": doc.summary,
                "average_confidence": doc.average_confidence(),
                "fact_types": fact_types,
                "source_turns": doc.source_turns,
            }),
        };

        self.store.insert_memory(&entry)?;
        debug!(
            "Stored rich session memory {} for session {} ({} facts)",
            id.0,
            session_id,
            doc.facts.len()
        );
        Ok(Some(id))
    }

    /// Remove duplicate facts, keeping the highest-confidence copy.
    pub fn deduplicate(&self, facts: Vec<ExtractedFact>) -> Vec<ExtractedFact> {
        let mut by_key: std::collections::HashMap<String, ExtractedFact> = std::collections::HashMap::new();
        for fact in facts {
            let key = normalize_content(&fact.content);
            if key.is_empty() {
                continue;
            }
            match by_key.get_mut(&key) {
                Some(existing) => {
                    if fact.confidence.score > existing.confidence.score {
                        *existing = fact;
                    }
                }
                None => {
                    by_key.insert(key, fact);
                }
            }
        }
        let mut out: Vec<ExtractedFact> = by_key.into_values().collect();
        out.sort_by(|a, b| b.confidence.score.partial_cmp(&a.confidence.score).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

const FACT_MARKERS: &[&str] = &[
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
    "supports",
];

const PREFERENCE_MARKERS: &[&str] = &[
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

/// Extract a factual sentence, returning the cleaned sentence if it looks like
/// a durable fact.
fn extract_fact_sentence(sentence: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    if lower.chars().count() < 8 || lower.chars().count() > 240 {
        return None;
    }
    if FACT_MARKERS.iter().any(|m| lower.contains(m)) {
        let cleaned = clean_sentence(sentence);
        if !cleaned.is_empty() {
            return Some(cleaned);
        }
    }
    None
}

/// Extract a preference sentence, returning the distilled preference.
fn extract_preference_sentence(sentence: &str) -> Option<String> {
    let lower = sentence.to_lowercase();
    if lower.chars().count() > 240 {
        return None;
    }
    for marker in PREFERENCE_MARKERS {
        if let Some(idx) = lower.find(marker) {
            let start = (idx + marker.len()).min(sentence.len());
            let Some(rest_slice) = sentence.get(start..) else {
                continue;
            };
            let rest = rest_slice.trim();
            if !rest.is_empty() {
                return Some(format!("{} {}", marker.trim(), rest));
            }
        }
    }
    None
}

/// Estimate how reliable a sentence is as a source of a fact.
fn estimate_sentence_confidence(sentence: &str, role: MessageRole) -> SourceConfidence {
    let lower = sentence.to_lowercase();
    let mut base = match role {
        MessageRole::User => 0.7,
        MessageRole::Assistant => 0.5,
        _ => 0.4,
    };
    // Strong first-person signals increase confidence.
    for strong in ["i prefer", "i like", "i love", "i want", "i need", "i use", "i avoid"] {
        if lower.contains(strong) {
            base += 0.15;
            break;
        }
    }
    // Explicit hedges decrease confidence.
    for hedge in ["maybe", "perhaps", "i think", "might", "possibly", "not sure"] {
        if lower.contains(hedge) {
            base -= 0.2;
            break;
        }
    }
    SourceConfidence::new(base)
}

/// Compute a confidence score for a tool skill based on usage count.
fn skill_confidence(tool: &str, count: usize) -> SourceConfidence {
    let _ = tool;
    let base = match count {
        0 => 0.2,
        1 => 0.5,
        2 => 0.65,
        3..=4 => 0.8,
        _ => 0.9,
    };
    SourceConfidence::new(base)
}

/// Extract recurring significant words as typed pattern facts.
fn extract_pattern_facts(messages: &[Message]) -> Vec<ExtractedFact> {
    let texts: Vec<String> = messages.iter().map(|m| m.text_content()).collect();
    let words = extract_pattern_words(&texts);
    words
        .into_iter()
        .map(|w| ExtractedFact {
            content: format!("Recurring topic: {}", w),
            fact_type: FactType::Pattern,
            confidence: SourceConfidence::low(),
            source_message_indices: Vec::new(),
        })
        .collect()
}

/// Extract the top recurring significant words from a set of texts.
fn extract_pattern_words(texts: &[String]) -> Vec<String> {
    let stop_words: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "and", "or", "but", "not", "this",
        "that", "it", "its", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our",
        "their", "what", "which", "who", "when", "where", "why", "how", "all", "can", "will",
        "would", "should", "just", "very", "please", "want", "need", "help",
    ]
    .iter()
    .cloned()
    .collect();

    let mut word_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for text in texts {
        let unique: std::collections::HashSet<String> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| {
                w.len() > 3
                    && !stop_words.contains(w)
                    && w.chars().all(|c| c.is_alphabetic())
            })
            .map(|w| w.to_string())
            .collect();
        for word in unique {
            *word_freq.entry(word).or_default() += 1;
        }
    }

    let total_docs = texts.len().max(1);
    let mut words: Vec<String> = word_freq
        .into_iter()
        .filter(|(_, count)| *count >= 2 && *count as f64 / total_docs as f64 > 0.15)
        .map(|(word, _)| word)
        .collect();
    words.sort();
    words.truncate(20);
    words
}

/// Normalize content for deduplication.
fn normalize_content(s: &str) -> String {
    s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strip leading filler words and trailing punctuation from a sentence.
fn clean_sentence(sentence: &str) -> String {
    sentence
        .trim()
        .trim_end_matches(|c: char| c == '.' || c == '!' || c == '?' || c == ',')
        .trim()
        .to_string()
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Build a short title from the first user message.
fn summarize_title(messages: &[Message]) -> String {
    for msg in messages {
        if msg.role == MessageRole::User {
            let text = msg.text_content();
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                let mut chars = trimmed.chars();
                let first: String = chars.by_ref().take(80).collect();
                let mut title = first;
                if chars.next().is_some() {
                    title.push('…');
                }
                return title;
            }
        }
    }
    "Session".to_string()
}

/// Build a short document summary from the first user/assistant exchange.
fn summarize_document(messages: &[Message]) -> String {
    let mut user_texts: Vec<String> = Vec::new();
    let mut assistant_texts: Vec<String> = Vec::new();
    for msg in messages {
        let text = msg.text_content();
        if text.trim().is_empty() {
            continue;
        }
        match msg.role {
            MessageRole::User => user_texts.push(text),
            MessageRole::Assistant => assistant_texts.push(text),
            _ => {}
        }
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(first_user) = user_texts.first() {
        parts.push(format!("User asked: {}", first_user.trim()));
    }
    if let Some(first_assistant) = assistant_texts.first() {
        parts.push(format!("Assistant responded: {}", first_assistant.trim()));
    }
    let mut joined = parts.join("\n");
    if joined.chars().count() > 512 {
        let truncated: String = joined.chars().take(512).collect();
        joined = truncated;
        joined.push('…');
    }
    joined
}

/// Build a summary line from the first user request and first assistant reply.
fn summarize_session(
    user_texts: &[String],
    assistant_texts: &[String],
    max_len: usize,
) -> String {
    let first_user = user_texts.first().map(|s| s.trim()).unwrap_or("").to_string();
    let first_assistant = assistant_texts.first().map(|s| s.trim()).unwrap_or("").to_string();

    let mut parts: Vec<String> = Vec::new();
    if !first_user.is_empty() {
        parts.push(format!("User asked: {}", first_user));
    }
    if !first_assistant.is_empty() {
        parts.push(format!("Assistant responded: {}", first_assistant));
    }

    let mut joined = parts.join("\n");
    if joined.chars().count() > max_len {
        let truncated: String = joined.chars().take(max_len).collect();
        joined = truncated;
        joined.push('…');
    }
    joined
}

/// Extract potential user preferences via simple pattern matching.
///
/// Heuristic: look for `"I (prefer|like|want|need|love|use|avoid)"` followed by
/// content up to a sentence boundary. This is deliberately conservative; an
/// LLM-backed extractor can replace it later.
fn extract_preferences(user_texts: &[String]) -> Vec<String> {
    let mut prefs: Vec<String> = Vec::new();
    for text in user_texts {
        let lower = text.to_lowercase();
        let markers = [
            "i prefer",
            "i like",
            "i love",
            "i want",
            "i need",
            "i use",
            "i avoid",
            "prefer",
        ];
        for marker in markers {
            if let Some(idx) = lower.find(marker) {
                let start = idx + marker.len();
                let rest = &text[start..];
                // Take up to the next sentence-ending punctuation or newline.
                let end = rest
                    .find(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
                    .unwrap_or(rest.len().min(160));
                let candidate = rest[..end].trim();
                if !candidate.is_empty() {
                    let pref = format!("{} {}", marker, candidate);
                    if !prefs.iter().any(|p| p == &pref) {
                        prefs.push(pref);
                    }
                }
                break;
            }
        }
    }
    prefs
}

/// Extract recurring significant words as patterns (stop-word filtered).
fn extract_patterns(user_texts: &[String], assistant_texts: &[String]) -> Vec<String> {
    let stop_words: std::collections::HashSet<&str> = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "and", "or", "but", "not", "this",
        "that", "it", "its", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our",
        "their", "what", "which", "who", "when", "where", "why", "how", "all", "can", "will",
        "would", "should", "just", "very", "please", "want", "need", "help",
    ]
    .iter()
    .cloned()
    .collect();

    let mut word_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for text in user_texts.iter().chain(assistant_texts.iter()) {
        let words: std::collections::HashSet<String> = text
            .to_lowercase()
            .split_whitespace()
            .filter(|w| {
                w.len() > 3
                    && !stop_words.contains(w)
                    && w.chars().all(|c| c.is_alphabetic())
            })
            .map(|w| w.to_string())
            .collect();
        for word in words {
            *word_freq.entry(word).or_default() += 1;
        }
    }

    let total_docs = (user_texts.len() + assistant_texts.len()).max(1);
    let mut patterns: Vec<String> = word_freq
        .into_iter()
        .filter(|(_, count)| *count >= 2 && *count as f64 / total_docs as f64 > 0.15)
        .map(|(word, _)| word)
        .collect();
    patterns.sort();
    patterns.truncate(20);
    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_session_document() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionSource::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            Message::user("I prefer concise answers and I use rust daily"),
            Message::assistant("Understood. Rust and concise answers it is."),
            Message::user("I also like dark mode interfaces"),
            Message::assistant("Noted, dark mode preference saved."),
        ];
        let doc = source.derive(session, agent, &messages).unwrap();
        assert!(!doc.summary.is_empty());
        assert!(!doc.preferences.is_empty());
        assert!(!doc.tags.is_empty());
    }

    #[test]
    fn test_derive_and_store() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionSource::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            Message::user("I prefer using tokio for async work"),
            Message::assistant("Great, tokio it is."),
        ];
        let id = source
            .derive_and_store(session, agent, &messages, serde_json::json!({}))
            .unwrap();
        assert!(id.is_some());

        let all = store.list_memories(&agent, None, 100, 0).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].memory_type, "session_document");
        assert_eq!(all[0].source, "session_source");
        assert_eq!(
            all[0].metadata["session_id"].as_str(),
            Some(&session.to_string())
        );
    }

    #[test]
    fn test_no_content_returns_none() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionSource::new(store);
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let id = source
            .derive_and_store(session, agent, &[], serde_json::json!({}))
            .unwrap();
        assert!(id.is_none());
    }

    #[test]
    fn test_extract_knowledge_facts() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store);
        let messages = vec![
            Message::user("My company is located in Berlin and uses rust."),
            Message::assistant("Noted. Berlin-based, rust shop."),
        ];
        let facts = source.extract_knowledge(&messages);
        assert!(!facts.is_empty());
        assert!(facts.iter().all(|f| f.fact_type == FactType::Knowledge));
        assert!(facts.iter().all(|f| f.confidence.score > 0.0 && f.confidence.score <= 1.0));
    }

    #[test]
    fn test_extract_preferences_typed() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store);
        let messages = vec![
            Message::user("I prefer dark mode and I love terse answers."),
        ];
        let facts = source.extract_preferences(&messages);
        assert!(!facts.is_empty());
        assert!(facts.iter().all(|f| f.fact_type == FactType::Preference));
    }

    #[test]
    fn test_extract_skills() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store);
        let messages = vec![
            Message::assistant("Running search"),
            Message {
                role: MessageRole::Assistant,
                content: vec![],
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![opensquilla_core::types::ToolCall::new(
                    "call_1",
                    "search_files",
                    serde_json::json!({}),
                )]),
                tool_result: None,
            },
        ];
        let facts = source.extract_skills(&messages);
        assert!(facts.iter().any(|f| f.content.contains("search_files")));
    }

    #[test]
    fn test_build_memory_document() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store);
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            Message::user("I prefer concise code and use rust daily."),
            Message::assistant("Great, I will keep answers concise."),
        ];
        let doc = source.build_memory_document(session, agent, &messages);
        assert!(!doc.facts.is_empty());
        assert!(!doc.summary.is_empty());
        assert!(doc.average_confidence() > 0.0);
        assert!(doc.fact_count(FactType::Preference) >= 1);
    }

    #[test]
    fn test_rich_derive_and_store() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store.clone());
        let agent = Uuid::new_v4();
        let session = SessionId::new();
        let messages = vec![
            Message::user("I prefer using tokio for async work"),
            Message::assistant("tokio it is."),
        ];
        let id = source.derive_and_store(session, agent, &messages).unwrap();
        assert!(id.is_some());
        let all = store.list_memories(&agent, None, 100, 0).unwrap();
        assert_eq!(all.len(), 1);
        assert!(all[0].content.contains("Preference"));
    }

    #[test]
    fn test_deduplicate_keeps_highest_confidence() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionMemorySource::new(store);
        let facts = vec![
            ExtractedFact {
                content: "user prefers dark mode".to_string(),
                fact_type: FactType::Preference,
                confidence: SourceConfidence::low(),
                source_message_indices: vec![0],
            },
            ExtractedFact {
                content: "User prefers dark mode".to_string(),
                fact_type: FactType::Preference,
                confidence: SourceConfidence::high(),
                source_message_indices: vec![1],
            },
        ];
        let deduped = source.deduplicate(facts);
        assert_eq!(deduped.len(), 1);
        assert!((deduped[0].confidence.score - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_to_content_includes_sections() {
        let store = MemoryStore::in_memory().unwrap();
        let source = SessionSource::new(store);
        let doc = source
            .derive(
                SessionId::new(),
                Uuid::new_v4(),
                &[Message::user("hello"), Message::assistant("world")],
            )
            .unwrap();
        let content = doc.to_content(true);
        assert!(content.contains("Session Summary"));
        assert!(content.contains("hello"));
    }
}
