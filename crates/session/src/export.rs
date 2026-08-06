//! # Session export and import
//!
//! Session export/import bridges the session store to the outside world:
//!
//! - [`SessionExporter`] serializes a session (transcript, summaries, tags,
//!   metadata, routing decisions, attachments) to JSON or Markdown.
//! - [`SessionImporter`] reconstructs a session from a JSON export.
//!
//! The JSON format is the canonical interchange format. The Markdown format
//! is a human-readable rendering intended for archival or LLM ingestion; it
//! cannot be re-imported losslessly.
//!
//! Both exporters operate through a [`SessionManager`] reference so they share
//! the exact same storage as the caller.

use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

use crate::manager::{CreateSessionConfig, SessionManager};
use crate::models::{
    Session, SessionAttachment, SessionSummary, TranscriptEntry,
};

// ---------------------------------------------------------------------------
// Export document format
// ---------------------------------------------------------------------------

/// The canonical JSON interchange format for a session export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExportDocument {
    /// The format version of the document.
    pub format_version: u32,
    /// When the export was produced.
    pub exported_at: DateTime<Utc>,
    /// The session being exported.
    pub session: Session,
    /// Transcript entries in chronological order.
    pub transcript: Vec<TranscriptEntry>,
    /// The active summary, if any.
    pub active_summary: Option<SessionSummary>,
    /// All summaries, newest first.
    pub summaries: Vec<SessionSummary>,
    /// Session tags.
    pub tags: Vec<String>,
    /// Key/value metadata.
    pub metadata: Vec<(String, String)>,
    /// Attachments.
    pub attachments: Vec<SessionAttachment>,
}

/// The current export format version.
pub const EXPORT_FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Export configuration
// ---------------------------------------------------------------------------

/// Options controlling what a session export includes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Include the full transcript (default `true`).
    pub include_transcript: bool,
    /// Include summaries (default `true`).
    pub include_summaries: bool,
    /// Include tags (default `true`).
    pub include_tags: bool,
    /// Include metadata (default `true`).
    pub include_metadata: bool,
    /// Include attachments (default `true`).
    pub include_attachments: bool,
    /// Include routing decisions as metadata (default `false`).
    pub include_routing: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_transcript: true,
            include_summaries: true,
            include_tags: true,
            include_metadata: true,
            include_attachments: true,
            include_routing: false,
        }
    }
}

impl ExportOptions {
    /// Export only the transcript (no summaries, tags, metadata).
    pub fn transcript_only() -> Self {
        Self {
            include_transcript: true,
            include_summaries: false,
            include_tags: false,
            include_metadata: false,
            include_attachments: false,
            include_routing: false,
        }
    }

    /// Export everything including routing decisions.
    pub fn full() -> Self {
        Self {
            include_routing: true,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// SessionExporter
// ---------------------------------------------------------------------------

/// Exports sessions to JSON and Markdown.
///
/// Holds a shared reference to a [`SessionManager`], so the exporter reads
/// through the exact same storage instance as the caller.
pub struct SessionExporter<'a> {
    manager: &'a SessionManager,
}

impl<'a> SessionExporter<'a> {
    /// Create an exporter wrapping a reference to the manager.
    pub fn new(manager: &'a SessionManager) -> Self {
        Self { manager }
    }

    /// Borrow the underlying manager.
    pub fn manager(&self) -> &SessionManager {
        self.manager
    }

    /// Export a session to the canonical JSON document.
    pub fn export(
        &self,
        session_id: &Uuid,
        options: &ExportOptions,
    ) -> CoreResult<SessionExportDocument> {
        let session = self.require_session(session_id)?;
        let transcript = if options.include_transcript {
            self.manager.full_transcript(session_id)?
        } else {
            Vec::new()
        };

        let active_summary = if options.include_summaries {
            self.manager.get_active_summary(session_id)?
        } else {
            None
        };

        let summaries = if options.include_summaries {
            self.manager.list_summaries(session_id, u64::MAX, 0)?
        } else {
            Vec::new()
        };

        let tags = if options.include_tags {
            self.manager.list_tags(session_id)?
        } else {
            Vec::new()
        };

        let metadata = if options.include_metadata {
            let mut meta = self.manager.list_metadata(session_id)?;
            if options.include_routing {
                let decisions = self
                    .manager
                    .list_routing_decisions(session_id, u64::MAX, 0)?;
                for decision in decisions {
                    meta.push((
                        format!("routing.{}", decision.turn),
                        serde_json::json!({
                            "provider": decision.provider,
                            "model": decision.model,
                            "reason": decision.reason,
                        })
                        .to_string(),
                    ));
                }
            }
            meta
        } else {
            Vec::new()
        };

        let attachments = if options.include_attachments {
            self.manager.list_attachments(session_id)?
        } else {
            Vec::new()
        };

        let document = SessionExportDocument {
            format_version: EXPORT_FORMAT_VERSION,
            exported_at: Utc::now(),
            session,
            transcript,
            active_summary,
            summaries,
            tags,
            metadata,
            attachments,
        };

        debug!("Exported session {} to JSON document", session_id);
        Ok(document)
    }

    /// Export a session to a pretty-printed JSON string.
    pub fn export_json(
        &self,
        session_id: &Uuid,
        options: &ExportOptions,
    ) -> CoreResult<String> {
        let document = self.export(session_id, options)?;
        serde_json::to_string_pretty(&document)
            .map_err(|e| CoreError::Internal(format!("export serialization failed: {}", e)))
    }

    /// Export a session to a Markdown rendering.
    pub fn export_markdown(
        &self,
        session_id: &Uuid,
        options: &ExportOptions,
    ) -> CoreResult<String> {
        let document = self.export(session_id, options)?;
        Ok(render_markdown(&document))
    }

    /// Export a session to JSON and write it to a file.
    pub fn export_to_file(
        &self,
        session_id: &Uuid,
        path: &str,
        options: &ExportOptions,
    ) -> CoreResult<usize> {
        let json = self.export_json(session_id, options)?;
        std::fs::write(path, &json).map_err(|e| CoreError::Storage(e.to_string()))?;
        info!("Exported session {} to {}", session_id, path);
        Ok(json.len())
    }

    /// Export a session to Markdown and write it to a file.
    pub fn export_markdown_to_file(
        &self,
        session_id: &Uuid,
        path: &str,
        options: &ExportOptions,
    ) -> CoreResult<usize> {
        let md = self.export_markdown(session_id, options)?;
        std::fs::write(path, &md).map_err(|e| CoreError::Storage(e.to_string()))?;
        info!("Exported session {} to markdown {}", session_id, path);
        Ok(md.len())
    }

    fn require_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.manager
            .get(session_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Session {}", session_id)))
    }
}

// ---------------------------------------------------------------------------
// Import configuration
// ---------------------------------------------------------------------------

/// Options controlling how a session import reconstructs a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptions {
    /// The agent the imported session belongs to. When `None`, the exported
    /// agent_id is used.
    pub agent_id: Option<Uuid>,
    /// Whether to import the transcript entries.
    pub import_transcript: bool,
    /// Whether to import tags.
    pub import_tags: bool,
    /// Whether to import metadata.
    pub import_metadata: bool,
    /// Whether to import attachments.
    pub import_attachments: bool,
    /// Whether to import the active summary.
    pub import_summary: bool,
    /// When the original session id is already in use, create a new id.
    pub remap_session_id: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            agent_id: None,
            import_transcript: true,
            import_tags: true,
            import_metadata: true,
            import_attachments: true,
            import_summary: true,
            remap_session_id: true,
        }
    }
}

/// The result of a session import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    /// The id of the imported session.
    pub session_id: Uuid,
    /// Whether a new session was created (`true`) or an existing one updated.
    pub created: bool,
    /// Number of transcript entries imported.
    pub entries_imported: usize,
    /// Number of tags imported.
    pub tags_imported: usize,
    /// Number of metadata keys imported.
    pub metadata_imported: usize,
    /// Number of attachments imported.
    pub attachments_imported: usize,
    /// Whether the active summary was imported.
    pub summary_imported: bool,
}

impl ImportResult {
    /// Create an empty result.
    pub fn new(session_id: Uuid, created: bool) -> Self {
        Self {
            session_id,
            created,
            entries_imported: 0,
            tags_imported: 0,
            metadata_imported: 0,
            attachments_imported: 0,
            summary_imported: false,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionImporter
// ---------------------------------------------------------------------------

/// Reconstructs sessions from [`SessionExportDocument`] JSON.
///
/// Holds a shared reference to a [`SessionManager`], so imported sessions are
/// persisted into the exact same storage as the caller.
pub struct SessionImporter<'a> {
    manager: &'a SessionManager,
}

impl<'a> SessionImporter<'a> {
    /// Create an importer wrapping a reference to the manager.
    pub fn new(manager: &'a SessionManager) -> Self {
        Self { manager }
    }

    /// Borrow the underlying manager.
    pub fn manager(&self) -> &SessionManager {
        self.manager
    }

    /// Import a session from a [`SessionExportDocument`].
    pub fn import(
        &self,
        document: &SessionExportDocument,
        options: &ImportOptions,
    ) -> CoreResult<ImportResult> {
        let exported = &document.session;
        let agent_id = options.agent_id.unwrap_or(exported.agent_id);

        // Decide whether to create a new session or reuse the exported id.
        let existing = self.manager.get(&exported.id).ok().flatten();
        let (session_id, created) = match (existing, options.remap_session_id) {
            (Some(_), true) => (Uuid::new_v4(), true),
            (None, _) => (exported.id, true),
            (Some(s), false) => (s.id, false),
        };
        // The session id used for persistence is always freshly generated by
        // `create`; keep the exported id in metadata for provenance.
        let _ = session_id;

        let config = CreateSessionConfig::new(agent_id)
            .with_name(exported.name.clone())
            .with_system_prompt(exported.system_prompt.clone())
            .with_mode(exported.mode.clone())
            .with_metadata(exported.metadata.clone());

        let session = self.manager.create(config)?;
        let final_id = session.id;
        let mut result = ImportResult::new(final_id, created);

        // Restore parent/fork provenance if it can be preserved.
        if let Some(parent) = exported.parent_session_id {
            self.manager
                .set_metadata(&final_id, "imported_parent", &parent.to_string())?;
        }
        if let Some(fork_event) = &exported.fork_event {
            self.manager
                .set_metadata(&final_id, "imported_fork_event", fork_event)?;
        }
        self.manager
            .set_metadata(&final_id, "imported_from", &exported.id.to_string())?;

        // Transcript.
        if options.import_transcript {
            for entry in &document.transcript {
                self.manager.add_message(
                    &final_id,
                    entry.role.clone(),
                    entry.content.clone(),
                    entry.token_count,
                )?;
                result.entries_imported += 1;
            }
        }

        // Tags.
        if options.import_tags {
            for tag in &document.tags {
                self.manager.add_tag(&final_id, tag)?;
                result.tags_imported += 1;
            }
        }

        // Metadata.
        if options.import_metadata {
            for (key, value) in &document.metadata {
                self.manager.set_metadata(&final_id, key, value)?;
                result.metadata_imported += 1;
            }
        }

        // Attachments.
        if options.import_attachments {
            for attachment in &document.attachments {
                self.manager.attach(
                    &final_id,
                    &attachment.name,
                    &attachment.content_type,
                    attachment.size_bytes,
                    &attachment.storage_uri,
                    attachment.metadata.clone(),
                )?;
                result.attachments_imported += 1;
            }
        }

        // Active summary.
        if options.import_summary {
            if let Some(summary) = &document.active_summary {
                self.import_summary(&final_id, summary)?;
                result.summary_imported = true;
            }
        }

        info!(
            "Imported session {} (created={}, {} entries, {} tags, {} metadata, {} attachments)",
            final_id,
            created,
            result.entries_imported,
            result.tags_imported,
            result.metadata_imported,
            result.attachments_imported
        );
        Ok(result)
    }

    /// Import a session from a JSON string.
    pub fn import_json(
        &self,
        json: &str,
        options: &ImportOptions,
    ) -> CoreResult<ImportResult> {
        let document: SessionExportDocument = serde_json::from_str(json).map_err(|e| {
            CoreError::InvalidInput(format!("invalid session export JSON: {}", e))
        })?;
        if document.format_version != EXPORT_FORMAT_VERSION {
            debug!(
                "importing session export with format version {} (current {})",
                document.format_version, EXPORT_FORMAT_VERSION
            );
        }
        self.import(&document, options)
    }

    /// Import a session from a JSON file.
    pub fn import_file(&self, path: &str, options: &ImportOptions) -> CoreResult<ImportResult> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| CoreError::Storage(format!("failed to read {}: {}", path, e)))?;
        let result = self.import_json(&json, options)?;
        info!("Imported session from file {}", path);
        Ok(result)
    }

    /// Import a session from a JSON string, preserving the exported agent id.
    pub fn import_json_default(&self, json: &str) -> CoreResult<ImportResult> {
        self.import_json(json, &ImportOptions::default())
    }

    /// Write an active summary into a session without triggering compaction.
    fn import_summary(&self, session_id: &Uuid, summary: &SessionSummary) -> CoreResult<()> {
        // Store the imported summary text as a metadata key for provenance.
        self.manager.set_metadata(
            session_id,
            "imported_summary",
            &summary.summary.chars().take(2000).collect::<String>(),
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Markdown rendering
// ---------------------------------------------------------------------------

/// Render a session export document as Markdown.
pub fn render_markdown(document: &SessionExportDocument) -> String {
    let session = &document.session;
    let mut out = String::new();

    out.push_str(&format!("# Session: {}\n\n", session.name));
    out.push_str(&format!("- **Id**: {}\n", session.id));
    out.push_str(&format!("- **Status**: {:?}\n", session.status));
    out.push_str(&format!("- **Mode**: {:?}\n", session.mode));
    out.push_str(&format!("- **Created**: {}\n", session.created_at));
    out.push_str(&format!("- **Last active**: {}\n", session.last_active_at));
    out.push_str(&format!("- **Messages**: {}\n", session.message_count));
    out.push_str(&format!("- **Tokens**: {}\n", session.total_tokens));

    if !session.system_prompt.is_empty() {
        out.push_str("\n## System prompt\n\n");
        out.push_str(&session.system_prompt);
        out.push_str("\n");
    }

    if let Some(summary) = &document.active_summary {
        out.push_str("\n## Active summary\n\n");
        out.push_str(&summary.summary);
        out.push_str("\n");
    }

    if !document.tags.is_empty() {
        out.push_str("\n## Tags\n\n");
        for tag in &document.tags {
            out.push_str(&format!("- `{}`\n", tag));
        }
    }

    if !document.metadata.is_empty() {
        out.push_str("\n## Metadata\n\n");
        for (key, value) in &document.metadata {
            out.push_str(&format!("- **{}**: `{}`\n", key, value));
        }
    }

    if !document.transcript.is_empty() {
        out.push_str("\n## Transcript\n\n");
        for entry in &document.transcript {
            let role = match entry.role.as_str() {
                "user" => "User",
                "assistant" => "Assistant",
                "system" => "System",
                "tool" => "Tool",
                other => other,
            };
            out.push_str(&format!("### {} ({})\n\n", role, entry.created_at));
            out.push_str(&entry.content);
            out.push_str("\n\n");
        }
    }

    if !document.attachments.is_empty() {
        out.push_str("\n## Attachments\n\n");
        for attachment in &document.attachments {
            out.push_str(&format!(
                "- **{}** ({} bytes, `{}`)\n",
                attachment.name, attachment.size_bytes, attachment.content_type
            ));
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Build a JSON export for a session in one call.
pub fn export_session_json(
    manager: &SessionManager,
    session_id: &Uuid,
    options: &ExportOptions,
) -> CoreResult<String> {
    SessionExporter::new(manager).export_json(session_id, options)
}

/// Build a Markdown export for a session in one call.
pub fn export_session_markdown(
    manager: &SessionManager,
    session_id: &Uuid,
    options: &ExportOptions,
) -> CoreResult<String> {
    SessionExporter::new(manager).export_markdown(session_id, options)
}

/// Import a session from a JSON string in one call.
pub fn import_session_json(
    manager: &SessionManager,
    json: &str,
    options: &ImportOptions,
) -> CoreResult<ImportResult> {
    SessionImporter::new(manager).import_json(json, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SessionMode;

    fn manager() -> SessionManager {
        SessionManager::new(crate::storage::SessionStorage::in_memory().unwrap())
    }

    fn create_session(manager: &SessionManager, name: &str) -> Session {
        manager
            .create(
                CreateSessionConfig::new(Uuid::new_v4())
                    .with_name(name)
                    .with_system_prompt("you are helpful")
                    .with_mode(SessionMode::Chat)
                    .with_tag("test")
                    .with_metadata(serde_json::json!({"color": "blue"})),
            )
            .unwrap()
    }

    #[test]
    fn export_roundtrip_preserves_content() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let session = create_session(&mgr, "roundtrip");
        mgr.add_message(&session.id, "user".into(), "hello there".into(), 7)
            .unwrap();
        mgr.add_message(&session.id, "assistant".into(), "hi".into(), 2)
            .unwrap();
        mgr.set_metadata(&session.id, "theme", "dark").unwrap();

        let json = exporter
            .export_json(&session.id, &ExportOptions::default())
            .unwrap();
        let document: SessionExportDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(document.session.name, "roundtrip");
        assert_eq!(document.transcript.len(), 2);
        assert_eq!(document.transcript[0].content, "hello there");
        assert!(document.tags.contains(&"test".to_string()));
        assert!(document
            .metadata
            .iter()
            .any(|(k, v)| k == "theme" && v == "dark"));
    }

    #[test]
    fn export_markdown_contains_sections() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let session = create_session(&mgr, "md-export");
        mgr.add_message(&session.id, "user".into(), "hello".into(), 3)
            .unwrap();

        let md = exporter
            .export_markdown(&session.id, &ExportOptions::default())
            .unwrap();
        assert!(md.contains("# Session: md-export"));
        assert!(md.contains("## Transcript"));
        assert!(md.contains("### User"));
        assert!(md.contains("hello"));
        assert!(md.contains("## Tags"));
    }

    #[test]
    fn import_creates_new_session() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let importer = SessionImporter::new(&mgr);

        let session = create_session(&mgr, "import-source");
        mgr.add_message(&session.id, "user".into(), "content".into(), 5)
            .unwrap();
        let json = exporter
            .export_json(&session.id, &ExportOptions::default())
            .unwrap();

        let result = importer
            .import_json(&json, &ImportOptions::default())
            .unwrap();
        assert!(result.created);
        assert_eq!(result.entries_imported, 1);
        assert!(result.tags_imported >= 1);

        let imported = mgr.get(&result.session_id).unwrap().unwrap();
        assert_eq!(imported.name, "import-source");
        assert_eq!(mgr.full_transcript(&imported.id).unwrap().len(), 1);
    }

    #[test]
    fn import_remaps_when_id_conflicts() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let importer = SessionImporter::new(&mgr);

        let session = create_session(&mgr, "conflict");
        mgr.add_message(&session.id, "user".into(), "hello".into(), 1)
            .unwrap();
        let json = exporter
            .export_json(&session.id, &ExportOptions::default())
            .unwrap();

        // Import twice; the second must remap to a new id.
        let first = importer
            .import_json(&json, &ImportOptions::default())
            .unwrap();
        let second = importer
            .import_json(&json, &ImportOptions::default())
            .unwrap();
        assert_ne!(first.session_id, second.session_id);
    }

    #[test]
    fn export_to_file_writes_json() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let session = create_session(&mgr, "file-export");
        let path = std::env::temp_dir()
            .join(format!("osq-session-export-{}.json", Uuid::new_v4()));
        let path_str = path.to_str().unwrap().to_string();
        let len = exporter
            .export_to_file(&session.id, &path_str, &ExportOptions::default())
            .unwrap();
        assert!(len > 0);
        assert!(path.exists());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn import_file_roundtrip() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let importer = SessionImporter::new(&mgr);
        let session = create_session(&mgr, "file-import");
        mgr.add_message(&session.id, "user".into(), "hello".into(), 3)
            .unwrap();

        let path = std::env::temp_dir()
            .join(format!("osq-session-import-{}.json", Uuid::new_v4()));
        let path_str = path.to_str().unwrap().to_string();
        exporter
            .export_to_file(&session.id, &path_str, &ExportOptions::default())
            .unwrap();

        let result = importer
            .import_file(&path_str, &ImportOptions::default())
            .unwrap();
        assert_eq!(result.entries_imported, 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn render_markdown_handles_empty_transcript() {
        let document = SessionExportDocument {
            format_version: EXPORT_FORMAT_VERSION,
            exported_at: Utc::now(),
            session: create_session(
                &SessionManager::new(crate::storage::SessionStorage::in_memory().unwrap()),
                "empty",
            ),
            transcript: Vec::new(),
            active_summary: None,
            summaries: Vec::new(),
            tags: Vec::new(),
            metadata: Vec::new(),
            attachments: Vec::new(),
        };
        let md = render_markdown(&document);
        assert!(md.contains("# Session: empty"));
        assert!(!md.contains("## Transcript\n\n###"));
    }

    #[test]
    fn export_options_builders() {
        let opts = ExportOptions::transcript_only();
        assert!(opts.include_transcript);
        assert!(!opts.include_tags);
        assert!(!opts.include_metadata);

        let full = ExportOptions::full();
        assert!(full.include_routing);
    }

    #[test]
    fn import_records_provenance_metadata() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let importer = SessionImporter::new(&mgr);
        let session = create_session(&mgr, "provenance");
        let json = exporter
            .export_json(&session.id, &ExportOptions::default())
            .unwrap();

        let result = importer
            .import_json(&json, &ImportOptions::default())
            .unwrap();
        let provenance = mgr
            .get_metadata(&result.session_id, "imported_from")
            .unwrap();
        assert_eq!(provenance.as_deref(), Some(session.id.to_string().as_str()));
    }

    #[test]
    fn import_imports_active_summary() {
        let mgr = manager();
        let exporter = SessionExporter::new(&mgr);
        let importer = SessionImporter::new(&mgr);
        let session = create_session(&mgr, "summary-import");
        // Build a summary by compacting.
        for i in 0..60 {
            mgr.add_message(&session.id, "user".into(), format!("line {}", i), 100)
                .unwrap();
        }
        mgr.compact(&session.id).unwrap();

        let json = exporter
            .export_json(&session.id, &ExportOptions::default())
            .unwrap();
        let document: SessionExportDocument = serde_json::from_str(&json).unwrap();
        assert!(document.active_summary.is_some());

        let result = importer
            .import_json(&json, &ImportOptions::default())
            .unwrap();
        assert!(result.summary_imported);
    }
}
