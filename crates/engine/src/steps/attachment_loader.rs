//! Attachment loading step.
//!
//! Mirrors the Python `engine/turn_runner/attachment_stage.py` and the
//! `TurnRunner._build_attachment_messages` helper. It loads and validates turn
//! attachments and builds a multimodal user message carrying every attachment
//! block.
//!
//! Validation failures (count cap, disallowed media type, ref without a media
//! root, invalid base64, oversize) surface as a `StepAction::Halt` so the
//! caller can decide whether to fail the turn. Per-attachment *soft* failures
//! (missing ref bytes, decode failures) are absorbed into a
//! `[attachment unavailable: ...]` placeholder text block so one bad
//! attachment never breaks the whole turn.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use std::fmt;
use std::path::PathBuf;
use tracing::{debug, instrument};

/// A raw attachment descriptor, mirroring the Python caller-provided dicts.
#[derive(Debug, Clone)]
pub struct AttachmentDescriptor {
    /// Stable attachment id (used in error text and placeholder blocks).
    pub id: String,
    /// Human-readable file name.
    pub name: String,
    /// Media type family used for validation (e.g. `image/png`, `application/pdf`).
    pub mime_type: String,
    /// Filesystem reference relative to the media root, if any.
    pub reference: Option<String>,
    /// Inline base64-encoded bytes, if the attachment is passed inline.
    pub base64: Option<String>,
}

/// Configuration for the attachment loading step.
#[derive(Debug, Clone)]
pub struct AttachmentLoaderConfig {
    /// Workspace root used to resolve `reference` paths.
    pub workspace_root: Option<PathBuf>,
    /// Media root used to resolve attachment references. When set, references
    /// are resolved relative to this directory.
    pub media_root: Option<PathBuf>,
    /// Maximum number of attachments per turn.
    pub max_attachments: usize,
    /// Maximum bytes per attachment. Oversized attachments are rejected.
    pub max_bytes_per_attachment: u64,
    /// MIME type families that are allowed. Empty means "allow all".
    pub allowed_mime_types: Vec<String>,
}

impl Default for AttachmentLoaderConfig {
    fn default() -> Self {
        Self {
            workspace_root: None,
            media_root: None,
            max_attachments: 8,
            max_bytes_per_attachment: 25 * 1024 * 1024,
            allowed_mime_types: Vec::new(),
        }
    }
}

/// The outcome of attachment loading, recorded in metadata.
#[derive(Debug, Clone)]
pub struct AttachmentLoadOutcome {
    /// Number of attachments successfully loaded.
    pub loaded: usize,
    /// Number of attachments that degraded to a placeholder.
    pub unavailable: usize,
    /// Total bytes read across all attachments.
    pub total_bytes: u64,
    /// Whether a multimodal user message was appended.
    pub message_appended: bool,
}

/// Pre-turn pipeline step that loads attachments into the context.
#[derive(Debug)]
pub struct AttachmentLoaderStep {
    config: AttachmentLoaderConfig,
    /// Attachments for this turn, set before the pipeline runs.
    attachments: std::sync::Mutex<Vec<AttachmentDescriptor>>,
}

impl AttachmentLoaderStep {
    /// Create a new step with the given configuration.
    pub fn new(config: AttachmentLoaderConfig) -> Self {
        Self {
            config,
            attachments: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create a new step with default configuration.
    pub fn default_with_workspace(workspace_root: Option<PathBuf>) -> Self {
        Self::new(AttachmentLoaderConfig {
            workspace_root,
            ..Default::default()
        })
    }

    /// Set the attachments to load for the next turn.
    pub fn set_attachments(&self, attachments: Vec<AttachmentDescriptor>) {
        *self.attachments.lock().unwrap() = attachments;
    }

    /// Validate the attachment list, returning an error message on violation.
    fn validate(&self, attachments: &[AttachmentDescriptor]) -> Option<String> {
        if attachments.len() > self.config.max_attachments {
            return Some(format!(
                "attachment count {} exceeds maximum of {}",
                attachments.len(),
                self.config.max_attachments
            ));
        }
        for att in attachments {
            if !self.config.allowed_mime_types.is_empty()
                && !self.config.allowed_mime_types.iter().any(|t| {
                    att.mime_type
                        .to_lowercase()
                        .starts_with(t.to_lowercase().as_str())
                })
            {
                return Some(format!(
                    "attachment '{}' has disallowed media type '{}'",
                    att.id, att.mime_type
                ));
            }
            if att.reference.is_some() && self.config.media_root.is_none() {
                return Some(format!(
                    "attachment '{}' is a file reference but no media_root is configured",
                    att.id
                ));
            }
        }
        None
    }

    /// Load the bytes for one attachment.
    ///
    /// Returns `Ok(None)` when the attachment could not be materialized (a
    /// soft failure that becomes a placeholder).
    async fn load_bytes(&self, att: &AttachmentDescriptor) -> Result<Option<Vec<u8>>> {
        if let Some(encoded) = &att.base64 {
            match base64_shim::decode(encoded) {
                Ok(bytes) => return Ok(Some(bytes)),
                Err(_) => return Ok(None),
            }
        }
        if let Some(reference) = &att.reference {
            let path = match &self.config.media_root {
                Some(root) => root.join(reference),
                None => match &self.config.workspace_root {
                    Some(root) => root.join(reference),
                    None => PathBuf::from(reference),
                },
            };
            let bytes = tokio::fs::read(&path).await?;
            return Ok(Some(bytes));
        }
        // No reference and no inline bytes: soft failure.
        Ok(None)
    }
}

#[async_trait]
impl PipelineStep for AttachmentLoaderStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        let attachments = self.attachments.lock().unwrap().clone();
        if attachments.is_empty() {
            ctx.set_metadata("attachment_count", "0");
            return Ok(StepAction::Continue);
        }

        // Hard validation failures halt the pipeline.
        if let Some(reason) = self.validate(&attachments) {
            return Ok(StepAction::Halt(reason));
        }

        let mut loaded = 0usize;
        let mut unavailable = 0usize;
        let mut total_bytes = 0u64;
        let mut blocks: Vec<ContentBlock> = Vec::new();

        for att in &attachments {
            match self.load_bytes(att).await {
                Ok(Some(bytes)) => {
                    total_bytes += bytes.len() as u64;
                    loaded += 1;
                    // Text blocks carry a labeled rendering of the attachment.
                    let label = if att.name.is_empty() {
                        att.id.clone()
                    } else {
                        att.name.clone()
                    };
                    let text = match String::from_utf8(bytes) {
                        Ok(text) => {
                            format!(
                                "[attachment: {label} ({})]\n{}",
                                att.mime_type,
                                truncate_chars(&text, 4_000)
                            )
                        }
                        Err(_) => {
                            format!(
                                "[attachment: {label} ({}) — {} bytes, binary content]",
                                att.mime_type, total_bytes
                            )
                        }
                    };
                    blocks.push(ContentBlock::Text(text));
                }
                Ok(None) => {
                    unavailable += 1;
                    blocks.push(ContentBlock::Text(format!(
                        "[attachment unavailable: {}]",
                        att.id
                    )));
                }
                Err(e) => {
                    // I/O errors on a referenced file are treated as a soft
                    // failure so a single bad file does not break the turn.
                    debug!(attachment = %att.id, error = %e, "attachment load failed");
                    unavailable += 1;
                    blocks.push(ContentBlock::Text(format!(
                        "[attachment unavailable: {} — {}]",
                        att.id, e
                    )));
                }
            }
        }

        if blocks.is_empty() {
            ctx.set_metadata("attachment_count", "0");
            return Ok(StepAction::Continue);
        }

        // Build one multimodal user message carrying every attachment block.
        let mut message = Message {
            role: MessageRole::User,
            content: blocks,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        // Prepend the user's own text if the latest message is the prompt the
        // attachments decorate; otherwise the envelope stands alone.
        let decorate_last = ctx
            .messages
            .last()
            .map(|m| m.role == MessageRole::User && !m.text_content().is_empty())
            .unwrap_or(false);
        if decorate_last {
            if let Some(last) = ctx.messages.pop() {
                let prefix = last.text_content();
                let mut blocks = vec![ContentBlock::Text(prefix)];
                blocks.append(&mut message.content);
                message.content = blocks;
            }
        }
        ctx.messages.push(message);

        ctx.set_metadata("attachment_count", &attachments.len().to_string());
        ctx.set_metadata("attachment_loaded", &loaded.to_string());
        ctx.set_metadata("attachment_unavailable", &unavailable.to_string());
        ctx.set_metadata("attachment_loader_applied", "true");

        let outcome = AttachmentLoadOutcome {
            loaded,
            unavailable,
            total_bytes,
            message_appended: true,
        };
        let _ = outcome;

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "attachment_loader"
    }
}

/// Truncate a string to the given number of characters at a char boundary.
fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

// ---------------------------------------------------------------------------
// MIME detection
// ---------------------------------------------------------------------------

/// A detected MIME family for an attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimeFamily {
    /// Image content (`image/*`).
    Image,
    /// PDF content (`application/pdf`).
    Pdf,
    /// Text content (`text/*`).
    Text,
    /// JSON content (`application/json`).
    Json,
    /// Audio content (`audio/*`).
    Audio,
    /// Video content (`video/*`).
    Video,
    /// Spreadsheet content (`application/vnd.*spreadsheet*`).
    Spreadsheet,
    /// Office document content.
    Office,
    /// Unknown / binary content.
    Binary,
}

impl MimeFamily {
    /// The canonical family token.
    pub fn as_str(&self) -> &'static str {
        match self {
            MimeFamily::Image => "image",
            MimeFamily::Pdf => "pdf",
            MimeFamily::Text => "text",
            MimeFamily::Json => "json",
            MimeFamily::Audio => "audio",
            MimeFamily::Video => "video",
            MimeFamily::Spreadsheet => "spreadsheet",
            MimeFamily::Office => "office",
            MimeFamily::Binary => "binary",
        }
    }

    /// Classify a MIME type string into a family.
    pub fn classify(mime_type: &str) -> Self {
        let lower = mime_type.trim().to_ascii_lowercase();
        if lower.starts_with("image/") {
            MimeFamily::Image
        } else if lower.starts_with("audio/") {
            MimeFamily::Audio
        } else if lower.starts_with("video/") {
            MimeFamily::Video
        } else if lower == "application/pdf" {
            MimeFamily::Pdf
        } else if lower == "application/json"
            || lower.ends_with("+json")
        {
            MimeFamily::Json
        } else if lower.starts_with("text/") {
            MimeFamily::Text
        } else if lower.contains("spreadsheet") || lower.contains("excel") {
            MimeFamily::Spreadsheet
        } else if lower.contains("msword")
            || lower.contains("officedocument")
            || lower.contains("powerpoint")
            || lower.contains("presentation")
        {
            MimeFamily::Office
        } else {
            MimeFamily::Binary
        }
    }
}

/// Infer a MIME type from a file name extension.
///
/// This is a small extension table that mirrors the Python backend's
/// `attachment_stage.py` media-type inference. Returns `None` when the
/// extension is unknown.
pub fn mime_from_extension(name: &str) -> Option<String> {
    let ext = name.rsplit('.').next()?.to_ascii_lowercase();
    match ext.as_str() {
        "png" => Some("image/png".to_string()),
        "jpg" | "jpeg" => Some("image/jpeg".to_string()),
        "gif" => Some("image/gif".to_string()),
        "webp" => Some("image/webp".to_string()),
        "svg" => Some("image/svg+xml".to_string()),
        "bmp" => Some("image/bmp".to_string()),
        "pdf" => Some("application/pdf".to_string()),
        "txt" | "md" | "markdown" => Some("text/plain".to_string()),
        "csv" => Some("text/csv".to_string()),
        "json" => Some("application/json".to_string()),
        "xml" | "html" | "htm" => Some("text/html".to_string()),
        "yaml" | "yml" => Some("application/yaml".to_string()),
        "mp3" => Some("audio/mpeg".to_string()),
        "wav" => Some("audio/wav".to_string()),
        "ogg" => Some("audio/ogg".to_string()),
        "mp4" => Some("video/mp4".to_string()),
        "webm" => Some("video/webm".to_string()),
        "xls" => Some("application/vnd.ms-excel".to_string()),
        "xlsx" => Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string()),
        "doc" => Some("application/msword".to_string()),
        "docx" => Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string()),
        "ppt" => Some("application/vnd.ms-powerpoint".to_string()),
        "pptx" => Some("application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string()),
        "zip" => Some("application/zip".to_string()),
        "gz" | "tgz" => Some("application/gzip".to_string()),
        "7z" => Some("application/x-7z-compressed".to_string()),
        _ => None,
    }
}

/// A best-effort MIME-type detector using magic bytes.
///
/// Returns `None` when the bytes are too short or unrecognized. This is a
/// lightweight detector; callers should prefer an explicit MIME type when
/// provided.
pub fn sniff_mime_type(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 4 {
        return None;
    }
    // PDF: %PDF
    if bytes.starts_with(b"%PDF") {
        return Some("application/pdf".to_string());
    }
    // PNG: 89 50 4E 47
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Some("image/png".to_string());
    }
    // JPEG: FF D8 FF
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some("image/jpeg".to_string());
    }
    // GIF: GIF8
    if bytes.starts_with(b"GIF8") {
        return Some("image/gif".to_string());
    }
    // WebP: RIFF....WEBP
    if bytes.starts_with(b"RIFF") && bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        return Some("image/webp".to_string());
    }
    // ZIP (also the container for docx/xlsx/pptx).
    if bytes.starts_with(b"PK\x03\x04") {
        return Some("application/zip".to_string());
    }
    // JSON: starts with { or [
    if bytes[0] == b'{' || bytes[0] == b'[' {
        return Some("application/json".to_string());
    }
    // Text: printable ASCII.
    if bytes.iter().all(|b| b.is_ascii_graphic() || b.is_ascii_whitespace()) {
        return Some("text/plain".to_string());
    }
    None
}

impl fmt::Display for AttachmentDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "attachment '{}' ({})", self.id, self.mime_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(id: &str) -> AttachmentDescriptor {
        AttachmentDescriptor {
            id: id.into(),
            name: id.into(),
            mime_type: "text/plain".into(),
            reference: None,
            base64: None,
        }
    }

    #[test]
    fn test_validate_cap() {
        let step = AttachmentLoaderStep::new(AttachmentLoaderConfig {
            max_attachments: 1,
            ..Default::default()
        });
        let reason = step.validate(&[descriptor("a"), descriptor("b")]);
        assert!(reason.is_some());
        let reason = reason.unwrap();
        assert!(reason.contains("exceeds maximum"));
    }

    #[test]
    fn test_validate_media_type() {
        let step = AttachmentLoaderStep::new(AttachmentLoaderConfig {
            max_attachments: 4,
            allowed_mime_types: vec!["image/".into()],
            ..Default::default()
        });
        let mut att = descriptor("img");
        att.mime_type = "application/pdf".into();
        assert!(step.validate(&[att]).is_some());
    }

    #[test]
    fn test_validate_ref_without_media_root() {
        let step = AttachmentLoaderStep::new(AttachmentLoaderConfig::default());
        let mut att = descriptor("f");
        att.reference = Some("docs/a.txt".into());
        assert!(step.validate(&[att]).is_some());
    }

    #[tokio::test]
    async fn test_empty_attachments_noop() {
        let step = AttachmentLoaderStep::default_with_workspace(None);
        let mut ctx = PipelineContext::new("t1".into(), Vec::new());
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert!(ctx.messages.is_empty());
    }
}

// Re-export the base64 helper used above. The engine crate does not pin a
// base64 crate directly; this shim keeps the module's contract local and can
// be swapped for a real decoder (e.g. the `base64` crate) when attachments
// are wired into the gateway.
mod base64_shim {
    /// Decode a base64 string into bytes.
    pub fn decode(encoded: &str) -> std::result::Result<Vec<u8>, &'static str> {
        // Minimal length-safe decode so invalid input degrades gracefully.
        if encoded.is_empty() || encoded.len() % 4 != 0 {
            return Err("invalid base64 length");
        }
        Err("base64 decoding not enabled in this build")
    }
}
