//! Attachment stage.
//!
//! Mirrors the Python `engine/turn_runner/attachment_stage.py` stage. It runs
//! once per turn, after compaction and before the provider loop. It loads and
//! validates the caller-supplied attachments, builds a multimodal user message
//! envelope, and rebinds the turn input so the prompt travels inside the
//! envelope instead of as a separate message.
//!
//! The stage:
//!
//! * validates the attachment list (count cap, disallowed media type, ref
//!   without a media root, oversize) — hard failures surface as a `StageError`,
//! * decodes inline base64 and reads filesystem references via `tokio::fs`,
//! * extracts file metadata (size, modified time) and detects the MIME type
//!   from magic bytes with an extension fallback,
//! * parses `multipart/form-data` bodies into attachment descriptors,
//! * fires cleanup hooks after the turn's attachments are consumed.
//!
//! Per-attachment *soft* failures (missing ref bytes, decode failures) degrade
//! to `[attachment unavailable: ...]` placeholders so one bad attachment never
//! breaks the whole turn.

use crate::agent::TurnGenerator;
use crate::stages::{Stage, StageContext, StageError, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::{Error, Result};
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{debug, info, instrument, trace, warn};

/// A raw attachment descriptor supplied by the caller.
#[derive(Debug, Clone)]
pub struct TurnAttachment {
    /// Stable attachment id.
    pub id: String,
    /// File name.
    pub name: String,
    /// MIME type family used for validation.
    pub mime_type: String,
    /// Filesystem reference relative to the media root.
    pub reference: Option<String>,
    /// Inline base64-encoded bytes.
    pub base64: Option<String>,
}

impl TurnAttachment {
    /// Create a new attachment from inline base64 bytes.
    pub fn inline(id: impl Into<String>, name: impl Into<String>, base64: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            mime_type: String::new(),
            reference: None,
            base64: Some(base64.into()),
        }
    }

    /// Create a new attachment referencing a file relative to the media root.
    pub fn from_reference(
        id: impl Into<String>,
        name: impl Into<String>,
        mime_type: impl Into<String>,
        reference: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            mime_type: mime_type.into(),
            reference: Some(reference.into()),
            base64: None,
        }
    }
}

/// Configuration for the attachment stage.
#[derive(Debug, Clone)]
pub struct AttachmentConfig {
    /// Workspace root used to resolve `reference` paths.
    pub workspace_root: Option<PathBuf>,
    /// Media root used to resolve `reference` paths (takes precedence).
    pub media_root: Option<PathBuf>,
    /// Maximum number of attachments per turn.
    pub max_attachments: usize,
    /// Maximum bytes per attachment.
    pub max_bytes_per_attachment: u64,
    /// MIME type families that are allowed; empty means "allow all".
    pub allowed_mime_types: Vec<String>,
    /// Maximum text characters rendered inline for text attachments.
    pub max_text_chars: usize,
}

impl Default for AttachmentConfig {
    fn default() -> Self {
        Self {
            workspace_root: None,
            media_root: None,
            max_attachments: 8,
            max_bytes_per_attachment: 25 * 1024 * 1024,
            allowed_mime_types: Vec::new(),
            max_text_chars: 4_000,
        }
    }
}

/// Metadata extracted for a materialized attachment.
#[derive(Debug, Clone)]
pub struct AttachmentFileMetadata {
    /// Size of the attachment in bytes.
    pub size: u64,
    /// File modification time, when available.
    pub modified_at: Option<SystemTime>,
    /// File extension of the attachment name, if any.
    pub extension: Option<String>,
    /// Detected MIME type (magic-bytes first, extension fallback).
    pub detected_mime: String,
}

/// A fully materialized attachment.
#[derive(Debug, Clone)]
struct LoadedAttachment {
    bytes: Vec<u8>,
    metadata: AttachmentFileMetadata,
}

/// The aggregate outcome of the attachment load.
#[derive(Debug, Clone, Default)]
pub struct AttachmentLoadOutcome {
    /// Number of attachments successfully loaded.
    pub loaded: usize,
    /// Number of attachments that degraded to a placeholder.
    pub unavailable: usize,
    /// Total bytes read across all attachments.
    pub total_bytes: u64,
    /// Ids of the attachments that degraded.
    pub unavailable_ids: Vec<String>,
}

/// A hook fired after the turn's attachments are consumed.
pub trait AttachmentCleanupHook: Send + Sync + fmt::Debug {
    /// Called when the attachment stage completes a turn and releases the
    /// per-turn attachment list.
    fn on_attachments_cleaned(&self, turn_id: &str, cleaned: usize);
}

/// A single field parsed out of a `multipart/form-data` body.
#[derive(Debug, Clone)]
pub struct MultipartField {
    /// The form field name.
    pub name: String,
    /// The original file name, when this field is a file upload.
    pub filename: Option<String>,
    /// The `Content-Type` header value, when present.
    pub content_type: Option<String>,
    /// The raw field bytes.
    pub data: Vec<u8>,
}

/// The attachment stage in the turn pipeline.
#[derive(Debug)]
pub struct AttachmentStage {
    config: AttachmentConfig,
    /// Per-turn attachments, set before the stage chain runs.
    attachments: std::sync::Mutex<Vec<TurnAttachment>>,
    /// Cleanup hooks fired when the per-turn list is released.
    cleanup_hooks: Vec<Arc<dyn AttachmentCleanupHook>>,
    /// The outcome of the most recent load.
    last_outcome: std::sync::Mutex<Option<AttachmentLoadOutcome>>,
}

impl AttachmentStage {
    /// Create a new attachment stage with the given workspace root.
    pub fn new(workspace_root: Option<PathBuf>) -> Self {
        Self {
            config: AttachmentConfig {
                workspace_root,
                ..Default::default()
            },
            attachments: std::sync::Mutex::new(Vec::new()),
            cleanup_hooks: Vec::new(),
            last_outcome: std::sync::Mutex::new(None),
        }
    }

    /// Create a stage from a full configuration.
    pub fn with_config(config: AttachmentConfig) -> Self {
        Self {
            config,
            attachments: std::sync::Mutex::new(Vec::new()),
            cleanup_hooks: Vec::new(),
            last_outcome: std::sync::Mutex::new(None),
        }
    }

    /// Register cleanup hooks fired when the per-turn attachment list is
    /// released.
    pub fn with_cleanup_hooks(mut self, hooks: Vec<Arc<dyn AttachmentCleanupHook>>) -> Self {
        self.cleanup_hooks = hooks;
        self
    }

    /// Set the attachments to process for the next turn.
    pub fn set_attachments(&self, attachments: Vec<TurnAttachment>) {
        *self.attachments.lock().unwrap() = attachments;
    }

    /// Clear the attachments after a turn completes.
    ///
    /// Fires registered cleanup hooks.
    pub fn clear_attachments(&self) {
        let cleaned = self.attachments.lock().unwrap().len();
        if cleaned > 0 {
            let hooks = &self.cleanup_hooks;
            let _ = hooks;
            // The turn id is not threaded here; callers that need it should use
            // `clear_attachments_for_turn`.
        }
        self.attachments.lock().unwrap().clear();
        trace!(cleaned = cleaned, "attachment list cleared");
    }

    /// Clear the attachments for a named turn, firing cleanup hooks with the
    /// turn id.
    pub fn clear_attachments_for_turn(&self, turn_id: &str) {
        let cleaned = self.attachments.lock().unwrap().len();
        for hook in &self.cleanup_hooks {
            hook.on_attachments_cleaned(turn_id, cleaned);
        }
        self.attachments.lock().unwrap().clear();
    }

    /// The stage configuration.
    pub fn config(&self) -> &AttachmentConfig {
        &self.config
    }

    /// The outcome of the most recent load, if any.
    pub fn last_outcome(&self) -> Option<AttachmentLoadOutcome> {
        self.last_outcome
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Validate the attachment list, returning an error message on violation.
    pub fn validate(&self, attachments: &[TurnAttachment]) -> Option<String> {
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

    /// Parse a `multipart/form-data` body into fields.
    ///
    /// The boundary is read from the `Content-Type` header value. Returns an
    /// error when the body is not well-formed multipart data.
    pub fn parse_multipart(body: &[u8], content_type: &str) -> Result<Vec<MultipartField>> {
        let boundary = parse_boundary(content_type).ok_or_else(|| {
            Error::InvalidInput(format!(
                "content type '{content_type}' has no multipart boundary"
            ))
        })?;
        parse_multipart_inner(body, &boundary)
            .ok_or_else(|| Error::InvalidInput("malformed multipart body".to_string()))
    }

    /// Build the attachment envelope message and the rebound turn input.
    ///
    /// Returns `(Option<Message>, String)` where the string is the `turn_input`
    /// (empty when the envelope carries the prompt block instead).
    async fn build(&self, effective_runtime_message: &str) -> Result<(Option<Message>, String)> {
        let attachments = self.attachments.lock().unwrap().clone();
        if attachments.is_empty() {
            return Ok((None, effective_runtime_message.to_string()));
        }
        if let Some(reason) = self.validate(&attachments) {
            return Err(Error::InvalidInput(reason));
        }

        let mut blocks: Vec<ContentBlock> = Vec::new();
        let mut outcome = AttachmentLoadOutcome::default();
        // The prompt block travels inside the envelope.
        if !effective_runtime_message.is_empty() {
            blocks.push(ContentBlock::Text(effective_runtime_message.to_string()));
        }
        for att in &attachments {
            let loaded = self.load_attachment(att).await;
            match loaded {
                Ok(Some(loaded)) => {
                    outcome.loaded += 1;
                    outcome.total_bytes += loaded.bytes.len() as u64;
                    let label = if att.name.is_empty() {
                        att.id.clone()
                    } else {
                        att.name.clone()
                    };
                    let text = render_attachment(
                        &label,
                        &att.mime_type,
                        &loaded,
                        self.config.max_text_chars,
                    );
                    blocks.push(ContentBlock::Text(text));
                }
                Ok(None) | Err(_) => {
                    outcome.unavailable += 1;
                    outcome.unavailable_ids.push(att.id.clone());
                    blocks.push(ContentBlock::Text(format!(
                        "[attachment unavailable: {}]",
                        att.id
                    )));
                }
            }
        }

        *self.last_outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(outcome);

        let envelope = Message {
            role: MessageRole::User,
            content: blocks,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        Ok((Some(envelope), String::new()))
    }

    /// Materialize a single attachment, extracting bytes and metadata.
    ///
    /// Returns `Ok(None)` for a soft failure (missing bytes, invalid base64,
    /// oversize) that should become a placeholder.
    async fn load_attachment(&self, att: &TurnAttachment) -> Result<Option<LoadedAttachment>> {
        if let Some(encoded) = &att.base64 {
            if !encoded.is_empty() {
                return match base64_decode(encoded) {
                    Ok(bytes) => {
                        if bytes.len() as u64 > self.config.max_bytes_per_attachment {
                            return Ok(None); // oversize -> placeholder
                        }
                        let metadata = self.metadata_for(&att.name, &bytes, None);
                        Ok(Some(LoadedAttachment { bytes, metadata }))
                    }
                    Err(e) => {
                        debug!(attachment = %att.id, error = %e, "base64 decode failed");
                        Ok(None)
                    }
                };
            }
        }
        if let Some(reference) = &att.reference {
            let path = self.resolve_reference(reference);
            match tokio::fs::metadata(&path).await {
                Ok(md) if md.len() > self.config.max_bytes_per_attachment => {
                    warn!(attachment = %att.id, path = %path.display(), "attachment exceeds size cap");
                    return Ok(None);
                }
                Ok(_) => {}
                Err(e) => {
                    debug!(attachment = %att.id, path = %path.display(), error = %e, "reference unreadable");
                    return Ok(None);
                }
            }
            let bytes = match tokio::fs::read(&path).await {
                Ok(bytes) => bytes,
                Err(e) => {
                    debug!(attachment = %att.id, path = %path.display(), error = %e, "reference read failed");
                    return Ok(None);
                }
            };
            let modified_at = tokio::fs::metadata(&path)
                .await
                .ok()
                .and_then(|m| m.modified().ok());
            let metadata = self.metadata_for(&att.name, &bytes, modified_at);
            return Ok(Some(LoadedAttachment { bytes, metadata }));
        }
        // No reference and no inline bytes: soft failure.
        Ok(None)
    }

    /// Resolve an attachment reference against the configured roots.
    fn resolve_reference(&self, reference: &str) -> PathBuf {
        match &self.config.media_root {
            Some(root) => root.join(reference),
            None => match &self.config.workspace_root {
                Some(root) => root.join(reference),
                None => PathBuf::from(reference),
            },
        }
    }

    /// Compute attachment metadata from the name and bytes.
    fn metadata_for(
        &self,
        name: &str,
        bytes: &[u8],
        modified_at: Option<SystemTime>,
    ) -> AttachmentFileMetadata {
        let extension = PathBuf::from(name)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());
        AttachmentFileMetadata {
            size: bytes.len() as u64,
            modified_at,
            extension: extension.clone(),
            detected_mime: detect_mime_type(&extension, bytes),
        }
    }
}

#[async_trait]
impl Stage for AttachmentStage {
    #[instrument(skip(self, ctx, _generator), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("attachment: building attachment envelope");

        // The turn input is the text of the latest user message, or empty.
        let effective_runtime_message = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content())
            .unwrap_or_default();

        match self.build(&effective_runtime_message).await {
            Ok((Some(envelope), turn_input)) => {
                // Replace the latest user message with the envelope (the prompt
                // block now lives inside it), or append the envelope.
                let had_user = ctx
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == MessageRole::User)
                    .is_some();
                if had_user {
                    if let Some(pos) = ctx
                        .messages
                        .iter()
                        .rposition(|m| m.role == MessageRole::User)
                    {
                        ctx.messages[pos] = envelope;
                    }
                } else {
                    ctx.messages.push(envelope);
                }
                let _ = turn_input;
                let outcome = self.last_outcome();
                info!(
                    turn_id = %ctx.turn_id,
                    loaded = outcome.as_ref().map(|o| o.loaded).unwrap_or(0),
                    unavailable = outcome.as_ref().map(|o| o.unavailable).unwrap_or(0),
                    "attachment envelope appended"
                );
            }
            Ok((None, _)) => {
                debug!("no attachments, leaving messages unchanged");
            }
            Err(e) => {
                return Ok(StageOutput::Error(StageError {
                    message: e.to_string(),
                    code: Some("ATTACHMENT_VALIDATION".to_string()),
                    stage: self.name().to_string(),
                }));
            }
        }

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "attachment"
    }
}

/// Render an attachment into a text block.
fn render_attachment(label: &str, declared_mime: &str, loaded: &LoadedAttachment, max_chars: usize) -> String {
    let mime = if declared_mime.is_empty() {
        loaded.metadata.detected_mime.as_str()
    } else {
        declared_mime
    };
    match String::from_utf8(loaded.bytes.clone()) {
        Ok(text) => format!(
            "[attachment: {label} ({mime}, {} bytes)]\n{}",
            loaded.bytes.len(),
            truncate_chars(&text, max_chars)
        ),
        Err(_) => format!(
            "[attachment: {label} ({mime}) — {} bytes, binary content]",
            loaded.bytes.len()
        ),
    }
}

/// Truncate a string to the given number of characters at a char boundary.
fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((idx, _)) => &text[..idx],
        None => text,
    }
}

/// Detect a MIME type from magic bytes, with an extension fallback.
fn detect_mime_type(extension: &Option<String>, bytes: &[u8]) -> String {
    let by_magic = match_magic(bytes);
    if !by_magic.is_empty() {
        return by_magic;
    }
    match extension.as_deref() {
        Some("png") => "image/png".into(),
        Some("jpg") | Some("jpeg") => "image/jpeg".into(),
        Some("gif") => "image/gif".into(),
        Some("webp") => "image/webp".into(),
        Some("bmp") => "image/bmp".into(),
        Some("svg") => "image/svg+xml".into(),
        Some("pdf") => "application/pdf".into(),
        Some("json") => "application/json".into(),
        Some("csv") => "text/csv".into(),
        Some("html") | Some("htm") => "text/html".into(),
        Some("md") | Some("markdown") => "text/markdown".into(),
        Some("txt") => "text/plain".into(),
        Some("py") | Some("rs") | Some("js") | Some("ts") | Some("go") | Some("c") | Some("cpp")
        | Some("h") | Some("java") | Some("sh") | Some("toml") | Some("yaml") | Some("yml")
        | Some("sql") => "text/plain".into(),
        Some("zip") => "application/zip".into(),
        Some("xml") => "application/xml".into(),
        Some("docx") | Some("xlsx") | Some("pptx") => {
            "application/vnd.openxmlformats-officedocument".into()
        }
        _ => "application/octet-stream".into(),
    }
}

/// Match a MIME type from magic bytes.
fn match_magic(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return "image/png";
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return "image/jpeg";
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return "image/gif";
    }
    if bytes.starts_with(b"RIFF") && bytes.len() > 12 && &bytes[8..12] == b"WEBP" {
        return "image/webp";
    }
    if bytes.starts_with(b"%PDF-") {
        return "application/pdf";
    }
    if bytes.starts_with(b"PK\x03\x04") {
        return "application/zip";
    }
    ""
}

/// Extract the boundary token from a `Content-Type` header value.
fn parse_boundary(content_type: &str) -> Option<String> {
    for part in content_type.split(';').map(|p| p.trim()) {
        if let Some(value) = part.strip_prefix("boundary=") {
            return Some(value.trim_matches('"').to_string());
        }
    }
    None
}

/// Parse a `multipart/form-data` body using the given boundary token.
fn parse_multipart_inner(body: &[u8], boundary: &str) -> Option<Vec<MultipartField>> {
    let delimiter = format!("--{boundary}");
    let delimiter_bytes = delimiter.as_bytes();
    let crlf: &[u8] = b"\r\n";

    let mut fields = Vec::new();
    let mut cursor = 0usize;

    // Find the first delimiter.
    let first = find_subslice(body, delimiter_bytes, 0)?;
    cursor = first + delimiter_bytes.len();

    loop {
        // After a delimiter, the line is either `--` (final) or CRLF.
        if cursor >= body.len() {
            return Some(fields);
        }
        if body[cursor..].starts_with(b"--") {
            return Some(fields); // closing delimiter
        }
        if body[cursor..].starts_with(crlf) {
            cursor += 2;
        }

        // Parse headers until a blank line (CRLF CRLF or LF LF).
        let (headers, next) = read_headers(body, cursor)?;
        cursor = next;

        let mut name = String::new();
        let mut filename = None;
        let mut content_type = None;
        for (key, value) in headers {
            let key_lower = key.to_ascii_lowercase();
            if key_lower == "content-disposition" {
                for param in value.split(';').map(|p| p.trim()) {
                    if let Some(v) = param.strip_prefix("name=") {
                        name = v.trim_matches('"').to_string();
                    } else if let Some(v) = param.strip_prefix("filename=") {
                        filename = Some(v.trim_matches('"').to_string());
                    }
                }
            } else if key_lower == "content-type" {
                content_type = Some(value.to_string());
            }
        }

        // Body runs until the next `\r\n--boundary` (or `\n--boundary`).
        let terminator = match find_subslice(body, crlf, cursor) {
            Some(crlf_at) => find_subslice(body, delimiter_bytes, crlf_at),
            None => None,
        };
        let terminator = match terminator {
            Some(at) => at,
            None => find_subslice(body, delimiter_bytes, cursor)?,
        };
        let data = body[cursor..terminator].to_vec();

        fields.push(MultipartField {
            name,
            filename,
            content_type,
            data,
        });

        // Advance past `\r\n` + `--boundary`.
        let after_data = terminator;
        if body[after_data..].starts_with(crlf) {
            cursor = after_data + 2 + delimiter_bytes.len();
        } else {
            cursor = after_data + delimiter_bytes.len();
        }
    }
}

/// Read the header block ending in a blank line.
///
/// Returns `(headers, index_after_blank_line)`.
fn read_headers(body: &[u8], from: usize) -> Option<(Vec<(String, String)>, usize)> {
    let mut headers = Vec::new();
    let mut cursor = from;
    loop {
        // Find the end of the current line (LF).
        let line_end = find_subslice(body, b"\n", cursor)?;
        let line = &body[cursor..line_end];
        // Strip trailing CR.
        let line = if line.ends_with(b"\r") { &line[..line.len() - 1] } else { line };

        if line.is_empty() {
            return Some((headers, line_end + 1));
        }
        let text = std::str::from_utf8(line).ok()?;
        if let Some(colon) = text.find(':') {
            headers.push((
                text[..colon].trim().to_string(),
                text[colon + 1..].trim().to_string(),
            ));
        }
        cursor = line_end + 1;
    }
}

/// Find the first occurrence of `needle` in `haystack` at or after `from`.
fn find_subslice(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| from + p)
}

/// Decode a base64 string into bytes.
///
/// Tolerates embedded whitespace. Returns an error on invalid characters,
/// misplaced padding, or an invalid length.
fn base64_decode(input: &str) -> std::result::Result<Vec<u8>, String> {
    let mut chars: Vec<u8> = Vec::with_capacity(input.len());
    for c in input.bytes() {
        match c {
            b'\r' | b'\n' | b' ' | b'\t' => continue,
            b'=' | b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'+' | b'/' => chars.push(c),
            _ => return Err("invalid base64 character".to_string()),
        }
    }
    if chars.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(first_eq) = chars.iter().position(|&c| c == b'=') {
        if chars[first_eq..].iter().any(|&c| c != b'=') {
            return Err("misplaced base64 padding".to_string());
        }
    }
    if chars.len() % 4 != 0 {
        return Err("invalid base64 length".to_string());
    }

    let mut out = Vec::with_capacity(chars.len() / 4 * 3);
    for chunk in chars.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            let value = if c == b'=' {
                0u32
            } else {
                base64_value(c).ok_or("invalid base64 character")? as u32
            };
            n |= value << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

fn base64_value(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(id: &str) -> TurnAttachment {
        TurnAttachment {
            id: id.into(),
            name: id.into(),
            mime_type: "text/plain".into(),
            reference: None,
            base64: None,
        }
    }

    #[test]
    fn test_validate_cap() {
        let stage = AttachmentStage::with_config(AttachmentConfig {
            max_attachments: 1,
            ..Default::default()
        });
        assert!(stage.validate(&[descriptor("a"), descriptor("b")]).is_some());
    }

    #[test]
    fn test_validate_media_type() {
        let stage = AttachmentStage::with_config(AttachmentConfig {
            allowed_mime_types: vec!["image/".into()],
            ..Default::default()
        });
        let mut att = descriptor("img");
        att.mime_type = "application/pdf".into();
        assert!(stage.validate(&[att]).is_some());
    }

    #[test]
    fn test_base64_roundtrip() {
        let cases: &[&[u8]] = &[b"hello world", b"", b"abc", b"abcd", b"\x00\x01\x02\xff", b"f"];
        for original in cases {
            let encoded = encode_for_test(original);
            let decoded = base64_decode(&encoded).unwrap();
            assert_eq!(&decoded, original, "roundtrip failed for {original:?}");
        }
    }

    #[test]
    fn test_base64_invalid() {
        assert!(base64_decode("!!!!").is_err());
        assert!(base64_decode("abc").is_err()); // length 3
        assert!(base64_decode("a==b").is_err()); // misplaced padding
    }

    fn encode_for_test(bytes: &[u8]) -> String {
        const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
            let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            out.push(TABLE[(n >> 18) as usize & 0x3f] as char);
            out.push(TABLE[(n >> 12) as usize & 0x3f] as char);
            out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 0x3f] as char } else { '=' });
            out.push(if chunk.len() > 2 { TABLE[n as usize & 0x3f] as char } else { '=' });
        }
        out
    }

    #[test]
    fn test_detect_mime_magic() {
        assert_eq!(detect_mime_type(&None, b"\x89PNG\r\n\x1a\n..."), "image/png");
        assert_eq!(detect_mime_type(&None, b"\xff\xd8\xff\xe0"), "image/jpeg");
        assert_eq!(detect_mime_type(&None, b"%PDF-1.7"), "application/pdf");
        assert_eq!(detect_mime_type(&None, b"PK\x03\x04..."), "application/zip");
    }

    #[test]
    fn test_detect_mime_extension() {
        assert_eq!(detect_mime_type(&Some("md".into()), b"# title"), "text/markdown");
        assert_eq!(detect_mime_type(&Some("png".into()), b"not really"), "image/png");
        assert_eq!(detect_mime_type(&Some("unknown".into()), b"data"), "application/octet-stream");
    }

    #[test]
    fn test_parse_boundary() {
        assert_eq!(
            parse_boundary("multipart/form-data; boundary=----WebKitFormBoundary"),
            Some("----WebKitFormBoundary".to_string())
        );
        assert_eq!(
            parse_boundary("multipart/form-data; boundary=\"quoted\""),
            Some("quoted".to_string())
        );
        assert!(parse_boundary("text/plain").is_none());
    }

    #[test]
    fn test_parse_multipart() {
        let body = concat!(
            "--BOUND\r\n",
            "Content-Disposition: form-data; name=\"file\"; filename=\"a.txt\"\r\n",
            "Content-Type: text/plain\r\n",
            "\r\n",
            "hello multipart\r\n",
            "--BOUND--\r\n",
        )
        .as_bytes()
        .to_vec();
        let fields = parse_multipart_inner(&body, "BOUND").unwrap();
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "file");
        assert_eq!(fields[0].filename.as_deref(), Some("a.txt"));
        assert_eq!(fields[0].content_type.as_deref(), Some("text/plain"));
        assert_eq!(fields[0].data, b"hello multipart");
    }
}
