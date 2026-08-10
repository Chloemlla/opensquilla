//! Attachment management.
//!
//! Mirrors the Python `attachments.py` module. Provides attachment metadata
//! storage, retrieval, size/type validation, and attachment-to-message
//! linking. Files themselves are stored on disk (see [`crate::uploads`]);
//! this module owns the metadata layer and the axum multipart upload handler
//! that accepts uploads and records them.
//!
//! NOTE: The axum `multipart` feature must be enabled on the `axum`
//! dependency for the upload handler to compile.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use uuid::Uuid;

/// Maximum allowed attachment size (default 25 MiB).
pub const DEFAULT_MAX_ATTACHMENT_BYTES: usize = 25 * 1024 * 1024;
/// MIME types that are always allowed.
pub const ALLOWED_MIME_PREFIXES: &[&str] = &[
    "image/",
    "text/",
    "application/pdf",
    "application/json",
    "application/x-json",
    "audio/",
    "video/",
];

/// The kind of attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentType {
    Image,
    Pdf,
    Code,
    Audio,
    Video,
    Text,
    Other,
}

impl AttachmentType {
    /// Infer an attachment type from a MIME content type.
    pub fn from_mime(mime: &str) -> Self {
        let mime = mime.to_ascii_lowercase();
        if mime.starts_with("image/") {
            AttachmentType::Image
        } else if mime == "application/pdf" {
            AttachmentType::Pdf
        } else if mime.starts_with("audio/") {
            AttachmentType::Audio
        } else if mime.starts_with("video/") {
            AttachmentType::Video
        } else if mime.starts_with("text/") || mime.contains("json") || mime.contains("code") {
            AttachmentType::Code
        } else {
            AttachmentType::Text
        }
    }

    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            AttachmentType::Image => "image",
            AttachmentType::Pdf => "pdf",
            AttachmentType::Code => "code",
            AttachmentType::Audio => "audio",
            AttachmentType::Video => "video",
            AttachmentType::Text => "text",
            AttachmentType::Other => "other",
        }
    }
}

/// Metadata for a stored attachment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentMeta {
    pub attachment_id: String,
    pub session_id: String,
    /// The message this attachment is linked to, if any.
    pub message_id: Option<String>,
    pub filename: String,
    pub content_type: String,
    pub attachment_type: AttachmentType,
    pub size_bytes: usize,
    /// Absolute or relative path to the stored file.
    pub storage_path: String,
    pub uploaded_at: DateTime<Utc>,
}

impl AttachmentMeta {
    /// Create metadata for a newly uploaded file.
    pub fn new(
        session_id: impl Into<String>,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        size_bytes: usize,
        storage_path: impl Into<String>,
    ) -> Self {
        let filename = filename.into();
        let content_type = content_type.into();
        Self {
            attachment_id: Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            message_id: None,
            filename: filename.clone(),
            content_type: content_type.clone(),
            attachment_type: AttachmentType::from_mime(&content_type),
            size_bytes,
            storage_path: storage_path.into(),
            uploaded_at: Utc::now(),
        }
    }
}

/// Validate an upload's size and MIME type.
pub fn validate_attachment(
    content_type: &str,
    size_bytes: usize,
    max_bytes: usize,
) -> Result<(), AppError> {
    if size_bytes == 0 {
        return Err(AppError::bad_request("Attachment must not be empty"));
    }
    if size_bytes > max_bytes {
        return Err(AppError::bad_request(format!(
            "Attachment size {size_bytes} exceeds maximum {max_bytes} bytes"
        )));
    }
    let allowed = ALLOWED_MIME_PREFIXES
        .iter()
        .any(|p| content_type.to_ascii_lowercase().starts_with(p));
    if !allowed {
        return Err(AppError::bad_request(format!(
            "Attachment content type '{content_type}' is not allowed"
        )));
    }
    Ok(())
}

/// In-memory metadata store for attachments.
///
/// Thread-safe and cheap to clone. The store does not move files; it only
/// records metadata and (via [`AttachmentStore::delete`]) removes the backing
/// file on disk.
#[derive(Clone, Default)]
pub struct AttachmentStore {
    attachments: std::sync::Arc<RwLock<std::collections::HashMap<String, AttachmentMeta>>>,
}

impl AttachmentStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a stored attachment's metadata.
    pub fn record(&self, meta: AttachmentMeta) -> String {
        let id = meta.attachment_id.clone();
        self.attachments.write().insert(id.clone(), meta);
        id
    }

    /// Get attachment metadata by id.
    pub fn get(&self, attachment_id: &str) -> Option<AttachmentMeta> {
        self.attachments.read().get(attachment_id).cloned()
    }

    /// List attachments for a session.
    pub fn list_for_session(&self, session_id: &str) -> Vec<AttachmentMeta> {
        let mut metas: Vec<AttachmentMeta> = self
            .attachments
            .read()
            .values()
            .filter(|m| m.session_id == session_id)
            .cloned()
            .collect();
        metas.sort_by_key(|b| std::cmp::Reverse(b.uploaded_at));
        metas
    }

    /// Link an attachment to a message.
    pub fn attach_to_message(&self, attachment_id: &str, message_id: &str) -> Result<(), AppError> {
        let mut guard = self.attachments.write();
        let meta = guard.get_mut(attachment_id).ok_or_else(|| {
            AppError::not_found(format!("Attachment '{attachment_id}' not found"))
        })?;
        meta.message_id = Some(message_id.to_string());
        Ok(())
    }

    /// Delete an attachment's metadata and backing file.
    pub fn delete(&self, attachment_id: &str) -> Result<bool, AppError> {
        let meta = self.attachments.write().remove(attachment_id);
        match meta {
            Some(meta) => {
                let path = Path::new(&meta.storage_path);
                if path.exists() {
                    std::fs::remove_file(path).map_err(|e| {
                        AppError::internal(format!("Failed to remove attachment file: {e}"))
                    })?;
                }
                info!(attachment_id = %attachment_id, "Attachment deleted");
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Remove all attachments for a session (and their files).
    pub fn delete_session(&self, session_id: &str) -> usize {
        let mut guard = self.attachments.write();
        let ids: Vec<String> = guard
            .iter()
            .filter(|(_, m)| m.session_id == session_id)
            .map(|(id, _)| id.clone())
            .collect();
        let mut removed = 0;
        for id in ids {
            if let Some(meta) = guard.remove(&id) {
                let path = Path::new(&meta.storage_path);
                if path.exists() {
                    std::fs::remove_file(path).ok();
                }
                removed += 1;
            }
        }
        removed
    }

    /// Return the total number of recorded attachments.
    pub fn len(&self) -> usize {
        self.attachments.read().len()
    }

    /// Return `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.attachments.read().is_empty()
    }
}

/// A fully-received attachment upload, ready to be persisted.
#[derive(Debug)]
pub struct AttachmentUpload {
    pub filename: String,
    pub content_type: String,
    pub bytes: Vec<u8>,
}

impl AttachmentUpload {
    /// Validate the upload against the store limits.
    pub fn validate(&self, max_bytes: usize) -> Result<(), AppError> {
        validate_attachment(&self.content_type, self.bytes.len(), max_bytes)
    }
}

/// Persist an upload to disk and record its metadata.
///
/// Returns the stored metadata. The file is written under `dir`, named with a
/// fresh UUID to avoid collisions and path-traversal via the original
/// filename.
pub fn store_attachment(
    store: &AttachmentStore,
    session_id: &str,
    upload: AttachmentUpload,
    dir: &Path,
) -> Result<AttachmentMeta, AppError> {
    upload.validate(DEFAULT_MAX_ATTACHMENT_BYTES)?;
    std::fs::create_dir_all(dir)
        .map_err(|e| AppError::internal(format!("Cannot create attachment dir: {e}")))?;

    let id = Uuid::new_v4().to_string();
    let safe_name = sanitize_filename(&upload.filename);
    let storage_path = dir.join(format!("{id}-{safe_name}"));
    std::fs::write(&storage_path, &upload.bytes)
        .map_err(|e| AppError::internal(format!("Failed to write attachment file: {e}")))?;

    let meta = AttachmentMeta::new(
        session_id,
        &upload.filename,
        &upload.content_type,
        upload.bytes.len(),
        storage_path.display().to_string(),
    );
    let recorded_id = store.record(meta.clone());
    debug!(
        attachment_id = %recorded_id,
        size = upload.bytes.len(),
        "Attachment stored"
    );
    Ok(meta)
}

/// axum multipart handler that accepts one or more file parts and stores
/// them as attachments.
///
/// Requires the `multipart` feature on `axum`.
pub async fn handle_multipart_upload(
    store: AttachmentStore,
    session_id: String,
    dir: PathBuf,
    mut multipart: axum::extract::Multipart,
) -> Result<Vec<AttachmentMeta>, AppError> {
    let mut metas = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Multipart parse error: {e}")))?
    {
        let filename = field.file_name().unwrap_or("unnamed").to_string();
        let content_type = field
            .content_type()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let data = field
            .bytes()
            .await
            .map_err(|e| AppError::bad_request(format!("Field read error: {e}")))?;
        let upload = AttachmentUpload {
            filename,
            content_type,
            bytes: data.to_vec(),
        };
        let meta = store_attachment(&store, &session_id, upload, &dir)?;
        metas.push(meta);
    }
    Ok(metas)
}

/// Sanitize a filename so it cannot traverse directories.
fn sanitize_filename(name: &str) -> String {
    let name = name.replace(['/', '\\'], "_");
    let name = name.trim_start_matches('.');
    let name = name.trim_start_matches('_');
    if name.is_empty() {
        "file".to_string()
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla-attachment-test-{tag}-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_attachment_type_from_mime() {
        assert_eq!(
            AttachmentType::from_mime("image/png"),
            AttachmentType::Image
        );
        assert_eq!(
            AttachmentType::from_mime("application/pdf"),
            AttachmentType::Pdf
        );
        assert_eq!(
            AttachmentType::from_mime("audio/mpeg"),
            AttachmentType::Audio
        );
        assert_eq!(
            AttachmentType::from_mime("video/mp4"),
            AttachmentType::Video
        );
        assert_eq!(
            AttachmentType::from_mime("text/markdown"),
            AttachmentType::Code
        );
        assert_eq!(AttachmentType::from_mime("x/other"), AttachmentType::Text);
    }

    #[test]
    fn test_validate_attachment_size_and_type() {
        assert!(validate_attachment("image/png", 100, 1024).is_ok());
        assert!(validate_attachment("image/png", 0, 1024).is_err());
        assert!(validate_attachment("image/png", 2048, 1024).is_err());
        assert!(validate_attachment("application/x-msdownload", 100, 1024).is_err());
    }

    #[test]
    fn test_store_attach_and_delete() {
        let dir = temp_dir("crud");
        let store = AttachmentStore::new();
        let upload = AttachmentUpload {
            filename: "a.png".into(),
            content_type: "image/png".into(),
            bytes: vec![1, 2, 3, 4],
        };
        let meta = store_attachment(&store, "s1", upload, &dir).unwrap();
        assert_eq!(meta.session_id, "s1");
        assert!(Path::new(&meta.storage_path).exists());

        store
            .attach_to_message(&meta.attachment_id, "msg-1")
            .unwrap();
        assert_eq!(
            store
                .get(&meta.attachment_id)
                .unwrap()
                .message_id
                .as_deref(),
            Some("msg-1")
        );
        assert_eq!(store.list_for_session("s1").len(), 1);

        assert!(store.delete(&meta.attachment_id).unwrap());
        assert!(!Path::new(&meta.storage_path).exists());
        assert!(store.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_session() {
        let dir = temp_dir("del-session");
        let store = AttachmentStore::new();
        for i in 0..3 {
            let upload = AttachmentUpload {
                filename: format!("f{i}.png"),
                content_type: "image/png".into(),
                bytes: vec![i as u8],
            };
            store_attachment(&store, "s1", upload, &dir).unwrap();
        }
        let upload = AttachmentUpload {
            filename: "other.png".into(),
            content_type: "image/png".into(),
            bytes: vec![9],
        };
        store_attachment(&store, "s2", upload, &dir).unwrap();
        assert_eq!(store.delete_session("s1"), 3);
        assert_eq!(store.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("../evil.png"), "evil.png");
        assert_eq!(sanitize_filename("a/b\\c.png"), "a_b_c.png");
        assert_eq!(sanitize_filename("...."), "file");
    }

    #[test]
    fn test_store_rejects_invalid_type() {
        let dir = temp_dir("reject");
        let store = AttachmentStore::new();
        let upload = AttachmentUpload {
            filename: "evil.exe".into(),
            content_type: "application/x-msdownload".into(),
            bytes: vec![0],
        };
        assert!(store_attachment(&store, "s1", upload, &dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_attach_missing_attachment() {
        let store = AttachmentStore::new();
        assert!(store.attach_to_message("nope", "m1").is_err());
    }
}
