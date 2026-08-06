//! File upload handling.
//!
//! Mirrors the Python `uploads.py` module. Provides low-level file upload
//! plumbing: multipart extraction, chunked streaming to disk, upload progress
//! tracking, upload cancellation, and a virus-scanning hook point.
//!
//! NOTE: The axum `multipart` feature must be enabled on the `axum`
//! dependency for the multipart extraction helper to compile.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Default maximum upload size (100 MiB).
pub const DEFAULT_MAX_UPLOAD_BYTES: u64 = 100 * 1024 * 1024;
/// Default chunk size for streaming uploads to disk.
pub const DEFAULT_CHUNK_BYTES: usize = 64 * 1024;

/// Status of an upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UploadStatus {
    /// Upload in progress.
    InProgress,
    /// Upload completed and file finalized.
    Completed,
    /// Upload was cancelled by the client.
    Cancelled,
    /// Upload failed.
    Failed,
}

impl UploadStatus {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            UploadStatus::InProgress => "in_progress",
            UploadStatus::Completed => "completed",
            UploadStatus::Cancelled => "cancelled",
            UploadStatus::Failed => "failed",
        }
    }
}

/// Progress of an in-flight upload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadProgress {
    pub upload_id: String,
    pub bytes_received: u64,
    pub total_bytes: Option<u64>,
    pub status: UploadStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl UploadProgress {
    /// Return the completion ratio, if the total is known.
    pub fn fraction(&self) -> Option<f64> {
        self.total_bytes
            .filter(|t| *t > 0)
            .map(|t| self.bytes_received as f64 / t as f64)
    }
}

/// A virus-scanning hook. Implementations should return `Err` if the content
/// is unsafe.
pub trait VirusScanner: Send + Sync {
    /// Scan a fully-received file. `path` points at the staged file on disk.
    fn scan(&self, path: &Path) -> Result<(), AppError>;
}

/// A no-op scanner used when scanning is not configured.
pub struct NoopScanner;

impl VirusScanner for NoopScanner {
    fn scan(&self, _path: &Path) -> Result<(), AppError> {
        Ok(())
    }
}

/// Tracks in-flight and completed uploads.
#[derive(Clone)]
pub struct UploadManager {
    dir: PathBuf,
    max_bytes: u64,
    progress: std::sync::Arc<RwLock<std::collections::HashMap<String, UploadProgress>>>,
    scanner: Arc<dyn VirusScanner>,
}

impl UploadManager {
    /// Create an upload manager writing to the given directory.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        Self::with_scanner(dir, Arc::new(NoopScanner))
    }

    /// Create an upload manager with a custom virus scanner.
    pub fn with_scanner(
        dir: impl Into<PathBuf>,
        scanner: Arc<dyn VirusScanner>,
    ) -> Result<Self, AppError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| AppError::internal(format!("Cannot create upload dir: {e}")))?;
        Ok(Self {
            dir,
            max_bytes: DEFAULT_MAX_UPLOAD_BYTES,
            progress: std::sync::Arc::new(RwLock::new(std::collections::HashMap::new())),
            scanner,
        })
    }

    /// Set a custom maximum upload size.
    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Begin a new upload. Returns the upload id and the staged file path.
    pub fn begin(&self) -> Result<(String, PathBuf), AppError> {
        let upload_id = Uuid::new_v4().to_string();
        let path = self.dir.join(format!("{upload_id}.part"));
        self.progress.write().insert(
            upload_id.clone(),
            UploadProgress {
                upload_id: upload_id.clone(),
                bytes_received: 0,
                total_bytes: None,
                status: UploadStatus::InProgress,
                started_at: Utc::now(),
                completed_at: None,
                error: None,
            },
        );
        Ok((upload_id, path))
    }

    /// Write a chunk to the staged file, updating progress. Returns the new
    /// byte count.
    pub fn write_chunk(
        &self,
        upload_id: &str,
        file: &mut std::fs::File,
        chunk: &[u8],
    ) -> Result<u64, AppError> {
        self.check_active(upload_id)?;
        let total = {
            let mut guard = self.progress.write();
            let progress = guard
                .get_mut(upload_id)
                .ok_or_else(|| AppError::not_found(format!("Unknown upload '{upload_id}'")))?;
            progress.bytes_received = progress
                .bytes_received
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| AppError::internal("Upload byte counter overflow"))?;
            progress.bytes_received
        };
        if total > self.max_bytes {
            let _ = self.fail(upload_id, "Upload exceeded maximum size");
            return Err(AppError::bad_request(format!(
                "Upload exceeds maximum size of {} bytes",
                self.max_bytes
            )));
        }
        file.write_all(chunk)
            .map_err(|e| AppError::internal(format!("Failed to write upload chunk: {e}")))?;
        Ok(total)
    }

    /// Complete an upload: finalize the file, run the virus scanner, and
    /// rename the staged file to its final name.
    pub fn complete(&self, upload_id: &str, final_name: &str) -> Result<PathBuf, AppError> {
        let staged = self.dir.join(format!("{upload_id}.part"));
        if !staged.exists() {
            return Err(AppError::not_found(format!(
                "Staged file for upload '{upload_id}' not found"
            )));
        }

        // Run the virus scanner on the staged file before finalizing.
        self.scanner.scan(&staged).map_err(|e| {
            let _ = self.fail(upload_id, "Virus scan rejected the upload");
            e
        })?;

        let safe = sanitize_filename(final_name);
        let final_path = self.dir.join(format!("{upload_id}-{safe}"));
        std::fs::rename(&staged, &final_path).map_err(|e| {
            let _ = self.fail(upload_id, "Failed to finalize upload");
            AppError::internal(format!("Failed to finalize upload: {e}"))
        })?;

        {
            let mut guard = self.progress.write();
            if let Some(p) = guard.get_mut(upload_id) {
                p.status = UploadStatus::Completed;
                p.completed_at = Some(Utc::now());
            }
        }
        info!(upload_id = %upload_id, path = %final_path.display(), "Upload completed");
        Ok(final_path)
    }

    /// Cancel an in-flight upload, removing the staged file.
    pub fn cancel(&self, upload_id: &str) -> Result<(), AppError> {
        let staged = self.dir.join(format!("{upload_id}.part"));
        if staged.exists() {
            std::fs::remove_file(&staged)
                .map_err(|e| AppError::internal(format!("Failed to remove staged file: {e}")))?;
        }
        {
            let mut guard = self.progress.write();
            if let Some(p) = guard.get_mut(upload_id) {
                p.status = UploadStatus::Cancelled;
                p.completed_at = Some(Utc::now());
            }
        }
        info!(upload_id = %upload_id, "Upload cancelled");
        Ok(())
    }

    /// Mark an upload as failed.
    pub fn fail(&self, upload_id: &str, error: &str) -> Result<(), AppError> {
        let staged = self.dir.join(format!("{upload_id}.part"));
        if staged.exists() {
            std::fs::remove_file(&staged).ok();
        }
        {
            let mut guard = self.progress.write();
            if let Some(p) = guard.get_mut(upload_id) {
                p.status = UploadStatus::Failed;
                p.completed_at = Some(Utc::now());
                p.error = Some(error.to_string());
            }
        }
        warn!(upload_id = %upload_id, error = %error, "Upload failed");
        Ok(())
    }

    /// Query the progress of an upload.
    pub fn progress(&self, upload_id: &str) -> Option<UploadProgress> {
        self.progress.read().get(upload_id).cloned()
    }

    /// List all tracked uploads.
    pub fn list(&self) -> Vec<UploadProgress> {
        let mut list: Vec<UploadProgress> = self.progress.read().values().cloned().collect();
        list.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        list
    }

    fn check_active(&self, upload_id: &str) -> Result<(), AppError> {
        let guard = self.progress.read();
        let progress = guard
            .get(upload_id)
            .ok_or_else(|| AppError::not_found(format!("Unknown upload '{upload_id}'")))?;
        if progress.status != UploadStatus::InProgress {
            return Err(AppError::bad_request(format!(
                "Upload '{upload_id}' is not in progress (status: {})",
                progress.status.as_str()
            )));
        }
        Ok(())
    }
}

/// A shared `std::fs::File` wrapper used for streaming multipart writes.
///
/// Because axum multipart fields are consumed sequentially, uploads are
/// usually written synchronously. This type exists so callers can share the
/// file handle if they choose to parallelize parts.
#[derive(Clone)]
pub struct UploadFileHandle {
    inner: Arc<Mutex<std::fs::File>>,
}

impl UploadFileHandle {
    /// Open a file handle for appending chunks.
    pub fn create(path: &Path) -> Result<Self, AppError> {
        let file = std::fs::File::create(path)
            .map_err(|e| AppError::internal(format!("Cannot create upload file: {e}")))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(file)),
        })
    }

    /// Append a chunk to the file.
    pub fn write_chunk(&self, chunk: &[u8]) -> Result<(), AppError> {
        let mut file = self.inner.lock();
        file.write_all(chunk)
            .map_err(|e| AppError::internal(format!("Cannot write chunk: {e}")))
    }

    /// Flush the file to disk.
    pub fn flush(&self) -> Result<(), AppError> {
        let mut file = self.inner.lock();
        file.flush()
            .map_err(|e| AppError::internal(format!("Cannot flush upload file: {e}")))
    }
}

/// axum multipart handler that streams an uploaded file to disk through an
/// [`UploadManager`].
///
/// Requires the `multipart` feature on `axum`.
pub async fn handle_upload(
    manager: UploadManager,
    mut multipart: axum::extract::Multipart,
) -> Result<serde_json::Value, AppError> {
    let mut result = Vec::new();
    while let Some(mut field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Multipart parse error: {e}")))?
    {
        let filename = field.file_name().unwrap_or("unnamed").to_string();
        let content_type = field
            .content_type()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "application/octet-stream".to_string());

        let (upload_id, staged) = manager.begin()?;
        let mut file_handle = UploadFileHandle::create(&staged)?;
        // axum 0.8 `Field::chunk()` is a Future returning
        // `Result<Option<Bytes>, MultipartError>`: `Ok(None)` marks the end of
        // the field. Await it directly for each chunk.
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|e| AppError::bad_request(format!("Chunk read error: {e}")))?
        {
            file_handle.write_chunk(&chunk)?;
        }
        file_handle.flush()?;
        drop(file_handle);

        let final_path = manager.complete(&upload_id, &filename)?;
        result.push(serde_json::json!({
            "upload_id": upload_id,
            "filename": filename,
            "content_type": content_type,
            "path": final_path.display().to_string(),
        }));
    }
    Ok(serde_json::json!({ "uploads": result, "count": result.len() }))
}

/// Sanitize a filename so it cannot traverse directories.
fn sanitize_filename(name: &str) -> String {
    let name = name.replace(['/', '\\'], "_");
    let name = name.trim_start_matches('.');
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
        let dir =
            std::env::temp_dir().join(format!("opensquilla-upload-test-{tag}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_begin_write_complete() {
        let dir = temp_dir("roundtrip");
        let manager = UploadManager::new(&dir).unwrap();
        let (id, staged) = manager.begin().unwrap();
        assert!(staged.ends_with(format!("{id}.part").as_str()));
        assert_eq!(
            manager.progress(&id).unwrap().status,
            UploadStatus::InProgress
        );

        let mut file = std::fs::File::create(&staged).unwrap();
        manager.write_chunk(&id, &mut file, b"hello").unwrap();
        manager.write_chunk(&id, &mut file, b" world").unwrap();
        assert_eq!(manager.progress(&id).unwrap().bytes_received, 11);
        drop(file);

        let final_path = manager.complete(&id, "notes.txt").unwrap();
        assert!(final_path.exists());
        let contents = std::fs::read_to_string(&final_path).unwrap();
        assert_eq!(contents, "hello world");
        assert_eq!(
            manager.progress(&id).unwrap().status,
            UploadStatus::Completed
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_cancel_removes_staged() {
        let dir = temp_dir("cancel");
        let manager = UploadManager::new(&dir).unwrap();
        let (id, staged) = manager.begin().unwrap();
        assert!(staged.exists());
        manager.cancel(&id).unwrap();
        assert!(!staged.exists());
        assert_eq!(
            manager.progress(&id).unwrap().status,
            UploadStatus::Cancelled
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_fail_marks_and_removes() {
        let dir = temp_dir("fail");
        let manager = UploadManager::new(&dir).unwrap();
        let (id, staged) = manager.begin().unwrap();
        assert!(staged.exists());
        manager.fail(&id, "boom").unwrap();
        assert!(!staged.exists());
        let progress = manager.progress(&id).unwrap();
        assert_eq!(progress.status, UploadStatus::Failed);
        assert_eq!(progress.error.as_deref(), Some("boom"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_max_size_enforced() {
        let dir = temp_dir("maxsize");
        let manager = UploadManager::new(&dir).unwrap().with_max_bytes(4);
        let (id, staged) = manager.begin().unwrap();
        let mut file = std::fs::File::create(&staged).unwrap();
        assert!(manager.write_chunk(&id, &mut file, b"12345").is_err());
        assert_eq!(manager.progress(&id).unwrap().status, UploadStatus::Failed);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_complete_runs_virus_scanner() {
        struct RejectScanner;
        impl VirusScanner for RejectScanner {
            fn scan(&self, _path: &Path) -> Result<(), AppError> {
                Err(AppError::forbidden("blocked by scanner"))
            }
        }
        let dir = temp_dir("scanner");
        let manager = UploadManager::with_scanner(&dir, Arc::new(RejectScanner)).unwrap();
        let (id, staged) = manager.begin().unwrap();
        let mut file = std::fs::File::create(&staged).unwrap();
        manager.write_chunk(&id, &mut file, b"data").unwrap();
        drop(file);
        let result = manager.complete(&id, "evil.txt");
        assert!(result.is_err());
        assert_eq!(manager.progress(&id).unwrap().status, UploadStatus::Failed);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_upload_file_handle_chunks() {
        let dir = temp_dir("handle");
        let path = dir.join("out.bin");
        let handle = UploadFileHandle::create(&path).unwrap();
        handle.write_chunk(b"ab").unwrap();
        handle.write_chunk(b"cd").unwrap();
        handle.flush().unwrap();
        drop(handle);
        assert_eq!(std::fs::read(&path).unwrap(), b"abcd");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_fraction() {
        let p = UploadProgress {
            upload_id: "u1".into(),
            bytes_received: 50,
            total_bytes: Some(100),
            status: UploadStatus::InProgress,
            started_at: Utc::now(),
            completed_at: None,
            error: None,
        };
        assert!((p.fraction().unwrap() - 0.5).abs() < 1e-9);
        let unknown = UploadProgress {
            total_bytes: None,
            ..p
        };
        assert!(unknown.fraction().is_none());
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("../x.txt"), "x.txt");
        assert_eq!(sanitize_filename("a/b\\c"), "a_b_c");
        assert_eq!(sanitize_filename(".."), "file");
    }
}
