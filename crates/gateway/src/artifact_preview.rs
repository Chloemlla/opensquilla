//! Artifact preview generation and caching.
//!
//! Mirrors the Python `artifact_preview.py` module. Generates lightweight
//! previews for generated files (images, PDFs, code), caches them, and hands
//! out time-limited access leases so that preview URLs cannot be shared
//! indefinitely.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Default lease duration for a preview URL.
pub const DEFAULT_LEASE_SECONDS: i64 = 60 * 30;
/// Default preview cache size (number of artifacts).
pub const DEFAULT_CACHE_LIMIT: usize = 256;

/// The kind of preview that can be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreviewKind {
    Image,
    Pdf,
    Code,
    Text,
    Unknown,
}

impl PreviewKind {
    /// Infer a preview kind from a file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "svg" | "bmp" => PreviewKind::Image,
            "pdf" => PreviewKind::Pdf,
            "rs" | "py" | "ts" | "js" | "tsx" | "jsx" | "go" | "c" | "cpp" | "h" | "hpp"
            | "toml" | "json" | "yaml" | "yml" | "sh" | "md" | "html" | "css" => {
                PreviewKind::Code
            }
            "txt" | "log" => PreviewKind::Text,
            _ => PreviewKind::Unknown,
        }
    }
}

/// A generated preview payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewData {
    pub artifact_id: String,
    pub kind: PreviewKind,
    /// MIME content type of the preview.
    pub content_type: String,
    /// Preview bytes (thumbnail, rendered PDF page, syntax-highlighted text).
    pub bytes: Vec<u8>,
    pub generated_at: DateTime<Utc>,
}

impl PreviewData {
    /// Size of the preview payload in bytes.
    pub fn size_bytes(&self) -> usize {
        self.bytes.len()
    }
}

/// A time-limited lease granting access to a cached preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewLease {
    pub lease_id: String,
    pub artifact_id: String,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

impl PreviewLease {
    /// Return `true` if the lease is still valid.
    pub fn is_valid(&self, now: DateTime<Utc>) -> bool {
        now < self.expires_at
    }
}

/// A callback that generates preview bytes for an artifact.
///
/// Returns a preview, or `None` if no preview can be generated for the
/// artifact (e.g. unsupported binary format).
pub trait PreviewGenerator: Send + Sync {
    /// Generate a preview for the given artifact file.
    fn generate(&self, artifact_id: &str, path: &Path) -> Option<PreviewData>;
}

/// Generates simple previews: images are returned as-is (downscaling is out
/// of scope), text/code files are returned as UTF-8 text, and PDFs are
/// represented by a placeholder text.
#[derive(Default)]
pub struct DefaultPreviewGenerator;

impl PreviewGenerator for DefaultPreviewGenerator {
    fn generate(&self, artifact_id: &str, path: &Path) -> Option<PreviewData> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();
        let kind = PreviewKind::from_extension(&ext);
        match kind {
            PreviewKind::Image => {
                let bytes = std::fs::read(path).ok()?;
                let content_type = match ext.as_str() {
                    "png" => "image/png",
                    "jpg" | "jpeg" => "image/jpeg",
                    "gif" => "image/gif",
                    "webp" => "image/webp",
                    "svg" => "image/svg+xml",
                    "bmp" => "image/bmp",
                    _ => "application/octet-stream",
                };
                Some(PreviewData {
                    artifact_id: artifact_id.to_string(),
                    kind,
                    content_type: content_type.to_string(),
                    bytes,
                    generated_at: Utc::now(),
                })
            }
            PreviewKind::Code | PreviewKind::Text => {
                let text = std::fs::read_to_string(path).ok()?;
                Some(PreviewData {
                    artifact_id: artifact_id.to_string(),
                    kind,
                    content_type: "text/plain; charset=utf-8".to_string(),
                    bytes: text.into_bytes(),
                    generated_at: Utc::now(),
                })
            }
            PreviewKind::Pdf => {
                // A real implementation would render the first PDF page. We
                // produce a small placeholder so the pipeline is exercised.
                let placeholder = format!("PDF preview for artifact {artifact_id} (rendering not enabled)");
                Some(PreviewData {
                    artifact_id: artifact_id.to_string(),
                    kind,
                    content_type: "text/plain; charset=utf-8".to_string(),
                    bytes: placeholder.into_bytes(),
                    generated_at: Utc::now(),
                })
            }
            PreviewKind::Unknown => None,
        }
    }
}

/// A cache entry pairing a preview with its lease.
#[derive(Debug, Clone)]
struct CacheEntry {
    preview: PreviewData,
    lease: Option<PreviewLease>,
    last_accessed: DateTime<Utc>,
}

/// Preview cache with time-limited access leases.
///
/// Previews are generated on first access, cached in memory (LRU-evicted at
/// the configured limit), and accessed through short-lived leases.
#[derive(Clone)]
pub struct PreviewCache {
    generator: Arc<dyn PreviewGenerator>,
    entries: std::sync::Arc<RwLock<HashMap<String, CacheEntry>>>,
    limit: usize,
}

impl PreviewCache {
    /// Create a cache with a custom generator.
    pub fn with_generator(generator: Arc<dyn PreviewGenerator>) -> Self {
        Self {
            generator,
            entries: std::sync::Arc::new(RwLock::new(HashMap::new())),
            limit: DEFAULT_CACHE_LIMIT,
        }
    }

    /// Create a cache with the default generator.
    pub fn new() -> Self {
        Self::with_generator(Arc::new(DefaultPreviewGenerator))
    }

    /// Set the cache size limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Generate and cache a preview for an artifact file.
    pub fn generate(
        &self,
        artifact_id: &str,
        path: &Path,
    ) -> Result<PreviewData, AppError> {
        {
            let guard = self.entries.read();
            if let Some(entry) = guard.get(artifact_id) {
                return Ok(entry.preview.clone());
            }
        }

        let preview = self
            .generator
            .generate(artifact_id, path)
            .ok_or_else(|| {
                AppError::bad_request(format!(
                    "No preview can be generated for artifact '{artifact_id}'"
                ))
            })?;

        self.evict_if_needed();
        {
            let mut guard = self.entries.write();
            guard.insert(
                artifact_id.to_string(),
                CacheEntry {
                    preview: preview.clone(),
                    lease: None,
                    last_accessed: Utc::now(),
                },
            );
        }
        debug!(artifact_id = %artifact_id, kind = ?preview.kind, "Preview cached");
        Ok(preview)
    }

    /// Get a cached preview without generating one.
    pub fn get(&self, artifact_id: &str) -> Option<PreviewData> {
        let mut guard = self.entries.write();
        let entry = guard.get_mut(artifact_id)?;
        entry.last_accessed = Utc::now();
        Some(entry.preview.clone())
    }

    /// Issue a time-limited access lease for a cached preview.
    pub fn issue_lease(
        &self,
        artifact_id: &str,
        ttl_seconds: i64,
    ) -> Result<PreviewLease, AppError> {
        let mut guard = self.entries.write();
        let entry = guard.get_mut(artifact_id).ok_or_else(|| {
            AppError::not_found(format!("No cached preview for artifact '{artifact_id}'"))
        })?;
        let now = Utc::now();
        let lease = PreviewLease {
            lease_id: Uuid::new_v4().to_string(),
            artifact_id: artifact_id.to_string(),
            issued_at: now,
            expires_at: now + Duration::seconds(ttl_seconds),
        };
        entry.lease = Some(lease.clone());
        Ok(lease)
    }

    /// Redeem a lease. Returns the preview if the lease is valid and matches.
    pub fn redeem(&self, lease: &PreviewLease) -> Result<PreviewData, AppError> {
        if !lease.is_valid(Utc::now()) {
            return Err(AppError::forbidden("Preview lease has expired"));
        }
        let guard = self.entries.read();
        let entry = guard.get(&lease.artifact_id).ok_or_else(|| {
            AppError::not_found(format!(
                "No cached preview for artifact '{}'",
                lease.artifact_id
            ))
        })?;
        match &entry.lease {
            Some(stored) if stored.lease_id == lease.lease_id => Ok(entry.preview.clone()),
            _ => Err(AppError::forbidden("Preview lease does not match")),
        }
    }

    /// Invalidate a cached preview.
    pub fn invalidate(&self, artifact_id: &str) -> bool {
        self.entries.write().remove(artifact_id).is_some()
    }

    /// Remove expired leases from the cache. Returns the number removed.
    pub fn purge_expired(&self) -> usize {
        let now = Utc::now();
        let mut guard = self.entries.write();
        let mut removed = 0;
        for (_, entry) in guard.iter_mut() {
            if let Some(lease) = &entry.lease {
                if !lease.is_valid(now) {
                    entry.lease = None;
                    removed += 1;
                }
            }
        }
        removed
    }

    /// Return the number of cached previews.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Evict least-recently-accessed entries when over the limit.
    fn evict_if_needed(&self) {
        let mut guard = self.entries.write();
        if guard.len() < self.limit {
            return;
        }
        let mut sorted: Vec<(String, DateTime<Utc>)> = guard
            .iter()
            .map(|(id, entry)| (id.clone(), entry.last_accessed))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));
        let to_evict = guard.len() - self.limit + 1;
        for (id, _) in sorted.into_iter().take(to_evict) {
            guard.remove(&id);
        }
        warn!(evicted = to_evict, "Preview cache evicted entries");
    }
}

impl Default for PreviewCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(tag: &str, contents: &[u8], ext: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("opensquilla-preview-test-{tag}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("artifact.{ext}"));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn test_preview_kind_from_extension() {
        assert_eq!(PreviewKind::from_extension("PNG"), PreviewKind::Image);
        assert_eq!(PreviewKind::from_extension("pdf"), PreviewKind::Pdf);
        assert_eq!(PreviewKind::from_extension("rs"), PreviewKind::Code);
        assert_eq!(PreviewKind::from_extension("txt"), PreviewKind::Text);
        assert_eq!(PreviewKind::from_extension("bin"), PreviewKind::Unknown);
    }

    #[test]
    fn test_generate_code_preview() {
        let path = temp_file("code", b"fn main() {}", "rs");
        let cache = PreviewCache::new();
        let preview = cache.generate("a1", &path).unwrap();
        assert_eq!(preview.kind, PreviewKind::Code);
        assert!(preview.size_bytes() > 0);
        assert!(cache.get("a1").is_some());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_generate_unknown_returns_error() {
        let path = temp_file("unknown", b"\x00\x01\x02", "bin");
        let cache = PreviewCache::new();
        assert!(cache.generate("a1", &path).is_err());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_lease_lifecycle() {
        let path = temp_file("lease", b"code", "py");
        let cache = PreviewCache::new();
        cache.generate("a1", &path).unwrap();
        let lease = cache.issue_lease("a1", 60).unwrap();
        assert!(lease.is_valid(Utc::now()));
        let preview = cache.redeem(&lease).unwrap();
        assert_eq!(preview.artifact_id, "a1");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_expired_lease_rejected() {
        let path = temp_file("expired", b"code", "py");
        let cache = PreviewCache::new();
        cache.generate("a1", &path).unwrap();
        // A negative TTL produces an already-expired lease.
        let lease = cache.issue_lease("a1", -10).unwrap();
        assert!(!lease.is_valid(Utc::now()));
        assert!(cache.redeem(&lease).is_err());
        assert_eq!(cache.purge_expired(), 1);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_lease_mismatch_rejected() {
        let path = temp_file("mismatch", b"code", "py");
        let cache = PreviewCache::new();
        cache.generate("a1", &path).unwrap();
        let lease = cache.issue_lease("a1", 60).unwrap();
        let forged = PreviewLease {
            lease_id: "forged".to_string(),
            ..lease.clone()
        };
        assert!(cache.redeem(&forged).is_err());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn test_invalidate_and_limit() {
        let cache = PreviewCache::new().with_limit(2);
        let p1 = temp_file("l1", b"a", "txt");
        let p2 = temp_file("l2", b"b", "txt");
        let p3 = temp_file("l3", b"c", "txt");
        cache.generate("a1", &p1).unwrap();
        cache.generate("a2", &p2).unwrap();
        assert_eq!(cache.len(), 2);
        cache.generate("a3", &p3).unwrap();
        // With limit 2, generating a third should evict one.
        assert!(cache.len() <= 2);
        assert!(cache.invalidate("a1") || cache.invalidate("a2") || cache.invalidate("a3"));
        std::fs::remove_dir_all(p1.parent().unwrap()).ok();
    }

    #[test]
    fn test_redeem_missing_artifact() {
        let lease = PreviewLease {
            lease_id: "x".into(),
            artifact_id: "missing".into(),
            issued_at: Utc::now(),
            expires_at: Utc::now() + Duration::seconds(60),
        };
        let cache = PreviewCache::new();
        assert!(cache.redeem(&lease).is_err());
    }
}
