//! Audio transcription.
//!
//! Mirrors the Python `audio_transcription.py` module. Receives audio uploads,
//! calls an external transcription API (Whisper or compatible), and returns
//! the transcribed text. Supports multiple audio formats via a configurable
//! set of allowed MIME types.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};
use uuid::Uuid;

/// Audio content types accepted by the transcription pipeline.
pub const SUPPORTED_AUDIO_MIME_TYPES: &[&str] = &[
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
    "audio/flac",
    "audio/aac",
    "audio/x-aac",
    "audio/amr",
    "audio/3gpp",
    "audio/x-opus+ogg",
];

/// The provider used for transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionProvider {
    /// OpenAI Whisper API (or compatible).
    Whisper,
    /// Local/offline engine (stub).
    Local,
}

/// The transcription API trait.
///
/// Implementations are expected to make an HTTP call to the external service
/// and return the transcribed text.
pub trait TranscriptionApi: Send + Sync {
    /// Transcribe audio bytes with the given MIME type.
    fn transcribe(&self, mime_type: &str, audio_bytes: &[u8]) -> Result<String, AppError>;
}

/// A stub API for tests and offline operation. Returns a deterministic
/// placeholder rather than calling a real service.
pub struct StubTranscriptionApi;

impl TranscriptionApi for StubTranscriptionApi {
    fn transcribe(&self, _mime_type: &str, audio_bytes: &[u8]) -> Result<String, AppError> {
        Ok(format!(
            "[transcribed {} bytes of audio]",
            audio_bytes.len()
        ))
    }
}

/// A record of a transcription request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRecord {
    pub id: String,
    pub session_id: String,
    pub mime_type: String,
    pub audio_size_bytes: usize,
    pub text: String,
    pub provider: TranscriptionProvider,
    pub duration_ms: u64,
    pub created_at: DateTime<Utc>,
}

/// The transcription service.
///
/// Validates the audio MIME type and size, calls the configured
/// [`TranscriptionApi`], and records the result. Clone is cheap.
#[derive(Clone)]
pub struct TranscriptionService {
    api: Arc<dyn TranscriptionApi>,
    provider: TranscriptionProvider,
    max_audio_bytes: usize,
    history: std::sync::Arc<RwLock<HashMap<String, TranscriptionRecord>>>,
}

impl TranscriptionService {
    /// Create a service backed by the given API.
    pub fn with_api(api: Arc<dyn TranscriptionApi>, provider: TranscriptionProvider) -> Self {
        Self {
            api,
            provider,
            max_audio_bytes: DEFAULT_MAX_AUDIO_BYTES,
            history: std::sync::Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a service using the stub API.
    pub fn new() -> Self {
        Self::with_api(Arc::new(StubTranscriptionApi), TranscriptionProvider::Local)
    }

    /// Set a custom maximum audio payload size.
    pub fn with_max_audio_bytes(mut self, bytes: usize) -> Self {
        self.max_audio_bytes = bytes;
        self
    }

    /// Validate an audio upload's MIME type and size.
    pub fn validate_audio(&self, mime_type: &str, size_bytes: usize) -> Result<(), AppError> {
        let supported = SUPPORTED_AUDIO_MIME_TYPES
            .iter()
            .any(|m| m.eq_ignore_ascii_case(mime_type));
        if !supported {
            return Err(AppError::bad_request(format!(
                "Unsupported audio MIME type '{mime_type}'"
            )));
        }
        if size_bytes == 0 {
            return Err(AppError::bad_request("Audio payload must not be empty"));
        }
        if size_bytes > self.max_audio_bytes {
            return Err(AppError::bad_request(format!(
                "Audio payload {size_bytes} bytes exceeds maximum {} bytes",
                self.max_audio_bytes
            )));
        }
        Ok(())
    }

    /// Transcribe an audio payload.
    ///
    /// Validates the MIME type, calls the API, records the result, and
    /// returns the transcription record.
    pub async fn transcribe(
        &self,
        session_id: &str,
        mime_type: &str,
        audio_bytes: Vec<u8>,
    ) -> Result<TranscriptionRecord, AppError> {
        self.validate_audio(mime_type, audio_bytes.len())?;

        let started = std::time::Instant::now();
        // The API trait is synchronous; call it via spawn_blocking so a slow
        // HTTP call does not block the async worker.
        let api = self.api.clone();
        let mime_type_owned = mime_type.to_string();
        let audio_bytes_clone = audio_bytes.clone();
        let audio_size = audio_bytes.len();
        let text = tokio::task::spawn_blocking(move || {
            api.transcribe(&mime_type_owned, &audio_bytes_clone)
        })
        .await
        .map_err(|e| AppError::internal(format!("Transcription task failed: {e}")))?
        .map_err(|e| {
            warn!(error = %e, "Transcription failed");
            e
        })?;

        let duration_ms = started.elapsed().as_millis() as u64;
        let record = TranscriptionRecord {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            mime_type: mime_type.to_string(),
            audio_size_bytes: audio_size,
            text,
            provider: self.provider,
            duration_ms,
            created_at: Utc::now(),
        };
        self.history
            .write()
            .insert(record.id.clone(), record.clone());
        info!(
            session_id = %session_id,
            duration_ms,
            bytes = audio_bytes.len(),
            "Audio transcribed"
        );
        Ok(record)
    }

    /// Look up a transcription record by id.
    pub fn get(&self, id: &str) -> Option<TranscriptionRecord> {
        self.history.read().get(id).cloned()
    }

    /// List transcription records for a session.
    pub fn list_for_session(&self, session_id: &str) -> Vec<TranscriptionRecord> {
        let mut records: Vec<TranscriptionRecord> = self
            .history
            .read()
            .values()
            .filter(|r| r.session_id == session_id)
            .cloned()
            .collect();
        records.sort_by_key(|b| std::cmp::Reverse(b.created_at));
        records
    }

    /// Return the provider in use.
    pub fn provider(&self) -> TranscriptionProvider {
        self.provider
    }
}

impl Default for TranscriptionService {
    fn default() -> Self {
        Self::new()
    }
}

/// Default maximum audio payload size (25 MiB).
pub const DEFAULT_MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validate_audio_mime() {
        let service = TranscriptionService::new();
        assert!(service.validate_audio("audio/mpeg", 100).is_ok());
        assert!(service.validate_audio("audio/wav", 100).is_ok());
        assert!(service.validate_audio("image/png", 100).is_err());
        assert!(service.validate_audio("audio/mpeg", 0).is_err());
    }

    #[tokio::test]
    async fn test_validate_size_limit() {
        let service = TranscriptionService::new().with_max_audio_bytes(10);
        assert!(service.validate_audio("audio/wav", 100).is_err());
    }

    #[tokio::test]
    async fn test_transcribe_with_stub() {
        let service = TranscriptionService::new();
        let record = service
            .transcribe("s1", "audio/wav", vec![1, 2, 3])
            .await
            .unwrap();
        assert!(record.text.contains("3 bytes"));
        assert_eq!(record.session_id, "s1");
        assert_eq!(record.provider, TranscriptionProvider::Local);
        assert!(record.duration_ms < 10_000);
    }

    #[tokio::test]
    async fn test_transcribe_unsupported_rejected() {
        let service = TranscriptionService::new();
        let result = service.transcribe("s1", "video/mp4", vec![1, 2, 3]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_history_lookup_and_list() {
        let service = TranscriptionService::new();
        let r1 = service
            .transcribe("s1", "audio/wav", vec![1])
            .await
            .unwrap();
        service
            .transcribe("s2", "audio/mpeg", vec![2])
            .await
            .unwrap();
        assert!(service.get(&r1.id).is_some());
        assert_eq!(service.list_for_session("s1").len(), 1);
    }

    #[tokio::test]
    async fn test_custom_api_failure_propagates() {
        struct FailingApi;
        impl TranscriptionApi for FailingApi {
            fn transcribe(&self, _mime: &str, _bytes: &[u8]) -> Result<String, AppError> {
                Err(AppError::internal("upstream transcription failed"))
            }
        }
        let service =
            TranscriptionService::with_api(Arc::new(FailingApi), TranscriptionProvider::Whisper);
        let result = service.transcribe("s1", "audio/wav", vec![1]).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("upstream"));
    }
}
