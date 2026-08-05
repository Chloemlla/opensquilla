//! Audio provider: text-to-speech, speech-to-text, voice management, and
//! ElevenLabs production audio APIs.
//!
//! Makes raw `reqwest` HTTP calls to audio APIs:
//!
//! * **ElevenLabs** — `POST /v1/text-to-speech/{voice}` (TTS),
//!   `POST /v1/speech-to-text` (STT), `/v1/voices` (voice management),
//!   `/v1/voices/add` (voice cloning), `/v1/speech-to-speech/{voice}` (voice
//!   conversion), `/v1/dubbing` (dubbing), `/v1/music` (music generation), and
//!   `/v1/user/subscription`.
//! * **OpenAI TTS** — `POST /v1/audio/speech` (tts-1, tts-1-hd).
//! * **OpenAI Whisper** — `POST /v1/audio/transcriptions` (speech-to-text).
//! * **Local** — a generic OpenAI-compatible endpoint.
//!
//! No SDK dependency. Streaming TTS is supported via
//! [`AudioProvider::stream_synthesize`], which yields audio byte chunks as
//! they arrive.

use crate::types::{ProviderError, ProviderResult};
use crate::util::{DEFAULT_TIMEOUT, RateLimiter, RetryConfig, check_status, with_retry};
use futures::{Stream, StreamExt};
use reqwest::Client;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, info};

/// Supported audio output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MP3 audio.
    Mp3,
    /// WAV audio (PCM in a RIFF container).
    Wav,
    /// OGG Opus.
    Opus,
    /// AAC (MPEG-4).
    Aac,
    /// FLAC lossless.
    Flac,
    /// Raw 16-bit PCM at 44.1kHz.
    Pcm,
}

impl Default for AudioFormat {
    fn default() -> Self {
        AudioFormat::Mp3
    }
}

impl AudioFormat {
    /// The file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Wav => "wav",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
            AudioFormat::Pcm => "pcm",
        }
    }

    /// The MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Aac => "audio/aac",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Pcm => "audio/L16",
        }
    }

    /// The OpenAI TTS `response_format` string.
    pub fn openai_format(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Wav => "wav",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
            AudioFormat::Pcm => "pcm",
        }
    }

    /// The ElevenLabs `output_format` string.
    ///
    /// ElevenLabs exposes a flat namespace of container + sample-rate
    /// combinations; unsupported formats are mapped to the closest equivalent
    /// and should be rejected via [`AudioFormat::is_elevenlabs_supported`]
    /// before the request is sent.
    pub fn elevenlabs_format(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3_44100_128",
            AudioFormat::Wav => "pcm_44100",
            AudioFormat::Opus => "opus_44100_128",
            AudioFormat::Aac => "mp3_44100_192",
            AudioFormat::Flac => "flac_44100",
            AudioFormat::Pcm => "pcm_16000",
        }
    }

    /// Whether ElevenLabs actually supports this output format.
    pub fn is_elevenlabs_supported(&self) -> bool {
        matches!(
            self,
            AudioFormat::Mp3 | AudioFormat::Wav | AudioFormat::Flac | AudioFormat::Pcm
        )
    }
}

/// The audio backend family a provider targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioProviderType {
    /// ElevenLabs production audio API.
    ElevenLabs,
    /// OpenAI TTS (`/v1/audio/speech`).
    OpenAiTts,
    /// OpenAI Whisper (`/v1/audio/transcriptions`).
    OpenAiStt,
    /// A generic OpenAI-compatible endpoint (usually local).
    Local,
}

impl AudioProviderType {
    /// The stable string name for this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioProviderType::ElevenLabs => "elevenlabs",
            AudioProviderType::OpenAiTts => "openai-tts",
            AudioProviderType::OpenAiStt => "openai-stt",
            AudioProviderType::Local => "local",
        }
    }

    /// Parse a backend name back into an [`AudioProviderType`].
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "elevenlabs" | "eleven" => Some(AudioProviderType::ElevenLabs),
            "openai-tts" | "openai_tts" | "openai" => Some(AudioProviderType::OpenAiTts),
            "openai-stt" | "openai_stt" | "whisper" => Some(AudioProviderType::OpenAiStt),
            "local" => Some(AudioProviderType::Local),
            _ => None,
        }
    }

    /// The default base URL for this backend.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            AudioProviderType::ElevenLabs => "https://api.elevenlabs.io",
            AudioProviderType::OpenAiTts | AudioProviderType::OpenAiStt => {
                "https://api.openai.com/v1"
            }
            AudioProviderType::Local => "http://localhost:8000/v1",
        }
    }

    /// A sensible default voice name.
    pub fn default_voice(&self) -> &'static str {
        match self {
            AudioProviderType::ElevenLabs => "Rachel",
            AudioProviderType::OpenAiTts
            | AudioProviderType::OpenAiStt
            | AudioProviderType::Local => "alloy",
        }
    }

    /// A sensible default model name.
    pub fn default_model(&self) -> &'static str {
        match self {
            AudioProviderType::ElevenLabs => "eleven_multilingual_v2",
            AudioProviderType::OpenAiTts => "tts-1",
            AudioProviderType::OpenAiStt => "whisper-1",
            AudioProviderType::Local => "local",
        }
    }
}

/// Configuration for an audio provider.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Which backend family to call.
    pub provider: AudioProviderType,
    /// API key.
    pub api_key: String,
    /// Base URL. Paths are appended with the backend's well-known layout.
    pub base_url: String,
    /// Default voice used when a request does not specify one.
    pub default_voice: String,
    /// Default model used when a request does not specify one.
    pub default_model: String,
    /// HTTP timeout for each request.
    pub timeout: Duration,
    /// Retry policy for transient failures.
    pub retry: RetryConfig,
    /// Explicit cost override in USD.
    pub cost_override: Option<f64>,
}

impl AudioConfig {
    /// Create a config with sensible defaults for the given backend.
    pub fn new(provider: AudioProviderType, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            base_url: provider.default_base_url().to_string(),
            default_voice: provider.default_voice().to_string(),
            default_model: provider.default_model().to_string(),
            timeout: DEFAULT_TIMEOUT,
            retry: RetryConfig::default(),
            cost_override: None,
        }
    }
}

/// A text-to-speech request (low-level form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    /// The text to synthesize.
    pub text: String,
    /// The model to use (e.g. "tts-1", "tts-1-hd", "eleven_multilingual_v2").
    pub model: String,
    /// The voice id / name to use.
    #[serde(default = "default_voice")]
    pub voice: String,
    /// Output audio format.
    #[serde(default)]
    pub format: AudioFormat,
    /// Speaking speed multiplier (0.25 - 4.0, OpenAI only).
    #[serde(default = "default_speed", skip_serializing_if = "is_default_speed")]
    pub speed: f64,
    /// Voice stability (0.0 - 1.0, ElevenLabs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    /// Voice similarity boost (0.0 - 1.0, ElevenLabs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f64>,
    /// Voice style exaggeration (0.0 - 1.0, ElevenLabs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f64>,
    /// Whether to use speaker boost (ElevenLabs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
    /// Language code override (ElevenLabs multilingual models).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language_code: Option<String>,
}

fn default_voice() -> String {
    "alloy".into()
}
fn default_speed() -> f64 {
    1.0
}
fn is_default_speed(s: &f64) -> bool {
    *s == 1.0
}

impl Default for TtsRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            model: "tts-1".into(),
            voice: default_voice(),
            format: AudioFormat::Mp3,
            speed: default_speed(),
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            language_code: None,
        }
    }
}

/// A TTS response containing the synthesized audio bytes.
#[derive(Debug, Clone)]
pub struct TtsResponse {
    /// The raw audio bytes.
    pub audio: Vec<u8>,
    /// The format of the audio.
    pub format: AudioFormat,
    /// The model used.
    pub model: String,
    /// Estimated cost in USD, when known.
    pub cost: Option<f64>,
    /// The generation id reported by the provider, when present.
    pub generation_id: Option<String>,
    /// The voice used.
    pub voice: Option<String>,
}

impl TtsResponse {
    /// Returns the number of bytes of audio.
    pub fn len(&self) -> usize {
        self.audio.len()
    }

    /// Returns `true` if the audio buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.audio.is_empty()
    }
}

/// Parameters for a high-level [`AudioProvider::text_to_speech`] call. `None`
/// fields fall back to the provider's [`AudioConfig`] defaults.
#[derive(Debug, Clone)]
pub struct TtsParams {
    /// The text to synthesize.
    pub text: String,
    /// Voice override.
    pub voice: Option<String>,
    /// Model override.
    pub model: Option<String>,
    /// Speaking speed multiplier.
    pub speed: Option<f32>,
    /// Output format override.
    pub format: Option<AudioFormat>,
    /// Language code override.
    pub language_code: Option<String>,
    /// Voice stability (ElevenLabs).
    pub stability: Option<f32>,
    /// Similarity boost (ElevenLabs).
    pub similarity_boost: Option<f32>,
    /// Style exaggeration (ElevenLabs).
    pub style: Option<f32>,
    /// Speaker boost (ElevenLabs).
    pub use_speaker_boost: Option<bool>,
}

impl Default for TtsParams {
    fn default() -> Self {
        Self {
            text: String::new(),
            voice: None,
            model: None,
            speed: Some(1.0),
            format: Some(AudioFormat::Mp3),
            language_code: None,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
        }
    }
}

impl TtsParams {
    /// Materialize into a low-level [`TtsRequest`], resolving `None` fields
    /// against the provider config.
    pub fn to_request(&self, config: &AudioConfig) -> TtsRequest {
        TtsRequest {
            text: self.text.clone(),
            model: self
                .model
                .clone()
                .unwrap_or_else(|| config.default_model.clone()),
            voice: self
                .voice
                .clone()
                .unwrap_or_else(|| config.default_voice.clone()),
            format: self.format.unwrap_or(AudioFormat::Mp3),
            speed: self.speed.unwrap_or(1.0) as f64,
            stability: self.stability.map(|x| x as f64),
            similarity_boost: self.similarity_boost.map(|x| x as f64),
            style: self.style.map(|x| x as f64),
            use_speaker_boost: self.use_speaker_boost,
            language_code: self.language_code.clone(),
        }
    }
}

/// A high-level text-to-speech result.
#[derive(Debug, Clone)]
pub struct TtsResult {
    /// The synthesized audio bytes.
    pub audio_data: Vec<u8>,
    /// The audio format.
    pub format: AudioFormat,
    /// Estimated audio duration in seconds.
    pub duration_seconds: f64,
    /// Estimated cost in USD, when known.
    pub cost: Option<f64>,
    /// The model that produced the audio.
    pub model: String,
    /// The voice used.
    pub voice: String,
    /// The provider's generation id, when reported.
    pub generation_id: Option<String>,
}

/// Parameters for [`AudioProvider::speech_to_text`].
#[derive(Debug, Clone)]
pub struct SttParams {
    /// The raw audio bytes to transcribe.
    pub audio_data: Vec<u8>,
    /// The source filename (e.g. "clip.wav").
    pub filename: String,
    /// The source MIME type (e.g. "audio/wav").
    pub mime_type: String,
    /// Model override (e.g. "whisper-1", "scribe_v2").
    pub model: Option<String>,
    /// ISO-639-1 language hint.
    pub language: Option<String>,
    /// A context prompt that guides the transcription.
    pub prompt: Option<String>,
}

/// A single timed transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttSegment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// The transcribed text for this segment.
    pub text: String,
}

/// A speech-to-text result.
#[derive(Debug, Clone)]
pub struct SttResult {
    /// The full transcribed text.
    pub text: String,
    /// The detected language, when reported.
    pub language: Option<String>,
    /// The audio duration in seconds, when known.
    pub duration_seconds: f64,
    /// Timed segments (OpenAI verbose JSON).
    pub segments: Vec<SttSegment>,
    /// The model used.
    pub model: String,
    /// Language detection confidence (ElevenLabs).
    pub language_probability: Option<f64>,
    /// Word-level transcriptions (ElevenLabs).
    pub words: Vec<String>,
}

/// A voice available from the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// The voice id.
    pub id: String,
    /// The display name.
    pub name: String,
    /// The voice category (ElevenLabs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// A human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A preview clip URL, when provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview_url: Option<String>,
}

/// Per-voice configuration (ElevenLabs).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VoiceSettings {
    /// Voice stability (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stability: Option<f32>,
    /// Similarity boost (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f32>,
    /// Style exaggeration (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f32>,
    /// Whether speaker boost is enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
    /// Speaking speed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

impl VoiceSettings {
    /// Defaults for OpenAI voices (no per-voice settings).
    pub fn openai_default() -> Self {
        Self::default()
    }

    /// ElevenLabs midpoint defaults.
    pub fn elevenlabs_default() -> Self {
        Self {
            stability: Some(0.5),
            similarity_boost: Some(0.75),
            style: Some(0.0),
            use_speaker_boost: Some(false),
            speed: Some(1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// ElevenLabs production audio request/result types
// ---------------------------------------------------------------------------

/// Parameters for [`AudioProvider::clone_voice`].
#[derive(Debug, Clone)]
pub struct VoiceCloneParams {
    /// A short sample of the target voice.
    pub sample_audio: Vec<u8>,
    /// The sample filename.
    pub sample_filename: String,
    /// The sample MIME type.
    pub sample_mime_type: String,
    /// The name to give the cloned voice.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Optional labels (serialized as JSON).
    pub labels: HashMap<String, String>,
}

/// Result of [`AudioProvider::clone_voice`].
#[derive(Debug, Clone)]
pub struct VoiceCloneResult {
    /// The provider id ("elevenlabs").
    pub provider: String,
    /// The new voice id.
    pub voice_id: String,
    /// The voice name.
    pub name: String,
    /// A preview clip URL, when returned.
    pub preview_url: Option<String>,
    /// Whether the voice requires verification before use.
    pub requires_verification: bool,
}

/// Parameters for [`AudioProvider::convert_voice`] (speech-to-speech).
#[derive(Debug, Clone)]
pub struct VoiceConversionParams {
    /// The source audio bytes.
    pub source_audio: Vec<u8>,
    /// The source filename.
    pub source_filename: String,
    /// The source MIME type.
    pub source_mime_type: String,
    /// The target voice id.
    pub target_voice: String,
    /// Model override.
    pub model: Option<String>,
    /// Output format.
    pub output_format: AudioFormat,
}

/// Result of [`AudioProvider::convert_voice`].
#[derive(Debug, Clone)]
pub struct VoiceConversionResult {
    /// The converted audio bytes.
    pub audio: Vec<u8>,
    /// The provider id.
    pub provider: String,
    /// The model used.
    pub model: String,
    /// The target voice.
    pub voice: String,
    /// The output format.
    pub format: AudioFormat,
    /// The output MIME type.
    pub mime_type: String,
}

/// Parameters for [`AudioProvider::create_dubbing`].
#[derive(Debug, Clone)]
pub struct DubbingParams {
    /// The source audio/video bytes.
    pub source: Vec<u8>,
    /// The source filename.
    pub filename: String,
    /// The source MIME type.
    pub mime_type: String,
    /// The target language code.
    pub target_language: String,
    /// The source language code, when known.
    pub source_language: Option<String>,
    /// An optional name for the dubbing job.
    pub name: Option<String>,
    /// The number of speakers, when known.
    pub num_speakers: Option<u32>,
    /// Whether to add a watermark.
    pub watermark: Option<bool>,
}

/// Result of [`AudioProvider::create_dubbing`].
#[derive(Debug, Clone)]
pub struct DubbingResult {
    /// The provider id.
    pub provider: String,
    /// The dubbing job id.
    pub dubbing_id: String,
    /// The job status ("submitted", "processing", "done", ...).
    pub status: String,
    /// The target language code.
    pub target_language: String,
}

/// Status of a dubbing job.
#[derive(Debug, Clone)]
pub struct DubbingStatus {
    /// The provider id.
    pub provider: String,
    /// The dubbing job id.
    pub dubbing_id: String,
    /// The job status.
    pub status: String,
}

/// Parameters for [`AudioProvider::generate_music`].
#[derive(Debug, Clone)]
pub struct MusicGenerationParams {
    /// The music prompt.
    pub prompt: String,
    /// Model override.
    pub model: Option<String>,
    /// Output format.
    pub output_format: AudioFormat,
    /// Optional lyrics.
    pub lyrics: Option<String>,
    /// Desired duration in seconds.
    pub duration_seconds: Option<f64>,
    /// Whether to force an instrumental (no vocals).
    pub force_instrumental: bool,
}

/// Result of [`AudioProvider::generate_music`].
#[derive(Debug, Clone)]
pub struct MusicGenerationResult {
    /// The generated audio bytes.
    pub audio: Vec<u8>,
    /// The provider id.
    pub provider: String,
    /// The model used.
    pub model: String,
    /// The output format.
    pub format: AudioFormat,
    /// The output MIME type.
    pub mime_type: String,
}

/// ElevenLabs subscription info.
#[derive(Debug, Clone)]
pub struct SubscriptionInfo {
    /// The provider id.
    pub provider: String,
    /// The tier / plan name.
    pub tier: Option<String>,
    /// The subscription status.
    pub status: Option<String>,
}

/// Search filters for [`AudioProvider::search_shared_voices`].
#[derive(Debug, Clone, Default)]
pub struct SharedVoicesQuery {
    /// Language filter.
    pub language: Option<String>,
    /// Accent filter.
    pub accent: Option<String>,
    /// Locale filter.
    pub locale: Option<String>,
    /// Gender filter.
    pub gender: Option<String>,
    /// Age group filter.
    pub age: Option<String>,
    /// Category filter.
    pub category: Option<String>,
    /// Free-text search.
    pub search: Option<String>,
    /// Page size (clamped to 1..=50).
    pub page_size: u32,
}

/// Provider for audio APIs.
#[derive(Clone)]
pub struct AudioProvider {
    client: Client,
    config: AudioConfig,
    limiter: Option<RateLimiter>,
}

impl AudioProvider {
    /// Create a provider from an explicit config.
    pub fn new(config: AudioConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create reqwest Client");
        Self {
            client,
            config,
            limiter: None,
        }
    }

    /// Create an ElevenLabs provider.
    pub fn elevenlabs(api_key: impl Into<String>) -> Self {
        Self::new(AudioConfig::new(AudioProviderType::ElevenLabs, api_key))
    }

    /// Create an OpenAI TTS provider.
    pub fn openai_tts(api_key: impl Into<String>) -> Self {
        Self::new(AudioConfig::new(AudioProviderType::OpenAiTts, api_key))
    }

    /// Create an OpenAI Whisper speech-to-text provider.
    pub fn openai_stt(api_key: impl Into<String>) -> Self {
        Self::new(AudioConfig::new(AudioProviderType::OpenAiStt, api_key))
    }

    /// Create a provider for a local OpenAI-compatible endpoint.
    pub fn local(base_url: impl Into<String>) -> Self {
        let mut config = AudioConfig::new(AudioProviderType::Local, "");
        config.base_url = base_url.into();
        Self::new(config)
    }

    /// Backward-compatible alias for [`Self::openai_tts`].
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::openai_tts(api_key)
    }

    /// Attach a rate limiter that gates every outbound request.
    pub fn with_limiter(mut self, limiter: RateLimiter) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.config.base_url = base_url.into();
        self
    }

    /// Override the default voice.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.config.default_voice = voice.into();
        self
    }

    /// Override the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Override the HTTP timeout (rebuilds the underlying client).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create reqwest Client");
        self
    }

    /// Override the retry policy.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.config.retry = retry;
        self
    }

    /// Set an explicit cost override.
    pub fn with_cost_override(mut self, cost: f64) -> Self {
        self.config.cost_override = Some(cost);
        self
    }

    /// The provider name (e.g. "elevenlabs", "openai-tts").
    pub fn name(&self) -> &str {
        self.config.provider.as_str()
    }

    /// The provider configuration.
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Build an absolute endpoint URL, tolerating a base that already ends in
    /// `/v1` when the path also starts with `/v1`.
    fn endpoint(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        if base.ends_with("/v1") && path.starts_with("/v1/") {
            format!("{base}{}", &path[3..])
        } else {
            format!("{base}{path}")
        }
    }

    /// Synthesize speech, returning the full audio buffer.
    pub async fn synthesize(&self, request: &TtsRequest) -> ProviderResult<TtsResponse> {
        debug!(
            target = "provider",
            provider = %self.name(),
            model = %request.model,
            "Synthesizing speech"
        );

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let mut resp = match self.config.provider {
            AudioProviderType::ElevenLabs => {
                with_retry(&self.config.retry, |_attempt| {
                    self.synthesize_elevenlabs(request)
                })
                .await?
            }
            AudioProviderType::OpenAiTts | AudioProviderType::Local => {
                with_retry(&self.config.retry, |_attempt| {
                    self.synthesize_openai(request)
                })
                .await?
            }
            AudioProviderType::OpenAiStt => {
                return Err(ProviderError::Config(
                    "OpenAI STT provider cannot perform text-to-speech".into(),
                ));
            }
        };

        if resp.cost.is_none() {
            resp.cost = estimate_tts_cost(&request.model, request.text.chars().count());
        }
        Ok(resp)
    }

    /// Stream speech synthesis, returning a stream of audio byte chunks.
    ///
    /// This is useful for low-latency playback: audio chunks can be fed to a
    /// decoder/player as they arrive rather than waiting for the full buffer.
    /// Streaming is not retried (the body cannot be re-read after the first
    /// chunk).
    pub async fn stream_synthesize(
        &self,
        request: &TtsRequest,
    ) -> ProviderResult<Box<dyn Stream<Item = Result<Vec<u8>, ProviderError>> + Send + Unpin>> {
        debug!(
            target = "provider",
            provider = %self.name(),
            model = %request.model,
            "Streaming speech synthesis"
        );

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let resp = match self.config.provider {
            AudioProviderType::ElevenLabs => self.send_elevenlabs(request).await?,
            AudioProviderType::OpenAiTts | AudioProviderType::Local => {
                self.send_openai(request).await?
            }
            AudioProviderType::OpenAiStt => {
                return Err(ProviderError::Config(
                    "OpenAI STT provider cannot perform text-to-speech".into(),
                ));
            }
        };

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "text to speech")?;
            unreachable!("check_status returned Ok for a non-success status");
        }

        let byte_stream = resp.bytes_stream();
        let mapped = byte_stream
            .map(|result| result.map(|b| b.to_vec()).map_err(ProviderError::Network))
            .boxed();
        Ok(Box::new(mapped))
    }

    /// High-level text-to-speech entry point.
    pub async fn text_to_speech(&self, params: &TtsParams) -> ProviderResult<TtsResult> {
        let request = params.to_request(&self.config);
        let response = self.synthesize(&request).await?;
        let duration = estimate_tts_duration(&response);
        let cost = self
            .config
            .cost_override
            .or(response.cost)
            .or(estimate_tts_cost(
                &request.model,
                request.text.chars().count(),
            ));
        Ok(TtsResult {
            audio_data: response.audio,
            format: response.format,
            duration_seconds: duration,
            cost,
            model: response.model,
            voice: response.voice.unwrap_or_else(|| request.voice.clone()),
            generation_id: response.generation_id,
        })
    }

    /// High-level streaming text-to-speech entry point.
    pub async fn stream_text_to_speech(
        &self,
        params: &TtsParams,
    ) -> ProviderResult<Box<dyn Stream<Item = Result<Vec<u8>, ProviderError>> + Send + Unpin>> {
        let request = params.to_request(&self.config);
        self.stream_synthesize(&request).await
    }

    /// Transcribe audio to text.
    pub async fn speech_to_text(&self, params: &SttParams) -> ProviderResult<SttResult> {
        debug!(
            target = "provider",
            provider = %self.name(),
            model = %params.model.as_deref().unwrap_or(&self.config.default_model),
            "Transcribing speech"
        );

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let result = match self.config.provider {
            AudioProviderType::ElevenLabs => {
                with_retry(&self.config.retry, |_attempt| {
                    self.transcribe_elevenlabs(params)
                })
                .await?
            }
            AudioProviderType::OpenAiStt | AudioProviderType::Local => {
                with_retry(&self.config.retry, |_attempt| {
                    self.transcribe_openai(params)
                })
                .await?
            }
            AudioProviderType::OpenAiTts => {
                return Err(ProviderError::Config(
                    "OpenAI TTS provider cannot perform speech-to-text".into(),
                ));
            }
        };
        Ok(result)
    }

    /// Convenience transcription that builds [`SttParams`] from raw audio.
    pub async fn transcribe(
        &self,
        audio_data: Vec<u8>,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> ProviderResult<SttResult> {
        let params = SttParams {
            audio_data,
            filename: filename.into(),
            mime_type: mime_type.into(),
            model: None,
            language: None,
            prompt: None,
        };
        self.speech_to_text(&params).await
    }

    /// List the voices available from the provider.
    pub async fn list_voices(&self) -> ProviderResult<Vec<Voice>> {
        match self.config.provider {
            AudioProviderType::ElevenLabs => {
                with_retry(&self.config.retry, |_attempt| self.list_elevenlabs_voices()).await
            }
            AudioProviderType::OpenAiTts | AudioProviderType::Local => Ok(default_openai_voices()),
            AudioProviderType::OpenAiStt => Err(ProviderError::Config(
                "OpenAI STT provider has no voices".into(),
            )),
        }
    }

    /// Fetch the configuration for a voice.
    pub async fn get_voice_settings(&self, voice_id: &str) -> ProviderResult<VoiceSettings> {
        match self.config.provider {
            AudioProviderType::ElevenLabs => {
                with_retry(&self.config.retry, |_attempt| {
                    self.get_elevenlabs_voice_settings(voice_id)
                })
                .await
            }
            AudioProviderType::OpenAiTts | AudioProviderType::Local => {
                Ok(VoiceSettings::openai_default())
            }
            AudioProviderType::OpenAiStt => Err(ProviderError::Config(
                "OpenAI STT provider has no voice settings".into(),
            )),
        }
    }

    // --- ElevenLabs production audio --------------------------------------

    /// Clone a voice from a sample (ElevenLabs `/v1/voices/add`).
    pub async fn clone_voice(&self, params: &VoiceCloneParams) -> ProviderResult<VoiceCloneResult> {
        let file_part = Part::bytes(params.sample_audio.clone())
            .file_name(params.sample_filename.clone())
            .mime_str(&params.sample_mime_type)
            .map_err(ProviderError::Network)?;
        let mut form = Form::new()
            .text("name", params.name.clone())
            .part("files", file_part);
        if let Some(description) = &params.description {
            form = form.text("description", description.clone());
        }
        if !params.labels.is_empty() {
            let labels =
                serde_json::to_string(&params.labels).map_err(ProviderError::Serialization)?;
            form = form.text("labels", labels);
        }

        let resp = self
            .client
            .post(self.endpoint("/v1/voices/add"))
            .header("xi-api-key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "voice clone")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let voice_id = data
            .get("voice_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                ProviderError::Provider("Voice clone provider returned no voice_id".into())
            })?;
        let name = data
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .map(String::from)
            .unwrap_or_else(|| params.name.clone());

        Ok(VoiceCloneResult {
            provider: "elevenlabs".into(),
            voice_id: voice_id.to_string(),
            name,
            preview_url: data
                .get("preview_url")
                .and_then(|v| v.as_str())
                .map(String::from),
            requires_verification: data
                .get("requires_verification")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }

    /// Convert speech from one voice to another (ElevenLabs
    /// `/v1/speech-to-speech/{voice}`).
    pub async fn convert_voice(
        &self,
        params: &VoiceConversionParams,
    ) -> ProviderResult<VoiceConversionResult> {
        if !params.output_format.is_elevenlabs_supported() {
            return Err(ProviderError::Config(format!(
                "ElevenLabs does not support {} output",
                params.output_format.extension()
            )));
        }
        let file_part = Part::bytes(params.source_audio.clone())
            .file_name(params.source_filename.clone())
            .mime_str(&params.source_mime_type)
            .map_err(ProviderError::Network)?;
        let model = params
            .model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());
        let form = Form::new()
            .text("model_id", model.clone())
            .part("audio", file_part);

        let resp = self
            .client
            .post(self.endpoint(&format!("/v1/speech-to-speech/{}", params.target_voice)))
            .query(&[("output_format", params.output_format.elevenlabs_format())])
            .header("xi-api-key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "voice conversion")?;
            unreachable!("check_status returned Ok for a non-success status");
        }
        let audio = resp.bytes().await.map_err(ProviderError::Network)?.to_vec();
        if audio.is_empty() {
            return Err(ProviderError::Provider(
                "Voice conversion provider returned no audio".into(),
            ));
        }

        Ok(VoiceConversionResult {
            audio,
            provider: "elevenlabs".into(),
            model,
            voice: params.target_voice.clone(),
            format: params.output_format,
            mime_type: params.output_format.mime_type().to_string(),
        })
    }

    /// Submit a dubbing job (ElevenLabs `/v1/dubbing`).
    pub async fn create_dubbing(&self, params: &DubbingParams) -> ProviderResult<DubbingResult> {
        let file_part = Part::bytes(params.source.clone())
            .file_name(params.filename.clone())
            .mime_str(&params.mime_type)
            .map_err(ProviderError::Network)?;
        let mut form = Form::new()
            .text("target_lang", params.target_language.clone())
            .part("file", file_part);
        if let Some(source_language) = &params.source_language {
            form = form.text("source_lang", source_language.clone());
        }
        if let Some(name) = &params.name {
            form = form.text("name", name.clone());
        }
        if let Some(num_speakers) = params.num_speakers {
            form = form.text("num_speakers", num_speakers.to_string());
        }
        if let Some(watermark) = params.watermark {
            form = form.text("watermark", watermark.to_string());
        }

        let resp = self
            .client
            .post(self.endpoint("/v1/dubbing"))
            .header("xi-api-key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "dubbing")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let dubbing_id = data
            .get("dubbing_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                ProviderError::Provider("Dubbing provider returned no dubbing_id".into())
            })?;
        let status_str = data
            .get("status")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("submitted")
            .to_string();

        Ok(DubbingResult {
            provider: "elevenlabs".into(),
            dubbing_id: dubbing_id.to_string(),
            status: status_str,
            target_language: params.target_language.clone(),
        })
    }

    /// Check a dubbing job's status (ElevenLabs `/v1/dubbing/{id}`).
    pub async fn dubbing_status(&self, dubbing_id: &str) -> ProviderResult<DubbingStatus> {
        let resp = self
            .client
            .get(self.endpoint(&format!("/v1/dubbing/{dubbing_id}")))
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "dubbing status")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let state = data
            .get("status")
            .or_else(|| data.get("state"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("unknown")
            .to_string();
        Ok(DubbingStatus {
            provider: "elevenlabs".into(),
            dubbing_id: dubbing_id.to_string(),
            status: state,
        })
    }

    /// Download a finished dubbing's audio (ElevenLabs
    /// `/v1/dubbing/{id}/audio/{lang}`).
    pub async fn download_dubbing(
        &self,
        dubbing_id: &str,
        language_code: &str,
    ) -> ProviderResult<Vec<u8>> {
        let resp = self
            .client
            .get(self.endpoint(&format!("/v1/dubbing/{dubbing_id}/audio/{language_code}")))
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "dubbing download")?;
            unreachable!("check_status returned Ok for a non-success status");
        }
        let audio = resp.bytes().await.map_err(ProviderError::Network)?.to_vec();
        if audio.is_empty() {
            return Err(ProviderError::Provider(
                "Dubbing download provider returned no audio".into(),
            ));
        }
        Ok(audio)
    }

    /// Generate music from a prompt (ElevenLabs `/v1/music`).
    pub async fn generate_music(
        &self,
        params: &MusicGenerationParams,
    ) -> ProviderResult<MusicGenerationResult> {
        if !params.output_format.is_elevenlabs_supported() {
            return Err(ProviderError::Config(format!(
                "ElevenLabs does not support {} output",
                params.output_format.extension()
            )));
        }
        let model = params
            .model
            .clone()
            .unwrap_or_else(|| self.config.default_model.clone());
        let mut prompt = params.prompt.clone();
        if let Some(lyrics) = &params.lyrics {
            prompt = format!("{prompt}\nLyrics:\n{lyrics}");
        }
        let mut body = serde_json::json!({
            "prompt": prompt,
            "model_id": model,
            "force_instrumental": params.force_instrumental,
        });
        if let Some(duration) = params.duration_seconds {
            body["music_length_ms"] = serde_json::json!((duration * 1000.0) as u64);
        }
        if let Some(lyrics) = &params.lyrics {
            body["lyrics"] = serde_json::json!(lyrics);
        }

        let resp = self
            .client
            .post(self.endpoint("/v1/music"))
            .query(&[("output_format", params.output_format.elevenlabs_format())])
            .header("xi-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "music generation")?;
            unreachable!("check_status returned Ok for a non-success status");
        }
        let audio = resp.bytes().await.map_err(ProviderError::Network)?.to_vec();
        if audio.is_empty() {
            return Err(ProviderError::Provider(
                "Music generation provider returned no audio".into(),
            ));
        }
        Ok(MusicGenerationResult {
            audio,
            provider: "elevenlabs".into(),
            model,
            format: params.output_format,
            mime_type: params.output_format.mime_type().to_string(),
        })
    }

    /// Fetch the ElevenLabs subscription (`/v1/user/subscription`).
    pub async fn get_subscription(&self) -> ProviderResult<SubscriptionInfo> {
        let resp = self
            .client
            .get(self.endpoint("/v1/user/subscription"))
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "ElevenLabs subscription")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let tier = data
            .get("tier")
            .or_else(|| data.get("plan"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let status_str = data
            .get("status")
            .and_then(|v| v.as_str())
            .map(String::from);
        Ok(SubscriptionInfo {
            provider: "elevenlabs".into(),
            tier,
            status: status_str,
        })
    }

    /// Search the ElevenLabs shared-voice library (`/v1/shared-voices`).
    pub async fn search_shared_voices(
        &self,
        query: &SharedVoicesQuery,
    ) -> ProviderResult<Vec<Voice>> {
        let page_size = query.page_size.clamp(1, 50);
        let mut params: Vec<(String, String)> =
            vec![("page_size".to_string(), page_size.to_string())];
        for (key, value) in [
            ("language", &query.language),
            ("accent", &query.accent),
            ("locale", &query.locale),
            ("gender", &query.gender),
            ("age", &query.age),
            ("category", &query.category),
            ("search", &query.search),
        ] {
            if let Some(v) = value {
                if !v.trim().is_empty() {
                    params.push((key.to_string(), v.trim().to_string()));
                }
            }
        }

        let resp = self
            .client
            .get(self.endpoint("/v1/shared-voices"))
            .query(&params)
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "shared voices search")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let voices = data
            .get("voices")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(parse_voice_item).collect())
            .unwrap_or_default();
        Ok(voices)
    }

    // --- Backend implementations ------------------------------------------

    async fn send_openai(&self, request: &TtsRequest) -> ProviderResult<reqwest::Response> {
        let body = build_openai_tts_body(request);
        self.client
            .post(self.endpoint("/audio/speech"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)
    }

    async fn synthesize_openai(&self, request: &TtsRequest) -> ProviderResult<TtsResponse> {
        let resp = self.send_openai(request).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "text to speech")?;
            unreachable!("check_status returned Ok for a non-success status");
        }
        let audio = resp.bytes().await.map_err(ProviderError::Network)?.to_vec();
        info!(
            target = "provider",
            provider = %self.name(),
            bytes = audio.len(),
            "Speech synthesized"
        );
        Ok(TtsResponse {
            audio,
            format: request.format,
            model: request.model.clone(),
            cost: None,
            generation_id: None,
            voice: Some(request.voice.clone()),
        })
    }

    async fn send_elevenlabs(&self, request: &TtsRequest) -> ProviderResult<reqwest::Response> {
        if !request.format.is_elevenlabs_supported() {
            return Err(ProviderError::Config(format!(
                "ElevenLabs does not support {} output",
                request.format.extension()
            )));
        }
        let url = format!(
            "{}/v1/text-to-speech/{}",
            self.config.base_url.trim_end_matches('/'),
            request.voice
        );
        let body = build_elevenlabs_tts_body(request);
        self.client
            .post(&url)
            .query(&[("output_format", request.format.elevenlabs_format())])
            .header("xi-api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", request.format.mime_type())
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)
    }

    async fn synthesize_elevenlabs(&self, request: &TtsRequest) -> ProviderResult<TtsResponse> {
        let resp = self.send_elevenlabs(request).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "text to speech")?;
            unreachable!("check_status returned Ok for a non-success status");
        }
        let generation_id = resp
            .headers()
            .get("x-generation-id")
            .and_then(|v| v.to_str().ok())
            .map(String::from);
        let audio = resp.bytes().await.map_err(ProviderError::Network)?.to_vec();
        info!(
            target = "provider",
            provider = %self.name(),
            bytes = audio.len(),
            "Speech synthesized"
        );
        Ok(TtsResponse {
            audio,
            format: request.format,
            model: request.model.clone(),
            cost: None,
            generation_id,
            voice: Some(request.voice.clone()),
        })
    }

    async fn transcribe_openai(&self, params: &SttParams) -> ProviderResult<SttResult> {
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let file_part = Part::bytes(params.audio_data.clone())
            .file_name(params.filename.clone())
            .mime_str(&params.mime_type)
            .map_err(ProviderError::Network)?;
        let mut form = Form::new()
            .text("model", model.to_string())
            .text("response_format", "verbose_json")
            .text("timestamp_granularities[]", "segment")
            .part("file", file_part);
        if let Some(language) = &params.language {
            form = form.text("language", language.clone());
        }
        if let Some(prompt) = &params.prompt {
            form = form.text("prompt", prompt.clone());
        }

        let resp = self
            .client
            .post(self.endpoint("/audio/transcriptions"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "speech to text")?;
        parse_openai_stt_json(&text, model)
    }

    async fn transcribe_elevenlabs(&self, params: &SttParams) -> ProviderResult<SttResult> {
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let file_part = Part::bytes(params.audio_data.clone())
            .file_name(params.filename.clone())
            .mime_str(&params.mime_type)
            .map_err(ProviderError::Network)?;
        let mut form = Form::new()
            .text("model_id", model.to_string())
            .part("file", file_part);
        if let Some(language) = &params.language {
            form = form.text("language_code", language.clone());
        }

        let resp = self
            .client
            .post(self.endpoint("/v1/speech-to-text"))
            .header("xi-api-key", &self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "speech to text")?;

        let mut result = parse_elevenlabs_stt_json(&text, model)?;
        // ElevenLabs does not report duration; estimate from WAV when possible.
        if params.mime_type.contains("wav") {
            if let Some(duration) = wav_duration_seconds(&params.audio_data) {
                result.duration_seconds = duration;
            }
        }
        Ok(result)
    }

    async fn list_elevenlabs_voices(&self) -> ProviderResult<Vec<Voice>> {
        let resp = self
            .client
            .get(self.endpoint("/v1/voices"))
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "ElevenLabs voices")?;
        parse_elevenlabs_voices_json(&text)
    }

    async fn get_elevenlabs_voice_settings(&self, voice_id: &str) -> ProviderResult<VoiceSettings> {
        let resp = self
            .client
            .get(self.endpoint(&format!("/v1/voices/{voice_id}/settings")))
            .header("xi-api-key", &self.config.api_key)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "ElevenLabs voice settings")?;
        parse_elevenlabs_voice_settings_json(&text)
    }
}

// ---------------------------------------------------------------------------
// Pure request builders and parsers (unit-testable without a network)
// ---------------------------------------------------------------------------

/// Build the OpenAI `/v1/audio/speech` JSON body.
pub fn build_openai_tts_body(request: &TtsRequest) -> serde_json::Value {
    serde_json::json!({
        "model": request.model,
        "input": request.text,
        "voice": request.voice,
        "response_format": request.format.openai_format(),
        "speed": request.speed,
    })
}

/// Build the ElevenLabs `/v1/text-to-speech/{voice}` JSON body.
pub fn build_elevenlabs_tts_body(request: &TtsRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "text": request.text,
        "model_id": request.model,
    });
    if let Some(lang) = &request.language_code {
        body["language_code"] = serde_json::json!(lang);
    }
    let mut settings = serde_json::Map::new();
    if let Some(stability) = request.stability {
        settings.insert("stability".into(), serde_json::json!(stability));
        settings.insert(
            "similarity_boost".into(),
            serde_json::json!(request.similarity_boost.unwrap_or(0.75)),
        );
    }
    if let Some(style) = request.style {
        settings.insert("style".into(), serde_json::json!(style));
    }
    if let Some(use_speaker_boost) = request.use_speaker_boost {
        settings.insert(
            "use_speaker_boost".into(),
            serde_json::json!(use_speaker_boost),
        );
    }
    if !settings.is_empty() {
        body["voice_settings"] = serde_json::Value::Object(settings);
    }
    body
}

/// Parse an OpenAI Whisper verbose-JSON transcription response.
pub fn parse_openai_stt_json(text: &str, model: &str) -> ProviderResult<SttResult> {
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let transcript = data
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::Provider("Speech to text provider returned no text".into()))?
        .to_string();
    let language = data
        .get("language")
        .and_then(|v| v.as_str())
        .map(String::from);
    let duration = data.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let segments = data
        .get("segments")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let start = s.get("start").and_then(|v| v.as_f64())?;
                    let end = s.get("end").and_then(|v| v.as_f64())?;
                    let text = s.get("text").and_then(|v| v.as_str())?.to_string();
                    Some(SttSegment { start, end, text })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(SttResult {
        text: transcript,
        language,
        duration_seconds: duration,
        segments,
        model: model.to_string(),
        language_probability: None,
        words: Vec::new(),
    })
}

/// Parse an ElevenLabs `/v1/speech-to-text` response.
pub fn parse_elevenlabs_stt_json(text: &str, model: &str) -> ProviderResult<SttResult> {
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let transcript = data
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProviderError::Provider("Speech to text provider returned no text".into()))?
        .to_string();
    let language = data
        .get("language_code")
        .and_then(|v| v.as_str())
        .map(String::from);
    let language_probability = data.get("language_probability").and_then(|v| v.as_f64());
    let words = data
        .get("words")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|w| w.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Ok(SttResult {
        text: transcript,
        language,
        duration_seconds: 0.0,
        segments: Vec::new(),
        model: model.to_string(),
        language_probability,
        words,
    })
}

/// Parse the ElevenLabs `/v1/voices` response into [`Voice`] entries.
pub fn parse_elevenlabs_voices_json(text: &str) -> ProviderResult<Vec<Voice>> {
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let voices = data
        .get("voices")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(parse_voice_item).collect())
        .unwrap_or_default();
    Ok(voices)
}

/// Parse one ElevenLabs voice object.
fn parse_voice_item(value: &serde_json::Value) -> Option<Voice> {
    let id = value.get("voice_id").and_then(|v| v.as_str())?.to_string();
    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_default();
    Some(Voice {
        id,
        name,
        category: value
            .get("category")
            .and_then(|v| v.as_str())
            .map(String::from),
        description: value
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from),
        preview_url: value
            .get("preview_url")
            .and_then(|v| v.as_str())
            .map(String::from),
    })
}

/// Parse the ElevenLabs `/v1/voices/{id}/settings` response.
pub fn parse_elevenlabs_voice_settings_json(text: &str) -> ProviderResult<VoiceSettings> {
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    Ok(VoiceSettings {
        stability: data
            .get("stability")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32),
        similarity_boost: data
            .get("similarity_boost")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32),
        style: data.get("style").and_then(|v| v.as_f64()).map(|x| x as f32),
        use_speaker_boost: data.get("use_speaker_boost").and_then(|v| v.as_bool()),
        speed: data.get("speed").and_then(|v| v.as_f64()).map(|x| x as f32),
    })
}

/// The standard OpenAI TTS voice set.
pub fn default_openai_voices() -> Vec<Voice> {
    const NAMES: &[&str] = &[
        "alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse",
    ];
    NAMES
        .iter()
        .map(|name| Voice {
            id: name.to_string(),
            name: name.to_string(),
            category: None,
            description: None,
            preview_url: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Audio format helpers and duration estimation
// ---------------------------------------------------------------------------

/// A wrapper around a WAV header followed by PCM samples, for providers that
/// return raw PCM. Prepends a minimal WAV header so the result is playable.
pub fn pcm_to_wav(pcm: &[u8], sample_rate: u32, channels: u16, bits_per_sample: u16) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let chunk_size = 36 + data_len;

    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(pcm);
    out
}

/// Strip a WAV header, returning the raw PCM sample bytes.
pub fn wav_to_pcm(wav: &[u8]) -> Option<Vec<u8>> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" {
        return None;
    }
    find_chunk(wav, b"data").map(|(offset, len)| wav[offset..offset + len].to_vec())
}

/// Locate a RIFF chunk by id, returning `(data_offset, data_len)`.
fn find_chunk(wav: &[u8], id: &[u8; 4]) -> Option<(usize, usize)> {
    let mut i = 12;
    while i + 8 <= wav.len() {
        if &wav[i..i + 4] == id {
            let len = u32::from_le_bytes([wav[i + 4], wav[i + 5], wav[i + 6], wav[i + 7]]) as usize;
            return Some((i + 8, len));
        }
        let len = u32::from_le_bytes([wav[i + 4], wav[i + 5], wav[i + 6], wav[i + 7]]) as usize;
        i += 8 + len + (len % 2);
    }
    None
}

/// Estimate the duration (seconds) of a WAV buffer from its header.
pub fn wav_duration_seconds(wav: &[u8]) -> Option<f64> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" {
        return None;
    }
    let (fmt_offset, fmt_len) = find_chunk(wav, b"fmt ")?;
    let (_, data_len) = find_chunk(wav, b"data")?;
    let chunk = &wav[fmt_offset..fmt_offset + fmt_len];
    if chunk.len() < 16 {
        return None;
    }
    let audio_format = u16::from_le_bytes([chunk[0], chunk[1]]);
    // 0x0001 = PCM, 0xFFFE = extensible (still PCM under the sub-format).
    if audio_format != 1 && audio_format != 0xfffe {
        return None;
    }
    let channels = u16::from_le_bytes([chunk[2], chunk[3]]) as f64;
    let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as f64;
    let bits = u16::from_le_bytes([chunk[14], chunk[15]]);
    let bytes_per_sample = (bits / 8).max(1) as f64;
    if sample_rate <= 0.0 || channels <= 0.0 {
        return None;
    }
    Some(data_len as f64 / (sample_rate * channels * bytes_per_sample))
}

/// Estimate the duration (seconds) of an audio buffer given its format,
/// byte length, sample rate, and channels. Compressed formats use a rough
/// typical-bitrate estimate because duration cannot be derived from length
/// alone.
pub fn estimate_duration_seconds(
    format: AudioFormat,
    bytes: usize,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
) -> Option<f64> {
    match format {
        AudioFormat::Wav | AudioFormat::Pcm => {
            let bytes_per_sample = ((bits_per_sample as usize).max(8)) / 8;
            let bytes_per_frame = bytes_per_sample * channels as usize;
            if bytes_per_frame == 0 {
                return None;
            }
            Some(bytes as f64 / (bytes_per_frame as f64 * sample_rate as f64))
        }
        AudioFormat::Mp3 => Some(bytes as f64 / (128_000.0 / 8.0)),
        AudioFormat::Opus => Some(bytes as f64 / (64_000.0 / 8.0)),
        AudioFormat::Aac => Some(bytes as f64 / (128_000.0 / 8.0)),
        AudioFormat::Flac => Some(bytes as f64 / (882_000.0 / 8.0)),
    }
}

/// Estimate the duration of a synthesized [`TtsResponse`].
fn estimate_tts_duration(response: &TtsResponse) -> f64 {
    match response.format {
        AudioFormat::Wav => wav_duration_seconds(&response.audio).unwrap_or(0.0),
        _ => estimate_duration_seconds(response.format, response.audio.len(), 44_100, 2, 16)
            .unwrap_or(0.0),
    }
}

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

/// Estimate the cost in USD of a TTS request from character count.
pub fn estimate_tts_cost(model: &str, text_len_chars: usize) -> Option<f64> {
    let chars_per_1k = text_len_chars as f64 / 1000.0;
    let m = model.to_ascii_lowercase();
    if m.contains("tts-1-hd") {
        Some(0.030 * chars_per_1k)
    } else if m.contains("tts-1") {
        Some(0.015 * chars_per_1k)
    } else if m.contains("eleven") {
        Some(0.30 * chars_per_1k)
    } else {
        None
    }
}

/// Estimate the cost in USD of a transcription request from audio duration.
pub fn estimate_stt_cost(duration_seconds: f64) -> f64 {
    duration_seconds / 60.0 * 0.006
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn form_has_field(form: &Form, name: &str) -> bool {
        form.fields().iter().any(|(n, _)| n.as_ref() == name)
    }

    fn make_wav(data_len: u32, sample_rate: u32, channels: u16, bits: u16) -> Vec<u8> {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * channels as u32 * bits as u32 / 8;
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        wav.extend_from_slice(&bits.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(&vec![0u8; data_len as usize]);
        wav
    }

    // --- AudioFormat -------------------------------------------------------

    #[test]
    fn audio_format_extensions() {
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Opus.extension(), "opus");
        assert_eq!(AudioFormat::Aac.extension(), "aac");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Pcm.extension(), "pcm");
    }

    #[test]
    fn audio_format_mime_types() {
        assert_eq!(AudioFormat::Mp3.mime_type(), "audio/mpeg");
        assert_eq!(AudioFormat::Wav.mime_type(), "audio/wav");
        assert_eq!(AudioFormat::Opus.mime_type(), "audio/opus");
        assert_eq!(AudioFormat::Pcm.mime_type(), "audio/L16");
    }

    #[test]
    fn audio_format_openai_strings() {
        assert_eq!(AudioFormat::Opus.openai_format(), "opus");
        assert_eq!(AudioFormat::Aac.openai_format(), "aac");
        assert_eq!(AudioFormat::Pcm.openai_format(), "pcm");
    }

    #[test]
    fn audio_format_elevenlabs_strings() {
        assert_eq!(AudioFormat::Mp3.elevenlabs_format(), "mp3_44100_128");
        assert_eq!(AudioFormat::Wav.elevenlabs_format(), "pcm_44100");
    }

    #[test]
    fn elevenlabs_format_support() {
        assert!(AudioFormat::Mp3.is_elevenlabs_supported());
        assert!(!AudioFormat::Opus.is_elevenlabs_supported());
        assert!(!AudioFormat::Aac.is_elevenlabs_supported());
    }

    // --- Provider type / config -------------------------------------------

    #[test]
    fn provider_type_roundtrip() {
        assert_eq!(
            AudioProviderType::from_str("elevenlabs"),
            Some(AudioProviderType::ElevenLabs)
        );
        assert_eq!(
            AudioProviderType::from_str("whisper"),
            Some(AudioProviderType::OpenAiStt)
        );
        assert_eq!(
            AudioProviderType::from_str("local"),
            Some(AudioProviderType::Local)
        );
        assert_eq!(AudioProviderType::from_str("bogus"), None);
        assert_eq!(AudioProviderType::OpenAiTts.as_str(), "openai-tts");
    }

    #[test]
    fn audio_config_defaults() {
        let cfg = AudioConfig::new(AudioProviderType::ElevenLabs, "xi-test");
        assert_eq!(cfg.base_url, "https://api.elevenlabs.io");
        assert_eq!(cfg.default_voice, "Rachel");
        assert_eq!(cfg.default_model, "eleven_multilingual_v2");

        let openai = AudioConfig::new(AudioProviderType::OpenAiTts, "sk-test");
        assert_eq!(openai.default_voice, "alloy");
        assert_eq!(openai.default_model, "tts-1");
    }

    #[test]
    fn provider_builder_methods() {
        let p = AudioProvider::openai_tts("sk-test")
            .with_voice("nova")
            .with_model("tts-1-hd")
            .with_base_url("https://example.com/v1");
        assert_eq!(p.config().default_voice, "nova");
        assert_eq!(p.config().default_model, "tts-1-hd");
        assert_eq!(p.config().base_url, "https://example.com/v1");
        assert_eq!(p.name(), "openai-tts");
    }

    // --- Request structs ---------------------------------------------------

    #[test]
    fn tts_request_default() {
        let req = TtsRequest::default();
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.voice, "alloy");
        assert_eq!(req.format, AudioFormat::Mp3);
        assert_eq!(req.speed, 1.0);
        assert!(req.stability.is_none());
    }

    #[test]
    fn tts_request_serialize() {
        let req = TtsRequest {
            text: "hello".into(),
            model: "tts-1-hd".into(),
            voice: "nova".into(),
            format: AudioFormat::Wav,
            speed: 1.5,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            language_code: None,
        };
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["model"], "tts-1-hd");
        assert_eq!(value["voice"], "nova");
        assert_eq!(value["format"], "wav");
        assert_eq!(value["speed"], 1.5);
    }

    #[test]
    fn tts_params_to_request_resolves_defaults() {
        let cfg = AudioConfig::new(AudioProviderType::OpenAiTts, "sk-test");
        let params = TtsParams {
            text: "hi".into(),
            ..TtsParams::default()
        };
        let req = params.to_request(&cfg);
        assert_eq!(req.model, "tts-1");
        assert_eq!(req.voice, "alloy");
        assert_eq!(req.format, AudioFormat::Mp3);

        let params = TtsParams {
            text: "hi".into(),
            voice: Some("echo".into()),
            model: Some("tts-1-hd".into()),
            format: Some(AudioFormat::Opus),
            ..TtsParams::default()
        };
        let req = params.to_request(&cfg);
        assert_eq!(req.voice, "echo");
        assert_eq!(req.model, "tts-1-hd");
        assert_eq!(req.format, AudioFormat::Opus);
    }

    // --- Request builders --------------------------------------------------

    #[test]
    fn build_openai_tts_body_basic() {
        let req = TtsRequest {
            text: "hello".into(),
            model: "tts-1".into(),
            voice: "alloy".into(),
            format: AudioFormat::Mp3,
            speed: 1.25,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            language_code: None,
        };
        let body = build_openai_tts_body(&req);
        assert_eq!(body["model"], "tts-1");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "mp3");
        assert_eq!(body["speed"], 1.25);
    }

    #[test]
    fn build_elevenlabs_tts_body_basic() {
        let req = TtsRequest {
            text: "hi".into(),
            model: "eleven_multilingual_v2".into(),
            voice: "Rachel".into(),
            format: AudioFormat::Mp3,
            speed: 1.0,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            language_code: None,
        };
        let body = build_elevenlabs_tts_body(&req);
        assert_eq!(body["text"], "hi");
        assert_eq!(body["model_id"], "eleven_multilingual_v2");
        assert!(body.get("voice_settings").is_none());
        assert!(body.get("language_code").is_none());
    }

    #[test]
    fn build_elevenlabs_tts_body_with_settings() {
        let req = TtsRequest {
            text: "hi".into(),
            model: "eleven_multilingual_v2".into(),
            voice: "Rachel".into(),
            format: AudioFormat::Mp3,
            speed: 1.0,
            stability: Some(0.7),
            similarity_boost: Some(0.8),
            style: Some(0.1),
            use_speaker_boost: Some(true),
            language_code: Some("en".into()),
        };
        let body = build_elevenlabs_tts_body(&req);
        assert_eq!(body["language_code"], "en");
        let settings = &body["voice_settings"];
        assert_eq!(settings["stability"], 0.7);
        assert_eq!(settings["similarity_boost"], 0.8);
        assert_eq!(settings["style"], 0.1);
        assert_eq!(settings["use_speaker_boost"], true);
    }

    // --- Response parsers --------------------------------------------------

    #[test]
    fn parse_openai_stt_json_verbose() {
        let text = json!({
            "text": "hello world",
            "language": "en",
            "duration": 2.0,
            "segments": [
                { "start": 0.0, "end": 1.0, "text": "hello" },
                { "start": 1.0, "end": 2.0, "text": "world" }
            ]
        })
        .to_string();
        let result = parse_openai_stt_json(&text, "whisper-1").unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.duration_seconds, 2.0);
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].text, "hello");
        assert_eq!(result.segments[0].start, 0.0);
    }

    #[test]
    fn parse_openai_stt_json_no_segments() {
        let text = json!({ "text": "plain" }).to_string();
        let result = parse_openai_stt_json(&text, "whisper-1").unwrap();
        assert_eq!(result.text, "plain");
        assert!(result.segments.is_empty());
    }

    #[test]
    fn parse_openai_stt_json_missing_text_errors() {
        assert!(parse_openai_stt_json(r#"{"foo":1}"#, "whisper-1").is_err());
    }

    #[test]
    fn parse_elevenlabs_stt_json() {
        let text = json!({
            "text": "hi there",
            "language_code": "en",
            "language_probability": 0.99,
            "words": [{"word": "hi"}, {"word": "there"}]
        })
        .to_string();
        let result = parse_elevenlabs_stt_json(&text, "scribe_v2").unwrap();
        assert_eq!(result.text, "hi there");
        assert_eq!(result.language.as_deref(), Some("en"));
        assert_eq!(result.language_probability, Some(0.99));
        // Non-string word objects are skipped by the string collector.
        assert!(result.words.is_empty());
    }

    #[test]
    fn parse_elevenlabs_voices_json() {
        let text = json!({
            "voices": [
                { "voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade", "description": "calm" },
                { "voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew", "preview_url": "https://x/preview.mp3" }
            ]
        })
        .to_string();
        let voices = parse_elevenlabs_voices_json(&text).unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].id, "21m00Tcm4TlvDq8ikWAM");
        assert_eq!(voices[0].name, "Rachel");
        assert_eq!(voices[0].category.as_deref(), Some("premade"));
        assert_eq!(
            voices[1].preview_url.as_deref(),
            Some("https://x/preview.mp3")
        );
    }

    #[test]
    fn parse_elevenlabs_voice_settings_json() {
        let text = json!({
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.1,
            "use_speaker_boost": true,
            "speed": 1.0
        })
        .to_string();
        let settings = parse_elevenlabs_voice_settings_json(&text).unwrap();
        assert_eq!(settings.stability, Some(0.5));
        assert_eq!(settings.similarity_boost, Some(0.75));
        assert_eq!(settings.use_speaker_boost, Some(true));
    }

    #[test]
    fn default_openai_voices_list() {
        let voices = default_openai_voices();
        assert!(voices.iter().any(|v| v.name == "alloy"));
        assert!(voices.iter().any(|v| v.name == "nova"));
        assert!(voices.iter().all(|v| v.category.is_none()));
    }

    // --- Form builders (via send-side helpers) -----------------------------

    #[test]
    fn stt_openai_form_has_expected_fields() {
        let file_part = Part::bytes(vec![0u8; 8])
            .file_name("clip.wav")
            .mime_str("audio/wav")
            .unwrap();
        let form = Form::new()
            .text("model", "whisper-1")
            .text("response_format", "verbose_json")
            .text("language", "en")
            .part("file", file_part);
        assert!(form_has_field(&form, "model"));
        assert!(form_has_field(&form, "file"));
        assert!(form_has_field(&form, "language"));
    }

    // --- WAV helpers -------------------------------------------------------

    #[test]
    fn pcm_to_wav_header() {
        let pcm = vec![0u8; 1600]; // 100ms at 16kHz mono 16-bit
        let wav = pcm_to_wav(&pcm, 16_000, 1, 16);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(
            u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]),
            1600
        );
        assert_eq!(wav.len(), 1644);
    }

    #[test]
    fn wav_to_pcm_roundtrip() {
        let wav = make_wav(1600, 16_000, 1, 16);
        let pcm = wav_to_pcm(&wav).unwrap();
        assert_eq!(pcm.len(), 1600);
    }

    #[test]
    fn wav_to_pcm_rejects_non_wav() {
        assert!(wav_to_pcm(b"nope").is_none());
    }

    #[test]
    fn wav_duration_parses_header() {
        // 1600 bytes, 16kHz, mono, 16-bit => 1600 / (16000 * 1 * 2) = 0.05s
        let wav = make_wav(1600, 16_000, 1, 16);
        let duration = wav_duration_seconds(&wav).unwrap();
        assert!((duration - 0.05).abs() < 0.001);
    }

    #[test]
    fn wav_duration_rejects_non_wav() {
        assert!(wav_duration_seconds(b"junk").is_none());
    }

    #[test]
    fn estimate_duration_wav() {
        let d = estimate_duration_seconds(AudioFormat::Wav, 1600, 16_000, 1, 16).unwrap();
        assert!((d - 0.05).abs() < 0.001);
    }

    #[test]
    fn estimate_duration_mp3() {
        let d = estimate_duration_seconds(AudioFormat::Mp3, 16_000, 44_100, 2, 16).unwrap();
        assert!(d > 0.0);
    }

    #[test]
    fn estimate_duration_pcm() {
        // 44100 * 2 * 2 = 176400 bytes/sec => 88200 bytes = 0.5s
        let d = estimate_duration_seconds(AudioFormat::Pcm, 88_200, 44_100, 2, 16).unwrap();
        assert!((d - 0.5).abs() < 0.001);
    }

    // --- Cost estimation ---------------------------------------------------

    #[test]
    fn tts_cost_openai() {
        assert_eq!(estimate_tts_cost("tts-1", 1000).unwrap(), 0.015);
        assert_eq!(estimate_tts_cost("tts-1-hd", 1000).unwrap(), 0.030);
        assert_eq!(estimate_tts_cost("tts-1", 2000).unwrap(), 0.030);
    }

    #[test]
    fn tts_cost_elevenlabs_and_unknown() {
        assert!(estimate_tts_cost("eleven_multilingual_v2", 1000).is_some());
        assert_eq!(estimate_tts_cost("custom-model", 1000), None);
    }

    #[test]
    fn stt_cost_scales_with_duration() {
        assert_eq!(estimate_stt_cost(60.0), 0.006);
        assert_eq!(estimate_stt_cost(120.0), 0.012);
    }
}
