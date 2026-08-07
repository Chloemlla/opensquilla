//! Image generation provider.
//!
//! Makes raw `reqwest` HTTP calls to image generation APIs across five backend
//! families and handles base64 / URL image responses. No SDK dependency —
//! matching the Python backend's approach.
//!
//! # Supported backends
//!
//! * **OpenAI / DALL-E** — `POST /v1/images/generations` (dall-e-2, dall-e-3,
//!   gpt-image-1) and `POST /v1/images/edits` for variations and inpainting.
//! * **Stability AI** — `POST /v2beta/stable-image/generate/{model}` (Stable
//!   Diffusion 3) and `/v2beta/stable-image/edit/inpaint` for inpainting.
//! * **Replicate** — `POST /v1/predictions` with output polling.
//! * **Prodia** — `POST /v1/sd/generate` with job polling.
//! * **Local** — a generic OpenAI-compatible endpoint served from
//!   `base_url` (e.g. a local inference server).
//!
//! Every backend dispatches through a unified entry point
//! [`ImageGenerationProvider::generate_image`], which applies retry logic,
//! optional rate limiting, content-safety checks, and per-model cost
//! estimation before returning an [`ImageGenerationResult`].

use crate::types::{ProviderError, ProviderResult};
use crate::util::{RateLimiter, RetryConfig, check_status, with_retry};
use futures::StreamExt;
use reqwest::Client;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

/// The image generation backend family a provider targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenProvider {
    /// OpenAI / DALL-E / gpt-image-1 (`/v1/images/generations`).
    OpenAi,
    /// Stability AI Stable Diffusion (`/v2beta/stable-image/generate`).
    StabilityAi,
    /// Replicate prediction API.
    Replicate,
    /// Prodia Stable Diffusion job API.
    Prodia,
    /// A generic OpenAI-compatible endpoint (usually local).
    Local,
}

impl ImageGenProvider {
    /// The stable string name for this backend.
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageGenProvider::OpenAi => "openai",
            ImageGenProvider::StabilityAi => "stability",
            ImageGenProvider::Replicate => "replicate",
            ImageGenProvider::Prodia => "prodia",
            ImageGenProvider::Local => "local",
        }
    }

    /// Parse a backend name back into an [`ImageGenProvider`].
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai" | "dall-e" | "dalle" => Some(ImageGenProvider::OpenAi),
            "stability" | "stabilityai" | "stability_ai" | "stable-diffusion" | "sd3" => {
                Some(ImageGenProvider::StabilityAi)
            }
            "replicate" => Some(ImageGenProvider::Replicate),
            "prodia" => Some(ImageGenProvider::Prodia),
            "local" => Some(ImageGenProvider::Local),
            _ => None,
        }
    }

    /// The default base URL for this backend.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            ImageGenProvider::OpenAi => "https://api.openai.com/v1",
            ImageGenProvider::StabilityAi => "https://api.stability.ai",
            ImageGenProvider::Replicate => "https://api.replicate.com/v1",
            ImageGenProvider::Prodia => "https://api.prodia.com",
            ImageGenProvider::Local => "http://localhost:8000/v1",
        }
    }

    /// A sensible default model name for this backend.
    pub fn default_model(&self) -> &'static str {
        match self {
            ImageGenProvider::OpenAi => "gpt-image-1",
            ImageGenProvider::StabilityAi => "sd3-large",
            ImageGenProvider::Replicate => "black-forest-labs/flux-1.1-pro",
            ImageGenProvider::Prodia => "3Guofeng3_v34.safetensors",
            ImageGenProvider::Local => "local",
        }
    }
}

/// Requested output dimensions for generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSize {
    /// 1024x1024.
    #[default]
    Square,
    /// 1792x1024 (DALL-E 3 wide).
    Wide,
    /// 1024x1792 (DALL-E 3 tall).
    Tall,
    /// Arbitrary width x height in pixels.
    Custom(u32, u32),
}

impl ImageSize {
    /// The pixel dimensions `(width, height)` for this size.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            ImageSize::Square => (1024, 1024),
            ImageSize::Wide => (1792, 1024),
            ImageSize::Tall => (1024, 1792),
            ImageSize::Custom(w, h) => (*w, *h),
        }
    }

    /// The reduced aspect ratio as `"W:H"` (OpenRouter / chat-completions
    /// image models).
    pub fn aspect_ratio(&self) -> String {
        let (w, h) = self.dimensions();
        let g = gcd(w, h);
        format!("{}:{}", w / g, h / g)
    }

    /// The reduced aspect ratio as an OpenRouter-style token such as `"1:1"`.
    pub fn aspect_ratio_token(&self) -> String {
        self.aspect_ratio()
    }
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (w, h) = self.dimensions();
        write!(f, "{w}x{h}")
    }
}

impl From<(u32, u32)> for ImageSize {
    /// Map canonical dimensions onto the named variants, otherwise a custom size.
    fn from(dims: (u32, u32)) -> Self {
        match dims {
            (1024, 1024) => ImageSize::Square,
            (1792, 1024) => ImageSize::Wide,
            (1024, 1792) => ImageSize::Tall,
            (w, h) => ImageSize::Custom(w, h),
        }
    }
}

impl From<ImageSize> for (u32, u32) {
    fn from(size: ImageSize) -> Self {
        size.dimensions()
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Output quality tier. Maps to DALL-E 3 `standard`/`hd` and to
/// gpt-image-1 `medium`/`high` depending on the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    /// Standard quality.
    #[default]
    Standard,
    /// HD / high quality.
    Hd,
}

impl ImageQuality {
    /// The DALL-E 3 quality string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageQuality::Standard => "standard",
            ImageQuality::Hd => "hd",
        }
    }
}

/// The desired response format for generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    /// Return a URL pointing to the generated image.
    Url,
    /// Return the image as a base64-encoded JSON string.
    #[default]
    B64Json,
}

/// A request to generate one or more images (low-level form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    /// The text prompt describing the desired image.
    pub prompt: String,
    /// The model to use (e.g. "dall-e-3", "stable-image-core").
    pub model: String,
    /// Number of images to generate (1-10, depending on model).
    #[serde(default = "default_n")]
    pub n: u32,
    /// Square size in pixels (e.g. 1024).
    #[serde(default = "default_size")]
    pub size: u32,
    /// Response format: URL or base64.
    #[serde(default)]
    pub response_format: ImageResponseFormat,
    /// Optional quality hint (e.g. "standard", "hd").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
    /// Optional style hint (e.g. "vivid", "natural").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<String>,
}

fn default_n() -> u32 {
    1
}
fn default_size() -> u32 {
    1024
}

impl Default for ImageGenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            model: "dall-e-3".into(),
            n: 1,
            size: 1024,
            response_format: ImageResponseFormat::B64Json,
            quality: None,
            style: None,
        }
    }
}

/// A single generated image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// Base64-encoded image data (present when `response_format` is `B64Json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// A URL to the generated image (present when `response_format` is `Url`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The revised prompt (DALL-E 3 may rewrite the prompt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    /// The deterministic seed, when the backend reports one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl GeneratedImage {
    /// Returns the raw bytes of the image, decoding base64 if necessary.
    pub fn bytes(&self) -> Option<Vec<u8>> {
        self.b64_json.as_deref().and_then(|s| decode_base64(s).ok())
    }

    /// Convert into a high-level [`ImageGenerationResult`].
    pub fn into_result(self, _model: String, cost: Option<f64>) -> ImageGenerationResult {
        ImageGenerationResult {
            url: self.url.unwrap_or_default(),
            b64_json: self.b64_json,
            revised_prompt: self.revised_prompt,
            seed: self.seed,
            cost,
        }
    }
}

/// The response from an image generation request (low-level form).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    /// The generated images.
    pub images: Vec<GeneratedImage>,
    /// The model that produced the images.
    pub model: String,
    /// Estimated cost in USD for the request, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

/// Parameters for a high-level [`ImageGenerationProvider::generate_image`]
/// call. `None` fields fall back to the provider's [`ImageGenConfig`] defaults.
#[derive(Debug, Clone)]
pub struct ImageGenerationParams {
    /// Model override (e.g. "dall-e-3", "sd3-large").
    pub model: Option<String>,
    /// Output size override.
    pub size: Option<ImageSize>,
    /// Output quality override.
    pub quality: Option<ImageQuality>,
    /// Style hint (OpenAI: "vivid" / "natural").
    pub style: Option<String>,
    /// Number of images to generate.
    pub n: u32,
    /// Deterministic seed (backend-dependent).
    pub seed: Option<u64>,
    /// Negative prompt (Stability / Prodia / Replicate).
    pub negative_prompt: Option<String>,
    /// Output format ("png", "jpeg", "webp").
    pub output_format: Option<String>,
    /// Timeout hint for the request. The effective client timeout is governed
    /// by `ImageGenConfig::timeout`.
    pub timeout_seconds: f64,
    /// Explicit cost override in USD.
    pub cost: Option<f64>,
}

impl Default for ImageGenerationParams {
    fn default() -> Self {
        Self {
            model: None,
            size: None,
            quality: None,
            style: None,
            n: 1,
            seed: None,
            negative_prompt: None,
            output_format: Some("png".into()),
            timeout_seconds: 180.0,
            cost: None,
        }
    }
}

/// A high-level image generation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResult {
    /// A URL to the image. Empty when only `b64_json` is present.
    pub url: String,
    /// Base64-encoded image data, when returned inline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    /// The revised prompt (DALL-E 3 may rewrite the prompt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    /// The deterministic seed, when the backend reports one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Estimated cost in USD for the request, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

impl ImageGenerationResult {
    /// Decode the inline base64 payload into bytes.
    pub fn bytes(&self) -> Option<Vec<u8>> {
        self.b64_json.as_deref().and_then(|s| decode_base64(s).ok())
    }

    /// A `data:` URL synthesized from the inline base64 payload.
    pub fn data_url(&self) -> Option<String> {
        self.b64_json
            .as_deref()
            .map(|b| format!("data:image/png;base64,{b}"))
    }

    /// Whether the result carries neither bytes nor a URL.
    pub fn is_empty(&self) -> bool {
        self.url.is_empty() && self.b64_json.is_none()
    }

    /// Resolve the image bytes, downloading the URL when no base64 payload is
    /// present.
    pub async fn fetch_bytes(&self, client: &Client) -> ProviderResult<Vec<u8>> {
        if let Some(bytes) = self.bytes() {
            return Ok(bytes);
        }
        if self.url.is_empty() {
            return Err(ProviderError::Provider(
                "Image generation result contains neither bytes nor a URL".into(),
            ));
        }
        let (_mime, bytes) = download_generated_image(client, &self.url).await?;
        Ok(bytes)
    }
}

/// Configuration for an image generation provider.
#[derive(Debug, Clone)]
pub struct ImageGenConfig {
    /// Which backend family to call.
    pub provider: ImageGenProvider,
    /// API key / bearer token.
    pub api_key: String,
    /// Base URL. Paths are appended with the backend's well-known layout.
    pub base_url: String,
    /// Default model used when a request does not specify one.
    pub default_model: String,
    /// Default output size.
    pub default_size: ImageSize,
    /// Default quality.
    pub default_quality: ImageQuality,
    /// HTTP timeout for each request.
    pub timeout: Duration,
    /// Retry policy for transient failures.
    pub retry: RetryConfig,
    /// Explicit per-image cost override in USD, when known.
    pub cost_override: Option<f64>,
}

impl ImageGenConfig {
    /// Create a config with sensible defaults for the given backend.
    pub fn new(provider: ImageGenProvider, api_key: impl Into<String>) -> Self {
        Self {
            provider,
            api_key: api_key.into(),
            base_url: provider.default_base_url().to_string(),
            default_model: provider.default_model().to_string(),
            default_size: ImageSize::Square,
            default_quality: ImageQuality::Standard,
            timeout: Duration::from_secs(180),
            retry: RetryConfig::default(),
            cost_override: None,
        }
    }
}

/// Provider for image generation APIs.
#[derive(Clone)]
pub struct ImageGenerationProvider {
    client: Client,
    config: ImageGenConfig,
    limiter: Option<RateLimiter>,
}

impl ImageGenerationProvider {
    /// Create a provider from an explicit config.
    pub fn new(config: ImageGenConfig) -> Self {
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

    /// Create an OpenAI / DALL-E provider targeting OpenAI's API.
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self::new(ImageGenConfig::new(ImageGenProvider::OpenAi, api_key))
    }

    /// Create a Stability AI provider.
    pub fn stability_ai(api_key: impl Into<String>) -> Self {
        Self::new(ImageGenConfig::new(ImageGenProvider::StabilityAi, api_key))
    }

    /// Create a Replicate provider.
    pub fn replicate(api_key: impl Into<String>) -> Self {
        Self::new(ImageGenConfig::new(ImageGenProvider::Replicate, api_key))
    }

    /// Create a Prodia provider.
    pub fn prodia(api_key: impl Into<String>) -> Self {
        Self::new(ImageGenConfig::new(ImageGenProvider::Prodia, api_key))
    }

    /// Create a provider for a local OpenAI-compatible endpoint.
    pub fn local(base_url: impl Into<String>) -> Self {
        let mut config = ImageGenConfig::new(ImageGenProvider::Local, "");
        config.base_url = base_url.into();
        Self::new(config)
    }

    /// Backward-compatible alias for [`Self::openai`].
    pub fn dall_e(api_key: impl Into<String>) -> Self {
        Self::openai(api_key)
    }

    /// Backward-compatible alias for [`Self::stability_ai`].
    pub fn stable_diffusion(api_key: impl Into<String>) -> Self {
        Self::stability_ai(api_key)
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

    /// Override the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.config.default_model = model.into();
        self
    }

    /// Override the default size.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.config.default_size = size;
        self
    }

    /// Override the default quality.
    pub fn with_quality(mut self, quality: ImageQuality) -> Self {
        self.config.default_quality = quality;
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

    /// Set an explicit per-image cost override.
    pub fn with_cost_override(mut self, cost: f64) -> Self {
        self.config.cost_override = Some(cost);
        self
    }

    /// The provider name (e.g. "openai", "stability").
    pub fn name(&self) -> &str {
        self.config.provider.as_str()
    }

    /// The provider configuration.
    pub fn config(&self) -> &ImageGenConfig {
        &self.config
    }

    /// The effective retry policy for this provider.
    pub fn retry_config(&self) -> RetryConfig {
        self.config.retry
    }

    /// Estimate the cost (USD) of a prospective [`generate_image`] call
    /// without sending any request.
    ///
    /// [`generate_image`]: Self::generate_image
    pub fn estimate_cost_for(&self, params: &ImageGenerationParams) -> Option<f64> {
        if let Some(cost) = self.config.cost_override {
            return Some(cost);
        }
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let size = params.size.unwrap_or(self.config.default_size);
        let quality = params.quality.unwrap_or(self.config.default_quality);
        estimate_image_cost(model, size, quality, params.n.max(1))
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

    /// Generate one or more images from a text prompt.
    ///
    /// This is the main entry point. It resolves `params` against the
    /// provider config, applies the rate limiter, retries transient failures,
    /// validates the response, and estimates cost.
    pub async fn generate_image(
        &self,
        prompt: &str,
        params: &ImageGenerationParams,
    ) -> ProviderResult<ImageGenerationResult> {
        debug!(
            target = "provider",
            provider = %self.name(),
            model = %params.model.as_deref().unwrap_or(&self.config.default_model),
            "Generating image"
        );

        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let size = params.size.unwrap_or(self.config.default_size);
        let quality = params.quality.unwrap_or(self.config.default_quality);

        let response = self
            .generate_inner(
                prompt,
                model,
                size,
                quality,
                params.style.as_deref(),
                params.n.max(1),
                params.seed,
                params.negative_prompt.as_deref(),
                params.output_format.as_deref(),
                ImageResponseFormat::B64Json,
            )
            .await?;

        let cost = params.cost.or(self.config.cost_override).or(response.cost);
        let first =
            response.images.into_iter().next().ok_or_else(|| {
                ProviderError::Provider("Image generation returned no images".into())
            })?;

        info!(
            target = "provider",
            provider = %self.name(),
            model = model,
            "Image generated"
        );
        Ok(first.into_result(model.to_string(), cost))
    }

    /// Generate a variation of an existing image.
    ///
    /// OpenAI and Stability use dedicated endpoints; Replicate runs an
    /// img2img-style prediction with the source image passed as a `data:`
    /// URL. Prodia does not support variations.
    pub async fn generate_image_variation(
        &self,
        image: &[u8],
        prompt: &str,
    ) -> ProviderResult<ImageGenerationResult> {
        let params = ImageGenerationParams::default();
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let size = params.size.unwrap_or(self.config.default_size);
        let quality = params.quality.unwrap_or(self.config.default_quality);
        let n = params.n.max(1);

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let response = match self.config.provider {
            ImageGenProvider::OpenAi | ImageGenProvider::Local => {
                with_retry(&self.config.retry, |_attempt| {
                    self.variation_openai(image, prompt, model, size, quality, n)
                })
                .await?
            }
            ImageGenProvider::StabilityAi => {
                with_retry(&self.config.retry, |_attempt| {
                    self.variation_stability(image, prompt, model, size, n)
                })
                .await?
            }
            ImageGenProvider::Replicate => {
                with_retry(&self.config.retry, |_attempt| {
                    self.variation_replicate(image, prompt, model, size, n)
                })
                .await?
            }
            ImageGenProvider::Prodia => {
                return Err(ProviderError::Provider(
                    "Prodia does not support image variation".into(),
                ));
            }
        };

        let cost = self.config.cost_override.or(response.cost);
        let first =
            response.images.into_iter().next().ok_or_else(|| {
                ProviderError::Provider("Image generation returned no images".into())
            })?;
        Ok(first.into_result(model.to_string(), cost))
    }

    /// Edit (inpaint) an image. `mask` is optional for OpenAI; Stability
    /// requires it for `/edit/inpaint`.
    pub async fn edit_image(
        &self,
        image: &[u8],
        mask: Option<&[u8]>,
        prompt: &str,
    ) -> ProviderResult<ImageGenerationResult> {
        let params = ImageGenerationParams::default();
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let size = params.size.unwrap_or(self.config.default_size);
        let quality = params.quality.unwrap_or(self.config.default_quality);
        let n = params.n.max(1);

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let response = match self.config.provider {
            ImageGenProvider::OpenAi | ImageGenProvider::Local => {
                with_retry(&self.config.retry, |_attempt| {
                    self.edit_openai(image, mask, prompt, model, size, quality, n)
                })
                .await?
            }
            ImageGenProvider::StabilityAi => {
                with_retry(&self.config.retry, |_attempt| {
                    self.edit_stability(image, mask, prompt, model, size)
                })
                .await?
            }
            ImageGenProvider::Replicate => {
                with_retry(&self.config.retry, |_attempt| {
                    self.edit_replicate(image, mask, prompt, model, size, n)
                })
                .await?
            }
            ImageGenProvider::Prodia => {
                return Err(ProviderError::Provider(
                    "Prodia does not support image inpainting".into(),
                ));
            }
        };

        let cost = self.config.cost_override.or(response.cost);
        let first =
            response.images.into_iter().next().ok_or_else(|| {
                ProviderError::Provider("Image generation returned no images".into())
            })?;
        Ok(first.into_result(model.to_string(), cost))
    }

    /// Generate via the low-level request form.
    pub async fn generate(
        &self,
        request: &ImageGenerationRequest,
    ) -> ProviderResult<ImageGenerationResponse> {
        let quality = match request.quality.as_deref() {
            Some("hd") | Some("high") => ImageQuality::Hd,
            _ => ImageQuality::Standard,
        };
        let size = ImageSize::Custom(request.size, request.size);
        self.generate_inner(
            &request.prompt,
            &request.model,
            size,
            quality,
            request.style.as_deref(),
            request.n.max(1),
            None,
            None,
            None,
            request.response_format,
        )
        .await
    }

    /// Shared dispatch across every backend.
    #[allow(clippy::too_many_arguments)]
    async fn generate_inner(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
        quality: ImageQuality,
        style: Option<&str>,
        n: u32,
        seed: Option<u64>,
        negative_prompt: Option<&str>,
        output_format: Option<&str>,
        response_format: ImageResponseFormat,
    ) -> ProviderResult<ImageGenerationResponse> {
        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }
        match self.config.provider {
            ImageGenProvider::OpenAi | ImageGenProvider::Local => {
                with_retry(&self.config.retry, |_attempt| {
                    self.generate_openai(
                        prompt,
                        model,
                        size,
                        quality,
                        style,
                        n,
                        seed,
                        response_format,
                    )
                })
                .await
            }
            ImageGenProvider::StabilityAi => {
                with_retry(&self.config.retry, |_attempt| {
                    self.generate_stability(
                        prompt,
                        model,
                        size,
                        quality,
                        n,
                        seed,
                        negative_prompt,
                        output_format,
                    )
                })
                .await
            }
            ImageGenProvider::Replicate => {
                with_retry(&self.config.retry, |_attempt| {
                    self.generate_replicate(prompt, model, size, n, seed, negative_prompt)
                })
                .await
            }
            ImageGenProvider::Prodia => {
                with_retry(&self.config.retry, |_attempt| {
                    self.generate_prodia(
                        prompt,
                        model,
                        size,
                        n,
                        seed,
                        negative_prompt,
                        output_format,
                    )
                })
                .await
            }
        }
    }

    // --- OpenAI / DALL-E -------------------------------------------------

    async fn generate_openai(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
        quality: ImageQuality,
        style: Option<&str>,
        n: u32,
        seed: Option<u64>,
        response_format: ImageResponseFormat,
    ) -> ProviderResult<ImageGenerationResponse> {
        let body = build_openai_body(
            prompt,
            model,
            size,
            quality,
            style,
            n,
            seed,
            response_format,
        );
        let resp = self
            .client
            .post(self.endpoint("/images/generations"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        let mut response = parse_openai_json(&text, model)?;
        response.cost = estimate_image_cost(model, size, quality, n);
        Ok(response)
    }

    async fn variation_openai(
        &self,
        image: &[u8],
        prompt: &str,
        model: &str,
        size: ImageSize,
        quality: ImageQuality,
        n: u32,
    ) -> ProviderResult<ImageGenerationResponse> {
        let form = build_openai_edit_form(image, None, prompt, model, size, n)?;
        let resp = self
            .client
            .post(self.endpoint("/images/edits"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        let mut response = parse_openai_json(&text, model)?;
        response.cost = estimate_image_cost(model, size, quality, n);
        Ok(response)
    }

    async fn edit_openai(
        &self,
        image: &[u8],
        mask: Option<&[u8]>,
        prompt: &str,
        model: &str,
        size: ImageSize,
        quality: ImageQuality,
        n: u32,
    ) -> ProviderResult<ImageGenerationResponse> {
        let form = build_openai_edit_form(image, mask, prompt, model, size, n)?;
        let resp = self
            .client
            .post(self.endpoint("/images/edits"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        let mut response = parse_openai_json(&text, model)?;
        response.cost = estimate_image_cost(model, size, quality, n);
        Ok(response)
    }

    // --- Stability AI ----------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    async fn generate_stability(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
        quality: ImageQuality,
        n: u32,
        seed: Option<u64>,
        negative_prompt: Option<&str>,
        output_format: Option<&str>,
    ) -> ProviderResult<ImageGenerationResponse> {
        let form = build_stability_form(
            prompt,
            size,
            quality,
            n,
            seed,
            negative_prompt,
            output_format,
        )?;
        let url = format!(
            "{}/v2beta/stable-image/generate/{}",
            self.config.base_url.trim_end_matches('/'),
            model
        );
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        let mut response = parse_stability_json(&text, model)?;
        response.cost = estimate_image_cost(model, size, quality, n);
        Ok(response)
    }

    async fn variation_stability(
        &self,
        image: &[u8],
        prompt: &str,
        model: &str,
        size: ImageSize,
        n: u32,
    ) -> ProviderResult<ImageGenerationResponse> {
        let url = format!(
            "{}/v2beta/stable-image/generate/{}",
            self.config.base_url.trim_end_matches('/'),
            model
        );
        let (w, h) = size.dimensions();
        let image_part = Part::bytes(image.to_vec())
            .file_name("image.png")
            .mime_str("image/png")
            .map_err(ProviderError::Network)?;
        let form = Form::new()
            .text("prompt", prompt.to_string())
            .text("mode", "image-to-image")
            .text("output_format", "png")
            .text("width", w.to_string())
            .text("height", h.to_string())
            .text("n", n.to_string())
            .part("image", image_part);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        parse_stability_json(&text, model)
    }

    async fn edit_stability(
        &self,
        image: &[u8],
        mask: Option<&[u8]>,
        prompt: &str,
        model: &str,
        size: ImageSize,
    ) -> ProviderResult<ImageGenerationResponse> {
        let url = format!(
            "{}/v2beta/stable-image/edit/inpaint",
            self.config.base_url.trim_end_matches('/')
        );
        let (w, h) = size.dimensions();
        let image_part = Part::bytes(image.to_vec())
            .file_name("image.png")
            .mime_str("image/png")
            .map_err(ProviderError::Network)?;
        let mut form = Form::new()
            .text("prompt", prompt.to_string())
            .text("output_format", "png")
            .text("width", w.to_string())
            .text("height", h.to_string())
            .part("image", image_part);
        if let Some(mask) = mask {
            let mask_part = Part::bytes(mask.to_vec())
                .file_name("mask.png")
                .mime_str("image/png")
                .map_err(ProviderError::Network)?;
            form = form.part("mask", mask_part);
        }
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Accept", "application/json")
            .multipart(form)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        parse_stability_json(&text, model)
    }

    // --- Replicate -------------------------------------------------------

    async fn generate_replicate(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
        n: u32,
        seed: Option<u64>,
        negative_prompt: Option<&str>,
    ) -> ProviderResult<ImageGenerationResponse> {
        let input = build_replicate_input(prompt, size, n, seed, negative_prompt, None, None);
        self.replicate_prediction(model, input).await
    }

    async fn variation_replicate(
        &self,
        image: &[u8],
        prompt: &str,
        model: &str,
        size: ImageSize,
        n: u32,
    ) -> ProviderResult<ImageGenerationResponse> {
        let image_url = encode_data_url(image, "image/png");
        let input = build_replicate_input(prompt, size, n, None, None, Some(&image_url), None);
        self.replicate_prediction(model, input).await
    }

    async fn edit_replicate(
        &self,
        image: &[u8],
        mask: Option<&[u8]>,
        prompt: &str,
        model: &str,
        size: ImageSize,
        n: u32,
    ) -> ProviderResult<ImageGenerationResponse> {
        let image_url = encode_data_url(image, "image/png");
        let mask_url = mask.map(|m| encode_data_url(m, "image/png"));
        let input = build_replicate_input(
            prompt,
            size,
            n,
            None,
            None,
            Some(&image_url),
            mask_url.as_deref(),
        );
        self.replicate_prediction(model, input).await
    }

    /// Create a Replicate prediction and poll it to completion.
    async fn replicate_prediction(
        &self,
        model: &str,
        input: serde_json::Map<String, serde_json::Value>,
    ) -> ProviderResult<ImageGenerationResponse> {
        // Accept either a bare version hash ("r_...") or an owner/name model
        // reference ("stability-ai/sdxl").
        let body = if model.contains(':') {
            serde_json::json!({ "version": model, "input": input })
        } else {
            serde_json::json!({ "model": model, "input": input })
        };

        let resp = self
            .client
            .post(self.endpoint("/predictions"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .header("Prefer", "wait=60")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;

        let initial: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let prediction = self.poll_replicate(&initial).await?;

        let output_urls = extract_string_array(&prediction["output"]);
        if output_urls.is_empty() {
            return Err(ProviderError::Provider(
                "Replicate returned no image outputs".into(),
            ));
        }
        let seed = prediction.get("seed").and_then(|v| v.as_u64());
        let images = output_urls
            .into_iter()
            .map(|url| GeneratedImage {
                b64_json: None,
                url: Some(url),
                revised_prompt: None,
                seed,
            })
            .collect();
        Ok(ImageGenerationResponse {
            images,
            model: model.to_string(),
            cost: None,
        })
    }

    /// Poll a Replicate prediction until it reaches a terminal state.
    async fn poll_replicate(
        &self,
        initial: &serde_json::Value,
    ) -> ProviderResult<serde_json::Value> {
        let status = initial.get("status").and_then(|v| v.as_str()).unwrap_or("");
        if matches!(status, "succeeded" | "failed" | "canceled") {
            return Ok(initial.clone());
        }
        let id = initial
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProviderError::Provider("Replicate prediction returned no id".into()))?;
        let get_url = initial
            .get("urls")
            .and_then(|u| u.get("get"))
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| self.endpoint(&format!("/predictions/{id}")));

        for _ in 0..120 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let resp = self
                .client
                .get(&get_url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .send()
                .await
                .map_err(ProviderError::Network)?;
            let status = resp.status();
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "image generation poll")?;
            let polled: serde_json::Value =
                serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
            match polled.get("status").and_then(|v| v.as_str()).unwrap_or("") {
                "succeeded" => return Ok(polled),
                "failed" => {
                    let error = polled
                        .get("error")
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "unknown error".into());
                    return Err(ProviderError::Provider(format!(
                        "Replicate prediction failed: {error}"
                    )));
                }
                "canceled" => {
                    return Err(ProviderError::Provider(
                        "Replicate prediction was canceled".into(),
                    ));
                }
                _ => {}
            }
        }
        Err(ProviderError::Timeout(
            "Replicate prediction timed out".into(),
        ))
    }

    // --- Prodia ----------------------------------------------------------

    async fn generate_prodia(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
        n: u32,
        seed: Option<u64>,
        negative_prompt: Option<&str>,
        output_format: Option<&str>,
    ) -> ProviderResult<ImageGenerationResponse> {
        let body = build_prodia_body(prompt, model, size, n, seed, negative_prompt, output_format);
        let resp = self
            .client
            .post(self.endpoint("/v1/sd/generate"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;
        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let job = data
            .get("job")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ProviderError::Provider("Prodia returned no job id".into()))?;
        let image_url = self.poll_prodia(job).await?;
        let image = GeneratedImage {
            b64_json: None,
            url: Some(image_url),
            revised_prompt: None,
            seed,
        };
        Ok(ImageGenerationResponse {
            images: vec![image],
            model: model.to_string(),
            cost: None,
        })
    }

    /// Poll a Prodia job until it reaches a terminal state.
    async fn poll_prodia(&self, job: &str) -> ProviderResult<String> {
        let url = self.endpoint(&format!("/v1/job/{job}"));
        for _ in 0..120 {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let resp = self
                .client
                .get(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .send()
                .await
                .map_err(ProviderError::Network)?;
            let status = resp.status();
            let text = resp.text().await.map_err(ProviderError::Network)?;
            check_status(status, &text, "image generation poll")?;
            let data: serde_json::Value =
                serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
            match data.get("status").and_then(|v| v.as_str()).unwrap_or("") {
                "succeeded" => {
                    return data
                        .get("image_url")
                        .and_then(|v| v.as_str())
                        .map(String::from)
                        .ok_or_else(|| {
                            ProviderError::Provider(
                                "Prodia job succeeded but returned no image_url".into(),
                            )
                        });
                }
                "failed" => {
                    return Err(ProviderError::Provider("Prodia job failed".into()));
                }
                _ => {}
            }
        }
        Err(ProviderError::Timeout("Prodia job timed out".into()))
    }

    /// Generate an image through a chat-completions endpoint that reports
    /// images natively (e.g. OpenRouter image models).
    ///
    /// Uses `modalities: ["image", "text"]` and extracts the first image URL
    /// from the assistant message. The image may be returned inline as a
    /// `data:` URL or as a remote URL.
    pub async fn generate_via_chat_completions(
        &self,
        prompt: &str,
        params: &ImageGenerationParams,
    ) -> ProviderResult<ImageGenerationResult> {
        let model = params
            .model
            .as_deref()
            .unwrap_or(&self.config.default_model);
        let size = params.size.unwrap_or(self.config.default_size);
        let body = build_chat_completions_image_body(prompt, model, size);

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let resp = self
            .client
            .post(self.endpoint("/chat/completions"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;

        let mut response = parse_chat_completions_image(&text, model)?;
        response.cost = self.config.cost_override.or(estimate_image_cost(
            model,
            size,
            self.config.default_quality,
            params.n.max(1),
        ));
        let first =
            response.images.into_iter().next().ok_or_else(|| {
                ProviderError::Provider("Image generation returned no images".into())
            })?;
        Ok(first.into_result(model.to_string(), response.cost))
    }

    /// Generate an image through the Qwen Token Plan multimodal-generation
    /// endpoint (`/services/aigc/multimodal-generation/generation`).
    pub async fn generate_qwen_token_plan(
        &self,
        prompt: &str,
        model: &str,
        size: ImageSize,
    ) -> ProviderResult<ImageGenerationResult> {
        let body = build_qwen_token_plan_body(prompt, model, size);

        if let Some(limiter) = &self.limiter {
            limiter.acquire().await?;
        }

        let resp = self
            .client
            .post(self.endpoint("/services/aigc/multimodal-generation/generation"))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Network)?;
        let status = resp.status();
        let text = resp.text().await.map_err(ProviderError::Network)?;
        check_status(status, &text, "image generation")?;

        let data: serde_json::Value =
            serde_json::from_str(&text).map_err(ProviderError::Serialization)?;
        let image_url = extract_qwen_token_plan_image_url(&data).ok_or_else(|| {
            ProviderError::Provider("Image generation provider returned no images".into())
        })?;
        let (_mime, image_bytes) = download_generated_image(&self.client, &image_url).await?;
        Ok(ImageGenerationResult {
            url: image_url,
            b64_json: Some(encode_base64(&image_bytes)),
            revised_prompt: None,
            seed: None,
            cost: self.config.cost_override,
        })
    }
}

// ---------------------------------------------------------------------------
// Pure request builders (unit-testable without a network)
// ---------------------------------------------------------------------------

/// Build the OpenAI `/images/generations` JSON body.
#[allow(clippy::too_many_arguments)]
pub fn build_openai_body(
    prompt: &str,
    model: &str,
    size: ImageSize,
    quality: ImageQuality,
    style: Option<&str>,
    n: u32,
    seed: Option<u64>,
    response_format: ImageResponseFormat,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "size": size.to_string(),
        "n": n,
        "response_format": match response_format {
            ImageResponseFormat::Url => "url",
            ImageResponseFormat::B64Json => "b64_json",
        },
        "quality": openai_quality_string(model, quality),
    });
    if let Some(style) = style {
        body["style"] = serde_json::json!(style);
    }
    if let Some(seed) = seed {
        body["seed"] = serde_json::json!(seed);
    }
    body
}

/// The quality string a given OpenAI model accepts.
///
/// gpt-image-1 uses `low`/`medium`/`high`; DALL-E 3 uses `standard`/`hd`.
pub fn openai_quality_string(model: &str, quality: ImageQuality) -> &'static str {
    if model.to_ascii_lowercase().contains("gpt-image") {
        match quality {
            ImageQuality::Standard => "medium",
            ImageQuality::Hd => "high",
        }
    } else {
        quality.as_str()
    }
}

/// Build the OpenAI `/images/edits` multipart form for variations/inpainting.
pub fn build_openai_edit_form(
    image: &[u8],
    mask: Option<&[u8]>,
    prompt: &str,
    model: &str,
    size: ImageSize,
    n: u32,
) -> ProviderResult<Form> {
    let image_part = Part::bytes(image.to_vec())
        .file_name("image.png")
        .mime_str("image/png")
        .map_err(ProviderError::Network)?;
    let mut form = Form::new()
        .text("prompt", prompt.to_string())
        .text("model", model.to_string())
        .text("size", size.to_string())
        .text("n", n.to_string())
        .text("response_format", "b64_json")
        .part("image", image_part);
    if let Some(mask) = mask {
        let mask_part = Part::bytes(mask.to_vec())
            .file_name("mask.png")
            .mime_str("image/png")
            .map_err(ProviderError::Network)?;
        form = form.part("mask", mask_part);
    }
    Ok(form)
}

/// Build the Stability AI `/v2beta/stable-image/generate` multipart form.
pub fn build_stability_form(
    prompt: &str,
    size: ImageSize,
    quality: ImageQuality,
    n: u32,
    seed: Option<u64>,
    negative_prompt: Option<&str>,
    output_format: Option<&str>,
) -> ProviderResult<Form> {
    let (w, h) = size.dimensions();
    let fmt = output_format.unwrap_or("png");
    let mut form = Form::new()
        .text("prompt", prompt.to_string())
        .text("output_format", fmt.to_string())
        .text("width", w.to_string())
        .text("height", h.to_string())
        .text("n", n.to_string());
    if let Some(seed) = seed {
        form = form.text("seed", seed.to_string());
    }
    if let Some(np) = negative_prompt {
        form = form.text("negative_prompt", np.to_string());
    }
    if matches!(quality, ImageQuality::Hd) {
        form = form.text("quality", "high");
    }
    Ok(form)
}

/// Build the Replicate prediction `input` object.
pub fn build_replicate_input(
    prompt: &str,
    size: ImageSize,
    n: u32,
    seed: Option<u64>,
    negative_prompt: Option<&str>,
    image_data_url: Option<&str>,
    mask_data_url: Option<&str>,
) -> serde_json::Map<String, serde_json::Value> {
    let (w, h) = size.dimensions();
    let mut input = serde_json::Map::new();
    input.insert("prompt".into(), serde_json::json!(prompt));
    input.insert("width".into(), serde_json::json!(w));
    input.insert("height".into(), serde_json::json!(h));
    if let Some(seed) = seed {
        input.insert("seed".into(), serde_json::json!(seed));
    }
    if let Some(np) = negative_prompt {
        input.insert("negative_prompt".into(), serde_json::json!(np));
    }
    if n > 1 {
        input.insert("num_outputs".into(), serde_json::json!(n));
    }
    if let Some(img) = image_data_url {
        input.insert("image".into(), serde_json::json!(img));
    }
    if let Some(mask) = mask_data_url {
        input.insert("mask".into(), serde_json::json!(mask));
    }
    input
}

/// Build the Prodia `/v1/sd/generate` JSON body.
pub fn build_prodia_body(
    prompt: &str,
    model: &str,
    size: ImageSize,
    n: u32,
    seed: Option<u64>,
    negative_prompt: Option<&str>,
    output_format: Option<&str>,
) -> serde_json::Value {
    let (w, h) = size.dimensions();
    let fmt = output_format.unwrap_or("png");
    serde_json::json!({
        "model": model,
        "prompt": prompt,
        "negative_prompt": negative_prompt.unwrap_or(""),
        "steps": 30,
        "cfg_scale": 7,
        "seed": seed.unwrap_or(0),
        "width": w,
        "height": h,
        "sampler": "DPM++ SDE Karras",
        "output_format": fmt,
        "n": n,
    })
}

/// Build a chat-completions body that requests an image modality (OpenRouter
/// image models such as `google/gemini-2.0-flash-image-preview`).
pub fn build_chat_completions_image_body(
    prompt: &str,
    model: &str,
    size: ImageSize,
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],
        "stream": false,
        "image_config": {
            "aspect_ratio": size.aspect_ratio_token(),
            "image_size": "1K",
        }
    })
}

/// Build a Qwen Token Plan multimodal-generation body.
pub fn build_qwen_token_plan_body(prompt: &str, model: &str, size: ImageSize) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "input": {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ]
        },
        "parameters": {
            "size": qwen_wire_size(size),
            "n": 1,
            "thinking_mode": false,
        }
    })
}

/// Render an [`ImageSize`] in the Qwen Token Plan wire format (`W*H`).
pub fn qwen_wire_size(size: ImageSize) -> String {
    let (w, h) = size.dimensions();
    format!("{w}*{h}")
}

// ---------------------------------------------------------------------------
// Response parsers
// ---------------------------------------------------------------------------

/// Parse an OpenAI `/images/generations` (or `/images/edits`) JSON response.
pub fn parse_openai_json(text: &str, model: &str) -> ProviderResult<ImageGenerationResponse> {
    validate_content_safety(text)?;
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let images: Vec<GeneratedImage> = data
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .map(|item| GeneratedImage {
                    b64_json: item
                        .get("b64_json")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    url: item.get("url").and_then(|v| v.as_str()).map(String::from),
                    revised_prompt: item
                        .get("revised_prompt")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    seed: item.get("seed").and_then(|v| v.as_u64()),
                })
                .collect()
        })
        .unwrap_or_default();
    if images.is_empty() {
        return Err(ProviderError::Provider(
            "Image generation provider returned no images".into(),
        ));
    }
    Ok(ImageGenerationResponse {
        images,
        model: model.to_string(),
        cost: None,
    })
}

/// Parse a Stability AI JSON envelope (`{"image": "<base64>", "seed": ...}`).
pub fn parse_stability_json(text: &str, model: &str) -> ProviderResult<ImageGenerationResponse> {
    validate_content_safety(text)?;
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let b64 = data.get("image").and_then(|v| v.as_str()).map(String::from);
    if b64.is_none() {
        return Err(ProviderError::Provider(
            "Stability image generation returned no image".into(),
        ));
    }
    let seed = data.get("seed").and_then(|v| v.as_u64());
    let image = GeneratedImage {
        b64_json: b64,
        url: None,
        revised_prompt: None,
        seed,
    };
    Ok(ImageGenerationResponse {
        images: vec![image],
        model: model.to_string(),
        cost: None,
    })
}

/// Extract the first image URL from an OpenRouter-style chat-completions
/// response (useful for chat-native image models that report `images` inside
/// the assistant message).
pub fn extract_openrouter_image_url(data: &serde_json::Value) -> Option<String> {
    for choice in data.get("choices")?.as_array()? {
        let message = choice.get("message")?;
        for image in message.get("images")?.as_array()? {
            let image_url = image
                .get("image_url")
                .or_else(|| image.get("imageUrl"))
                .unwrap_or(&serde_json::Value::Null);
            if let Some(url) = image_url.get("url").and_then(|v| v.as_str()) {
                if !url.is_empty() {
                    return Some(url.to_string());
                }
            }
        }
    }
    None
}

/// Extract the first image URL from a Qwen Token Plan response.
///
/// The payload nests the image inside `output.choices[].message.content[]`,
/// where each item carries an `image` (or `image_url`) string field.
pub fn extract_qwen_token_plan_image_url(data: &serde_json::Value) -> Option<String> {
    let output = data.get("output")?;
    for choice in output.get("choices")?.as_array()? {
        let message = choice.get("message")?;
        for item in message.get("content")?.as_array()? {
            if let Some(url) = item
                .get("image")
                .or_else(|| item.get("image_url"))
                .and_then(|v| v.as_str())
            {
                if !url.is_empty() {
                    return Some(url.to_string());
                }
            }
        }
    }
    None
}

/// Parse a chat-completions image response (OpenRouter-style) into an
/// [`ImageGenerationResponse`].
///
/// Inline `data:` URLs are normalized into a base64 payload; remote URLs are
/// kept as `url`.
pub fn parse_chat_completions_image(
    text: &str,
    model: &str,
) -> ProviderResult<ImageGenerationResponse> {
    validate_content_safety(text)?;
    let data: serde_json::Value =
        serde_json::from_str(text).map_err(ProviderError::Serialization)?;
    let image_url = extract_openrouter_image_url(&data).ok_or_else(|| {
        ProviderError::Provider("Image generation provider returned no images".into())
    })?;

    let image = match decode_data_url(&image_url) {
        Ok((_mime, bytes)) => GeneratedImage {
            b64_json: Some(encode_base64(&bytes)),
            url: None,
            revised_prompt: None,
            seed: None,
        },
        Err(_) => GeneratedImage {
            b64_json: None,
            url: Some(image_url),
            revised_prompt: None,
            seed: None,
        },
    };
    Ok(ImageGenerationResponse {
        images: vec![image],
        model: model.to_string(),
        cost: None,
    })
}

/// Collect string items from a JSON value that is either a single string or an
/// array of strings.
fn extract_string_array(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Secure image download
// ---------------------------------------------------------------------------

/// Maximum bytes to download for a generated image.
pub const GENERATED_IMAGE_DOWNLOAD_LIMIT: usize = 20 * 1024 * 1024;
/// Maximum redirects to follow when downloading a generated image.
pub const GENERATED_IMAGE_REDIRECT_LIMIT: u32 = 3;

/// Download a signed generated-image URL with an SSRF-oriented guard: only
/// `https` URLs without userinfo or fragments, a size cap, and a redirect cap.
pub async fn download_generated_image(
    client: &Client,
    image_url: &str,
) -> ProviderResult<(String, Vec<u8>)> {
    let parsed = reqwest::Url::parse(image_url)
        .map_err(|_| ProviderError::Provider("unsafe generated image URL".into()))?;
    if parsed.scheme() != "https"
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err(ProviderError::Provider("unsafe generated image URL".into()));
    }

    let mut current_url = image_url.to_string();
    for _ in 0..=GENERATED_IMAGE_REDIRECT_LIMIT {
        let resp = client
            .get(&current_url)
            .send()
            .await
            .map_err(ProviderError::Network)?;

        if let Some(location) = resp.headers().get(reqwest::header::LOCATION) {
            let loc = location
                .to_str()
                .map_err(|_| ProviderError::Provider("redirect without location".into()))?;
            let base = reqwest::Url::parse(&current_url)
                .map_err(|_| ProviderError::Provider("invalid redirect base".into()))?;
            current_url = base
                .join(loc)
                .map_err(|_| ProviderError::Provider("invalid redirect location".into()))?
                .to_string();
            continue;
        }

        let status = resp.status();
        if !status.is_success() {
            return Err(ProviderError::Provider(format!(
                "image download HTTP {status}"
            )));
        }
        if let Some(content_length) = resp.content_length() {
            if content_length as usize > GENERATED_IMAGE_DOWNLOAD_LIMIT {
                return Err(ProviderError::Provider(
                    "generated image exceeds download limit".into(),
                ));
            }
        }

        // Read content type before consuming resp
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("image/png")
            .split(';')
            .next()
            .unwrap_or("image/png")
            .trim()
            .to_ascii_lowercase();
        let mime = if content_type.starts_with("image/") {
            content_type
        } else {
            "image/png".into()
        };

        let mut bytes = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(ProviderError::Network)?;
            bytes.extend_from_slice(&chunk);
            if bytes.len() > GENERATED_IMAGE_DOWNLOAD_LIMIT {
                return Err(ProviderError::Provider(
                    "generated image exceeds download limit".into(),
                ));
            }
        }

        return Ok((mime, bytes));
    }
    Err(ProviderError::Provider(
        "generated image exceeded the redirect limit".into(),
    ))
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Whether a provider response body signals that the request was blocked by a
/// content-safety policy.
pub fn contains_safety_issue(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    const MARKERS: &[&str] = &[
        "content_policy_violation",
        "content policy violation",
        "content_filter",
        "safety system",
        "prompt blocked",
        "flagged for content",
        "unsafe image",
    ];
    MARKERS.iter().any(|m| lower.contains(m))
}

/// Reject responses blocked by a provider content-safety policy.
pub fn validate_content_safety(text: &str) -> ProviderResult<()> {
    if contains_safety_issue(text) {
        return Err(ProviderError::Provider(
            "Image generation was blocked by the provider's content safety policy".into(),
        ));
    }
    Ok(())
}

/// Parse image dimensions from PNG, JPEG, GIF, and WebP headers without
/// decoding the full image.
pub fn image_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() >= 24 && bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        // PNG: width at offset 16, height at offset 20 (big-endian).
        let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        return Some((w, h));
    }
    if bytes.len() >= 3 && bytes.starts_with(&[0xff, 0xd8]) {
        return jpeg_dimensions(bytes);
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return webp_dimensions(bytes);
    }
    if bytes.len() >= 10 && (bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        let w = u16::from_le_bytes([bytes[6], bytes[7]]) as u32;
        let h = u16::from_le_bytes([bytes[8], bytes[9]]) as u32;
        return Some((w, h));
    }
    None
}

/// Parse JPEG dimensions by scanning for a SOF marker.
fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2;
    while i + 9 < bytes.len() {
        if bytes[i] != 0xff {
            i += 1;
            continue;
        }
        let marker = bytes[i + 1];
        // Standalone markers carry no length field.
        if marker == 0xd8 || marker == 0xd9 || (0xd0..=0xd7).contains(&marker) || marker == 0x01 {
            i += 2;
            continue;
        }
        if i + 3 >= bytes.len() {
            return None;
        }
        let seg_len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        let is_sof = (0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
        if is_sof {
            if i + 9 >= bytes.len() {
                return None;
            }
            let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
            let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
            return Some((w, h));
        }
        i += 2 + seg_len;
    }
    None
}

/// Parse WebP dimensions from the container header.
fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 30 {
        return None;
    }
    match &bytes[12..16] {
        b"VP8X" => {
            let w = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], 0]) & 0xff_ffff;
            let h = u32::from_le_bytes([bytes[27], bytes[28], bytes[29], 0]) & 0xff_ffff;
            Some((w + 1, h + 1))
        }
        b"VP8 " => {
            let w = u16::from_le_bytes([bytes[26], bytes[27]]) as u32 & 0x3fff;
            let h = u16::from_le_bytes([bytes[28], bytes[29]]) as u32 & 0x3fff;
            Some((w, h))
        }
        b"VP8L" => {
            let b0 = bytes[21] as u32;
            let b1 = bytes[22] as u32;
            let b2 = bytes[23] as u32;
            let b3 = bytes[24] as u32;
            let w = ((b1 & 0x3f) << 8) | b0;
            let h = ((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6);
            Some((w + 1, h + 1))
        }
        _ => None,
    }
}

/// Validate that `bytes` looks like a decodable image and return its
/// dimensions.
pub fn validate_image_bytes(bytes: &[u8]) -> Result<(u32, u32), String> {
    if bytes.is_empty() {
        return Err("image is empty".into());
    }
    image_dimensions(bytes).ok_or_else(|| "image bytes are not a recognized image format".into())
}

/// Validate that `bytes` decodes to an image close to the requested size.
///
/// A small tolerance (8px) is applied because backends do not always honor the
/// requested dimensions exactly.
pub fn validate_image_size(bytes: &[u8], expected: ImageSize) -> Result<(), String> {
    let (w, h) = validate_image_bytes(bytes)?;
    let (ew, eh) = expected.dimensions();
    if (w as i64 - ew as i64).unsigned_abs() > 8 || (h as i64 - eh as i64).unsigned_abs() > 8 {
        return Err(format!(
            "image dimensions {w}x{h} do not match requested {ew}x{eh}"
        ));
    }
    Ok(())
}

/// Detect the MIME type of encoded image bytes from their magic header.
///
/// Returns `None` for anything that does not look like a known raster format.
pub fn image_mime_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

/// Validate that encoded bytes decode to an image of the given MIME type.
pub fn validate_image_mime(bytes: &[u8], expected_mime: &str) -> Result<(), String> {
    let detected = image_mime_type(bytes)
        .ok_or_else(|| "image bytes are not a recognized image format".to_string())?;
    if detected != expected_mime && expected_mime != "image/png" {
        // Backends frequently report a generic image/*; accept a prefix match.
        let expected_prefix = expected_mime.split('/').next().unwrap_or("");
        if !detected.starts_with(expected_prefix) {
            return Err(format!(
                "image MIME type {detected} does not match expected {expected_mime}"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cost estimation
// ---------------------------------------------------------------------------

/// Estimate the total cost in USD for generating `n` images.
pub fn estimate_image_cost(
    model: &str,
    size: ImageSize,
    quality: ImageQuality,
    n: u32,
) -> Option<f64> {
    let per = per_image_cost(model, size, quality)?;
    Some(per * n.max(1) as f64)
}

/// Estimate the cost in USD of a single image for a known model + size.
///
/// Pricing is drawn from the public OpenAI rate card; Stability is estimated
/// from a typical credit cost. Unknown models return `None`.
pub fn per_image_cost(model: &str, size: ImageSize, quality: ImageQuality) -> Option<f64> {
    let m = model.to_ascii_lowercase();
    let (w, h) = size.dimensions();
    let area = w as f64 * h as f64;

    if m.contains("gpt-image") {
        let cost = if area <= 1024.0 * 1024.0 {
            0.12
        } else if area <= 1536.0 * 1024.0 {
            0.18
        } else {
            0.24
        };
        return Some(cost);
    }
    if m.contains("dall-e-3") || m.contains("dalle-3") {
        return match (quality, w, h) {
            (ImageQuality::Standard, 1024, 1024) => Some(0.040),
            (ImageQuality::Standard, 1792, 1024) => Some(0.080),
            (ImageQuality::Standard, 1024, 1792) => Some(0.080),
            (ImageQuality::Hd, 1024, 1024) => Some(0.080),
            (ImageQuality::Hd, 1792, 1024) => Some(0.120),
            (ImageQuality::Hd, 1024, 1792) => Some(0.120),
            _ => None,
        };
    }
    if m.contains("dall-e-2") || m.contains("dalle-2") {
        return match (w, h) {
            (1024, 1024) => Some(0.020),
            (512, 512) => Some(0.018),
            (256, 256) => Some(0.016),
            _ => None,
        };
    }
    if m.contains("stable-image") || m.contains("sd3") || m.contains("stable-diffusion") {
        return Some(0.065);
    }
    None
}

// ---------------------------------------------------------------------------
// Base64 / data-url helpers
// ---------------------------------------------------------------------------

/// Decode a base64 string into bytes, tolerating whitespace.
pub fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(input.trim())
        .map_err(|e| e.to_string())
}

/// Encode bytes into a base64 string.
pub fn encode_base64(input: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(input)
}

/// Encode bytes into a `data:` URL.
pub fn encode_data_url(bytes: &[u8], mime: &str) -> String {
    format!("data:{mime};base64,{}", encode_base64(bytes))
}

/// Decode a `data:` URL into its MIME type and bytes.
pub fn decode_data_url(data_url: &str) -> Result<(String, Vec<u8>), String> {
    let (prefix, encoded) = data_url
        .split_once(',')
        .ok_or_else(|| "unsupported image URL".to_string())?;
    if !prefix.contains(";base64") {
        return Err("unsupported image URL".into());
    }
    let mime = prefix
        .strip_prefix("data:")
        .and_then(|p| p.split(';').next())
        .unwrap_or("image/png")
        .to_string();
    let bytes = decode_base64(encoded)?;
    Ok((mime, bytes))
}

// ---------------------------------------------------------------------------
// Fallback generation
// ---------------------------------------------------------------------------

/// A failed generation attempt recorded during [`generate_with_fallbacks`].
#[derive(Debug, Clone)]
pub struct GenerationAttempt {
    /// The provider name.
    pub provider: String,
    /// The model that was attempted.
    pub model: String,
    /// The error message.
    pub error: String,
}

/// Try a list of providers in order, returning the first successful image.
///
/// Each failure is recorded in a [`GenerationAttempt`]; if every candidate
/// fails, an aggregated error summarizing all attempts is returned. This
/// mirrors the Python backend's `generate_with_fallbacks` entry point.
pub async fn generate_with_fallbacks(
    providers: &[ImageGenerationProvider],
    prompt: &str,
    params: &ImageGenerationParams,
) -> ProviderResult<ImageGenerationResult> {
    let mut attempts: Vec<GenerationAttempt> = Vec::new();
    let mut last_error: Option<ProviderError> = None;

    for provider in providers {
        let provider_name = provider.name().to_string();
        let model = params
            .model
            .clone()
            .unwrap_or_else(|| provider.config().default_model.clone());
        match provider.generate_image(prompt, params).await {
            Ok(result) => {
                if !result.is_empty() {
                    return Ok(result);
                }
                attempts.push(GenerationAttempt {
                    provider: provider_name,
                    model,
                    error: "provider returned an empty image".into(),
                });
            }
            Err(err) => {
                let message = err.to_string();
                attempts.push(GenerationAttempt {
                    provider: provider_name,
                    model,
                    error: message,
                });
                last_error = Some(err);
            }
        }
    }

    if attempts.len() <= 1 {
        if let Some(err) = last_error {
            return Err(err);
        }
    }
    let summary = attempts
        .iter()
        .map(|a| format!("{}/{}: {}", a.provider, a.model, a.error))
        .collect::<Vec<_>>()
        .join(" | ");
    Err(ProviderError::Provider(format!(
        "All image generation models failed ({}): {summary}",
        attempts.len()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn form_has_field(form: &Form, name: &str) -> bool {
        // reqwest 0.12 does not expose the form fields directly; the `Debug`
        // impl renders each part as `("name", Part { ... })`, so a field is
        // present iff its quoted name appears in that output.
        format!("{form:?}").contains(&format!("\"{name}\""))
    }

    // --- Provider / config defaults --------------------------------------

    #[test]
    fn provider_enum_roundtrip() {
        assert_eq!(
            ImageGenProvider::parse("openai"),
            Some(ImageGenProvider::OpenAi)
        );
        assert_eq!(
            ImageGenProvider::parse("StabilityAi"),
            Some(ImageGenProvider::StabilityAi)
        );
        assert_eq!(
            ImageGenProvider::parse("replicate"),
            Some(ImageGenProvider::Replicate)
        );
        assert_eq!(
            ImageGenProvider::parse("prodia"),
            Some(ImageGenProvider::Prodia)
        );
        assert_eq!(
            ImageGenProvider::parse("local"),
            Some(ImageGenProvider::Local)
        );
        assert_eq!(ImageGenProvider::parse("nope"), None);
        assert_eq!(ImageGenProvider::OpenAi.as_str(), "openai");
    }

    #[test]
    fn config_defaults_per_backend() {
        let cfg = ImageGenConfig::new(ImageGenProvider::OpenAi, "sk-test");
        assert_eq!(cfg.base_url, "https://api.openai.com/v1");
        assert_eq!(cfg.default_model, "gpt-image-1");
        assert_eq!(cfg.default_size, ImageSize::Square);
        assert_eq!(cfg.retry.max_attempts, 3);

        let prodia = ImageGenConfig::new(ImageGenProvider::Prodia, "pk-test");
        assert_eq!(prodia.default_model, "3Guofeng3_v34.safetensors");
    }

    #[test]
    fn provider_builder_methods() {
        let p = ImageGenerationProvider::openai("sk-test")
            .with_model("dall-e-3")
            .with_size(ImageSize::Wide)
            .with_quality(ImageQuality::Hd)
            .with_base_url("https://example.com/v1");
        assert_eq!(p.config().default_model, "dall-e-3");
        assert_eq!(p.config().default_size, ImageSize::Wide);
        assert_eq!(p.config().base_url, "https://example.com/v1");
        assert_eq!(p.name(), "openai");
    }

    // --- ImageSize helpers -----------------------------------------------

    #[test]
    fn image_size_dimensions() {
        assert_eq!(ImageSize::Square.dimensions(), (1024, 1024));
        assert_eq!(ImageSize::Wide.dimensions(), (1792, 1024));
        assert_eq!(ImageSize::Tall.dimensions(), (1024, 1792));
        assert_eq!(ImageSize::Custom(512, 768).dimensions(), (512, 768));
    }

    #[test]
    fn image_size_to_string() {
        assert_eq!(ImageSize::Square.to_string(), "1024x1024");
        assert_eq!(ImageSize::Wide.to_string(), "1792x1024");
        assert_eq!(ImageSize::Custom(640, 480).to_string(), "640x480");
    }

    #[test]
    fn image_size_aspect_ratio() {
        assert_eq!(ImageSize::Square.aspect_ratio(), "1:1");
        assert_eq!(ImageSize::Wide.aspect_ratio(), "7:4");
        assert_eq!(ImageSize::Tall.aspect_ratio(), "4:7");
        assert_eq!(ImageSize::Custom(1920, 1080).aspect_ratio(), "16:9");
    }

    #[test]
    fn image_quality_str() {
        assert_eq!(ImageQuality::Standard.as_str(), "standard");
        assert_eq!(ImageQuality::Hd.as_str(), "hd");
    }

    // --- Request structs --------------------------------------------------

    #[test]
    fn request_default() {
        let req = ImageGenerationRequest::default();
        assert_eq!(req.model, "dall-e-3");
        assert_eq!(req.n, 1);
        assert_eq!(req.size, 1024);
        assert_eq!(req.response_format, ImageResponseFormat::B64Json);
    }

    #[test]
    fn request_serialize() {
        let req = ImageGenerationRequest {
            prompt: "a cat".into(),
            model: "dall-e-3".into(),
            n: 2,
            size: 512,
            response_format: ImageResponseFormat::Url,
            quality: Some("hd".into()),
            style: None,
        };
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["prompt"], "a cat");
        assert_eq!(value["n"], 2);
        assert_eq!(value["response_format"], "url");
        assert_eq!(value["quality"], "hd");
        assert!(value.get("style").is_none());
    }

    #[test]
    fn image_gen_params_default() {
        let p = ImageGenerationParams::default();
        assert_eq!(p.n, 1);
        assert_eq!(p.output_format.as_deref(), Some("png"));
        assert!(p.model.is_none());
    }

    // --- OpenAI request builder ------------------------------------------

    #[test]
    fn build_openai_body_dalle3() {
        let body = build_openai_body(
            "a cat",
            "dall-e-3",
            ImageSize::Wide,
            ImageQuality::Hd,
            Some("vivid"),
            1,
            None,
            ImageResponseFormat::B64Json,
        );
        assert_eq!(body["model"], "dall-e-3");
        assert_eq!(body["size"], "1792x1024");
        assert_eq!(body["quality"], "hd");
        assert_eq!(body["style"], "vivid");
        assert_eq!(body["response_format"], "b64_json");
    }

    #[test]
    fn build_openai_body_gpt_image_maps_quality() {
        let body = build_openai_body(
            "photo",
            "gpt-image-1",
            ImageSize::Square,
            ImageQuality::Standard,
            None,
            1,
            Some(42),
            ImageResponseFormat::Url,
        );
        assert_eq!(body["quality"], "medium");
        assert_eq!(body["seed"], 42);
        assert_eq!(body["response_format"], "url");
    }

    #[test]
    fn openai_quality_string_model_specific() {
        assert_eq!(
            openai_quality_string("gpt-image-1", ImageQuality::Hd),
            "high"
        );
        assert_eq!(openai_quality_string("dall-e-3", ImageQuality::Hd), "hd");
        assert_eq!(
            openai_quality_string("gpt-image-1", ImageQuality::Standard),
            "medium"
        );
    }

    #[test]
    fn build_openai_edit_form_basic() {
        let form =
            build_openai_edit_form(b"img", None, "fix it", "gpt-image-1", ImageSize::Square, 1)
                .unwrap();
        assert!(form_has_field(&form, "prompt"));
        assert!(form_has_field(&form, "image"));
        assert!(!form_has_field(&form, "mask"));
    }

    #[test]
    fn build_openai_edit_form_with_mask() {
        let form = build_openai_edit_form(
            b"img",
            Some(b"mask"),
            "fix it",
            "dall-e-2",
            ImageSize::Square,
            1,
        )
        .unwrap();
        assert!(form_has_field(&form, "mask"));
    }

    // --- Stability builder ------------------------------------------------

    #[test]
    fn build_stability_form_basic() {
        let form = build_stability_form(
            "a castle",
            ImageSize::Custom(640, 640),
            ImageQuality::Standard,
            1,
            Some(7),
            Some("blurry"),
            Some("png"),
        )
        .unwrap();
        assert!(form_has_field(&form, "prompt"));
        assert!(form_has_field(&form, "width"));
        assert!(form_has_field(&form, "height"));
        assert!(form_has_field(&form, "seed"));
        assert!(form_has_field(&form, "negative_prompt"));
        assert!(!form_has_field(&form, "quality"));
    }

    #[test]
    fn build_stability_form_hd_adds_quality() {
        let form = build_stability_form(
            "x",
            ImageSize::Square,
            ImageQuality::Hd,
            1,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(form_has_field(&form, "quality"));
    }

    // --- Replicate / Prodia builders --------------------------------------

    #[test]
    fn build_replicate_input_basic() {
        let input = build_replicate_input(
            "a dog",
            ImageSize::Square,
            2,
            Some(3),
            Some("ugly"),
            None,
            None,
        );
        assert_eq!(input["width"], 1024);
        assert_eq!(input["height"], 1024);
        assert_eq!(input["num_outputs"], 2);
        assert_eq!(input["seed"], 3);
        assert_eq!(input["negative_prompt"], "ugly");
    }

    #[test]
    fn build_replicate_input_with_image() {
        let input = build_replicate_input(
            "img2img",
            ImageSize::Square,
            1,
            None,
            None,
            Some("data:image/png;base64,abc"),
            Some("data:image/png;base64,mask"),
        );
        assert_eq!(input["image"], "data:image/png;base64,abc");
        assert_eq!(input["mask"], "data:image/png;base64,mask");
    }

    #[test]
    fn build_prodia_body_basic() {
        let body = build_prodia_body(
            "a plane",
            "v1",
            ImageSize::Custom(512, 512),
            1,
            None,
            None,
            None,
        );
        assert_eq!(body["model"], "v1");
        assert_eq!(body["width"], 512);
        assert_eq!(body["height"], 512);
        assert_eq!(body["steps"], 30);
    }

    // --- Response parsers --------------------------------------------------

    #[test]
    fn parse_openai_json_b64() {
        let text = json!({
            "created": 1,
            "data": [{
                "b64_json": "QUJD",
                "revised_prompt": "a revised prompt",
                "seed": 99
            }]
        })
        .to_string();
        let resp = parse_openai_json(&text, "dall-e-3").unwrap();
        assert_eq!(resp.images.len(), 1);
        assert_eq!(resp.images[0].b64_json.as_deref(), Some("QUJD"));
        assert_eq!(
            resp.images[0].revised_prompt.as_deref(),
            Some("a revised prompt")
        );
        assert_eq!(resp.images[0].seed, Some(99));
        assert_eq!(resp.images[0].bytes().unwrap(), b"ABC");
    }

    #[test]
    fn parse_openai_json_url() {
        let text = json!({
            "data": [{ "url": "https://example.com/i.png" }]
        })
        .to_string();
        let resp = parse_openai_json(&text, "gpt-image-1").unwrap();
        assert_eq!(
            resp.images[0].url.as_deref(),
            Some("https://example.com/i.png")
        );
        assert!(resp.images[0].b64_json.is_none());
    }

    #[test]
    fn parse_openai_json_empty_data_errors() {
        let text = json!({ "data": [] }).to_string();
        let err = parse_openai_json(&text, "dall-e-3").unwrap_err();
        assert!(err.to_string().contains("no images"));
    }

    #[test]
    fn parse_openai_json_malformed_errors() {
        assert!(parse_openai_json("{not json", "dall-e-3").is_err());
    }

    #[test]
    fn parse_openai_json_safety_blocked() {
        let text = json!({
            "error": { "message": "content_policy_violation: prompt blocked" }
        })
        .to_string();
        let err = parse_openai_json(&text, "dall-e-3").unwrap_err();
        assert!(err.to_string().to_lowercase().contains("safety"));
    }

    #[test]
    fn parse_stability_json_basic() {
        let text = json!({ "image": "QUJD", "seed": 5 }).to_string();
        let resp = parse_stability_json(&text, "sd3-large").unwrap();
        assert_eq!(resp.images[0].b64_json.as_deref(), Some("QUJD"));
        assert_eq!(resp.images[0].seed, Some(5));
    }

    #[test]
    fn parse_stability_json_missing_image_errors() {
        let text = json!({ "finish_reason": "ERROR" }).to_string();
        let err = parse_stability_json(&text, "sd3-large").unwrap_err();
        assert!(err.to_string().contains("no image"));
    }

    #[test]
    fn extract_openrouter_image_url_works() {
        let data = json!({
            "choices": [{
                "message": {
                    "images": [ { "image_url": { "url": "https://cdn/x.png" } } ]
                }
            }]
        });
        assert_eq!(
            extract_openrouter_image_url(&data).as_deref(),
            Some("https://cdn/x.png")
        );
    }

    #[test]
    fn extract_openrouter_image_url_none() {
        let data = json!({ "choices": [] });
        assert_eq!(extract_openrouter_image_url(&data), None);
    }

    // --- Validation --------------------------------------------------------

    #[test]
    fn png_dimensions_parsed() {
        let mut b = vec![0u8; 24];
        b[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        b[12..16].copy_from_slice(b"IHDR");
        b[16..20].copy_from_slice(&128u32.to_be_bytes());
        b[20..24].copy_from_slice(&64u32.to_be_bytes());
        assert_eq!(image_dimensions(&b), Some((128, 64)));
    }

    #[test]
    fn jpeg_dimensions_parsed() {
        fn jpeg_bytes(w: u16, h: u16) -> Vec<u8> {
            let mut b = vec![0u8; 40];
            b[0] = 0xff;
            b[1] = 0xd8;
            b[2] = 0xff;
            b[3] = 0xe0; // APP0
            b[4] = 0x00;
            b[5] = 0x10;
            b[20] = 0xff;
            b[21] = 0xc0; // SOF0
            b[22] = 0x00;
            b[23] = 0x0b;
            b[24] = 0x08;
            b[25] = (h >> 8) as u8;
            b[26] = (h & 0xff) as u8;
            b[27] = (w >> 8) as u8;
            b[28] = (w & 0xff) as u8;
            b
        }
        assert_eq!(image_dimensions(&jpeg_bytes(320, 240)), Some((320, 240)));
    }

    #[test]
    fn gif_and_webp_dimensions_parsed() {
        let mut gif = vec![0u8; 10];
        gif[..6].copy_from_slice(b"GIF89a");
        gif[6] = 10;
        gif[7] = 0;
        gif[8] = 20;
        gif[9] = 0;
        assert_eq!(image_dimensions(&gif), Some((10, 20)));

        let mut webp = vec![0u8; 30];
        webp[..4].copy_from_slice(b"RIFF");
        webp[8..12].copy_from_slice(b"WEBP");
        webp[12..16].copy_from_slice(b"VP8X");
        webp[24] = 99; // width-1 = 99 => 100
        webp[27] = 49; // height-1 = 49 => 50
        assert_eq!(image_dimensions(&webp), Some((100, 50)));
    }

    #[test]
    fn invalid_bytes_have_no_dimensions() {
        assert_eq!(image_dimensions(b""), None);
        assert_eq!(image_dimensions(b"not an image"), None);
        assert!(validate_image_bytes(b"").is_err());
    }

    #[test]
    fn image_mime_detection() {
        let mut png = vec![0u8; 24];
        png[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(image_mime_type(&png), Some("image/png"));
        assert_eq!(image_mime_type(&[0xff, 0xd8, 0xff]), Some("image/jpeg"));
        assert_eq!(image_mime_type(b"GIF89a..."), Some("image/gif"));
        assert_eq!(image_mime_type(b"BM..."), Some("image/bmp"));
        assert_eq!(image_mime_type(b"plain text"), None);
    }

    #[test]
    fn validate_image_mime_matches() {
        let mut png = vec![0u8; 24];
        png[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert!(validate_image_mime(&png, "image/png").is_ok());
        // Prefix match: a generic image/* expectation is tolerated.
        assert!(validate_image_mime(&png, "image/octet-stream").is_ok());
        assert!(validate_image_mime(&png, "video/mp4").is_err());
    }

    #[test]
    fn image_size_tuple_conversions() {
        assert_eq!(ImageSize::from((1024, 1024)), ImageSize::Square);
        assert_eq!(ImageSize::from((1792, 1024)), ImageSize::Wide);
        assert_eq!(ImageSize::from((1024, 1792)), ImageSize::Tall);
        assert_eq!(ImageSize::from((640, 480)), ImageSize::Custom(640, 480));
        let (w, h): (u32, u32) = ImageSize::Square.into();
        assert_eq!((w, h), (1024, 1024));
    }

    #[test]
    fn image_size_serde_roundtrip() {
        for size in [
            ImageSize::Square,
            ImageSize::Wide,
            ImageSize::Tall,
            ImageSize::Custom(640, 480),
        ] {
            let value = serde_json::to_value(size).unwrap();
            let back: ImageSize = serde_json::from_value(value).unwrap();
            assert_eq!(back, size);
        }
    }

    #[test]
    fn image_quality_serde_roundtrip() {
        assert_eq!(
            serde_json::from_value::<ImageQuality>(json!("hd")).unwrap(),
            ImageQuality::Hd
        );
        assert_eq!(
            serde_json::to_value(ImageQuality::Standard).unwrap(),
            json!("standard")
        );
    }

    #[test]
    fn validate_image_size_checks_dims() {
        let mut png = vec![0u8; 24];
        png[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        png[16..20].copy_from_slice(&1024u32.to_be_bytes());
        png[20..24].copy_from_slice(&1024u32.to_be_bytes());
        assert!(validate_image_size(&png, ImageSize::Square).is_ok());
        assert!(validate_image_size(&png, ImageSize::Wide).is_err());
    }

    #[test]
    fn content_safety_detection() {
        assert!(contains_safety_issue("content_policy_violation happened"));
        assert!(contains_safety_issue("your prompt was blocked"));
        assert!(!contains_safety_issue("a beautiful landscape"));
    }

    // --- Cost estimation ----------------------------------------------------

    #[test]
    fn cost_dalle3_standard() {
        let cost = per_image_cost("dall-e-3", ImageSize::Square, ImageQuality::Standard);
        assert_eq!(cost, Some(0.040));
    }

    #[test]
    fn cost_dalle3_hd() {
        let cost = per_image_cost("dall-e-3", ImageSize::Square, ImageQuality::Hd);
        assert_eq!(cost, Some(0.080));
        let wide = per_image_cost("dall-e-3", ImageSize::Wide, ImageQuality::Hd);
        assert_eq!(wide, Some(0.120));
    }

    #[test]
    fn cost_dalle2() {
        assert_eq!(
            per_image_cost("dall-e-2", ImageSize::Square, ImageQuality::Standard),
            Some(0.020)
        );
        assert_eq!(
            per_image_cost(
                "dall-e-2",
                ImageSize::Custom(512, 512),
                ImageQuality::Standard
            ),
            Some(0.018)
        );
        assert_eq!(
            per_image_cost(
                "dall-e-2",
                ImageSize::Custom(256, 256),
                ImageQuality::Standard
            ),
            Some(0.016)
        );
    }

    #[test]
    fn cost_gpt_image() {
        assert_eq!(
            per_image_cost("gpt-image-1", ImageSize::Square, ImageQuality::Standard),
            Some(0.12)
        );
        assert_eq!(
            per_image_cost("gpt-image-1", ImageSize::Wide, ImageQuality::Standard),
            Some(0.18)
        );
    }

    #[test]
    fn cost_unknown_model_is_none() {
        assert_eq!(
            per_image_cost("made-up-model", ImageSize::Square, ImageQuality::Standard),
            None
        );
    }

    #[test]
    fn estimate_cost_scales_with_n() {
        assert_eq!(
            estimate_image_cost("dall-e-3", ImageSize::Square, ImageQuality::Standard, 3),
            Some(0.12)
        );
    }

    #[test]
    fn provider_estimate_cost_for_uses_params_and_override() {
        let p = ImageGenerationProvider::openai("sk-test")
            .with_model("dall-e-3")
            .with_size(ImageSize::Square);
        let params = ImageGenerationParams::default();
        assert_eq!(p.estimate_cost_for(&params), Some(0.040));

        let params = ImageGenerationParams {
            quality: Some(ImageQuality::Hd),
            ..ImageGenerationParams::default()
        };
        assert_eq!(p.estimate_cost_for(&params), Some(0.080));

        let p2 = p.with_cost_override(9.99);
        assert_eq!(
            p2.estimate_cost_for(&ImageGenerationParams::default()),
            Some(9.99)
        );
        assert_eq!(p2.retry_config().max_attempts, 3);
    }

    // --- Base64 / data-url helpers ------------------------------------------

    #[test]
    fn decode_encode_base64_roundtrip() {
        let original = b"hello image world";
        let encoded = encode_base64(original);
        let decoded = decode_base64(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn data_url_roundtrip() {
        let url = encode_data_url(b"abc", "image/png");
        assert!(url.starts_with("data:image/png;base64,"));
        let (mime, bytes) = decode_data_url(&url).unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, b"abc");
    }

    #[test]
    fn decode_data_url_rejects_non_base64() {
        assert!(decode_data_url("https://example.com/x.png").is_err());
        assert!(decode_data_url("data:image/png,abc").is_err());
    }

    // --- Result / generated image helpers -------------------------------------

    #[test]
    fn generated_image_bytes() {
        let img = GeneratedImage {
            b64_json: Some(encode_base64(b"pixel data")),
            url: None,
            revised_prompt: None,
            seed: None,
        };
        assert_eq!(img.bytes().unwrap(), b"pixel data");
    }

    #[test]
    fn generated_image_into_result() {
        let img = GeneratedImage {
            b64_json: Some("QUJD".into()),
            url: None,
            revised_prompt: Some("rp".into()),
            seed: Some(1),
        };
        let r = img.into_result("dall-e-3".into(), Some(0.04));
        assert_eq!(r.url, "");
        assert_eq!(r.b64_json.as_deref(), Some("QUJD"));
        assert_eq!(r.cost, Some(0.04));
        assert!(!r.is_empty());
        assert_eq!(r.bytes().unwrap(), b"ABC");
        assert!(r.data_url().unwrap().starts_with("data:image/png;base64,"));
    }

    // --- extract helpers -------------------------------------------------------

    #[test]
    fn extract_string_array_variants() {
        assert_eq!(extract_string_array(&json!("a")), vec!["a".to_string()]);
        assert_eq!(
            extract_string_array(&json!(["a", "b"])),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(extract_string_array(&json!([1, 2])), Vec::<String>::new());
        assert_eq!(extract_string_array(&json!(null)), Vec::<String>::new());
    }

    // --- Chat-completions / Qwen Token Plan builders ---------------------------

    #[test]
    fn build_chat_completions_image_body_basic() {
        let body = build_chat_completions_image_body(
            "a cat",
            "google/gemini-2.0-flash-image-preview",
            ImageSize::Square,
        );
        assert_eq!(body["model"], "google/gemini-2.0-flash-image-preview");
        assert_eq!(body["messages"][0]["content"], "a cat");
        assert_eq!(body["modalities"][0], "image");
        assert_eq!(body["stream"], false);
        assert_eq!(body["image_config"]["aspect_ratio"], "1:1");
        assert_eq!(body["image_config"]["image_size"], "1K");
    }

    #[test]
    fn qwen_wire_size_format() {
        assert_eq!(qwen_wire_size(ImageSize::Square), "1024*1024");
        assert_eq!(qwen_wire_size(ImageSize::Custom(768, 512)), "768*512");
    }

    #[test]
    fn build_qwen_token_plan_body_basic() {
        let body = build_qwen_token_plan_body("a castle", "wan2.7-image", ImageSize::Square);
        assert_eq!(body["model"], "wan2.7-image");
        assert_eq!(
            body["input"]["messages"][0]["content"][0]["text"],
            "a castle"
        );
        assert_eq!(body["parameters"]["size"], "1024*1024");
        assert_eq!(body["parameters"]["thinking_mode"], false);
    }

    #[test]
    fn extract_qwen_token_plan_image_url_works() {
        let data = json!({
            "output": {
                "choices": [{
                    "message": {
                        "content": [
                            {"text": "here you go"},
                            {"image": "https://cdn/qwen.png"}
                        ]
                    }
                }]
            }
        });
        assert_eq!(
            extract_qwen_token_plan_image_url(&data).as_deref(),
            Some("https://cdn/qwen.png")
        );
    }

    #[test]
    fn extract_qwen_token_plan_image_url_none() {
        assert_eq!(
            extract_qwen_token_plan_image_url(&json!({"output": {}})),
            None
        );
    }

    #[test]
    fn parse_chat_completions_image_data_url() {
        let text = json!({
            "choices": [{
                "message": {
                    "images": [ { "image_url": { "url": "data:image/png;base64,QUJD" } } ]
                }
            }]
        })
        .to_string();
        let resp = parse_chat_completions_image(&text, "gemini-image").unwrap();
        assert_eq!(resp.images[0].b64_json.as_deref(), Some("QUJD"));
        assert!(resp.images[0].url.is_none());
    }

    #[test]
    fn parse_chat_completions_image_remote_url() {
        let text = json!({
            "choices": [{
                "message": {
                    "images": [ { "image_url": { "url": "https://cdn/out.png" } } ]
                }
            }]
        })
        .to_string();
        let resp = parse_chat_completions_image(&text, "gemini-image").unwrap();
        assert_eq!(resp.images[0].url.as_deref(), Some("https://cdn/out.png"));
        assert!(resp.images[0].b64_json.is_none());
    }

    #[test]
    fn parse_chat_completions_image_no_images_errors() {
        let text = json!({ "choices": [] }).to_string();
        assert!(parse_chat_completions_image(&text, "gemini-image").is_err());
    }

    #[tokio::test]
    async fn fallbacks_aggregate_failures() {
        // Invalid base URLs fail fast at URL parse time — no network involved.
        let p1 = ImageGenerationProvider::openai("key1")
            .with_retry(RetryConfig::no_retry())
            .with_base_url("http://[::1");
        let p2 = ImageGenerationProvider::openai("key2")
            .with_retry(RetryConfig::no_retry())
            .with_base_url("http://[::1");
        let providers = vec![p1, p2];
        let params = ImageGenerationParams::default();
        let err = generate_with_fallbacks(&providers, "a cat", &params)
            .await
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("All image generation models failed (2)"),
            "unexpected message: {message}"
        );
    }

    // --- Secure download URL guard (fails before any network I/O) --------------

    #[tokio::test]
    async fn download_rejects_non_https_urls() {
        let client = Client::new();
        assert!(
            download_generated_image(&client, "http://example.com/i.png")
                .await
                .is_err()
        );
        assert!(
            download_generated_image(&client, "https://user:pw@example.com/i.png")
                .await
                .is_err()
        );
        assert!(
            download_generated_image(&client, "data:image/png;base64,QUJD")
                .await
                .is_err()
        );
        assert!(
            download_generated_image(&client, "not a url")
                .await
                .is_err()
        );
    }

    // --- Serde round-trips ------------------------------------------------------

    #[test]
    fn image_gen_provider_serde_roundtrip() {
        assert_eq!(
            serde_json::from_value::<ImageGenProvider>(json!("stability_ai")).unwrap(),
            ImageGenProvider::StabilityAi
        );
        assert_eq!(
            serde_json::to_value(ImageGenProvider::Prodia).unwrap(),
            json!("prodia")
        );
    }

    #[test]
    fn image_generation_response_serde() {
        let resp = ImageGenerationResponse {
            images: vec![GeneratedImage {
                b64_json: Some("QUJD".into()),
                url: None,
                revised_prompt: None,
                seed: Some(1),
            }],
            model: "dall-e-3".into(),
            cost: Some(0.04),
        };
        let value = serde_json::to_value(&resp).unwrap();
        assert_eq!(value["model"], "dall-e-3");
        assert_eq!(value["cost"], 0.04);
        assert_eq!(value["images"][0]["b64_json"], "QUJD");
    }

    #[test]
    fn image_generation_result_serde() {
        let result = ImageGenerationResult {
            url: "".into(),
            b64_json: Some("QUJD".into()),
            revised_prompt: Some("better".into()),
            seed: Some(5),
            cost: Some(0.04),
        };
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(value["revised_prompt"], "better");
        assert_eq!(value["seed"], 5);
        let back: ImageGenerationResult = serde_json::from_value(value).unwrap();
        assert_eq!(back.b64_json.as_deref(), Some("QUJD"));
    }

    #[test]
    fn cost_scales_for_gpt_image() {
        assert_eq!(
            estimate_image_cost("gpt-image-1", ImageSize::Square, ImageQuality::Standard, 2),
            Some(0.24)
        );
    }

    // --- Extra edge cases -------------------------------------------------------

    #[test]
    fn base64_decoding_tolerates_whitespace() {
        let encoded = "QUJD\nREVG\t";
        let decoded = decode_base64(encoded).unwrap();
        assert_eq!(decoded, b"ABCDEF");
    }

    #[test]
    fn validate_image_size_has_tolerance() {
        fn png_bytes(w: u32, h: u32) -> Vec<u8> {
            let mut b = vec![0u8; 24];
            b[..8].copy_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
            b[12..16].copy_from_slice(b"IHDR");
            b[16..20].copy_from_slice(&w.to_be_bytes());
            b[20..24].copy_from_slice(&h.to_be_bytes());
            b
        }
        // 1020x1024 is within the 8px tolerance of 1024x1024.
        assert!(validate_image_size(&png_bytes(1020, 1024), ImageSize::Square).is_ok());
        // 512x512 is far from 1024x1024.
        assert!(validate_image_size(&png_bytes(512, 512), ImageSize::Square).is_err());
    }

    #[test]
    fn safety_markers_are_detected() {
        assert!(contains_safety_issue("the request was flagged for content"));
        assert!(contains_safety_issue("content filter triggered"));
        assert!(contains_safety_issue("SAFETY SYSTEM INTERVENTION"));
        assert!(!contains_safety_issue("a peaceful meadow"));
    }

    #[test]
    fn build_openai_body_omits_optional_fields() {
        let body = build_openai_body(
            "x",
            "dall-e-3",
            ImageSize::Square,
            ImageQuality::Standard,
            None,
            1,
            None,
            ImageResponseFormat::default(),
        );
        assert!(body.get("style").is_none());
        assert!(body.get("seed").is_none());
        assert_eq!(body["response_format"], "b64_json");
    }
}
