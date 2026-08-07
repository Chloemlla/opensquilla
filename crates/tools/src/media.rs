//! Media tools: image processing, PDF reading, text-to-speech (TTS).
//!
//! Provides tools for:
//! - Image processing (resize, convert, format info) via the `image` crate
//! - PDF text extraction via the `lopdf` crate
//! - Text-to-speech via HTTP API (e.g., ElevenLabs)

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use base64::Engine;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

/// Tool for image processing operations.
pub struct ImageTool {
    /// Allowed base directory for reading images.
    allowed_base: PathBuf,
    /// Maximum image file size in bytes.
    max_image_size: u64,
}

impl ImageTool {
    /// Create a new image tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            max_image_size: 20 * 1024 * 1024, // 20 MB
        }
    }

    /// Resolve a file path safely.
    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new(
                "PATH_INVALID",
                format!("Cannot access path '{}': {}", path_str, e),
            )
        })?;
        if !canonical.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!("Path '{}' is outside the allowed base", path_str),
            ));
        }
        Ok(canonical)
    }

    /// Get image information (dimensions, format, color type).
    fn image_info(path: &std::path::Path) -> ToolResult<ToolOutput> {
        let reader = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?;

        let format = reader
            .format()
            .map(|f| format!("{:?}", f))
            .unwrap_or_default();
        let (width, height) = reader.into_dimensions().map_err(|e| {
            ToolError::new("IMAGE_ERROR", format!("Failed to read dimensions: {}", e))
        })?;

        let metadata = std::fs::metadata(path)
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read metadata: {}", e)))?;

        let data = serde_json::json!({
            "width": width,
            "height": height,
            "format": format,
            "size_bytes": metadata.len(),
            "path": path.to_string_lossy(),
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }

    /// Resize an image to the given dimensions.
    fn resize_image(
        path: &std::path::Path,
        width: u32,
        height: u32,
        output_path: &std::path::Path,
    ) -> ToolResult<ToolOutput> {
        let img = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?
            .decode()
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to decode image: {}", e)))?;

        let resized = img.resize_exact(width, height, image::imageops::FilterType::Lanczos3);

        // Ensure output directory exists.
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        resized
            .save(output_path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to save image: {}", e)))?;

        let data = serde_json::json!({
            "original_width": img.width(),
            "original_height": img.height(),
            "new_width": width,
            "new_height": height,
            "output_path": output_path.to_string_lossy(),
        });

        Ok(ToolOutput::success(format!(
            "Resized image from {}x{} to {}x{}",
            img.width(),
            img.height(),
            width,
            height
        ))
        .with_data(data))
    }

    /// Convert an image to a different format.
    fn convert_image(
        path: &std::path::Path,
        format: &str,
        output_path: &std::path::Path,
    ) -> ToolResult<ToolOutput> {
        let img = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?
            .decode()
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to decode image: {}", e)))?;

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        match format.to_lowercase().as_str() {
            "png" => img.save(output_path),
            "jpeg" | "jpg" => img.save(output_path),
            "gif" => img.save(output_path),
            "webp" => img.save(output_path),
            "bmp" => img.save(output_path),
            "tiff" => img.save(output_path),
            "ico" => img.save(output_path),
            "pnm" => img.save(output_path),
            "qoi" => img.save(output_path),
            _ => {
                return Err(ToolError::invalid_args(format!(
                    "Unsupported output format: '{}'. Supported: png, jpeg, gif, webp, bmp, tiff, ico, pnm, qoi",
                    format
                )));
            }
        }
        .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to convert image: {}", e)))?;

        let data = serde_json::json!({
            "format": format,
            "output_path": output_path.to_string_lossy(),
        });

        Ok(ToolOutput::success(format!("Converted image to {} format", format)).with_data(data))
    }

    /// Crop an image to a rectangle.
    fn crop_image(
        path: &std::path::Path,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        output_path: &std::path::Path,
    ) -> ToolResult<ToolOutput> {
        let img = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?
            .decode()
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to decode image: {}", e)))?;

        let (img_width, img_height) = (img.width(), img.height());
        if x >= img_width || y >= img_height {
            return Err(ToolError::invalid_args(format!(
                "Crop origin ({},{}) is outside the image bounds ({}x{})",
                x, y, img_width, img_height
            )));
        }
        let width = width.min(img_width - x);
        let height = height.min(img_height - y);

        let cropped = img.crop_imm(x, y, width, height);

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        cropped.save(output_path).map_err(|e| {
            ToolError::new(
                "IMAGE_ERROR",
                format!("Failed to save cropped image: {}", e),
            )
        })?;

        let data = serde_json::json!({
            "original_width": img_width,
            "original_height": img_height,
            "crop_x": x,
            "crop_y": y,
            "crop_width": width,
            "crop_height": height,
            "output_path": output_path.to_string_lossy(),
        });

        Ok(ToolOutput::success(format!(
            "Cropped image to {}x{} at ({},{})",
            width, height, x, y
        ))
        .with_data(data))
    }

    /// Rotate an image by a number of degrees (multiples of 90).
    fn rotate_image(
        path: &std::path::Path,
        degrees: u32,
        output_path: &std::path::Path,
    ) -> ToolResult<ToolOutput> {
        let img = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?
            .decode()
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to decode image: {}", e)))?;

        let rotated = match degrees % 360 {
            0 => img.clone(),
            90 => img.rotate90(),
            180 => img.rotate180(),
            270 => img.rotate270(),
            other => {
                return Err(ToolError::invalid_args(format!(
                    "Unsupported rotation: {} degrees (must be a multiple of 90)",
                    other
                )));
            }
        };

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        rotated.save(output_path).map_err(|e| {
            ToolError::new(
                "IMAGE_ERROR",
                format!("Failed to save rotated image: {}", e),
            )
        })?;

        let data = serde_json::json!({
            "degrees": degrees,
            "output_path": output_path.to_string_lossy(),
        });

        Ok(ToolOutput::success(format!("Rotated image by {} degrees", degrees)).with_data(data))
    }

    /// Get image metadata including basic EXIF information.
    ///
    /// Uses the `image` crate's metadata support and parses the JPEG APP1
    /// segment to detect embedded EXIF data.
    fn image_metadata(path: &std::path::Path) -> ToolResult<ToolOutput> {
        let reader = image::ImageReader::open(path)
            .map_err(|e| ToolError::new("IMAGE_ERROR", format!("Failed to open image: {}", e)))?;

        let format = reader
            .format()
            .map(|f| format!("{:?}", f))
            .unwrap_or_default();
        let (width, height) = reader.into_dimensions().map_err(|e| {
            ToolError::new("IMAGE_ERROR", format!("Failed to read dimensions: {}", e))
        })?;

        let file_metadata = std::fs::metadata(path)
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read metadata: {}", e)))?;

        // Basic EXIF-style metadata extraction from JPEG APP1 header.
        let mut exif = serde_json::Map::new();
        if let Ok(bytes) = std::fs::read(path) {
            if bytes.len() > 4 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
                // JPEG - try to find EXIF APP1 marker (FF E1).
                let mut offset = 2;
                while offset + 4 < bytes.len() {
                    if bytes[offset] == 0xFF {
                        let marker = bytes[offset + 1];
                        let size =
                            u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
                        if marker == 0xE1 && offset + 4 + size <= bytes.len() {
                            exif.insert("has_exif".to_string(), serde_json::json!(true));
                            let tiff_start = offset + 4;
                            if tiff_start + 8 <= bytes.len()
                                && &bytes[tiff_start..tiff_start + 4] == b"Exif"
                            {
                                exif.insert("exif_version".to_string(), serde_json::json!("2.x"));
                            }
                            break;
                        } else if marker != 0xD8 && marker != 0xD9 {
                            offset += 2 + size;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
        }

        let data = serde_json::json!({
            "width": width,
            "height": height,
            "format": format,
            "size_bytes": file_metadata.len(),
            "path": path.to_string_lossy(),
            "exif": exif,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }
}

#[async_trait]
impl Tool for ImageTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "image",
                concat!(
                    "Process images: get information, resize, crop, rotate, convert between formats, ",
                    "and read metadata (including EXIF detection). ",
                    "Supports PNG, JPEG, GIF, WebP, BMP, TIFF, ICO, PNM, and QOI formats.",
),
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The operation to perform")
                            .enum_values(vec![
                                "info".to_string(),
                                "metadata".to_string(),
                                "resize".to_string(),
                                "crop".to_string(),
                                "rotate".to_string(),
                                "convert".to_string(),
                            ]),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::required_string("Path to the image file"),
                    ),
                    (
                        "width".to_string(),
                        ParameterDefinition::integer(
                            "Target width in pixels (required for resize)",
                        ),
                    ),
                    (
                        "height".to_string(),
                        ParameterDefinition::integer(
                            "Target height in pixels (required for resize)",
                        ),
                    ),
                    (
                        "format".to_string(),
                        ParameterDefinition::string(
                            "Target format (required for convert: png, jpeg, gif, webp, bmp, tiff, ico, pnm, qoi)",
                        ),
                    ),
                    (
                        "output_path".to_string(),
                        ParameterDefinition::string("Output path for the result image"),
                    ),
                    (
                        "x".to_string(),
                        ParameterDefinition::integer("Crop origin X (required for crop)"),
                    ),
                    (
                        "y".to_string(),
                        ParameterDefinition::integer("Crop origin Y (required for crop)"),
                    ),
                    (
                        "degrees".to_string(),
                        ParameterDefinition::integer(
                            "Rotation degrees, multiple of 90 (required for rotate)",
                        ),
                    ),
                ]),
            )
            .category("media")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?
            .to_string();

        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'path' parameter"))?
            .to_string();

        let path = self.resolve_path(&path_str)?;

        // Image decode/encode is CPU-bound and touches the filesystem; run it
        // on a blocking thread to avoid stalling the async executor. The audit
        // found these calls ran inline on the runtime.
        match operation.as_str() {
            "info" => {
                let path_clone = path.clone();
                tokio::task::spawn_blocking(move || Self::image_info(&path_clone))
                    .await
                    .map_err(|e| {
                        ToolError::new("IMAGE_ERROR", format!("Image info task failed: {}", e))
                    })?
            }
            "metadata" => {
                let path_clone = path.clone();
                tokio::task::spawn_blocking(move || Self::image_metadata(&path_clone))
                    .await
                    .map_err(|e| {
                        ToolError::new("IMAGE_ERROR", format!("Image metadata task failed: {}", e))
                    })?
            }
            "resize" => {
                let width = params["width"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'width' for resize"))?
                    as u32;
                let height = params["height"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'height' for resize"))?
                    as u32;
                let output = params["output_path"].as_str().unwrap_or(&path_str);
                let output_path = self.resolve_path(output)?;
                let path_clone = path.clone();
                let output_path_clone = output_path.clone();
                tokio::task::spawn_blocking(move || {
                    Self::resize_image(&path_clone, width, height, &output_path_clone)
                })
                .await
                .map_err(|e| {
                    ToolError::new("IMAGE_ERROR", format!("Image resize task failed: {}", e))
                })?
            }
            "crop" => {
                let x = params["x"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'x' for crop"))?
                    as u32;
                let y = params["y"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'y' for crop"))?
                    as u32;
                let width = params["width"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'width' for crop"))?
                    as u32;
                let height = params["height"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'height' for crop"))?
                    as u32;
                let output = params["output_path"].as_str().unwrap_or(&path_str);
                let output_path = self.resolve_path(output)?;
                let path_clone = path.clone();
                let output_path_clone = output_path.clone();
                tokio::task::spawn_blocking(move || {
                    Self::crop_image(&path_clone, x, y, width, height, &output_path_clone)
                })
                .await
                .map_err(|e| {
                    ToolError::new("IMAGE_ERROR", format!("Image crop task failed: {}", e))
                })?
            }
            "rotate" => {
                let degrees = params["degrees"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'degrees' for rotate"))?
                    as u32;
                let output = params["output_path"].as_str().unwrap_or(&path_str);
                let output_path = self.resolve_path(output)?;
                let path_clone = path.clone();
                let output_path_clone = output_path.clone();
                tokio::task::spawn_blocking(move || {
                    Self::rotate_image(&path_clone, degrees, &output_path_clone)
                })
                .await
                .map_err(|e| {
                    ToolError::new("IMAGE_ERROR", format!("Image rotate task failed: {}", e))
                })?
            }
            "convert" => {
                let format = params["format"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'format' for convert"))?
                    .to_string();
                let output = params["output_path"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'output_path' for convert"))?;
                let output_path = self.resolve_path(output)?;
                let path_clone = path.clone();
                let output_path_clone = output_path.clone();
                tokio::task::spawn_blocking(move || {
                    Self::convert_image(&path_clone, &format, &output_path_clone)
                })
                .await
                .map_err(|e| {
                    ToolError::new("IMAGE_ERROR", format!("Image convert task failed: {}", e))
                })?
            }
            other => Err(ToolError::invalid_args(format!(
                "Unknown operation: {}",
                other
            ))),
        }
    }
}

/// Tool for reading PDF files.
pub struct PdfTool {
    /// Allowed base directory.
    allowed_base: PathBuf,
    /// Maximum PDF file size in bytes.
    max_pdf_size: u64,
}

impl PdfTool {
    /// Create a new PDF tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            max_pdf_size: 50 * 1024 * 1024, // 50 MB
        }
    }

    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new(
                "PATH_INVALID",
                format!("Cannot access path '{}': {}", path_str, e),
            )
        })?;
        if !canonical.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!("Path '{}' is outside the allowed base", path_str),
            ));
        }
        Ok(canonical)
    }
}

#[async_trait]
impl Tool for PdfTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "pdf",
                concat!(
                    "Read and extract text content from PDF files, or get PDF metadata ",
                    "such as page count. The 'info' operation returns document metadata; ",
                    "the 'extract' operation returns text content page by page.",
                ),
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::string("Operation: extract (default) or info")
                            .default(serde_json::json!("extract")),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::required_string("Path to the PDF file"),
                    ),
                    (
                        "page_start".to_string(),
                        ParameterDefinition::integer("Starting page number (1-based)")
                            .default(serde_json::json!(1)),
                    ),
                    (
                        "page_end".to_string(),
                        ParameterDefinition::integer("Ending page number (inclusive)"),
                    ),
                ]),
            )
            .category("media")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'path' parameter"))?
            .to_string();

        let path = self.resolve_path(&path_str)?;

        let metadata = std::fs::metadata(&path).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read file metadata: {}", e))
        })?;

        if metadata.len() > self.max_pdf_size {
            return Err(ToolError::new(
                "FILE_TOO_LARGE",
                format!(
                    "PDF too large: {} bytes (max {})",
                    metadata.len(),
                    self.max_pdf_size
                ),
            ));
        }

        let operation = params["operation"].as_str().unwrap_or("extract");

        if operation == "info" {
            // Return metadata only (page count, size) without extracting text.
            let path_for_task = path.clone();
            let page_count = tokio::task::spawn_blocking(move || -> ToolResult<u32> {
                let doc = lopdf::Document::load(&path_for_task).map_err(|e| {
                    ToolError::new("PDF_ERROR", format!("Failed to open PDF: {}", e))
                })?;
                Ok(doc.get_pages().len() as u32)
            })
            .await
            .map_err(|e| ToolError::new("PDF_ERROR", format!("PDF info task failed: {}", e)))??;

            let data = serde_json::json!({
                "path": path_str,
                "page_count": page_count,
                "size_bytes": metadata.len(),
                "pdf_version": "unknown",
            });

            return Ok(ToolOutput::success(
                serde_json::to_string_pretty(&data).unwrap_or_default(),
            )
            .with_data(data));
        }

        let page_start = params["page_start"].as_i64().unwrap_or(1).max(1) as u32;
        let page_end = params["page_end"].as_i64().map(|n| n as u32);

        // PDF parsing is CPU-bound and reads from disk; run it on a blocking
        // thread so the async executor is not stalled.
        let path_for_task = path.clone();
        let path_str_for_task = path_str.clone();
        let (content, page_count, pages_extracted) =
            tokio::task::spawn_blocking(move || -> ToolResult<(String, u32, u32)> {
                let doc = lopdf::Document::load(&path_for_task).map_err(|e| {
                    ToolError::new("PDF_ERROR", format!("Failed to open PDF: {}", e))
                })?;

                let page_count = doc.get_pages().len() as u32;
                let end_page = page_end.unwrap_or(page_count).min(page_count);

                let mut content = String::new();
                content.push_str(&format!(
                    "PDF: {}\nPages: {}-{} of {}\n\n",
                    path_str_for_task, page_start, end_page, page_count
                ));

                for page_num in page_start..=end_page {
                    if let Ok(text) = doc.extract_text(&[page_num]) {
                        content.push_str(&format!("--- Page {} ---\n", page_num));
                        content.push_str(text.trim());
                        content.push('\n');
                    }
                }

                let pages_extracted = end_page.saturating_sub(page_start) + 1;
                Ok((content, page_count, pages_extracted))
            })
            .await
            .map_err(|e| {
                ToolError::new("PDF_ERROR", format!("PDF extraction task failed: {}", e))
            })??;

        let data = serde_json::json!({
            "path": path_str,
            "page_count": page_count,
            "pages_extracted": pages_extracted,
            "size_bytes": metadata.len(),
        });

        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for text-to-speech conversion.
pub struct TtsTool {
    /// API key for the TTS service.
    api_key: Option<String>,
    /// Base URL for the TTS API.
    api_url: String,
}

impl TtsTool {
    /// Create a new TTS tool with the given API key.
    pub fn new(api_key: Option<String>) -> Self {
        Self {
            api_key,
            api_url: "https://api.elevenlabs.io/v1/text-to-speech".to_string(),
        }
    }

    /// Set a custom API base URL.
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }
}

#[async_trait]
impl Tool for TtsTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "tts",
                concat!(
                    "Convert text to speech using an ElevenLabs-compatible TTS API. ",
                    "Returns the audio as a base64-encoded MP3 file.",
                ),
                HashMap::from([
                    (
                        "text".to_string(),
                        ParameterDefinition::required_string("The text to convert to speech"),
                    ),
                    (
                        "voice".to_string(),
                        ParameterDefinition::string("Voice ID or name")
                            .default(serde_json::json!("21m00Tcm4TlvDq8ikWAM")),
                    ),
                    (
                        "stability".to_string(),
                        ParameterDefinition::integer("Voice stability (0-100)")
                            .default(serde_json::json!(50)),
                    ),
                    (
                        "similarity_boost".to_string(),
                        ParameterDefinition::integer("Voice similarity boost (0-100)")
                            .default(serde_json::json!(75)),
                    ),
                ]),
            )
            .category("media")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let text = params["text"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'text' parameter"))?;

        if text.len() > 5000 {
            return Err(ToolError::new(
                "TEXT_TOO_LONG",
                format!("Text too long: {} characters (max 5000)", text.len()),
            ));
        }

        let voice = params["voice"].as_str().unwrap_or("21m00Tcm4TlvDq8ikWAM");
        let stability = params["stability"].as_i64().unwrap_or(50) as f64 / 100.0;
        let similarity_boost = params["similarity_boost"].as_i64().unwrap_or(75) as f64 / 100.0;

        let api_key = self.api_key.as_deref().ok_or_else(|| {
            ToolError::new("CONFIG_ERROR", "TTS API key is not configured".to_string())
        })?;

        let url = format!("{}/{}/stream", self.api_url, voice);

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("xi-api-key", api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                }
            }))
            .send()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("TTS request failed: {}", e)))?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::new(
                "TTS_ERROR",
                format!("TTS API returned status {}: {}", status, body),
            ));
        }

        let audio_bytes = resp
            .bytes()
            .await
            .map_err(|e| ToolError::new("HTTP_ERROR", format!("Failed to read audio: {}", e)))?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(&audio_bytes);

        let data = serde_json::json!({
            "audio_format": "mp3",
            "audio_size_bytes": audio_bytes.len(),
            "audio_base64": encoded,
            "voice": voice,
        });

        Ok(ToolOutput::success(format!(
            "Generated {} bytes of audio data",
            audio_bytes.len()
        ))
        .with_data(data)
        .with_mime_type("audio/mpeg"))
    }
}

/// Tool for audio transcription via a speech-to-text API.
///
/// Sends an audio file to an OpenAI-Whisper-compatible transcription endpoint
/// and returns the transcribed text.
pub struct TranscriptionTool {
    /// API key for the transcription service.
    api_key: Option<String>,
    /// Base URL for the transcription API.
    api_url: String,
    /// Allowed base directory for audio files.
    allowed_base: PathBuf,
}

impl TranscriptionTool {
    /// Create a new transcription tool.
    pub fn new(allowed_base: PathBuf, api_key: Option<String>) -> Self {
        Self {
            api_key,
            api_url: "https://api.openai.com/v1/audio/transcriptions".to_string(),
            allowed_base,
        }
    }

    /// Set a custom API base URL.
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new(
                "PATH_INVALID",
                format!("Cannot access path '{}': {}", path_str, e),
            )
        })?;
        if !canonical.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!("Path '{}' is outside the allowed base", path_str),
            ));
        }
        Ok(canonical)
    }
}

#[async_trait]
impl Tool for TranscriptionTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "transcribe_audio",
                concat!(
                    "Transcribe an audio file to text using an OpenAI-Whisper-compatible API. ",
                    "Supports mp3, mp4, mpeg, mpga, m4a, wav, and webm audio files.",
                ),
                HashMap::from([
                    (
                        "path".to_string(),
                        ParameterDefinition::required_string("Path to the audio file"),
                    ),
                    (
                        "language".to_string(),
                        ParameterDefinition::string(
                            "Optional language hint (ISO-639-1, e.g. 'en')",
                        ),
                    ),
                    (
                        "prompt".to_string(),
                        ParameterDefinition::string("Optional prompt to guide the transcription"),
                    ),
                ]),
            )
            .category("media")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'path' parameter"))?;

        let path = self.resolve_path(path_str)?;

        if !path.is_file() {
            return Err(ToolError::new(
                "NOT_A_FILE",
                format!("'{}' is not a file", path.display()),
            ));
        }

        let metadata = std::fs::metadata(&path).map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read file metadata: {}", e))
        })?;
        if metadata.len() > 25 * 1024 * 1024 {
            return Err(ToolError::new(
                "FILE_TOO_LARGE",
                format!("Audio file too large: {} bytes (max 25 MB)", metadata.len()),
            ));
        }

        let api_key = self.api_key.as_deref().ok_or_else(|| {
            ToolError::new(
                "CONFIG_ERROR",
                "Transcription API key is not configured".to_string(),
            )
        })?;

        let language = params["language"].as_str().unwrap_or("");
        let prompt = params["prompt"].as_str().unwrap_or("");

        // Read the file bytes.
        let audio_bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read audio file: {}", e)))?;

        // Determine the mime type from the file extension.
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let mime_type = match ext.as_str() {
            "mp3" => "audio/mpeg",
            "mp4" => "audio/mp4",
            "mpeg" => "audio/mpeg",
            "mpga" => "audio/mpeg",
            "m4a" => "audio/m4a",
            "wav" => "audio/wav",
            "webm" => "audio/webm",
            "ogg" => "audio/ogg",
            _ => "audio/mpeg",
        };

        // Build a multipart request.
        let client = reqwest::Client::new();
        let part = reqwest::multipart::Part::bytes(audio_bytes)
            .file_name(
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            )
            .mime_str(mime_type)
            .map_err(|e| {
                ToolError::new("HTTP_ERROR", format!("Failed to build multipart: {}", e))
            })?;

        let mut form = reqwest::multipart::Form::new().part("file", part);
        form = form.text("model", "whisper-1");
        if !language.is_empty() {
            form = form.text("language", language.to_string());
        }
        if !prompt.is_empty() {
            form = form.text("prompt", prompt.to_string());
        }

        let resp = client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                ToolError::new("HTTP_ERROR", format!("Transcription request failed: {}", e))
            })?;

        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::new(
                "TRANSCRIPTION_ERROR",
                format!("API returned status {}: {}", status, body),
            ));
        }

        let body: serde_json::Value = resp.json().await.map_err(|e| {
            ToolError::new("HTTP_ERROR", format!("Failed to parse response: {}", e))
        })?;

        let text = body["text"].as_str().unwrap_or("").to_string();

        let data = serde_json::json!({
            "path": path_str,
            "language": language,
            "duration_seconds": null,
            "transcription": text,
        });

        Ok(ToolOutput::success(text).with_data(data))
    }
}

/// Combined media tool that dispatches to image, pdf, or tts sub-tools.
pub struct MediaTool {
    image: ImageTool,
    pdf: PdfTool,
    tts: TtsTool,
}

impl MediaTool {
    pub fn new(allowed_base: PathBuf, tts_api_key: Option<String>) -> Self {
        Self {
            image: ImageTool::new(allowed_base.clone()),
            pdf: PdfTool::new(allowed_base),
            tts: TtsTool::new(tts_api_key),
        }
    }
}

#[async_trait]
impl Tool for MediaTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "media",
                "Process media files: images (info, resize, convert), PDFs (text extraction), and TTS (text-to-speech).",
                HashMap::from([
                    (
                        "type".to_string(),
                        ParameterDefinition::required_string("The media type")
                            .enum_values(vec!["image".to_string(), "pdf".to_string(), "tts".to_string()]),
                    ),
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The operation to perform"),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::string("Path to the media file (for image and pdf)"),
                    ),
                    (
                        "text".to_string(),
                        ParameterDefinition::string("Text to convert to speech (for tts)"),
                    ),
                ]),
            )
            .category("media")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let media_type = params["type"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'type' parameter"))?;

        match media_type {
            "image" => self.image.execute(params).await,
            "pdf" => self.pdf.execute(params).await,
            "tts" => self.tts.execute(params).await,
            other => Err(ToolError::invalid_args(format!(
                "Unknown media type: {}",
                other
            ))),
        }
    }
}
