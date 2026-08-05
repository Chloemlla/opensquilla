//! Session export.
//!
//! Exports a session's transcript and metadata to portable formats. Supports
//! JSON (lossless) and Markdown (human-readable) exports, plus a shared
//! directory-based export used for backups and debugging.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};

/// The export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportFormat {
    /// Lossless JSON document.
    Json,
    /// Human-readable Markdown transcript.
    Markdown,
    /// JSONL (one message per line) for tooling.
    Jsonl,
}

impl ExportFormat {
    /// Parse a format from a string.
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(ExportFormat::Json),
            "markdown" | "md" => Ok(ExportFormat::Markdown),
            "jsonl" => Ok(ExportFormat::Jsonl),
            other => Err(AppError::bad_request(format!(
                "Unknown export format '{other}'"
            ))),
        }
    }

    /// The file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Markdown => "md",
            ExportFormat::Jsonl => "jsonl",
        }
    }
}

/// Metadata attached to every export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMeta {
    pub session_id: String,
    pub exported_at: DateTime<Utc>,
    pub format: ExportFormat,
    pub message_count: usize,
}

/// A single message in an exported transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMessage {
    pub role: String,
    pub content: String,
    pub model: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub meta: Option<serde_json::Value>,
}

/// The full exported session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExport {
    pub meta: ExportMeta,
    pub session: serde_json::Value,
    pub messages: Vec<ExportedMessage>,
}

impl SessionExport {
    /// Serialize the export to bytes in the requested format.
    pub fn to_bytes(&self, format: ExportFormat) -> Result<Vec<u8>, AppError> {
        match format {
            ExportFormat::Json => {
                serde_json::to_vec_pretty(self)
                    .map_err(|e| AppError::internal(format!("Export serialization failed: {e}")))
            }
            ExportFormat::Jsonl => {
                let mut out = Vec::new();
                for msg in &self.messages {
                    let line = serde_json::to_string(msg)
                        .map_err(|e| AppError::internal(format!("Export serialization failed: {e}")))?;
                    out.extend_from_slice(line.as_bytes());
                    out.push(b'\n');
                }
                Ok(out)
            }
            ExportFormat::Markdown => Ok(self.to_markdown().into_bytes()),
        }
    }

    /// Render the export as a Markdown transcript.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("# Session {}\n\n", self.meta.session_id));
        out.push_str(&format!(
            "Exported at {}\n\n",
            self.meta.exported_at.to_rfc3339()
        ));
        for msg in &self.messages {
            out.push_str(&format!("## {}\n\n", msg.role));
            out.push_str(&msg.content);
            out.push_str("\n\n");
        }
        out
    }
}

/// Exports sessions to disk.
///
/// Each export is written as a file under the configured export directory,
/// named `<session_id>_<timestamp>.<ext>` so multiple exports of the same
/// session do not collide.
#[derive(Debug, Clone)]
pub struct SessionExporter {
    dir: PathBuf,
}

impl SessionExporter {
    /// Create an exporter rooted at the given directory, creating it if
    /// missing.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, AppError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| AppError::internal(format!("Cannot create export dir: {e}")))?;
        Ok(Self { dir })
    }

    /// Create an exporter using a default directory under the data local dir.
    pub fn default_path() -> Result<Self, AppError> {
        let base = dirs::data_local_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::new(base.join("exports"))
    }

    /// Build a [`SessionExport`] from a session record and message list.
    pub fn build_export(
        session_id: &str,
        session: serde_json::Value,
        messages: Vec<ExportedMessage>,
    ) -> SessionExport {
        SessionExport {
            meta: ExportMeta {
                session_id: session_id.to_string(),
                exported_at: Utc::now(),
                format: ExportFormat::Json,
                message_count: messages.len(),
            },
            session,
            messages,
        }
    }

    /// Write an export to disk in the given format. Returns the path written.
    pub fn write(
        &self,
        mut export: SessionExport,
        format: ExportFormat,
    ) -> Result<PathBuf, AppError> {
        export.meta.format = format;
        let bytes = export.to_bytes(format)?;
        let stem = format!(
            "{}_export_{}",
            sanitize_session_id(&export.meta.session_id),
            export.meta.exported_at.format("%Y%m%dT%H%M%S")
        );
        let path = self.dir.join(format!("{stem}.{}", format.extension()));
        std::fs::write(&path, bytes)
            .map_err(|e| AppError::internal(format!("Failed to write export: {e}")))?;
        Ok(path)
    }

    /// Return the export directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Sanitize a session id for use as a filename.
fn sanitize_session_id(id: &str) -> String {
    let mut safe = String::with_capacity(id.len());
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            safe.push(ch);
        } else {
            safe.push('_');
        }
    }
    safe
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla-export-test-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sample_messages() -> Vec<ExportedMessage> {
        vec![
            ExportedMessage {
                role: "user".into(),
                content: "Hello".into(),
                model: None,
                timestamp: None,
                meta: None,
            },
            ExportedMessage {
                role: "assistant".into(),
                content: "Hi there".into(),
                model: Some("gpt-4o".into()),
                timestamp: Some(Utc::now()),
                meta: Some(serde_json::json!({"tokens": 12})),
            },
        ]
    }

    #[test]
    fn test_format_parse() {
        assert_eq!(ExportFormat::parse("json").unwrap(), ExportFormat::Json);
        assert_eq!(
            ExportFormat::parse("markdown").unwrap(),
            ExportFormat::Markdown
        );
        assert_eq!(ExportFormat::parse("md").unwrap(), ExportFormat::Markdown);
        assert_eq!(ExportFormat::parse("jsonl").unwrap(), ExportFormat::Jsonl);
        assert!(ExportFormat::parse("xml").is_err());
    }

    #[test]
    fn test_export_to_markdown() {
        let export = SessionExporter::build_export("s1", serde_json::json!({}), sample_messages());
        let md = export.to_markdown();
        assert!(md.contains("# Session s1"));
        assert!(md.contains("## user"));
        assert!(md.contains("Hi there"));
    }

    #[test]
    fn test_export_json_roundtrip() {
        let export = SessionExporter::build_export("s1", serde_json::json!({}), sample_messages());
        let bytes = export.to_bytes(ExportFormat::Json).unwrap();
        let parsed: SessionExport = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.meta.message_count, 2);
        assert_eq!(parsed.messages[1].role, "assistant");
    }

    #[test]
    fn test_export_jsonl() {
        let export = SessionExporter::build_export("s1", serde_json::json!({}), sample_messages());
        let bytes = export.to_bytes(ExportFormat::Jsonl).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn test_write_export_to_disk() {
        let dir = temp_dir("write");
        let exporter = SessionExporter::new(&dir).unwrap();
        let export = SessionExporter::build_export("s1", serde_json::json!({}), sample_messages());
        let path = exporter.write(export, ExportFormat::Json).unwrap();
        assert!(path.exists());
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("json"));
        let md = exporter
            .write(
                SessionExporter::build_export("s1", serde_json::json!({}), sample_messages()),
                ExportFormat::Markdown,
            )
            .unwrap();
        assert!(md.exists());
        assert_eq!(md.extension().and_then(|e| e.to_str()), Some("md"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_missing_dir_created() {
        let base = temp_dir("nested");
        let dir = base.join("a").join("b");
        let exporter = SessionExporter::new(&dir).unwrap();
        assert!(dir.is_dir());
        std::fs::remove_dir_all(&base).ok();
    }
}
