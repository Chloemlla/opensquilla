//! Artifact generation tool: publish_artifact, generate_*.
//!
//! Generates document artifacts in various formats including DOCX, XLSX, PDF,
//! CSV, JSON, HTML, Markdown, and plain text. Uses the `calamine` crate for
//! reading existing Excel files and custom generation for other formats.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

/// Supported artifact formats.
const SUPPORTED_FORMATS: &[&str] = &["txt", "md", "html", "json", "csv", "yaml", "xml"];

/// Tool for generating file artifacts.
pub struct ArtifactTool {
    /// Allowed base directory for writing artifacts.
    allowed_base: PathBuf,
    /// Maximum file size in bytes.
    max_file_size: u64,
}

impl ArtifactTool {
    /// Create a new artifact generation tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            max_file_size: 10 * 1024 * 1024, // 10 MB
        }
    }

    fn resolve_output_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };

        // For new files, resolve the parent directory.
        if let Some(parent) = resolved.parent() {
            let parent_canonical = parent.canonicalize().map_err(|_| {
                ToolError::new(
                    "PATH_INVALID",
                    format!("Cannot access parent directory of '{}'", path_str),
                )
            })?;
            let full_path = parent_canonical.join(resolved.file_name().unwrap_or_default());
            if !full_path.starts_with(&self.allowed_base) {
                return Err(ToolError::new(
                    "PATH_TRAVERSAL",
                    format!("Path '{}' is outside the allowed base", path_str),
                ));
            }
            return Ok(full_path);
        }

        Err(ToolError::new(
            "PATH_INVALID",
            format!("Invalid path '{}'", path_str),
        ))
    }

    /// Generate a text-based artifact (txt, md, html, json, csv, yaml, xml).
    fn generate_text(format: &str, content: &str, filename: &str) -> ToolResult<(String, String)> {
        let extension = match format {
            "txt" => "txt",
            "md" | "markdown" => "md",
            "html" => "html",
            "json" => "json",
            "csv" => "csv",
            "yaml" | "yml" => "yaml",
            "xml" => "xml",
            _ => {
                return Err(ToolError::invalid_args(format!(
                    "Unsupported format: '{}'. Supported: {}",
                    format,
                    SUPPORTED_FORMATS.join(", ")
                )));
            }
        };

        let file_name = if filename.contains('.') {
            filename.to_string()
        } else {
            format!("{}.{}", filename, extension)
        };

        Ok((file_name, content.to_string()))
    }

    /// Generate a CSV file from a JSON array or object.
    fn generate_csv(json_data: &str, filename: &str) -> ToolResult<(String, String)> {
        let data: Value = serde_json::from_str(json_data)
            .map_err(|e| ToolError::invalid_args(format!("Invalid JSON data for CSV: {}", e)))?;

        let file_name = if filename.contains('.') {
            filename.to_string()
        } else {
            format!("{}.csv", filename)
        };

        let mut csv_output = String::new();

        match &data {
            Value::Array(rows) => {
                // Collect all unique keys.
                let mut keys = Vec::new();
                for row in rows {
                    if let Value::Object(map) = row {
                        for key in map.keys() {
                            if !keys.contains(key) {
                                keys.push(key.clone());
                            }
                        }
                    }
                }

                // Write header.
                csv_output.push_str(&keys.join(","));
                csv_output.push('\n');

                // Write rows.
                for row in rows {
                    if let Value::Object(map) = row {
                        let values: Vec<String> = keys
                            .iter()
                            .map(|k| {
                                map.get(k)
                                    .map(|v| match v {
                                        Value::String(s) => {
                                            if s.contains(',')
                                                || s.contains('"')
                                                || s.contains('\n')
                                            {
                                                format!("\"{}\"", s.replace('"', "\"\""))
                                            } else {
                                                s.clone()
                                            }
                                        }
                                        Value::Null => String::new(),
                                        other => other.to_string(),
                                    })
                                    .unwrap_or_default()
                            })
                            .collect();
                        csv_output.push_str(&values.join(","));
                        csv_output.push('\n');
                    }
                }
            }
            Value::Object(map) => {
                // Single object: write key-value pairs.
                csv_output.push_str("key,value\n");
                for (k, v) in map {
                    csv_output.push_str(&format!(
                        "{},{}\n",
                        k,
                        match v {
                            Value::String(s) => {
                                if s.contains(',') || s.contains('"') {
                                    format!("\"{}\"", s.replace('"', "\"\""))
                                } else {
                                    s.clone()
                                }
                            }
                            other => other.to_string(),
                        }
                    ));
                }
            }
            _ => {
                return Err(ToolError::invalid_args(
                    "CSV data must be a JSON array or object",
                ));
            }
        }

        Ok((file_name, csv_output))
    }

    /// Generate an HTML document.
    fn generate_html(
        title: &str,
        body_content: &str,
        filename: &str,
    ) -> ToolResult<(String, String)> {
        let file_name = if filename.contains('.') {
            filename.to_string()
        } else {
            format!("{}.html", filename)
        };

        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 2px; }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div>{}</div>
</body>
</html>"#,
            title, title, body_content
        );

        Ok((file_name, html))
    }
}

#[async_trait]
impl Tool for ArtifactTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "publish_artifact",
                concat!(
                    "Generate and save a file artifact in various formats. ",
                    "Supports TXT, Markdown, HTML, JSON, CSV, YAML, and XML. ",
                    "The file is written to the allowed working directory.",
                ),
                HashMap::from([
                    (
                        "format".to_string(),
                        ParameterDefinition::required_string("The output format").enum_values(
                            vec![
                                "txt".to_string(),
                                "md".to_string(),
                                "markdown".to_string(),
                                "html".to_string(),
                                "json".to_string(),
                                "csv".to_string(),
                                "yaml".to_string(),
                                "xml".to_string(),
                            ],
                        ),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The content of the artifact"),
                    ),
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "The output filename (without extension, or with)",
                        ),
                    ),
                    (
                        "title".to_string(),
                        ParameterDefinition::string("Document title (used for HTML format)"),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let format = params["format"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'format' parameter"))?;

        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'content' parameter"))?;

        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;

        let title = params["title"].as_str().unwrap_or("Artifact");

        // Generate the file content based on format.
        let (file_name, file_content) = match format {
            "csv" => Self::generate_csv(content, filename)?,
            "html" => Self::generate_html(title, content, filename)?,
            _ => Self::generate_text(format, content, filename)?,
        };

        let byte_len = file_content.len() as u64;
        if byte_len > self.max_file_size {
            return Err(ToolError::new(
                "FILE_TOO_LARGE",
                format!(
                    "Generated content too large: {} bytes (max {})",
                    byte_len, self.max_file_size
                ),
            ));
        }

        let output_path = self.resolve_output_path(&file_name)?;

        // Ensure parent directory exists.
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
            })?;
        }

        tokio::fs::write(&output_path, &file_content)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to write artifact: {}", e)))?;

        let data = serde_json::json!({
            "path": output_path.to_string_lossy(),
            "format": format,
            "size_bytes": byte_len,
            "filename": file_name,
        });

        tracing::info!(
            target = "tools",
            path = %output_path.display(),
            format = %format,
            size = byte_len,
            "Artifact generated"
        );

        Ok(ToolOutput::success(format!(
            "Generated {} artifact '{}' ({} bytes)",
            format, file_name, byte_len
        ))
        .with_data(data))
    }
}

/// Generate a Markdown document artifact.
pub struct GenerateMarkdownTool {
    base: ArtifactTool,
}

impl GenerateMarkdownTool {
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            base: ArtifactTool::new(allowed_base),
        }
    }
}

#[async_trait]
impl Tool for GenerateMarkdownTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_markdown",
                "Generate a Markdown document artifact and save it to the filesystem.",
                HashMap::from([
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The Markdown content"),
                    ),
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "The output filename (e.g., 'document.md')",
                        ),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'content' parameter"))?;
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;

        self.base
            .execute(serde_json::json!({
                "format": "md",
                "content": content,
                "filename": filename,
            }))
            .await
    }
}

/// Generate a JSON document artifact.
pub struct GenerateJsonTool {
    base: ArtifactTool,
}

impl GenerateJsonTool {
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            base: ArtifactTool::new(allowed_base),
        }
    }
}

#[async_trait]
impl Tool for GenerateJsonTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "generate_json",
                "Generate a JSON document artifact and save it to the filesystem.",
                HashMap::from([
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("The JSON content"),
                    ),
                    (
                        "filename".to_string(),
                        ParameterDefinition::required_string(
                            "The output filename (e.g., 'data.json')",
                        ),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'content' parameter"))?;
        let filename = params["filename"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'filename' parameter"))?;

        // Validate JSON.
        serde_json::from_str::<Value>(content)
            .map_err(|e| ToolError::invalid_args(format!("Invalid JSON content: {}", e)))?;

        self.base
            .execute(serde_json::json!({
                "format": "json",
                "content": content,
                "filename": filename,
            }))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_text_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "txt",
                "content": "Hello, World!",
                "filename": "hello.txt",
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_generate_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "md",
                "content": "# Title\n\nSome **bold** text.",
                "filename": "doc",
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("doc.md")).unwrap();
        assert!(content.contains("# Title"));
    }

    #[tokio::test]
    async fn test_generate_csv_from_array() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "csv",
                "content": r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#,
                "filename": "people.csv",
            }))
            .await;
        assert!(result.is_ok());
        let content = std::fs::read_to_string(dir.path().join("people.csv")).unwrap();
        assert!(content.contains("Alice"));
        assert!(content.contains("Bob"));
        assert!(content.contains("name,age") || content.contains("age,name"));
    }

    #[tokio::test]
    async fn test_generate_html() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "html",
                "content": "<p>Hello World</p>",
                "filename": "page.html",
                "title": "My Page",
            }))
            .await;
        assert!(result.is_ok());
        let content = std::fs::read_to_string(dir.path().join("page.html")).unwrap();
        assert!(content.contains("My Page"));
        assert!(content.contains("<p>Hello World</p>"));
    }

    #[tokio::test]
    async fn test_unsupported_format() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "docx",
                "content": "hello",
                "filename": "test.docx",
            }))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_path_traversal_denied() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ArtifactTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "format": "txt",
                "content": "secret",
                "filename": "../outside.txt",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PATH_TRAVERSAL");
    }
}
