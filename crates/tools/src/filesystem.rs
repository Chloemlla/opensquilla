//! Filesystem tools: read_file, write_file, edit_file.
//!
//! Provides safe file I/O operations with path traversal protection via
//! canonicalization. All paths are resolved relative to an allowed base
//! directory to prevent directory traversal attacks.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Tool for filesystem operations (read, write, edit, list).
pub struct FilesystemTool {
    /// Base directory that all paths must be under.
    allowed_base: PathBuf,
    /// Maximum file size in bytes for reading.
    max_read_size: u64,
    /// Maximum file size in bytes for writing.
    max_write_size: u64,
}

impl FilesystemTool {
    /// Create a new filesystem tool with the given allowed base directory.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self {
            allowed_base,
            max_read_size: 10 * 1024 * 1024, // 10 MB
            max_write_size: 1 * 1024 * 1024, // 1 MB
        }
    }

    /// Resolve and validate a path against the allowed base directory.
    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);

        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };

        // Canonicalize to resolve symlinks and '..' components.
        let canonical = resolved.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                // For new files (write), canonicalize the parent directory.
                if let Some(parent) = resolved.parent() {
                    let parent_canonical = parent.canonicalize().map_err(|_| {
                        ToolError::new(
                            "PATH_INVALID",
                            format!("Cannot access parent directory of '{}'", path_str),
                        )
                    })?;
                    return Ok(parent_canonical.join(resolved.file_name().unwrap_or_default()));
                }
            }
            ToolError::new(
                "PATH_INVALID",
                format!("Cannot access path '{}': {}", path_str, e),
            )
        })?;

        // Check traversal protection: must be within allowed_base.
        if !canonical.starts_with(&self.allowed_base) {
            return Err(ToolError::new(
                "PATH_TRAVERSAL",
                format!(
                    "Path '{}' resolves outside the allowed base directory",
                    path_str
                ),
            ));
        }

        Ok(canonical)
    }

    /// Read a file and return its contents.
    async fn read_file(&self, path: &Path) -> ToolResult<ToolOutput> {
        let metadata = fs::metadata(path).await.map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read file metadata: {}", e))
        })?;

        if !metadata.is_file() {
            return Err(ToolError::new(
                "NOT_A_FILE",
                format!("Path '{}' is not a file", path.display()),
            ));
        }

        if metadata.len() > self.max_read_size {
            return Err(ToolError::new(
                "FILE_TOO_LARGE",
                format!(
                    "File too large: {} bytes (max {})",
                    metadata.len(),
                    self.max_read_size
                ),
            ));
        }

        let content = fs::read_to_string(path)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read file: {}", e)))?;

        let data = serde_json::json!({
            "size": metadata.len(),
            "path": path.to_string_lossy(),
        });

        Ok(ToolOutput::success(content).with_data(data))
    }

    /// Write content to a file, creating parent directories as needed.
    async fn write_file(&self, path: &Path, content: &str) -> ToolResult<ToolOutput> {
        let byte_len = content.len() as u64;
        if byte_len > self.max_write_size {
            return Err(ToolError::new(
                "CONTENT_TOO_LARGE",
                format!(
                    "Content too large: {} bytes (max {})",
                    byte_len, self.max_write_size
                ),
            ));
        }

        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
            })?;
        }

        fs::write(path, content)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to write file: {}", e)))?;

        tracing::info!(target = "tools", path = %path.display(), bytes = byte_len, "File written");

        let data = serde_json::json!({
            "size": byte_len,
            "path": path.to_string_lossy(),
        });

        Ok(
            ToolOutput::success(format!("Wrote {} bytes to {}", byte_len, path.display()))
                .with_data(data),
        )
    }

    /// Edit a file by applying line-based replacements.
    async fn edit_file(
        &self,
        path: &Path,
        old_string: &str,
        new_string: &str,
    ) -> ToolResult<ToolOutput> {
        let content = fs::read_to_string(path).await.map_err(|e| {
            ToolError::new(
                "IO_ERROR",
                format!("Failed to read file for editing: {}", e),
            )
        })?;

        if !content.contains(old_string) {
            return Err(ToolError::new(
                "STRING_NOT_FOUND",
                format!(
                    "The string to replace was not found in '{}'",
                    path.display()
                ),
            ));
        }

        let new_content = content.replace(old_string, new_string);

        fs::write(path, &new_content).await.map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to write edited file: {}", e))
        })?;

        let data = serde_json::json!({
            "path": path.to_string_lossy(),
            "replacements": 1,
        });

        Ok(
            ToolOutput::success(format!("Edited {}: replaced 1 occurrence", path.display()))
                .with_data(data),
        )
    }

    /// List directory contents.
    async fn list_dir(&self, path: &Path) -> ToolResult<ToolOutput> {
        let mut entries = Vec::new();
        let mut read_dir = fs::read_dir(path)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read directory: {}", e)))?;

        while let Some(entry) = read_dir.next_entry().await.map_err(|e| {
            ToolError::new("IO_ERROR", format!("Failed to read directory entry: {}", e))
        })? {
            let metadata = entry.metadata().await.ok();
            entries.push(serde_json::json!({
                "name": entry.file_name().to_string_lossy(),
                "is_dir": metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                "is_file": metadata.as_ref().map(|m| m.is_file()).unwrap_or(false),
                "size": metadata.as_ref().map(|m| m.len()).unwrap_or(0),
            }));
        }

        let data = serde_json::json!({
            "path": path.to_string_lossy(),
            "entries": entries,
            "count": entries.len(),
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }
}

#[async_trait]
impl Tool for FilesystemTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "filesystem",
                "Read, write, edit, and list files on the local filesystem. \
                 All paths are relative to the allowed working directory.",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string(
                            "The operation: read, write, edit, list",
                        )
                        .enum_values(vec![
                            "read".to_string(),
                            "write".to_string(),
                            "edit".to_string(),
                            "list".to_string(),
                        ]),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::required_string("Path to the file or directory"),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::string("Content to write (for write operation)"),
                    ),
                    (
                        "old_string".to_string(),
                        ParameterDefinition::string("String to replace (for edit operation)"),
                    ),
                    (
                        "new_string".to_string(),
                        ParameterDefinition::string("Replacement string (for edit operation)"),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?;

        let path_str = params["path"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'path' parameter"))?;

        let path = self.resolve_path(path_str)?;

        match operation {
            "read" => self.read_file(&path).await,
            "write" => {
                let content = params["content"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'content' for write operation")
                })?;
                self.write_file(&path, content).await
            }
            "edit" => {
                let old_string = params["old_string"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'old_string' for edit operation")
                })?;
                let new_string = params["new_string"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'new_string' for edit operation")
                })?;
                self.edit_file(&path, old_string, new_string).await
            }
            "list" => {
                if !path.is_dir() {
                    return Err(ToolError::new(
                        "NOT_A_DIRECTORY",
                        format!("Path '{}' is not a directory", path_str),
                    ));
                }
                self.list_dir(&path).await
            }
            other => Err(ToolError::invalid_args(format!(
                "Unknown operation: {}",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_and_read_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "write",
                "path": "test.txt",
                "content": "Hello, World!"
            }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("Wrote"));

        let result = tool
            .execute(serde_json::json!({
                "operation": "read",
                "path": "test.txt",
            }))
            .await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_edit_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        tool.execute(serde_json::json!({
            "operation": "write",
            "path": "edit_test.txt",
            "content": "Hello, World!"
        }))
        .await
        .unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "edit",
                "path": "edit_test.txt",
                "old_string": "World",
                "new_string": "Rust"
            }))
            .await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(dir.path().join("edit_test.txt")).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_path_traversal_denied() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "read",
                "path": "../etc/passwd",
            }))
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "PATH_TRAVERSAL");
    }

    #[tokio::test]
    async fn test_list_directory() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "list",
                "path": ".",
            }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("a.txt"));
    }
}
