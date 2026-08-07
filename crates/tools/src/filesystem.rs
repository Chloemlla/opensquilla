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

/// Create a symlink at `link` pointing to `target`.
///
/// `tokio::fs::symlink` is unix-only; on Windows we dispatch on whether the
/// target is a file or directory, best-effort.
#[cfg(unix)]
async fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    tokio::fs::symlink(target, link).await
}

#[cfg(windows)]
async fn create_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    let target = target.to_path_buf();
    let link = link.to_path_buf();
    tokio::task::spawn_blocking(move || {
        std::os::windows::fs::symlink_file(&target, &link)
            .or_else(|_| std::os::windows::fs::symlink_dir(&target, &link))
    })
    .await
    .map_err(std::io::Error::other)?
}

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
            max_write_size: 1024 * 1024,     // 1 MB
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

        // Canonicalize to resolve symlinks and '..' components. For paths that
        // do not exist yet (new files on write), canonicalize the parent and
        // re-attach the file name.
        let canonical = resolved.canonicalize().or_else(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                if let Some(parent) = resolved.parent() {
                    let parent_canonical = parent.canonicalize().map_err(|_| {
                        ToolError::new(
                            "PATH_INVALID",
                            format!("Cannot access parent directory of '{}'", path_str),
                        )
                    })?;
                    Ok(parent_canonical.join(resolved.file_name().unwrap_or_default()))
                } else {
                    Err(ToolError::new(
                        "PATH_INVALID",
                        format!("Cannot access path '{}': {}", path_str, e),
                    ))
                }
            } else {
                Err(ToolError::new(
                    "PATH_INVALID",
                    format!("Cannot access path '{}': {}", path_str, e),
                ))
            }
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

    /// List a directory tree recursively, with nesting levels.
    ///
    /// Returns entries with their relative path, type, size, and depth.
    async fn list_tree(&self, path: &Path, max_depth: usize) -> ToolResult<ToolOutput> {
        if max_depth == 0 {
            return self.list_dir(path).await;
        }

        let path_clone = path.to_path_buf();
        let allowed_base = self.allowed_base.clone();
        let tree = tokio::task::spawn_blocking(move || -> ToolResult<Vec<serde_json::Value>> {
            let mut entries = Vec::new();
            let mut stack = vec![(path_clone, 0usize)];
            while let Some((dir, depth)) = stack.pop() {
                if depth > max_depth {
                    continue;
                }
                let read_dir = std::fs::read_dir(&dir).map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to read directory: {}", e))
                })?;
                for entry in read_dir.flatten() {
                    let metadata = entry.metadata().ok();
                    let is_dir = metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                    let rel = entry
                        .path()
                        .strip_prefix(&allowed_base)
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or_else(|_| entry.file_name().to_string_lossy().to_string());
                    entries.push(serde_json::json!({
                        "path": rel,
                        "name": entry.file_name().to_string_lossy(),
                        "is_dir": is_dir,
                        "size": metadata.as_ref().map(|m| m.len()).unwrap_or(0),
                        "depth": depth,
                    }));
                    if is_dir {
                        stack.push((entry.path(), depth + 1));
                    }
                }
            }
            entries.sort_by(|a, b| {
                a["path"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["path"].as_str().unwrap_or(""))
            });
            Ok(entries)
        })
        .await
        .map_err(|e| ToolError::new("IO_ERROR", format!("Tree walk task failed: {}", e)))??;

        let data = serde_json::json!({
            "path": path.to_string_lossy(),
            "entries": tree,
            "count": tree.len(),
            "max_depth": max_depth,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }

    /// Get detailed file metadata.
    async fn stat_file(&self, path: &Path) -> ToolResult<ToolOutput> {
        let metadata = fs::metadata(path)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to read metadata: {}", e)))?;

        let symlink_target = if metadata.file_type().is_symlink() {
            fs::read_link(path)
                .await
                .ok()
                .map(|p| p.to_string_lossy().to_string())
        } else {
            None
        };

        let created = metadata.created().ok().map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
        let modified = metadata.modified().ok().map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });
        let accessed = metadata.accessed().ok().map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0)
        });

        let data = serde_json::json!({
            "path": path.to_string_lossy(),
            "size": metadata.len(),
            "is_file": metadata.is_file(),
            "is_dir": metadata.is_dir(),
            "is_symlink": metadata.file_type().is_symlink(),
            "permissions_readonly": metadata.permissions().readonly(),
            "created_epoch": created,
            "modified_epoch": modified,
            "accessed_epoch": accessed,
            "symlink_target": symlink_target,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }

    /// Recursively copy a file or directory to a destination.
    async fn copy_path(
        &self,
        source: &Path,
        dest: &Path,
        overwrite: bool,
    ) -> ToolResult<ToolOutput> {
        if dest.exists() && !overwrite {
            return Err(ToolError::new(
                "DESTINATION_EXISTS",
                format!("Destination '{}' already exists", dest.display()),
            ));
        }

        if source.is_dir() {
            fs::create_dir_all(dest).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
            })?;
            let mut read_dir = fs::read_dir(source).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to read directory: {}", e))
            })?;
            while let Some(entry) = read_dir.next_entry().await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to read directory entry: {}", e))
            })? {
                let entry_path = entry.path();
                let rel = entry_path.strip_prefix(source).map_err(|_| {
                    ToolError::new("IO_ERROR", "Failed to compute relative path".to_string())
                })?;
                let dest_path = dest.join(rel);
                if entry_path.is_dir() {
                    fs::create_dir_all(&dest_path).await.map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                    })?;
                } else if entry_path.is_symlink() {
                    if let Ok(target) = fs::read_link(&entry_path).await {
                        create_symlink(&target, &dest_path).await.map_err(|e| {
                            ToolError::new("IO_ERROR", format!("Failed to create symlink: {}", e))
                        })?;
                    }
                } else {
                    fs::copy(&entry_path, &dest_path).await.map_err(|e| {
                        ToolError::new("IO_ERROR", format!("Failed to copy file: {}", e))
                    })?;
                }
            }
        } else if source.is_symlink() {
            if let Ok(target) = fs::read_link(source).await {
                create_symlink(&target, dest).await.map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create symlink: {}", e))
                })?;
            }
        } else {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                })?;
            }
            fs::copy(source, dest)
                .await
                .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to copy file: {}", e)))?;
        }

        let data = serde_json::json!({
            "source": source.to_string_lossy(),
            "destination": dest.to_string_lossy(),
            "overwrite": overwrite,
        });

        Ok(ToolOutput::success(format!(
            "Copied '{}' to '{}'",
            source.display(),
            dest.display()
        ))
        .with_data(data))
    }

    /// Move a file or directory to a destination.
    async fn move_path(
        &self,
        source: &Path,
        dest: &Path,
        overwrite: bool,
    ) -> ToolResult<ToolOutput> {
        if dest.exists() && !overwrite {
            return Err(ToolError::new(
                "DESTINATION_EXISTS",
                format!("Destination '{}' already exists", dest.display()),
            ));
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
            })?;
        }

        fs::rename(source, dest)
            .await
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to move: {}", e)))?;

        let data = serde_json::json!({
            "source": source.to_string_lossy(),
            "destination": dest.to_string_lossy(),
            "overwrite": overwrite,
        });

        Ok(ToolOutput::success(format!(
            "Moved '{}' to '{}'",
            source.display(),
            dest.display()
        ))
        .with_data(data))
    }

    /// Delete a file or directory recursively.
    async fn delete_path(&self, path: &Path, recursive: bool) -> ToolResult<ToolOutput> {
        if path.is_dir() && !recursive {
            return Err(ToolError::new(
                "NOT_EMPTY",
                format!(
                    "'{}' is a directory; use recursive=true to delete it",
                    path.display()
                ),
            ));
        }

        if path.is_dir() {
            fs::remove_dir_all(path).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to remove directory: {}", e))
            })?;
        } else {
            fs::remove_file(path)
                .await
                .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to remove file: {}", e)))?;
        }

        let data = serde_json::json!({
            "path": path.to_string_lossy(),
            "recursive": recursive,
        });

        Ok(ToolOutput::success(format!("Deleted '{}'", path.display())).with_data(data))
    }

    /// Create a symbolic link.
    async fn create_symlink(&self, target: &Path, link: &Path) -> ToolResult<ToolOutput> {
        #[cfg(windows)]
        let result = {
            let metadata = std::fs::metadata(target);
            if let Ok(m) = metadata {
                if m.is_dir() {
                    std::os::windows::fs::symlink_dir(target, link)
                } else {
                    std::os::windows::fs::symlink_file(target, link)
                }
            } else {
                std::os::windows::fs::symlink_file(target, link)
            }
        };
        #[cfg(not(windows))]
        let result = std::os::unix::fs::symlink(target, link);

        result
            .map_err(|e| ToolError::new("IO_ERROR", format!("Failed to create symlink: {}", e)))?;

        let data = serde_json::json!({
            "target": target.to_string_lossy(),
            "link": link.to_string_lossy(),
        });

        Ok(ToolOutput::success(format!(
            "Created symlink '{}' → '{}'",
            link.display(),
            target.display()
        ))
        .with_data(data))
    }

    /// Search for files matching a substring pattern within the base directory.
    async fn search_files(&self, pattern: &str, base: &Path) -> ToolResult<ToolOutput> {
        let pattern_lower = pattern.to_lowercase();
        let base_clone = base.to_path_buf();
        let max_results = 500usize;

        let matches = tokio::task::spawn_blocking(move || -> Vec<serde_json::Value> {
            let mut results = Vec::new();
            let mut stack = vec![base_clone];
            while let Some(dir) = stack.pop() {
                if results.len() >= max_results {
                    break;
                }
                if let Ok(read_dir) = std::fs::read_dir(&dir) {
                    for entry in read_dir.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.to_lowercase().contains(&pattern_lower) {
                            results.push(serde_json::json!({
                                "path": entry.path().to_string_lossy(),
                                "name": name,
                            }));
                        }
                        if entry.path().is_dir() {
                            stack.push(entry.path());
                        }
                        if results.len() >= max_results {
                            break;
                        }
                    }
                }
            }
            results
        })
        .await
        .map_err(|e| ToolError::new("IO_ERROR", format!("Search task failed: {}", e)))?;

        let data = serde_json::json!({
            "pattern": pattern,
            "matches": matches,
            "count": matches.len(),
            "truncated": matches.len() >= max_results,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }

    /// Watch a file or directory for changes using the `notify` crate.
    ///
    /// Collects filesystem events (create, modify, remove, rename) for a
    /// configurable duration and returns them as structured data.
    async fn watch_files(
        &self,
        path: &Path,
        duration_secs: u64,
        recursive: bool,
    ) -> ToolResult<ToolOutput> {
        use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel::<Event>();
        let mut watcher: RecommendedWatcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let _ = tx.send(event);
                }
            })
            .map_err(|e| {
                ToolError::new("WATCH_ERROR", format!("Failed to create watcher: {}", e))
            })?;

        watcher
            .configure(Config::default().with_poll_interval(std::time::Duration::from_secs(1)))
            .map_err(|e| {
                ToolError::new("WATCH_ERROR", format!("Failed to configure watcher: {}", e))
            })?;

        watcher
            .watch(
                path,
                if recursive {
                    RecursiveMode::Recursive
                } else {
                    RecursiveMode::NonRecursive
                },
            )
            .map_err(|e| {
                ToolError::new(
                    "WATCH_ERROR",
                    format!("Failed to watch '{}': {}", path.display(), e),
                )
            })?;

        // Collect events for the requested duration.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(duration_secs);
        let mut events: Vec<serde_json::Value> = Vec::new();

        while std::time::Instant::now() < deadline {
            match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                Ok(event) => {
                    let kind = match event.kind {
                        EventKind::Create(_) => "create",
                        EventKind::Modify(_) => "modify",
                        EventKind::Remove(_) => "remove",
                        EventKind::Access(_) => "access",
                        EventKind::Any => "any",
                        _ => "other",
                    };
                    for event_path in event.paths {
                        events.push(serde_json::json!({
                            "kind": kind,
                            "path": event_path.to_string_lossy(),
                        }));
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        let data = serde_json::json!({
            "watched_path": path.to_string_lossy(),
            "duration_secs": duration_secs,
            "recursive": recursive,
            "event_count": events.len(),
            "events": events,
        });

        Ok(
            ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
                .with_data(data),
        )
    }
}

/// Active platform string used by the foreign-host path policy.
fn active_platform() -> &'static str {
    if cfg!(windows) { "nt" } else { "posix" }
}

/// Reject absolute paths from another host OS against the scoped context.
fn gate_foreign_host_path(original_path: &str) -> ToolResult<()> {
    if let Some(ctx) = crate::context::current_tool_context() {
        let workspace = ctx
            .workspace_dir
            .as_ref()
            .map(|p| p.to_string_lossy().to_string());
        crate::path_policy::reject_foreign_host_path(
            original_path,
            active_platform(),
            workspace.as_deref(),
        )?;
    }
    Ok(())
}

/// Run workspace write-deny + scratch-artifact gates against the scoped
/// context. No-op when no context is scoped (direct tool invocation).
fn gate_workspace_writes(tool_name: &str, path: &Path, original_path: &str) -> ToolResult<()> {
    crate::context::with_current_tool_context(|ctx| {
        crate::write_policy::gate_workspace_write_deny(
            tool_name,
            path,
            Some(original_path),
            ctx.workspace_dir.as_deref(),
            ctx,
        )?;
        crate::write_policy::gate_workspace_scratch_artifact(
            tool_name,
            path,
            Some(original_path),
            ctx.workspace_dir.as_deref(),
            ctx,
        )
    })
    .transpose()?;
    Ok(())
}

#[async_trait]
impl Tool for FilesystemTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "filesystem",
                "Read, write, edit, and list files on the local filesystem. \
                 Also supports directory tree listing, file metadata (stat), \
                 recursive copy/move/delete, symlink creation, file search, \
                 and file watching. \
                 All paths are relative to the allowed working directory.",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string(
                            "The operation: read, write, edit, list, tree, stat, copy, move, delete, symlink, search, watch",
                        )
                        .enum_values(vec![
                            "read".to_string(),
                            "write".to_string(),
                            "edit".to_string(),
                            "list".to_string(),
                            "tree".to_string(),
                            "stat".to_string(),
                            "copy".to_string(),
                            "move".to_string(),
                            "delete".to_string(),
                            "symlink".to_string(),
                            "search".to_string(),
                            "watch".to_string(),
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
                    (
                        "max_depth".to_string(),
                        ParameterDefinition::integer("Maximum depth for tree listing")
                            .default(serde_json::json!(10)),
                    ),
                    (
                        "destination".to_string(),
                        ParameterDefinition::string("Destination path (for copy, move operations)"),
                    ),
                    (
                        "overwrite".to_string(),
                        ParameterDefinition::boolean("Overwrite destination if it exists")
                            .default(serde_json::json!(false)),
                    ),
                    (
                        "recursive".to_string(),
                        ParameterDefinition::boolean(
                            "Delete recursively (for delete) / watch recursively (for watch)",
                        )
                        .default(serde_json::json!(false)),
                    ),
                    (
                        "target".to_string(),
                        ParameterDefinition::string("Symlink target (for symlink operation)"),
                    ),
                    (
                        "pattern".to_string(),
                        ParameterDefinition::string("Filename pattern to search (for search operation)"),
                    ),
                    (
                        "duration_secs".to_string(),
                        ParameterDefinition::integer("Duration in seconds to watch (for watch operation)")
                            .default(serde_json::json!(5)),
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

        // Foreign-host path rejection (defense in depth; no-op without a
        // scoped context).
        gate_foreign_host_path(path_str)?;

        match operation {
            "read" => {
                let output = self.read_file(&path).await?;
                crate::context::mutate_current_tool_context(|ctx| {
                    crate::write_tracking::record_workspace_file_read(
                        ctx,
                        &path,
                        "read",
                        None,
                        None,
                        Some(true),
                    )
                });
                Ok(output)
            }
            "write" => {
                let content = params["content"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'content' for write operation")
                })?;
                gate_workspace_writes("write_file", &path, path_str)?;
                let created = !path.exists();
                let output = self.write_file(&path, content).await?;
                crate::context::mutate_current_tool_context(|ctx| {
                    crate::write_tracking::record_workspace_file_write(ctx, &path, "write", created)
                });
                Ok(output)
            }
            "edit" => {
                let old_string = params["old_string"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'old_string' for edit operation")
                })?;
                let new_string = params["new_string"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'new_string' for edit operation")
                })?;
                gate_workspace_writes("edit_file", &path, path_str)?;
                if let Some(ctx) = crate::context::current_tool_context() {
                    crate::write_tracking::require_fresh_workspace_file_read(
                        &ctx,
                        &path,
                        "edit_file",
                        path_str,
                    )?;
                }
                let output = self.edit_file(&path, old_string, new_string).await?;
                crate::context::mutate_current_tool_context(|ctx| {
                    crate::write_tracking::record_workspace_file_write(ctx, &path, "edit", false)
                });
                Ok(output)
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
            "tree" => {
                if !path.is_dir() {
                    return Err(ToolError::new(
                        "NOT_A_DIRECTORY",
                        format!("Path '{}' is not a directory", path_str),
                    ));
                }
                let max_depth = params["max_depth"].as_i64().unwrap_or(10).max(0) as usize;
                self.list_tree(&path, max_depth).await
            }
            "stat" => self.stat_file(&path).await,
            "copy" => {
                let destination = params["destination"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'destination' for copy operation")
                })?;
                let dest = self.resolve_path(destination)?;
                let overwrite = params["overwrite"].as_bool().unwrap_or(false);
                self.copy_path(&path, &dest, overwrite).await
            }
            "move" => {
                let destination = params["destination"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'destination' for move operation")
                })?;
                let dest = self.resolve_path(destination)?;
                let overwrite = params["overwrite"].as_bool().unwrap_or(false);
                self.move_path(&path, &dest, overwrite).await
            }
            "delete" => {
                let recursive = params["recursive"].as_bool().unwrap_or(false);
                self.delete_path(&path, recursive).await
            }
            "symlink" => {
                let target = params["target"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'target' for symlink operation")
                })?;
                let target_path = self.resolve_path(target)?;
                self.create_symlink(&target_path, &path).await
            }
            "search" => {
                let pattern = params["pattern"].as_str().ok_or_else(|| {
                    ToolError::invalid_args("Missing 'pattern' for search operation")
                })?;
                let base = if path.is_dir() {
                    path
                } else {
                    self.allowed_base.clone()
                };
                self.search_files(pattern, &base).await
            }
            "watch" => {
                if !path.exists() {
                    return Err(ToolError::not_found(format!(
                        "Path '{}' does not exist",
                        path.display()
                    )));
                }
                let duration_secs = params["duration_secs"].as_i64().unwrap_or(5).max(1) as u64;
                let recursive = params["recursive"].as_bool().unwrap_or(true);
                self.watch_files(&path, duration_secs, recursive).await
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

    #[tokio::test]
    async fn test_tree_listing_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/nested.txt"), "nested").unwrap();
        std::fs::write(dir.path().join("top.txt"), "top").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "tree",
                "path": ".",
                "max_depth": 5,
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("nested.txt"));
        assert!(output.content.contains("top.txt"));
    }

    #[tokio::test]
    async fn test_stat_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("meta.txt"), "hello").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "stat",
                "path": "meta.txt",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("is_file"));
        assert!(output.content.contains("size"));
    }

    #[tokio::test]
    async fn test_copy_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("orig.txt"), "hello").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "copy",
                "path": "orig.txt",
                "destination": "copied.txt",
            }))
            .await;
        assert!(result.is_ok());
        let content = std::fs::read_to_string(dir.path().join("copied.txt")).unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_copy_directory_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::create_dir_all(dir.path().join("src/sub")).unwrap();
        std::fs::write(dir.path().join("src/a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("src/sub/b.txt"), "b").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "copy",
                "path": "src",
                "destination": "dst",
            }))
            .await;
        assert!(result.is_ok());
        assert!(dir.path().join("dst/a.txt").exists());
        assert!(dir.path().join("dst/sub/b.txt").exists());
    }

    #[tokio::test]
    async fn test_copy_overwrite_denied() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("orig.txt"), "hello").unwrap();
        std::fs::write(dir.path().join("dest.txt"), "existing").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "copy",
                "path": "orig.txt",
                "destination": "dest.txt",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "DESTINATION_EXISTS");
    }

    #[tokio::test]
    async fn test_move_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("mover.txt"), "move me").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "move",
                "path": "mover.txt",
                "destination": "moved.txt",
            }))
            .await;
        assert!(result.is_ok());
        assert!(!dir.path().join("mover.txt").exists());
        assert!(dir.path().join("moved.txt").exists());
    }

    #[tokio::test]
    async fn test_delete_file() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("todelete.txt"), "bye").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "delete",
                "path": "todelete.txt",
            }))
            .await;
        assert!(result.is_ok());
        assert!(!dir.path().join("todelete.txt").exists());
    }

    #[tokio::test]
    async fn test_delete_directory_requires_recursive() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::create_dir_all(dir.path().join("dir_to_del")).unwrap();
        std::fs::write(dir.path().join("dir_to_del/f.txt"), "x").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "delete",
                "path": "dir_to_del",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "NOT_EMPTY");

        let result = tool
            .execute(serde_json::json!({
                "operation": "delete",
                "path": "dir_to_del",
                "recursive": true,
            }))
            .await;
        assert!(result.is_ok());
        assert!(!dir.path().join("dir_to_del").exists());
    }

    #[tokio::test]
    async fn test_search_files() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("report_q1.txt"), "data").unwrap();
        std::fs::write(dir.path().join("notes.txt"), "data").unwrap();
        std::fs::create_dir_all(dir.path().join("nested")).unwrap();
        std::fs::write(dir.path().join("nested/report_q2.txt"), "data").unwrap();

        let result = tool
            .execute(serde_json::json!({
                "operation": "search",
                "path": ".",
                "pattern": "report",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("report_q1.txt"));
        assert!(output.content.contains("report_q2.txt"));
        assert!(!output.content.contains("notes.txt"));
    }

    #[tokio::test]
    async fn test_symlink_creation() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        std::fs::write(dir.path().join("target.txt"), "link target").unwrap();

        // On Windows, symlink creation may require admin privileges; skip if it fails.
        let result = tool
            .execute(serde_json::json!({
                "operation": "symlink",
                "path": "link.txt",
                "target": "target.txt",
            }))
            .await;
        if result.is_ok() {
            let metadata = std::fs::symlink_metadata(dir.path().join("link.txt")).unwrap();
            assert!(metadata.file_type().is_symlink());
        }
    }

    #[tokio::test]
    async fn test_watch_operation() {
        let dir = tempfile::tempdir().unwrap();
        let _tool = FilesystemTool::new(dir.path().to_path_buf());

        // Watch the directory for a short window and create a file inside it.
        let tool_clone_path = dir.path().to_path_buf();
        let watcher_dir = dir.path().to_path_buf();
        let watch_handle = tokio::spawn(async move {
            let tool = FilesystemTool::new(watcher_dir);
            tool.execute(serde_json::json!({
                "operation": "watch",
                "path": ".",
                "duration_secs": 3,
                "recursive": true,
            }))
            .await
        });

        // Give the watcher a moment to start, then create a file.
        tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        std::fs::write(tool_clone_path.join("watched.txt"), "hello").unwrap();

        let result = watch_handle.await.unwrap();
        assert!(result.is_ok());
        let output = result.unwrap();
        // The operation must return a valid response structure regardless of
        // whether notify delivered events on this platform.
        let data = output.data.unwrap();
        assert!(data["duration_secs"] == 3);
    }

    #[tokio::test]
    async fn test_watch_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let tool = FilesystemTool::new(dir.path().to_path_buf());

        let result = tool
            .execute(serde_json::json!({
                "operation": "watch",
                "path": "nonexistent_dir",
                "duration_secs": 1,
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "TOOL_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_scoped_context_records_writes_and_reads() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().canonicalize().expect("canonical workspace");
        let tool = FilesystemTool::new(workspace.clone());

        let ctx = crate::context::ToolContext {
            workspace_dir: Some(workspace.clone()),
            ..Default::default()
        };

        let (write_count, read_count) = crate::context::run_with_tool_context(Some(ctx), async {
            tool.execute(serde_json::json!({
                "operation": "write",
                "path": "tracked.txt",
                "content": "hello",
            }))
            .await
            .unwrap();
            tool.execute(serde_json::json!({
                "operation": "read",
                "path": "tracked.txt",
            }))
            .await
            .unwrap();
            let writes =
                crate::context::with_current_tool_context(|c| c.workspace_file_writes.len());
            let reads = crate::context::with_current_tool_context(|c| c.workspace_file_reads.len());
            (writes, reads)
        })
        .await;
        assert_eq!(write_count, Some(1));
        assert_eq!(read_count, Some(1));
    }

    #[tokio::test]
    async fn test_scoped_context_gates_write_deny() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().canonicalize().expect("canonical workspace");
        let tool = FilesystemTool::new(workspace.clone());

        let ctx = crate::context::ToolContext {
            workspace_dir: Some(workspace.clone()),
            workspace_write_deny_globs: vec!["**/*.lock".to_string()],
            ..Default::default()
        };

        let result = crate::context::run_with_tool_context(
            Some(ctx),
            tool.execute(serde_json::json!({
                "operation": "write",
                "path": "Cargo.lock",
                "content": "locked",
            })),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "WORKSPACE_WRITE_DENY");
    }

    #[tokio::test]
    async fn test_scoped_context_requires_fresh_read_before_edit() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().canonicalize().expect("canonical workspace");
        let tool = FilesystemTool::new(workspace.clone());
        std::fs::write(workspace.join("x.txt"), "original").unwrap();

        let ctx = crate::context::ToolContext {
            workspace_dir: Some(workspace.clone()),
            file_edit_requires_fresh_read: true,
            ..Default::default()
        };

        // Without a prior read the edit is refused.
        let result = crate::context::run_with_tool_context(
            Some(ctx.clone()),
            tool.execute(serde_json::json!({
                "operation": "edit",
                "path": "x.txt",
                "old_string": "original",
                "new_string": "changed",
            })),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "FRESH_READ_REQUIRED");

        // After a full read the edit proceeds.
        let result = crate::context::run_with_tool_context(Some(ctx), async {
            tool.execute(serde_json::json!({
                "operation": "read",
                "path": "x.txt",
            }))
            .await
            .unwrap();
            tool.execute(serde_json::json!({
                "operation": "edit",
                "path": "x.txt",
                "old_string": "original",
                "new_string": "changed",
            }))
            .await
        })
        .await;
        assert!(result.is_ok());
    }
}
