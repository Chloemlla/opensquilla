//! File and directory diffing tools with multiple output formats.
//!
//! Provides line-by-line file diffs (using an LCS algorithm), directory tree
//! diffs that compare two directory trees and report added/removed/modified
//! files, and multiple output formats (unified, context, JSON).
//!
//! This is the Rust counterpart of the Python `diff_tools.py` module.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// The output format for a diff.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DiffFormat {
    /// Standard unified diff (default).
    Unified,
    /// Context diff (with 3 lines of context by default).
    Context,
    /// JSON-structured diff.
    Json,
    /// A human-readable side-by-side summary.
    Summary,
}

impl Default for DiffFormat {
    fn default() -> Self {
        DiffFormat::Unified
    }
}

/// A single line change in a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LineChange {
    /// A line that is the same in both versions (context).
    #[serde(rename = "context")]
    Context { text: String, old_num: usize, new_num: usize },
    /// A line added in the new version.
    #[serde(rename = "added")]
    Added { text: String, new_num: usize },
    /// A line removed from the old version.
    #[serde(rename = "removed")]
    Removed { text: String, old_num: usize },
}

/// A hunk of changes in a file diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    /// The starting line in the old file (1-based).
    pub old_start: usize,
    /// The number of lines in the old file.
    pub old_count: usize,
    /// The starting line in the new file (1-based).
    pub new_start: usize,
    /// The number of lines in the new file.
    pub new_count: usize,
    /// The line changes in this hunk.
    pub lines: Vec<LineChange>,
}

/// A complete diff for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    /// The old file path.
    pub old_path: String,
    /// The new file path.
    pub new_path: String,
    /// The hunks of the diff.
    pub hunks: Vec<DiffHunk>,
    /// Whether the file was added (no old content).
    pub is_added: bool,
    /// Whether the file was deleted (no new content).
    pub is_deleted: bool,
}

/// The status of a file in a directory comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStatus {
    /// File exists only in the new tree.
    Added,
    /// File exists only in the old tree.
    Removed,
    /// File content differs between trees.
    Modified,
    /// File is identical in both trees.
    Unchanged,
}

/// An entry in a directory diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirDiffEntry {
    /// The relative path of the file.
    pub path: String,
    /// The status of the file.
    pub status: FileStatus,
    /// Size in bytes in the old tree (if present).
    pub old_size: Option<u64>,
    /// Size in bytes in the new tree (if present).
    pub new_size: Option<u64>,
}

/// The result of comparing two directories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirDiff {
    /// The old directory path.
    pub old_dir: String,
    /// The new directory path.
    pub new_dir: String,
    /// The entries in the diff.
    pub entries: Vec<DirDiffEntry>,
    /// Number of added files.
    pub added: usize,
    /// Number of removed files.
    pub removed: usize,
    /// Number of modified files.
    pub modified: usize,
    /// Number of unchanged files.
    pub unchanged: usize,
}

/// Compute the LCS table for two slices of strings.
fn lcs_table(a: &[String], b: &[String]) -> Vec<Vec<usize>> {
    let m = a.len();
    let n = b.len();
    let mut table = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
            } else {
                table[i][j] = table[i - 1][j].max(table[i][j - 1]);
            }
        }
    }
    table
}

/// Compute the line-level diff between two texts using LCS.
///
/// Returns a list of `LineChange` items. Each line is tagged as context,
/// added, or removed. Line numbers are 1-based.
pub fn diff_lines(old: &str, new: &str) -> Vec<LineChange> {
    let old_lines: Vec<String> = old.lines().map(String::from).collect();
    let new_lines: Vec<String> = new.lines().map(String::from).collect();

    let table = lcs_table(&old_lines, &new_lines);
    let mut changes = Vec::new();
    let mut i = old_lines.len();
    let mut j = new_lines.len();

    // Walk the table backwards to reconstruct the diff.
    let mut reversed = Vec::new();
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
            reversed.push(LineChange::Context {
                text: old_lines[i - 1].clone(),
                old_num: i,
                new_num: j,
            });
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || table[i][j - 1] >= table[i - 1][j]) {
            reversed.push(LineChange::Added {
                text: new_lines[j - 1].clone(),
                new_num: j,
            });
            j -= 1;
        } else if i > 0 {
            reversed.push(LineChange::Removed {
                text: old_lines[i - 1].clone(),
                old_num: i,
            });
            i -= 1;
        } else {
            break;
        }
    }
    reversed.reverse();
    changes = reversed;
    changes
}

/// Group line changes into hunks with configurable context size.
///
/// Adjacent changes within `context` lines of each other are grouped into
/// the same hunk.
pub fn group_into_hunks(changes: &[LineChange], context: usize) -> Vec<DiffHunk> {
    if changes.is_empty() {
        return Vec::new();
    }

    // Find indices of all non-context changes.
    let change_indices: Vec<usize> = changes
        .iter()
        .enumerate()
        .filter_map(|(i, c)| match c {
            LineChange::Added { .. } | LineChange::Removed { .. } => Some(i),
            _ => None,
        })
        .collect();

    if change_indices.is_empty() {
        return Vec::new();
    }

    let mut hunks = Vec::new();
    let mut current_start = change_indices[0].saturating_sub(context);
    let mut current_end = (change_indices[0] + context).min(changes.len() - 1);

    for &idx in &change_indices[1..] {
        let hunk_start = idx.saturating_sub(context);
        let hunk_end = (idx + context).min(changes.len() - 1);
        if hunk_start <= current_end {
            // Merge into current hunk.
            current_end = hunk_end;
        } else {
            // Flush current hunk.
            hunks.push(changes[current_start..=current_end].to_vec());
            current_start = hunk_start;
            current_end = hunk_end;
        }
    }
    hunks.push(changes[current_start..=current_end].to_vec());

    // Convert line groups into DiffHunks.
    hunks
        .into_iter()
        .map(|lines| {
            let old_start = lines
                .iter()
                .filter_map(|c| match c {
                    LineChange::Context { old_num, .. } | LineChange::Removed { old_num, .. } => {
                        Some(*old_num)
                    }
                    _ => None,
                })
                .min()
                .unwrap_or(1);
            let new_start = lines
                .iter()
                .filter_map(|c| match c {
                    LineChange::Context { new_num, .. } | LineChange::Added { new_num, .. } => {
                        Some(*new_num)
                    }
                    _ => None,
                })
                .min()
                .unwrap_or(1);
            let old_count = lines
                .iter()
                .filter(|c| matches!(c, LineChange::Context { .. } | LineChange::Removed { .. }))
                .count();
            let new_count = lines
                .iter()
                .filter(|c| matches!(c, LineChange::Context { .. } | LineChange::Added { .. }))
                .count();
            DiffHunk {
                old_start,
                old_count,
                new_start,
                new_count,
                lines,
            }
        })
        .collect()
}

/// Render a file diff as a unified diff string.
pub fn render_unified(diff: &FileDiff) -> String {
    let mut out = String::new();
    out.push_str(&format!("--- {}\n", diff.old_path));
    out.push_str(&format!("+++ {}\n", diff.new_path));
    for hunk in &diff.hunks {
        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
        ));
        for line in &hunk.lines {
            match line {
                LineChange::Context { text, .. } => {
                    out.push(' ');
                    out.push_str(text);
                }
                LineChange::Added { text, .. } => {
                    out.push('+');
                    out.push_str(text);
                }
                LineChange::Removed { text, .. } => {
                    out.push('-');
                    out.push_str(text);
                }
            }
            out.push('\n');
        }
    }
    out
}

/// Render a file diff as a context diff string.
pub fn render_context(diff: &FileDiff) -> String {
    let mut out = String::new();
    out.push_str(&format!("*** {}\n", diff.old_path));
    out.push_str(&format!("--- {}\n", diff.new_path));
    for hunk in &diff.hunks {
        out.push_str(&format!(
            "***************\n*** {},{}\n",
            hunk.old_start,
            hunk.old_start + hunk.old_count.saturating_sub(1)
        ));
        for line in &hunk.lines {
            match line {
                LineChange::Context { text, .. } => {
                    out.push_str("  ");
                    out.push_str(text);
                }
                LineChange::Removed { text, .. } => {
                    out.push_str("- ");
                    out.push_str(text);
                }
                _ => {}
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {},{}\n",
            hunk.new_start,
            hunk.new_start + hunk.new_count.saturating_sub(1)
        ));
        for line in &hunk.lines {
            match line {
                LineChange::Context { text, .. } => {
                    out.push_str("  ");
                    out.push_str(text);
                }
                LineChange::Added { text, .. } => {
                    out.push_str("+ ");
                    out.push_str(text);
                }
                _ => {}
            }
            out.push('\n');
        }
    }
    out
}

/// Render a file diff as a human-readable summary.
pub fn render_summary(diff: &FileDiff) -> String {
    let added = diff
        .hunks
        .iter()
        .flat_map(|h| h.lines.iter())
        .filter(|l| matches!(l, LineChange::Added { .. }))
        .count();
    let removed = diff
        .hunks
        .iter()
        .flat_map(|h| h.lines.iter())
        .filter(|l| matches!(l, LineChange::Removed { .. }))
        .count();
    let mut out = String::new();
    out.push_str(&format!("File: {} → {}\n", diff.old_path, diff.new_path));
    if diff.is_added {
        out.push_str("  Status: ADDED\n");
    } else if diff.is_deleted {
        out.push_str("  Status: DELETED\n");
    } else {
        out.push_str(&format!("  Status: MODIFIED\n"));
    }
    out.push_str(&format!("  +{} additions, -{} deletions\n", added, removed));
    out.push_str(&format!("  {} hunk(s)\n", diff.hunks.len()));
    out
}

/// Compute a file diff between two texts.
pub fn diff_texts(old: &str, new: &str, old_path: &str, new_path: &str) -> FileDiff {
    let changes = diff_lines(old, new);
    let hunks = group_into_hunks(&changes, 3);
    let is_added = old.is_empty() && !new.is_empty();
    let is_deleted = !old.is_empty() && new.is_empty();
    FileDiff {
        old_path: old_path.to_string(),
        new_path: new_path.to_string(),
        hunks,
        is_added,
        is_deleted,
    }
}

/// Walk a directory and return a map of relative path → file metadata.
///
/// Returns only regular files (not directories). Symlinks are followed.
fn walk_directory(root: &Path) -> std::io::Result<BTreeMap<String, u64>> {
    let mut files = BTreeMap::new();
    let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    walk_inner(&canonical_root, &canonical_root, &mut files)?;
    Ok(files)
}

fn walk_inner(
    current: &Path,
    root: &Path,
    files: &mut BTreeMap<String, u64>,
) -> std::io::Result<()> {
    if !current.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            walk_inner(&path, root, files)?;
        } else if metadata.is_file() {
            let rel = path
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|| path.to_string_lossy().to_string());
            files.insert(rel, metadata.len());
        }
    }
    Ok(())
}

/// Compare two directories and return the differences.
///
/// Files are compared by relative path. Files with the same path but
/// different sizes are marked as `Modified`. To detect content-identical
/// files with different sizes, use [`compare_directories_by_content`].
pub fn compare_directories(old_dir: &Path, new_dir: &Path) -> ToolResult<DirDiff> {
    let old_files = walk_directory(old_dir).map_err(|e| {
        ToolError::new(
            "IO_ERROR",
            format!("Failed to walk '{}': {}", old_dir.display(), e),
        )
    })?;
    let new_files = walk_directory(new_dir).map_err(|e| {
        ToolError::new(
            "IO_ERROR",
            format!("Failed to walk '{}': {}", new_dir.display(), e),
        )
    })?;

    let mut entries = Vec::new();
    let mut added = 0;
    let mut removed = 0;
    let mut modified = 0;
    let mut unchanged = 0;

    let all_paths: std::collections::BTreeSet<&String> =
        old_files.keys().chain(new_files.keys()).collect();

    for path in all_paths {
        let old_size = old_files.get(path).copied();
        let new_size = new_files.get(path).copied();
        let status = match (old_size, new_size) {
            (None, Some(new)) => {
                added += 1;
                FileStatus::Added
            }
            (Some(_), None) => {
                removed += 1;
                FileStatus::Removed
            }
            (Some(old), Some(new)) if old != new => {
                modified += 1;
                FileStatus::Modified
            }
            (Some(_), Some(_)) => {
                unchanged += 1;
                FileStatus::Unchanged
            }
            (None, None) => continue,
        };
        entries.push(DirDiffEntry {
            path: path.clone(),
            status,
            old_size,
            new_size,
        });
    }

    Ok(DirDiff {
        old_dir: old_dir.to_string_lossy().to_string(),
        new_dir: new_dir.to_string_lossy().to_string(),
        entries,
        added,
        removed,
        modified,
        unchanged,
    })
}

/// Compare two directories by content (hash), not just size.
///
/// Files with the same size but different content are marked `Modified`.
pub fn compare_directories_by_content(old_dir: &Path, new_dir: &Path) -> ToolResult<DirDiff> {
    use sha2::{Digest, Sha256};
    let mut hash_map_old: HashMap<String, [u8; 32]> = HashMap::new();
    let mut hash_map_new: HashMap<String, [u8; 32]> = HashMap::new();
    let mut size_map_old: HashMap<String, u64> = HashMap::new();
    let mut size_map_new: HashMap<String, u64> = HashMap::new();

    let old_files = walk_directory(old_dir).map_err(|e| {
        ToolError::new(
            "IO_ERROR",
            format!("Failed to walk '{}': {}", old_dir.display(), e),
        )
    })?;
    let new_files = walk_directory(new_dir).map_err(|e| {
        ToolError::new(
            "IO_ERROR",
            format!("Failed to walk '{}': {}", new_dir.display(), e),
        )
    })?;

    for (rel, size) in &old_files {
        let full = old_dir.join(rel);
        if let Ok(bytes) = std::fs::read(&full) {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            hash_map_old.insert(rel.clone(), hasher.finalize().into());
            size_map_old.insert(rel.clone(), *size);
        }
    }
    for (rel, size) in &new_files {
        let full = new_dir.join(rel);
        if let Ok(bytes) = std::fs::read(&full) {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            hash_map_new.insert(rel.clone(), hasher.finalize().into());
            size_map_new.insert(rel.clone(), *size);
        }
    }

    let mut entries = Vec::new();
    let mut added = 0;
    let mut removed = 0;
    let mut modified = 0;
    let mut unchanged = 0;

    let all_paths: std::collections::BTreeSet<&String> =
        hash_map_old.keys().chain(hash_map_new.keys()).collect();

    for path in all_paths {
        let old_hash = hash_map_old.get(path);
        let new_hash = hash_map_new.get(path);
        let old_size = size_map_old.get(path).copied();
        let new_size = size_map_new.get(path).copied();
        let status = match (old_hash, new_hash) {
            (None, Some(_)) => {
                added += 1;
                FileStatus::Added
            }
            (Some(_), None) => {
                removed += 1;
                FileStatus::Removed
            }
            (Some(o), Some(n)) if o != n => {
                modified += 1;
                FileStatus::Modified
            }
            (Some(_), Some(_)) => {
                unchanged += 1;
                FileStatus::Unchanged
            }
            (None, None) => continue,
        };
        entries.push(DirDiffEntry {
            path: path.clone(),
            status,
            old_size,
            new_size,
        });
    }

    Ok(DirDiff {
        old_dir: old_dir.to_string_lossy().to_string(),
        new_dir: new_dir.to_string_lossy().to_string(),
        entries,
        added,
        removed,
        modified,
        unchanged,
    })
}

/// Tool for diffing two files or texts.
pub struct DiffTool {
    allowed_base: PathBuf,
}

impl DiffTool {
    /// Create a new diff tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self { allowed_base }
    }

    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new("PATH_INVALID", format!("Cannot access path '{}': {}", path_str, e))
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
impl Tool for DiffTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "diff_files",
                "Diff two files or text strings and return the differences. "
                    + "Supports unified, context, JSON, and summary output formats.",
                HashMap::from([
                    (
                        "old_path".to_string(),
                        ParameterDefinition::string("Path to the old file. Either this or old_text is required."),
                    ),
                    (
                        "new_path".to_string(),
                        ParameterDefinition::string("Path to the new file. Either this or new_text is required."),
                    ),
                    (
                        "old_text".to_string(),
                        ParameterDefinition::string("The old text content."),
                    ),
                    (
                        "new_text".to_string(),
                        ParameterDefinition::string("The new text content."),
                    ),
                    (
                        "format".to_string(),
                        ParameterDefinition::string("Output format: unified, context, json, summary")
                            .default(serde_json::json!("unified")),
                    ),
                    (
                        "context".to_string(),
                        ParameterDefinition::integer("Number of context lines (default 3)")
                            .default(serde_json::json!(3)),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let old_text = if let Some(s) = params["old_text"].as_str() {
            s.to_string()
        } else if let Some(p) = params["old_path"].as_str() {
            let path = self.resolve_path(p)?;
            tokio::fs::read_to_string(&path).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to read '{}': {}", path.display(), e))
            })?
        } else {
            return Err(ToolError::invalid_args("Either 'old_text' or 'old_path' is required"));
        };

        let new_text = if let Some(s) = params["new_text"].as_str() {
            s.to_string()
        } else if let Some(p) = params["new_path"].as_str() {
            let path = self.resolve_path(p)?;
            tokio::fs::read_to_string(&path).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to read '{}': {}", path.display(), e))
            })?
        } else {
            return Err(ToolError::invalid_args("Either 'new_text' or 'new_path' is required"));
        };

        let format = params["format"].as_str().unwrap_or("unified");
        let context = params["context"].as_i64().unwrap_or(3).max(0) as usize;

        let old_path = params["old_path"].as_str().unwrap_or("old");
        let new_path = params["new_path"].as_str().unwrap_or("new");

        let changes = diff_lines(&old_text, &new_text);
        let hunks = group_into_hunks(&changes, context);
        let diff = FileDiff {
            old_path: old_path.to_string(),
            new_path: new_path.to_string(),
            hunks,
            is_added: old_text.is_empty() && !new_text.is_empty(),
            is_deleted: !old_text.is_empty() && new_text.is_empty(),
        };

        let (content, mime) = match format {
            "unified" => (render_unified(&diff), "text/x-diff"),
            "context" => (render_context(&diff), "text/x-diff"),
            "json" => (
                serde_json::to_string_pretty(&diff).unwrap_or_default(),
                "application/json",
            ),
            "summary" => (render_summary(&diff), "text/plain"),
            other => {
                return Err(ToolError::invalid_args(format!(
                    "Unknown format: '{}'. Supported: unified, context, json, summary",
                    other
                )))
            }
        };

        let added = diff
            .hunks
            .iter()
            .flat_map(|h| h.lines.iter())
            .filter(|l| matches!(l, LineChange::Added { .. }))
            .count();
        let removed = diff
            .hunks
            .iter()
            .flat_map(|h| h.lines.iter())
            .filter(|l| matches!(l, LineChange::Removed { .. }))
            .count();

        let data = serde_json::json!({
            "format": format,
            "hunks": diff.hunks.len(),
            "additions": added,
            "deletions": removed,
            "is_added": diff.is_added,
            "is_deleted": diff.is_deleted,
        });

        Ok(ToolOutput::success(content)
            .with_mime_type(mime)
            .with_data(data))
    }
}

/// Tool for diffing two directories.
pub struct DirDiffTool {
    allowed_base: PathBuf,
}

impl DirDiffTool {
    /// Create a new directory diff tool.
    pub fn new(allowed_base: PathBuf) -> Self {
        Self { allowed_base }
    }

    fn resolve_path(&self, path_str: &str) -> ToolResult<PathBuf> {
        let path = PathBuf::from(path_str);
        let resolved = if path.is_relative() {
            self.allowed_base.join(&path)
        } else {
            path
        };
        let canonical = resolved.canonicalize().map_err(|e| {
            ToolError::new("PATH_INVALID", format!("Cannot access path '{}': {}", path_str, e))
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
impl Tool for DirDiffTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "diff_directories",
                "Compare two directories and report added, removed, and modified files. "
                    + "Supports size-based (fast) or content-hash-based (accurate) comparison.",
                HashMap::from([
                    (
                        "old_dir".to_string(),
                        ParameterDefinition::required_string("Path to the old directory"),
                    ),
                    (
                        "new_dir".to_string(),
                        ParameterDefinition::required_string("Path to the new directory"),
                    ),
                    (
                        "method".to_string(),
                        ParameterDefinition::string("Comparison method: size or content")
                            .default(serde_json::json!("size")),
                    ),
                    (
                        "format".to_string(),
                        ParameterDefinition::string("Output format: json or summary")
                            .default(serde_json::json!("summary")),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let old_dir_str = params["old_dir"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'old_dir' parameter"))?;
        let new_dir_str = params["new_dir"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'new_dir' parameter"))?;

        let old_dir = self.resolve_path(old_dir_str)?;
        let new_dir = self.resolve_path(new_dir_str)?;

        if !old_dir.is_dir() {
            return Err(ToolError::new(
                "NOT_A_DIRECTORY",
                format!("'{}' is not a directory", old_dir.display()),
            ));
        }
        if !new_dir.is_dir() {
            return Err(ToolError::new(
                "NOT_A_DIRECTORY",
                format!("'{}' is not a directory", new_dir.display()),
            ));
        }

        let method = params["method"].as_str().unwrap_or("size");
        let format = params["format"].as_str().unwrap_or("summary");

        let dir_diff = if method == "content" {
            // Content comparison is CPU/IO-bound; run on a blocking thread.
            let old = old_dir.clone();
            let new = new_dir.clone();
            tokio::task::spawn_blocking(move || compare_directories_by_content(&old, &new))
                .await
                .map_err(|e| {
                    ToolError::new("DIFF_ERROR", format!("Diff task failed: {}", e))
                })??
        } else {
            let old = old_dir.clone();
            let new = new_dir.clone();
            tokio::task::spawn_blocking(move || compare_directories(&old, &new))
                .await
                .map_err(|e| {
                    ToolError::new("DIFF_ERROR", format!("Diff task failed: {}", e))
                })??
        };

        let content = match format {
            "json" => serde_json::to_string_pretty(&dir_diff).unwrap_or_default(),
            "summary" => {
                let mut out = String::new();
                out.push_str(&format!(
                    "Directory diff: {} → {}\n\n",
                    dir_diff.old_dir, dir_diff.new_dir
                ));
                out.push_str(&format!(
                    "  {} added, {} removed, {} modified, {} unchanged\n\n",
                    dir_diff.added, dir_diff.removed, dir_diff.modified, dir_diff.unchanged
                ));
                for entry in &dir_diff.entries {
                    if matches!(entry.status, FileStatus::Unchanged) {
                        continue;
                    }
                    let marker = match entry.status {
                        FileStatus::Added => "+",
                        FileStatus::Removed => "-",
                        FileStatus::Modified => "M",
                        FileStatus::Unchanged => " ",
                    };
                    out.push_str(&format!("  {} {}\n", marker, entry.path));
                }
                out
            }
            other => {
                return Err(ToolError::invalid_args(format!(
                    "Unknown format: '{}'. Supported: json, summary",
                    other
                )))
            }
        };

        let data = serde_json::json!({
            "added": dir_diff.added,
            "removed": dir_diff.removed,
            "modified": dir_diff.modified,
            "unchanged": dir_diff.unchanged,
            "total": dir_diff.entries.len(),
            "method": method,
        });

        Ok(ToolOutput::success(content).with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_lines_no_changes() {
        let changes = diff_lines("hello\nworld", "hello\nworld");
        assert_eq!(changes.len(), 2);
        assert!(matches!(changes[0], LineChange::Context { .. }));
    }

    #[test]
    fn test_diff_lines_addition() {
        let changes = diff_lines("hello\nworld", "hello\nnew\nworld");
        let has_addition = changes.iter().any(|c| matches!(c, LineChange::Added { .. }));
        assert!(has_addition);
    }

    #[test]
    fn test_diff_lines_removal() {
        let changes = diff_lines("hello\nmiddle\nworld", "hello\nworld");
        let has_removal = changes.iter().any(|c| matches!(c, LineChange::Removed { .. }));
        assert!(has_removal);
    }

    #[test]
    fn test_diff_texts_unified_format() {
        let diff = diff_texts("hello\nworld", "hello\nrust", "old.txt", "new.txt");
        let rendered = render_unified(&diff);
        assert!(rendered.contains("--- old.txt"));
        assert!(rendered.contains("+++ new.txt"));
        assert!(rendered.contains("-world"));
        assert!(rendered.contains("+rust"));
    }

    #[test]
    fn test_diff_texts_summary_format() {
        let diff = diff_texts("hello\nworld", "hello\nrust", "old.txt", "new.txt");
        let rendered = render_summary(&diff);
        assert!(rendered.contains("MODIFIED"));
        assert!(rendered.contains("addition"));
        assert!(rendered.contains("deletion"));
    }

    #[test]
    fn test_group_into_hunks() {
        let changes = diff_lines(
            "line1\nline2\nline3\nline4\nline5",
            "line1\nCHANGED\nline3\nline4\nline5",
        );
        let hunks = group_into_hunks(&changes, 3);
        assert_eq!(hunks.len(), 1);
    }

    #[tokio::test]
    async fn test_diff_files_text_mode() {
        let tool = DiffTool::new(PathBuf::from("."));
        let result = tool
            .execute(serde_json::json!({
                "old_text": "hello\nworld",
                "new_text": "hello\nrust",
                "format": "unified",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.content.contains("+rust"));
        assert!(output.content.contains("-world"));
    }

    #[tokio::test]
    async fn test_diff_files_json_format() {
        let tool = DiffTool::new(PathBuf::from("."));
        let result = tool
            .execute(serde_json::json!({
                "old_text": "hello\nworld",
                "new_text": "hello\nrust",
                "format": "json",
            }))
            .await;
        assert!(result.is_ok());
        let output = result.unwrap();
        let parsed: FileDiff = serde_json::from_str(&output.content).unwrap();
        assert!(!parsed.hunks.is_empty());
    }

    #[tokio::test]
    async fn test_diff_directories() {
        let old_dir = tempfile::tempdir().unwrap();
        let new_dir = tempfile::tempdir().unwrap();

        std::fs::write(old_dir.path().join("same.txt"), "same").unwrap();
        std::fs::write(new_dir.path().join("same.txt"), "same").unwrap();
        std::fs::write(old_dir.path().join("removed.txt"), "old").unwrap();
        std::fs::write(new_dir.path().join("added.txt"), "new").unwrap();
        std::fs::write(old_dir.path().join("modified.txt"), "old content").unwrap();
        std::fs::write(new_dir.path().join("modified.txt"), "new content different").unwrap();

        let tool = DirDiffTool::new(old_dir.path().to_path_buf());
        let result = tool
            .execute(serde_json::json!({
                "old_dir": ".",
                "new_dir": new_dir.path().to_string_lossy(),
                "method": "size",
                "format": "json",
            }))
            .await;
        // The new_dir is outside old_dir's allowed base, so this will fail
        // path validation. Let's use a tool rooted at the parent.
        if result.is_err() {
            // Expected: path traversal. Re-test with a tool rooted at the
            // temp parent.
            let parent = old_dir.path().parent().unwrap().to_path_buf();
            let tool = DirDiffTool::new(parent);
            let result = tool
                .execute(serde_json::json!({
                    "old_dir": old_dir.path().to_string_lossy(),
                    "new_dir": new_dir.path().to_string_lossy(),
                    "method": "size",
                    "format": "json",
                }))
                .await;
            assert!(result.is_ok());
            let output = result.unwrap();
            let diff: DirDiff = serde_json::from_str(&output.content).unwrap();
            assert!(diff.added >= 1);
            assert!(diff.removed >= 1);
            assert!(diff.modified >= 1);
        }
    }
}
