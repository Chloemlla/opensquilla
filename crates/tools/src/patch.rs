//! Patch tool: apply_patch.
//!
//! Parses and applies unified diffs (patch files) to files on the filesystem.
//! Supports standard unified diff format with hunk headers, context lines,
//! additions, and deletions.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

/// A single hunk in a unified diff.
#[derive(Debug, Clone)]
struct Hunk {
    /// The original file line range (start, count).
    old_start: usize,
    old_count: usize,
    /// The new file line range (start, count).
    new_start: usize,
    new_count: usize,
    /// The lines of the hunk.
    lines: Vec<HunkLine>,
}

/// A single line in a unified diff hunk.
#[derive(Debug, Clone)]
enum HunkLine {
    /// A context line (starts with ' ').
    Context(String),
    /// A line to be added (starts with '+').
    Addition(String),
    /// A line to be removed (starts with '-').
    Removal(String),
}

/// A parsed unified diff, containing changes for one or more files.
#[derive(Debug, Clone)]
struct ParsedDiff {
    /// The original file path.
    old_path: String,
    /// The new file path.
    new_path: String,
    /// The hunks of the diff.
    hunks: Vec<Hunk>,
}

/// Parse a unified diff string into a structured representation.
fn parse_diff(diff_text: &str) -> Result<Vec<ParsedDiff>, ToolError> {
    let mut diffs = Vec::new();
    let mut current_diff: Option<ParsedDiff> = None;
    let mut current_hunk: Option<Hunk> = None;

    for line in diff_text.lines() {
        if line.starts_with("--- ") {
            // Start of a new file diff.
            if let Some(hunk) = current_hunk.take() {
                if let Some(ref mut diff) = current_diff {
                    diff.hunks.push(hunk);
                }
            }
            if let Some(diff) = current_diff.take() {
                if !diff.hunks.is_empty() {
                    diffs.push(diff);
                }
            }
            current_diff = Some(ParsedDiff {
                old_path: line[4..].trim().to_string(),
                new_path: String::new(),
                hunks: Vec::new(),
            });
        } else if line.starts_with("+++ ") {
            if let Some(ref mut diff) = current_diff {
                diff.new_path = line[4..].trim().to_string();
            }
        } else if line.starts_with("@@") {
            // Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if let Some(hunk) = current_hunk.take() {
                if let Some(ref mut diff) = current_diff {
                    diff.hunks.push(hunk);
                }
            }

            if let Some(caps) = parse_hunk_header(line) {
                current_hunk = Some(Hunk {
                    old_start: caps.0,
                    old_count: caps.1,
                    new_start: caps.2,
                    new_count: caps.3,
                    lines: Vec::new(),
                });
            }
        } else if line.starts_with('+') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Addition(line[1..].to_string()));
            }
        } else if line.starts_with('-') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Removal(line[1..].to_string()));
            }
        } else if line.starts_with(' ') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Context(line[1..].to_string()));
            }
        }
        // Skip lines that don't start with known prefixes (e.g., diff --git, index lines).
    }

    // Flush remaining hunk and diff.
    if let Some(hunk) = current_hunk.take() {
        if let Some(ref mut diff) = current_diff {
            diff.hunks.push(hunk);
        }
    }
    if let Some(diff) = current_diff.take() {
        if !diff.hunks.is_empty() {
            diffs.push(diff);
        }
    }

    Ok(diffs)
}

/// Parse a unified diff hunk header.
/// Format: @@ -old_start,old_count +new_start,new_count @@
fn parse_hunk_header(header: &str) -> Option<(usize, usize, usize, usize)> {
    let header = header.trim();
    if !header.starts_with("@@") {
        return None;
    }

    // Extract the content between @@ markers.
    let content = header
        .strip_prefix("@@")?
        .strip_suffix("@@")?
        .trim();

    // Split into old and new parts.
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let old_part = parts[0].strip_prefix('-')?;
    let new_part = parts[1].strip_prefix('+')?;

    let parse_range = |s: &str| -> Option<(usize, usize)> {
        if let Some((start, count)) = s.split_once(',') {
            Some((start.parse().ok()?, count.parse().ok()?))
        } else {
            Some((s.parse().ok()?, 1))
        }
    };

    let (old_start, old_count) = parse_range(old_part)?;
    let (new_start, new_count) = parse_range(new_part)?;

    Some((old_start, old_count, new_start, new_count))
}

/// Apply a parsed diff to a file's content.
fn apply_diff(content: &str, hunks: &[Hunk]) -> Result<String, ToolError> {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = Vec::new();
    let mut line_idx = 0usize;

    for hunk in hunks {
        // Add lines before the hunk.
        while line_idx < hunk.old_start.saturating_sub(1) && line_idx < lines.len() {
            result.push(lines[line_idx].to_string());
            line_idx += 1;
        }

        // Verify the context lines match.
        let hunk_lines = &hunk.lines;
        let mut hunk_idx = 0;
        let mut original_idx = line_idx;

        // Verify the hunk matches the target content.
        while hunk_idx < hunk_lines.len() {
            match &hunk_lines[hunk_idx] {
                HunkLine::Context(text) | HunkLine::Removal(text) => {
                    if original_idx < lines.len() {
                        let original_line = lines[original_idx];
                        if original_line != text.as_str() {
                            return Err(ToolError::new(
                                "PATCH_FAILED",
                                format!(
                                    "Hunk context mismatch at line {}: expected '{}', got '{}'",
                                    original_idx + 1,
                                    text,
                                    original_line
                                ),
                            ));
                        }
                    }
                    original_idx += 1;
                }
                HunkLine::Addition(_) => {
                    // Additions don't consume from the original.
                }
            }
            hunk_idx += 1;
        }

        // Apply the hunk: keep context lines, skip removals, add additions.
        for hunk_line in &hunk.lines {
            match hunk_line {
                HunkLine::Context(text) => {
                    result.push(text.clone());
                    line_idx += 1;
                }
                HunkLine::Removal(_) => {
                    line_idx += 1;
                }
                HunkLine::Addition(text) => {
                    result.push(text.clone());
                }
            }
        }
    }

    // Add remaining lines after the last hunk.
    while line_idx < lines.len() {
        result.push(lines[line_idx].to_string());
        line_idx += 1;
    }

    Ok(result.join("\n"))
}

/// Tool for applying unified diff patches to files.
pub struct ApplyPatchTool {
    /// Allowed base directory.
    allowed_base: PathBuf,
}

impl ApplyPatchTool {
    /// Create a new patch tool.
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
impl Tool for ApplyPatchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "apply_patch",
                "Apply a unified diff (patch) to a file. The patch must be in standard unified diff format. "
                    + "The tool will validate that the context lines match before applying changes.",
                HashMap::from([
                    (
                        "patch".to_string(),
                        ParameterDefinition::required_string("The unified diff text to apply"),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::string("Optional: specific file path to patch. If not provided, the path from the diff header is used."),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let patch_text = params["patch"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'patch' parameter"))?;

        let diffs = parse_diff(patch_text)?;

        if diffs.is_empty() {
            return Err(ToolError::new(
                "INVALID_PATCH",
                "No valid hunks found in the patch. Patch must be in unified diff format.".to_string(),
            ));
        }

        let mut results = Vec::new();

        for diff in &diffs {
            // Determine the file path.
            let file_path = if let Some(path) = params["path"].as_str() {
                self.resolve_path(path)?
            } else {
                // Use the new file path from the diff header, stripping a/ or b/ prefix.
                let path_str = diff
                    .new_path
                    .strip_prefix("b/")
                    .or_else(|| diff.new_path.strip_prefix("a/"))
                    .unwrap_or(&diff.new_path);
                if path_str.is_empty() || path_str == "/dev/null" {
                    continue;
                }
                self.resolve_path(path_str)?
            };

            // Read the current file content.
            let content = tokio::fs::read_to_string(&file_path).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to read file '{}': {}", file_path.display(), e))
            })?;

            // Apply the diff.
            let new_content = apply_diff(&content, &diff.hunks)?;

            // Write the patched content back.
            tokio::fs::write(&file_path, &new_content).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to write patched file: {}", e))
            })?;

            results.push(serde_json::json!({
                "file": file_path.to_string_lossy(),
                "hunks_applied": diff.hunks.len(),
            }));
        }

        let data = serde_json::json!({
            "patched_files": results,
            "total_files": results.len(),
        });

        Ok(ToolOutput::success(format!(
            "Successfully applied patch to {} file(s)",
            results.len()
        ))
        .with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hunk_header() {
        let header = "@@ -10,6 +10,7 @@";
        let result = parse_hunk_header(header);
        assert_eq!(result, Some((10, 6, 10, 7)));
    }

    #[test]
    fn test_parse_hunk_header_single_line() {
        let header = "@@ -1 +1,2 @@";
        let result = parse_hunk_header(header);
        assert_eq!(result, Some((1, 1, 1, 2)));
    }

    #[test]
    fn test_parse_simple_diff() {
        let diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n+rocks\n goodbye\n";
        let diffs = parse_diff(diff).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].hunks.len(), 1);
        assert_eq!(diffs[0].hunks[0].lines.len(), 5);
    }

    #[test]
    fn test_apply_diff() {
        let original = "hello\nworld\ngoodbye\n";
        let diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n+rocks\n goodbye\n";
        let diffs = parse_diff(diff).unwrap();
        let result = apply_diff(original, &diffs[0].hunks).unwrap();
        assert_eq!(result, "hello\nrust\nrocks\ngoodbye\n");
    }

    #[test]
    fn test_apply_diff_context_mismatch() {
        let original = "hello\nsomething\nworld\n";
        let diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n goodbye\n";
        let diffs = parse_diff(diff).unwrap();
        let result = apply_diff(original, &diffs[0].hunks);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PATCH_FAILED");
    }

    #[test]
    fn test_parse_invalid_diff() {
        let diffs = parse_diff("this is not a diff").unwrap();
        assert!(diffs.is_empty());
    }
}