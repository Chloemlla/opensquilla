//! Revision-based source read and line edit helpers.
//!
//! Mirrors the Python `opensquilla.tools.source_edit_contract` module:
//! build a model-facing read receipt with plain source lines, apply
//! inclusive 1-based line edits to source text, and produce bounded unified
//! diff summaries for source edits.

use crate::diff::{diff_texts, render_unified};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Raised when a source edit contract input cannot be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceEditContractError {
    pub message: String,
}

impl std::fmt::Display for SourceEditContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SourceEditContractError {}

fn contract_error(message: impl Into<String>) -> SourceEditContractError {
    SourceEditContractError {
        message: message.into(),
    }
}

/// Default number of source lines included in a read receipt.
pub const DEFAULT_SOURCE_READ_LINES: usize = 200;

/// Return a stable short revision token for the current file bytes.
pub fn source_revision_for_path(path: &Path) -> Result<String, SourceEditContractError> {
    let bytes = std::fs::read(path)
        .map_err(|e| contract_error(format!("cannot read {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("file_{:x}", hasher.finalize()).chars().take(17).collect())
}

fn line_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.lines().count()
    }
}

fn validate_line_range(
    start_line: i64,
    end_line: i64,
    line_count: usize,
) -> Result<(usize, usize), SourceEditContractError> {
    if start_line < 1 || end_line < 1 {
        return Err(contract_error("line ranges must be positive"));
    }
    if start_line > end_line {
        return Err(contract_error("start_line must be less than or equal to end_line"));
    }
    let end = end_line as usize;
    if end > line_count {
        return Err(contract_error(format!(
            "line range {start_line}-{end_line} exceeds file length {line_count}"
        )));
    }
    Ok((start_line as usize, end))
}

/// Build a model-facing read receipt with plain source lines.
///
/// `end_line == None` reads up to [`DEFAULT_SOURCE_READ_LINES`] lines from
/// `start_line`.
pub fn build_line_receipt(
    path: &Path,
    start_line: i64,
    end_line: Option<i64>,
    display_path: &str,
) -> Result<serde_json::Value, SourceEditContractError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| contract_error(format!("cannot read {}: {e}", path.display())))?;
    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();
    let effective_end_line = match end_line {
        None => (start_line as usize)
            .saturating_add(DEFAULT_SOURCE_READ_LINES)
            .saturating_sub(1)
            .min(total_lines) as i64,
        Some(end) => end,
    };
    let (start, end) = validate_line_range(start_line, effective_end_line, total_lines)?;
    let revision = source_revision_for_path(path)?;
    let receipt_lines: Vec<serde_json::Value> = (start..=end)
        .map(|line_number| {
            serde_json::json!({
                "line": line_number,
                "text": lines[line_number - 1],
            })
        })
        .collect();
    Ok(serde_json::json!({
        "status": "success",
        "path": display_path,
        "revision": revision,
        "range": [start, end],
        "total_lines": total_lines,
        "lines": receipt_lines,
    }))
}

fn replacement_lines(replacement: &serde_json::Value, index: usize) -> Result<Vec<String>, SourceEditContractError> {
    let Some(replacement) = replacement.as_str() else {
        return Err(contract_error(format!(
            "edits[{index}].replacement must be a string"
        )));
    };
    if replacement.is_empty() {
        return Ok(Vec::new());
    }
    Ok(split_lines_keepends(replacement))
}

fn split_lines_keepends(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut remaining = text;
    while let Some(pos) = remaining.find('\n') {
        lines.push(remaining[..=pos].to_string());
        remaining = &remaining[pos + 1..];
    }
    if !remaining.is_empty() {
        lines.push(remaining.to_string());
    }
    lines
}

fn normalized_edits(
    original: &str,
    edits: &[serde_json::Value],
) -> Result<Vec<(usize, usize, Vec<String>)>, SourceEditContractError> {
    if edits.is_empty() {
        return Err(contract_error("edits must be a non-empty array"));
    }
    let line_count = line_count(original);
    let mut normalized: Vec<(usize, usize, Vec<String>)> = Vec::new();
    for (index, edit) in edits.iter().enumerate() {
        let Some(edit) = edit.as_object() else {
            return Err(contract_error(format!("edits[{index}] must be an object")));
        };
        let start_line = edit
            .get("start_line")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| contract_error("line ranges must use integer start_line and end_line"))?;
        let end_line = edit
            .get("end_line")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| contract_error("line ranges must use integer start_line and end_line"))?;
        let (start, end) = validate_line_range(start_line, end_line, line_count)?;
        let replacement = replacement_lines(edit.get("replacement").unwrap_or(&serde_json::Value::Null), index)?;
        normalized.push((start, end, replacement));
    }

    normalized.sort_by_key(|item| item.0);
    let mut previous_end = 0usize;
    for (start, end, _) in &normalized {
        if *start <= previous_end {
            return Err(contract_error("edits must not overlap"));
        }
        previous_end = *end;
    }
    Ok(normalized)
}

/// Apply inclusive 1-based line edits to source text.
pub fn apply_line_edits(
    original: &str,
    edits: &[serde_json::Value],
) -> Result<String, SourceEditContractError> {
    let normalized = normalized_edits(original, edits)?;
    let mut lines = split_lines_keepends(original);
    for (start, end, replacement) in normalized.iter().rev() {
        let range_end = *end; // inclusive end -> exclusive slice bound
        lines.splice(start - 1..range_end, replacement.iter().cloned());
    }
    Ok(lines.concat())
}

/// Return a bounded unified diff summary for a source edit.
///
/// When the diff exceeds `max_chars`, the tail is truncated with a
/// `[diff_summary_truncated: omitted_chars=N]` marker.
pub fn build_diff_summary(
    before: &str,
    after: &str,
    path: &str,
    max_chars: usize,
) -> String {
    let diff = render_unified(&diff_texts(before, after, &format!("a/{path}"), &format!("b/{path}")));
    if diff.chars().count() <= max_chars {
        return diff;
    }
    let omitted = diff.chars().count() - max_chars;
    let truncated: String = diff.chars().take(max_chars).collect();
    format!("{truncated}\n[diff_summary_truncated: omitted_chars={omitted}]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_revision_is_stable() {
        let temp = tempfile::NamedTempFile::new().expect("temp");
        std::fs::write(temp.path(), b"hello world").expect("write");
        let rev = source_revision_for_path(temp.path()).expect("revision");
        assert_eq!(rev.len(), 17);
        assert!(rev.starts_with("file_"));
        let rev2 = source_revision_for_path(temp.path()).expect("revision");
        assert_eq!(rev, rev2);
    }

    #[test]
    fn build_line_receipt_ranges() {
        let temp = tempfile::NamedTempFile::new().expect("temp");
        std::fs::write(temp.path(), b"a\nb\nc\nd\ne").expect("write");
        let receipt = build_line_receipt(temp.path(), 2, Some(4), "src/x.py").expect("receipt");
        assert_eq!(receipt["status"], "success");
        assert_eq!(receipt["total_lines"], 5);
        assert_eq!(receipt["range"], serde_json::json!([2, 4]));
        let lines = receipt["lines"].as_array().unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["text"], "b");
        assert_eq!(lines[2]["text"], "d");
    }

    #[test]
    fn build_line_receipt_default_window() {
        let temp = tempfile::NamedTempFile::new().expect("temp");
        let content: String = (1..=300).map(|i| format!("line{i}\n")).collect();
        std::fs::write(temp.path(), content).expect("write");
        let receipt = build_line_receipt(temp.path(), 1, None, "src/x.py").expect("receipt");
        let lines = receipt["lines"].as_array().unwrap();
        assert_eq!(lines.len(), DEFAULT_SOURCE_READ_LINES);
    }

    #[test]
    fn line_range_validation() {
        let temp = tempfile::NamedTempFile::new().expect("temp");
        std::fs::write(temp.path(), b"a\nb\nc").expect("write");
        assert!(build_line_receipt(temp.path(), 0, Some(1), "x").is_err());
        assert!(build_line_receipt(temp.path(), 3, Some(2), "x").is_err());
        assert!(build_line_receipt(temp.path(), 1, Some(99), "x").is_err());
        assert!(build_line_receipt(temp.path(), 1, Some(3), "x").is_ok());
    }

    #[test]
    fn apply_line_edits_replaces_range() {
        let original = "a\nb\nc\nd\ne";
        let edits = serde_json::json!([
            {"start_line": 2, "end_line": 3, "replacement": "X\nY"}
        ]);
        let result = apply_line_edits(original, &edits.as_array().unwrap().clone()).expect("apply");
        // Matches the Python splitlines(keepends=True) contract: the
        // replacement's final line (no trailing newline) merges with "d\n".
        assert_eq!(result, "a\nX\nYd\ne");
    }

    #[test]
    fn apply_multiple_line_edits() {
        let original = "a\nb\nc\nd\ne\nf";
        let edits = serde_json::json!([
            {"start_line": 2, "end_line": 2, "replacement": "B"},
            {"start_line": 5, "end_line": 5, "replacement": "E"}
        ]);
        let result = apply_line_edits(original, &edits.as_array().unwrap().clone()).expect("apply");
        assert_eq!(result, "a\nBc\nd\nEf");
    }

    #[test]
    fn apply_line_edits_rejects_overlap() {
        let original = "a\nb\nc\nd";
        let edits = serde_json::json!([
            {"start_line": 2, "end_line": 3, "replacement": "X"},
            {"start_line": 3, "end_line": 4, "replacement": "Y"}
        ]);
        let result = apply_line_edits(original, &edits.as_array().unwrap().clone());
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("overlap"));
    }

    #[test]
    fn apply_line_edits_rejects_bad_replacement() {
        let original = "a\nb";
        let edits = serde_json::json!([
            {"start_line": 1, "end_line": 1, "replacement": 42}
        ]);
        assert!(apply_line_edits(original, &edits.as_array().unwrap().clone()).is_err());
    }

    #[test]
    fn build_diff_summary_truncates() {
        let before = "a\nb\nc\nd\ne";
        let after = "a\nB\nc\nd\ne\nf\ng\nh";
        let summary = build_diff_summary(before, after, "src/x.rs", 40);
        assert!(summary.contains("[diff_summary_truncated"));
        let long = build_diff_summary(before, after, "src/x.rs", 4000);
        assert!(!long.contains("[diff_summary_truncated"));
        assert!(long.contains("--- a/src/x.rs"));
        assert!(long.contains("+++ b/src/x.rs"));
    }
}
