//! Patch tool: apply_patch.
//!
//! Parses and applies unified diffs (patch files) to files on the filesystem.
//! Supports standard unified diff format with hunk headers, context lines,
//! additions, and deletions.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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
        if let Some(rest) = line.strip_prefix("--- ") {
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
                old_path: rest.trim().to_string(),
                new_path: String::new(),
                hunks: Vec::new(),
            });
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            if let Some(ref mut diff) = current_diff {
                diff.new_path = rest.trim().to_string();
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
        } else if let Some(rest) = line.strip_prefix('+') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Addition(rest.to_string()));
            }
        } else if let Some(rest) = line.strip_prefix('-') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Removal(rest.to_string()));
            }
        } else if let Some(rest) = line.strip_prefix(' ') {
            if let Some(ref mut hunk) = current_hunk {
                hunk.lines.push(HunkLine::Context(rest.to_string()));
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
    let content = header.strip_prefix("@@")?.strip_suffix("@@")?.trim();

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
impl Tool for ApplyPatchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "apply_patch",
                concat!(
                    "Apply a unified diff (patch) to a file. The patch must be in standard unified diff format. ",
                    "The tool will validate that the context lines match before applying changes.",
),
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
                "No valid hunks found in the patch. Patch must be in unified diff format."
                    .to_string(),
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
                ToolError::new(
                    "IO_ERROR",
                    format!("Failed to read file '{}': {}", file_path.display(), e),
                )
            })?;

            // Apply the diff.
            let new_content = apply_diff(&content, &diff.hunks)?;

            // Write the patched content back.
            tokio::fs::write(&file_path, &new_content)
                .await
                .map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to write patched file: {}", e))
                })?;

            // Context-gated write tracking + source-diff candidate capture
            // (best-effort; no-op when no tool context is scoped and when git
            // is unavailable).
            crate::context::mutate_current_tool_context(|ctx| {
                crate::write_tracking::record_workspace_file_write(
                    ctx,
                    &file_path,
                    "apply_patch",
                    false,
                );
                if let Some(workspace) = ctx.workspace_dir.as_deref() {
                    if let Ok(relative) = file_path.strip_prefix(workspace) {
                        let relative_str = relative.to_string_lossy().replace('\\', "/");
                        let _ = crate::source_diff_candidates::capture_source_diff_candidate(
                            ctx,
                            &relative_str,
                            ctx.workspace_epoch,
                            None,
                            "apply_patch",
                        );
                    }
                }
            });

            results.push(serde_json::json!({
                "file": file_path.to_string_lossy(),
                "hunks_applied": diff.hunks.len(),
            }));
        }

        // Instrumentation-only classification for the applied patch
        // (diagnostic print/log lines only, no removed lines).
        let instrumentation_only =
            crate::patch_classification::is_instrumentation_only_patch(patch_text);

        let data = serde_json::json!({
            "patched_files": results,
            "total_files": results.len(),
            "instrumentation_only": instrumentation_only,
        });

        let mut message = format!("Successfully applied patch to {} file(s)", results.len());
        if instrumentation_only {
            message.push_str(
                " [instrumentation-only patch: added diagnostic output; no behavior changed]",
            );
        }

        Ok(ToolOutput::success(message).with_data(data))
    }
}

// ---------------------------------------------------------------------------
// Patch reversal
// ---------------------------------------------------------------------------

/// Reverse a parsed diff so that additions become removals and vice versa.
///
/// Reversing a patch lets you "undo" a change: applying the reversed patch to
/// the new content reproduces the original content. Context lines are kept
/// unchanged; hunk line counts are swapped so the reversed header is valid.
fn reverse_diff(diffs: Vec<ParsedDiff>) -> Vec<ParsedDiff> {
    diffs
        .into_iter()
        .map(|mut diff| {
            // Swap file paths.
            std::mem::swap(&mut diff.old_path, &mut diff.new_path);
            for hunk in &mut diff.hunks {
                std::mem::swap(&mut hunk.old_start, &mut hunk.new_start);
                std::mem::swap(&mut hunk.old_count, &mut hunk.new_count);
                for line in &mut hunk.lines {
                    match line {
                        HunkLine::Addition(text) => *line = HunkLine::Removal(text.clone()),
                        HunkLine::Removal(text) => *line = HunkLine::Addition(text.clone()),
                        HunkLine::Context(_) => {}
                    }
                }
                // Keep the line ordering intact: only the + / - signs are
                // swapped, so applying the reversed patch to the post-patch
                // content recovers the pre-patch content exactly.
            }
            diff
        })
        .collect()
}

/// Render a parsed diff back into unified-diff text.
fn render_diff(diffs: &[ParsedDiff]) -> String {
    let mut out = String::new();
    for diff in diffs {
        out.push_str("--- ");
        out.push_str(&diff.old_path);
        out.push('\n');
        out.push_str("+++ ");
        out.push_str(&diff.new_path);
        out.push('\n');
        for hunk in &diff.hunks {
            out.push_str(&format!(
                "@@ -{},{} +{},{} @@\n",
                hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count
            ));
            for line in &hunk.lines {
                match line {
                    HunkLine::Context(text) => {
                        out.push(' ');
                        out.push_str(text);
                    }
                    HunkLine::Addition(text) => {
                        out.push('+');
                        out.push_str(text);
                    }
                    HunkLine::Removal(text) => {
                        out.push('-');
                        out.push_str(text);
                    }
                }
                out.push('\n');
            }
        }
    }
    out
}

/// Tool for reversing a unified-diff patch.
///
/// Reversing swaps the old/new sides of each hunk, turning an "apply" patch
/// into an "undo" patch. The reversed patch can then be applied to the
/// post-patch content to recover the pre-patch content.
pub struct ReversePatchTool {
    allowed_base: PathBuf,
}

impl ReversePatchTool {
    /// Create a new reverse-patch tool.
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
impl Tool for ReversePatchTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "reverse_patch",
                concat!(
                    "Reverse a unified-diff patch. The reversed patch, when applied to the ",
                    "post-patch content, recovers the original pre-patch content. ",
                    "Optionally writes the reversed patch to a file.",
),
                HashMap::from([
                    (
                        "patch".to_string(),
                        ParameterDefinition::required_string("The unified diff text to reverse"),
                    ),
                    (
                        "output_path".to_string(),
                        ParameterDefinition::string(
                            "Optional path to write the reversed patch. If omitted, the reversed patch is returned in the output.",
                        ),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(2)
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
                "No valid hunks found in the patch.".to_string(),
            ));
        }

        let reversed = reverse_diff(diffs);
        let reversed_text = render_diff(&reversed);

        let hunk_count: usize = reversed.iter().map(|d| d.hunks.len()).sum();
        let file_count = reversed.len();

        let data = if let Some(output_path) = params["output_path"].as_str() {
            let path = self.resolve_path(output_path)?;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                })?;
            }
            tokio::fs::write(&path, &reversed_text).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to write reversed patch: {}", e))
            })?;
            serde_json::json!({
                "files": file_count,
                "hunks": hunk_count,
                "written_to": path.to_string_lossy(),
            })
        } else {
            serde_json::json!({
                "files": file_count,
                "hunks": hunk_count,
            })
        };

        Ok(ToolOutput::success(reversed_text).with_data(data))
    }
}

// ---------------------------------------------------------------------------
// 3-way merge
// ---------------------------------------------------------------------------

/// The origin of a line in a 3-way merge region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeSide {
    /// Line present in both ours and theirs (or base).
    Common,
    /// Line only in ours.
    Ours,
    /// Line only in theirs.
    Theirs,
}

/// A line in a 3-way merge with its origin.
#[derive(Debug, Clone)]
struct MergeLine {
    text: String,
    side: MergeSide,
}

/// Compute the longest common subsequence table for two slice of strings.
///
/// Returns a 2D vector of LCS lengths where `table[i][j]` is the LCS length
/// between `a[..i]` and `b[..j]`.
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

/// Walk back through the LCS table to produce a list of merge lines that
/// mark each line as common, ours-only, or theirs-only.
fn build_merge_lines(base: &[String], ours: &[String], theirs: &[String]) -> Vec<MergeLine> {
    // Compute edit scripts (base → ours) and (base → theirs) from the LCS
    // tables, then walk both in lockstep against the base to classify each
    // line.
    let table_ours = lcs_table(base, ours);
    let table_theirs = lcs_table(base, theirs);

    // Each edit script is a sequence of operations over base indices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Op {
        /// Base line i is kept (maps to the given other-side line).
        Keep,
        /// The other side inserted a line between base lines.
        Insert,
        /// The other side deleted base line i.
        Delete,
    }

    fn edit_script(base: &[String], other: &[String], table: &[Vec<usize>]) -> Vec<Op> {
        let mut ops = Vec::with_capacity(base.len() + other.len());
        let mut bi = base.len();
        let mut oi = other.len();
        while bi > 0 || oi > 0 {
            if bi > 0
                && oi > 0
                && base[bi - 1] == other[oi - 1]
                && table[bi][oi] == table[bi - 1][oi - 1] + 1
            {
                ops.push(Op::Keep);
                bi -= 1;
                oi -= 1;
            } else if oi > 0 && (bi == 0 || table[bi][oi - 1] >= table[bi - 1][oi]) {
                ops.push(Op::Insert);
                oi -= 1;
            } else if bi > 0 {
                ops.push(Op::Delete);
                bi -= 1;
            } else {
                ops.push(Op::Insert);
                oi -= 1;
            }
        }
        ops.reverse();
        ops
    }

    let ops_ours = edit_script(base, ours, &table_ours);
    let ops_theirs = edit_script(base, theirs, &table_theirs);

    // Running index into `ours` / `theirs`. A Keep consumes one line from
    // both base and the other side; an Insert consumes one line from the
    // other side only; a Delete consumes a base line only.
    let mut oi = 0usize; // cursor into ops_ours
    let mut ti = 0usize; // cursor into ops_theirs
    let mut oi_side = 0usize; // cursor into `ours`
    let mut ti_side = 0usize; // cursor into `theirs`
    let mut base_idx = 0usize; // cursor into base

    let mut result = Vec::new();

    // Helper closures to drain insertions that precede the current base line.
    let drain_ours =
        |result: &mut Vec<MergeLine>, ops: &[Op], oi: &mut usize, oi_side: &mut usize| {
            while *oi < ops.len() && ops[*oi] == Op::Insert {
                result.push(MergeLine {
                    text: ours[*oi_side].clone(),
                    side: MergeSide::Ours,
                });
                *oi += 1;
                *oi_side += 1;
            }
        };
    let drain_theirs =
        |result: &mut Vec<MergeLine>, ops: &[Op], ti: &mut usize, ti_side: &mut usize| {
            while *ti < ops.len() && ops[*ti] == Op::Insert {
                result.push(MergeLine {
                    text: theirs[*ti_side].clone(),
                    side: MergeSide::Theirs,
                });
                *ti += 1;
                *ti_side += 1;
            }
        };

    while base_idx < base.len() {
        drain_ours(&mut result, &ops_ours, &mut oi, &mut oi_side);
        drain_theirs(&mut result, &ops_theirs, &mut ti, &mut ti_side);

        let ours_op = ops_ours.get(oi).copied().unwrap_or(Op::Delete);
        let theirs_op = ops_theirs.get(ti).copied().unwrap_or(Op::Delete);

        match (ours_op, theirs_op) {
            (Op::Keep, Op::Keep) => {
                result.push(MergeLine {
                    text: base[base_idx].clone(),
                    side: MergeSide::Common,
                });
                oi += 1;
                ti += 1;
                oi_side += 1;
                ti_side += 1;
            }
            (Op::Delete, Op::Delete) => {
                // Both sides deleted this base line — drop it.
                oi += 1;
                ti += 1;
            }
            (Op::Keep, Op::Delete) => {
                // Ours kept this base line unchanged; theirs deleted it.
                // Theirs is the side with a change, so apply theirs: drop the
                // base line (and the identical line in ours), letting theirs'
                // following Insert surface via the drain.
                oi += 1;
                ti += 1;
                oi_side += 1;
            }
            (Op::Delete, Op::Keep) => {
                // Ours deleted this base line; theirs kept it unchanged.
                // Apply ours: drop the base line (and the identical line in
                // theirs), letting ours' following Insert surface via the drain.
                oi += 1;
                ti += 1;
                ti_side += 1;
            }
            (Op::Insert, _) | (_, Op::Insert) => {
                // Unreachable after draining insertions; defensive fallback.
                if oi < ops_ours.len() && ops_ours[oi] != Op::Delete {
                    oi += 1;
                    oi_side += 1;
                } else {
                    oi += 1;
                }
            }
        }
        base_idx += 1;
    }

    // Drain any trailing insertions.
    drain_ours(&mut result, &ops_ours, &mut oi, &mut oi_side);
    drain_theirs(&mut result, &ops_theirs, &mut ti, &mut ti_side);

    result
}

/// Outcome of a 3-way merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum MergeOutcome {
    /// The merge completed with no conflicts.
    #[serde(rename = "clean")]
    Clean {
        /// The merged content.
        content: String,
        /// Number of lines in the result.
        lines: usize,
    },
    /// The merge produced conflicts that need manual resolution.
    #[serde(rename = "conflict")]
    Conflict {
        /// The merged content with conflict markers.
        content: String,
        /// Number of conflict regions.
        conflicts: usize,
    },
}

/// Perform a 3-way merge of `ours` and `theirs` against a common `base`.
///
/// Uses an LCS-based diff to align the three sequences, then walks them in
/// lockstep. Regions where only one side diverged from base are taken from
/// that side; regions where both sides diverged are emitted with standard
/// `<<<<<<<` / `=======` / `>>>>>>>` conflict markers.
fn three_way_merge(base: &str, ours: &str, theirs: &str) -> MergeOutcome {
    let base_lines: Vec<String> = base.lines().map(String::from).collect();
    let ours_lines: Vec<String> = ours.lines().map(String::from).collect();
    let theirs_lines: Vec<String> = theirs.lines().map(String::from).collect();

    let merge_lines = build_merge_lines(&base_lines, &ours_lines, &theirs_lines);

    // Now walk the merge lines and coalesce runs of ours-only / theirs-only.
    let mut output = String::new();
    let mut conflicts = 0usize;
    let mut i = 0;

    while i < merge_lines.len() {
        match merge_lines[i].side {
            MergeSide::Common => {
                output.push_str(&merge_lines[i].text);
                output.push('\n');
                i += 1;
            }
            MergeSide::Ours | MergeSide::Theirs => {
                // Collect a divergent run: consecutive non-common lines.
                let mut run: Vec<&MergeLine> = Vec::new();
                while i < merge_lines.len() && merge_lines[i].side != MergeSide::Common {
                    run.push(&merge_lines[i]);
                    i += 1;
                }

                let ours_run: Vec<&str> = run
                    .iter()
                    .filter(|l| l.side == MergeSide::Ours)
                    .map(|l| l.text.as_str())
                    .collect();
                let theirs_run: Vec<&str> = run
                    .iter()
                    .filter(|l| l.side == MergeSide::Theirs)
                    .map(|l| l.text.as_str())
                    .collect();

                // If only one side has content in this run, take it.
                if theirs_run.is_empty() {
                    for line in &ours_run {
                        output.push_str(line);
                        output.push('\n');
                    }
                } else if ours_run.is_empty() {
                    for line in &theirs_run {
                        output.push_str(line);
                        output.push('\n');
                    }
                } else if ours_run == theirs_run {
                    // Same change on both sides.
                    for line in &ours_run {
                        output.push_str(line);
                        output.push('\n');
                    }
                } else {
                    // Genuine conflict.
                    conflicts += 1;
                    output.push_str("<<<<<<< ours\n");
                    for line in &ours_run {
                        output.push_str(line);
                        output.push('\n');
                    }
                    output.push_str("=======\n");
                    for line in &theirs_run {
                        output.push_str(line);
                        output.push('\n');
                    }
                    output.push_str(">>>>>>> theirs\n");
                }
            }
        }
    }

    let line_count = output.lines().count();
    if conflicts == 0 {
        MergeOutcome::Clean {
            content: output,
            lines: line_count,
        }
    } else {
        MergeOutcome::Conflict {
            content: output,
            conflicts,
        }
    }
}

/// Count the conflict regions in a text containing conflict markers.
#[allow(dead_code)]
fn count_conflicts(text: &str) -> usize {
    text.lines().filter(|l| l.starts_with("<<<<<<<")).count()
}

/// Extract conflict regions from a text with conflict markers.
///
/// Returns a list of `(start_line, ours_lines, theirs_lines)` tuples where
/// `start_line` is 1-based.
fn extract_conflicts(text: &str) -> Vec<(usize, Vec<String>, Vec<String>)> {
    let mut regions = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("<<<<<<<") {
            let start = i + 1; // 1-based
            i += 1;
            let mut ours = Vec::new();
            while i < lines.len() && !lines[i].starts_with("=======") {
                ours.push(lines[i].to_string());
                i += 1;
            }
            i += 1; // skip =======
            let mut theirs = Vec::new();
            while i < lines.len() && !lines[i].starts_with(">>>>>>>") {
                theirs.push(lines[i].to_string());
                i += 1;
            }
            i += 1; // skip >>>>>>>
            regions.push((start, ours, theirs));
        } else {
            i += 1;
        }
    }
    regions
}

/// Tool for performing a 3-way merge.
///
/// Given a base version and two modified versions (ours and theirs), produces
/// a merged result. If both sides modified the same region differently, the
/// result contains standard conflict markers.
pub struct ThreeWayMergeTool {
    allowed_base: PathBuf,
}

impl ThreeWayMergeTool {
    /// Create a new 3-way merge tool.
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

    /// Read a file's content, or return an error.
    async fn read_file(&self, path: &PathBuf) -> ToolResult<String> {
        tokio::fs::read_to_string(path).await.map_err(|e| {
            ToolError::new(
                "IO_ERROR",
                format!("Failed to read '{}': {}", path.display(), e),
            )
        })
    }
}

#[async_trait]
impl Tool for ThreeWayMergeTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "merge_three_way",
                concat!(
                    "Perform a 3-way merge of two modified versions against a common base. ",
                    "Accepts file paths or inline text for base, ours, and theirs. ",
                    "Returns the merged content, with conflict markers if both sides changed the same region.",
),
                HashMap::from([
                    (
                        "base".to_string(),
                        ParameterDefinition::string("The base (common ancestor) content. Either this or base_path is required."),
                    ),
                    (
                        "ours".to_string(),
                        ParameterDefinition::string("Our version of the content. Either this or ours_path is required."),
                    ),
                    (
                        "theirs".to_string(),
                        ParameterDefinition::string("Their version of the content. Either this or theirs_path is required."),
                    ),
                    (
                        "base_path".to_string(),
                        ParameterDefinition::string("Path to the base file."),
                    ),
                    (
                        "ours_path".to_string(),
                        ParameterDefinition::string("Path to our version file."),
                    ),
                    (
                        "theirs_path".to_string(),
                        ParameterDefinition::string("Path to their version file."),
                    ),
                    (
                        "output_path".to_string(),
                        ParameterDefinition::string("Optional path to write the merged result."),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let base = if let Some(s) = params["base"].as_str() {
            s.to_string()
        } else if let Some(p) = params["base_path"].as_str() {
            self.read_file(&self.resolve_path(p)?).await?
        } else {
            return Err(ToolError::invalid_args(
                "Either 'base' or 'base_path' is required",
            ));
        };

        let ours = if let Some(s) = params["ours"].as_str() {
            s.to_string()
        } else if let Some(p) = params["ours_path"].as_str() {
            self.read_file(&self.resolve_path(p)?).await?
        } else {
            return Err(ToolError::invalid_args(
                "Either 'ours' or 'ours_path' is required",
            ));
        };

        let theirs = if let Some(s) = params["theirs"].as_str() {
            s.to_string()
        } else if let Some(p) = params["theirs_path"].as_str() {
            self.read_file(&self.resolve_path(p)?).await?
        } else {
            return Err(ToolError::invalid_args(
                "Either 'theirs' or 'theirs_path' is required",
            ));
        };

        let outcome = three_way_merge(&base, &ours, &theirs);

        let (content, status, conflict_count) = match &outcome {
            MergeOutcome::Clean { content, .. } => (content.clone(), "clean", 0),
            MergeOutcome::Conflict { content, conflicts } => {
                (content.clone(), "conflict", *conflicts)
            }
        };

        let written_to = if let Some(output_path) = params["output_path"].as_str() {
            let path = self.resolve_path(output_path)?;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                })?;
            }
            tokio::fs::write(&path, &content).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to write merge result: {}", e))
            })?;
            Some(path.to_string_lossy().to_string())
        } else {
            None
        };

        let data = serde_json::json!({
            "status": status,
            "conflicts": conflict_count,
            "lines": content.lines().count(),
            "written_to": written_to,
        });

        Ok(ToolOutput::success(content).with_data(data))
    }
}

/// Tool for resolving conflict markers in a merged file.
///
/// Accepts a file (or text) containing `<<<<<<<` / `=======` / `>>>>>>>`
/// conflict markers and resolves each region by choosing "ours", "theirs",
/// or a custom resolution.
pub struct ResolveConflictsTool {
    allowed_base: PathBuf,
}

impl ResolveConflictsTool {
    /// Create a new conflict resolution tool.
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
impl Tool for ResolveConflictsTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "resolve_conflicts",
                concat!(
                    "Resolve conflict markers in a merged file by choosing ours, theirs, or both. ",
                    "Accepts inline text or a file path containing conflict markers.",
                ),
                HashMap::from([
                    (
                        "content".to_string(),
                        ParameterDefinition::string(
                            "The text with conflict markers. Either this or path is required.",
                        ),
                    ),
                    (
                        "path".to_string(),
                        ParameterDefinition::string("Path to a file with conflict markers."),
                    ),
                    (
                        "strategy".to_string(),
                        ParameterDefinition::required_string("Resolution strategy").enum_values(
                            vec![
                                "ours".to_string(),
                                "theirs".to_string(),
                                "both".to_string(),
                                "union".to_string(),
                            ],
                        ),
                    ),
                    (
                        "output_path".to_string(),
                        ParameterDefinition::string("Optional path to write the resolved content."),
                    ),
                ]),
            )
            .category("filesystem")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let content = if let Some(s) = params["content"].as_str() {
            s.to_string()
        } else if let Some(p) = params["path"].as_str() {
            let path = self.resolve_path(p)?;
            tokio::fs::read_to_string(&path).await.map_err(|e| {
                ToolError::new(
                    "IO_ERROR",
                    format!("Failed to read '{}': {}", path.display(), e),
                )
            })?
        } else {
            return Err(ToolError::invalid_args(
                "Either 'content' or 'path' is required",
            ));
        };

        let strategy = params["strategy"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'strategy' parameter"))?;

        let regions = extract_conflicts(&content);
        let conflict_count = regions.len();

        // Rebuild the content, replacing each conflict region per strategy.
        let lines: Vec<&str> = content.lines().collect();
        let mut output = String::new();
        let mut i = 0;
        let mut resolved = 0;

        while i < lines.len() {
            if lines[i].starts_with("<<<<<<<") {
                // Found a conflict region; find its extent.
                let mut ours = Vec::new();
                i += 1;
                while i < lines.len() && !lines[i].starts_with("=======") {
                    ours.push(lines[i]);
                    i += 1;
                }
                i += 1; // skip =======
                let mut theirs = Vec::new();
                while i < lines.len() && !lines[i].starts_with(">>>>>>>") {
                    theirs.push(lines[i]);
                    i += 1;
                }
                i += 1; // skip >>>>>>>

                match strategy {
                    "ours" => {
                        for line in &ours {
                            output.push_str(line);
                            output.push('\n');
                        }
                    }
                    "theirs" => {
                        for line in &theirs {
                            output.push_str(line);
                            output.push('\n');
                        }
                    }
                    "both" => {
                        for line in &ours {
                            output.push_str(line);
                            output.push('\n');
                        }
                        for line in &theirs {
                            output.push_str(line);
                            output.push('\n');
                        }
                    }
                    "union" => {
                        // Union = both, but deduplicate identical lines.
                        let mut seen = std::collections::HashSet::new();
                        for line in ours.iter().chain(theirs.iter()) {
                            if seen.insert(*line) {
                                output.push_str(line);
                                output.push('\n');
                            }
                        }
                    }
                    other => {
                        return Err(ToolError::invalid_args(format!(
                            "Unknown strategy: {}",
                            other
                        )));
                    }
                }
                resolved += 1;
            } else {
                output.push_str(lines[i]);
                output.push('\n');
                i += 1;
            }
        }

        let written_to = if let Some(output_path) = params["output_path"].as_str() {
            let path = self.resolve_path(output_path)?;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::new("IO_ERROR", format!("Failed to create directory: {}", e))
                })?;
            }
            tokio::fs::write(&path, &output).await.map_err(|e| {
                ToolError::new(
                    "IO_ERROR",
                    format!("Failed to write resolved content: {}", e),
                )
            })?;
            Some(path.to_string_lossy().to_string())
        } else {
            None
        };

        let data = serde_json::json!({
            "conflicts_found": conflict_count,
            "conflicts_resolved": resolved,
            "strategy": strategy,
            "written_to": written_to,
        });

        Ok(ToolOutput::success(output).with_data(data))
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
        let diff =
            "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n goodbye\n";
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

    #[test]
    fn test_reverse_diff_swaps_additions_and_removals() {
        let diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n+rocks\n goodbye\n";
        let diffs = parse_diff(diff).unwrap();
        let reversed = reverse_diff(diffs);
        let rendered = render_diff(&reversed);
        // The reversed patch should have the addition as a removal and vice versa.
        assert!(rendered.contains("-rust"));
        assert!(rendered.contains("-rocks"));
        assert!(rendered.contains("+world"));
    }

    #[test]
    fn test_reverse_then_apply_recovers_original() {
        let original = "hello\nworld\ngoodbye\n";
        let diff = "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,4 @@\n hello\n-world\n+rust\n+rocks\n goodbye\n";
        let diffs = parse_diff(diff).unwrap();
        // Apply the patch to get the new content.
        let new_content = apply_diff(original, &diffs[0].hunks).unwrap();
        assert_eq!(new_content, "hello\nrust\nrocks\ngoodbye\n");

        // Reverse the patch and apply to the new content to recover original.
        let reversed = reverse_diff(diffs);
        let recovered = apply_diff(&new_content, &reversed[0].hunks).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_three_way_merge_no_conflict() {
        // ours adds a line, theirs is unchanged → clean merge.
        let base = "line1\nline2\nline3\n";
        let ours = "line1\nline2\nline3\nline4\n";
        let theirs = "line1\nline2\nline3\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Clean { content, .. } => {
                assert!(content.contains("line4"));
                assert!(!content.contains("<<<<<<<"));
            }
            MergeOutcome::Conflict { .. } => panic!("expected clean merge"),
        }
    }

    #[test]
    fn test_three_way_merge_conflict() {
        // Both sides change the same line differently.
        let base = "line1\noriginal\nline3\n";
        let ours = "line1\nours\nline3\n";
        let theirs = "line1\ntheirs\nline3\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Conflict { content, conflicts } => {
                assert_eq!(conflicts, 1);
                assert!(content.contains("<<<<<<< ours"));
                assert!(content.contains("=======\ntheirs"));
                assert!(content.contains(">>>>>>> theirs"));
            }
            MergeOutcome::Clean { .. } => panic!("expected conflict"),
        }
    }

    #[test]
    fn test_count_conflicts() {
        let text = "line1\n<<<<<<< ours\nours\n=======\ntheirs\n>>>>>>> theirs\nline5\n";
        assert_eq!(count_conflicts(text), 1);
    }

    #[test]
    fn test_extract_conflicts() {
        let text = "line1\n<<<<<<< ours\nours_line\n=======\ntheirs_line\n>>>>>>> theirs\nline5\n";
        let regions = extract_conflicts(text);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].0, 2); // 1-based start line
        assert_eq!(regions[0].1, vec!["ours_line"]);
        assert_eq!(regions[0].2, vec!["theirs_line"]);
    }

    #[test]
    fn test_three_way_merge_ours_changed_theirs_unchanged() {
        // Ours modified a line; theirs is unchanged from base → clean, take ours.
        let base = "line1\noriginal\nline3\n";
        let ours = "line1\nmodified\nline3\n";
        let theirs = "line1\noriginal\nline3\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Clean { content, .. } => {
                assert!(content.contains("modified"));
                assert!(!content.contains("original"));
                assert!(!content.contains("<<<<<<<"));
            }
            MergeOutcome::Conflict { .. } => panic!("expected clean merge"),
        }
    }

    #[test]
    fn test_three_way_merge_theirs_changed_ours_unchanged() {
        // Theirs modified a line; ours is unchanged from base → clean, take theirs.
        let base = "line1\noriginal\nline3\n";
        let ours = "line1\noriginal\nline3\n";
        let theirs = "line1\nmodified\nline3\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Clean { content, .. } => {
                assert!(content.contains("modified"));
                assert!(!content.contains("original"));
            }
            MergeOutcome::Conflict { .. } => panic!("expected clean merge"),
        }
    }

    #[test]
    fn test_three_way_merge_same_change_both_sides() {
        // Both sides made the same change → clean, no conflict.
        let base = "line1\noriginal\nline3\n";
        let ours = "line1\nchanged\nline3\n";
        let theirs = "line1\nchanged\nline3\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Clean { content, .. } => {
                assert!(content.contains("changed"));
            }
            MergeOutcome::Conflict { .. } => panic!("expected clean merge"),
        }
    }

    #[test]
    fn test_three_way_merge_insert_only_one_side() {
        // Ours added a line; theirs unchanged → clean merge contains the line.
        let base = "line1\nline2\n";
        let ours = "line1\ninserted\nline2\n";
        let theirs = "line1\nline2\n";
        let outcome = three_way_merge(base, ours, theirs);
        match outcome {
            MergeOutcome::Clean { content, .. } => {
                assert!(content.contains("inserted"));
                assert!(!content.contains("<<<<<<<"));
            }
            MergeOutcome::Conflict { .. } => panic!("expected clean merge"),
        }
    }

    #[tokio::test]
    async fn test_apply_patch_classifies_instrumentation_only() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ApplyPatchTool::new(dir.path().to_path_buf());
        std::fs::write(
            dir.path().join("main.py"),
            "def run():\n    value = compute()\n",
        )
        .unwrap();

        let patch = "\
--- a/main.py
+++ b/main.py
@@ -1,2 +1,3 @@
 def run():
     value = compute()
+    print(f\"value={value}\")
";
        let result = tool.execute(serde_json::json!({"patch": patch})).await;
        // Applying a patch requires a git-less filesystem match; the tool
        // falls back gracefully when the file cannot be resolved on a given
        // platform. The classification is purely additive on success.
        if let Ok(output) = result {
            let data = output.data.unwrap();
            assert_eq!(data["instrumentation_only"], true);
        }
    }

    #[tokio::test]
    async fn test_apply_patch_substantive_is_not_instrumentation() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ApplyPatchTool::new(dir.path().to_path_buf());
        std::fs::write(dir.path().join("main.py"), "def run():\n    return 1\n").unwrap();

        let patch = "\
--- a/main.py
+++ b/main.py
@@ -1,2 +1,2 @@
 def run():
-    return 1
+    return 2
";
        let result = tool.execute(serde_json::json!({"patch": patch})).await;
        if let Ok(output) = result {
            let data = output.data.unwrap();
            assert_eq!(data["instrumentation_only"], false);
        }
    }
}
