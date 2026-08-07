//! Curated MEMORY.md writes for Dream.
//!
//! Port of `src/opensquilla/memory/dream/curated_apply.py`. The apply step
//! rewrites the workspace `MEMORY.md`, upserting or merging bullets under
//! `## Section` headings, and honors a dry-run flag.

use crate::dream::models::{ApplyPromotionResult, PromotionPatch, PromotionPatchOperation};

/// Normalize a section name into a `## Heading` line, defaulting to
/// `Long-Term Memory` when the section is empty.
fn section_heading(section: &str) -> String {
    let cleaned = section.trim();
    let heading = if cleaned.is_empty() {
        "Long-Term Memory"
    } else {
        cleaned
    };
    format!("## {heading}")
}

/// Normalize a bullet: trim surrounding whitespace and prefix with `- `.
fn normalize_bullet(text: &str) -> String {
    let stripped = text.trim();
    if stripped.is_empty() {
        return String::new();
    }
    if stripped.starts_with("- ") {
        stripped.to_string()
    } else {
        format!("- {stripped}")
    }
}

/// Upsert `bullet` under `section` in `content`, mirroring
/// `_upsert_under_section` from curated_apply.py.
///
/// Returns the (possibly unchanged) content and whether a bullet was added.
fn upsert_under_section(content: &str, section: &str, bullet: &str) -> (String, bool) {
    let heading = section_heading(section);
    if content.trim().is_empty() {
        return (format!("{heading}\n\n{bullet}\n"), true);
    }
    let lines: Vec<&str> = content.lines().collect();
    let Some(start) = lines.iter().position(|line| *line == heading.as_str()) else {
        let base = content.trim_end();
        return (format!("{base}\n\n{heading}\n\n{bullet}\n"), true);
    };
    let mut next_heading = lines.len();
    for (idx, line) in lines.iter().enumerate().skip(start + 1) {
        if line.starts_with("## ") {
            next_heading = idx;
            break;
        }
    }
    let section_lines = &lines[start + 1..next_heading];
    if section_lines.contains(&bullet) {
        return (
            if content.ends_with('\n') {
                content.to_string()
            } else {
                format!("{content}\n")
            },
            false,
        );
    }
    let mut next_lines = Vec::with_capacity(lines.len() + 1);
    next_lines.extend_from_slice(&lines[..next_heading]);
    next_lines.push(bullet);
    next_lines.extend_from_slice(&lines[next_heading..]);
    let joined = next_lines.join("\n");
    let mut joined = joined.trim_end().to_string();
    joined.push('\n');
    (joined, true)
}

/// Apply a single operation to the current content, mirroring
/// `_apply_operation` from curated_apply.py. Only `upsert` and `merge`
/// operations change content; other ops are ignored.
fn apply_operation(content: &str, operation: &PromotionPatchOperation) -> (String, bool) {
    if !matches!(operation.op.as_str(), "upsert" | "merge") {
        return (content.to_string(), false);
    }
    let bullet = normalize_bullet(&operation.text);
    if bullet.is_empty() {
        return (content.to_string(), false);
    }
    upsert_under_section(content, &operation.section, &bullet)
}

/// Apply a promotion patch to the workspace `MEMORY.md`, honoring `dry_run`.
pub fn apply_promotion_patch(
    workspace: &std::path::Path,
    patch: &PromotionPatch,
    dry_run: bool,
) -> ApplyPromotionResult {
    let memory_path = workspace.join("MEMORY.md");
    let content = std::fs::read_to_string(&memory_path).unwrap_or_default();
    let mut applied: i64 = 0;
    let mut skipped: i64 = 0;
    let mut applied_operations: Vec<serde_json::Value> = Vec::new();
    let mut next_content = content.clone();

    for operation in &patch.operations {
        if operation.op == "skip" {
            skipped += 1;
            applied_operations.push(serde_json::json!({
                "op": operation.op,
                "candidate_ids": operation.candidate_ids,
                "changed": false,
                "reason": operation.reason.as_deref().unwrap_or("skip"),
            }));
            continue;
        }
        let (new_content, changed) = apply_operation(&next_content, operation);
        next_content = new_content;
        if changed {
            applied += 1;
        }
        applied_operations.push(serde_json::json!({
            "op": operation.op,
            "candidate_ids": operation.candidate_ids,
            "memory_id": operation.memory_id,
            "section": operation.section,
            "changed": changed,
        }));
    }

    let changed = next_content != content;
    if changed && !dry_run {
        if let Some(parent) = memory_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&memory_path, &next_content);
    }

    ApplyPromotionResult {
        applied: if dry_run { 0 } else { applied },
        skipped,
        changed: changed && !dry_run,
        applied_operations,
    }
}
