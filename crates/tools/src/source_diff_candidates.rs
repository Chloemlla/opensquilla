//! In-memory source diff candidate ledger for coding-agent recovery.
//!
//! Mirrors the Python `opensquilla.tools.source_diff_candidates` module:
//! captures the current git diff for a changed source path so a later
//! destructive action can restore it. The ledger lives on the tool context.

use crate::context::ToolContext;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Maximum number of candidates retained in the ledger.
pub const MAX_CANDIDATES: usize = 8;
/// Maximum patch size captured per candidate.
pub const MAX_PATCH_CHARS: usize = 64_000;

const CANDIDATE_MODES: &[&str] = &["off", "log", "warn_model"];

fn candidate_mode(ctx: &ToolContext) -> &'static str {
    let value = ctx
        .source_diff_candidate_mode
        .trim()
        .to_ascii_lowercase();
    if CANDIDATE_MODES.contains(&value.as_str()) {
        match value.as_str() {
            "off" => "off",
            "warn_model" => "warn_model",
            _ => "log",
        }
    } else {
        "log"
    }
}

fn normalize_relative_path(path: &str) -> String {
    let mut text = path.trim().to_string().replace('\\', "/");
    while text.starts_with("./") {
        text = text[2..].to_string();
    }
    Path::new(&text)
        .to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches('/')
        .to_string()
}

fn git_diff_for_path(workspace: &Path, relative_path: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["diff", "--", relative_path])
        .current_dir(workspace)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Capture the current git diff for a changed source path.
///
/// Capture is intentionally best-effort: failures return `None` and are never
/// raised into the source-edit tool path.
pub fn capture_source_diff_candidate(
    ctx: &mut ToolContext,
    relative_path: &str,
    workspace_epoch: u32,
    receipt_id: Option<&str>,
    tool_name: &str,
) -> Option<Value> {
    if candidate_mode(ctx) == "off" {
        return None;
    }
    let workspace = ctx.workspace_dir.clone()?;
    let path = normalize_relative_path(relative_path);
    if path.is_empty() {
        return None;
    }
    let patch = git_diff_for_path(&workspace, &path)?;
    if patch.trim().is_empty() || patch.chars().count() > MAX_PATCH_CHARS {
        return None;
    }

    ctx.source_diff_candidate_counter += 1;
    let counter = ctx.source_diff_candidate_counter;
    let mut hasher = Sha256::new();
    hasher.update(patch.as_bytes());
    let patch_sha256 = format!("{:x}", hasher.finalize());
    let candidate = json!({
        "candidate_id": format!("srcdiff-{counter}"),
        "paths": [path],
        "patch": patch,
        "patch_sha256": patch_sha256,
        "workspace_epoch": workspace_epoch,
        "receipt_id": receipt_id,
        "tool_name": tool_name,
        "lost": false,
        "lost_reason": null,
        "lost_command": null,
        "restored": false,
    });
    ctx.source_diff_candidates.push(candidate.clone());
    if ctx.source_diff_candidates.len() > MAX_CANDIDATES {
        let overflow = ctx.source_diff_candidates.len() - MAX_CANDIDATES;
        ctx.source_diff_candidates.drain(0..overflow);
    }
    Some(candidate)
}

/// Mark recoverable candidates whose paths were targeted by a destructive
/// action.
pub fn mark_source_diff_candidates_lost(
    ctx: &mut ToolContext,
    paths: &[&str],
    reason: &str,
    command: Option<&str>,
) -> Vec<Value> {
    let targets: std::collections::BTreeSet<String> = paths
        .iter()
        .map(|p| normalize_relative_path(p))
        .filter(|p| !p.is_empty())
        .collect();
    if targets.is_empty() {
        return Vec::new();
    }
    let mut marked: Vec<Value> = Vec::new();
    for candidate in &mut ctx.source_diff_candidates {
        if candidate.get("lost").and_then(|v| v.as_bool()).unwrap_or(false)
            || candidate.get("restored").and_then(|v| v.as_bool()).unwrap_or(false)
        {
            continue;
        }
        let candidate_paths: std::collections::BTreeSet<String> = candidate
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|p| normalize_relative_path(p))
                    .collect()
            })
            .unwrap_or_default();
        if candidate_paths.is_disjoint(&targets) {
            continue;
        }
        candidate["lost"] = json!(true);
        candidate["lost_reason"] = json!(reason);
        if let Some(command) = command {
            candidate["lost_command"] = json!(command);
        }
        marked.push(candidate.clone());
    }
    marked
}

/// Return the newest candidate that has not been restored.
pub fn latest_recoverable_source_candidate(ctx: &ToolContext) -> Option<Value> {
    ctx.source_diff_candidates
        .iter()
        .rev()
        .find(|candidate| {
            let restored = candidate
                .get("restored")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let has_patch = candidate.get("patch").and_then(|v| v.as_str()).is_some();
            !restored && has_patch
        })
        .cloned()
}

/// Return lost candidate ids that overlap current lost source paths.
pub fn recoverable_lost_source_candidate_ids(
    candidates: &[Value],
    lost_source_paths: &[&str],
) -> Vec<String> {
    let lost_paths: std::collections::BTreeSet<String> = lost_source_paths
        .iter()
        .map(|p| normalize_relative_path(p))
        .filter(|p| !p.is_empty())
        .collect();
    if lost_paths.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<String> = Vec::new();
    for candidate in candidates {
        let lost = candidate.get("lost").and_then(|v| v.as_bool()).unwrap_or(false);
        let restored = candidate
            .get("restored")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if !lost || restored {
            continue;
        }
        let candidate_paths: std::collections::BTreeSet<String> = candidate
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|p| normalize_relative_path(p))
                    .collect()
            })
            .unwrap_or_default();
        if candidate_paths.is_disjoint(&lost_paths) {
            continue;
        }
        if let Some(candidate_id) = candidate.get("candidate_id").and_then(|v| v.as_str()) {
            if !result.contains(&candidate_id.to_string()) {
                result.push(candidate_id.to_string());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn candidate_ledger_captures_and_trims() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut ctx = ToolContext {
            workspace_dir: Some(temp.path().to_path_buf()),
            ..Default::default()
        };
        // Not a git repo: capture fails gracefully.
        let candidate = capture_source_diff_candidate(&mut ctx, "src/x.rs", 1, Some("mut-1"), "edit_source");
        assert!(candidate.is_none());
        assert!(ctx.source_diff_candidates.is_empty());
    }

    #[test]
    fn candidate_mode_off_skips_capture() {
        let mut ctx = ToolContext {
            workspace_dir: Some(PathBuf::from("/workspace")),
            source_diff_candidate_mode: "off".to_string(),
            ..Default::default()
        };
        assert!(capture_source_diff_candidate(&mut ctx, "a.rs", 1, None, "edit_source").is_none());
    }

    #[test]
    fn normalize_relative_path_behavior() {
        assert_eq!(normalize_relative_path("./src/x.rs"), "src/x.rs");
        assert_eq!(normalize_relative_path("a\\b\\c.py"), "a/b/c.py");
        assert_eq!(normalize_relative_path(""), "");
    }

    #[test]
    fn mark_lost_and_latest_recoverable() {
        let mut ctx = ToolContext::default();
        ctx.source_diff_candidates = vec![
            json!({
                "candidate_id": "srcdiff-1",
                "paths": ["src/a.rs"],
                "patch": "diff --git a/src/a.rs b/src/a.rs",
                "lost": false,
                "restored": false,
            }),
            json!({
                "candidate_id": "srcdiff-2",
                "paths": ["src/b.rs"],
                "patch": "diff --git a/src/b.rs b/src/b.rs",
                "lost": false,
                "restored": false,
            }),
        ];
        let marked = mark_source_diff_candidates_lost(&mut ctx, &["src/a.rs"], "git_checkout", Some("git checkout HEAD"));
        assert_eq!(marked.len(), 1);
        assert_eq!(marked[0]["candidate_id"], "srcdiff-1");
        assert!(ctx.source_diff_candidates[0]["lost"].as_bool().unwrap());

        let latest = latest_recoverable_source_candidate(&ctx).expect("latest");
        assert_eq!(latest["candidate_id"], "srcdiff-2");
    }

    #[test]
    fn recoverable_lost_ids() {
        let candidates = vec![
            json!({
                "candidate_id": "srcdiff-1",
                "paths": ["src/a.rs"],
                "lost": true,
                "restored": false,
            }),
            json!({
                "candidate_id": "srcdiff-2",
                "paths": ["src/b.rs"],
                "lost": true,
                "restored": true,
            }),
        ];
        let ids = recoverable_lost_source_candidate_ids(&candidates, &["src/a.rs"]);
        assert_eq!(ids, vec!["srcdiff-1"]);
    }
}
