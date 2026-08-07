//! Candidate scanning and lightweight signal classification for Dream.
//!
//! Port of `src/opensquilla/memory/dream/candidates.py`.

use sha2::{Digest, Sha256};
use std::path::Path;

use crate::dream::models::RawDreamCandidate;
use crate::dream::quarantine::{is_quarantined_path, is_quarantined_text};

const SNIPPET_MAX_CHARS: usize = 4000;

fn workspace_relative(workspace: &Path, path: &Path) -> String {
    match path.strip_prefix(workspace) {
        Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
        Err(_) => path.to_string_lossy().replace('\\', "/"),
    }
}

fn normalize_snippet(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sha256(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// Extract a `YYYY-MM-DD` prefix from a file stem, when the stem looks dated.
fn source_day(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let bytes = stem.as_bytes();
    if stem.len() >= 10 && bytes[4] == b'-' && bytes[7] == b'-' {
        let candidate = &stem[..10];
        if candidate
            .split('-')
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
        {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Classify the signal kind of a candidate snippet.
pub fn classify_signal(text: &str) -> String {
    let lowered = text.to_lowercase();
    if lowered.contains("memory:") || lowered.contains("remember that") {
        return "manual".to_string();
    }
    const CORRECTION: &[&str] = &["do not", "don't", "rejected", "wrong", "instead"];
    if CORRECTION.iter().any(|m| lowered.contains(m)) {
        return "correction".to_string();
    }
    const FAILURE: &[&str] = &["failed", "error", "exception", "traceback", "rollback"];
    if FAILURE.iter().any(|m| lowered.contains(m)) {
        return "failure".to_string();
    }
    const POSITIVE: &[&str] = &["prefers", "accepted", "successful", "works", "use "];
    if POSITIVE.iter().any(|m| lowered.contains(m)) {
        return "positive".to_string();
    }
    "neutral".to_string()
}

/// Scan workspace memory files modified after `cursor`, producing raw
/// promotion candidates ordered by modification time.
pub fn scan_dream_candidates(
    workspace: &Path,
    cursor: f64,
    max_batch_size: usize,
    agent_id: &str,
    quarantine_enabled: bool,
) -> Vec<RawDreamCandidate> {
    let memory_dir = workspace.join("memory");
    if !memory_dir.is_dir() {
        return Vec::new();
    }
    let entries = match std::fs::read_dir(&memory_dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };

    let mut candidates: Vec<(f64, RawDreamCandidate)> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let Some(name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        let is_md = path
            .extension()
            .map(|e| e.to_string_lossy().eq_ignore_ascii_case("md"))
            .unwrap_or(false);
        if name.starts_with('.') || name == "MEMORY.md" || !is_md {
            continue;
        }
        if mtime <= cursor {
            continue;
        }
        let rel_path = workspace_relative(workspace, &path);
        if quarantine_enabled && is_quarantined_path(&rel_path) {
            continue;
        }
        let raw = match std::fs::read(&path) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => continue,
        };
        if quarantine_enabled && is_quarantined_text(&raw) {
            continue;
        }
        let mut snippet = normalize_snippet(&raw);
        if snippet.chars().count() > SNIPPET_MAX_CHARS {
            snippet = snippet.chars().take(SNIPPET_MAX_CHARS).collect::<String>();
            snippet = snippet.trim_end().to_string();
        }
        if snippet.is_empty() {
            continue;
        }
        let mtime_ns = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        candidates.push((
            mtime,
            RawDreamCandidate {
                agent_id: agent_id.to_string(),
                source_path: rel_path,
                source_kind: "memory_file".to_string(),
                source_mtime_ns: mtime_ns,
                source_size: metadata.len() as i64,
                snippet: snippet.clone(),
                snippet_sha256: sha256(&snippet),
                claim_sha256: sha256(&normalize_snippet(&snippet).to_lowercase()),
                source_day: source_day(&path),
                signal_kind: classify_signal(&snippet),
            },
        ));
    }

    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(max_batch_size);
    candidates
        .into_iter()
        .map(|(_mtime, candidate)| candidate)
        .collect()
}
