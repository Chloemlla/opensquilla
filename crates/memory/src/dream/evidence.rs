//! Promotion evidence store for Dream.
//!
//! Mirrors `src/opensquilla/memory/dream/evidence.py`. The evidence store
//! accumulates per-candidate signal counts and persists to
//! `memory/.dream_state/promotion_evidence.json`.

use crate::dream::models::{PromotionEvidenceEntry, PromotionEvidenceStore, RawDreamCandidate};
use sha2::{Digest, Sha256};

/// A decoded JSON object (map of string keys to values).
type JsonObject = serde_json::Map<String, serde_json::Value>;

/// Path of the persisted promotion evidence store.
pub fn promotion_evidence_path(workspace: &std::path::Path) -> std::path::PathBuf {
    workspace
        .join("memory")
        .join(".dream_state")
        .join("promotion_evidence.json")
}

/// Collapse runs of whitespace in a snippet to single spaces.
fn normalize_snippet(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Hex SHA-256 digest of `text`.
fn sha256(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

/// Deterministic candidate id: SHA-256 of `agent_id` and the claim hash,
/// newline-separated (mirrors `_candidate_id` in evidence.py).
fn candidate_id(agent_id: &str, claim_sha: &str) -> String {
    sha256(&format!("{agent_id}\n{claim_sha}"))
}

/// Stringify a scalar JSON value (string, number, bool). Returns `None` for
/// null, arrays and objects.
fn string_of(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Optional string field: kept only when it is a JSON string, mirroring the
/// `isinstance(raw.get(...), str)` checks in evidence.py.
fn opt_str(obj: &JsonObject, key: &str) -> Option<String> {
    obj.get(key).and_then(|v| v.as_str()).map(String::from)
}

/// String field with a fallback, mirroring `str(raw.get(key) or default)`:
/// missing, null and empty-string values fall back to `default`.
fn str_or(obj: &JsonObject, key: &str, default: &str) -> String {
    match obj.get(key) {
        Some(serde_json::Value::String(s)) if !s.is_empty() => s.clone(),
        Some(value) => string_of(value)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default.to_string()),
        None => default.to_string(),
    }
}

/// Non-empty strings from a JSON array field.
fn str_list(obj: &JsonObject, key: &str) -> Vec<String> {
    match obj.get(key) {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
        _ => Vec::new(),
    }
}

/// Coerce a JSON value to `i64` the way Python's `int()` does: booleans become
/// 0/1, floats are truncated, numeric strings are parsed. Returns `None` for
/// values that cannot be coerced (mirrors `ValueError`/`TypeError`).
fn json_as_i64(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                i64::try_from(u).ok()
            } else if let Some(f) = n.as_f64() {
                if f.is_finite() {
                    Some(f.trunc() as i64)
                } else {
                    None
                }
            } else {
                None
            }
        }
        serde_json::Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        serde_json::Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

/// Integer field mirroring `int(raw.get(key) or 0)`. Returns `None` when the
/// stored value cannot be coerced (dropping the whole entry). When `clamp` is
/// set the result is clamped to zero, mirroring `max(0, ...)`.
fn int_field(obj: &JsonObject, key: &str, clamp: bool) -> Option<i64> {
    let value = match obj.get(key) {
        None | Some(serde_json::Value::Null) => return Some(0),
        Some(serde_json::Value::String(s)) if s.is_empty() => return Some(0),
        Some(value) => value,
    };
    json_as_i64(value).map(|i| if clamp { i.max(0) } else { i })
}

/// Build an evidence entry from a decoded JSON object, dropping malformed
/// entries (mirrors `_entry_from_dict` in evidence.py).
fn entry_from_dict(raw: &serde_json::Value) -> Option<PromotionEvidenceEntry> {
    let obj = raw.as_object()?;
    let candidate_id = string_of(obj.get("candidate_id")?)?;
    let source_path = string_of(obj.get("source_path")?)?;
    Some(PromotionEvidenceEntry {
        candidate_id,
        agent_id: str_or(obj, "agent_id", "main"),
        source_path,
        source_kind: str_or(obj, "source_kind", "memory_file"),
        source_mtime_ns: int_field(obj, "source_mtime_ns", false)?,
        source_size: int_field(obj, "source_size", false)?,
        snippet: str_or(obj, "snippet", ""),
        snippet_sha256: str_or(obj, "snippet_sha256", ""),
        claim_sha256: str_or(obj, "claim_sha256", ""),
        first_seen_at: str_or(obj, "first_seen_at", ""),
        last_seen_at: str_or(obj, "last_seen_at", ""),
        seen_count: int_field(obj, "seen_count", true)?,
        positive_signal_count: int_field(obj, "positive_signal_count", true)?,
        correction_signal_count: int_field(obj, "correction_signal_count", true)?,
        failure_signal_count: int_field(obj, "failure_signal_count", true)?,
        manual_signal_count: int_field(obj, "manual_signal_count", true)?,
        source_days: str_list(obj, "source_days"),
        status: str_or(obj, "status", "candidate"),
        promoted_at: opt_str(obj, "promoted_at"),
        rejected_at: opt_str(obj, "rejected_at"),
        last_skip_reason: opt_str(obj, "last_skip_reason"),
    })
}

/// Load the promotion evidence store, returning a fresh store when missing or
/// corrupt. Malformed entries are dropped individually.
pub fn load_evidence_store(workspace: &std::path::Path) -> PromotionEvidenceStore {
    let path = promotion_evidence_path(workspace);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => return PromotionEvidenceStore::default(),
    };
    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(raw) => raw,
        Err(_) => return PromotionEvidenceStore::default(),
    };
    if !raw.is_object() {
        return PromotionEvidenceStore::default();
    }
    let mut store = PromotionEvidenceStore::default();
    if let Some(updated_at) = raw.get("updated_at") {
        store.updated_at = string_of(updated_at)
            .filter(|s| !s.is_empty())
            .unwrap_or_default();
    }
    if let Some(entries) = raw.get("entries").and_then(|v| v.as_object()) {
        for (key, value) in entries {
            if let Some(entry) = entry_from_dict(value) {
                store.entries.insert(key.clone(), entry);
            }
        }
    }
    store
}

/// Persist the promotion evidence store atomically (tmp file + rename).
pub fn write_evidence_store(
    workspace: &std::path::Path,
    store: &PromotionEvidenceStore,
) -> std::io::Result<()> {
    let path = promotion_evidence_path(workspace);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let payload = serde_json::to_string_pretty(store).map_err(std::io::Error::other)?;
    std::fs::write(&tmp, format!("{payload}\n"))?;
    std::fs::rename(&tmp, &path)
}

/// Increment the counter matching the candidate's signal kind. Unknown kinds
/// (e.g. "neutral") are ignored, mirroring evidence.py.
fn increment_signal(entry: &mut PromotionEvidenceEntry, signal_kind: &str) {
    match signal_kind {
        "positive" => entry.positive_signal_count += 1,
        "correction" => entry.correction_signal_count += 1,
        "failure" => entry.failure_signal_count += 1,
        "manual" => entry.manual_signal_count += 1,
        _ => {}
    }
}

/// Fold raw candidates into the evidence store, incrementing seen counts and
/// signal counters, and persist when `persist` is true.
pub fn update_promotion_evidence(
    workspace: &std::path::Path,
    candidates: &[RawDreamCandidate],
    now_iso: &str,
    persist: bool,
) -> PromotionEvidenceStore {
    let mut store = load_evidence_store(workspace);
    for candidate in candidates {
        let snippet = candidate.snippet.trim().to_string();
        if snippet.is_empty() {
            continue;
        }
        let snippet_sha = if candidate.snippet_sha256.is_empty() {
            sha256(&snippet)
        } else {
            candidate.snippet_sha256.clone()
        };
        let claim_sha = if candidate.claim_sha256.is_empty() {
            sha256(&normalize_snippet(&snippet).to_lowercase())
        } else {
            candidate.claim_sha256.clone()
        };
        let id = candidate_id(&candidate.agent_id, &claim_sha);
        let entry = store
            .entries
            .entry(id.clone())
            .or_insert_with(|| PromotionEvidenceEntry {
                candidate_id: id.clone(),
                agent_id: candidate.agent_id.clone(),
                source_path: candidate.source_path.clone(),
                source_kind: candidate.source_kind.clone(),
                source_mtime_ns: candidate.source_mtime_ns,
                source_size: candidate.source_size,
                snippet: snippet.clone(),
                snippet_sha256: snippet_sha.clone(),
                claim_sha256: claim_sha.clone(),
                first_seen_at: now_iso.to_string(),
                last_seen_at: now_iso.to_string(),
                seen_count: 0,
                positive_signal_count: 0,
                correction_signal_count: 0,
                failure_signal_count: 0,
                manual_signal_count: 0,
                source_days: Vec::new(),
                status: "candidate".to_string(),
                promoted_at: None,
                rejected_at: None,
                last_skip_reason: None,
            });
        entry.last_seen_at = now_iso.to_string();
        entry.source_path = candidate.source_path.clone();
        entry.source_kind = candidate.source_kind.clone();
        entry.source_mtime_ns = candidate.source_mtime_ns;
        entry.source_size = candidate.source_size;
        entry.snippet = snippet;
        entry.snippet_sha256 = snippet_sha;
        entry.claim_sha256 = claim_sha;
        entry.seen_count += 1;
        if let Some(day) = &candidate.source_day {
            if !entry.source_days.contains(day) {
                entry.source_days.push(day.clone());
            }
        }
        increment_signal(entry, &candidate.signal_kind);
    }
    store.updated_at = now_iso.to_string();
    if persist {
        // The runner reports write failures separately via `write_evidence_store`.
        let _ = write_evidence_store(workspace, &store);
    }
    store
}

/// Mark candidates as promoted in the evidence store.
pub fn mark_evidence_promoted(
    store: &mut PromotionEvidenceStore,
    candidate_ids: &[String],
    now_iso: &str,
) {
    for candidate_id in candidate_ids {
        if let Some(entry) = store.entries.get_mut(candidate_id) {
            entry.status = "promoted".to_string();
            entry.promoted_at = Some(now_iso.to_string());
            entry.last_skip_reason = None;
        }
    }
}

/// Record a skip reason for a candidate in the evidence store.
pub fn mark_evidence_skipped(store: &mut PromotionEvidenceStore, candidate_id: &str, reason: &str) {
    if let Some(entry) = store.entries.get_mut(candidate_id) {
        entry.last_skip_reason = Some(reason.to_string());
    }
}

/// Mark candidates as represented (no curated change) in the evidence store.
pub fn mark_evidence_represented(
    store: &mut PromotionEvidenceStore,
    candidate_ids: &[String],
    reason: &str,
) {
    for candidate_id in candidate_ids {
        if let Some(entry) = store.entries.get_mut(candidate_id) {
            entry.status = "represented".to_string();
            entry.last_skip_reason = Some(reason.to_string());
        }
    }
}
