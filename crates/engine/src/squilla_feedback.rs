//! Squilla router self-learning: feedback capture and dataset export.
//!
//! Mirrors the pure data/storage layers of the Python
//! `squilla_router/self_learning/` subsystem (`feedback.py`, `store.py`,
//! `schema.py`). The offline training pipeline (`train.py`, `train_worker.py`,
//! `evaluate.py`, `dataset.py` alignment) hard-depends on numpy/LightGBM and is
//! intentionally skipped in the Rust port.
//!
//! Storage is append-only JSONL under `<home>/router/`:
//!
//! ```text
//! router/data/<agent_id>/feedback.jsonl         explicit ratings
//! router/data/<agent_id>/samples-YYYYMMDD.jsonl captured turns
//! router/datasets/<agent_id>/<label>.jsonl      exported datasets
//! ```
//!
//! Writes are best-effort: a capture failure must never fail a turn.

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Global kill-switch env var, mirroring `OPENSQUILLA_ROUTER_SELFLEARN_DISABLED`.
pub const ENV_DISABLE: &str = "OPENSQUILLA_ROUTER_SELFLEARN_DISABLED";
/// Feedback JSONL schema version (additive-only).
pub const FEEDBACK_SCHEMA_VERSION: u32 = 1;
/// Feedback file name inside the agent data directory.
pub const FEEDBACK_FILENAME: &str = "feedback.jsonl";
/// Allowed explicit ratings; `neutral` revokes a previous rating.
pub const RATINGS: &[&str] = &["up", "down", "neutral"];
/// Executed-decision kinds; `ensemble` ratings judge the whole chain.
pub const EXECUTED_KINDS: &[&str] = &["single", "ensemble"];
/// Router sample schema version.
pub const SAMPLE_SCHEMA_VERSION: u32 = 1;
/// Captured text-feature dimension (storage constant; no numpy in Rust).
pub const FEATURES_390_DIM: usize = 390;

/// One appended rating row. Revisions append; readers merge last-write-wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackRow {
    pub decision_id: String,
    pub session_key: String,
    pub turn_index: i64,
    pub rating: String,
    pub executed_kind: String,
    pub ts: String,
    pub decision_ts: String,
    pub schema_version: u32,
}

/// The effective (post-merge) rating for one routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackEntry {
    pub rating: String,
    pub executed_kind: String,
}

/// Aggregate counts for gates, rollback monitoring, and status RPCs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeedbackStats {
    pub total: usize,
    pub up: usize,
    pub down: usize,
    /// Single-model slice; the rollback monitor keys on this only.
    pub total_single: usize,
    pub down_single: usize,
}

impl FeedbackStats {
    /// Single-model down-vote rate — numerator AND denominator sliced.
    pub fn downvote_rate(&self) -> f64 {
        if self.total_single == 0 {
            0.0
        } else {
            self.down_single as f64 / self.total_single as f64
        }
    }
}

/// One captured routing turn, matching `RouterTrainSample`'s decision fields.
///
/// Feature vectors are base64 float16 in Python; the Rust port keeps the
/// decision/heuristic fields and omits the numpy-encoded feature blobs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RouterTrainSample {
    pub session_key: String,
    pub turn_index: i64,
    pub ts: String,
    pub feature_schema_version: String,
    pub route_class: String,
    pub final_route_class: String,
    pub routed_tier: String,
    pub probabilities: Vec<f64>,
    pub margin: f64,
    pub confidence: f64,
    pub complaint_detected: bool,
    pub anti_downgrade_applied: bool,
    pub confidence_gate_applied: bool,
    pub large_context_floor_applied: bool,
    pub image_route: bool,
    pub exploration: bool,
    pub decision_id: Option<String>,
    pub schema_version: u32,
}

/// True when the global kill-switch env var is set truthy.
pub fn self_learning_disabled_by_env() -> bool {
    let value = std::env::var(ENV_DISABLE).unwrap_or_default();
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Sanitize an agent id into a single safe path segment.
fn safe_agent_id(agent_id: &str) -> String {
    let cleaned: String = agent_id
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        "default".to_string()
    } else {
        cleaned
    };
    let mut s = cleaned;
    s.truncate(128);
    s
}

/// The single root holding all self-learning artifacts under `base`.
pub fn router_data_root(base: &Path) -> PathBuf {
    base.join("router")
}

/// The per-agent captured-sample directory.
pub fn agent_data_dir(base: &Path, agent_id: &str) -> PathBuf {
    router_data_root(base)
        .join("data")
        .join(safe_agent_id(agent_id))
}

/// Path to the per-agent feedback JSONL.
pub fn feedback_path(base: &Path, agent_id: &str) -> PathBuf {
    agent_data_dir(base, agent_id).join(FEEDBACK_FILENAME)
}

/// Path to a per-day captured-samples JSONL.
pub fn samples_path(base: &Path, agent_id: &str, day: &str) -> PathBuf {
    agent_data_dir(base, agent_id).join(format!("samples-{day}.jsonl"))
}

/// Append one rating row; `neutral` revokes a previous rating.
pub fn write_feedback(
    base: &Path,
    agent_id: &str,
    decision_id: &str,
    session_key: &str,
    turn_index: i64,
    rating: &str,
    executed_kind: &str,
) -> Result<PathBuf> {
    if !RATINGS.contains(&rating) {
        anyhow::bail!("rating must be one of {RATINGS:?}");
    }
    let kind = if EXECUTED_KINDS.contains(&executed_kind) {
        executed_kind
    } else {
        "single"
    };
    let stamp = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let row = FeedbackRow {
        decision_id: decision_id.to_string(),
        session_key: session_key.to_string(),
        turn_index,
        rating: rating.to_string(),
        executed_kind: kind.to_string(),
        ts: stamp.clone(),
        decision_ts: stamp,
        schema_version: FEEDBACK_SCHEMA_VERSION,
    };
    let path = feedback_path(base, agent_id);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut fh = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(fh, "{}", serde_json::to_string(&row)?)?;
    Ok(path)
}

/// Read every feedback row, skipping malformed lines.
fn read_rows(path: &Path) -> Result<Vec<FeedbackRow>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(row) = serde_json::from_str::<FeedbackRow>(line) {
            rows.push(row);
        }
    }
    Ok(rows)
}

/// File-order merge: the last row per decision_id is the effective rating.
fn merged_rows(base: &Path, agent_id: &str) -> Result<HashMap<String, FeedbackRow>> {
    let mut merged: HashMap<String, FeedbackRow> = HashMap::new();
    for row in read_rows(&feedback_path(base, agent_id))? {
        if !row.decision_id.is_empty() {
            merged.insert(row.decision_id.clone(), row);
        }
    }
    Ok(merged)
}

/// Effective feedback keyed by decision_id for the sample join.
///
/// Last write per decision wins; `neutral` (revoked) entries are dropped.
pub fn load_feedback_map(base: &Path, agent_id: &str) -> Result<HashMap<String, FeedbackEntry>> {
    let mut out = HashMap::new();
    for (id, row) in merged_rows(base, agent_id)? {
        if row.rating == "up" || row.rating == "down" {
            let kind = if EXECUTED_KINDS.contains(&row.executed_kind.as_str()) {
                row.executed_kind.clone()
            } else {
                "single".to_string()
            };
            out.insert(
                id,
                FeedbackEntry {
                    rating: row.rating,
                    executed_kind: kind,
                },
            );
        }
    }
    Ok(out)
}

/// Aggregate effective ratings, optionally restricted to a decision window.
pub fn scan_feedback_stats(
    base: &Path,
    agent_id: &str,
    since_ts: Option<&str>,
) -> Result<FeedbackStats> {
    let mut total = 0usize;
    let mut up = 0usize;
    let mut down = 0usize;
    let mut total_single = 0usize;
    let mut down_single = 0usize;
    for row in merged_rows(base, agent_id)?.values() {
        if row.rating != "up" && row.rating != "down" {
            continue;
        }
        let window_ts = if row.decision_ts.is_empty() {
            &row.ts
        } else {
            &row.decision_ts
        };
        if let Some(since) = since_ts {
            if window_ts.as_str() <= since {
                continue;
            }
        }
        let is_single = row.executed_kind != "ensemble";
        total += 1;
        if is_single {
            total_single += 1;
        }
        if row.rating == "up" {
            up += 1;
        } else {
            down += 1;
            if is_single {
                down_single += 1;
            }
        }
    }
    Ok(FeedbackStats {
        total,
        up,
        down,
        total_single,
        down_single,
    })
}

/// Append one captured turn sample as a JSON line.
pub fn write_sample(base: &Path, agent_id: &str, sample: &RouterTrainSample) -> Result<PathBuf> {
    let day = Utc::now().format("%Y%m%d").to_string();
    let path = samples_path(base, agent_id, &day);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut fh = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    writeln!(fh, "{}", serde_json::to_string(sample)?)?;
    Ok(path)
}

/// Read all captured samples for an agent, in file/line order.
pub fn iter_samples(base: &Path, agent_id: &str) -> Result<Vec<RouterTrainSample>> {
    let dir = agent_data_dir(base, agent_id);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str());
        if name.is_some_and(|n| n.starts_with("samples-") && n.ends_with(".jsonl")) {
            files.push(path);
        }
    }
    files.sort();
    let mut samples = Vec::new();
    for path in files {
        for line in std::fs::read_to_string(&path)?.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(sample) = serde_json::from_str::<RouterTrainSample>(line) {
                samples.push(sample);
            }
        }
    }
    Ok(samples)
}

/// Export all captured samples for an agent as one JSONL dataset.
pub fn export_samples_jsonl(base: &Path, agent_id: &str, label: &str) -> Result<PathBuf> {
    let samples = iter_samples(base, agent_id)?;
    let dir = router_data_root(base)
        .join("datasets")
        .join(safe_agent_id(agent_id));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{label}.jsonl"));
    let mut fh = std::fs::File::create(&path)?;
    for sample in &samples {
        writeln!(fh, "{}", serde_json::to_string(sample)?)?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_base() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("squilla_feedback_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn feedback_last_write_wins_and_neutral_revokes() {
        let base = temp_base();
        write_feedback(&base, "agent-1", "d1", "s1", 0, "down", "single").unwrap();
        write_feedback(&base, "agent-1", "d1", "s1", 0, "up", "single").unwrap();
        write_feedback(&base, "agent-1", "d2", "s2", 1, "down", "single").unwrap();
        write_feedback(&base, "agent-1", "d2", "s2", 1, "neutral", "single").unwrap();

        let map = load_feedback_map(&base, "agent-1").unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("d1").map(|e| e.rating.as_str()), Some("up"));
        assert!(!map.contains_key("d2"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn feedback_stats_uses_merged_rows() {
        let base = temp_base();
        write_feedback(&base, "agent-1", "d1", "s1", 0, "down", "single").unwrap();
        write_feedback(&base, "agent-1", "d2", "s1", 1, "up", "ensemble").unwrap();
        write_feedback(&base, "agent-1", "d3", "s2", 0, "down", "single").unwrap();
        // Revision collapses into one decision.
        write_feedback(&base, "agent-1", "d3", "s2", 0, "up", "single").unwrap();

        let stats = scan_feedback_stats(&base, "agent-1", None).unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.up, 2);
        assert_eq!(stats.down, 1);
        assert_eq!(stats.total_single, 2);
        assert_eq!(stats.down_single, 1);
        assert_eq!(stats.downvote_rate(), 0.5);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn sample_roundtrip_and_dataset_export() {
        let base = temp_base();
        let sample = RouterTrainSample {
            session_key: "s1".to_string(),
            turn_index: 0,
            ts: "2026-08-08T00:00:00Z".to_string(),
            feature_schema_version: "v1".to_string(),
            route_class: "R1".to_string(),
            final_route_class: "R1".to_string(),
            routed_tier: "c1".to_string(),
            probabilities: vec![0.1, 0.6, 0.2, 0.1],
            margin: 0.4,
            confidence: 0.8,
            decision_id: Some("d9".to_string()),
            ..Default::default()
        };
        write_sample(&base, "agent-1", &sample).unwrap();

        let samples = iter_samples(&base, "agent-1").unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].session_key, "s1");
        assert_eq!(samples[0].decision_id.as_deref(), Some("d9"));

        let path = export_samples_jsonl(&base, "agent-1", "samples").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let parsed: RouterTrainSample = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(parsed.routed_tier, "c1");
        assert_eq!(parsed.confidence, 0.8);
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn safe_agent_id_sanitizes_path_segments() {
        assert_eq!(safe_agent_id(".."), "default");
        assert_eq!(safe_agent_id("  "), "default");
        assert_eq!(safe_agent_id("a/b:../c"), "a_b_.._c");
    }
}
