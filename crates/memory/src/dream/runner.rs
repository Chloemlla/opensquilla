//! Per-agent Dream runner (evidence-gated consolidation).
//!
//! Mirrors `src/opensquilla/memory/dream/runner.py`. Each run scans workspace
//! memory files past the persisted cursor, folds them into the promotion
//! evidence store, ranks candidates, rehydrates the survivors, asks the
//! injected [`DreamLlm`] for a curated MEMORY.md patch, applies it (honoring a
//! dry-run flag), persists the evidence store and cursor, then emits a JSONL
//! log row plus a receipt.

use sha2::{Digest, Sha256};
use std::collections::HashSet;

use super::DreamLlm;
use crate::dream::candidates::scan_dream_candidates;
use crate::dream::curated_apply::apply_promotion_patch;
use crate::dream::evidence::{
    mark_evidence_promoted, mark_evidence_represented, mark_evidence_skipped,
    update_promotion_evidence, write_evidence_store,
};
use crate::dream::models::{ApplyPromotionResult, PromotionPatch, PromotionPatchOperation};
use crate::dream::prompts::{parse_promotion_patch, promotion_patch_prompt};
use crate::dream::ranking::rank_promotion_candidates;
use crate::dream::receipts::write_dream_receipt;
use crate::dream::rehydrate::rehydrate_candidate;

/// Timestamp (UTC epoch seconds) of the last successful Dream batch, persisted
/// at `<memory_dir>/.dream_cursor`.
#[derive(Debug, Clone)]
pub struct DreamCursor {
    memory_dir: std::path::PathBuf,
}

impl DreamCursor {
    pub fn new(memory_dir: &std::path::Path) -> Self {
        Self {
            memory_dir: memory_dir.to_path_buf(),
        }
    }

    /// Path of the persisted cursor file.
    pub fn path(&self) -> std::path::PathBuf {
        self.memory_dir.join(".dream_cursor")
    }

    /// Load the persisted cursor, defaulting to `0.0` when absent or corrupt.
    pub fn load(&self) -> f64 {
        match std::fs::read_to_string(self.path()) {
            Ok(text) => text.trim().parse::<f64>().unwrap_or(0.0),
            Err(_) => 0.0,
        }
    }

    /// Persist the cursor.
    pub fn save(&self, ts: f64) {
        if std::fs::create_dir_all(&self.memory_dir).is_ok() {
            std::fs::write(self.path(), format!("{ts}\n")).ok();
        }
    }

    /// Delete the cursor file; a missing file is the expected idle state.
    pub fn reset(&self) {
        std::fs::remove_file(self.path()).ok();
    }
}

/// Outcome of a Dream run — emitted to logs and receipts.
#[derive(Debug, Clone, Default)]
pub struct DreamResult {
    pub files_considered: usize,
    pub files_processed: usize,
    pub evidence_status: String,
    pub apply_status: String,
    pub evidence_ms: u64,
    pub apply_ms: u64,
    pub provider_calls: usize,
    pub error: Option<String>,
    pub cursor_before: f64,
    pub cursor_after: f64,
    pub memory_md_sha_before: Option<String>,
    pub memory_md_sha_after: Option<String>,
    pub input_slimming: String,
    pub promotion_prompt_chars: usize,
    pub dry_run: bool,
    pub edit_receipt_path: Option<String>,
}

/// Runtime configuration for the Dream runner.
#[derive(Debug, Clone)]
pub struct DreamRunnerConfig {
    pub max_batch_size: usize,
    pub min_batch_size: usize,
    pub evidence_min_score: f64,
    pub evidence_negative_recurrence_threshold: i64,
    pub evidence_min_seen_count: i64,
    pub evidence_quarantine_enabled: bool,
    pub evidence_curated_writes_enabled: bool,
    pub input_slimming: String,
    pub preview_mode: bool,
    pub dry_run: bool,
}

impl Default for DreamRunnerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 20,
            min_batch_size: 1,
            evidence_min_score: 0.55,
            evidence_negative_recurrence_threshold: 2,
            evidence_min_seen_count: 1,
            evidence_quarantine_enabled: true,
            evidence_curated_writes_enabled: true,
            input_slimming: "off".to_string(),
            preview_mode: false,
            dry_run: false,
        }
    }
}

/// Per-agent Dream runner, constructed once per cron invocation.
#[derive(Clone)]
pub struct DreamRunner {
    workspace: std::path::PathBuf,
    agent_id: String,
    config: DreamRunnerConfig,
    llm: std::sync::Arc<dyn DreamLlm>,
    cursor: DreamCursor,
}

/// SHA-256 of the workspace MEMORY.md, when it exists.
fn sha256_of_memory_md(runner: &DreamRunner) -> Option<String> {
    if let Ok(bytes) = std::fs::read(runner.memory_md()) {
        Some(hex::encode(Sha256::digest(&bytes)))
    } else {
        None
    }
}

impl DreamRunner {
    pub fn new(
        workspace: &std::path::Path,
        agent_id: impl Into<String>,
        config: DreamRunnerConfig,
        llm: std::sync::Arc<dyn DreamLlm>,
    ) -> Self {
        let memory_dir = workspace.join("memory");
        Self {
            workspace: workspace.to_path_buf(),
            agent_id: agent_id.into(),
            config,
            llm,
            cursor: DreamCursor::new(&memory_dir),
        }
    }

    pub fn workspace(&self) -> &std::path::Path {
        &self.workspace
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn config(&self) -> &DreamRunnerConfig {
        &self.config
    }

    /// Count how many files are pending a dream run after the current cursor.
    pub fn pending_candidate_count(&self) -> usize {
        scan_dream_candidates(
            &self.workspace,
            self.cursor.load(),
            self.config.max_batch_size,
            &self.agent_id,
            self.config.evidence_quarantine_enabled,
        )
        .len()
    }

    /// Run the single evidence-gated Dream consolidation path.
    pub async fn run(&self) -> DreamResult {
        let dry_run = self.config.preview_mode || self.config.dry_run;
        let cursor_before = self.cursor.load();
        let mut result = DreamResult {
            cursor_before,
            memory_md_sha_before: sha256_of_memory_md(self),
            input_slimming: self.config.input_slimming.clone(),
            dry_run,
            ..Default::default()
        };

        let raw = scan_dream_candidates(
            &self.workspace,
            cursor_before,
            self.config.max_batch_size,
            &self.agent_id,
            self.config.evidence_quarantine_enabled,
        );
        result.files_considered = raw.len();

        if raw.len() < self.config.min_batch_size {
            result.cursor_after = cursor_before;
            self.emit_log(&result);
            return result;
        }

        let now_iso = chrono::Utc::now().to_rfc3339();
        let evidence_start = std::time::Instant::now();
        let (mut store, ranked) = {
            let store = update_promotion_evidence(&self.workspace, &raw, &now_iso, !dry_run);
            let ranked = rank_promotion_candidates(
                &store,
                self.config.evidence_min_score,
                self.config.evidence_negative_recurrence_threshold,
                self.config.evidence_min_seen_count,
                Some(self.config.max_batch_size),
            );
            (store, ranked)
        };
        result.evidence_status = "ok".to_string();
        result.evidence_ms = evidence_start.elapsed().as_millis() as u64;

        if ranked.is_empty() {
            let max_mtime = raw
                .iter()
                .map(|c| c.source_mtime_ns as f64 / 1e9)
                .fold(cursor_before, f64::max);
            if !dry_run {
                if let Err(e) = write_evidence_store(&self.workspace, &store) {
                    result.evidence_status = "error".to_string();
                    result.error = Some(format!("evidence: {e}"));
                    result.cursor_after = cursor_before;
                    self.emit_log(&result);
                    return result;
                }
                result.files_processed = raw.len();
                self.cursor.save(max_mtime);
                result.cursor_after = max_mtime;
            } else {
                result.cursor_after = cursor_before;
            }
            result.apply_status = "skipped".to_string();
            let candidate_paths: Vec<String> = raw.iter().map(|c| c.source_path.clone()).collect();
            result.edit_receipt_path = Some(write_dream_receipt(
                &self.workspace,
                &self.artifact_id(),
                &self.agent_id,
                dry_run,
                &candidate_paths,
                raw.len(),
                &[],
                &[],
                &ApplyPromotionResult::default(),
                "",
                cursor_before,
                result.cursor_after,
            ));
            self.emit_log(&result);
            return result;
        }

        let apply_start = std::time::Instant::now();
        let artifact_id = self.artifact_id();
        let candidate_paths: Vec<String> = raw.iter().map(|c| c.source_path.clone()).collect();
        let mut skipped_candidates: Vec<serde_json::Value> = Vec::new();
        let mut memory_backup_path = String::new();

        let apply_outcome: Result<(), String> = async {
            let current_memory = std::fs::read_to_string(self.memory_md()).unwrap_or_default();
            let prompt = promotion_patch_prompt(&current_memory, &ranked);
            result.promotion_prompt_chars = prompt.len();
            let text = self
                .llm
                .complete("", &prompt)
                .await
                .map_err(|e| e.to_string())?;
            result.provider_calls = 1;
            let patch = parse_promotion_patch(&text, &ranked)?;

            let mut live: HashSet<String> = HashSet::new();
            for candidate in &ranked {
                let rehydrated = rehydrate_candidate(&self.workspace, candidate);
                if rehydrated.ok {
                    live.insert(candidate.candidate_id.clone());
                } else {
                    let reason = rehydrated
                        .reason
                        .clone()
                        .unwrap_or_else(|| "rehydrate_failed".to_string());
                    skipped_candidates.push(serde_json::json!({
                        "candidate_id": candidate.candidate_id,
                        "reason": reason,
                    }));
                    mark_evidence_skipped(&mut store, &candidate.candidate_id, &reason);
                }
            }

            let mut filtered_operations: Vec<PromotionPatchOperation> = Vec::new();
            for mut operation in patch.operations {
                if operation.op == "skip" {
                    filtered_operations.push(operation);
                    continue;
                }
                let live_ids: Vec<String> = operation
                    .candidate_ids
                    .iter()
                    .filter(|id| live.contains(*id))
                    .cloned()
                    .collect();
                if live_ids.is_empty() {
                    continue;
                }
                operation.candidate_ids = live_ids;
                filtered_operations.push(operation);
            }
            let filtered_patch = PromotionPatch {
                operations: filtered_operations,
            };

            if !dry_run && self.config.evidence_curated_writes_enabled {
                memory_backup_path = self.backup_memory_md(&artifact_id);
            }
            let applied = apply_promotion_patch(
                &self.workspace,
                &filtered_patch,
                dry_run || !self.config.evidence_curated_writes_enabled,
            );

            if !dry_run {
                let mut promoted_ids: Vec<String> = Vec::new();
                let mut represented_ids: Vec<String> = Vec::new();
                for applied_operation in &applied.applied_operations {
                    let is_curated = matches!(
                        applied_operation.get("op").and_then(|v| v.as_str()),
                        Some("upsert") | Some("merge")
                    );
                    if !is_curated {
                        continue;
                    }
                    let Some(serde_json::Value::Array(raw_candidate_ids)) =
                        applied_operation.get("candidate_ids")
                    else {
                        continue;
                    };
                    let changed =
                        applied_operation.get("changed").and_then(|v| v.as_bool()) == Some(true);
                    let ids = raw_candidate_ids
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from));
                    if changed {
                        promoted_ids.extend(ids);
                    } else {
                        represented_ids.extend(ids);
                    }
                }
                let mut promoted_set: HashSet<String> = HashSet::new();
                promoted_ids.retain(|id| promoted_set.insert(id.clone()));
                represented_ids.retain(|id| !promoted_set.contains(id));
                mark_evidence_promoted(&mut store, &promoted_ids, &now_iso);
                mark_evidence_represented(&mut store, &represented_ids, "no_curated_change");
                write_evidence_store(&self.workspace, &store).map_err(|e| e.to_string())?;
                let max_mtime = raw
                    .iter()
                    .map(|c| c.source_mtime_ns as f64 / 1e9)
                    .fold(cursor_before, f64::max);
                result.files_processed = raw.len();
                self.cursor.save(max_mtime);
                result.cursor_after = max_mtime;
            } else {
                result.cursor_after = cursor_before;
            }

            result.memory_md_sha_after = sha256_of_memory_md(self);
            result.apply_status = "ok".to_string();
            result.apply_ms = apply_start.elapsed().as_millis() as u64;
            result.edit_receipt_path = Some(write_dream_receipt(
                &self.workspace,
                &artifact_id,
                &self.agent_id,
                dry_run,
                &candidate_paths,
                raw.len(),
                &ranked,
                &skipped_candidates,
                &applied,
                &memory_backup_path,
                cursor_before,
                result.cursor_after,
            ));
            Ok(())
        }
        .await;

        if let Err(e) = apply_outcome {
            result.apply_status = "error".to_string();
            result.apply_ms = apply_start.elapsed().as_millis() as u64;
            result.error = Some(format!("apply: {e}"));
            result.cursor_after = cursor_before;
        }

        self.emit_log(&result);
        result
    }

    fn memory_dir(&self) -> std::path::PathBuf {
        self.workspace.join("memory")
    }

    fn memory_md(&self) -> std::path::PathBuf {
        self.workspace.join("MEMORY.md")
    }

    fn artifact_id(&self) -> String {
        format!("{}-{}", self.agent_id, chrono::Utc::now().timestamp_millis())
    }

    fn workspace_relative(&self, path: &std::path::Path) -> String {
        match path.strip_prefix(&self.workspace) {
            Ok(rel) => rel.to_string_lossy().replace('\\', "/"),
            Err(_) => path.display().to_string(),
        }
    }

    fn backup_memory_md(&self, artifact_id: &str) -> String {
        let backup_dir = self.memory_dir().join(".dream_backups").join(artifact_id);
        let backup_path = backup_dir.join("MEMORY.md");
        std::fs::create_dir_all(&backup_dir).ok();
        let content = std::fs::read(self.memory_md()).unwrap_or_default();
        std::fs::write(&backup_path, content).ok();
        self.workspace_relative(&backup_path)
    }

    fn emit_log(&self, result: &DreamResult) {
        let log_dir = self.workspace.join("logs");
        if std::fs::create_dir_all(&log_dir).is_err() {
            return;
        }
        let today = chrono::Utc::now().format("%Y-%m-%d");
        let path = log_dir.join(format!("dream-{}-{}.jsonl", self.agent_id, today));
        let row = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "agent_id": self.agent_id,
            "cursor_before": result.cursor_before,
            "cursor_after": result.cursor_after,
            "files_considered": result.files_considered,
            "files_processed": result.files_processed,
            "evidence_ms": result.evidence_ms,
            "evidence_status": result.evidence_status,
            "apply_ms": result.apply_ms,
            "apply_status": result.apply_status,
            "provider_calls": result.provider_calls,
            "memory_md_sha_before": result.memory_md_sha_before,
            "memory_md_sha_after": result.memory_md_sha_after,
            "input_slimming": result.input_slimming,
            "promotion_prompt_chars": result.promotion_prompt_chars,
            "dry_run": result.dry_run,
            "edit_receipt_path": result.edit_receipt_path,
            "error": result.error,
        });
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            use std::io::Write;
            let line = format!("{}\n", serde_json::to_string(&row).unwrap_or_default());
            file.write_all(line.as_bytes()).ok();
        }
    }
}
