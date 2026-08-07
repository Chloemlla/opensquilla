//! Per-agent Dream runner (evidence-gated consolidation).
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/runner.py`. The
//! clustering/merging path is implemented on [`super::DreamEngine`]; this
//! module stubs the higher-level evidence → rank → rehydrate → apply → receipt
//! pipeline that the Python runner orchestrates per cron invocation.

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
    ///
    /// TODO(parity): implement file load / parse from runner.py.
    pub fn load(&self) -> f64 {
        0.0
    }

    /// Persist the cursor.
    ///
    /// TODO(parity): implement file write from runner.py.
    pub fn save(&self, ts: f64) {
        let _ = ts;
    }

    /// Delete the cursor file.
    ///
    /// TODO(parity): implement unlink from runner.py.
    pub fn reset(&self) {
        let _ = ();
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

/// Per-agent Dream runner, constructed once per cron invocation.
#[derive(Debug, Clone)]
pub struct DreamRunner {
    workspace: std::path::PathBuf,
    agent_id: String,
}

impl DreamRunner {
    pub fn new(workspace: &std::path::Path, agent_id: impl Into<String>) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            agent_id: agent_id.into(),
        }
    }

    pub fn workspace(&self) -> &std::path::Path {
        &self.workspace
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Count how many files are pending a dream run after the current cursor.
    ///
    /// TODO(parity): implement candidate scanning from runner.py.
    pub fn pending_candidate_count(&self) -> usize {
        0
    }

    /// Run the single evidence-gated Dream consolidation path.
    ///
    /// TODO(parity): implement the evidence → rank → rehydrate → apply →
    /// receipt pipeline from runner.py.
    pub async fn run(&self) -> DreamResult {
        DreamResult::default()
    }
}
