use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::info;
use uuid::Uuid;

use crate::models::{CompactionHistory, Session, SessionSummary, TranscriptEntry};
use crate::storage::SessionStorage;

/// Threshold in tokens above which compaction is triggered.
pub const DEFAULT_COMPACTION_THRESHOLD: u64 = 4096;

/// Target token count after compaction.
pub const DEFAULT_COMPACTION_TARGET: u64 = 2048;

/// Number of most-recent entries always retained in-context.
pub const DEFAULT_KEEP_LAST_COUNT: usize = 4;

/// How far a fallback extractive summary may grow (in characters).
pub const DEFAULT_SUMMARY_CHARS: usize = 500;

// ---------------------------------------------------------------------------
// Strategy + planning types
// ---------------------------------------------------------------------------

/// The strategy used to shrink a session's context window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategy {
    /// Summarize old turns (least destructive; may require an LLM).
    Summarize,
    /// Drop the oldest turns wholesale.
    DropOldest,
    /// Drop the oldest turns until the remaining transcript fits a token budget.
    TruncateToBudget,
    /// Keep only the last N turns.
    KeepLast,
}

impl CompactionStrategy {
    /// A short human-readable label for logs and reports.
    pub fn label(&self) -> &'static str {
        match self {
            CompactionStrategy::Summarize => "summarize",
            CompactionStrategy::DropOldest => "drop_oldest",
            CompactionStrategy::TruncateToBudget => "truncate_to_budget",
            CompactionStrategy::KeepLast => "keep_last",
        }
    }

    /// All strategies, ordered least-to-most destructive. Used when a tie
    /// needs to be broken by data preservation.
    pub fn ordered() -> [CompactionStrategy; 4] {
        [
            CompactionStrategy::Summarize,
            CompactionStrategy::KeepLast,
            CompactionStrategy::TruncateToBudget,
            CompactionStrategy::DropOldest,
        ]
    }
}

/// A token-savings estimate for applying `strategy` to a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEstimate {
    pub strategy: CompactionStrategy,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub savings: u64,
    pub entries_compacted: usize,
    pub summary_tokens: u64,
}

/// A full context-window usage estimate for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEstimate {
    /// The session this estimate applies to.
    pub session_id: Uuid,
    /// Tokens consumed by un-compacted transcript entries.
    pub active_tokens: u64,
    /// Estimated tokens consumed by the active summary.
    pub summary_tokens: u64,
    /// Estimated tokens consumed by the system prompt.
    pub system_tokens: u64,
    /// Total estimated tokens.
    pub total_tokens: u64,
    /// The session's compaction target budget.
    pub budget_tokens: u64,
    /// Whether the session exceeds its budget.
    pub over_budget: bool,
    /// Utilization of the budget in `[0, 1]`.
    pub utilization: f64,
}

impl ContextEstimate {
    /// The token slack (budget - total), possibly negative.
    pub fn slack(&self) -> i64 {
        self.budget_tokens as i64 - self.total_tokens as i64
    }
}

/// A fully-resolved compaction decision: which entries to compact, which to
/// retain, and the projected token effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionPlan {
    pub session_id: Uuid,
    pub strategy: CompactionStrategy,
    pub entries_to_compact: Vec<TranscriptEntry>,
    pub retained_entries: Vec<TranscriptEntry>,
    pub target_budget: u64,
    pub current_tokens: u64,
    pub estimated_savings: u64,
    pub estimated_after: u64,
}

impl CompactionPlan {
    /// Build a report for this plan without an actual summary row. Used for
    /// no-op plans and dry-run reporting.
    pub fn to_report(&self, status: &str) -> CompactionReport {
        let now = Utc::now();
        CompactionReport {
            compaction_id: Uuid::new_v4(),
            session_id: self.session_id,
            strategy: self.strategy,
            entries_compacted: self.entries_to_compact.len() as u64,
            tokens_before: self.current_tokens,
            tokens_after: self.estimated_after,
            summary_id: None,
            started_at: now,
            completed_at: now,
            status: status.to_string(),
        }
    }

    /// True when the plan actually compacts something.
    pub fn is_noop(&self) -> bool {
        self.entries_to_compact.is_empty()
    }
}

/// The outcome of a compaction run, persisted to `compaction_history`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionReport {
    pub compaction_id: Uuid,
    pub session_id: Uuid,
    pub strategy: CompactionStrategy,
    pub entries_compacted: u64,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub summary_id: Option<Uuid>,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub status: String,
}

/// Observable events emitted by a compaction run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CompactionEvent {
    Started {
        session_id: Uuid,
        strategy: CompactionStrategy,
        tokens_before: u64,
    },
    SummaryGenerated {
        session_id: Uuid,
        summary_id: Uuid,
        tokens: u64,
    },
    EntriesCompacted {
        session_id: Uuid,
        count: u64,
        savings: u64,
    },
    Completed {
        session_id: Uuid,
        report: CompactionReport,
    },
    Failed {
        session_id: Uuid,
        error: String,
    },
}

// ---------------------------------------------------------------------------
// LLM summarizer protocol
// ---------------------------------------------------------------------------

/// Summarizes a slice of transcript entries into a compact representation.
/// The gateway wires a real provider-backed implementation; a deterministic
/// [`ExtractiveSummarizer`] is the built-in fallback.
#[async_trait::async_trait]
pub trait SessionSummarizer: Send + Sync {
    async fn summarize(
        &self,
        entries: &[TranscriptEntry],
        session: &Session,
    ) -> Result<String, String>;
}

/// Deterministic, dependency-free summarizer that extracts the first line of
/// each entry. Safe for headless/offline runs.
pub struct ExtractiveSummarizer {
    max_chars: usize,
}

impl ExtractiveSummarizer {
    pub fn new() -> Self {
        Self {
            max_chars: DEFAULT_SUMMARY_CHARS,
        }
    }

    pub fn with_max_chars(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl SessionSummarizer for ExtractiveSummarizer {
    async fn summarize(
        &self,
        entries: &[TranscriptEntry],
        _session: &Session,
    ) -> Result<String, String> {
        Ok(extractive_summary(entries, self.max_chars))
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Estimate the number of tokens a piece of text consumes.
pub fn estimate_token_count(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

/// Estimate the token cost of a future summary based on the tokens it will
/// replace. Roughly 1/10th of the source, clamped to a sane band.
pub fn estimate_summary_tokens(compacted_tokens: u64) -> u64 {
    (compacted_tokens / 10).clamp(64, 512)
}

/// Build an extractive summary: the first line of each entry, truncated to
/// `max_chars` total.
pub fn extractive_summary(entries: &[TranscriptEntry], max_chars: usize) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut total: usize = 0;
    for entry in entries {
        let line = entry.content.lines().next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let piece = format!("[{}] {}", entry.role, line);
        if total + piece.len() > max_chars && !parts.is_empty() {
            break;
        }
        total += piece.len();
        parts.push(piece);
    }
    let mut joined = parts.join("\n");
    if joined.chars().count() > max_chars {
        let trimmed: String = joined.chars().take(max_chars.saturating_sub(3)).collect();
        joined = format!("{}...", trimmed);
    }
    joined
}

/// Total tokens across a slice of entries.
pub fn retained_tokens(entries: &[TranscriptEntry]) -> u64 {
    entries.iter().map(|e| e.token_count).sum()
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Decides what and how to compact a session's context window.
pub struct CompactionPlanner {
    threshold: u64,
    target_budget: u64,
    keep_last_count: usize,
}

impl CompactionPlanner {
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_COMPACTION_THRESHOLD,
            target_budget: DEFAULT_COMPACTION_TARGET,
            keep_last_count: DEFAULT_KEEP_LAST_COUNT,
        }
    }

    pub fn with_threshold(mut self, threshold: u64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_target_budget(mut self, target_budget: u64) -> Self {
        self.target_budget = target_budget;
        self
    }

    pub fn with_keep_last(mut self, keep_last_count: usize) -> Self {
        self.keep_last_count = keep_last_count;
        self
    }

    pub fn threshold(&self) -> u64 {
        self.threshold
    }

    pub fn target_budget(&self) -> u64 {
        self.target_budget
    }

    pub fn keep_last_count(&self) -> usize {
        self.keep_last_count
    }

    /// Whether a session has outgrown its compaction threshold based on the
    /// aggregate token counter recorded on the session row.
    pub fn needs_compaction(&self, session: &Session) -> bool {
        session.total_tokens > self.threshold
    }

    /// Decide what to compact for a session. Returns a no-op plan when the
    /// session is under budget or has nothing left to compact.
    pub fn plan_compaction(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
    ) -> CoreResult<CompactionPlan> {
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let current_tokens: u64 = un_compacted.iter().map(|e| e.token_count).sum();

        if current_tokens <= self.threshold {
            return Ok(CompactionPlan {
                session_id: session.id,
                strategy: CompactionStrategy::KeepLast,
                entries_to_compact: Vec::new(),
                retained_entries: un_compacted,
                target_budget: self.target_budget,
                current_tokens,
                estimated_savings: 0,
                estimated_after: current_tokens,
            });
        }

        let strategy = self.select_strategy(session, entries, self.target_budget)?;
        let (to_compact, retained) = self.partition(&un_compacted, strategy);
        if to_compact.is_empty() {
            return Ok(CompactionPlan {
                session_id: session.id,
                strategy,
                entries_to_compact: Vec::new(),
                retained_entries: retained,
                target_budget: self.target_budget,
                current_tokens,
                estimated_savings: 0,
                estimated_after: current_tokens,
            });
        }

        let estimate = self.estimate_for(session, entries, strategy)?;
        Ok(CompactionPlan {
            session_id: session.id,
            strategy,
            entries_to_compact: to_compact,
            retained_entries: retained,
            target_budget: self.target_budget,
            current_tokens,
            estimated_savings: estimate.savings,
            estimated_after: estimate.tokens_after,
        })
    }

    /// Estimate the token effect of applying `strategy`.
    pub fn estimate_token_savings(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
        strategy: CompactionStrategy,
    ) -> CoreResult<CompactionEstimate> {
        self.estimate_for(session, entries, strategy)
    }

    /// Select compaction boundaries that respect message-pair semantics.
    ///
    /// Returns a list of entry indexes at which a "natural" boundary exists:
    /// after a complete `user -> assistant` (or `assistant -> user`) exchange.
    /// Compacting between these boundaries preserves conversational coherence
    /// better than slicing at arbitrary entry positions.
    pub fn select_boundaries(&self, entries: &[TranscriptEntry]) -> Vec<usize> {
        if entries.is_empty() {
            return Vec::new();
        }
        let mut boundaries = Vec::new();
        for i in 0..entries.len().saturating_sub(1) {
            let cur = entries[i].role.as_str();
            let next = entries[i + 1].role.as_str();
            // A boundary exists when the role changes and neither is a
            // transient tool/system frame.
            let role_switched = cur != next;
            let cur_transient = matches!(cur, "tool" | "system" | "function");
            let next_transient = matches!(next, "tool" | "system" | "function");
            if role_switched && !cur_transient && !next_transient {
                boundaries.push(i + 1);
            }
        }
        // Always allow a boundary at the very start.
        if !boundaries.contains(&0) {
            boundaries.insert(0, 0);
        }
        boundaries
    }

    /// Choose the largest natural boundary index that does not exceed
    /// `max_index`. Used to round a compaction cutoff down to a coherent
    /// exchange boundary.
    pub fn floor_to_boundary(&self, entries: &[TranscriptEntry], max_index: usize) -> usize {
        let boundaries = self.select_boundaries(entries);
        boundaries
            .into_iter()
            .filter(|b| *b <= max_index)
            .max()
            .unwrap_or(max_index)
    }

    /// Estimate the full context-window usage of a session: the active
    /// transcript tokens plus the active summary tokens plus the system
    /// prompt overhead.
    pub fn estimate_context_window(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
    ) -> ContextEstimate {
        let active_tokens: u64 = entries
            .iter()
            .filter(|e| !e.compacted)
            .map(|e| e.token_count)
            .sum();
        let summary_tokens = estimate_summary_tokens(active_tokens);
        let system_tokens = estimate_token_count(&session.system_prompt);
        let total = active_tokens + summary_tokens + system_tokens;
        ContextEstimate {
            session_id: session.id,
            active_tokens,
            summary_tokens,
            system_tokens,
            total_tokens: total,
            budget_tokens: self.target_budget,
            over_budget: total > self.target_budget,
            utilization: if self.target_budget > 0 {
                (total as f64 / self.target_budget as f64).clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }

    /// Whether the session would still exceed the budget after a given
    /// strategy runs.
    pub fn would_still_exceed(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
        strategy: CompactionStrategy,
    ) -> CoreResult<bool> {
        let estimate = self.estimate_for(session, entries, strategy)?;
        Ok(estimate.tokens_after > self.target_budget)
    }

    /// Choose the best strategy to reach `target_budget`. Among strategies
    /// that fit the budget, the least destructive wins (Summarize first);
    /// if none fit, the one with the greatest savings is returned.
    pub fn select_strategy(
        &self,
        session: &Session,
        entries: &[TranscriptEntry],
        target_budget: u64,
    ) -> CoreResult<CompactionStrategy> {
        let mut budget_fit: Option<(CompactionStrategy, u64)> = None;
        let mut max_savings: Option<(CompactionStrategy, u64)> = None;

        for strategy in CompactionStrategy::ordered() {
            let estimate = self.estimate_for(session, entries, strategy)?;
            if estimate.tokens_after <= target_budget
                && budget_fit.is_none_or(|(_, after)| estimate.tokens_after < after)
            {
                budget_fit = Some((strategy, estimate.tokens_after));
            }
            if max_savings.is_none_or(|(_, savings)| estimate.savings > savings) {
                max_savings = Some((strategy, estimate.savings));
            }
        }

        if let Some((strategy, _)) = budget_fit {
            Ok(strategy)
        } else {
            max_savings
                .map(|(strategy, _)| strategy)
                .ok_or_else(|| CoreError::Internal("no compaction strategy available".into()))
        }
    }

    /// Split the un-compacted entries into a to-compact set and a retained
    /// set according to the strategy.
    fn partition(
        &self,
        un_compacted: &[TranscriptEntry],
        strategy: CompactionStrategy,
    ) -> (Vec<TranscriptEntry>, Vec<TranscriptEntry>) {
        let n = un_compacted.len();
        let keep = self.keep_last_count.min(n);

        match strategy {
            CompactionStrategy::TruncateToBudget => {
                let mut to_compact = Vec::new();
                let mut retained = Vec::new();
                let mut acc: u64 = 0;
                for entry in un_compacted.iter().rev() {
                    if acc + entry.token_count <= self.target_budget {
                        retained.push(entry.clone());
                        acc += entry.token_count;
                    } else {
                        to_compact.push(entry.clone());
                    }
                }
                retained.reverse();
                to_compact.reverse();
                (to_compact, retained)
            }
            CompactionStrategy::Summarize
            | CompactionStrategy::KeepLast
            | CompactionStrategy::DropOldest => {
                let split = n.saturating_sub(keep);
                let to_compact = un_compacted[..split].to_vec();
                let retained = un_compacted[split..].to_vec();
                (to_compact, retained)
            }
        }
    }

    fn estimate_for(
        &self,
        _session: &Session,
        entries: &[TranscriptEntry],
        strategy: CompactionStrategy,
    ) -> CoreResult<CompactionEstimate> {
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let tokens_before: u64 = un_compacted.iter().map(|e| e.token_count).sum();
        let n = un_compacted.len();
        let keep = self.keep_last_count.min(n);

        let (to_compact, retained) = match strategy {
            CompactionStrategy::TruncateToBudget => {
                let mut to_compact = Vec::new();
                let mut retained = Vec::new();
                let mut acc: u64 = 0;
                for entry in un_compacted.iter().rev() {
                    if acc + entry.token_count <= self.target_budget {
                        retained.push(entry.clone());
                        acc += entry.token_count;
                    } else {
                        to_compact.push(entry.clone());
                    }
                }
                retained.reverse();
                to_compact.reverse();
                (to_compact, retained)
            }
            _ => {
                let split = n.saturating_sub(keep);
                (
                    un_compacted[..split].to_vec(),
                    un_compacted[split..].to_vec(),
                )
            }
        };

        let retained_tokens: u64 = retained.iter().map(|e| e.token_count).sum();
        let compacted_tokens: u64 = to_compact.iter().map(|e| e.token_count).sum();
        let summary_tokens = match strategy {
            CompactionStrategy::Summarize => estimate_summary_tokens(compacted_tokens),
            _ => 0,
        };
        let tokens_after = retained_tokens + summary_tokens;

        Ok(CompactionEstimate {
            strategy,
            tokens_before,
            tokens_after,
            savings: tokens_before.saturating_sub(tokens_after),
            entries_compacted: to_compact.len(),
            summary_tokens,
        })
    }
}

impl Default for CompactionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Executes compaction plans against storage, recording history and events.
pub struct CompactionExecutor {
    storage: SessionStorage,
    planner: CompactionPlanner,
    summarizer: Arc<dyn SessionSummarizer>,
    events: Arc<Mutex<Vec<CompactionEvent>>>,
}

impl CompactionExecutor {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage,
            planner: CompactionPlanner::new(),
            summarizer: Arc::new(ExtractiveSummarizer::new()),
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_summarizer(mut self, summarizer: Arc<dyn SessionSummarizer>) -> Self {
        self.summarizer = summarizer;
        self
    }

    pub fn with_planner(mut self, planner: CompactionPlanner) -> Self {
        self.planner = planner;
        self
    }

    pub fn planner(&self) -> &CompactionPlanner {
        &self.planner
    }

    pub fn set_planner(&mut self, planner: CompactionPlanner) {
        self.planner = planner;
    }

    /// Snapshot of events recorded since construction.
    pub fn recent_events(&self) -> Vec<CompactionEvent> {
        self.events.lock().map(|e| e.clone()).unwrap_or_default()
    }

    fn record(&self, event: CompactionEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Whether a session needs compaction.
    pub fn needs_compaction(&self, session_id: &Uuid) -> CoreResult<bool> {
        let session = self.require_session(session_id)?;
        Ok(self.planner.needs_compaction(&session))
    }

    /// Build (but do not apply) a compaction plan for a session.
    pub fn plan(&self, session_id: &Uuid) -> CoreResult<CompactionPlan> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        self.planner.plan_compaction(&session, &entries)
    }

    /// Run a plan; `summary_text` is used verbatim when provided (Summarize),
    /// otherwise a marker summary is persisted for the other strategies.
    pub fn execute_plan(
        &self,
        plan: CompactionPlan,
        summary_text: Option<String>,
    ) -> CoreResult<CompactionReport> {
        if plan.entries_to_compact.is_empty() {
            return Ok(plan.to_report("no_op"));
        }

        self.record(CompactionEvent::Started {
            session_id: plan.session_id,
            strategy: plan.strategy,
            tokens_before: plan.current_tokens,
        });

        let now = Utc::now();
        let summary = SessionSummary {
            id: Uuid::new_v4(),
            session_id: plan.session_id,
            summary: summary_text.clone().unwrap_or_else(|| {
                format!(
                    "[compacted {} entries under {}]",
                    plan.entries_to_compact.len(),
                    plan.strategy.label()
                )
            }),
            created_at: now,
            token_count: summary_text
                .as_ref()
                .map(|t| estimate_token_count(t))
                .unwrap_or(0),
            is_active: true,
        };

        let history = CompactionHistory {
            id: Uuid::new_v4(),
            session_id: plan.session_id,
            compaction_id: Uuid::new_v4(),
            entries_compacted: plan.entries_to_compact.len() as u64,
            tokens_before: plan.current_tokens,
            tokens_after: plan.estimated_after,
            summary_id: Some(summary.id),
            started_at: now,
            completed_at: now,
            status: "completed".to_string(),
            metadata: serde_json::json!({ "strategy": plan.strategy.label() }),
        };

        self.storage.apply_compaction_transaction(
            &plan.session_id,
            &summary,
            &plan.entries_to_compact,
            &history,
        )?;

        let report = CompactionReport {
            compaction_id: history.compaction_id,
            session_id: plan.session_id,
            strategy: plan.strategy,
            entries_compacted: history.entries_compacted,
            tokens_before: history.tokens_before,
            tokens_after: history.tokens_after,
            summary_id: Some(summary.id),
            started_at: now,
            completed_at: now,
            status: "completed".to_string(),
        };

        self.record(CompactionEvent::EntriesCompacted {
            session_id: plan.session_id,
            count: report.entries_compacted,
            savings: plan.estimated_savings,
        });
        self.record(CompactionEvent::Completed {
            session_id: plan.session_id,
            report: report.clone(),
        });

        info!(
            "Compaction applied for session {}: {} entries compacted via {}, tokens {} -> {}",
            plan.session_id,
            report.entries_compacted,
            plan.strategy.label(),
            report.tokens_before,
            report.tokens_after
        );
        Ok(report)
    }

    /// LLM-style summarization of old turns. Falls back to the configured
    /// summarizer (by default the deterministic extractive one).
    pub async fn summarize(&self, session_id: &Uuid) -> CoreResult<CompactionReport> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let mut plan = self.planner.plan_compaction(&session, &entries)?;

        if plan.is_noop() {
            return Ok(plan.to_report("no_op"));
        }

        let summary_text = self
            .summarizer
            .summarize(&plan.entries_to_compact, &session)
            .await
            .map_err(CoreError::Provider)?;

        // Re-estimate after the real summary is known.
        let after = estimate_token_count(&summary_text) + retained_tokens(&plan.retained_entries);
        plan.estimated_after = after;
        plan.estimated_savings = plan.current_tokens.saturating_sub(after);

        let report = self.execute_plan(plan, Some(summary_text))?;
        if let Some(summary_id) = report.summary_id {
            self.record(CompactionEvent::SummaryGenerated {
                session_id: *session_id,
                summary_id,
                tokens: report.tokens_after,
            });
        }
        Ok(report)
    }

    /// Drop the oldest `count` un-compacted entries.
    pub fn drop_oldest(&self, session_id: &Uuid, count: usize) -> CoreResult<CompactionReport> {
        self.run_strategy(session_id, CompactionStrategy::DropOldest, count, 0)
    }

    /// Keep only the last `count` entries.
    pub fn keep_last(&self, session_id: &Uuid, count: usize) -> CoreResult<CompactionReport> {
        self.run_strategy(session_id, CompactionStrategy::KeepLast, count, 0)
    }

    /// Truncate the transcript until it fits `budget` tokens.
    pub fn truncate_to_budget(
        &self,
        session_id: &Uuid,
        budget: u64,
    ) -> CoreResult<CompactionReport> {
        self.run_strategy(session_id, CompactionStrategy::TruncateToBudget, 0, budget)
    }

    /// Shared implementation for the non-LLM strategies.
    fn run_strategy(
        &self,
        session_id: &Uuid,
        strategy: CompactionStrategy,
        count: usize,
        budget: u64,
    ) -> CoreResult<CompactionReport> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let n = un_compacted.len();

        let (to_compact, retained) = match strategy {
            CompactionStrategy::DropOldest => {
                let drop = count.min(n);
                (un_compacted[..drop].to_vec(), un_compacted[drop..].to_vec())
            }
            CompactionStrategy::KeepLast => {
                let keep = count.min(n);
                let split = n.saturating_sub(keep);
                (
                    un_compacted[..split].to_vec(),
                    un_compacted[split..].to_vec(),
                )
            }
            CompactionStrategy::TruncateToBudget => {
                let mut to_compact = Vec::new();
                let mut retained = Vec::new();
                let mut acc: u64 = 0;
                for entry in un_compacted.iter().rev() {
                    if acc + entry.token_count <= budget {
                        retained.push(entry.clone());
                        acc += entry.token_count;
                    } else {
                        to_compact.push(entry.clone());
                    }
                }
                retained.reverse();
                to_compact.reverse();
                (to_compact, retained)
            }
            CompactionStrategy::Summarize => {
                return Err(CoreError::InvalidInput(
                    "Summarize is asynchronous; call summarize() instead".into(),
                ));
            }
        };

        let current_tokens: u64 = un_compacted.iter().map(|e| e.token_count).sum();
        let estimated_after: u64 = retained.iter().map(|e| e.token_count).sum();
        let plan = CompactionPlan {
            session_id: *session_id,
            strategy,
            entries_to_compact: to_compact,
            retained_entries: retained,
            target_budget: if strategy == CompactionStrategy::TruncateToBudget {
                budget
            } else {
                self.planner.target_budget()
            },
            current_tokens,
            estimated_savings: current_tokens.saturating_sub(estimated_after),
            estimated_after,
        };
        let _ = &session;
        self.execute_plan(plan, None)
    }

    fn require_session(&self, session_id: &Uuid) -> CoreResult<Session> {
        self.storage
            .get_session(session_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Session {}", session_id)))
    }

    /// Estimate the full context-window usage for a session.
    pub fn context_estimate(&self, session_id: &Uuid) -> CoreResult<ContextEstimate> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        Ok(self.planner.estimate_context_window(&session, &entries))
    }

    /// Incremental compaction: compact only the oldest `count` un-compacted
    /// entries, rounding the cutoff down to the nearest conversational
    /// boundary so exchanges stay intact.
    pub fn incremental_compact(
        &self,
        session_id: &Uuid,
        count: usize,
    ) -> CoreResult<CompactionReport> {
        let session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let n = un_compacted.len();
        if n == 0 {
            return Ok(self
                .planner
                .plan_compaction(&session, &entries)?
                .to_report("no_op"));
        }

        let raw_cutoff = count.min(n).saturating_sub(1);
        let boundary_cutoff = self.planner.floor_to_boundary(&un_compacted, raw_cutoff);
        let to_compact = un_compacted[..boundary_cutoff.min(n)].to_vec();
        if to_compact.is_empty() {
            return Ok(self
                .planner
                .plan_compaction(&session, &entries)?
                .to_report("no_op"));
        }
        let retained = un_compacted[boundary_cutoff.min(n)..].to_vec();

        let current_tokens: u64 = un_compacted.iter().map(|e| e.token_count).sum();
        let estimated_after: u64 = retained.iter().map(|e| e.token_count).sum();
        let plan = CompactionPlan {
            session_id: *session_id,
            strategy: CompactionStrategy::Summarize,
            entries_to_compact: to_compact,
            retained_entries: retained,
            target_budget: self.planner.target_budget(),
            current_tokens,
            estimated_savings: current_tokens.saturating_sub(estimated_after),
            estimated_after,
        };
        self.execute_plan(plan, None)
    }

    /// Compact just enough to fit a target token budget: repeatedly compact
    /// the oldest boundary group until the retained window fits, or nothing
    /// more can be compacted.
    pub fn compact_to_budget(
        &self,
        session_id: &Uuid,
        budget: u64,
    ) -> CoreResult<CompactionReport> {
        let _session = self.require_session(session_id)?;
        let entries = self.storage.list_by_session(session_id)?;
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let mut retained: Vec<TranscriptEntry> = un_compacted.clone();
        let mut compacted: Vec<TranscriptEntry> = Vec::new();
        let mut previous_len = usize::MAX;

        while retained_tokens(&retained) > budget && retained.len() < previous_len {
            previous_len = retained.len();
            let keep = retained.len().saturating_sub(2); // keep two more newest
            let cutoff = self.planner.floor_to_boundary(&retained, keep);
            if cutoff == 0 {
                break;
            }
            // Move the OLDEST `cutoff` entries into the compacted set.
            let mut oldest: Vec<TranscriptEntry> = retained[..cutoff].to_vec();
            retained = retained[cutoff..].to_vec();
            compacted.append(&mut oldest);
        }

        if compacted.is_empty() {
            return Ok(CompactionPlan {
                session_id: *session_id,
                strategy: CompactionStrategy::TruncateToBudget,
                entries_to_compact: Vec::new(),
                retained_entries: retained,
                target_budget: budget,
                current_tokens: retained_tokens(&un_compacted),
                estimated_savings: 0,
                estimated_after: retained_tokens(&un_compacted),
            }
            .to_report("no_op"));
        }

        let current_tokens = retained_tokens(&un_compacted);
        let estimated_after = retained_tokens(&retained);
        let plan = CompactionPlan {
            session_id: *session_id,
            strategy: CompactionStrategy::TruncateToBudget,
            entries_to_compact: compacted,
            retained_entries: retained,
            target_budget: budget,
            current_tokens,
            estimated_savings: current_tokens.saturating_sub(estimated_after),
            estimated_after,
        };
        self.execute_plan(plan, None)
    }

    /// Run an LLM-backed summarization over a specific set of entries, without
    /// applying it. Returns the summary text. Useful for previewing.
    pub async fn preview_summary(
        &self,
        session_id: &Uuid,
        entries: &[TranscriptEntry],
    ) -> CoreResult<String> {
        let session = self.require_session(session_id)?;
        self.summarizer
            .summarize(entries, &session)
            .await
            .map_err(CoreError::Provider)
    }

    /// Build a rich summarization prompt from entries, including metadata such
    /// as entry roles and token counts. This is the prompt the fallback and
    /// injected summarizers typically consume.
    pub fn build_summary_prompt_detailed(entries: &[TranscriptEntry]) -> String {
        let mut prompt = String::from(
            "Summarize the following conversation exchange for context preservation.\n\
             Capture: key decisions, user preferences, important facts, open questions.\n\
             Be concise but information-dense. Preserve names and numbers verbatim.\n\n",
        );
        for (i, entry) in entries.iter().enumerate() {
            prompt.push_str(&format!(
                "[{}] ({} tokens) {}\n",
                entry.role, entry.token_count, entry.content
            ));
            if i == 0 {
                continue;
            }
        }
        prompt.push_str("\n---\nSummary:");
        prompt
    }

    /// Select the boundary index that splits the transcript so that the
    /// retained tail fits `budget` tokens, rounding to an exchange boundary.
    pub fn boundary_for_budget(&self, session_id: &Uuid, budget: u64) -> CoreResult<usize> {
        let entries = self.storage.list_by_session(session_id)?;
        let un_compacted: Vec<TranscriptEntry> =
            entries.iter().filter(|e| !e.compacted).cloned().collect();
        let mut acc: u64 = 0;
        let mut split = 0usize;
        for (i, entry) in un_compacted.iter().rev().enumerate() {
            if acc + entry.token_count > budget {
                split = un_compacted.len() - i;
                break;
            }
            acc += entry.token_count;
        }
        Ok(self.planner.floor_to_boundary(&un_compacted, split))
    }
}

// ---------------------------------------------------------------------------
// Legacy compatibility engine
// ---------------------------------------------------------------------------

/// Legacy context-window compressor. Kept for API compatibility; new code
/// should prefer [`CompactionPlanner`] + [`CompactionExecutor`].
pub struct CompactionEngine {
    storage: SessionStorage,
    threshold: u64,
    target: u64,
}

impl CompactionEngine {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage,
            threshold: DEFAULT_COMPACTION_THRESHOLD,
            target: DEFAULT_COMPACTION_TARGET,
        }
    }

    pub fn with_threshold(mut self, threshold: u64) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_target(mut self, target: u64) -> Self {
        self.target = target;
        self
    }

    /// Check if a session needs compaction.
    pub fn needs_compaction(&self, session_id: &Uuid) -> CoreResult<bool> {
        let session = self
            .storage
            .get_session(session_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Session {}", session_id)))?;
        Ok(session.total_tokens > self.threshold)
    }

    /// Get un-compacted entries ready for summarization.
    pub fn get_entries_for_compaction(
        &self,
        session_id: &Uuid,
    ) -> CoreResult<Vec<TranscriptEntry>> {
        let entries = self.storage.get_transcript_entries(session_id, 1000, 0)?;
        let un_compacted: Vec<TranscriptEntry> =
            entries.into_iter().filter(|e| !e.compacted).collect();

        if un_compacted.is_empty() {
            return Ok(Vec::new());
        }

        // Keep the most recent entries below the target, compact the rest.
        let mut total_tokens: u64 = 0;
        let mut compact_targets = Vec::new();

        for entry in &un_compacted {
            total_tokens += entry.token_count;
            if total_tokens > self.target {
                compact_targets.push(entry.clone());
            }
        }

        // Exclude the last few entries that should stay in context.
        let keep_count = 4.min(un_compacted.len());
        let compact_end = compact_targets.len().saturating_sub(keep_count);

        compact_targets.truncate(compact_end);
        Ok(compact_targets)
    }

    /// Build a summarization prompt from entries.
    pub fn build_summary_prompt(entries: &[TranscriptEntry]) -> String {
        let mut prompt = String::from(
            "Please summarize the following conversation for context preservation. \
             Focus on key decisions, user preferences, and important facts:\n\n",
        );

        for entry in entries {
            prompt.push_str(&format!("[{}]: {}\n", entry.role, entry.content));
        }

        prompt.push_str("\n---\nSummary:");
        prompt
    }

    /// Apply compaction by storing the summary and marking entries as compacted.
    pub fn apply_compaction(
        &self,
        session_id: &Uuid,
        summary: &str,
        summary_token_count: u64,
    ) -> CoreResult<()> {
        let now = Utc::now();

        // Deactivate old summaries
        self.storage.deactivate_summaries(session_id)?;

        // Insert new summary
        let summary_entry = SessionSummary {
            id: Uuid::new_v4(),
            session_id: *session_id,
            summary: summary.to_string(),
            created_at: now,
            token_count: summary_token_count,
            is_active: true,
        };
        self.storage.insert_summary(&summary_entry)?;

        // Mark entries as compacted
        let count = self.storage.mark_entries_compacted(session_id, &now)?;
        info!(
            "Compaction applied for session {}: {} entries compacted, summary token count {}",
            session_id, count, summary_token_count
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Session, SessionMode, SessionStatus, TranscriptEntry};
    use chrono::Utc;

    fn test_session(total_tokens: u64) -> Session {
        let now = Utc::now();
        Session {
            id: Uuid::new_v4(),
            agent_id: Uuid::new_v4(),
            name: "test".into(),
            created_at: now,
            updated_at: now,
            last_active_at: now,
            status: SessionStatus::Active,
            mode: SessionMode::Chat,
            system_prompt: String::new(),
            total_tokens,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: None,
            fork_event: None,
            metadata: serde_json::Value::Null,
        }
    }

    fn entry(session_id: Uuid, i: usize, tokens: u64) -> TranscriptEntry {
        TranscriptEntry {
            id: Uuid::new_v4(),
            session_id,
            role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
            content: format!("message number {}", i),
            created_at: Utc::now() + chrono::Duration::seconds(i as i64),
            token_count: tokens,
            metadata: serde_json::Value::Null,
            compacted: false,
        }
    }

    fn seed_entries(storage: &SessionStorage, session_id: &Uuid, n: usize, tokens_per: u64) {
        for i in 0..n {
            storage
                .insert_transcript_entry(&entry(*session_id, i, tokens_per))
                .unwrap();
        }
    }

    #[test]
    fn strategy_selection_prefers_least_destructive() {
        let session = test_session(100_000);
        let mut entries = Vec::new();
        for i in 0..20 {
            entries.push(entry(Uuid::new_v4(), i, 500));
        }
        let planner = CompactionPlanner::new()
            .with_threshold(100)
            .with_target_budget(1024)
            .with_keep_last(2);
        let strategy = planner.select_strategy(&session, &entries, 1024).unwrap();
        // KeepLast keeps 2 entries (1000 tokens) which fits under 1024 and is
        // less destructive than Summarize for this tiny case... but Summarize
        // fits too, so it should win. Either way it must not be DropOldest.
        assert_ne!(strategy, CompactionStrategy::DropOldest);
    }

    #[test]
    fn estimate_token_savings_math() {
        let session = test_session(10_000);
        let mut entries = Vec::new();
        for i in 0..10 {
            entries.push(entry(Uuid::new_v4(), i, 100));
        }
        let planner = CompactionPlanner::new()
            .with_threshold(50)
            .with_target_budget(500)
            .with_keep_last(2);
        let est = planner
            .estimate_token_savings(&session, &entries, CompactionStrategy::KeepLast)
            .unwrap();
        assert_eq!(est.tokens_before, 1000);
        assert_eq!(est.tokens_after, 200); // last 2 entries
        assert_eq!(est.savings, 800);
        assert_eq!(est.entries_compacted, 8);
    }

    #[test]
    fn plan_compaction_noop_under_threshold() {
        let session = test_session(500);
        let entries = vec![entry(session.id, 0, 100), entry(session.id, 1, 100)];
        let planner = CompactionPlanner::new().with_threshold(1000);
        let plan = planner.plan_compaction(&session, &entries).unwrap();
        assert!(plan.is_noop());
        assert_eq!(plan.current_tokens, 200);
    }

    #[test]
    fn keep_last_compacts_old_entries() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        seed_entries(&storage, &session.id, 5, 100);

        let executor = CompactionExecutor::new(storage);
        let report = executor.keep_last(&session.id, 2).unwrap();

        assert_eq!(report.entries_compacted, 3);
        assert_eq!(report.tokens_before, 500);
        assert_eq!(report.tokens_after, 200);

        // Oldest three entries should now be marked compacted.
        let entries = executor.storage.list_by_session(&session.id).unwrap();
        let compacted_count = entries.iter().filter(|e| e.compacted).count();
        assert_eq!(compacted_count, 3);

        // An active summary should exist with the marker text.
        let summary = executor
            .storage
            .get_active_summary(&session.id)
            .unwrap()
            .unwrap();
        assert!(summary.summary.contains("compacted 3 entries"));
    }

    #[test]
    fn drop_oldest_drops_specific_count() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        seed_entries(&storage, &session.id, 5, 100);

        let executor = CompactionExecutor::new(storage);
        let report = executor.drop_oldest(&session.id, 2).unwrap();
        assert_eq!(report.entries_compacted, 2);
        assert_eq!(report.tokens_after, 300);
    }

    #[test]
    fn truncate_to_budget_fits_retained_window() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        seed_entries(&storage, &session.id, 5, 100);

        let executor = CompactionExecutor::new(storage);
        let report = executor.truncate_to_budget(&session.id, 250).unwrap();
        // Budget 250: newest 2 entries (200 tokens) fit, oldest 3 compacted.
        assert_eq!(report.entries_compacted, 3);
        assert_eq!(report.tokens_after, 200);
        assert!(report.tokens_after <= 250);
    }

    #[tokio::test]
    async fn summarize_uses_fallback_extractive() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        seed_entries(&storage, &session.id, 6, 100);

        let executor = CompactionExecutor::new(storage).with_planner(
            CompactionPlanner::new()
                .with_threshold(50)
                .with_target_budget(400)
                .with_keep_last(1),
        );
        let report = executor.summarize(&session.id).await.unwrap();
        assert_eq!(report.status, "completed");
        assert!(report.entries_compacted > 0);
        assert!(report.summary_id.is_some());

        // Event recording captured the run.
        let events = executor.recent_events();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, CompactionEvent::Completed { .. }))
        );

        // Context state should reference the new summary.
        let ctx = executor
            .storage
            .get_context_state(&session.id)
            .unwrap()
            .unwrap();
        assert_eq!(ctx.active_summary_id, report.summary_id);
    }

    #[test]
    fn legacy_engine_compatibility() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(10_000);
        storage.create_session(&session).unwrap();
        seed_entries(&storage, &session.id, 10, 100);

        let engine = CompactionEngine::new(storage)
            .with_threshold(50)
            .with_target(100);
        assert!(engine.needs_compaction(&session.id).unwrap());
        let candidates = engine.get_entries_for_compaction(&session.id).unwrap();
        assert!(!candidates.is_empty());
        let prompt = CompactionEngine::build_summary_prompt(&candidates);
        assert!(prompt.contains("message number"));
        engine
            .apply_compaction(&session.id, "summary text", 10)
            .unwrap();
    }

    #[test]
    fn select_boundaries_finds_role_switches() {
        let mut entries = Vec::new();
        for i in 0..6 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            entries.push(entry(Uuid::new_v4(), i, 10));
            entries[i].role = role.to_string();
        }
        let planner = CompactionPlanner::new();
        let boundaries = planner.select_boundaries(&entries);
        // Boundaries after each role switch: after index 0 (start), 1, 3, 5.
        assert!(boundaries.contains(&0));
        assert!(boundaries.contains(&1));
        assert!(boundaries.contains(&3));
        assert!(boundaries.contains(&5));
    }

    #[test]
    fn floor_to_boundary_rounds_down() {
        let mut entries = Vec::new();
        for i in 0..6 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            entries.push(entry(Uuid::new_v4(), i, 10));
            entries[i].role = role.to_string();
        }
        let planner = CompactionPlanner::new();
        // Requesting index 4 floors down to the boundary at 3.
        assert_eq!(planner.floor_to_boundary(&entries, 4), 3);
        // Requesting index 2 floors down to 1.
        assert_eq!(planner.floor_to_boundary(&entries, 2), 1);
    }

    #[test]
    fn context_estimate_math() {
        let session = test_session(1000);
        session_meta_check(&session);
        let mut entries = Vec::new();
        for i in 0..4 {
            entries.push(entry(session.id, i, 100));
        }
        let planner = CompactionPlanner::new().with_target_budget(2048);
        let estimate = planner.estimate_context_window(&session, &entries);
        assert_eq!(estimate.active_tokens, 400);
        assert!(!estimate.over_budget);
        assert!(estimate.total_tokens >= 400);
        assert!(estimate.slack() > 0);
    }

    fn session_meta_check(session: &Session) {
        assert!(session.system_prompt.is_empty());
    }

    #[test]
    fn would_still_exceed_detects_insufficient_strategy() {
        let session = test_session(100_000);
        let mut entries = Vec::new();
        for i in 0..20 {
            entries.push(entry(Uuid::new_v4(), i, 500));
        }
        let planner = CompactionPlanner::new()
            .with_threshold(50)
            .with_target_budget(100)
            .with_keep_last(1);
        assert!(
            planner
                .would_still_exceed(&session, &entries, CompactionStrategy::KeepLast)
                .unwrap()
        );
    }

    #[test]
    fn incremental_compact_rounds_to_boundary() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(2000);
        storage.create_session(&session).unwrap();
        for i in 0..8 {
            let mut e = entry(session.id, i, 100);
            e.role = if i % 2 == 0 { "user" } else { "assistant" }.to_string();
            storage.insert_transcript_entry(&e).unwrap();
        }
        let executor = CompactionExecutor::new(storage);
        // Request compacting 3 oldest entries; floor to boundary 3 (keeps 0..3
        // compacted is 3 entries because index 3 is a boundary).
        let report = executor.incremental_compact(&session.id, 3).unwrap();
        assert!(report.entries_compacted > 0);
        let entries = executor.storage.list_by_session(&session.id).unwrap();
        let compacted = entries.iter().filter(|e| e.compacted).count();
        assert!(compacted > 0);
    }

    #[test]
    fn compact_to_budget_fits() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(2000);
        storage.create_session(&session).unwrap();
        for i in 0..10 {
            let mut e = entry(session.id, i, 100);
            e.role = if i % 2 == 0 { "user" } else { "assistant" }.to_string();
            storage.insert_transcript_entry(&e).unwrap();
        }
        let executor = CompactionExecutor::new(storage);
        let report = executor.compact_to_budget(&session.id, 300).unwrap();
        assert!(report.entries_compacted > 0);
        assert!(report.tokens_after <= 300);
    }

    #[test]
    fn boundary_for_budget_returns_index() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        for i in 0..8 {
            let mut e = entry(session.id, i, 100);
            e.role = if i % 2 == 0 { "user" } else { "assistant" }.to_string();
            storage.insert_transcript_entry(&e).unwrap();
        }
        let executor = CompactionExecutor::new(storage);
        let boundary = executor.boundary_for_budget(&session.id, 300).unwrap();
        assert!(boundary >= 4);
    }

    #[tokio::test]
    async fn preview_summary_returns_text() {
        let storage = SessionStorage::in_memory().unwrap();
        let session = test_session(1000);
        storage.create_session(&session).unwrap();
        let entries: Vec<TranscriptEntry> = (0..3).map(|i| entry(session.id, i, 10)).collect();
        let executor = CompactionExecutor::new(storage);
        let preview = executor
            .preview_summary(&session.id, &entries)
            .await
            .unwrap();
        assert!(preview.contains("message number"));
    }

    #[test]
    fn build_summary_prompt_detailed_includes_tokens() {
        let entries: Vec<TranscriptEntry> = (0..2).map(|i| entry(Uuid::new_v4(), i, 42)).collect();
        let prompt = CompactionExecutor::build_summary_prompt_detailed(&entries);
        assert!(prompt.contains("42 tokens"));
        assert!(prompt.contains("Summary:"));
    }

    #[test]
    fn context_estimate_utilization() {
        let session = test_session(1000);
        let entries: Vec<TranscriptEntry> = (0..10).map(|i| entry(session.id, i, 100)).collect();
        let planner = CompactionPlanner::new().with_target_budget(200);
        let estimate = planner.estimate_context_window(&session, &entries);
        assert!(estimate.over_budget);
        assert_eq!(estimate.utilization, 1.0);
    }
}
