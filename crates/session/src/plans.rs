use chrono::{DateTime, Utc};
use opensquilla_core::error::CoreError;
use opensquilla_core::result::CoreResult;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;
use uuid::Uuid;

use crate::models::{PlanRevision, PlanRun, PlanRunStatus, PlanStatus};
use crate::storage::SessionStorage;

// ---------------------------------------------------------------------------
// Step model
// ---------------------------------------------------------------------------

/// Status of an individual step within a plan revision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Proposed,
    Approved,
    Rejected,
    InProgress,
    Completed,
    Blocked,
}

/// A single actionable step in a plan. Steps are stored in the revision's
/// `metadata["steps"]` array so the `plan_revisions` row stays versionable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub text: String,
    pub status: PlanStepStatus,
    pub dependencies: Vec<String>,
    pub assignee: Option<String>,
    pub approved_at: Option<DateTime<Utc>>,
    pub rejection_reason: Option<String>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl PlanStep {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            status: PlanStepStatus::Proposed,
            dependencies: Vec::new(),
            assignee: None,
            approved_at: None,
            rejection_reason: None,
            completed_at: None,
        }
    }

    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    pub fn with_assignee(mut self, assignee: impl Into<String>) -> Self {
        self.assignee = Some(assignee.into());
        self
    }
}

/// A recorded approval (or rejection) of a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepApproval {
    pub step_id: String,
    pub plan_revision_id: Uuid,
    pub approved: bool,
    pub reason: Option<String>,
    pub approved_by: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// A plan revision together with its parsed steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSnapshot {
    pub revision: PlanRevision,
    pub steps: Vec<PlanStep>,
}

/// Execution metrics for a completed/abandoned plan run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRunReport {
    pub run: PlanRun,
    pub completed_steps: usize,
    pub total_steps: usize,
    pub duration_ms: u64,
}

/// Parse a plan outline into steps. Recognizes `- `, `* `, `+ ` bullets and
/// `1. ` / `1) ` numbered markers; continuation lines are folded into the
/// preceding step.
pub fn parse_plan(text: &str) -> Vec<PlanStep> {
    let mut steps: Vec<PlanStep> = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if is_step_line(trimmed) {
            let id = (steps.len() + 1).to_string();
            steps.push(PlanStep::new(id, strip_marker(trimmed)));
        } else if let Some(last) = steps.last_mut() {
            last.text.push(' ');
            last.text.push_str(trimmed);
        }
    }
    steps
}

fn is_step_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('-')
        || t.starts_with('*')
        || t.starts_with('+')
        || t.chars().next().map_or(false, |c| c.is_ascii_digit())
}

fn strip_marker(line: &str) -> String {
    let t = line.trim_start();
    if let Some(rest) = t
        .strip_prefix("- ")
        .or_else(|| t.strip_prefix("* "))
        .or_else(|| t.strip_prefix("+ "))
    {
        return rest.trim().to_string();
    }
    let chars: Vec<char> = t.chars().collect();
    let mut i = 0;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i > 0 && i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        let rest = rest.trim_start();
        if let Some(after) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
            return after.trim().to_string();
        }
    }
    t.to_string()
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanError {
    InvalidTransition,
    NotDraft,
    NotActive,
    AlreadyCompleted,
    AlreadyCancelled,
    StepNotFound(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::InvalidTransition => write!(f, "Invalid state transition"),
            PlanError::NotDraft => write!(f, "Plan is not in draft status"),
            PlanError::NotActive => write!(f, "Plan is not active"),
            PlanError::AlreadyCompleted => write!(f, "Plan is already completed"),
            PlanError::AlreadyCancelled => write!(f, "Plan is already cancelled"),
            PlanError::StepNotFound(id) => write!(f, "Plan step '{}' not found", id),
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PlanTransition {
    Create,
    Activate,
    Update,
    Complete,
    Cancel,
    Supersede,
}

/// Collaborative plan state machine backed by the `plan_revisions` and
/// `plan_runs` tables.
pub struct PlanStateMachine {
    storage: Arc<SessionStorage>,
}

impl PlanStateMachine {
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Construct a plan state machine sharing an existing `Arc<SessionStorage>`.
    /// This lets tool layers (which already share session storage behind an
    /// `Arc`) construct a machine without cloning the non-`Clone` storage.
    pub fn new_with_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }

    // --- Revision lifecycle ---

    /// Create a new plan revision. If the plan text parses into steps, they
    /// are recorded in `metadata["steps"]`.
    pub fn create_plan(
        &self,
        session_id: Uuid,
        plan: String,
        metadata: serde_json::Value,
    ) -> CoreResult<PlanRevision> {
        let mut metadata = metadata;
        if metadata.get("steps").is_none() {
            let steps = parse_plan(&plan);
            if !steps.is_empty() {
                let steps_val = steps_to_value(&steps);
                if metadata.is_null() {
                    metadata = serde_json::json!({ "steps": steps_val });
                } else if let Some(obj) = metadata.as_object_mut() {
                    obj.insert("steps".to_string(), steps_val);
                }
            }
        }

        let revision = PlanRevision {
            id: Uuid::new_v4(),
            session_id,
            plan,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            version: 1,
            parent_revision_id: None,
            metadata,
        };

        self.storage.insert_plan_revision(&revision)?;
        info!(
            "Created plan revision {} for session {}",
            revision.id, session_id
        );
        Ok(revision)
    }

    /// Create a plan from a free-form goal. The goal is parsed into steps and
    /// stored both as the plan text and in metadata.
    pub fn create_plan_from_goal(&self, session_id: Uuid, goal: &str) -> CoreResult<PlanRevision> {
        let steps = parse_plan(goal);
        let plan_text = if steps.is_empty() {
            goal.to_string()
        } else {
            let mut out = String::new();
            for (i, step) in steps.iter().enumerate() {
                out.push_str(&format!("{}. {}\n", i + 1, step.text));
            }
            out.trim_end().to_string()
        };
        let metadata = serde_json::json!({
            "goal": goal,
            "steps": steps_to_value(&steps),
        });
        self.create_plan(session_id, plan_text, metadata)
    }

    /// Get a plan revision by ID.
    pub fn get_revision(&self, revision_id: &Uuid) -> CoreResult<PlanRevision> {
        self.storage
            .get_plan_revision(revision_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Plan revision {}", revision_id)))
    }

    /// Get the latest plan revision for a session.
    pub fn get_latest_plan(&self, session_id: &Uuid) -> CoreResult<Option<PlanRevision>> {
        self.storage.get_latest_plan(session_id)
    }

    /// List all revisions for a session, newest version first.
    pub fn list_revisions(&self, session_id: &Uuid) -> CoreResult<Vec<PlanRevision>> {
        self.storage.list_plan_revisions(session_id, u64::MAX, 0)
    }

    /// Activate a plan (Draft -> Active).
    pub fn activate_plan(&self, revision_id: &Uuid) -> CoreResult<PlanRevision> {
        let current = self.get_revision(revision_id)?;
        if current.status != PlanStatus::Draft {
            return Err(CoreError::InvalidInput(PlanError::NotDraft.to_string()));
        }
        self.storage
            .update_plan_status(revision_id, &PlanStatus::Active)?;
        let mut updated = current;
        updated.status = PlanStatus::Active;
        info!("Activated plan revision {}", revision_id);
        Ok(updated)
    }

    /// Create a new revision superseding an old one (legacy signature).
    pub fn update_plan(
        &self,
        session_id: Uuid,
        parent_revision_id: &Uuid,
        new_plan: String,
        metadata: serde_json::Value,
    ) -> CoreResult<PlanRevision> {
        let parent = self.get_revision(parent_revision_id)?;
        if parent.status != PlanStatus::Active {
            return Err(CoreError::InvalidInput(PlanError::NotActive.to_string()));
        }

        self.storage
            .update_plan_status(parent_revision_id, &PlanStatus::Superseded)?;

        let mut metadata = metadata;
        if metadata.get("steps").is_none() {
            let steps = parse_plan(&new_plan);
            if !steps.is_empty() {
                if let Some(obj) = metadata.as_object_mut() {
                    obj.insert("steps".to_string(), steps_to_value(&steps));
                }
            }
        }

        let revision = PlanRevision {
            id: Uuid::new_v4(),
            session_id,
            plan: new_plan,
            status: PlanStatus::Active,
            created_at: Utc::now(),
            version: parent.version + 1,
            parent_revision_id: Some(parent.id),
            metadata,
        };

        self.storage.insert_plan_revision(&revision)?;
        info!(
            "Updated plan: new revision {} (v{}) superseding {}",
            revision.id, revision.version, parent_revision_id
        );
        Ok(revision)
    }

    /// Revise an active plan: supersede it and create a new revision with an
    /// incremented version and re-parsed steps.
    pub fn revise_plan(
        &self,
        plan_id: &Uuid,
        revision_text: String,
        reason: Option<&str>,
    ) -> CoreResult<PlanRevision> {
        let parent = self.get_revision(plan_id)?;
        if parent.status != PlanStatus::Active {
            return Err(CoreError::InvalidInput(PlanError::NotActive.to_string()));
        }

        self.storage
            .update_plan_status(plan_id, &PlanStatus::Superseded)?;

        let steps = parse_plan(&revision_text);
        let mut metadata = serde_json::json!({
            "steps": steps_to_value(&steps),
            "revised_from": parent.id.to_string(),
        });
        if let Some(reason) = reason {
            if let Some(obj) = metadata.as_object_mut() {
                obj.insert("revision_reason".to_string(), serde_json::json!(reason));
            }
        }

        let revision = PlanRevision {
            id: Uuid::new_v4(),
            session_id: parent.session_id,
            plan: revision_text,
            status: PlanStatus::Active,
            created_at: Utc::now(),
            version: parent.version + 1,
            parent_revision_id: Some(parent.id),
            metadata,
        };
        self.storage.insert_plan_revision(&revision)?;
        info!(
            "Revised plan: new revision {} (v{})",
            revision.id, revision.version
        );
        Ok(revision)
    }

    /// Mark a plan as completed (Active -> Completed).
    pub fn complete_plan(&self, revision_id: &Uuid) -> CoreResult<PlanRevision> {
        let current = self.get_revision(revision_id)?;
        if current.status != PlanStatus::Active {
            return Err(CoreError::InvalidInput(PlanError::NotActive.to_string()));
        }
        self.storage
            .update_plan_status(revision_id, &PlanStatus::Completed)?;
        let mut updated = current;
        updated.status = PlanStatus::Completed;
        info!("Completed plan revision {}", revision_id);
        Ok(updated)
    }

    /// Cancel a plan (any status -> Cancelled).
    pub fn cancel_plan(&self, revision_id: &Uuid) -> CoreResult<PlanRevision> {
        let current = self.get_revision(revision_id)?;
        if current.status == PlanStatus::Completed {
            return Err(CoreError::InvalidInput(
                PlanError::AlreadyCompleted.to_string(),
            ));
        }
        if current.status == PlanStatus::Cancelled {
            return Err(CoreError::InvalidInput(
                PlanError::AlreadyCancelled.to_string(),
            ));
        }
        self.storage
            .update_plan_status(revision_id, &PlanStatus::Cancelled)?;
        let mut updated = current;
        updated.status = PlanStatus::Cancelled;
        info!("Cancelled plan revision {}", revision_id);
        Ok(updated)
    }

    /// List all valid transitions for a given status.
    pub fn valid_transitions(status: &PlanStatus) -> Vec<PlanTransition> {
        match status {
            PlanStatus::Draft => vec![PlanTransition::Activate, PlanTransition::Cancel],
            PlanStatus::Active => vec![
                PlanTransition::Update,
                PlanTransition::Complete,
                PlanTransition::Cancel,
            ],
            PlanStatus::Completed => vec![],
            PlanStatus::Cancelled => vec![],
            PlanStatus::Superseded => vec![],
        }
    }

    /// Check if a transition is valid.
    pub fn is_valid_transition(from: &PlanStatus, transition: &PlanTransition) -> bool {
        Self::valid_transitions(from)
            .iter()
            .any(|t| t == transition)
    }

    // --- Steps ---

    /// Load a revision plus its parsed steps.
    pub fn snapshot(&self, revision_id: &Uuid) -> CoreResult<PlanSnapshot> {
        let revision = self.get_revision(revision_id)?;
        let steps = steps_from_value(
            revision
                .metadata
                .get("steps")
                .unwrap_or(&serde_json::Value::Null),
        );
        Ok(PlanSnapshot { revision, steps })
    }

    /// List the steps of a plan revision.
    pub fn list_steps(&self, revision_id: &Uuid) -> CoreResult<Vec<PlanStep>> {
        Ok(self.snapshot(revision_id)?.steps)
    }

    /// Get a single step by ID.
    pub fn get_step(&self, revision_id: &Uuid, step_id: &str) -> CoreResult<PlanStep> {
        self.snapshot(revision_id)?
            .steps
            .into_iter()
            .find(|s| s.id == step_id)
            .ok_or_else(|| {
                CoreError::InvalidInput(PlanError::StepNotFound(step_id.to_string()).to_string())
            })
    }

    /// Approve a step (Proposed -> Approved).
    pub fn approve_step(
        &self,
        revision_id: &Uuid,
        step_id: &str,
        approved_by: Option<&str>,
    ) -> CoreResult<StepApproval> {
        let mut snapshot = self.snapshot(revision_id)?;
        self.require_active(&snapshot.revision)?;

        {
            let step = find_mut_step(&mut snapshot.steps, step_id)?;
            if step.status == PlanStepStatus::Rejected {
                return Err(CoreError::InvalidInput(
                    "Cannot approve a rejected step; revise the plan first".into(),
                ));
            }
            if step.status == PlanStepStatus::Completed {
                return Err(CoreError::InvalidInput("Step is already completed".into()));
            }
            step.status = PlanStepStatus::Approved;
            step.approved_at = Some(Utc::now());
            step.rejection_reason = None;
        }

        let approval = StepApproval {
            step_id: step_id.to_string(),
            plan_revision_id: *revision_id,
            approved: true,
            reason: None,
            approved_by: approved_by.map(|s| s.to_string()),
            created_at: Utc::now(),
        };
        append_approval(&mut snapshot, &approval);
        self.save_snapshot(&snapshot)?;
        info!("Approved step {} of plan {}", step_id, revision_id);
        Ok(approval)
    }

    /// Reject a step (Proposed/Approved -> Rejected).
    pub fn reject_step(
        &self,
        revision_id: &Uuid,
        step_id: &str,
        reason: &str,
    ) -> CoreResult<StepApproval> {
        let mut snapshot = self.snapshot(revision_id)?;
        self.require_active(&snapshot.revision)?;

        {
            let step = find_mut_step(&mut snapshot.steps, step_id)?;
            if step.status == PlanStepStatus::Completed {
                return Err(CoreError::InvalidInput(
                    "Cannot reject a completed step".into(),
                ));
            }
            step.status = PlanStepStatus::Rejected;
            step.approved_at = None;
            step.rejection_reason = Some(reason.to_string());
        }

        let approval = StepApproval {
            step_id: step_id.to_string(),
            plan_revision_id: *revision_id,
            approved: false,
            reason: Some(reason.to_string()),
            approved_by: None,
            created_at: Utc::now(),
        };
        append_approval(&mut snapshot, &approval);
        self.save_snapshot(&snapshot)?;
        info!("Rejected step {} of plan {}", step_id, revision_id);
        Ok(approval)
    }

    /// Mark a step in-progress (Approved -> InProgress).
    pub fn start_step(&self, revision_id: &Uuid, step_id: &str) -> CoreResult<PlanStep> {
        let mut snapshot = self.snapshot(revision_id)?;
        self.require_active(&snapshot.revision)?;

        {
            let step = find_mut_step(&mut snapshot.steps, step_id)?;
            if step.status != PlanStepStatus::Approved {
                return Err(CoreError::InvalidInput(
                    "Only approved steps can be started".into(),
                ));
            }
            step.status = PlanStepStatus::InProgress;
        }
        self.save_snapshot(&snapshot)?;
        Ok(snapshot
            .steps
            .into_iter()
            .find(|s| s.id == step_id)
            .ok_or_else(|| {
                CoreError::InvalidInput(PlanError::StepNotFound(step_id.to_string()).to_string())
            })?)
    }

    /// Mark a step completed (InProgress/Approved -> Completed).
    pub fn complete_step(&self, revision_id: &Uuid, step_id: &str) -> CoreResult<PlanStep> {
        let mut snapshot = self.snapshot(revision_id)?;
        self.require_active(&snapshot.revision)?;

        {
            let step = find_mut_step(&mut snapshot.steps, step_id)?;
            if step.status == PlanStepStatus::Rejected {
                return Err(CoreError::InvalidInput(
                    "Cannot complete a rejected step".into(),
                ));
            }
            if step.status == PlanStepStatus::Completed {
                return Err(CoreError::InvalidInput("Step is already completed".into()));
            }
            step.status = PlanStepStatus::Completed;
            step.completed_at = Some(Utc::now());
        }
        self.save_snapshot(&snapshot)?;
        info!("Completed step {} of plan {}", step_id, revision_id);
        Ok(snapshot
            .steps
            .into_iter()
            .find(|s| s.id == step_id)
            .ok_or_else(|| {
                CoreError::InvalidInput(PlanError::StepNotFound(step_id.to_string()).to_string())
            })?)
    }

    /// History of approvals recorded against a revision.
    pub fn approval_log(&self, revision_id: &Uuid) -> CoreResult<Vec<StepApproval>> {
        let revision = self.get_revision(revision_id)?;
        Ok(revision
            .metadata
            .get("approvals")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|v| serde_json::from_value::<StepApproval>(v.clone()).ok())
            .collect())
    }

    // --- Runs ---

    /// Create a run against an active plan revision.
    pub fn create_run(&self, revision_id: &Uuid) -> CoreResult<PlanRun> {
        let revision = self.get_revision(revision_id)?;
        self.require_active(&revision)?;

        let run = PlanRun {
            id: Uuid::new_v4(),
            plan_revision_id: revision.id,
            session_id: revision.session_id,
            status: PlanRunStatus::Pending,
            started_at: Utc::now(),
            completed_at: None,
            agent_task_id: None,
            result: serde_json::Value::Null,
        };
        self.storage.insert_plan_run(&run)?;
        info!("Created plan run {} for revision {}", run.id, revision_id);
        Ok(run)
    }

    pub fn get_run(&self, run_id: &Uuid) -> CoreResult<Option<PlanRun>> {
        self.storage.get_plan_run(run_id)
    }

    pub fn require_run(&self, run_id: &Uuid) -> CoreResult<PlanRun> {
        self.storage
            .get_plan_run(run_id)?
            .ok_or_else(|| CoreError::NotFound(format!("Plan run {}", run_id)))
    }

    /// Start a run (Pending -> Running).
    pub fn start_run(&self, run_id: &Uuid) -> CoreResult<PlanRun> {
        let mut run = self.require_run(run_id)?;
        if run.status != PlanRunStatus::Pending {
            return Err(CoreError::InvalidInput("Run is not pending".into()));
        }
        let now = Utc::now();
        self.storage
            .update_plan_run_status(run_id, &PlanRunStatus::Running, Some(&now))?;
        run.status = PlanRunStatus::Running;
        run.started_at = now;
        info!("Started plan run {}", run_id);
        Ok(run)
    }

    /// Complete a run successfully (Running -> Succeeded).
    pub fn complete_run(&self, run_id: &Uuid, result: serde_json::Value) -> CoreResult<PlanRun> {
        let mut run = self.require_run(run_id)?;
        self.require_running(&run)?;
        let now = Utc::now();
        self.storage
            .update_plan_run_status(run_id, &PlanRunStatus::Succeeded, Some(&now))?;
        self.storage.update_plan_run_result(run_id, &result)?;
        run.status = PlanRunStatus::Succeeded;
        run.completed_at = Some(now);
        run.result = result;
        info!("Completed plan run {}", run_id);
        Ok(run)
    }

    /// Fail a run (Running -> Failed).
    pub fn fail_run(&self, run_id: &Uuid, error: &str) -> CoreResult<PlanRun> {
        let mut run = self.require_run(run_id)?;
        self.require_running(&run)?;
        let now = Utc::now();
        self.storage
            .update_plan_run_status(run_id, &PlanRunStatus::Failed, Some(&now))?;
        self.storage
            .update_plan_run_result(run_id, &serde_json::json!({ "error": error }))?;
        run.status = PlanRunStatus::Failed;
        run.completed_at = Some(now);
        run.result = serde_json::json!({ "error": error });
        info!("Failed plan run {}", run_id);
        Ok(run)
    }

    /// Cancel a run (Pending/Running -> Cancelled).
    pub fn cancel_run(&self, run_id: &Uuid) -> CoreResult<PlanRun> {
        let mut run = self.require_run(run_id)?;
        if matches!(run.status, PlanRunStatus::Succeeded | PlanRunStatus::Failed) {
            return Err(CoreError::InvalidInput("Run is already finished".into()));
        }
        let now = Utc::now();
        self.storage
            .update_plan_run_status(run_id, &PlanRunStatus::Cancelled, Some(&now))?;
        run.status = PlanRunStatus::Cancelled;
        run.completed_at = Some(now);
        info!("Cancelled plan run {}", run_id);
        Ok(run)
    }

    /// List runs for a plan revision, newest first.
    pub fn list_runs(&self, revision_id: &Uuid) -> CoreResult<Vec<PlanRun>> {
        self.storage.list_plan_runs(revision_id, u64::MAX, 0)
    }

    /// List runs for a session, newest first.
    pub fn list_runs_by_session(&self, session_id: &Uuid) -> CoreResult<Vec<PlanRun>> {
        self.storage.list_plan_runs_by_session(session_id)
    }

    /// Execution report for a run.
    pub fn run_report(&self, run_id: &Uuid) -> CoreResult<PlanRunReport> {
        let run = self.require_run(run_id)?;
        let snapshot = self.snapshot(&run.plan_revision_id)?;
        let completed_steps = snapshot
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Completed)
            .count();
        let duration_ms = run
            .completed_at
            .map(|c| (c - run.started_at).num_milliseconds().max(0))
            .unwrap_or(0) as u64;
        Ok(PlanRunReport {
            run,
            completed_steps,
            total_steps: snapshot.steps.len(),
            duration_ms,
        })
    }

    // --- Internal helpers ---

    fn require_active(&self, revision: &PlanRevision) -> CoreResult<()> {
        if revision.status != PlanStatus::Active {
            return Err(CoreError::InvalidInput(PlanError::NotActive.to_string()));
        }
        Ok(())
    }

    fn require_running(&self, run: &PlanRun) -> CoreResult<()> {
        if !matches!(run.status, PlanRunStatus::Running | PlanRunStatus::Pending) {
            return Err(CoreError::InvalidInput("Run is not in progress".into()));
        }
        Ok(())
    }

    fn save_snapshot(&self, snapshot: &PlanSnapshot) -> CoreResult<()> {
        let mut revision = snapshot.revision.clone();
        let mut meta = revision.metadata.as_object().cloned().unwrap_or_default();
        meta.insert("steps".to_string(), steps_to_value(&snapshot.steps));
        revision.metadata = serde_json::Value::Object(meta);
        self.storage.update_plan_revision(&revision)
    }

    /// Public wrapper around the internal snapshot persistence, used by
    /// callers that construct or mutate a [`PlanSnapshot`] directly.
    pub fn save_snapshot_public(&self, snapshot: &PlanSnapshot) -> CoreResult<()> {
        self.save_snapshot(snapshot)
    }

    // -----------------------------------------------------------------------
    // Progress & analytics
    // -----------------------------------------------------------------------

    /// Compute the progress of a plan revision as a percentage `[0, 100]`
    /// based on the ratio of completed to total steps.
    pub fn plan_progress(&self, revision_id: &Uuid) -> CoreResult<f64> {
        let snapshot = self.snapshot(revision_id)?;
        if snapshot.steps.is_empty() {
            return Ok(0.0);
        }
        let completed = snapshot
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Completed)
            .count();
        Ok(completed as f64 * 100.0 / snapshot.steps.len() as f64)
    }

    /// Count steps by status for a revision.
    pub fn step_status_counts(
        &self,
        revision_id: &Uuid,
    ) -> CoreResult<std::collections::HashMap<PlanStepStatus, usize>> {
        let snapshot = self.snapshot(revision_id)?;
        let mut counts = std::collections::HashMap::new();
        for step in &snapshot.steps {
            *counts.entry(step.status.clone()).or_insert(0) += 1;
        }
        Ok(counts)
    }

    /// The next runnable step ids: steps that are `Approved` with all their
    /// dependencies completed. Ordered by id.
    pub fn next_runnable_steps(&self, revision_id: &Uuid) -> CoreResult<Vec<String>> {
        let snapshot = self.snapshot(revision_id)?;
        let completed: std::collections::HashSet<&str> = snapshot
            .steps
            .iter()
            .filter(|s| s.status == PlanStepStatus::Completed)
            .map(|s| s.id.as_str())
            .collect();

        let mut runnable: Vec<PlanStep> = snapshot
            .steps
            .iter()
            .filter(|s| {
                s.status == PlanStepStatus::Approved
                    && s.dependencies.iter().all(|d| completed.contains(d.as_str()))
            })
            .cloned()
            .collect();
        runnable.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(runnable.into_iter().map(|s| s.id).collect())
    }

    /// Reorder the steps of a revision. The provided `step_ids` must be a
    /// permutation of the existing step ids. Reordering updates `metadata`
    /// only; it does not create a new revision.
    pub fn reorder_steps(&self, revision_id: &Uuid, step_ids: &[String]) -> CoreResult<()> {
        let mut snapshot = self.snapshot(revision_id)?;
        self.require_active(&snapshot.revision)?;

        let existing: std::collections::HashSet<String> =
            snapshot.steps.iter().map(|s| s.id.clone()).collect();
        let requested: std::collections::HashSet<&String> = step_ids.iter().collect();
        if requested.len() != existing.len() || requested.iter().any(|id| !existing.contains(*id)) {
            return Err(CoreError::InvalidInput(
                "step_ids must be a permutation of the existing step ids".into(),
            ));
        }

        let by_id: std::collections::HashMap<String, PlanStep> = snapshot
            .steps
            .drain(..)
            .map(|s| (s.id.clone(), s))
            .collect();
        snapshot.steps = step_ids
            .iter()
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        self.save_snapshot(&snapshot)
    }

    /// Validate that a step's dependencies are acyclic and refer to existing
    /// steps. Returns the set of dependency cycles found (empty when valid).
    pub fn validate_step_dependencies(
        &self,
        revision_id: &Uuid,
    ) -> CoreResult<Vec<Vec<String>>> {
        let snapshot = self.snapshot(revision_id)?;
        let graph: std::collections::HashMap<String, Vec<String>> = snapshot
            .steps
            .iter()
            .map(|s| (s.id.clone(), s.dependencies.clone()))
            .collect();

        // Count in-degrees only for dependencies that actually exist.
        let mut indegree: std::collections::HashMap<String, usize> = graph
            .iter()
            .map(|(id, deps)| {
                (
                    id.clone(),
                    deps.iter().filter(|d| graph.contains_key(*d)).count(),
                )
            })
            .collect();

        let mut queue: Vec<String> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut visited = 0usize;
        while let Some(id) = queue.pop() {
            visited += 1;
            let deps_copy: Vec<String> = graph.get(&id).cloned().unwrap_or_default();
            for dep in deps_copy {
                if let Some(count) = indegree.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        queue.push(dep.clone());
                    }
                }
            }
        }

        let mut cycles = Vec::new();
        if visited < graph.len() {
            let remaining: Vec<String> = indegree
                .iter()
                .filter(|(_, d)| **d > 0)
                .map(|(id, _)| id.clone())
                .collect();
            if !remaining.is_empty() {
                cycles.push(remaining);
            }
        }
        Ok(cycles)
    }

    // -----------------------------------------------------------------------
    // Plan export / import
    // -----------------------------------------------------------------------

    /// Export a plan revision (with steps, runs, and approvals) to JSON.
    pub fn export_plan(&self, revision_id: &Uuid) -> CoreResult<String> {
        let snapshot = self.snapshot(revision_id)?;
        let runs = self.list_runs(revision_id)?;
        let approvals = self.approval_log(revision_id)?;
        let export = PlanExport {
            revision: snapshot.revision,
            steps: snapshot.steps,
            runs,
            approvals,
            exported_at: Utc::now(),
        };
        serde_json::to_string_pretty(&export)
            .map_err(|e| CoreError::Internal(format!("plan export serialization failed: {}", e)))
    }

    /// Import a plan from an exported JSON string into a session.
    ///
    /// Creates a new revision in the given session with the exported plan
    /// text and steps. The original revision id is recorded in metadata.
    pub fn import_plan(&self, session_id: Uuid, json: &str) -> CoreResult<PlanRevision> {
        let export: PlanExport = serde_json::from_str(json)
            .map_err(|e| CoreError::InvalidInput(format!("invalid plan export JSON: {}", e)))?;
        let mut metadata = serde_json::json!({
            "steps": steps_to_value(&export.steps),
            "imported_from": export.revision.id.to_string(),
            "imported_at": Utc::now().to_rfc3339(),
        });
        if let Some(obj) = metadata.as_object_mut() {
            for (k, v) in export
                .revision
                .metadata
                .as_object()
                .unwrap_or(&serde_json::Map::new())
            {
                if k != "steps" {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }
        self.create_plan(session_id, export.revision.plan.clone(), metadata)
    }

    /// Import a plan and immediately activate it.
    pub fn import_plan_active(&self, session_id: Uuid, json: &str) -> CoreResult<PlanRevision> {
        let revision = self.import_plan(session_id, json)?;
        self.activate_plan(&revision.id)
    }

    /// Whether all steps of a revision are completed.
    pub fn is_complete(&self, revision_id: &Uuid) -> CoreResult<bool> {
        let snapshot = self.snapshot(revision_id)?;
        Ok(!snapshot.steps.is_empty()
            && snapshot
                .steps
                .iter()
                .all(|s| s.status == PlanStepStatus::Completed))
    }

    /// A human-readable one-line status string for a plan.
    pub fn status_line(&self, revision_id: &Uuid) -> CoreResult<String> {
        let snapshot = self.snapshot(revision_id)?;
        let progress = self.plan_progress(revision_id)?;
        let counts = self.step_status_counts(revision_id)?;
        Ok(format!(
            "plan '{}' status={:?} progress={:.0}% steps={} completed={}",
            snapshot.revision.plan.chars().take(40).collect::<String>(),
            snapshot.revision.status,
            progress,
            snapshot.steps.len(),
            counts.get(&PlanStepStatus::Completed).copied().unwrap_or(0)
        ))
    }
}

/// The complete serializable state of a plan (revision + steps + runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanExport {
    pub revision: PlanRevision,
    pub steps: Vec<PlanStep>,
    pub runs: Vec<PlanRun>,
    pub approvals: Vec<StepApproval>,
    pub exported_at: DateTime<Utc>,
}

fn find_mut_step<'a>(steps: &'a mut [PlanStep], step_id: &str) -> CoreResult<&'a mut PlanStep> {
    steps.iter_mut().find(|s| s.id == step_id).ok_or_else(|| {
        CoreError::InvalidInput(PlanError::StepNotFound(step_id.to_string()).to_string())
    })
}

/// Append an approval record to a snapshot's metadata before it is saved.
fn append_approval(snapshot: &mut PlanSnapshot, approval: &StepApproval) {
    let mut meta = snapshot
        .revision
        .metadata
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut approvals = meta
        .get("approvals")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    approvals.push(serde_json::to_value(approval).unwrap_or(serde_json::Value::Null));
    meta.insert("approvals".to_string(), serde_json::Value::Array(approvals));
    snapshot.revision.metadata = serde_json::Value::Object(meta);
}

fn steps_to_value(steps: &[PlanStep]) -> serde_json::Value {
    serde_json::to_value(steps).unwrap_or_else(|_| serde_json::Value::Array(Vec::new()))
}

fn steps_from_value(value: &serde_json::Value) -> Vec<PlanStep> {
    serde_json::from_value(value.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Session, SessionMode, SessionStatus};

    fn machine() -> PlanStateMachine {
        PlanStateMachine::new(SessionStorage::in_memory().unwrap())
    }

    fn seeded_session(m: &PlanStateMachine) -> Uuid {
        let id = Uuid::new_v4();
        let session = Session {
            id,
            agent_id: Uuid::new_v4(),
            name: "plan test".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            last_active_at: Utc::now(),
            status: SessionStatus::Active,
            mode: SessionMode::Chat,
            system_prompt: String::new(),
            total_tokens: 0,
            total_cost_usd: 0.0,
            message_count: 0,
            parent_session_id: None,
            fork_event: None,
            metadata: serde_json::Value::Null,
        };
        m.storage.create_session(&session).unwrap();
        id
    }

    #[test]
    fn parse_plan_recognizes_bullets_and_numbers() {
        let steps = parse_plan("- research the problem\n- draft a proposal\n- get sign-off");
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].text, "research the problem");
        assert_eq!(steps[2].text, "get sign-off");

        let numbered = parse_plan("1. setup\n2. implement\n3. test");
        assert_eq!(numbered.len(), 3);
        assert_eq!(numbered[1].text, "implement");
    }

    #[test]
    fn parse_plan_folds_continuation_lines() {
        let steps = parse_plan("- first step\n  with more detail\n- second step");
        assert_eq!(steps.len(), 2);
        assert!(steps[0].text.contains("more detail"));
    }

    #[test]
    fn create_activate_complete_lifecycle() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two")
            .unwrap();
        assert_eq!(plan.status, PlanStatus::Draft);
        assert_eq!(plan.version, 1);
        assert_eq!(m.list_steps(&plan.id).unwrap().len(), 2);

        let active = m.activate_plan(&plan.id).unwrap();
        assert_eq!(active.status, PlanStatus::Active);

        // Activating a non-draft plan fails.
        assert!(m.activate_plan(&plan.id).is_err());

        let completed = m.complete_plan(&plan.id).unwrap();
        assert_eq!(completed.status, PlanStatus::Completed);

        // Completing again fails.
        assert!(m.complete_plan(&plan.id).is_err());
    }

    #[test]
    fn revise_plan_supersedes_and_bumps_version() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- initial approach")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();

        let revised = m
            .revise_plan(
                &plan.id,
                "- revised approach\n- extra step".to_string(),
                Some("scope change"),
            )
            .unwrap();
        assert_eq!(revised.version, 2);
        assert_eq!(revised.parent_revision_id, Some(plan.id));
        assert_eq!(m.list_steps(&revised.id).unwrap().len(), 2);

        // The old revision is now superseded.
        let old = m.get_revision(&plan.id).unwrap();
        assert_eq!(old.status, PlanStatus::Superseded);
    }

    #[test]
    fn step_approve_reject_complete_flow() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();

        let approval = m.approve_step(&plan.id, "1", Some("lead")).unwrap();
        assert!(approval.approved);
        assert_eq!(approval.approved_by.as_deref(), Some("lead"));

        let step = m.get_step(&plan.id, "1").unwrap();
        assert_eq!(step.status, PlanStepStatus::Approved);
        assert!(step.approved_at.is_some());

        m.reject_step(&plan.id, "2", "out of scope").unwrap();
        let rejected = m.get_step(&plan.id, "2").unwrap();
        assert_eq!(rejected.status, PlanStepStatus::Rejected);
        assert_eq!(rejected.rejection_reason.as_deref(), Some("out of scope"));

        // Approving a rejected step fails.
        assert!(m.approve_step(&plan.id, "2", None).is_err());

        // Completing a rejected step fails.
        assert!(m.complete_step(&plan.id, "2").is_err());

        // A missing step is a StepNotFound error.
        let err = m.approve_step(&plan.id, "nope", None).unwrap_err();
        assert!(matches!(err, CoreError::InvalidInput(_)));

        assert_eq!(m.approval_log(&plan.id).unwrap().len(), 2);
    }

    #[test]
    fn step_ops_require_active_plan() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m.create_plan_from_goal(session_id, "- step").unwrap();
        // Draft plan: approvals are not allowed.
        assert!(m.approve_step(&plan.id, "1", None).is_err());
    }

    #[test]
    fn run_lifecycle_and_report() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();
        m.approve_step(&plan.id, "1", None).unwrap();
        m.complete_step(&plan.id, "1").unwrap();

        let run = m.create_run(&plan.id).unwrap();
        assert_eq!(run.status, PlanRunStatus::Pending);

        let running = m.start_run(&run.id).unwrap();
        assert_eq!(running.status, PlanRunStatus::Running);

        let completed = m
            .complete_run(&run.id, serde_json::json!({ "ok": true }))
            .unwrap();
        assert_eq!(completed.status, PlanRunStatus::Succeeded);
        assert!(completed.completed_at.is_some());
        assert_eq!(completed.result["ok"], serde_json::json!(true));

        let report = m.run_report(&run.id).unwrap();
        assert_eq!(report.completed_steps, 1);
        assert_eq!(report.total_steps, 2);

        // Finishing an already-finished run fails.
        assert!(m.complete_run(&run.id, serde_json::Value::Null).is_err());
    }

    #[test]
    fn run_fail_and_cancel() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m.create_plan_from_goal(session_id, "- step").unwrap();
        m.activate_plan(&plan.id).unwrap();

        let run = m.create_run(&plan.id).unwrap();
        m.start_run(&run.id).unwrap();
        let failed = m.fail_run(&run.id, "boom").unwrap();
        assert_eq!(failed.status, PlanRunStatus::Failed);
        assert_eq!(failed.result["error"], serde_json::json!("boom"));

        let run2 = m.create_run(&plan.id).unwrap();
        m.start_run(&run2.id).unwrap();
        let cancelled = m.cancel_run(&run2.id).unwrap();
        assert_eq!(cancelled.status, PlanRunStatus::Cancelled);
    }

    #[test]
    fn list_revisions_orders_newest_first() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m.create_plan_from_goal(session_id, "- step").unwrap();
        m.activate_plan(&plan.id).unwrap();
        m.revise_plan(&plan.id, "- revised".to_string(), None).unwrap();

        let revisions = m.list_revisions(&session_id).unwrap();
        assert_eq!(revisions.len(), 2);
        assert_eq!(revisions[0].version, 2);
        assert_eq!(revisions[1].version, 1);
    }

    #[test]
    fn plan_progress_tracks_completion() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two\n- step three")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();

        assert_eq!(m.plan_progress(&plan.id).unwrap(), 0.0);
        m.approve_step(&plan.id, "1", None).unwrap();
        m.complete_step(&plan.id, "1").unwrap();
        assert_eq!(m.plan_progress(&plan.id).unwrap(), 100.0 / 3.0);
    }

    #[test]
    fn step_status_counts_groups() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();
        m.approve_step(&plan.id, "1", None).unwrap();
        m.complete_step(&plan.id, "1").unwrap();
        m.reject_step(&plan.id, "2", "nope").unwrap();

        let counts = m.step_status_counts(&plan.id).unwrap();
        assert_eq!(counts.get(&PlanStepStatus::Completed).copied().unwrap_or(0), 1);
        assert_eq!(counts.get(&PlanStepStatus::Rejected).copied().unwrap_or(0), 1);
    }

    #[test]
    fn next_runnable_steps_requires_deps() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m.create_plan(session_id, "- first\n- second\n- third".to_string(), serde_json::Value::Null).unwrap();
        m.activate_plan(&plan.id).unwrap();

        // Approve all steps.
        m.approve_step(&plan.id, "1", None).unwrap();
        m.approve_step(&plan.id, "2", None).unwrap();
        m.approve_step(&plan.id, "3", None).unwrap();

        // All three runnable since no dependencies set.
        let runnable = m.next_runnable_steps(&plan.id).unwrap();
        assert_eq!(runnable.len(), 3);

        m.complete_step(&plan.id, "1").unwrap();
        let runnable2 = m.next_runnable_steps(&plan.id).unwrap();
        assert_eq!(runnable2.len(), 2);
    }

    #[test]
    fn reorder_steps_requires_permutation() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- a\n- b\n- c")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();

        m.reorder_steps(&plan.id, &["3".to_string(), "1".to_string(), "2".to_string()])
            .unwrap();
        let steps = m.list_steps(&plan.id).unwrap();
        assert_eq!(steps[0].id, "3");

        // A non-permutation is rejected.
        assert!(m.reorder_steps(&plan.id, &["1".to_string(), "2".to_string()]).is_err());
        assert!(m.reorder_steps(&plan.id, &["1".to_string(), "2".to_string(), "ghost".to_string()]).is_err());
    }

    #[test]
    fn validate_step_dependencies_detects_cycle() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m.create_plan(session_id, "- a\n- b".to_string(), serde_json::Value::Null).unwrap();
        m.activate_plan(&plan.id).unwrap();

        // Build a cycle via metadata manipulation.
        let mut snapshot = m.snapshot(&plan.id).unwrap();
        snapshot.steps[0].dependencies = vec!["2".to_string()];
        snapshot.steps[1].dependencies = vec!["1".to_string()];
        m.save_snapshot_public(&snapshot).unwrap();

        let cycles = m.validate_step_dependencies(&plan.id).unwrap();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn export_import_plan_roundtrip() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one\n- step two")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();
        m.approve_step(&plan.id, "1", Some("lead")).unwrap();

        let json = m.export_plan(&plan.id).unwrap();

        let imported = m.import_plan(seeded_session(&m), &json).unwrap();
        assert_eq!(imported.plan, plan.plan);
        assert_eq!(m.list_steps(&imported.id).unwrap().len(), 2);

        let active = m.import_plan_active(seeded_session(&m), &json).unwrap();
        assert_eq!(active.status, PlanStatus::Active);
    }

    #[test]
    fn is_complete_all_steps_done() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- only step")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();
        assert!(!m.is_complete(&plan.id).unwrap());
        m.approve_step(&plan.id, "1", None).unwrap();
        m.complete_step(&plan.id, "1").unwrap();
        assert!(m.is_complete(&plan.id).unwrap());
    }

    #[test]
    fn status_line_is_readable() {
        let m = machine();
        let session_id = seeded_session(&m);
        let plan = m
            .create_plan_from_goal(session_id, "- step one")
            .unwrap();
        m.activate_plan(&plan.id).unwrap();
        let line = m.status_line(&plan.id).unwrap();
        assert!(line.contains("progress="));
        assert!(line.contains("steps=1"));
    }
}
