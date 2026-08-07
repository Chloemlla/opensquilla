//! Plan-control tools: submit_plan, request_user_input, plan_run_checkpoint.
//!
//! These tools interact with the session plan state machine
//! ([`opensquilla_session::PlanStateMachine`]) to submit plans, request user
//! clarification, and checkpoint plan-run progress.
//!
//! The Python `plan_control.py` reads a runtime `ToolContext` for plan-mode
//! gating, interaction mode, and the active plan revision. The Rust tools
//! crate has no global contextvar, so each tool takes the relevant ids
//! (`session_id`, `plan_run_id`) as explicit parameters. The plan state
//! machine's public API is the single source of truth for revision/run
//! lifecycle.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_session::PlanStateMachine;
use opensquilla_session::SessionStorage;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Map a session-crate error to a tool error.
fn map_plan_error(op: &str, err: impl std::fmt::Display) -> ToolError {
    ToolError::new("PLAN_ERROR", format!("Plan {} failed: {}", op, err))
}

/// Maximum character limits mirroring the Python `session.plans` constants.
const MAX_PLAN_TITLE_CHARS: usize = 200;
const MAX_PLAN_MARKDOWN_CHARS: usize = 20_000;
const MAX_PLAN_STEP_TITLE_CHARS: usize = 200;
const MAX_PLAN_STEPS: usize = 50;
const MAX_PLAN_STEP_ID_CHARS: usize = 64;
const MAX_PLAN_STEP_REASON_CHARS: usize = 1_000;

const MAX_QUESTION_COUNT: usize = 3;
const MAX_QUESTION_ID_CHARS: usize = 80;
const MAX_QUESTION_HEADER_CHARS: usize = 80;
const MAX_QUESTION_TEXT_CHARS: usize = 1_000;
const MIN_OPTION_COUNT: usize = 2;
const MAX_OPTION_COUNT: usize = 3;
const MAX_OPTION_LABEL_CHARS: usize = 120;
const MAX_OPTION_DESCRIPTION_CHARS: usize = 500;

/// Clean and validate a text field, returning the trimmed string.
fn clean_text(value: &str, field: &str, max_chars: usize) -> Result<String, ToolError> {
    let text = value.trim();
    if text.is_empty() {
        return Err(ToolError::invalid_args(format!("{} is required", field)));
    }
    if text.chars().count() > max_chars {
        return Err(ToolError::invalid_args(format!(
            "{} must be at most {} characters",
            field, max_chars
        )));
    }
    Ok(text.to_string())
}

// ===========================================================================
// submit_plan
// ===========================================================================

/// Tool for submitting a complete structured plan, creating a new revision.
pub struct SubmitPlanTool {
    storage: Arc<SessionStorage>,
}

impl SubmitPlanTool {
    /// Create the tool from a shared session storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for SubmitPlanTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "submit_plan",
                concat!(
                    "Submit the complete structured plan for the current session. ",
                    "Creates a new immutable plan revision and returns its id.",
                ),
                HashMap::from([
                    (
                        "session_id".to_string(),
                        ParameterDefinition::required_string("The session UUID the plan belongs to"),
                    ),
                    (
                        "title".to_string(),
                        ParameterDefinition::required_string("Short plan title"),
                    ),
                    (
                        "markdown".to_string(),
                        ParameterDefinition::required_string(
                            "Complete human-readable plan (markdown)",
                        ),
                    ),
                    (
                        "steps".to_string(),
                        ParameterDefinition::array(
                            "Ordered implementation steps for the complete plan",
                            ParameterDefinition::string("step title"),
                        ),
                    ),
                ]),
            )
            .category("plan")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let session_raw = params["session_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'session_id'"))?;
        let session_id = Uuid::parse_str(session_raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'session_id' UUID '{}': {}", session_raw, e))
        })?;

        let title = clean_text(
            params["title"].as_str().unwrap_or(""),
            "title",
            MAX_PLAN_TITLE_CHARS,
        )?;
        let markdown = clean_text(
            params["markdown"].as_str().unwrap_or(""),
            "markdown",
            MAX_PLAN_MARKDOWN_CHARS,
        )?;

        // Normalize steps: accept an array of strings or objects with a title.
        let steps_input = params["steps"].as_array().cloned().unwrap_or_default();
        if steps_input.is_empty() {
            return Err(ToolError::invalid_args("steps must contain at least one item"));
        }
        if steps_input.len() > MAX_PLAN_STEPS {
            return Err(ToolError::invalid_args(format!(
                "steps must contain at most {} items",
                MAX_PLAN_STEPS
            )));
        }

        let mut steps_text = String::new();
        for (i, s) in steps_input.iter().enumerate() {
            let step_title = if let Some(t) = s.as_str() {
                clean_text(t, &format!("steps[{}]", i), MAX_PLAN_STEP_TITLE_CHARS)?
            } else if let Some(obj) = s.as_object() {
                let t = obj
                    .get("title")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ToolError::invalid_args(format!("steps[{}].title is required", i))
                    })?;
                clean_text(t, &format!("steps[{}].title", i), MAX_PLAN_STEP_TITLE_CHARS)?
            } else {
                return Err(ToolError::invalid_args(format!(
                    "steps[{}] must be a string or object",
                    i
                )));
            };
            steps_text.push_str(&format!("{}. {}\n", i + 1, step_title));
        }
        let plan_text = format!("# {}\n\n{}\n\n## Steps\n\n{}", title, markdown, steps_text);

        let storage = self.storage.clone();
        let title_for_closure = title.clone();
        let revision = tokio::task::spawn_blocking(move || -> Result<_, ToolError> {
            let machine = PlanStateMachine::new_with_arc(storage);
            let metadata = serde_json::json!({ "title": title_for_closure });
            machine
                .create_plan(session_id, plan_text, metadata)
                .map_err(|e| map_plan_error("create", e))
        })
        .await
        .map_err(|e| ToolError::new("PLAN_ERROR", format!("Plan create task failed: {}", e)))??;

        let data = serde_json::json!({
            "status": "plan_submitted",
            "title": title,
            "session_id": session_id.to_string(),
            "revision_id": revision.id.to_string(),
            "version": revision.version,
            "step_count": steps_input.len(),
        });
        Ok(ToolOutput::success_with_data(
            format!("Plan '{}' submitted (revision {})", title, revision.id),
            data,
        ))
    }
}

// ===========================================================================
// request_user_input
// ===========================================================================

/// Tool for asking one to three concise questions when a missing user decision
/// materially changes the plan.
pub struct RequestUserInputTool;

impl RequestUserInputTool {
    /// Create a new request_user_input tool.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RequestUserInputTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for RequestUserInputTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "request_user_input",
                concat!(
                    "Ask one to three concise questions when a missing user decision ",
                    "materially changes the plan. Returns a structured clarification request.",
                ),
                HashMap::from([(
                    "questions".to_string(),
                    ParameterDefinition::array(
                        "One to three question objects (id, question, optional header, optional options)",
                        ParameterDefinition::string("question object"),
                    ),
                )]),
            )
            .category("plan")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let questions = params["questions"]
            .as_array()
            .cloned()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'questions'"))?;
        if !(1..=MAX_QUESTION_COUNT).contains(&questions.len()) {
            return Err(ToolError::invalid_args(format!(
                "questions must contain between 1 and {} items",
                MAX_QUESTION_COUNT
            )));
        }

        let mut normalized: Vec<Value> = Vec::new();
        let mut fields: Vec<Value> = Vec::new();
        let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (index, raw) in questions.iter().enumerate() {
            let obj = raw.as_object().ok_or_else(|| {
                ToolError::invalid_args(format!("questions[{}] must be an object", index))
            })?;

            let question_id = clean_text(
                obj.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                &format!("questions[{}].id", index),
                MAX_QUESTION_ID_CHARS,
            )?;
            if seen_ids.contains(&question_id) {
                return Err(ToolError::invalid_args("question ids must be unique"));
            }
            seen_ids.insert(question_id.clone());

            let question_text = clean_text(
                obj.get("question").and_then(|v| v.as_str()).unwrap_or(""),
                &format!("questions[{}].question", index),
                MAX_QUESTION_TEXT_CHARS,
            )?;

            let header = obj
                .get("header")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if !header.is_empty() && header.chars().count() > MAX_QUESTION_HEADER_CHARS {
                return Err(ToolError::invalid_args(format!(
                    "questions[{}].header must be at most {} characters",
                    index, MAX_QUESTION_HEADER_CHARS
                )));
            }

            let raw_options = obj.get("options").and_then(|v| v.as_array());
            let mut choices: Vec<String> = Vec::new();
            let mut normalized_options: Vec<Value> = Vec::new();
            if let Some(opts) = raw_options {
                if !(MIN_OPTION_COUNT..=MAX_OPTION_COUNT).contains(&opts.len()) {
                    return Err(ToolError::invalid_args(format!(
                        "questions[{}].options must contain {} or {} items",
                        index, MIN_OPTION_COUNT, MAX_OPTION_COUNT
                    )));
                }
                let mut seen_labels: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for (oi, opt) in opts.iter().enumerate() {
                    let opt_obj = opt.as_object().ok_or_else(|| {
                        ToolError::invalid_args(format!(
                            "questions[{}].options[{}] must be an object",
                            index, oi
                        ))
                    })?;
                    let label = clean_text(
                        opt_obj.get("label").and_then(|v| v.as_str()).unwrap_or(""),
                        &format!("questions[{}].options[{}].label", index, oi),
                        MAX_OPTION_LABEL_CHARS,
                    )?;
                    if seen_labels.contains(&label) {
                        return Err(ToolError::invalid_args(format!(
                            "questions[{}].option labels must be unique",
                            index
                        )));
                    }
                    seen_labels.insert(label.clone());

                    let description = opt_obj
                        .get("description")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if description.chars().count() > MAX_OPTION_DESCRIPTION_CHARS {
                        return Err(ToolError::invalid_args(format!(
                            "questions[{}].options[{}].description must be at most {} characters",
                            index, oi, MAX_OPTION_DESCRIPTION_CHARS
                        )));
                    }
                    let mut norm_opt = serde_json::Map::new();
                    norm_opt.insert("label".into(), Value::String(label.clone()));
                    if !description.is_empty() {
                        norm_opt.insert("description".into(), Value::String(description));
                    }
                    normalized_options.push(Value::Object(norm_opt));
                    choices.push(label);
                }
            }

            let mut norm_q = serde_json::Map::new();
            norm_q.insert("id".into(), Value::String(question_id.clone()));
            norm_q.insert("question".into(), Value::String(question_text.clone()));
            if !header.is_empty() {
                norm_q.insert("header".into(), Value::String(header.clone()));
            }
            if !normalized_options.is_empty() {
                norm_q.insert("options".into(), Value::Array(normalized_options.clone()));
            }
            normalized.push(Value::Object(norm_q));

            let mut field = serde_json::Map::new();
            field.insert("name".into(), Value::String(question_id));
            field.insert("prompt".into(), Value::String(question_text));
            field.insert(
                "type".into(),
                Value::String(if choices.is_empty() {
                    "string".to_string()
                } else {
                    "enum".to_string()
                }),
            );
            field.insert("required".into(), Value::Bool(true));
            field.insert(
                "choices".into(),
                Value::Array(choices.iter().map(|c| Value::String(c.clone())).collect()),
            );
            if !header.is_empty() {
                field.insert("header".into(), Value::String(header));
            }
            if !normalized_options.is_empty() {
                field.insert("options".into(), Value::Array(normalized_options.clone()));
                field.insert("allow_other".into(), Value::Bool(true));
            }
            fields.push(Value::Object(field));
        }

        let data = serde_json::json!({
            "status": "input_required",
            "kind": "user_input",
            "paused": true,
            "step": "plan",
            "clarify_schema": {
                "mode": "form",
                "presentation": "plan_questionnaire_v1",
                "intro": "The plan needs a decision before it can be completed.",
                "fields": fields,
            },
            "questions": normalized,
        });
        Ok(ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
            .with_data(data))
    }
}

// ===========================================================================
// plan_run_checkpoint
// ===========================================================================

/// Tool for persisting progress for a PlanRun attached to an implementation
/// turn.
pub struct PlanRunCheckpointTool {
    storage: Arc<SessionStorage>,
}

impl PlanRunCheckpointTool {
    /// Create the tool from a shared session storage handle.
    pub fn new(storage: SessionStorage) -> Self {
        Self {
            storage: Arc::new(storage),
        }
    }

    /// Create the tool from an existing `Arc<SessionStorage>`.
    pub fn from_arc(storage: Arc<SessionStorage>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl Tool for PlanRunCheckpointTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "plan_run_checkpoint",
                concat!(
                    "Persist progress for the PlanRun attached to this implementation turn. ",
                    "Checkpoint the current step immediately after it reaches the stated result. ",
                    "A blocked checkpoint ends the turn.",
                ),
                HashMap::from([
                    (
                        "plan_run_id".to_string(),
                        ParameterDefinition::required_string("The UUID of the active plan run"),
                    ),
                    (
                        "step_id".to_string(),
                        ParameterDefinition::required_string("The plan step whose state changed"),
                    ),
                    (
                        "step_status".to_string(),
                        ParameterDefinition::string("New step status: completed, blocked, or skipped")
                            .enum_values(vec![
                                "completed".into(),
                                "blocked".into(),
                                "skipped".into(),
                            ]),
                    ),
                    (
                        "reason".to_string(),
                        ParameterDefinition::string(
                            "Required explanation when blocked or skipped",
                        ),
                    ),
                ]),
            )
            .category("plan")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let run_raw = params["plan_run_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'plan_run_id'"))?;
        let run_id = Uuid::parse_str(run_raw).map_err(|e| {
            ToolError::invalid_args(format!("Invalid 'plan_run_id' UUID '{}': {}", run_raw, e))
        })?;

        let step_id = clean_text(
            params["step_id"].as_str().unwrap_or(""),
            "step_id",
            MAX_PLAN_STEP_ID_CHARS,
        )?;

        let status_raw = params["step_status"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let reason = params["reason"].as_str().unwrap_or("").trim().to_string();

        match status_raw.as_str() {
            "completed" | "blocked" | "skipped" => {}
            _ => {
                return Err(ToolError::invalid_args(
                    "step_status must be completed, blocked, or skipped",
                ));
            }
        }

        if (status_raw == "blocked" || status_raw == "skipped") && reason.is_empty() {
            return Err(ToolError::invalid_args(format!(
                "reason is required when step_status is {}",
                status_raw
            )));
        }
        if !reason.is_empty() && reason.chars().count() > MAX_PLAN_STEP_REASON_CHARS {
            return Err(ToolError::invalid_args(format!(
                "reason must be at most {} characters",
                MAX_PLAN_STEP_REASON_CHARS
            )));
        }

        let storage = self.storage.clone();
        let status_owned = status_raw.clone();
        let result = tokio::task::spawn_blocking(move || -> Result<Value, ToolError> {
            let machine = PlanStateMachine::new_with_arc(storage);
            let run = machine
                .get_run(&run_id)
                .map_err(|e| map_plan_error("get_run", e))?
                .ok_or_else(|| {
                    ToolError::new(
                        "PLAN_RUN_NOT_FOUND",
                        format!("The active PlanRun '{}' no longer exists", run_id),
                    )
                })?;

            // Map the checkpoint status onto the plan step lifecycle.
            // `completed` -> complete_step; `blocked`/`skipped` -> reject_step
            // with the reason, then fail the run if blocked.
            let revision_id = run.plan_revision_id;
            match status_owned.as_str() {
                "completed" => {
                    machine
                        .complete_step(&revision_id, &step_id)
                        .map_err(|e| map_plan_error("complete_step", e))?;
                }
                "blocked" | "skipped" => {
                    machine
                        .reject_step(&revision_id, &step_id, &reason)
                        .map_err(|e| map_plan_error("reject_step", e))?;
                    if status_owned == "blocked" {
                        machine
                            .fail_run(&run_id, &reason)
                            .map_err(|e| map_plan_error("fail_run", e))?;
                    }
                }
                _ => unreachable!(),
            }

            let report = machine
                .run_report(&run_id)
                .map_err(|e| map_plan_error("run_report", e))?;
            Ok(serde_json::json!({
                "status": "checkpoint_recorded",
                "plan_run": {
                    "run_id": run.id.to_string(),
                    "status": run.status,
                    "completed_steps": report.completed_steps,
                    "total_steps": report.total_steps,
                    "duration_ms": report.duration_ms,
                },
            }))
        })
        .await
        .map_err(|e| ToolError::new("PLAN_ERROR", format!("Checkpoint task failed: {}", e)))??;

        Ok(ToolOutput::success(serde_json::to_string_pretty(&result).unwrap_or_default())
            .with_data(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_session::models::{Session, SessionMode, SessionStatus};

    fn test_storage() -> Arc<SessionStorage> {
        Arc::new(SessionStorage::in_memory().expect("in-memory session storage"))
    }

    fn seeded_session(storage: &SessionStorage) -> Uuid {
        let id = Uuid::new_v4();
        let session = Session {
            id,
            agent_id: Uuid::new_v4(),
            name: "plan test".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            last_active_at: chrono::Utc::now(),
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
        storage.create_session(&session).expect("create session");
        id
    }

    #[tokio::test]
    async fn test_submit_plan_creates_revision() {
        let storage = test_storage();
        let session_id = seeded_session(&storage);
        let tool = SubmitPlanTool::from_arc(storage);

        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "title": "My Plan",
                "markdown": "Do the work carefully.",
                "steps": ["first step", "second step"],
            }))
            .await;
        assert!(result.is_ok(), "submit failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("plan_submitted"));
        assert_eq!(data["step_count"], 2);
        assert!(!data["revision_id"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_submit_plan_rejects_empty_steps() {
        let storage = test_storage();
        let session_id = seeded_session(&storage);
        let tool = SubmitPlanTool::from_arc(storage);

        let result = tool
            .execute(serde_json::json!({
                "session_id": session_id.to_string(),
                "title": "My Plan",
                "markdown": "Do the work.",
                "steps": [],
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_request_user_input_validates_questions() {
        let tool = RequestUserInputTool::new();

        // Empty questions array rejected.
        let result = tool.execute(serde_json::json!({ "questions": [] })).await;
        assert!(result.is_err());

        // Valid single question succeeds.
        let result = tool
            .execute(serde_json::json!({
                "questions": [{
                    "id": "q1",
                    "question": "Which approach?",
                    "options": [
                        { "label": "A" },
                        { "label": "B" },
                    ],
                }],
            }))
            .await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("input_required"));
        assert_eq!(data["clarify_schema"]["fields"][0]["type"], "enum");
    }

    #[tokio::test]
    async fn test_request_user_input_rejects_duplicate_ids() {
        let tool = RequestUserInputTool::new();
        let result = tool
            .execute(serde_json::json!({
                "questions": [
                    { "id": "dup", "question": "one" },
                    { "id": "dup", "question": "two" },
                ],
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_plan_run_checkpoint_requires_reason_when_blocked() {
        let storage = test_storage();
        let tool = PlanRunCheckpointTool::from_arc(storage);

        let result = tool
            .execute(serde_json::json!({
                "plan_run_id": Uuid::new_v4().to_string(),
                "step_id": "1",
                "step_status": "blocked",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_plan_run_checkpoint_run_not_found() {
        let storage = test_storage();
        let tool = PlanRunCheckpointTool::from_arc(storage);

        let result = tool
            .execute(serde_json::json!({
                "plan_run_id": Uuid::new_v4().to_string(),
                "step_id": "1",
                "step_status": "completed",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "PLAN_RUN_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_plan_run_checkpoint_completed_flow() {
        let storage = test_storage();
        let session_id = seeded_session(&storage);
        let machine = PlanStateMachine::new_with_arc(storage.clone());
        // Create + activate a plan with one step.
        let revision = machine
            .create_plan_from_goal(session_id, "- step one")
            .expect("create plan");
        machine.activate_plan(&revision.id).expect("activate");
        machine
            .approve_step(&revision.id, "1", None)
            .expect("approve");
        let run = machine.create_run(&revision.id).expect("create run");
        machine.start_run(&run.id).expect("start run");

        let tool = PlanRunCheckpointTool::from_arc(storage);
        let result = tool
            .execute(serde_json::json!({
                "plan_run_id": run.id.to_string(),
                "step_id": "1",
                "step_status": "completed",
            }))
            .await;
        assert!(result.is_ok(), "checkpoint failed: {:?}", result.err());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["status"], serde_json::json!("checkpoint_recorded"));
        assert_eq!(data["plan_run"]["completed_steps"], 1);
    }
}
