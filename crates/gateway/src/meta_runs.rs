//! Meta-skill run history RPC handlers.
//!
//! Provides `rpc_meta_runs` for inspecting meta-skill run history. Run
//! records are persisted in an in-memory store keyed by skill id and run id,
//! modeling the lifecycle of a meta-skill DAG execution.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// The status of a meta-skill run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// A single step's outcome within a meta-skill run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step_id: String,
    pub step_name: String,
    pub step_type: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: RunStatus,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

/// A complete meta-skill run record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaRun {
    pub id: Uuid,
    pub skill_id: String,
    pub skill_name: String,
    pub status: RunStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub steps: Vec<StepRecord>,
    pub outputs: HashMap<String, serde_json::Value>,
    pub error: Option<String>,
}

impl MetaRun {
    /// Create a new run record for a skill.
    pub fn new(skill_id: &str, skill_name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            skill_id: skill_id.to_string(),
            skill_name: skill_name.to_string(),
            status: RunStatus::Running,
            started_at: Utc::now(),
            finished_at: None,
            steps: Vec::new(),
            outputs: HashMap::new(),
            error: None,
        }
    }
}

/// In-memory store of meta-skill runs.
#[derive(Clone, Default)]
pub struct MetaRunStore {
    runs: Arc<Mutex<Vec<MetaRun>>>,
}

impl MetaRunStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new run.
    pub fn insert(&self, run: MetaRun) {
        self.runs.lock().push(run);
    }

    /// Update a run in place.
    pub fn update<F>(&self, run_id: Uuid, f: F) -> Option<MetaRun>
    where
        F: FnOnce(&mut MetaRun),
    {
        let mut runs = self.runs.lock();
        for run in runs.iter_mut() {
            if run.id == run_id {
                f(run);
                return Some(run.clone());
            }
        }
        None
    }

    /// Get a run by id.
    pub fn get(&self, run_id: Uuid) -> Option<MetaRun> {
        self.runs.lock().iter().find(|r| r.id == run_id).cloned()
    }

    /// List runs, optionally filtered by skill id.
    pub fn list(&self, skill_id: Option<&str>, limit: usize) -> Vec<MetaRun> {
        let runs = self.runs.lock();
        let filtered: Vec<MetaRun> = runs
            .iter()
            .filter(|r| skill_id.map_or(true, |sid| r.skill_id == sid))
            .cloned()
            .collect();
        filtered.into_iter().rev().take(limit).collect()
    }

    /// Delete a run by id.
    pub fn delete(&self, run_id: Uuid) -> bool {
        let mut runs = self.runs.lock();
        let before = runs.len();
        runs.retain(|r| r.id != run_id);
        runs.len() < before
    }
}

fn parse_run_status(s: &str) -> Result<RunStatus, AppError> {
    match s.to_ascii_lowercase().as_str() {
        "running" => Ok(RunStatus::Running),
        "succeeded" => Ok(RunStatus::Succeeded),
        "failed" => Ok(RunStatus::Failed),
        "cancelled" => Ok(RunStatus::Cancelled),
        other => Err(AppError::bad_request(format!(
            "Unknown run status '{other}'"
        ))),
    }
}

/// Register meta-run RPC handlers on the given registry.
pub fn register_meta_runs_handlers(registry: &mut RpcRegistry, store: MetaRunStore) {
    let store = Arc::new(store);

    // meta_runs.start — record the start of a new meta-skill run
    registry.register(rpc_handler("meta_runs.start", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let skill_id = params
                    .get("skill_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'skill_id' parameter"))?;
                let skill_name = params
                    .get("skill_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(skill_id)
                    .to_string();
                let run = MetaRun::new(skill_id, &skill_name);
                store.insert(run.clone());
                Ok(serde_json::to_value(run).map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // meta_runs.add_step — append a step record to a run
    registry.register(rpc_handler("meta_runs.add_step", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let run_id = parse_run_id(&params)?;
                let step_id = params
                    .get("step_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'step_id' parameter"))?
                    .to_string();
                let step_name = params
                    .get("step_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&step_id)
                    .to_string();
                let step_type = params
                    .get("step_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("agent")
                    .to_string();

                let record = StepRecord {
                    step_id,
                    step_name,
                    step_type,
                    started_at: Utc::now(),
                    finished_at: None,
                    status: RunStatus::Running,
                    result: None,
                    error: None,
                };

                let updated = store.update(run_id, |run| run.steps.push(record));
                match updated {
                    Some(run) => {
                        Ok(serde_json::to_value(run)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Run {run_id} not found"))),
                }
            }
        }
    }));

    // meta_runs.complete_step — mark a step as finished
    registry.register(rpc_handler("meta_runs.complete_step", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let run_id = parse_run_id(&params)?;
                let step_id = params
                    .get("step_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'step_id' parameter"))?;
                let status = params
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(parse_run_status)
                    .unwrap_or(Ok(RunStatus::Succeeded))?;
                let result = params.get("result").cloned();
                let error = params
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                let updated = store.update(run_id, |run| {
                    for step in run.steps.iter_mut() {
                        if step.step_id == step_id {
                            step.finished_at = Some(Utc::now());
                            step.status = status;
                            step.result = result.clone();
                            step.error = error.clone();
                            break;
                        }
                    }
                });
                match updated {
                    Some(run) => {
                        Ok(serde_json::to_value(run)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Run {run_id} not found"))),
                }
            }
        }
    }));

    // meta_runs.finish — mark a run as finished
    registry.register(rpc_handler("meta_runs.finish", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let run_id = parse_run_id(&params)?;
                let status = params
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(parse_run_status)
                    .unwrap_or(Ok(RunStatus::Succeeded))?;
                let error = params
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let outputs = params
                    .get("outputs")
                    .and_then(|v| v.as_object())
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                let updated = store.update(run_id, |run| {
                    run.status = status;
                    run.finished_at = Some(Utc::now());
                    run.error = error;
                    run.outputs = outputs;
                });
                match updated {
                    Some(run) => {
                        Ok(serde_json::to_value(run)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Run {run_id} not found"))),
                }
            }
        }
    }));

    // meta_runs.get — fetch a run by id
    registry.register(rpc_handler("meta_runs.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let run_id = parse_run_id(&params)?;
                match store.get(run_id) {
                    Some(run) => {
                        Ok(serde_json::to_value(run)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    None => Err(AppError::not_found(format!("Run {run_id} not found"))),
                }
            }
        }
    }));

    // meta_runs.list — list runs, optionally filtered by skill id
    registry.register(rpc_handler("meta_runs.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let skill_id = params.get("skill_id").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
                let runs = store.list(skill_id, limit);
                Ok(serde_json::json!({
                    "runs": runs,
                    "count": runs.len(),
                }))
            }
        }
    }));

    // meta_runs.delete — delete a run record
    registry.register(rpc_handler("meta_runs.delete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let run_id = parse_run_id(&params)?;
                if store.delete(run_id) {
                    Ok(serde_json::json!({"deleted": true, "id": run_id.to_string()}))
                } else {
                    Err(AppError::not_found(format!("Run {run_id} not found")))
                }
            }
        }
    }));
}

fn parse_run_id(params: &serde_json::Value) -> Result<Uuid, AppError> {
    params
        .get("run_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'run_id' parameter"))
        .and_then(|s| Uuid::parse_str(s).map_err(|_| AppError::bad_request("Invalid run_id")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_meta_run_lifecycle() {
        let store = MetaRunStore::new();
        let mut registry = RpcRegistry::new();
        register_meta_runs_handlers(&mut registry, store);

        // Start a run
        let r = registry
            .dispatch(
                "meta_runs.start",
                serde_json::json!({"skill_id": "code-review", "skill_name": "Code Review"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        let run_id = resp["id"].as_str().unwrap().to_string();
        assert_eq!(resp["status"], "running");

        // Add a step
        let r = registry
            .dispatch(
                "meta_runs.add_step",
                serde_json::json!({
                    "run_id": run_id,
                    "step_id": "s1",
                    "step_name": "Analyze",
                    "step_type": "llm_chat",
                }),
            )
            .await;
        assert!(r.unwrap().is_ok());

        // Complete the step
        let r = registry
            .dispatch(
                "meta_runs.complete_step",
                serde_json::json!({
                    "run_id": run_id,
                    "step_id": "s1",
                    "status": "succeeded",
                    "result": {"verdict": "ok"},
                }),
            )
            .await;
        assert!(r.unwrap().is_ok());

        // Finish the run
        let r = registry
            .dispatch(
                "meta_runs.finish",
                serde_json::json!({"run_id": run_id, "status": "succeeded"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "succeeded");
    }

    #[tokio::test]
    async fn test_meta_runs_list_filtered() {
        let store = MetaRunStore::new();
        let mut registry = RpcRegistry::new();
        register_meta_runs_handlers(&mut registry, store);

        registry
            .dispatch("meta_runs.start", serde_json::json!({"skill_id": "a"}))
            .await
            .unwrap()
            .unwrap();
        registry
            .dispatch("meta_runs.start", serde_json::json!({"skill_id": "b"}))
            .await
            .unwrap()
            .unwrap();

        let r = registry
            .dispatch("meta_runs.list", serde_json::json!({"skill_id": "a"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }
}
