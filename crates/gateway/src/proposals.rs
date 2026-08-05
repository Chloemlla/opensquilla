//! Proposals RPC handlers.
//!
//! Provides `rpc_proposals` for meta-skill proposals: creating, listing,
//! accepting, and rejecting proposals produced by meta-skill runs.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// The status of a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Accepted,
    Rejected,
    Expired,
}

/// A meta-skill proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: Uuid,
    pub skill_id: String,
    pub run_id: Option<String>,
    pub title: String,
    pub description: String,
    pub proposed_action: serde_json::Value,
    pub status: ProposalStatus,
    pub created_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

/// In-memory proposal store.
#[derive(Clone, Default)]
pub struct ProposalStore {
    proposals: Arc<Mutex<Vec<Proposal>>>,
}

impl ProposalStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a proposal.
    pub fn insert(&self, proposal: Proposal) {
        self.proposals.lock().push(proposal);
    }

    /// Update a proposal in place.
    pub fn update<F>(&self, id: Uuid, f: F) -> Option<Proposal>
    where
        F: FnOnce(&mut Proposal),
    {
        let mut proposals = self.proposals.lock();
        for p in proposals.iter_mut() {
            if p.id == id {
                f(p);
                return Some(p.clone());
            }
        }
        None
    }

    /// Get a proposal by id.
    pub fn get(&self, id: Uuid) -> Option<Proposal> {
        self.proposals.lock().iter().find(|p| p.id == id).cloned()
    }

    /// List proposals, optionally filtered by skill id or status.
    pub fn list(
        &self,
        skill_id: Option<&str>,
        status: Option<ProposalStatus>,
        limit: usize,
    ) -> Vec<Proposal> {
        let proposals = self.proposals.lock();
        let filtered: Vec<Proposal> = proposals
            .iter()
            .filter(|p| skill_id.map_or(true, |sid| p.skill_id == sid))
            .filter(|p| status.map_or(true, |s| p.status == s))
            .cloned()
            .collect();
        filtered.into_iter().rev().take(limit).collect()
    }

    /// Delete a proposal.
    pub fn delete(&self, id: Uuid) -> bool {
        let mut proposals = self.proposals.lock();
        let before = proposals.len();
        proposals.retain(|p| p.id != id);
        proposals.len() < before
    }
}

fn parse_status(s: &str) -> Result<ProposalStatus, AppError> {
    match s.to_ascii_lowercase().as_str() {
        "pending" => Ok(ProposalStatus::Pending),
        "accepted" => Ok(ProposalStatus::Accepted),
        "rejected" => Ok(ProposalStatus::Rejected),
        "expired" => Ok(ProposalStatus::Expired),
        other => Err(AppError::bad_request(format!(
            "Unknown proposal status '{other}'"
        ))),
    }
}

/// Register proposals RPC handlers on the given registry.
pub fn register_proposals_handlers(registry: &mut RpcRegistry, store: ProposalStore) {
    let store = Arc::new(store);

    // proposals.create — create a new proposal
    registry.register(rpc_handler("proposals.create", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let skill_id = params
                    .get("skill_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'skill_id' parameter"))?;
                let title = params
                    .get("title")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'title' parameter"))?;
                let description = params
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let run_id = params
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let proposed_action = params
                    .get("proposed_action")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let metadata: HashMap<String, String> = params
                    .get("metadata")
                    .and_then(|v| v.as_object())
                    .map(|m| {
                        m.iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                            .collect()
                    })
                    .unwrap_or_default();

                let proposal = Proposal {
                    id: Uuid::new_v4(),
                    skill_id: skill_id.to_string(),
                    run_id,
                    title: title.to_string(),
                    description,
                    proposed_action,
                    status: ProposalStatus::Pending,
                    created_at: Utc::now(),
                    decided_at: None,
                    metadata,
                };
                store.insert(proposal.clone());
                Ok(
                    serde_json::to_value(proposal)
                        .map_err(|e| AppError::internal(e.to_string()))?,
                )
            }
        }
    }));

    // proposals.get — fetch a proposal by id
    registry.register(rpc_handler("proposals.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_proposal_id(&params)?;
                match store.get(id) {
                    Some(proposal) => Ok(serde_json::to_value(proposal)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Proposal {id} not found"))),
                }
            }
        }
    }));

    // proposals.list — list proposals with optional filters
    registry.register(rpc_handler("proposals.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let skill_id = params.get("skill_id").and_then(|v| v.as_str());
                let status = params
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(parse_status)
                    .transpose()?;
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
                let proposals = store.list(skill_id, status, limit);
                Ok(serde_json::json!({
                    "proposals": proposals,
                    "count": proposals.len(),
                }))
            }
        }
    }));

    // proposals.accept — accept a pending proposal
    registry.register(rpc_handler("proposals.accept", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_proposal_id(&params)?;
                let updated = store.update(id, |p| {
                    p.status = ProposalStatus::Accepted;
                    p.decided_at = Some(Utc::now());
                });
                match updated {
                    Some(proposal) => Ok(serde_json::to_value(proposal)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Proposal {id} not found"))),
                }
            }
        }
    }));

    // proposals.reject — reject a pending proposal
    registry.register(rpc_handler("proposals.reject", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_proposal_id(&params)?;
                let updated = store.update(id, |p| {
                    p.status = ProposalStatus::Rejected;
                    p.decided_at = Some(Utc::now());
                });
                match updated {
                    Some(proposal) => Ok(serde_json::to_value(proposal)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Proposal {id} not found"))),
                }
            }
        }
    }));

    // proposals.delete — delete a proposal
    registry.register(rpc_handler("proposals.delete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let id = parse_proposal_id(&params)?;
                if store.delete(id) {
                    Ok(serde_json::json!({"deleted": true, "id": id.to_string()}))
                } else {
                    Err(AppError::not_found(format!("Proposal {id} not found")))
                }
            }
        }
    }));
}

fn parse_proposal_id(params: &serde_json::Value) -> Result<Uuid, AppError> {
    params
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))
        .and_then(|s| Uuid::parse_str(s).map_err(|_| AppError::bad_request("Invalid proposal id")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proposals_lifecycle() {
        let store = ProposalStore::new();
        let mut registry = RpcRegistry::new();
        register_proposals_handlers(&mut registry, store);

        let params = serde_json::json!({
            "skill_id": "code-review",
            "title": "Refactor module X",
            "description": "Proposed refactoring",
            "proposed_action": {"action": "refactor", "target": "module_x"},
        });
        let r = registry.dispatch("proposals.create", params).await;
        let resp = r.unwrap().unwrap();
        let id = resp["id"].as_str().unwrap().to_string();
        assert_eq!(resp["status"], "pending");

        let r = registry
            .dispatch("proposals.accept", serde_json::json!({"id": id}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "accepted");

        let r = registry
            .dispatch("proposals.list", serde_json::json!({"status": "accepted"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_proposals_reject() {
        let store = ProposalStore::new();
        let mut registry = RpcRegistry::new();
        register_proposals_handlers(&mut registry, store);

        let params = serde_json::json!({
            "skill_id": "s",
            "title": "t",
        });
        let resp = registry
            .dispatch("proposals.create", params)
            .await
            .unwrap()
            .unwrap();
        let id = resp["id"].as_str().unwrap().to_string();

        let r = registry
            .dispatch("proposals.reject", serde_json::json!({"id": id}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["status"], "rejected");
    }
}
