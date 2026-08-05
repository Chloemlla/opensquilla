use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use opensquilla_core::config::Config;

/// An entry in the security audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique audit entry ID.
    pub id: String,
    /// Timestamp of the event.
    pub timestamp: DateTime<Utc>,
    /// The action that was performed.
    pub action: String,
    /// The actor who performed the action (user, system, etc.).
    pub actor: String,
    /// The resource affected by the action.
    pub resource: String,
    /// Whether the action was allowed or denied.
    pub result: AuditResult,
    /// Additional details about the event.
    pub details: HashMap<String, String>,
}

/// Result of an audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditResult {
    Allowed,
    Denied,
    Error,
}

/// Security audit log for recording security-relevant events.
#[derive(Debug, Clone)]
pub struct AuditLog {
    entries: Arc<RwLock<Vec<AuditEntry>>>,
    retention_count: usize,
}

impl AuditLog {
    /// Create a new audit log.
    pub fn new(config: &Config) -> Self {
        let retention = config
            .get("audit.retention_count")
            .unwrap_or_else(|| "10000".to_string())
            .parse::<usize>()
            .unwrap_or(10000);

        info!("Audit log initialized with retention of {retention} entries");

        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            retention_count: retention,
        }
    }

    /// Record an audit entry.
    pub async fn record(&self, entry: AuditEntry) {
        let mut entries = self.entries.write().await;
        entries.push(entry);

        // Trim to retention limit
        if entries.len() > self.retention_count {
            let excess = entries.len() - self.retention_count;
            entries.drain(0..excess);
        }
    }

    /// Record a security-relevant action.
    pub async fn record_action(
        &self,
        action: &str,
        actor: &str,
        resource: &str,
        result: AuditResult,
        details: HashMap<String, String>,
    ) {
        let entry = AuditEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action: action.to_string(),
            actor: actor.to_string(),
            resource: resource.to_string(),
            result,
            details,
        };

        let action_desc = format!("{action} on {resource} by {actor}");
        match result {
            AuditResult::Allowed => {
                debug!("Audit: {action_desc} allowed");
            }
            AuditResult::Denied => {
                warn!("Audit: {action_desc} denied");
            }
            AuditResult::Error => {
                warn!("Audit: {action_desc} error");
            }
        }

        self.record(entry).await;
    }

    /// Record a permission grant.
    pub async fn record_permission_grant(
        &self,
        actor: &str,
        permission: &str,
        resource: &str,
    ) {
        self.record_action(
            "permission.grant",
            actor,
            resource,
            AuditResult::Allowed,
            HashMap::from([("permission".to_string(), permission.to_string())]),
        )
        .await;
    }

    /// Record a permission denial.
    pub async fn record_permission_denial(
        &self,
        actor: &str,
        permission: &str,
        resource: &str,
    ) {
        self.record_action(
            "permission.deny",
            actor,
            resource,
            AuditResult::Denied,
            HashMap::from([("permission".to_string(), permission.to_string())]),
        )
        .await;
    }

    /// Record a configuration change.
    pub async fn record_config_change(&self, actor: &str, key: &str, old_value: &str, new_value: &str) {
        self.record_action(
            "config.change",
            actor,
            key,
            AuditResult::Allowed,
            HashMap::from([
                ("old_value".to_string(), old_value.to_string()),
                ("new_value".to_string(), new_value.to_string()),
            ]),
        )
        .await;
    }

    /// Record a user authentication event.
    pub async fn record_auth_event(
        &self,
        username: &str,
        success: bool,
        method: &str,
    ) {
        let result = if success {
            AuditResult::Allowed
        } else {
            AuditResult::Denied
        };

        self.record_action(
            "auth",
            username,
            "authentication",
            result,
            HashMap::from([("method".to_string(), method.to_string())]),
        )
        .await;
    }

    /// Record an error during audit.
    pub async fn record_error(&self, actor: &str, action: &str, resource: &str, error: &str) {
        self.record_action(
            action,
            actor,
            resource,
            AuditResult::Error,
            HashMap::from([("error".to_string(), error.to_string())]),
        )
        .await;
    }

    /// Get all audit entries.
    pub async fn get_entries(&self) -> Vec<AuditEntry> {
        self.entries.read().await.clone()
    }

    /// Get entries for a specific actor.
    pub async fn get_entries_by_actor(&self, actor: &str) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.actor == actor)
            .cloned()
            .collect()
    }

    /// Get entries for a specific action.
    pub async fn get_entries_by_action(&self, action: &str) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.action == action)
            .cloned()
            .collect()
    }

    /// Get denied entries.
    pub async fn get_denied_entries(&self) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.result == AuditResult::Denied)
            .cloned()
            .collect()
    }

    /// Get entries within a time range.
    pub async fn get_entries_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Clear all audit entries.
    pub async fn clear(&self) {
        let mut entries = self.entries.write().await;
        entries.clear();
        info!("Audit log cleared");
    }

    /// Export audit log as JSON.
    pub async fn export_json(&self) -> String {
        let entries = self.entries.read().await;
        serde_json::to_string_pretty(&*entries).unwrap_or_default()
    }

    /// Get the number of entries.
    pub async fn count(&self) -> usize {
        let entries = self.entries.read().await;
        entries.len()
    }
}