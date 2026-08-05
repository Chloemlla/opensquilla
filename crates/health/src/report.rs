use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Overall health status of a subsystem or the whole application.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Fully operational.
    Ok,
    /// Operational with reduced capacity or a non-fatal issue.
    Degraded,
    /// Failing; the subsystem is unavailable.
    Critical,
}

impl HealthStatus {
    /// Whether this status counts as healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Ok)
    }

    /// A stable string identifier for this status.
    pub fn as_str(&self) -> &'static str {
        match self {
            HealthStatus::Ok => "ok",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Critical => "critical",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Health of a single subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    /// Subsystem name (e.g. `config`, `database`, `providers`).
    pub name: String,
    /// The computed status.
    pub status: HealthStatus,
    /// Optional human-readable message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Optional structured detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

impl SubsystemHealth {
    /// A healthy subsystem report.
    pub fn ok(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Ok,
            message: None,
            detail: None,
        }
    }

    /// A degraded subsystem report.
    pub fn degraded(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Degraded,
            message: Some(message.into()),
            detail: None,
        }
    }

    /// A critical subsystem report.
    pub fn critical(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Critical,
            message: Some(message.into()),
            detail: None,
        }
    }

    /// Attach structured detail to this report.
    pub fn with_detail(mut self, detail: serde_json::Value) -> Self {
        self.detail = Some(detail);
        self
    }
}

/// Aggregate health report across all subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall status (Critical if any subsystem is critical, else Degraded if
    /// any is degraded, else Ok).
    pub overall: HealthStatus,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,
    /// Per-subsystem reports.
    pub subsystems: Vec<SubsystemHealth>,
}

impl HealthReport {
    /// Build a report from per-subsystem results, computing the overall status.
    pub fn new(subsystems: Vec<SubsystemHealth>) -> Self {
        let overall = if subsystems
            .iter()
            .any(|s| s.status == HealthStatus::Critical)
        {
            HealthStatus::Critical
        } else if subsystems
            .iter()
            .any(|s| s.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        };
        Self {
            overall,
            generated_at: Utc::now(),
            subsystems,
        }
    }

    /// Look up a single subsystem report by name.
    pub fn subsystem(&self, name: &str) -> Option<&SubsystemHealth> {
        self.subsystems.iter().find(|s| s.name == name)
    }

    /// The number of subsystems in a failing state.
    pub fn failing_count(&self) -> usize {
        self.subsystems
            .iter()
            .filter(|s| s.status != HealthStatus::Ok)
            .count()
    }
}
