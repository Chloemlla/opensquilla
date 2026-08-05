//! Health check endpoint and subsystem health aggregation.
//!
//! Subsystems register themselves with a [`HealthRegistry`] by implementing
//! [`HealthCheck`]. [`HealthRegistry::build_report`] runs every registered
//! check, measures per-component latency, aggregates issues, and produces a
//! [`HealthReport`] with an overall status.
//!
//! [`health_router`] mounts the standard `/health` (JSON report) and `/healthz`
//! (plain-text liveness) endpoints on an axum router.
//!
//! See the `tests` module at the bottom of this file for end-to-end examples.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

/// Overall health status of a component or of the whole system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Everything is working.
    Healthy,
    /// Working with warnings; some subsystem is degraded.
    Degraded,
    /// A critical subsystem is failing.
    Unhealthy,
}

impl HealthStatus {
    fn name(self) -> &'static str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded",
            HealthStatus::Unhealthy => "unhealthy",
        }
    }
}

/// The health of a single subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name, e.g. "config", "provider", "session", "memory".
    pub name: String,
    /// Status of this component.
    pub status: HealthStatus,
    /// Human-readable description.
    pub description: String,
    /// Measured latency of the check in milliseconds.
    pub latency_ms: u64,
    /// Additional structured details.
    pub details: HashMap<String, String>,
}

impl ComponentHealth {
    /// Build a healthy component report.
    pub fn healthy(name: &str, description: &str) -> Self {
        Self::new(name, HealthStatus::Healthy, description)
    }

    /// Build a component report with a specific status.
    pub fn new(name: &str, status: HealthStatus, description: &str) -> Self {
        Self {
            name: name.to_string(),
            status,
            description: description.to_string(),
            latency_ms: 0,
            details: HashMap::new(),
        }
    }

    /// Attach a detail key/value pair.
    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }
}

/// Severity of a health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    /// A failing component; the system cannot operate normally.
    Critical,
    /// A degraded component; operation continues with reduced capability.
    Warning,
    /// Informational note.
    Info,
}

/// A single issue discovered during a health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// The component the issue belongs to.
    pub component: String,
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Human-readable message.
    pub message: String,
    /// Optional remediation suggestion.
    pub suggestion: Option<String>,
}

/// The aggregated result of running all registered health checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall health status.
    pub status: HealthStatus,
    /// Timestamp of the check.
    pub timestamp: DateTime<Utc>,
    /// Application uptime in seconds.
    pub uptime_seconds: u64,
    /// Per-component health results.
    pub components: Vec<ComponentHealth>,
    /// Aggregated issues found.
    pub issues: Vec<HealthIssue>,
}

/// A subsystem health check.
#[async_trait]
pub trait HealthCheck: Send + Sync + std::fmt::Debug {
    /// The name of this component, reported in the health output.
    fn name(&self) -> &'static str;
    /// Run the check and return the component health.
    async fn check(&self) -> ComponentHealth;
}

/// A registry of subsystem health checks.
///
/// Components register once at startup; [`HealthRegistry::build_report`]
/// runs all of them and aggregates the result.
#[derive(Debug, Clone)]
pub struct HealthRegistry {
    checks: Arc<RwLock<Vec<Box<dyn HealthCheck>>>>,
    start_time: DateTime<Utc>,
}

impl Default for HealthRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthRegistry {
    /// Create an empty health registry.
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(Vec::new())),
            start_time: Utc::now(),
        }
    }

    /// Register a subsystem health check.
    pub async fn register(&self, check: Box<dyn HealthCheck>) {
        let name = check.name().to_string();
        self.checks.write().await.push(check);
        info!("Registered health check: {}", name);
    }

    /// The number of registered checks.
    pub async fn len(&self) -> usize {
        self.checks.read().await.len()
    }

    /// True when no checks are registered.
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Run all registered checks and aggregate them into a [`HealthReport`].
    ///
    /// Each check runs sequentially; per-component latency is measured by the
    /// registry. Overall status is `Unhealthy` if any component is unhealthy,
    /// `Degraded` if any is degraded, otherwise `Healthy`.
    pub async fn build_report(&self) -> HealthReport {
        let checks = self.checks.read().await;
        let mut components = Vec::with_capacity(checks.len());

        for check in checks.iter() {
            let start = Instant::now();
            let mut component = check.check().await;
            component.latency_ms = start.elapsed().as_millis() as u64;
            components.push(component);
        }
        drop(checks);

        let issues = aggregate_issues(&components);
        let status = overall_status(&components);

        let uptime = (Utc::now() - self.start_time)
            .num_seconds()
            .max(0) as u64;

        HealthReport {
            status,
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            components,
            issues,
        }
    }

    /// Run all checks and return just the overall status (liveness probe).
    pub async fn overall_status(&self) -> HealthStatus {
        self.build_report().await.status
    }

    /// The application start time.
    pub fn start_time(&self) -> DateTime<Utc> {
        self.start_time
    }

    /// The application uptime.
    pub fn uptime(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }
}

/// Convert a collection of component health results into aggregated issues.
fn aggregate_issues(components: &[ComponentHealth]) -> Vec<HealthIssue> {
    let mut issues = Vec::new();
    for component in components {
        if component.status == HealthStatus::Healthy {
            continue;
        }
        let severity = match component.status {
            HealthStatus::Unhealthy => IssueSeverity::Critical,
            HealthStatus::Degraded => IssueSeverity::Warning,
            HealthStatus::Healthy => IssueSeverity::Info,
        };
        issues.push(HealthIssue {
            component: component.name.clone(),
            severity,
            message: component.description.clone(),
            suggestion: None,
        });
    }
    issues
}

/// Compute the overall status from component health results.
fn overall_status(components: &[ComponentHealth]) -> HealthStatus {
    if components.iter().any(|c| c.status == HealthStatus::Unhealthy) {
        HealthStatus::Unhealthy
    } else if components.iter().any(|c| c.status == HealthStatus::Degraded) {
        HealthStatus::Degraded
    } else {
        HealthStatus::Healthy
    }
}

/// Build an axum router serving `/health` (JSON report) and `/healthz`
/// (plain-text liveness) endpoints.
///
/// The returned router carries no state (`Router<()>`), so it can be merged
/// into a larger router with `Router::merge` or nested with `Router::nest`.
pub fn health_router(registry: Arc<HealthRegistry>) -> axum::Router {
    use axum::routing::get;

    axum::Router::new()
        .route(
            "/health",
            get({
                let registry = registry.clone();
                move || {
                    let registry = registry.clone();
                    async move {
                        let report = registry.build_report().await;
                        let status = http_status_for(report.status);
                        (status, axum::Json(report))
                    }
                }
            }),
        )
        .route(
            "/healthz",
            get({
                let registry = registry.clone();
                move || {
                    let registry = registry.clone();
                    async move {
                        let status = registry.overall_status().await;
                        let code = http_status_for(status);
                        (code, status.name().to_string())
                    }
                }
            }),
        )
}

fn http_status_for(status: HealthStatus) -> axum::http::StatusCode {
    match status {
        HealthStatus::Healthy | HealthStatus::Degraded => axum::http::StatusCode::OK,
        HealthStatus::Unhealthy => axum::http::StatusCode::SERVICE_UNAVAILABLE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct OkCheck;

    #[async_trait]
    impl HealthCheck for OkCheck {
        fn name(&self) -> &'static str {
            "ok"
        }
        async fn check(&self) -> ComponentHealth {
            ComponentHealth::healthy("ok", "fine")
        }
    }

    #[derive(Debug)]
    struct BadCheck;

    #[async_trait]
    impl HealthCheck for BadCheck {
        fn name(&self) -> &'static str {
            "bad"
        }
        async fn check(&self) -> ComponentHealth {
            ComponentHealth::new("bad", HealthStatus::Unhealthy, "broken")
        }
    }

    #[derive(Debug)]
    struct WarnCheck;

    #[async_trait]
    impl HealthCheck for WarnCheck {
        fn name(&self) -> &'static str {
            "warn"
        }
        async fn check(&self) -> ComponentHealth {
            ComponentHealth::new("warn", HealthStatus::Degraded, "slow")
        }
    }

    #[tokio::test]
    async fn healthy_registry_reports_healthy() {
        let registry = HealthRegistry::new();
        registry.register(Box::new(OkCheck)).await;
        let report = registry.build_report().await;
        assert_eq!(report.status, HealthStatus::Healthy);
        assert!(report.issues.is_empty());
        assert_eq!(report.components.len(), 1);
        assert!(report.uptime_seconds >= 0);
    }

    #[tokio::test]
    async fn unhealthy_component_drives_overall_status() {
        let registry = HealthRegistry::new();
        registry.register(Box::new(OkCheck)).await;
        registry.register(Box::new(BadCheck)).await;
        let report = registry.build_report().await;
        assert_eq!(report.status, HealthStatus::Unhealthy);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].severity, IssueSeverity::Critical);
    }

    #[tokio::test]
    async fn degraded_component_drives_overall_status() {
        let registry = HealthRegistry::new();
        registry.register(Box::new(WarnCheck)).await;
        let report = registry.build_report().await;
        assert_eq!(report.status, HealthStatus::Degraded);
        assert_eq!(report.issues[0].severity, IssueSeverity::Warning);
    }

    #[tokio::test]
    async fn empty_registry_is_healthy() {
        let registry = HealthRegistry::new();
        assert!(registry.is_empty().await);
        let report = registry.build_report().await;
        assert_eq!(report.status, HealthStatus::Healthy);
    }
}
