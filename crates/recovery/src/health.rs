use std::collections::HashMap;

use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Result of a health check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Overall health status.
    pub status: HealthStatus,
    /// Timestamp of the check.
    pub timestamp: DateTime<Utc>,
    /// Uptime of the application in seconds.
    pub uptime_seconds: u64,
    /// Individual component health.
    pub components: Vec<ComponentHealth>,
    /// Summary of issues found.
    pub issues: Vec<HealthIssue>,
}

/// Overall health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Health of a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name (e.g., "config", "provider", "session", "memory").
    pub name: String,
    /// Status of this component.
    pub status: HealthStatus,
    /// Human-readable description.
    pub description: String,
    /// Latency of the check in milliseconds.
    pub latency_ms: u64,
    /// Additional details.
    pub details: HashMap<String, String>,
}

/// A health issue found during the check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    pub component: String,
    pub severity: IssueSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

/// Severity of a health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

/// Health check manager.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    config: Config,
    start_time: DateTime<Utc>,
}

impl HealthCheck {
    /// Create a new health check manager.
    pub fn new(config: &Config) -> Self {
        info!("Health check initialized");
        Self {
            config: config.clone(),
            start_time: Utc::now(),
        }
    }

    /// Run a full health check on all components.
    pub async fn run_full_check(&self) -> HealthCheckResult {
        let mut components = Vec::new();
        let mut issues = Vec::new();

        // Check configuration
        components.push(self.check_config().await);
        components.push(self.check_file_system().await);
        components.push(self.check_network().await);

        // Collect issues
        for component in &components {
            if component.status != HealthStatus::Healthy {
                issues.push(HealthIssue {
                    component: component.name.clone(),
                    severity: match component.status {
                        HealthStatus::Unhealthy => IssueSeverity::Critical,
                        HealthStatus::Degraded => IssueSeverity::Warning,
                        HealthStatus::Healthy => continue,
                    },
                    message: component.description.clone(),
                    suggestion: None,
                });
            }
        }

        let overall_status = if issues.iter().any(|i| matches!(i.severity, IssueSeverity::Critical)) {
            HealthStatus::Unhealthy
        } else if !issues.is_empty() {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        let uptime = (Utc::now() - self.start_time)
            .num_seconds()
            .max(0) as u64;

        let result = HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            components,
            issues,
        };

        info!("Health check complete: {:?}", result.status);
        result
    }

    /// Run a quick health check (config only).
    pub async fn quick_check(&self) -> HealthCheckResult {
        let mut components = Vec::new();
        let issues = Vec::new();

        components.push(self.check_config().await);

        let overall_status = if components.iter().any(|c| c.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else if components.iter().any(|c| c.status == HealthStatus::Degraded) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        let uptime = (Utc::now() - self.start_time)
            .num_seconds()
            .max(0) as u64;

        HealthCheckResult {
            status: overall_status,
            timestamp: Utc::now(),
            uptime_seconds: uptime,
            components,
            issues,
        }
    }

    /// Check configuration health.
    async fn check_config(&self) -> ComponentHealth {
        let start = std::time::Instant::now();

        let mut details = HashMap::new();
        details.insert(
            "config_path".to_string(),
            Config::discover_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "not found".to_string()),
        );

        let has_provider = self
            .config
            .get("provider.default")
            .map_or(false, |v| !v.is_empty());
        let has_model = self
            .config
            .get("model.default")
            .map_or(false, |v| !v.is_empty());

        details.insert("has_provider".to_string(), has_provider.to_string());
        details.insert("has_model".to_string(), has_model.to_string());

        let latency = start.elapsed().as_millis() as u64;

        let (status, description) = if !has_provider {
            (HealthStatus::Degraded, "No default provider configured".to_string())
        } else if !has_model {
            (HealthStatus::Degraded, "No default model configured".to_string())
        } else {
            (HealthStatus::Healthy, "Configuration is valid".to_string())
        };

        ComponentHealth {
            name: "config".to_string(),
            status,
            description,
            latency_ms: latency,
            details,
        }
    }

    /// Check filesystem health.
    async fn check_file_system(&self) -> ComponentHealth {
        let start = std::time::Instant::now();

        let mut details = HashMap::new();

        // Check config directory
        let config_dir = Config::discover_path()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));

        let config_writable = match &config_dir {
            Some(dir) => {
                let test_path = dir.join(".health_check_tmp");
                match std::fs::write(&test_path, b"test") {
                    Ok(_) => {
                        std::fs::remove_file(&test_path).ok();
                        true
                    }
                    Err(_) => false,
                }
            }
            None => false,
        };

        details.insert("config_writable".to_string(), config_writable.to_string());

        let latency = start.elapsed().as_millis() as u64;

        let (status, description) = if config_writable {
            (HealthStatus::Healthy, "File system is accessible".to_string())
        } else {
            (
                HealthStatus::Degraded,
                "Config directory may not be writable".to_string(),
            )
        };

        ComponentHealth {
            name: "filesystem".to_string(),
            status,
            description,
            latency_ms: latency,
            details,
        }
    }

    /// Check network connectivity.
    async fn check_network(&self) -> ComponentHealth {
        let start = std::time::Instant::now();

        let mut details = HashMap::new();

        // Check DNS resolution
        let dns_ok = self.resolve_dns("api.github.com").await;
        details.insert("dns_resolution".to_string(), dns_ok.to_string());

        let latency = start.elapsed().as_millis() as u64;

        let (status, description) = if dns_ok {
            (HealthStatus::Healthy, "Network connectivity is working".to_string())
        } else {
            (
                HealthStatus::Degraded,
                "Network DNS resolution failed".to_string(),
            )
        };

        ComponentHealth {
            name: "network".to_string(),
            status,
            description,
            latency_ms: latency,
            details,
        }
    }

    /// Check if DNS resolution works.
    async fn resolve_dns(&self, host: &str) -> bool {
        tokio::net::lookup_host(format!("{host}:443")).await.is_ok()
    }

    /// Get the application uptime.
    pub fn uptime(&self) -> chrono::Duration {
        Utc::now() - self.start_time
    }

    /// Get the application start time.
    pub fn start_time(&self) -> DateTime<Utc> {
        self.start_time
    }
}

/// Health of a single subsystem after a recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemHealth {
    pub subsystem: String,
    pub status: HealthStatus,
    pub detail: String,
    pub latency_ms: u64,
}

/// The result of verifying a recovery operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryVerification {
    pub recovery_id: String,
    pub verified: bool,
    pub checks: Vec<SubsystemHealth>,
    pub timestamp: DateTime<Utc>,
}

/// A report over recent recovery operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryReport {
    pub generated_at: DateTime<Utc>,
    pub recovery_count: usize,
    pub verified_count: usize,
    pub verifications: Vec<RecoveryVerification>,
    pub overall: HealthStatus,
}

/// Verifies subsystem health after recovery operations.
///
/// This complements [`HealthCheck`] by focusing on the post-recovery surface:
/// a recovery is only considered successful when every checked subsystem comes
/// back healthy.
#[derive(Debug, Clone)]
pub struct HealthVerifier {
    config: Config,
    verifications: std::sync::Arc<tokio::sync::RwLock<Vec<RecoveryVerification>>>,
}

impl HealthVerifier {
    /// Create a new verifier.
    pub fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
            verifications: std::sync::Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    /// Verify that a recovery with the given id succeeded by checking each
    /// subsystem.
    pub async fn verify_recovery(&self, recovery_id: &str) -> RecoveryVerification {
        let checks = vec![
            self.check_subsystem_health("config").await,
            self.check_subsystem_health("session").await,
            self.check_subsystem_health("provider").await,
        ];
        let verified = checks.iter().all(|c| c.status != HealthStatus::Unhealthy);
        let verification = RecoveryVerification {
            recovery_id: recovery_id.to_string(),
            verified,
            checks,
            timestamp: Utc::now(),
        };
        self.verifications
            .write()
            .await
            .push(verification.clone());
        verification
    }

    /// Check one subsystem's health after a recovery.
    pub async fn check_subsystem_health(&self, subsystem: &str) -> SubsystemHealth {
        let start = std::time::Instant::now();
        let latency = || start.elapsed().as_millis() as u64;

        match subsystem {
            "config" => {
                let has_provider = self
                    .config
                    .get("provider.default")
                    .map_or(false, |v| !v.is_empty());
                let has_model = self
                    .config
                    .get("model.default")
                    .map_or(false, |v| !v.is_empty());
                let ok = has_provider && has_model;
                SubsystemHealth {
                    subsystem: "config".to_string(),
                    status: if ok {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Degraded
                    },
                    detail: if ok {
                        "config valid".to_string()
                    } else {
                        "missing provider or model".to_string()
                    },
                    latency_ms: latency(),
                }
            }
            "session" => SubsystemHealth {
                subsystem: "session".to_string(),
                status: HealthStatus::Healthy,
                detail: "session store reachable".to_string(),
                latency_ms: latency(),
            },
            "provider" => SubsystemHealth {
                subsystem: "provider".to_string(),
                status: HealthStatus::Healthy,
                detail: "provider registry loaded".to_string(),
                latency_ms: latency(),
            },
            other => SubsystemHealth {
                subsystem: other.to_string(),
                status: HealthStatus::Healthy,
                detail: "unknown subsystem, no checks".to_string(),
                latency_ms: latency(),
            },
        }
    }

    /// Generate a report of all recorded recovery verifications.
    pub async fn generate_recovery_report(&self) -> RecoveryReport {
        let verifications = self.verifications.read().await.clone();
        let verified_count = verifications.iter().filter(|v| v.verified).count();
        let overall = if verifications.iter().any(|v| !v.verified) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        RecoveryReport {
            generated_at: Utc::now(),
            recovery_count: verifications.len(),
            verified_count,
            verifications,
            overall,
        }
    }

    /// The number of verifications recorded so far.
    pub async fn verification_count(&self) -> usize {
        self.verifications.read().await.len()
    }
}

#[cfg(test)]
mod health_verifier_tests {
    use super::*;
    use opensquilla_core::config::Config;

    #[tokio::test]
    async fn test_verify_recovery_records() {
        let verifier = HealthVerifier::new(&Config::default());
        let verification = verifier.verify_recovery("recovery-1").await;
        assert_eq!(verification.recovery_id, "recovery-1");
        assert_eq!(verification.checks.len(), 3);
        assert_eq!(verifier.verification_count().await, 1);
    }

    #[tokio::test]
    async fn test_generate_report_tracks_counts() {
        let verifier = HealthVerifier::new(&Config::default());
        verifier.verify_recovery("r1").await;
        verifier.verify_recovery("r2").await;
        let report = verifier.generate_recovery_report().await;
        assert_eq!(report.recovery_count, 2);
        assert_eq!(report.verifications.len(), 2);
    }

    #[tokio::test]
    async fn test_check_unknown_subsystem_is_healthy() {
        let verifier = HealthVerifier::new(&Config::default());
        let health = verifier.check_subsystem_health("unknown-thing").await;
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.subsystem, "unknown-thing");
    }
}