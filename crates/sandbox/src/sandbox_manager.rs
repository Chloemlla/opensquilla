//! High-level sandbox orchestration.
//!
//! [`SandboxManager`] is the primary application-facing entry point: it owns a
//! default [`Sandbox`] backend, an optional [`NetworkProxy`], an optional
//! [`MetricsCollector`], a [`ProfileRegistry`] and a [`GovernanceCoordinator`],
//! and exposes a single `run` API that:
//!
//! 1. resolves the sandbox profile for an operation,
//! 2. submits the operation to the governance gate (when the profile or the
//!    caller requires approval),
//! 3. applies the policy to the network proxy,
//! 4. executes the command through the backend,
//! 5. records metrics,
//! 6. and returns a rich [`SandboxOutcome`].
//!
//! The manager is cheap to clone (all state is `Arc`), so a single instance
//! can be shared across RPC handlers, CLI subcommands and the Tauri bridge.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::governance::GovernanceCoordinator;
use crate::metrics::{ExecutionMetrics, MetricsCollector};
use crate::network::{NetworkProxy, default_blocked_ranges};
use crate::policy::{SandboxLevel, SandboxPolicy, SandboxResult, policy_summary};
use crate::profile::{ProfileRegistry, SandboxProfile};
use crate::Sandbox;

/// The outcome of a managed sandbox run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SandboxOutcome {
    /// The profile id that was used.
    pub profile_id: String,
    /// The policy summary (for audit).
    pub policy: String,
    /// The raw sandbox result.
    pub result: SandboxResult,
    /// The execution metrics snapshot.
    pub metrics: ExecutionMetrics,
    /// The governance request id, when the run required approval.
    pub approval_request_id: Option<String>,
    /// The approval decision when one was required.
    pub approval_status: Option<String>,
    /// Backend name that executed the run.
    pub backend: String,
}

/// Configuration for a single managed run.
#[derive(Debug, Clone)]
pub struct RunRequest<'a> {
    /// Operation descriptor, e.g. `"code_execution"` or `"git"`.
    pub operation: &'a str,
    /// Command to execute.
    pub command: &'a str,
    /// Command arguments.
    pub args: &'a [&'a str],
    /// Optional environment override.
    pub env: Option<HashMap<String, String>>,
    /// Optional working directory.
    pub working_dir: Option<&'a str>,
    /// Profile id to use; when `None` the profile is derived from the
    /// operation.
    pub profile: Option<&'a str>,
    /// Policy overrides merged on top of the profile.
    pub overrides: Option<&'a SandboxPolicy>,
    /// When `true`, the run is submitted to the governance gate first.
    pub require_approval: bool,
    /// Paths the operation touches (used for escalation routing).
    pub touched_paths: &'a [&'a str],
    /// Human-readable reason shown to approvers.
    pub reason: &'a str,
    /// The approver identity when the caller can approve directly (skips the
    /// queue). Rare; most callers pass `None`.
    pub auto_approve_as: Option<&'a str>,
}

impl<'a> RunRequest<'a> {
    /// A minimal request without approval.
    pub fn simple(operation: &'a str, command: &'a str, args: &'a [&'a str]) -> Self {
        Self {
            operation,
            command,
            args,
            env: None,
            working_dir: None,
            profile: None,
            overrides: None,
            require_approval: false,
            touched_paths: &[],
            reason: "",
            auto_approve_as: None,
        }
    }
}

/// A managed sandbox environment.
///
/// The backend is wrapped in a `tokio::sync::Mutex` because the [`Sandbox`]
/// trait's `execute` methods take `&mut self` (they accumulate audit state).
/// The manager itself is cheap to clone — all state is `Arc` — so a single
/// instance can be shared across RPC handlers, CLI subcommands and the Tauri
/// bridge. Concurrent `run` calls serialise on the backend mutex.
#[derive(Clone)]
pub struct SandboxManager {
    backend: Arc<tokio::sync::Mutex<Box<dyn Sandbox>>>,
    profiles: Arc<ProfileRegistry>,
    governance: Option<Arc<GovernanceCoordinator>>,
    metrics: Option<MetricsCollector>,
    proxy: Option<NetworkProxy>,
    default_level: SandboxLevel,
}

impl SandboxManager {
    /// Create a manager with a default backend and builtin profiles.
    pub fn new() -> Self {
        Self {
            backend: Arc::new(tokio::sync::Mutex::new(crate::default_sandbox())),
            profiles: Arc::new(ProfileRegistry::with_builtins()),
            governance: None,
            metrics: None,
            proxy: None,
            default_level: SandboxLevel::Standard,
        }
    }

    /// Create a manager with an explicit backend.
    pub fn with_backend(backend: Box<dyn Sandbox>) -> Self {
        Self {
            backend: Arc::new(tokio::sync::Mutex::new(backend)),
            profiles: Arc::new(ProfileRegistry::with_builtins()),
            governance: None,
            metrics: None,
            proxy: None,
            default_level: SandboxLevel::Standard,
        }
    }

    /// Attach a governance coordinator.
    pub fn with_governance(mut self, governance: GovernanceCoordinator) -> Self {
        self.governance = Some(Arc::new(governance));
        self
    }

    /// Attach a metrics collector.
    pub fn with_metrics(mut self, metrics: MetricsCollector) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set the default isolation level.
    pub fn with_default_level(mut self, level: SandboxLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Register an additional profile.
    pub fn register_profile(&self, profile: SandboxProfile) {
        self.profiles.register(profile);
    }

    /// The backend name.
    pub async fn backend_name(&self) -> &'static str {
        self.backend.lock().await.name()
    }

    /// The backend's health status.
    pub async fn health(&self) -> bool {
        self.backend.lock().await.health_check()
    }

    /// Resolve the profile to use for a request.
    pub fn resolve_profile(&self, request: &RunRequest<'_>) -> SandboxProfile {
        if let Some(id) = request.profile {
            if let Some(profile) = self.profiles.get(id) {
                return profile.clone();
            }
            // Fall back to the builtin of the same id, or the operation's
            // recommended profile.
            if let Some(builtin) = SandboxProfile::by_id(id) {
                return builtin;
            }
        }
        self.profiles.resolve_for_operation(request.operation)
    }

    /// Run an operation through the manager.
    pub async fn run(&self, request: &RunRequest<'_>) -> Result<SandboxOutcome, String> {
        let profile = self.resolve_profile(request);
        let mut policy = profile.policy.clone();
        if let Some(overrides) = request.overrides {
            policy = policy.merge(overrides);
        }
        policy.validate().map_err(|e| format!("policy validation failed: {e}"))?;

        // Governance gate.
        let args_owned: Vec<String> = request.args.iter().map(|s| s.to_string()).collect();
        let mut approval_request_id: Option<String> = None;
        let mut approval_status: Option<String> = None;
        if request.require_approval {
            if let Some(governance) = &self.governance {
                // Check the post-rejection guard first.
                if governance
                    .queue
                    .is_rejected(request.operation, request.command, &args_owned)
                    .await
                {
                    return Err(
                        "operation was previously rejected and is still inside its cooldown window"
                            .to_string(),
                    );
                }
                let id = governance
                    .submit(
                        request.operation,
                        request.command,
                        &args_owned,
                        request.reason,
                        request.touched_paths,
                    )
                    .await?;
                approval_request_id = Some(id.clone());

                if let Some(approver) = request.auto_approve_as {
                    governance.approve(&id, approver).await?;
                    approval_status = Some("approved".to_string());
                } else {
                    approval_status = Some("pending".to_string());
                    return Err(format!(
                        "approval required; request {id} is pending (auto-approval not configured)"
                    ));
                }
            } else {
                return Err(
                    "governance approval requested but no coordinator is configured".to_string(),
                );
            }
        }

        // Apply the policy to the proxy.
        if let Some(proxy) = &self.proxy {
            let ranges = default_blocked_ranges().unwrap_or_default();
            proxy.apply_policy(&policy.network, &ranges).await;
        }

        // Execute.
        let execution_id = uuid::Uuid::new_v4().to_string();
        let start = std::time::Instant::now();
        let mut backend = self.backend.lock().await;
        let backend_name = backend.name();
        let result = match &request.env {
            Some(env) => {
                backend
                    .execute_with_env(
                        request.command,
                        request.args,
                        env.clone(),
                        request.working_dir,
                        &policy,
                    )
                    .await
            }
            None => backend.execute(request.command, request.args, &policy).await,
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let result = result.map_err(|e| {
            if let Some(metrics) = &self.metrics {
                let m = ExecutionMetrics::new(&execution_id, backend_name)
                    .with_duration(std::time::Duration::from_millis(duration_ms))
                    .with_exit(-1, false)
                    .with_level(format!("{:?}", policy.level));
                let metrics_clone = metrics.clone();
                tokio::spawn(async move {
                    metrics_clone.record(m).await;
                });
            }
            e
        })?;

        // Record metrics.
        if let Some(metrics) = &self.metrics {
            let m = ExecutionMetrics::new(&execution_id, backend_name)
                .with_duration(std::time::Duration::from_millis(result.duration_ms))
                .with_exit(result.exit_code, result.exit_code == -1 && result.stderr.contains("timed out"))
                .with_level(format!("{:?}", policy.level));
            metrics.record(m).await;
        }

        Ok(SandboxOutcome {
            profile_id: profile.id,
            policy: policy_summary(&policy),
            result: result.clone(),
            metrics: ExecutionMetrics::new(&execution_id, backend_name)
                .with_duration(std::time::Duration::from_millis(result.duration_ms))
                .with_exit(result.exit_code, false)
                .with_level(format!("{:?}", policy.level)),
            approval_request_id,
            approval_status,
            backend: backend_name.to_string(),
        })
    }

    /// Health-check all components.
    pub async fn health_report(&self) -> serde_json::Value {
        let backend_name = self.backend.lock().await.name();
        let backend_ok = self.backend.lock().await.health_check();
        serde_json::json!({
            "backend": backend_name,
            "backend_ok": backend_ok,
            "profiles": self.profiles.all().len(),
            "governance": self.governance.is_some(),
            "metrics": self.metrics.is_some(),
            "proxy": self.proxy.is_some(),
            "default_level": format!("{:?}", self.default_level),
        })
    }
}

impl Default for SandboxManager {
    fn default() -> Self {
        Self::new()
    }
}

/// A builder that spins up a fully-wired manager in one call: default backend,
/// proxy (optional), governance, metrics and a ledger path.
pub struct SandboxBuilder {
    governance: bool,
    metrics: bool,
    proxy: bool,
    ledger_path: Option<PathBuf>,
    default_level: SandboxLevel,
}

impl Default for SandboxBuilder {
    fn default() -> Self {
        Self {
            governance: false,
            metrics: true,
            proxy: true,
            ledger_path: None,
            default_level: SandboxLevel::Standard,
        }
    }
}

impl SandboxBuilder {
    /// Create a builder with sane defaults (metrics + proxy on).
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the governance coordinator.
    pub fn with_governance(mut self) -> Self {
        self.governance = true;
        self
    }

    /// Enable metrics collection.
    pub fn with_metrics(mut self) -> Self {
        self.metrics = true;
        self
    }

    /// Enable the network proxy (when the backend supports proxy mode).
    pub fn with_proxy(mut self) -> Self {
        self.proxy = true;
        self
    }

    /// Set a rejection-ledger path for governance persistence.
    pub fn with_ledger(mut self, path: impl Into<PathBuf>) -> Self {
        self.ledger_path = Some(path.into());
        self
    }

    /// Set the default isolation level.
    pub fn with_level(mut self, level: SandboxLevel) -> Self {
        self.default_level = level;
        self
    }

    /// Build the manager.
    pub async fn build(self) -> Result<SandboxManager, String> {
        let mut manager = SandboxManager::with_backend(crate::default_sandbox())
            .with_default_level(self.default_level);

        if self.proxy {
            let proxy = NetworkProxy::new(crate::network::NetworkConfig::default()).await?;
            manager.proxy = Some(proxy);
        }
        if self.metrics {
            manager.metrics = Some(MetricsCollector::new());
        }
        if self.governance {
            let mut coordinator = GovernanceCoordinator::new();
            if let Some(path) = self.ledger_path {
                coordinator.queue = coordinator.queue.clone().with_ledger(path).await;
                coordinator
                    .queue
                    .start_persistence_worker()
                    .map_err(|e| format!("governance persistence worker: {e}"))?;
            }
            manager.governance = Some(Arc::new(coordinator));
        }

        Ok(manager)
    }
}

/// A minimal trait mirroring [`crate::Sandbox`] for the managed layer, so
/// callers that want the orchestration surface without the raw backend can
/// depend on this.
#[async_trait]
pub trait ManagedSandbox: Send + Sync {
    /// Run an operation and return the outcome.
    async fn run<'a>(&self, request: &RunRequest<'a>) -> Result<SandboxOutcome, String>;
}

#[async_trait]
impl ManagedSandbox for SandboxManager {
    async fn run<'a>(&self, request: &RunRequest<'a>) -> Result<SandboxOutcome, String> {
        SandboxManager::run(self, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resolve_profile_uses_builtins() {
        let manager = SandboxManager::new();
        let req = RunRequest::simple("code_execution", "python", &[]);
        let profile = manager.resolve_profile(&req);
        assert_eq!(profile.id, "strict_code_execution");
    }

    #[tokio::test]
    async fn builder_builds() {
        let manager = SandboxBuilder::new()
            .with_proxy()
            .with_metrics()
            .build()
            .await
            .unwrap();
        assert!(manager.health_report().await["backend_ok"].is_boolean());
    }

    #[tokio::test]
    async fn approval_without_coordinator_fails_cleanly() {
        let manager = SandboxManager::new();
        let req = RunRequest {
            require_approval: true,
            ..RunRequest::simple("file_read", "cat", &["/etc/hostname"])
        };
        let err = manager.run(&req).await.unwrap_err();
        assert!(err.contains("governance"));
    }
}
