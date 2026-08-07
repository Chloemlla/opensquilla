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
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::Sandbox;
use crate::config::{Backend, NetworkDefault, SandboxSettings};
use crate::denial_attribution::{SandboxRunOutcome, is_likely_sandbox_denied};
use crate::governance::GovernanceCoordinator;
use crate::managed_proxy_env::{
    extend_env_allowlist_with_proxy_vars, managed_proxy_env_for_backend,
};
use crate::metrics::{ExecutionMetrics, MetricsCollector};
use crate::network::{NetworkProxy, ProxyHandle, default_blocked_ranges};
use crate::policy::{NetworkPolicy, SandboxLevel, SandboxPolicy, SandboxResult, policy_summary};
use crate::profile::{ProfileRegistry, SandboxProfile};
use crate::run_mode::RunMode;
use crate::run_mode_policy::Principal;

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
    /// The resolved run mode for this run (`standard` / `trusted` / `full`),
    /// when a run mode or principal was supplied to the request.
    pub run_mode: Option<String>,
    /// Whether the failed run was attributed to a sandbox denial by
    /// [`crate::denial_attribution::is_likely_sandbox_denied`].
    pub denied_by_sandbox: bool,
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
    /// Requested run mode alias (`"standard"`, `"trusted"`, `"full"`, or any
    /// alias accepted by [`crate::run_mode::normalize_run_mode`]). When
    /// `principal` is present the mode is coerced against the principal's
    /// allowed set; a resolved `full` mode executes on the host (noop backend).
    pub run_mode: Option<&'a str>,
    /// The requesting principal; when present, run-mode admission follows
    /// [`crate::run_mode_policy`] (owners may select FULL, others are coerced).
    pub principal: Option<Principal>,
    /// The workspace root, used for sensitive-path workspace-aware checks.
    pub workspace: Option<&'a str>,
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
            run_mode: None,
            principal: None,
            workspace: None,
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
    profiles: Arc<Mutex<ProfileRegistry>>,
    governance: Option<Arc<GovernanceCoordinator>>,
    metrics: Option<MetricsCollector>,
    proxy: Option<NetworkProxy>,
    default_level: SandboxLevel,
    settings: Option<SandboxSettings>,
    inject_proxy_env: bool,
    proxy_handle: Arc<tokio::sync::Mutex<Option<ProxyHandle>>>,
    proxy_addr: Arc<tokio::sync::Mutex<Option<SocketAddr>>>,
}

impl SandboxManager {
    /// Create a manager with a default backend and builtin profiles.
    pub fn new() -> Self {
        Self {
            backend: Arc::new(tokio::sync::Mutex::new(crate::default_sandbox())),
            profiles: Arc::new(Mutex::new(ProfileRegistry::with_builtins())),
            governance: None,
            metrics: None,
            proxy: None,
            default_level: SandboxLevel::Standard,
            settings: None,
            inject_proxy_env: false,
            proxy_handle: Arc::new(tokio::sync::Mutex::new(None)),
            proxy_addr: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Create a manager with an explicit backend.
    pub fn with_backend(backend: Box<dyn Sandbox>) -> Self {
        Self {
            backend: Arc::new(tokio::sync::Mutex::new(backend)),
            profiles: Arc::new(Mutex::new(ProfileRegistry::with_builtins())),
            governance: None,
            metrics: None,
            proxy: None,
            default_level: SandboxLevel::Standard,
            settings: None,
            inject_proxy_env: false,
            proxy_handle: Arc::new(tokio::sync::Mutex::new(None)),
            proxy_addr: Arc::new(tokio::sync::Mutex::new(None)),
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

    /// Attach sandbox settings (builder-internal helper; use
    /// [`SandboxBuilder::with_settings`] from outside).
    pub fn with_settings_opt(mut self, settings: Option<SandboxSettings>) -> Self {
        self.settings = settings;
        self
    }

    /// Register an additional profile.
    pub fn register_profile(&self, profile: SandboxProfile) {
        self.profiles.lock().unwrap().register(profile);
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
            if let Some(profile) = self.profiles.lock().unwrap().get(id) {
                return profile.clone();
            }
            // Fall back to the builtin of the same id, or the operation's
            // recommended profile.
            if let Some(builtin) = SandboxProfile::by_id(id) {
                return builtin;
            }
        }
        self.profiles
            .lock()
            .unwrap()
            .resolve_for_operation(request.operation)
    }

    /// Run an operation through the manager.
    pub async fn run(&self, request: &RunRequest<'_>) -> Result<SandboxOutcome, String> {
        let profile = self.resolve_profile(request);
        let mut policy = profile.policy.clone();
        if let Some(overrides) = request.overrides {
            policy = policy.merge(overrides);
        }
        policy
            .validate()
            .map_err(|e| format!("policy validation failed: {e}"))?;

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

        // Apply the policy to the proxy, honouring the settings' network
        // default and the built-in default allowlist / package bundles when the
        // manager was configured with settings.
        if let Some(proxy) = &self.proxy {
            let ranges = default_blocked_ranges().unwrap_or_default();
            let mut net_policy = policy.network.clone();
            if let Some(settings) = &self.settings {
                if settings.network_default == NetworkDefault::None {
                    net_policy = NetworkPolicy::None;
                }
            }
            proxy.apply_policy(&net_policy, &ranges).await;
            if let Some(settings) = &self.settings {
                proxy.set_default_allowlist_enabled(
                    settings.network_default == NetworkDefault::ProxyAllowlist,
                );
                let bundles = crate::package_bundles::default_package_bundle_ids();
                proxy.set_enabled_bundles(&bundles).await;
            }
        }

        // Resolve the run mode against the principal, when supplied.
        let resolved_run_mode = Self::resolve_run_mode(request);
        let effective = self.settings.as_ref().map(|s| s.validate_combination());
        // FULL run mode (explicit or principal-coerced) and settings that turn
        // sandboxing off both mean host execution via the noop backend.
        let host_execution = effective
            .as_ref()
            .map(|e| !e.sandbox_enabled)
            .unwrap_or(false)
            || resolved_run_mode == Some(RunMode::Full);

        // Execute.
        let execution_id = uuid::Uuid::new_v4().to_string();
        let start = std::time::Instant::now();

        // Managed-proxy env injection (opt-in via the builder). When active,
        // the proxy is started lazily and every package-manager / HTTP client
        // proxy variable is pointed at it; the policy allowlist is extended so
        // backend env filtering keeps the injected variables.
        let mut run_env = request.env.clone();
        let mut run_policy = policy.clone();
        if let Some(settings) = &self.settings {
            if settings.network_default == NetworkDefault::None {
                run_policy.network = NetworkPolicy::None;
            }
        }
        if self.inject_proxy_env
            && self.proxy.is_some()
            && matches!(run_policy.network, NetworkPolicy::ProxyAllowlist(_))
        {
            if let Some(addr) = self.ensure_proxy_started().await {
                let backend_probe = if host_execution {
                    "noop"
                } else {
                    self.backend.lock().await.name()
                };
                let proxy_env = managed_proxy_env_for_backend(
                    Some(backend_probe),
                    &addr.ip().to_string(),
                    addr.port(),
                );
                let env = run_env.get_or_insert_with(HashMap::new);
                for (k, v) in proxy_env {
                    env.insert(k, v);
                }
                extend_env_allowlist_with_proxy_vars(&mut run_policy.env_allowlist, true);
            }
        }

        let backend_name: &'static str;
        let result = if host_execution {
            let mut noop = crate::noop::NoopSandbox::new();
            backend_name = "noop";
            execute_request(&mut noop, request, &run_env, &run_policy).await
        } else {
            let mut backend = self.backend.lock().await;
            backend_name = backend.name();
            execute_request(&mut **backend, request, &run_env, &run_policy).await
        };
        let duration_ms = start.elapsed().as_millis() as u64;

        let result = result.inspect_err(|_| {
            if let Some(metrics) = &self.metrics {
                let m = ExecutionMetrics::new(&execution_id, backend_name)
                    .with_duration(std::time::Duration::from_millis(duration_ms))
                    .with_exit(-1, false)
                    .with_level(format!("{:?}", run_policy.level));
                let metrics_clone = metrics.clone();
                tokio::spawn(async move {
                    metrics_clone.record(m).await;
                });
            }
        })?;

        // Attribute a non-zero exit to a sandbox denial (only for sandboxed
        // runs; the noop/host path is never attributed).
        let denied_by_sandbox = !host_execution
            && result.exit_code != 0
            && is_likely_sandbox_denied(&SandboxRunOutcome::new(
                result.exit_code,
                &result.stdout,
                &result.stderr,
                backend_name,
            ));

        // Record metrics.
        if let Some(metrics) = &self.metrics {
            let m = ExecutionMetrics::new(&execution_id, backend_name)
                .with_duration(std::time::Duration::from_millis(result.duration_ms))
                .with_exit(
                    result.exit_code,
                    result.exit_code == -1 && result.stderr.contains("timed out"),
                )
                .with_level(format!("{:?}", run_policy.level));
            metrics.record(m).await;
        }

        Ok(SandboxOutcome {
            profile_id: profile.id,
            policy: policy_summary(&run_policy),
            result: result.clone(),
            metrics: ExecutionMetrics::new(&execution_id, backend_name)
                .with_duration(std::time::Duration::from_millis(result.duration_ms))
                .with_exit(result.exit_code, false)
                .with_level(format!("{:?}", run_policy.level)),
            approval_request_id,
            approval_status,
            backend: backend_name.to_string(),
            run_mode: resolved_run_mode.map(|m| m.as_str().to_string()),
            denied_by_sandbox,
        })
    }

    /// Resolve the requested run mode against the principal's allowed set.
    ///
    /// When a principal is present, [`crate::run_mode_policy`] admission
    /// applies: owners may select any mode, non-owners are coerced to their
    /// default. Without a principal the raw alias is normalized (invalid
    /// aliases are treated as unset).
    fn resolve_run_mode(request: &RunRequest<'_>) -> Option<RunMode> {
        if let Some(principal) = &request.principal {
            Some(crate::run_mode_policy::coerce_run_mode_for_principal(
                request.run_mode,
                principal,
            ))
        } else {
            request
                .run_mode
                .and_then(|m| crate::run_mode::normalize_run_mode(Some(m), RunMode::Trusted).ok())
        }
    }

    /// Lazily start the managed network proxy and cache its bound address.
    ///
    /// Returns `None` when no proxy is configured or it failed to start; the
    /// caller then simply skips proxy-env injection.
    async fn ensure_proxy_started(&self) -> Option<SocketAddr> {
        if let Some(addr) = *self.proxy_addr.lock().await {
            return Some(addr);
        }
        let proxy = self.proxy.as_ref()?;
        let mut addr_guard = self.proxy_addr.lock().await;
        if let Some(addr) = *addr_guard {
            return Some(addr);
        }
        let handle = proxy.start().await.ok()?;
        let addr = proxy.bound_addr().await?;
        *self.proxy_handle.lock().await = Some(handle);
        *addr_guard = Some(addr);
        Some(addr)
    }

    /// Health-check all components.
    pub async fn health_report(&self) -> serde_json::Value {
        let backend_name = self.backend.lock().await.name();
        let backend_ok = self.backend.lock().await.health_check();
        serde_json::json!({
            "backend": backend_name,
            "backend_ok": backend_ok,
            "profiles": self.profiles.lock().unwrap().all().len(),
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
    settings: Option<SandboxSettings>,
    inject_proxy_env: bool,
}

impl Default for SandboxBuilder {
    fn default() -> Self {
        Self {
            governance: false,
            metrics: true,
            proxy: true,
            ledger_path: None,
            default_level: SandboxLevel::Standard,
            settings: None,
            inject_proxy_env: false,
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

    /// Attach sandbox settings. The settings' [`EffectiveMode`] then drives
    /// backend selection, sandbox enablement and the default network posture
    /// (see [`SandboxManager`]).
    pub fn with_settings(mut self, settings: SandboxSettings) -> Self {
        self.settings = Some(settings);
        self
    }

    /// Inject the managed-proxy environment into every proxied run. When the
    /// policy is `ProxyAllowlist`, the proxy is started lazily and the package
    /// manager / HTTP client proxy variables are pointed at it.
    pub fn with_managed_proxy_env(mut self) -> Self {
        self.inject_proxy_env = true;
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
        let settings = self.settings;
        let (backend, default_level) = match &settings {
            Some(settings) => {
                let effective = settings.validate_combination();
                let backend = select_backend(effective.backend);
                let level = security_to_sandbox_level(effective.default_level);
                (backend, level)
            }
            None => (crate::default_sandbox(), self.default_level),
        };
        let mut manager = SandboxManager::with_backend(backend)
            .with_default_level(default_level)
            .with_settings_opt(settings.clone());
        manager.inject_proxy_env = self.inject_proxy_env;

        if self.proxy {
            let proxy = NetworkProxy::new(crate::network::NetworkConfig::default()).await?;
            if let Some(settings) = &settings {
                proxy.set_default_allowlist_enabled(
                    settings.network_default == NetworkDefault::ProxyAllowlist,
                );
                let bundles = crate::package_bundles::default_package_bundle_ids();
                proxy.set_enabled_bundles(&bundles).await;
            }
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

/// Execute a request on a concrete backend with an optional environment.
async fn execute_request(
    backend: &mut dyn Sandbox,
    request: &RunRequest<'_>,
    env: &Option<HashMap<String, String>>,
    policy: &SandboxPolicy,
) -> Result<SandboxResult, String> {
    match env {
        Some(env) => {
            backend
                .execute_with_env(
                    request.command,
                    request.args,
                    env.clone(),
                    request.working_dir,
                    policy,
                )
                .await
        }
        None => backend.execute(request.command, request.args, policy).await,
    }
}

/// Map a configured [`Backend`] to a concrete sandbox backend instance.
fn select_backend(backend: Backend) -> Box<dyn Sandbox> {
    match backend {
        Backend::Auto => crate::default_sandbox(),
        Backend::Bubblewrap => Box::new(crate::linux::LinuxSandbox::new()),
        Backend::Seatbelt => Box::new(crate::macos::MacOsSandbox::new()),
        Backend::Noop => Box::new(crate::noop::NoopSandbox::new()),
        Backend::WindowsDefault => Box::new(crate::windows::WindowsSandbox::new()),
    }
}

/// Map the configured security level to the policy sandbox level.
///
/// `Disabled` (legacy mode) is mapped to Standard: host isolation is off, but
/// the policy engine still runs with a standard-level rule set.
fn security_to_sandbox_level(level: crate::config::SecurityLevel) -> SandboxLevel {
    match level {
        crate::config::SecurityLevel::Disabled | crate::config::SecurityLevel::Standard => {
            SandboxLevel::Standard
        }
        crate::config::SecurityLevel::Strict => SandboxLevel::Strict,
        crate::config::SecurityLevel::Locked => SandboxLevel::Locked,
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

    /// A command + args that succeed on every platform (used for the host/noop
    /// execution path in tests).
    fn ok_noop_command() -> (&'static str, &'static [&'static str]) {
        #[cfg(windows)]
        {
            ("cmd.exe", &["/C", "exit", "0"])
        }
        #[cfg(not(windows))]
        {
            ("sh", &["-c", "exit 0"])
        }
    }

    /// A deterministic backend for tests: no subprocess, fixed exit/output.
    struct FakeSandbox {
        name: &'static str,
        exit_code: i32,
        stderr: String,
    }

    impl FakeSandbox {
        fn ok(name: &'static str) -> Self {
            Self {
                name,
                exit_code: 0,
                stderr: String::new(),
            }
        }

        fn denied(name: &'static str) -> Self {
            Self {
                name,
                exit_code: 1,
                stderr: "sandbox: permission denied by policy".to_string(),
            }
        }

        fn result(&self) -> SandboxResult {
            SandboxResult {
                exit_code: self.exit_code,
                stdout: String::new(),
                stderr: self.stderr.clone(),
                duration_ms: 0,
                audit_log: Vec::new(),
            }
        }
    }

    #[async_trait]
    impl Sandbox for FakeSandbox {
        async fn execute(
            &mut self,
            _command: &str,
            _args: &[&str],
            _policy: &SandboxPolicy,
        ) -> Result<SandboxResult, String> {
            Ok(self.result())
        }

        async fn execute_with_env(
            &mut self,
            _command: &str,
            _args: &[&str],
            _env: HashMap<String, String>,
            _working_dir: Option<&str>,
            _policy: &SandboxPolicy,
        ) -> Result<SandboxResult, String> {
            Ok(self.result())
        }

        fn health_check(&self) -> bool {
            true
        }

        fn name(&self) -> &'static str {
            self.name
        }

        fn audit_log(&self) -> Vec<crate::policy::AuditEntry> {
            Vec::new()
        }
    }

    fn owner_principal() -> Principal {
        Principal {
            is_owner: true,
            role: Some("owner".to_string()),
            scopes: vec!["run:full".to_string()],
            authenticated: true,
        }
    }

    fn member_principal() -> Principal {
        Principal {
            is_owner: false,
            role: Some("member".to_string()),
            scopes: vec!["run:trusted".to_string()],
            authenticated: true,
        }
    }

    #[tokio::test]
    async fn non_owner_full_run_mode_is_coerced() {
        let manager = SandboxManager::with_backend(Box::new(FakeSandbox::ok("fake")));
        let req = RunRequest {
            run_mode: Some("full"),
            principal: Some(member_principal()),
            ..RunRequest::simple("shell", "true", &[])
        };
        let outcome = manager.run(&req).await.unwrap();
        // FULL is not selectable by a non-owner: coerced to TRUSTED and
        // executed sandboxed on the shared backend.
        assert_eq!(outcome.run_mode.as_deref(), Some("trusted"));
        assert_eq!(outcome.backend, "fake");
        assert!(!outcome.denied_by_sandbox);
    }

    #[tokio::test]
    async fn owner_full_run_mode_executes_on_host() {
        let manager = SandboxManager::with_backend(Box::new(FakeSandbox::ok("fake")));
        let (cmd, args) = ok_noop_command();
        let req = RunRequest {
            run_mode: Some("full"),
            principal: Some(owner_principal()),
            ..RunRequest::simple("shell", cmd, args)
        };
        let outcome = manager.run(&req).await.unwrap();
        assert_eq!(outcome.run_mode.as_deref(), Some("full"));
        // FULL host access bypasses the sandbox backend entirely.
        assert_eq!(outcome.backend, "noop");
    }

    #[tokio::test]
    async fn denial_attribution_classified() {
        let manager = SandboxManager::with_backend(Box::new(FakeSandbox::denied("fake")));
        let req = RunRequest::simple("shell", "touch /protected", &["/protected"]);
        let outcome = manager.run(&req).await.unwrap();
        assert!(outcome.denied_by_sandbox);
        // The noop/host path is never attributed, even for a denial-looking
        // failure.
        assert!(!is_likely_sandbox_denied(&SandboxRunOutcome::new(
            1,
            "",
            "sandbox: permission denied",
            "noop",
        )));
    }

    #[tokio::test]
    async fn settings_select_noop_backend() {
        let manager = SandboxBuilder::new()
            .with_proxy()
            .with_settings(SandboxSettings {
                backend: crate::config::Backend::Noop,
                ..SandboxSettings::default()
            })
            .build()
            .await
            .unwrap();
        assert_eq!(manager.backend_name().await, "noop");
    }

    #[tokio::test]
    async fn settings_disable_sandbox_forces_host_execution() {
        let manager = SandboxManager::with_backend(Box::new(FakeSandbox::ok("fake")));
        let settings = SandboxSettings {
            sandbox: false,
            security_grading: false,
            ..SandboxSettings::default()
        };
        let (cmd, args) = ok_noop_command();
        let req = RunRequest {
            workspace: Some("/tmp"),
            ..RunRequest::simple("shell", cmd, args)
        };
        // Same request, with and without settings: with sandbox disabled the
        // run goes through the noop/host path, without settings it uses the
        // configured backend.
        let without = manager.run(&req).await.unwrap();
        assert_eq!(without.backend, "fake");
        let manager = manager.with_settings_opt(Some(settings));
        let with = manager.run(&req).await.unwrap();
        assert_eq!(with.backend, "noop");
    }

    #[tokio::test]
    async fn settings_default_allowlist_and_bundles_opt_in() {
        let manager = SandboxBuilder::new()
            .with_proxy()
            .with_settings(SandboxSettings::default())
            .build()
            .await
            .unwrap();
        let proxy = manager.proxy.as_ref().unwrap();
        assert!(proxy.is_domain_allowed("github.com").await);
        assert!(proxy.is_domain_allowed("pypi.org").await);
        assert!(!proxy.is_domain_allowed("evil.example.com").await);
    }
}
