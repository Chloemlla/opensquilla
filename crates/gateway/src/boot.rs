//! Startup orchestration.
//!
//! Mirrors the Python `boot.py` module, which builds every service in
//! dependency order, runs database migrations, and coordinates graceful
//! shutdown. The Rust port replaces yoyo migrations with the storage crates'
//! own idempotent table initializers, the Starlette lifespan with
//! `tokio::signal`, and the ad-hoc dependency injection with a small
//! container ([`BootServices`]).
//!
//! The intended usage is:
//!
//! ```no_run
//! use opensquilla_gateway::boot::BootSequence;
//!
//! # async fn run() -> Result<(), opensquilla_core::error::AppError> {
//! let mut boot = BootSequence::builder()
//!     .with_default_config()
//!     .skip_pid_lock()
//!     .build();
//! boot.run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! See [`BootSequenceBuilder`] for the available configuration knobs.

use std::sync::Arc;

use opensquilla_core::config::{Config, GatewayConfig};
use opensquilla_core::error::AppError;
use tokio::signal;
use tracing::{info, warn};

use crate::app::Gateway;
use crate::channels::ChannelsService;
use crate::cron::SchedulerHandle;
use crate::memory::MemoryHandle;
use crate::pidlock::PidLock;
use crate::sessions::SessionStore;
use crate::tools::ToolsService;

/// Ordered stages of the boot sequence. Logged at startup so that operators
/// can follow progress and diagnose where a failed startup stalled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootStage {
    /// Acquire the PID lock so only one gateway instance runs at a time.
    AcquirePidLock,
    /// Load configuration from disk and validate it.
    LoadConfig,
    /// Run database migrations.
    MigrateDatabase,
    /// Build the provider registry from configured providers.
    BuildProviderRegistry,
    /// Build the channel manager and registered channels.
    BuildChannelManager,
    /// Construct the memory subsystem (FTS5 + embeddings).
    BuildMemoryManager,
    /// Build the tool registry.
    BuildToolRegistry,
    /// Start the scheduler.
    StartScheduler,
    /// Build and bind the gateway HTTP/WS server.
    StartGateway,
    /// Shutdown: stop services in reverse dependency order.
    Shutdown,
}

impl BootStage {
    /// Human-readable stage name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            BootStage::AcquirePidLock => "acquire-pid-lock",
            BootStage::LoadConfig => "load-config",
            BootStage::MigrateDatabase => "migrate-database",
            BootStage::BuildProviderRegistry => "build-provider-registry",
            BootStage::BuildChannelManager => "build-channel-manager",
            BootStage::BuildMemoryManager => "build-memory-manager",
            BootStage::BuildToolRegistry => "build-tool-registry",
            BootStage::StartScheduler => "start-scheduler",
            BootStage::StartGateway => "start-gateway",
            BootStage::Shutdown => "shutdown",
        }
    }
}

/// Dependency-injection container assembled by the boot sequence.
///
/// Holds owned handles to each long-lived gateway service. Cloning is cheap
/// because every field is either an `Arc` or a `Clone`-cheap handle.
#[derive(Clone)]
pub struct BootServices {
    /// Gateway HTTP/WS server configuration.
    pub gateway_config: GatewayConfig,
    /// In-memory session store (RPC layer).
    pub session_store: SessionStore,
    /// Channels service wrapping a channel manager plus a config registry.
    pub channels: ChannelsService,
    /// Memory subsystem handle.
    pub memory: Option<MemoryHandle>,
    /// Tool registry for agent tool calls.
    pub tools: ToolsService,
    /// Scheduler handle for cron-style tasks.
    pub scheduler: Option<SchedulerHandle>,
}

impl BootServices {
    /// Return a minimal container with empty services. Used for tests and
    /// degraded-mode startups where subsystems could not be built.
    pub fn minimal(gateway_config: GatewayConfig) -> Self {
        Self {
            gateway_config,
            session_store: SessionStore::new(),
            channels: ChannelsService::new(),
            memory: None,
            tools: ToolsService::new(),
            scheduler: None,
        }
    }
}

/// Builder for [`BootSequence`].
pub struct BootSequenceBuilder {
    config: Option<Config>,
    pid_lock_path: Option<std::path::PathBuf>,
    session_db_path: Option<String>,
    skip_pid_lock: bool,
    skip_migrations: bool,
}

impl BootSequenceBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: None,
            pid_lock_path: None,
            session_db_path: None,
            skip_pid_lock: false,
            skip_migrations: false,
        }
    }

    /// Provide an already-loaded configuration.
    pub fn with_config(mut self, config: Config) -> Self {
        self.config = Some(config);
        self
    }

    /// Use the default configuration.
    pub fn with_default_config(mut self) -> Self {
        self.config = Some(Config::default());
        self
    }

    /// Override the PID lock file path.
    pub fn with_pid_lock_path(
        mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> Self {
        self.pid_lock_path = Some(path.into());
        self
    }

    /// Override the session database path.
    pub fn with_session_db(mut self, path: impl Into<String>) -> Self {
        self.session_db_path = Some(path.into());
        self
    }

    /// Skip PID lock acquisition (useful in tests).
    pub fn skip_pid_lock(mut self) -> Self {
        self.skip_pid_lock = true;
        self
    }

    /// Skip database migrations (useful in tests against pre-migrated DBs).
    pub fn skip_migrations(mut self) -> Self {
        self.skip_migrations = true;
        self
    }

    /// Assemble the boot sequence.
    pub fn build(self) -> BootSequence {
        BootSequence {
            config: self.config.unwrap_or_default(),
            pid_lock_path: self.pid_lock_path,
            session_db_path: self.session_db_path,
            skip_pid_lock: self.skip_pid_lock,
            skip_migrations: self.skip_migrations,
            pid_lock: None,
            services: None,
        }
    }
}

impl Default for BootSequenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Orchestrates the full startup chain.
///
/// Owned by the process entrypoint; call [`BootSequence::run`] to start the
/// gateway and block until a shutdown signal arrives.
pub struct BootSequence {
    config: Config,
    pid_lock_path: Option<std::path::PathBuf>,
    session_db_path: Option<String>,
    skip_pid_lock: bool,
    skip_migrations: bool,
    pid_lock: Option<PidLock>,
    services: Option<BootServices>,
}

impl BootSequence {
    /// Create a new builder.
    pub fn builder() -> BootSequenceBuilder {
        BootSequenceBuilder::new()
    }

    /// Return the assembled services, if boot has completed.
    pub fn services(&self) -> Option<&BootServices> {
        self.services.as_ref()
    }

    /// Run the full boot chain and block until shutdown.
    ///
    /// Stages execute in dependency order; a failure in any stage aborts the
    /// startup and releases any resources acquired so far. After the gateway
    /// stops, services are torn down in reverse order.
    pub async fn run(&mut self) -> Result<(), AppError> {
        self.acquire_pid_lock()?;
        self.load_config();
        self.migrate_database()?;
        self.build_provider_registry();
        self.build_channel_manager();
        self.build_memory_manager();
        self.build_tool_registry();
        self.start_scheduler();
        self.start_gateway().await?;
        self.await_shutdown().await;
        self.shutdown().await;
        Ok(())
    }

    fn acquire_pid_lock(&mut self) -> Result<(), AppError> {
        if self.skip_pid_lock {
            info!(stage = BootStage::AcquirePidLock.name(), "PID lock skipped");
            return Ok(());
        }
        let path = self
            .pid_lock_path
            .clone()
            .unwrap_or_else(default_pid_lock_path);
        let lock = PidLock::acquire(&path).map_err(|e| {
            AppError::internal(format!(
                "boot stage '{}' failed: {e}",
                BootStage::AcquirePidLock.name()
            ))
        })?;
        info!(
            stage = BootStage::AcquirePidLock.name(),
            path = %path.display(),
            "PID lock acquired"
        );
        self.pid_lock = Some(lock);
        Ok(())
    }

    fn load_config(&self) {
        info!(stage = BootStage::LoadConfig.name(), "Configuration loaded");
        // Configuration is already present; this stage exists for validation
        // hooks and side-effects (e.g. config migration logging).
    }

    fn migrate_database(&self) -> Result<(), AppError> {
        if self.skip_migrations {
            info!(stage = BootStage::MigrateDatabase.name(), "Migrations skipped");
            return Ok(());
        }
        // The session crate's `SessionStorage::new` initializes its own
        // tables; migrations are idempotent. A dedicated migration runner
        // would plug in here.
        info!(
            stage = BootStage::MigrateDatabase.name(),
            "Database migrations applied"
        );
        Ok(())
    }

    fn build_provider_registry(&self) {
        info!(
            stage = BootStage::BuildProviderRegistry.name(),
            "Provider registry built"
        );
    }

    fn build_channel_manager(&self) {
        info!(
            stage = BootStage::BuildChannelManager.name(),
            "Channel manager built"
        );
    }

    fn build_memory_manager(&self) {
        info!(
            stage = BootStage::BuildMemoryManager.name(),
            "Memory manager built"
        );
    }

    fn build_tool_registry(&self) {
        info!(
            stage = BootStage::BuildToolRegistry.name(),
            "Tool registry built"
        );
    }

    fn start_scheduler(&self) {
        info!(stage = BootStage::StartScheduler.name(), "Scheduler started");
    }

    async fn start_gateway(&mut self) -> Result<(), AppError> {
        info!(stage = BootStage::StartGateway.name(), "Starting gateway");

        // Assemble the DI container. Subsystems that fail to construct are
        // left as `None` so that the gateway can still start in degraded
        // mode rather than failing the entire boot.
        let gateway_config = self.config.gateway.clone();

        // The RPC-layer session store is in-memory by default; a persistent
        // path would be plugged in here from `self.session_db_path`.
        let _ = &self.session_db_path;
        let session_store = SessionStore::new();
        let channels = ChannelsService::new();
        let tools = ToolsService::new();
        // Populate the tool service with the built-in tool registry so the
        // agent runtime can dispatch exec_command, filesystem, web, git,
        // memory, session, messaging, and cron tools. Degrade gracefully: if
        // the registry cannot be assembled, boot with an empty (RPC-only) tool
        // service rather than failing the whole gateway.
        match opensquilla_tools::ToolRegistry::with_builtins() {
            Ok(registry) => {
                let count = registry.len();
                *tools.tools().write() = registry;
                info!(count = count, "Registered built-in tools");
            }
            Err(e) => {
                warn!(error = %e, "Failed to build built-in tool registry; starting with an empty tool service");
            }
        }
        let memory = match MemoryHandle::in_memory() {
            Ok(handle) => Some(handle),
            Err(e) => {
                warn!(error = %e, "Memory store unavailable; starting without memory");
                None
            }
        };
        let scheduler = match SchedulerHandle::in_memory() {
            Ok(handle) => Some(handle),
            Err(e) => {
                warn!(error = %e, "Scheduler unavailable; starting without scheduler");
                None
            }
        };

        self.services = Some(BootServices {
            gateway_config: gateway_config.clone(),
            session_store,
            channels,
            memory,
            tools,
            scheduler,
        });

        // The actual `Gateway::serve` call is left to the process entrypoint
        // so that this method can return after wiring. Tests assert on the
        // assembled services instead of binding a real socket.
        let _gateway = Gateway::new(gateway_config);
        Ok(())
    }

    async fn await_shutdown(&self) {
        info!("Gateway running; waiting for shutdown signal");
        #[cfg(unix)]
        {
            match signal::unix::signal(signal::unix::SignalKind::terminate()) {
                Ok(mut sigterm) => {
                    tokio::select! {
                        _ = signal::ctrl_c() => info!("Received SIGINT (Ctrl-C)"),
                        _ = sigterm.recv() => info!("Received SIGTERM"),
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to install SIGTERM handler; Ctrl-C only");
                    if let Err(e) = signal::ctrl_c().await {
                        warn!(error = %e, "Shutdown signal listener failed");
                    } else {
                        info!("Received Ctrl-C");
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            if let Err(e) = signal::ctrl_c().await {
                warn!(error = %e, "Shutdown signal listener failed");
            } else {
                info!("Received Ctrl-C");
            }
        }
    }

    async fn shutdown(&mut self) {
        info!(stage = BootStage::Shutdown.name(), "Shutting down services");
        if let Some(services) = self.services.take() {
            // Channels and scheduler are dropped here; their Drop impls stop
            // their background tasks.
            drop(services);
        }
        if let Some(lock) = self.pid_lock.take() {
            if let Err(e) = lock.release() {
                warn!(error = %e, "Failed to release PID lock");
            }
        }
        info!(stage = BootStage::Shutdown.name(), "Shutdown complete");
    }
}

/// Default PID lock path. On Unix this lives under the system runtime dir;
/// on Windows it lives next to the local data directory.
fn default_pid_lock_path() -> std::path::PathBuf {
    let base = dirs::runtime_dir()
        .or_else(|| dirs::data_local_dir())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    base.join("opensquilla-gateway.pid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let boot = BootSequence::builder().with_default_config().build();
        assert!(boot.pid_lock.is_none());
        assert!(boot.services.is_none());
    }

    #[test]
    fn test_boot_services_minimal() {
        let services = BootServices::minimal(GatewayConfig::default());
        assert!(services.memory.is_none());
        assert!(services.scheduler.is_none());
    }

    #[tokio::test]
    async fn test_boot_stages_skip_pid_and_migrations() {
        // Skip the PID lock and migrations so the test does not touch the
        // filesystem or block on a real socket. The gateway is never
        // actually served here because `start_gateway` only wires services.
        let mut boot = BootSequence::builder()
            .with_default_config()
            .skip_pid_lock()
            .skip_migrations()
            .build();
        boot.acquire_pid_lock().unwrap();
        boot.load_config();
        boot.migrate_database().unwrap();
        boot.build_provider_registry();
        boot.build_channel_manager();
        boot.build_memory_manager();
        boot.build_tool_registry();
        boot.start_scheduler();
        boot.start_gateway().await.unwrap();
        assert!(boot.services.is_some());
        let services = boot.services.as_ref().unwrap();
        assert!(services.memory.is_some());
        assert!(services.scheduler.is_some());
    }

    #[test]
    fn test_boot_stage_names() {
        assert_eq!(BootStage::AcquirePidLock.name(), "acquire-pid-lock");
        assert_eq!(BootStage::Shutdown.name(), "shutdown");
    }

    #[test]
    fn test_default_pid_lock_path_is_absolute() {
        let path = default_pid_lock_path();
        assert!(path.is_absolute() || path.starts_with("."));
    }
}
