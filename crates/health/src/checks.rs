use std::path::PathBuf;

use async_trait::async_trait;

use crate::check::HealthCheck;
use crate::report::SubsystemHealth;

/// Checks that the gateway configuration loads and parses.
pub struct ConfigCheck {
    /// Path to the config file, if one is known.
    pub config_path: Option<PathBuf>,
}

#[async_trait]
impl HealthCheck for ConfigCheck {
    fn name(&self) -> &str {
        "config"
    }

    async fn check(&self) -> SubsystemHealth {
        match &self.config_path {
            Some(path) => {
                if !path.exists() {
                    return SubsystemHealth::critical(
                        "config",
                        format!("Config file not found: {}", path.display()),
                    );
                }
                match opensquilla_core::config::Config::from_file(path) {
                    Ok(_) => SubsystemHealth::ok("config"),
                    Err(e) => {
                        SubsystemHealth::degraded("config", format!("Config file invalid: {e}"))
                    }
                }
            }
            None => SubsystemHealth::ok("config"),
        }
    }
}

/// Checks database connectivity by opening the SQLite file and running a probe.
pub struct DatabaseCheck {
    /// Path to the SQLite database file.
    pub db_path: PathBuf,
}

#[async_trait]
impl HealthCheck for DatabaseCheck {
    fn name(&self) -> &str {
        "database"
    }

    async fn check(&self) -> SubsystemHealth {
        match rusqlite::Connection::open(&self.db_path) {
            Ok(conn) => match conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0)) {
                Ok(_) => SubsystemHealth::ok("database"),
                Err(e) => SubsystemHealth::degraded(
                    "database",
                    format!("Database opened but probe query failed: {e}"),
                ),
            },
            Err(e) => SubsystemHealth::critical(
                "database",
                format!("Cannot open database {}: {e}", self.db_path.display()),
            ),
        }
    }
}

/// Checks that at least one LLM provider is configured.
pub struct ProvidersCheck {
    /// Number of configured providers.
    pub configured_providers: usize,
}

#[async_trait]
impl HealthCheck for ProvidersCheck {
    fn name(&self) -> &str {
        "providers"
    }

    async fn check(&self) -> SubsystemHealth {
        if self.configured_providers > 0 {
            SubsystemHealth::ok("providers")
        } else {
            SubsystemHealth::degraded("providers", "No providers configured")
        }
    }
}

/// Checks whether the sandbox is enabled.
pub struct SandboxCheck {
    /// Whether the sandbox subsystem is enabled.
    pub enabled: bool,
}

#[async_trait]
impl HealthCheck for SandboxCheck {
    fn name(&self) -> &str {
        "sandbox"
    }

    async fn check(&self) -> SubsystemHealth {
        if self.enabled {
            SubsystemHealth::ok("sandbox")
        } else {
            SubsystemHealth::degraded("sandbox", "Sandbox disabled")
        }
    }
}

/// Checks that at least one messaging channel is configured.
pub struct ChannelsCheck {
    /// Number of configured channels.
    pub configured_channels: usize,
}

#[async_trait]
impl HealthCheck for ChannelsCheck {
    fn name(&self) -> &str {
        "channels"
    }

    async fn check(&self) -> SubsystemHealth {
        if self.configured_channels > 0 {
            SubsystemHealth::ok("channels")
        } else {
            SubsystemHealth::degraded("channels", "No channels configured")
        }
    }
}
