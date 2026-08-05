use std::path::PathBuf;
use std::sync::Arc;

use crate::check::HealthCheck;
use crate::report::HealthReport;

/// Options used to build a full health report.
#[derive(Debug, Clone, Default)]
pub struct ReportOptions {
    /// Path to the config file, if any.
    pub config_path: Option<PathBuf>,
    /// Path to the SQLite database file.
    pub db_path: Option<PathBuf>,
    /// Number of configured providers.
    pub provider_count: usize,
    /// Whether the sandbox is enabled.
    pub sandbox_enabled: bool,
    /// Number of configured channels.
    pub channel_count: usize,
}

/// Aggregate the health of all subsystems into a single report.
///
/// Runs the standard set of subsystem checks (config, database, providers,
/// sandbox, channels) concurrently and folds them into one report.
pub async fn build_report(options: &ReportOptions) -> HealthReport {
    let checks: Vec<Arc<dyn HealthCheck>> = vec![
        Arc::new(crate::checks::ConfigCheck {
            config_path: options.config_path.clone(),
        }),
        Arc::new(crate::checks::DatabaseCheck {
            db_path: options
                .db_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("opensquilla.db")),
        }),
        Arc::new(crate::checks::ProvidersCheck {
            configured_providers: options.provider_count,
        }),
        Arc::new(crate::checks::SandboxCheck {
            enabled: options.sandbox_enabled,
        }),
        Arc::new(crate::checks::ChannelsCheck {
            configured_channels: options.channel_count,
        }),
    ];

    let subsystems = futures::future::join_all(checks.iter().map(|c| c.check())).await;
    HealthReport::new(subsystems)
}
