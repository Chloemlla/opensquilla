use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use tracing::{debug, info};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use opensquilla_core::config::Config;

/// Global logger instance.
static LOGGER_INITIALIZED: OnceLock<bool> = OnceLock::new();

/// Configuration for the logger.
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Log level (e.g., "info", "debug", "warn", "error").
    pub level: String,
    /// Whether to output logs in JSON format.
    pub json_format: bool,
    /// Optional file path for log output.
    pub log_file: Option<PathBuf>,
    /// Whether to include file and line information.
    pub with_location: bool,
    /// Whether to include thread IDs.
    pub with_thread_ids: bool,
}

impl Default for LoggerConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_format: false,
            log_file: None,
            with_location: true,
            with_thread_ids: false,
        }
    }
}

/// Initialize the global logger with a given log level.
pub fn init_logger(level: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = LoggerConfig {
        level: level.to_string(),
        ..Default::default()
    };
    init_logger_with_config(&config)
}

/// Initialize the global logger from the application config.
pub fn init_logger_from_config(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let level = config
        .get("log.level")
        .unwrap_or_else(|| "info".to_string());
    let json_format = config
        .get("log.json")
        .unwrap_or_else(|| "false".to_string())
        == "true";
    let log_file = config.get("log.file").map(PathBuf::from);

    let logger_config = LoggerConfig {
        level,
        json_format,
        log_file,
        ..Default::default()
    };

    init_logger_with_config(&logger_config)
}

/// Initialize the logger with a specific configuration.
pub fn init_logger_with_config(config: &LoggerConfig) -> Result<(), Box<dyn std::error::Error>> {
    if LOGGER_INITIALIZED.set(true).is_err() {
        debug!("Logger already initialized");
        return Ok(());
    }

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    let mut layers = Vec::new();

    // Console output layer
    if config.json_format {
        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_span_events(FmtSpan::CLOSE)
            .with_thread_ids(config.with_thread_ids)
            .with_file(config.with_location)
            .with_line_number(config.with_location)
            .with_filter(env_filter.clone());

        layers.push(json_layer.boxed());
    } else {
        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_thread_ids(config.with_thread_ids)
            .with_file(config.with_location)
            .with_line_number(config.with_location)
            .with_filter(env_filter.clone());

        layers.push(fmt_layer.boxed());
    }

    // File output layer
    if let Some(ref log_file) = config.log_file {
        if let Some(parent) = log_file.parent() {
            fs::create_dir_all(parent).ok();
        }

        let file_appender = tracing_appender::rolling::daily(
            log_file.parent().unwrap_or(std::path::Path::new(".")),
            log_file
                .file_name()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("opensquilla.log"),
        );

        let file_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_target(true)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(file_appender)
            .with_filter(env_filter);

        layers.push(file_layer.boxed());
    }

    let subscriber = tracing_subscriber::Registry::default().with(layers);

    subscriber.init();

    info!(
        "Logger initialized (level={}, json={}, file={})",
        config.level,
        config.json_format,
        config
            .log_file
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    Ok(())
}

/// The Logger struct for runtime log management.
#[derive(Debug, Clone, Default)]
pub struct Logger {
    config: LoggerConfig,
}

impl Logger {
    /// Create a new Logger instance.
    pub fn new(config: &LoggerConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Create a logger from the application config.
    pub fn from_config(config: &Config) -> Self {
        let level = config
            .get("log.level")
            .unwrap_or_else(|| "info".to_string());
        let json_format = config
            .get("log.json")
            .unwrap_or_else(|| "false".to_string())
            == "true";
        let log_file = config.get("log.file").map(PathBuf::from);

        Self {
            config: LoggerConfig {
                level,
                json_format,
                log_file,
                ..Default::default()
            },
        }
    }

    /// Get the current log level.
    pub fn level(&self) -> &str {
        &self.config.level
    }

    /// Check if JSON format is enabled.
    pub fn is_json_format(&self) -> bool {
        self.config.json_format
    }

    /// Get the log file path if configured.
    pub fn log_file(&self) -> Option<&PathBuf> {
        self.config.log_file.as_ref()
    }
}
