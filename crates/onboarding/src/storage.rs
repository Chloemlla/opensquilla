use std::path::PathBuf;

use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Configuration storage backend.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// Store config in a file (YAML or JSON).
    File {
        path: PathBuf,
        format: StorageFormat,
    },
    /// Store config in environment variables.
    Environment,
}

/// Configuration file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageFormat {
    Yaml,
    Json,
}

/// Configuration storage for onboarding.
#[derive(Debug, Clone)]
pub struct ConfigStorage {
    backend: StorageBackend,
}

impl ConfigStorage {
    /// Create a new config storage from the application config.
    pub fn new(config: &Config) -> Result<Self, ConfigStorageError> {
        let config_path = Config::discover_path().map_err(|e| {
            ConfigStorageError::InitError(format!("Cannot discover config path: {e}"))
        })?;

        let format = match config_path.extension().and_then(|e| e.to_str()) {
            Some("yaml" | "yml") => StorageFormat::Yaml,
            Some("json") => StorageFormat::Json,
            _ => StorageFormat::Yaml, // Default
        };

        Ok(Self {
            backend: StorageBackend::File {
                path: config_path,
                format,
            },
        })
    }

    /// Create a storage with a specific file path.
    pub fn with_path(path: PathBuf) -> Self {
        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("yaml" | "yml") => StorageFormat::Yaml,
            Some("json") => StorageFormat::Json,
            _ => StorageFormat::Yaml,
        };

        Self {
            backend: StorageBackend::File { path, format },
        }
    }

    /// Save the configuration to storage.
    pub async fn save_config(&self, config: &Config) -> Result<(), ConfigStorageError> {
        match &self.backend {
            StorageBackend::File { path, format } => {
                // Ensure parent directory exists
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        ConfigStorageError::WriteError(format!(
                            "Cannot create config directory: {e}"
                        ))
                    })?;
                }

                let entries = config.list();
                let config_map: std::collections::BTreeMap<String, String> =
                    entries.into_iter().collect();

                match format {
                    StorageFormat::Yaml => {
                        let yaml_str = serde_yaml::to_string(&config_map)
                            .map_err(|e| ConfigStorageError::SerializationError(e.to_string()))?;
                        std::fs::write(path, &yaml_str).map_err(|e| {
                            ConfigStorageError::WriteError(format!("Cannot write config: {e}"))
                        })?;
                    }
                    StorageFormat::Json => {
                        let json_str = serde_json::to_string_pretty(&config_map)
                            .map_err(|e| ConfigStorageError::SerializationError(e.to_string()))?;
                        std::fs::write(path, &json_str).map_err(|e| {
                            ConfigStorageError::WriteError(format!("Cannot write config: {e}"))
                        })?;
                    }
                }

                info!("Configuration saved to {}", path.display());
            }
            StorageBackend::Environment => {
                // Environment variable storage is read-only
                warn!("Cannot save config to environment variables");
            }
        }
        Ok(())
    }

    /// Load the configuration from storage.
    pub async fn load_config(&self) -> Result<Config, ConfigStorageError> {
        match &self.backend {
            StorageBackend::File { path, format } => {
                if !path.exists() {
                    return Err(ConfigStorageError::NotFound(path.display().to_string()));
                }

                let content = std::fs::read_to_string(path).map_err(|e| {
                    ConfigStorageError::ReadError(format!("Cannot read config: {e}"))
                })?;

                match format {
                    StorageFormat::Yaml => {
                        let config_map: std::collections::BTreeMap<String, String> =
                            serde_yaml::from_str(&content).map_err(|e| {
                                ConfigStorageError::DeserializationError(e.to_string())
                            })?;
                        let mut config = Config::default();
                        for (key, value) in config_map {
                            config.set(&key, &value).ok();
                        }
                        debug!("Configuration loaded from {}", path.display());
                        Ok(config)
                    }
                    StorageFormat::Json => {
                        let config_map: std::collections::BTreeMap<String, String> =
                            serde_json::from_str(&content).map_err(|e| {
                                ConfigStorageError::DeserializationError(e.to_string())
                            })?;
                        let mut config = Config::default();
                        for (key, value) in config_map {
                            config.set(&key, &value).ok();
                        }
                        debug!("Configuration loaded from {}", path.display());
                        Ok(config)
                    }
                }
            }
            StorageBackend::Environment => {
                let mut config = Config::default();
                for (key, value) in std::env::vars() {
                    if key.starts_with("OPEN") || key.starts_with("OPEN_SQUILLA_") {
                        let config_key = key
                            .to_lowercase()
                            .trim_start_matches("open_squilla_")
                            .replace('_', ".");
                        config.set(&config_key, &value).ok();
                    }
                }
                info!("Configuration loaded from environment variables");
                Ok(config)
            }
        }
    }

    /// Check if the storage backend is available.
    pub fn is_available(&self) -> bool {
        match &self.backend {
            StorageBackend::File { path, .. } => path.parent().map_or(false, |p| p.exists()),
            StorageBackend::Environment => true,
        }
    }

    /// Get the storage path if it's a file backend.
    pub fn path(&self) -> Option<&PathBuf> {
        match &self.backend {
            StorageBackend::File { path, .. } => Some(path),
            StorageBackend::Environment => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigStorageError {
    #[error("Configuration file not found: {0}")]
    NotFound(String),

    #[error("Cannot read configuration: {0}")]
    ReadError(String),

    #[error("Cannot write configuration: {0}")]
    WriteError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Initialization error: {0}")]
    InitError(String),
}
