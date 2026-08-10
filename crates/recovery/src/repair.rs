use std::path::PathBuf;

use opensquilla_core::config::{Config, LlmConfig, ProviderConfig};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::health::has_default_model;

/// Types of configuration issues that can be repaired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigIssue {
    MissingKey,
    InvalidValue,
    DeprecatedKey,
    InvalidFormat,
    MissingProvider,
    MissingApiKey,
}

/// A detected configuration issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigIssueReport {
    pub issue_type: ConfigIssue,
    pub key: String,
    pub description: String,
    pub severity: IssueSeverity,
    pub can_auto_fix: bool,
}

/// Severity of a configuration issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

/// Configuration repair manager.
#[derive(Debug, Clone)]
pub struct ConfigRepair {
    config: Config,
    config_path: PathBuf,
}

impl ConfigRepair {
    /// Create a new config repair manager.
    pub fn new(config: &Config) -> Result<Self, ConfigRepairError> {
        let config_path = Config::discover_path()
            .map_err(|e| ConfigRepairError::ConfigError(format!("Cannot find config: {e}")))?;

        Ok(Self {
            config: config.clone(),
            config_path,
        })
    }

    /// Scan the configuration for issues.
    pub fn scan(&self) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        // Check for missing API keys.
        for provider in &self.config.providers {
            if provider
                .api_key
                .as_deref()
                .is_none_or(|k| k.trim().is_empty())
            {
                issues.push(ConfigIssueReport {
                    issue_type: ConfigIssue::MissingApiKey,
                    key: format!("providers.{}.api_key", provider.name),
                    description: format!("API key missing for provider '{}'", provider.name),
                    severity: IssueSeverity::Error,
                    can_auto_fix: false,
                });
            }
        }

        // Check for default model.
        if !has_default_model(&self.config) {
            issues.push(ConfigIssueReport {
                issue_type: ConfigIssue::MissingKey,
                key: "model.default".to_string(),
                description: "Default model not configured".to_string(),
                severity: IssueSeverity::Warning,
                can_auto_fix: true,
            });
        }

        // Check for default provider (no primary `llm` provider designated).
        if llm_provider_missing(&self.config) {
            issues.push(ConfigIssueReport {
                issue_type: ConfigIssue::MissingKey,
                key: "provider.default".to_string(),
                description: "Default provider not configured".to_string(),
                severity: IssueSeverity::Warning,
                can_auto_fix: first_configured_provider(&self.config).is_some(),
            });
        }

        issues
    }

    /// Auto-fix configurable issues.
    pub async fn auto_fix(&mut self) -> Result<Vec<ConfigIssueReport>, ConfigRepairError> {
        let issues = self.scan();
        let mut fixed = Vec::new();

        for issue in &issues {
            if !issue.can_auto_fix {
                continue;
            }

            if let ConfigIssue::MissingKey = issue.issue_type {
                    if issue.key == "model.default" {
                        self.config
                            .llm
                            .get_or_insert_with(LlmConfig::default)
                            .model = "gpt-4o".to_string();
                        fixed.push(issue.clone());
                        info!("Auto-fixed: set model.default to gpt-4o");
                    } else if issue.key == "provider.default" {
                        // Try to find the first configured provider.
                        if let Some(provider) = self.find_first_configured_provider() {
                            self.config
                                .llm
                                .get_or_insert_with(LlmConfig::default)
                                .provider = provider.clone();
                            fixed.push(issue.clone());
                            info!("Auto-fixed: set provider.default to {provider}");
                        }
                    }
                }
            }
        }

        if !fixed.is_empty() {
            self.config.save().ok();
        }

        Ok(fixed)
    }

    /// Find the first configured provider.
    fn find_first_configured_provider(&self) -> Option<String> {
        self.config
            .providers
            .iter()
            .find(|p| p.api_key.as_deref().is_some_and(|k| !k.trim().is_empty()))
            .map(|p| p.name.clone())
    }

    /// Backup the current configuration.
    pub async fn backup(&self) -> Result<PathBuf, ConfigRepairError> {
        let backup_path = self.config_path.with_extension(format!(
            "bak.{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ));

        if self.config_path.exists() {
            std::fs::copy(&self.config_path, &backup_path)
                .map_err(|e| ConfigRepairError::IoError(format!("Cannot create backup: {e}")))?;
            info!("Configuration backed up to {}", backup_path.display());
        }

        Ok(backup_path)
    }

    /// Restore from a backup file.
    pub async fn restore_from_backup(
        &mut self,
        backup_path: &PathBuf,
    ) -> Result<(), ConfigRepairError> {
        if !backup_path.exists() {
            return Err(ConfigRepairError::BackupNotFound(
                backup_path.display().to_string(),
            ));
        }

        std::fs::copy(backup_path, &self.config_path)
            .map_err(|e| ConfigRepairError::IoError(format!("Cannot restore from backup: {e}")))?;

        // Reload config
        self.config = Config::load()
            .map_err(|e| ConfigRepairError::ConfigError(format!("Cannot reload config: {e}")))?;

        info!("Configuration restored from {}", backup_path.display());
        Ok(())
    }

    /// Validate the configuration and return whether it's valid.
    pub fn is_valid(&self) -> bool {
        self.scan()
            .iter()
            .all(|i| i.severity != IssueSeverity::Error)
    }

    /// Get the config path.
    pub fn config_path(&self) -> &PathBuf {
        &self.config_path
    }
}

impl ConfigRepair {
    /// Validate and repair a configuration in place, returning the issues fixed.
    ///
    /// Fixes applied:
    ///
    /// 1. a missing default provider is set from the first configured provider,
    /// 2. a missing default model is set to a sensible fallback.
    pub async fn repair_config(config: &mut Config) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        if llm_provider_missing(config) {
            if let Some(provider) = first_configured_provider(config) {
                config
                    .llm
                    .get_or_insert_with(LlmConfig::default)
                    .provider = provider.clone();
                issues.push(make_issue(
                    ConfigIssue::MissingKey,
                    "provider.default",
                    &format!("Set provider.default to {provider}"),
                    true,
                ));
            }
        }

        if !has_default_model(config) {
            config
                .llm
                .get_or_insert_with(LlmConfig::default)
                .model = "gpt-4o".to_string();
            issues.push(make_issue(
                ConfigIssue::MissingKey,
                "model.default",
                "Set model.default to gpt-4o",
                true,
            ));
        }

        issues
    }

    /// Repair a single provider's configuration: ensure the `api_key` and
    /// `default_model` fields exist, then persist.
    pub async fn repair_provider_config(&mut self, provider: &str) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        if let Some(provider_config) = self
            .config
            .providers
            .iter_mut()
            .find(|p| p.name == provider)
        {
            if provider_config
                .api_key
                .as_deref()
                .is_none_or(|k| k.trim().is_empty())
            {
                provider_config.api_key = Some(String::new());
                issues.push(make_issue(
                    ConfigIssue::MissingApiKey,
                    &format!("providers.{provider}.api_key"),
                    &format!("API key missing for provider '{provider}'"),
                    false,
                ));
            }

            if provider_config
                .default_model
                .as_deref()
                .is_none_or(|m| m.trim().is_empty())
            {
                provider_config.default_model = provider_config.models.first().cloned();
                issues.push(make_issue(
                    ConfigIssue::MissingKey,
                    &format!("providers.{provider}.default_model"),
                    &format!("Model not configured for provider '{provider}'"),
                    true,
                ));
            }
        }

        if !issues.is_empty() {
            self.config.save().ok();
        }
        issues
    }

    /// Repair a single channel's configuration. A typed `ChannelConfig` always
    /// carries `enabled` and `channel_type`, so there is nothing left to add;
    /// the method exists to keep the repair surface uniform.
    pub async fn repair_channel_config(_channel: &str) -> Vec<ConfigIssueReport> {
        Vec::new()
    }

    /// Migrate legacy env-style provider keys into typed provider entries.
    /// Top-level legacy config keys cannot survive a typed `Config` load, so
    /// the process environment is the only remaining source of those values.
    pub async fn auto_migrate_config(&mut self) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        let legacy_env: [(&str, &str, &str); 2] = [
            ("OPENAI_API_KEY", "openai", "openai"),
            ("ANTHROPIC_API_KEY", "anthropic", "anthropic"),
        ];
        for (env, name, provider_type) in legacy_env {
            let Ok(key) = std::env::var(env) else {
                continue;
            };
            if key.trim().is_empty() || self.config.providers.iter().any(|p| p.name == name) {
                continue;
            }
            self.config.providers.push(ProviderConfig {
                name: name.to_string(),
                provider_type: provider_type.to_string(),
                api_key: Some(key),
                base_url: None,
                models: Vec::new(),
                default_model: None,
                max_retries: 3,
                timeout_secs: 60,
            });
            issues.push(make_issue(
                ConfigIssue::DeprecatedKey,
                env,
                &format!("Migrated {env} into a configured provider"),
                true,
            ));
        }

        if !issues.is_empty() {
            self.config.save().ok();
        }
        issues
    }
}

/// Whether no primary `llm` provider is designated.
fn llm_provider_missing(config: &Config) -> bool {
    config
        .llm
        .as_ref()
        .is_none_or(|l| l.provider.trim().is_empty())
}

/// Find the first provider with a non-empty `api_key`.
fn first_configured_provider(config: &Config) -> Option<String> {
    config
        .providers
        .iter()
        .find(|p| p.api_key.as_deref().is_some_and(|k| !k.trim().is_empty()))
        .map(|p| p.name.clone())
}

/// Build a lightweight [`ConfigIssueReport`] for a fix that was applied.
///
/// A free function (not a `&self` method) so it can be called while
/// `self.config.providers` is still mutably borrowed.
fn make_issue(
    issue_type: ConfigIssue,
    key: &str,
    description: &str,
    can_auto_fix: bool,
) -> ConfigIssueReport {
    ConfigIssueReport {
        issue_type,
        key: key.to_string(),
        description: description.to_string(),
        severity: IssueSeverity::Warning,
        can_auto_fix,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigRepairError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Backup not found: {0}")]
    BackupNotFound(String),

    #[error("IO error: {0}")]
    IoError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str, api_key: Option<&str>, models: &[&str]) -> ProviderConfig {
        ProviderConfig {
            name: name.to_string(),
            provider_type: name.to_string(),
            api_key: api_key.map(|s| s.to_string()),
            base_url: None,
            models: models.iter().map(|m| m.to_string()).collect(),
            default_model: models.first().map(|m| m.to_string()),
            max_retries: 3,
            timeout_secs: 60,
        }
    }

    fn repair_with(config: Config) -> ConfigRepair {
        ConfigRepair {
            config,
            config_path: PathBuf::from("opensquilla.toml"),
        }
    }

    #[test]
    fn test_first_configured_provider() {
        let mut config = Config::default();
        config
            .providers
            .push(provider("openai", Some("sk-test"), &["gpt-4o"]));
        config
            .providers
            .push(provider("anthropic", None, &["claude-sonnet-4"]));
        assert_eq!(
            first_configured_provider(&config).as_deref(),
            Some("openai")
        );
    }

    #[test]
    fn test_first_configured_provider_none() {
        let config = Config::default();
        assert_eq!(first_configured_provider(&config), None);
    }

    #[test]
    fn test_scan_reports_missing_defaults() {
        let repair = repair_with(Config::default());
        let issues = repair.scan();
        assert!(issues.iter().any(|i| i.issue_type == ConfigIssue::MissingKey));
        assert!(issues.iter().any(|i| i.key == "provider.default"));
        assert!(issues.iter().any(|i| i.key == "model.default"));
    }

    #[tokio::test]
    async fn test_repair_config_fixes_missing_defaults() {
        let mut config = Config::default();
        // A configured provider with an API key but no models: the primary
        // `llm` provider is missing and repairable.
        config
            .providers
            .push(provider("openai", Some("sk-test"), &[]));

        let issues = ConfigRepair::repair_config(&mut config).await;
        assert!(!issues.is_empty());
        assert_eq!(
            config.llm.as_ref().map(|l| l.provider.as_str()),
            Some("openai")
        );
        assert!(config
            .llm
            .as_ref()
            .is_some_and(|l| !l.model.trim().is_empty()));
        assert!(issues.iter().any(|i| i.key == "provider.default"));
    }

    #[tokio::test]
    async fn test_auto_migrate_config_reads_env_keys() {
        let mut repair = repair_with(Config::default());
        // `auto_migrate_config` persists via `Config::save()`, which resolves
        // through `discover_path()`. Point it at a temp file that already
        // exists (discover_path only honors an existing path) so the write
        // never lands on a real user config.
        let temp_config = std::env::temp_dir().join("opensquilla-test-config.toml");
        std::fs::write(&temp_config, "").expect("create temp config");
        // SAFETY: edition 2024 marks set_var unsafe; this is a single-threaded
        // unit test that restores the environment before returning.
        unsafe { std::env::set_var("OPENSQUILLA_CONFIG", &temp_config) };
        unsafe { std::env::set_var("OPENAI_API_KEY", "sk-env") };
        let issues = repair.auto_migrate_config().await;
        unsafe {
            std::env::remove_var("OPENSQUILLA_CONFIG");
            std::env::remove_var("OPENAI_API_KEY");
        }
        let _ = std::fs::remove_file(&temp_config);
        assert!(issues.iter().any(|i| i.key == "OPENAI_API_KEY"));
        assert!(repair
            .config
            .providers
            .iter()
            .any(|p| p.name == "openai" && p.api_key.as_deref() == Some("sk-env")));
    }
}
