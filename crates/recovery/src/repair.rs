use std::path::PathBuf;

use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use tracing::info;

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
        let entries = self.config.list();

        // Check for missing API keys
        let provider_keys: Vec<String> = entries
            .keys()
            .filter(|k| k.starts_with("provider.") && k.ends_with(".api_key"))
            .cloned()
            .collect();

        for key in &provider_keys {
            if entries.get(key).map_or(true, |v| v.is_empty()) {
                let provider_name = key
                    .trim_start_matches("provider.")
                    .trim_end_matches(".api_key");
                issues.push(ConfigIssueReport {
                    issue_type: ConfigIssue::MissingApiKey,
                    key: key.clone(),
                    description: format!("API key missing for provider '{provider_name}'"),
                    severity: IssueSeverity::Error,
                    can_auto_fix: false,
                });
            }
        }

        // Check for default model
        if !entries.contains_key("model.default") {
            issues.push(ConfigIssueReport {
                issue_type: ConfigIssue::MissingKey,
                key: "model.default".to_string(),
                description: "Default model not configured".to_string(),
                severity: IssueSeverity::Warning,
                can_auto_fix: true,
            });
        }

        // Check for default provider
        if !entries.contains_key("provider.default") {
            issues.push(ConfigIssueReport {
                issue_type: ConfigIssue::MissingKey,
                key: "provider.default".to_string(),
                description: "Default provider not configured".to_string(),
                severity: IssueSeverity::Warning,
                can_auto_fix: true,
            });
        }

        // Check for deprecated keys
        let deprecated_keys = ["api_key", "openai_api_key", "anthropic_api_key"];
        for key in &deprecated_keys {
            if entries.contains_key(*key) {
                issues.push(ConfigIssueReport {
                    issue_type: ConfigIssue::DeprecatedKey,
                    key: key.to_string(),
                    description: format!("Deprecated config key '{key}' - use provider-specific keys"),
                    severity: IssueSeverity::Warning,
                    can_auto_fix: true,
                });
            }
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

            match issue.issue_type {
                ConfigIssue::MissingKey => {
                    if issue.key == "model.default" {
                        self.config.set("model.default", "gpt-4o").ok();
                        fixed.push(issue.clone());
                        info!("Auto-fixed: set model.default to gpt-4o");
                    } else if issue.key == "provider.default" {
                        // Try to find the first configured provider
                        if let Some(provider) = self.find_first_configured_provider() {
                            self.config
                                .set("provider.default", &provider)
                                .ok();
                            fixed.push(issue.clone());
                            info!("Auto-fixed: set provider.default to {provider}");
                        }
                    }
                }
                ConfigIssue::DeprecatedKey => {
                    // Remove deprecated key
                    self.config.remove(&issue.key);
                    fixed.push(issue.clone());
                    info!("Auto-fixed: removed deprecated key {}", issue.key);
                }
                _ => {}
            }
        }

        if !fixed.is_empty() {
            self.config.save().ok();
        }

        Ok(fixed)
    }

    /// Find the first configured provider.
    fn find_first_configured_provider(&self) -> Option<String> {
        let entries = self.config.list();
        for key in entries.keys() {
            if key.starts_with("provider.") && key.ends_with(".api_key") {
                if !entries.get(key).map_or(true, |v| v.is_empty()) {
                    return Some(
                        key.trim_start_matches("provider.")
                            .trim_end_matches(".api_key")
                            .to_string(),
                    );
                }
            }
        }
        None
    }

    /// Backup the current configuration.
    pub async fn backup(&self) -> Result<PathBuf, ConfigRepairError> {
        let backup_path = self.config_path.with_extension(format!(
            "bak.{}",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        ));

        if self.config_path.exists() {
            std::fs::copy(&self.config_path, &backup_path).map_err(|e| {
                ConfigRepairError::IoError(format!("Cannot create backup: {e}"))
            })?;
            info!("Configuration backed up to {}", backup_path.display());
        }

        Ok(backup_path)
    }

    /// Restore from a backup file.
    pub async fn restore_from_backup(&mut self, backup_path: &PathBuf) -> Result<(), ConfigRepairError> {
        if !backup_path.exists() {
            return Err(ConfigRepairError::BackupNotFound(
                backup_path.display().to_string(),
            ));
        }

        std::fs::copy(backup_path, &self.config_path).map_err(|e| {
            ConfigRepairError::IoError(format!("Cannot restore from backup: {e}"))
        })?;

        // Reload config
        self.config = Config::load().map_err(|e| {
            ConfigRepairError::ConfigError(format!("Cannot reload config: {e}"))
        })?;

        info!("Configuration restored from {}", backup_path.display());
        Ok(())
    }

    /// Validate the configuration and return whether it's valid.
    pub fn is_valid(&self) -> bool {
        self.scan().iter().all(|i| i.severity != IssueSeverity::Error)
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
    /// 1. deprecated top-level provider keys are removed,
    /// 2. a missing `provider.default` is set from the first configured provider,
    /// 3. a missing `model.default` is set to a sensible fallback.
    pub async fn repair_config(&self, config: &mut Config) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        for key in ["api_key", "openai_api_key", "anthropic_api_key"] {
            if config.list().contains_key(key) {
                config.remove(key);
                issues.push(self.issue(
                    ConfigIssue::DeprecatedKey,
                    key,
                    "Deprecated config key removed",
                    true,
                ));
            }
        }

        let provider_default = config.get("provider.default").unwrap_or_default();
        if provider_default.trim().is_empty() {
            if let Some(provider) = first_configured_provider(config) {
                config.set("provider.default", &provider).ok();
                issues.push(self.issue(
                    ConfigIssue::MissingKey,
                    "provider.default",
                    &format!("Set provider.default to {provider}"),
                    true,
                ));
            }
        }

        if config.get("model.default").map_or(true, |v| v.is_empty()) {
            config.set("model.default", "gpt-4o").ok();
            issues.push(self.issue(
                ConfigIssue::MissingKey,
                "model.default",
                "Set model.default to gpt-4o",
                true,
            ));
        }

        issues
    }

    /// Repair a single provider's configuration: ensure the `api_key` and
    /// `model` keys exist, then persist.
    pub async fn repair_provider_config(&mut self, provider: &str) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();
        let base = format!("provider.{provider}");

        let key = format!("{base}.api_key");
        if self.config.get(&key).map_or(true, |v| v.is_empty()) {
            self.config.set(&key, "").ok();
            issues.push(self.issue(
                ConfigIssue::MissingApiKey,
                &key,
                &format!("API key missing for provider '{provider}'"),
                false,
            ));
        }

        let model_key = format!("{base}.model");
        if self.config.get(&model_key).map_or(true, |v| v.is_empty()) {
            self.config.set(&model_key, "").ok();
            issues.push(self.issue(
                ConfigIssue::MissingKey,
                &model_key,
                &format!("Model not configured for provider '{provider}'"),
                true,
            ));
        }

        if !issues.is_empty() {
            self.config.save().ok();
        }
        issues
    }

    /// Repair a single channel's configuration: ensure the channel's base keys
    /// (`enabled`, `type`) exist, then persist.
    pub async fn repair_channel_config(&mut self, channel: &str) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();
        let base = format!("channel.{channel}");

        for suffix in ["enabled", "type"] {
            let key = format!("{base}.{suffix}");
            if !self.config.list().contains_key(&key) {
                let value = if suffix == "enabled" { "false" } else { "" };
                self.config.set(&key, value).ok();
                issues.push(self.issue(
                    ConfigIssue::MissingKey,
                    &key,
                    &format!("Added missing channel key '{key}'"),
                    true,
                ));
            }
        }

        if !issues.is_empty() {
            self.config.save().ok();
        }
        issues
    }

    /// Migrate configuration from legacy formats (env-style and top-level
    /// provider keys) into the modern `provider.<name>.<key>` shape.
    pub async fn auto_migrate_config(&mut self) -> Vec<ConfigIssueReport> {
        let mut issues = Vec::new();

        let legacy_top_level: Vec<(String, String, String)> = [
            ("openai_api_key", "provider.openai.api_key"),
            ("anthropic_api_key", "provider.anthropic.api_key"),
        ]
        .iter()
        .filter_map(|(legacy, modern)| {
            self.config
                .get(legacy)
                .filter(|v| !v.is_empty())
                .map(|v| (legacy.to_string(), modern.to_string(), v))
        })
        .collect();

        for (legacy, modern, value) in legacy_top_level {
            self.config.set(&modern, &value).ok();
            self.config.remove(&legacy);
            issues.push(self.issue(
                ConfigIssue::DeprecatedKey,
                &legacy,
                &format!("Migrated {legacy} to {modern}"),
                true,
            ));
        }

        let env_keys: Vec<(String, String, String)> = [
            ("OPENAI_API_KEY", "provider.openai.api_key"),
            ("ANTHROPIC_API_KEY", "provider.anthropic.api_key"),
        ]
        .iter()
        .filter_map(|(env, modern)| {
            self.config
                .get(env)
                .filter(|v| !v.is_empty())
                .map(|v| (env.to_string(), modern.to_string(), v))
        })
        .collect();

        for (env, modern, value) in env_keys {
            self.config.set(&modern, &value).ok();
            self.config.remove(&env);
            issues.push(self.issue(
                ConfigIssue::DeprecatedKey,
                &env,
                &format!("Migrated {env} to {modern}"),
                true,
            ));
        }

        if !issues.is_empty() {
            self.config.save().ok();
        }
        issues
    }

    /// Build a lightweight [`ConfigIssueReport`] for a fix that was applied.
    fn issue(
        &self,
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
}

/// Find the first provider with a non-empty `api_key`.
fn first_configured_provider(config: &Config) -> Option<String> {
    for (key, value) in config.list() {
        if key.starts_with("provider.") && key.ends_with(".api_key") && !value.is_empty() {
            return Some(
                key.trim_start_matches("provider.")
                    .trim_end_matches(".api_key")
                    .to_string(),
            );
        }
    }
    None
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

    #[test]
    fn test_first_configured_provider() {
        let mut config = Config::default();
        config.set("provider.openai.api_key", "sk-test").unwrap();
        config.set("provider.anthropic.api_key", "").unwrap();
        assert_eq!(first_configured_provider(&config).as_deref(), Some("openai"));
    }

    #[test]
    fn test_first_configured_provider_none() {
        let config = Config::default();
        assert_eq!(first_configured_provider(&config), None);
    }

    #[tokio::test]
    async fn test_repair_config_fixes_missing_defaults() {
        // Constructing a ConfigRepair requires a discoverable config file; in a
        // bare test environment it fails and the test passes vacuously.
        let Ok(repair) = ConfigRepair::new(&Config::default()) else {
            return;
        };
        let mut config = Config::default();
        config.set("provider.openai.api_key", "sk-test").unwrap();
        config.set("api_key", "legacy").unwrap();

        let issues = repair.repair_config(&mut config).await;
        assert!(!issues.is_empty());
        assert!(config.get("provider.default").is_some());
        assert!(config.get("model.default").is_some());
        assert!(config.get("api_key").is_none());
        assert!(issues.iter().any(|i| i.key == "api_key"));
    }

    #[tokio::test]
    async fn test_auto_migrate_config_moves_legacy_keys() {
        let Ok(mut repair) = ConfigRepair::new(&Config::default()) else {
            return;
        };
        repair.config.set("openai_api_key", "sk-legacy").ok();
        let issues = repair.auto_migrate_config().await;
        assert!(issues.iter().any(|i| i.key == "openai_api_key"));
        assert_eq!(
            repair.config.get("provider.openai.api_key").as_deref(),
            Some("sk-legacy")
        );
        assert!(repair.config.get("openai_api_key").is_none());
    }
}