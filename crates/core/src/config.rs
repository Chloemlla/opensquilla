use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level configuration for the OpenSquilla gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Gateway server configuration.
    pub gateway: GatewayConfig,
    /// Configured LLM providers.
    pub providers: Vec<ProviderConfig>,
    /// Communication channels (e.g., CLI, Slack, Discord).
    pub channels: Vec<ChannelConfig>,
    /// Model configuration and routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelConfig>,
    /// Sandbox configuration for secure tool execution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxConfig>,
    /// Skills and capability configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<SkillsConfig>,
    /// Scheduled task configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler: Option<SchedulerConfig>,
    /// Observability and telemetry configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observability: Option<ObservabilityConfig>,
}

impl Config {
    /// Load the configuration from a TOML file at the given path.
    pub fn from_file(path: impl Into<PathBuf>) -> crate::error::Result<Self> {
        let path = path.into();
        let contents = std::fs::read_to_string(&path).map_err(|e| {
            crate::error::Error::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        toml::from_str(&contents).map_err(|e| {
            crate::error::Error::Config(format!("Failed to parse config: {}", e))
        })
    }

    /// Load the configuration from a YAML file at the given path.
    pub fn from_yaml_file(path: impl Into<PathBuf>) -> crate::error::Result<Self> {
        let path = path.into();
        let contents = std::fs::read_to_string(&path).map_err(|e| {
            crate::error::Error::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;
        serde_yaml::from_str(&contents).map_err(|e| {
            crate::error::Error::Config(format!("Failed to parse YAML config: {}", e))
        })
    }

    /// Find a provider configuration by name.
    pub fn find_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Find a channel configuration by name.
    pub fn find_channel(&self, name: &str) -> Option<&ChannelConfig> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// Discover the configuration file path by checking common locations.
    pub fn discover_path() -> crate::error::Result<PathBuf> {
        // 1. Explicit env var
        if let Ok(path) = std::env::var("OPENSQUILLA_CONFIG") {
            let path = PathBuf::from(path);
            if path.exists() {
                return Ok(path);
            }
        }

        // 2. Current directory
        let local = PathBuf::from("opensquilla.toml");
        if local.exists() {
            return Ok(local);
        }

        // 3. Platform config directory
        if let Some(config_dir) = dirs::config_dir() {
            let path = config_dir
                .join("opensquilla")
                .join("opensquilla.toml");
            if path.exists() {
                return Ok(path);
            }
        }

        // 4. Home directory dotfile
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".opensquilla").join("opensquilla.toml");
            if path.exists() {
                return Ok(path);
            }
        }

        Err(crate::error::Error::Config(
            "No configuration file found. Checked OPENSQUILLA_CONFIG, ./opensquilla.toml, \
             and platform config directories."
                .to_string(),
        ))
    }

    /// Load the configuration from the discovered path.
    pub fn load() -> crate::error::Result<Self> {
        let path = Self::discover_path()?;
        Self::from_file(path)
    }

    /// Get a flat dotted-path key value from the configuration.
    ///
    /// For example, `get("scheduler.enabled")` returns "true" if the
    /// scheduler section has `enabled = true`.
    pub fn get(&self, key: &str) -> Option<String> {
        let flat = self.to_flat_map();
        flat.get(key).cloned()
    }

    /// List all configuration values as a flat dotted-path map.
    pub fn list(&self) -> HashMap<String, String> {
        self.to_flat_map()
    }

    /// Set a flat dotted-path key value in the configuration.
    pub fn set(&mut self, key: &str, value: &str) -> crate::error::Result<()> {
        let mut value_map = self.to_value_map();
        let keys: Vec<&str> = key.split('.').collect();

        let mut current = &mut value_map;
        for (i, part) in keys.iter().enumerate() {
            if i == keys.len() - 1 {
                current[&**part] = serde_json::Value::String(value.to_string());
            } else {
                current = current
                    .entry((*part).to_string())
                    .or_insert_with(|| serde_json::Value::Object(Default::default()))
                    .as_object_mut()
                    .ok_or_else(|| {
                        crate::error::Error::Config(format!(
                            "Cannot set key '{}': intermediate '{}' is not an object",
                            key, part
                        ))
                    })?;
            }
        }

        self.from_value_map(&value_map)
    }

    /// Remove a flat dotted-path key from the configuration.
    pub fn remove(&mut self, key: &str) {
        let mut value_map = self.to_value_map();
        let keys: Vec<&str> = key.split('.').collect();

        if keys.len() == 1 {
            value_map.remove(keys[0]);
        } else {
            let mut current = &mut value_map;
            for part in &keys[..keys.len() - 1] {
                match current.get_mut(*part) {
                    Some(serde_json::Value::Object(map)) => current = map,
                    _ => return,
                }
            }
            current.remove(keys[keys.len() - 1]);
        }

        let _ = self.from_value_map(&value_map);
    }

    /// Save the configuration to the discovered path as TOML.
    pub fn save(&self) -> crate::error::Result<()> {
        let path = Self::discover_path()?;
        self.save_to(&path)
    }

    /// Save the configuration to a specific path as TOML.
    pub fn save_to(&self, path: &PathBuf) -> crate::error::Result<()> {
        let contents = toml::to_string(self).map_err(|e| {
            crate::error::Error::Config(format!("Failed to serialize config: {}", e))
        })?;
        std::fs::write(path, contents).map_err(|e| {
            crate::error::Error::Config(format!(
                "Failed to write config {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(())
    }

    /// Serialize the config to a flat dotted-path string map.
    fn to_flat_map(&self) -> HashMap<String, String> {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        let mut flat = HashMap::new();
        flatten_json(&value, "", &mut flat);
        flat
    }

    /// Serialize the config to a serde_json object for mutation.
    fn to_value_map(&self) -> serde_json::Map<String, serde_json::Value> {
        serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    }

    /// Rebuild the config from a serde_json object after mutation.
    fn from_value_map(&mut self, value_map: &serde_json::Map<String, serde_json::Value>) -> crate::error::Result<()> {
        let value = serde_json::Value::Object(value_map.clone());
        let new_config: Self = serde_json::from_value(value).map_err(|e| {
            crate::error::Error::Config(format!("Failed to rebuild config after mutation: {}", e))
        })?;
        *self = new_config;
        Ok(())
    }
}

/// Recursively flatten a serde_json value into dotted-path keys.
fn flatten_json(value: &serde_json::Value, prefix: &str, out: &mut HashMap<String, String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                flatten_json(v, &key, out);
            }
        }
        serde_json::Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                let key = format!("{}.{}", prefix, i);
                flatten_json(v, &key, out);
            }
        }
        serde_json::Value::Bool(b) => {
            out.insert(prefix.to_string(), b.to_string());
        }
        serde_json::Value::Number(n) => {
            out.insert(prefix.to_string(), n.to_string());
        }
        serde_json::Value::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        serde_json::Value::Null => {}
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gateway: GatewayConfig::default(),
            providers: Vec::new(),
            channels: Vec::new(),
            models: None,
            sandbox: None,
            skills: None,
            scheduler: None,
            observability: None,
        }
    }
}

/// Gateway server binding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// The host address to bind to.
    pub host: String,
    /// The port to listen on.
    pub port: u16,
    /// Maximum number of concurrent connections.
    pub max_connections: u32,
    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
    /// Allowed CORS origins for HTTP requests.
    pub cors_origins: Vec<String>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            request_timeout_secs: 120,
            cors_origins: vec!["*".to_string()],
        }
    }
}

/// Configuration for an LLM provider (e.g., OpenAI, Anthropic, DeepSeek).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// A unique name for this provider instance.
    pub name: String,
    /// The provider type identifier (e.g., "openai", "anthropic", "deepseek").
    pub provider_type: String,
    /// API key for authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Optional base URL override for the API endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// List of model identifiers available through this provider.
    pub models: Vec<String>,
    /// The default model to use when none is specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Maximum number of retry attempts for failed requests.
    pub max_retries: u32,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

/// Configuration for a communication channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// A unique name for this channel.
    pub name: String,
    /// The channel type identifier (e.g., "cli", "slack", "discord", "web").
    pub channel_type: String,
    /// Whether this channel is enabled.
    pub enabled: bool,
    /// Channel-specific configuration key-value pairs.
    pub config: HashMap<String, String>,
}

/// Model routing and capability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model to use across all providers.
    pub default_model: Option<String>,
    /// Model routing rules for specific use cases.
    #[serde(default)]
    pub routing_rules: Vec<RoutingRule>,
}

/// A routing rule that maps a use case to a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// The use case pattern (e.g., "chat", "code", "reasoning").
    pub use_case: String,
    /// The model to use for this use case.
    pub model: String,
    /// Optional provider to use for this routing rule.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
}

/// Sandbox configuration for secure tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Whether the sandbox is enabled.
    pub enabled: bool,
    /// Sandbox type (e.g., "docker", "container", "process").
    pub sandbox_type: String,
    /// Timeout for sandbox execution in seconds.
    pub timeout_secs: u64,
    /// Resource limits for the sandbox.
    #[serde(default)]
    pub resource_limits: ResourceLimits,
}

/// Resource limits for sandboxed execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory in megabytes.
    #[serde(default = "default_memory_mb")]
    pub max_memory_mb: u64,
    /// Maximum CPU quota as a percentage.
    #[serde(default = "default_cpu_percent")]
    pub max_cpu_percent: u64,
    /// Maximum disk space in megabytes.
    #[serde(default = "default_disk_mb")]
    pub max_disk_mb: u64,
}

fn default_memory_mb() -> u64 {
    512
}

fn default_cpu_percent() -> u64 {
    100
}

fn default_disk_mb() -> u64 {
    1024
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: default_memory_mb(),
            max_cpu_percent: default_cpu_percent(),
            max_disk_mb: default_disk_mb(),
        }
    }
}

/// Skills and capability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillsConfig {
    /// Directories to search for skill definitions.
    #[serde(default)]
    pub skill_dirs: Vec<String>,
    /// Whether skills are enabled by default.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Maximum execution time for a skill in seconds.
    #[serde(default = "default_skill_timeout")]
    pub max_execution_time_secs: u64,
}

fn default_enabled() -> bool {
    true
}

fn default_skill_timeout() -> u64 {
    300
}

/// Scheduled task configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Whether the scheduler is enabled.
    pub enabled: bool,
    /// Check interval for scheduled tasks in seconds.
    #[serde(default = "default_check_interval")]
    pub check_interval_secs: u64,
    /// Maximum number of concurrent scheduled tasks.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_tasks: u32,
}

fn default_check_interval() -> u64 {
    60
}

fn default_max_concurrent() -> u32 {
    10
}

/// Observability and telemetry configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Whether observability is enabled.
    pub enabled: bool,
    /// Logging level (e.g., "debug", "info", "warn", "error").
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// Whether to enable OpenTelemetry tracing.
    #[serde(default)]
    pub tracing_enabled: bool,
    /// Whether to enable metrics collection.
    #[serde(default)]
    pub metrics_enabled: bool,
}

fn default_log_level() -> String {
    "info".to_string()
}