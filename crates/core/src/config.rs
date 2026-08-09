use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level configuration for the OpenSquilla gateway.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Operator-facing control UI preferences (e.g. channel message locale).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub control_ui: Option<ControlUiConfig>,
    /// Primary LLM provider configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm: Option<LlmConfig>,
    /// LLM provider profiles for non-primary providers, keyed by provider id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_profiles: Option<HashMap<String, LlmProfile>>,
    /// LLM ensemble configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_ensemble: Option<LlmEnsembleConfig>,
    /// Search provider name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_provider: Option<String>,
    /// Search API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_api_key: Option<String>,
    /// Search API key environment variable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_api_key_env: Option<String>,
    /// Maximum number of search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_max_results: Option<u32>,
    /// Search proxy URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_proxy: Option<String>,
    /// Whether to use the environment proxy for search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_use_env_proxy: Option<bool>,
    /// Search fallback policy ("off" or "network").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_fallback_policy: Option<String>,
    /// Whether search diagnostics are enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_diagnostics: Option<bool>,
    /// Image generation configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_generation: Option<ImageGenerationConfig>,
    /// Audio configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
    /// Memory configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryConfig>,
    /// Squilla router configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub squilla_router: Option<SquillaRouterConfig>,
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
        toml::from_str(&contents)
            .map_err(|e| crate::error::Error::Config(format!("Failed to parse config: {}", e)))
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
        serde_yaml::from_str(&contents)
            .map_err(|e| crate::error::Error::Config(format!("Failed to parse YAML config: {}", e)))
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
            let path = config_dir.join("opensquilla").join("opensquilla.toml");
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

        self.apply_value_map(&value_map)
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

        let _ = self.apply_value_map(&value_map);
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
            crate::error::Error::Config(format!("Failed to write config {}: {}", path.display(), e))
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
    fn apply_value_map(
        &mut self,
        value_map: &serde_json::Map<String, serde_json::Value>,
    ) -> crate::error::Result<()> {
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

/// Operator-facing control UI preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ControlUiConfig {
    /// The gateway-wide default locale for channel system messages.
    pub default_locale: String,
}

impl Default for ControlUiConfig {
    fn default() -> Self {
        Self {
            default_locale: "en".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Onboarding configuration sections (Python-to-Rust migration)
// ---------------------------------------------------------------------------

/// Primary LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Provider identifier (e.g. "tokenrhythm").
    #[serde(default = "default_tokenrhythm")]
    pub provider: String,
    /// Default model identifier.
    #[serde(default = "default_llm_model")]
    pub model: String,
    /// API key for authentication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable name for the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Base URL override for the API endpoint.
    #[serde(default = "default_llm_base_url")]
    pub base_url: String,
    /// HTTP proxy URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Maximum tokens for responses (0 = auto-resolve from model catalog).
    #[serde(default)]
    pub max_tokens: u32,
    /// Context window size in tokens (0 = auto-resolve).
    #[serde(default)]
    pub context_window_tokens: u32,
    /// Temperature for generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Top-p sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Thinking level override (off|minimal|low|medium|high|xhigh|adaptive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Provider request proof budget in characters (0 = derive from context budget).
    #[serde(default)]
    pub provider_request_proof_max_chars: u32,
    /// OpenRouter-only: map model id -> upstream provider name.
    #[serde(default)]
    pub provider_routing: HashMap<String, String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "tokenrhythm".to_string(),
            model: "deepseek-v4-pro".to_string(),
            api_key: None,
            api_key_env: None,
            base_url: "https://tokenrhythm.studio/v1".to_string(),
            proxy: None,
            max_tokens: 0,
            context_window_tokens: 0,
            temperature: None,
            top_p: None,
            thinking: None,
            provider_request_proof_max_chars: 0,
            provider_routing: HashMap::new(),
        }
    }
}

/// Credential profile for a non-primary or session-pinned LLM deployment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmProfile {
    /// Model override for this deployment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// API key for this profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable name for the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Rotation pool of environment variable names (never key values).
    #[serde(default)]
    pub api_key_env_pool: Vec<String>,
    /// Base URL override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// HTTP proxy URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
}

/// A single candidate in an LLM ensemble lineup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleCandidate {
    /// Provider identifier for this candidate.
    pub provider: String,
    /// Model identifier for this candidate.
    pub model: String,
    /// Source of the candidate ("custom" or "legacy_model_options").
    #[serde(default)]
    pub source: String,
    /// Whether this candidate is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Advisory role label (""|primary|contrast|fast_check|critic|aggregator).
    #[serde(default)]
    pub role: String,
    /// Per-candidate thinking level override.
    #[serde(default)]
    pub thinking_level: String,
}

impl Default for EnsembleCandidate {
    fn default() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            source: "custom".to_string(),
            enabled: true,
            role: String::new(),
            thinking_level: String::new(),
        }
    }
}

/// LLM ensemble configuration (b5_fusion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmEnsembleConfig {
    /// Whether the ensemble is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Ensemble mode (currently only "b5_fusion").
    #[serde(default)]
    pub mode: String,
    /// Selection mode for the ensemble lineup.
    #[serde(default)]
    pub selection_mode: String,
    /// Whether to expose tool schemas to proposers.
    #[serde(default)]
    pub proposer_tools: bool,
    /// Minimum number of successful proposers required.
    #[serde(default = "default_one_u32")]
    pub min_successful_proposers: u32,
    /// Policy when all proposers fail ("fallback_single" or "error").
    #[serde(default)]
    pub all_failed_policy: String,
    /// Legacy model options list.
    #[serde(default)]
    pub model_options: Vec<String>,
    /// Custom candidate lineup.
    #[serde(default)]
    pub candidates: Vec<EnsembleCandidate>,
    /// Maximum characters per candidate.
    #[serde(default = "default_24000")]
    pub candidate_max_chars: u32,
    /// Timeout for proposer phase in seconds.
    #[serde(default = "default_3600")]
    pub proposer_timeout_seconds: f64,
    /// Timeout for aggregator phase in seconds.
    #[serde(default = "default_3600")]
    pub aggregator_timeout_seconds: f64,
    /// Whether to shuffle candidates before selection.
    #[serde(default = "default_true")]
    pub shuffle_candidates: bool,
    /// Whether to record candidate outputs.
    #[serde(default)]
    pub record_candidates: bool,
}

impl Default for LlmEnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "b5_fusion".to_string(),
            selection_mode: "static_openrouter_b5".to_string(),
            proposer_tools: false,
            min_successful_proposers: 1,
            all_failed_policy: "fallback_single".to_string(),
            model_options: Vec::new(),
            candidates: Vec::new(),
            candidate_max_chars: 24_000,
            proposer_timeout_seconds: 3600.0,
            aggregator_timeout_seconds: 3600.0,
            shuffle_candidates: true,
            record_candidates: false,
        }
    }
}

/// Image generation provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImageGenProvider {
    /// Base URL for the provider API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// API key for authentication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable name for the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
}

/// Image generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationConfig {
    /// Whether image generation is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Binding mode ("custom" or "follow_llm").
    #[serde(default)]
    pub binding: String,
    /// Primary image generation model identifier.
    #[serde(default)]
    pub primary: String,
    /// Fallback model identifiers in order of preference.
    #[serde(default)]
    pub fallbacks: Vec<String>,
    /// Default image size (e.g. "1024x1024").
    #[serde(default)]
    pub size: String,
    /// Timeout in seconds for image generation requests.
    #[serde(default = "default_180")]
    pub timeout_seconds: f64,
    /// Output format (png, jpeg, webp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    /// Per-provider image generation configurations.
    #[serde(default)]
    pub providers: HashMap<String, ImageGenProvider>,
}

impl Default for ImageGenerationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            binding: "custom".to_string(),
            primary: "openai/gpt-image-1".to_string(),
            fallbacks: Vec::new(),
            size: "1024x1024".to_string(),
            timeout_seconds: 180.0,
            output_format: None,
            providers: HashMap::new(),
        }
    }
}

/// Audio TTS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTtsConfig {
    /// TTS model identifier.
    #[serde(default)]
    pub model: String,
    /// Voice identifier.
    #[serde(default)]
    pub voice: String,
    /// Language code (ISO 639-1).
    #[serde(default)]
    pub language_code: String,
    /// Output audio format.
    #[serde(default)]
    pub output_format: String,
    /// Timeout in seconds for TTS requests.
    #[serde(default = "default_120")]
    pub timeout_seconds: f64,
    /// Voice stability (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<f64>,
    /// Similarity boost (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_boost: Option<f64>,
    /// Style exaggeration (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub style: Option<f64>,
    /// Whether to use speaker boost.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
    /// Playback speed multiplier.
    #[serde(default = "default_1")]
    pub speed: f64,
}

impl Default for AudioTtsConfig {
    fn default() -> Self {
        Self {
            model: "eleven_multilingual_v2".to_string(),
            voice: "21m00Tcm4TlvDq8ikWAM".to_string(),
            language_code: String::new(),
            output_format: "mp3_44100_128".to_string(),
            timeout_seconds: 120.0,
            stability: None,
            similarity_boost: None,
            style: None,
            use_speaker_boost: None,
            speed: 1.0,
        }
    }
}

/// Audio provider configuration (elevenlabs).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioProvider {
    /// Base URL for the provider API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// API key for authentication.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable name for the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Speech-to-text model identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_to_text_model: Option<String>,
    /// Voice conversion model identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice_conversion_model: Option<String>,
    /// Music generation model identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub music_model: Option<String>,
    /// Music output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub music_output_format: Option<String>,
}

/// Audio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Whether audio is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// TTS configuration.
    #[serde(default)]
    pub tts: AudioTtsConfig,
    /// Per-provider audio configurations.
    #[serde(default)]
    pub providers: HashMap<String, AudioProvider>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tts: AudioTtsConfig::default(),
            providers: HashMap::new(),
        }
    }
}

/// Local memory embedding settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryEmbeddingLocalConfig {
    /// Path to the ONNX model directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_dir: Option<String>,
}

/// OpenAI-compatible remote memory embedding settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryEmbeddingRemoteConfig {
    /// API key for the remote embedding service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Environment variable name for the API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Base URL for the remote embedding service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// HTTP headers for the remote embedding service.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Model identifier for the remote embedding service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Embedding dimensions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// Ollama memory embedding settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryEmbeddingOllamaConfig {
    /// Base URL for the Ollama API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Model identifier for Ollama embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Memory embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEmbeddingConfig {
    /// Embedding provider ("auto"|"none"|"local"|"openai"|"openai-compatible"|"ollama").
    #[serde(default)]
    pub provider: String,
    /// Legacy mode field (overridden by provider when set).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Model override for embeddings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// API key for the embedding service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Base URL for the embedding service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Local embedding settings.
    #[serde(default)]
    pub local: MemoryEmbeddingLocalConfig,
    /// Remote embedding settings.
    #[serde(default)]
    pub remote: MemoryEmbeddingRemoteConfig,
    /// Ollama embedding settings.
    #[serde(default)]
    pub ollama: MemoryEmbeddingOllamaConfig,
}

impl Default for MemoryEmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "auto".to_string(),
            mode: None,
            model: None,
            api_key: None,
            base_url: None,
            local: MemoryEmbeddingLocalConfig::default(),
            remote: MemoryEmbeddingRemoteConfig::default(),
            ollama: MemoryEmbeddingOllamaConfig::default(),
        }
    }
}

/// Memory cost configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryCostConfig {
    /// Query embedding cache mode ("off"|"shadow"|"on").
    #[serde(default)]
    pub query_embedding_cache: String,
}

/// Memory configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory cost configuration.
    #[serde(default)]
    pub cost: MemoryCostConfig,
    /// Markdown memory source location ("state" or "workspace").
    #[serde(default)]
    pub source: String,
    /// Retrieval mode ("hybrid" or "fts_only").
    #[serde(default)]
    pub retrieval_mode: String,
    /// Embedding configuration.
    #[serde(default)]
    pub embedding: MemoryEmbeddingConfig,
    /// Background sync interval in minutes (0 = disabled).
    #[serde(default)]
    pub sync_interval_minutes: f64,
    /// Whether session source is enabled.
    #[serde(default)]
    pub session_source_enabled: bool,
    /// Maximum characters for passive memory injection into system prompt.
    #[serde(default = "default_4000")]
    pub inject_limit: u32,
    /// Maximum file size in KB (0 = disabled).
    #[serde(default = "default_1024")]
    pub max_file_size_kb: u32,
    /// Maximum total memory size in KB (0 = disabled).
    #[serde(default = "default_102400")]
    pub max_total_size_kb: u32,
    /// Maximum number of memory files.
    #[serde(default = "default_500")]
    pub max_files: u32,
    /// Entry TTL in days (0 = no auto-prune).
    #[serde(default)]
    pub entry_ttl_days: u32,
    /// Background TTL sweep interval in minutes.
    #[serde(default = "default_60")]
    pub ttl_sweep_interval_minutes: f64,
    /// Whether flush is enabled.
    #[serde(default)]
    pub flush_enabled: bool,
    /// Whether to pre-compact before flush.
    #[serde(default)]
    pub flush_pre_compaction: bool,
    /// Flush timeout in seconds.
    #[serde(default = "default_15")]
    pub flush_timeout_seconds: f64,
    /// Background flush timeout in seconds.
    #[serde(default = "default_120")]
    pub flush_background_timeout_seconds: f64,
    /// Initial backoff for flush retries in seconds.
    #[serde(default = "default_30")]
    pub flush_backoff_initial_seconds: f64,
    /// Maximum backoff for flush retries in seconds.
    #[serde(default = "default_300")]
    pub flush_backoff_max_seconds: f64,
    /// Maximum archive size in bytes for flush.
    #[serde(default = "default_800000")]
    pub flush_archive_max_bytes: u32,
    /// Whether compaction requires a safe receipt.
    #[serde(default)]
    pub flush_compaction_requires_safe_receipt: bool,
    /// Compaction safety mode ("protect"|"best_effort"|"block"|"off").
    #[serde(default)]
    pub flush_compaction_safety_mode: String,
    /// Whether repair is enabled.
    #[serde(default = "default_true")]
    pub repair_enabled: bool,
    /// Repair interval in seconds.
    #[serde(default = "default_60")]
    pub repair_interval_seconds: f64,
    /// Maximum items to repair per tick.
    #[serde(default = "default_5_u32")]
    pub repair_max_items_per_tick: u32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            cost: MemoryCostConfig::default(),
            source: "workspace".to_string(),
            retrieval_mode: "hybrid".to_string(),
            embedding: MemoryEmbeddingConfig::default(),
            sync_interval_minutes: 0.0,
            session_source_enabled: false,
            inject_limit: 4000,
            max_file_size_kb: 1024,
            max_total_size_kb: 102400,
            max_files: 500,
            entry_ttl_days: 0,
            ttl_sweep_interval_minutes: 60.0,
            flush_enabled: false,
            flush_pre_compaction: false,
            flush_timeout_seconds: 15.0,
            flush_background_timeout_seconds: 120.0,
            flush_backoff_initial_seconds: 30.0,
            flush_backoff_max_seconds: 300.0,
            flush_archive_max_bytes: 800_000,
            flush_compaction_requires_safe_receipt: false,
            flush_compaction_safety_mode: "protect".to_string(),
            repair_enabled: true,
            repair_interval_seconds: 60.0,
            repair_max_items_per_tick: 5,
        }
    }
}

/// Squilla router configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquillaRouterConfig {
    /// Whether the router is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Whether automatic thinking is enabled.
    #[serde(default = "default_true")]
    pub auto_thinking: bool,
    /// Rollout phase ("observe"|"prompt_only"|"full").
    #[serde(default)]
    pub rollout_phase: String,
    /// Routing strategy identifier.
    #[serde(default)]
    pub strategy: String,
    /// Tier profile identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_profile: Option<String>,
    /// Preset binding ("follow_primary"|"custom").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_binding: Option<String>,
    /// Visual mode for the router.
    #[serde(default)]
    pub visual_mode: String,
    /// Whether cross-provider tier execution is enabled.
    #[serde(default)]
    pub cross_provider_tiers: bool,
    /// Policy for tier-provider mismatch ("route"|"veto").
    #[serde(default)]
    pub tier_provider_mismatch: String,
    /// Default tier identifier.
    #[serde(default)]
    pub default_tier: String,
    /// Confidence threshold for routing decisions.
    #[serde(default = "default_05")]
    pub confidence_threshold: f64,
    /// Margin for high-confidence tier selection.
    #[serde(default = "default_005")]
    pub confidence_high_tier_margin: f64,
    /// Routing timeout in seconds.
    #[serde(default = "default_5")]
    pub routing_timeout_seconds: f64,
    /// Whether KV cache anti-downgrade is enabled.
    #[serde(default = "default_true")]
    pub kv_cache_anti_downgrade_enabled: bool,
    /// KV cache anti-downgrade window in seconds.
    #[serde(default = "default_600")]
    pub kv_cache_anti_downgrade_window_seconds: u32,
    /// Whether complaint upgrade is enabled.
    #[serde(default = "default_true")]
    pub complaint_upgrade_enabled: bool,
    /// Number of complaint upgrade steps.
    #[serde(default = "default_one_u32")]
    pub complaint_upgrade_steps: u32,
    /// Maximum characters for complaint upgrade.
    #[serde(default = "default_160")]
    pub complaint_upgrade_max_chars: u32,
    /// Whether router runtime is required.
    #[serde(default = "default_true")]
    pub require_router_runtime: bool,
    /// Decision record retention in days.
    #[serde(default = "default_30_u32")]
    pub decision_retention_days: u32,
    /// Whether on-device calibration is enabled.
    #[serde(default)]
    pub calibration_enabled: bool,
    /// Estimated output savings percentage.
    #[serde(default = "default_003")]
    pub estimated_output_savings_pct: f64,
    /// Whether upgrade to C3 compaction is enabled.
    #[serde(default = "default_true")]
    pub upgrade_to_c3_compaction_enabled: bool,
    /// Vision history lookback turns.
    #[serde(default = "default_8")]
    pub vision_history_lookback_turns: u32,
    /// Vision history candidate turns.
    #[serde(default = "default_8")]
    pub vision_history_candidate_turns: u32,
    /// Vision sticky follow-up turns.
    #[serde(default = "default_3")]
    pub vision_sticky_followup_turns: u32,
    /// Whether vision follow-up gate is enabled.
    #[serde(default = "default_true")]
    pub vision_followup_gate_enabled: bool,
    /// Vision follow-up gate tier.
    #[serde(default)]
    pub vision_followup_gate_tier: String,
    /// Vision follow-up gate model override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision_followup_gate_model: Option<String>,
    /// Vision follow-up gate timeout in seconds.
    #[serde(default = "default_10")]
    pub vision_followup_gate_timeout_seconds: f64,
    /// Vision follow-up gate max output tokens.
    #[serde(default = "default_512")]
    pub vision_followup_gate_max_output_tokens: u32,
    /// Vision follow-up gate fallback recent turns.
    #[serde(default = "default_2")]
    pub vision_followup_gate_fallback_recent_turns: u32,
    /// Vision follow-up gate unknown policy.
    #[serde(default)]
    pub vision_followup_gate_unknown_policy: String,
}

impl Default for SquillaRouterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_thinking: true,
            rollout_phase: "full".to_string(),
            strategy: "v4_phase3".to_string(),
            tier_profile: None,
            preset_binding: None,
            visual_mode: "real_candidates".to_string(),
            cross_provider_tiers: false,
            tier_provider_mismatch: "route".to_string(),
            default_tier: "c1".to_string(),
            confidence_threshold: 0.5,
            confidence_high_tier_margin: 0.05,
            routing_timeout_seconds: 5.0,
            kv_cache_anti_downgrade_enabled: true,
            kv_cache_anti_downgrade_window_seconds: 600,
            complaint_upgrade_enabled: true,
            complaint_upgrade_steps: 1,
            complaint_upgrade_max_chars: 160,
            require_router_runtime: true,
            decision_retention_days: 30,
            calibration_enabled: false,
            estimated_output_savings_pct: 0.03,
            upgrade_to_c3_compaction_enabled: true,
            vision_history_lookback_turns: 8,
            vision_history_candidate_turns: 8,
            vision_sticky_followup_turns: 3,
            vision_followup_gate_enabled: true,
            vision_followup_gate_tier: "c0".to_string(),
            vision_followup_gate_model: None,
            vision_followup_gate_timeout_seconds: 10.0,
            vision_followup_gate_max_output_tokens: 512,
            vision_followup_gate_fallback_recent_turns: 2,
            vision_followup_gate_unknown_policy: "image_if_recent".to_string(),
        }
    }
}

// Default-value helper functions for onboarding structs.
fn default_tokenrhythm() -> String {
    "tokenrhythm".to_string()
}

fn default_llm_model() -> String {
    "deepseek-v4-pro".to_string()
}

fn default_llm_base_url() -> String {
    "https://tokenrhythm.studio/v1".to_string()
}

fn default_true() -> bool { true }
fn default_one_u32() -> u32 { 1 }
fn default_24000() -> u32 { 24_000 }
fn default_3600() -> f64 { 3600.0 }
fn default_180() -> f64 { 180.0 }
fn default_120() -> f64 { 120.0 }
fn default_1() -> f64 { 1.0 }
fn default_4000() -> u32 { 4000 }
fn default_1024() -> u32 { 1024 }
fn default_102400() -> u32 { 102_400 }
fn default_500() -> u32 { 500 }
fn default_60() -> f64 { 60.0 }
fn default_15() -> f64 { 15.0 }
fn default_30() -> f64 { 30.0 }
fn default_300() -> f64 { 300.0 }
fn default_800000() -> u32 { 800_000 }
fn default_5() -> f64 { 5.0 }
fn default_5_u32() -> u32 { 5 }
fn default_05() -> f64 { 0.5 }
fn default_005() -> f64 { 0.05 }
fn default_003() -> f64 { 0.03 }
fn default_600() -> u32 { 600 }
fn default_160() -> u32 { 160 }
fn default_30_u32() -> u32 { 30 }
fn default_8() -> u32 { 8 }
fn default_3() -> u32 { 3 }
fn default_10() -> f64 { 10.0 }
fn default_512() -> u32 { 512 }
fn default_2() -> u32 { 2 }

impl Config {
    /// The resolved `control_ui.default_locale`, defaulting to `"en"`.
    pub fn default_locale(&self) -> &str {
        self.control_ui
            .as_ref()
            .map(|c| c.default_locale.as_str())
            .unwrap_or("en")
    }
}
