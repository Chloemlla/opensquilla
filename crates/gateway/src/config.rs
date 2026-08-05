//! Config CRUD RPC handlers.
//!
//! Provides RPC handlers for reading, updating, listing, patching, resetting,
//! validating, importing, and exporting configuration, plus model routing
//! control. The store is backed by [`opensquilla_core::config::Config`] for the
//! structured sections and a flat overlay for arbitrary key/value entries, and
//! wraps a [`ModelRouter`] for routing rules and per-session holds.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use opensquilla_core::error::AppError;
use parking_lot::{Mutex, RwLock};

use crate::model_routing::{ModelRouter, ModelRouterConfig, RouteRequest, RoutingRule};
use crate::rpc::{rpc_handler, RpcRegistry};

/// A config store combining a structured [`Config`], a flat key/value overlay,
/// and a model router.
#[derive(Clone)]
pub struct ConfigStore {
    entries: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    config: Arc<Mutex<Config>>,
    router: Arc<RwLock<ModelRouter>>,
}

impl Default for ConfigStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigStore {
    /// Create a new config store with the default configuration.
    pub fn new() -> Self {
        Self::from_config(Config::default())
    }

    /// Create a config store from an existing [`Config`].
    pub fn from_config(config: Config) -> Self {
        let router = ModelRouter::new(router_config_from_config(&config));
        Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(Mutex::new(config)),
            router: Arc::new(RwLock::new(router)),
        }
    }

    /// Load the configuration from the discovered path.
    pub fn load() -> Result<Self, AppError> {
        let config = Config::load().map_err(|e| AppError::internal(e.to_string()))?;
        Ok(Self::from_config(config))
    }

    // -----------------------------------------------------------------------
    // Flat key/value CRUD
    // -----------------------------------------------------------------------

    /// Get a configuration value by key. Overlay entries win over the
    /// structured config's flat view.
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        {
            let entries = self.entries.lock();
            if let Some(value) = entries.get(key) {
                return Some(value.clone());
            }
        }
        let config = self.config.lock();
        config.get(key).map(serde_json::Value::String)
    }

    /// Set a configuration value by key.
    pub fn set(&self, key: String, value: serde_json::Value) {
        self.entries.lock().insert(key, value);
        self.sync_router();
    }

    /// Delete a configuration entry by key.
    pub fn delete(&self, key: &str) -> bool {
        let removed = self.entries.lock().remove(key).is_some();
        {
            let mut config = self.config.lock();
            config.remove(key);
        }
        self.sync_router();
        removed
    }

    /// List all configuration entries as dotted keys, sorted.
    pub fn list(&self) -> Vec<(String, serde_json::Value)> {
        let mut result: HashMap<String, serde_json::Value> = HashMap::new();
        {
            let config = self.config.lock();
            for (k, v) in config.list() {
                result.insert(k, serde_json::Value::String(v));
            }
        }
        {
            let entries = self.entries.lock();
            for (k, v) in entries.iter() {
                result.insert(k.clone(), v.clone());
            }
        }
        let mut sorted: Vec<(String, serde_json::Value)> = result.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        sorted
    }

    /// Apply a patch (object of key → value) to the store.
    pub fn patch(&self, values: &serde_json::Map<String, serde_json::Value>) -> usize {
        let mut entries = self.entries.lock();
        for (k, v) in values {
            entries.insert(k.clone(), v.clone());
        }
        drop(entries);
        self.sync_router();
        values.len()
    }

    /// Reset the store to the default configuration.
    pub fn reset(&self) {
        self.entries.lock().clear();
        *self.config.lock() = Config::default();
        self.sync_router();
    }

    /// Return the full structured config serialized to JSON.
    pub fn get_all(&self) -> serde_json::Value {
        let config = self.config.lock();
        serde_json::to_value(&*config).unwrap_or(serde_json::Value::Null)
    }

    /// Return the flat overlay entries.
    pub fn overrides(&self) -> HashMap<String, serde_json::Value> {
        self.entries.lock().clone()
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate the configuration, returning a list of issues (empty = valid).
    pub fn validate(&self) -> Vec<String> {
        let config = self.config.lock();
        let mut issues = Vec::new();

        if config.providers.is_empty() {
            issues.push("No providers are configured".to_string());
        }

        for provider in &config.providers {
            if provider.models.is_empty() {
                issues.push(format!(
                    "Provider '{}' has no models configured",
                    provider.name
                ));
            }
            if provider.api_key.is_none() && provider.base_url.is_none() {
                issues.push(format!(
                    "Provider '{}' has neither an API key nor a base URL",
                    provider.name
                ));
            }
        }

        let mut names: HashSet<String> = HashSet::new();
        for provider in &config.providers {
            if !names.insert(provider.name.clone()) {
                issues.push(format!("Duplicate provider name '{}'", provider.name));
            }
        }

        if let Some(models) = &config.models {
            if let Some(default) = &models.default_model {
                let found = config
                    .providers
                    .iter()
                    .any(|p| p.models.iter().any(|m| m == default));
                if !found {
                    issues.push(format!(
                        "Default model '{default}' is not available from any configured provider"
                    ));
                }
            }
        }

        if config.gateway.port == 0 {
            issues.push("Gateway port must not be 0".to_string());
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Import / export / save
    // -----------------------------------------------------------------------

    /// Import configuration from a JSON or YAML string.
    ///
    /// Returns the number of top-level keys merged into the overlay.
    pub fn import(&self, format: &str, content: &str) -> Result<usize, AppError> {
        let value: serde_json::Value = match format.to_ascii_lowercase().as_str() {
            "json" => serde_json::from_str(content)
                .map_err(|e| AppError::bad_request(format!("Invalid JSON config: {e}")))?,
            "yaml" | "yml" => serde_yaml::from_str(content)
                .map_err(|e| AppError::bad_request(format!("Invalid YAML config: {e}")))?,
            other => {
                return Err(AppError::bad_request(format!(
                    "Unsupported config format '{other}' (expected json or yaml)"
                )));
            }
        };
        let object = value
            .as_object()
            .ok_or_else(|| AppError::bad_request("Config import must be a JSON object"))?;
        let count = object.len();
        let mut entries = self.entries.lock();
        for (k, v) in object {
            entries.insert(k.clone(), v.clone());
        }
        drop(entries);
        self.sync_router();
        Ok(count)
    }

    /// Export the configuration as a JSON or YAML string.
    ///
    /// The structured config is emitted with the flat overlay entries merged on
    /// top so imported overrides are reflected in the export.
    pub fn export(&self, format: &str) -> Result<String, AppError> {
        let mut root = self.get_all();
        if let Some(obj) = root.as_object_mut() {
            for (k, v) in self.entries.lock().iter() {
                obj.insert(k.clone(), v.clone());
            }
        }
        match format.to_ascii_lowercase().as_str() {
            "json" => serde_json::to_string_pretty(&root)
                .map_err(|e| AppError::internal(format!("JSON export failed: {e}"))),
            "yaml" | "yml" => serde_yaml::to_string(&root)
                .map_err(|e| AppError::internal(format!("YAML export failed: {e}"))),
            other => Err(AppError::bad_request(format!(
                "Unsupported config format '{other}' (expected json or yaml)"
            ))),
        }
    }

    /// Persist the configuration. When `path` is given it is written there;
    /// otherwise the discovered config path is used.
    pub fn save(&self, path: Option<PathBuf>) -> Result<(), AppError> {
        let config = self.config.lock();
        match path {
            Some(path) => config
                .save_to(&path)
                .map_err(|e| AppError::internal(e.to_string())),
            None => config
                .save()
                .map_err(|e| AppError::internal(e.to_string())),
        }
    }

    // -----------------------------------------------------------------------
    // Model routing
    // -----------------------------------------------------------------------

    /// Rebuild the router config from the current structured config.
    fn sync_router(&self) {
        let config = self.config.lock();
        let router_config = router_config_from_config(&config);
        self.router.write().set_config(router_config);
    }

    /// Return the current routing rules and defaults.
    pub fn routing_view(&self) -> serde_json::Value {
        let router = self.router.read();
        let rc = router.config();
        serde_json::json!({
            "default_model": rc.default_model,
            "fallback_chain": rc.fallback_chain,
            "rules": rc.rules,
        })
    }

    /// Add or replace a routing rule.
    pub fn set_routing_rule(&self, rule: RoutingRule) {
        let router = self.router.read();
        let mut rc = router.config();
        if let Some(existing) = rc.rules.iter_mut().find(|r| r.name == rule.name) {
            *existing = rule;
        } else {
            rc.rules.push(rule);
        }
        rc.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        router.set_config(rc);
    }

    /// Pin a session to a model until the given expiry.
    pub fn set_routing_override(
        &self,
        session_id: &str,
        model: &str,
        provider: Option<String>,
        reason: &str,
        expires_at: Option<DateTime<Utc>>,
    ) {
        let router = self.router.read();
        router.hold_session(session_id, model, provider, reason, expires_at);
    }

    /// Release a session's routing hold.
    pub fn release_routing_override(&self, session_id: &str) -> bool {
        let router = self.router.read();
        router.release_hold(session_id).is_some()
    }

    /// Record a routing decision for a request and return the outcome.
    pub fn route(&self, request: &RouteRequest) -> crate::model_routing::RouteOutcome {
        let router = self.router.read();
        router.route(request)
    }

    /// Return recent routing decisions.
    pub fn routing_decisions(&self, session_id: Option<&str>, limit: usize) -> Vec<crate::routing::RoutingDecision> {
        let router = self.router.read();
        router.decisions(session_id, limit)
    }
}

/// Map a structured [`Config`] onto a [`ModelRouterConfig`].
fn router_config_from_config(config: &Config) -> ModelRouterConfig {
    let default_model = config
        .models
        .as_ref()
        .and_then(|m| m.default_model.clone())
        .unwrap_or_else(|| "gpt-4o".to_string());
    let rules = config
        .models
        .as_ref()
        .map(|m| {
            m.routing_rules
                .iter()
                .map(|r| RoutingRule {
                    name: r.use_case.clone(),
                    session_pattern: None,
                    requested_model_contains: Some(r.use_case.clone()),
                    model: r.model.clone(),
                    provider: r.provider.clone(),
                    priority: 0,
                })
                .collect()
        })
        .unwrap_or_default();
    ModelRouterConfig {
        default_model,
        fallback_chain: Vec::new(),
        rules,
    }
}

/// Register config RPC handlers on the given registry.
pub fn register_config_handlers(registry: &mut RpcRegistry, config_store: ConfigStore) {
    let config_store = Arc::new(config_store);

    // config.get — retrieve a single config value
    registry.register(rpc_handler("config.get", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                match store.get(key) {
                    Some(value) => Ok(serde_json::json!({"key": key, "value": value})),
                    None => Err(AppError::not_found(format!("Config key '{key}' not found"))),
                }
            }
        }
    }));

    // config.set — set a configuration value
    registry.register(rpc_handler("config.set", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?
                    .to_string();
                let value = params
                    .get("value")
                    .ok_or_else(|| AppError::bad_request("Missing 'value' parameter"))?
                    .clone();
                store.set(key.clone(), value.clone());
                Ok(serde_json::json!({"key": key, "value": value, "status": "set"}))
            }
        }
    }));

    // config.delete — delete a configuration entry
    registry.register(rpc_handler("config.delete", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                if store.delete(key) {
                    Ok(serde_json::json!({"key": key, "deleted": true}))
                } else {
                    Err(AppError::not_found(format!("Config key '{key}' not found")))
                }
            }
        }
    }));

    // config.list — list all configuration entries
    registry.register(rpc_handler("config.list", {
        let store = config_store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let entries = store.list();
                let result: Vec<serde_json::Value> = entries
                    .into_iter()
                    .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
                    .collect();
                Ok(serde_json::json!({"entries": result, "count": result.len()}))
            }
        }
    }));

    // config.patch — apply multiple key/value pairs at once
    registry.register(rpc_handler("config.patch", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let values = params
                    .get("values")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| AppError::bad_request("Missing 'values' object"))?
                    .clone();
                let applied = store.patch(&values);
                Ok(serde_json::json!({"applied": applied}))
            }
        }
    }));

    // config.reset — restore the default configuration
    registry.register(rpc_handler("config.reset", {
        let store = config_store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                store.reset();
                Ok(serde_json::json!({"reset": true}))
            }
        }
    }));

    // config.get_all — full structured config as JSON
    registry.register(rpc_handler("config.get_all", {
        let store = config_store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                Ok(serde_json::json!({
                    "config": store.get_all(),
                    "overrides": store.overrides(),
                }))
            }
        }
    }));

    // config.validate — run schema/dependency checks
    registry.register(rpc_handler("config.validate", {
        let store = config_store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let issues = store.validate();
                Ok(serde_json::json!({
                    "valid": issues.is_empty(),
                    "issues": issues,
                    "issue_count": issues.len(),
                }))
            }
        }
    }));

    // config.import — import JSON/YAML configuration
    registry.register(rpc_handler("config.import", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let format = params
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("json");
                let content = params
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'content' parameter"))?;
                let imported = store.import(format, content)?;
                Ok(serde_json::json!({"imported": true, "keys": imported}))
            }
        }
    }));

    // config.export — export configuration as JSON/YAML
    registry.register(rpc_handler("config.export", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let format = params
                    .get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("json");
                let content = store.export(format)?;
                Ok(serde_json::json!({"format": format, "content": content}))
            }
        }
    }));

    // config.save — persist configuration to disk
    registry.register(rpc_handler("config.save", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let path = params.get("path").and_then(|v| v.as_str()).map(PathBuf::from);
                store.save(path)?;
                Ok(serde_json::json!({"saved": true}))
            }
        }
    }));

    // config.routing.list — current routing rules and defaults
    registry.register(rpc_handler("config.routing.list", {
        let store = config_store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                Ok(store.routing_view())
            }
        }
    }));

    // config.routing.set_rule — add or replace a routing rule
    registry.register(rpc_handler("config.routing.set_rule", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?
                    .to_string();
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?
                    .to_string();
                let rule = RoutingRule {
                    name,
                    session_pattern: params.get("session_pattern").and_then(|v| v.as_str()).map(String::from),
                    requested_model_contains: params
                        .get("requested_model_contains")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    model,
                    provider: params.get("provider").and_then(|v| v.as_str()).map(String::from),
                    priority: params.get("priority").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
                };
                store.set_routing_rule(rule.clone());
                Ok(serde_json::json!({"rule": rule}))
            }
        }
    }));

    // config.routing.override — pin a session to a model
    registry.register(rpc_handler("config.routing.override", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?;
                let provider = params.get("provider").and_then(|v| v.as_str()).map(String::from);
                let reason = params
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("manual override")
                    .to_string();
                let expires_at = params
                    .get("expires_at")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));
                store.set_routing_override(session_id, model, provider.clone(), &reason, expires_at);
                Ok(serde_json::json!({
                    "held": true,
                    "session_id": session_id,
                    "model": model,
                    "provider": provider,
                }))
            }
        }
    }));

    // config.routing.release — release a session's routing hold
    registry.register(rpc_handler("config.routing.release", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let released = store.release_routing_override(session_id);
                Ok(serde_json::json!({"session_id": session_id, "released": released}))
            }
        }
    }));

    // config.routing.decisions — recent routing decisions
    registry.register(rpc_handler("config.routing.decisions", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params.get("session_id").and_then(|v| v.as_str());
                let limit = params
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(20) as usize;
                let decisions = store.routing_decisions(session_id, limit);
                Ok(serde_json::json!({
                    "decisions": decisions,
                    "count": decisions.len(),
                }))
            }
        }
    }));

    // config.routing.route — make a routing decision for a request
    registry.register(rpc_handler("config.routing.route", {
        let store = config_store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let requested_model = params
                    .get("requested_model")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let purpose = params
                    .get("purpose")
                    .and_then(|v| v.as_str())
                    .unwrap_or("chat")
                    .to_string();
                let outcome = store.route(&RouteRequest {
                    session_id,
                    requested_model,
                    purpose,
                });
                Ok(serde_json::to_value(outcome)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_config_crud() {
        let store = ConfigStore::new();
        let mut registry = RpcRegistry::new();
        register_config_handlers(&mut registry, store.clone());

        // Set
        let params = serde_json::json!({"key": "theme", "value": "dark"});
        let result = registry.dispatch("config.set", params).await;
        assert!(result.unwrap().is_ok());

        // Get
        let params = serde_json::json!({"key": "theme"});
        let result = registry.dispatch("config.get", params).await;
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["value"], "dark");

        // List
        let result = registry.dispatch("config.list", serde_json::Value::Null).await;
        let resp = result.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() >= 1);

        // Patch
        let result = registry
            .dispatch(
                "config.patch",
                serde_json::json!({"values": {"a": 1, "b": true}}),
            )
            .await;
        let resp = result.unwrap().unwrap();
        assert_eq!(resp["applied"], 2);

        // Delete
        let params = serde_json::json!({"key": "theme"});
        let result = registry.dispatch("config.delete", params).await;
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_config_validate_default() {
        let store = ConfigStore::new();
        let mut registry = RpcRegistry::new();
        register_config_handlers(&mut registry, store);

        let r = registry.dispatch("config.validate", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        // Default config has no providers configured, so there will be issues.
        assert_eq!(resp["valid"], false);
        assert!(resp["issues"].is_array());
    }

    #[tokio::test]
    async fn test_config_import_export_roundtrip() {
        let store = ConfigStore::new();
        let mut registry = RpcRegistry::new();
        register_config_handlers(&mut registry, store.clone());

        // Import a JSON config with a provider.
        let content = serde_json::json!({
            "providers": [
                {
                    "name": "openai",
                    "provider_type": "openai",
                    "models": ["gpt-4o"],
                    "max_retries": 3,
                    "timeout_secs": 60,
                }
            ],
        })
        .to_string();
        let r = registry
            .dispatch(
                "config.import",
                serde_json::json!({"format": "json", "content": content}),
            )
            .await;
        assert!(r.unwrap().is_ok());

        // Export to YAML should succeed and reference the provider.
        let r = registry
            .dispatch("config.export", serde_json::json!({"format": "yaml"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["content"].as_str().unwrap().contains("openai"));
    }

    #[tokio::test]
    async fn test_config_routing_override_and_route() {
        let store = ConfigStore::new();
        let mut registry = RpcRegistry::new();
        register_config_handlers(&mut registry, store);

        // Route without a hold → default model.
        let r = registry
            .dispatch(
                "config.routing.route",
                serde_json::json!({"session_id": "s1", "purpose": "chat"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["selected_model"], "gpt-4o");
        assert_eq!(resp["strategy"], "default");

        // Pin s1 to claude.
        let r = registry
            .dispatch(
                "config.routing.override",
                serde_json::json!({"session_id": "s1", "model": "claude-sonnet-4", "provider": "anthropic"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["held"], true);

        // Route again → hold wins.
        let r = registry
            .dispatch(
                "config.routing.route",
                serde_json::json!({"session_id": "s1", "requested_model": "gpt-4o"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["selected_model"], "claude-sonnet-4");
        assert_eq!(resp["strategy"], "hold");

        // Decisions recorded.
        let r = registry
            .dispatch("config.routing.decisions", serde_json::json!({"session_id": "s1"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() >= 2);

        // Release the hold.
        let r = registry
            .dispatch(
                "config.routing.release",
                serde_json::json!({"session_id": "s1"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["released"], true);
    }

    #[tokio::test]
    async fn test_config_routing_set_rule() {
        let store = ConfigStore::new();
        let mut registry = RpcRegistry::new();
        register_config_handlers(&mut registry, store);

        let r = registry
            .dispatch(
                "config.routing.set_rule",
                serde_json::json!({
                    "name": "code",
                    "requested_model_contains": "code",
                    "model": "claude-opus-4",
                    "priority": 10,
                }),
            )
            .await;
        assert!(r.unwrap().is_ok());

        // Route with a request containing "code" → the rule matches.
        let r = registry
            .dispatch(
                "config.routing.route",
                serde_json::json!({"session_id": "s2", "requested_model": "code-model"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["selected_model"], "claude-opus-4");
        assert_eq!(resp["rule_used"], "code");
    }
}
