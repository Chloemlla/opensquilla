//! Secrets / key management RPC handlers.
//!
//! Provides `rpc_secrets` for API key management. Keys are held in memory
//! keyed by provider name; secret material is never logged and is masked in
//! all responses.

use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A stored secret descriptor. The actual value is kept separately and never
/// serialized into responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretRecord {
    pub key: String,
    pub provider: String,
    pub set_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub masked_preview: String,
}

/// In-memory secrets store. The secret values live in a separate map that is
/// never exposed via the API.
#[derive(Clone, Default)]
pub struct SecretsStore {
    records: Arc<Mutex<HashMap<String, SecretRecord>>>,
    values: Arc<Mutex<HashMap<String, String>>>,
}

impl SecretsStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a secret for a provider.
    pub fn set(&self, key: &str, provider: &str, value: &str) {
        let masked = mask(value);
        let record = SecretRecord {
            key: key.to_string(),
            provider: provider.to_string(),
            set_at: chrono::Utc::now(),
            last_used: None,
            masked_preview: masked,
        };
        self.records.lock().insert(key.to_string(), record);
        self.values
            .lock()
            .insert(key.to_string(), value.to_string());
    }

    /// Get the actual secret value (for internal use only, not via RPC).
    pub fn get_value(&self, key: &str) -> Option<String> {
        let value = self.values.lock().get(key).cloned();
        if value.is_some() {
            if let Some(record) = self.records.lock().get_mut(key) {
                record.last_used = Some(chrono::Utc::now());
            }
        }
        value
    }

    /// Get the masked record for a key.
    pub fn get_record(&self, key: &str) -> Option<SecretRecord> {
        self.records.lock().get(key).cloned()
    }

    /// Delete a secret.
    pub fn delete(&self, key: &str) -> bool {
        let removed = self.records.lock().remove(key).is_some();
        self.values.lock().remove(key);
        removed
    }

    /// List all secret records (masked).
    pub fn list(&self) -> Vec<SecretRecord> {
        let mut records: Vec<SecretRecord> = self.records.lock().values().cloned().collect();
        records.sort_by(|a, b| a.key.cmp(&b.key));
        records
    }

    /// Check if a secret exists.
    pub fn exists(&self, key: &str) -> bool {
        self.records.lock().contains_key(key)
    }
}

/// Mask a secret value, showing only the first and last characters.
fn mask(value: &str) -> String {
    let len = value.chars().count();
    if len <= 4 {
        return "*".repeat(len);
    }
    let first: String = value.chars().take(2).collect();
    let last: String = value
        .chars()
        .rev()
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{first}{}{last}", "*".repeat(len.saturating_sub(4)))
}

/// Register secrets RPC handlers on the given registry.
pub fn register_secrets_handlers(registry: &mut RpcRegistry, store: SecretsStore) {
    let store = Arc::new(store);

    // secrets.set — store a secret for a provider
    registry.register(rpc_handler("secrets.set", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                let provider = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let value = params
                    .get("value")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'value' parameter"))?;

                store.set(key, &provider, value);
                let record = store
                    .get_record(key)
                    .ok_or_else(|| AppError::internal("Failed to store secret"))?;
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // secrets.get — fetch the masked record for a key (never the value)
    registry.register(rpc_handler("secrets.get", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                match store.get_record(key) {
                    Some(record) => Ok(serde_json::to_value(record)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Secret '{key}' not found"))),
                }
            }
        }
    }));

    // secrets.list — list all secret records (masked)
    registry.register(rpc_handler("secrets.list", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let records = store.list();
                Ok(serde_json::json!({
                    "secrets": records,
                    "count": records.len(),
                }))
            }
        }
    }));

    // secrets.delete — delete a secret
    registry.register(rpc_handler("secrets.delete", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                if store.delete(key) {
                    Ok(serde_json::json!({"deleted": true, "key": key}))
                } else {
                    Err(AppError::not_found(format!("Secret '{key}' not found")))
                }
            }
        }
    }));

    // secrets.exists — check if a secret is set without revealing it
    registry.register(rpc_handler("secrets.exists", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let key = params
                    .get("key")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'key' parameter"))?;
                let exists = store.exists(key);
                Ok(serde_json::json!({"key": key, "exists": exists}))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_secrets_set_get_masked() {
        let store = SecretsStore::new();
        let mut registry = RpcRegistry::new();
        register_secrets_handlers(&mut registry, store);

        let params = serde_json::json!({
            "key": "openai_api_key",
            "provider": "openai",
            "value": "sk-abcdefgh1234567890",
        });
        let r = registry.dispatch("secrets.set", params).await;
        let resp = r.unwrap().unwrap();
        // The response should be masked, not the raw value.
        assert_ne!(resp["masked_preview"], "sk-abcdefgh1234567890");
        assert!(resp["masked_preview"].as_str().unwrap().contains("*"));

        let r = registry
            .dispatch("secrets.get", serde_json::json!({"key": "openai_api_key"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["provider"], "openai");
        assert!(resp["masked_preview"].as_str().unwrap().contains("*"));
    }

    #[tokio::test]
    async fn test_secrets_list_and_delete() {
        let store = SecretsStore::new();
        let mut registry = RpcRegistry::new();
        register_secrets_handlers(&mut registry, store);

        let _ = registry
            .dispatch(
                "secrets.set",
                serde_json::json!({"key": "k1", "value": "secret-value-1"}),
            )
            .await
            .unwrap();
        let _ = registry
            .dispatch(
                "secrets.set",
                serde_json::json!({"key": "k2", "value": "secret-value-2"}),
            )
            .await
            .unwrap();

        let r = registry
            .dispatch("secrets.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 2);

        let r = registry
            .dispatch("secrets.delete", serde_json::json!({"key": "k1"}))
            .await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("secrets.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);
    }

    #[tokio::test]
    async fn test_secrets_exists() {
        let store = SecretsStore::new();
        let mut registry = RpcRegistry::new();
        register_secrets_handlers(&mut registry, store);

        let r = registry
            .dispatch("secrets.exists", serde_json::json!({"key": "missing"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["exists"], false);

        let _ = registry
            .dispatch(
                "secrets.set",
                serde_json::json!({"key": "present", "value": "v"}),
            )
            .await
            .unwrap();
        let r = registry
            .dispatch("secrets.exists", serde_json::json!({"key": "present"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["exists"], true);
    }
}
