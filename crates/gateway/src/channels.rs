//! Channels RPC handlers.
//!
//! Provides `rpc_channels` for channel management CRUD, backed by the
//! channels crate's [`ChannelManager`] and its [`ChannelConfig`] type.

use opensquilla_channels::manager::ChannelManager;
use opensquilla_channels::types::{ChannelConfig, ChannelType, OutgoingMessage};
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::rpc::{RpcRegistry, rpc_handler};

/// Stored channel descriptor. We keep a parallel config registry because
/// `ChannelManager` owns the live handles and does not expose their configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRecord {
    pub channel_id: String,
    pub name: String,
    pub channel_type: String,
    pub enabled: bool,
    pub config: serde_json::Value,
}

/// A channels service wrapping a [`ChannelManager`] plus a config registry.
#[derive(Clone)]
pub struct ChannelsService {
    manager: Arc<ChannelManager>,
    records: Arc<Mutex<std::collections::HashMap<String, ChannelRecord>>>,
}

impl ChannelsService {
    /// Create a new channels service with an empty manager.
    pub fn new() -> Self {
        Self {
            manager: Arc::new(ChannelManager::new()),
            records: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Access the underlying channel manager.
    pub fn manager(&self) -> &ChannelManager {
        &self.manager
    }

    /// Upsert a channel record.
    pub fn upsert(&self, record: ChannelRecord) {
        self.records
            .lock()
            .insert(record.channel_id.clone(), record);
    }

    /// Get a channel record by id.
    pub fn get(&self, channel_id: &str) -> Option<ChannelRecord> {
        self.records.lock().get(channel_id).cloned()
    }

    /// Remove a channel record.
    pub fn remove(&self, channel_id: &str) -> Option<ChannelRecord> {
        self.records.lock().remove(channel_id)
    }

    /// List all channel records.
    pub fn list(&self) -> Vec<ChannelRecord> {
        let mut records: Vec<ChannelRecord> = self.records.lock().values().cloned().collect();
        records.sort_by(|a, b| a.channel_id.cmp(&b.channel_id));
        records
    }
}

impl Default for ChannelsService {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a [`ChannelType`] from a string.
fn parse_channel_type(s: &str) -> Result<ChannelType, AppError> {
    match s.to_ascii_lowercase().as_str() {
        "slack" => Ok(ChannelType::Slack),
        "discord" => Ok(ChannelType::Discord),
        "telegram" => Ok(ChannelType::Telegram),
        "feishu" => Ok(ChannelType::Feishu),
        "dingtalk" => Ok(ChannelType::DingTalk),
        "qq" => Ok(ChannelType::QQ),
        "wecom" => Ok(ChannelType::WeCom),
        "matrix" => Ok(ChannelType::Matrix),
        "msteams" => Ok(ChannelType::MSTeams),
        "terminal" => Ok(ChannelType::Terminal),
        "websocket" => Ok(ChannelType::WebSocket),
        other => Err(AppError::bad_request(format!(
            "Unknown channel type '{other}'"
        ))),
    }
}

/// Register channels RPC handlers on the given registry.
pub fn register_channels_handlers(registry: &mut RpcRegistry, service: ChannelsService) {
    let service = Arc::new(service);

    // channels.create — register a new channel record
    registry.register(rpc_handler("channels.create", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?
                    .to_string();
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&channel_id)
                    .to_string();
                let type_str = params
                    .get("channel_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_type' parameter"))?;
                let _channel_type = parse_channel_type(type_str)?;
                let enabled = params
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let config = params
                    .get("config")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                if service.get(&channel_id).is_some() {
                    return Err(AppError::bad_request(format!(
                        "Channel '{channel_id}' already exists"
                    )));
                }

                let record = ChannelRecord {
                    channel_id: channel_id.to_string(),
                    name,
                    channel_type: type_str.to_string(),
                    enabled,
                    config,
                };
                service.upsert(record.clone());
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // channels.get — fetch a channel record by id
    registry.register(rpc_handler("channels.get", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                match service.get(channel_id) {
                    Some(record) => Ok(serde_json::to_value(record)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!(
                        "Channel '{channel_id}' not found"
                    ))),
                }
            }
        }
    }));

    // channels.list — list all channel records
    registry.register(rpc_handler("channels.list", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let records = service.list();
                Ok(serde_json::json!({
                    "channels": records,
                    "count": records.len(),
                }))
            }
        }
    }));

    // channels.update — update a channel record's name/enabled/config
    registry.register(rpc_handler("channels.update", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                let mut record = service.get(channel_id).ok_or_else(|| {
                    AppError::not_found(format!("Channel '{channel_id}' not found"))
                })?;
                if let Some(name) = params.get("name").and_then(|v| v.as_str()) {
                    record.name = name.to_string();
                }
                if let Some(enabled) = params.get("enabled").and_then(|v| v.as_bool()) {
                    record.enabled = enabled;
                }
                if let Some(config) = params.get("config") {
                    record.config = config.clone();
                }
                if let Some(type_str) = params.get("channel_type").and_then(|v| v.as_str()) {
                    let _ = parse_channel_type(type_str)?;
                    record.channel_type = type_str.to_string();
                }
                service.upsert(record.clone());
                serde_json::to_value(record).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // channels.delete — remove a channel record and unregister the handle
    registry.register(rpc_handler("channels.delete", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                match service.remove(channel_id) {
                    Some(_) => {
                        service.manager().remove(channel_id);
                        Ok(serde_json::json!({"deleted": true, "channel_id": channel_id}))
                    }
                    None => Err(AppError::not_found(format!(
                        "Channel '{channel_id}' not found"
                    ))),
                }
            }
        }
    }));

    // channels.send — send a message to a registered channel
    registry.register(rpc_handler("channels.send", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                let text = params
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'text' parameter"))?;

                let record = service.get(channel_id).ok_or_else(|| {
                    AppError::not_found(format!("Channel '{channel_id}' not found"))
                })?;
                if !record.enabled {
                    return Err(AppError::bad_request(format!(
                        "Channel '{channel_id}' is disabled"
                    )));
                }

                let channel_type = parse_channel_type(&record.channel_type)?;
                let message =
                    OutgoingMessage::new(channel_id.to_string(), channel_type, text.to_string());
                service
                    .manager()
                    .send(channel_id, &message)
                    .map_err(AppError::bad_request)?;
                Ok(serde_json::json!({"sent": true, "channel_id": channel_id}))
            }
        }
    }));

    // channels.register_handle — initialize a channel from config and register it
    registry.register(rpc_handler("channels.register_handle", {
        let service = service.clone();
        move |params| {
            let service = service.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                let type_str = params
                    .get("channel_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_type' parameter"))?;
                let channel_type = parse_channel_type(type_str)?;
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(channel_id)
                    .to_string();
                let enabled = params
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let config_value = params
                    .get("config")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let cfg = ChannelConfig {
                    channel_type,
                    channel_id: channel_id.to_string(),
                    name: name.clone(),
                    enabled,
                    config: config_value.clone(),
                };

                // init_channel returns an error for unsupported types (feishu, etc.)
                match service.manager().init_channel(cfg) {
                    Ok(_handle) => {
                        let record = ChannelRecord {
                            channel_id: channel_id.to_string(),
                            name,
                            channel_type: type_str.to_string(),
                            enabled,
                            config: config_value,
                        };
                        service.upsert(record.clone());
                        // The handle is now owned and registered by the manager.
                        Ok(serde_json::to_value(record)
                            .map_err(|e| AppError::internal(e.to_string()))?)
                    }
                    Err(e) => Err(AppError::bad_request(format!(
                        "Failed to initialize channel: {e}"
                    ))),
                }
            }
        }
    }));

    // channels.list_handles — list currently registered live channel handles
    registry.register(rpc_handler("channels.list_handles", {
        let service = service.clone();
        move |_params| {
            let service = service.clone();
            async move {
                let handles: Vec<String> = service.manager().list_channels();
                Ok(serde_json::json!({
                    "handles": handles,
                    "count": handles.len(),
                }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channels_crud() {
        let service = ChannelsService::new();
        let mut registry = RpcRegistry::new();
        register_channels_handlers(&mut registry, service);

        let params = serde_json::json!({
            "channel_id": "ch-1",
            "name": "Test Channel",
            "channel_type": "terminal",
            "enabled": true,
        });
        let r = registry.dispatch("channels.create", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("channels.get", serde_json::json!({"channel_id": "ch-1"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["name"], "Test Channel");

        let r = registry
            .dispatch("channels.list", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["count"], 1);

        let r = registry
            .dispatch(
                "channels.update",
                serde_json::json!({"channel_id": "ch-1", "enabled": false}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["enabled"], false);

        let r = registry
            .dispatch("channels.delete", serde_json::json!({"channel_id": "ch-1"}))
            .await;
        assert!(r.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_channels_create_duplicate() {
        let service = ChannelsService::new();
        let mut registry = RpcRegistry::new();
        register_channels_handlers(&mut registry, service);

        let params = serde_json::json!({
            "channel_id": "ch-1",
            "channel_type": "terminal",
        });
        let _ = registry
            .dispatch("channels.create", params.clone())
            .await
            .unwrap();
        let r = registry.dispatch("channels.create", params).await;
        assert!(r.unwrap().is_err());
    }
}
