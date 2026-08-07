//! Channel manager — registry, lifecycle, outbox and tool channels.
//!
//! The [`ChannelManager`] is the boot-time orchestrator for all channel
//! adapters. It mirrors the Python `ChannelManager`:
//!
//! 1. **Registration**: [`ChannelManager::from_config`] constructs adapters
//!    from [`ChannelConfig`]s; [`ChannelManager::init_channel`] builds a
//!    single adapter and registers it.
//! 2. **Lifecycle**: every channel is wrapped in a [`ManagedChannel`] that
//!    can be started / stopped / restarted. Adapters with a background event
//!    loop (Socket Mode, Gateway, long polling, streams) carry start/stop
//!    hooks captured at construction time.
//! 3. **Outbox**: [`ChannelManager::install_outbox`] installs a
//!    [`DeliveryStore`]-backed worker so [`ChannelManager::send`] persists
//!    messages before dispatching (at-least-once delivery).
//! 4. **Tool channels**: [`ChannelManager::register_tool_channel`] exposes
//!    channels as callable message tools for the agent runtime.

use dashmap::{DashMap, DashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::delivery::{DeliveryStore, OutboxEntry, OutboxWorker};
use crate::dingtalk::DingTalkChannel;
use crate::discord::DiscordChannel;
use crate::feishu::FeishuChannel;
use crate::matrix::MatrixChannel;
use crate::msteams::MSTeamsChannel;
use crate::qq::QQChannel;
use crate::slack::SlackChannel;
use crate::telegram::TelegramChannel;
use crate::terminal::TerminalChannel;
use crate::types::{ChannelConfig, ChannelHandle, ChannelType, IncomingMessage, OutgoingMessage};
use crate::websocket::WebSocketChannel;
use crate::wecom::WeComChannel;

/// A start hook that launches a channel's background event loop.
pub type StartHook =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> + Send + Sync>;
/// A stop hook that shuts down a channel's background event loop.
pub type StopHook = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

/// The lifecycle state of a [`ManagedChannel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelState {
    Stopped,
    Starting,
    Running,
    Stopping,
    Failed,
}

/// A channel handle wrapped with lifecycle hooks.
#[derive(Clone)]
pub struct ManagedChannel {
    handle: ChannelHandle,
    state: Arc<tokio::sync::Mutex<ChannelState>>,
    start: Option<StartHook>,
    stop: Option<StopHook>,
}

impl ManagedChannel {
    /// Wrap a handle with no lifecycle hooks.
    pub fn new(handle: ChannelHandle) -> Self {
        Self {
            handle,
            state: Arc::new(tokio::sync::Mutex::new(ChannelState::Stopped)),
            start: None,
            stop: None,
        }
    }

    /// Wrap a handle with optional start/stop hooks.
    pub fn with_lifecycle(
        handle: ChannelHandle,
        start: Option<StartHook>,
        stop: Option<StopHook>,
    ) -> Self {
        Self {
            handle,
            state: Arc::new(tokio::sync::Mutex::new(ChannelState::Stopped)),
            start,
            stop,
        }
    }

    /// The underlying channel handle.
    pub fn handle(&self) -> &ChannelHandle {
        &self.handle
    }

    /// Whether this channel has a background lifecycle (start/stop hooks).
    pub fn has_lifecycle(&self) -> bool {
        self.start.is_some() || self.stop.is_some()
    }

    /// The current lifecycle state.
    pub async fn state(&self) -> ChannelState {
        *self.state.lock().await
    }

    /// Start the channel's background loop, if any.
    pub async fn start(&self) -> Result<(), String> {
        {
            let mut state = self.state.lock().await;
            if *state == ChannelState::Running || *state == ChannelState::Starting {
                return Ok(());
            }
            *state = ChannelState::Starting;
        }
        let result = match &self.start {
            Some(hook) => hook().await,
            None => Ok(()),
        };
        let mut state = self.state.lock().await;
        *state = match &result {
            Ok(()) => ChannelState::Running,
            Err(_) => ChannelState::Failed,
        };
        result
    }

    /// Stop the channel's background loop, if any.
    pub async fn stop(&self) {
        {
            let mut state = self.state.lock().await;
            if *state == ChannelState::Stopped || *state == ChannelState::Stopping {
                return;
            }
            *state = ChannelState::Stopping;
        }
        if let Some(hook) = &self.stop {
            hook().await;
        }
        *self.state.lock().await = ChannelState::Stopped;
    }

    /// Restart the channel (stop then start).
    pub async fn restart(&self) -> Result<(), String> {
        self.stop().await;
        self.start().await
    }
}

/// Registry of channel adapters with lifecycle management.
#[derive(Default)]
pub struct ChannelManager {
    channels: DashMap<String, ChannelHandle>,
    managed: DashMap<String, ManagedChannel>,
    message_handlers:
        DashMap<String, Arc<dyn Fn(IncomingMessage) -> Result<(), String> + Send + Sync>>,
    outbox: Arc<std::sync::Mutex<Option<Arc<DeliveryStore>>>>,
    outbox_worker: Arc<std::sync::Mutex<Option<Arc<OutboxWorker>>>>,
    tool_channels: DashSet<String>,
}

impl ChannelManager {
    /// Create an empty channel manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a manager from a list of channel configs, initializing and
    /// registering each enabled channel.
    pub fn from_config(configs: &[ChannelConfig]) -> Result<Self, String> {
        let manager = Self::new();
        for config in configs {
            if !config.enabled {
                info!("Skipping disabled channel {}", config.channel_id);
                continue;
            }
            manager.init_channel(config.clone())?;
        }
        Ok(manager)
    }

    /// Register an already-constructed channel handle.
    pub fn register(&self, channel: ChannelHandle) {
        let id = channel.channel_id().to_string();
        self.channels.insert(id.clone(), channel.clone());
        self.managed
            .insert(id.clone(), ManagedChannel::new(channel));
        info!("Registered channel {id}");
    }

    /// Register a message handler for a channel.
    pub fn on_message<F>(&self, channel_id: &str, handler: F)
    where
        F: Fn(IncomingMessage) -> Result<(), String> + Send + Sync + 'static,
    {
        self.message_handlers
            .insert(channel_id.to_string(), Arc::new(handler));
    }

    /// Initialize a channel from config and register it, returning the live
    /// handle.
    pub fn init_channel(&self, config: ChannelConfig) -> Result<ChannelHandle, String> {
        let (handle, start, stop) = Self::build_adapter(config)?;
        let id = handle.channel_id().to_string();
        let managed = ManagedChannel::with_lifecycle(handle.clone(), start, stop);
        self.channels.insert(id.clone(), handle.clone());
        self.managed.insert(id.clone(), managed);
        info!("Registered channel {id}");
        Ok(handle)
    }

    /// Construct a concrete adapter plus its lifecycle hooks.
    fn build_adapter(
        config: ChannelConfig,
    ) -> Result<(ChannelHandle, Option<StartHook>, Option<StopHook>), String> {
        match config.channel_type {
            ChannelType::Slack => {
                let ch = Arc::new(SlackChannel::new(config)?);
                let socket = ch.socket_mode_enabled();
                let handle: ChannelHandle = ch.clone();
                if !socket {
                    return Ok((handle, None, None));
                }
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_socket_mode().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_socket_mode().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::Discord => {
                let ch = Arc::new(DiscordChannel::new(config)?);
                let handle: ChannelHandle = ch.clone();
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_gateway().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_gateway().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::Telegram => {
                let use_webhook = config
                    .config
                    .get("use_webhook")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let ch = Arc::new(TelegramChannel::new(config)?);
                let handle: ChannelHandle = ch.clone();
                if use_webhook {
                    return Ok((handle, None, None));
                }
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_polling().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_polling().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::WeCom => {
                let ch = Arc::new(WeComChannel::new(config)?);
                let handle: ChannelHandle = ch;
                Ok((handle, None, None))
            }
            ChannelType::Feishu => {
                let ch = Arc::new(FeishuChannel::new(config)?);
                let handle: ChannelHandle = ch;
                Ok((handle, None, None))
            }
            ChannelType::DingTalk => {
                let ch = Arc::new(DingTalkChannel::new(config)?);
                let handle: ChannelHandle = ch.clone();
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_stream().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_stream().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::QQ => {
                let ch = Arc::new(QQChannel::new(config)?);
                let handle: ChannelHandle = ch.clone();
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_websocket().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_websocket().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::Matrix => {
                let ch = Arc::new(MatrixChannel::new(config)?);
                let handle: ChannelHandle = ch.clone();
                let (start_ch, stop_ch) = (ch.clone(), ch);
                let start: StartHook = Arc::new(move || {
                    let c = start_ch.clone();
                    Box::pin(async move { c.start_sync().await })
                });
                let stop: StopHook = Arc::new(move || {
                    let c = stop_ch.clone();
                    Box::pin(async move { c.stop_sync().await })
                });
                Ok((handle, Some(start), Some(stop)))
            }
            ChannelType::MSTeams => {
                let ch = Arc::new(MSTeamsChannel::new(config)?);
                let handle: ChannelHandle = ch;
                Ok((handle, None, None))
            }
            ChannelType::Terminal => {
                let ch = Arc::new(TerminalChannel::new(config)?);
                let handle: ChannelHandle = ch;
                Ok((handle, None, None))
            }
            ChannelType::WebSocket => {
                let ch = Arc::new(WebSocketChannel::new(config)?);
                let handle: ChannelHandle = ch;
                Ok((handle, None, None))
            }
            other => Err(format!("Unsupported channel type: {other:?}")),
        }
    }

    // -- lifecycle ----------------------------------------------------------

    /// Start a channel's background loop.
    pub async fn start_channel(&self, channel_id: &str) -> Result<(), String> {
        let managed = self
            .managed
            .get(channel_id)
            .map(|m| m.clone())
            .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
        managed.start().await
    }

    /// Stop a channel's background loop.
    pub async fn stop_channel(&self, channel_id: &str) -> Result<(), String> {
        let managed = self
            .managed
            .get(channel_id)
            .map(|m| m.clone())
            .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
        managed.stop().await;
        Ok(())
    }

    /// Restart a channel's background loop.
    pub async fn restart_channel(&self, channel_id: &str) -> Result<(), String> {
        let managed = self
            .managed
            .get(channel_id)
            .map(|m| m.clone())
            .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
        managed.restart().await
    }

    /// Start all channels with lifecycle hooks.
    pub async fn start_all(&self) -> Vec<(String, Result<(), String>)> {
        let mut results = Vec::new();
        for entry in self.managed.iter() {
            let id = entry.key().clone();
            let managed = entry.value().clone();
            if managed.has_lifecycle() {
                let result = managed.start().await;
                results.push((id, result));
            }
        }
        results
    }

    /// Stop all channels with lifecycle hooks.
    pub async fn stop_all(&self) {
        for entry in self.managed.iter() {
            let managed = entry.value().clone();
            if managed.has_lifecycle() {
                managed.stop().await;
            }
        }
    }

    /// The current lifecycle state of a channel.
    pub async fn channel_state(&self, channel_id: &str) -> Option<ChannelState> {
        let managed = self.managed.get(channel_id).map(|m| m.clone())?;
        Some(managed.state().await)
    }

    /// The ids of all channels that have a background lifecycle.
    pub fn lifecycle_channels(&self) -> Vec<String> {
        self.managed
            .iter()
            .filter(|e| e.value().has_lifecycle())
            .map(|e| e.key().clone())
            .collect()
    }

    // -- outbox -------------------------------------------------------------

    /// Install a delivery store and start the outbox worker. Once installed,
    /// [`ChannelManager::send`] persists messages before dispatching.
    pub fn install_outbox(
        &self,
        store: Arc<DeliveryStore>,
        _poll_interval: Duration,
    ) -> Result<(), String> {
        let channels = self.channels.clone();
        let send =
            move |entry: &OutboxEntry| -> Pin<Box<dyn Future<Output = Result<(), String>> + Send>> {
                let channels = channels.clone();
                let channel_id = entry.channel_id.clone();
                // Extract owned data before the async block so the returned future
                // is 'static (the outbox worker requires a Send + 'static send fn).
                let msg = match entry.to_outgoing() {
                    Some(msg) => msg,
                    None => {
                        return Box::pin(async move {
                            Err(format!("Invalid outbox payload for {channel_id}"))
                        });
                    }
                };
                Box::pin(async move {
                    let channel = channels
                        .get(&channel_id)
                        .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
                    let channel = channel.clone();
                    channel.send_message(&msg).await
                })
            };
        let worker = OutboxWorker::new(
            store.clone(),
            send,
            format!("manager-{}", store.instance_id()),
        );
        worker.start();
        *self.outbox.lock().unwrap() = Some(store);
        *self.outbox_worker.lock().unwrap() = Some(Arc::new(worker));
        info!("Outbox installed; persistent delivery enabled");
        Ok(())
    }

    /// Whether a delivery outbox is installed.
    pub fn has_outbox(&self) -> bool {
        self.outbox.lock().unwrap().is_some()
    }

    /// The installed delivery store, if any.
    pub fn outbox_store(&self) -> Option<Arc<DeliveryStore>> {
        self.outbox.lock().unwrap().clone()
    }

    /// Stop the outbox worker (if running).
    pub async fn stop_outbox(&self) {
        if let Some(worker) = self.outbox_worker.lock().unwrap().clone() {
            worker.stop().await;
        }
    }

    // -- tool channels ------------------------------------------------------

    /// Register a channel as a callable message tool.
    pub fn register_tool_channel(&self, channel_id: &str) -> bool {
        if self.channels.contains_key(channel_id) {
            self.tool_channels.insert(channel_id.to_string());
            info!("Registered tool channel {channel_id}");
            true
        } else {
            warn!("Cannot register tool channel: {channel_id} not found");
            false
        }
    }

    /// Remove a channel from the tool registry.
    pub fn unregister_tool_channel(&self, channel_id: &str) {
        self.tool_channels.remove(channel_id);
    }

    /// List the ids of all tool-registered channels.
    pub fn tool_channels(&self) -> Vec<String> {
        self.tool_channels.iter().map(|e| e.clone()).collect()
    }

    /// Send a message through a tool channel.
    pub fn send_as_tool(&self, channel_id: &str, text: &str) -> Result<(), String> {
        if !self.tool_channels.contains(channel_id) {
            return Err(format!(
                "Channel {channel_id} is not registered as a tool channel"
            ));
        }
        let channel = self
            .channels
            .get(channel_id)
            .map(|e| e.clone())
            .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
        let msg = OutgoingMessage::new(
            channel_id.to_string(),
            channel.channel_type(),
            text.to_string(),
        );
        let cid = channel_id.to_string();
        tokio::spawn(async move {
            if let Err(e) = channel.send_message(&msg).await {
                warn!("Tool send failed to {cid}: {e}");
            }
        });
        Ok(())
    }

    // -- sending ------------------------------------------------------------

    /// Send a message to a channel.
    ///
    /// When an outbox is installed, the message is persisted and delivered by
    /// the outbox worker (at-least-once). Otherwise it is dispatched
    /// immediately in a background task.
    pub fn send(&self, channel_id: &str, message: &OutgoingMessage) -> Result<(), String> {
        if let Some(outbox) = self.outbox.lock().unwrap().clone() {
            let channel_id = channel_id.to_string();
            let msg = message.clone();
            tokio::spawn(async move {
                if let Err(e) = outbox.enqueue_outgoing(&channel_id, &msg, None).await {
                    warn!("Outbox enqueue failed for {channel_id}: {e}");
                }
            });
            return Ok(());
        }
        self.send_immediate(channel_id, message)
    }

    /// Send a message immediately, bypassing any installed outbox.
    pub fn send_immediate(
        &self,
        channel_id: &str,
        message: &OutgoingMessage,
    ) -> Result<(), String> {
        let channel = self
            .channels
            .get(channel_id)
            .map(|e| e.clone())
            .ok_or_else(|| format!("Channel not found: {channel_id}"))?;
        let msg = message.clone();
        let cid = channel_id.to_string();
        tokio::spawn(async move {
            if let Err(e) = channel.send_message(&msg).await {
                warn!("Failed to send message to {cid}: {e}");
            }
        });
        Ok(())
    }

    /// Send a message to all registered channels.
    pub fn broadcast(&self, message: &OutgoingMessage) {
        for entry in self.channels.iter() {
            let channel = entry.value().clone();
            let msg = message.clone();
            let channel_id = entry.key().clone();
            tokio::spawn(async move {
                if let Err(e) = channel.send_message(&msg).await {
                    warn!("Broadcast failed for {channel_id}: {e}");
                }
            });
        }
    }

    /// Handle an incoming message by dispatching to the registered handler.
    pub fn handle_incoming(&self, message: IncomingMessage) -> Result<(), String> {
        if let Some(handler) = self.message_handlers.get(&message.channel_id) {
            handler(message)
        } else {
            warn!(
                "No handler for incoming message on channel {}",
                message.channel_id
            );
            Ok(())
        }
    }

    // -- registry queries ---------------------------------------------------

    /// Get a channel handle by id.
    pub fn get(&self, channel_id: &str) -> Option<ChannelHandle> {
        self.channels.get(channel_id).map(|e| e.clone())
    }

    /// Whether a channel is registered.
    pub fn has_channel(&self, channel_id: &str) -> bool {
        self.channels.contains_key(channel_id)
    }

    /// List all registered channel ids.
    pub fn list_channels(&self) -> Vec<String> {
        self.channels.iter().map(|e| e.key().clone()).collect()
    }

    /// Number of registered channels.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Whether the manager is empty.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Remove a channel and all its registrations.
    pub fn remove(&self, channel_id: &str) {
        self.channels.remove(channel_id);
        self.managed.remove(channel_id);
        self.message_handlers.remove(channel_id);
        self.tool_channels.remove(channel_id);
        info!("Removed channel {channel_id}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn slack_config() -> ChannelConfig {
        ChannelConfig {
            channel_type: ChannelType::Slack,
            channel_id: "slack-c1".to_string(),
            name: "Slack Test".to_string(),
            enabled: true,
            config: json!({ "bot_token": "xoxb-test" }),
        }
    }

    fn terminal_config() -> ChannelConfig {
        ChannelConfig {
            channel_type: ChannelType::Terminal,
            channel_id: "term-c1".to_string(),
            name: "Terminal".to_string(),
            enabled: true,
            config: json!({ "color": false }),
        }
    }

    #[test]
    fn test_init_and_query() {
        let manager = ChannelManager::new();
        let handle = manager.init_channel(slack_config()).unwrap();
        assert_eq!(handle.channel_id(), "slack-c1");
        assert!(manager.has_channel("slack-c1"));
        assert_eq!(manager.len(), 1);
        assert!(manager.get("slack-c1").is_some());
    }

    #[test]
    fn test_from_config_respects_enabled() {
        let mut disabled = terminal_config();
        disabled.enabled = false;
        let manager = ChannelManager::from_config(&[slack_config(), disabled]).unwrap();
        assert_eq!(manager.len(), 1);
        assert!(manager.has_channel("slack-c1"));
        assert!(!manager.has_channel("term-c1"));
    }

    #[test]
    fn test_init_unsupported_type() {
        let manager = ChannelManager::new();
        let err = manager.init_channel(ChannelConfig {
            channel_type: ChannelType::Custom("nope".to_string()),
            channel_id: "x".to_string(),
            name: "x".to_string(),
            enabled: true,
            config: json!({}),
        });
        assert!(err.is_err());
    }

    #[test]
    fn test_register_remove() {
        let manager = ChannelManager::new();
        let handle = manager.init_channel(slack_config()).unwrap();
        manager.remove("slack-c1");
        assert!(!manager.has_channel("slack-c1"));
        assert_eq!(manager.len(), 0);
        let _ = handle;
    }

    #[tokio::test]
    async fn test_lifecycle_slack_no_hooks() {
        let manager = ChannelManager::new();
        manager.init_channel(slack_config()).unwrap();
        let state = manager.channel_state("slack-c1").await.unwrap();
        assert_eq!(state, ChannelState::Stopped);
        // No socket mode → start is a no-op.
        manager.start_channel("slack-c1").await.unwrap();
        let state = manager.channel_state("slack-c1").await.unwrap();
        assert_eq!(state, ChannelState::Running);
        manager.stop_channel("slack-c1").await.unwrap();
    }

    #[tokio::test]
    async fn test_start_all() {
        let manager = ChannelManager::new();
        manager.init_channel(slack_config()).unwrap();
        let results = manager.start_all().await;
        assert_eq!(results.len(), 0); // no lifecycle hooks
        assert!(manager.lifecycle_channels().is_empty());
    }

    #[test]
    fn test_tool_channels() {
        let manager = ChannelManager::new();
        manager.init_channel(slack_config()).unwrap();
        assert!(!manager.register_tool_channel("missing"));
        assert!(manager.register_tool_channel("slack-c1"));
        assert_eq!(manager.tool_channels(), vec!["slack-c1".to_string()]);
        manager.unregister_tool_channel("slack-c1");
        assert!(manager.tool_channels().is_empty());
    }

    #[tokio::test]
    async fn test_send_fires_and_forget() {
        let manager = ChannelManager::new();
        manager.init_channel(slack_config()).unwrap();
        let msg =
            OutgoingMessage::new("slack-c1".to_string(), ChannelType::Slack, "hi".to_string());
        manager.send("slack-c1", &msg).unwrap();
        assert!(manager.send("missing", &msg).is_err());
    }

    #[tokio::test]
    async fn test_outbox_install() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("manager_outbox_{}.sqlite", uuid::Uuid::new_v4()));
        let store = Arc::new(DeliveryStore::open(&path, "manager-test".to_string()).unwrap());
        let manager = ChannelManager::new();
        manager.init_channel(terminal_config()).unwrap();
        manager
            .install_outbox(store.clone(), Duration::from_millis(10))
            .unwrap();
        assert!(manager.has_outbox());
        assert!(manager.outbox_store().is_some());
        manager.stop_outbox().await;
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_message_handlers() {
        let manager = ChannelManager::new();
        manager.init_channel(slack_config()).unwrap();
        let called = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = called.clone();
        manager.on_message("slack-c1", move |_msg| {
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        });
        let msg = IncomingMessage {
            id: uuid::Uuid::new_v4(),
            channel_id: "slack-c1".to_string(),
            channel_type: ChannelType::Slack,
            user_id: "u".to_string(),
            user_name: None,
            text: "hi".to_string(),
            thread_id: None,
            attachments: Vec::new(),
            timestamp: chrono::Utc::now(),
            raw: json!({}),
        };
        manager.handle_incoming(msg).unwrap();
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    }
}
