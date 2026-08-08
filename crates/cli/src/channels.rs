//! Channel management commands.
//!
//! Implements the `channels` subcommand. Channels are declared in the
//! configuration; each command initializes a [`ChannelManager`] and reflects the
//! configured channels, their runtime status, and a live connectivity test.

use anyhow::{Context, Result};
use opensquilla_channels::manager::ChannelManager;
use opensquilla_channels::types::{
    ChannelConfig as CrateChannelConfig, ChannelHandle, ChannelType,
};
use opensquilla_core::config::Config;
use tracing::info;

/// List configured channels with their runtime status.
pub async fn list_channels() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();

    if config.channels.is_empty() {
        println!("No channels configured.");
        println!("Add a [[channels]] section to your config to enable one.");
        return Ok(());
    }

    println!("Channels:");
    println!("{:-<70}", "");
    println!(
        "{:<20} {:<18} {:<10} Runtime",
        "Name", "Type", "Enabled"
    );
    println!("{:-<70}", "");
    for cfg in &config.channels {
        let runtime = match init_channel(&manager, cfg) {
            Ok(_) => "initialized",
            Err(_) => "error",
        };
        println!(
            "{:<20} {:<18} {:<10} {}",
            cfg.name,
            cfg.channel_type,
            if cfg.enabled { "yes" } else { "no" },
            runtime
        );
    }
    Ok(())
}

/// Show the status of a single configured channel.
pub async fn show_channel_status(name: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let channel_name = name.unwrap_or_else(|| {
        config
            .channels
            .first()
            .map(|c| c.name.clone())
            .unwrap_or_default()
    });

    let cfg = config
        .find_channel(&channel_name)
        .ok_or_else(|| anyhow::anyhow!("Channel '{channel_name}' is not configured"))?;

    println!("Channel:  {}", cfg.name);
    println!("Type:     {}", cfg.channel_type);
    println!("Enabled:  {}", cfg.enabled);
    match init_channel(&manager, cfg) {
        Ok(handle) => {
            println!("Status:   connected");
            println!("Channel id: {}", handle.channel_id());
        }
        Err(e) => {
            println!("Status:   error");
            println!("Error:    {e}");
        }
    }
    Ok(())
}

/// Test a configured channel by initializing it and reporting connectivity.
pub async fn test_channel(name: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let cfg = config
        .find_channel(&name)
        .ok_or_else(|| anyhow::anyhow!("Channel '{name}' is not configured"))?;

    println!("Testing channel '{name}'...");
    match init_channel(&manager, cfg) {
        Ok(handle) => {
            println!(
                "OK: channel '{}' ({:?}) initialized",
                handle.name(),
                handle.channel_type()
            );
            info!("Channel {name} test passed");
            Ok(())
        }
        Err(e) => anyhow::bail!("Channel '{name}' test failed: {e}"),
    }
}

/// Initialize a configured channel and register it with the manager.
pub async fn connect_channel(kind: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let cfg = config
        .find_channel(&kind)
        .ok_or_else(|| anyhow::anyhow!("Channel '{kind}' is not configured"))?;
    let handle = init_channel(&manager, cfg)?;
    println!(
        "Connected channel: {} ({:?})",
        handle.name(),
        handle.channel_type()
    );
    Ok(())
}

/// Remove a channel from the runtime manager.
pub async fn disconnect_channel(id: String) -> Result<()> {
    let manager = ChannelManager::new();
    manager.remove(&id);
    println!("Disconnected channel: {id}");
    Ok(())
}

/// Send a test message to a configured channel.
pub async fn send_message(name: String, message: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let cfg = config
        .find_channel(&name)
        .ok_or_else(|| anyhow::anyhow!("Channel '{name}' is not configured"))?;
    let handle = init_channel(&manager, cfg)?;

    let outgoing = opensquilla_channels::types::OutgoingMessage::new(
        handle.channel_id().to_string(),
        handle.channel_type(),
        message.clone(),
    );
    handle
        .send_message(&outgoing)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to send message: {e}"))?;
    println!("{} Message sent to '{name}': {message}", crate::table::ok());
    info!("Message sent to channel {name}");
    Ok(())
}

/// Start all enabled channels.
pub async fn start_all_channels() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let mut started = 0u32;
    let mut failed = 0u32;

    for cfg in &config.channels {
        if !cfg.enabled {
            continue;
        }
        match init_channel(&manager, cfg) {
            Ok(handle) => {
                println!(
                    "{} Started channel: {} ({:?})",
                    crate::table::ok(),
                    handle.name(),
                    handle.channel_type()
                );
                started += 1;
            }
            Err(e) => {
                println!(
                    "{} Failed to start channel {}: {e}",
                    crate::table::fail(),
                    cfg.name
                );
                failed += 1;
            }
        }
    }

    println!();
    println!("Started {started} channel(s), {failed} failed.");
    Ok(())
}

/// Stop all channels.
pub async fn stop_all_channels() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = ChannelManager::new();
    let mut stopped = 0u32;

    for cfg in &config.channels {
        if init_channel(&manager, cfg).is_ok() {
            stopped += 1;
        }
    }

    println!("Stopped {stopped} channel(s).");
    Ok(())
}

/// Initialize a channel from its config declaration.
fn init_channel(
    manager: &ChannelManager,
    cfg: &opensquilla_core::config::ChannelConfig,
) -> Result<ChannelHandle> {
    let channel_type = parse_channel_type(&cfg.channel_type)?;
    let config = CrateChannelConfig {
        channel_type,
        channel_id: cfg.name.clone(),
        name: cfg.name.clone(),
        enabled: cfg.enabled,
        config: serde_json::Value::Object(
            cfg.config
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                .collect(),
        ),
    };
    manager
        .init_channel(config)
        .map_err(|e| anyhow::anyhow!("Failed to initialize channel '{}': {e}", cfg.name))
}

/// Map a config type string onto the channels crate's `ChannelType`.
fn parse_channel_type(kind: &str) -> Result<ChannelType> {
    Ok(match kind.to_lowercase().as_str() {
        "slack" => ChannelType::Slack,
        "discord" => ChannelType::Discord,
        "telegram" => ChannelType::Telegram,
        "feishu" | "lark" => ChannelType::Feishu,
        "dingtalk" => ChannelType::DingTalk,
        "qq" => ChannelType::QQ,
        "wecom" | "wechat_work" => ChannelType::WeCom,
        "matrix" => ChannelType::Matrix,
        "msteams" | "teams" => ChannelType::MSTeams,
        "terminal" => ChannelType::Terminal,
        "websocket" | "ws" => ChannelType::WebSocket,
        _ => ChannelType::Custom(kind.to_string()),
    })
}
