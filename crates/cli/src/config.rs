//! Configuration management commands.
//!
//! Implements the `config` subcommand. All operations are direct calls into the
//! [`opensquilla_core::config::Config`] type (Mode B): values are read, written
//! and removed via dotted-path keys, and the whole configuration can be
//! imported from or exported to a file for migration between environments.

use std::path::Path;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use tracing::info;

/// Read a single configuration value by dotted-path key.
pub async fn get_config(key: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    match config.get(&key) {
        Some(value) => {
            println!("{key} = {value}");
            Ok(())
        }
        None => anyhow::bail!("Config key '{key}' not found"),
    }
}

/// Write a single configuration value by dotted-path key.
pub async fn set_config(key: String, value: String) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;
    config
        .set(&key, &value)
        .map_err(|e| anyhow::anyhow!("Failed to set '{key}': {e}"))?;
    config.save().context("Failed to save configuration")?;
    info!("Config {key} set to {value}");
    println!("Set {key} = {value}");
    Ok(())
}

/// Remove a configuration key and save.
pub async fn unset_config(key: String) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;
    config.remove(&key);
    config.save().context("Failed to save configuration")?;
    info!("Config key {key} removed");
    println!("Removed {key}");
    Ok(())
}

/// Open the configuration file in the user's default editor, then validate it.
pub async fn edit_config() -> Result<()> {
    let config_path = Config::discover_path().context("No config file found")?;
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| {
            #[cfg(windows)]
            {
                "notepad".to_string()
            }
            #[cfg(not(windows))]
            {
                "nano".to_string()
            }
        });

    info!(
        "Opening config at {} with editor {}",
        config_path.display(),
        editor
    );

    let status = std::process::Command::new(&editor)
        .arg(&config_path)
        .status()
        .context("Failed to launch editor")?;

    if !status.success() {
        anyhow::bail!("Editor exited with error");
    }

    // Reload the config to surface any TOML errors the edit may have caused.
    Config::load().context("Config file contains errors after edit")?;
    println!("Config updated successfully.");
    Ok(())
}

/// Print every configuration value as a flat dotted-path table.
pub async fn list_config() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let entries = config.list();
    if entries.is_empty() {
        println!("No configuration entries found.");
        return Ok(());
    }
    println!("Configuration:");
    println!("{:-<40}", "");
    for (key, value) in entries {
        println!("{key:20} = {value}");
    }
    println!("{:-<40}", "");
    Ok(())
}

/// Print the resolved configuration file path.
pub async fn show_config_path() -> Result<()> {
    let path = Config::discover_path().context("No config file found")?;
    println!("{}", path.display());
    Ok(())
}

/// Import a configuration from a TOML, YAML, or JSON file and save it to the
/// discovered config location.
pub async fn import_config(path: String) -> Result<()> {
    let ext = Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("toml")
        .to_lowercase();

    let config = match ext.as_str() {
        "yaml" | "yml" => Config::from_yaml_file(&path)
            .with_context(|| format!("Failed to parse YAML config at {path}"))?,
        "json" => {
            let contents =
                std::fs::read_to_string(&path).with_context(|| format!("Failed to read {path}"))?;
            serde_json::from_str::<Config>(&contents)
                .with_context(|| format!("Failed to parse JSON config at {path}"))?
        }
        _ => {
            Config::from_file(&path).with_context(|| format!("Failed to parse config at {path}"))?
        }
    };

    config.save().context("Failed to write config")?;
    let dest = Config::discover_path().context("No destination config path")?;
    info!("Imported config from {path} to {}", dest.display());
    println!("Imported configuration from {path} to {}", dest.display());
    Ok(())
}

/// Export the current configuration to a TOML or JSON file.
pub async fn export_config(path: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let ext = Path::new(&path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("toml")
        .to_lowercase();

    let contents = match ext.as_str() {
        "json" => {
            serde_json::to_string_pretty(&config).context("Failed to serialize config to JSON")?
        }
        _ => toml::to_string_pretty(&config).context("Failed to serialize config to TOML")?,
    };

    std::fs::write(&path, contents).with_context(|| format!("Failed to write {path}"))?;
    println!("Exported configuration to {path}");
    Ok(())
}
