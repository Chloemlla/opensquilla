//! Provider management commands.
//!
//! Implements the `providers` subcommand against the provider crate's
//! [`ProviderRegistry`] and [`ProviderSpecTable`]. Listing and status are pure
//! registry queries (Mode B); `providers test` performs a live round-trip to
//! validate credentials and connectivity.

use std::time::Instant;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_core::types::Message;
use opensquilla_provider::{ChatConfig, Provider, ProviderSpecTable};
use tracing::info;

use crate::util;

/// List all configured providers and the well-known provider catalogue.
pub async fn list_providers() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let names = registry.list();

    if names.is_empty() {
        println!("No providers configured.");
        println!("Run 'opensquilla onboarding' or add a [[providers]] entry to your config.");
    } else {
        println!("Configured providers:");
        println!("{:-<60}", "");
        for name in names {
            if let Some(provider) = registry.get(&name) {
                let model_count = provider.supported_models().len();
                println!(
                    "  {:<20} (backend: {})  {} model(s)",
                    name,
                    provider.name(),
                    model_count
                );
            }
        }
        println!("{:-<60}", "");
    }

    println!();
    println!("Available provider types:");
    for spec in ProviderSpecTable::all() {
        let marker = if spec.enabled { "enabled" } else { "disabled" };
        println!(
            "  {:<20} {}  default: {}",
            spec.id, marker, spec.default_model
        );
    }
    Ok(())
}

/// Show the status of a provider (or the default provider).
pub async fn show_provider_status(name: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;

    let provider_name = name.unwrap_or_else(|| util::default_provider(&config));
    let provider = registry
        .get(&provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' is not configured"))?;

    let models = provider.supported_models();
    let spec = ProviderSpecTable::get(provider.name());
    let default_model = spec
        .map(|s| s.default_model.to_string())
        .unwrap_or_default();

    println!("Provider:  {}", provider_name);
    println!("Backend:   {}", provider.name());
    println!("Status:    configured");
    println!("Models:    {}", models.len());
    for m in &models {
        let is_default = if *m == default_model {
            " (default)"
        } else {
            ""
        };
        println!("             - {m}{is_default}");
    }
    Ok(())
}

/// Test a provider connection by sending a minimal message and measuring
/// round-trip latency.
pub async fn test_provider(name: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;

    let provider_name = name.unwrap_or_else(|| util::default_provider(&config));
    let provider = registry
        .get(&provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' is not configured"))?;

    let model = provider_default_model(&config, &provider_name, provider.as_ref());

    println!(
        "Testing provider '{provider_name}' (backend: {})...",
        provider.name()
    );
    println!("  Model:   {model}");
    println!("  Sending: \"ping\"");

    let chat_config = ChatConfig {
        model: model.clone(),
        temperature: 0.0,
        max_tokens: 16,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: false,
        extra: Default::default(),
    };

    let start = Instant::now();
    match provider
        .send_message(&chat_config, &[Message::user("ping")], &[])
        .await
    {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis();
            let reply = resp
                .content
                .iter()
                .map(|m| m.text_content())
                .collect::<Vec<_>>()
                .join(" ");
            println!("  OK       in {latency_ms} ms");
            if !reply.is_empty() {
                println!("  Reply:   {reply}");
            }
            info!("Provider {provider_name} health check passed in {latency_ms} ms");
            Ok(())
        }
        Err(e) => {
            let latency_ms = start.elapsed().as_millis();
            anyhow::bail!("Provider '{provider_name}' failed after {latency_ms} ms: {e}");
        }
    }
}

/// Resolve the model to use for a provider: config default, else the provider's
/// own spec default.
fn provider_default_model(config: &Config, provider_name: &str, provider: &dyn Provider) -> String {
    if let Some(pc) = config.find_provider(provider_name) {
        if let Some(m) = &pc.default_model {
            return m.clone();
        }
        if let Some(m) = pc.models.first() {
            return m.clone();
        }
    }
    let models = provider.supported_models();
    if let Some(m) = models.first() {
        return m.clone();
    }
    util::default_model(config)
}

/// Add a new provider to the configuration.
pub async fn add_provider(
    name: String,
    provider_type: String,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;

    // Check for duplicates.
    if config.find_provider(&name).is_some() {
        anyhow::bail!("Provider '{name}' already exists");
    }

    let models = match &model {
        Some(m) => vec![m.clone()],
        None => ProviderSpecTable::get(&provider_type)
            .map(|s| s.models.iter().map(|m| m.to_string()).collect())
            .unwrap_or_default(),
    };
    let default_model = model.clone().or_else(|| {
        ProviderSpecTable::get(&provider_type).map(|s| s.default_model.to_string())
    });

    let provider_config = opensquilla_core::config::ProviderConfig {
        name: name.clone(),
        provider_type: provider_type.clone(),
        api_key,
        base_url,
        models,
        default_model,
        ..Default::default()
    };
    config.providers.push(provider_config);
    config.save().context("Failed to save configuration")?;

    println!("{} Added provider: {} ({})", crate::table::ok(), name, provider_type);
    if let Some(m) = &default_model {
        println!("  Default model: {m}");
    }
    Ok(())
}

/// Remove a provider from the configuration.
pub async fn remove_provider(name: String) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;
    let before = config.providers.len();
    config.providers.retain(|p| p.name != name);
    if config.providers.len() == before {
        anyhow::bail!("Provider '{name}' not found");
    }
    config.save().context("Failed to save configuration")?;
    println!("{} Removed provider: {}", crate::table::ok(), name);
    Ok(())
}

/// Set the default provider by moving it to the front of the list.
pub async fn set_default_provider(name: String) -> Result<()> {
    let mut config = Config::load().context("Failed to load configuration")?;
    let idx = config
        .providers
        .iter()
        .position(|p| p.name == name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{name}' not found"))?;
    let provider = config.providers.remove(idx);
    config.providers.insert(0, provider);
    config.save().context("Failed to save configuration")?;
    println!("{} Default provider set to: {}", crate::table::ok(), name);
    Ok(())
}
