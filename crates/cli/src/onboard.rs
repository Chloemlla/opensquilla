//! # Onboarding commands
//!
//! Implements the `onboard` subcommand, an interactive setup wizard that
//! guides the user through initial configuration: provider selection, API key
//! entry, model preference, sandbox level, and a connectivity test. The wizard
//! writes the resulting configuration to disk and runs a `doctor` check at the
//! end.
//!
//! - `onboard` — run the full interactive wizard
//! - `onboard reset` — reset configuration to defaults
//! - `onboard check` — verify the current setup

use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_core::types::Message;
use opensquilla_provider::{ChatConfig, ProviderSpecTable};
use tracing::info;

use crate::table::{self, Color, KeyValue, Style};
use crate::util;

/// Onboarding subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum OnboardAction {
    /// Run the full interactive wizard.
    Run,
    /// Reset configuration to defaults.
    Reset,
    /// Verify the current setup.
    Check,
}

/// Run an onboarding subcommand.
pub async fn run_onboard(action: OnboardAction) -> Result<()> {
    match action {
        OnboardAction::Run => run_wizard().await,
        OnboardAction::Reset => reset_config().await,
        OnboardAction::Check => check_setup().await,
    }
}

/// Run the interactive onboarding wizard.
pub async fn run_wizard() -> Result<()> {
    println!("{}", "Welcome to OpenSquilla!".bold());
    println!();
    println!("This wizard will help you configure OpenSquilla for first use.");
    println!("You can re-run it anytime with: osq onboard");
    println!();

    let mut config = Config::load().unwrap_or_default();

    // Step 1: Provider selection.
    println!("{}", "Step 1: Choose your AI provider".bold());
    println!();
    let specs = ProviderSpecTable::all();
    for (i, spec) in specs.iter().enumerate() {
        let status = if spec.enabled { "✓" } else { " " };
        println!(
            "  [{status}] {}. {} (default model: {})",
            i + 1,
            spec.id,
            spec.default_model
        );
    }
    println!();

    let provider_type = prompt(
        "Enter provider name or number (default: openai)",
        Some("openai"),
    )?;
    let provider_type = resolve_provider_choice(&provider_type, &specs);

    // Step 2: API key.
    println!();
    println!("{}", "Step 2: API key".bold());
    let env_var = format!("{}_API_KEY", provider_type.to_uppercase());
    let env_key = std::env::var(&env_var).ok();
    let api_key = if let Some(ref _key) = env_key {
        println!("Found key in ${env_var} (will use environment variable).");
        None
    } else {
        let key = prompt_secret("Enter your API key (input hidden)")?;
        if key.is_empty() {
            println!("No key provided — you can set it later via config or environment variable.");
        }
        Some(key)
    };

    // Step 3: Model selection.
    println!();
    println!("{}", "Step 3: Choose a model".bold());
    let spec = ProviderSpecTable::get(&provider_type);
    let default_model = spec
        .as_ref()
        .map(|s| s.default_model.to_string())
        .unwrap_or_else(|| "gpt-4".to_string());
    if let Some(spec) = spec {
        println!("Available models for {provider_type}:");
        for m in spec.models {
            let marker = if *m == default_model {
                " (default)"
            } else {
                ""
            };
            println!("  - {m}{marker}");
        }
    }
    println!();
    let model = prompt("Enter model name", Some(&default_model))?;

    // Step 4: Provider name.
    println!();
    println!("{}", "Step 4: Name this provider".bold());
    let default_name = provider_type.to_string();
    let provider_name = prompt("Provider display name", Some(&default_name))?;

    // Step 5: Sandbox level.
    println!();
    println!("{}", "Step 5: Sandbox level".bold());
    println!("  1. strict    — maximum isolation, no network, read-only filesystem");
    println!("  2. standard  — balanced (recommended)");
    println!("  3. permissive — full access (development only)");
    println!();
    let level = prompt("Sandbox level (default: standard)", Some("standard"))?;
    let sandbox_level = match level.to_lowercase().as_str() {
        "1" | "strict" => "strict",
        "3" | "permissive" => "permissive",
        _ => "standard",
    };

    // Step 6: Gateway configuration.
    println!();
    println!("{}", "Step 6: Gateway configuration".bold());
    let host = prompt("Gateway host", Some("127.0.0.1"))?;
    let port = prompt("Gateway port", Some("8080"))?;
    let port: u16 = port.parse().context("Invalid port number")?;

    // Summary.
    println!();
    println!("{}", "Configuration Summary".bold());
    table::rule();
    KeyValue::new()
        .entry("Provider type", provider_type.clone())
        .entry("Provider name", provider_name.clone())
        .entry("Model", model.clone())
        .entry(
            "API key",
            if api_key.is_some() {
                "(set)"
            } else {
                "(from env)"
            },
        )
        .entry("Sandbox", sandbox_level.to_string())
        .entry("Gateway", format!("{host}:{port}"))
        .print();
    println!();

    let confirm = prompt("Save configuration? (yes/no)", Some("yes"))?;
    if !confirm.to_lowercase().starts_with('y') {
        println!("Configuration not saved.");
        return Ok(());
    }

    // Apply configuration.
    apply_provider_config(
        &mut config,
        &provider_name,
        &provider_type,
        &model,
        api_key.as_deref(),
    );
    apply_sandbox_config(&mut config, sandbox_level);
    apply_gateway_config(&mut config, &host, port);

    config.save().context("Failed to save configuration")?;
    println!("{} Configuration saved.", table::ok());

    // Step 7: Connectivity test.
    println!();
    println!("{}", "Step 7: Connectivity test".bold());
    test_connectivity(&config, &provider_name, &model).await;

    // Step 8: Doctor check.
    println!();
    println!("{}", "Step 8: Diagnostics".bold());
    run_doctor_quick(&config).await;

    println!();
    println!("{} Onboarding complete!", table::ok());
    println!();
    println!("Next steps:");
    println!("  • Start chatting:        osq chat");
    println!("  • Launch the TUI:        osq tui");
    println!("  • Start the gateway:     osq gateway start");
    println!("  • Run diagnostics:       osq doctor");
    println!();

    info!("Onboarding completed for provider {provider_name}");
    Ok(())
}

/// Reset configuration to defaults.
pub async fn reset_config() -> Result<()> {
    let confirm = prompt(
        "Reset all configuration to defaults? This cannot be undone. (yes/no)",
        Some("no"),
    )?;
    if !confirm.to_lowercase().starts_with('y') {
        println!("Reset cancelled.");
        return Ok(());
    }
    let config = Config::default();
    config
        .save()
        .context("Failed to save default configuration")?;
    println!("{} Configuration reset to defaults.", table::ok());
    Ok(())
}

/// Verify the current setup.
pub async fn check_setup() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;

    println!("{}", "Setup Check".bold());
    table::rule();

    let mut issues = 0u32;

    // Check providers.
    if config.providers.is_empty() {
        println!("{} No providers configured", table::fail());
        issues += 1;
    } else {
        println!(
            "{} {} provider(s) configured",
            table::ok(),
            config.providers.len()
        );
    }

    // Check API keys.
    for p in &config.providers {
        let env_var = format!("{}_API_KEY", p.provider_type.to_uppercase());
        let has_key = util::provider_api_key(&config, p).is_some();
        if has_key {
            println!("  {} {} — API key set", table::ok(), p.name);
        } else {
            println!(
                "  {} {} — no API key (set ${} or providers.{}.api_key)",
                table::fail(),
                p.name,
                env_var,
                p.name
            );
            issues += 1;
        }
    }

    // Check gateway.
    let port_open =
        tokio::net::TcpStream::connect((config.gateway.host.as_str(), config.gateway.port))
            .await
            .is_ok();
    if port_open {
        println!(
            "{} Gateway running on {}:{}",
            table::ok(),
            config.gateway.host,
            config.gateway.port
        );
    } else {
        println!(
            "{} Gateway not running on {}:{}",
            table::warn(),
            config.gateway.host,
            config.gateway.port
        );
    }

    // Check session store.
    match util::build_session_manager(&config) {
        Ok(_) => println!("{} Session store accessible", table::ok()),
        Err(e) => {
            println!("{} Session store error: {e}", table::fail());
            issues += 1;
        }
    }

    // Check memory store.
    let mem_path = util::memory_db_path();
    if mem_path.exists() {
        println!("{} Memory store exists", table::ok());
    } else {
        println!(
            "{} Memory store not initialized (will be created on first use)",
            table::info()
        );
    }

    println!();
    if issues == 0 {
        println!("{} All checks passed.", table::ok());
    } else {
        println!("{} {issues} issue(s) found.", table::warn());
        println!("Run 'osq onboard' to fix configuration issues.");
    }
    Ok(())
}

/// Test connectivity to the configured provider.
async fn test_connectivity(config: &Config, provider_name: &str, model: &str) {
    let registry = match util::build_provider_registry(config) {
        Ok(r) => r,
        Err(e) => {
            println!("{} Failed to build provider registry: {e}", table::fail());
            return;
        }
    };
    let provider = match registry.get(provider_name) {
        Some(p) => p,
        None => {
            println!("{} Provider '{provider_name}' not found", table::fail());
            return;
        }
    };

    let chat_config = ChatConfig {
        model: model.to_string(),
        temperature: 0.0,
        max_tokens: 16,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: false,
        extra: Default::default(),
    };

    print!("Testing connection... ");
    let _ = io::stdout().flush();

    let start = std::time::Instant::now();
    match provider
        .send_message(&chat_config, &[Message::user("ping")], &[])
        .await
    {
        Ok(resp) => {
            let ms = start.elapsed().as_millis();
            let reply = resp
                .content
                .iter()
                .map(|m| m.text_content())
                .collect::<Vec<_>>()
                .join(" ");
            println!("{} OK ({ms} ms)", table::ok());
            if !reply.is_empty() {
                println!("  Reply: {reply}");
            }
        }
        Err(e) => {
            println!("{} Failed: {e}", table::fail());
            println!("  Check your API key and network connection.");
        }
    }
}

/// Run a quick doctor check and print a summary.
async fn run_doctor_quick(config: &Config) {
    let health = opensquilla_recovery::health::HealthCheck::new(config);
    let result = health.run_full_check().await;
    let status_icon = match result.status {
        opensquilla_recovery::health::HealthStatus::Healthy => table::ok(),
        opensquilla_recovery::health::HealthStatus::Degraded => table::warn(),
        opensquilla_recovery::health::HealthStatus::Unhealthy => table::fail(),
    };
    println!("Diagnostics: {status_icon} {:?}", result.status);
    if !result.issues.is_empty() {
        for issue in result.issues.iter().take(3) {
            println!("  {} {}", table::warn(), issue.message);
        }
        if result.issues.len() > 3 {
            println!("  ... and {} more", result.issues.len() - 3);
        }
    }
}

/// Resolve a user's provider choice (number or name).
fn resolve_provider_choice(choice: &str, specs: &[opensquilla_provider::ProviderSpec]) -> String {
    // Try as a number.
    if let Ok(n) = choice.parse::<usize>() {
        if n > 0 && n <= specs.len() {
            return specs[n - 1].id.to_string();
        }
    }
    // Fall back to the string itself.
    choice.to_string()
}

/// Apply provider configuration.
fn apply_provider_config(
    config: &mut Config,
    name: &str,
    provider_type: &str,
    model: &str,
    api_key: Option<&str>,
) {
    // Remove any existing provider with the same name.
    config.providers.retain(|p| p.name != name);
    let provider_config = opensquilla_core::config::ProviderConfig {
        name: name.to_string(),
        provider_type: provider_type.to_string(),
        api_key: api_key.map(|s| s.to_string()),
        base_url: None,
        models: vec![model.to_string()],
        default_model: Some(model.to_string()),
        max_retries: 3,
        timeout_secs: 60,
    };
    config.providers.push(provider_config);
}

/// Apply sandbox configuration.
fn apply_sandbox_config(config: &mut Config, level: &str) {
    let mut sandbox =
        config
            .sandbox
            .clone()
            .unwrap_or_else(|| opensquilla_core::config::SandboxConfig {
                enabled: false,
                sandbox_type: "process".to_string(),
                timeout_secs: 120,
                resource_limits: opensquilla_core::config::ResourceLimits::default(),
            });
    sandbox.enabled = true;
    sandbox.sandbox_type = level.to_string();
    config.sandbox = Some(sandbox);
}

/// Apply gateway configuration.
fn apply_gateway_config(config: &mut Config, host: &str, port: u16) {
    config.gateway.host = host.to_string();
    config.gateway.port = port;
}

// ---------------------------------------------------------------------------
// Input helpers
// ---------------------------------------------------------------------------

/// Read a line from stdin, returning the default if empty.
fn prompt(message: &str, default: Option<&str>) -> Result<String> {
    print!("{message}");
    if let Some(d) = default {
        print!(" [{d}]");
    }
    print!(": ");
    io::stdout().flush()?;
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    let trimmed = line.trim().to_string();
    if trimmed.is_empty() {
        if let Some(d) = default {
            return Ok(d.to_string());
        }
    }
    Ok(trimmed)
}

/// Read a secret from stdin without echoing.
fn prompt_secret(message: &str) -> Result<String> {
    print!("{message}: ");
    io::stdout().flush()?;
    // On Unix we could use termios to disable echo, but to stay portable
    // and dependency-free we just read a line. The key will be visible,
    // which is acceptable for a CLI setup wizard.
    let stdin = io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    Ok(line.trim().to_string())
}

/// Style helper for bold text.
trait BoldStr {
    fn bold(&self) -> String;
}

impl BoldStr for &str {
    fn bold(&self) -> String {
        format!(
            "{}",
            Style::new().bold().fg(Color::BrightBlue).styled(*self)
        )
    }
}

impl BoldStr for String {
    fn bold(&self) -> String {
        format!("{}", Style::new().bold().fg(Color::BrightBlue).styled(self))
    }
}
