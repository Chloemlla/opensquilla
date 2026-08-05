//! Model catalog commands.
//!
//! Implements the `models` subcommand. Lists models supported by the configured
//! providers and shows capability details (context window, tool/vision support,
//! pricing) from the provider crate's [`ModelCatalog`].

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_provider::model_catalog::{seed_static, ModelCatalog};

use crate::util;

/// List the models available from a provider (or all configured providers).
pub async fn list_models(provider: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let catalog = ModelCatalog::new();

    if let Some(provider_name) = provider {
        let p = registry
            .get(&provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' is not configured"))?;
        print_provider_models(
            &catalog,
            &provider_name,
            p.name(),
            &p.supported_models(),
            &config,
        );
        return Ok(());
    }

    let names = registry.list();
    if names.is_empty() {
        println!("No providers configured.");
        return Ok(());
    }

    for name in names {
        if let Some(p) = registry.get(&name) {
            print_provider_models(&catalog, &name, p.name(), &p.supported_models(), &config);
            println!();
        }
    }
    Ok(())
}

/// Show detailed capability information for a single model.
pub async fn show_model(name: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let catalog = ModelCatalog::new();

    for provider_name in registry.list() {
        let Some(p) = registry.get(&provider_name) else {
            continue;
        };
        let backend = p.name();
        if p.supported_models().iter().any(|m| m == &name) {
            println!("Model:    {name}");
            println!("Provider: {provider_name}");
            println!("Backend:  {backend}");
            print_capabilities(&catalog, backend, &name);
            return Ok(());
        }
    }

    // Not found in a configured provider; still surface static catalogue info.
    if let Some(spec) = opensquilla_provider::ProviderSpecTable::get(
        config
            .providers
            .first()
            .map(|p| p.provider_type.as_str())
            .unwrap_or("openai"),
    ) {
        println!("Model:    {name}");
        println!("Provider: (static catalogue)");
        println!("Backend:  {}", spec.id);
        print_capabilities(&catalog, spec.id, &name);
        return Ok(());
    }

    anyhow::bail!("Model '{name}' not found in any configured provider")
}

fn print_provider_models(
    catalog: &ModelCatalog,
    provider_name: &str,
    backend: &str,
    models: &[String],
    config: &Config,
) {
    println!("Models available from {provider_name} ({}):", backend);
    println!("{:-<50}", "");
    for model in models {
        let is_default = if *model == util::default_model(config) {
            " (default)"
        } else {
            ""
        };
        let caps = catalog.get(backend, model);
        let detail = match caps {
            Some(c) if c.context_window.is_some() => {
                format!("  ctx={}k", c.context_window.unwrap() / 1000)
            }
            _ => String::new(),
        };
        println!("  - {model}{is_default} {detail}");
    }
    println!("{:-<50}", "");
}

fn print_capabilities(catalog: &ModelCatalog, backend: &str, model: &str) {
    seed_static(catalog, backend);
    match catalog.get(backend, model) {
        Some(caps) => {
            println!("  Label:        {}", caps.label.as_deref().unwrap_or(model));
            if let Some(ctx) = caps.context_window {
                println!("  Context:      {ctx} tokens");
            }
            if let Some(out) = caps.max_output_tokens {
                println!("  Max output:   {out} tokens");
            }
            println!("  Tools:        {}", yes_no(caps.supports_tools));
            println!("  Vision:       {}", yes_no(caps.supports_vision));
            println!("  Audio:        {}", yes_no(caps.supports_audio));
            println!("  Reasoning:    {}", yes_no(caps.supports_reasoning));
            println!("  Streaming:    {}", yes_no(caps.supports_streaming));
            if let Some(price) = caps.input_price_per_million {
                println!("  Input price:  ${price:.2} / 1M tokens");
            }
            if let Some(price) = caps.output_price_per_million {
                println!("  Output price: ${price:.2} / 1M tokens");
            }
            if !caps.tags.is_empty() {
                println!("  Tags:         {}", caps.tags.join(", "));
            }
        }
        None => {
            println!("  (no capability metadata available for this model)");
        }
    }
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}
