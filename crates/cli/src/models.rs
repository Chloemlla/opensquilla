//! Model catalog commands.
//!
//! Implements the `models` subcommand. Lists models supported by the configured
//! providers and shows capability details (context window, tool/vision support,
//! pricing) from the provider crate's [`ModelCatalog`].

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_provider::model_catalog::{ModelCatalog, seed_static};

use crate::util;

/// List the models available from a provider (or all configured providers).
pub async fn list_models(provider: Option<String>) -> Result<()> {
    list_models_filtered(provider, false, false).await
}

/// List models with optional provider and capability filters.
pub async fn list_models_filtered(
    provider: Option<String>,
    tools: bool,
    vision: bool,
) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let catalog = ModelCatalog::new();

    if let Some(provider_name) = provider {
        let p = registry
            .get(&provider_name)
            .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' is not configured"))?;
        print_provider_models_filtered(
            &catalog,
            &provider_name,
            p.name(),
            &p.supported_models(),
            &config,
            tools,
            vision,
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
            print_provider_models_filtered(
                &catalog,
                &name,
                p.name(),
                &p.supported_models(),
                &config,
                tools,
                vision,
            );
            println!();
        }
    }
    Ok(())
}

/// Print provider models applying capability filters.
fn print_provider_models_filtered(
    catalog: &ModelCatalog,
    provider_name: &str,
    backend: &str,
    models: &[String],
    config: &Config,
    tools: bool,
    vision: bool,
) {
    println!("Models available from {provider_name} ({backend}):");
    println!("{:-<50}", "");
    for model in models {
        let caps = catalog.get(backend, model);

        // Apply filters.
        if tools {
            if !caps.as_ref().is_some_and(|c| c.supports_tools) {
                continue;
            }
        }
        if vision {
            if !caps.as_ref().is_some_and(|c| c.supports_vision) {
                continue;
            }
        }

        let is_default = if *model == util::default_model(config) {
            " (default)"
        } else {
            ""
        };
        let detail = match &caps {
            Some(c) if c.context_window.is_some() => {
                format!("  ctx={}k", c.context_window.unwrap() / 1000)
            }
            _ => String::new(),
        };
        let badges = match &caps {
            Some(c) => {
                let mut b = Vec::new();
                if c.supports_tools {
                    b.push("tools");
                }
                if c.supports_vision {
                    b.push("vision");
                }
                if c.supports_reasoning {
                    b.push("reasoning");
                }
                if b.is_empty() {
                    String::new()
                } else {
                    format!("  [{}]", b.join(","))
                }
            }
            None => String::new(),
        };
        println!("  - {model}{is_default} {detail}{badges}");
    }
    println!("{:-<50}", "");
}

/// Compare two or more models side by side.
pub async fn compare_models(models: Vec<String>) -> Result<()> {
    if models.len() < 2 {
        anyhow::bail!("Provide at least two models to compare");
    }
    let config = Config::load().context("Failed to load configuration")?;
    let registry = util::build_provider_registry(&config)?;
    let catalog = ModelCatalog::new();

    println!("Model Comparison");
    let mut table = crate::table::Table::new()
        .border(crate::table::TableBorder::Header)
        .column(crate::table::Column::new("Attribute"))
        .column(crate::table::Column::new("Model 1"))
        .column(crate::table::Column::new("Model 2"));

    if models.len() > 2 {
        // Extend for up to 4 models.
        for i in 3..=models.len().min(4) {
            table = table.column(crate::table::Column::new(format!("Model {i}")));
        }
    }

    let specs: Vec<_> = models
        .iter()
        .map(|m| {
            for name in registry.list() {
                if let Some(p) = registry.get(&name) {
                    if p.supported_models().iter().any(|x| x == m) {
                        return (m.clone(), p.name().to_string());
                    }
                }
            }
            (m.clone(), "unknown".to_string())
        })
        .collect();

    for (i, spec) in specs.iter().enumerate() {
        seed_static(&catalog, &spec.1);
    }

    // Build comparison rows.
    let attrs = [
        (
            "Backend",
            specs
                .iter()
                .map(|(_, b)| b.clone())
                .collect::<Vec<String>>(),
        ),
        (
            "Context window",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .and_then(|c| c.context_window)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".to_string())
                })
                .collect(),
        ),
        (
            "Max output",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .and_then(|c| c.max_output_tokens)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "—".to_string())
                })
                .collect(),
        ),
        (
            "Tools",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .map(|c| if c.supports_tools { "yes" } else { "no" }.to_string())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect(),
        ),
        (
            "Vision",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .map(|c| if c.supports_vision { "yes" } else { "no" }.to_string())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect(),
        ),
        (
            "Reasoning",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .map(|c| if c.supports_reasoning { "yes" } else { "no" }.to_string())
                        .unwrap_or_else(|| "?".to_string())
                })
                .collect(),
        ),
        (
            "Input $/1M",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .and_then(|c| c.input_price_per_million)
                        .map(|v| format!("${v:.2}"))
                        .unwrap_or_else(|| "—".to_string())
                })
                .collect(),
        ),
        (
            "Output $/1M",
            specs
                .iter()
                .map(|(m, b)| {
                    catalog
                        .get(b, m)
                        .and_then(|c| c.output_price_per_million)
                        .map(|v| format!("${v:.2}"))
                        .unwrap_or_else(|| "—".to_string())
                })
                .collect(),
        ),
    ];

    for (attr, vals) in &attrs {
        let cells: Vec<String> = std::iter::once(attr.to_string())
            .chain(vals.iter().cloned())
            .collect();
        table = table.row_owned(cells);
    }
    table.print();
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
    if b { "yes" } else { "no" }
}
