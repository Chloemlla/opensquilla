//! # Web search commands
//!
//! Implements the `search` subcommand using the search crate's provider
//! registry (Brave, Tavily, DuckDuckGo, Exa, Bocha, IQS). Searches run against
//! whichever provider has credentials configured.
//!
//! - `search <query>` — run a web search
//! - `search providers` — list configured search providers
//! - `search with <provider> <query>` — search with a specific provider

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_search::registry::SearchRegistry;
use opensquilla_search::types::{SearchOptions, SearchRequest};

use crate::table::{self, Alignment, Color, Column, KeyValue, Style, Table};

/// Search subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum SearchAction {
    /// Run a web search with the default provider.
    Query {
        query: String,
        #[arg(long, default_value = "5")]
        results: usize,
    },
    /// List configured search providers.
    Providers,
    /// Search with a specific provider.
    With {
        provider: String,
        query: String,
        #[arg(long, default_value = "5")]
        results: usize,
    },
    /// Show the status of all search providers.
    Status,
}

/// Run a search subcommand.
pub async fn run_search(action: SearchAction) -> Result<()> {
    match action {
        SearchAction::Query { query, results } => search_query(&query, None, results).await,
        SearchAction::Providers => list_providers().await,
        SearchAction::With {
            provider,
            query,
            results,
        } => search_query(&query, Some(&provider), results).await,
        SearchAction::Status => provider_status().await,
    }
}

/// Run a web search.
pub async fn search_query(query: &str, provider: Option<&str>, max_results: usize) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = SearchRegistry::new(&config);

    let request = SearchRequest {
        query: query.to_string(),
        options: SearchOptions {
            max_results,
            ..Default::default()
        },
    };

    let start = std::time::Instant::now();
    let response = match provider {
        Some(name) => registry
            .search(name, &request)
            .await
            .map_err(|e| anyhow::anyhow!("Search provider '{name}' failed: {e}"))?,
        None => registry
            .search_default(&request)
            .await
            .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?,
    };
    let elapsed = start.elapsed();

    let provider_name = match provider {
        Some(name) => name.to_string(),
        None => registry.default_provider().to_string(),
    };

    println!("Search results for '{query}'");
    println!("Provider: {provider_name}  ({elapsed:?})");
    println!();

    if response.results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("#").align(Alignment::Right))
        .column(Column::new("Title").max_width(50))
        .column(Column::new("URL").max_width(60));

    for (i, result) in response.results.iter().enumerate() {
        table = table.row_owned(vec![
            (i + 1).to_string(),
            result.title.clone(),
            result.url.clone(),
        ]);
    }
    table.print();

    // Print snippets below the table.
    if response.results.iter().any(|r| !r.snippet.is_empty()) {
        println!();
        for (i, result) in response.results.iter().enumerate() {
            if !result.snippet.is_empty() {
                println!("  {}. {} — {}", i + 1, result.title, result.snippet);
            }
        }
    }

    if let Some(total) = response.total_results {
        println!();
        println!("{total} total result(s)");
    }
    Ok(())
}

/// List configured search providers.
pub async fn list_providers() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = SearchRegistry::new(&config);

    let names = registry.list_providers();
    if names.is_empty() {
        println!("No search providers configured.");
        return Ok(());
    }

    println!("Search Providers");
    println!("{:-<50}", "");
    for name in names {
        let marker = if name == registry.default_provider() {
            " (default)"
        } else {
            ""
        };
        println!("  - {name}{marker}");
    }
    println!("{:-<50}", "");
    Ok(())
}

/// Show search provider status.
pub async fn provider_status() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let registry = SearchRegistry::new(&config);

    let ready = registry.list_ready_providers();
    let all = registry.list_providers();

    println!("Search Provider Status");
    println!();
    KeyValue::new()
        .entry("Providers", all.len().to_string())
        .entry("Ready", ready.len().to_string())
        .entry("Default", registry.default_provider().to_string())
        .print();

    println!();
    let mut table = Table::from_headers(&["Provider", "Status"]).border(table::TableBorder::Header);
    for name in all {
        let status = if ready.contains(&name) {
            "ready".to_string()
        } else {
            "no credentials".to_string()
        };
        table = table.row_owned(vec![name.to_string(), status.clone()]);
        let _ = status;
    }
    table.print();
    Ok(())
}

/// Bold helper for headers.
#[allow(dead_code)]
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
