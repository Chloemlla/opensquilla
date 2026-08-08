//! # Tool registry commands
//!
//! Implements the `tools` subcommand for inspecting and testing the built-in
//! tool registry. Tools are the executable capabilities the agent can call
//! (shell, filesystem, git, web, memory, session management, etc.).
//!
//! - `tools list` — list all registered tools grouped by category
//! - `tools show <name>` — show a tool's definition and parameters
//! - `tools test <name>` — dry-run a tool with sample arguments
//! - `tools categories` — list tool categories

use anyhow::{Context, Result};
use opensquilla_tools::registry::ToolRegistry;

use crate::table::{self, Alignment, Color, Column, KeyValue, Style, Table};

/// Tool subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum ToolAction {
    /// List all registered tools.
    List {
        /// Filter by category.
        category: Option<String>,
    },
    /// Show a tool's definition.
    Show { name: String },
    /// Dry-run a tool with sample arguments.
    Test {
        name: String,
        /// JSON arguments for the tool.
        args: Option<String>,
    },
    /// List tool categories.
    Categories,
}

/// Run a tools subcommand.
pub async fn run_tool(action: ToolAction) -> Result<()> {
    match action {
        ToolAction::List { category } => list_tools(category).await,
        ToolAction::Show { name } => show_tool(name).await,
        ToolAction::Test { name, args } => test_tool(name, args).await,
        ToolAction::Categories => list_categories().await,
    }
}

/// Build the tool registry with all built-in tools.
fn build_registry() -> Result<ToolRegistry> {
    ToolRegistry::with_builtins().map_err(|e| anyhow::anyhow!("Failed to build tool registry: {e}"))
}

/// List all registered tools, optionally filtered by category.
pub async fn list_tools(category: Option<String>) -> Result<()> {
    let registry = build_registry()?;

    if registry.is_empty() {
        println!("No tools registered.");
        return Ok(());
    }

    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("Tool"))
        .column(Column::new("Category").max_width(20))
        .column(Column::new("Risk").align(Alignment::Right))
        .column(Column::new("Confirmation"))
        .column(Column::new("Description").max_width(50));

    let mut names = registry.tool_names();
    names.sort();
    for name in names {
        let Some(tool) = registry.get(&name) else {
            continue;
        };
        let def = tool.definition();
        let cat = def
            .category
            .clone()
            .unwrap_or_else(|| "general".to_string());
        if let Some(ref filter) = category {
            if !cat.to_lowercase().contains(&filter.to_lowercase()) {
                continue;
            }
        }
        table = table.row_owned(vec![
            def.name.clone(),
            cat,
            def.risk_level.to_string(),
            if def.requires_confirmation {
                "yes"
            } else {
                "no"
            }
            .to_string(),
            def.description.chars().take(48).collect(),
        ]);
    }

    println!("Tools ({})", registry.len());
    println!();
    table.print();
    Ok(())
}

/// Show a single tool's definition and parameters.
pub async fn show_tool(name: String) -> Result<()> {
    let registry = build_registry()?;
    let tool = registry
        .get(&name)
        .ok_or_else(|| anyhow::anyhow!("Tool '{name}' not found"))?;
    let def = tool.definition();

    println!("Tool: {}", def.name);
    println!();
    KeyValue::new()
        .entry("Description", def.description.clone())
        .entry(
            "Category",
            def.category.clone().unwrap_or_else(|| "general".into()),
        )
        .entry("Risk level", def.risk_level.to_string())
        .entry(
            "Confirmation",
            if def.requires_confirmation {
                "required"
            } else {
                "none"
            }
            .to_string(),
        )
        .entry(
            "Behavior",
            if def.risk_level == 0 {
                "safe".to_string()
            } else if def.risk_level <= 2 {
                "moderate".to_string()
            } else {
                "potentially unsafe".to_string()
            },
        )
        .print();

    if !def.parameters.is_empty() {
        println!();
        println!("Parameters:");
        let mut table = Table::from_headers(&["Name", "Type", "Required", "Description"])
            .border(table::TableBorder::Header);
        for (param_name, p) in &def.parameters {
            table = table.row_owned(vec![
                param_name.clone(),
                p.param_type.clone(),
                if p.required { "yes" } else { "no" }.to_string(),
                p.description.clone().unwrap_or_default(),
            ]);
        }
        table.print();
    }
    Ok(())
}

/// Dry-run a tool with sample arguments.
pub async fn test_tool(name: String, args: Option<String>) -> Result<()> {
    let registry = build_registry()?;
    let tool = registry
        .get(&name)
        .ok_or_else(|| anyhow::anyhow!("Tool '{name}' not found"))?;

    let args_json: serde_json::Value = match args {
        Some(a) => {
            serde_json::from_str(&a).with_context(|| format!("Invalid JSON arguments: {a}"))?
        }
        None => serde_json::json!({}),
    };

    println!("Tool test: {name}");
    println!("  Arguments: {}", args_json);
    println!();

    // Validate against the tool's parameter schema.
    let def = tool.definition();
    for (param_name, p) in &def.parameters {
        if p.required && !args_json.get(param_name).is_some() {
            println!(
                "{} Missing required parameter '{}'",
                table::fail(),
                param_name
            );
            return Ok(());
        }
    }

    println!("{} Parameter validation passed", table::ok());

    // Show what the tool would receive.
    let definition = def.to_json_schema();
    println!(
        "  JSON schema: {}",
        serde_json::to_string_pretty(&definition)?
    );

    // If args were provided and the tool accepts them, print a dry-run notice.
    if !args_json.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        println!();
        println!(
            "{} Dry-run: not executing tool (use the agent to run real calls)",
            table::info()
        );
    }
    Ok(())
}

/// List tool categories with counts.
pub async fn list_categories() -> Result<()> {
    let registry = build_registry()?;
    let by_cat = registry.by_category();

    let mut cats: Vec<(&str, usize)> = by_cat
        .iter()
        .map(|(cat, tools)| (*cat, tools.len()))
        .collect();
    cats.sort_by_key(|b| std::cmp::Reverse(b.1));

    println!("Tool Categories ({})", cats.len());
    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("Category"))
        .column(Column::new("Tools").align(Alignment::Right));
    for (cat, count) in &cats {
        table = table.row_owned(vec![cat.to_string(), count.to_string()]);
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
