//! # Project initialization commands
//!
//! Implements the `init` subcommand for scaffolding a new OpenSquilla project
//! directory. Creates a `.opensquilla/` directory with a default config file,
//! a skills directory, a data directory for SQLite databases, and a `.gitignore`
//! to keep runtime artifacts out of version control.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use tracing::info;

use crate::table::{self, Color, KeyValue, Style};

/// Init subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum InitAction {
    /// Initialize a new project in the current or given directory.
    Create {
        directory: Option<String>,
        name: Option<String>,
        force: bool,
    },
    /// Show what would be created without writing anything.
    Preview { directory: Option<String> },
}

/// Run an init subcommand.
pub async fn run_init(action: InitAction) -> Result<()> {
    match action {
        InitAction::Create {
            directory,
            name,
            force,
        } => init_project(directory, name, force).await,
        InitAction::Preview { directory } => preview_init(directory).await,
    }
}

/// Initialize a new OpenSquilla project.
pub async fn init_project(
    directory: Option<String>,
    name: Option<String>,
    force: bool,
) -> Result<()> {
    let dir = directory
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let project_name = name.unwrap_or_else(|| {
        dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("opensquilla-project")
            .to_string()
    });

    println!("{}", "Initializing OpenSquilla project".bold());
    println!();
    KeyValue::new()
        .entry("Directory", dir.display().to_string())
        .entry("Name", project_name.clone())
        .print();
    println!();

    if !dir.exists() {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
    }

    let opensquilla_dir = dir.join(".opensquilla");
    if opensquilla_dir.exists() && !force {
        anyhow::bail!(
            "{} already exists. Use --force to overwrite.",
            opensquilla_dir.display()
        );
    }

    // Create directory structure.
    let dirs = [
        opensquilla_dir.clone(),
        opensquilla_dir.join("skills"),
        opensquilla_dir.join("data"),
        opensquilla_dir.join("logs"),
        opensquilla_dir.join("agents"),
    ];
    for d in &dirs {
        std::fs::create_dir_all(d)
            .with_context(|| format!("Failed to create directory: {}", d.display()))?;
        println!("{} Created {}", table::ok(), d.display());
    }

    // Write config file.
    let config_path = opensquilla_dir.join("config.toml");
    let config = default_project_config(&project_name);
    let config_contents =
        toml::to_string_pretty(&config).context("Failed to serialize default config")?;
    std::fs::write(&config_path, &config_contents)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;
    println!("{} Created {}", table::ok(), config_path.display());

    // Write .gitignore.
    let gitignore_path = dir.join(".gitignore");
    let gitignore = default_gitignore();
    if !gitignore_path.exists() || force {
        std::fs::write(&gitignore_path, &gitignore)
            .with_context(|| format!("Failed to write {}", gitignore_path.display()))?;
        println!("{} Created {}", table::ok(), gitignore_path.display());
    }

    // Write a sample skill.
    let sample_skill_path = opensquilla_dir.join("skills").join("example.md");
    let sample_skill = sample_skill_content();
    std::fs::write(&sample_skill_path, &sample_skill)
        .with_context(|| format!("Failed to write {}", sample_skill_path.display()))?;
    println!("{} Created {}", table::ok(), sample_skill_path.display());

    // Write a project README.
    let readme_path = dir.join("OPENQUILLA.md");
    if !readme_path.exists() || force {
        let readme = project_readme(&project_name);
        std::fs::write(&readme_path, &readme)
            .with_context(|| format!("Failed to write {}", readme_path.display()))?;
        println!("{} Created {}", table::ok(), readme_path.display());
    }

    println!();
    println!("{} Project initialized successfully!", table::ok());
    println!();
    println!("Next steps:");
    println!("  1. Edit .opensquilla/config.toml to add your API keys");
    println!("  2. Run 'osq onboard' for the interactive setup wizard");
    println!("  3. Start chatting with: osq chat");
    println!("  4. Or launch the TUI: osq tui");
    println!();

    info!("Project initialized at {}", dir.display());
    Ok(())
}

/// Preview what would be created.
pub async fn preview_init(directory: Option<String>) -> Result<()> {
    let dir = directory
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    println!("{}", "Project Structure Preview".bold());
    println!();
    println!("{} would contain:", dir.display());
    println!();

    let tree = crate::table::TreeNode::new(&dir.display().to_string())
        .child(
            crate::table::TreeNode::new(".opensquilla/")
                .leaf("config.toml")
                .leaf("skills/")
                .leaf("data/")
                .leaf("logs/")
                .leaf("agents/"),
        )
        .leaf(".gitignore")
        .leaf("OPENQUILLA.md");
    tree.print();

    println!();
    println!("Run with --force to overwrite existing files.");
    Ok(())
}

/// Build a default project config.
fn default_project_config(name: &str) -> Config {
    let mut config = Config::default();
    config
        .providers
        .push(opensquilla_core::config::ProviderConfig {
            name: "openai".to_string(),
            provider_type: "openai".to_string(),
            api_key: None,
            base_url: None,
            models: vec!["gpt-4o-mini".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            max_retries: 3,
            timeout_secs: 60,
        });
    config.skills = Some(opensquilla_core::config::SkillsConfig {
        skill_dirs: vec![".opensquilla/skills".to_string()],
        enabled: true,
        max_execution_time_secs: 300,
    });
    config
}

/// Default .gitignore content for an OpenSquilla project.
fn default_gitignore() -> String {
    r#"# OpenSquilla runtime artifacts
.opensquilla/data/
.opensquilla/logs/
*.db
*.db-journal
*.db-wal
*.db-shm
gateway.pid

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
"#
    .to_string()
}

/// Sample skill content.
fn sample_skill_content() -> String {
    r#"---
name: example
description: An example skill showing the frontmatter format
version: 1.0.0
author: OpenSquilla
tags:
  - example
  - template
---

# Example Skill

This is an example skill. Skills are markdown files with YAML frontmatter.

## Steps

1. Read the user's request
2. Analyze the task
3. Execute the appropriate action
4. Report the result

## Usage

```
/osq skill example
```

Edit this file or add new `.md` files to the `skills/` directory to create
your own skills.
"#
    .to_string()
}

/// Project README content.
fn project_readme(name: &str) -> String {
    format!(
        r#"# {name}

This project is configured for [OpenSquilla](https://github.com/opensquilla/opensquilla).

## Quick Start

```bash
# Start chatting
osq chat

# Launch the TUI
osq tui

# Start the gateway
osq gateway start
```

## Configuration

Edit `.opensquilla/config.toml` to configure providers, models, channels,
and sandbox settings.

## Structure

- `.opensquilla/config.toml` — main configuration
- `.opensquilla/skills/` — custom skills
- `.opensquilla/data/` — SQLite databases (sessions, memory, scheduler)
- `.opensquilla/logs/` — log files
- `.opensquilla/agents/` — agent profiles

## Learn More

- `osq --help` — see all commands
- `osq onboard` — interactive setup wizard
- `osq doctor` — run diagnostics
"#
    )
}

/// Bold helper.
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
