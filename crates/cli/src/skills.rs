//! Skills management commands.
//!
//! Implements the `skills` subcommand. Installed skills are discovered with a
//! [`SkillLoader`] scanning the configured skill directories plus the managed
//! skills directory; installation, uninstallation, and search go through a
//! [`SkillHub`] backed by the GitHub and ClawHub sources.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_skills::bundled::load_bundled_skills;
use opensquilla_skills::hub::{GitHubSource, SkillHub, SkillMeta};
use opensquilla_skills::loader::SkillLoader;
use opensquilla_skills::types::SkillLayer;
use tracing::info;

use crate::util;

/// List installed skills across all layers.
pub async fn list_skills() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let loader = build_loader(&config)?;
    let skills = loader
        .get_skills(None)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to scan skills: {e}"))?;

    if skills.is_empty() {
        println!("No skills installed.");
        return Ok(());
    }

    println!("Installed skills ({})", skills.len());
    println!("{:-<70}", "");
    println!(
        "{:<28} {:<24} {:<10} {}",
        "Name", "Layer", "Version", "Description"
    );
    println!("{:-<70}", "");
    for skill in &skills {
        let version = skill.version.as_deref().unwrap_or("—");
        let desc: String = skill.description.chars().take(30).collect();
        println!(
            "{:<28} {:<24} {:<10} {}",
            skill.name, skill.layer, version, desc
        );
    }
    Ok(())
}

/// Show the details of a single skill.
pub async fn show_skill(name: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let loader = build_loader(&config)?;
    let skill = loader
        .get_skill(&name)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to load skill: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Skill '{name}' not found"))?;

    println!("Skill:        {}", skill.name);
    println!("  Description: {}", skill.description);
    println!("  Layer:       {}", skill.layer);
    println!("  Version:     {}", skill.version.as_deref().unwrap_or("—"));
    println!("  Author:      {}", skill.author.as_deref().unwrap_or("—"));
    println!("  Kind:        {:?}", skill.kind);
    println!("  Tags:        {}", skill.tags.join(", "));
    println!("  Steps:       {}", skill.steps.len());
    Ok(())
}

/// Install a skill from a GitHub repo, ClawHub, or a local directory.
pub async fn install_skill(source: String) -> Result<()> {
    let hub = build_hub()?;

    if let Some(local) = source.strip_prefix("path:") {
        let path = Path::new(local);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| local.to_string());
        let skill = hub
            .install_from_local(path, &name)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to install from local path: {e}"))?;
        println!(
            "Installed skill: {} v{}",
            skill.name,
            skill.version.as_deref().unwrap_or("?")
        );
        return Ok(());
    }

    if let Some(spec) = source.strip_prefix("github:") {
        let (repo, path) = match spec.split_once(':') {
            Some((r, p)) => (r, p),
            None => (spec, "."),
        };
        let name = repo.rsplit('/').next().unwrap_or(repo).to_string();
        let skill = hub
            .install_from_github(repo, path, &name)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to install from GitHub: {e}"))?;
        println!(
            "Installed skill: {} v{}",
            skill.name,
            skill.version.as_deref().unwrap_or("?")
        );
        return Ok(());
    }

    // Bare identifier: discover across registered sources, install the top hit.
    let metas = hub
        .discover(&source, None)
        .await
        .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;
    let meta = metas
        .first()
        .ok_or_else(|| anyhow::anyhow!("No skill found matching '{source}'"))?;
    println!(
        "Installing '{}' from {} ({})",
        meta.name, meta.source_id, meta.identifier
    );
    let result = hub
        .install(&meta.identifier, &meta.source_id, true)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to install skill: {e}"))?;
    if result.success {
        println!("Installed: {}", result.name);
        Ok(())
    } else {
        anyhow::bail!("Install failed: {}", result.message)
    }
}

/// Uninstall a skill by name.
pub async fn uninstall_skill(name: String) -> Result<()> {
    let hub = build_hub()?;
    hub.uninstall(&name)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to uninstall skill: {e}"))?;
    info!("Skill {name} uninstalled");
    println!("Uninstalled skill: {name}");
    Ok(())
}

/// Search the skill hub for skills matching a query.
pub async fn search_skills(query: String) -> Result<()> {
    let hub = build_hub()?;
    let metas = hub
        .discover(&query, None)
        .await
        .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;

    if metas.is_empty() {
        println!("No skills found for '{query}'.");
        return Ok(());
    }

    println!("Search results for '{query}':");
    println!("{:-<80}", "");
    for meta in &metas {
        print_skill_meta(meta);
        println!();
    }
    println!("{:-<80}", "");
    Ok(())
}

fn print_skill_meta(meta: &SkillMeta) {
    println!("  {} (v{})", meta.name, meta.version);
    println!(
        "    {}  [{}, trust: {}]",
        meta.description, meta.source_id, meta.trust_level
    );
    if !meta.tags.is_empty() {
        println!("    tags: {}", meta.tags.join(", "));
    }
}

/// Build a `SkillLoader` covering bundled, configured, and managed skills.
fn build_loader(config: &Config) -> Result<SkillLoader> {
    let loader = SkillLoader::new();

    if let Some(skills_cfg) = config.skills.as_ref() {
        for dir in &skills_cfg.skill_dirs {
            loader.register_layer_dir(SkillLayer::Extra, Path::new(dir).to_path_buf());
        }
    }

    let managed = util::skills_dir();
    std::fs::create_dir_all(&managed).ok();
    loader.register_layer_dir(SkillLayer::Managed, managed);

    let bundled = load_bundled_skills();
    loader.register_skills(bundled);

    loader
        .scan_all()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to scan skill directories: {e}"))?;
    Ok(loader)
}

/// Build a `SkillHub` with the GitHub source (and ClawHub if configured).
fn build_hub() -> Result<SkillHub> {
    let managed = util::skills_dir();
    std::fs::create_dir_all(&managed).ok();
    let mut hub = SkillHub::new(managed)
        .map_err(|e| anyhow::anyhow!("Failed to initialize skill hub: {e}"))?;

    let github_token = std::env::var("GITHUB_TOKEN").ok();
    hub.register_source(Arc::new(GitHubSource::new(github_token)));

    if let Ok(url) = std::env::var("CLAWHUB_URL") {
        match opensquilla_skills::hub::ClawHubSource::new(&url) {
            Ok(source) => {
                hub.register_source(Arc::new(source));
            }
            Err(e) => {
                anyhow::bail!("Invalid CLAWHUB_URL: {e}");
            }
        }
    }
    Ok(hub)
}
