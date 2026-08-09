//! Uninstall commands.
//!
//! Implements the `uninstall` subcommand against the uninstall crate's
//! inventory scanner and plan executor. Default posture: remove the program,
//! keep user data. Deletion of data is opt-in via `--purge-state` /
//! `--purge-config` / `--purge-all`. A total wipe (`--purge-all`) requires the
//! confirmation phrase on every surface.

use std::path::PathBuf;

use anyhow::{Context, Result};
use opensquilla_uninstall::inventory::{InventoryItem, InventoryItemKind, scan_install};
use opensquilla_uninstall::plan::UninstallPlan;
use opensquilla_uninstall::execute;

use crate::util;

/// The phrase required before `--purge-all` is honored.
pub const PURGE_ALL_CONFIRM_PHRASE: &str = "permanently delete all opensquilla data";

/// Uninstall subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum UninstallAction {
    /// Uninstall OpenSquilla (default: keep user data).
    Run {
        /// Show what would be removed and kept; do nothing.
        #[arg(long)]
        dry_run: bool,
        /// Skip the interactive confirmation prompt.
        #[arg(short, long)]
        yes: bool,
        /// Emit a machine-readable plan/result.
        #[arg(long)]
        json: bool,
        /// Also delete runtime state (sessions, scheduler, memory, logs, cache).
        #[arg(long)]
        purge_state: bool,
        /// Also delete configuration and secrets.
        #[arg(long)]
        purge_config: bool,
        /// Delete ALL OpenSquilla user data (implies state + config).
        #[arg(long)]
        purge_all: bool,
        /// Confirmation phrase required for --purge-all on non-interactive surfaces.
        #[arg(long)]
        confirm_purge_all: Option<String>,
    },
}

/// Run an uninstall subcommand.
pub async fn run_uninstall(action: UninstallAction) -> Result<()> {
    match action {
        UninstallAction::Run {
            dry_run,
            yes,
            json,
            purge_state,
            purge_config,
            purge_all,
            confirm_purge_all,
        } => {
            uninstall_run(
                dry_run,
                yes,
                json,
                purge_state,
                purge_config,
                purge_all,
                confirm_purge_all,
            )
            .await
        }
    }
}

struct PurgeOptions {
    purge_state: bool,
    purge_config: bool,
    purge_all: bool,
}

impl PurgeOptions {
    fn any(&self) -> bool {
        self.purge_state || self.purge_config || self.purge_all
    }
}

/// Scan the install, build a plan, and optionally execute it.
async fn uninstall_run(
    dry_run: bool,
    yes: bool,
    json: bool,
    purge_state: bool,
    purge_config: bool,
    purge_all: bool,
    confirm_purge_all: Option<String>,
) -> Result<()> {
    let opts = PurgeOptions {
        purge_state,
        purge_config,
        purge_all,
    };

    let base_dir = install_base_dir();
    let items = scan_install(&base_dir)
        .map_err(|e| anyhow::anyhow!("Failed to scan install at {}: {e}", base_dir.display()))?;
    let plan = build_plan(items, &opts);

    if dry_run {
        let actions = opensquilla_uninstall::dry_run(&plan);
        render_plan(&plan, &base_dir);
        if json {
            util::print_json(&plan_json(&plan, &base_dir, true))?;
        }
        let _ = actions;
        return Ok(());
    }

    // purge-all demands the typed phrase on every surface.
    if purge_all {
        let provided = confirm_purge_all.unwrap_or_default();
        if provided.trim() != PURGE_ALL_CONFIRM_PHRASE {
            anyhow::bail!(
                "--purge-all requires confirmation: pass --confirm-purge-all \"{}\".",
                PURGE_ALL_CONFIRM_PHRASE
            );
        }
    } else if !yes && json {
        anyhow::bail!(
            "Refusing to uninstall without --yes on a non-interactive (--json) surface. Re-run with --yes."
        );
    }

    if plan.is_empty() {
        println!("Nothing to uninstall at {}", base_dir.display());
        return Ok(());
    }

    execute(&plan).map_err(|e| anyhow::anyhow!("Uninstall execution failed: {e}"))?;

    if json {
        util::print_json(&plan_json(&plan, &base_dir, false))?;
    } else {
        println!("Uninstall complete. Removed {} item(s) from {}.", plan.action_count(), base_dir.display());
        if !plan.kept_paths.is_empty() {
            println!("Kept {} user-data path(s).", plan.kept_paths.len());
        }
    }
    Ok(())
}

/// Resolve the install base directory to scan.
fn install_base_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OPENSQUILLA_UNINSTALL_BASE") {
        return PathBuf::from(p);
    }
    util::data_dir()
}

/// Build an [`UninstallPlan`] honoring the purge flags.
///
/// The uninstall crate's `from_inventory` keeps both config and database when
/// `keep_user_data` is true, which is too coarse for separate state/config
/// purges. We therefore filter the inventory first so only the categories the
/// caller wants removed reach the plan builder.
fn build_plan(items: Vec<InventoryItem>, opts: &PurgeOptions) -> UninstallPlan {
    let to_delete: Vec<InventoryItem> = items
        .into_iter()
        .filter(|item| should_delete(item.kind, opts))
        .collect();
    UninstallPlan::from_inventory(to_delete, false)
}

fn should_delete(kind: InventoryItemKind, opts: &PurgeOptions) -> bool {
    match kind {
        InventoryItemKind::File | InventoryItemKind::Directory => true,
        InventoryItemKind::Config => opts.purge_config || opts.purge_all,
        InventoryItemKind::Database => opts.purge_state || opts.purge_all,
        InventoryItemKind::Log | InventoryItemKind::Cache => opts.purge_state || opts.purge_all,
    }
}

fn plan_json(plan: &UninstallPlan, base: &PathBuf, dry: bool) -> serde_json::Value {
    let actions: Vec<_> = plan
        .actions
        .iter()
        .map(|a| serde_json::json!({ "kind": format!("{:?}", a.kind), "path": a.path.display().to_string() }))
        .collect();
    serde_json::json!({
        "dry_run": dry,
        "base": base.display().to_string(),
        "action_count": plan.action_count(),
        "total_bytes": plan.total_bytes,
        "kept_paths": plan.kept_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "actions": actions,
    })
}

fn render_plan(plan: &UninstallPlan, base: &PathBuf) {
    println!("Uninstall plan for {}", base.display());
    println!("{:-<60}", "");
    println!("  Will delete: {} item(s)", plan.action_count());
    println!("  Will free:   {} bytes", util::human_bytes(plan.total_bytes));
    for action in &plan.actions {
        println!("    - {} {}", format!("{:?}", action.kind), action.path.display());
    }
    if !plan.kept_paths.is_empty() {
        println!("  Will keep {} user-data path(s):", plan.kept_paths.len());
        for p in &plan.kept_paths {
            println!("    + {}", p.display());
        }
    }
    println!("{:-<60}", "");
}

/// Convenience serialization for the dry-run action list.
#[allow(dead_code)]
fn action_kind_str(kind: opensquilla_uninstall::plan::UninstallActionKind) -> &'static str {
    match kind {
        opensquilla_uninstall::plan::UninstallActionKind::DeleteFile => "delete_file",
        opensquilla_uninstall::plan::UninstallActionKind::DeleteDirectory => "delete_directory",
        opensquilla_uninstall::plan::UninstallActionKind::Keep => "keep",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(purge_state: bool, purge_config: bool, purge_all: bool) -> PurgeOptions {
        PurgeOptions {
            purge_state,
            purge_config,
            purge_all,
        }
    }

    #[test]
    fn test_should_delete_default_keeps_data() {
        assert!(should_delete(InventoryItemKind::File, &opts(false, false, false)));
        assert!(!should_delete(InventoryItemKind::Config, &opts(false, false, false)));
        assert!(!should_delete(InventoryItemKind::Database, &opts(false, false, false)));
        assert!(!should_delete(InventoryItemKind::Log, &opts(false, false, false)));
    }

    #[test]
    fn test_should_delete_purge_state() {
        let o = opts(true, false, false);
        assert!(should_delete(InventoryItemKind::Database, &o));
        assert!(should_delete(InventoryItemKind::Log, &o));
        assert!(!should_delete(InventoryItemKind::Config, &o));
    }

    #[test]
    fn test_should_delete_purge_config() {
        let o = opts(false, true, false);
        assert!(should_delete(InventoryItemKind::Config, &o));
        assert!(!should_delete(InventoryItemKind::Database, &o));
    }

    #[test]
    fn test_should_delete_purge_all() {
        let o = opts(false, false, true);
        assert!(should_delete(InventoryItemKind::Config, &o));
        assert!(should_delete(InventoryItemKind::Database, &o));
        assert!(should_delete(InventoryItemKind::File, &o));
    }
}