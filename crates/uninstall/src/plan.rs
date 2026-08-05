use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::inventory::{InventoryItem, InventoryItemKind};

/// A single deletion action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallAction {
    /// Path affected by this action.
    pub path: PathBuf,
    /// How the path is handled.
    pub kind: UninstallActionKind,
}

/// The kind of uninstall action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UninstallActionKind {
    /// Delete a single file.
    DeleteFile,
    /// Recursively delete a directory.
    DeleteDirectory,
    /// Leave the path in place.
    Keep,
}

/// A safe uninstall plan derived from an inventory scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallPlan {
    /// The actions to execute, in order.
    pub actions: Vec<UninstallAction>,
    /// Total bytes that would be freed.
    pub total_bytes: u64,
    /// Paths preserved (user data).
    pub kept_paths: Vec<PathBuf>,
    /// When the plan was generated.
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl UninstallPlan {
    /// Build a plan from an inventory scan.
    ///
    /// When `keep_user_data` is true, config and database files are preserved
    /// rather than deleted. Directories that contain files being deleted are
    /// removed implicitly and do not get their own action (avoids double
    /// deletes).
    pub fn from_inventory(items: Vec<InventoryItem>, keep_user_data: bool) -> Self {
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        let mut kept = Vec::new();
        let mut total = 0u64;

        for item in items {
            let preserve = keep_user_data
                && matches!(
                    item.kind,
                    InventoryItemKind::Config | InventoryItemKind::Database
                );
            if preserve {
                kept.push(item.path.clone());
                continue;
            }
            total += item.size_bytes;
            match item.kind {
                InventoryItemKind::Directory => dirs.push(item.path),
                _ => files.push(UninstallAction {
                    path: item.path,
                    kind: UninstallActionKind::DeleteFile,
                }),
            }
        }

        // Only emit a DeleteDirectory action for directories that do not
        // contain any file that is itself being removed.
        let file_paths: HashSet<PathBuf> = files.iter().map(|a| a.path.clone()).collect();
        let mut actions = files;
        for dir in dirs {
            let contains_removed_file = file_paths.iter().any(|p| p.starts_with(&dir));
            if !contains_removed_file {
                actions.push(UninstallAction {
                    path: dir,
                    kind: UninstallActionKind::DeleteDirectory,
                });
            }
        }

        Self {
            actions,
            total_bytes: total,
            kept_paths: kept,
            generated_at: opensquilla_core::time::now(),
        }
    }

    /// The number of actions in the plan.
    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Whether the plan has no work to do.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}
