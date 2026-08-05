use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// An item that may be removed during uninstall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryItem {
    /// Path to the item.
    pub path: PathBuf,
    /// Kind of item.
    pub kind: InventoryItemKind,
    /// Size in bytes (0 for directories).
    pub size_bytes: u64,
}

/// The kind of an inventory item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InventoryItemKind {
    File,
    Directory,
    Config,
    Database,
    Log,
    Cache,
}

/// Scan the installation at `base_dir` and return the removable items.
///
/// Walks the directory tree depth-first and classifies each entry.
pub fn scan_install(base_dir: impl AsRef<Path>) -> crate::Result<Vec<InventoryItem>> {
    let base = base_dir.as_ref();
    if !base.exists() {
        return Ok(Vec::new());
    }

    let mut items = Vec::new();
    let walker = walkdir::WalkDir::new(base)
        .min_depth(1)
        .sort_by_file_name();
    for entry in walker {
        let entry = entry.map_err(|e| crate::Error::Io(e.to_string()))?;
        let path = entry.path();
        if entry.file_type().is_dir() {
            items.push(InventoryItem {
                path: path.to_path_buf(),
                kind: InventoryItemKind::Directory,
                size_bytes: 0,
            });
        } else {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let kind = classify(path);
            items.push(InventoryItem {
                path: path.to_path_buf(),
                kind,
                size_bytes: size,
            });
        }
    }
    Ok(items)
}

/// Classify a file path by its extension/name.
fn classify(path: &Path) -> InventoryItemKind {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if name.ends_with(".db") || name.ends_with(".sqlite") || name.ends_with(".sqlite3") {
        InventoryItemKind::Database
    } else if name.ends_with(".log") {
        InventoryItemKind::Log
    } else if name.ends_with(".toml")
        || name.ends_with(".yaml")
        || name.ends_with(".yml")
        || name.ends_with(".json")
    {
        InventoryItemKind::Config
    } else if path.to_string_lossy().to_lowercase().contains("cache") {
        InventoryItemKind::Cache
    } else {
        InventoryItemKind::File
    }
}
