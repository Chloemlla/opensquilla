use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Serializable distribution workspace state.
///
/// Persisted as `dist_state.json` (or TOML) inside the workspace/state
/// directory and read back on subsequent launches.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DistState {
    /// The current distribution version.
    pub version: String,
    /// The workspace root this state describes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_path: Option<PathBuf>,
    /// Whether the workspace has been bootstrapped.
    pub bootstrapped: bool,
    /// Whether a full uninstall was previously prepared.
    pub uninstall_prepared: bool,
    /// Last state update timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl DistState {
    /// Create a fresh state for the given version.
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            workspace_path: None,
            bootstrapped: false,
            uninstall_prepared: false,
            updated_at: Some(opensquilla_core::time::now()),
        }
    }

    /// Read the workspace state from a JSON file.
    pub fn read(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| {
            crate::Error::Io(format!("Failed to read {}: {e}", path.display()))
        })?;
        let mut state: Self = serde_json::from_str(&contents)
            .map_err(|e| crate::Error::Parse(format!("Invalid dist state: {e}")))?;
        state.updated_at = Some(opensquilla_core::time::now());
        Ok(state)
    }

    /// Write the workspace state to a JSON file, creating parent directories.
    pub fn write(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        self.write_json(path)
    }

    /// Write the workspace state to a JSON file.
    pub fn write_json(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let path = path.as_ref();
        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Parse(e.to_string()))?;
        write_state_file(path, &contents)
    }

    /// Write the workspace state to a TOML file.
    pub fn write_toml(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let path = path.as_ref();
        let contents = toml::to_string(self).map_err(|e| crate::Error::Parse(e.to_string()))?;
        write_state_file(path, &contents)
    }
}

fn write_state_file(path: &Path, contents: &str) -> crate::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            crate::Error::Io(format!("Failed to create {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(path, contents).map_err(|e| {
        crate::Error::Io(format!("Failed to write {}: {e}", path.display()))
    })?;
    Ok(())
}
