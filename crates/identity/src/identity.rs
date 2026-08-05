use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Terminal identity: identifies the user/agent pair in a terminal session.
///
/// Persisted as a JSON file (e.g. `$STATE_DIR/identity.json`) and used to
/// seed prompts, workspaces, and bootstrap state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// The identity name (e.g. `alice@workstation`).
    pub name: String,
    /// The software version that created this identity.
    pub version: String,
    /// The workspace path associated with this identity.
    pub workspace_path: PathBuf,
    /// When this identity was created.
    pub created_at: DateTime<Utc>,
    /// Optional machine hostname.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
}

impl Identity {
    /// Create a new identity with the current timestamp.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        workspace_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            workspace_path: workspace_path.into(),
            created_at: Utc::now(),
            hostname: hostname_from_env(),
        }
    }

    /// Load an identity from a JSON file at the given path.
    pub fn load(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| crate::Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        serde_json::from_str(&contents).map_err(|e| {
            crate::Error::Parse(format!("Failed to parse identity {}: {e}", path.display()))
        })
    }

    /// Save the identity to a JSON file at the given path, creating parent
    /// directories as needed.
    pub fn save(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let path = path.as_ref();
        let contents =
            serde_json::to_string_pretty(self).map_err(|e| crate::Error::Parse(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| crate::Error::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        std::fs::write(path, contents).map_err(|e| crate::Error::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Ok(())
    }

    /// Render the prompt line (e.g. `alice@workstation ~/repo $ `) using the
    /// given prompt template.
    pub fn display_prompt(&self, template: &crate::PromptTemplate) -> String {
        template.render(self)
    }
}

fn hostname_from_env() -> Option<String> {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .ok()
        .filter(|s| !s.is_empty())
}
