use std::path::{Path, PathBuf};

/// A discovered workspace root.
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Absolute path to the workspace root.
    pub root: PathBuf,
    /// Display name (last path component).
    pub name: String,
}

/// Marker files/directories that identify a workspace root.
const WORKSPACE_MARKERS: &[&str] = &[
    ".git",
    ".hg",
    "AGENTS.md",
    "SOUL.md",
    ".agents",
    "Cargo.toml",
    "pyproject.toml",
    "package.json",
];

impl Workspace {
    /// Discover a workspace by walking up from `start`, looking for a marker
    /// file. Falls back to the current directory when nothing is found.
    pub fn discover(start: impl AsRef<Path>) -> crate::Result<Self> {
        let start = start.as_ref();
        let start = if start.is_file() {
            start.parent().unwrap_or(start)
        } else {
            start
        };

        let mut dir = Some(start.to_path_buf());
        while let Some(current) = dir {
            if is_workspace_marker(&current) {
                return Ok(Self::from_path(current));
            }
            dir = current.parent().map(|p| p.to_path_buf());
        }

        // Fall back to the home directory.
        if let Some(home) = dirs::home_dir() {
            if home.exists() {
                return Ok(Self::from_path(home));
            }
        }

        Err(crate::Error::Workspace(format!(
            "No workspace marker found from {}",
            start.display()
        )))
    }

    /// Build a workspace from an explicit path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let root = path.into();
        let name = root
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "workspace".into());
        Self { root, name }
    }

    /// Expand `~` in a user-supplied path.
    pub fn expand_user(path: impl AsRef<str>) -> PathBuf {
        PathBuf::from(shellexpand::tilde(path.as_ref()).to_string())
    }
}

fn is_workspace_marker(dir: &Path) -> bool {
    WORKSPACE_MARKERS.iter().any(|m| dir.join(m).exists())
}
