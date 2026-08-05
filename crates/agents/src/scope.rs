use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A capability scope granted to an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentScope {
    /// Full access to all capabilities.
    Full,
    /// Write access to a defined set of workspaces only.
    Workspace,
    /// Read-only access.
    ReadOnly,
    /// No capabilities beyond basic messaging.
    Restricted,
}

impl AgentScope {
    /// Whether the agent may write to the file system.
    pub fn allows_fs_write(&self) -> bool {
        matches!(self, AgentScope::Full | AgentScope::Workspace)
    }

    /// Whether the agent may access the network.
    pub fn allows_network(&self) -> bool {
        matches!(self, AgentScope::Full)
    }

    /// Whether the agent may invoke tools at all.
    pub fn allows_tools(&self) -> bool {
        !matches!(self, AgentScope::Restricted)
    }

    /// A stable string identifier for this scope.
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentScope::Full => "full",
            AgentScope::Workspace => "workspace",
            AgentScope::ReadOnly => "read_only",
            AgentScope::Restricted => "restricted",
        }
    }
}

impl std::fmt::Display for AgentScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Scoped workspace directories an agent may access.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeConfig {
    /// Directories the agent may write to.
    pub writable_paths: Vec<PathBuf>,
    /// Directories the agent may read from.
    pub readable_paths: Vec<PathBuf>,
    /// Whether the agent may spawn subprocesses.
    pub allow_subprocess: bool,
}

impl ScopeConfig {
    /// Whether the agent may write to the given path.
    pub fn can_write(&self, path: &Path) -> bool {
        self.writable_paths.iter().any(|p| path.starts_with(p))
    }

    /// Whether the agent may read the given path.
    pub fn can_read(&self, path: &Path) -> bool {
        self.can_write(path) || self.readable_paths.iter().any(|p| path.starts_with(p))
    }

    /// Add a writable directory.
    pub fn with_writable(mut self, path: impl Into<PathBuf>) -> Self {
        self.writable_paths.push(path.into());
        self
    }

    /// Add a readable directory.
    pub fn with_readable(mut self, path: impl Into<PathBuf>) -> Self {
        self.readable_paths.push(path.into());
        self
    }
}
