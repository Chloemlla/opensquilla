use std::path::{Path, PathBuf};

use dashmap::DashMap;
use opensquilla_core::types::AgentId;
use serde::{Deserialize, Serialize};

use crate::limits::AgentLimits;
use crate::scope::{AgentScope, ScopeConfig};

/// A registered agent definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Unique agent id.
    pub id: AgentId,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The capability scope granted to this agent.
    pub scope: AgentScope,
    /// Resource limits for this agent.
    pub limits: AgentLimits,
    /// Scoped path access.
    pub scope_config: ScopeConfig,
    /// Explicit capability names the agent may use (`*` means all).
    #[serde(default)]
    pub capabilities: Vec<String>,
}

impl AgentDefinition {
    /// Create a new agent definition with restricted defaults.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: AgentId::new(),
            name: name.into(),
            description: None,
            scope: AgentScope::Restricted,
            limits: AgentLimits::restricted(),
            scope_config: ScopeConfig::default(),
            capabilities: Vec::new(),
        }
    }

    /// Grant a scope to this agent.
    pub fn with_scope(mut self, scope: AgentScope) -> Self {
        self.scope = scope;
        self
    }

    /// Add a capability name to this agent.
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }
}

/// Thread-safe registry of agent definitions, keyed by agent id.
///
/// By default a registry is in-memory only. Enable disk persistence with
/// [`AgentRegistry::with_persistence`]; mutations then flush the registry to a
/// JSON file at the configured path, mirroring the Python registry's
/// `persist_changes` behavior.
pub struct AgentRegistry {
    agents: DashMap<AgentId, AgentDefinition>,
    /// When set, mutations flush the registry JSON to this path.
    config_path: Option<PathBuf>,
    /// When `false`, mutations skip the disk flush even if `config_path` is set.
    /// Defaults to `true` (matches the Python registry).
    persist_changes: bool,
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentRegistry {
    /// Create an empty in-memory registry (no disk persistence).
    pub fn new() -> Self {
        Self {
            agents: DashMap::new(),
            config_path: None,
            persist_changes: true,
        }
    }

    /// Create a registry that persists mutations to `path`.
    ///
    /// Equivalent to the Python registry constructed with
    /// `persist_changes=True, config_path=path`. The file is *not* read here;
    /// use [`AgentRegistry::load_from_disk`] to hydrate from an existing file.
    pub fn with_persistence(path: PathBuf) -> Self {
        Self {
            agents: DashMap::new(),
            config_path: Some(path),
            persist_changes: true,
        }
    }

    /// Whether this registry flushes mutations to disk.
    pub fn persists(&self) -> bool {
        self.persist_changes && self.config_path.is_some()
    }

    /// The configured persistence path, if any.
    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    /// Serialize the current registry contents to the configured path.
    ///
    /// Returns an error if persistence is disabled or the write fails.
    /// Holding a DashMap read guard across the file write is safe here:
    /// the write is synchronous and no register/unregister caller can
    /// deadlock against this method (they call it *after* releasing their
    /// own write guard).
    pub fn save_to_disk(&self) -> crate::Result<()> {
        let Some(path) = self.config_path.as_deref() else {
            return Err(crate::Error::Persistence(
                "no config_path configured".into(),
            ));
        };
        let snapshot: Vec<AgentDefinition> = self.agents.iter().map(|r| r.clone()).collect();
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| crate::Error::Persistence(e.to_string()))?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| crate::Error::Persistence(e.to_string()))?;
            }
        }
        std::fs::write(path, json).map_err(|e| crate::Error::Persistence(e.to_string()))?;
        Ok(())
    }

    /// Load a registry from a JSON file written by [`save_to_disk`].
    ///
    /// The returned registry inherits `path` as its `config_path` with
    /// `persist_changes = true`, so subsequent mutations continue to flush
    /// to the same file. A missing file is treated as an empty registry
    /// (the next mutation will create it) rather than an error, matching
    /// first-run behavior.
    pub fn load_from_disk(path: &Path) -> crate::Result<Self> {
        let registry = Self::with_persistence(path.to_path_buf());
        match std::fs::read_to_string(path) {
            Ok(data) if data.trim().is_empty() => Ok(registry),
            Ok(data) => {
                let agents: Vec<AgentDefinition> = serde_json::from_str(&data)
                    .map_err(|e| crate::Error::Persistence(e.to_string()))?;
                for def in agents {
                    registry.agents.insert(def.id, def);
                }
                Ok(registry)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(registry),
            Err(e) => Err(crate::Error::Persistence(e.to_string())),
        }
    }

    /// Best-effort flush used by mutations. A failed flush logs a warning
    /// but does *not* fail the caller — the in-memory mutation has already
    /// succeeded. This mirrors the Python registry's fire-and-forget `_persist`.
    fn persist_if_enabled(&self) {
        if !self.persists() {
            return;
        }
        if let Err(e) = self.save_to_disk() {
            tracing::warn!(error = %e, "agent registry persistence failed");
        }
    }

    /// Register an agent definition.
    pub fn register(&self, def: AgentDefinition) -> crate::Result<()> {
        if self.agents.contains_key(&def.id) {
            return Err(crate::Error::AlreadyRegistered(def.id.to_string()));
        }
        self.agents.insert(def.id, def);
        self.persist_if_enabled();
        Ok(())
    }

    /// Unregister an agent by id.
    pub fn unregister(&self, id: &AgentId) -> crate::Result<()> {
        self.agents
            .remove(id)
            .map(|_| {
                self.persist_if_enabled();
            })
            .ok_or_else(|| crate::Error::NotFound(id.to_string()))
    }

    /// Get a clone of an agent definition.
    pub fn get(&self, id: &AgentId) -> Option<AgentDefinition> {
        self.agents.get(id).map(|r| r.clone())
    }

    /// Get a mutable reference to an agent definition.
    pub fn get_mut(
        &self,
        id: &AgentId,
    ) -> Option<dashmap::mapref::one::RefMut<'_, AgentId, AgentDefinition>> {
        self.agents.get_mut(id)
    }

    /// List all registered agents.
    pub fn list(&self) -> Vec<AgentDefinition> {
        self.agents.iter().map(|r| r.clone()).collect()
    }

    /// The number of registered agents.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    /// Whether no agents are registered.
    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Whether the agent is allowed to use the named tool, considering both
    /// its scope and its explicit capability list.
    pub fn can_use_tool(&self, id: &AgentId, tool_name: &str) -> bool {
        let Some(def) = self.get(id) else {
            return false;
        };
        if !def.scope.allows_tools() {
            return false;
        }
        def.capabilities.iter().any(|c| c == tool_name || c == "*")
    }

    /// Whether the agent may write to the given path.
    pub fn can_write(&self, id: &AgentId, path: &std::path::Path) -> bool {
        self.get(id)
            .map(|def| def.scope.allows_fs_write() && def.scope_config.can_write(path))
            .unwrap_or(false)
    }
}
