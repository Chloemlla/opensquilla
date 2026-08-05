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
#[derive(Default)]
pub struct AgentRegistry {
    agents: DashMap<AgentId, AgentDefinition>,
}

impl AgentRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            agents: DashMap::new(),
        }
    }

    /// Register an agent definition.
    pub fn register(&self, def: AgentDefinition) -> crate::Result<()> {
        if self.agents.contains_key(&def.id) {
            return Err(crate::Error::AlreadyRegistered(def.id.to_string()));
        }
        self.agents.insert(def.id, def);
        Ok(())
    }

    /// Unregister an agent by id.
    pub fn unregister(&self, id: &AgentId) -> crate::Result<()> {
        self.agents
            .remove(id)
            .map(|_| ())
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
