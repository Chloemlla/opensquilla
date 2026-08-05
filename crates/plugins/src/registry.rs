use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::plugin::{Plugin, PluginInfo};

/// Thread-safe registry of loaded plugins.
///
/// Registration is keyed by plugin id; a plugin can be loaded only once.
#[derive(Default)]
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, Arc<dyn Plugin>>>,
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
        }
    }

    /// Load a plugin into the registry.
    pub fn register(&self, plugin: Arc<dyn Plugin>) -> crate::Result<()> {
        let id = plugin.id().to_string();
        let mut plugins = self.plugins.write().map_err(|_| crate::Error::Poisoned)?;
        if plugins.contains_key(&id) {
            return Err(crate::Error::AlreadyLoaded(id));
        }
        plugin.on_load();
        plugins.insert(id, plugin);
        Ok(())
    }

    /// Unload a plugin by id.
    pub fn unregister(&self, id: &str) -> crate::Result<()> {
        let mut plugins = self.plugins.write().map_err(|_| crate::Error::Poisoned)?;
        let plugin = plugins
            .remove(id)
            .ok_or_else(|| crate::Error::NotFound(id.to_string()))?;
        plugin.on_unload();
        Ok(())
    }

    /// Look up a loaded plugin by id.
    pub fn get(&self, id: &str) -> Option<Arc<dyn Plugin>> {
        self.plugins.read().ok()?.get(id).cloned()
    }

    /// List all loaded plugins.
    pub fn list(&self) -> Vec<PluginInfo> {
        self.plugins
            .read()
            .map(|plugins| plugins.values().map(|p| p.info()).collect())
            .unwrap_or_default()
    }

    /// The number of loaded plugins.
    pub fn len(&self) -> usize {
        self.plugins.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Whether no plugins are loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether a plugin with the given id is loaded.
    pub fn contains(&self, id: &str) -> bool {
        self.plugins
            .read()
            .map(|p| p.contains_key(id))
            .unwrap_or(false)
    }
}
