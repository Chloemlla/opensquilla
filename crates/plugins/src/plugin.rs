use serde::{Deserialize, Serialize};

/// Metadata describing a loaded plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Unique plugin identifier.
    pub id: String,
    /// Human-readable plugin name.
    pub name: String,
    /// Plugin version.
    pub version: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// The plugin trait implemented by TokenJuice plugins.
///
/// A plugin is a self-contained unit of behavior that can be loaded, used, and
/// unloaded at runtime without restarting the gateway.
pub trait Plugin: Send + Sync {
    /// Unique plugin identifier.
    fn id(&self) -> &str;

    /// Human-readable plugin name.
    fn name(&self) -> &str;

    /// Plugin version.
    fn version(&self) -> &str;

    /// Optional plugin description.
    fn description(&self) -> Option<&str> {
        None
    }

    /// Called when the plugin is loaded into the registry.
    fn on_load(&self) {}

    /// Called when the plugin is unloaded from the registry.
    fn on_unload(&self) {}

    /// Describe this plugin.
    fn info(&self) -> PluginInfo {
        PluginInfo {
            id: self.id().to_string(),
            name: self.name().to_string(),
            version: self.version().to_string(),
            description: self.description().map(|d| d.to_string()),
        }
    }
}
