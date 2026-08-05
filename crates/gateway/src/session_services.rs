//! Session service dependency injection.
//!
//! Mirrors the Python `session_services.py` module. Provides a service
//! locator ([`SessionServices`]) that bundles the shared dependencies a
//! session needs at runtime: storage, memory, provider routing, tools, and
//! channels.
//!
//! The gateway crate deliberately does not depend on the `opensquilla-provider`
//! crate directly (providers are owned by the engine runtime). Instead the
//! provider surface is exposed here through a trait object
//! ([`SessionProviderHandle`]) so that the engine can inject a concrete
//! provider registry while the session layer stays decoupled.

use std::sync::Arc;

use opensquilla_core::error::AppError;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::channels::ChannelsService;
use crate::memory::MemoryHandle;
use crate::sessions::SessionStore;
use crate::tools::ToolsService;

/// Opaque handle to the provider layer as seen by the session services.
///
/// The concrete implementation lives in the engine runtime; this trait lets
/// the gateway reference providers without taking a compile-time dependency
/// on the provider crate.
pub trait SessionProviderHandle: Send + Sync {
    /// Return the list of model identifiers currently available.
    fn available_models(&self) -> Vec<String>;

    /// Return the default model id, if one is configured.
    fn default_model(&self) -> Option<String>;

    /// Resolve a model id to its provider name, if known.
    fn provider_for(&self, model: &str) -> Option<String>;
}

/// A no-op provider handle used as a default and in tests.
#[derive(Debug, Default, Clone)]
pub struct NullProviderHandle;

impl SessionProviderHandle for NullProviderHandle {
    fn available_models(&self) -> Vec<String> {
        Vec::new()
    }
    fn default_model(&self) -> Option<String> {
        None
    }
    fn provider_for(&self, _model: &str) -> Option<String> {
        None
    }
}

/// Snapshot of the provider state for serialization in API responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSnapshot {
    pub available_models: Vec<String>,
    pub default_model: Option<String>,
}

/// The set of services a session needs at runtime.
///
/// Acts as a service locator: each field is an `Arc` so the struct can be
/// cloned cheaply and handed to per-session tasks. Build one with
/// [`SessionServicesBuilder`] and then clone it into each session.
#[derive(Clone)]
pub struct SessionServices {
    /// In-memory session metadata store (RPC layer).
    pub storage: Arc<SessionStore>,
    /// Memory subsystem handle.
    pub memory: Arc<MemoryHandle>,
    /// Provider routing surface.
    pub providers: Arc<dyn SessionProviderHandle>,
    /// Tool registry for agent tool calls.
    pub tools: Arc<ToolsService>,
    /// Channel manager for outbound messaging.
    pub channels: Arc<ChannelsService>,
}

impl SessionServices {
    /// Create a new builder for assembling a service bundle.
    pub fn builder() -> SessionServicesBuilder {
        SessionServicesBuilder::default()
    }

    /// Return a snapshot of the provider state for API responses.
    pub fn provider_snapshot(&self) -> ProviderSnapshot {
        ProviderSnapshot {
            available_models: self.providers.available_models(),
            default_model: self.providers.default_model(),
        }
    }

    /// Validate that the configured default model is actually available.
    pub fn validate_default_model(&self) -> Result<(), AppError> {
        let Some(default) = self.providers.default_model() else {
            return Ok(());
        };
        let models = self.providers.available_models();
        if models.iter().any(|m| m == &default) {
            Ok(())
        } else {
            Err(AppError::internal(format!(
                "Configured default model '{default}' is not in the available models list"
            )))
        }
    }
}

impl std::fmt::Debug for SessionServices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionServices")
            .field("storage", &"SessionStore")
            .field("memory", &"MemoryHandle")
            .field("providers", &"dyn SessionProviderHandle")
            .field("tools", &"ToolsService")
            .field("channels", &"ChannelsService")
            .finish()
    }
}

/// Builder for [`SessionServices`].
///
/// Provides a service-locator pattern: each dependency is set independently,
/// with sensible (null) defaults so a partially-configured bundle can still
/// be constructed for testing.
#[derive(Default)]
pub struct SessionServicesBuilder {
    storage: Option<Arc<SessionStore>>,
    memory: Option<Arc<MemoryHandle>>,
    providers: Option<Arc<dyn SessionProviderHandle>>,
    tools: Option<Arc<ToolsService>>,
    channels: Option<Arc<ChannelsService>>,
}

impl SessionServicesBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the session storage.
    pub fn with_storage(mut self, storage: Arc<SessionStore>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the memory handle.
    pub fn with_memory(mut self, memory: Arc<MemoryHandle>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Set the provider handle.
    pub fn with_providers(mut self, providers: Arc<dyn SessionProviderHandle>) -> Self {
        self.providers = Some(providers);
        self
    }

    /// Set the tools service.
    pub fn with_tools(mut self, tools: Arc<ToolsService>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the channels service.
    pub fn with_channels(mut self, channels: Arc<ChannelsService>) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Assemble the service bundle, filling any unset field with a default.
    pub fn build(self) -> SessionServices {
        // Memory handle needs to be constructed; fall back to an in-memory
        // store if the caller did not supply one.
        let memory = match self.memory {
            Some(m) => m,
            None => Arc::new(
                MemoryHandle::in_memory().expect("in-memory memory store should always construct"),
            ),
        };
        SessionServices {
            storage: self
                .storage
                .unwrap_or_else(|| Arc::new(SessionStore::new())),
            memory,
            providers: self
                .providers
                .unwrap_or_else(|| Arc::new(NullProviderHandle)),
            tools: self.tools.unwrap_or_else(|| Arc::new(ToolsService::new())),
            channels: self
                .channels
                .unwrap_or_else(|| Arc::new(ChannelsService::new())),
        }
    }
}

/// A registry of per-session service bundles.
///
/// Useful when the gateway needs to look up the services for a specific
/// session id (service-locator pattern). Sessions that do not have an
/// explicit bundle fall back to a shared default.
#[derive(Clone)]
pub struct SessionServiceRegistry {
    default: Arc<SessionServices>,
    overrides: Arc<RwLock<std::collections::HashMap<String, Arc<SessionServices>>>>,
}

impl SessionServiceRegistry {
    /// Create a new registry with the given default service bundle.
    pub fn new(default: SessionServices) -> Self {
        Self {
            default: Arc::new(default),
            overrides: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Register a per-session override.
    pub fn set(&self, session_id: impl Into<String>, services: Arc<SessionServices>) {
        self.overrides.write().insert(session_id.into(), services);
    }

    /// Look up the services for a session, falling back to the default.
    pub fn get(&self, session_id: &str) -> Arc<SessionServices> {
        self.overrides
            .read()
            .get(session_id)
            .cloned()
            .unwrap_or_else(|| self.default.clone())
    }

    /// Remove a per-session override.
    pub fn remove(&self, session_id: &str) -> Option<Arc<SessionServices>> {
        self.overrides.write().remove(session_id)
    }

    /// Return the default service bundle.
    pub fn default_services(&self) -> &Arc<SessionServices> {
        &self.default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let services = SessionServicesBuilder::new().build();
        assert!(services.providers.default_model().is_none());
        assert!(services.providers.available_models().is_empty());
    }

    #[test]
    fn test_builder_with_providers() {
        struct StaticProvider;
        impl SessionProviderHandle for StaticProvider {
            fn available_models(&self) -> Vec<String> {
                vec!["gpt-4o".into(), "claude-sonnet-4".into()]
            }
            fn default_model(&self) -> Option<String> {
                Some("gpt-4o".into())
            }
            fn provider_for(&self, model: &str) -> Option<String> {
                match model {
                    "gpt-4o" => Some("openai".into()),
                    "claude-sonnet-4" => Some("anthropic".into()),
                    _ => None,
                }
            }
        }
        let services = SessionServices::builder()
            .with_providers(Arc::new(StaticProvider))
            .build();
        let snapshot = services.provider_snapshot();
        assert_eq!(snapshot.default_model.as_deref(), Some("gpt-4o"));
        assert_eq!(snapshot.available_models.len(), 2);
        assert!(services.validate_default_model().is_ok());
    }

    #[test]
    fn test_validate_default_model_missing() {
        struct BrokenProvider;
        impl SessionProviderHandle for BrokenProvider {
            fn available_models(&self) -> Vec<String> {
                vec!["claude-sonnet-4".into()]
            }
            fn default_model(&self) -> Option<String> {
                Some("gpt-4o".into())
            }
            fn provider_for(&self, _: &str) -> Option<String> {
                None
            }
        }
        let services = SessionServices::builder()
            .with_providers(Arc::new(BrokenProvider))
            .build();
        assert!(services.validate_default_model().is_err());
    }

    #[test]
    fn test_registry_override() {
        let default = SessionServicesBuilder::new().build();
        let registry = SessionServiceRegistry::new(default);

        struct Custom;
        impl SessionProviderHandle for Custom {
            fn available_models(&self) -> Vec<String> {
                vec!["custom-model".into()]
            }
            fn default_model(&self) -> Option<String> {
                None
            }
            fn provider_for(&self, _: &str) -> Option<String> {
                None
            }
        }
        let custom = Arc::new(
            SessionServices::builder()
                .with_providers(Arc::new(Custom))
                .build(),
        );
        registry.set("s1", custom);

        assert_eq!(
            registry.get("s1").providers.available_models(),
            vec!["custom-model".to_string()]
        );
        // Unknown session falls back to default.
        assert!(registry.get("s2").providers.available_models().is_empty());
        assert!(registry.remove("s1").is_some());
        assert!(registry.get("s1").providers.available_models().is_empty());
    }

    #[test]
    fn test_null_provider_handle_default() {
        let handle = NullProviderHandle;
        assert!(handle.available_models().is_empty());
        assert!(handle.default_model().is_none());
        assert!(handle.provider_for("anything").is_none());
    }
}
