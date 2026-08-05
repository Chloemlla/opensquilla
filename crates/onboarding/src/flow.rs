use std::collections::HashMap;

use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::channels::ChannelSetup;
use crate::providers::ProviderSpec;
use crate::storage::ConfigStorage;

/// State of the setup wizard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupState {
    /// Welcome screen
    Welcome,
    /// Provider configuration
    ProviderSelection,
    /// Provider API key setup
    ProviderAuth,
    /// Model selection
    ModelSelection,
    /// Channel configuration
    ChannelSetup,
    /// Sandbox configuration
    SandboxConfig,
    /// Review and confirm
    Review,
    /// Setup complete
    Complete,
    /// Setup cancelled
    Cancelled,
}

/// A step in the setup flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupStep {
    pub state: SetupState,
    pub title: String,
    pub description: String,
    pub completed: bool,
    pub skipped: bool,
}

/// The setup flow wizard for initial configuration.
#[derive(Debug, Clone)]
pub struct SetupFlow {
    config: Config,
    current_state: SetupState,
    steps: Vec<SetupStep>,
    selected_provider: Option<ProviderSpec>,
    channel_configs: HashMap<String, ChannelSetup>,
    errors: Vec<String>,
}

impl SetupFlow {
    /// Create a new setup flow.
    pub fn new(config: &Config) -> Self {
        let steps = vec![
            SetupStep {
                state: SetupState::Welcome,
                title: "Welcome".to_string(),
                description: "Welcome to OpenSquilla! Let's get you set up.".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::ProviderSelection,
                title: "Provider Selection".to_string(),
                description: "Choose an AI provider to use.".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::ProviderAuth,
                title: "API Key Setup".to_string(),
                description: "Enter your API key for the selected provider.".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::ModelSelection,
                title: "Model Selection".to_string(),
                description: "Choose which model to use by default.".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::ChannelSetup,
                title: "Channel Setup".to_string(),
                description: "Connect messaging channels (optional).".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::SandboxConfig,
                title: "Sandbox Configuration".to_string(),
                description: "Configure sandbox execution environment (optional).".to_string(),
                completed: false,
                skipped: false,
            },
            SetupStep {
                state: SetupState::Review,
                title: "Review".to_string(),
                description: "Review your configuration before saving.".to_string(),
                completed: false,
                skipped: false,
            },
        ];

        Self {
            config: config.clone(),
            current_state: SetupState::Welcome,
            steps,
            selected_provider: None,
            channel_configs: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Get the current flow state.
    pub fn current_state(&self) -> SetupState {
        self.current_state
    }

    /// Get all steps.
    pub fn steps(&self) -> &[SetupStep] {
        &self.steps
    }

    /// Advance to the next step.
    pub fn advance(&mut self) -> Result<(), OnboardingError> {
        let next = match self.current_state {
            SetupState::Welcome => SetupState::ProviderSelection,
            SetupState::ProviderSelection => {
                if self.selected_provider.is_none() {
                    return Err(OnboardingError::ValidationError(
                        "No provider selected".to_string(),
                    ));
                }
                SetupState::ProviderAuth
            }
            SetupState::ProviderAuth => {
                if !self.has_api_key() {
                    return Err(OnboardingError::ValidationError(
                        "API key not provided".to_string(),
                    ));
                }
                SetupState::ModelSelection
            }
            SetupState::ModelSelection => SetupState::ChannelSetup,
            SetupState::ChannelSetup => SetupState::SandboxConfig,
            SetupState::SandboxConfig => SetupState::Review,
            SetupState::Review => SetupState::Complete,
            SetupState::Complete | SetupState::Cancelled => {
                return Err(OnboardingError::FlowEnded);
            }
        };

        // Mark current step as completed
        if let Some(step) = self
            .steps
            .iter_mut()
            .find(|s| s.state == self.current_state)
        {
            step.completed = true;
        }

        self.current_state = next;
        info!("Setup flow advanced to {:?}", self.current_state);
        Ok(())
    }

    /// Go back to the previous step.
    pub fn go_back(&mut self) {
        let prev = match self.current_state {
            SetupState::ProviderSelection => SetupState::Welcome,
            SetupState::ProviderAuth => SetupState::ProviderSelection,
            SetupState::ModelSelection => SetupState::ProviderAuth,
            SetupState::ChannelSetup => SetupState::ModelSelection,
            SetupState::SandboxConfig => SetupState::ChannelSetup,
            SetupState::Review => SetupState::SandboxConfig,
            _ => return,
        };

        self.current_state = prev;
        debug!("Setup flow went back to {:?}", self.current_state);
    }

    /// Skip the current step.
    pub fn skip(&mut self) {
        if let Some(step) = self
            .steps
            .iter_mut()
            .find(|s| s.state == self.current_state)
        {
            step.skipped = true;
        }
        let _ = self.advance();
        info!("Skipped step {:?}", self.current_state);
    }

    /// Select a provider.
    pub fn select_provider(&mut self, provider: ProviderSpec) {
        self.selected_provider = Some(provider);
        debug!("Provider selected");
    }

    /// Set the API key for the selected provider.
    pub fn set_api_key(&mut self, key: &str) {
        if let Some(ref provider) = self.selected_provider {
            self.config
                .set(&format!("provider.{}.api_key", provider.name), key)
                .ok();
            info!("API key set for provider {}", provider.name);
        }
    }

    /// Check if the API key is set.
    pub fn has_api_key(&self) -> bool {
        if let Some(ref provider) = self.selected_provider {
            !self
                .config
                .get(&format!("provider.{}.api_key", provider.name))
                .unwrap_or_default()
                .is_empty()
        } else {
            false
        }
    }

    /// Set the default model.
    pub fn set_default_model(&mut self, model: &str) {
        self.config.set("model.default", model).ok();
        info!("Default model set to {model}");
    }

    /// Add a channel configuration.
    pub fn add_channel(&mut self, name: &str, setup: ChannelSetup) {
        self.channel_configs.insert(name.to_string(), setup);
        debug!("Channel {name} configured");
    }

    /// Get the current error messages.
    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Complete the setup and save configuration.
    pub async fn complete(mut self) -> Result<(), OnboardingError> {
        if self.current_state != SetupState::Review {
            return Err(OnboardingError::ValidationError(
                "Cannot complete before reaching the review step".to_string(),
            ));
        }

        // Save channel configurations
        for (name, setup) in &self.channel_configs {
            let config_key = format!("channel.{}.enabled", name);
            self.config.set(&config_key, "true").ok();

            for (key, value) in &setup.settings {
                let channel_key = format!("channel.{name}.{key}");
                self.config.set(&channel_key, value).ok();
            }
        }

        // Mark review as complete
        if let Some(step) = self
            .steps
            .iter_mut()
            .find(|s| s.state == self.current_state)
        {
            step.completed = true;
        }

        self.current_state = SetupState::Complete;

        // Save to storage
        let storage = ConfigStorage::new(&self.config)?;
        storage.save_config(&self.config).await?;

        info!("Setup completed successfully");
        Ok(())
    }

    /// Cancel the setup flow.
    pub fn cancel(&mut self) {
        self.current_state = SetupState::Cancelled;
        info!("Setup cancelled by user");
    }

    /// Check if the setup is complete.
    pub fn is_complete(&self) -> bool {
        self.current_state == SetupState::Complete
    }

    /// Check if the setup is cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.current_state == SetupState::Cancelled
    }

    /// Get progress percentage.
    pub fn progress(&self) -> f32 {
        let completed = self
            .steps
            .iter()
            .filter(|s| s.completed || s.skipped)
            .count();
        if self.steps.is_empty() {
            0.0
        } else {
            completed as f32 / self.steps.len() as f32 * 100.0
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OnboardingError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Setup flow has ended")]
    FlowEnded,

    #[error("Storage error: {0}")]
    StorageError(String),
}
