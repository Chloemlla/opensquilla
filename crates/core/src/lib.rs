//! # OpenSquilla Core
//!
//! Core types, configuration, and error handling for the OpenSquilla gateway.
//! This crate provides the fundamental building blocks used by all other crates
//! in the OpenSquilla ecosystem.

/// Core message types: Message, ContentBlock, ToolCall, ToolResult, Usage, etc.
pub mod types;

/// Configuration system: Config, ProviderConfig, ChannelConfig, etc. with serde support.
pub mod config;

/// Error types: Error enum with thiserror derive, and a Result<T> alias.
pub mod error;

/// Model types: ModelInfo, ModelCapabilities, ProviderSpec, ModelPricing, etc.
pub mod model;

/// Event types: StreamEvent, TurnEvent, ToolEvent, and GatewayEvent for streaming and lifecycle.
pub mod events;

/// ID creation helpers.
pub mod id;

/// Time utilities for working with timestamps and durations.
pub mod time;

/// Type alias for core results.
pub mod result;

/// Convenience prelude for crate-level re-exports.
pub mod prelude;

/// Re-export the most commonly used types at the crate root for convenience.
pub use types::{
    AgentId, ContentBlock, Conversation, GenerationParameters, GenerationRequest, JobId, MemoryId,
    Message, MessageId, MessageRole, RateLimit, SessionId, ToolCall, ToolResult, Usage, UserId,
};

pub use config::{
    ChannelConfig, Config, GatewayConfig, ModelConfig, ObservabilityConfig, ProviderConfig,
    ResourceLimits, SandboxConfig, SchedulerConfig, SkillsConfig,
};

pub use error::{Error, Result};

pub use model::{
    AuthMethod, ModelCapabilities, ModelInfo, ModelPricing, ModelRegistry, ModelSelectionStrategy,
    ProviderSpec,
};

pub use events::{
    ContentBlockDelta, GatewayEvent, MessageDelta, StreamEvent, ToolEvent, TurnEvent,
};
