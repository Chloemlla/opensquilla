//! Prompt-cache annotation step.
//!
//! Mirrors the Python `engine/steps/prompt_cache.py` step. It annotates the
//! system prompt with provider cache breakpoints and records cache metrics
//! (hashes, char counts) into pipeline metadata for observability.
//!
//! The Python step reads a `prompt_cache` config block and a dual-track
//! `(base, dynamic)` system-prompt tuple. The Rust `PipelineContext` has no
//! config or split system prompt, so cache enablement is driven by metadata
//! and the whole system message text is treated as the cache base.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::MessageRole;
use sha2::{Digest, Sha256};
use tracing::{debug, instrument};

/// Configuration for the prompt-cache step.
#[derive(Debug, Clone)]
pub struct PromptCacheConfig {
    /// Whether prompt caching is enabled.
    pub enabled: bool,
    /// The cache mode (e.g. `"off"`, `"anthropic"`, `"auto"`).
    pub mode: String,
}

impl Default for PromptCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "off".to_string(),
        }
    }
}

/// Pre-turn pipeline step that annotates the system prompt for caching.
#[derive(Debug)]
pub struct PromptCacheStep {
    config: PromptCacheConfig,
}

impl PromptCacheStep {
    /// Create a new step with default configuration (disabled).
    pub fn new() -> Self {
        Self {
            config: PromptCacheConfig::default(),
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: PromptCacheConfig) -> Self {
        Self { config }
    }
}

impl Default for PromptCacheStep {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a 16-character hex hash of the given text (SHA-256 truncated).
fn hash16(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|b| format!("{:02x}", b)).collect();
    hex[..16].to_string()
}

#[async_trait]
impl PipelineStep for PromptCacheStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        // The Python step reads a `prompt_cache` config block. The Rust
        // PipelineContext has no config, so enablement is driven by the
        // `cache_enabled` metadata flag set by the runtime, or by the step's
        // own config.
        let enabled = self.config.enabled
            || ctx
                .get_metadata("cache_enabled")
                .map(|v| v == "true")
                .unwrap_or(false);
        if !enabled {
            return Ok(StepAction::Continue);
        }

        let mode = if !self.config.mode.is_empty() && self.config.mode != "off" {
            self.config.mode.clone()
        } else if let Some(m) = ctx.get_metadata("cache_mode") {
            m.clone()
        } else {
            "auto".to_string()
        };
        if mode == "off" {
            return Ok(StepAction::Continue);
        }

        debug!(mode = %mode, "prompt_cache.applying");
        ctx.set_metadata("cache_enabled", "true");
        ctx.set_metadata("cache_mode", &mode);

        // Record the resolved model for cache-key telemetry. The Python step
        // reads `ctx.model`; the Rust pipeline carries it as `resolved_model`
        // metadata (written by ModelSelectStep).
        let resolved_model = ctx
            .get_metadata("resolved_model")
            .cloned()
            .or_else(|| ctx.get_metadata("model").cloned())
            .unwrap_or_default();
        ctx.set_metadata("resolved_model", &resolved_model);

        // TODO(parity): the Python step records a dual-track cache key using
        // `parse_agent_id(ctx.session_key)`, `ctx.provider.provider_name`, and
        // `ctx.metadata["platform_markdown_hint"]`. The Rust PipelineContext
        // has no `session_key` or `provider` field. The agent id and provider
        // are carried in metadata when the runtime sets them; we use those
        // when present.
        let agent_id = ctx.get_metadata("agent_id").cloned().unwrap_or_default();
        let provider_after_rewrite = ctx
            .get_metadata("provider_name")
            .cloned()
            .unwrap_or_default();
        let channel_pinned = ctx
            .get_metadata("platform_markdown_hint")
            .map(|v| !v.is_empty())
            .unwrap_or(false);

        let legacy_hash = hash16(&format!("{agent_id}\0{resolved_model}"));
        let shadow_hash = hash16(&format!(
            "{agent_id}\0{resolved_model}\0{provider_after_rewrite}\0{channel_pinned}"
        ));

        ctx.set_metadata("provider_after_rewrite", &provider_after_rewrite);
        ctx.set_metadata("cache_legacy_hash", &legacy_hash);
        ctx.set_metadata("cache_shadow_final_hash", &shadow_hash);
        // TODO(parity): collision detection against a process-global
        // `_LEGACY_TO_SHADOW` dict is omitted (no persistent state across
        // turns in the Rust pipeline).
        ctx.set_metadata("cache_key_collision", "false");

        // Record cache metrics from the system prompt text. The Python step
        // splits the system prompt into (base, dynamic); the Rust pipeline has
        // a single system message, so we treat its full text as the base.
        let system_text = ctx
            .messages
            .iter()
            .find(|m| m.role == MessageRole::System)
            .map(|m| m.text_content())
            .unwrap_or_default();

        if !system_text.is_empty() {
            ctx.set_metadata("cache_base_prompt", &system_text);
            ctx.set_metadata("cache_base_chars", system_text.chars().count().to_string());
            ctx.set_metadata("cache_base_hash", hash16(&system_text));
        }

        // The Python step sets `cache_last_tool = True` when tool_defs are
        // present. The Rust pipeline carries tool availability as metadata.
        if ctx
            .get_metadata("has_tool_defs")
            .map(|v| v == "true")
            .unwrap_or(false)
        {
            ctx.set_metadata("cache_last_tool", "true");
        }

        // TODO(parity): `cache_dynamic_prompt`, `cache_dynamic_chars`, and
        // `cache_dynamic_hash` are omitted — the Rust PipelineContext has no
        // split (base, dynamic) system prompt.

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "prompt_cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{Message, MessageRole};

    #[tokio::test]
    async fn test_disabled_is_noop() {
        let step = PromptCacheStep::new();
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert!(ctx.get_metadata("cache_enabled").is_none());
    }

    #[tokio::test]
    async fn test_enabled_via_metadata() {
        let step = PromptCacheStep::new();
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        ctx.set_metadata("cache_enabled", "true");
        ctx.set_metadata("resolved_model", "claude-sonnet");
        ctx.add_message(Message::system("You are helpful."));
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("cache_enabled").map(String::as_str),
            Some("true")
        );
        assert!(ctx.get_metadata("cache_base_hash").is_some());
        assert!(
            ctx.get_metadata("cache_base_hash")
                .map(|h| h.len() == 16)
                .unwrap_or(false)
        );
    }

    #[tokio::test]
    async fn test_enabled_via_config() {
        let step = PromptCacheStep::with_config(PromptCacheConfig {
            enabled: true,
            mode: "anthropic".into(),
        });
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        ctx.add_message(Message {
            role: MessageRole::System,
            content: vec![opensquilla_core::types::ContentBlock::Text { text: 
                "system prompt".into(),
             }],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        });
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("cache_mode").map(String::as_str),
            Some("anthropic")
        );
        assert!(ctx.get_metadata("cache_base_chars").is_some());
    }
}
