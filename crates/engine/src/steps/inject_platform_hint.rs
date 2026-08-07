//! Platform-hint injection step.
//!
//! Mirrors the Python `engine/steps/inject_platform_hint.py` step. It appends a
//! channel-specific rendering hint to the system prompt suffix when the
//! channel kind benefits from markdown guidance.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use tracing::{debug, instrument};

/// Configuration for the platform-hint injection step.
#[derive(Debug, Clone)]
pub struct InjectPlatformHintConfig {
    /// Master switch. When false the step is a complete no-op.
    pub enabled: bool,
}

impl Default for InjectPlatformHintConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Pre-turn pipeline step that injects a channel rendering hint.
#[derive(Debug, Default)]
pub struct InjectPlatformHintStep {
    config: InjectPlatformHintConfig,
}

impl InjectPlatformHintStep {
    /// Create a new step with default configuration (enabled).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: InjectPlatformHintConfig) -> Self {
        Self { config }
    }

    /// Return the markdown rendering hint for the given channel kind, if any.
    ///
    /// Mirrors `opensquilla.channels.registry.markdown_render_hint_for`. The
    /// Rust channels crate does not yet expose this function, so the mapping is
    /// inlined here.
    fn render_hint_for(&self, channel_kind: &str) -> Option<String> {
        match channel_kind {
            "dingtalk" => Some(
                "Render responses in Markdown. DingTalk supports headings, bold, \
                 italic, code blocks, lists, and links."
                    .to_string(),
            ),
            "discord" => Some(
                "Render responses in Markdown. Discord supports bold, italic, \
                 underline, strikethrough, code blocks, inline code, lists, and \
                 links."
                    .to_string(),
            ),
            "slack" => Some(
                "Render responses in Slack's mrkdwn format. Use *bold* not \
                 **bold**, _italic_ not *italic*, and ```code blocks```."
                    .to_string(),
            ),
            "wecom" | "wechat" => Some(
                "Render responses in plain Markdown. WeCom supports bold, code \
                 blocks, and basic lists. Avoid complex tables."
                    .to_string(),
            ),
            "qq" => Some(
                "Render responses in Markdown. QQ supports basic markdown \
                 formatting including bold, code blocks, and lists."
                    .to_string(),
            ),
            "msteams" => Some(
                "Render responses in Markdown. MS Teams supports headings, bold, \
                 italic, code blocks, lists, and links."
                    .to_string(),
            ),
            "matrix" => Some(
                "Render responses in Markdown. Matrix supports bold, italic, \
                 code blocks, lists, and links."
                    .to_string(),
            ),
            "webhook" => Some(
                "Render responses in Markdown. The webhook channel forwards \
                 markdown to the downstream integration."
                    .to_string(),
            ),
            _ => None,
        }
    }
}

#[async_trait]
impl PipelineStep for InjectPlatformHintStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if !self.config.enabled {
            debug!("inject_platform_hint disabled, skipping");
            ctx.set_metadata("inject_platform_hint__applied", "false");
            return Ok(StepAction::Continue);
        }

        let channel_kind = ctx
            .get_metadata("channel_kind")
            .map(|s| s.trim().to_lowercase())
            .unwrap_or_default();

        let Some(hint) = self.render_hint_for(&channel_kind) else {
            debug!(channel_kind = %channel_kind, "no render hint for channel");
            ctx.set_metadata("inject_platform_hint__applied", "false");
            return Ok(StepAction::Continue);
        };

        let block = format!("## Channel Rendering\n\n{hint}");

        // Append the rendering hint block to the last system message, or insert
        // a fresh system message when none exists. The Python step appends to
        // the "uncached suffix" of a (base, suffix) system-prompt tuple; the
        // Rust pipeline has no such split, so we append to the system message
        // text directly.
        if let Some(idx) = ctx
            .messages
            .iter()
            .rposition(|m| m.role == MessageRole::System)
        {
            let existing = ctx.messages[idx].text_content();
            let combined = if existing.is_empty() {
                block.clone()
            } else {
                format!("{existing}\n\n{block}")
            };
            ctx.messages[idx] = Message::text(MessageRole::System, combined);
        } else {
            ctx.add_message(Message::system(block));
        }

        ctx.set_metadata("platform_markdown_hint", &channel_kind);
        debug!(channel_kind = %channel_kind, "platform hint injected");

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "inject_platform_hint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    #[tokio::test]
    async fn test_disabled_is_noop() {
        let step = InjectPlatformHintStep::with_config(InjectPlatformHintConfig { enabled: false });
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert_eq!(
            ctx.get_metadata("inject_platform_hint__applied")
                .map(String::as_str),
            Some("false")
        );
    }

    #[tokio::test]
    async fn test_unknown_channel_skips() {
        let step = InjectPlatformHintStep::new();
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        ctx.set_metadata("channel_kind", "unknown_chan");
        step.execute(&mut ctx).await.unwrap();
        assert!(ctx.get_metadata("platform_markdown_hint").is_none());
    }

    #[tokio::test]
    async fn test_injects_hint_for_slack() {
        let step = InjectPlatformHintStep::new();
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        ctx.set_metadata("channel_kind", "slack");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("platform_markdown_hint")
                .map(String::as_str),
            Some("slack")
        );
        assert!(
            ctx.messages
                .iter()
                .any(|m| m.text_content().contains("Channel Rendering"))
        );
    }
}
