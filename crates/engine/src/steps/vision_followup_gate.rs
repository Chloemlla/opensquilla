//! Vision follow-up gate step.
//!
//! Mirrors the Python `engine/steps/vision_followup_gate.py` step. It is a
//! semantic gate for text-only follow-ups to historical images: when the
//! history contains a recent image but the current turn has no image
//! attachment, the gate decides whether the model needs to reuse the previous
//! image.
//!
//! The Python step calls an auxiliary LLM provider to classify the turn, with
//! deterministic fast-paths for explicit opt-out / explicit previous-image
//! references. The Rust port implements the deterministic fast-paths and the
//! metadata bookkeeping. The LLM-backed classification is left as a
//! `TODO(parity)` because the Rust `PipelineContext` has no provider handle.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use regex::Regex;
use std::sync::OnceLock;
use tracing::{debug, instrument};

/// English image-reference regex.
fn image_ref_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(image|picture|photo|screenshot|screen|diagram)\b").expect("valid regex")
    })
}

/// English explicit opt-out regex.
fn image_optout_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?i)\b(?:do\s+not|don't|dont|without|no\s+need\s+to)\b",
            r".{0,80}\b(?:use|inspect|look\s+at|view|analy[sz]e|consider)\b",
            r".{0,80}\b(?:image|picture|photo|screenshot|screen|diagram)\b",
            "|",
            r"\bignore\b.{0,80}\b(?:image|picture|photo|screenshot|screen|diagram)\b",
            "|",
            r"(?:不要|不用|无需|不需要|别).{0,40}(?:看|使用|参考|分析|检查).{0,40}(?:图|图片|截图|照片)",
            "|",
            r"(?:忽略|无视).{0,40}(?:图|图片|截图|照片)",
        ))
        .expect("valid regex")
    })
}

/// English previous-image reference regex.
fn previous_image_ref_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?i)\b(?:previous|last|earlier|above|that|the)\b",
            r".{0,50}\b(?:image|picture|photo|screenshot|screen|diagram)\b",
            "|",
            r"\b(?:image|picture|photo|screenshot|screen|diagram)\b",
            r".{0,50}\b(?:above|before|earlier)\b",
        ))
        .expect("valid regex")
    })
}

/// Chinese image reference substrings.
const ZH_IMAGE_REFS: &[&str] = &["图", "图片", "截图", "照片"];

/// Chinese previous-image reference substrings.
const ZH_PREVIOUS_IMAGE_REFS: &[&str] = &[
    "上一张图",
    "上一张图片",
    "上张图",
    "上张图片",
    "刚才那张图",
    "刚才那张图片",
    "前面那张图",
    "前面那张图片",
    "之前那张图",
    "之前那张图片",
    "那张图",
    "那张图片",
];

/// Configuration for the vision follow-up gate step.
#[derive(Debug, Clone)]
pub struct VisionFollowupGateConfig {
    /// Whether the gate is enabled.
    pub enabled: bool,
    /// The unknown-policy fallback: `"image_if_recent"` or `"text_only"`.
    pub unknown_policy: String,
    /// When the unknown policy is `image_if_recent`, how many recent turns
    /// after an image still count as "recent".
    pub fallback_recent_turns: u32,
}

impl Default for VisionFollowupGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            unknown_policy: "image_if_recent".to_string(),
            fallback_recent_turns: 2,
        }
    }
}

/// Pre-turn pipeline step that gates text-only follow-ups to historical images.
#[derive(Debug)]
pub struct VisionFollowupGateStep {
    config: VisionFollowupGateConfig,
}

impl VisionFollowupGateStep {
    /// Create a new step with default configuration.
    pub fn new() -> Self {
        Self {
            config: VisionFollowupGateConfig::default(),
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: VisionFollowupGateConfig) -> Self {
        Self { config }
    }

    /// Read the current user message text from the pipeline context.
    fn current_user_text(&self, ctx: &PipelineContext) -> String {
        use opensquilla_core::types::MessageRole;
        ctx.messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content())
            .unwrap_or_default()
    }

    /// Read an integer metadata value.
    fn turns_since_last_image(&self, ctx: &PipelineContext) -> Option<i64> {
        ctx.get_metadata("router_turns_since_last_image")
            .and_then(|v| v.parse::<i64>().ok())
    }

    /// Check whether the candidate window has expired.
    fn candidate_window_expired(&self, ctx: &PipelineContext) -> bool {
        let turns = match self.turns_since_last_image(ctx) {
            Some(t) => t,
            None => return false,
        };
        let candidate_turns = ctx
            .get_metadata("router_vision_candidate_turns")
            .and_then(|v| v.parse::<i64>().ok());
        match candidate_turns {
            Some(c) if c > 0 => turns >= c,
            _ => false,
        }
    }

    /// Check whether the current text explicitly opts out of image use.
    fn current_text_explicitly_opts_out(&self, text: &str) -> bool {
        if text.trim().is_empty() {
            return false;
        }
        if !image_ref_re().is_match(text) && !ZH_IMAGE_REFS.iter().any(|r| text.contains(r)) {
            return false;
        }
        image_optout_re().is_match(text)
    }

    /// Check whether the current text explicitly references a previous image.
    fn current_text_explicitly_requests_previous_image(&self, text: &str) -> bool {
        if text.trim().is_empty() {
            return false;
        }
        if ZH_PREVIOUS_IMAGE_REFS.iter().any(|r| text.contains(r)) {
            return true;
        }
        previous_image_ref_re().is_match(text)
    }

    /// Apply the explicit opt-out decision to the context.
    fn apply_explicit_opt_out(&self, ctx: &mut PipelineContext) {
        ctx.set_metadata("router_vision_followup_gate_decision", "text_only");
        ctx.set_metadata("router_vision_followup_gate_confidence", "1.0");
        ctx.set_metadata(
            "router_vision_followup_gate_reason",
            "current turn explicitly opts out of image use",
        );
        ctx.set_metadata("router_vision_followup_gate_source", "explicit_opt_out");
        ctx.set_metadata("router_vision_followup_needs_image", "false");
    }

    /// Apply the explicit previous-image-request decision to the context.
    fn apply_explicit_previous_image_request(&self, ctx: &mut PipelineContext) {
        ctx.set_metadata("router_vision_followup_gate_decision", "needs_image");
        ctx.set_metadata("router_vision_followup_gate_confidence", "1.0");
        ctx.set_metadata(
            "router_vision_followup_gate_reason",
            "current turn explicitly references a previous image",
        );
        ctx.set_metadata(
            "router_vision_followup_gate_source",
            "explicit_image_reference",
        );
        ctx.set_metadata("router_vision_followup_needs_image", "true");
    }

    /// Apply the unknown-fallback decision to the context.
    fn apply_unknown_fallback(&self, ctx: &mut PipelineContext, source: &str, reason: &str) {
        let turns = self.turns_since_last_image(ctx);
        let needs_image = self.config.unknown_policy == "image_if_recent"
            && turns.is_some()
            && turns.unwrap() <= self.config.fallback_recent_turns as i64;

        ctx.set_metadata("router_vision_followup_gate_decision", "unknown");
        ctx.set_metadata("router_vision_followup_gate_confidence", "0.0");
        ctx.set_metadata("router_vision_followup_gate_reason", reason);
        ctx.set_metadata("router_vision_followup_gate_source", source);
        ctx.set_metadata(
            "router_vision_followup_needs_image",
            if needs_image { "true" } else { "false" },
        );
        if needs_image {
            ctx.set_metadata("router_vision_followup_fallback", "image_if_recent");
        }
    }
}

impl Default for VisionFollowupGateStep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStep for VisionFollowupGateStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if !self.config.enabled {
            ctx.set_metadata("router_vision_followup_gate_decision", "disabled");
            return Ok(StepAction::Continue);
        }

        // If the current turn already has an image attachment, the gate is
        // not applicable.
        // TODO(parity): the Python step checks `ctx.attachments` for image
        // media types. The Rust PipelineContext has no `attachments` field;
        // we check the `current_turn_has_image` metadata flag instead, which
        // the runtime is expected to set.
        if ctx
            .get_metadata("current_turn_has_image")
            .map(|v| v == "true")
            .unwrap_or(false)
        {
            ctx.set_metadata("router_vision_followup_gate_decision", "current_image");
            return Ok(StepAction::Continue);
        }

        // If the history has no recent image, the gate is not applicable.
        let history_has_recent_image = ctx
            .get_metadata("router_history_has_recent_image")
            .map(|v| v == "true")
            .unwrap_or(false);
        if !history_has_recent_image {
            ctx.set_metadata("router_vision_followup_gate_decision", "not_applicable");
            return Ok(StepAction::Continue);
        }

        // If the candidate window has expired, the gate is not applicable.
        if self.candidate_window_expired(ctx) {
            ctx.set_metadata("router_vision_followup_gate_decision", "not_applicable");
            ctx.set_metadata(
                "router_vision_followup_gate_reason",
                "candidate_window_expired",
            );
            return Ok(StepAction::Continue);
        }

        let text = self.current_user_text(ctx);

        // Deterministic fast-path: explicit opt-out.
        if self.current_text_explicitly_opts_out(&text) {
            self.apply_explicit_opt_out(ctx);
            return Ok(StepAction::Continue);
        }

        // Deterministic fast-path: explicit previous-image reference.
        if self.current_text_explicitly_requests_previous_image(&text) {
            self.apply_explicit_previous_image_request(ctx);
            return Ok(StepAction::Continue);
        }

        // TODO(parity): the Python step calls an auxiliary LLM provider to
        // classify the turn as needs_image / text_only / unknown. The Rust
        // PipelineContext has no provider handle, so we fall back to the
        // unknown policy. When a provider is wired into the pipeline, this
        // branch should call `_call_gate_provider` and `_apply_gate_decision`.
        debug!("vision_followup_gate: no LLM provider, applying unknown fallback");
        self.apply_unknown_fallback(ctx, "no_provider", "no LLM provider available for gate");

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "vision_followup_gate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    fn ctx_with_user(text: &str) -> PipelineContext {
        PipelineContext::new("t1".into(), vec![Message::user(text)])
    }

    #[tokio::test]
    async fn test_disabled_skips() {
        let step = VisionFollowupGateStep::with_config(VisionFollowupGateConfig {
            enabled: false,
            ..Default::default()
        });
        let mut ctx = ctx_with_user("hello");
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("disabled")
        );
    }

    #[tokio::test]
    async fn test_current_image_skips() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("what is this");
        ctx.set_metadata("current_turn_has_image", "true");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("current_image")
        );
    }

    #[tokio::test]
    async fn test_no_history_image_skips() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("what is this");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("not_applicable")
        );
    }

    #[tokio::test]
    async fn test_explicit_optout() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("do not look at the image");
        ctx.set_metadata("router_history_has_recent_image", "true");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("text_only")
        );
        assert_eq!(
            ctx.get_metadata("router_vision_followup_needs_image")
                .map(String::as_str),
            Some("false")
        );
    }

    #[tokio::test]
    async fn test_explicit_previous_image() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("look at the previous image again");
        ctx.set_metadata("router_history_has_recent_image", "true");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("needs_image")
        );
        assert_eq!(
            ctx.get_metadata("router_vision_followup_needs_image")
                .map(String::as_str),
            Some("true")
        );
    }

    #[tokio::test]
    async fn test_zh_previous_image() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("看一下上一张图");
        ctx.set_metadata("router_history_has_recent_image", "true");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("needs_image")
        );
    }

    #[tokio::test]
    async fn test_unknown_fallback_recent() {
        let step = VisionFollowupGateStep::new();
        let mut ctx = ctx_with_user("what is the meaning of life");
        ctx.set_metadata("router_history_has_recent_image", "true");
        ctx.set_metadata("router_turns_since_last_image", "1");
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("router_vision_followup_gate_decision")
                .map(String::as_str),
            Some("unknown")
        );
        assert_eq!(
            ctx.get_metadata("router_vision_followup_needs_image")
                .map(String::as_str),
            Some("true")
        );
    }
}
