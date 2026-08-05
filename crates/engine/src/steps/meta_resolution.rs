//! Meta-instruction resolution step.
//!
//! Mirrors the Python `engine/steps/meta_resolution.py` step. It inspects the
//! user's message for meta-instruction triggers (a leading `/meta` command or
//! a configured trigger phrase), resolves the matching meta-skill from the
//! loaded skill catalog, and leaves a *soft hint* in pipeline metadata so
//! downstream steps (skills filter, prompt assembly) can surface the matched
//! workflow to the model.
//!
//! The meta-resolution signal is deliberately soft: the step never takes over
//! the turn. It records `meta_match` / `meta_match_trigger` in the pipeline
//! metadata and optionally injects a short guidance block into the system
//! prompt. The model makes the final semantic judgment about whether to invoke
//! `meta_invoke(name=...)`.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use tracing::{debug, instrument};

/// Configuration for meta-instruction resolution.
#[derive(Debug, Clone)]
pub struct MetaResolutionConfig {
    /// Master switch. When false the step is a complete no-op.
    pub enabled: bool,
    /// Command-style trigger prefix (e.g. `/meta`). A message starting with
    /// this prefix is treated as a meta-instruction.
    pub trigger_prefix: String,
    /// Additional trigger phrases that count as meta-instruction signals.
    pub trigger_phrases: Vec<String>,
    /// Messages longer than this many characters are never treated as a
    /// meta-instruction (defends against noisy turns).
    pub max_message_chars: usize,
    /// Whether to inject a short guidance block into the system prompt when a
    /// meta-skill was matched.
    pub inject_guidance: bool,
}

impl Default for MetaResolutionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_prefix: "/meta".to_string(),
            trigger_phrases: vec!["meta workflow".to_string()],
            max_message_chars: 2_000,
            inject_guidance: true,
        }
    }
}

/// A minimal view of a meta-skill entry as produced by the skill loader.
///
/// The engine crate does not depend on the skills crate, so the step works
/// against this lightweight projection. Real skill catalogs can be adapted
/// onto this shape cheaply.
#[derive(Debug, Clone)]
pub struct MetaSkill {
    /// The skill identifier (stable across renames).
    pub id: String,
    /// The skill name, used to build the `meta_invoke(name=...)` hint.
    pub name: String,
    /// A short description rendered in the guidance block.
    pub description: String,
}

/// The result of meta-instruction resolution, recorded into metadata.
#[derive(Debug, Clone)]
pub struct MetaResolutionOutcome {
    /// Whether a meta-instruction was detected and resolved.
    pub matched: bool,
    /// The name of the matched meta-skill, if any.
    pub matched_name: Option<String>,
    /// The trigger text that produced the match.
    pub trigger: Option<String>,
    /// Whether the guidance block was injected into the system prompt.
    pub guidance_injected: bool,
}

/// Pre-turn pipeline step that resolves meta-instructions.
#[derive(Debug)]
pub struct MetaResolutionStep {
    config: MetaResolutionConfig,
    catalog: Vec<MetaSkill>,
}

impl MetaResolutionStep {
    /// Create a new step with default configuration and an empty catalog.
    pub fn new() -> Self {
        Self {
            config: MetaResolutionConfig::default(),
            catalog: Vec::new(),
        }
    }

    /// Create a step with the given configuration.
    pub fn with_config(config: MetaResolutionConfig) -> Self {
        Self {
            config,
            catalog: Vec::new(),
        }
    }

    /// Register the meta-skill catalog used for resolution.
    pub fn with_catalog(mut self, catalog: Vec<MetaSkill>) -> Self {
        self.catalog = catalog;
        self
    }

    /// Detect whether the given user text carries a meta-instruction trigger.
    ///
    /// Returns the trigger text when a signal is present.
    pub fn detect_trigger(&self, message: &str) -> Option<String> {
        let trimmed = message.trim();
        if trimmed.is_empty() {
            return None;
        }
        if self.config.max_message_chars > 0
            && trimmed.chars().count() > self.config.max_message_chars
        {
            return None;
        }
        let lowered = trimmed.to_lowercase();
        let prefix = self.config.trigger_prefix.to_lowercase();
        if !prefix.is_empty() && lowered.starts_with(prefix.as_str()) {
            return Some(self.config.trigger_prefix.clone());
        }
        self.config
            .trigger_phrases
            .iter()
            .find(|phrase| lowered.contains(phrase.to_lowercase().as_str()))
            .cloned()
    }

    /// Resolve a trigger to the best matching meta-skill in the catalog.
    ///
    /// Matches by name substring first (so `meta_invoke(name="code-task")`
    /// resolves for a `/meta code-task` message), then by id.
    pub fn resolve(&self, trigger: &str, message: &str) -> Option<&MetaSkill> {
        let lowered = message.to_lowercase();
        self.catalog
            .iter()
            .find(|s| {
                !s.name.is_empty()
                    && (lowered.contains(s.name.to_lowercase().as_str())
                        || (trigger.starts_with(self.config.trigger_prefix.as_str())
                            && lowered
                                .trim()
                                .trim_start_matches(self.config.trigger_prefix.as_str())
                                .trim()
                                .starts_with(s.name.as_str())))
            })
            .or_else(|| {
                self.catalog
                    .iter()
                    .find(|s| !s.id.is_empty() && lowered.contains(s.id.to_lowercase().as_str()))
            })
    }
}

impl Default for MetaResolutionStep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStep for MetaResolutionStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if !self.config.enabled {
            debug!("meta_resolution disabled, skipping");
            return Ok(StepAction::Continue);
        }

        // Find the most recent user message as the resolution surface.
        let user_text = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content());

        let Some(message) = user_text else {
            debug!("no user message found, skipping");
            return Ok(StepAction::Continue);
        };

        let Some(trigger) = self.detect_trigger(&message) else {
            debug!("no meta-instruction trigger detected");
            ctx.set_metadata("meta_match", "");
            ctx.set_metadata("meta_match_trigger", "");
            return Ok(StepAction::Continue);
        };

        let Some(skill) = self.resolve(&trigger, &message) else {
            debug!(trigger = %trigger, "meta trigger detected but no matching skill");
            ctx.set_metadata("meta_match_trigger", &trigger);
            return Ok(StepAction::Continue);
        };

        debug!(
            trigger = %trigger,
            skill = %skill.name,
            "meta-instruction resolved"
        );

        let mut guidance_injected = false;
        if self.config.inject_guidance {
            let guidance = format!(
                "The user is invoking a meta-workflow. When you are confident the \
                 workflow applies, call the `meta_invoke` tool with name=\"{}\". \
                 {}",
                skill.name, skill.description
            );
            ctx.add_message(Message::system(guidance));
            guidance_injected = true;
        }

        // Soft hints consumed by the skills filter and prompt assembly.
        ctx.set_metadata("meta_match", &skill.name);
        ctx.set_metadata("meta_match_trigger", &trigger);
        ctx.set_metadata("meta_match_candidates", &skill.id);
        ctx.set_metadata("meta_resolution_applied", "true");

        let outcome = MetaResolutionOutcome {
            matched: true,
            matched_name: Some(skill.name.clone()),
            trigger: Some(trigger),
            guidance_injected,
        };
        let _ = outcome;

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "meta_resolution"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_prefix_trigger() {
        let step = MetaResolutionStep::new();
        assert!(step.detect_trigger("/meta code-task").is_some());
        assert!(step.detect_trigger("normal question").is_none());
    }

    #[test]
    fn test_detect_long_message_ignored() {
        let step = MetaResolutionStep::new();
        let long = "x".repeat(3000);
        assert!(step.detect_trigger(&long).is_none());
    }

    #[test]
    fn test_resolve_by_name() {
        let step = MetaResolutionStep::new().with_catalog(vec![
            MetaSkill {
                id: "code-task".into(),
                name: "code-task".into(),
                description: "Run a coding task".into(),
            },
        ]);
        let matched = step.resolve("/meta", "/meta code-task fix the tests");
        assert_eq!(matched.map(|s| s.name.as_str()), Some("code-task"));
    }
}
