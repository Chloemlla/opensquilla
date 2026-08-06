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

/// Parsed SKILL.md frontmatter, mirroring the Python skills loader's
/// `_parse_skill_frontmatter`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SkillFrontmatter {
    /// The skill name.
    pub name: String,
    /// A short description.
    pub description: String,
    /// The skill kind (`"skill"` or `"meta"`).
    pub kind: String,
    /// Whether the skill is always visible.
    pub always: bool,
    /// Whether the model may invoke this skill directly.
    pub allow_model_invocation: bool,
    /// Required tools.
    pub requires_tools: Vec<String>,
    /// Skills this skill requires.
    pub requires_skills: Vec<String>,
    /// Toolsets this skill falls back to.
    pub fallback_for_toolsets: Vec<String>,
    /// Whether the skill auto-triggers on matching messages.
    pub auto_trigger: bool,
    /// Trigger phrases for auto-trigger.
    pub triggers: Vec<String>,
    /// The original raw frontmatter.
    pub raw: String,
}

/// Parse a SKILL.md document's YAML frontmatter block.
///
/// The frontmatter is the `---`-delimited block at the top of the file.
/// Missing or malformed frontmatter yields a default [`SkillFrontmatter`]
/// with `name` derived from the first heading line.
pub fn parse_skill_frontmatter(content: &str) -> SkillFrontmatter {
    let raw = extract_frontmatter(content).unwrap_or_default();
    if raw.is_empty() {
        // No frontmatter: derive the name from the first `# heading`.
        let name = content
            .lines()
            .find(|l| l.trim_start().starts_with('#'))
            .map(|l| l.trim_start_matches('#').trim().to_string())
            .unwrap_or_default();
        return SkillFrontmatter {
            name,
            ..Default::default()
        };
    }

    let mut frontmatter = SkillFrontmatter {
        raw: raw.clone(),
        ..Default::default()
    };

    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("---") {
            continue;
        }
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let key = key.trim().to_lowercase();
        let value = value.trim().to_string();
        match key.as_str() {
            "name" => frontmatter.name = unquote(&value),
            "description" => frontmatter.description = unquote(&value),
            "kind" => frontmatter.kind = unquote(&value),
            "always" => frontmatter.always = parse_bool(&value),
            "allow_model_invocation" => frontmatter.allow_model_invocation = parse_bool(&value),
            "requires_tools" => frontmatter.requires_tools = parse_list(&value),
            "requires_skills" => frontmatter.requires_skills = parse_list(&value),
            "fallback_for_toolsets" => frontmatter.fallback_for_toolsets = parse_list(&value),
            "auto_trigger" => frontmatter.auto_trigger = parse_bool(&value),
            "triggers" => frontmatter.triggers = parse_list(&value),
            _ => {}
        }
    }

    frontmatter
}

/// Extract the `---`-delimited frontmatter block from a document.
pub fn extract_frontmatter(content: &str) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() || !lines[0].trim().starts_with("---") {
        return None;
    }
    let mut block = Vec::new();
    for line in lines.iter().skip(1) {
        if line.trim().starts_with("---") {
            return Some(block.join("\n"));
        }
        block.push(*line);
    }
    None
}

/// Convert a parsed [`SkillFrontmatter`] into a [`MetaSkill`].
impl From<SkillFrontmatter> for MetaSkill {
    fn from(fm: SkillFrontmatter) -> Self {
        Self {
            id: fm.name.clone(),
            name: fm.name,
            description: fm.description,
        }
    }
}

/// Convert a parsed [`SkillFrontmatter`] into a `crate::steps::SkillSpec`.
///
/// This bridges the frontmatter parser to the skills filter so a SKILL.md
/// catalog can feed the gate directly.
pub fn frontmatter_to_skill_spec(
    fm: &SkillFrontmatter,
) -> crate::steps::SkillSpec {
    crate::steps::SkillSpec {
        id: fm.name.clone(),
        name: fm.name.clone(),
        description: fm.description.clone(),
        kind: if fm.kind.is_empty() {
            "skill".to_string()
        } else {
            fm.kind.clone()
        },
        always: fm.always,
        disable_model_invocation: !fm.allow_model_invocation,
        requires_tools: fm.requires_tools.clone(),
        fallback_for_toolsets: fm.fallback_for_toolsets.clone(),
    }
}

/// Strip surrounding quotes from a string.
fn unquote(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

/// Parse a boolean value from YAML-ish text.
fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "yes" | "1" | "on"
    )
}

/// Parse a list value from YAML-ish text.
///
/// Accepts `[a, b, c]`, `["a", "b"]`, and comma-separated scalar values.
fn parse_list(value: &str) -> Vec<String> {
    let value = value.trim();
    let inner = value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .unwrap_or(value);
    inner
        .split(',')
        .map(|item| unquote(item.trim()))
        .filter(|s| !s.is_empty())
        .collect()
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
        let step = MetaResolutionStep::new().with_catalog(vec![MetaSkill {
            id: "code-task".into(),
            name: "code-task".into(),
            description: "Run a coding task".into(),
        }]);
        let matched = step.resolve("/meta", "/meta code-task fix the tests");
        assert_eq!(matched.map(|s| s.name.as_str()), Some("code-task"));
    }
}
