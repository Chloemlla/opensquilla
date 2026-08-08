//! Skills filtering step.
//!
//! Mirrors the Python `engine/steps/skills_filter.py` step. It gates the
//! loaded skill catalog deterministically (no LLM), hides meta-skills when the
//! auto-trigger is off, pins "always" skills and any meta-matched workflow so
//! retrieval cannot drop them, optionally retrieves a subset by relevance, and
//! injects the surviving skills into the system prompt as an
//! `<available_skills>` block.
//!
//! The gate is pure over the skill specs and the set of available tool names;
//! retrieval and prompt injection are the only places that consult config.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use tracing::{debug, instrument};

/// A skill spec, matching the Python `skills/types.py` `SkillSpec` shape that
/// the gate consumes.
#[derive(Debug, Clone)]
pub struct SkillSpec {
    /// Stable skill identifier.
    pub id: String,
    /// Skill name (used for `meta_invoke(name=...)` and pinning).
    pub name: String,
    /// One-line description rendered in `<available_skills>`.
    pub description: String,
    /// Skill kind: `"skill"` (default) or `"meta"`.
    pub kind: String,
    /// Always-visible skills bypass the relevance filter.
    pub always: bool,
    /// Skills that must never be invoked directly by the model.
    pub disable_model_invocation: bool,
    /// Tools the skill requires; the skill is gated out if any is missing.
    pub requires_tools: Vec<String>,
    /// Toolsets that, when present, make this skill redundant.
    pub fallback_for_toolsets: Vec<String>,
}

impl SkillSpec {
    /// A skill that is always visible.
    pub fn always(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            kind: "skill".to_string(),
            always: true,
            disable_model_invocation: false,
            requires_tools: Vec::new(),
            fallback_for_toolsets: Vec::new(),
        }
    }

    /// A standard (filterable) skill.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            always: false,
            ..Self::always(id, name)
        }
    }
}

/// Configuration for the skills filtering step.
#[derive(Debug, Clone)]
pub struct SkillsFilterConfig {
    /// Whether relevance retrieval is enabled at all.
    pub filter_enabled: bool,
    /// Maximum number of characters in the injected skills prompt.
    pub max_skills_prompt_chars: usize,
    /// Injection mode: `"system"`, `"user_message"`, or `"user_context"`.
    pub injection_mode: String,
    /// Top-k skills retained by the retriever.
    pub filter_top_k: usize,
    /// Skill ids the operator has disabled.
    pub disabled: Vec<String>,
    /// When true, only skills named in `disabled` are gated (test/simulator).
    pub memory_only: bool,
}

impl Default for SkillsFilterConfig {
    fn default() -> Self {
        Self {
            filter_enabled: false,
            max_skills_prompt_chars: 8_000,
            injection_mode: "system".to_string(),
            filter_top_k: 5,
            disabled: Vec::new(),
            memory_only: false,
        }
    }
}

/// The outcome of the skills filter, recorded in metadata.
#[derive(Debug, Clone)]
pub struct SkillsFilterOutcome {
    /// Total skills in the catalog before gating.
    pub total: usize,
    /// Skills that passed the deterministic gate.
    pub gated: usize,
    /// Skills actually rendered into `<available_skills>`.
    pub rendered: usize,
    /// Number of characters written into the skills prompt.
    pub prompt_chars: usize,
}

/// Pre-turn pipeline step that gates, filters, and injects skills.
#[derive(Debug)]
pub struct SkillsFilterStep {
    config: SkillsFilterConfig,
    /// The loaded skill catalog (a live projection of the skills loader).
    catalog: Vec<SkillSpec>,
    /// Names of tools available to the agent this turn.
    available_tools: Vec<String>,
}

impl SkillsFilterStep {
    /// Create a new step with default configuration.
    pub fn new() -> Self {
        Self {
            config: SkillsFilterConfig::default(),
            catalog: Vec::new(),
            available_tools: Vec::new(),
        }
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: SkillsFilterConfig) -> Self {
        self.config = config;
        self
    }

    /// Register the skill catalog.
    pub fn with_catalog(mut self, catalog: Vec<SkillSpec>) -> Self {
        self.catalog = catalog;
        self
    }

    /// Register the set of tools available this turn.
    pub fn with_available_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    /// Deterministic skill gate (pure, no I/O).
    ///
    /// A skill is retained when it is model-invocable, passes eligibility
    /// (simplified here to the operator-disabled set), its required tools are
    /// all present, and it is not redundant with an already-present toolset.
    pub fn deterministic_gate(&self, skills: &[SkillSpec]) -> Vec<SkillSpec> {
        let available: Vec<String> = self.available_tools.clone();
        let disabled: Vec<String> = self.config.disabled.clone();
        skills
            .iter()
            .filter(|s| {
                if s.disable_model_invocation {
                    return false;
                }
                if disabled.iter().any(|d| d == &s.id || d == &s.name) {
                    return false;
                }
                if !s.requires_tools.is_empty()
                    && !s.requires_tools.iter().all(|t| available.contains(t))
                {
                    return false;
                }
                if s.fallback_for_toolsets
                    .iter()
                    .any(|t| available.contains(t))
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect()
    }

    /// Compute an eligibility score for a skill against the current turn.
    ///
    /// The score aggregates several signals:
    /// * whether the skill's required tools are all present (+0.4),
    /// * whether the skill is a meta skill and meta is enabled (+0.2),
    /// * whether the skill's id/name appears in the pipeline metadata
    ///   (`meta_match`, `requested_skill`) (+0.3),
    /// * whether the skill is pinned (`always`) (+0.1).
    ///
    /// Returns a score in `[0, 1]`.
    pub fn eligibility_score(&self, skill: &SkillSpec, ctx: &PipelineContext) -> f64 {
        let mut score = 0.0f64;
        let available = &self.available_tools;

        // Required tools present.
        if skill.requires_tools.is_empty()
            || skill.requires_tools.iter().all(|t| available.contains(t))
        {
            score += 0.4;
        }

        // Meta skills score higher when meta is enabled.
        let meta_enabled = ctx.get_metadata("meta_skill_enabled").is_some();
        if skill.kind == "meta" && meta_enabled {
            score += 0.2;
        }

        // A meta-match or explicit request boosts the skill.
        if let Some(match_name) = ctx.get_metadata("meta_match") {
            if skill.name == *match_name || skill.id == *match_name {
                score += 0.3;
            }
        }
        if let Some(requested) = ctx.get_metadata("requested_skill") {
            if skill.name == *requested || skill.id == *requested {
                score += 0.3;
            }
        }

        // Pinned (always) skills get a small boost.
        if skill.always {
            score += 0.1;
        }

        score.min(1.0)
    }

    /// Rank a skill catalog by eligibility score against the current context.
    ///
    /// Returns the skills sorted by descending eligibility score.
    pub fn rank_by_eligibility(
        &self,
        skills: &[SkillSpec],
        ctx: &PipelineContext,
    ) -> Vec<SkillSpec> {
        let mut scored: Vec<(f64, SkillSpec)> = skills
            .iter()
            .map(|s| (self.eligibility_score(s, ctx), s.clone()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(_, s)| s).collect()
    }

    /// Determine whether a skill is eligible for the current turn.
    ///
    /// This is a stronger check than the deterministic gate: it also verifies
    /// the skill is not disabled by pipeline metadata (`disabled_skill_ids`)
    /// and that meta skills are gated behind the meta-enable flag.
    pub fn is_eligible(&self, skill: &SkillSpec, ctx: &PipelineContext) -> bool {
        // Operator-disabled skills.
        if self
            .config
            .disabled
            .iter()
            .any(|d| d == &skill.id || d == &skill.name)
        {
            return false;
        }
        // Turn-scoped disabled skills.
        if let Some(disabled) = ctx.get_metadata("disabled_skill_ids") {
            if disabled
                .split(',')
                .any(|d| d.trim() == skill.id || d.trim() == skill.name)
            {
                return false;
            }
        }
        // Meta skills require the meta flag.
        if skill.kind == "meta" && ctx.get_metadata("meta_skill_enabled").is_none() {
            return false;
        }
        // Model invocation gating.
        if skill.disable_model_invocation {
            return false;
        }
        // Required tools.
        if !skill.requires_tools.is_empty()
            && !skill
                .requires_tools
                .iter()
                .all(|t| self.available_tools.contains(t))
        {
            return false;
        }
        // Fallback redundancy.
        if skill
            .fallback_for_toolsets
            .iter()
            .any(|t| self.available_tools.contains(t))
        {
            return false;
        }
        true
    }

    /// Filter a catalog to only eligible skills for the turn.
    pub fn eligible_skills<'a>(
        &self,
        skills: &'a [SkillSpec],
        ctx: &PipelineContext,
    ) -> Vec<&'a SkillSpec> {
        skills.iter().filter(|s| self.is_eligible(s, ctx)).collect()
    }

    /// Build the `<available_skills>` prompt block from the final skill list.
    ///
    /// Renders each skill as `<name>description</name>`, truncated to
    /// `max_skills_prompt_chars`. Pinned skills are rendered first.
    pub fn render_prompt(&self, pinned: &[&SkillSpec], filtered: &[&SkillSpec]) -> String {
        let mut out = String::from("<available_skills>\n");
        let mut budget = self.config.max_skills_prompt_chars;
        let push_skill = |out: &mut String, budget: &mut usize, s: &SkillSpec| -> bool {
            let entry = format!("<name>{}</name>\n{}", s.name, s.description);
            let cost = entry.len();
            if *budget < cost && !out.is_empty() {
                return false;
            }
            *budget = budget.saturating_sub(cost);
            out.push_str(&entry);
            out.push('\n');
            true
        };
        for skill in pinned {
            push_skill(&mut out, &mut budget, skill);
        }
        for skill in filtered {
            if budget == 0 {
                break;
            }
            push_skill(&mut out, &mut budget, skill);
        }
        out.push_str("</available_skills>\n");
        out
    }
}

impl Default for SkillsFilterStep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStep for SkillsFilterStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if self.config.memory_only {
            ctx.set_metadata("filtered_skill_ids", "");
            ctx.set_metadata("skill_count", "0");
            ctx.set_metadata("skills_prompt_chars", "0");
            debug!("skills_filter skipped (memory_only)");
            return Ok(StepAction::Continue);
        }

        if self.catalog.is_empty() {
            ctx.set_metadata("skill_count", "0");
            return Ok(StepAction::Continue);
        }

        let total = self.catalog.len();

        // Deterministic gate.
        let gated = self.deterministic_gate(&self.catalog);

        // Hide meta-skills when auto-trigger is unavailable. The engine does
        // not own the meta subsystem's enable flags, so this uses the catalog
        // metadata hint: meta skills are hidden unless explicitly enabled.
        let meta_skill_enabled = ctx.get_metadata("meta_skill_enabled").is_some();
        let gated: Vec<SkillSpec> = if meta_skill_enabled {
            gated
        } else {
            gated.into_iter().filter(|s| s.kind != "meta").collect()
        };

        // Pin always-visible skills and the meta-matched workflow.
        let mut pinned: Vec<SkillSpec> = gated.iter().filter(|s| s.always).cloned().collect();
        let mut filterable: Vec<SkillSpec> = gated.iter().filter(|s| !s.always).cloned().collect();

        if let Some(hint) = ctx.get_metadata("meta_match").cloned() {
            let already = pinned.iter().any(|s| s.name == hint);
            if !already {
                if let Some(pos) = filterable.iter().position(|s| s.name == hint) {
                    pinned.push(filterable.remove(pos));
                }
            }
        }

        // Relevance retrieval (simplified: stable scoring when enabled).
        let filtered: Vec<SkillSpec> = if self.config.filter_enabled {
            let top = self.config.filter_top_k.max(1);
            filterable.into_iter().take(top).collect()
        } else {
            filterable
        };

        // Render and inject the skills prompt.
        let pinned_refs: Vec<&SkillSpec> = pinned.iter().collect();
        let filtered_refs: Vec<&SkillSpec> = filtered.iter().collect();
        let prompt = self.render_prompt(&pinned_refs, &filtered_refs);

        let final_len = pinned.len() + filtered.len();
        if !prompt.is_empty() {
            let combined = match self.config.injection_mode.as_str() {
                "user_message" | "user_context" => {
                    // Append as a fresh system message so the model always
                    // sees the skills block regardless of injection mode.
                    ctx.add_message(Message {
                        role: MessageRole::System,
                        content: vec![opensquilla_core::types::ContentBlock::Text(prompt.clone())],
                        name: None,
                        tool_call_id: None,
                        tool_calls: None,
                        tool_result: None,
                    });
                    prompt.clone()
                }
                _ => prompt.clone(),
            };
            ctx.set_metadata("skills_context_prompt", combined);
        }

        ctx.set_metadata("skill_count", final_len.to_string());
        ctx.set_metadata("skills_prompt_chars", prompt.len().to_string());
        ctx.set_metadata("skills_rendered_count", final_len.to_string());
        ctx.set_metadata("skills_injection_mode", &self.config.injection_mode);
        ctx.set_metadata(
            "filtered_skill_ids",
            filtered
                .iter()
                .map(|s| s.id.clone())
                .collect::<Vec<_>>()
                .join(","),
        );

        let outcome = SkillsFilterOutcome {
            total,
            gated: gated.len(),
            rendered: final_len,
            prompt_chars: prompt.len(),
        };
        let _ = outcome;

        debug!(
            total = total,
            gated = gated.len(),
            rendered = final_len,
            "skills_filter applied"
        );

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "skills_filter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_catalog() -> Vec<SkillSpec> {
        vec![
            SkillSpec::always("always-a", "always-a"),
            SkillSpec::new("git", "git").with_requires(vec!["git_exec".into()]),
            SkillSpec::new("meta-flow", "meta-flow"),
            SkillSpec {
                id: "needs-missing".into(),
                name: "needs-missing".into(),
                description: String::new(),
                kind: "skill".into(),
                always: false,
                disable_model_invocation: false,
                requires_tools: vec!["missing-tool".into()],
                fallback_for_toolsets: Vec::new(),
            },
        ]
    }

    #[test]
    fn test_gate_filters_requires_tools() {
        let step = SkillsFilterStep::new()
            .with_catalog(sample_catalog())
            .with_available_tools(vec!["git_exec".to_string()]);
        let gated = step.deterministic_gate(&step.catalog);
        let names: Vec<&str> = gated.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"always-a"));
        assert!(names.contains(&"git"));
        assert!(!names.contains(&"needs-missing"));
    }

    #[test]
    fn test_render_prompt_pins_first() {
        let catalog = sample_catalog();
        let always: Vec<&SkillSpec> = catalog.iter().filter(|s| s.always).collect();
        let step = SkillsFilterStep::new();
        let prompt = step.render_prompt(&always, &[]);
        assert!(prompt.starts_with("<available_skills>"));
        assert!(prompt.contains("always-a"));
        assert!(prompt.ends_with("</available_skills>\n"));
    }
}

impl SkillSpec {
    /// Builder: add a required tool.
    pub fn with_requires(mut self, tools: impl Into<Vec<String>>) -> Self {
        self.requires_tools = tools.into();
        self
    }

    /// Builder: mark as a meta skill.
    pub fn meta(mut self) -> Self {
        self.kind = "meta".to_string();
        self
    }
}
