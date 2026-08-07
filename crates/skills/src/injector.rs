//! # Skill injection
//!
//! The [`SkillInjector`] renders the `<available_skills>` block that is placed
//! into the model's system prompt, and is responsible for deciding *which*
//! skills survive into that block under a hard token budget.
//!
//! Responsibilities:
//!
//! - Render skills in XML (default), Markdown, or plain-text form.
//! - Enforce a token budget by ranking and truncating the lowest-value skills.
//! - Filter skills by eligibility (delegating to
//!   [`crate::eligibility::EligibilityChecker`]).
//! - Activate skills dynamically based on the current conversation context
//!   (user message, available tools, pinned skills).
//!
//! The injection logic is pure over the input skills and context; it performs
//! no I/O of its own.

use crate::eligibility::EligibilityChecker;
use crate::types::{SkillMatch, SkillScope, SkillSpec, rank_skills};
use std::collections::HashSet;

/// The rendering format for the injected skills block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkillListFormat {
    /// `<available_skills>` XML block (the default).
    #[default]
    Xml,
    /// A compact Markdown list.
    Markdown,
    /// Plain text, one skill per line.
    Plain,
}

/// Tuning knobs for the injector.
#[derive(Debug, Clone)]
pub struct InjectorConfig {
    /// Maximum number of tokens the rendered block may consume.
    pub max_tokens: usize,
    /// Whether to render step-level usage for meta-skills.
    pub include_usage: bool,
    /// Whether to render the `requires` block.
    pub include_requirements: bool,
    /// Whether to render tags and metadata.
    pub include_metadata: bool,
    /// Whether to run eligibility checks before rendering.
    pub filter_eligible: bool,
    /// The rendering format.
    pub format: SkillListFormat,
    /// Maximum description length in characters.
    pub description_max_chars: usize,
    /// Include disabled skills when `true` (usually `false`).
    pub include_disabled: bool,
    /// Group the rendered list by layer.
    pub group_by_layer: bool,
    /// Footer text appended after the block.
    pub footer_text: String,
}

impl Default for InjectorConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            include_usage: true,
            include_requirements: true,
            include_metadata: true,
            filter_eligible: false,
            format: SkillListFormat::Xml,
            description_max_chars: 400,
            include_disabled: false,
            group_by_layer: false,
            footer_text: String::new(),
        }
    }
}

/// The conversation context used to decide which skills are relevant.
#[derive(Debug, Clone, Default)]
pub struct InjectionContext {
    /// The user's current message, if any.
    pub user_message: Option<String>,
    /// Recent conversation turns (plain text), most recent first.
    pub conversation: Vec<String>,
    /// Tools available to the agent this turn.
    pub available_tools: Vec<String>,
    /// The scope the session is running in.
    pub scope: SkillScope,
    /// Skill ids the operator disabled.
    pub disabled: HashSet<String>,
    /// Skill ids that must always be rendered.
    pub pinned: HashSet<String>,
    /// Optional query terms used to boost relevance scoring.
    pub query_terms: Vec<String>,
}

impl InjectionContext {
    /// An empty context.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Create a context from just a user message.
    pub fn from_message(message: impl Into<String>) -> Self {
        Self {
            user_message: Some(message.into()),
            ..Self::default()
        }
    }

    /// The combined text used for relevance scoring.
    pub fn text(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(msg) = &self.user_message {
            parts.push(msg);
        }
        for turn in &self.conversation {
            parts.push(turn);
        }
        let query = self.query_terms.join(" ");
        parts.push(&query);
        parts.join("\n")
    }
}

/// The skill injector.
pub struct SkillInjector {
    config: InjectorConfig,
    /// Optional eligibility checker used when `filter_eligible` is enabled.
    eligibility: Option<EligibilityChecker>,
}

impl Default for SkillInjector {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillInjector {
    /// Create an injector with default configuration.
    pub fn new() -> Self {
        Self {
            config: InjectorConfig::default(),
            eligibility: None,
        }
    }

    /// Create an injector with a custom token budget.
    pub fn with_token_budget(max_tokens: usize) -> Self {
        Self {
            config: InjectorConfig {
                max_tokens,
                ..InjectorConfig::default()
            },
            eligibility: None,
        }
    }

    /// Build an injector from a full configuration.
    pub fn with_config(config: InjectorConfig) -> Self {
        Self {
            config,
            eligibility: None,
        }
    }

    /// Enable eligibility filtering using a shared checker.
    pub fn with_eligibility(mut self, checker: EligibilityChecker) -> Self {
        self.eligibility = Some(checker);
        self
    }

    /// The injector configuration.
    pub fn config(&self) -> &InjectorConfig {
        &self.config
    }

    /// Update the injector configuration.
    pub fn set_config(&mut self, config: InjectorConfig) {
        self.config = config;
    }

    /// The maximum token budget.
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }

    // -----------------------------------------------------------------------
    // Token estimation
    // -----------------------------------------------------------------------

    /// Estimate the token count of a string.
    ///
    /// The estimator is a deliberate approximation of a BPE tokenizer: Latin
    /// text costs ~4 characters per token, CJK characters cost ~1 token each,
    /// and whitespace/control runs are cheap. This is monotonic in practice and
    /// deterministic, which is what the budget logic needs.
    pub fn estimate_tokens(&self, text: &str) -> usize {
        estimate_string_tokens(text)
    }

    /// Estimate the token count of a rendered skills block.
    pub fn estimate_block_tokens(&self, block: &str) -> usize {
        self.estimate_tokens(block)
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    /// Render the `<available_skills>` XML block for a list of skills.
    pub fn render_skills_block(&self, skills: &[&SkillSpec]) -> String {
        let entries: Vec<String> = skills.iter().map(|s| self.render_entry(s)).collect();
        self.wrap_block(entries)
    }

    /// Render skills, respecting the token budget by truncating the
    /// lowest-priority skills.
    pub fn render_within_budget(&self, skills: &[SkillSpec]) -> String {
        self.render_within_budget_impl(skills, &InjectionContext::empty())
    }

    /// Render the skills block as a system prompt addition.
    pub fn render_system_prompt(&self, skills: &[SkillSpec]) -> String {
        let block = self.render_within_budget(skills);
        format!(
            r#"You have access to the following skills. Use them when appropriate:

{block}

To use a skill, reference it by its id in your response.
"#
        )
    }

    /// Full pipeline: filter by context, rank, trim to budget, render.
    ///
    /// This is the primary entry point for the turn runner. It:
    /// 1. removes disabled and (optionally) ineligible skills,
    /// 2. pins always-on skills first,
    /// 3. ranks the rest against the conversation context,
    /// 4. truncates to the token budget, and
    /// 5. renders the surviving skills in the configured format.
    pub fn render(&self, skills: &[SkillSpec], context: &InjectionContext) -> String {
        let selected = self.select_skills(skills, context);
        self.render_within_budget_impl(&selected, context)
    }

    /// Render a full system-prompt wrapper using context-aware selection.
    pub fn render_prompt_for(&self, skills: &[SkillSpec], context: &InjectionContext) -> String {
        let block = self.render(skills, context);
        format!(
            r#"You have access to the following skills. Use them when appropriate:

{block}

To use a skill, reference it by its id in your response.
"#
        )
    }

    /// Select the skills that should be injected for a given context, without
    /// rendering. Honors pinning, disabling, eligibility, and relevance.
    pub fn select_skills(
        &self,
        skills: &[SkillSpec],
        context: &InjectionContext,
    ) -> Vec<SkillSpec> {
        let mut candidates: Vec<SkillSpec> = skills
            .iter()
            .filter(|s| !s.disabled || self.config.include_disabled)
            .filter(|s| !context.disabled.contains(&s.id))
            .filter(|s| s.in_scope(context.scope))
            .filter(|s| {
                if self.config.filter_eligible {
                    self.eligibility
                        .as_ref()
                        .map(|c| c.is_eligible(&s.requires).unwrap_or(false))
                        .unwrap_or(true)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        if self.config.filter_eligible && self.eligibility.is_none() {
            // Eligibility requested but no checker provided: construct one.
            let checker = EligibilityChecker::new();
            candidates.retain(|s| checker.is_eligible(&s.requires).unwrap_or(true));
        }

        // Sort pinned first, then by layer priority.
        candidates.sort_by(|a, b| {
            let a_pinned = context.pinned.contains(&a.id) || a.is_always();
            let b_pinned = context.pinned.contains(&b.id) || b.is_always();
            b_pinned
                .cmp(&a_pinned)
                .then_with(|| b.layer.priority().cmp(&a.layer.priority()))
        });

        candidates
    }

    /// Rank skills by relevance to the context (dynamic activation).
    pub fn rank_for_context(
        &self,
        skills: &[SkillSpec],
        context: &InjectionContext,
    ) -> Vec<SkillMatch> {
        let text = context.text();
        if text.trim().is_empty() {
            // No context: rank by layer priority and pin status.
            let mut ranked: Vec<SkillMatch> = skills
                .iter()
                .cloned()
                .map(|s| {
                    let pinned = context.pinned.contains(&s.id) || s.is_always();
                    SkillMatch {
                        skill: s,
                        score: if pinned { 1.0 } else { 0.0 },
                        matched_fields: Vec::new(),
                        matched_terms: Vec::new(),
                    }
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.skill.layer.priority().cmp(&a.skill.layer.priority()))
            });
            return ranked;
        }

        // Combine lexical scoring with tool availability and pin bonuses.
        let mut ranked = rank_skills(&text, skills);
        for m in &mut ranked {
            if context.pinned.contains(&m.skill.id) || m.skill.is_always() {
                m.score = (m.score + 0.5).min(1.0);
            }
            // A skill that references an available tool gets a small boost.
            let tools = m.skill.tool_names();
            if tools.iter().any(|t| context.available_tools.contains(t)) {
                m.score = (m.score + 0.1).min(1.0);
            }
        }
        ranked
    }

    /// Render a single skill entry in the configured format.
    pub fn render_entry(&self, skill: &SkillSpec) -> String {
        match self.config.format {
            SkillListFormat::Xml => self.render_entry_xml(skill),
            SkillListFormat::Markdown => self.render_entry_markdown(skill),
            SkillListFormat::Plain => self.render_entry_plain(skill),
        }
    }

    /// Render the block wrapper around a list of rendered entries.
    pub fn wrap_block(&self, entries: Vec<String>) -> String {
        match self.config.format {
            SkillListFormat::Xml => {
                let mut xml = String::from("<available_skills>\n");
                for entry in entries {
                    xml.push_str(&entry);
                }
                xml.push_str("</available_skills>\n");
                if !self.config.footer_text.is_empty() {
                    xml.push_str(&self.config.footer_text);
                    xml.push('\n');
                }
                xml
            }
            SkillListFormat::Markdown => {
                let mut md = String::from("## Available Skills\n\n");
                for entry in entries {
                    md.push_str(&entry);
                }
                if !self.config.footer_text.is_empty() {
                    md.push('\n');
                    md.push_str(&self.config.footer_text);
                    md.push('\n');
                }
                md
            }
            SkillListFormat::Plain => {
                let mut out = String::from("Available skills:\n");
                for entry in entries {
                    out.push_str(&entry);
                }
                if !self.config.footer_text.is_empty() {
                    out.push_str(&self.config.footer_text);
                    out.push('\n');
                }
                out
            }
        }
    }

    /// Truncate rendered entries to the token budget, dropping from the end.
    pub fn trim_to_budget(&self, entries: Vec<(SkillSpec, String)>) -> Vec<(SkillSpec, String)> {
        let mut budget = self.config.max_tokens;
        let wrapper_cost = self.estimate_tokens(&self.wrap_block(Vec::new()));
        budget = budget.saturating_sub(wrapper_cost);

        let mut kept = Vec::new();
        for (spec, entry) in entries {
            let cost = self.estimate_tokens(&entry);
            if budget < cost && !kept.is_empty() {
                break;
            }
            budget = budget.saturating_sub(cost);
            kept.push((spec, entry));
        }
        kept
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn render_within_budget_impl(
        &self,
        skills: &[SkillSpec],
        context: &InjectionContext,
    ) -> String {
        let mut sorted: Vec<&SkillSpec> = skills.iter().collect();
        sorted.sort_by(|a, b| {
            let a_pinned = context.pinned.contains(&a.id) || a.is_always();
            let b_pinned = context.pinned.contains(&b.id) || b.is_always();
            b_pinned
                .cmp(&a_pinned)
                .then_with(|| b.layer.priority().cmp(&a.layer.priority()))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut entries: Vec<(SkillSpec, String)> = Vec::new();
        let mut estimated_tokens = 0usize;
        let wrapper = self.wrap_block(Vec::new());
        estimated_tokens += self.estimate_tokens(&wrapper);

        for skill in &sorted {
            let entry = self.render_entry(skill);
            let entry_tokens = self.estimate_tokens(&entry);
            if estimated_tokens + entry_tokens <= self.config.max_tokens {
                entries.push(((*skill).clone(), entry));
                estimated_tokens += entry_tokens;
            } else {
                break;
            }
        }

        if self.config.group_by_layer {
            entries.sort_by_key(|a| std::cmp::Reverse(a.0.layer.priority()));
        }

        let rendered: Vec<String> = entries.into_iter().map(|(_, e)| e).collect();
        self.wrap_block(rendered)
    }

    fn render_entry_xml(&self, skill: &SkillSpec) -> String {
        let mut entry = String::new();
        entry.push_str(&format!(
            "  <skill id=\"{}\" name=\"{}\">\n",
            escape_xml(&skill.id),
            escape_xml(&skill.name)
        ));
        entry.push_str(&format!(
            "    <description>{}</description>\n",
            escape_xml(&self.trim_description(&skill.description))
        ));

        if let Some(ref version) = skill.version {
            entry.push_str(&format!("    <version>{}</version>\n", escape_xml(version)));
        }
        if self.config.include_metadata && !skill.tags.is_empty() {
            let tags: Vec<&str> = skill.tags.iter().map(|t| t.as_str()).collect();
            entry.push_str(&format!(
                "    <tags>{}</tags>\n",
                escape_xml(&tags.join(", "))
            ));
        }
        if self.config.include_metadata {
            if let Some(ref license) = skill.license {
                entry.push_str(&format!("    <license>{}</license>\n", escape_xml(license)));
            }
            if let Some(ref homepage) = skill.homepage {
                entry.push_str(&format!(
                    "    <homepage>{}</homepage>\n",
                    escape_xml(homepage)
                ));
            }
        }

        if self.config.include_requirements {
            entry.push_str(&self.render_requires_xml(skill));
        }

        if self.config.include_usage && skill.is_meta() {
            entry.push_str("    <usage>\n");
            for step in &skill.steps {
                entry.push_str(&format!(
                    "      <step id=\"{}\" type=\"{}\">{}</step>\n",
                    escape_xml(&step.id),
                    step.step_type,
                    escape_xml(&step.name)
                ));
            }
            entry.push_str("    </usage>\n");
        }

        entry.push_str("  </skill>\n");
        entry
    }

    fn render_requires_xml(&self, skill: &SkillSpec) -> String {
        let mut out = String::new();
        let requires = &skill.requires;
        if let Some(ref os) = requires.os {
            if !os.is_empty() {
                out.push_str(&format!("    <os>{}</os>\n", escape_xml(&os.join(", "))));
            }
        }
        if let Some(ref binaries) = requires.binaries {
            if !binaries.is_empty() {
                out.push_str(&format!(
                    "    <binaries>{}</binaries>\n",
                    escape_xml(&binaries.join(", "))
                ));
            }
        }
        if let Some(ref env_vars) = requires.env_vars {
            if !env_vars.is_empty() {
                out.push_str(&format!(
                    "    <env_vars>{}</env_vars>\n",
                    escape_xml(&env_vars.join(", "))
                ));
            }
        }
        if let Some(ref files) = requires.files {
            if !files.is_empty() {
                out.push_str(&format!(
                    "    <files>{}</files>\n",
                    escape_xml(&files.join(", "))
                ));
            }
        }
        out
    }

    fn render_entry_markdown(&self, skill: &SkillSpec) -> String {
        let mut out = String::new();
        out.push_str(&format!("- **{}**", escape_md(&skill.name)));
        if let Some(ref version) = skill.version {
            out.push_str(&format!(" (v{})", escape_md(version)));
        }
        out.push_str(&format!(" — `{}`", escape_md(&skill.id)));
        out.push('\n');
        let desc = self.trim_description(&skill.description);
        if !desc.is_empty() {
            out.push_str(&format!("  {}\n", escape_md(&desc)));
        }
        if self.config.include_usage && skill.is_meta() {
            out.push_str(&format!("  _meta-skill: {} steps_\n", skill.steps.len()));
        }
        out
    }

    fn render_entry_plain(&self, skill: &SkillSpec) -> String {
        let mut out = String::new();
        out.push_str(&format!("{} [{}]", skill.name, skill.id));
        if let Some(ref version) = skill.version {
            out.push_str(&format!(" v{version}"));
        }
        out.push_str(": ");
        out.push_str(&self.trim_description(&skill.description));
        out.push('\n');
        out
    }

    fn trim_description(&self, description: &str) -> String {
        let max = self.config.description_max_chars;
        if description.chars().count() <= max {
            description.to_string()
        } else {
            let cut: String = description.chars().take(max).collect();
            format!("{cut}…")
        }
    }
}

/// Estimate the token count of a string with a deterministic heuristic.
pub fn estimate_string_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut tokens = 0usize;
    let mut cjk_run = 0usize;
    let mut ascii_run = 0usize;

    for c in text.chars() {
        if is_cjk(c) {
            // CJK chars are dense: ~1 token each.
            ascii_run = 0;
            cjk_run += 1;
            if cjk_run >= 1 {
                tokens += 1;
                cjk_run = 0;
            }
        } else {
            cjk_run = 0;
            ascii_run += 1;
            // Latin/ASCII: ~4 chars per token.
            if ascii_run >= 4 {
                tokens += 1;
                ascii_run = 0;
            }
        }
    }
    if ascii_run > 0 {
        tokens += 1;
    }
    tokens.max(1)
}

/// Whether a character is in a CJK-ish block that BPE-style tokenizers treat
/// as dense.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{30FF}' // Hiragana + Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
    )
}

/// XML-escape a string.
pub fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Escape Markdown emphasis characters.
pub fn escape_md(s: &str) -> String {
    s.replace('*', "\\*")
        .replace('_', "\\_")
        .replace('`', "\\`")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SkillLayer;

    fn spec(id: &str, layer: SkillLayer, description: &str) -> SkillSpec {
        let mut s = SkillSpec::new(
            id.to_string(),
            id.to_string(),
            description.to_string(),
            layer,
        );
        s.version = Some("1.0.0".to_string());
        s.tags = vec!["test".to_string()];
        s
    }

    #[test]
    fn render_xml_block() {
        let injector = SkillInjector::new();
        let a = spec("alpha", SkillLayer::Bundled, "Alpha skill");
        let block = injector.render_skills_block(&[&a]);
        assert!(block.starts_with("<available_skills>"));
        assert!(block.contains("<skill id=\"alpha\""));
        assert!(block.contains("<description>Alpha skill</description>"));
        assert!(block.ends_with("</available_skills>\n"));
    }

    #[test]
    fn xml_escaping() {
        let injector = SkillInjector::new();
        let mut a = spec("a", SkillLayer::Bundled, "has <angle> & ampersand");
        a.name = "A & B <C>".to_string();
        let block = injector.render_skills_block(&[&a]);
        assert!(block.contains("&lt;angle&gt;"));
        assert!(block.contains("&amp;"));
        assert!(!block.contains("A & B <C>"));
    }

    #[test]
    fn token_budget_truncates() {
        let injector = SkillInjector::with_token_budget(200);
        let skills: Vec<SkillSpec> = (0..20)
            .map(|i| {
                spec(
                    &format!("skill-{i}"),
                    SkillLayer::Bundled,
                    &format!("Skill number {i} with a longer description to consume tokens"),
                )
            })
            .collect();
        let block = injector.render_within_budget(&skills);
        let tokens = injector.estimate_tokens(&block);
        assert!(tokens <= 200, "block used {tokens} tokens");
        // The highest-priority skill is always included.
        assert!(block.contains("skill-0") || block.contains("skill-19"));
    }

    #[test]
    fn estimate_tokens_cjk() {
        let injector = SkillInjector::new();
        let latin = injector.estimate_tokens("the quick brown fox jumps over the lazy dog");
        assert!(latin > 5 && latin < 20);
        let cjk = injector.estimate_tokens("这是一个技能描述用来测试");
        assert!(cjk >= 9, "CJK counted as {cjk}");
    }

    #[test]
    fn select_respects_disabled() {
        let injector = SkillInjector::new();
        let mut a = spec("a", SkillLayer::Bundled, "A");
        a.disabled = true;
        let b = spec("b", SkillLayer::Bundled, "B");
        let ctx = InjectionContext::empty();
        let selected = injector.select_skills(&[a, b], &ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "b");
    }

    #[test]
    fn select_respects_context_disabled() {
        let injector = SkillInjector::new();
        let a = spec("a", SkillLayer::Bundled, "A");
        let b = spec("b", SkillLayer::Bundled, "B");
        let mut ctx = InjectionContext::empty();
        ctx.disabled.insert("a".to_string());
        let selected = injector.select_skills(&[a, b], &ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].id, "b");
    }

    #[test]
    fn pinned_skills_come_first() {
        let injector = SkillInjector::new();
        let a = spec("alpha", SkillLayer::Bundled, "A");
        let b = spec("beta", SkillLayer::Bundled, "B");
        let mut ctx = InjectionContext::empty();
        ctx.pinned.insert("beta".to_string());
        let selected = injector.select_skills(&[a, b], &ctx);
        assert_eq!(selected[0].id, "beta");
    }

    #[test]
    fn rank_uses_context() {
        let injector = SkillInjector::new();
        let git = spec("git", SkillLayer::Bundled, "interact with git repositories");
        let memory = spec("memory", SkillLayer::Bundled, "search durable memory");
        let ctx = InjectionContext::from_message("help me commit my git changes");
        let ranked = injector.rank_for_context(&[git, memory], &ctx);
        assert_eq!(ranked[0].skill.id, "git");
    }

    #[test]
    fn markdown_format() {
        let injector = SkillInjector::with_config(InjectorConfig {
            format: SkillListFormat::Markdown,
            ..InjectorConfig::default()
        });
        let a = spec("alpha", SkillLayer::Bundled, "Alpha skill");
        let block = injector.render_skills_block(&[&a]);
        assert!(block.contains("## Available Skills"));
        assert!(block.contains("- **alpha**"));
    }

    #[test]
    fn trim_description_respects_max() {
        let injector = SkillInjector::with_config(InjectorConfig {
            description_max_chars: 10,
            ..InjectorConfig::default()
        });
        let long = "this description is much longer than ten characters";
        let trimmed = injector.trim_description(long);
        assert!(trimmed.chars().count() <= 11);
        assert!(trimmed.ends_with('…'));
    }

    #[test]
    fn empty_skills_produce_empty_block() {
        let injector = SkillInjector::new();
        let block = injector.render_skills_block(&[]);
        assert_eq!(block, "<available_skills>\n</available_skills>\n");
    }

    #[test]
    fn render_prompt_includes_footer() {
        let injector = SkillInjector::with_config(InjectorConfig {
            footer_text: "Footer here.".to_string(),
            ..InjectorConfig::default()
        });
        let a = spec("a", SkillLayer::Bundled, "A");
        let block = injector.render_skills_block(&[&a]);
        assert!(block.contains("Footer here."));
    }
}
