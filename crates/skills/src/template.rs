//! # Prompt template management and rendering
//!
//! A thin layer over the [`tera`] template engine providing:
//!
//! - A [`TemplateRegistry`] that stores named prompt templates and renders them
//!   against an arbitrary JSON context.
//! - A library of reusable built-in prompt templates (system prompts, tool
//!   descriptions, classifier instructions, summarizers) that ship with the
//!   binary.
//! - Bilingual (English / Chinese) prompt support via a [`BilingualPrompt`]
//!   that picks the matching language at render time.
//! - Custom filter and function registration so templates can call into
//!   skill-specific helpers (`truncate`, `word_count`, `json_pretty`, …).
//!
//! The registry is deliberately engine-agnostic: it owns a [`tera::Tera`]
//! instance and exposes a small, well-typed surface so callers do not have to
//! depend on tera directly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tera::{Tera, Value};
use thiserror::Error;
use tracing::{debug, warn};

/// Errors raised by the template engine.
#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("template '{name}' not found")]
    NotFound { name: String },
    #[error("template parse error: {0}")]
    Parse(String),
    #[error("template render error: {0}")]
    Render(String),
    #[error("template '{name}' already registered")]
    AlreadyExists { name: String },
}

/// A named prompt template with optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Unique template name (e.g. `"system.helpful_assistant"`).
    pub name: String,
    /// The template body (Tera/Jinja2 syntax).
    pub body: String,
    /// Human-readable description of when to use this template.
    #[serde(default)]
    pub description: String,
    /// Declared input variables (informational; not enforced at render time).
    #[serde(default)]
    pub variables: Vec<String>,
    /// The output language code (e.g. `"en"`, `"zh"`). `"any"` means the
    /// template is language-agnostic.
    #[serde(default = "default_language")]
    pub language: String,
    /// Optional tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Schema version of the template structure itself.
    #[serde(default = "default_template_version")]
    pub version: u32,
}

fn default_language() -> String {
    "any".to_string()
}

fn default_template_version() -> u32 {
    1
}

impl PromptTemplate {
    /// Create a new template.
    pub fn new(name: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            body: body.into(),
            description: String::new(),
            variables: Vec::new(),
            language: default_language(),
            tags: Vec::new(),
            version: 1,
        }
    }

    /// Builder: set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder: set the declared variables.
    pub fn with_variables(mut self, vars: Vec<String>) -> Self {
        self.variables = vars;
        self
    }

    /// Builder: set the language.
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// Builder: add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// A bilingual prompt that renders in the user's preferred language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilingualPrompt {
    /// The English template body.
    pub en: String,
    /// The Chinese template body.
    pub zh: String,
    /// Which language to render by default when none is requested.
    #[serde(default = "default_language")]
    pub default_language: String,
}

impl BilingualPrompt {
    /// Create a bilingual prompt with both language bodies.
    pub fn new(en: impl Into<String>, zh: impl Into<String>) -> Self {
        Self {
            en: en.into(),
            zh: zh.into(),
            default_language: "en".to_string(),
        }
    }

    /// Pick the template body for the given language code.
    pub fn body_for(&self, language: &str) -> &str {
        match language.trim().to_lowercase().as_str() {
            "zh" | "chinese" | "cn" | "zh-cn" | "zh-tw" => &self.zh,
            _ => &self.en,
        }
    }

    /// The body for the default language.
    pub fn default_body(&self) -> &str {
        self.body_for(&self.default_language)
    }
}

/// The template registry. Owns a [`tera::Tera`] instance and a map of named
/// templates.
#[derive(Debug)]
pub struct TemplateRegistry {
    templates: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    tera: Arc<RwLock<Tera>>,
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for TemplateRegistry {
    fn clone(&self) -> Self {
        Self {
            templates: self.templates.clone(),
            tera: self.tera.clone(),
        }
    }
}

impl TemplateRegistry {
    /// Create an empty registry with the standard filters and functions
    /// pre-registered.
    pub fn new() -> Self {
        let mut tera = Tera::default();
        register_standard_filters(&mut tera);
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            tera: Arc::new(RwLock::new(tera)),
        }
    }

    /// Register a template. Returns an error if the template body fails to
    /// parse or the name is already taken.
    pub fn register(&self, template: PromptTemplate) -> Result<(), TemplateError> {
        {
            let mut tera = self.tera.write().map_err(|e| {
                TemplateError::Parse(format!("tera lock poisoned: {e}"))
            })?;
            tera.add_raw_template(&template.name, &template.body)
                .map_err(|e| TemplateError::Parse(e.to_string()))?;
        }
        let mut map = self.templates.write().map_err(|e| {
            TemplateError::Parse(format!("templates lock poisoned: {e}"))
        })?;
        if map.contains_key(&template.name) {
            return Err(TemplateError::AlreadyExists {
                name: template.name.clone(),
            });
        }
        map.insert(template.name.clone(), template);
        Ok(())
    }

    /// Register a template, replacing any existing one with the same name.
    pub fn replace(&self, template: PromptTemplate) -> Result<(), TemplateError> {
        {
            let mut tera = self.tera.write().map_err(|e| {
                TemplateError::Parse(format!("tera lock poisoned: {e}"))
            })?;
            tera.add_raw_template(&template.name, &template.body)
                .map_err(|e| TemplateError::Parse(e.to_string()))?;
        }
        let mut map = self.templates.write().map_err(|e| {
            TemplateError::Parse(format!("templates lock poisoned: {e}"))
        })?;
        map.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template by name.
    pub fn unregister(&self, name: &str) -> Option<PromptTemplate> {
        {
            let mut tera = self.tera.write().ok()?;
            tera.templates.remove(name);
        }
        self.templates.write().ok()?.remove(name)
    }

    /// Look up a template by name.
    pub fn get(&self, name: &str) -> Option<PromptTemplate> {
        self.templates.read().ok()?.get(name).cloned()
    }

    /// Whether a template is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.templates.read().map(|m| m.contains_key(name)).unwrap_or(false)
    }

    /// The number of registered templates.
    pub fn len(&self) -> usize {
        self.templates.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All registered template names.
    pub fn names(&self) -> Vec<String> {
        self.templates
            .read()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Render a named template against a JSON context.
    pub fn render(
        &self,
        name: &str,
        context: &serde_json::Value,
    ) -> Result<String, TemplateError> {
        let tera = self.tera.read().map_err(|e| {
            TemplateError::Render(format!("tera lock poisoned: {e}"))
        })?;
        if !tera.templates.contains_key(name) {
            return Err(TemplateError::NotFound {
                name: name.to_string(),
            });
        }
        let ctx = tera::Context::from_serialize(context)
            .map_err(|e| TemplateError::Render(e.to_string()))?;
        tera.render(name, &ctx)
            .map_err(|e| TemplateError::Render(e.to_string()))
    }

    /// Render a raw template string against a context, without registering it.
    pub fn render_str(
        &self,
        template: &str,
        context: &serde_json::Value,
    ) -> Result<String, TemplateError> {
        let tera = self.tera.read().map_err(|e| {
            TemplateError::Render(format!("tera lock poisoned: {e}"))
        })?;
        let ctx = tera::Context::from_serialize(context)
            .map_err(|e| TemplateError::Render(e.to_string()))?;
        tera.render_str(template, &ctx)
            .map_err(|e| TemplateError::Render(e.to_string()))
    }

    /// Render a bilingual prompt in the requested language.
    pub fn render_bilingual(
        &self,
        prompt: &BilingualPrompt,
        language: &str,
        context: &serde_json::Value,
    ) -> Result<String, TemplateError> {
        let body = prompt.body_for(language);
        self.render_str(body, context)
    }

    /// Register a custom Tera filter.
    pub fn register_filter(
        &self,
        name: &str,
        filter: impl tera::Filter<Value> + Send + Sync + 'static,
    ) -> Result<(), TemplateError> {
        let mut tera = self.tera.write().map_err(|e| {
            TemplateError::Parse(format!("tera lock poisoned: {e}"))
        })?;
        tera.register_filter(name, filter);
        Ok(())
    }

    /// Register a custom Tera function.
    pub fn register_function(
        &self,
        name: &str,
        func: impl tera::Function + Send + Sync + 'static,
    ) -> Result<(), TemplateError> {
        let mut tera = self.tera.write().map_err(|e| {
            TemplateError::Parse(format!("tera lock poisoned: {e}"))
        })?;
        tera.register_function(name, func);
        Ok(())
    }

    /// Render a template, falling back to the raw body on any error. This is
    /// the safe entry point used by the meta orchestrator.
    pub fn render_or_raw(&self, name: &str, context: &serde_json::Value) -> String {
        match self.get(name) {
            Some(template) => match self.render_str(&template.body, context) {
                Ok(rendered) => rendered,
                Err(e) => {
                    warn!("template '{}' render failed: {}", name, e);
                    template.body.clone()
                }
            },
            None => {
                debug!("template '{}' not found, returning empty", name);
                String::new()
            }
        }
    }

    /// Load the built-in prompt templates into this registry.
    pub fn load_builtins(&self) -> Result<usize, TemplateError> {
        for template in builtin_templates() {
            self.register(template)?;
        }
        Ok(builtin_templates().len())
    }
}

/// Register the standard set of Tera filters and functions used across the
/// skill system.
fn register_standard_filters(tera: &mut Tera) {
    tera.register_filter("truncate", filter_truncate);
    tera.register_filter("word_count", filter_word_count);
    tera.register_filter("char_count", filter_char_count);
    tera.register_filter("json_pretty", filter_json_pretty);
    tera.register_filter("lowercase", filter_lowercase);
    tera.register_filter("uppercase", filter_uppercase);
    tera.register_filter("trim", filter_trim);
    tera.register_filter("default", filter_default);
    tera.register_filter("split_lines", filter_split_lines);
    tera.register_filter("first_line", filter_first_line);
    tera.register_filter("indent", filter_indent);

    tera.register_function("uuid", fn_uuid);
    tera.register_function("now_iso", fn_now_iso);
    tera.register_function("ts", fn_ts);
    tera.register_function("env", fn_env);
    tera.register_function("concat", fn_concat);
}

// --- Standard filters ---

fn filter_truncate(value: &Value, args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    let max = args
        .get("length")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(100);
    let ellipsis = args
        .get("ellipsis")
        .and_then(|v| v.as_str())
        .unwrap_or("...");
    if s.chars().count() <= max {
        return Ok(Value::String(s));
    }
    let take = max.saturating_sub(ellipsis.chars().count()).max(0);
    let truncated: String = s.chars().take(take).collect();
    Ok(Value::String(format!("{}{}", truncated, ellipsis)))
}

fn filter_word_count(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    let count = s.split_whitespace().count();
    Ok(Value::Number(count.into()))
}

fn filter_char_count(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    Ok(Value::Number(s.chars().count().into()))
}

fn filter_json_pretty(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    Ok(Value::String(s))
}

fn filter_lowercase(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::String(value_as_string(value).to_lowercase()))
}

fn filter_uppercase(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::String(value_as_string(value).to_uppercase()))
}

fn filter_trim(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::String(value_as_string(value).trim().to_string()))
}

fn filter_default(value: &Value, args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    match value {
        Value::Null | Value::String(s) if s.is_empty() => {
            let default = args
                .get("default")
                .cloned()
                .unwrap_or(Value::String(String::new()));
            Ok(default)
        }
        _ => Ok(value.clone()),
    }
}

fn filter_split_lines(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    let lines: Vec<Value> = s.lines().map(|l| Value::String(l.to_string())).collect();
    Ok(Value::Array(lines))
}

fn filter_first_line(value: &Value, _args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    let first = s.lines().next().unwrap_or("").to_string();
    Ok(Value::String(first))
}

fn filter_indent(value: &Value, args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let s = value_as_string(value);
    let n = args
        .get("n")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(2);
    let prefix: String = " ".repeat(n);
    let indented: String = s
        .lines()
        .map(|l| format!("{}{}", prefix, l))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Value::String(indented))
}

fn value_as_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// --- Standard functions ---

fn fn_uuid(_args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::String(uuid::Uuid::new_v4().to_string()))
}

fn fn_now_iso(_args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::String(chrono::Utc::now().to_rfc3339()))
}

fn fn_ts(_args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    Ok(Value::Number(chrono::Utc::now().timestamp().into()))
}

fn fn_env(args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let name = args
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| tera::Error::msg("env() requires a 'name' argument"))?;
    let default = args.get("default").and_then(|v| v.as_str()).unwrap_or("");
    let value = std::env::var(name).unwrap_or_else(|_| default.to_string());
    Ok(Value::String(value))
}

fn fn_concat(args: &HashMap<String, Value>) -> Result<Value, tera::Error> {
    let mut out = String::new();
    for (key, value) in args {
        if key.starts_with("_") {
            continue;
        }
        out.push_str(&value_as_string(value));
    }
    Ok(Value::String(out))
}

/// The library of built-in prompt templates.
pub fn builtin_templates() -> Vec<PromptTemplate> {
    vec![
        PromptTemplate::new(
            "system.helpful_assistant",
            "You are a helpful, harmless, and honest AI assistant. {{ instructions }}",
        )
        .with_description("A generic helpful assistant system prompt")
        .with_variables(vec!["instructions".to_string()])
        .with_tag("system"),
        PromptTemplate::new(
            "system.skill_executor",
            "You are executing the skill '{{ skill_name }}'.\n\n{{ description }}\n\nFollow the skill's instructions precisely.",
        )
        .with_description("System prompt for executing a named skill")
        .with_variables(vec!["skill_name".to_string(), "description".to_string()])
        .with_tag("system"),
        PromptTemplate::new(
            "classifier.label",
            "You are a deterministic classifier. Read the input and decide which single label applies.\nReply with EXACTLY ONE of: {{ choices | join(', ') }}\nDo not add quotes, punctuation, prefixes, or explanations — emit only the label.\n\nInput: {{ input }}",
        )
        .with_description("Constrained LLM label classifier prompt")
        .with_variables(vec!["choices".to_string(), "input".to_string()])
        .with_tag("classify"),
        PromptTemplate::new(
            "summarizer.compact",
            "Summarize the following conversation for context preservation.\nFocus on key decisions, user preferences, and important facts.\nBe concise — at most {{ max_words }} words.\n\n{% for entry in entries %}[{{ entry.role }}]: {{ entry.content }}\n{% endfor %}\n---\nSummary:",
        )
        .with_description("Conversation summarizer prompt")
        .with_variables(vec!["entries".to_string(), "max_words".to_string()])
        .with_tag("summarize"),
        PromptTemplate::new(
            "tool.description",
            "Tool: {{ name }}\nDescription: {{ description }}\n{% if parameters %}Parameters:\n{{ parameters | json_pretty }}\n{% endif %}",
        )
        .with_description("Tool description template for system prompts")
        .with_variables(vec!["name".to_string(), "description".to_string(), "parameters".to_string()])
        .with_tag("tool"),
        PromptTemplate::new(
            "available_skills",
            "<available_skills>\n{% for skill in skills %}- {{ skill.id }}: {{ skill.description }}\n{% endfor %}</available_skills>",
        )
        .with_description("Available skills listing for system prompts")
        .with_variables(vec!["skills".to_string()])
        .with_tag("listing"),
        PromptTemplate::new(
            "bilingual.greeting",
            "{% if language == 'zh' %}你好{{ name }}{% else %}Hello {{ name }}{% endif %}",
        )
        .with_description("A bilingual greeting template")
        .with_variables(vec!["language".to_string(), "name".to_string()])
        .with_language("any")
        .with_tag("bilingual"),
    ]
}

/// Render a one-off template string against a small JSON context. Falls back to
/// the raw template on any parse or render error.
pub fn render_template_str(template: &str, context: &serde_json::Value) -> String {
    let mut tera = Tera::default();
    register_standard_filters(&mut tera);
    let ctx = match tera::Context::from_serialize(context) {
        Ok(c) => c,
        Err(_) => return template.to_string(),
    };
    tera.render_str(template, &ctx).unwrap_or_else(|e| {
        debug!("template render failed, returning raw: {}", e);
        template.to_string()
    })
}

/// Extract the list of `{{ var }}` variable references in a template string.
/// This is a lightweight scan and may produce false positives for escaped
/// sequences, but it is sufficient for documentation purposes.
pub fn extract_variables(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let re = regex::Regex::new(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)").unwrap();
    for caps in re.captures_iter(template) {
        let name = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        if !name.is_empty() && !out.contains(&name.to_string()) {
            out.push(name.to_string());
        }
    }
    out
}

/// Validate that a template parses without errors. Returns a list of parse
/// errors (empty when the template is valid).
pub fn validate_template(template: &str) -> Vec<String> {
    let mut tera = Tera::default();
    register_standard_filters(&mut tera);
    match tera.add_raw_template("__validation__", template) {
        Ok(_) => Vec::new(),
        Err(e) => vec![e.to_string()],
    }
}

/// Render a template against a simple key-value context map.
pub fn render_with_map(template: &str, vars: &HashMap<String, serde_json::Value>) -> String {
    let context = serde_json::Value::Object(
        vars.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    );
    render_template_str(template, &context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn registry_register_and_render() {
        let reg = TemplateRegistry::new();
        let tpl = PromptTemplate::new("greet", "Hello, {{ name }}!").with_variables(vec!["name".to_string()]);
        reg.register(tpl).unwrap();
        let rendered = reg.render("greet", &json!({"name": "World"})).unwrap();
        assert_eq!(rendered, "Hello, World!");
    }

    #[test]
    fn registry_rejects_duplicate() {
        let reg = TemplateRegistry::new();
        reg.register(PromptTemplate::new("dup", "body"))
            .unwrap();
        let err = reg.register(PromptTemplate::new("dup", "other")).unwrap_err();
        assert!(matches!(err, TemplateError::AlreadyExists { .. }));
    }

    #[test]
    fn render_str_works() {
        let reg = TemplateRegistry::new();
        let out = reg.render_str("{{ x }} + {{ y }}", &json!({"x": 1, "y": 2})).unwrap();
        assert_eq!(out, "1 + 2");
    }

    #[test]
    fn render_str_falls_back_on_error() {
        let reg = TemplateRegistry::new();
        let out = reg.render_str("{{ x ", &json!({})).unwrap_err();
        assert!(matches!(out, TemplateError::Render(_)));
    }

    #[test]
    fn builtin_templates_load() {
        let reg = TemplateRegistry::new();
        let count = reg.load_builtins().unwrap();
        assert!(count > 0);
        assert!(reg.contains("system.helpful_assistant"));
        assert!(reg.contains("classifier.label"));
    }

    #[test]
    fn truncate_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ text | truncate(length=5) }}",
                &json!({"text": "hello world"}),
            )
            .unwrap();
        assert_eq!(out, "he...");
    }

    #[test]
    fn word_count_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str("{{ text | word_count }}", &json!({"text": "one two three"}))
            .unwrap();
        assert_eq!(out, "3");
    }

    #[test]
    fn json_pretty_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str("{{ obj | json_pretty }}", &json!({"obj": {"a": 1}}))
            .unwrap();
        assert!(out.contains("\"a\": 1"));
    }

    #[test]
    fn default_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ x | default(default='fallback') }}",
                &json!({"x": ""}),
            )
            .unwrap();
        assert_eq!(out, "fallback");
    }

    #[test]
    fn uuid_and_ts_functions_work() {
        let reg = TemplateRegistry::new();
        let uuid = reg.render_str("{{ uuid() }}", &json!({})).unwrap();
        assert_eq!(uuid.len(), 36);
        let ts = reg.render_str("{{ ts() }}", &json!({})).unwrap();
        assert!(ts.parse::<i64>().is_ok());
    }

    #[test]
    fn env_function_reads_env() {
        std::env::set_var("OSQ_TEMPLATE_TEST", "hello-env");
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ env(name='OSQ_TEMPLATE_TEST') }}",
                &json!({}),
            )
            .unwrap();
        assert_eq!(out, "hello-env");
        std::env::remove_var("OSQ_TEMPLATE_TEST");
    }

    #[test]
    fn bilingual_prompt_picks_language() {
        let prompt = BilingualPrompt::new("Hello", "你好");
        assert_eq!(prompt.body_for("en"), "Hello");
        assert_eq!(prompt.body_for("zh"), "你好");
        assert_eq!(prompt.body_for("zh-CN"), "你好");
        assert_eq!(prompt.body_for("fr"), "Hello");
    }

    #[test]
    fn bilingual_render_works() {
        let reg = TemplateRegistry::new();
        let prompt = BilingualPrompt::new("Hello {{ name }}", "你好 {{ name }}");
        let en = reg.render_bilingual(&prompt, "en", &json!({"name": "World"})).unwrap();
        assert_eq!(en, "Hello World");
        let zh = reg.render_bilingual(&prompt, "zh", &json!({"name": "World"})).unwrap();
        assert_eq!(zh, "你好 World");
    }

    #[test]
    fn extract_variables_finds_refs() {
        let vars = extract_variables("Hello {{ name }}, your score is {{ score }} and team is {{ team.name }}");
        assert!(vars.contains(&"name".to_string()));
        assert!(vars.contains(&"score".to_string()));
        assert!(vars.contains(&"team.name".to_string()));
    }

    #[test]
    fn validate_template_catches_error() {
        let errs = validate_template("{{ unclosed");
        assert!(!errs.is_empty());
    }

    #[test]
    fn validate_template_passes_clean() {
        let errs = validate_template("Hello {{ name }}");
        assert!(errs.is_empty());
    }

    #[test]
    fn render_with_map_works() {
        let mut map = HashMap::new();
        map.insert("name".to_string(), json!("Alice"));
        let out = render_with_map("Hi {{ name }}", &map);
        assert_eq!(out, "Hi Alice");
    }

    #[test]
    fn render_or_raw_falls_back() {
        let reg = TemplateRegistry::new();
        let tpl = PromptTemplate::new("bad", "Hello {{ name }}");
        reg.register(tpl).unwrap();
        let out = reg.render_or_raw("bad", &json!({}));
        assert!(out.contains("Hello"));
    }

    #[test]
    fn unregister_removes_template() {
        let reg = TemplateRegistry::new();
        reg.register(PromptTemplate::new("temp", "body")).unwrap();
        assert!(reg.contains("temp"));
        reg.unregister("temp");
        assert!(!reg.contains("temp"));
    }

    #[test]
    fn replace_overrides_existing() {
        let reg = TemplateRegistry::new();
        reg.register(PromptTemplate::new("x", "first")).unwrap();
        reg.replace(PromptTemplate::new("x", "second")).unwrap();
        let out = reg.render_str("{{ x }}", &json!({})).unwrap_or_default();
        // render_str does not use registered templates, so just check the stored body
        let stored = reg.get("x").unwrap();
        assert_eq!(stored.body, "second");
        let _ = out;
    }

    #[test]
    fn indent_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ text | indent(n=2) }}",
                &json!({"text": "line1\nline2"}),
            )
            .unwrap();
        assert!(out.contains("  line1"));
        assert!(out.contains("  line2"));
    }

    #[test]
    fn split_lines_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ text | split_lines | length }}",
                &json!({"text": "a\nb\nc"}),
            )
            .unwrap();
        assert_eq!(out, "3");
    }

    #[test]
    fn first_line_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg
            .render_str(
                "{{ text | first_line }}",
                &json!({"text": "first line\nsecond line"}),
            )
            .unwrap();
        assert_eq!(out, "first line");
    }

    #[test]
    fn lowercase_and_uppercase_filters_work() {
        let reg = TemplateRegistry::new();
        let lower = reg.render_str("{{ text | lowercase }}", &json!({"text": "HeLLo"})).unwrap();
        assert_eq!(lower, "hello");
        let upper = reg.render_str("{{ text | uppercase }}", &json!({"text": "HeLLo"})).unwrap();
        assert_eq!(upper, "HELLO");
    }

    #[test]
    fn trim_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg.render_str("{{ text | trim }}", &json!({"text": "  hello  "})).unwrap();
        assert_eq!(out, "hello");
    }

    #[test]
    fn char_count_filter_works() {
        let reg = TemplateRegistry::new();
        let out = reg.render_str("{{ text | char_count }}", &json!({"text": "hello"})).unwrap();
        assert_eq!(out, "5");
    }

    #[test]
    fn names_lists_templates() {
        let reg = TemplateRegistry::new();
        reg.register(PromptTemplate::new("a", "x")).unwrap();
        reg.register(PromptTemplate::new("b", "y")).unwrap();
        let names = reg.names();
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn classifier_label_renders() {
        let reg = TemplateRegistry::new();
        reg.load_builtins().unwrap();
        let out = reg
            .render(
                "classifier.label",
                &json!({"choices": ["yes", "no"], "input": "do you want tea?"}),
            )
            .unwrap();
        assert!(out.contains("yes, no"));
        assert!(out.contains("do you want tea?"));
    }

    #[test]
    fn available_skills_renders() {
        let reg = TemplateRegistry::new();
        reg.load_builtins().unwrap();
        let out = reg
            .render(
                "available_skills",
                &json!({"skills": [{"id": "git", "description": "vcs"}, {"id": "web", "description": "search"}]}),
            )
            .unwrap();
        assert!(out.contains("git: vcs"));
        assert!(out.contains("web: search"));
    }
}
