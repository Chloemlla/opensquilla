//! # Skill manifest schema, validation, and versioning
//!
//! [`SkillManifest`] is the wire type produced by deserializing a `SKILL.md`
//! frontmatter block. This module adds a focused layer on top of the raw
//! [`crate::types::SkillManifest`] struct:
//!
//! - A versioned [`ManifestSchema`] descriptor with compatibility checks.
//! - A [`ManifestValidator`] that enforces the full OpenClaw skill manifest
//!   schema (required fields, kind/step consistency, dependency cycles, …) and
//!   returns a structured [`ValidationReport`].
//! - Serialization helpers that round-trip a manifest through YAML while
//!   preserving unknown keys and stable key ordering.
//!
//! The validator is deliberately permissive about *unknown* keys (so
//! community-authored manifests keep parsing) but strict about the keys it
//! does understand: a meta-skill with no steps is an error, a `tool_call`
//! step without a `tool` is an error, and so on.

use crate::types::{
    SkillAuthor, SkillDependency, SkillKind, SkillLayer, SkillLicense, SkillManifest,
    SkillMetadata, SkillRequires, SkillSpec, SkillStep, SkillVersion, SkillVisibility, StepType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// The current manifest schema version.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Supported manifest schema versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestSchema {
    /// The original, unversioned frontmatter shape. Treated as `v1`.
    Legacy,
    /// Version 1 — the schema documented in the OpenClaw skill spec.
    V1,
}

impl ManifestSchema {
    /// Detect the schema version from a parsed manifest. Any manifest with an
    /// explicit `schema_version` of `1` (or no `schema_version` at all) maps
    /// to [`ManifestSchema::V1`]; an unknown value maps to [`ManifestSchema::Legacy`].
    pub fn from_manifest(manifest: &SkillManifest) -> Self {
        let raw = manifest.extra.get("schema_version").and_then(|v| {
            v.as_u64()
                .map(|n| n as u32)
                .or_else(|| v.as_str().and_then(|s| s.parse::<u32>().ok()))
        });
        match raw {
            Some(1) | None => ManifestSchema::V1,
            _ => ManifestSchema::Legacy,
        }
    }

    /// The numeric schema version.
    pub fn version(self) -> u32 {
        match self {
            ManifestSchema::Legacy => 0,
            ManifestSchema::V1 => 1,
        }
    }

    /// Whether manifests authored against this schema are still readable by
    /// the current binary.
    pub fn is_supported(self) -> bool {
        matches!(self, ManifestSchema::V1 | ManifestSchema::Legacy)
    }
}

impl std::fmt::Display for ManifestSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestSchema::Legacy => write!(f, "legacy"),
            ManifestSchema::V1 => write!(f, "v1"),
        }
    }
}

/// A single validation problem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// The field path the issue applies to, e.g. `steps[1].tool`.
    pub field: String,
    /// Human-readable description.
    pub message: String,
}

/// Issue severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// The manifest is unusable; loaders must reject it.
    Error,
    /// The manifest is usable but the issue should be fixed.
    Warning,
    /// Informational note; no action required.
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Info => write!(f, "info"),
        }
    }
}

/// The result of validating a manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    /// The schema version the manifest was validated against.
    pub schema: Option<ManifestSchema>,
    /// All issues found, in evaluation order.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether the report contains any error-severity issues.
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Only the error-severity issues.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect()
    }

    /// Only the warning-severity issues.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect()
    }

    /// Whether the manifest passed validation entirely (no issues at all).
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// A flat list of `"severity: field — message"` strings for logging.
    pub fn to_log_lines(&self) -> Vec<String> {
        self.issues
            .iter()
            .map(|i| format!("{}: {} — {}", i.severity, i.field, i.message))
            .collect()
    }
}

/// Errors raised by manifest serialization/deserialization helpers.
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("YAML serialization error: {0}")]
    Yaml(String),
    #[error("JSON serialization error: {0}")]
    Json(String),
    #[error("validation failed: {0}")]
    Validation(String),
}

/// Validates [`SkillManifest`] instances against the OpenClaw schema.
///
/// The validator is constructed with a [`ManifestValidatorConfig`] that tunes
/// which checks are strict and which are advisory.
#[derive(Debug, Clone)]
pub struct ManifestValidator {
    config: ManifestValidatorConfig,
}

/// Tunable knobs for [`ManifestValidator`].
#[derive(Debug, Clone)]
pub struct ManifestValidatorConfig {
    /// Require the `id` field to be a non-empty slug.
    pub require_id: bool,
    /// Require the `name` field.
    pub require_name: bool,
    /// Require the `description` field.
    pub require_description: bool,
    /// Require the `version` field to parse as a [`SkillVersion`].
    pub require_version: bool,
    /// Reject manifests whose `requires` block references unknown binaries.
    pub warn_on_missing_binary_requires: bool,
    /// Enforce that meta-skills declare at least one step.
    pub require_meta_steps: bool,
    /// Enforce that non-meta skills do *not* declare steps.
    pub reject_non_meta_steps: bool,
    /// Enforce step id uniqueness within a manifest.
    pub require_unique_step_ids: bool,
    /// Enforce that every `depends_on` refers to an existing step id.
    pub require_known_dependencies: bool,
    /// Reject dependency cycles in the step DAG.
    pub reject_cycles: bool,
    /// Enforce that `tool_call` steps declare a `tool`.
    pub require_tool_for_tool_call: bool,
    /// Enforce that `skill_exec` steps declare a `skill`.
    pub require_skill_for_skill_exec: bool,
    /// Enforce that `llm_classify` steps declare `output_choices`.
    pub require_choices_for_classify: bool,
    /// Warn when a manifest carries unknown top-level keys.
    pub warn_on_unknown_keys: bool,
}

impl Default for ManifestValidatorConfig {
    fn default() -> Self {
        Self {
            require_id: true,
            require_name: true,
            require_description: true,
            require_version: false,
            warn_on_missing_binary_requires: false,
            require_meta_steps: true,
            reject_non_meta_steps: true,
            require_unique_step_ids: true,
            require_known_dependencies: true,
            reject_cycles: true,
            require_tool_for_tool_call: true,
            require_skill_for_skill_exec: true,
            require_choices_for_classify: true,
            warn_on_unknown_keys: false,
        }
    }
}

impl Default for ManifestValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ManifestValidator {
    /// Create a validator with the default config.
    pub fn new() -> Self {
        Self {
            config: ManifestValidatorConfig::default(),
        }
    }

    /// Create a validator with a custom config.
    pub fn with_config(config: ManifestValidatorConfig) -> Self {
        Self { config }
    }

    /// The active validator configuration.
    pub fn config(&self) -> &ManifestValidatorConfig {
        &self.config
    }

    /// Validate a manifest, returning a structured report.
    pub fn validate(&self, manifest: &SkillManifest) -> ValidationReport {
        let mut report = ValidationReport {
            schema: Some(ManifestSchema::from_manifest(manifest)),
            issues: Vec::new(),
        };

        self.validate_identifiers(manifest, &mut report);
        self.validate_kind(manifest, &mut report);
        self.validate_steps(manifest, &mut report);
        self.validate_dependencies(manifest, &mut report);
        self.validate_args(manifest, &mut report);
        self.validate_contexts(manifest, &mut report);
        self.validate_metadata(manifest, &mut report);
        self.validate_unknown_keys(manifest, &mut report);

        report
    }

    /// Validate a manifest and return `Ok(())` when it has no errors, or
    /// `Err(ManifestError::Validation(..))` describing the errors.
    pub fn validate_or_error(&self, manifest: &SkillManifest) -> Result<(), ManifestError> {
        let report = self.validate(manifest);
        if report.has_errors() {
            let msgs: Vec<String> = report.errors().iter().map(|i| i.message.clone()).collect();
            Err(ManifestError::Validation(msgs.join("; ")))
        } else {
            Ok(())
        }
    }

    fn validate_identifiers(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        if self.config.require_id {
            match &manifest.id {
                None => report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: "id".to_string(),
                    message: "skill id is required".to_string(),
                }),
                Some(id) if id.is_empty() => report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: "id".to_string(),
                    message: "skill id must not be empty".to_string(),
                }),
                Some(id) if !is_valid_slug(id) => report.issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    field: "id".to_string(),
                    message: format!("skill id '{}' should be a lowercase slug (a-z0-9_-)", id),
                }),
                _ => {}
            }
        }

        if self.config.require_name && manifest.name.as_deref().is_none_or(|n| n.is_empty()) {
            report.issues.push(ValidationIssue {
                severity: Severity::Error,
                field: "name".to_string(),
                message: "skill name is required".to_string(),
            });
        }

        if self.config.require_description
            && manifest.description.as_deref().is_none_or(|d| d.is_empty())
        {
            report.issues.push(ValidationIssue {
                severity: Severity::Warning,
                field: "description".to_string(),
                message: "skill description is recommended".to_string(),
            });
        }

        if self.config.require_version {
            if let Some(v) = &manifest.version {
                if v.parse::<SkillVersion>().is_err() {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: "version".to_string(),
                        message: format!("version '{}' is not a valid semantic version", v),
                    });
                }
            } else {
                report.issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    field: "version".to_string(),
                    message: "version is recommended".to_string(),
                });
            }
        }
    }

    fn validate_kind(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        let is_meta = manifest.kind.as_ref().is_some_and(|k| k.is_meta());
        let has_steps = !manifest.steps.is_empty();

        if is_meta && self.config.require_meta_steps && !has_steps {
            report.issues.push(ValidationIssue {
                severity: Severity::Error,
                field: "steps".to_string(),
                message: "meta-skill declares no steps".to_string(),
            });
        }
        if !is_meta && has_steps && self.config.reject_non_meta_steps {
            report.issues.push(ValidationIssue {
                severity: Severity::Warning,
                field: "steps".to_string(),
                message: "non-meta skill declares steps; set kind: meta to use them".to_string(),
            });
        }
    }

    fn validate_steps(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        let mut seen: HashMap<String, usize> = HashMap::new();
        for (i, step) in manifest.steps.iter().enumerate() {
            let path = format!("steps[{}]", i);

            if step.id.is_empty() {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: format!("{}.id", path),
                    message: "step id is required".to_string(),
                });
            }
            if self.config.require_unique_step_ids {
                if let Some(prev) = seen.get(&step.id) {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: format!("{}.id", path),
                        message: format!(
                            "step id '{}' is duplicated (first seen at steps[{}])",
                            step.id, prev
                        ),
                    });
                } else {
                    seen.insert(step.id.clone(), i);
                }
            }

            self.validate_step_kind(step, &path, report);
            self.validate_step_output(step, &path, report);
        }
    }

    fn validate_step_kind(&self, step: &SkillStep, path: &str, report: &mut ValidationReport) {
        match step.step_type {
            StepType::ToolCall => {
                if self.config.require_tool_for_tool_call
                    && step.tool.as_deref().is_none_or(|t| t.is_empty())
                {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: format!("{}.tool", path),
                        message: format!("step '{}' (tool_call) has no tool", step.id),
                    });
                }
            }
            StepType::SkillExec => {
                if self.config.require_skill_for_skill_exec
                    && step.skill.as_deref().is_none_or(|s| s.is_empty())
                {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: format!("{}.skill", path),
                        message: format!("step '{}' (skill_exec) has no skill", step.id),
                    });
                }
            }
            StepType::LlmClassify => {
                if self.config.require_choices_for_classify && step.output_choices.is_empty() {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: format!("{}.output_choices", path),
                        message: format!("step '{}' (llm_classify) has no output_choices", step.id),
                    });
                }
            }
            StepType::LlmChat | StepType::Agent => {
                if step.prompt.as_deref().is_none_or(|p| p.is_empty())
                    && !step.with_args.contains_key("task")
                    && !step.with_args.contains_key("prompt")
                {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        field: format!("{}.prompt", path),
                        message: format!(
                            "step '{}' ({}) has no prompt or task",
                            step.id, step.step_type
                        ),
                    });
                }
            }
            StepType::UserInput => {
                // user_input steps are validated by their `with_args.fields` shape.
                if !step.with_args.contains_key("fields") {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Info,
                        field: format!("{}.with_args.fields", path),
                        message: format!(
                            "step '{}' (user_input) has no fields; handler will receive an empty list",
                            step.id
                        ),
                    });
                }
            }
        }
    }

    fn validate_step_output(&self, step: &SkillStep, path: &str, report: &mut ValidationReport) {
        if let Some(out) = &step.output {
            if let Some(route) = &out.route_to {
                if route.is_empty() {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Error,
                        field: format!("{}.output.route_to", path),
                        message: format!("step '{}' has empty route_to", step.id),
                    });
                }
            }
            if let Some(pick) = &out.pick {
                if pick.is_empty() {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        field: format!("{}.output.pick", path),
                        message: format!("step '{}' has empty pick list", step.id),
                    });
                }
            }
            let _ = out;
        }
    }

    fn validate_dependencies(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        let ids: std::collections::HashSet<&str> =
            manifest.steps.iter().map(|s| s.id.as_str()).collect();
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        for step in &manifest.steps {
            graph.insert(step.id.as_str(), Vec::new());
        }

        for (i, step) in manifest.steps.iter().enumerate() {
            let path = format!("steps[{}].depends_on", i);
            if let Some(deps) = &step.depends_on {
                for dep in deps {
                    if !ids.contains(dep.as_str()) {
                        if self.config.require_known_dependencies {
                            report.issues.push(ValidationIssue {
                                severity: Severity::Error,
                                field: path.clone(),
                                message: format!(
                                    "step '{}' depends on unknown step '{}'",
                                    step.id, dep
                                ),
                            });
                        }
                    } else if let Some(adj) = graph.get_mut(step.id.as_str()) {
                        adj.push(dep.as_str());
                    }
                }
            }
        }

        if self.config.reject_cycles {
            if let Some(cycle) = detect_cycle(&graph) {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: "steps.depends_on".to_string(),
                    message: format!(
                        "dependency cycle detected among steps: {}",
                        cycle.join(" -> ")
                    ),
                });
            }
        }
    }

    fn validate_args(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        let mut seen: HashMap<String, usize> = HashMap::new();
        for (i, arg) in manifest.args.iter().enumerate() {
            if arg.name.is_empty() {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: format!("args[{}].name", i),
                    message: "argument name is required".to_string(),
                });
                continue;
            }
            if let Some(prev) = seen.get(&arg.name) {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: format!("args[{}].name", i),
                    message: format!(
                        "argument '{}' is duplicated (first seen at args[{}])",
                        arg.name, prev
                    ),
                });
            } else {
                seen.insert(arg.name.clone(), i);
            }
            if !arg.enum_values.is_empty()
                && arg.default.is_some()
                && !arg
                    .enum_values
                    .iter()
                    .any(|v| Some(serde_json::Value::String(v.clone())) == arg.default)
            {
                report.issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    field: format!("args[{}].default", i),
                    message: format!(
                        "argument '{}' default is not one of its enum_values",
                        arg.name
                    ),
                });
            }
        }
    }

    fn validate_contexts(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        let mut seen: HashMap<String, usize> = HashMap::new();
        for (i, ctx) in manifest.contexts.iter().enumerate() {
            if ctx.name.is_empty() {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: format!("contexts[{}].name", i),
                    message: "context name is required".to_string(),
                });
                continue;
            }
            if let Some(prev) = seen.get(&ctx.name) {
                report.issues.push(ValidationIssue {
                    severity: Severity::Error,
                    field: format!("contexts[{}].name", i),
                    message: format!(
                        "context '{}' is duplicated (first seen at contexts[{}])",
                        ctx.name, prev
                    ),
                });
            } else {
                seen.insert(ctx.name.clone(), i);
            }
        }
    }

    fn validate_metadata(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        if let Some(meta) = &manifest.metadata {
            if let Some(lang) = &meta.language {
                if !is_valid_language_code(lang) {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Info,
                        field: "metadata.language".to_string(),
                        message: format!(
                            "language code '{}' is not a recognized ISO-639-1 code",
                            lang
                        ),
                    });
                }
            }
            if let Some(temp) = meta.temperature {
                if !(0.0..=2.0).contains(&temp) {
                    report.issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        field: "metadata.temperature".to_string(),
                        message: format!(
                            "temperature {} is outside the typical [0, 2] range",
                            temp
                        ),
                    });
                }
            }
        }
    }

    fn validate_unknown_keys(&self, manifest: &SkillManifest, report: &mut ValidationReport) {
        if !self.config.warn_on_unknown_keys {
            return;
        }
        let known: std::collections::HashSet<&str> = [
            "id",
            "name",
            "kind",
            "description",
            "version",
            "author",
            "license",
            "homepage",
            "layer",
            "visibility",
            "scope",
            "tags",
            "requires",
            "metadata",
            "allowed-tools",
            "disable-model-invocation",
            "disable_model_invocation",
            "steps",
            "outputs",
            "contexts",
            "args",
            "dependencies",
            "changelog",
        ]
        .iter()
        .copied()
        .collect();
        for key in manifest.extra.keys() {
            if !known.contains(key.as_str()) {
                report.issues.push(ValidationIssue {
                    severity: Severity::Info,
                    field: key.clone(),
                    message: format!("unknown top-level key '{}'", key),
                });
            }
        }
    }
}

/// Detect a cycle in a dependency graph using DFS. Returns the cycle path if
/// one exists.
fn detect_cycle<'a>(graph: &HashMap<&'a str, Vec<&'a str>>) -> Option<Vec<&'a str>> {
    let mut state: HashMap<&str, u8> = HashMap::new(); // 0=unvisited, 1=in-progress, 2=done
    for node in graph.keys() {
        if state.get(node).copied().unwrap_or(0) == 0 {
            let mut path: Vec<&'a str> = Vec::new();
            if let Some(cycle) = dfs_cycle(graph, node, &mut state, &mut path) {
                return Some(cycle);
            }
        }
    }
    None
}

fn dfs_cycle<'a>(
    graph: &HashMap<&'a str, Vec<&'a str>>,
    node: &'a str,
    state: &mut HashMap<&'a str, u8>,
    path: &mut Vec<&'a str>,
) -> Option<Vec<&'a str>> {
    state.insert(node, 1);
    path.push(node);
    if let Some(adj) = graph.get(node) {
        for neighbor in adj {
            let s = state.get(neighbor).copied().unwrap_or(0);
            if s == 1 {
                let start = path.iter().position(|n| *n == *neighbor);
                if let Some(start_idx) = start {
                    let cycle: Vec<&'a str> = path[start_idx..].to_vec();
                    return Some(cycle);
                }
                return Some(vec![*neighbor, node]);
            }
            if s == 0 {
                if let Some(cycle) = dfs_cycle(graph, neighbor, state, path) {
                    return Some(cycle);
                }
            }
        }
    }
    path.pop();
    state.insert(node, 2);
    None
}

/// Whether a string is a valid lowercase slug.
fn is_valid_slug(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
        && s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
}

/// Whether a string looks like a 2-letter ISO-639-1 or 3-letter ISO-639-2
/// language code.
fn is_valid_language_code(s: &str) -> bool {
    let s = s.trim();
    s.len() == 2 || s.len() == 3
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a manifest to a stable, sorted YAML string.
///
/// Unknown keys are preserved (via the `extra` map) but the well-known fields
/// are written in a fixed order so diffs stay readable.
pub fn manifest_to_yaml(manifest: &SkillManifest) -> Result<String, ManifestError> {
    let value = manifest_to_ordered_value(manifest);
    serde_yaml::to_string(&value).map_err(|e| ManifestError::Yaml(e.to_string()))
}

/// Serialize a manifest to a pretty-printed JSON string.
pub fn manifest_to_json(manifest: &SkillManifest) -> Result<String, ManifestError> {
    let value = manifest_to_ordered_value(manifest);
    serde_json::to_string_pretty(&value).map_err(|e| ManifestError::Json(e.to_string()))
}

/// Convert a manifest into an ordered [`serde_json::Value`] preserving unknown
/// keys at the end. The well-known fields are emitted in declaration order.
pub fn manifest_to_ordered_value(manifest: &SkillManifest) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    macro_rules! opt_str {
        ($field:expr, $key:literal) => {
            if let Some(v) = $field {
                map.insert($key.to_string(), serde_json::Value::String(v.clone()));
            }
        };
    }
    opt_str!(manifest.id.as_ref(), "id");
    opt_str!(manifest.name.as_ref(), "name");
    if let Some(kind) = &manifest.kind {
        map.insert(
            "kind".to_string(),
            serde_json::to_value(kind).unwrap_or(serde_json::Value::Null),
        );
    }
    opt_str!(manifest.description.as_ref(), "description");
    opt_str!(manifest.version.as_ref(), "version");
    if !matches!(manifest.author, Some(serde_json::Value::Null) | None) {
        if let Some(author) = &manifest.author {
            map.insert("author".to_string(), author.clone());
        }
    }
    opt_str!(manifest.license.as_ref(), "license");
    opt_str!(manifest.homepage.as_ref(), "homepage");
    opt_str!(manifest.layer.as_ref(), "layer");
    opt_str!(manifest.visibility.as_ref(), "visibility");
    opt_str!(manifest.scope.as_ref(), "scope");
    if !manifest.tags.is_empty() {
        map.insert(
            "tags".to_string(),
            serde_json::to_value(&manifest.tags).unwrap_or(serde_json::Value::Null),
        );
    }
    if let Some(requires) = &manifest.requires {
        map.insert(
            "requires".to_string(),
            serde_json::to_value(requires).unwrap_or(serde_json::Value::Null),
        );
    }
    if let Some(metadata) = &manifest.metadata {
        map.insert(
            "metadata".to_string(),
            serde_json::to_value(metadata).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.allowed_tools.is_empty() {
        map.insert(
            "allowed-tools".to_string(),
            serde_json::to_value(&manifest.allowed_tools).unwrap_or(serde_json::Value::Null),
        );
    }
    if manifest.disable_model_invocation {
        map.insert(
            "disable_model_invocation".to_string(),
            serde_json::Value::Bool(true),
        );
    }
    if !manifest.steps.is_empty() {
        map.insert(
            "steps".to_string(),
            serde_json::to_value(&manifest.steps).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.outputs.is_empty() {
        map.insert(
            "outputs".to_string(),
            serde_json::to_value(&manifest.outputs).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.contexts.is_empty() {
        map.insert(
            "contexts".to_string(),
            serde_json::to_value(&manifest.contexts).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.args.is_empty() {
        map.insert(
            "args".to_string(),
            serde_json::to_value(&manifest.args).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.dependencies.is_empty() {
        map.insert(
            "dependencies".to_string(),
            serde_json::to_value(&manifest.dependencies).unwrap_or(serde_json::Value::Null),
        );
    }
    if !manifest.changelog.is_empty() {
        map.insert(
            "changelog".to_string(),
            serde_json::to_value(&manifest.changelog).unwrap_or(serde_json::Value::Null),
        );
    }
    for (k, v) in &manifest.extra {
        map.insert(k.clone(), v.clone());
    }
    serde_json::Value::Object(map)
}

/// Deserialize a manifest from a YAML string.
pub fn manifest_from_yaml(yaml: &str) -> Result<SkillManifest, ManifestError> {
    serde_yaml::from_str(yaml).map_err(|e| ManifestError::Yaml(e.to_string()))
}

/// Deserialize a manifest from a JSON string.
pub fn manifest_from_json(json: &str) -> Result<SkillManifest, ManifestError> {
    serde_json::from_str(json).map_err(|e| ManifestError::Json(e.to_string()))
}

/// Build a minimal valid manifest programmatically.
pub fn minimal_manifest(id: &str, name: &str, description: &str) -> SkillManifest {
    SkillManifest {
        id: Some(id.to_string()),
        name: Some(name.to_string()),
        description: Some(description.to_string()),
        ..Default::default()
    }
}

/// Build a meta-skill manifest with the given steps.
pub fn meta_manifest(
    id: &str,
    name: &str,
    description: &str,
    steps: Vec<SkillStep>,
) -> SkillManifest {
    SkillManifest {
        id: Some(id.to_string()),
        name: Some(name.to_string()),
        description: Some(description.to_string()),
        kind: Some(SkillKind::Meta),
        steps,
        ..Default::default()
    }
}

/// Merge two manifests: `base` provides defaults and `override_` wins on every
/// field it sets. Lists and maps are merged element-wise.
pub fn merge_manifests(base: &SkillManifest, override_: &SkillManifest) -> SkillManifest {
    let mut out = base.clone();
    if override_.id.is_some() {
        out.id = override_.id.clone();
    }
    if override_.name.is_some() {
        out.name = override_.name.clone();
    }
    if override_.kind.is_some() {
        out.kind = override_.kind.clone();
    }
    if override_.description.is_some() {
        out.description = override_.description.clone();
    }
    if override_.version.is_some() {
        out.version = override_.version.clone();
    }
    if override_.author.is_some() {
        out.author = override_.author.clone();
    }
    if override_.license.is_some() {
        out.license = override_.license.clone();
    }
    if override_.homepage.is_some() {
        out.homepage = override_.homepage.clone();
    }
    if override_.layer.is_some() {
        out.layer = override_.layer.clone();
    }
    if override_.visibility.is_some() {
        out.visibility = override_.visibility.clone();
    }
    if override_.scope.is_some() {
        out.scope = override_.scope.clone();
    }
    if !override_.tags.is_empty() {
        let mut tags = base.tags.clone();
        for t in &override_.tags {
            if !tags.contains(t) {
                tags.push(t.clone());
            }
        }
        out.tags = tags;
    }
    if override_.requires.is_some() {
        out.requires = override_.requires.clone();
    }
    if override_.metadata.is_some() {
        out.metadata = override_.metadata.clone();
    }
    if !override_.allowed_tools.is_empty() {
        let mut tools = base.allowed_tools.clone();
        for t in &override_.allowed_tools {
            if !tools.contains(t) {
                tools.push(t.clone());
            }
        }
        out.allowed_tools = tools;
    }
    if override_.disable_model_invocation {
        out.disable_model_invocation = true;
    }
    if !override_.steps.is_empty() {
        out.steps = override_.steps.clone();
    }
    if !override_.outputs.is_empty() {
        for (k, v) in &override_.outputs {
            out.outputs.insert(k.clone(), v.clone());
        }
    }
    if !override_.contexts.is_empty() {
        let mut seen: std::collections::HashSet<String> =
            base.contexts.iter().map(|c| c.name.clone()).collect();
        for ctx in &override_.contexts {
            if seen.insert(ctx.name.clone()) {
                out.contexts.push(ctx.clone());
            }
        }
    }
    if !override_.args.is_empty() {
        let mut seen: std::collections::HashSet<String> =
            base.args.iter().map(|a| a.name.clone()).collect();
        for arg in &override_.args {
            if seen.insert(arg.name.clone()) {
                out.args.push(arg.clone());
            }
        }
    }
    if !override_.dependencies.is_empty() {
        out.dependencies
            .extend(override_.dependencies.iter().cloned());
    }
    if !override_.changelog.is_empty() {
        out.changelog.extend(override_.changelog.iter().cloned());
    }
    for (k, v) in &override_.extra {
        out.extra.insert(k.clone(), v.clone());
    }
    out
}

/// Compute the set of tags declared anywhere in a manifest: top-level tags,
/// metadata.classification, and metadata.triggers.
pub fn collect_tags(manifest: &SkillManifest) -> Vec<String> {
    let mut tags: Vec<String> = manifest.tags.clone();
    if let Some(meta) = &manifest.metadata {
        if let Some(classification) = &meta.classification {
            if !tags.contains(classification) {
                tags.push(classification.clone());
            }
        }
        for t in &meta.triggers {
            if !tags.contains(t) {
                tags.push(t.clone());
            }
        }
    }
    tags.sort();
    tags.dedup();
    tags
}

/// Extract a list of all tool names referenced by a manifest (explicit
/// allowed-tools, requires.tools, and Tool dependencies).
pub fn collect_tool_names(manifest: &SkillManifest) -> Vec<String> {
    let mut tools: Vec<String> = manifest.allowed_tools.clone();
    if let Some(requires) = &manifest.requires {
        if let Some(req_tools) = &requires.tools {
            for t in req_tools {
                if !tools.contains(t) {
                    tools.push(t.clone());
                }
            }
        }
    }
    for dep in &manifest.dependencies {
        if let SkillDependency::Tool { name, .. } = dep {
            if !tools.contains(name) {
                tools.push(name.clone());
            }
        }
    }
    tools.sort();
    tools.dedup();
    tools
}

/// Convert a manifest into a compact summary used by the hub index.
pub fn manifest_summary(manifest: &SkillManifest) -> ManifestSummary {
    ManifestSummary {
        id: manifest.id.clone().unwrap_or_default(),
        name: manifest.name.clone().unwrap_or_default(),
        description: manifest.description.clone().unwrap_or_default(),
        version: manifest.version.clone().unwrap_or_default(),
        kind: manifest.kind.clone().unwrap_or_default(),
        tags: collect_tags(manifest),
        tools: collect_tool_names(manifest),
        step_count: manifest.steps.len(),
        schema: ManifestSchema::from_manifest(manifest).version(),
    }
}

/// A compact summary of a manifest, used for indexing and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub kind: SkillKind,
    pub tags: Vec<String>,
    pub tools: Vec<String>,
    pub step_count: usize,
    pub schema: u32,
}

/// Upgrade a legacy manifest to the current schema by filling in defaults.
pub fn upgrade_manifest(mut manifest: SkillManifest) -> SkillManifest {
    if manifest.kind.is_none() && !manifest.steps.is_empty() {
        manifest.kind = Some(SkillKind::Meta);
    }
    if manifest.id.is_none()
        && let Some(name) = &manifest.name
    {
        manifest.id = Some(name.to_lowercase().replace(' ', "_"));
    }
    if manifest.name.is_none() && manifest.id.is_some() {
        manifest.name = manifest.id.clone();
    }
    manifest
}

/// Extract a [`SkillAuthor`] from the manifest's free-form `author` field.
pub fn extract_author(manifest: &SkillManifest) -> Option<SkillAuthor> {
    match &manifest.author {
        Some(serde_json::Value::String(s)) => Some(SkillAuthor::new(s.clone())),
        Some(serde_json::Value::Object(map)) => {
            let name = map.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let mut author = SkillAuthor::new(name);
            author.email = map.get("email").and_then(|v| v.as_str()).map(String::from);
            author.url = map.get("url").and_then(|v| v.as_str()).map(String::from);
            author.organization = map
                .get("organization")
                .and_then(|v| v.as_str())
                .map(String::from);
            if name.is_empty() { None } else { Some(author) }
        }
        _ => None,
    }
}

/// Extract a [`SkillLicense`] from the manifest's `license` field.
pub fn extract_license(manifest: &SkillManifest) -> Option<SkillLicense> {
    manifest
        .license
        .as_ref()
        .map(|s| SkillLicense::new(s.clone()))
}

/// Extract a [`SkillRequires`] from the manifest, defaulting when absent.
pub fn extract_requires(manifest: &SkillManifest) -> SkillRequires {
    manifest.requires.clone().unwrap_or_default()
}

/// Extract a [`SkillMetadata`] from the manifest, defaulting when absent.
pub fn extract_metadata(manifest: &SkillManifest) -> SkillMetadata {
    manifest.metadata.clone().unwrap_or_default()
}

/// Extract the [`SkillVisibility`] declared by the manifest.
pub fn extract_visibility(manifest: &SkillManifest) -> SkillVisibility {
    manifest
        .visibility
        .as_deref()
        .and_then(SkillVisibility::from_str_loose)
        .unwrap_or(SkillVisibility::Personal)
}

/// Extract the [`SkillLayer`] declared by the manifest (defaulting to Managed).
pub fn extract_layer(manifest: &SkillManifest) -> SkillLayer {
    manifest
        .layer
        .as_deref()
        .and_then(SkillLayer::from_str_loose)
        .unwrap_or(SkillLayer::Managed)
}

/// Convert a manifest into a fully-populated [`SkillSpec`] for the given layer
/// and source path. This is the public, dependency-free variant of
/// [`crate::loader::manifest_to_spec`]; both produce equivalent specs.
pub fn manifest_to_spec_public(
    manifest: SkillManifest,
    layer: SkillLayer,
    source_path: Option<String>,
    body: String,
    raw_frontmatter: String,
) -> Result<SkillSpec, ManifestError> {
    let validator = ManifestValidator::new();
    validator.validate_or_error(&manifest)?;

    let id = manifest.id.clone().unwrap_or_default();
    let name = manifest.name.clone().unwrap_or_else(|| id.clone());
    let description = manifest.description.clone().unwrap_or_default();

    let mut spec = SkillSpec::new(id, name, description, layer);
    // All reads via `&manifest` must finish before any field is moved out of
    // the manifest below.
    spec.author = extract_author(&manifest).map(|a| a.name);
    spec.license = manifest.license.clone();
    spec.homepage = manifest.homepage.clone();
    spec.tags = collect_tags(&manifest);
    spec.allowed_tools = collect_tool_names(&manifest);
    spec.requires = extract_requires(&manifest);
    spec.metadata = Some(extract_metadata(&manifest));
    spec.visibility = extract_visibility(&manifest);

    spec.kind = manifest.kind.unwrap_or(SkillKind::Skill);
    spec.version = manifest.version;
    spec.steps = manifest.steps;
    spec.outputs = manifest.outputs;
    spec.raw_frontmatter = raw_frontmatter;
    spec.source_path = source_path;
    spec.body = body;
    spec.disable_model_invocation = manifest.disable_model_invocation;
    spec.contexts = manifest.contexts;
    spec.args = manifest.args;
    spec.dependencies = manifest.dependencies;
    spec.scope = manifest
        .scope
        .as_deref()
        .and_then(crate::types::SkillScope::from_str_loose)
        .unwrap_or_default();
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_manifest() -> SkillManifest {
        minimal_manifest("my-skill", "My Skill", "A test skill")
    }

    #[test]
    fn schema_detection() {
        let m = basic_manifest();
        assert_eq!(ManifestSchema::from_manifest(&m), ManifestSchema::V1);

        let mut m = basic_manifest();
        m.extra
            .insert("schema_version".to_string(), serde_json::json!(1));
        assert_eq!(ManifestSchema::from_manifest(&m), ManifestSchema::V1);

        let mut m = basic_manifest();
        m.extra
            .insert("schema_version".to_string(), serde_json::json!(99));
        assert_eq!(ManifestSchema::from_manifest(&m), ManifestSchema::Legacy);
    }

    #[test]
    fn validator_passes_clean_manifest() {
        let m = basic_manifest();
        let report = ManifestValidator::new().validate(&m);
        assert!(!report.has_errors(), "{:?}", report.issues);
    }

    #[test]
    fn validator_catches_missing_id() {
        let mut m = basic_manifest();
        m.id = None;
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(report.issues.iter().any(|i| i.field == "id"));
    }

    #[test]
    fn validator_catches_meta_without_steps() {
        let mut m = basic_manifest();
        m.kind = Some(SkillKind::Meta);
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(report.issues.iter().any(|i| i.field == "steps"));
    }

    #[test]
    fn validator_catches_duplicate_step_ids() {
        let mut m = meta_manifest(
            "wf",
            "Workflow",
            "desc",
            vec![
                SkillStep::new("s1", StepType::Agent),
                SkillStep::new("s1", StepType::LlmChat),
            ],
        );
        m.steps[1].prompt = Some("hello".to_string());
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.field == "steps[1].id" && i.message.contains("duplicated"))
        );
    }

    #[test]
    fn validator_catches_tool_call_without_tool() {
        let m = meta_manifest(
            "wf",
            "Workflow",
            "desc",
            vec![SkillStep::new("s1", StepType::ToolCall)],
        );
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(report.issues.iter().any(|i| i.field == "steps[0].tool"));
    }

    #[test]
    fn validator_catches_unknown_dependency() {
        let mut step = SkillStep::new("s1", StepType::Agent);
        step.prompt = Some("do thing".to_string());
        step.depends_on = Some(vec!["ghost".to_string()]);
        let m = meta_manifest("wf", "Workflow", "desc", vec![step]);
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.field == "steps[0].depends_on")
        );
    }

    #[test]
    fn validator_catches_dependency_cycle() {
        let mut a = SkillStep::new("a", StepType::Agent);
        a.prompt = Some("a".to_string());
        a.depends_on = Some(vec!["b".to_string()]);
        let mut b = SkillStep::new("b", StepType::Agent);
        b.prompt = Some("b".to_string());
        b.depends_on = Some(vec!["a".to_string()]);
        let m = meta_manifest("wf", "Workflow", "desc", vec![a, b]);
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(report.issues.iter().any(|i| i.message.contains("cycle")));
    }

    #[test]
    fn validator_catches_llm_classify_without_choices() {
        let mut step = SkillStep::new("s1", StepType::LlmClassify);
        step.prompt = Some("classify".to_string());
        let m = meta_manifest("wf", "Workflow", "desc", vec![step]);
        let report = ManifestValidator::new().validate(&m);
        assert!(report.has_errors());
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.field == "steps[0].output_choices")
        );
    }

    #[test]
    fn yaml_roundtrip_preserves_fields() {
        let mut m = basic_manifest();
        m.version = Some("1.2.3".to_string());
        m.tags = vec!["test".to_string()];
        let yaml = manifest_to_yaml(&m).unwrap();
        let back = manifest_from_yaml(&yaml).unwrap();
        assert_eq!(back.id, m.id);
        assert_eq!(back.name, m.name);
        assert_eq!(back.version, m.version);
        assert_eq!(back.tags, m.tags);
    }

    #[test]
    fn json_roundtrip_preserves_fields() {
        let mut m = basic_manifest();
        m.version = Some("2.0.0".to_string());
        let json = manifest_to_json(&m).unwrap();
        let back = manifest_from_json(&json).unwrap();
        assert_eq!(back.id, m.id);
        assert_eq!(back.version, m.version);
    }

    #[test]
    fn merge_manifests_overrides_correctly() {
        let mut base = basic_manifest();
        base.tags = vec!["a".to_string()];
        base.description = Some("base desc".to_string());

        let mut over = basic_manifest();
        over.description = Some("overridden desc".to_string());
        over.tags = vec!["b".to_string()];

        let merged = merge_manifests(&base, &over);
        assert_eq!(merged.description, Some("overridden desc".to_string()));
        assert!(merged.tags.contains(&"a".to_string()));
        assert!(merged.tags.contains(&"b".to_string()));
    }

    #[test]
    fn collect_tags_dedups() {
        let mut m = basic_manifest();
        m.tags = vec!["rust".to_string(), "async".to_string()];
        m.metadata = Some(SkillMetadata {
            classification: Some("memory".to_string()),
            triggers: vec!["rust".to_string()],
            ..Default::default()
        });
        let tags = collect_tags(&m);
        assert!(tags.contains(&"async".to_string()));
        assert!(tags.contains(&"memory".to_string()));
        assert_eq!(tags.iter().filter(|t| **t == "rust").count(), 1);
    }

    #[test]
    fn collect_tool_names_from_all_sources() {
        let mut m = basic_manifest();
        m.allowed_tools = vec!["read_file".to_string()];
        m.requires = Some(SkillRequires {
            tools: Some(vec!["git".to_string()]),
            ..Default::default()
        });
        m.dependencies = vec![SkillDependency::Tool {
            name: "web_search".to_string(),
            version: None,
        }];
        let tools = collect_tool_names(&m);
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"git".to_string()));
        assert!(tools.contains(&"web_search".to_string()));
    }

    #[test]
    fn upgrade_manifest_fills_kind() {
        let mut m = basic_manifest();
        m.kind = None;
        m.steps = vec![SkillStep::new("s1", StepType::Agent)];
        let upgraded = upgrade_manifest(m);
        assert_eq!(upgraded.kind, Some(SkillKind::Meta));
    }

    #[test]
    fn extract_author_from_string() {
        let mut m = basic_manifest();
        m.author = Some(serde_json::Value::String("Jane Doe".to_string()));
        let author = extract_author(&m).unwrap();
        assert_eq!(author.name, "Jane Doe");
    }

    #[test]
    fn extract_author_from_object() {
        let mut m = basic_manifest();
        m.author = Some(serde_json::json!({
            "name": "Jane Doe",
            "email": "jane@example.com",
            "organization": "Acme",
        }));
        let author = extract_author(&m).unwrap();
        assert_eq!(author.name, "Jane Doe");
        assert_eq!(author.email.as_deref(), Some("jane@example.com"));
        assert_eq!(author.organization.as_deref(), Some("Acme"));
    }

    #[test]
    fn manifest_summary_compacts() {
        let mut m = basic_manifest();
        m.version = Some("1.0.0".to_string());
        m.tags = vec!["test".to_string()];
        m.allowed_tools = vec!["git".to_string()];
        let summary = manifest_summary(&m);
        assert_eq!(summary.id, "my-skill");
        assert_eq!(summary.version, "1.0.0");
        assert!(summary.tags.contains(&"test".to_string()));
        assert!(summary.tools.contains(&"git".to_string()));
        assert_eq!(summary.step_count, 0);
    }

    #[test]
    fn cycle_detection_on_acyclic_graph() {
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        graph.insert("a", vec!["b"]);
        graph.insert("b", vec!["c"]);
        graph.insert("c", vec![]);
        assert!(detect_cycle(&graph).is_none());
    }

    #[test]
    fn cycle_detection_finds_cycle() {
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        graph.insert("a", vec!["b"]);
        graph.insert("b", vec!["c"]);
        graph.insert("c", vec!["a"]);
        let cycle = detect_cycle(&graph).unwrap();
        assert!(cycle.contains(&"a"));
        assert!(cycle.contains(&"b"));
        assert!(cycle.contains(&"c"));
    }

    #[test]
    fn is_valid_slug_works() {
        assert!(is_valid_slug("my-skill"));
        assert!(is_valid_slug("git_workflow"));
        assert!(!is_valid_slug(""));
        assert!(!is_valid_slug("My Skill"));
        assert!(!is_valid_slug("-leading-dash"));
    }

    #[test]
    fn manifest_to_spec_public_produces_valid_spec() {
        let mut m = basic_manifest();
        m.version = Some("1.0.0".to_string());
        let spec =
            manifest_to_spec_public(m, SkillLayer::Managed, None, String::new(), String::new())
                .unwrap();
        assert_eq!(spec.id, "my-skill");
        assert_eq!(spec.name, "My Skill");
        assert_eq!(spec.layer, SkillLayer::Managed);
    }
}
