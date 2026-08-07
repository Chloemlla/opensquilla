//! Skill management tools: skill_list, skill_view, skill_search_community,
//! skill_install_community, install_skill_deps, skill_create, skill_edit,
//! skill_delete.
//!
//! These tools wrap `opensquilla-skills`' [`SkillLoader`] (local catalog) and
//! [`SkillHub`] (community distribution). The loader is shared (wrapped in
//! `Arc`) so a skill created by `skill_create` is visible to `skill_list` and
//! `skill_view`.
//!
//! The mutation tools (`skill_create`, `skill_edit`, `skill_delete`) operate
//! on the workspace layer only, mirroring the Python `skill_tools.py` policy.
//! Community install/deps paths that need network or subprocess execution are
//! implemented as best-effort: the schema and validation are complete, but the
//! network/subprocess piece is `TODO`-commented and returns an error `Value`
//! when the required infrastructure is unavailable.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use opensquilla_skills::SkillHub;
use opensquilla_skills::SkillLayer;
use opensquilla_skills::SkillLoader;
use opensquilla_skills::SkillSpec;
use opensquilla_skills::hub::SkillMeta;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Valid skill name pattern: lowercase alphanumeric + hyphens, 1-63 chars.
fn is_valid_skill_name(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    for c in chars {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
            return false;
        }
    }
    !name.is_empty() && name.len() <= 63
}

/// Strip characters that could inject YAML structure.
fn sanitize_yaml_value(value: &str) -> String {
    value.replace('\n', " ").replace('\r', " ").trim().to_string()
}

/// Render a SKILL.md file from parts. Frontmatter is hand-formatted with
/// quoting to round-trip safely through the loader's YAML parser.
fn render_skill_md(
    name: &str,
    description: &str,
    content: &str,
    triggers: Option<&[String]>,
) -> String {
    let safe_desc = sanitize_yaml_value(description);
    let mut fm = format!("name: {}\n", quote_yaml_scalar(name));
    fm.push_str(&format!("description: {}\n", quote_yaml_scalar(&safe_desc)));
    if let Some(trigs) = triggers {
        if !trigs.is_empty() {
            fm.push_str("triggers:\n");
            for t in trigs {
                fm.push_str(&format!("  - {}\n", quote_yaml_scalar(&sanitize_yaml_value(t))));
            }
        }
    }
    format!("---\n{}---\n\n{}", fm, content)
}

/// Quote a YAML scalar if it contains characters that could be parsed as
/// structure (`:`, `#`, `[`, `{`, leading quotes, leading `-`).
fn quote_yaml_scalar(value: &str) -> String {
    let needs_quote = value
        .chars()
        .next()
        .map(|c| matches!(c, '"' | '\'' | '-' | '?' | ':' | '[' | ']' | '{' | '}' | '#' | '&' | '*' | '!' | '|' | '>' | '%' | '@' | '`'))
        .unwrap_or(false)
        || value.contains(':')
        || value.contains(" #")
        || value.is_empty();
    if needs_quote || value.is_empty() {
        format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
    } else {
        value.to_string()
    }
}

/// Serialize a skill spec to a JSON value for tool output.
fn skill_to_json(skill: &SkillSpec) -> Value {
    serde_json::json!({
        "name": skill.name,
        "id": skill.id,
        "description": skill.description,
        "kind": skill.kind.to_string(),
        "layer": skill.layer.to_string(),
        "version": skill.version,
        "author": skill.author,
        "disabled": skill.disabled,
    })
}

/// Serialize a community search result.
fn community_result_to_dict(meta: &SkillMeta, installed: &std::collections::HashSet<String>) -> Value {
    serde_json::json!({
        "name": meta.name,
        "description": meta.description,
        "version": meta.version,
        "author": meta.author,
        "source": meta.source_id,
        "trust_level": meta.trust_level,
        "identifier": meta.identifier,
        "installed": installed.contains(&meta.name) || installed.contains(&meta.identifier),
    })
}

/// Collect the set of installed skill names from the loader.
async fn installed_skill_names(loader: &SkillLoader) -> std::collections::HashSet<String> {
    loader
        .get_skills(None)
        .await
        .into_iter()
        .map(|s| s.name.clone())
        .collect()
}

// ===========================================================================
// skill_list
// ===========================================================================

/// Tool for listing all available skills.
pub struct SkillListTool {
    loader: Arc<SkillLoader>,
}

impl SkillListTool {
    /// Create the tool from a shared skill loader.
    pub fn new(loader: SkillLoader) -> Self {
        Self {
            loader: Arc::new(loader),
        }
    }

    /// Create the tool from an existing `Arc<SkillLoader>`.
    pub fn from_arc(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

#[async_trait]
impl Tool for SkillListTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_list",
                "List all available skills with name, description, and layer.",
                HashMap::new(),
            )
            .category("skills")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, _params: Value) -> ToolResult {
        let skills = self.loader.get_skills(None).await;
        let skills: Vec<SkillSpec> = skills.into_iter().filter(|s| !s.disabled).collect();
        if skills.is_empty() {
            return Ok(ToolOutput::success("No skills installed."));
        }

        let items: Vec<Value> = skills.iter().map(skill_to_json).collect();
        let data = serde_json::json!({
            "count": items.len(),
            "skills": items,
        });

        let mut lines = vec![format!("Available skills ({}):", items.len())];
        let mut sorted = skills.clone();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        for s in &sorted {
            lines.push(format!("  - {}: {}", s.name, s.description));
        }
        Ok(ToolOutput::success(lines.join("\n")).with_data(data))
    }
}

// ===========================================================================
// skill_view
// ===========================================================================

/// Tool for viewing a skill's SKILL.md content.
pub struct SkillViewTool {
    loader: Arc<SkillLoader>,
}

impl SkillViewTool {
    /// Create the tool from a shared skill loader.
    pub fn new(loader: SkillLoader) -> Self {
        Self {
            loader: Arc::new(loader),
        }
    }

    /// Create the tool from an existing `Arc<SkillLoader>`.
    pub fn from_arc(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

#[async_trait]
impl Tool for SkillViewTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_view",
                concat!(
                    "Read a skill's SKILL.md content by name. Optionally read a supporting ",
                    "file from the skill directory.",
                ),
                HashMap::from([
                    (
                        "name".to_string(),
                        ParameterDefinition::required_string("Exact skill name to view"),
                    ),
                    (
                        "file_path".to_string(),
                        ParameterDefinition::string(
                            "Optional sub-file path (e.g. references/, scripts/)",
                        ),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?;

        // Try by name first, then by id.
        let skills = self.loader.get_skills(None).await;
        let skill = skills
            .iter()
            .find(|s| s.name == name || s.id == name)
            .cloned();

        let skill = skill.ok_or_else(|| {
            ToolError::new(
                "SKILL_NOT_FOUND",
                format!(
                    "Skill not found: {}. Use skill_list to inspect available skills.",
                    name
                ),
            )
        })?;

        let file_path = params["file_path"].as_str();
        if let Some(fp) = file_path {
            let normalized = fp.trim().trim_start_matches("./");
            if normalized.is_empty() || normalized == "SKILL.md" {
                let body = if skill.body.is_empty() {
                    format!("(Skill '{}' has no body content)", name)
                } else {
                    skill.body.clone()
                };
                return Ok(ToolOutput::success(body)
                    .with_data(serde_json::json!({ "name": name, "file": "SKILL.md" })));
            }
            // Read a supporting file from the skill's source directory.
            let base = skill.source_path.as_deref().unwrap_or("");
            if base.is_empty() {
                return Err(ToolError::new(
                    "FILE_NOT_FOUND",
                    format!("Skill '{}' has no on-disk directory; cannot read '{}'", name, fp),
                ));
            }
            let base_dir = std::path::Path::new(base)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            let target = base_dir.join(normalized);
            // Prevent path traversal outside the skill directory.
            if !target.starts_with(base_dir) {
                return Err(ToolError::new(
                    "FILE_NOT_FOUND",
                    format!("File path escapes skill directory: {}", fp),
                ));
            }
            match std::fs::read_to_string(&target) {
                Ok(content) => Ok(ToolOutput::success(content)
                    .with_data(serde_json::json!({ "name": name, "file": normalized }))),
                Err(_) => Err(ToolError::new(
                    "FILE_NOT_FOUND",
                    format!("File not found in skill '{}': {}", name, fp),
                )),
            }
        } else {
            let body = if skill.body.is_empty() {
                format!("(Skill '{}' has no body content)", name)
            } else {
                skill.body.clone()
            };
            Ok(ToolOutput::success(body)
                .with_data(serde_json::json!({ "name": name, "file": "SKILL.md" })))
        }
    }
}

// ===========================================================================
// skill_search_community
// ===========================================================================

/// Tool for searching community skill sources (e.g. ClawHub).
pub struct SkillSearchCommunityTool {
    loader: Arc<SkillLoader>,
    hub: Arc<SkillHub>,
}

impl SkillSearchCommunityTool {
    /// Create the tool from a shared loader and hub.
    pub fn new(loader: SkillLoader, hub: SkillHub) -> Self {
        Self {
            loader: Arc::new(loader),
            hub: Arc::new(hub),
        }
    }

    /// Create the tool from existing `Arc`s.
    pub fn from_arc(loader: Arc<SkillLoader>, hub: Arc<SkillHub>) -> Self {
        Self { loader, hub }
    }
}

#[async_trait]
impl Tool for SkillSearchCommunityTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_search_community",
                concat!(
                    "Search Community skill sources such as ClawHub. Use this when the user asks ",
                    "to find, search, browse, or locate installable skills from the community ",
                    "marketplace.",
                ),
                HashMap::from([
                    (
                        "query".to_string(),
                        ParameterDefinition::required_string("Search query for Community skills"),
                    ),
                    (
                        "source".to_string(),
                        ParameterDefinition::string(
                            "Source id to search, usually 'clawhub'. Use 'all' to search all sources.",
                        )
                        .default(serde_json::json!("clawhub")),
                    ),
                    (
                        "limit".to_string(),
                        ParameterDefinition::integer("Maximum number of results to return")
                            .default(serde_json::json!(10)),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(1)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let query = params["query"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'query'"))?
            .trim()
            .to_string();
        if query.is_empty() {
            return Err(ToolError::invalid_args("query must not be empty"));
        }
        let limit = params["limit"].as_i64().unwrap_or(10).clamp(1, 100) as usize;
        let source_raw = params["source"].as_str().unwrap_or("clawhub").trim();
        let source_id: Option<&str> = match source_raw {
            "" | "all" | "*" => None,
            s => Some(s),
        };

        let results = self
            .hub
            .discover(&query, source_id)
            .await
            .map_err(|e| ToolError::new("SKILL_SEARCH_FAILED", format!("Community search failed: {}", e)))?;

        let installed = installed_skill_names(&self.loader).await;
        let items: Vec<Value> = results
            .iter()
            .take(limit)
            .map(|m| community_result_to_dict(m, &installed))
            .collect();

        let data = serde_json::json!({
            "status": "ok",
            "query": query,
            "source": source_id.unwrap_or("all"),
            "count": items.len(),
            "results": items,
        });
        Ok(ToolOutput::success(serde_json::to_string_pretty(&data).unwrap_or_default())
            .with_data(data))
    }
}

// ===========================================================================
// skill_install_community
// ===========================================================================

/// Tool for installing a community skill from ClawHub or another source.
pub struct SkillInstallCommunityTool {
    hub: Arc<SkillHub>,
}

impl SkillInstallCommunityTool {
    /// Create the tool from a shared hub.
    pub fn new(hub: SkillHub) -> Self {
        Self { hub: Arc::new(hub) }
    }

    /// Create the tool from an existing `Arc<SkillHub>`.
    pub fn from_arc(hub: Arc<SkillHub>) -> Self {
        Self { hub }
    }
}

#[async_trait]
impl Tool for SkillInstallCommunityTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_install_community",
                concat!(
                    "Install a Community skill from ClawHub or another configured source. ",
                    "Use only when the user clearly asked to install a specific skill identifier ",
                    "or chose one exact result from skill_search_community.",
                ),
                HashMap::from([
                    (
                        "identifier".to_string(),
                        ParameterDefinition::required_string(
                            "Exact source identifier or slug returned by skill_search_community",
                        ),
                    ),
                    (
                        "source".to_string(),
                        ParameterDefinition::string("Source id, usually 'clawhub'")
                            .default(serde_json::json!("clawhub")),
                    ),
                    (
                        "force".to_string(),
                        ParameterDefinition::boolean(
                            "Override a dangerous security scan only after the user explicitly asks",
                        )
                        .default(serde_json::json!(false)),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let identifier = params["identifier"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'identifier'"))?
            .trim()
            .to_string();
        if identifier.is_empty() {
            return Err(ToolError::invalid_args("identifier must not be empty"));
        }
        let source = params["source"].as_str().unwrap_or("clawhub").trim();
        let source = if source.is_empty() { "clawhub" } else { source };
        let force = params["force"].as_bool().unwrap_or(false);

        let result = self
            .hub
            .install(&identifier, source, force)
            .await
            .map_err(|e| {
                ToolError::new(
                    "SKILL_INSTALL_FAILED",
                    format!("Community install failed: {}", e),
                )
            })?;

        let data = serde_json::json!({
            "status": if result.success { "installed" } else { "failed" },
            "success": result.success,
            "name": result.name,
            "identifier": identifier,
            "source": source,
            "message": result.message,
            "path": result.path,
            "scan_verdict": result.scan.as_ref().map(|s| s.verdict.clone()),
        });
        let content = if result.success {
            format!("Installed skill '{}' from {}", result.name, source)
        } else {
            format!("Install failed: {}", result.message)
        };
        Ok(ToolOutput::success(content).with_data(data))
    }
}

// ===========================================================================
// install_skill_deps
// ===========================================================================

/// Tool for previewing or installing a skill dependency declared in skill
/// metadata.
pub struct InstallSkillDepsTool {
    loader: Arc<SkillLoader>,
}

impl InstallSkillDepsTool {
    /// Create the tool from a shared loader.
    pub fn new(loader: SkillLoader) -> Self {
        Self {
            loader: Arc::new(loader),
        }
    }

    /// Create the tool from an existing `Arc<SkillLoader>`.
    pub fn from_arc(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

#[async_trait]
impl Tool for InstallSkillDepsTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "install_skill_deps",
                concat!(
                    "Preview or install a skill dependency declared in skill metadata. ",
                    "Supports brew, node, go, and uv install specs. This does not install ",
                    "Community skills; use skill_install_community for ClawHub installs.",
                ),
                HashMap::from([
                    (
                        "skill_name".to_string(),
                        ParameterDefinition::required_string(
                            "Exact skill name containing the install metadata",
                        ),
                    ),
                    (
                        "install_id".to_string(),
                        ParameterDefinition::required_string(
                            "Install spec id from the skill metadata install list",
                        ),
                    ),
                    (
                        "confirmed".to_string(),
                        ParameterDefinition::boolean(
                            "When false, return preview JSON. When true, execute the install.",
                        )
                        .default(serde_json::json!(false)),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let skill_name = params["skill_name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'skill_name'"))?;
        let install_id = params["install_id"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'install_id'"))?;
        let confirmed = params["confirmed"].as_bool().unwrap_or(false);

        let skills = self.loader.get_skills(None).await;
        let skill = skills
            .iter()
            .find(|s| s.name == skill_name || s.id == skill_name)
            .cloned()
            .ok_or_else(|| {
                ToolError::new(
                    "SKILL_NOT_FOUND",
                    format!("Skill not found: {}", skill_name),
                )
            })?;

        // The install specs live in metadata.extra["install"] as a JSON array.
        // Each spec: { id, kind, package|formula|module, label }.
        let install_specs = skill
            .metadata
            .as_ref()
            .and_then(|m| m.extra.get("install"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if install_specs.is_empty() {
            return Err(ToolError::new(
                "NO_INSTALL_METADATA",
                format!("Skill '{}' has no install metadata", skill_name),
            ));
        }

        let spec = install_specs.iter().find_map(|s| {
            let id = s.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let fallback = format!(
                "{}-{}",
                s.get("kind").and_then(|v| v.as_str()).unwrap_or(""),
                install_specs.iter().position(|x| x == s)?
            );
            if id == install_id || fallback == install_id {
                Some(s.clone())
            } else {
                None
            }
        }).ok_or_else(|| {
            ToolError::new(
                "INSTALL_SPEC_NOT_FOUND",
                format!(
                    "Install spec '{}' not found for skill '{}'",
                    install_id, skill_name
                ),
            )
        })?;

        let kind = spec.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        let package = spec
            .get("package")
            .or_else(|| spec.get("formula"))
            .or_else(|| spec.get("module"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let label = spec
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or("Install dependency");

        // Build the argv for the install spec.
        let argv = build_install_argv(kind, package)?;
        if !confirmed {
            let data = serde_json::json!({
                "status": "preview",
                "skill_name": skill_name,
                "install_id": install_id,
                "kind": kind,
                "label": label,
                "argv": argv,
            });
            return Ok(ToolOutput::success(
                serde_json::to_string_pretty(&data).unwrap_or_default(),
            )
            .with_data(data));
        }

        // TODO: Execute the install argv in a sandboxed subprocess (the Python
        // layer uses asyncio.create_subprocess_exec with a 120s timeout). The
        // tools crate's shell/exec infrastructure could be used here, but
        // wiring it requires a sandbox handle. For now return an error Value
        // so the agent can fall back to manual install.
        Err(ToolError::new(
            "INSTALL_EXEC_UNAVAILABLE",
            format!(
                "Executing skill dep installs requires a sandboxed subprocess (not yet wired). Preview argv: {:?}. Install manually or use exec_command.",
                argv
            ),
        ))
    }
}

/// Build the argv for an install spec, validating the package name.
fn build_install_argv(kind: &str, package: &str) -> Result<Vec<String>, ToolError> {
    if package.is_empty() {
        return Err(ToolError::invalid_args(format!(
            "Missing install package for kind '{}'",
            kind
        )));
    }
    if package.starts_with('-') {
        return Err(ToolError::invalid_args(format!(
            "Unsafe install value for {}: {}",
            kind, package
        )));
    }
    match kind {
        "brew" => {
            validate_install_value(package, &BREW_FORMULA_RE, "formula")?;
            Ok(vec!["brew".into(), "install".into(), package.into()])
        }
        "node" => {
            validate_install_value(package, &NODE_PACKAGE_RE, "package")?;
            Ok(vec![
                "npm".into(),
                "install".into(),
                "-g".into(),
                "--ignore-scripts".into(),
                package.into(),
            ])
        }
        "go" => {
            validate_install_value(package, &GO_MODULE_RE, "module")?;
            let module = if package.contains('@') {
                package.to_string()
            } else {
                format!("{}@latest", package)
            };
            Ok(vec!["go".into(), "install".into(), module])
        }
        "uv" => {
            validate_install_value(package, &UV_PACKAGE_RE, "package")?;
            Ok(vec!["uv".into(), "tool".into(), "install".into(), package.into()])
        }
        "download" => Err(ToolError::new(
            "INSTALL_KIND_DEFERRED",
            "Install kind 'download' is deferred and cannot be executed",
        )),
        other => Err(ToolError::invalid_args(format!(
            "Unsupported install kind: {}",
            other
        ))),
    }
}

fn validate_install_value(value: &str, pattern: &regex::Regex, label: &str) -> Result<(), ToolError> {
    if !pattern.is_match(value) {
        return Err(ToolError::invalid_args(format!(
            "Unsafe install value for {}: {}",
            label, value
        )));
    }
    Ok(())
}

static BREW_FORMULA_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"^[A-Za-z0-9][A-Za-z0-9/_@.+-]*$").unwrap());
static NODE_PACKAGE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^(?:@[A-Za-z0-9][A-Za-z0-9._-]*/)?[A-Za-z0-9][A-Za-z0-9._-]*$").unwrap()
});
static GO_MODULE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._~/-]*(?:@[A-Za-z0-9][A-Za-z0-9._~+-]*)?$").unwrap()
});
static UV_PACKAGE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?$").unwrap()
});

// ===========================================================================
// skill_create
// ===========================================================================

/// Tool for creating a new local authored skill in the workspace layer.
pub struct SkillCreateTool {
    loader: Arc<SkillLoader>,
    workspace_dir: PathBuf,
}

impl SkillCreateTool {
    /// Create the tool from a shared loader and the workspace skills directory.
    pub fn new(loader: SkillLoader, workspace_dir: PathBuf) -> Self {
        Self {
            loader: Arc::new(loader),
            workspace_dir,
        }
    }

    /// Create the tool from existing `Arc<SkillLoader>` and a workspace dir.
    pub fn from_arc(loader: Arc<SkillLoader>, workspace_dir: PathBuf) -> Self {
        Self {
            loader,
            workspace_dir,
        }
    }
}

#[async_trait]
impl Tool for SkillCreateTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_create",
                concat!(
                    "Create a new local authored skill in the workspace layer. ",
                    "Writes a SKILL.md file with frontmatter and body content. ",
                    "Do not use this for Community or ClawHub installs.",
                ),
                HashMap::from([
                    (
                        "name".to_string(),
                        ParameterDefinition::required_string(
                            "Skill name (lowercase, hyphens allowed, e.g. 'my-helper')",
                        ),
                    ),
                    (
                        "description".to_string(),
                        ParameterDefinition::required_string("One-line description of what the skill does"),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::required_string("Skill body content (markdown)"),
                    ),
                    (
                        "triggers".to_string(),
                        ParameterDefinition::array(
                            "Optional trigger phrases for auto-activation",
                            ParameterDefinition::string("trigger"),
                        ),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?;
        let description = params["description"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'description'"))?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'content'"))?;

        if !is_valid_skill_name(name) {
            return Err(ToolError::invalid_args(format!(
                "Invalid skill name: '{}'. Use lowercase letters, digits, and hyphens (e.g. 'my-helper').",
                name
            )));
        }
        if description.trim().is_empty() {
            return Err(ToolError::invalid_args("Description must not be empty"));
        }
        if content.trim().is_empty() {
            return Err(ToolError::invalid_args("Content must not be empty"));
        }

        // Check for name collision against the live catalog.
        let existing = self.loader.get_skills(None).await;
        if existing.iter().any(|s| s.name == name || s.id == name) {
            return Err(ToolError::new(
                "SKILL_EXISTS",
                format!(
                    "Skill '{}' already exists. Use skill_edit to modify it, or choose a different name.",
                    name
                ),
            ));
        }

        let triggers: Option<Vec<String>> = params["triggers"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());

        let skill_dir = self.workspace_dir.join(name);
        let skill_file = skill_dir.join("SKILL.md");
        if skill_file.exists() {
            return Err(ToolError::new(
                "SKILL_EXISTS",
                format!("Skill '{}' already exists at {}", name, skill_file.display()),
            ));
        }

        std::fs::create_dir_all(&skill_dir).map_err(|e| {
            ToolError::new(
                "SKILL_CREATE_FAILED",
                format!("Failed to create skill directory: {}", e),
            )
        })?;

        let skill_md =
            render_skill_md(name, description, content, triggers.as_deref());
        std::fs::write(&skill_file, skill_md).map_err(|e| {
            ToolError::new(
                "SKILL_CREATE_FAILED",
                format!("Failed to write SKILL.md: {}", e),
            )
        })?;

        let data = serde_json::json!({
            "name": name,
            "path": skill_file.to_string_lossy(),
            "created": true,
        });
        Ok(ToolOutput::success_with_data(
            format!("Skill '{}' created at {}", name, skill_file.display()),
            data,
        ))
    }
}

// ===========================================================================
// skill_edit
// ===========================================================================

/// Tool for editing an existing skill's content or description (workspace
/// layer only).
pub struct SkillEditTool {
    loader: Arc<SkillLoader>,
}

impl SkillEditTool {
    /// Create the tool from a shared loader.
    pub fn new(loader: SkillLoader) -> Self {
        Self {
            loader: Arc::new(loader),
        }
    }

    /// Create the tool from an existing `Arc<SkillLoader>`.
    pub fn from_arc(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

#[async_trait]
impl Tool for SkillEditTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_edit",
                concat!(
                    "Edit an existing skill's content or description. ",
                    "Only workspace-layer skills can be edited.",
                ),
                HashMap::from([
                    (
                        "name".to_string(),
                        ParameterDefinition::required_string("Exact name of the skill to edit"),
                    ),
                    (
                        "content".to_string(),
                        ParameterDefinition::string("New body content (replaces existing)"),
                    ),
                    (
                        "description".to_string(),
                        ParameterDefinition::string("New description (keeps existing if omitted)"),
                    ),
                    (
                        "triggers".to_string(),
                        ParameterDefinition::array(
                            "New trigger list (keeps existing if omitted)",
                            ParameterDefinition::string("trigger"),
                        ),
                    ),
                ]),
            )
            .category("skills")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?;

        let skills = self.loader.get_skills(None).await;
        let existing = skills
            .iter()
            .find(|s| s.name == name || s.id == name)
            .cloned()
            .ok_or_else(|| {
                ToolError::new("SKILL_NOT_FOUND", format!("Skill not found: {}", name))
            })?;

        // Only workspace-layer skills are mutable.
        if existing.layer != SkillLayer::Workspace {
            return Err(ToolError::new(
                "SKILL_NOT_MUTABLE",
                format!(
                    "Skill '{}' is in layer '{}' and cannot be edited. Only workspace-layer skills can be modified. Create a workspace override with skill_create instead.",
                    name,
                    existing.layer
                ),
            ));
        }

        let new_content = params["content"].as_str();
        let new_description = params["description"].as_str();
        let new_triggers: Option<Vec<String>> = params["triggers"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());

        if new_content.is_none() && new_description.is_none() && new_triggers.is_none() {
            return Err(ToolError::invalid_args(
                "Nothing to edit — provide content, description, or triggers",
            ));
        }

        let description = new_description.unwrap_or(&existing.description).to_string();
        let content = new_content
            .map(String::from)
            .unwrap_or_else(|| existing.body.clone());
        let triggers: Option<Vec<String>> = match new_triggers {
            Some(t) => Some(t),
            None => existing
                .metadata
                .as_ref()
                .and_then(|m| {
                    if m.triggers.is_empty() {
                        None
                    } else {
                        Some(m.triggers.clone())
                    }
                }),
        };

        let skill_file = existing
            .source_path
            .as_deref()
            .map(PathBuf::from)
            .ok_or_else(|| {
                ToolError::new(
                    "SKILL_NOT_MUTABLE",
                    format!("Skill '{}' has no on-disk file path", name),
                )
            })?;
        if !skill_file.exists() {
            return Err(ToolError::new(
                "SKILL_FILE_MISSING",
                format!("Skill file missing: {}", skill_file.display()),
            ));
        }

        let skill_md = render_skill_md(name, &description, &content, triggers.as_deref());
        std::fs::write(&skill_file, skill_md).map_err(|e| {
            ToolError::new(
                "SKILL_EDIT_FAILED",
                format!("Failed to write SKILL.md: {}", e),
            )
        })?;

        let data = serde_json::json!({
            "name": name,
            "path": skill_file.to_string_lossy(),
            "updated": true,
        });
        Ok(ToolOutput::success_with_data(
            format!("Skill '{}' updated", name),
            data,
        ))
    }
}

// ===========================================================================
// skill_delete
// ===========================================================================

/// Tool for deleting a skill from the workspace layer.
pub struct SkillDeleteTool {
    loader: Arc<SkillLoader>,
}

impl SkillDeleteTool {
    /// Create the tool from a shared loader.
    pub fn new(loader: SkillLoader) -> Self {
        Self {
            loader: Arc::new(loader),
        }
    }

    /// Create the tool from an existing `Arc<SkillLoader>`.
    pub fn from_arc(loader: Arc<SkillLoader>) -> Self {
        Self { loader }
    }
}

#[async_trait]
impl Tool for SkillDeleteTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "skill_delete",
                concat!(
                    "Delete a skill from the workspace layer. ",
                    "Cannot delete bundled or managed skills.",
                ),
                HashMap::from([(
                    "name".to_string(),
                    ParameterDefinition::required_string("Exact name of the skill to delete"),
                )]),
            )
            .category("skills")
            .risk_level(2)
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing required parameter 'name'"))?;

        let skills = self.loader.get_skills(None).await;
        let existing = skills
            .iter()
            .find(|s| s.name == name || s.id == name)
            .cloned()
            .ok_or_else(|| {
                ToolError::new("SKILL_NOT_FOUND", format!("Skill not found: {}", name))
            })?;

        if existing.layer != SkillLayer::Workspace {
            return Err(ToolError::new(
                "SKILL_NOT_MUTABLE",
                format!(
                    "Skill '{}' is in layer '{}' and cannot be deleted. Only workspace-layer skills can be removed.",
                    name,
                    existing.layer
                ),
            ));
        }

        let source_path = existing
            .source_path
            .as_deref()
            .map(PathBuf::from)
            .ok_or_else(|| {
                ToolError::new(
                    "SKILL_NOT_MUTABLE",
                    format!("Skill '{}' has no on-disk directory", name),
                )
            })?;
        // The skill directory is the parent of the SKILL.md file.
        let skill_dir = source_path.parent().ok_or_else(|| {
            ToolError::new(
                "SKILL_DELETE_FAILED",
                format!("Cannot resolve skill directory for '{}'", name),
            )
        })?;
        if !skill_dir.exists() {
            return Err(ToolError::new(
                "SKILL_DIR_MISSING",
                format!("Skill directory missing: {}", skill_dir.display()),
            ));
        }

        std::fs::remove_dir_all(skill_dir).map_err(|e| {
            ToolError::new(
                "SKILL_DELETE_FAILED",
                format!("Failed to delete skill directory: {}", e),
            )
        })?;

        let data = serde_json::json!({
            "name": name,
            "deleted": true,
        });
        Ok(ToolOutput::success_with_data(
            format!("Skill '{}' deleted from workspace layer", name),
            data,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_loader() -> Arc<SkillLoader> {
        Arc::new(SkillLoader::new())
    }

    fn test_workspace_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla-skill-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).expect("create temp workspace dir");
        dir
    }

    #[tokio::test]
    async fn test_skill_list_empty() {
        let list = SkillListTool::from_arc(test_loader());
        let result = list.execute(serde_json::json!({})).await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("No skills installed"));
    }

    #[tokio::test]
    async fn test_skill_view_not_found() {
        let view = SkillViewTool::from_arc(test_loader());
        let result = view
            .execute(serde_json::json!({ "name": "nonexistent" }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SKILL_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_skill_create_and_list_and_view() {
        let loader = test_loader();
        let workspace = test_workspace_dir();
        let create = SkillCreateTool::from_arc(loader.clone(), workspace.clone());
        let list = SkillListTool::from_arc(loader.clone());
        let view = SkillViewTool::from_arc(loader.clone());

        let result = create
            .execute(serde_json::json!({
                "name": "my-helper",
                "description": "A test helper skill",
                "content": "Do the thing.",
            }))
            .await;
        assert!(result.is_ok(), "create failed: {:?}", result.err());

        // Register the newly created skill into the loader so list/view see it.
        let skill_dir = workspace.join("my-helper");
        let (spec, _warnings) = loader
            .parse_skill_md(
                &std::fs::read_to_string(skill_dir.join("SKILL.md")).unwrap(),
                SkillLayer::Workspace,
                Some(skill_dir.join("SKILL.md")),
            )
            .expect("parse created skill");
        loader.register_skills(vec![spec]).await;

        let result = list.execute(serde_json::json!({})).await;
        assert!(result.is_ok());
        let data = result.unwrap().data.unwrap();
        assert_eq!(data["count"].as_u64(), Some(1));

        let result = view
            .execute(serde_json::json!({ "name": "my-helper" }))
            .await;
        assert!(result.is_ok());
        assert!(result.unwrap().content.contains("Do the thing."));

        std::fs::remove_dir_all(workspace).ok();
    }

    #[tokio::test]
    async fn test_skill_create_invalid_name() {
        let loader = test_loader();
        let workspace = test_workspace_dir();
        let create = SkillCreateTool::from_arc(loader, workspace.clone());
        let result = create
            .execute(serde_json::json!({
                "name": "Bad Name!",
                "description": "desc",
                "content": "content",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
        std::fs::remove_dir_all(workspace).ok();
    }

    #[tokio::test]
    async fn test_skill_edit_requires_workspace_layer() {
        let loader = test_loader();
        // Register a bundled-layer skill that cannot be edited.
        let mut spec = SkillSpec::new(
            "bundled-skill".into(),
            "Bundled Skill".into(),
            "desc".into(),
            SkillLayer::Bundled,
        );
        spec.source_path = Some("/tmp/skill/SKILL.md".into());
        loader.register_skills(vec![spec]).await;

        let edit = SkillEditTool::from_arc(loader);
        let result = edit
            .execute(serde_json::json!({
                "name": "Bundled Skill",
                "content": "new content",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SKILL_NOT_MUTABLE");
    }

    #[tokio::test]
    async fn test_skill_delete_not_found() {
        let loader = test_loader();
        let delete = SkillDeleteTool::from_arc(loader);
        let result = delete
            .execute(serde_json::json!({ "name": "ghost" }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SKILL_NOT_FOUND");
    }

    #[tokio::test]
    async fn test_install_skill_deps_no_metadata() {
        let loader = test_loader();
        let spec = SkillSpec::new(
            "no-deps".into(),
            "No Deps".into(),
            "desc".into(),
            SkillLayer::Bundled,
        );
        loader.register_skills(vec![spec]).await;

        let tool = InstallSkillDepsTool::from_arc(loader);
        let result = tool
            .execute(serde_json::json!({
                "skill_name": "No Deps",
                "install_id": "brew-0",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "NO_INSTALL_METADATA");
    }

    #[test]
    fn test_is_valid_skill_name() {
        assert!(is_valid_skill_name("my-helper"));
        assert!(is_valid_skill_name("a"));
        assert!(is_valid_skill_name("abc123"));
        assert!(!is_valid_skill_name("My-Helper"));
        assert!(!is_valid_skill_name("-leading"));
        assert!(!is_valid_skill_name("has space"));
        assert!(!is_valid_skill_name(""));
        assert!(!is_valid_skill_name("0leading-digit"));
    }

    #[test]
    fn test_render_skill_md_round_trips() {
        let md = render_skill_md(
            "my-helper",
            "A skill: with a colon",
            "Body content",
            Some(&["trigger one".into()]),
        );
        assert!(md.starts_with("---\n"));
        assert!(md.contains("name: my-helper"));
        assert!(md.contains("Body content"));
        // The colon in the description must be quoted.
        assert!(md.contains("\"A skill: with a colon\""));
    }
}
