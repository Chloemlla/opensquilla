//! # Skill domain types
//!
//! Core data types for the OpenSquilla skill system. A skill is a directory of
//! files whose entry point is a `SKILL.md` document with a YAML frontmatter
//! block. The frontmatter is deserialized into [`SkillManifest`], then
//! normalized into a runtime [`SkillSpec`] that the loader, injector,
//! eligibility checker, meta orchestrator, and hub all share.
//!
//! All enums in this module carry full serde support so they can round-trip
//! through YAML (SKILL.md frontmatter), JSON (RPC payloads), and TOML (config)
//! unchanged. Where the SKILL.md ecosystem uses string aliases that differ from
//! the canonical Rust names, serde `alias` attributes are used so real-world
//! frontmatter keeps parsing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// SkillLayer
// ---------------------------------------------------------------------------

/// The priority layer of a skill in the 6-layer coverage system.
///
/// Layers, from lowest to highest priority:
///
/// 1. `EXTRA`    — external extra directories
/// 2. `BUNDLED`  — built-in skills compiled into the binary
/// 3. `MANAGED`  — community-installed skills (`$STATE_DIR/skills/`)
/// 4. `PERSONAL` — user-installed skills (`~/.agents/skills/`)
/// 5. `PROJECT`  — workspace skills (`{workspace}/.agents/skills/`)
/// 6. `WORKSPACE`— workspace root skills (`{workspace}/skills/`)
///
/// When two layers define a skill with the same `id`, the higher-priority
/// layer wins.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "UPPERCASE")]
pub enum SkillLayer {
    /// External extra directories (lowest priority)
    #[serde(alias = "extra", alias = "extras")]
    Extra,
    /// Bundled built-in skills
    #[serde(alias = "bundled", alias = "builtin", alias = "built-in")]
    #[default]
    Bundled,
    /// Community-managed installations
    #[serde(alias = "managed")]
    Managed,
    /// Personal user installations
    #[serde(alias = "personal", alias = "user")]
    Personal,
    /// Project workspace skills
    #[serde(alias = "project")]
    Project,
    /// Workspace root skills (highest priority)
    #[serde(alias = "workspace", alias = "root")]
    Workspace,
}

impl SkillLayer {
    /// Priority order as a number (lower = higher priority on override).
    pub fn priority(&self) -> u8 {
        match self {
            SkillLayer::Extra => 0,
            SkillLayer::Bundled => 1,
            SkillLayer::Managed => 2,
            SkillLayer::Personal => 3,
            SkillLayer::Project => 4,
            SkillLayer::Workspace => 5,
        }
    }

    /// All six layers in ascending priority order.
    pub const ALL: [SkillLayer; 6] = [
        SkillLayer::Extra,
        SkillLayer::Bundled,
        SkillLayer::Managed,
        SkillLayer::Personal,
        SkillLayer::Project,
        SkillLayer::Workspace,
    ];

    /// The user-authored layers (`PERSONAL`, `PROJECT`, `WORKSPACE`). These are
    /// the layers an operator can write to directly; the others are read-only.
    pub fn is_writable(&self) -> bool {
        matches!(
            self,
            SkillLayer::Personal | SkillLayer::Project | SkillLayer::Workspace
        )
    }

    /// The system-owned layers (`EXTRA`, `BUNDLED`, `MANAGED`).
    pub fn is_system(&self) -> bool {
        !self.is_writable()
    }

    /// Parse a layer from any of the accepted string spellings.
    pub fn from_str_loose(s: &str) -> Option<SkillLayer> {
        match s.trim().to_ascii_lowercase().as_str() {
            "extra" | "extras" => Some(SkillLayer::Extra),
            "bundled" | "builtin" | "built-in" | "built_in" => Some(SkillLayer::Bundled),
            "managed" => Some(SkillLayer::Managed),
            "personal" | "user" => Some(SkillLayer::Personal),
            "project" => Some(SkillLayer::Project),
            "workspace" | "root" => Some(SkillLayer::Workspace),
            _ => None,
        }
    }
}

impl fmt::Display for SkillLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkillLayer::Extra => write!(f, "extra"),
            SkillLayer::Bundled => write!(f, "bundled"),
            SkillLayer::Managed => write!(f, "managed"),
            SkillLayer::Personal => write!(f, "personal"),
            SkillLayer::Project => write!(f, "project"),
            SkillLayer::Workspace => write!(f, "workspace"),
        }
    }
}

impl FromStr for SkillLayer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SkillLayer::from_str_loose(s).ok_or_else(|| format!("Unknown skill layer '{s}'"))
    }
}

// ---------------------------------------------------------------------------
// SkillKind
// ---------------------------------------------------------------------------

/// The kind of a skill, determining its behavior.
///
/// - [`SkillKind::Skill`] — a regular skill providing
///   instructions and tools.
/// - [`SkillKind::Meta`] / [`SkillKind::MetaSop`] — a meta-skill defining a DAG
///   workflow of steps.
/// - [`SkillKind::Tool`] — a bundled tool wrapper.
/// - [`SkillKind::Composite`] — a bundle of multiple skills.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillKind {
    /// A regular skill providing instructions and tools
    #[default]
    #[serde(alias = "basic", alias = "instruction")]
    Skill,
    /// A meta-skill defining a DAG workflow
    #[serde(alias = "workflow", alias = "dag")]
    Meta,
    /// A standard operating procedure meta-skill
    #[serde(alias = "sop", alias = "meta_sop")]
    MetaSop,
    /// A bundled tool
    Tool,
    /// A composite of multiple skills
    Composite,
}

impl SkillKind {
    /// Whether this kind denotes a meta-skill (a DAG workflow).
    pub fn is_meta(&self) -> bool {
        matches!(self, SkillKind::Meta | SkillKind::MetaSop)
    }

    /// Whether this kind is a plain instruction skill the model can invoke
    /// directly.
    pub fn is_invocable(&self) -> bool {
        !self.is_meta() && !matches!(self, SkillKind::Tool)
    }

    /// Parse a kind from a frontmatter string, tolerant of aliases.
    pub fn from_str_loose(s: &str) -> Option<SkillKind> {
        match s.trim().to_ascii_lowercase().as_str() {
            "skill" | "basic" | "instruction" | "prompt" => Some(SkillKind::Skill),
            "meta" | "workflow" | "dag" => Some(SkillKind::Meta),
            "meta_sop" | "metasop" | "sop" => Some(SkillKind::MetaSop),
            "tool" => Some(SkillKind::Tool),
            "composite" | "bundle" => Some(SkillKind::Composite),
            _ => None,
        }
    }
}

impl fmt::Display for SkillKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SkillKind::Skill => "skill",
            SkillKind::Meta => "meta",
            SkillKind::MetaSop => "meta_sop",
            SkillKind::Tool => "tool",
            SkillKind::Composite => "composite",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// SkillVisibility & SkillScope
// ---------------------------------------------------------------------------

/// Who can see and use a skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillVisibility {
    /// Visible only to the owning agent / session.
    Private,
    /// Visible to the current user across sessions.
    #[default]
    Personal,
    /// Visible to everyone on the workspace / host.
    Shared,
    /// Published to a registry.
    Public,
}

impl SkillVisibility {
    pub fn from_str_loose(s: &str) -> Option<SkillVisibility> {
        match s.trim().to_ascii_lowercase().as_str() {
            "private" | "session" => Some(SkillVisibility::Private),
            "personal" | "user" => Some(SkillVisibility::Personal),
            "shared" | "workspace" => Some(SkillVisibility::Shared),
            "public" | "published" => Some(SkillVisibility::Public),
            _ => None,
        }
    }
}

impl fmt::Display for SkillVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SkillVisibility::Private => "private",
            SkillVisibility::Personal => "personal",
            SkillVisibility::Shared => "shared",
            SkillVisibility::Public => "public",
        };
        f.write_str(s)
    }
}

/// The scope within which a skill is active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkillScope {
    /// Active for every agent run.
    #[default]
    Global,
    /// Active within a specific workspace.
    Workspace,
    /// Active within a specific project.
    Project,
    /// Active within a specific agent.
    Agent,
    /// Active within a single session only.
    Session,
}

impl SkillScope {
    pub fn from_str_loose(s: &str) -> Option<SkillScope> {
        match s.trim().to_ascii_lowercase().as_str() {
            "global" | "all" => Some(SkillScope::Global),
            "workspace" => Some(SkillScope::Workspace),
            "project" => Some(SkillScope::Project),
            "agent" => Some(SkillScope::Agent),
            "session" => Some(SkillScope::Session),
            _ => None,
        }
    }
}

impl fmt::Display for SkillScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SkillScope::Global => "global",
            SkillScope::Workspace => "workspace",
            SkillScope::Project => "project",
            SkillScope::Agent => "agent",
            SkillScope::Session => "session",
        };
        f.write_str(s)
    }
}

// ---------------------------------------------------------------------------
// SkillAuthor, SkillLicense, SkillVersion
// ---------------------------------------------------------------------------

/// Author metadata for a skill.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillAuthor {
    /// Display name.
    pub name: String,
    /// Email address, if any.
    pub email: Option<String>,
    /// URL of the author's homepage or profile.
    pub url: Option<String>,
    /// Organization the author belongs to.
    pub organization: Option<String>,
}

impl SkillAuthor {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            email: None,
            url: None,
            organization: None,
        }
    }
}

impl fmt::Display for SkillAuthor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// License identifier. Accepts SPDX short ids (`MIT`, `Apache-2.0`) and
/// free-form strings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillLicense {
    /// SPDX identifier or free-form license name.
    pub id: String,
    /// License text or URL to the license.
    pub url: Option<String>,
}

impl SkillLicense {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: None,
        }
    }
}

impl fmt::Display for SkillLicense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// A semantic version with helpers for constraint matching.
///
/// Supports `MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`. Ordering ignores build
/// metadata and treats prerelease identifiers as lower than the release.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SkillVersion {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

impl SkillVersion {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            prerelease: None,
            build: None,
        }
    }

    /// Whether this version is a stable release (no prerelease tag).
    pub fn is_stable(&self) -> bool {
        self.prerelease.is_none()
    }

    /// Match against a constraint string. Supports `==`, `!=`, `>=`, `<=`,
    /// `>`, `<`, `~=` (compatible: same major/minor, patch >=), `^` (caret),
    /// comma-separated AND lists, and bare exact versions.
    pub fn matches_constraint(&self, constraint: &str) -> bool {
        let c = constraint.trim();
        if c.is_empty() {
            return true;
        }
        for part in c.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if !self.matches_part(part) {
                return false;
            }
        }
        true
    }

    fn matches_part(&self, part: &str) -> bool {
        for (op, rest) in [
            (">=", part.strip_prefix(">=")),
            ("<=", part.strip_prefix("<=")),
            ("==", part.strip_prefix("==")),
            ("!=", part.strip_prefix("!=")),
            (">", part.strip_prefix('>')),
            ("<", part.strip_prefix('<')),
            ("~=", part.strip_prefix("~=")),
            ("^", part.strip_prefix('^')),
        ] {
            if let Some(rest) = rest {
                let rhs = rest.trim();
                let Ok(other) = Self::from_str(rhs) else {
                    return false;
                };
                return match op {
                    ">=" => *self >= other,
                    "<=" => *self <= other,
                    "==" => *self == other,
                    "!=" => *self != other,
                    ">" => *self > other,
                    "<" => *self < other,
                    "~=" => {
                        self.major == other.major
                            && self.minor == other.minor
                            && self.patch >= other.patch
                    }
                    "^" => {
                        if other.major > 0 {
                            self.major == other.major && *self >= other
                        } else if other.minor > 0 {
                            self.major == 0
                                && self.minor == other.minor
                                && self.patch >= other.patch
                        } else {
                            self.major == 0 && self.minor == 0 && self.patch >= other.patch
                        }
                    }
                    _ => false,
                };
            }
        }
        // Bare version: exact match.
        Self::from_str(part).map(|v| v == *self).unwrap_or(false)
    }
}

impl Default for SkillVersion {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 1,
            patch: 0,
            prerelease: None,
            build: None,
        }
    }
}

impl FromStr for SkillVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty version string".to_string());
        }
        let s = s.strip_prefix('v').unwrap_or(s);
        let (core, build) = match s.split_once('+') {
            Some((c, b)) => (c, Some(b.to_string())),
            None => (s, None),
        };
        let (core, prerelease) = match core.split_once('-') {
            Some((c, p)) => (c, Some(p.to_string())),
            None => (core, None),
        };
        let parts: Vec<&str> = core.split('.').collect();
        if parts.is_empty() || parts.len() > 3 {
            return Err(format!("invalid version '{s}'"));
        }
        let parse = |p: &str| -> Result<u64, String> {
            p.parse::<u64>()
                .map_err(|_| format!("invalid version '{s}'"))
        };
        let major = parse(parts[0])?;
        let minor = if parts.len() > 1 { parse(parts[1])? } else { 0 };
        let patch = if parts.len() > 2 { parse(parts[2])? } else { 0 };
        Ok(Self {
            major,
            minor,
            patch,
            prerelease,
            build,
        })
    }
}

impl fmt::Display for SkillVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(p) = &self.prerelease {
            write!(f, "-{p}")?;
        }
        if let Some(b) = &self.build {
            write!(f, "+{b}")?;
        }
        Ok(())
    }
}

impl PartialOrd for SkillVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SkillVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then_with(|| self.minor.cmp(&other.minor))
            .then_with(|| self.patch.cmp(&other.patch))
            .then_with(|| match (&self.prerelease, &other.prerelease) {
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
                (Some(a), Some(b)) => a.cmp(b),
            })
    }
}

// ---------------------------------------------------------------------------
// SkillRequires
// ---------------------------------------------------------------------------

/// Requirements for a skill to be eligible on a host.
///
/// Every field is optional; an unset field imposes no constraint. The
/// [`crate::eligibility::EligibilityChecker`] evaluates these against the
/// current host.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SkillRequires {
    /// Required operating systems (`"linux"`, `"macos"`, `"windows"`,
    /// `"unix"`, `"any"`, …). An empty list means no constraint.
    #[serde(default)]
    pub os: Option<Vec<String>>,
    /// Required binary executables, checked via `which`/`where`.
    #[serde(default)]
    pub binaries: Option<Vec<String>>,
    /// Required environment variables (set and non-empty).
    #[serde(default)]
    pub env_vars: Option<Vec<String>>,
    /// Required capabilities (`"network"`, `"filesystem"`, `"git"`, …).
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    /// Minimum skill version for a compatible tool.
    #[serde(default)]
    pub min_version: Option<String>,
    /// Required files or directories that must exist (absolute or relative to
    /// the workspace).
    #[serde(default)]
    pub files: Option<Vec<String>>,
    /// Required tool names that must be present in the tool registry.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Version constraints on tools, e.g. `{ "python": ">=3.11" }`.
    #[serde(default)]
    pub tool_versions: Option<HashMap<String, String>>,
    /// Required CPU architectures (`"x86_64"`, `"aarch64"`, `"arm64"`).
    #[serde(default)]
    pub arch: Option<Vec<String>>,
    /// Minimum free memory required, in MiB.
    #[serde(default)]
    pub min_memory_mb: Option<u64>,
    /// Whether network access is required.
    #[serde(default)]
    pub network: Option<bool>,
}

impl SkillRequires {
    /// Whether this requires set imposes no constraints at all.
    pub fn is_empty(&self) -> bool {
        self.os.as_deref().is_none_or(|v| v.is_empty())
            && self.binaries.as_deref().is_none_or(|v| v.is_empty())
            && self.env_vars.as_deref().is_none_or(|v| v.is_empty())
            && self.capabilities.as_deref().is_none_or(|v| v.is_empty())
            && self.min_version.is_none()
            && self.files.as_deref().is_none_or(|v| v.is_empty())
            && self.tools.as_deref().is_none_or(|v| v.is_empty())
            && self.tool_versions.as_ref().is_none_or(|v| v.is_empty())
            && self.arch.as_deref().is_none_or(|v| v.is_empty())
            && self.min_memory_mb.is_none()
            && self.network.is_none()
    }
}

// ---------------------------------------------------------------------------
// SkillDependency
// ---------------------------------------------------------------------------

/// A dependency of a skill on another skill, tool, or package.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SkillDependency {
    /// Depends on another skill by id.
    Skill { id: String, version: Option<String> },
    /// Depends on a tool registered in the tool registry.
    Tool {
        name: String,
        version: Option<String>,
    },
    /// Depends on an OS package.
    Package {
        name: String,
        manager: Option<String>,
    },
    /// Depends on an environment variable being set.
    Env { name: String },
}

impl SkillDependency {
    /// The dependency key used for deduplication.
    pub fn key(&self) -> String {
        match self {
            SkillDependency::Skill { id, .. } => format!("skill:{id}"),
            SkillDependency::Tool { name, .. } => format!("tool:{name}"),
            SkillDependency::Package { name, .. } => format!("package:{name}"),
            SkillDependency::Env { name } => format!("env:{name}"),
        }
    }
}

// ---------------------------------------------------------------------------
// StepType / SkillStep / StepOutput
// ---------------------------------------------------------------------------

/// The type of a meta-skill step.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepType {
    /// Run a sub-agent with a prompt
    Agent,
    /// Classify input using an LLM
    LlmClassify,
    /// Chat with an LLM
    LlmChat,
    /// Call a registered tool
    ToolCall,
    /// Execute a sub-skill
    SkillExec,
    /// Request user input
    UserInput,
}

impl StepType {
    /// Parse a step type from a string, tolerant of minor spellings.
    pub fn from_str_loose(s: &str) -> Option<StepType> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "agent" | "subagent" | "sub_agent" => Some(StepType::Agent),
            "llm_classify" | "classify" => Some(StepType::LlmClassify),
            "llm_chat" | "llm" | "chat" => Some(StepType::LlmChat),
            "tool_call" | "tool" | "call_tool" => Some(StepType::ToolCall),
            "skill_exec" | "skill" | "exec_skill" => Some(StepType::SkillExec),
            "user_input" | "input" | "ask_user" => Some(StepType::UserInput),
            _ => None,
        }
    }
}

impl fmt::Display for StepType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepType::Agent => write!(f, "agent"),
            StepType::LlmClassify => write!(f, "llm_classify"),
            StepType::LlmChat => write!(f, "llm_chat"),
            StepType::ToolCall => write!(f, "tool_call"),
            StepType::SkillExec => write!(f, "skill_exec"),
            StepType::UserInput => write!(f, "user_input"),
        }
    }
}

/// Output routing configuration for a step.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepOutput {
    /// Variable name to store the output in
    #[serde(default)]
    pub var: Option<String>,
    /// Route to a specific step on success
    #[serde(default)]
    pub route_to: Option<String>,
    /// Route to a specific step on failure
    #[serde(default)]
    pub route_on_error: Option<String>,
    /// Whether the output should be exposed to the caller as a final result.
    #[serde(default)]
    pub export: bool,
    /// Output keys to extract from a JSON result object, if any.
    #[serde(default)]
    pub pick: Option<Vec<String>>,
}

/// A skill step in a meta-skill DAG workflow.
///
/// Steps are deserialized from the `steps:` list in a meta-skill's frontmatter.
/// New fields are additive and `#[serde(default)]` so older manifests keep
/// parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillStep {
    /// Unique step ID
    pub id: String,
    /// Human-readable name
    #[serde(default)]
    pub name: String,
    /// Step type: agent, llm_classify, llm_chat, tool_call, skill_exec, user_input
    #[serde(rename = "type")]
    pub step_type: StepType,
    /// The prompt or instruction for this step
    #[serde(default)]
    pub prompt: Option<String>,
    /// The tool to call (for tool_call steps)
    #[serde(default)]
    pub tool: Option<String>,
    /// The skill to execute (for skill_exec steps)
    #[serde(default)]
    pub skill: Option<String>,
    /// Conditional expression for when this step should run
    #[serde(default)]
    pub when: Option<String>,
    /// Dependencies: step IDs that must complete before this one
    #[serde(default, alias = "depends-on", alias = "dependencies")]
    pub depends_on: Option<Vec<String>>,
    /// Maximum retries
    #[serde(default, alias = "max-retries")]
    pub max_retries: Option<u32>,
    /// Timeout in seconds
    #[serde(default, alias = "timeout-secs", alias = "timeout")]
    pub timeout_secs: Option<u64>,
    /// Output routing
    #[serde(default)]
    pub output: Option<StepOutput>,
    /// Per-step arguments (Jinja-rendered against inputs + outputs).
    #[serde(default, alias = "with-args", alias = "args")]
    pub with_args: HashMap<String, serde_json::Value>,
    /// Arguments passed verbatim to the named tool for `tool_call` steps.
    #[serde(default, alias = "tool-args")]
    pub tool_args: HashMap<String, serde_json::Value>,
    /// Closed set of valid labels for `llm_classify` steps.
    #[serde(default, alias = "output-choices", alias = "choices")]
    pub output_choices: Vec<String>,
    /// Backoff base in milliseconds between retries (doubles each attempt).
    #[serde(default, alias = "retry-backoff-ms")]
    pub retry_backoff_ms: Option<u64>,
    /// Priority hint for the scheduler (higher runs first among ready steps).
    #[serde(default)]
    pub priority: Option<i32>,
    /// Whether this step must run even if a downstream branch already failed.
    #[serde(default)]
    pub critical: bool,
    /// Free-form metadata attached to the step.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SkillStep {
    /// Create a minimal step.
    pub fn new(id: impl Into<String>, step_type: StepType) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            step_type,
            prompt: None,
            tool: None,
            skill: None,
            when: None,
            depends_on: None,
            max_retries: None,
            timeout_secs: None,
            output: None,
            with_args: HashMap::new(),
            tool_args: HashMap::new(),
            output_choices: Vec::new(),
            retry_backoff_ms: None,
            priority: None,
            critical: false,
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SkillContext, SkillArg, SkillMetadata
// ---------------------------------------------------------------------------

/// A named, reusable prompt context within a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillContext {
    /// Short name used to reference the context.
    pub name: String,
    /// Prompt body.
    #[serde(default)]
    pub description: String,
    /// Argument names this context accepts.
    #[serde(default)]
    pub arguments: Vec<String>,
}

/// An input argument declared by a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillArg {
    /// Argument name.
    pub name: String,
    /// Argument description.
    #[serde(default)]
    pub description: String,
    /// Whether the argument is required.
    #[serde(default)]
    pub required: bool,
    /// JSON-schema style type hint: `string`, `number`, `boolean`, `array`.
    #[serde(default)]
    pub type_: Option<String>,
    /// Default value.
    #[serde(default)]
    pub default: Option<serde_json::Value>,
    /// Allowed enum values.
    #[serde(default)]
    pub enum_values: Vec<String>,
}

/// Nested `metadata:` block from SKILL.md frontmatter.
///
/// All fields are optional and tolerant of unknown keys so real-world
/// manifests do not fail to parse.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillMetadata {
    /// High-level classification, e.g. `"coding"`, `"research"`.
    #[serde(default)]
    pub classification: Option<String>,
    /// Triggers that should cause the skill to be activated.
    #[serde(default)]
    pub triggers: Vec<String>,
    /// Models the skill is optimized for.
    #[serde(default)]
    pub model: Vec<String>,
    /// The mode or persona context.
    #[serde(default)]
    pub mode: Option<String>,
    /// Whether the skill should always be injected.
    #[serde(default)]
    pub always: bool,
    /// Language code (e.g. `zh`, `en`) the skill targets.
    #[serde(default)]
    pub language: Option<String>,
    /// Recommended temperature for LLM steps.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Tool names the skill may use.
    #[serde(default, alias = "allowed-tools")]
    pub allowed_tools: Vec<String>,
    /// Whether the operator must sign off on the skill's actions.
    #[serde(default, alias = "require-sign-off", alias = "require_sign_off")]
    pub require_sign_off: bool,
    /// Unknown metadata keys are preserved verbatim.
    #[serde(default, flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// SkillManifest
// ---------------------------------------------------------------------------

/// The full SKILL.md frontmatter block, mirroring the OpenClaw / Claude skill
/// manifest schema.
///
/// This is the *wire* type — what `serde_yaml` produces directly from the
/// frontmatter. The loader normalizes it into a [`SkillSpec`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique skill identifier (slug).
    #[serde(default)]
    pub id: Option<String>,
    /// Human-readable name.
    #[serde(default)]
    pub name: Option<String>,
    /// Skill kind.
    #[serde(default)]
    pub kind: Option<SkillKind>,
    /// One-line description.
    #[serde(default)]
    pub description: Option<String>,
    /// Semantic version string.
    #[serde(default)]
    pub version: Option<String>,
    /// Author name or object.
    #[serde(default)]
    pub author: Option<serde_json::Value>,
    /// License string.
    #[serde(default)]
    pub license: Option<String>,
    /// Homepage / repository URL.
    #[serde(default)]
    pub homepage: Option<String>,
    /// Layer hint (overridden by the loader based on the scanned directory).
    #[serde(default)]
    pub layer: Option<String>,
    /// Visibility.
    #[serde(default)]
    pub visibility: Option<String>,
    /// Scope.
    #[serde(default)]
    pub scope: Option<String>,
    /// Tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Requirements.
    #[serde(default)]
    pub requires: Option<SkillRequires>,
    /// Nested metadata.
    #[serde(default)]
    pub metadata: Option<SkillMetadata>,
    /// Allowed tools.
    #[serde(default, alias = "allowed-tools")]
    pub allowed_tools: Vec<String>,
    /// Whether the model may invoke this skill directly.
    #[serde(
        default,
        alias = "disable-model-invocation",
        alias = "disable_model_invocation"
    )]
    pub disable_model_invocation: bool,
    /// Meta-skill DAG steps.
    #[serde(default)]
    pub steps: Vec<SkillStep>,
    /// Meta-skill output variable mappings.
    #[serde(default)]
    pub outputs: HashMap<String, String>,
    /// Named contexts.
    #[serde(default)]
    pub contexts: Vec<SkillContext>,
    /// Declared input arguments.
    #[serde(default)]
    pub args: Vec<SkillArg>,
    /// Dependencies.
    #[serde(default)]
    pub dependencies: Vec<SkillDependency>,
    /// Changelog entries keyed by version.
    #[serde(default)]
    pub changelog: Vec<SkillChange>,
    /// Anything else in the frontmatter is preserved verbatim.
    #[serde(default, flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A changelog entry in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillChange {
    /// Version the entry applies to.
    pub version: String,
    /// Summary of what changed.
    pub description: String,
    /// When the change was made (ISO-8601).
    #[serde(default)]
    pub date: Option<String>,
}

// ---------------------------------------------------------------------------
// SkillSpec
// ---------------------------------------------------------------------------

/// Complete skill specification, the runtime form used across the crate.
///
/// Parsed from SKILL.md frontmatter by the loader, or constructed
/// programmatically (e.g. bundled skills). `body` holds the markdown content
/// that follows the frontmatter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSpec {
    /// Unique skill identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Skill kind
    #[serde(default)]
    pub kind: SkillKind,
    /// Brief description
    pub description: String,
    /// Version string
    #[serde(default)]
    pub version: Option<String>,
    /// Author
    #[serde(default)]
    pub author: Option<String>,
    /// The layer this skill belongs to
    #[serde(default = "default_layer")]
    pub layer: SkillLayer,
    /// Requirements for eligibility
    #[serde(default)]
    pub requires: SkillRequires,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// For meta-skills: the DAG workflow steps
    #[serde(default)]
    pub steps: Vec<SkillStep>,
    /// For meta-skills: the output variable mappings
    #[serde(default)]
    pub outputs: HashMap<String, String>,
    /// The raw frontmatter for reference
    #[serde(skip)]
    pub raw_frontmatter: String,
    /// The file path this skill was loaded from
    #[serde(skip)]
    pub source_path: Option<String>,
    /// The markdown body following the frontmatter.
    #[serde(skip)]
    pub body: String,
    /// Visibility.
    #[serde(default)]
    pub visibility: SkillVisibility,
    /// Scope.
    #[serde(default)]
    pub scope: SkillScope,
    /// License string.
    #[serde(default)]
    pub license: Option<String>,
    /// Homepage URL.
    #[serde(default)]
    pub homepage: Option<String>,
    /// Nested metadata block.
    #[serde(default)]
    pub metadata: Option<SkillMetadata>,
    /// Allowed tools.
    #[serde(default, alias = "allowed-tools")]
    pub allowed_tools: Vec<String>,
    /// Whether the model may invoke this skill directly.
    #[serde(default, alias = "disable-model-invocation")]
    pub disable_model_invocation: bool,
    /// Named contexts.
    #[serde(default)]
    pub contexts: Vec<SkillContext>,
    /// Declared input arguments.
    #[serde(default)]
    pub args: Vec<SkillArg>,
    /// Dependencies.
    #[serde(default)]
    pub dependencies: Vec<SkillDependency>,
    /// Whether this skill is currently disabled.
    #[serde(default)]
    pub disabled: bool,
    /// When the skill was loaded / installed (ISO-8601).
    #[serde(default)]
    pub loaded_at: Option<String>,
    /// Last scan mtime, used for cache invalidation (nanoseconds since epoch).
    #[serde(skip)]
    pub mtime_ns: Option<u64>,
}

impl SkillSpec {
    /// Create a new skill spec with minimal fields.
    pub fn new(id: String, name: String, description: String, layer: SkillLayer) -> Self {
        Self {
            id,
            name,
            kind: SkillKind::Skill,
            description,
            version: None,
            author: None,
            layer,
            requires: SkillRequires::default(),
            tags: Vec::new(),
            steps: Vec::new(),
            outputs: HashMap::new(),
            raw_frontmatter: String::new(),
            source_path: None,
            body: String::new(),
            visibility: SkillVisibility::Personal,
            scope: SkillScope::Global,
            license: None,
            homepage: None,
            metadata: None,
            allowed_tools: Vec::new(),
            disable_model_invocation: false,
            contexts: Vec::new(),
            args: Vec::new(),
            dependencies: Vec::new(),
            disabled: false,
            loaded_at: None,
            mtime_ns: None,
        }
    }

    /// Check if this skill is a meta-skill (has a DAG workflow).
    pub fn is_meta(&self) -> bool {
        self.kind.is_meta()
    }

    /// Whether the model may invoke this skill directly.
    pub fn is_model_invocable(&self) -> bool {
        !self.disable_model_invocation
    }

    /// Whether this skill is "always on" (injected regardless of relevance).
    pub fn is_always(&self) -> bool {
        self.metadata.as_ref().map(|m| m.always).unwrap_or(false)
    }

    /// Whether the skill is active in a given scope.
    pub fn in_scope(&self, scope: SkillScope) -> bool {
        matches!(self.scope, SkillScope::Global) || self.scope == scope
    }

    /// The skill's parsed version, if any.
    pub fn version_parsed(&self) -> Option<SkillVersion> {
        self.version
            .as_deref()
            .and_then(|v| SkillVersion::from_str(v).ok())
    }

    /// All tool names the skill may use: explicit `allowed_tools`, tool
    /// requirements, and dependencies of type `Tool`.
    pub fn tool_names(&self) -> Vec<String> {
        let mut out = self.allowed_tools.clone();
        out.extend(self.requires.tools.iter().flatten().cloned());
        for dep in &self.dependencies {
            if let SkillDependency::Tool { name, .. } = dep {
                out.push(name.clone());
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// True if every skill the loader needs is present: an id, a name, and
    /// either a description or a body.
    pub fn is_valid(&self) -> bool {
        !self.id.is_empty()
            && !self.name.is_empty()
            && (!self.description.is_empty() || !self.body.is_empty())
    }

    /// Validate and return a list of human-readable problems. An empty vector
    /// means the spec is valid.
    pub fn validation_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.id.is_empty() {
            issues.push("skill id is required".to_string());
        }
        if self.name.is_empty() {
            issues.push("skill name is required".to_string());
        }
        if self.description.is_empty() && self.body.is_empty() {
            issues.push("skill needs a description or a body".to_string());
        }
        if self.is_meta() && self.steps.is_empty() {
            issues.push(format!("meta-skill '{}' has no steps", self.id));
        }
        if !self.is_meta() && !self.steps.is_empty() {
            issues.push(format!(
                "non-meta skill '{}' declares steps but kind is not meta",
                self.id
            ));
        }
        issues
    }
}

// ---------------------------------------------------------------------------
// SkillMatch / SkillFilter
// ---------------------------------------------------------------------------

/// How well a skill matched a search query, used for relevance ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillMatch {
    /// The skill that matched.
    pub skill: SkillSpec,
    /// Relevance score in `[0, 1]`.
    pub score: f64,
    /// Fields of the skill that produced the match.
    pub matched_fields: Vec<String>,
    /// Query terms that matched.
    pub matched_terms: Vec<String>,
}

impl Default for SkillMatch {
    fn default() -> Self {
        Self {
            skill: SkillSpec {
                id: String::new(),
                name: String::new(),
                kind: SkillKind::default(),
                description: String::new(),
                version: None,
                author: None,
                layer: SkillLayer::default(),
                requires: SkillRequires::default(),
                tags: Vec::new(),
                steps: Vec::new(),
                outputs: HashMap::new(),
                raw_frontmatter: String::new(),
                source_path: None,
                body: String::new(),
                visibility: SkillVisibility::default(),
                scope: SkillScope::default(),
                license: None,
                homepage: None,
                metadata: None,
                allowed_tools: Vec::new(),
                disable_model_invocation: false,
                contexts: Vec::new(),
                args: Vec::new(),
                dependencies: Vec::new(),
                disabled: false,
                loaded_at: None,
                mtime_ns: None,
            },
            score: 0.0,
            matched_fields: Vec::new(),
            matched_terms: Vec::new(),
        }
    }
}

impl SkillMatch {
    pub fn new(skill: SkillSpec) -> Self {
        Self {
            skill,
            score: 0.0,
            matched_fields: Vec::new(),
            matched_terms: Vec::new(),
        }
    }
}

/// Criteria for filtering the skill catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SkillFilter {
    /// Restrict to a specific layer.
    #[serde(default)]
    pub layer: Option<SkillLayer>,
    /// Restrict to a set of layers.
    #[serde(default)]
    pub layers: Vec<SkillLayer>,
    /// Restrict to a specific kind.
    #[serde(default)]
    pub kind: Option<SkillKind>,
    /// Only include skills with every one of these tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Only include skills with any of these tags.
    #[serde(default)]
    pub any_tag: Vec<String>,
    /// Include only meta-skills (`Some(true)`) or only regular skills
    /// (`Some(false)`); `None` = no constraint.
    #[serde(default)]
    pub meta_only: Option<bool>,
    /// Include only skills requiring no host eligibility.
    #[serde(default)]
    pub eligible_only: bool,
    /// Substring to search in id, name, and description.
    #[serde(default)]
    pub query: Option<String>,
    /// Include disabled skills (default `false`).
    #[serde(default)]
    pub include_disabled: bool,
    /// Restrict to a visibility.
    #[serde(default)]
    pub visibility: Option<SkillVisibility>,
    /// Restrict to a scope.
    #[serde(default)]
    pub scope: Option<SkillScope>,
    /// Maximum number of results.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl SkillFilter {
    /// Apply this filter to a list of specs, returning matching skills.
    pub fn apply(&self, skills: &[SkillSpec]) -> Vec<SkillSpec> {
        let mut out: Vec<SkillSpec> = skills.iter().filter(|s| self.matches(s)).cloned().collect();
        if let Some(limit) = self.limit {
            out.truncate(limit);
        }
        out
    }

    /// Whether a single spec passes this filter.
    pub fn matches(&self, skill: &SkillSpec) -> bool {
        if skill.disabled && !self.include_disabled {
            return false;
        }
        if let Some(layer) = self.layer {
            if skill.layer != layer {
                return false;
            }
        }
        if !self.layers.is_empty() && !self.layers.contains(&skill.layer) {
            return false;
        }
        if let Some(kind) = &self.kind {
            if skill.kind != *kind {
                return false;
            }
        }
        if !self.tags.is_empty()
            && !self
                .tags
                .iter()
                .all(|t| skill.tags.iter().any(|st| st == t))
        {
            return false;
        }
        if !self.any_tag.is_empty()
            && !self
                .any_tag
                .iter()
                .any(|t| skill.tags.iter().any(|st| st == t))
        {
            return false;
        }
        if let Some(meta_only) = self.meta_only {
            if skill.is_meta() != meta_only {
                return false;
            }
        }
        if let Some(vis) = self.visibility {
            if skill.visibility != vis {
                return false;
            }
        }
        if let Some(scope) = self.scope {
            if skill.scope != scope {
                return false;
            }
        }
        if let Some(query) = &self.query {
            let q = query.to_lowercase();
            let hit = skill.id.to_lowercase().contains(&q)
                || skill.name.to_lowercase().contains(&q)
                || skill.description.to_lowercase().contains(&q)
                || skill.tags.iter().any(|t| t.to_lowercase().contains(&q));
            if !hit {
                return false;
            }
        }
        true
    }
}

/// Rank skills against a free-text query using lightweight term scoring.
///
/// The scorer is deterministic and pure (no LLM). Terms are matched against the
/// skill id, name, description, and tags, with exact id/name matches weighted
/// highest.
pub fn rank_skills(query: &str, skills: &[SkillSpec]) -> Vec<SkillMatch> {
    let query = query.trim().to_lowercase();
    if query.is_empty() {
        return skills
            .iter()
            .cloned()
            .map(|s| SkillMatch {
                skill: s,
                score: 0.0,
                matched_fields: Vec::new(),
                matched_terms: Vec::new(),
            })
            .collect();
    }

    let terms: Vec<&str> = query.split_whitespace().filter(|t| t.len() >= 2).collect();

    let mut results: Vec<SkillMatch> = skills
        .iter()
        .map(|s| {
            let mut m = SkillMatch::new(s.clone());
            let mut score = 0.0f64;

            // Exact id / name match is a strong signal.
            if s.id.to_lowercase() == query {
                score += 1.0;
                m.matched_fields.push("id".to_string());
                m.matched_terms.push(query.clone());
            }
            if s.name.to_lowercase() == query {
                score += 0.9;
                m.matched_fields.push("name".to_string());
                m.matched_terms.push(query.clone());
            }

            for term in &terms {
                let term = term.to_lowercase();
                if s.id.to_lowercase().contains(&term) {
                    score += 0.4;
                    m.matched_fields.push("id".to_string());
                    m.matched_terms.push(term.clone());
                }
                if s.name.to_lowercase().contains(&term) {
                    score += 0.35;
                    m.matched_fields.push("name".to_string());
                    m.matched_terms.push(term.clone());
                }
                if s.description.to_lowercase().contains(&term) {
                    score += 0.2;
                    m.matched_fields.push("description".to_string());
                    m.matched_terms.push(term.clone());
                }
                for tag in &s.tags {
                    if tag.to_lowercase().contains(&term) {
                        score += 0.15;
                        m.matched_fields.push("tag".to_string());
                        m.matched_terms.push(term.clone());
                        break;
                    }
                }
                // Trigger words in metadata also count.
                if let Some(meta) = &s.metadata {
                    if meta
                        .triggers
                        .iter()
                        .any(|t| t.to_lowercase().contains(&term))
                    {
                        score += 0.3;
                        m.matched_fields.push("trigger".to_string());
                        m.matched_terms.push(term.clone());
                    }
                }
            }

            m.score = score.min(1.0);
            m
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.skill.layer.priority().cmp(&a.skill.layer.priority()))
    });
    results
}

/// Default skill layer used when a manifest omits `layer`.
fn default_layer() -> SkillLayer {
    SkillLayer::Managed
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, layer: SkillLayer) -> SkillSpec {
        SkillSpec::new(id.to_string(), id.to_string(), "desc".to_string(), layer)
    }

    #[test]
    fn layer_priority_order() {
        assert!(SkillLayer::Extra.priority() < SkillLayer::Bundled.priority());
        assert!(SkillLayer::Bundled.priority() < SkillLayer::Managed.priority());
        assert!(SkillLayer::Managed.priority() < SkillLayer::Personal.priority());
        assert!(SkillLayer::Personal.priority() < SkillLayer::Project.priority());
        assert!(SkillLayer::Project.priority() < SkillLayer::Workspace.priority());
    }

    #[test]
    fn layer_from_str_loose() {
        assert_eq!(SkillLayer::from_str_loose("EXTRA"), Some(SkillLayer::Extra));
        assert_eq!(SkillLayer::from_str_loose("extra"), Some(SkillLayer::Extra));
        assert_eq!(
            SkillLayer::from_str_loose("BUNDLED"),
            Some(SkillLayer::Bundled)
        );
        assert_eq!(
            SkillLayer::from_str_loose("builtin"),
            Some(SkillLayer::Bundled)
        );
        assert_eq!(
            SkillLayer::from_str_loose("managed"),
            Some(SkillLayer::Managed)
        );
        assert_eq!(
            SkillLayer::from_str_loose("user"),
            Some(SkillLayer::Personal)
        );
        assert_eq!(
            SkillLayer::from_str_loose("project"),
            Some(SkillLayer::Project)
        );
        assert_eq!(
            SkillLayer::from_str_loose("workspace"),
            Some(SkillLayer::Workspace)
        );
        assert_eq!(SkillLayer::from_str_loose("nope"), None);
    }

    #[test]
    fn layer_serde_roundtrip() {
        for layer in SkillLayer::ALL {
            let json = serde_json::to_string(&layer).unwrap();
            let back: SkillLayer = serde_json::from_str(&json).unwrap();
            assert_eq!(layer, back);
        }
    }

    #[test]
    fn kind_alias_parsing() {
        assert_eq!(SkillKind::from_str_loose("meta"), Some(SkillKind::Meta));
        assert_eq!(
            SkillKind::from_str_loose("meta_sop"),
            Some(SkillKind::MetaSop)
        );
        assert_eq!(SkillKind::from_str_loose("basic"), Some(SkillKind::Skill));
        assert_eq!(SkillKind::from_str_loose("workflow"), Some(SkillKind::Meta));
        let meta: SkillKind = serde_yaml::from_str("meta").unwrap();
        assert_eq!(meta, SkillKind::Meta);
        let sop: SkillKind = serde_yaml::from_str("meta_sop").unwrap();
        assert_eq!(sop, SkillKind::MetaSop);
    }

    #[test]
    fn version_parse_and_display() {
        let v: SkillVersion = "1.2.3".parse().unwrap();
        assert_eq!(v.to_string(), "1.2.3");
        let v: SkillVersion = "v2.0.0-beta.1+build5".parse().unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.prerelease.as_deref(), Some("beta.1"));
        assert_eq!(v.build.as_deref(), Some("build5"));
    }

    #[test]
    fn version_ordering() {
        let v1: SkillVersion = "1.0.0".parse().unwrap();
        let v2: SkillVersion = "1.0.1".parse().unwrap();
        let v1beta: SkillVersion = "1.0.0-beta".parse().unwrap();
        assert!(v1 < v2);
        assert!(v1beta < v1);
        assert!(v1 > v1beta);
    }

    #[test]
    fn version_constraint_matching() {
        let v: SkillVersion = "1.4.2".parse().unwrap();
        assert!(v.matches_constraint(">=1.0.0"));
        assert!(v.matches_constraint(">=1.4, <2.0"));
        assert!(v.matches_constraint("==1.4.2"));
        assert!(!v.matches_constraint("==1.4.3"));
        assert!(v.matches_constraint("~=1.4"));
        assert!(v.matches_constraint("^1"));
        assert!(!v.matches_constraint("^2"));
        assert!(v.matches_constraint("!=1.5"));
    }

    #[test]
    fn requires_is_empty() {
        assert!(SkillRequires::default().is_empty());
        let r = SkillRequires {
            os: Some(vec!["linux".to_string()]),
            ..Default::default()
        };
        assert!(!r.is_empty());
    }

    #[test]
    fn manifest_roundtrip() {
        let yaml = r#"
name: Test Skill
description: A skill for testing
kind: meta
version: 1.2.3
tags: [testing, demo]
metadata:
  classification: coding
  always: true
steps:
  - id: step1
    name: First
    type: llm_chat
    prompt: "Hello"
  - id: step2
    name: Second
    type: tool_call
    tool: read_file
    depends_on: [step1]
"#;
        let manifest: SkillManifest = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(manifest.name.as_deref(), Some("Test Skill"));
        assert_eq!(manifest.kind, Some(SkillKind::Meta));
        assert_eq!(manifest.steps.len(), 2);
        assert_eq!(manifest.steps[1].step_type, StepType::ToolCall);
        assert!(manifest.metadata.as_ref().unwrap().always);
    }

    #[test]
    fn spec_validation_issues() {
        let s = SkillSpec::new(
            String::new(),
            "name".into(),
            "desc".into(),
            SkillLayer::Bundled,
        );
        assert!(!s.is_valid());
        assert!(!s.validation_issues().is_empty());

        let ok = SkillSpec::new(
            "id".into(),
            "name".into(),
            "desc".into(),
            SkillLayer::Bundled,
        );
        assert!(ok.is_valid());
        assert!(ok.validation_issues().is_empty());

        let mut meta = ok.clone();
        meta.kind = SkillKind::Meta;
        assert!(!meta.validation_issues().is_empty()); // no steps
        meta.steps.push(SkillStep::new("s1", StepType::Agent));
        assert!(meta.validation_issues().is_empty());
    }

    #[test]
    fn filter_matches_criteria() {
        let mut a = spec("git", SkillLayer::Bundled);
        a.tags = vec!["vcs".to_string(), "tool".to_string()];
        let mut b = spec("memory", SkillLayer::Bundled);
        b.tags = vec!["recall".to_string()];

        let filter = SkillFilter {
            tags: vec!["vcs".to_string()],
            ..SkillFilter::default()
        };
        let matches = filter.apply(&[a.clone(), b.clone()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "git");

        let query = SkillFilter {
            query: Some("mem".to_string()),
            ..SkillFilter::default()
        };
        let matches = query.apply(&[a.clone(), b.clone()]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "memory");
    }

    #[test]
    fn rank_skills_scores() {
        let mut git = spec("git", SkillLayer::Bundled);
        git.description = "interact with git repositories".to_string();
        git.tags = vec!["vcs".to_string()];
        let mut web = spec("web_search", SkillLayer::Bundled);
        web.description = "search the web".to_string();
        web.tags = vec!["search".to_string()];

        let ranked = rank_skills("git", &[git.clone(), web.clone()]);
        assert_eq!(ranked[0].skill.id, "git");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn skill_tool_names_collected() {
        let mut s = spec("x", SkillLayer::Bundled);
        s.allowed_tools = vec!["read_file".to_string()];
        s.requires.tools = Some(vec!["git".to_string()]);
        s.dependencies = vec![SkillDependency::Tool {
            name: "web_search".to_string(),
            version: None,
        }];
        let names = s.tool_names();
        assert!(names.contains(&"read_file".to_string()));
        assert!(names.contains(&"git".to_string()));
        assert!(names.contains(&"web_search".to_string()));
    }
}
