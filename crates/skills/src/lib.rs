//! # OpenSquilla Skills
//!
//! Skill system: SKILL.md frontmatter parsing, 6-layer skill registry,
//! meta-skill DAG orchestrator, eligibility checking, and hub distribution.
//!
//! A *skill* is a directory whose entry point is a `SKILL.md` document with a
//! YAML frontmatter block. Skills are data (Markdown + YAML), not Python
//! modules. This crate provides:
//!
//! - [`loader`] — discovery, parsing, caching, and hot-reload of skills across
//!   the six layers (`EXTRA`, `BUNDLED`, `MANAGED`, `PERSONAL`, `PROJECT`,
//!   `WORKSPACE`).
//! - [`types`] — the core data model ([`SkillSpec`], [`SkillManifest`],
//!   eligibility requirements, filter/match types).
//! - [`injector`] — rendering `<available_skills>` into system prompts under a
//!   token budget, with context-aware activation.
//! - [`eligibility`] — runtime host checks (OS, arch, binaries, env, files,
//!   versions, capabilities).
//! - [`meta`] — the meta-skill DAG orchestrator (6 step executors, concurrent
//!   scheduling, `when` conditions, Tera template rendering, event streaming).
//! - [`hub`] — distribution: ClawHub/GitHub/local sources, the installer,
//!   security scanner, and the lockfile.
//! - [`bundled`] — the built-in skill catalog compiled into the binary.

pub mod loader;
pub mod injector;
pub mod types;
pub mod eligibility;
pub mod meta;
pub mod hub;
pub mod manifest;
pub mod template;

/// Bundled built-in skills, defined as Rust data structures in code.
#[path = "../bundled/mod.rs"]
pub mod bundled;

// --- loader ---------------------------------------------------------------

pub use loader::{
    LoadReport, LoadWarning, LoaderConfig, SkillLoadError, SkillLoader, extract_frontmatter,
    manifest_to_spec,
};

// --- types ----------------------------------------------------------------

pub use types::{
    SkillAuthor, SkillChange, SkillContext, SkillDependency, SkillFilter, SkillKind, SkillLicense,
    SkillLayer, SkillMatch, SkillManifest, SkillMetadata, SkillRequires, SkillScope, SkillSpec,
    SkillStep, SkillVersion, SkillVisibility, StepOutput, StepType, rank_skills,
};

// --- injector -------------------------------------------------------------

pub use injector::{
    InjectorConfig, InjectionContext, SkillInjector, SkillListFormat, escape_md, escape_xml,
};

// --- eligibility ----------------------------------------------------------

pub use eligibility::{
    CheckStatus, CurrentHost, EligibilityChecker, EligibilityReport, RequirementKind,
};

// --- meta -----------------------------------------------------------------

pub use meta::{
    AgentExecutor, Dag, ExecutionContext, LlmChat, LlmChatExecutor, LlmClassifyExecutor,
    MetaEvent, MetaOrchestrator, MetaRun, SkillExecExecutor, SkillResolver, StepExecutor,
    SubAgentRunner, ToolCallExecutor, ToolInvoker, UserInputExecutor, UserInputHandler,
    coerce_to_choice, evaluate_when, is_truthy, render_args, render_template, spawn_orchestrator,
};

// --- hub ------------------------------------------------------------------

pub use hub::{
    ClawHubSource, GitHubSource, InstallOptions, InstallResult, LocalDirSource, LockEntry, LockFile,
    ScanFinding, ScanResult, ScanStrategy, SecurityScanner, SecurityWarning, Severity, SkillBundle,
    SkillHub, SkillInstaller, SkillMeta, SkillSearchIndex, SkillSource, TrustLevel,
};

// --- manifest --------------------------------------------------------------

pub use manifest::{
    extract_author, extract_layer, extract_license, extract_metadata, extract_requires,
    extract_visibility, manifest_from_json, manifest_from_yaml, manifest_summary,
    manifest_to_json, manifest_to_spec_public, manifest_to_yaml, merge_manifests, meta_manifest,
    minimal_manifest, upgrade_manifest, collect_tags, collect_tool_names, ManifestError,
    ManifestSchema, ManifestSummary, ManifestValidator, ManifestValidatorConfig, ValidationIssue,
    ValidationReport, MANIFEST_SCHEMA_VERSION,
};

// --- template --------------------------------------------------------------

pub use template::{
    builtin_templates, extract_variables, render_template_str, render_with_map,
    validate_template, BilingualPrompt, PromptTemplate, TemplateError as TemplateRenderError,
    TemplateRegistry,
};

// --- bundled --------------------------------------------------------------

pub use bundled::{
    get_bundled_skill, load_bundled_skills, bundled_skill_count, BundledSkillDef, BUNDLED_SKILLS,
};
