//! Helpers for request-scoped workspace write deny rules.
//!
//! Mirrors the Python `opensquilla.tools.write_policy` module: matches write
//! targets against the context's `workspace_write_deny_globs` (protecting
//! specific workspace-relative paths from mutation) and detects new root-level
//! diagnostic artifacts that must be written under the scratch directory
//! instead. Both checks are bypassed under Full Host Access semantics.

use crate::context::ToolContext;
use crate::policy_config::fnmatchcase;
use crate::registry::ToolError;
use regex::Regex;
use std::path::{Path, PathBuf};

/// Env levers in the workspace-write-deny family.
const WRITE_DENY_EFFECT_ENV: &str = "OPENSQUILLA_WORKSPACE_WRITE_DENY_EFFECT";
const WRITE_DENY_EFFECT_MODES: &[&str] = &["off", "warn", "revert"];
const WRITE_DENY_TRACKED_ONLY_ENV: &str = "OPENSQUILLA_WORKSPACE_WRITE_DENY_TRACKED_ONLY";
const WRITE_DENY_SYMLINK_GUARD_ENV: &str = "OPENSQUILLA_WORKSPACE_WRITE_DENY_SYMLINK_GUARD";
const WRITE_DENY_BOOL_ENVS: &[&str] = &[
    "OPENSQUILLA_WORKSPACE_WRITE_DENY_TRACKED_ONLY",
    "OPENSQUILLA_WORKSPACE_WRITE_DENY_SYMLINK_GUARD",
    "OPENSQUILLA_WORKSPACE_WRITE_DENY_HOST_SHELL",
    "OPENSQUILLA_WORKSPACE_WRITE_DENY_COMMAND_TARGETS",
    "OPENSQUILLA_WORKSPACE_WRITE_DENY_INTERPRETER_TARGETS",
];
const WRITE_DENY_TRUE_VALUES: &[&str] = &["1", "true", "yes", "on", "enabled"];
const WRITE_DENY_FALSE_VALUES: &[&str] = &["", "0", "false", "no", "off", "disabled"];

/// Post-execution effect enforcement mode: off (default), warn, or revert.
///
/// Dispatch-time reads fail safe (unrecognized -> off); strict rejection of
/// unrecognized values happens once at bootstrap via
/// [`validate_workspace_write_deny_env`].
pub fn workspace_write_deny_effect_mode() -> &'static str {
    let raw = std::env::var(WRITE_DENY_EFFECT_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if WRITE_DENY_EFFECT_MODES.contains(&raw.as_str()) {
        return match raw.as_str() {
            "warn" => "warn",
            "revert" => "revert",
            _ => "off",
        };
    }
    "off"
}

/// Whether write-deny enforcement is scoped to git-tracked paths only.
pub fn workspace_write_deny_tracked_only() -> bool {
    env_bool_flag(WRITE_DENY_TRACKED_ONLY_ENV)
}

/// Whether the symlink guard keeps matching against the lexical path view.
pub fn workspace_write_deny_symlink_guard() -> bool {
    env_bool_flag(WRITE_DENY_SYMLINK_GUARD_ENV)
}

fn env_bool_flag(name: &str) -> bool {
    let raw = std::env::var(name)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    WRITE_DENY_TRUE_VALUES.contains(&raw.as_str())
}

/// Strictly validate the write-deny env lever family; raise on typos.
///
/// Called from engine bootstrap so an unrecognized value stops the run at
/// startup. The tools layer itself stays lenient (fail-safe to off) because
/// it can run outside the engine.
pub fn validate_workspace_write_deny_env() -> Result<(), ToolError> {
    let raw = std::env::var(WRITE_DENY_EFFECT_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !raw.is_empty() && !WRITE_DENY_EFFECT_MODES.contains(&raw.as_str()) {
        return Err(ToolError::new(
            "WORKSPACE_WRITE_DENY_ENV",
            format!(
                "{WRITE_DENY_EFFECT_ENV} must be one of {}; got {raw:?}",
                WRITE_DENY_EFFECT_MODES.join(", ")
            ),
        ));
    }
    for name in WRITE_DENY_BOOL_ENVS {
        let value = std::env::var(*name)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let valid = WRITE_DENY_TRUE_VALUES.contains(&value.as_str())
            || WRITE_DENY_FALSE_VALUES.contains(&value.as_str());
        if !valid {
            let allowed = WRITE_DENY_TRUE_VALUES
                .iter()
                .chain(WRITE_DENY_FALSE_VALUES.iter())
                .copied()
                .collect::<Vec<&str>>()
                .join(", ");
            return Err(ToolError::new(
                "WORKSPACE_WRITE_DENY_ENV",
                format!("{name} must be a boolean flag (one of {allowed}); got {value:?}"),
            ));
        }
    }
    Ok(())
}

/// A workspace write-deny match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceWriteDenyMatch {
    /// The matching glob pattern.
    pub pattern: String,
    /// The original path spelling.
    pub path: String,
    /// The resolved path.
    pub resolved_path: String,
}

/// A scratch-artifact match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceScratchArtifactMatch {
    /// The original path spelling.
    pub path: String,
    /// The resolved path.
    pub resolved_path: String,
    /// The configured scratch directory.
    pub scratch_dir: String,
}

/// Root diagnostic artifact patterns: new files with these names should be
/// written under scratch, not the workspace root.
fn root_diagnostic_artifact_regexes() -> &'static [Regex] {
    static RE: std::sync::LazyLock<Vec<Regex>> = std::sync::LazyLock::new(|| {
        vec![
            Regex::new(r"(?i)^(?:debug|repro|reproduce|scratch|verify|inspect|investigate|trace|analy[sz]e|analysis)(?:[_.-].*)?\.(?:py|js|mjs|cjs|ts|rb|php|sh|txt|md|json|ya?ml|patch|diff|zsh)$")
                .expect("valid root pattern 1"),
            Regex::new(r"(?i)^(?:check|fix|test)[_.-](?:bug|debug|failure|failing|issue|local|repro|scratch|temp|test|tmp|verify|php|py|js|ts)(?:[_.-].*)?\.(?:py|js|mjs|cjs|ts|rb|php|sh|txt|md|json|ya?ml|patch|diff|zsh)$")
                .expect("valid root pattern 2"),
        ]
    });
    &RE
}

fn workspace_root(ctx: &ToolContext) -> Option<PathBuf> {
    ctx.workspace_dir
        .clone()
        .map(|p| p.canonicalize().ok().unwrap_or(p))
}

/// Normalize a workspace-relative path spelling.
fn normalize_relative(text: &str) -> String {
    let normalized = text.replace('\\', "/");
    let mut out = normalized.as_str();
    while let Some(stripped) = out.strip_prefix("./") {
        out = stripped;
    }
    out.trim_start_matches('/').to_string()
}

/// Candidate spellings matched against deny globs.
fn candidate_strings(
    resolved: &Path,
    original_path: &str,
    workspace: Option<&Path>,
    as_directory: bool,
) -> Vec<String> {
    let mut candidates: Vec<String> = vec![
        normalize_relative(original_path),
        resolved.to_string_lossy().replace('\\', "/"),
    ];
    if let Some(workspace) = workspace {
        if let Ok(relative) = resolved.strip_prefix(workspace) {
            let relative = relative.to_string_lossy().replace('\\', "/");
            if !relative.is_empty() {
                candidates.push(relative.clone());
                candidates.push(format!("./{relative}"));
            }
        }
    }
    if as_directory {
        candidates = candidates
            .into_iter()
            .map(|candidate| format!("{}/", candidate.trim_end_matches('/')))
            .collect();
    }
    // Deduplicate preserving order.
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|c| seen.insert(c.clone()));
    candidates
}

/// Lexical (symlink-free) workspace view of a path spelling, or `None`.
fn lexical_workspace_path(original: &str, workspace: &Path) -> Option<PathBuf> {
    let raw = original.replace('\\', "/");
    let expanded = PathBuf::from(raw);
    let path = if expanded.is_absolute() {
        expanded
    } else {
        workspace.join(expanded)
    };
    // Normalize `..` lexically without resolving symlinks.
    let mut components: Vec<std::path::Component> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !matches!(components.last(), Some(std::path::Component::Normal(_))) {
                    return None;
                }
                components.pop();
            }
            other => components.push(other),
        }
    }
    let lexical: PathBuf = components.iter().collect();
    if lexical.strip_prefix(workspace).is_ok() {
        Some(lexical)
    } else {
        None
    }
}

/// Whether git tracks the path; lookup failures fail closed (tracked).
fn workspace_path_is_git_tracked(
    workspace: &Path,
    relative_path: &str,
    as_directory: bool,
) -> bool {
    let result = if as_directory {
        run_git(workspace, &["ls-files", "--", &format!("{relative_path}/")])
    } else {
        run_git(
            workspace,
            &["ls-files", "--error-unmatch", "--", relative_path],
        )
    };
    match result {
        Some(Output {
            status: code,
            stdout,
        }) => {
            if as_directory {
                code == 0 && !stdout.trim().is_empty()
            } else if code == 0 {
                true
            } else {
                // --error-unmatch exits 1 for a valid repo with no matching
                // tracked file; any other status is a lookup failure.
                code == 1
            }
        }
        None => true,
    }
}

struct Output {
    status: i32,
    stdout: String,
}

fn run_git(workspace: &Path, args: &[&str]) -> Option<Output> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .ok()?;
    Some(Output {
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
    })
}

/// Return the deny rule matching a write target, if any.
///
/// Patterns are opt-in and intentionally match both the original spelling and
/// the active-workspace-relative path when a workspace is available.
#[allow(clippy::too_many_arguments)]
pub fn match_workspace_write_deny(
    path: &Path,
    original_path: Option<&str>,
    workspace: Option<&Path>,
    ctx: &ToolContext,
    as_directory: bool,
) -> Option<WorkspaceWriteDenyMatch> {
    if ctx.full_host_access_active() {
        return None;
    }
    if ctx.workspace_write_deny_globs.is_empty() {
        return None;
    }
    let resolved = path
        .canonicalize()
        .ok()
        .unwrap_or_else(|| path.to_path_buf());
    let workspace = workspace
        .map(Path::to_path_buf)
        .or_else(|| workspace_root(ctx));
    let original = match original_path {
        Some(p) => p.to_string(),
        None => path.to_string_lossy().to_string(),
    };

    let mut resolved = resolved;
    if let Some(workspace) = &workspace {
        if resolved.strip_prefix(workspace).is_err() {
            // A workspace-internal spelling can resolve outside the workspace
            // when a path component is a symlink. The symlink guard keeps
            // matching against the lexical (non-resolved) view.
            if !workspace_write_deny_symlink_guard() {
                return None;
            }
            let lexical = lexical_workspace_path(&original, workspace)?;
            resolved = lexical;
        }
    }

    let candidates = candidate_strings(&resolved, &original, workspace.as_deref(), as_directory);
    for pattern in &ctx.workspace_write_deny_globs {
        let normalized_pattern = pattern.replace('\\', "/");
        let normalized_pattern = normalized_pattern
            .strip_prefix("./")
            .unwrap_or(&normalized_pattern)
            .to_string();
        for candidate in &candidates {
            let normalized_candidate = normalize_relative(candidate);
            if fnmatchcase(&normalized_pattern, &normalized_candidate)
                || fnmatchcase(&normalized_pattern, &format!("/{normalized_candidate}"))
            {
                if let Some(workspace) = &workspace {
                    if workspace_write_deny_tracked_only() {
                        let relative = resolved
                            .strip_prefix(workspace)
                            .map(|p| p.to_string_lossy().replace('\\', "/"))
                            .unwrap_or_default();
                        if !workspace_path_is_git_tracked(workspace, &relative, as_directory) {
                            // Tracked-only mode: deny globs protect files under
                            // version control; files the agent created itself
                            // stay writable.
                            return None;
                        }
                    }
                }
                return Some(WorkspaceWriteDenyMatch {
                    pattern: pattern.clone(),
                    path: original.clone(),
                    resolved_path: resolved.to_string_lossy().to_string(),
                });
            }
        }
    }
    None
}

/// Return a match for new root diagnostic artifacts that belong in scratch.
///
/// The check is intentionally narrow: it only applies when a scratch directory
/// is configured, only for new root-level files inside the workspace, and never
/// for paths already under the scratch directory.
#[allow(clippy::too_many_arguments)]
pub fn match_workspace_scratch_artifact(
    path: &Path,
    original_path: Option<&str>,
    workspace: Option<&Path>,
    ctx: &ToolContext,
) -> Option<WorkspaceScratchArtifactMatch> {
    if ctx.full_host_access_active() {
        return None;
    }
    let scratch_dir = ctx.scratch_dir.clone()?;
    let workspace = workspace
        .map(Path::to_path_buf)
        .or_else(|| workspace_root(ctx))?;
    let resolved = path
        .canonicalize()
        .ok()
        .unwrap_or_else(|| path.to_path_buf());
    let scratch = scratch_dir.canonicalize().ok().unwrap_or(scratch_dir);
    if resolved.strip_prefix(&scratch).is_ok() {
        return None;
    }
    let relative = resolved.strip_prefix(&workspace).ok()?;
    let relative_posix = relative.to_string_lossy().replace('\\', "/");
    if relative_posix.contains('/') || resolved.exists() {
        return None;
    }
    if !root_diagnostic_artifact_regexes()
        .iter()
        .any(|re| re.is_match(&relative_posix))
    {
        return None;
    }
    let original = match original_path {
        Some(p) => p.to_string(),
        None => path.to_string_lossy().to_string(),
    };
    Some(WorkspaceScratchArtifactMatch {
        path: original,
        resolved_path: resolved.to_string_lossy().to_string(),
        scratch_dir: scratch.to_string_lossy().to_string(),
    })
}

/// Build the blocked payload for a scratch-artifact violation.
pub fn workspace_scratch_artifact_block(
    tool_name: &str,
    matched: &WorkspaceScratchArtifactMatch,
    command: Option<&str>,
) -> serde_json::Value {
    let message = format!(
        "{tool_name} blocked creation of a temporary diagnostic artifact in \
         the workspace root: {}. Temporary reproduction, debug, verification, or \
         candidate-patch files must be written under the configured scratch \
         directory instead: {}.",
        matched.path, matched.scratch_dir
    );
    let mut payload = serde_json::json!({
        "status": "blocked",
        "reason": "workspace_scratch_artifact",
        "tool": tool_name,
        "path": matched.path,
        "resolved_path": matched.resolved_path,
        "scratch_dir": matched.scratch_dir,
        "message": message,
        "retryable": true,
    });
    if let Some(command) = command {
        payload["command"] = serde_json::json!(command);
        payload["target"] = serde_json::json!(matched.path);
    }
    payload
}

/// Raise a `ToolError` when `path` is a blocked scratch artifact.
pub fn gate_workspace_scratch_artifact(
    tool_name: &str,
    path: &Path,
    original_path: Option<&str>,
    workspace: Option<&Path>,
    ctx: &ToolContext,
) -> Result<(), ToolError> {
    if let Some(matched) = match_workspace_scratch_artifact(path, original_path, workspace, ctx) {
        let payload = workspace_scratch_artifact_block(tool_name, &matched, None);
        let message = payload["message"].as_str().unwrap_or("blocked").to_string();
        return Err(ToolError::new("WORKSPACE_SCRATCH_ARTIFACT", message));
    }
    Ok(())
}

fn deny_retry_guidance(ctx: &ToolContext) -> String {
    // Opt-in override for the remediation sentence appended to deny messages.
    if let Ok(override_value) = std::env::var("OPENSQUILLA_WORKSPACE_WRITE_DENY_GUIDANCE") {
        let override_value = override_value.trim().to_string();
        if !override_value.is_empty() {
            return format!(" {override_value}");
        }
    }
    match &ctx.scratch_dir {
        Some(scratch) => format!(
            " Temporary reproduction, debug, verification, or candidate-patch files \
             must be written under the configured scratch directory instead: {}.",
            scratch.to_string_lossy()
        ),
        None => String::new(),
    }
}

/// Build the blocked payload for a workspace write-deny violation.
pub fn workspace_write_deny_block(
    tool_name: &str,
    matched: &WorkspaceWriteDenyMatch,
    command: Option<&str>,
    ctx: &ToolContext,
) -> serde_json::Value {
    let mut guidance = deny_retry_guidance(ctx);
    if let Some(mirror) = verify_mirror_path(&matched.path, &matched.resolved_path, ctx) {
        if ctx.scratch_verify_mirror_active {
            guidance.push_str(&format!(
                " To exercise this file's checks without modifying it, copy it to the \
                 writable mirror {mirror} first, keep the mirror copy identical to the \
                 workspace original, and add any new checks as separate files under the \
                 same verify-mirror directory."
            ));
        }
    }
    let message = format!(
        "{tool_name} blocked by workspace write deny policy: {} matches {}.{}",
        matched.path, matched.pattern, guidance
    );
    let mut payload = serde_json::json!({
        "status": "blocked",
        "reason": "workspace_write_deny",
        "tool": tool_name,
        "path": matched.path,
        "resolved_path": matched.resolved_path,
        "matched_pattern": matched.pattern,
        "message": message,
        "retryable": false,
    });
    if let Some(command) = command {
        payload["command"] = serde_json::json!(command);
        payload["target"] = serde_json::json!(matched.path);
    }
    payload
}

/// Raise a `ToolError` when `path` matches a workspace write-deny rule.
pub fn gate_workspace_write_deny(
    tool_name: &str,
    path: &Path,
    original_path: Option<&str>,
    workspace: Option<&Path>,
    ctx: &ToolContext,
) -> Result<(), ToolError> {
    if let Some(matched) = match_workspace_write_deny(path, original_path, workspace, ctx, false) {
        let payload = workspace_write_deny_block(tool_name, &matched, None, ctx);
        let message = payload["message"].as_str().unwrap_or("blocked").to_string();
        return Err(ToolError::new("WORKSPACE_WRITE_DENY", message));
    }
    Ok(())
}

/// Writable mirror path for a deny-blocked workspace file, or `None`.
///
/// Mirrors live under `<scratch_dir>/verify-mirror/<workspace-relative-path>`.
pub fn verify_mirror_path(
    match_path: &str,
    resolved_path: &str,
    ctx: &ToolContext,
) -> Option<String> {
    let _ = match_path;
    let scratch_dir = ctx.scratch_dir.clone()?;
    let workspace = workspace_root(ctx)?;
    let resolved = Path::new(resolved_path)
        .canonicalize()
        .ok()
        .unwrap_or_else(|| Path::new(resolved_path).to_path_buf());
    let relative = resolved.strip_prefix(&workspace).ok()?;
    let relative_posix = relative.to_string_lossy().replace('\\', "/");
    if relative_posix.is_empty() {
        return None;
    }
    let scratch = scratch_dir.canonicalize().ok().unwrap_or(scratch_dir);
    Some(
        scratch
            .join("verify-mirror")
            .join(relative)
            .to_string_lossy()
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(globs: &[&str]) -> ToolContext {
        ToolContext {
            workspace_dir: Some(PathBuf::from("/workspace")),
            workspace_write_deny_globs: globs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn matches_deny_glob_via_relative_path() {
        let ctx = ctx_with(&["**/*.lock"]);
        let matched = match_workspace_write_deny(
            Path::new("/workspace/Cargo.lock"),
            Some("Cargo.lock"),
            None,
            &ctx,
            false,
        );
        assert!(matched.is_some());
        assert_eq!(matched.unwrap().pattern, "**/*.lock");
    }

    #[test]
    fn non_matching_path_passes() {
        let ctx = ctx_with(&["**/*.lock"]);
        let matched = match_workspace_write_deny(
            Path::new("/workspace/src/main.rs"),
            Some("src/main.rs"),
            None,
            &ctx,
            false,
        );
        assert!(matched.is_none());
    }

    #[test]
    fn no_globs_means_no_match() {
        let ctx = ctx_with(&[]);
        let matched = match_workspace_write_deny(
            Path::new("/workspace/Cargo.lock"),
            Some("Cargo.lock"),
            None,
            &ctx,
            false,
        );
        assert!(matched.is_none());
    }

    #[test]
    fn full_host_access_bypasses_deny() {
        let mut ctx = ctx_with(&["**/*.lock"]);
        ctx.run_mode = Some(crate::context::RunMode::Full);
        let matched = match_workspace_write_deny(
            Path::new("/workspace/Cargo.lock"),
            Some("Cargo.lock"),
            None,
            &ctx,
            false,
        );
        assert!(matched.is_none());
    }

    #[test]
    fn directory_operand_matches_with_slash() {
        let ctx = ctx_with(&["dist/**"]);
        let matched = match_workspace_write_deny(
            Path::new("/workspace/dist"),
            Some("dist"),
            None,
            &ctx,
            true,
        );
        assert!(matched.is_some());
    }

    #[test]
    fn scratch_artifact_detection() {
        let ctx = ToolContext {
            workspace_dir: Some(PathBuf::from("/workspace")),
            scratch_dir: Some(PathBuf::from("/scratch")),
            ..Default::default()
        };
        // A new root-level debug script is flagged.
        let matched = match_workspace_scratch_artifact(
            Path::new("/workspace/debug_issue.py"),
            Some("debug_issue.py"),
            None,
            &ctx,
        );
        assert!(matched.is_some());

        // Existing files are not flagged.
        let temp = tempfile::NamedTempFile::new().expect("temp");
        let existing =
            match_workspace_scratch_artifact(temp.path(), Some("debug_x.py"), None, &ctx);
        assert!(existing.is_none());

        // Subdirectory files are not flagged.
        let nested = match_workspace_scratch_artifact(
            Path::new("/workspace/sub/debug_x.py"),
            Some("sub/debug_x.py"),
            None,
            &ctx,
        );
        assert!(nested.is_none());

        // Files under scratch are not flagged.
        let in_scratch = match_workspace_scratch_artifact(
            Path::new("/scratch/debug_x.py"),
            Some("debug_x.py"),
            None,
            &ctx,
        );
        assert!(in_scratch.is_none());
    }

    #[test]
    fn scratch_artifact_block_payload() {
        let matched = WorkspaceScratchArtifactMatch {
            path: "debug.py".to_string(),
            resolved_path: "/workspace/debug.py".to_string(),
            scratch_dir: "/scratch".to_string(),
        };
        let payload =
            workspace_scratch_artifact_block("exec_command", &matched, Some("python debug.py"));
        assert_eq!(payload["status"], "blocked");
        assert_eq!(payload["reason"], "workspace_scratch_artifact");
        assert_eq!(payload["retryable"], true);
        assert_eq!(payload["command"], "python debug.py");
        assert!(payload["message"].as_str().unwrap().contains("/scratch"));
    }

    #[test]
    fn write_deny_block_payload() {
        let matched = WorkspaceWriteDenyMatch {
            pattern: "**/*.lock".to_string(),
            path: "Cargo.lock".to_string(),
            resolved_path: "/workspace/Cargo.lock".to_string(),
        };
        let payload =
            workspace_write_deny_block("write_file", &matched, None, &ctx_with(&["**/*.lock"]));
        assert_eq!(payload["status"], "blocked");
        assert_eq!(payload["reason"], "workspace_write_deny");
        assert_eq!(payload["retryable"], false);
        assert_eq!(payload["matched_pattern"], "**/*.lock");
    }

    #[test]
    fn gate_functions_return_errors() {
        let ctx = ctx_with(&["**/*.lock"]);
        assert!(
            gate_workspace_write_deny(
                "write_file",
                Path::new("/workspace/Cargo.lock"),
                Some("Cargo.lock"),
                None,
                &ctx,
            )
            .is_err()
        );
        assert!(
            gate_workspace_write_deny(
                "write_file",
                Path::new("/workspace/src/main.rs"),
                Some("src/main.rs"),
                None,
                &ctx,
            )
            .is_ok()
        );
    }

    #[test]
    fn verify_mirror_path_under_scratch() {
        let ctx = ToolContext {
            workspace_dir: Some(PathBuf::from("/workspace")),
            scratch_dir: Some(PathBuf::from("/scratch")),
            ..Default::default()
        };
        let mirror = verify_mirror_path("tests/x.rs", "/workspace/tests/x.rs", &ctx);
        assert!(mirror.unwrap().contains("verify-mirror"));
    }
}
