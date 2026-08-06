//! Denylist of host paths that should never be touched without explicit
//! operator trust.
//!
//! Port of `src/opensquilla/sandbox/sensitive_paths.py`. Certain host paths
//! are classed as sensitive (SSH keys, cloud credentials, system
//! configuration) and must not fall under the ordinary "requires approval"
//! flow: users clicking *approve* under pressure have been a reliable source
//! of incidents, so these paths are hard-blocked at the tool boundary and only
//! the explicit `/elevated full` operator mode can override them.
//!
//! The list is a best-effort floor. It is not a substitute for OS-level
//! permissions. The whole layer can be no-op'd with the
//! `OPENSQUILLA_SENSITIVE_PATHS_DISABLED` environment variable (for trusted
//! single-operator environments / E2E testing).

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

use crate::destructive_intents::{expand_tilde, posix_split, whitespace_split};

/// Directory prefixes whose contents must not be read/written/deleted by the
/// agent in default mode. Strings starting with `~` expand to the current
/// user's home at check time.
const SENSITIVE_PREFIXES: &[&str] = &[
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.docker/config",
    "~/.kube",
    "~/.npmrc",
    "~/.pypirc",
    "~/.netrc",
    "~/.gnupg",
    "~/.password-store",
    "/etc/shadow",
    "/etc/gshadow",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh",
    "/boot",
    "/sys",
    "/proc",
    "/dev",
    "/root",
    "/var/log",
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/private/var/run/docker.sock",
    "/lib/systemd",
    "/usr/lib/systemd",
];

/// Linux-runtime variant used to build mount-time deny roots (no `/dev` —
/// sandboxed runtimes need device nodes inside the mount namespace).
const LINUX_RUNTIME_SENSITIVE_PREFIXES: &[&str] = &[
    "~/.ssh",
    "~/.aws",
    "~/.azure",
    "~/.config/gcloud",
    "~/.docker/config",
    "~/.kube",
    "~/.npmrc",
    "~/.pypirc",
    "~/.netrc",
    "~/.gnupg",
    "~/.password-store",
    "/etc/shadow",
    "/etc/gshadow",
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh",
    "/boot",
    "/sys",
    "/proc",
    "/root",
    "/var/log",
    "/var/run/docker.sock",
    "/run/docker.sock",
    "/lib/systemd",
    "/usr/lib/systemd",
];

/// Exact filename tails we never want mutated, regardless of parent directory.
const SENSITIVE_SUFFIXES: &[&str] = &[
    "/id_rsa",
    "/id_ed25519",
    "/id_ecdsa",
    "/id_dsa",
    "/known_hosts",
    "/authorized_keys",
    "/.env",
    "/.env.local",
    "/.env.development",
    "/.env.production",
    "/.env.test",
    "/.bash_history",
    "/.zsh_history",
    "/.mysql_history",
    "/.psql_history",
];

/// Workspace parent exceptions: a workspace nested under one of these markers
/// remains usable (the broad deny prefix is relaxed to leaf-level checks).
const WORKSPACE_PARENT_EXCEPTION_MARKERS: &[&str] = &["/root"];

const TOKEN_EDGE_CHARS: &str = " \t\r\n'\"`$(){}[]<>;,|&";

/// Matches absolute or tilde-prefixed path-like tokens in free-form text.
static ABSOLUTE_OR_TILDE_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:~)?/(?:[^\s'"`$(){}\[\]<>;,|&]+)"#).expect("valid regex")
});

/// Matches literal `.env` / `.env.local` style tokens, regardless of parent
/// directory, when surrounded by token boundaries. The leading/trailing
/// boundary chars are consumed as part of the match (the Rust regex crate does
/// not support look-around); callers read the named `path` group.
static DOTENV_LITERAL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(?:^|[\s'"`$(){}\[\]<>;,|&])(?P<path>(?:[^\s'"`$(){}\[\]<>;,|&]*/)?\.env(?:\.[A-Za-z0-9_.-]+)?)(?:$|[\s'"`$(){}\[\]<>;,|&])"#,
    )
    .expect("valid regex")
});

/// Whether the whole sensitive-path block layer is disabled via
/// `OPENSQUILLA_SENSITIVE_PATHS_DISABLED`.
fn sensitive_paths_disabled() -> bool {
    std::env::var("OPENSQUILLA_SENSITIVE_PATHS_DISABLED")
        .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Runtime platform detection (mirrors Python's `os.name == "nt"`).
fn is_windows_runtime() -> bool {
    std::env::consts::OS == "windows"
}

/// Expand `~` and resolve to absolute without requiring existence.
fn expand(path: &str) -> String {
    let expanded = expand_tilde(path);
    let p = Path::new(&expanded);
    // `has_root` treats `/abs` as rooted (Rust's `is_absolute` is false for it
    // on Windows, but Python pathlib considers it absolute).
    if !p.is_absolute() && !p.has_root() {
        if let Ok(cwd) = std::env::current_dir() {
            return crate::whitelist::normalise_path(&cwd.join(p).to_string_lossy());
        }
    }
    crate::whitelist::normalise_path(&expanded)
}

/// The comparison form of a path: expanded, backslashes normalized, lowercased
/// on Windows.
fn comparison_path(path: &str) -> String {
    let normalized = expand(path).replace('\\', "/");
    if is_windows_runtime() {
        normalized.to_lowercase()
    } else {
        normalized
    }
}

/// Every comparison candidate for a path (expanded, raw, tilde-expanded home).
fn comparison_path_candidates(path: &str) -> Vec<String> {
    let mut out = vec![comparison_path(path)];
    let raw = path.trim().replace('\\', "/");
    if !raw.is_empty() {
        out.push(if is_windows_runtime() {
            raw.to_lowercase()
        } else {
            raw.clone()
        });
    }
    if raw.starts_with("~/") {
        let home = crate::destructive_intents::home_dir().replace('\\', "/");
        let expanded_home = format!("{}/{}", home.trim_end_matches('/'), &raw[2..]);
        out.push(if is_windows_runtime() {
            expanded_home.to_lowercase()
        } else {
            expanded_home
        });
    }
    // Deduplicate preserving order.
    let mut seen = std::collections::HashSet::new();
    out.into_iter().filter(|c| seen.insert(c.clone())).collect()
}

/// True when `text` looks like a rooted path (`/...` or `~/...`, not `//`).
fn looks_like_rooted_path_text(path: &str) -> bool {
    let normalized = path.trim().replace('\\', "/");
    (normalized.starts_with('/') || normalized.starts_with("~/")) && !normalized.starts_with("//")
}

/// The basename of a path (POSIX rules, lowercase, trailing slash stripped).
fn path_name(path: &str) -> String {
    let normalized = path.trim().replace('\\', "/");
    let normalized = normalized.trim_end_matches('/');
    let name = normalized.rsplit('/').next().unwrap_or("");
    name.to_lowercase()
}

/// True when `path` equals `root` or is strictly under it.
fn path_contains(path: &str, root: &str) -> bool {
    if path.is_empty() || root.is_empty() {
        return false;
    }
    let normalized_path = path.trim_end_matches('/');
    let normalized_root = root.trim_end_matches('/');
    normalized_path == normalized_root || normalized_path.starts_with(&format!("{normalized_root}/"))
}

/// Return the matched sensitive marker for an absolute/tilde path, or `None`.
pub fn is_sensitive_path(path: &str) -> Option<String> {
    if sensitive_paths_disabled() {
        return None;
    }
    if path.is_empty() {
        return None;
    }
    let candidates = comparison_path_candidates(path);

    // `/root/.ssh` is matched even when the enclosing `/root` deny prefix was
    // relaxed for a workspace carve-out.
    for expanded in &candidates {
        if expanded == "/root/.ssh"
            || expanded.starts_with("/root/.ssh/")
            || expanded.ends_with("/root/.ssh")
            || expanded.contains("/root/.ssh/")
        {
            return Some("~/.ssh".to_string());
        }
    }

    for prefix in SENSITIVE_PREFIXES {
        for expanded in &candidates {
            for normalized in comparison_path_candidates(prefix) {
                if expanded == &normalized || expanded.starts_with(&format!("{normalized}/")) {
                    return Some((*prefix).to_string());
                }
            }
        }
    }

    for suffix in SENSITIVE_SUFFIXES {
        let normalized_suffix = if is_windows_runtime() {
            suffix.to_lowercase()
        } else {
            (*suffix).to_string()
        };
        if candidates.iter().any(|c| c.ends_with(&normalized_suffix)) {
            return Some((*suffix).to_string());
        }
    }

    let name = path_name(path);
    if name == ".env" || name.starts_with(".env.") {
        return Some("/.env*".to_string());
    }
    None
}

/// True when `path` resolves inside `workspace`.
fn workspace_contains(path: &str, workspace: Option<&str>) -> bool {
    let Some(workspace) = workspace else {
        return false;
    };
    let candidate = PathBuf::from(expand(path));
    let root = PathBuf::from(expand(workspace));
    if candidate.starts_with(&root) {
        return true;
    }
    let candidate_paths = comparison_path_candidates(path);
    let workspace_paths = comparison_path_candidates(workspace);
    candidate_paths
        .iter()
        .any(|c| workspace_paths.iter().any(|r| path_contains(c, r)))
}

/// True when `workspace` is strictly nested under `marker` (only consulted for
/// markers in [`WORKSPACE_PARENT_EXCEPTION_MARKERS`]).
fn workspace_nested_under_marker(workspace: Option<&str>, marker: &str) -> bool {
    let Some(workspace) = workspace else {
        return false;
    };
    if !WORKSPACE_PARENT_EXCEPTION_MARKERS.contains(&marker) {
        return false;
    }
    let root = PathBuf::from(expand(workspace));
    let marker_root = PathBuf::from(expand(marker));
    if root == marker_root {
        return false;
    }
    if root.starts_with(&marker_root) {
        return true;
    }
    for workspace_text in comparison_path_candidates(workspace) {
        for marker_text in comparison_path_candidates(marker) {
            if workspace_text != marker_text && path_contains(&workspace_text, &marker_text) {
                return true;
            }
        }
    }
    false
}

/// Leaf-level sensitive marker (suffix or `.env`) without the broad prefixes.
fn sensitive_leaf_marker(path: &str) -> Option<String> {
    let candidates = comparison_path_candidates(path);
    for suffix in SENSITIVE_SUFFIXES {
        let normalized_suffix = if is_windows_runtime() {
            suffix.to_lowercase()
        } else {
            (*suffix).to_string()
        };
        if candidates.iter().any(|c| c.ends_with(&normalized_suffix)) {
            return Some((*suffix).to_string());
        }
    }
    let name = path_name(path);
    if name == ".env" || name.starts_with(".env.") {
        return Some("/.env*".to_string());
    }
    None
}

/// Return a sensitive marker, honoring the active workspace boundary.
///
/// Container deployments commonly place OpenSquilla's default workspace under
/// `/root/.opensquilla/workspace`. The broad `/root` deny prefix should not
/// make that configured workspace unusable, but credential-like leaf files
/// such as `.env` and private-key names remain blocked.
pub fn sensitive_path_marker(path: &str, workspace: Option<&str>) -> Option<String> {
    let text = path.trim().to_string();
    let expanded = expand_tilde(&text);
    let raw = Path::new(&expanded);
    if !text.is_empty()
        && !text.starts_with('~')
        && !raw.is_absolute()
        && !looks_like_rooted_path_text(&text)
    {
        return sensitive_leaf_marker(&text);
    }

    let marker = is_sensitive_path(path)?;
    if workspace_contains(path, workspace) && workspace_nested_under_marker(workspace, &marker) {
        return sensitive_leaf_marker(path);
    }
    Some(marker)
}

/// Return the first sensitive path marker appearing in free-form text.
///
/// Intentionally conservative glue for shell/Python-code scanners; structured
/// callers should resolve concrete paths and call [`is_sensitive_path`]
/// directly.
pub fn sensitive_path_in_text(text: &str, workspace: Option<&str>) -> Option<String> {
    if sensitive_paths_disabled() {
        return None;
    }
    if text.is_empty() {
        return None;
    }

    let mut candidates: Vec<String> = Vec::new();
    match posix_split(text) {
        Ok(tokens) => candidates.extend(tokens),
        Err(_) => candidates.extend(whitespace_split(text)),
    }
    candidates.extend(whitespace_split(text));

    let mut with_context: Vec<(String, usize)> = Vec::new();
    for m in ABSOLUTE_OR_TILDE_PATH_RE.find_iter(text) {
        with_context.push((m.as_str().to_string(), m.start()));
    }
    for caps in DOTENV_LITERAL_RE.captures_iter(text) {
        if let Some(path_match) = caps.name("path") {
            with_context.push((path_match.as_str().to_string(), path_match.start()));
        }
    }

    for raw in candidates {
        if raw.contains("://") {
            continue;
        }
        let candidate = raw.trim_matches(|c| TOKEN_EDGE_CHARS.contains(c));
        if candidate.is_empty() {
            continue;
        }
        if let Some(marker) = sensitive_path_marker(candidate, workspace) {
            return Some(marker);
        }
    }

    for (raw, start) in with_context {
        let candidate = raw.trim_matches(|c| TOKEN_EDGE_CHARS.contains(c));
        if candidate.is_empty() || candidate.starts_with("//") || candidate.contains("://") {
            continue;
        }
        if start >= 3 && text.get(start - 3..start).is_some_and(|s| s == "://") {
            continue;
        }
        if let Some(marker) = sensitive_path_marker(candidate, workspace) {
            return Some(marker);
        }
    }
    None
}

/// Return the first sensitive marker for any destructive target, or `None`.
///
/// Multi-target commands (`rm /tmp/ok /etc/bad`) are each checked — the
/// presence of a single sensitive path is enough to block the whole command.
pub fn sensitive_target_in_command(
    command: &str,
    workspace: Option<&str>,
    cwd: Option<&str>,
) -> Option<String> {
    if sensitive_paths_disabled() {
        return None;
    }
    let effective_workspace: Option<String> = workspace
        .map(|s| s.to_string())
        .or_else(|| cwd.map(|s| s.to_string()))
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|d| d.to_string_lossy().into_owned())
        });
    for (_kind, target) in
        crate::destructive_intents::extract_intents(command, effective_workspace.as_deref())
    {
        if let Some(marker) = sensitive_path_marker(&target, effective_workspace.as_deref()) {
            return Some(marker);
        }
    }
    None
}

/// Shape of a hard-block result returned to the caller / model.
///
/// The model-facing `message` is intentionally terse and tells the agent not
/// to retry — `retryable: false` should be enough for a well-behaved model to
/// stop paraphrasing the same dangerous intent.
pub fn build_block_envelope(
    command: &str,
    sensitive_marker: &str,
    tool_name: Option<&str>,
) -> serde_json::Value {
    let tool = tool_name.filter(|t| !t.is_empty());
    serde_json::json!({
        "status": "blocked",
        "reason": "sensitive_path",
        "tool": tool,
        "command": command,
        "sensitive_path": sensitive_marker,
        "message": format!(
            "Refusing to operate on sensitive host path: {sensitive_marker}. This is a hard-block regardless of user approval. If this is truly intended, the operator must set /elevated full and retry."
        ),
        "retryable": false,
    })
}

/// The deny roots a Linux runtime should mount read-only/deny, derived from
/// the runtime prefix list. When the workspace is nested under `/root`, only
/// the credential-like children of `/root` are denied so the workspace stays
/// usable.
pub fn linux_runtime_sensitive_deny_roots(workspace: Option<&str>) -> Vec<PathBuf> {
    if sensitive_paths_disabled() {
        return Vec::new();
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    for raw in LINUX_RUNTIME_SENSITIVE_PREFIXES {
        let root = PathBuf::from(expand_tilde(raw));
        if let Some(ws) = workspace {
            if path_contains(&comparison_path(ws), &comparison_path(raw)) {
                roots.extend(sensitive_children_for_workspace_parent(&root));
                continue;
            }
        }
        roots.push(root);
    }
    let mut seen = std::collections::HashSet::new();
    roots.into_iter().filter(|r| seen.insert(r.clone())).collect()
}

/// For a `/root` workspace parent, the credential-like children still denied.
fn sensitive_children_for_workspace_parent(root: &Path) -> Vec<PathBuf> {
    if root.to_string_lossy().replace('\\', "/") != "/root" {
        return Vec::new();
    }
    vec![
        root.join(".ssh"),
        root.join(".aws"),
        root.join(".azure"),
        root.join(".config").join("gcloud"),
        root.join(".docker").join("config"),
        root.join(".kube"),
        root.join(".npmrc"),
        root.join(".pypirc"),
        root.join(".netrc"),
        root.join(".gnupg"),
        root.join(".password-store"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssh_keys_are_sensitive() {
        assert_eq!(is_sensitive_path("/home/u/.ssh/id_rsa"), Some("/id_rsa".to_string()));
        assert_eq!(is_sensitive_path("/home/u/.ssh/id_ed25519"), Some("/id_ed25519".to_string()));
        assert_eq!(is_sensitive_path("/root/.ssh/id_rsa"), Some("~/.ssh".to_string()));
        assert_eq!(is_sensitive_path("~/.ssh/config"), Some("~/.ssh".to_string()));
    }

    #[test]
    fn system_roots_are_sensitive() {
        assert_eq!(is_sensitive_path("/etc/shadow"), Some("/etc/shadow".to_string()));
        assert_eq!(is_sensitive_path("/etc/sudoers.d/x"), Some("/etc/sudoers.d".to_string()));
        assert_eq!(is_sensitive_path("/var/log/syslog"), Some("/var/log".to_string()));
        assert_eq!(is_sensitive_path("/proc/self/maps"), Some("/proc".to_string()));
    }

    #[test]
    fn dotenv_and_credential_files_are_sensitive_anywhere() {
        assert_eq!(is_sensitive_path("/tmp/.env"), Some("/.env".to_string()));
        assert_eq!(is_sensitive_path("/tmp/.env.production"), Some("/.env.production".to_string()));
        // A file named `.env` in any directory matches by name.
        assert_eq!(sensitive_path_marker("some_dir/.env", None), Some("/.env".to_string()));
        assert_eq!(sensitive_path_marker("some_dir/.env.local", None), Some("/.env.local".to_string()));
        // Unknown `.env.*` variants fall back to the wildcard marker.
        assert_eq!(sensitive_path_marker("some_dir/.env.staging", None), Some("/.env*".to_string()));
    }

    #[test]
    fn ordinary_paths_are_not_sensitive() {
        assert_eq!(is_sensitive_path("/home/u/docs/notes.txt"), None);
        assert_eq!(is_sensitive_path("/tmp/x"), None);
        assert_eq!(is_sensitive_path(""), None);
    }

    #[test]
    fn relative_paths_resolve_to_leaf() {
        // Relative paths return leaf markers without expanding.
        assert_eq!(sensitive_path_marker("rel/path/.env", None), Some("/.env".to_string()));
        assert_eq!(sensitive_path_marker("rel/path/id_rsa", None), Some("/id_rsa".to_string()));
    }

    #[test]
    fn workspace_under_root_keeps_leaf_checks() {
        let ws = Some("/root/.opensquilla/workspace");
        // Broad /root deny is relaxed for the workspace...
        assert_eq!(sensitive_path_marker("/root/.opensquilla/workspace/x.txt", ws), None);
        // ...but leaf credential files stay blocked.
        assert_eq!(
            sensitive_path_marker("/root/.opensquilla/workspace/.env", ws),
            Some("/.env".to_string())
        );
        assert_eq!(
            sensitive_path_marker("/root/.opensquilla/workspace/id_rsa", ws),
            Some("/id_rsa".to_string())
        );
    }

    #[test]
    fn text_scanner_finds_paths() {
        let text = "please read /home/u/.ssh/id_rsa and report";
        assert_eq!(
            sensitive_path_in_text(text, None),
            Some("/id_rsa".to_string())
        );
        let text = "cat /etc/shadow now";
        assert_eq!(
            sensitive_path_in_text(text, None),
            Some("/etc/shadow".to_string())
        );
    }

    #[test]
    fn text_scanner_ignores_urls() {
        let text = "see https://example.com/x for details";
        assert_eq!(sensitive_path_in_text(text, None), None);
    }

    #[test]
    fn command_targets_are_scanned() {
        assert_eq!(
            sensitive_target_in_command("rm /tmp/ok /etc/shadow", None, None),
            Some("/etc/shadow".to_string())
        );
        assert_eq!(
            sensitive_target_in_command("rm /tmp/ok", None, None),
            None
        );
    }

    #[test]
    fn block_envelope_shape() {
        let env = build_block_envelope("rm /etc/shadow", "/etc/shadow", Some("bash"));
        assert_eq!(env["status"], "blocked");
        assert_eq!(env["reason"], "sensitive_path");
        assert_eq!(env["retryable"], serde_json::Value::Bool(false));
        assert_eq!(env["tool"], "bash");
        assert_eq!(env["sensitive_path"], "/etc/shadow");
    }

    #[test]
    fn runtime_deny_roots_cover_credentials() {
        let roots = linux_runtime_sensitive_deny_roots(None);
        let text: Vec<String> = roots
            .iter()
            .map(|r| r.to_string_lossy().replace('\\', "/"))
            .collect();
        assert!(text.iter().any(|r| r.ends_with("/etc/shadow")));
        assert!(text.iter().any(|r| r.ends_with("/proc")));
    }
}
