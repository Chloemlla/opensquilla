//! Destructive-command intent extraction for sensitive-path detection.
//!
//! Port of `src/opensquilla/sandbox/destructive_intents.py`. Pulls delete /
//! remove targets out of a shell or Python command so
//! [`crate::sensitive_paths`] can check each target against the sensitive-path
//! deny list. This is pure parsing — it holds no state and makes no approval
//! decision.
//!
//! Scope: only *delete* intents for now, mirroring the Python port.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

/// Python-flavoured delete call patterns. Each captures a single quoted path.
static PY_DELETE_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r#"\bos\.(?:remove|unlink|rmdir|removedirs)\s*\(\s*["']([^"']+)["']"#)
            .expect("valid regex"),
        Regex::new(r#"\bshutil\.rmtree\s*\(\s*["']([^"']+)["']"#).expect("valid regex"),
        Regex::new(
            r#"\b(?:pathlib\.)?Path\s*\(\s*["']([^"']+)["']\s*\)\s*\.(?:unlink|rmdir)\s*\("#,
        )
        .expect("valid regex"),
    ]
});

/// Shell command separators that terminate a single `rm` invocation.
const SHELL_SEPARATORS: &[&str] = &[";", "&&", "||", "|", "&"];

/// Best-effort absolute-path normalization.
///
/// Leaves non-path tokens alone so `*` or variable references don't get
/// expanded into something wrong.
fn norm_path(raw: &str, base_dir: Option<&str>) -> String {
    if raw.is_empty() || raw.starts_with('$') || raw.starts_with('`') || raw == "*" || raw == "-" {
        return raw.to_string();
    }
    let mut path = expand_tilde(raw);
    // `has_root` is true for `/abs` and `C:\abs`; Rust's `is_absolute` alone is
    // false for `/abs` on Windows, whereas Python pathlib treats it as rooted.
    let is_rooted = |p: &Path| p.is_absolute() || p.has_root();
    if let Some(base) = base_dir {
        let p = Path::new(&path);
        if !is_rooted(p) {
            path = Path::new(base).join(p).to_string_lossy().into_owned();
        }
    }
    if !is_rooted(Path::new(&path)) {
        if let Ok(cwd) = std::env::current_dir() {
            path = cwd.join(&path).to_string_lossy().into_owned();
        }
    }
    crate::whitelist::normalise_path(&path)
}

/// Expand a leading `~` to the current user's home directory.
pub(crate) fn expand_tilde(path: &str) -> String {
    if path == "~" {
        return home_dir();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return format!("{}/{}", home_dir().trim_end_matches(['/', '\\']), rest);
    }
    if let Some(rest) = path.strip_prefix("~\\") {
        return format!("{}\\{}", home_dir().trim_end_matches(['/', '\\']), rest);
    }
    path.to_string()
}

/// The current user's home directory, or an empty string when unavailable.
pub(crate) fn home_dir() -> String {
    for var in ["HOME", "USERPROFILE"] {
        if let Ok(v) = std::env::var(var) {
            if !v.is_empty() {
                return v;
            }
        }
    }
    if let Ok(v) = std::env::var("HOMEDRIVE") {
        if let Ok(p) = std::env::var("HOMEPATH") {
            let joined = format!("{v}{p}");
            if !joined.is_empty() {
                return joined;
            }
        }
    }
    String::new()
}

/// Extract every non-flag argument out of an `rm` invocation.
///
/// Handles `rm a b c`, `rm -rf /a /b`, quoted paths, and stops at shell
/// separators. Falls back to whitespace split on unbalanced quotes.
fn extract_rm_targets(command: &str) -> Vec<String> {
    static RM_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\brm\b([^\n]*)").expect("valid regex"));
    let Some(caps) = RM_RE.captures(command) else {
        return Vec::new();
    };
    let mut tail = caps
        .get(1)
        .map(|m| m.as_str().to_string())
        .unwrap_or_default();

    // Cut at the first shell separator so `rm foo; ls bar` doesn't pick up
    // `ls`/`bar`.
    let mut cut = tail.len();
    for sep in SHELL_SEPARATORS {
        if let Some(idx) = tail.find(sep) {
            if idx < cut {
                cut = idx;
            }
        }
    }
    tail.truncate(cut);
    let tail = tail.trim().to_string();
    if tail.is_empty() {
        return Vec::new();
    }

    let mut token_sets: Vec<Vec<String>> = Vec::new();
    match posix_split(&tail) {
        Ok(tokens) => token_sets.push(tokens),
        Err(_) => token_sets.push(whitespace_split(&tail)),
    }
    if tail.contains('\\') && has_windows_escape(&tail) {
        // Windows-style paths: quote characters are kept literal, so a plain
        // whitespace split preserves `C:\Program Files\...`.
        token_sets.push(whitespace_split(&tail));
    }

    let mut targets: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for tokens in token_sets {
        for token in tokens {
            if token.is_empty() || token.starts_with('-') || seen.contains(&token) {
                continue;
            }
            seen.insert(token.clone());
            targets.push(token);
        }
    }
    targets
}

/// True when the tail looks like a Windows path fragment: a backslash preceded
/// by start/whitespace and followed by a non-whitespace char.
fn has_windows_escape(tail: &str) -> bool {
    let chars: Vec<char> = tail.chars().collect();
    for i in 0..chars.len() {
        if chars[i] == '\\' {
            let prev_ok = i == 0 || chars[i - 1].is_whitespace();
            let next_ok = i + 1 < chars.len() && !chars[i + 1].is_whitespace();
            if prev_ok && next_ok {
                return true;
            }
        }
    }
    false
}

/// POSIX-style shell word splitting (single/double quotes, backslash escapes).
/// Returns an error on unbalanced quotes so callers can fall back.
pub(crate) fn posix_split(input: &str) -> Result<Vec<String>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_token = false;
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            '\'' => {
                in_token = true;
                i += 1;
                loop {
                    if i >= chars.len() {
                        return Err("unbalanced single quote".to_string());
                    }
                    if chars[i] == '\'' {
                        i += 1;
                        break;
                    }
                    current.push(chars[i]);
                    i += 1;
                }
            }
            '"' => {
                in_token = true;
                i += 1;
                loop {
                    if i >= chars.len() {
                        return Err("unbalanced double quote".to_string());
                    }
                    match chars[i] {
                        '"' => {
                            i += 1;
                            break;
                        }
                        '\\' if i + 1 < chars.len() => {
                            let next = chars[i + 1];
                            match next {
                                '$' | '`' | '"' | '\\' => current.push(next),
                                '\n' => { /* line continuation */ }
                                other => {
                                    current.push('\\');
                                    current.push(other);
                                }
                            }
                            i += 2;
                        }
                        c => {
                            current.push(c);
                            i += 1;
                        }
                    }
                }
            }
            '\\' => {
                in_token = true;
                if i + 1 >= chars.len() {
                    return Err("trailing backslash".to_string());
                }
                let next = chars[i + 1];
                if next == '\n' {
                    // line continuation: drop both
                } else {
                    current.push(next);
                }
                i += 2;
            }
            c if c.is_whitespace() => {
                if in_token {
                    tokens.push(std::mem::take(&mut current));
                    in_token = false;
                }
                i += 1;
            }
            c => {
                in_token = true;
                current.push(c);
                i += 1;
            }
        }
    }
    if in_token {
        tokens.push(current);
    }
    Ok(tokens)
}

/// Whitespace split fallback (quote characters kept literal).
pub(crate) fn whitespace_split(input: &str) -> Vec<String> {
    input.split_whitespace().map(|s| s.to_string()).collect()
}

/// Return every recognized destructive intent, deduped and normalized.
///
/// `rm /a /b /c` yields three `("delete", "/a")`-style intents;
/// `shutil.rmtree('a'); os.remove('b')` yields two; a plain echo yields none.
pub fn extract_intents(command: &str, base_dir: Option<&str>) -> Vec<(String, String)> {
    if command.is_empty() {
        return Vec::new();
    }
    let mut paths: Vec<String> = Vec::new();
    paths.extend(extract_rm_targets(command));
    for pattern in PY_DELETE_PATTERNS.iter() {
        for caps in pattern.captures_iter(command) {
            if let Some(m) = caps.get(1) {
                paths.push(m.as_str().to_string());
            }
        }
    }

    let mut result: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for raw in paths {
        let intent = ("delete".to_string(), norm_path(&raw, base_dir));
        if seen.contains(&intent) {
            continue;
        }
        seen.insert(intent.clone());
        result.push(intent);
    }
    result
}

/// First extracted intent, or `None`. Convenience for single-target callers.
pub fn extract_intent(command: &str) -> Option<(String, String)> {
    let intents = extract_intents(command, None);
    intents.first().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_echo_yields_nothing() {
        assert!(extract_intents("echo hello", None).is_empty());
    }

    #[test]
    fn rm_multi_targets() {
        let intents = extract_intents("rm /a /b /c", None);
        let paths: Vec<&str> = intents.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(paths, vec!["/a", "/b", "/c"]);
        assert!(intents.iter().all(|(k, _)| k == "delete"));
    }

    #[test]
    fn rm_flags_are_skipped() {
        let intents = extract_intents("rm -rf /a /b", None);
        let paths: Vec<&str> = intents.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(paths, vec!["/a", "/b"]);
    }

    #[test]
    fn rm_stops_at_shell_separator() {
        let intents = extract_intents("rm /tmp/ok; ls /etc/passwd", None);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].1, "/tmp/ok");
    }

    #[test]
    fn rm_quoted_paths() {
        let intents = extract_intents("rm '/a b' /c", None);
        let paths: Vec<&str> = intents.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(paths, vec!["/a b", "/c"]);
    }

    #[test]
    fn python_delete_patterns() {
        let command = "shutil.rmtree('venv'); os.remove('x.txt'); Path('y.txt').unlink()";
        let intents = extract_intents(command, Some("/workspace"));
        let paths: Vec<String> = intents.into_iter().map(|(_, p)| p).collect();
        assert!(paths.contains(&"/workspace/venv".to_string()));
        assert!(paths.contains(&"/workspace/x.txt".to_string()));
        assert!(paths.contains(&"/workspace/y.txt".to_string()));
    }

    #[test]
    fn relative_paths_join_base_dir() {
        let intents = extract_intents("rm build", Some("/workspace"));
        assert_eq!(intents[0].1, "/workspace/build");
    }

    #[test]
    fn variable_and_glob_tokens_left_alone() {
        let intents = extract_intents("rm $TMPDIR/* -", None);
        let paths: Vec<&str> = intents.iter().map(|(_, p)| p.as_str()).collect();
        // `-` and `$TMPDIR/*` are left alone / skipped.
        assert!(paths.iter().any(|p| p.contains("$TMPDIR")));
    }

    #[test]
    fn intents_are_deduped() {
        let intents = extract_intents("rm /a /a /b", None);
        assert_eq!(intents.len(), 2);
    }

    #[test]
    fn single_intent_convenience() {
        assert_eq!(
            extract_intent("rm /x"),
            Some(("delete".to_string(), "/x".to_string()))
        );
        assert_eq!(extract_intent("echo hi"), None);
    }
}
