//! Filesystem path whitelist/blacklist engine with glob patterns.
//!
//! This module implements a higher-level path-access engine than the simple
//! prefix-matching in [`crate::policy::FilesystemPolicy`]. It supports:
//!
//! - Glob patterns (`*`, `**`, `?`, `[...]`) for flexible path matching.
//! - Per-rule access mode (read, write, both) and an explicit deny.
//! - Rule precedence: explicit deny always wins over allow; more specific
//!   (longer) patterns win over less specific ones.
//! - Case-insensitive matching on Windows, case-sensitive on POSIX.
//! - Normalisation of `.` / `..` segments and redundant separators.
//!
//! The engine is pure: it has no I/O and no global state, so it can be unit
//! tested in isolation and composed into platform backends without surprises.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// The access mode granted by an allow rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AccessMode {
    /// Read access only.
    Read,
    /// Write access only.
    Write,
    /// Both read and write access.
    ReadWrite,
}

impl AccessMode {
    /// Does this mode grant read access?
    pub fn allows_read(self) -> bool {
        matches!(self, AccessMode::Read | AccessMode::ReadWrite)
    }

    /// Does this mode grant write access?
    pub fn allows_write(self) -> bool {
        matches!(self, AccessMode::Write | AccessMode::ReadWrite)
    }

    /// Combine two modes (union semantics). Used when multiple allow rules
    /// match the same path.
    pub fn union(self, other: AccessMode) -> AccessMode {
        match (self, other) {
            (AccessMode::ReadWrite, _) | (_, AccessMode::ReadWrite) => AccessMode::ReadWrite,
            (AccessMode::Read, AccessMode::Read) => AccessMode::Read,
            (AccessMode::Write, AccessMode::Write) => AccessMode::Write,
            (AccessMode::Read, AccessMode::Write) | (AccessMode::Write, AccessMode::Read) => {
                AccessMode::ReadWrite
            }
        }
    }
}

/// A single path access rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathRule {
    /// The glob pattern, e.g. `/home/**/.ssh/*` or `C:/Users/*/Documents/**`.
    pub pattern: String,
    /// The access granted by this rule, or `None` for an explicit deny.
    pub mode: Option<AccessMode>,
    /// Free-form reason for audit logs.
    pub reason: String,
}

impl PathRule {
    /// Create an allow rule.
    pub fn allow(pattern: impl Into<String>, mode: AccessMode) -> Self {
        Self {
            pattern: pattern.into(),
            mode: Some(mode),
            reason: String::new(),
        }
    }

    /// Create a deny rule.
    pub fn deny(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            mode: None,
            reason: String::new(),
        }
    }

    /// Attach a human-readable reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Is this a deny rule?
    pub fn is_deny(&self) -> bool {
        self.mode.is_none()
    }
}

/// The outcome of a path access check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessVerdict {
    /// Access is allowed with the given mode.
    Allow(AccessMode),
    /// Access is explicitly denied. The matching deny rule's reason is
    /// included for audit.
    Deny(String),
    /// No rule matched; the caller should fall back to its default policy.
    NoMatch,
}

impl AccessVerdict {
    /// Was access allowed?
    pub fn is_allow(&self) -> bool {
        matches!(self, AccessVerdict::Allow(_))
    }

    /// Was access explicitly denied?
    pub fn is_deny(&self) -> bool {
        matches!(self, AccessVerdict::Deny(_))
    }
}

/// A compiled path-access rule set.
///
/// Rules are evaluated in order of specificity (pattern length descending) so
/// that the most specific match wins. Deny rules always take precedence over
/// allow rules regardless of specificity, mirroring the semantics of the
/// platform sandboxes (bwrap/Seatbelt/Windows ACLs).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathWhitelist {
    rules: Vec<PathRule>,
}

impl PathWhitelist {
    /// Create an empty whitelist.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a whitelist from a slice of rules. The rules are sorted into
    /// evaluation order (denies first, then by descending specificity).
    pub fn from_rules(rules: Vec<PathRule>) -> Self {
        let mut wl = Self { rules };
        wl.sort_rules();
        wl
    }

    /// Add a rule and re-sort.
    pub fn add(&mut self, rule: PathRule) {
        self.rules.push(rule);
        self.sort_rules();
    }

    /// Add an allow rule (builder style).
    pub fn with_allow(mut self, pattern: impl Into<String>, mode: AccessMode) -> Self {
        self.rules.push(PathRule::allow(pattern, mode));
        self.sort_rules();
        self
    }

    /// Add a deny rule (builder style).
    pub fn with_deny(mut self, pattern: impl Into<String>) -> Self {
        self.rules.push(PathRule::deny(pattern));
        self.sort_rules();
        self
    }

    /// Number of rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Is the rule set empty?
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterate over the rules in evaluation order.
    pub fn rules(&self) -> &[PathRule] {
        &self.rules
    }

    /// Check read access for a path.
    pub fn check_read(&self, path: &str) -> AccessVerdict {
        self.check(path, AccessIntent::Read)
    }

    /// Check write access for a path.
    pub fn check_write(&self, path: &str) -> AccessVerdict {
        self.check(path, AccessIntent::Write)
    }

    /// Check access for a path with a given intent.
    pub fn check(&self, path: &str, intent: AccessIntent) -> AccessVerdict {
        let normalised = normalise_path(path);
        for rule in &self.rules {
            if !glob_match(&rule.pattern, &normalised) {
                continue;
            }
            match rule.mode {
                None => {
                    return AccessVerdict::Deny(if rule.reason.is_empty() {
                        format!("path '{}' matches deny rule '{}'", path, rule.pattern)
                    } else {
                        rule.reason.clone()
                    });
                }
                Some(mode) => {
                    let granted = match intent {
                        AccessIntent::Read => mode.allows_read(),
                        AccessIntent::Write => mode.allows_write(),
                    };
                    if granted {
                        return AccessVerdict::Allow(mode);
                    }
                    // The rule matched but does not grant the requested
                    // intent; keep scanning for a more permissive rule.
                }
            }
        }
        AccessVerdict::NoMatch
    }

    /// Return all rules whose pattern matches `path`, in evaluation order.
    /// Useful for diagnostics and audit.
    pub fn matches_for(&self, path: &str) -> Vec<&PathRule> {
        let normalised = normalise_path(path);
        self.rules
            .iter()
            .filter(|r| glob_match(&r.pattern, &normalised))
            .collect()
    }

    /// Merge another whitelist into this one (union of rules).
    pub fn merge(&mut self, other: &PathWhitelist) {
        for rule in &other.rules {
            if !self
                .rules
                .iter()
                .any(|r| r.pattern == rule.pattern && r.mode == rule.mode)
            {
                self.rules.push(rule.clone());
            }
        }
        self.sort_rules();
    }

    fn sort_rules(&mut self) {
        // Denies first (most specific deny wins), then allows by descending
        // specificity. Specificity is approximated by the pattern's literal
        // (non-glob) length.
        self.rules.sort_by(|a, b| {
            let a_deny = a.is_deny();
            let b_deny = b.is_deny();
            match (a_deny, b_deny) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => {
                    let la = specificity(&a.pattern);
                    let lb = specificity(&b.pattern);
                    lb.cmp(&la) // descending
                }
            }
        });
    }
}

/// The access intent of a check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessIntent {
    Read,
    Write,
}

/// Approximate specificity of a pattern: the number of literal (non-glob)
/// characters. Longer literals are more specific.
fn specificity(pattern: &str) -> usize {
    pattern
        .chars()
        .filter(|c| !matches!(c, '*' | '?' | '[' | ']'))
        .count()
}

/// Normalise a path: collapse redundant separators, resolve `.` segments,
/// and trim trailing separators. `..` segments are resolved where possible;
/// unresolved `..` at the root are preserved.
pub fn normalise_path(path: &str) -> String {
    let is_windows = is_windows_path(path);
    let sep = if is_windows { '\\' } else { '/' };
    let alt_sep = if is_windows { '/' } else { '\\' };

    // Split on both separators and drive letter.
    let (drive, rest) = if is_windows {
        let d = &path[..2];
        let r = &path[2..];
        (Some(d), r)
    } else {
        (None, path)
    };

    let absolute = rest.starts_with('/') || rest.starts_with('\\');
    let mut segments: Vec<&str> = Vec::new();
    for seg in rest.split([sep, alt_sep]) {
        if seg.is_empty() || seg == "." {
            continue;
        }
        if seg == ".." {
            if let Some(last) = segments.last() {
                if *last != ".." {
                    segments.pop();
                    continue;
                }
            }
            // At the root, keep the `..` (it has no effect).
            if absolute {
                continue;
            }
            segments.push("..");
        } else {
            segments.push(seg);
        }
    }

    let mut out = String::new();
    if let Some(d) = drive {
        out.push_str(d);
    }
    if absolute {
        out.push(sep);
    }
    out.push_str(&segments.join(&sep.to_string()));
    if out.is_empty() { ".".to_string() } else { out }
}

/// Glob-match a pattern against a path.
///
/// Supports:
/// - `*` matches any number of characters except a path separator.
/// - `**` matches any number of characters including separators (but only
///   between path segments, i.e. `a/**/b`).
/// - `?` matches a single non-separator character.
/// - `[abc]` and `[!abc]` character classes.
/// - `{a,b,c}` alternation (expanded before matching).
///
/// Matching is case-insensitive on Windows, case-sensitive elsewhere.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    // Expand brace alternations into a list of patterns and match any.
    let expansions = expand_braces(pattern);
    for p in expansions {
        if glob_match_single(&p, path) {
            return true;
        }
    }
    false
}

/// Expand `{a,b,c}` alternations into a Cartesian product of patterns.
fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };
    // Find the matching close brace.
    let mut depth = 0i32;
    let mut close = None;
    for (i, c) in pattern[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close) = close else {
        return vec![pattern.to_string()];
    };
    let prefix = &pattern[..open];
    let inner = &pattern[open + 1..close];
    let suffix = &pattern[close + 1..];

    let alternatives: Vec<&str> = split_top_level(inner, ',');
    let mut out = Vec::new();
    for alt in alternatives {
        let combined = format!("{prefix}{alt}{suffix}");
        out.extend(expand_braces(&combined));
    }
    out
}

/// Split on `sep` at the top brace level only.
fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Match a single (brace-free) glob pattern against a path.
fn glob_match_single(pattern: &str, path: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = path.chars().collect();
    let case_insensitive = is_windows_path(pattern) || is_windows_path(path);
    glob_rec(&pat, 0, &txt, 0, case_insensitive)
}

/// Recursive glob matcher with backtracking for `*` and `**`.
fn glob_rec(pat: &[char], pi: usize, txt: &[char], ti: usize, ci: bool) -> bool {
    let mut pi = pi;
    let mut ti = ti;

    loop {
        if pi >= pat.len() {
            return ti >= txt.len();
        }
        match pat[pi] {
            '*' => {
                // Check for `**` (globstar).
                if pi + 1 < pat.len() && pat[pi + 1] == '*' {
                    // `**` matches across separators. Consume the `**` and any
                    // following separator if present, then try to match the
                    // rest at every position.
                    let mut next_pi = pi + 2;
                    // Optional separator after `**`: `a/**/b` should match
                    // `a/b` too.
                    if next_pi < pat.len() && is_sep(pat[next_pi]) {
                        next_pi += 1;
                    }
                    if next_pi >= pat.len() {
                        return true; // trailing `**` matches everything
                    }
                    // Try matching the remainder at each position from ti on.
                    for skip in ti..=txt.len() {
                        if glob_rec(pat, next_pi, txt, skip, ci) {
                            return true;
                        }
                    }
                    return false;
                }
                // Single `*` matches any run of non-separator characters.
                let next_pi = pi + 1;
                if next_pi >= pat.len() {
                    // Trailing `*` matches the rest if no separator remains.
                    return !txt[ti..].iter().any(|c| is_sep(*c));
                }
                // Try zero-length match and progressively longer matches.
                for skip in ti..=txt.len() {
                    // `*` cannot cross a separator: stop if we hit one.
                    if skip > ti && is_sep(txt[skip - 1]) {
                        break;
                    }
                    if glob_rec(pat, next_pi, txt, skip, ci) {
                        return true;
                    }
                }
                return false;
            }
            '?' => {
                if ti >= txt.len() || is_sep(txt[ti]) {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
            '[' => {
                if ti >= txt.len() {
                    return false;
                }
                let (matched, next_pi) = match_class(pat, pi, txt[ti], ci);
                if !matched {
                    return false;
                }
                pi = next_pi;
                ti += 1;
            }
            c => {
                if ti >= txt.len() {
                    return false;
                }
                if !char_eq(c, txt[ti], ci) {
                    return false;
                }
                pi += 1;
                ti += 1;
            }
        }
    }
}

/// Parse a `[...]` or `[!...]` character class starting at `pat[start]`.
/// Returns `(matched, index_after_closing_bracket)`.
fn match_class(pat: &[char], start: usize, ch: char, ci: bool) -> (bool, usize) {
    let mut i = start + 1;
    let mut negate = false;
    if i < pat.len() && (pat[i] == '!' || pat[i] == '^') {
        negate = true;
        i += 1;
    }
    let mut matched = false;
    while i < pat.len() && pat[i] != ']' {
        let lo = pat[i];
        // Range: `a-z`.
        if i + 2 < pat.len() && pat[i + 1] == '-' && pat[i + 2] != ']' {
            let hi = pat[i + 2];
            let c = ch as u32;
            let lo_v = lo as u32;
            let hi_v = hi as u32;
            let (lo_v, hi_v) = if lo_v <= hi_v {
                (lo_v, hi_v)
            } else {
                (hi_v, lo_v)
            };
            if c >= lo_v && c <= hi_v {
                matched = true;
            }
            if ci {
                // Also check case-folded.
                let folded = if ch.is_ascii_uppercase() {
                    ch.to_ascii_lowercase() as u32
                } else if ch.is_ascii_lowercase() {
                    ch.to_ascii_uppercase() as u32
                } else {
                    c
                };
                if folded >= lo_v && folded <= hi_v {
                    matched = true;
                }
            }
            i += 3;
        } else {
            if char_eq(lo, ch, ci) {
                matched = true;
            }
            i += 1;
        }
    }
    // Skip the closing `]` if present.
    if i < pat.len() && pat[i] == ']' {
        i += 1;
    }
    (matched ^ negate, i)
}

fn char_eq(a: char, b: char, ci: bool) -> bool {
    if ci {
        a.eq_ignore_ascii_case(&b)
    } else {
        a == b
    }
}

fn is_sep(c: char) -> bool {
    c == '/' || c == '\\'
}

fn is_windows_path(path: &str) -> bool {
    path.len() >= 2 && path.as_bytes()[1] == b':' && path.as_bytes()[0].is_ascii_alphabetic()
}

/// Verify that a real path on disk is permitted by the whitelist for the given
/// intent. Resolves symlinks before checking so that a symlink escape cannot
/// bypass the rules.
pub fn verify_path_on_disk(wl: &PathWhitelist, path: &Path, intent: AccessIntent) -> AccessVerdict {
    let canonical = match std::fs::canonicalize(path) {
        Ok(p) => p.to_string_lossy().to_string(),
        Err(_) => path.to_string_lossy().to_string(),
    };
    match intent {
        AccessIntent::Read => wl.check_read(&canonical),
        AccessIntent::Write => wl.check_write(&canonical),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_basic() {
        assert!(glob_match("/home/*/docs", "/home/alice/docs"));
        assert!(!glob_match("/home/*/docs", "/home/alice/pictures"));
        assert!(glob_match("/home/**/*.txt", "/home/alice/docs/notes.txt"));
        assert!(glob_match("/home/**/*.txt", "/home/notes.txt"));
        assert!(glob_match("a/?", "a/b"));
        assert!(!glob_match("a/?", "a/bc"));
        assert!(glob_match("a/[abc]", "a/b"));
        assert!(!glob_match("a/[abc]", "a/d"));
        assert!(glob_match("a/[!abc]", "a/d"));
        assert!(glob_match("{a,b}/c", "a/c"));
        assert!(glob_match("{a,b}/c", "b/c"));
        assert!(!glob_match("{a,b}/c", "c/c"));
    }

    #[test]
    fn deny_precedence() {
        let wl = PathWhitelist::new()
            .with_allow("/home/**", AccessMode::ReadWrite)
            .with_deny("/home/*/.ssh/**");
        assert_eq!(
            wl.check_read("/home/alice/docs/x"),
            AccessVerdict::Allow(AccessMode::ReadWrite)
        );
        assert!(wl.check_read("/home/alice/.ssh/id_rsa").is_deny());
    }

    #[test]
    fn normalise() {
        assert_eq!(normalise_path("/a/b/./c/../d/"), "/a/b/d");
        assert_eq!(normalise_path("/a//b///c"), "/a/b/c");
        // A relative path that resolves to the current directory becomes ".".
        assert_eq!(normalise_path("a/b/../.."), ".");
    }

    #[test]
    fn windows_case_insensitive() {
        let wl = PathWhitelist::new().with_allow("C:/Users/**", AccessMode::Read);
        assert!(wl.check_read("c:/users/alice/file").is_allow());
    }
}
