//! Rule selection ported from `tokenjuice/matcher.py`.
//!
//! [`select_rule`] walks priority-sorted rules and returns the first whose
//! [`rule_matches`] criteria are satisfied. [`command_argv`] tokenizes a command
//! string (or accepts a pre-tokenized argv). The strict-matcher and cd-unwrap
//! env levers are preserved for parity with the Python backend.

use std::path::Path;

use regex::Regex;

use crate::tokenjuice::types::{Rule, RuleMatch};

/// Env var enabling strict `gitSubcommands` / `argvIncludesAny` enforcement.
const MATCHER_STRICT_ENV: &str = "OPENSQUILLA_TOOLCOMP_MATCHER_STRICT";
/// Env var enabling leading `cd <dir> &&` prefix stripping before selection.
const CD_UNWRAP_ENV: &str = "OPENSQUILLA_TOOLCOMP_CD_UNWRAP";

const TRUE_ENV_VALUES: &[&str] = &["1", "true", "yes", "on", "enabled"];

/// Git global options that consume the next argv entry.
const GIT_GLOBAL_OPTIONS_WITH_VALUE: &[&str] = &[
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--super-prefix",
    "--config-env",
];

const GIT_GLOBAL_OPTION_INLINE_PREFIXES: &[&str] = &[
    "--git-dir=",
    "--work-tree=",
    "--namespace=",
    "--exec-path=",
    "--super-prefix=",
    "--config-env=",
];

fn env_flag(var: &str) -> bool {
    matches!(
        std::env::var(var)
            .ok()
            .map(|v| v.trim().to_lowercase())
            .as_deref(),
        Some(v) if TRUE_ENV_VALUES.contains(&v)
    )
}

fn matcher_strict_enabled() -> bool {
    env_flag(MATCHER_STRICT_ENV)
}

fn cd_unwrap_enabled() -> bool {
    env_flag(CD_UNWRAP_ENV)
}

/// Tokenize `command` into argv, honoring a pre-supplied `argv` when given.
///
/// Ports Python's `command_argv`: returns `argv` as-is if present, otherwise
/// shell-splits the command. The shell split uses a small POSIX-ish tokenizer
/// (quotes, backslash escapes, whitespace). On a malformed quote it falls back
/// to whitespace splitting, matching Python's `except ValueError: command.split()`.
pub fn command_argv(command: Option<&str>, argv: Option<&[String]>) -> Vec<String> {
    if let Some(argv) = argv {
        return argv.to_vec();
    }
    let Some(command) = command else {
        return Vec::new();
    };
    shell_split(command)
}

/// Minimal POSIX-ish shell tokenizer sufficient for rule matching (quotes,
/// backslash escapes, whitespace). Not a full `shlex` — it mirrors the subset
/// the Python `shlex.split` produces for typical commands.
fn shell_split(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_token = false;
    let mut quote: Option<char> = None;
    let mut escaping = false;
    for ch in command.chars() {
        if escaping {
            current.push(ch);
            escaping = false;
            continue;
        }
        match ch {
            '\\' if quote != Some('\'') => {
                escaping = true;
                in_token = true;
            }
            c @ ('"' | '\'') => {
                if Some(c) == quote {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(c);
                    in_token = true;
                } else {
                    current.push(c);
                }
            }
            c if c.is_whitespace() && quote.is_none() => {
                if in_token {
                    tokens.push(std::mem::take(&mut current));
                    in_token = false;
                }
            }
            c => {
                current.push(c);
                in_token = true;
            }
        }
    }
    if in_token || !current.is_empty() {
        tokens.push(current);
    }
    if quote.is_some() {
        // Unterminated quote: fall back to whitespace split, matching Python.
        return command.split_whitespace().map(|s| s.to_string()).collect();
    }
    tokens
}

/// The basename of `argv[0]`, stripping a single layer of surrounding quotes.
/// Returns `None` for an empty argv.
pub fn command_name(argv: &[String]) -> Option<String> {
    let first = argv.first()?;
    let mut first = first.as_str();
    if first.starts_with('\'') || first.starts_with('"') {
        first = &first[1..];
    }
    if first.ends_with('\'') || first.ends_with('"') {
        first = &first[..first.len() - 1];
    }
    Some(Path::new(first).file_name()?.to_string_lossy().into_owned())
}

/// Extract the git subcommand verb, skipping global options that consume a
/// value (e.g. `git -C path status` => `status`). Returns `None` if argv0 is
/// not `git` or no subcommand is found.
pub fn git_subcommand(argv: &[String]) -> Option<String> {
    if command_name(argv).as_deref() != Some("git") {
        return None;
    }
    let mut index = 1;
    while index < argv.len() {
        let arg = &argv[index];
        if arg.is_empty() {
            index += 1;
            continue;
        }
        if GIT_GLOBAL_OPTIONS_WITH_VALUE.contains(&arg.as_str()) {
            index += 2;
            continue;
        }
        if GIT_GLOBAL_OPTION_INLINE_PREFIXES
            .iter()
            .any(|p| arg.starts_with(p))
        {
            index += 1;
            continue;
        }
        if arg.starts_with('-') {
            index += 1;
            continue;
        }
        return Some(arg.clone());
    }
    None
}

fn contains_all(argv: &[String], needles: &[String]) -> bool {
    needles.iter().all(|n| argv.contains(n))
}

fn contains_command_text(command: &str, needles: &[String]) -> bool {
    let lowered = command.to_lowercase();
    needles.iter().all(|n| lowered.contains(&n.to_lowercase()))
}

/// Whether a rule's match criteria are satisfied for the given tool invocation.
///
/// Faithful port of `matcher.rule_matches`. An empty [`RuleMatch`] matches
/// everything (the generic fallback). `exit_code` is `1` for errors, `0`
/// otherwise (the engine bridge maps `is_error` to exit code).
pub fn rule_matches(
    rule: &Rule,
    tool_name: &str,
    command: Option<&str>,
    argv: &[String],
    content: &str,
    exit_code: i64,
) -> bool {
    let m: &RuleMatch = &rule.r#match;
    let normalized_tool = if command.is_some() { "exec" } else { tool_name };

    if !m.tool_names.is_empty()
        && !m
            .tool_names
            .iter()
            .any(|t| t == normalized_tool || t == tool_name)
    {
        return false;
    }

    if !m.argv0.is_empty() && (argv.is_empty() || !m.argv0.contains(&argv[0])) {
        return false;
    }

    if !m.git_subcommands.is_empty()
        && matcher_strict_enabled()
        && !m
            .git_subcommands
            .contains(&git_subcommand(argv).unwrap_or_default())
    {
        return false;
    }

    if !m.argv_includes.is_empty()
        && !m
            .argv_includes
            .iter()
            .any(|entry| contains_all(argv, entry))
    {
        return false;
    }

    if !m.argv_includes_any.is_empty()
        && matcher_strict_enabled()
        && !m
            .argv_includes_any
            .iter()
            .any(|entry| contains_all(argv, entry))
    {
        return false;
    }

    let command_text = command
        .map(|c| c.to_string())
        .unwrap_or_else(|| argv.join(" "));

    if !m.command_includes.is_empty() && !contains_command_text(&command_text, &m.command_includes)
    {
        return false;
    }

    if !m.command_includes_any.is_empty()
        && !m
            .command_includes_any
            .iter()
            .any(|needle| command_text.to_lowercase().contains(&needle.to_lowercase()))
    {
        return false;
    }

    if let Some(regex) = &m.command_regex {
        if let Ok(re) = Regex::new(regex) {
            if !re.is_match(&command_text) {
                return false;
            }
        } else {
            return false;
        }
    }

    if !m.exit_codes.is_empty() && !m.exit_codes.contains(&exit_code) {
        return false;
    }

    if let Some(regex) = &m.output_regex {
        if let Ok(re) = Regex::new(&format!("(?m){regex}")) {
            if !re.is_match(content) {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

/// Strip a leading `cd <dir> &&` (or `pushd <dir> &&`) chain from a command,
/// returning the effective command. Returns `None` if no such prefix is present
/// or the prefix is unsafe to strip (unquoted operator/redirection).
///
/// Ports `matcher._match_leading_cd_chain` + `strip_leading_cd_prefix` (up to
/// 8 chained unwraps).
pub fn strip_leading_cd_prefix(command: &str) -> Option<String> {
    let cd_re = Regex::new(r"^\s*(?:cd|pushd)[ \t]+").ok()?;
    let stop_chars = ['&', '|', ';', '<', '>', '\n'];
    let mut current = command.trim().to_string();
    for _ in 0..8 {
        let unwrapped = match_leading_cd_chain(&current, &cd_re, &stop_chars);
        match unwrapped {
            Some(u) => current = u,
            None => return Some(current),
        }
    }
    Some(current)
}

fn match_leading_cd_chain(command: &str, cd_re: &Regex, stop_chars: &[char]) -> Option<String> {
    let kw = cd_re.find(command)?;
    let bytes = command.as_bytes();
    let mut index = kw.end();
    let mut quote: Option<char> = None;
    let mut escaping = false;
    let mut saw_arg = false;
    while index < command.len() {
        let char_start = index;
        let ch = command[char_start..].chars().next()?;
        let len = ch.len_utf8();
        if escaping {
            escaping = false;
        } else if ch == '\\' {
            escaping = true;
        } else if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else if ch == '\'' || ch == '"' {
            quote = Some(ch);
        } else if stop_chars.contains(&ch) {
            return None;
        } else if ch.is_whitespace() {
            break;
        }
        saw_arg = true;
        index += len;
        let _ = bytes; // bytes kept for clarity; indexing uses char len
    }
    if !saw_arg {
        return None;
    }
    // Skip horizontal whitespace between arg and `&&`.
    while index < command.len() {
        let ch = command[index..].chars().next()?;
        if ch == ' ' || ch == '\t' {
            index += ch.len_utf8();
        } else {
            break;
        }
    }
    if command[index..].starts_with("&&") {
        let tail = command[index + 2..].trim();
        if tail.is_empty() {
            None
        } else {
            Some(tail.to_string())
        }
    } else {
        None
    }
}

/// Select the first matching rule from a priority-sorted slice.
///
/// `rules` must already be sorted (see [`crate::tokenjuice::rules::sort_rules`]).
/// When the cd-unwrap env lever is on, a leading `cd <dir> &&` prefix is
/// stripped from `command` before matching. Returns `None` if no rule matches.
pub fn select_rule<'a>(
    rules: &'a [Rule],
    tool_name: &str,
    command: Option<&str>,
    arguments: Option<&serde_json::Value>,
    content: &str,
    exit_code: i64,
) -> Option<&'a Rule> {
    let argv_from_args = arguments
        .and_then(|v| v.get("argv"))
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            let strs: Vec<String> = arr
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect();
            if strs.len() == arr.len() {
                Some(strs)
            } else {
                None
            }
        });

    let mut command = command.map(|c| c.to_string());
    let mut argv = command_argv(command.as_deref(), argv_from_args.as_deref());

    if let Some(cmd) = &command {
        if cd_unwrap_enabled() {
            if let Some(unwrapped) = strip_leading_cd_prefix(cmd) {
                if unwrapped != *cmd.trim() {
                    command = Some(unwrapped.clone());
                    argv = command_argv(Some(&unwrapped), None);
                }
            }
        }
    }

    rules.iter().find(|rule| {
        rule_matches(
            rule,
            tool_name,
            command.as_deref(),
            &argv,
            content,
            exit_code,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenjuice::types::{
        Counter, Failure, Filters, OutputMatch, Rule, RuleMatch, Summarize, Transforms,
    };
    use serde_json::json;

    fn rule(id: &str, m: RuleMatch, priority: i64) -> Rule {
        Rule {
            id: id.to_string(),
            family: "test".to_string(),
            r#match: m,
            transforms: Transforms {
                strip_ansi: true,
                trim_empty_edges: true,
                dedupe_adjacent: true,
            },
            filters: Filters::default(),
            summarize: Summarize {
                head: Some(8),
                tail: Some(4),
            },
            failure: Failure {
                preserve_on_failure: true,
                head: Some(12),
                tail: Some(12),
            },
            counters: vec![Counter {
                name: "error".to_string(),
                pattern: "error".to_string(),
                flags: Some("i".to_string()),
            }],
            output_matches: vec![OutputMatch {
                pattern: "x".to_string(),
                message: "m".to_string(),
                flags: None,
            }],
            on_empty: None,
            counter_source: "postKeep".to_string(),
            priority,
        }
    }

    #[test]
    fn command_argv_uses_supplied_argv() {
        let argv = vec!["git".to_string(), "status".to_string()];
        assert_eq!(
            command_argv(Some("ignored"), Some(&argv)),
            vec!["git".to_string(), "status".to_string()]
        );
    }

    #[test]
    fn command_argv_splits_shell_command() {
        let argv = command_argv(Some("git -C 'path' status"), None);
        assert_eq!(
            argv,
            vec![
                "git".to_string(),
                "-C".to_string(),
                "path".to_string(),
                "status".to_string()
            ]
        );
    }

    #[test]
    fn command_argv_empty_when_none() {
        assert!(command_argv(None, None).is_empty());
    }

    #[test]
    fn git_subcommand_skips_global_option_with_value() {
        let argv = vec![
            "git".to_string(),
            "-C".to_string(),
            "repo".to_string(),
            "status".to_string(),
        ];
        assert_eq!(git_subcommand(&argv), Some("status".to_string()));
    }

    #[test]
    fn rule_matches_empty_match_always() {
        let r = rule("generic/fallback", RuleMatch::default(), 0);
        assert!(rule_matches(
            &r,
            "exec",
            Some("ls"),
            &command_argv(Some("ls"), None),
            "x",
            0
        ));
    }

    #[test]
    fn rule_matches_tool_names_and_argv0() {
        let r = rule(
            "filesystem/ls",
            RuleMatch {
                tool_names: vec!["exec".to_string()],
                argv0: vec!["ls".to_string()],
                ..Default::default()
            },
            0,
        );
        assert!(rule_matches(
            &r,
            "exec",
            Some("ls -la"),
            &command_argv(Some("ls -la"), None),
            "x",
            0
        ));
        // Wrong argv0.
        assert!(!rule_matches(
            &r,
            "exec",
            Some("cat f"),
            &command_argv(Some("cat f"), None),
            "x",
            0
        ));
        // tool_name only (no command) does not satisfy toolNames=["exec"] for arbitrary tool.
        assert!(!rule_matches(&r, "read", None, &[], "x", 0));
    }

    #[test]
    fn rule_matches_argv_includes() {
        let r = rule(
            "git/status",
            RuleMatch {
                argv0: vec!["git".to_string()],
                argv_includes: vec![vec!["status".to_string()]],
                ..Default::default()
            },
            0,
        );
        assert!(rule_matches(
            &r,
            "exec",
            Some("git status"),
            &command_argv(Some("git status"), None),
            "x",
            0
        ));
        assert!(!rule_matches(
            &r,
            "exec",
            Some("git log"),
            &command_argv(Some("git log"), None),
            "x",
            0
        ));
    }

    #[test]
    fn rule_matches_command_includes_any() {
        let r = rule(
            "git/diff",
            RuleMatch {
                command_includes_any: vec!["git diff ".to_string()],
                ..Default::default()
            },
            0,
        );
        assert!(rule_matches(
            &r,
            "exec",
            Some("ls && git diff HEAD"),
            &command_argv(Some("ls && git diff HEAD"), None),
            "x",
            0
        ));
        assert!(!rule_matches(
            &r,
            "exec",
            Some("ls -la"),
            &command_argv(Some("ls -la"), None),
            "x",
            0
        ));
    }

    #[test]
    fn select_rule_priority_orders_then_first_match_wins() {
        let fallback = rule("generic/fallback", RuleMatch::default(), 0);
        let ls = rule(
            "filesystem/ls",
            RuleMatch {
                tool_names: vec!["exec".to_string()],
                argv0: vec!["ls".to_string()],
                ..Default::default()
            },
            10,
        );
        // Sort: fallback last, higher priority first.
        let rules = crate::tokenjuice::rules::sort_rules(vec![fallback, ls]);
        let chosen = select_rule(&rules, "exec", Some("ls -la"), None, "x", 0);
        assert_eq!(chosen.map(|r| r.id.as_str()), Some("filesystem/ls"));
    }

    #[test]
    fn select_rule_reads_argv_from_arguments() {
        let npm = rule(
            "install/npm-install",
            RuleMatch {
                tool_names: vec!["exec".to_string()],
                argv0: vec!["npm".to_string()],
                argv_includes: vec![vec!["install".to_string()]],
                ..Default::default()
            },
            10,
        );
        let fallback = rule("generic/fallback", RuleMatch::default(), 0);
        let rules = crate::tokenjuice::rules::sort_rules(vec![npm, fallback]);
        let args = json!({ "argv": ["npm", "install"] });
        let chosen = select_rule(&rules, "exec", None, Some(&args), "x", 0);
        assert_eq!(chosen.map(|r| r.id.as_str()), Some("install/npm-install"));
    }

    #[test]
    fn strip_leading_cd_prefix_unwraps_chain() {
        let unwrapped = strip_leading_cd_prefix("cd repo && git status");
        assert_eq!(unwrapped.as_deref(), Some("git status"));
    }

    #[test]
    fn strip_leading_cd_prefix_none_when_no_prefix() {
        let unwrapped = strip_leading_cd_prefix("git status");
        assert_eq!(unwrapped.as_deref(), Some("git status"));
    }
}
