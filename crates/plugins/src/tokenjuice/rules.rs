//! Rule loading and built-in defaults, ported from `tokenjuice/rules.py`.
//!
//! [`load_rules`] reads JSON rule files from a directory (recursively, skipping
//! any `fixtures` directory, matching Python's `_iter_json_files`). [`default_rules`]
//! returns a curated built-in set embedded as JSON constants so the reducer
//! works with zero configuration. [`sort_rules`] orders rules the same way
//! Python does: `generic/fallback` last, then by descending priority, then by id.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::tokenjuice::types::{OutputMatch, Rule};

/// Built-in default rule JSON payloads. These are a curated subset of the
/// Python rule set covering the most common command families (generic fallback,
/// git, search, build, install, filesystem, devops). Kept inline so the crate
/// works with zero external config and has no build-time dependency on the
/// Python source tree.
const BUILTIN_RULE_JSON: &[&str] = &[
    // --- generic (lowest priority; fallback must sort last) ---
    r#"{
        "id": "generic/help",
        "family": "help",
        "description": "Preserve command help output so agents can inspect available commands and flags.",
        "priority": 25,
        "match": {
            "toolNames": ["exec"],
            "argvIncludesAny": [["--help"], ["help"]],
            "commandIncludesAny": [" --help", " help"]
        },
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "summarize": {"head": 80, "tail": 40},
        "failure": {"preserveOnFailure": true, "head": 80, "tail": 40}
    }"#,
    // --- git ---
    r#"{
        "id": "git/status",
        "family": "git-status",
        "description": "Compact human-readable git status output.",
        "onEmpty": "working tree clean",
        "priority": 10,
        "match": {"argv0": ["git"], "argvIncludes": [["status"]]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {
            "skipPatterns": [
                "^On branch ", "^Your branch is ",
                "^and have \\d+ and \\d+ different commits each.*$",
                "^\\(use \"git .+\" to .+\\)$",
                "^no changes added to commit.*$",
                "^nothing added to commit but untracked files present.*$",
                "^nothing to commit, working tree clean$",
                "^use \"git .+\" to .+"
            ]
        },
        "summarize": {"head": 10, "tail": 4},
        "failure": {"preserveOnFailure": true, "head": 12, "tail": 12},
        "counters": [
            {"name": "modified file", "pattern": "^(?:M:|\\s*modified:|[ MTRU][MTRU]\\s+|[MTRU][ MTRU]\\s+)"},
            {"name": "new file", "pattern": "^(?:A:|\\s*new file:|A.\\s+|.A\\s+)"},
            {"name": "deleted file", "pattern": "^(?:D:|\\s*deleted:|D.\\s+|.D\\s+)"},
            {"name": "untracked file", "pattern": "^(?:\\?\\?:|\\?\\?\\s+|\\s*untracked files:)"}
        ]
    }"#,
    r#"{
        "id": "git/log-oneline",
        "family": "git-history",
        "description": "Compact git log --oneline output while preserving commits.",
        "priority": 10,
        "match": {"argv0": ["git"], "argvIncludes": [["log"], ["--oneline"]]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "summarize": {"head": 8, "tail": 6},
        "failure": {"preserveOnFailure": true, "head": 10, "tail": 10},
        "counters": [{"name": "commit", "pattern": "^[a-f0-9]{7,}\\s", "flags": "m"}]
    }"#,
    r#"{
        "id": "git/diff",
        "family": "git-diff",
        "description": "Compact full git diff patch output while preserving file and hunk headers plus changed lines.",
        "priority": 10,
        "match": {
            "toolNames": ["exec"],
            "commandIncludesAny": ["git diff ", "&& git diff ", "; git diff ", "\ngit diff "]
        },
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {
            "keepPatterns": [
                "^diff --git\\s+.+", "^new file mode\\s+.+", "^deleted file mode\\s+.+",
                "^similarity index\\s+.+", "^rename from\\s+.+", "^rename to\\s+.+",
                "^Binary files .+ differ$", "^---\\s+.+", "^\\+\\+\\+\\s+.+", "^@@\\s+.+",
                "^\\+(?!\\+\\+\\+).+", "^-(?!---).+", "^\\\\ No newline at end of file$",
                "^\\s*\\d+ files? changed.+", "^\\s*create mode .+", "^\\s*delete mode .+"
            ]
        },
        "summarize": {"head": 20, "tail": 12},
        "failure": {"preserveOnFailure": true, "head": 24, "tail": 16},
        "counters": [
            {"name": "changed file", "pattern": "^diff --git\\s", "flags": "m"},
            {"name": "hunk", "pattern": "^@@\\s", "flags": "m"},
            {"name": "added line", "pattern": "^\\+(?!\\+\\+\\+).+", "flags": "m"},
            {"name": "removed line", "pattern": "^-(?!---).+", "flags": "m"}
        ]
    }"#,
    // --- search ---
    r#"{
        "id": "search/rg",
        "family": "search",
        "description": "Compact ripgrep output while preserving match lines.",
        "priority": 10,
        "match": {"argv0": ["rg"], "toolNames": ["exec"]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {
            "keepPatterns": [
                "^.+:\\d+[: -].+", "^.+:.+",
                "error|warn|binary file|permission denied|no such file",
                "^\\d+ matches?$", "^\\d+ files? matched$"
            ]
        },
        "summarize": {"head": 10, "tail": 6},
        "failure": {"preserveOnFailure": true, "head": 12, "tail": 12},
        "counters": [{"name": "match", "pattern": ".+:.+"}]
    }"#,
    // --- build ---
    r#"{
        "id": "build/tsc",
        "family": "build-typescript",
        "description": "Compact TypeScript compiler output while preserving real diagnostics.",
        "priority": 10,
        "match": {"toolNames": ["exec"], "commandIncludes": ["tsc"]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {
            "skipPatterns": [
                "^Files:\\s+\\d+", "^Lines of Library:\\s+\\d+", "^Lines of Definitions:\\s+\\d+",
                "^Lines of TypeScript:\\s+\\d+", "^Lines of JavaScript:\\s+\\d+",
                "^Lines of JSON:\\s+\\d+", "^Lines of Other:\\s+\\d+", "^Identifiers:\\s+\\d+",
                "^Symbols:\\s+\\d+", "^Types:\\s+\\d+", "^Instantiations:\\s+\\d+",
                "^Memory used:\\s+.+", "^Assignability cache size:\\s+\\d+",
                "^Identity cache size:\\s+\\d+", "^Subtype cache size:\\s+\\d+",
                "^Strict subtype cache size:\\s+\\d+", "^I/O Read time:\\s+.+", "^Parse time:\\s+.+",
                "^ResolveModule time:\\s+.+", "^ResolveLibrary time:\\s+.+", "^Program time:\\s+.+",
                "^Bind time:\\s+.+", "^Check time:\\s+.+", "^transformTime time:\\s+.+",
                "^commentTime time:\\s+.+", "^I/O Write time:\\s+.+", "^printTime time:\\s+.+",
                "^Emit time:\\s+.+", "^Total time:\\s+.+", "^Watching for file changes\\."
            ],
            "keepPatterns": [
                "^.+\\(\\d+,\\d+\\):\\s+error TS\\d+: .+",
                "^.+\\(\\d+,\\d+\\):\\s+warning TS\\d+: .+",
                "^Found \\d+ errors?.+", "^error TS\\d+: .+"
            ]
        },
        "summarize": {"head": 4, "tail": 4},
        "failure": {"preserveOnFailure": true, "head": 4, "tail": 6},
        "counters": [
            {"name": "typescript error", "pattern": "TS\\d+"},
            {"name": "error", "pattern": "error", "flags": "i"}
        ]
    }"#,
    // --- install (uses matchOutput alias) ---
    r#"{
        "id": "install/npm-install",
        "family": "dependency-install",
        "description": "Compact npm install output while preserving warnings and audit summaries.",
        "onEmpty": "npm install: ok",
        "priority": 10,
        "matchOutput": [{"pattern": "up to date, audited \\d+ package", "message": "npm install: up to date", "flags": "i"}],
        "match": {"toolNames": ["exec"], "argv0": ["npm"], "argvIncludes": [["install"]]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {"skipPatterns": ["^npm notice .+"]},
        "summarize": {"head": 10, "tail": 8},
        "failure": {"preserveOnFailure": true, "head": 14, "tail": 14},
        "counters": [
            {"name": "warning", "pattern": "warn", "flags": "i"},
            {"name": "vulnerability", "pattern": "vulnerabilit", "flags": "i"}
        ]
    }"#,
    // --- filesystem ---
    r#"{
        "id": "filesystem/ls",
        "family": "filesystem-listing",
        "description": "Compact ls output for directory listings.",
        "priority": 10,
        "match": {"toolNames": ["exec"], "argv0": ["ls"]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "summarize": {"head": 8, "tail": 6},
        "failure": {"preserveOnFailure": true, "head": 12, "tail": 10},
        "counters": [{"name": "item", "pattern": "^(?!total\\s+\\d+).+\\S.*$"}]
    }"#,
    // --- devops ---
    r#"{
        "id": "devops/docker-logs",
        "family": "container-logs",
        "description": "Compact docker logs output while preserving early and late log lines.",
        "priority": 10,
        "match": {"toolNames": ["exec"], "argv0": ["docker"], "argvIncludes": [["logs"]]},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "filters": {
            "keepPatterns": [
                "error|warn|fatal|panic|exception|traceback|timeout|refused|fail",
                "^Caused by:", "^Traceback"
            ]
        },
        "summarize": {"head": 8, "tail": 8},
        "failure": {"preserveOnFailure": true, "head": 14, "tail": 14},
        "counters": [
            {"name": "error", "pattern": "error", "flags": "i"},
            {"name": "warning", "pattern": "warn", "flags": "i"}
        ]
    }"#,
    // --- generic fallback (must sort last) ---
    r#"{
        "id": "generic/fallback",
        "family": "generic",
        "description": "Generic fallback reducer for line-oriented output.",
        "priority": 0,
        "match": {},
        "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
        "summarize": {"head": 200, "tail": 200},
        "failure": {"preserveOnFailure": true, "head": 50, "tail": 50},
        "counters": [
            {"name": "error", "pattern": "error", "flags": "i"},
            {"name": "warning", "pattern": "warning", "flags": "i"}
        ]
    }"#,
];

/// Parse a single rule JSON value into a [`Rule`], mirroring Python's
/// `_load_rule`. Accepts both `outputMatches` (loader key) and `matchOutput`
/// (bundled JSON key) for output substitutions. Returns `None` on a missing
/// or non-string id, matching Python's filter.
pub fn parse_rule(value: &Value) -> Option<Rule> {
    let obj = value.as_object()?;
    let id = obj.get("id")?.as_str()?;
    if id.is_empty() {
        return None;
    }

    // Manually assemble to honor the matchOutput alias and lenient defaults.
    let family = obj
        .get("family")
        .and_then(Value::as_str)
        .unwrap_or("generic")
        .to_string();
    let r#match = obj
        .get("match")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let transforms = obj
        .get("transforms")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let filters = obj
        .get("filters")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let summarize = obj
        .get("summarize")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let failure = obj
        .get("failure")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let counters = obj
        .get("counters")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let output_matches: Vec<OutputMatch> = Rule::output_matches_from(value);
    let on_empty = obj.get("onEmpty").and_then(Value::as_str).map(String::from);
    let counter_source = obj
        .get("counterSource")
        .and_then(Value::as_str)
        .unwrap_or("postKeep")
        .to_string();
    let priority = obj.get("priority").and_then(Value::as_i64).unwrap_or(0);

    Some(Rule {
        id: id.to_string(),
        family,
        r#match,
        transforms,
        filters,
        summarize,
        failure,
        counters,
        output_matches,
        on_empty,
        counter_source,
        priority,
    })
}

/// The built-in default rule set, parsed and priority-sorted. Works with no
/// external configuration.
pub fn default_rules() -> Vec<Rule> {
    let rules: Vec<Rule> = BUILTIN_RULE_JSON
        .iter()
        .filter_map(|json| serde_json::from_str::<Value>(json).ok())
        .filter_map(|v| parse_rule(&v))
        .collect();
    sort_rules(rules)
}

/// Recursively read `.json` rule files from `dir` (skipping any `fixtures`
/// directory), parse each, and return them priority-sorted. Errors reading
/// individual files are logged and skipped, matching Python's best-effort
/// loader; a missing directory returns the empty vector with a debug log.
pub fn load_rules(dir: &Path) -> Vec<Rule> {
    let mut rules = Vec::new();
    if !dir.exists() {
        tracing::debug!(target: "tokenjuice", dir = %dir.display(), "rules directory absent; using no file rules");
        return rules;
    }
    iter_json_files(dir, &mut |path| match fs::read_to_string(path) {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(value) => {
                if let Some(rule) = parse_rule(&value) {
                    rules.push(rule);
                } else {
                    tracing::warn!(target: "tokenjuice", path = %path.display(), "rule file missing id; skipped");
                }
            }
            Err(err) => {
                tracing::warn!(target: "tokenjuice", path = %path.display(), error = %err, "invalid rule JSON; skipped");
            }
        },
        Err(err) => {
            tracing::warn!(target: "tokenjuice", path = %path.display(), error = %err, "unreadable rule file; skipped");
        }
    });
    sort_rules(rules)
}

/// Walk a directory recursively, invoking `f` for each `*.json` file. Any
/// directory named `fixtures` is skipped (mirroring Python's `_iter_json_files`).
fn iter_json_files(root: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name == "fixtures" {
            continue;
        }
        if path.is_dir() {
            iter_json_files(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            f(&path);
        }
    }
}

/// Sort rules the way Python's `load_rules` does: `generic/fallback` last,
/// then descending priority, then ascending id for a stable order.
pub fn sort_rules(mut rules: Vec<Rule>) -> Vec<Rule> {
    rules.sort_by(|a, b| {
        let fallback = |r: &Rule| r.id == "generic/fallback";
        match (fallback(a), fallback(b)) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            _ => b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)),
        }
    });
    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let base =
            std::env::temp_dir().join(format!("tokenjuice-rules-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        base
    }

    #[test]
    fn default_rules_loads_and_sorts_fallback_last() {
        let rules = default_rules();
        assert!(rules.len() >= 9);
        assert_eq!(rules.last().unwrap().id, "generic/fallback");
        // Higher priority (help=25) precedes lower priority (fallback=0).
        let help_idx = rules.iter().position(|r| r.id == "generic/help").unwrap();
        let status_idx = rules.iter().position(|r| r.id == "git/status").unwrap();
        let fallback_idx = rules
            .iter()
            .position(|r| r.id == "generic/fallback")
            .unwrap();
        assert!(help_idx < fallback_idx);
        assert!(status_idx < fallback_idx);
    }

    #[test]
    fn default_rules_parse_match_output_alias() {
        let rules = default_rules();
        let npm = rules
            .iter()
            .find(|r| r.id == "install/npm-install")
            .unwrap();
        assert_eq!(npm.output_matches.len(), 1);
        assert_eq!(npm.output_matches[0].message, "npm install: up to date");
    }

    #[test]
    fn load_rules_reads_directory_recursively() {
        let dir = tmp_dir("recursive");
        let sub = dir.join("git");
        fs::create_dir_all(&sub).unwrap();
        let mut f = fs::File::create(sub.join("status.json")).unwrap();
        f.write_all(
            br#"{"id":"git/status","family":"git-status","match":{"argv0":["git"],"argvIncludes":[["status"]]},"transforms":{"dedupeAdjacent":true},"summarize":{"head":10,"tail":4}}"#,
        ).unwrap();

        let rules = load_rules(&dir);
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "git/status");
        assert_eq!(rules[0].summarize.head, Some(10));
    }

    #[test]
    fn load_rules_skips_fixtures_and_bad_json() {
        let dir = tmp_dir("skip");
        let fixtures = dir.join("fixtures");
        fs::create_dir_all(&fixtures).unwrap();
        let mut f = fs::File::create(fixtures.join("ignored.json")).unwrap();
        f.write_all(br#"{"id":"ignored"}"#).unwrap();
        let mut f = fs::File::create(dir.join("good.json")).unwrap();
        f.write_all(br#"{"id":"good","family":"x"}"#).unwrap();
        let mut f = fs::File::create(dir.join("broken.json")).unwrap();
        f.write_all(b"{not json").unwrap();

        let rules = load_rules(&dir);
        let ids: Vec<&str> = rules.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["good"]);
    }

    #[test]
    fn load_rules_missing_dir_returns_empty() {
        let rules = load_rules(std::path::Path::new("/nonexistent/tokenjuice/rules-xyz"));
        assert!(rules.is_empty());
    }

    #[test]
    fn parse_rule_rejects_missing_id() {
        let v: Value = serde_json::from_str(r#"{"family":"x"}"#).unwrap();
        assert!(parse_rule(&v).is_none());
    }
}
