//! Rule application ported from `tokenjuice/reducer.py`.
//!
//! [`reduce_with_rule`] returns the compact summary plus named counter facts;
//! [`format_inline`] (ported from `plugin._format_inline`) stitches the summary
//! and facts into the final inline text.

use std::collections::BTreeMap;

use regex::Regex;

use crate::tokenjuice::formatters::{
    count_pattern, dedupe_adjacent, head_tail, strip_ansi, trim_empty_edges,
};
use crate::tokenjuice::types::{Counter, OutputMatch, Rule};

/// Translate a Python-style flag string (`"i"`, `"m"`, `"im"`) into the Rust
/// `regex` crate's inline-flag suffix. Unknown flags are ignored, matching the
/// lenient Python behaviour. Returns `"iMs"`-style letters suitable for
/// `(?FLAGS:...)` or empty when no flags are set.
pub fn compile_flags(flags: &str) -> String {
    let mut out = String::new();
    if flags.contains('i') {
        out.push('i');
    }
    if flags.contains('m') {
        out.push('m');
    }
    if flags.contains('s') {
        out.push('s');
    }
    out
}

/// Compile a pattern with the given flags into `(?flags:pattern)`. Bad patterns
/// yield `None`; callers skip them, matching Python's `except re.error: continue`.
fn compile(pattern: &str, flags: &str) -> Option<Regex> {
    let suffix = compile_flags(flags);
    let source = if suffix.is_empty() {
        pattern.to_string()
    } else {
        format!("(?{suffix}:{pattern})")
    };
    Regex::new(&source).ok()
}

/// Equivalent of Python `reducer._apply_output_matches`: return the first
/// `OutputMatch.message` whose pattern matches anywhere in `text` (multiline).
pub fn apply_output_matches(rule: &Rule, text: &str) -> Option<String> {
    for entry in &rule.output_matches {
        let OutputMatch {
            pattern,
            message,
            flags,
        } = entry;
        if let Some(re) = compile(pattern, flags.as_deref().unwrap_or("")) {
            if re.is_match(text) {
                return Some(message.clone());
            }
        }
    }
    None
}

/// Resolve the head/tail window for the given exit code. On non-zero exit with
/// `failure.preserveOnFailure`, the failure window is used; otherwise the
/// summarize window. Defaults mirror Python (`head=8`, `tail=8` for summarize;
/// `head=12`, `tail=12` for failure).
pub fn summarize_window(rule: &Rule, exit_code: i64) -> (usize, usize) {
    let to_usize = |v: Option<i64>| v.map(|n| n.max(0) as usize);
    if exit_code != 0 && rule.failure.preserve_on_failure {
        let head = to_usize(rule.failure.head).unwrap_or(12);
        let tail = to_usize(rule.failure.tail).unwrap_or(12);
        // `_FAILURE_PRESERVE_ENV` lever: when enabled, the failure window is
        // grown to at least the summarize window. We read the env here to keep
        // the public signature lean; the default (unset) is off, matching Python.
        if failure_preserve_enabled() {
            let s_head = to_usize(rule.summarize.head).unwrap_or(8);
            let s_tail = to_usize(rule.summarize.tail).unwrap_or(8);
            return (head.max(s_head), tail.max(s_tail));
        }
        (head, tail)
    } else {
        (
            to_usize(rule.summarize.head).unwrap_or(8),
            to_usize(rule.summarize.tail).unwrap_or(8),
        )
    }
}

fn failure_preserve_enabled() -> bool {
    matches!(
        std::env::var("OPENSQUILLA_TOOLCOMP_FAILURE_PRESERVE")
            .ok()
            .map(|v| v.trim().to_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on" | "enabled")
    )
}

/// Reduce `raw_text` under `rule`, returning `(summary, facts)`.
///
/// This is the faithful port of `reducer.reduce_with_rule`. The summary is the
/// windowed, deduped, filtered text; `facts` maps each counter name to its
/// match count. An [`apply_output_matches`] hit short-circuits with an empty
/// facts map, and an `on_empty` rule whose every line is filtered away returns
/// the on-empty text with an empty facts map.
pub fn reduce_with_rule(
    rule: &Rule,
    raw_text: &str,
    exit_code: i64,
) -> (String, BTreeMap<String, usize>) {
    let text = if rule.transforms.strip_ansi {
        strip_ansi(raw_text)
    } else {
        raw_text.to_string()
    };

    if let Some(message) = apply_output_matches(rule, &text) {
        return (message, BTreeMap::new());
    }

    let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();

    if rule.transforms.trim_empty_edges {
        lines = trim_empty_edges(&lines);
    }
    if rule.transforms.dedupe_adjacent {
        lines = dedupe_adjacent(&lines);
    }

    // Counters may run over the pre-keep or post-keep line set.
    let counter_lines = lines.clone();

    let skip_patterns: Vec<Regex> = rule
        .filters
        .skip_patterns
        .iter()
        .filter_map(|p| compile(p, ""))
        .collect();
    if !skip_patterns.is_empty() {
        lines.retain(|line| !skip_patterns.iter().any(|re| re.is_match(line)));
    }

    let keep_patterns: Vec<Regex> = rule
        .filters
        .keep_patterns
        .iter()
        .filter_map(|p| compile(p, ""))
        .collect();
    if !keep_patterns.is_empty() {
        let kept: Vec<String> = lines
            .iter()
            .filter(|line| keep_patterns.iter().any(|re| re.is_match(line)))
            .cloned()
            .collect();
        if !kept.is_empty() {
            lines = kept;
        }
    }

    if lines.is_empty() {
        if let Some(on_empty) = &rule.on_empty {
            return (on_empty.clone(), BTreeMap::new());
        }
    }

    let fact_source = if rule.counter_source == "preKeep" {
        &counter_lines
    } else {
        &lines
    };
    let mut facts: BTreeMap<String, usize> = BTreeMap::new();
    for counter in &rule.counters {
        let Counter {
            name,
            pattern,
            flags,
        } = counter;
        let count = count_pattern(fact_source, pattern, flags.as_deref().unwrap_or(""));
        facts.insert(name.clone(), count);
    }

    let (head, tail) = summarize_window(rule, exit_code);
    let compacted = head_tail(&lines, head, tail);
    (compacted.join("\n").trim().to_string(), facts)
}

/// Stitch the summary and counter facts into the final inline text, ported
/// from `plugin._format_inline`. A non-zero exit prepends an `exit N` line;
/// non-zero counter facts are joined as `name: count` (semicolon-separated).
pub fn format_inline(summary: &str, facts: &BTreeMap<String, usize>, exit_code: i64) -> String {
    let mut parts: Vec<String> = Vec::new();
    if exit_code != 0 {
        parts.push(format!("exit {exit_code}"));
    }
    let non_zero: Vec<String> = facts
        .iter()
        .filter(|(_, count)| **count > 0)
        .map(|(name, count)| format!("{name}: {count}"))
        .collect();
    if !non_zero.is_empty() {
        parts.push(non_zero.join("; "));
    }
    parts.push(summary.to_string());
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenjuice::types::{Counter, Failure, Filters, Rule, Summarize, Transforms};

    fn head_tail_rule(head: i64, tail: i64) -> Rule {
        Rule {
            id: "test/head-tail".to_string(),
            family: "test".to_string(),
            r#match: Default::default(),
            transforms: Transforms {
                strip_ansi: false,
                trim_empty_edges: false,
                dedupe_adjacent: false,
            },
            filters: Filters::default(),
            summarize: Summarize {
                head: Some(head),
                tail: Some(tail),
            },
            failure: Failure::default(),
            counters: vec![],
            output_matches: vec![],
            on_empty: None,
            counter_source: "postKeep".to_string(),
            priority: 0,
        }
    }

    #[test]
    fn keeps_head_and_tail() {
        let rule = head_tail_rule(2, 1);
        let content = (0..6)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (summary, facts) = reduce_with_rule(&rule, &content, 0);
        assert!(facts.is_empty());
        assert_eq!(summary, "line 0\nline 1\n... omitted 3 lines ...\nline 5");
    }

    #[test]
    fn dedupes_adjacent() {
        let rule = Rule {
            transforms: Transforms {
                strip_ansi: false,
                trim_empty_edges: false,
                dedupe_adjacent: true,
            },
            summarize: Summarize {
                head: Some(100),
                tail: Some(100),
            },
            ..head_tail_rule(100, 100)
        };
        let content = "x\nx\nx\ny";
        let (summary, _) = reduce_with_rule(&rule, content, 0);
        assert_eq!(summary, "x\ny");
    }

    #[test]
    fn skip_patterns_drop_lines() {
        let rule = Rule {
            filters: Filters {
                skip_patterns: vec!["^npm notice ".to_string()],
                keep_patterns: vec![],
            },
            summarize: Summarize {
                head: Some(100),
                tail: Some(100),
            },
            ..head_tail_rule(100, 100)
        };
        let content = "npm notice log\nreal line\nnpm notice audit\nother";
        let (summary, _) = reduce_with_rule(&rule, content, 0);
        assert_eq!(summary, "real line\nother");
    }

    #[test]
    fn counters_count_post_keep_by_default() {
        let rule = Rule {
            filters: Filters {
                skip_patterns: vec![],
                keep_patterns: vec!["error".to_string()],
            },
            summarize: Summarize {
                head: Some(100),
                tail: Some(100),
            },
            counters: vec![Counter {
                name: "error".to_string(),
                pattern: "error".to_string(),
                flags: Some("i".to_string()),
            }],
            ..head_tail_rule(100, 100)
        };
        // Two "error" lines; keep-patterns keep both => post-keep count is 2.
        let content = "error one\nfine\nerror two";
        let (summary, facts) = reduce_with_rule(&rule, content, 0);
        assert_eq!(summary, "error one\nerror two");
        assert_eq!(facts.get("error"), Some(&2));
    }

    #[test]
    fn on_empty_returns_marker() {
        let rule = Rule {
            filters: Filters {
                skip_patterns: vec![".*".to_string()],
                keep_patterns: vec![],
            },
            on_empty: Some("working tree clean".to_string()),
            summarize: Summarize {
                head: Some(8),
                tail: Some(4),
            },
            ..head_tail_rule(8, 4)
        };
        let (summary, facts) = reduce_with_rule(&rule, "On branch main\nnothing to commit", 0);
        assert_eq!(summary, "working tree clean");
        assert!(facts.is_empty());
    }

    #[test]
    fn output_match_short_circuits() {
        let rule = Rule {
            output_matches: vec![OutputMatch {
                pattern: "up to date, audited \\d+ package".to_string(),
                message: "npm install: up to date".to_string(),
                flags: Some("i".to_string()),
            }],
            summarize: Summarize {
                head: Some(100),
                tail: Some(100),
            },
            ..head_tail_rule(100, 100)
        };
        let content = "up to date, audited 42 packages in 1s";
        let (summary, facts) = reduce_with_rule(&rule, content, 0);
        assert_eq!(summary, "npm install: up to date");
        assert!(facts.is_empty());
    }

    #[test]
    fn failure_window_used_on_error() {
        let rule = Rule {
            failure: Failure {
                preserve_on_failure: true,
                head: Some(1),
                tail: Some(1),
            },
            summarize: Summarize {
                head: Some(8),
                tail: Some(8),
            },
            ..head_tail_rule(8, 8)
        };
        let content = (0..10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (summary, _) = reduce_with_rule(&rule, &content, 1);
        assert_eq!(summary, "line 0\n... omitted 8 lines ...\nline 9");
    }

    #[test]
    fn format_inline_joins_exit_facts_summary() {
        let mut facts = BTreeMap::new();
        facts.insert("error".to_string(), 3);
        facts.insert("warning".to_string(), 0);
        let inline = format_inline("compacted output", &facts, 1);
        assert_eq!(inline, "exit 1\nerror: 3\ncompacted output");
    }

    #[test]
    fn format_inline_omits_zero_facts() {
        let mut facts = BTreeMap::new();
        facts.insert("error".to_string(), 0);
        let inline = format_inline("ok", &facts, 0);
        assert_eq!(inline, "ok");
    }
}
