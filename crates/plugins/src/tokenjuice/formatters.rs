//! Line-oriented formatters ported from `tokenjuice/formatters.py`.
//!
//! Each function mirrors its Python counterpart: [`strip_ansi`], [`trim_empty_edges`],
//! [`dedupe_adjacent`], [`head_tail`], [`count_pattern`].

use regex::Regex;

use crate::tokenjuice::reducer::compile_flags;

/// Compiled CSI/OSC ANSI escape stripper, equivalent to the Python `ANSI_RE`.
pub fn ansi_regex() -> Regex {
    Regex::new(r"\x1b(?:[@-Z\\-_]|\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~])")
        .expect("hardcoded ANSI regex must compile")
}

/// Remove ANSI escape sequences from `text`.
pub fn strip_ansi(text: &str) -> String {
    ansi_regex().replace_all(text, "").into_owned()
}

/// Drop leading and trailing lines that are empty or whitespace-only.
pub fn trim_empty_edges(lines: &[String]) -> Vec<String> {
    let mut start = 0usize;
    let mut end = lines.len();
    while start < end && lines[start].trim().is_empty() {
        start += 1;
    }
    while end > start && lines[end - 1].trim().is_empty() {
        end -= 1;
    }
    lines[start..end].to_vec()
}

/// Collapse runs of identical adjacent lines to a single line.
pub fn dedupe_adjacent(lines: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(lines.len());
    let mut last: Option<&str> = None;
    for line in lines {
        if Some(line.as_str()) != last {
            out.push(line.clone());
        }
        last = Some(line.as_str());
    }
    out
}

/// Keep the first `head` and last `tail` lines, inserting an
/// `"... omitted N lines ..."` marker when lines are dropped.
pub fn head_tail(lines: &[String], head: usize, tail: usize) -> Vec<String> {
    if lines.len() <= head + tail {
        return lines.to_vec();
    }
    let omitted = lines.len() - head - tail;
    let mut out = Vec::with_capacity(head + tail + 1);
    out.extend_from_slice(&lines[..head]);
    out.push(format!("... omitted {omitted} lines ..."));
    out.extend_from_slice(&lines[lines.len() - tail..]);
    out
}

/// Count lines where `pattern` matches anywhere. `flags` mirrors the Python
/// flag string: `i` => case-insensitive, `m` => multiline.
pub fn count_pattern(lines: &[String], pattern: &str, flags: &str) -> usize {
    let flag_suffix = compile_flags(flags);
    let source = if flag_suffix.is_empty() {
        pattern.to_string()
    } else {
        format!("(?{flag_suffix}:{pattern})")
    };
    let re = match Regex::new(&source) {
        Ok(re) => re,
        Err(err) => {
            tracing::warn!(target: "tokenjuice", pattern = pattern, error = %err, "invalid counter pattern; counting as zero");
            return 0;
        }
    };
    lines.iter().filter(|line| re.is_match(line)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_codes() {
        let s = "\x1b[31mred\x1b[0m line";
        assert_eq!(strip_ansi(s), "red line");
    }

    #[test]
    fn trims_blank_edges_only() {
        let lines = vec![
            "".to_string(),
            "  ".to_string(),
            "a".to_string(),
            "".to_string(),
            "b".to_string(),
            "".to_string(),
        ];
        assert_eq!(
            trim_empty_edges(&lines),
            vec!["a".to_string(), "".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn dedupes_only_adjacent() {
        let lines = vec![
            "x".to_string(),
            "x".to_string(),
            "y".to_string(),
            "x".to_string(),
        ];
        assert_eq!(
            dedupe_adjacent(&lines),
            vec!["x".to_string(), "y".to_string(), "x".to_string()]
        );
    }

    #[test]
    fn head_tail_inserts_marker() {
        let lines: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let out = head_tail(&lines, 2, 2);
        assert_eq!(
            out,
            vec![
                "0".to_string(),
                "1".to_string(),
                "... omitted 6 lines ...".to_string(),
                "8".to_string(),
                "9".to_string()
            ]
        );
    }

    #[test]
    fn head_tail_noop_when_small() {
        let lines = vec!["a".to_string(), "b".to_string()];
        assert_eq!(head_tail(&lines, 8, 6), lines);
    }

    #[test]
    fn counts_case_insensitive() {
        let lines = vec![
            "Error here".to_string(),
            "ok".to_string(),
            "error again".to_string(),
        ];
        assert_eq!(count_pattern(&lines, "error", "i"), 2);
        assert_eq!(count_pattern(&lines, "error", ""), 1);
    }
}
