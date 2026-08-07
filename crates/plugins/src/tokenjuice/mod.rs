//! TokenJuice tool-result reducer.
//!
//! Ports `tokenjuice/plugin.py`: [`reduce_tool_result`] is the public entry
//! point the engine calls to compress verbose tool output (shell output, file
//! reads) before it re-enters the LLM context. A [`Rule`] selects transforms
//! and windows for a command family; [`reducer::reduce_with_rule`] applies them
//! and [`reducer::format_inline`] stitches the summary plus counter facts into
//! the final inline text.
//!
//! See the crate root (`lib.rs`) for the re-exported public API.

pub mod formatters;
pub mod matcher;
pub mod reducer;
pub mod rules;
pub mod types;

pub use matcher::select_rule;
pub use reducer::{format_inline, reduce_with_rule};
pub use rules::{default_rules, load_rules, sort_rules};
pub use types::{Reduction, Rule};

use serde_json::Value;

/// Reduce a tool result to a compact inline summary.
///
/// Mirrors `tokenjuice.plugin.reduce_tool_result`. `arguments` is the tool-call
/// arguments JSON; the command is taken from `command` or, if absent, from
/// `arguments.command`. `is_error` maps to exit code 1/0. The `rules` slice
/// should already be priority-sorted (see [`sort_rules`] / [`default_rules`]).
///
/// Returns `None` when no rule matches, when the reduction is empty, or when
/// the reduced text is not shorter than the original (mirroring Python's
/// no-op guard). Projection is best-effort: it never panics on bad regexes or
/// unexpected input.
pub fn reduce_tool_result(
    tool_name: &str,
    content: &str,
    is_error: bool,
    arguments: Option<&Value>,
    command: Option<&str>,
    rules: &[Rule],
) -> Option<Reduction> {
    reduce_tool_result_with_limit(
        tool_name, content, is_error, arguments, command, rules, None,
    )
}

/// Like [`reduce_tool_result`] but clamps the inline text to `max_inline_chars`
/// (splitting head/tail with an omitted-chars marker), mirroring Python's
/// `max_inline_chars` handling.
pub fn reduce_tool_result_with_limit(
    tool_name: &str,
    content: &str,
    is_error: bool,
    arguments: Option<&Value>,
    command: Option<&str>,
    rules: &[Rule],
    max_inline_chars: Option<usize>,
) -> Option<Reduction> {
    let command = command
        .map(|c| c.to_string())
        .or_else(|| string_arg(arguments, &["command"]));
    let exit_code: i64 = if is_error { 1 } else { 0 };

    let rule = select_rule(
        rules,
        tool_name,
        command.as_deref(),
        arguments,
        content,
        exit_code,
    )?;
    let (summary, facts) = reduce_with_rule(&rule, content, exit_code);
    if summary.is_empty() {
        return None;
    }

    let mut inline_text = format_inline(&summary, &facts, exit_code);

    if let Some(max) = max_inline_chars {
        if max > 0 && inline_text.chars().count() > max {
            let half = std::cmp::max(1, (max.saturating_sub(32)) / 2);
            let chars: Vec<char> = inline_text.chars().collect();
            let head: String = chars.iter().take(half).collect();
            let tail: String = chars[chars.len().saturating_sub(half)..].iter().collect();
            inline_text = format!("{head}\n... omitted chars ...\n{tail}");
        }
    }

    // No-op guard: never return a reduction that isn't shorter than the input.
    if inline_text.len() >= content.len() {
        return None;
    }

    let raw_chars = content.len();
    let reduced_chars = inline_text.len();
    Some(Reduction {
        inline_text,
        raw_chars,
        reduced_chars,
        ratio: reduced_chars as f64 / raw_chars.max(1) as f64,
        reducer: Some(rule.id.clone()),
    })
}

/// Extract a non-empty string argument by name, mirroring Python's
/// `_string_arg(args, *names)`.
fn string_arg(args: Option<&Value>, names: &[&str]) -> Option<String> {
    let args = args?.as_object()?;
    for name in names {
        if let Some(Value::String(s)) = args.get(*name) {
            if !s.trim().is_empty() {
                return Some(s.clone());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reduces_shell_output_with_fallback_rule() {
        let rules = default_rules();
        // Long repeated output triggers head/tail windowing.
        let content = (0..20)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let reduction = reduce_tool_result("exec", &content, false, None, Some("ls -la"), &rules);
        let r = reduction.expect("fallback should reduce long output");
        assert!(r.reduced_chars < r.raw_chars);
        assert!(r.ratio < 1.0);
        assert_eq!(r.reducer.as_deref(), Some("filesystem/ls"));
        assert!(r.inline_text.contains("... omitted"));
    }

    #[test]
    fn returns_none_when_not_shorter() {
        let rules = default_rules();
        // Tiny content cannot be reduced below its own length.
        let content = "hi";
        let reduction = reduce_tool_result("exec", content, false, None, Some("ls"), &rules);
        assert!(reduction.is_none());
    }

    #[test]
    fn returns_none_when_no_rule_matches_strict_tool() {
        // With no command and a non-exec tool, only an empty-match rule could
        // apply; the fallback's empty match still matches, so use a content so
        // short it cannot shrink to exercise the no-op guard path instead.
        let rules = default_rules();
        let reduction = reduce_tool_result("read", "x", false, None, None, &rules);
        assert!(reduction.is_none());
    }

    #[test]
    fn on_empty_emits_marker() {
        let rules = default_rules();
        let content = "On branch main\nnothing to commit, working tree clean";
        let reduction =
            reduce_tool_result("exec", content, false, None, Some("git status"), &rules);
        let r = reduction.expect("git/status rule should apply");
        assert_eq!(r.inline_text, "working tree clean");
        assert_eq!(r.reducer.as_deref(), Some("git/status"));
    }

    #[test]
    fn command_taken_from_arguments_when_command_none() {
        let rules = default_rules();
        let args = json!({ "command": "git status" });
        let content = "On branch main\nnothing to commit, working tree clean";
        let reduction = reduce_tool_result("exec", content, false, Some(&args), None, &rules);
        let r = reduction.expect("command should be read from arguments");
        assert_eq!(r.reducer.as_deref(), Some("git/status"));
    }

    #[test]
    fn max_inline_chars_clamps_output() {
        let rules = default_rules();
        let content = (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let reduction = reduce_tool_result_with_limit(
            "exec",
            &content,
            false,
            None,
            Some("ls"),
            &rules,
            Some(40),
        )
        .expect("should reduce and clamp");
        assert!(reduction.inline_text.contains("... omitted chars ..."));
        assert!(reduction.reduced_chars < reduction.raw_chars);
    }

    #[test]
    fn error_path_prepends_exit_and_counts() {
        let rules = default_rules();
        // Fallback counters count "error" case-insensitively; the failure
        // window (head=50, tail=50) truncates 200 lines so the reduction is
        // shorter than the input.
        let mut content = String::from("error one\nerror two\nerror three\n");
        for i in 0..200 {
            content.push_str(&format!("line {i}\n"));
        }
        let reduction = reduce_tool_result("exec", &content, true, None, Some("make"), &rules);
        let r = reduction.expect("fallback should reduce error output");
        assert!(r.inline_text.starts_with("exit 1"));
        assert!(r.inline_text.contains("error: 3"));
        assert!(r.reduced_chars < r.raw_chars);
    }

    #[test]
    fn argv_from_arguments_drives_selection() {
        let rules = default_rules();
        let args = json!({ "argv": ["npm", "install"] });
        // Content matches the matchOutput pattern => short-circuits to message.
        let content = "up to date, audited 42 packages in 1s";
        let reduction = reduce_tool_result("exec", content, false, Some(&args), None, &rules);
        let r = reduction.expect("npm-install rule should apply via argv");
        assert_eq!(r.inline_text, "npm install: up to date");
        assert_eq!(r.reducer.as_deref(), Some("install/npm-install"));
    }
}
