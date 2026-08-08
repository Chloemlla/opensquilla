//! Rule-based routing flags (5 booleans) for the Phase 3 router.
//!
//! Port of `runtime_src/src/router/flags.py::compute_flags`: keyword matching
//! and regex-pattern detection, all driven by `config::FlagRules`. The
//! `long_context` flag can additionally be triggered by heavy accumulated
//! context (mirrors the `context` metadata enhancement in the Python).

use std::sync::LazyLock;

use regex::Regex;

use crate::squilla_router::config::FlagRules;
use crate::squilla_router::features::ContextMetadata;

/// The five routing flags computed from a turn's text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    pub high_risk: bool,
    pub long_context: bool,
    pub debug: bool,
    pub repo_arch: bool,
    pub strict_format: bool,
}

static CODE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```[\s\S]*?```").expect("static code-block regex"));
static LOG_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}.*\n){3,}|(^\[?(INFO|WARN|ERROR|DEBUG)\]?\s.*\n){3,}",
    )
    .expect("static log-block regex")
});
static FILE_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)(?:^|[\s"'`(])([a-zA-Z_][\w.-]*/[\w./-]+\.[\w]+)"#)
        .expect("static file-path regex")
});

/// Whether any keyword appears, case-insensitively, as a substring of `text`.
/// Mirrors `flags.py::_has_keyword`.
fn has_keyword(text: &str, keywords: &[String]) -> bool {
    let text_lower = text.to_lowercase();
    keywords
        .iter()
        .any(|kw| text_lower.contains(&kw.to_lowercase()))
}

/// Whether any pattern matches anywhere in `text` (`flags.py::_has_pattern`).
/// Patterns that fail to compile are skipped, so a malformed config regex does
/// not break routing.
fn has_pattern(text: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|p| Regex::new(p).is_ok_and(|re| re.is_match(text)))
}

/// Total code-point length of all fenced code blocks in `text`.
fn code_block_total_len(text: &str) -> usize {
    CODE_BLOCK_RE
        .find_iter(text)
        .map(|m| m.as_str().chars().count())
        .sum()
}

/// Total code-point length of all timestamped / level-prefixed log runs.
fn log_block_total_len(text: &str) -> usize {
    LOG_BLOCK_RE
        .find_iter(text)
        .map(|m| m.as_str().chars().count())
        .sum()
}

/// Count of file-path references (captured groups) in `text`.
fn file_ref_count(text: &str) -> usize {
    FILE_PATH_RE
        .captures_iter(text)
        .filter_map(|c| c.get(1))
        .count()
}

/// Compute the five routing flags from raw text (port of
/// `flags.py::compute_flags`). `heavy_context_tokens` mirrors
/// `context_rules.heavy_context_tokens`; when the accumulated context estimate
/// exceeds it, `long_context` is forced on even if the current text is short.
pub fn compute_flags(
    text: &str,
    rules: &FlagRules,
    context: Option<&ContextMetadata>,
    heavy_context_tokens: usize,
) -> Flags {
    let high_risk = has_keyword(text, &rules.high_risk.keywords_zh)
        || has_keyword(text, &rules.high_risk.keywords_en);

    let debug =
        has_keyword(text, &rules.debug.keywords) || has_pattern(text, &rules.debug.patterns);

    let repo_arch = has_keyword(text, &rules.repo_arch.keywords);

    let strict_format = has_keyword(text, &rules.strict_format.keywords);

    let lc = &rules.long_context;
    let mut long_context = text.chars().count() >= lc.char_threshold
        || code_block_total_len(text) >= lc.code_block_threshold
        || log_block_total_len(text) >= lc.log_block_threshold
        || file_ref_count(text) >= lc.file_ref_threshold;

    if let Some(ctx) = context {
        if ctx.context_tokens_est as usize > heavy_context_tokens {
            long_context = true;
        }
    }

    Flags {
        high_risk,
        long_context,
        debug,
        repo_arch,
        strict_format,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::squilla_router::config::{
        DebugRules, HighRiskRules, KeywordRules, LongContextRules,
    };

    fn sample_rules() -> FlagRules {
        FlagRules {
            high_risk: HighRiskRules {
                keywords_zh: vec!["生产".into(), "部署".into()],
                keywords_en: vec!["deploy".into(), "rollback".into()],
            },
            debug: DebugRules {
                keywords: vec!["error".into(), "bug".into()],
                patterns: vec![r"Traceback \(most recent".into(), "stderr:".into()],
            },
            repo_arch: KeywordRules {
                keywords: vec!["architecture".into()],
            },
            strict_format: KeywordRules {
                keywords: vec!["JSON".into(), "只返回".into()],
            },
            long_context: LongContextRules {
                char_threshold: 100,
                code_block_threshold: 40,
                log_block_threshold: 40,
                file_ref_threshold: 2,
            },
        }
    }

    #[test]
    fn no_keywords_produces_all_false() {
        let flags = compute_flags("hello world", &sample_rules(), None, 2000);
        assert_eq!(flags, Flags::default());
    }

    #[test]
    fn high_risk_keywords_match_zh_and_en() {
        let rules = sample_rules();
        assert!(compute_flags("请部署到生产环境", &rules, None, 2000).high_risk);
        assert!(compute_flags("please deploy and rollback", &rules, None, 2000).high_risk);
        assert!(!compute_flags("just a question", &rules, None, 2000).high_risk);
    }

    #[test]
    fn keyword_matching_is_case_insensitive() {
        let rules = sample_rules();
        assert!(compute_flags("Please DEPLOY now", &rules, None, 2000).high_risk);
        assert!(compute_flags("there is a BUG", &rules, None, 2000).debug);
    }

    #[test]
    fn debug_flag_via_pattern_search() {
        let rules = sample_rules();
        assert!(compute_flags("Traceback (most recent call last):", &rules, None, 2000).debug);
        assert!(compute_flags("stderr: boom", &rules, None, 2000).debug);
    }

    #[test]
    fn repo_arch_and_strict_format_flags() {
        let rules = sample_rules();
        assert!(compute_flags("the new architecture", &rules, None, 2000).repo_arch);
        assert!(compute_flags("请只返回 JSON", &rules, None, 2000).strict_format);
    }

    #[test]
    fn long_context_via_char_threshold() {
        let rules = sample_rules();
        let text = "x".repeat(120);
        assert!(compute_flags(&text, &rules, None, 2000).long_context);
        assert!(!compute_flags("short", &rules, None, 2000).long_context);
    }

    #[test]
    fn long_context_via_code_block_length() {
        let rules = sample_rules();
        let block = format!("```\n{}\n```", "y".repeat(60));
        assert!(compute_flags(&block, &rules, None, 2000).long_context);
    }

    #[test]
    fn long_context_via_log_block_length() {
        let rules = sample_rules();
        let log =
            "2026-08-08 10:00:00 first\n2026-08-08 10:00:01 second\n2026-08-08 10:00:02 third\n";
        assert!(compute_flags(log, &rules, None, 2000).long_context);
    }

    #[test]
    fn long_context_via_file_ref_count() {
        let rules = sample_rules();
        let text = "see src/main.rs and src/lib.rs for details";
        assert!(compute_flags(text, &rules, None, 2000).long_context);
        assert!(!compute_flags("only src/main.rs", &rules, None, 2000).long_context);
    }

    #[test]
    fn long_context_enhanced_by_heavy_context() {
        let rules = sample_rules();
        let ctx = ContextMetadata {
            context_tokens_est: 5_000,
            ..Default::default()
        };
        assert!(compute_flags("short", &rules, Some(&ctx), 2000).long_context);

        let within_budget = ContextMetadata {
            context_tokens_est: 1_000,
            ..Default::default()
        };
        assert!(!compute_flags("short", &rules, Some(&within_budget), 2000).long_context);
    }
}
