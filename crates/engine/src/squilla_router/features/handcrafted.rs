//! Hand-crafted text features for the Phase 3 router.
//!
//! Port of `runtime_src/src/router/features.py::extract_handcrafted` plus its
//! helpers `_char_type_ratios` and `_keyword_count`.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use super::HC_DIMS;

const DEBUG_KW: &[&str] = &[
    "error",
    "bug",
    "exception",
    "traceback",
    "failed",
    "root cause",
    "报错",
    "根因",
    "修复",
    "stack trace",
    "debug",
];
const RESEARCH_KW: &[&str] = &[
    "调研",
    "research",
    "对比",
    "compare",
    "survey",
    "分析报告",
    "competitive analysis",
    "综述",
];
const ARCH_KW: &[&str] = &[
    "architecture",
    "架构",
    "重构",
    "refactor",
    "monorepo",
    "codebase",
    "module",
    "dependency",
];
const COMPARE_KW: &[&str] = &["对比", "compare", "audit", "审计", "review", "评估"];
const PLANNING_KW: &[&str] = &[
    "plan",
    "规划",
    "roadmap",
    "设计方案",
    "workflow",
    "pipeline",
    "步骤",
    "step by step",
];
const STRICT_FMT_KW: &[&str] = &[
    "JSON",
    "YAML",
    "CSV",
    "schema",
    "只返回",
    "不要解释",
    "按格式",
    "only return",
    "no explanation",
];
const HIGH_RISK_KW: &[&str] = &[
    "deploy",
    "rollback",
    "migration",
    "delete",
    "overwrite",
    "production",
    "生产",
    "部署",
    "删除",
    "客户",
    "法务",
    "财务",
];
const PRODUCTION_KW: &[&str] = &["production", "生产", "prod", "线上", "正式环境"];
const CUSTOMER_KW: &[&str] = &["customer", "客户", "用户邮件", "client"];
const DELETE_KW: &[&str] = &[
    "delete",
    "remove",
    "drop",
    "truncate",
    "删除",
    "清空",
    "覆盖",
    "overwrite",
];
const FORMAL_KW: &[&str] = &["formal", "正式", "official", "公文", "合同", "法律"];
const CONSTRAINT_KW: &[&str] = &[
    "必须",
    "不能",
    "不要",
    "只能",
    "must",
    "shall",
    "required",
    "forbidden",
    "不允许",
    "至少",
    "最多",
];
const TEACHING_KW: &[&str] = &[
    "how does",
    "explain",
    "what is",
    "why does",
    "how to",
    "教我",
    "解释",
    "为什么",
    "怎么",
    "是什么",
    "how can",
    "tell me about",
    "walk me through",
    "介绍",
    "说明",
];
const IMPLEMENT_KW: &[&str] = &[
    "implement",
    "write function",
    "write a",
    "create a",
    "写个",
    "实现",
    "用法",
    "帮我写",
    "生成代码",
    "add a",
    "build a",
    "make a",
    "写一个",
    "编写",
];

static CODE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```[\s\S]*?```").expect("static code-block regex"));
static JSON_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"\{[\s\S]*?["'][\w]+["']\s*:"#).expect("static JSON regex"));
static YAML_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[\w_]+:\s+\S").expect("static YAML regex"));
static CSV_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[^,\n]+,[^,\n]+,[^,\n]+").expect("static CSV regex"));
static TABLE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\|.*\|.*\|").expect("static table regex"));
static FILE_PATH_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?m)(?:^|[\s"'`(])([a-zA-Z_][\w.-]*/[\w./-]+\.[\w]+)"#)
        .expect("static file-path regex")
});
static URL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+").expect("static URL regex"));
static LOG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)(\d{4}[-/]\d{2}[-/]\d{2}[\sT]\d{2}:\d{2}.*\n){3,}|(^\[?(INFO|WARN|ERROR|DEBUG)\]?\s.*\n){3,}",
    )
    .expect("static log regex")
});
static SHELL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\$\s+\w|^>\s+\w|```(?:bash|sh|shell)").expect("static shell regex")
});
static TRACEBACK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"Traceback \(most recent|stderr:|\.py", line \d+"#)
        .expect("static traceback regex")
});
static BULLET_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[\s]*[-*]\s").expect("static bullet regex"));
static NUMBERED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^[\s]*\d+[.)]\s").expect("static numbered regex"));
static QUOTED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["'`](.*?)["'`]"#).expect("static quoted regex"));

/// Ratio of Chinese characters, ASCII alphabetic characters, and
/// code/punctuation characters in `text`.
///
/// Each count is divided by the total character count (characters, not bytes)
/// and all three ratios are `0.0` for empty input. Mirrors
/// `features.py::_char_type_ratios`.
pub fn char_type_ratios(text: &str) -> (f64, f64, f64) {
    if text.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = text.chars().count();
    let zh = text
        .chars()
        .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        .count();
    let en = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let code = text
        .chars()
        .filter(|c| "{}[]();=<>|&!@#$%^*~`\\".contains(*c))
        .count();
    (
        zh as f64 / n as f64,
        en as f64 / n as f64,
        code as f64 / n as f64,
    )
}

/// Count how many of `keywords` appear, case-insensitively, as a substring of
/// `text`. Mirrors `features.py::_keyword_count`.
pub fn keyword_count(text: &str, keywords: &[&str]) -> u32 {
    let text_lower = text.to_lowercase();
    keywords
        .iter()
        .filter(|kw| text_lower.contains(&kw.to_lowercase()))
        .count() as u32
}

/// Extract the 51-dimensional hand-crafted feature vector from `text`.
///
/// Ported index-for-index from `features.py::extract_handcrafted`: basic
/// length/word/line stats, language ratios, structural regex signals,
/// punctuation counts, keyword signals, risk signals, file/tool signals,
/// intensity, and the R1-specific teaching/implementation signals.
pub fn extract_handcrafted(text: &str) -> [f64; HC_DIMS] {
    let mut feats = [0.0_f64; HC_DIMS];

    // Basic (0-3).
    feats[0] = text.chars().count() as f64;
    let words: Vec<&str> = text.split_whitespace().collect();
    feats[1] = words.len() as f64;
    feats[2] = text.split('\n').count() as f64;
    feats[3] = feats[0] / feats[2].max(1.0);

    // Language (4-7).
    let (zh, en, code) = char_type_ratios(text);
    feats[4] = zh;
    feats[5] = en;
    feats[6] = code;
    feats[7] = if zh > 0.1 && en > 0.1 { 1.0 } else { 0.0 };

    // Structure (8-14).
    let code_blocks: Vec<&str> = CODE_BLOCK_RE.find_iter(text).map(|m| m.as_str()).collect();
    feats[8] = if code_blocks.is_empty() { 0.0 } else { 1.0 };
    feats[9] = code_blocks.len() as f64;
    feats[10] = code_blocks.iter().map(|b| b.chars().count() as f64).sum();
    feats[11] = if JSON_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[12] = if YAML_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[13] = if CSV_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[14] = if TABLE_RE.is_match(text) { 1.0 } else { 0.0 };

    // Punctuation (15-18).
    feats[15] = (text.matches('?').count() + text.matches('\u{ff1f}').count()) as f64;
    feats[16] = (text.matches('!').count() + text.matches('\u{ff01}').count()) as f64;
    feats[17] = BULLET_RE.find_iter(text).count() as f64;
    feats[18] = NUMBERED_RE.find_iter(text).count() as f64;

    // Keyword signals (22-27).
    feats[22] = keyword_count(text, DEBUG_KW) as f64;
    feats[23] = keyword_count(text, RESEARCH_KW) as f64;
    feats[24] = keyword_count(text, ARCH_KW) as f64;
    feats[25] = keyword_count(text, COMPARE_KW) as f64;
    feats[26] = keyword_count(text, PLANNING_KW) as f64;
    feats[27] = keyword_count(text, STRICT_FMT_KW) as f64;

    // Risk (28-32).
    feats[28] = keyword_count(text, HIGH_RISK_KW) as f64;
    feats[29] = keyword_count(text, PRODUCTION_KW) as f64;
    feats[30] = keyword_count(text, CUSTOMER_KW) as f64;
    feats[31] = keyword_count(text, DELETE_KW) as f64;
    feats[32] = keyword_count(text, FORMAL_KW) as f64;

    // File/tool (33-37).
    feats[33] = if FILE_PATH_RE.is_match(text) {
        1.0
    } else {
        0.0
    };
    feats[34] = if URL_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[35] = if LOG_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[36] = if SHELL_RE.is_match(text) { 1.0 } else { 0.0 };
    feats[37] = if TRACEBACK_RE.is_match(text) {
        1.0
    } else {
        0.0
    };

    // Intensity (38-40).
    feats[38] = keyword_count(text, CONSTRAINT_KW) as f64;
    let quoted_len: usize = QUOTED_RE
        .captures_iter(text)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str().chars().count())
        .sum();
    feats[39] = quoted_len as f64 / text.chars().count().max(1) as f64;
    let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
    feats[40] = unique_words.len() as f64 / words.len().max(1) as f64;

    // R1-specific signals (41-50).
    feats[41] = keyword_count(text, TEACHING_KW) as f64;
    feats[42] = keyword_count(text, IMPLEMENT_KW) as f64;
    let unique_file_refs: HashSet<&str> = FILE_PATH_RE
        .captures_iter(text)
        .filter_map(|c| c.get(1))
        .map(|m| m.as_str())
        .collect();
    let n_files = unique_file_refs.len();
    feats[43] = if n_files == 0 { 1.0 } else { 0.0 };
    feats[44] = if (1..=2).contains(&n_files) { 1.0 } else { 0.0 };
    feats[45] = if n_files >= 3 { 1.0 } else { 0.0 };
    let has_debug = keyword_count(text, DEBUG_KW) > 0;
    feats[46] = if !code_blocks.is_empty() && !has_debug {
        1.0
    } else {
        0.0
    };
    let text_len = text.chars().count();
    feats[47] = if text_len < 200 { 1.0 } else { 0.0 };
    feats[48] = if (200..=1000).contains(&text_len) {
        1.0
    } else {
        0.0
    };
    feats[49] = if text_len > 1000 { 1.0 } else { 0.0 };
    let total_kw =
        feats[22] + feats[23] + feats[24] + feats[25] + feats[26] + feats[27] + feats[28];
    feats[50] = if total_kw < 2.0 { 1.0 } else { 0.0 };

    feats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_type_ratios_mixed_text() {
        let (zh, en, code) = char_type_ratios("hello世界 fn()");
        assert!(zh > 0.0 && en > 0.0 && code > 0.0);
        assert!((zh + en + code - 1.0).abs() < 1e-9);
    }

    #[test]
    fn char_type_ratios_empty_is_zero() {
        assert_eq!(char_type_ratios(""), (0.0, 0.0, 0.0));
    }

    #[test]
    fn keyword_count_is_substring_case_insensitive() {
        let text = "Please explain the ERROR and help me debug";
        assert_eq!(keyword_count(text, DEBUG_KW), 2);
        assert_eq!(keyword_count("no keywords here", DEBUG_KW), 0);
    }

    #[test]
    fn empty_text_has_default_layout() {
        let feats = extract_handcrafted("");
        assert_eq!(feats[0], 0.0);
        assert_eq!(feats[1], 0.0);
        assert_eq!(feats[2], 1.0); // "".split('\n') yields one element
        assert_eq!(feats[3], 0.0);
        assert_eq!(feats[43], 1.0); // no file references
        assert_eq!(feats[47], 1.0); // length < 200
        assert_eq!(feats[50], 1.0); // total keyword signals < 2
        for i in (0..HC_DIMS).filter(|&i| i != 2 && i != 43 && i != 47 && i != 50) {
            assert_eq!(feats[i], 0.0, "feats[{i}] should be 0");
        }
    }

    #[test]
    fn code_block_structure_features() {
        let text = "```python\nprint('hi')\n```\n\n```sql\nSELECT 1\n```";
        let feats = extract_handcrafted(text);
        assert_eq!(feats[8], 1.0); // has a code block
        assert_eq!(feats[9], 2.0); // two code blocks
        assert!(feats[10] > 0.0); // total block length
        assert_eq!(feats[46], 1.0); // code without debug keywords
    }

    #[test]
    fn quoted_ratio_and_unique_words() {
        let text = "say \"hello world\" and 'goodbye'";
        let feats = extract_handcrafted(text);
        // Inner quoted lengths are 11 ("hello world") and 7 ("goodbye").
        assert!((feats[39] - 18.0 / 31.0).abs() < 1e-9);
        // All five words are distinct after lowercasing.
        assert!((feats[40] - 1.0).abs() < 1e-9);
    }
}
