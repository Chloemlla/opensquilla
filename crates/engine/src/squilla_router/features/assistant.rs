//! Assistant-turn handcrafted signals (12-dim channel).
//!
//! Port of `v4_features.py::extract_assistant_handcrafted`: clarification /
//! refusal / self-doubt / code / numbered-list regex signals over the previous
//! assistant turn, plus normalized usage stats.

use super::PrevAssistantUsage;
use regex::Regex;
use std::sync::LazyLock;

const ASST_HC_DIMS: usize = super::ASST_HC_DIMS;

static RE_CLAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:能否|请\s*提供|需要(?:更多|具体).{0,8}信息|could you (?:clarify|provide)|please (?:specify|provide)|clarify which)")
        .expect("valid clarification regex")
});
static RE_REFUSAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:I cannot|I can't help|对不起.{0,5}无法|抱歉.{0,5}不能|作为(?:AI|大语言模型))",
    )
    .expect("valid refusal regex")
});
static RE_SELF_DOUBT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:我不(?:确定|清楚)|可能(?:不太|不一定)|not sure|might not be|I'm not entirely)",
    )
    .expect("valid self-doubt regex")
});
static RE_CODE_INLINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`[^`]{4,}`").expect("valid inline-code regex"));
static RE_NUMBERED_LIST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*\d+[\.、]\s").expect("valid numbered-list regex"));

fn btof(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// `np.log1p(value) / divisor` (Python `_normalize_log_usage`).
pub(crate) fn normalize_log_usage(value: u64) -> f64 {
    (value as f64).ln_1p() / 10.0
}

/// Fraction of CJK (`U+4E00..=U+9FFF`) chars in `text` (Python `_zh_char_ratio`).
pub(crate) fn zh_char_ratio(text: &str) -> f64 {
    let n = text.chars().count();
    if n == 0 {
        return 0.0;
    }
    let zh = text
        .chars()
        .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
        .count();
    zh as f64 / n as f64
}

/// Port of `v4_features.py::extract_assistant_handcrafted`.
pub fn extract_assistant_handcrafted(
    prev_assistant_text: Option<&str>,
    prev_assistant_usage: Option<&PrevAssistantUsage>,
    current_user_text: &str,
) -> [f64; ASST_HC_DIMS] {
    let Some(t) = prev_assistant_text else {
        return [0.0; ASST_HC_DIMS];
    };
    let denom = current_user_text.chars().count().max(1);
    let len_ratio = (t.chars().count() as f64 / denom as f64).min(5.0) / 5.0;
    let input = prev_assistant_usage
        .map(|u| u.input_tokens)
        .unwrap_or(1)
        .max(1);
    let cached = prev_assistant_usage.map(|u| u.cached_tokens).unwrap_or(0);
    let cached_ratio = cached as f64 / input as f64;
    [
        1.0,
        btof(RE_CLAR.is_match(t)),
        btof(RE_REFUSAL.is_match(t)),
        btof(RE_SELF_DOUBT.is_match(t)),
        btof(t.contains("```") || RE_CODE_INLINE.is_match(t)),
        btof(RE_NUMBERED_LIST.is_match(t)),
        normalize_log_usage(prev_assistant_usage.map(|u| u.output_tokens).unwrap_or(0)),
        normalize_log_usage(
            prev_assistant_usage
                .map(|u| u.reasoning_tokens)
                .unwrap_or(0),
        ),
        normalize_log_usage(prev_assistant_usage.map(|u| u.duration_ms).unwrap_or(0)),
        len_ratio,
        zh_char_ratio(t),
        cached_ratio,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage() -> PrevAssistantUsage {
        PrevAssistantUsage {
            input_tokens: 100,
            output_tokens: 50,
            cached_tokens: 40,
            reasoning_tokens: 30,
            duration_ms: 2000,
        }
    }

    #[test]
    fn none_prev_asst_yields_zeros() {
        let out = extract_assistant_handcrafted(None, None, "hi");
        assert_eq!(out, [0.0; ASST_HC_DIMS]);
    }

    #[test]
    fn signals_and_usage_normalized() {
        let out = extract_assistant_handcrafted(
            Some("I'm not sure, could you clarify? ```rust\nfn f() {}\n```"),
            Some(&usage()),
            "short",
        );
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 1.0); // clarification
        assert_eq!(out[3], 1.0); // self-doubt
        assert_eq!(out[4], 1.0); // code block
        // log1p(50)/10, log1p(30)/10, log1p(2000)/10
        assert!((out[6] - 50f64.ln_1p() / 10.0).abs() < 1e-9);
        assert!((out[7] - 30f64.ln_1p() / 10.0).abs() < 1e-9);
        assert!((out[8] - 2000f64.ln_1p() / 10.0).abs() < 1e-9);
        // cached ratio 40/100
        assert!((out[11] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn zh_ratio_counts_only_cjk() {
        assert_eq!(zh_char_ratio(""), 0.0);
        assert!((zh_char_ratio("你好a") - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(zh_char_ratio("abc"), 0.0);
    }

    #[test]
    fn normalize_log_usage_is_log1p_over_ten() {
        assert_eq!(normalize_log_usage(0), 0.0);
        assert!((normalize_log_usage(10) - 11f64.ln() / 10.0).abs() < 1e-9);
    }
}
