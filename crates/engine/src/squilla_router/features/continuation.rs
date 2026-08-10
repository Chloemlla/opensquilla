//! Continuation-prompt channel (2-dim).
//!
//! Port of `v4_features.py::extract_continuation_features`: flags short
//! "please continue" cues and normalizes the previous output-token count.

use super::PrevAssistantUsage;
use super::assistant::normalize_log_usage;
use regex::Regex;
use std::sync::LazyLock;

const CONT_DIMS: usize = super::CONT_DIMS;

static RE_CONTINUATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:请继续|继续|接着|续写|展开一下|再说|more|continue|go on|carry on|next)")
        .expect("valid continuation regex")
});

fn btof(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Port of `v4_features.py::extract_continuation_features`.
pub fn extract_continuation_features(
    prev_assistant_usage: Option<&PrevAssistantUsage>,
    current_user_text: &str,
) -> [f64; CONT_DIMS] {
    let text = current_user_text.trim();
    let is_short = text.chars().count() <= 24;
    let has_cue = !text.is_empty() && is_short && RE_CONTINUATION.is_match(text);
    [
        btof(has_cue),
        normalize_log_usage(prev_assistant_usage.map(|u| u.output_tokens).unwrap_or(0)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cue_requires_short_text_and_match() {
        let out = extract_continuation_features(None, "继续");
        assert_eq!(out[0], 1.0);
        let out = extract_continuation_features(None, "请继续讨论这个非常长的主题并且详细说明具体内容需求");
        assert_eq!(out[0], 0.0); // too long (> 24 chars)
        let out = extract_continuation_features(None, "unrelated text");
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn output_tokens_log_normalized() {
        let usage = PrevAssistantUsage {
            output_tokens: 50,
            ..Default::default()
        };
        let out = extract_continuation_features(Some(&usage), "继续");
        assert!((out[1] - 50f64.ln_1p() / 10.0).abs() < 1e-9);
    }
}
