//! Reasoning-heavy prompt cues (5-dim channel).
//!
//! Port of `v4_features.py::extract_reasoning_features`: reasoning keyword
//! signal, question density, prompt-length log, and prior reasoning/duration
//! usage logs.

use super::PrevAssistantUsage;
use super::assistant::normalize_log_usage;
use regex::Regex;
use std::sync::LazyLock;

const REASONING_DIMS: usize = super::REASONING_DIMS;

static RE_REASONING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:why|compare|trade[ -]?off|analy[sz]e|architecture|reasoning|design|解释|原因|对比|分析|架构|设计|权衡)",
    )
    .expect("valid reasoning regex")
});

fn btof(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Port of `v4_features.py::extract_reasoning_features`.
pub fn extract_reasoning_features(
    prev_assistant_usage: Option<&PrevAssistantUsage>,
    current_user_text: &str,
) -> [f64; REASONING_DIMS] {
    let text = current_user_text.trim();
    let n = text.chars().count();
    let qmarks = (text.matches('?').count() + text.matches('？').count()) as f64;
    [
        btof(RE_REASONING.is_match(text)),
        (qmarks / (n.max(1) as f64) * 20.0).min(1.0),
        (n as f64).ln_1p() / 10.0,
        normalize_log_usage(
            prev_assistant_usage
                .map(|u| u.reasoning_tokens)
                .unwrap_or(0),
        ),
        normalize_log_usage(prev_assistant_usage.map(|u| u.duration_ms).unwrap_or(0)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_cue_and_question_density() {
        let out = extract_reasoning_features(None, "为什么? why? compare A与");
        assert_eq!(out[0], 1.0);
        // 2 qmarks / 20 chars * 20 = 2, clamped to 1.0
        assert_eq!(out[1], 1.0);
        assert!((out[2] - 20f64.ln_1p() / 10.0).abs() < 1e-9);
    }

    #[test]
    fn prior_reasoning_and_duration_logs() {
        let usage = PrevAssistantUsage {
            reasoning_tokens: 100,
            duration_ms: 3000,
            ..Default::default()
        };
        let out = extract_reasoning_features(Some(&usage), "analyze");
        assert!((out[3] - 100f64.ln_1p() / 10.0).abs() < 1e-9);
        assert!((out[4] - 3000f64.ln_1p() / 10.0).abs() < 1e-9);
    }
}
