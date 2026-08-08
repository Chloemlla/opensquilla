//! Squilla router post-processing controllers.
//!
//! Pure functions ported from `src/opensquilla/squilla_router/controller.py`:
//! derive the thinking mode (T0-T3) and prompt policy (P0-P2) from a
//! synthetic 4-class probability vector plus turn flags, localize prompt
//! hints, and normalize contradictory decisions. No I/O and no model runtime
//! dependency.

use std::collections::HashMap;

/// The canonical text tier ladder, lowest to highest (`router_tiers.TEXT_TIERS`).
pub const TIER_ORDER: [&str; 4] = ["c0", "c1", "c2", "c3"];

/// Synthetic peak probability used for fallback one-hot vectors.
pub const SYNTHETIC_PEAK: f64 = 0.85;

/// Per-class difficulty weights (`DIFFICULTY_WEIGHTS`).
pub const DIFFICULTY_WEIGHTS: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

/// Flags that push the thinking mode to T3 (`_DEEP_FLAGS`).
pub const DEEP_FLAGS: [&str; 3] = ["high_risk", "debug", "long_context"];

/// Flags that force a full prompt (`_FULL_PROMPT_FLAGS`).
pub const FULL_PROMPT_FLAGS: [&str; 4] = ["high_risk", "long_context", "debug", "strict_format"];

/// Flags that block prompt compression (`_COMPRESS_BLOCK_FLAGS`).
pub const COMPRESS_BLOCK_FLAGS: [&str; 3] = ["high_risk", "strict_format", "debug"];

/// Localized P0 prompt hints (`_PROMPT_HINTS["P0"]`).
pub const P0_HINT_ZH: &str = "直接作答，缩短思考长度，避免无关展开。";
pub const P0_HINT_EN: &str = "Answer directly, keep thinking short, avoid irrelevant expansion.";

const DEFAULT_T3_MIN_IDX: usize = 2;
const DEFAULT_T0_MAX_IDX: usize = 0;
const DEFAULT_T0_MIN_MARGIN: f64 = 0.5;
const DEFAULT_T1_MAX_IDX: usize = 1;
const DEFAULT_T1_MIN_MARGIN: f64 = 0.4;
const DEFAULT_MAX_DIFFICULTY: f64 = 0.8;
const DEFAULT_MIN_MARGIN: f64 = 0.4;

/// A localized prompt hint (mirrors a `{"hint_zh": ..., "hint_en": ...}` dict).
#[derive(Debug, Clone, Copy)]
pub struct PromptHint {
    pub zh: &'static str,
    pub en: &'static str,
}

const P0_HINT: PromptHint = PromptHint {
    zh: P0_HINT_ZH,
    en: P0_HINT_EN,
};

/// Lookup abstraction for the Python controller's `flags: dict | object | None`.
pub trait FlagLookup {
    fn flag_is_set(&self, name: &str) -> bool;
}

impl FlagLookup for HashMap<String, bool> {
    fn flag_is_set(&self, name: &str) -> bool {
        self.get(name).copied().unwrap_or(false)
    }
}

/// True when any of `flag_names` is set in `flags` (`_has_any_flag`).
pub fn has_any_flag(flags: Option<&dyn FlagLookup>, flag_names: &[&str]) -> bool {
    let Some(flags) = flags else {
        return false;
    };
    for name in flag_names {
        if flags.flag_is_set(name) {
            return true;
        }
    }
    false
}

/// Return a synthetic 4-class probability vector peaking on `tier`.
pub fn synthetic_one_hot(tier: &str) -> Vec<f64> {
    synthetic_one_hot_with_peak(tier, SYNTHETIC_PEAK)
}

/// `synthetic_one_hot` with an explicit peak probability.
pub fn synthetic_one_hot_with_peak(tier: &str, dominant: f64) -> Vec<f64> {
    let n = TIER_ORDER.len();
    let residual = (1.0 - dominant) / (n.saturating_sub(1).max(1) as f64);
    let idx = TIER_ORDER.iter().position(|t| *t == tier).unwrap_or(1);
    let mut probs = vec![residual; n];
    probs[idx] = dominant;
    probs
}

/// Weighted difficulty score of a probability vector (`compute_difficulty`).
pub fn compute_difficulty(probs: &[f64]) -> f64 {
    probs
        .iter()
        .zip(DIFFICULTY_WEIGHTS)
        .map(|(p, w)| w * p)
        .sum()
}

/// Margin between the top two classes (`compute_margin`).
pub fn compute_margin(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return probs.first().copied().unwrap_or(0.0);
    }
    let mut ordered: Vec<f64> = probs.to_vec();
    ordered.sort_by(|a, b| b.total_cmp(a));
    (ordered[0] - ordered[1]).max(0.0)
}

/// Index of the most probable class, keeping the leftmost on ties (Python's
/// `max(range(n), key=...)` picks the first maximum).
fn top1_index(probs: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, p) in probs.iter().enumerate() {
        if *p > probs[best] {
            best = i;
        }
    }
    best
}

/// Derive the thinking mode (T0-T3) using default thresholds.
pub fn derive_thinking_mode(probs: &[f64], flags: Option<&dyn FlagLookup>) -> String {
    derive_thinking_mode_with(
        probs,
        flags,
        DEFAULT_T3_MIN_IDX,
        DEFAULT_T0_MAX_IDX,
        DEFAULT_T0_MIN_MARGIN,
        DEFAULT_T1_MAX_IDX,
        DEFAULT_T1_MIN_MARGIN,
    )
}

/// `derive_thinking_mode` with explicit thresholds.
pub fn derive_thinking_mode_with(
    probs: &[f64],
    flags: Option<&dyn FlagLookup>,
    t3_min_idx: usize,
    t0_max_idx: usize,
    t0_min_margin: f64,
    t1_max_idx: usize,
    t1_min_margin: f64,
) -> String {
    let top1_idx = top1_index(probs);
    let margin = compute_margin(probs);
    if top1_idx >= TIER_ORDER.len() - 1 {
        return "T3".to_string();
    }
    if top1_idx >= t3_min_idx && has_any_flag(flags, &DEEP_FLAGS) {
        return "T3".to_string();
    }
    if top1_idx <= t0_max_idx && margin >= t0_min_margin {
        return "T0".to_string();
    }
    if top1_idx <= t1_max_idx && margin >= t1_min_margin {
        return "T1".to_string();
    }
    "T2".to_string()
}

/// Derive the prompt policy (P0-P2) using default thresholds.
pub fn derive_prompt_policy(probs: &[f64], flags: Option<&dyn FlagLookup>) -> String {
    derive_prompt_policy_with(probs, flags, DEFAULT_MAX_DIFFICULTY, DEFAULT_MIN_MARGIN)
}

/// `derive_prompt_policy` with explicit thresholds.
pub fn derive_prompt_policy_with(
    probs: &[f64],
    flags: Option<&dyn FlagLookup>,
    max_difficulty: f64,
    min_margin: f64,
) -> String {
    if has_any_flag(flags, &FULL_PROMPT_FLAGS) {
        return "P2".to_string();
    }
    let difficulty = compute_difficulty(probs);
    let margin = compute_margin(probs);
    if difficulty <= max_difficulty
        && margin >= min_margin
        && !has_any_flag(flags, &COMPRESS_BLOCK_FLAGS)
    {
        return "P0".to_string();
    }
    "P1".to_string()
}

/// Forbid the contradictory THINK_DEEP + P0 (compress) combination.
///
/// Python semantics: T2/T3 combined with P0 is normalized to P1; any other
/// combination is returned unchanged. (The pre-port Rust variant forced
/// T0 -> P0, which does not match `controller.normalize_decisions`.)
pub fn normalize_decisions(thinking_mode: &str, prompt_policy: &str) -> (String, String) {
    if matches!(thinking_mode, "T2" | "T3") && prompt_policy == "P0" {
        return (thinking_mode.to_string(), "P1".to_string());
    }
    (thinking_mode.to_string(), prompt_policy.to_string())
}

/// Map a thinking mode to its reasoning-effort level (`_THINKING_MODE_LEVEL`).
pub fn thinking_mode_to_level(mode: Option<&str>) -> Option<&'static str> {
    match mode? {
        "T1" => Some("low"),
        "T2" => Some("medium"),
        "T3" => Some("high"),
        _ => None,
    }
}

/// True when a char falls in any CJK range (`_CJK_RANGES`).
fn is_cjk(ch: char) -> bool {
    let code = ch as u32;
    (0x4e00..=0x9fff).contains(&code)
        || (0x3400..=0x4dbf).contains(&code)
        || (0xf900..=0xfaff).contains(&code)
}

/// Return `"zh"` when the prompt is substantially CJK, else `"en"`.
///
/// Python's `prompt_hint_locale` also tallies ASCII letters (`latin_count`)
/// but never reads them, so only the CJK count is ported.
pub fn prompt_hint_locale(text: Option<&str>) -> &'static str {
    let Some(text) = text else {
        return "en";
    };
    let cjk_count = text.chars().filter(|ch| is_cjk(*ch)).count();
    if cjk_count >= 2 { "zh" } else { "en" }
}

/// First non-empty hint, mirroring Python's `a or b or None`.
fn first_non_empty<'a>(primary: &'a str, fallback: &'a str) -> Option<&'a str> {
    if !primary.is_empty() {
        Some(primary)
    } else if !fallback.is_empty() {
        Some(fallback)
    } else {
        None
    }
}

/// The prompt hint configured for a policy id (`_PROMPT_HINTS.get(policy)`).
pub fn prompt_hint_for(policy: &str) -> Option<&'static PromptHint> {
    match policy {
        "P0" => Some(&P0_HINT),
        _ => None,
    }
}

/// Select hint_zh or hint_en by the input language.
pub fn select_localized_prompt_hint(hint: &PromptHint, text: Option<&str>) -> Option<&'static str> {
    if prompt_hint_locale(text) == "zh" {
        first_non_empty(hint.zh, hint.en)
    } else {
        first_non_empty(hint.en, hint.zh)
    }
}

/// Resolve the prompt hint for a policy, localized to `text` when provided.
pub fn get_prompt_hint(policy: Option<&str>, text: Option<&str>) -> Option<&'static str> {
    let hint = prompt_hint_for(policy?)?;
    if text.is_none() {
        return first_non_empty(hint.en, hint.zh);
    }
    select_localized_prompt_hint(hint, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(pairs: &[(&str, bool)]) -> HashMap<String, bool> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn has_any_flag_detects_present_flags() {
        assert!(!has_any_flag(None, &DEEP_FLAGS));
        let deep = flags(&[("high_risk", true)]);
        assert!(has_any_flag(Some(&deep), &DEEP_FLAGS));
        let blocked = flags(&[("debug", true)]);
        assert!(has_any_flag(Some(&blocked), &COMPRESS_BLOCK_FLAGS));
        let unset = flags(&[("long_context", false)]);
        assert!(!has_any_flag(Some(&unset), &DEEP_FLAGS));
    }

    #[test]
    fn synthetic_one_hot_peaks_on_requested_tier() {
        let probs = synthetic_one_hot("c2");
        assert_eq!(probs.len(), 4);
        assert_eq!(probs[2], SYNTHETIC_PEAK);
        assert_eq!(probs[0], (1.0 - SYNTHETIC_PEAK) / 3.0);
        assert_eq!(synthetic_one_hot("c9")[1], SYNTHETIC_PEAK);
    }

    #[test]
    fn difficulty_is_weighted_sum() {
        assert_eq!(compute_difficulty(&[0.25, 0.25, 0.25, 0.25]), 1.5);
        assert_eq!(compute_difficulty(&[1.0, 0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn margin_handles_short_and_sorted_inputs() {
        assert_eq!(compute_margin(&[]), 0.0);
        assert_eq!(compute_margin(&[0.7]), 0.7);
        assert_eq!(compute_margin(&[0.6, 0.4]), 0.2);
        assert_eq!(compute_margin(&[0.4, 0.6]), 0.2);
        assert_eq!(compute_margin(&[0.5, 0.5]), 0.0);
    }

    #[test]
    fn thinking_mode_covers_all_tiers() {
        assert_eq!(derive_thinking_mode(&[0.0, 0.1, 0.2, 0.7], None), "T3");
        let deep = flags(&[("high_risk", true)]);
        assert_eq!(
            derive_thinking_mode(&[0.1, 0.1, 0.7, 0.1], Some(&deep)),
            "T3"
        );
        assert_eq!(derive_thinking_mode(&[0.1, 0.1, 0.7, 0.1], None), "T2");
        assert_eq!(derive_thinking_mode(&[0.8, 0.2, 0.0, 0.0], None), "T0");
        assert_eq!(derive_thinking_mode(&[0.2, 0.6, 0.2, 0.0], None), "T1");
        assert_eq!(derive_thinking_mode(&[0.4, 0.4, 0.2, 0.0], None), "T2");
    }

    #[test]
    fn prompt_policy_covers_all_classes() {
        let full = flags(&[("strict_format", true)]);
        assert_eq!(
            derive_prompt_policy(&[0.7, 0.3, 0.0, 0.0], Some(&full)),
            "P2"
        );
        assert_eq!(derive_prompt_policy(&[0.7, 0.3, 0.0, 0.0], None), "P0");
        assert_eq!(derive_prompt_policy(&[0.4, 0.6, 0.0, 0.0], None), "P1");
        let blocked = flags(&[("debug", true)]);
        assert_eq!(
            derive_prompt_policy(&[0.7, 0.3, 0.0, 0.0], Some(&blocked)),
            "P2"
        );
        let neutral = flags(&[("some_other", true)]);
        assert_eq!(
            derive_prompt_policy(&[0.7, 0.3, 0.0, 0.0], Some(&neutral)),
            "P0"
        );
    }

    #[test]
    fn normalize_decisions_forbids_deep_compress() {
        assert_eq!(
            normalize_decisions("T2", "P0"),
            ("T2".to_string(), "P1".to_string())
        );
        assert_eq!(
            normalize_decisions("T3", "P0"),
            ("T3".to_string(), "P1".to_string())
        );
        assert_eq!(
            normalize_decisions("T0", "P0"),
            ("T0".to_string(), "P0".to_string())
        );
        assert_eq!(
            normalize_decisions("T1", "P1"),
            ("T1".to_string(), "P1".to_string())
        );
    }

    #[test]
    fn thinking_level_mapping() {
        assert_eq!(thinking_mode_to_level(None), None);
        assert_eq!(thinking_mode_to_level(Some("T0")), None);
        assert_eq!(thinking_mode_to_level(Some("T1")), Some("low"));
        assert_eq!(thinking_mode_to_level(Some("T2")), Some("medium"));
        assert_eq!(thinking_mode_to_level(Some("T3")), Some("high"));
        assert_eq!(thinking_mode_to_level(Some("T9")), None);
    }

    #[test]
    fn hint_locale_detects_cjk() {
        assert_eq!(prompt_hint_locale(None), "en");
        assert_eq!(prompt_hint_locale(Some("")), "en");
        assert_eq!(prompt_hint_locale(Some("answer directly")), "en");
        assert_eq!(prompt_hint_locale(Some("你")), "en");
        assert_eq!(prompt_hint_locale(Some("你好世界")), "zh");
        assert_eq!(prompt_hint_locale(Some("你a好")), "zh");
    }

    #[test]
    fn localized_hint_selection() {
        assert_eq!(
            select_localized_prompt_hint(&P0_HINT, Some("你好")),
            Some(P0_HINT_ZH)
        );
        assert_eq!(
            select_localized_prompt_hint(&P0_HINT, Some("hi")),
            Some(P0_HINT_EN)
        );
    }

    #[test]
    fn prompt_hint_resolution() {
        assert_eq!(get_prompt_hint(None, None), None);
        assert_eq!(get_prompt_hint(Some("P9"), None), None);
        assert_eq!(get_prompt_hint(Some("P0"), None), Some(P0_HINT_EN));
        assert_eq!(
            get_prompt_hint(Some("P0"), Some("你好世界")),
            Some(P0_HINT_ZH)
        );
        assert_eq!(
            get_prompt_hint(Some("P0"), Some("answer directly")),
            Some(P0_HINT_EN)
        );
    }
}
