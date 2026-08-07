//! Model-family reasoning format hints.
//!
//! Mirrors the Python backend's `engine/reasoning_hint.py`. Given a resolved
//! model id, these helpers identify whether the model belongs to a known
//! reasoning-capable family and, if so, what prompt hint to surface about the
//! `<think>` / `<final>` tag convention.

/// The prompt hint returned for reasoning-capable models.
pub const REASONING_HINT: &str = "For reasoning-capable models, keep private reasoning inside <think>...</think> when needed and put the user-visible answer inside <final>...</final>.";

/// Substrings that mark a resolved model id as reasoning-capable.
pub const REASONING_MODEL_MARKERS: &[&str] =
    &["gpt-5", "codex", "glm-4.7", "glm-4.6", "deepseek-r1"];

/// Return the reasoning-capable model family for a resolved model id, or `None`.
///
/// Mirrors `engine/reasoning_hint.model_family`. The returned value is the
/// marker substring that matched (e.g. `"deepseek-r1"`).
pub fn model_family(resolved_model: &str) -> Option<String> {
    let normalized = resolved_model.trim().to_lowercase();
    if normalized.is_empty() {
        return None;
    }
    for marker in REASONING_MODEL_MARKERS {
        if normalized.contains(marker) {
            return Some((*marker).to_string());
        }
    }
    None
}

/// Return the prompt hint for reasoning-capable models, or `None`.
///
/// Mirrors `engine/reasoning_hint.reasoning_tag_hint`.
pub fn reasoning_tag_hint(resolved_model: &str) -> Option<&'static str> {
    if model_family(resolved_model).is_some() {
        Some(REASONING_HINT)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_family_detected() {
        assert_eq!(
            model_family("deepseek-r1-0528"),
            Some("deepseek-r1".to_string())
        );
        assert_eq!(model_family("openai/gpt-5"), Some("gpt-5".to_string()));
        assert_eq!(model_family("GLM-4.6"), Some("glm-4.6".to_string()));
    }

    #[test]
    fn test_unknown_family_returns_none() {
        assert_eq!(model_family("gpt-4o"), None);
        assert_eq!(model_family("claude-3-5-sonnet"), None);
    }

    #[test]
    fn test_empty_model_returns_none() {
        assert_eq!(model_family(""), None);
        assert_eq!(model_family("   "), None);
    }

    #[test]
    fn test_hint_only_for_reasoning_family() {
        assert_eq!(reasoning_tag_hint("codex-1"), Some(REASONING_HINT));
        assert_eq!(reasoning_tag_hint("gpt-4o"), None);
    }

    #[test]
    fn test_case_insensitive_matching() {
        assert_eq!(
            model_family("DEEPSEEK-R1-TURBO"),
            Some("deepseek-r1".to_string())
        );
    }
}
