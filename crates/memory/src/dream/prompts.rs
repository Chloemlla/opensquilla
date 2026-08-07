//! Dream provider prompts and constrained patch parsing.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/prompts.py`. The
//! provider prompt asks the LLM to return a constrained JSON `operations`
//! array; parsing validates the operations against the ranked candidates.

use crate::dream::models::{PromotionCandidate, PromotionPatch};

/// Build the LLM prompt that asks for a MEMORY.md promotion patch.
///
/// TODO(parity): implement the current-MEMORY.md block + ranked candidate
/// listing + allowed-operations instructions from prompts.py.
pub fn promotion_patch_prompt(
    current_memory_md: &str,
    candidates: &[PromotionCandidate],
) -> String {
    let _ = (current_memory_md, candidates);
    String::new()
}

/// Parse an LLM JSON response into a constrained [`PromotionPatch`].
///
/// TODO(parity): implement JSON extraction, op filtering and the `["auto"]`
/// candidate-id expansion from prompts.py.
pub fn parse_promotion_patch(
    text: &str,
    candidates: &[PromotionCandidate],
) -> std::result::Result<PromotionPatch, String> {
    let _ = (text, candidates);
    Err("TODO(parity): parse_promotion_patch not implemented".to_string())
}
