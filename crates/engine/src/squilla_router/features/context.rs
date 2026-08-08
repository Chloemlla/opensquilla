//! Session/tool context channel (10-dim).
//!
//! Port of `features.py::extract_context_features` (`runtime_src/src/router/
//! features.py`): clamped numeric counters, boolean flags, and two derived
//! conversation-state signals.

use super::{CTX_DIMS, ContextMetadata};

/// Clamp bounds for the normalized numerics (mirrors the Python min/max caps).
const TURN_INDEX_MAX: u32 = 20;
const CONTEXT_TOKENS_MAX: u32 = 20_000;
const N_TOOLS_MAX: u32 = 5;
const TOOL_RESULT_LEN_MAX: u32 = 5_000;
/// Turn-index threshold for `is_deep_conversation`.
const DEEP_TURN_THRESHOLD: u32 = 4;
/// Estimated-token threshold for `is_heavy_context`.
const HEAVY_CTX_THRESHOLD: u32 = 2_000;

fn btof(b: bool) -> f64 {
    if b { 1.0 } else { 0.0 }
}

/// Port of `features.py::extract_context_features`.
///
/// The caller owns the `Option`, so this always computes from `ctx`; the
/// Python all-zeros fallback for `None` is represented by
/// `ContextMetadata::default()`. The numerics are already unsigned in Rust, so
/// the Python `max(x, 0)` lower clamp is a type-level guarantee.
pub fn extract_context_features(ctx: &ContextMetadata) -> [f64; CTX_DIMS] {
    [
        ctx.turn_index.min(TURN_INDEX_MAX) as f64 / 20.0,
        ctx.context_tokens_est.min(CONTEXT_TOKENS_MAX) as f64 / 20_000.0,
        ctx.n_tools.min(N_TOOLS_MAX) as f64 / 5.0,
        ctx.tool_result_length.min(TOOL_RESULT_LEN_MAX) as f64 / 5_000.0,
        btof(ctx.has_code_block),
        btof(ctx.has_file_reference),
        btof(ctx.has_url),
        btof(ctx.has_tool_results),
        btof(ctx.turn_index >= DEEP_TURN_THRESHOLD),
        btof(ctx.context_tokens_est > HEAVY_CTX_THRESHOLD),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layout_is_all_zeros() {
        let out = extract_context_features(&ContextMetadata::default());
        assert_eq!(out, [0.0; CTX_DIMS]);
    }

    #[test]
    fn numerics_clamp_to_upper_bounds() {
        let c = ContextMetadata {
            turn_index: 40,
            context_tokens_est: 40_000,
            n_tools: 10,
            tool_result_length: 10_000,
            ..Default::default()
        };
        let out = extract_context_features(&c);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn numerics_normalize_mid_values() {
        let c = ContextMetadata {
            turn_index: 10,
            context_tokens_est: 10_000,
            n_tools: 2,
            tool_result_length: 2_500,
            ..Default::default()
        };
        let out = extract_context_features(&c);
        assert_eq!(out[0], 0.5);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 0.4);
        assert_eq!(out[3], 0.5);
    }

    #[test]
    fn boolean_flags_emit_zero_one() {
        let c = ContextMetadata {
            has_code_block: true,
            has_file_reference: true,
            has_url: true,
            has_tool_results: true,
            ..Default::default()
        };
        let out = extract_context_features(&c);
        assert_eq!(out[4], 1.0);
        assert_eq!(out[5], 1.0);
        assert_eq!(out[6], 1.0);
        assert_eq!(out[7], 1.0);
    }

    #[test]
    fn derived_signals_hit_thresholds() {
        let deep = ContextMetadata {
            turn_index: 4,
            ..Default::default()
        };
        assert_eq!(extract_context_features(&deep)[8], 1.0);
        let not_deep = ContextMetadata {
            turn_index: 3,
            ..Default::default()
        };
        assert_eq!(extract_context_features(&not_deep)[8], 0.0);

        let heavy = ContextMetadata {
            context_tokens_est: 2_001,
            ..Default::default()
        };
        assert_eq!(extract_context_features(&heavy)[9], 1.0);
        let boundary = ContextMetadata {
            context_tokens_est: 2_000,
            ..Default::default()
        };
        assert_eq!(extract_context_features(&boundary)[9], 0.0);
    }
}
