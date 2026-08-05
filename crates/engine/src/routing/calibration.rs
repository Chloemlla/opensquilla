//! Deterministic, on-device router calibration.
//!
//! Mirrors the Python `engine/routing/calibration.py`. The bundled ML
//! classifier is frozen; this module aggregates local, prompt-free router
//! decision records into a small, hard-clamped adjustment to a single scalar —
//! the confidence-gate `confidence_threshold` — plus a per-tier confidence
//! bias. The routing policy reads the resulting [`CalibrationState`] as an
//! argument (never a file) and applies it as a bias in
//! [`crate::routing::confidence_gate`].
//!
//! Hard clamps (pinned):
//!
//! * `|per_class_bias[tier]| <= 0.15`
//! * the adjusted `confidence_threshold` stays within `[0.3, 0.7]`
//! * the stored threshold adjustment stays within `[-0.20, 0.20]`

use crate::routing::{TEXT_TIERS, normalize_text_tier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Schema version written into the calibration file.
pub const SCHEMA_VERSION: u32 = 1;
/// The default calibration file name.
pub const CALIBRATION_FILENAME: &str = "router_calibration.json";

/// Per-class bias clamp (pinned).
pub const BIAS_CLAMP: f64 = 0.15;
/// Effective threshold floor (pinned).
pub const THRESHOLD_FLOOR: f64 = 0.3;
/// Effective threshold ceiling (pinned).
pub const THRESHOLD_CEIL: f64 = 0.7;
/// Stored threshold-adjustment clamp (pinned).
pub const THRESHOLD_ADJUST_CLAMP: f64 = 0.20;

/// Aggregation gains and shrinkage.
const BIAS_GAIN: f64 = 0.15;
const THRESHOLD_GAIN: f64 = 0.20;
const MIN_SAMPLES_PER_CLASS: usize = 20;
const MIN_SAMPLES_GLOBAL: usize = 50;
const PRIOR_WEIGHT: f64 = 0.5;
const ROUND_DP: i32 = 4;
/// Only records within this age (relative to the injected `now`) contribute.
const WINDOW_MS: i64 = 30 * 24 * 60 * 60 * 1000;

/// A small, clamped router-threshold adjustment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Per-tier bias in `[-0.15, 0.15]`, keyed by canonical text tier.
    #[serde(default)]
    pub per_class_bias: HashMap<String, f64>,
    /// Shift to the base confidence threshold (clamped at read time).
    #[serde(default)]
    pub threshold_adjust: f64,
    /// How many decision records fed the aggregation.
    #[serde(default)]
    pub sample_count: usize,
    /// When the state was generated (epoch ms).
    #[serde(default)]
    pub generated_at_ms: i64,
}

impl CalibrationState {
    /// A zero-adjustment state — applying it is a no-op.
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Returns true when applying this state is a no-op.
    pub fn is_neutral(&self) -> bool {
        self.threshold_adjust == 0.0 && self.per_class_bias.values().all(|v| *v == 0.0)
    }
}

fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

fn round_dp(value: f64, dp: i32) -> f64 {
    let factor = 10f64.powi(dp);
    (value * factor).round() / factor
}

fn finite_float(value: &serde_json::Value) -> Option<f64> {
    match value {
        serde_json::Value::Number(n) => n.as_f64().filter(|f| f.is_finite()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Read-time application (pure; used by the routing policy)
// ---------------------------------------------------------------------------

/// Return the confidence-gate threshold after calibration.
///
/// With `None` this is exactly `base`. With a state, the adjusted threshold is
/// HARD-clamped into `[0.3, 0.7]`.
pub fn effective_threshold(base: f64, state: Option<&CalibrationState>) -> f64 {
    match state {
        None => base,
        Some(state) => clamp(
            base + state.threshold_adjust,
            THRESHOLD_FLOOR,
            THRESHOLD_CEIL,
        ),
    }
}

/// Return `confidence` biased by the tier's clamped per-class adjustment.
///
/// With `None` this returns `confidence` unchanged. Otherwise the bias is
/// HARD-clamped to `[-0.15, 0.15]` and the result stays in `[0, 1]`.
pub fn apply_bias(confidence: f64, tier: &str, state: Option<&CalibrationState>) -> f64 {
    let Some(state) = state else {
        return confidence;
    };
    let key = normalize_text_tier(tier).unwrap_or_else(|| tier.to_string());
    let bias = clamp(
        state.per_class_bias.get(&key).copied().unwrap_or(0.0),
        -BIAS_CLAMP,
        BIAS_CLAMP,
    );
    clamp(confidence + bias, 0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Aggregation (pure, deterministic)
// ---------------------------------------------------------------------------

fn stage_applied(trail: &serde_json::Value, stage: &str) -> bool {
    let Some(arr) = trail.as_array() else {
        return false;
    };
    arr.iter().any(|entry| {
        entry.get("stage").and_then(|v| v.as_str()) == Some(stage)
            && entry
                .get("applied")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
    })
}

fn flag_present(flags: &serde_json::Value, token: &str) -> bool {
    flags
        .as_array()
        .map_or(false, |arr| arr.iter().any(|v| v.as_str() == Some(token)))
}

/// Extract `(gated, complained, pinned)` from one decision record.
fn record_signals(record: &serde_json::Value) -> (bool, bool, bool) {
    let trail = record.get("trail");
    let flags = record.get("flags");
    let source = record.get("source").and_then(|v| v.as_str());

    let gated = trail.map_or(false, |t| stage_applied(t, "confidence_gate"))
        || flags.map_or(false, |f| flag_present(f, "confidence_gate_applied"));
    let complained = trail.map_or(false, |t| stage_applied(t, "complaint_upgrade"))
        || flags.map_or(false, |f| flag_present(f, "complaint_upgrade_applied"));
    let pinned = source == Some("router_control_hold")
        || flags.map_or(false, |f| flag_present(f, "router_control_hold_applied"))
        || flags.map_or(false, |f| flag_present(f, "router_control_hold"));
    (gated, complained, pinned)
}

/// Turn decision records into a clamped [`CalibrationState`].
///
/// Pure and deterministic given `records`, `now` (epoch ms), and `prior`.
/// Records older than the 30-day window relative to `now` are ignored; records
/// without a recognizable `proposed_tier` do not contribute. `prior` (when
/// given and non-neutral) is blended 50/50 for run-to-run stability.
pub fn aggregate_calibration(
    records: &[serde_json::Value],
    now: i64,
    prior: Option<&CalibrationState>,
) -> CalibrationState {
    let window_start = now - WINDOW_MS;
    let mut per_class_vote: HashMap<String, f64> = HashMap::new();
    let mut per_class_count: HashMap<String, usize> = HashMap::new();
    for tier in TEXT_TIERS {
        per_class_vote.insert(tier.to_string(), 0.0);
        per_class_count.insert(tier.to_string(), 0);
    }
    let mut global_vote = 0.0f64;
    let mut considered = 0usize;

    for record in records {
        if let Some(ts) = finite_float(&record["ts_ms"]) {
            if (ts as i64) < window_start {
                continue;
            }
        }
        let tier = record
            .get("proposed_tier")
            .and_then(|v| v.as_str())
            .and_then(normalize_text_tier);
        let Some(tier) = tier else {
            continue;
        };
        if !per_class_vote.contains_key(&tier) {
            continue;
        }
        let (gated, complained, pinned) = record_signals(record);
        considered += 1;
        if let Some(count) = per_class_count.get_mut(&tier) {
            *count += 1;
        }
        if gated && complained {
            if let Some(vote) = per_class_vote.get_mut(&tier) {
                *vote += 1.0;
            }
        } else if gated && !complained {
            if let Some(vote) = per_class_vote.get_mut(&tier) {
                *vote -= 1.0;
            }
        }
        if complained {
            global_vote -= 1.0;
        }
        if pinned {
            global_vote -= 1.0;
        }
        if gated && !complained {
            global_vote += 1.0;
        }
    }

    let mut per_class_bias: HashMap<String, f64> = HashMap::new();
    for tier in TEXT_TIERS {
        let count = *per_class_count.get(tier).unwrap_or(&0);
        if count == 0 {
            continue;
        }
        let raw = BIAS_GAIN * per_class_vote.get(tier).copied().unwrap_or(0.0)
            / (count.max(MIN_SAMPLES_PER_CLASS) as f64);
        let bias = round_dp(clamp(raw, -BIAS_CLAMP, BIAS_CLAMP), ROUND_DP);
        if bias != 0.0 {
            per_class_bias.insert(tier.to_string(), bias);
        }
    }

    let threshold_adjust = if considered == 0 {
        0.0
    } else {
        let raw = THRESHOLD_GAIN * global_vote / (considered.max(MIN_SAMPLES_GLOBAL) as f64);
        round_dp(
            clamp(raw, -THRESHOLD_ADJUST_CLAMP, THRESHOLD_ADJUST_CLAMP),
            ROUND_DP,
        )
    };

    let computed = CalibrationState {
        per_class_bias,
        threshold_adjust,
        sample_count: considered,
        generated_at_ms: now,
    };

    match prior {
        Some(prior) if !prior.is_neutral() => blend(prior, &computed, now),
        _ => computed,
    }
}

/// Blend a prior state with a freshly computed one, 50/50 per field.
fn blend(prior: &CalibrationState, computed: &CalibrationState, now: i64) -> CalibrationState {
    let mut tiers: Vec<String> = prior.per_class_bias.keys().cloned().collect();
    for tier in computed.per_class_bias.keys() {
        if !tiers.contains(tier) {
            tiers.push(tier.clone());
        }
    }
    tiers.sort();

    let mut blended_bias: HashMap<String, f64> = HashMap::new();
    for tier in &tiers {
        let value = PRIOR_WEIGHT * prior.per_class_bias.get(tier).copied().unwrap_or(0.0)
            + (1.0 - PRIOR_WEIGHT) * computed.per_class_bias.get(tier).copied().unwrap_or(0.0);
        let clamped = round_dp(clamp(value, -BIAS_CLAMP, BIAS_CLAMP), ROUND_DP);
        if clamped != 0.0 {
            blended_bias.insert(tier.clone(), clamped);
        }
    }

    let blended_threshold = round_dp(
        clamp(
            PRIOR_WEIGHT * prior.threshold_adjust
                + (1.0 - PRIOR_WEIGHT) * computed.threshold_adjust,
            -THRESHOLD_ADJUST_CLAMP,
            THRESHOLD_ADJUST_CLAMP,
        ),
        ROUND_DP,
    );

    CalibrationState {
        per_class_bias: blended_bias,
        threshold_adjust: blended_threshold,
        sample_count: computed.sample_count,
        generated_at_ms: now,
    }
}

// ---------------------------------------------------------------------------
// Load / save (atomic; tolerate missing/corrupt)
// ---------------------------------------------------------------------------

/// The default path to `router_calibration.json` (under the current state dir).
pub fn calibration_path() -> PathBuf {
    PathBuf::from(CALIBRATION_FILENAME)
}

/// Load the calibration state; a missing or corrupt file yields a neutral
/// state. Never returns an error.
pub fn load_calibration(path: Option<&Path>) -> CalibrationState {
    let default_path = calibration_path();
    let path = path.unwrap_or(&default_path);
    let Ok(raw) = std::fs::read_to_string(path) else {
        return CalibrationState::neutral();
    };
    match serde_json::from_str::<CalibrationState>(&raw) {
        Ok(state) => state,
        Err(_) => CalibrationState::neutral(),
    }
}

/// Atomically write `state` to the calibration file; return the path.
///
/// The write is atomic via a temp file + rename so a concurrent reader never
/// observes a partially-written payload.
pub fn save_calibration(
    state: &CalibrationState,
    path: Option<&Path>,
) -> Result<PathBuf, opensquilla_core::error::Error> {
    let path = path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(calibration_path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let payload = serde_json::to_string_pretty(state)?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &payload)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_neutral_state_is_noop() {
        let state = CalibrationState::neutral();
        assert!(state.is_neutral());
        assert_eq!(effective_threshold(0.5, Some(&state)), 0.5);
        assert_eq!(apply_bias(0.5, "c2", Some(&state)), 0.5);
    }

    #[test]
    fn test_effective_threshold_clamped() {
        let state = CalibrationState {
            threshold_adjust: 10.0,
            ..Default::default()
        };
        // Clamped into [0.3, 0.7].
        assert_eq!(effective_threshold(0.5, Some(&state)), 0.7);
        let state = CalibrationState {
            threshold_adjust: -10.0,
            ..Default::default()
        };
        assert_eq!(effective_threshold(0.5, Some(&state)), 0.3);
    }

    #[test]
    fn test_apply_bias_clamped() {
        let mut bias = HashMap::new();
        bias.insert("c2".to_string(), 5.0); // way over the clamp
        let state = CalibrationState {
            per_class_bias: bias,
            ..Default::default()
        };
        // Bias clamped to +0.15, result in [0, 1].
        assert!((apply_bias(0.5, "c2", Some(&state)) - 0.65).abs() < 1e-9);
        assert!((apply_bias(0.9, "c2", Some(&state)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_clean_gate_downgrade() {
        // A clean confidence-gate downgrade (no complaint) -> negative bias.
        let record = json!({
            "ts_ms": 1_700_000_000_000i64,
            "proposed_tier": "c2",
            "trail": [{"stage": "confidence_gate", "applied": true}],
        });
        let state = aggregate_calibration(&[record], 1_700_000_000_000i64, None);
        let bias = state.per_class_bias.get("c2").copied().unwrap_or(0.0);
        assert!(bias < 0.0);
    }

    #[test]
    fn test_aggregate_ignores_old_records() {
        let record = json!({
            "ts_ms": 1_000_000_000i64, // way older than the 30d window
            "proposed_tier": "c2",
        });
        let state = aggregate_calibration(&[record], 1_700_000_000_000i64, None);
        assert_eq!(state.sample_count, 0);
        assert!(state.is_neutral());
    }
}
