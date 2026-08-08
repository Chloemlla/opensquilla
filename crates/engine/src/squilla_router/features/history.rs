//! History channel (16-dim) plus the trajectory classifier.
//!
//! Port of `features.py::extract_hist_features` and `trajectory.py::classify`:
//! summarizes the recent route-decision sequence (last/max/dominant route
//! index, switch count) and one-hot encodes the inferred trajectory shape.

use super::route_class_idx;
use super::{HIST_DIMS, PrevRouteDecision};
use std::collections::HashMap;

/// Shape of the difficulty/route trend over prior decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trajectory {
    ColdStart,
    StableLow,
    StableHigh,
    Escalating,
    Descaling,
    Oscillating,
    Unclear,
    Mixed,
}

impl Trajectory {
    /// The 8 canonical names, in declaration order.
    pub fn as_str(&self) -> &'static str {
        match self {
            Trajectory::ColdStart => "COLD_START",
            Trajectory::StableLow => "STABLE_LOW",
            Trajectory::StableHigh => "STABLE_HIGH",
            Trajectory::Escalating => "ESCALATING",
            Trajectory::Descaling => "DESCALING",
            Trajectory::Oscillating => "OSCILLATING",
            Trajectory::Unclear => "UNCLEAR",
            Trajectory::Mixed => "MIXED",
        }
    }

    /// One-hot slot in the 16-dim vector: `8 + declaration index`.
    fn slot(&self) -> usize {
        match self {
            Trajectory::ColdStart => 8,
            Trajectory::StableLow => 9,
            Trajectory::StableHigh => 10,
            Trajectory::Escalating => 11,
            Trajectory::Descaling => 12,
            Trajectory::Oscillating => 13,
            Trajectory::Unclear => 14,
            Trajectory::Mixed => 15,
        }
    }
}

/// Route index for the `[0]`/`[3]` feature slots: unknown classes map to `-1`.
fn route_idx(class: &str) -> i64 {
    match route_class_idx(class) {
        Some(i) => i as i64,
        None => -1,
    }
}

/// Route index for the dominant/switches counts: unknown classes map to `0`
/// (mirrors the Python `_ROUTE_TO_IDX.get(route, 0)`).
fn route_idx_or_zero(class: &str) -> i64 {
    match route_class_idx(class) {
        Some(i) => i as i64,
        None => 0,
    }
}

/// Port of `trajectory.py::classify` with the default `delta_threshold = 0.3`.
pub fn classify_trajectory(history: &[PrevRouteDecision]) -> Trajectory {
    classify_trajectory_with(history, 0.3)
}

/// Port of `trajectory.py::classify` with an explicit `delta_threshold`.
pub fn classify_trajectory_with(history: &[PrevRouteDecision], delta_threshold: f64) -> Trajectory {
    if history.is_empty() {
        return Trajectory::ColdStart;
    }
    if history.len() < 2 {
        return Trajectory::Unclear;
    }

    let all_low = history
        .iter()
        .all(|d| d.route_class == "R0" || d.route_class == "R1");
    let all_high = history
        .iter()
        .all(|d| d.route_class == "R2" || d.route_class == "R3");
    if all_low {
        return Trajectory::StableLow;
    }
    if all_high {
        return Trajectory::StableHigh;
    }

    let nonzero: Vec<i8> = history
        .windows(2)
        .filter_map(|w| {
            let delta = w[1].difficulty - w[0].difficulty;
            if delta > delta_threshold {
                Some(1)
            } else if delta < -delta_threshold {
                Some(-1)
            } else {
                None
            }
        })
        .collect();

    if nonzero.len() >= 2 && nonzero.iter().all(|&s| s == 1) {
        return Trajectory::Escalating;
    }
    if nonzero.len() >= 2 && nonzero.iter().all(|&s| s == -1) {
        return Trajectory::Descaling;
    }

    let direction_changes = nonzero.windows(2).filter(|w| w[0] != w[1]).count();
    if direction_changes >= 2 {
        return Trajectory::Oscillating;
    }

    Trajectory::Mixed
}

/// Extract the 16-dim history channel. Port of
/// `features.py::extract_hist_features` (trajectory inferred internally).
pub fn extract_hist_features(history: &[PrevRouteDecision]) -> [f64; HIST_DIMS] {
    let trajectory = classify_trajectory(history);
    let mut vec = [0.0; HIST_DIMS];

    if history.is_empty() {
        vec[0] = -1.0;
        vec[3] = -1.0;
        vec[4] = 1.0;
        vec[6] = -1.0;
        vec[7] = 0.0;
    } else {
        let last = &history[history.len() - 1];
        vec[0] = route_idx(&last.route_class) as f64;
        vec[1] = last.difficulty;
        vec[2] = last.margin;
        vec[3] = history
            .iter()
            .map(|d| route_idx(&d.route_class))
            .max()
            .unwrap_or(-1) as f64;
        vec[4] = (history.len() + 1) as f64;
        vec[5] = history.len() as f64;

        let ridx: Vec<i64> = history
            .iter()
            .map(|d| route_idx_or_zero(&d.route_class))
            .collect();
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for r in &ridx {
            *counts.entry(*r).or_insert(0) += 1;
        }
        let max_count = counts.values().max().copied().unwrap_or_default();
        let dominant = counts
            .iter()
            .filter(|(_, c)| **c == max_count)
            .map(|(r, _)| *r)
            .max()
            .unwrap_or(-1);
        vec[6] = dominant as f64;

        let switches = ridx.windows(2).filter(|w| w[0] != w[1]).count();
        vec[7] = switches as f64;
    }

    vec[trajectory.slot()] = 1.0;
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(route_class: &str, difficulty: f64, margin: f64) -> PrevRouteDecision {
        PrevRouteDecision {
            route_class: route_class.to_string(),
            difficulty,
            margin,
        }
    }

    #[test]
    fn trajectory_as_str_names() {
        assert_eq!(Trajectory::ColdStart.as_str(), "COLD_START");
        assert_eq!(Trajectory::StableLow.as_str(), "STABLE_LOW");
        assert_eq!(Trajectory::StableHigh.as_str(), "STABLE_HIGH");
        assert_eq!(Trajectory::Escalating.as_str(), "ESCALATING");
        assert_eq!(Trajectory::Descaling.as_str(), "DESCALING");
        assert_eq!(Trajectory::Oscillating.as_str(), "OSCILLATING");
        assert_eq!(Trajectory::Unclear.as_str(), "UNCLEAR");
        assert_eq!(Trajectory::Mixed.as_str(), "MIXED");
    }

    #[test]
    fn empty_history_cold_start_and_defaults() {
        assert_eq!(classify_trajectory(&[]), Trajectory::ColdStart);
        let f = extract_hist_features(&[]);
        assert_eq!(f[0], -1.0);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
        assert_eq!(f[3], -1.0);
        assert_eq!(f[4], 1.0);
        assert_eq!(f[5], 0.0);
        assert_eq!(f[6], -1.0);
        assert_eq!(f[7], 0.0);
        // COLD_START one-hot at slot 8, rest of the one-hot block zero.
        assert_eq!(f[8], 1.0);
        assert_eq!(f[9..16].iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn single_decision_unclear_and_layout() {
        let h = vec![dec("R2", 0.8, 0.5)];
        assert_eq!(classify_trajectory(&h), Trajectory::Unclear);
        let f = extract_hist_features(&h);
        assert_eq!(f[0], 2.0); // last route idx
        assert_eq!(f[1], 0.8); // last difficulty
        assert_eq!(f[2], 0.5); // last margin
        assert_eq!(f[3], 2.0); // max route idx
        assert_eq!(f[4], 2.0); // len + 1
        assert_eq!(f[5], 1.0); // len
        assert_eq!(f[6], 2.0); // dominant
        assert_eq!(f[7], 0.0); // switches
        assert_eq!(f[14], 1.0); // UNCLEAR slot
        assert_eq!(f[8..16].iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn dual_history_stable_low_and_high() {
        let low = vec![dec("R0", 0.2, 0.1), dec("R1", 0.4, 0.3)];
        assert_eq!(classify_trajectory(&low), Trajectory::StableLow);
        let f = extract_hist_features(&low);
        assert_eq!(f[9], 1.0); // STABLE_LOW
        assert_eq!(f[6], 1.0); // R0/R1 tie -> highest idx wins

        let high = vec![dec("R2", 0.9, 0.4), dec("R3", 0.95, 0.6)];
        assert_eq!(classify_trajectory(&high), Trajectory::StableHigh);
        let f = extract_hist_features(&high);
        assert_eq!(f[10], 1.0); // STABLE_HIGH
        assert_eq!(f[6], 3.0);
    }

    #[test]
    fn escalating_difficulty() {
        let h = vec![
            dec("R0", 0.0, 0.0),
            dec("R1", 0.5, 0.2),
            dec("R2", 1.0, 0.5),
        ];
        assert_eq!(classify_trajectory(&h), Trajectory::Escalating);
        let f = extract_hist_features(&h);
        assert_eq!(f[11], 1.0); // ESCALATING slot
        assert_eq!(f[8..16].iter().sum::<f64>(), 1.0);
    }

    #[test]
    fn descaling_difficulty() {
        let h = vec![
            dec("R2", 1.0, 0.5),
            dec("R1", 0.5, 0.2),
            dec("R0", 0.0, 0.0),
        ];
        assert_eq!(classify_trajectory(&h), Trajectory::Descaling);
        let f = extract_hist_features(&h);
        assert_eq!(f[12], 1.0); // DESCALING slot
    }

    #[test]
    fn oscillating_sign_changes() {
        let h = vec![
            dec("R0", 0.0, 0.0),
            dec("R2", 1.0, 0.5),
            dec("R1", 0.0, 0.0),
            dec("R3", 1.0, 0.5),
        ];
        assert_eq!(classify_trajectory(&h), Trajectory::Oscillating);
        let f = extract_hist_features(&h);
        assert_eq!(f[13], 1.0); // OSCILLATING slot
    }

    #[test]
    fn mixed_trajectory_when_signals_below_threshold() {
        let h = vec![
            dec("R0", 0.0, 0.0),
            dec("R1", 0.1, 0.05),
            dec("R2", 0.2, 0.1),
        ];
        assert_eq!(classify_trajectory(&h), Trajectory::Mixed);
        let f = extract_hist_features(&h);
        assert_eq!(f[15], 1.0); // MIXED slot
    }

    #[test]
    fn classify_with_custom_threshold() {
        let h = vec![
            dec("R0", 0.0, 0.0),
            dec("R1", 0.1, 0.0),
            dec("R2", 0.2, 0.0),
        ];
        assert_eq!(classify_trajectory(&h), Trajectory::Mixed); // default 0.3
        assert_eq!(classify_trajectory_with(&h, 0.05), Trajectory::Escalating);
    }

    #[test]
    fn dominant_route_tie_breaks_to_highest() {
        let h = vec![
            dec("R1", 0.4, 0.2),
            dec("R1", 0.4, 0.2),
            dec("R2", 0.8, 0.5),
            dec("R2", 0.8, 0.5),
        ];
        let f = extract_hist_features(&h);
        assert_eq!(f[6], 2.0); // tie between 1 and 2 -> 2
    }

    #[test]
    fn switches_count_adjacent_route_changes() {
        let h = vec![
            dec("R0", 0.1, 0.0),
            dec("R1", 0.4, 0.2),
            dec("R1", 0.4, 0.2),
            dec("R2", 0.9, 0.5),
        ];
        let f = extract_hist_features(&h);
        assert_eq!(f[7], 2.0); // R0->R1, R1->R1, R1->R2
    }

    #[test]
    fn max_route_and_last_decision_fields() {
        let h = vec![
            dec("R0", 0.1, 0.0),
            dec("R2", 0.6, 0.4),
            dec("R1", 0.3, 0.2),
        ];
        let f = extract_hist_features(&h);
        assert_eq!(f[0], 1.0); // last is R1
        assert_eq!(f[1], 0.3);
        assert_eq!(f[2], 0.2);
        assert_eq!(f[3], 2.0); // max over R0/R2/R1
        assert_eq!(f[4], 4.0); // len + 1
        assert_eq!(f[5], 3.0); // len
    }

    #[test]
    fn unknown_route_class_maps_to_minus_one() {
        let h = vec![dec("RX", 0.5, 0.3)];
        assert_eq!(classify_trajectory(&h), Trajectory::Unclear);
        let f = extract_hist_features(&h);
        assert_eq!(f[0], -1.0);
        assert_eq!(f[3], -1.0);
        assert_eq!(f[6], 0.0); // unknown maps to 0 in the dominant count
    }
}
