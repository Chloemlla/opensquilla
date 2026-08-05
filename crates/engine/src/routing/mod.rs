//! Routing policy engine.
//!
//! Mirrors the Python backend's `engine/routing/` package: a deterministic,
//! post-classifier policy that decides the final routing tier from a
//! classification decision plus turn facts. The pipeline preserves the exact
//! legacy ordering:
//!
//! 1. [`confidence_gate`] — low classifier confidence falls back to the
//!    configured default tier (with a margin discount for above-default tiers).
//! 2. [`complaint_upgrade`] — a short message containing a known complaint term
//!    upgrades the tier.
//! 3. [`anti_downgrade`] — within the KV-cache window, never route below the
//!    previous turn's final tier.
//! 4. [`capability_gate`] — walk the working tier UP when the model catalog
//!    gives a definite signal that its model cannot serve the turn.
//! 5. [`bind`] — record the finalized routing trail and rebind to the final
//!    tier's configured model.
//! 6. [`large_context_floor`] — turns carrying large material contexts are
//!    floored to c2/c3 regardless of the classified tier.
//! 7. [`budget_gate`] — warn or cap when accumulated session spend crosses the
//!    configured limit (additive, default-off).
//! 8. [`provider_mismatch`] — flag-only by default; `veto` mode rebinds to the
//!    nearest tier that executes on the active provider.
//!
//! [`RoutingPolicyEngine`] runs these stages in order over a [`PolicyInputs`]
//! value. Tier helpers (`normalize_text_tier`, `tier_index`, route-class
//! maps) mirror `router_tiers.py` so config written against legacy `t0`-`t3`
//! ids keeps working.

pub mod calibration;
pub mod health_ledger;
pub mod selector;

use crate::routing::calibration::{apply_bias, effective_threshold};
use serde_json::{json, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tier constants and helpers (mirrors `router_tiers.py`)
// ---------------------------------------------------------------------------

/// The canonical text tier ladder, lowest to highest.
pub const TEXT_TIERS: [&str; 4] = ["c0", "c1", "c2", "c3"];
/// The default text tier.
pub const DEFAULT_TEXT_TIER: &str = "c1";
/// The highest text tier.
pub const HIGHEST_TEXT_TIER: &str = "c3";
/// The image tier id.
pub const IMAGE_TIER: &str = "image_model";

/// Legacy tier aliases (`t0` -> `c0`, ...).
const LEGACY_TEXT_TIER_ALIASES: [(&str, &str); 4] =
    [("t0", "c0"), ("t1", "c1"), ("t2", "c2"), ("t3", "c3")];

/// Route-class id to canonical tier.
pub const ROUTE_CLASS_TO_TIER: [(&str, &str); 4] =
    [("R0", "c0"), ("R1", "c1"), ("R2", "c2"), ("R3", "c3")];

/// Canonical tier to route-class id.
pub const TIER_TO_ROUTE_CLASS: [(&str, &str); 4] =
    [("c0", "R0"), ("c1", "R1"), ("c2", "R2"), ("c3", "R3")];

/// Thinking-mode ordering used by controller reconciliation.
const THINKING_MODE_ORDER: [(&str, u8); 4] = [("T0", 0), ("T1", 1), ("T2", 2), ("T3", 3)];

/// Large-context floor thresholds (mirrors `policy_data.py`).
pub const LARGE_CONTEXT_T2_FLOOR_TOKENS: u64 = 25_000;
pub const LARGE_CONTEXT_T3_FLOOR_TOKENS: u64 = 80_000;
pub const LARGE_CONTEXT_T3_CONTEXT_RATIO: f64 = 0.40;
pub const DEFAULT_CONTEXT_WINDOW_TOKENS: u64 = 200_000;

/// Complaint terms, zh/en only (mirrors `policy_data.py`).
const COMPLAINT_TERMS: &[&str] = &[
    "不对", "不行", "不对劲", "还是不对", "完全不对", "不是这样", "你搞错了", "你说错了",
    "回答错了", "理解错了", "搞错重点了", "错了", "答非所问", "没理解", "没听懂", "太差",
    "太敷衍", "敷衍", "没用", "废话", "离谱", "乱说", "瞎说", "胡扯", "答得太差", "质量太差",
    "不满意", "胡说", "漏了", "遗漏了", "没提到", "没覆盖", "跑题了", "偏题了", "不是我要的",
    "没按要求", "没有按要求", "重写", "重新来", "重新回答", "再来一版", "换个说法", "重新组织",
    "按我说的重来", "你没有回答", "垃圾", "傻逼", "sb", "蠢", "废物", "滚", "妈的", "操", "艹",
    "wrong", "incorrect", "not correct", "you are wrong", "completely wrong", "totally wrong",
    "not what i asked", "you misunderstood", "that's not right", "this is not right",
    "bad answer", "terrible answer", "awful answer", "horrible answer", "poor answer",
    "lazy answer", "low quality", "poor quality", "try again", "redo", "rewrite",
    "start over", "answer again", "you missed", "missed the point", "off topic",
    "irrelevant", "not helpful", "garbage", "trash", "crap", "sucks", "stupid", "idiot",
    "moron", "dumb", "pathetic", "ridiculous", "fuck", "fucking", "shit", "damn", "wtf",
    "asshole", "bullshit", "nonsense", "useless",
];

/// Normalize a tier value to its canonical text tier id, accepting legacy
/// `t0`-`t3` aliases. Returns `None` for unknown or empty values.
pub fn normalize_text_tier(value: &str) -> Option<String> {
    let tier = value.trim().to_lowercase();
    if tier.is_empty() {
        return None;
    }
    if TEXT_TIERS.contains(&tier.as_str()) {
        return Some(tier);
    }
    LEGACY_TEXT_TIER_ALIASES
        .iter()
        .find(|(alias, _)| *alias == tier.as_str())
        .map(|(_, canonical)| canonical.to_string())
}

/// Return 0-3 for known text tiers; -1 for unknown values.
pub fn tier_index(value: &str) -> i32 {
    match normalize_text_tier(value) {
        Some(tier) => TEXT_TIERS
            .iter()
            .position(|t| *t == tier)
            .map_or(-1, |i| i as i32),
        None => -1,
    }
}

/// Map a canonical tier to its route-class id (`c2` -> `R2`).
pub fn route_class_for_tier(tier: &str) -> Option<String> {
    let normalized = normalize_text_tier(tier).unwrap_or_else(|| tier.to_string());
    TIER_TO_ROUTE_CLASS
        .iter()
        .find(|(t, _)| *t == normalized)
        .map(|(_, rc)| rc.to_string())
}

/// Map a route-class id to its canonical tier (`R2` -> `c2`).
pub fn tier_for_route_class(route_class: Option<&str>) -> Option<String> {
    let rc = route_class?;
    ROUTE_CLASS_TO_TIER
        .iter()
        .find(|(r, _)| *r == rc)
        .map(|(_, tier)| tier.to_string())
}

/// Order tiers by the canonical c0<c1<c2<c3 ladder, not config/TOML order.
fn canonical_order(valid_tiers: &[String]) -> Vec<String> {
    let mut tiers: Vec<&String> = valid_tiers.iter().collect();
    tiers.sort_by_key(|name| {
        let idx = tier_index(name);
        if idx >= 0 {
            (0, idx as usize)
        } else {
            (1, 0)
        }
    });
    tiers.into_iter().cloned().collect()
}

/// The ladder position of `tier` within `valid_tiers`, or -1.
fn tier_index_in(tier: &str, valid_tiers: &[String]) -> i32 {
    let normalized = normalize_text_tier(tier).unwrap_or_else(|| tier.to_string());
    let ordered = canonical_order(valid_tiers);
    ordered
        .iter()
        .position(|t| *t == normalized)
        .map_or(-1, |i| i as i32)
}

/// Move `tier` up the ladder by `steps` (never below c0, never above the top).
fn upgrade_tier(tier: &str, valid_tiers: &[String], steps: i32) -> String {
    let ordered = canonical_order(valid_tiers);
    let normalized = normalize_text_tier(tier).unwrap_or_else(|| tier.to_string());
    match ordered.iter().position(|t| *t == normalized) {
        Some(i) => {
            let target = i as i32 + steps.max(0);
            let last = ordered.len() - 1;
            ordered[target.min(last as i32) as usize].clone()
        }
        None => tier.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Router configuration and routing decision
// ---------------------------------------------------------------------------

/// The routing-policy configuration, mirroring the `squilla_router` config
/// object's attributes.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Classifier confidence below which a non-default tier falls back.
    pub confidence_threshold: f64,
    /// Discount applied to the cutoff for tiers above the default.
    pub confidence_high_tier_margin: f64,
    /// The default tier id (e.g. `c1`).
    pub default_tier: Option<String>,
    /// Number of ladder steps a complaint upgrade moves the tier.
    pub complaint_upgrade_steps: i32,
    /// Messages longer than this many characters are never complaint-detected.
    pub complaint_upgrade_max_chars: usize,
    /// Master switch for complaint upgrades.
    pub complaint_upgrade_enabled: bool,
    /// Whether the KV-cache anti-downgrade hold is active.
    pub kv_cache_anti_downgrade_enabled: bool,
    /// Window (seconds) in which the previous turn's tier is held.
    pub kv_cache_anti_downgrade_window_seconds: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            confidence_high_tier_margin: 0.05,
            default_tier: None,
            complaint_upgrade_steps: 1,
            complaint_upgrade_max_chars: 160,
            complaint_upgrade_enabled: true,
            kv_cache_anti_downgrade_enabled: true,
            kv_cache_anti_downgrade_window_seconds: 600.0,
        }
    }
}

/// A typed view over one router tier entry.
#[derive(Debug, Clone, Default)]
pub struct TierConfig {
    /// Provider serving this tier (empty means "the active provider").
    pub provider: String,
    /// Model bound to this tier.
    pub model: String,
    /// Human-readable description.
    pub description: String,
    /// Thinking level enforced for the tier.
    pub thinking_level: Option<String>,
    /// Whether the tier's model supports images.
    pub supports_image: bool,
    /// Whether the tier is image-only (bypasses the confidence gate).
    pub image_only: bool,
}

impl TierConfig {
    /// Build from a JSON tier entry, tolerant of `null`/non-object values.
    pub fn from_value(value: &Value) -> Self {
        let get = |key: &str| -> Value {
            match value {
                Value::Object(map) => map.get(key).cloned().unwrap_or(Value::Null),
                _ => Value::Null,
            }
        };
        let thinking = get("thinking_level");
        let provider = get("provider");
        let model = get("model");
        Self {
            provider: provider
                .as_str()
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            model: model.as_str().map(|s| s.trim().to_string()).unwrap_or_default(),
            description: get("description")
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_default(),
            thinking_level: match thinking {
                Value::Null => None,
                v => {
                    let s = v.as_str().map(|s| s.trim()).unwrap_or_default();
                    if s.is_empty() {
                        None
                    } else {
                        Some(s.to_string())
                    }
                }
            },
            supports_image: get("supports_image").as_bool().unwrap_or(false),
            image_only: get("image_only").as_bool().unwrap_or(false),
        }
    }
}

/// Definite catalog facts for one tier's model. `None` on either field means
/// the catalog gave no definite signal — the capability gate never acts on a
/// `None`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierCapability {
    /// Whether the tier's model definitely supports vision.
    pub supports_vision: Option<bool>,
    /// The tier's model context window, when definitely known.
    pub context_window: Option<u64>,
}

/// The result of router classification.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The routed text tier (e.g. `c2`).
    pub tier: String,
    /// The model bound to that tier.
    pub model: String,
    /// Classifier confidence for the decision.
    pub confidence: f64,
    /// Decision source: `image_route` | `v4_phase3` | `heuristic` |
    /// `large_context_floor` | `budget_cap` | `default` | ...
    pub source: String,
}

impl RoutingDecision {
    /// Create a new routing decision.
    pub fn new(
        tier: impl Into<String>,
        model: impl Into<String>,
        confidence: f64,
        source: impl Into<String>,
    ) -> Self {
        Self {
            tier: tier.into(),
            model: model.into(),
            confidence,
            source: source.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Detect complaint terms in a message. Long messages (over `max_chars`) are
/// ignored.
pub fn detect_complaint(message: &str, max_chars: Option<usize>) -> Vec<String> {
    let text = message.trim();
    if let Some(max) = max_chars {
        if max > 0 && text.chars().count() > max {
            return Vec::new();
        }
    }
    let lowered = text.to_lowercase();
    COMPLAINT_TERMS
        .iter()
        .filter(|term| lowered.contains(&**term))
        .map(|s| s.to_string())
        .collect()
}

/// Find the most recent routing-history entry within the window.
pub fn previous_final_entry<'a>(
    routing_history: &'a [HashMap<String, Value>],
    now: f64,
    window: f64,
) -> Option<&'a HashMap<String, Value>> {
    if routing_history.is_empty() {
        return None;
    }
    let cutoff = now - window;
    for entry in routing_history.iter().rev() {
        let ts = entry.get("_ts").and_then(|v| v.as_f64()).unwrap_or(now);
        if ts >= cutoff {
            return Some(entry);
        }
    }
    None
}

/// The previous turn's final tier, normalized.
pub fn previous_final_tier(entry: Option<&HashMap<String, Value>>) -> Option<String> {
    let entry = entry?;
    if let Some(tier) = entry.get("final_tier").and_then(|v| v.as_str()) {
        return Some(normalize_text_tier(tier).unwrap_or_else(|| tier.to_string()));
    }
    let rc = entry
        .get("final_route_class")
        .or_else(|| entry.get("route_class"))
        .and_then(|v| v.as_str());
    tier_for_route_class(rc)
}

/// A process-relative monotonic clock in seconds, in the spirit of Python's
/// `time.monotonic`.
pub fn monotonic_now() -> f64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_secs_f64()
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

/// Result of the confidence gate.
#[derive(Debug, Clone)]
pub struct ConfidenceGateResult {
    /// The tier after the gate.
    pub tier: String,
    /// Whether the gate downgraded to the default tier.
    pub applied: bool,
    /// The effective threshold used.
    pub threshold: f64,
    /// The default tier the gate falls back to.
    pub default_tier: Option<String>,
}

/// Fall back to the default tier when classifier confidence is too low.
///
/// `calibration` is additive and default-off: `None` makes the gate
/// byte-identical to the uncalibrated path.
pub fn confidence_gate(
    tier: &str,
    confidence: f64,
    router_cfg: &RouterConfig,
    valid_tiers: &[String],
    tiers: &HashMap<String, TierConfig>,
    calibration: Option<&CalibrationState>,
) -> ConfidenceGateResult {
    let threshold = effective_threshold(router_cfg.confidence_threshold, calibration);
    let Some(default_tier) = router_cfg.default_tier.as_deref() else {
        return ConfidenceGateResult {
            tier: tier.to_string(),
            applied: false,
            threshold,
            default_tier: None,
        };
    };
    let default_tier = normalize_text_tier(default_tier).unwrap_or_else(|| default_tier.to_string());

    // Image-only tiers bypass the gate.
    if tiers.get(tier).map_or(false, |t| t.image_only) {
        return ConfidenceGateResult {
            tier: tier.to_string(),
            applied: false,
            threshold,
            default_tier: Some(default_tier),
        };
    }

    let gate_confidence = apply_bias(confidence, tier, calibration);
    let tier_rank = tier_index_in(tier, valid_tiers);
    let default_rank = tier_index_in(&default_tier, valid_tiers);
    let cutoff = if tier_rank > default_rank {
        threshold - router_cfg.confidence_high_tier_margin
    } else {
        threshold
    };

    if gate_confidence < cutoff && tier_rank >= 0 && default_rank >= 0 && tier != default_tier {
        return ConfidenceGateResult {
            tier: default_tier.clone(),
            applied: true,
            threshold,
            default_tier: Some(default_tier),
        };
    }

    ConfidenceGateResult {
        tier: tier.to_string(),
        applied: false,
        threshold,
        default_tier: Some(default_tier),
    }
}

/// Result of the complaint upgrade.
#[derive(Debug, Clone)]
pub struct ComplaintUpgradeResult {
    /// The tier after the upgrade.
    pub tier: String,
    /// The complaint terms detected.
    pub terms: Vec<String>,
    /// Whether the tier was upgraded.
    pub applied: bool,
    /// The number of ladder steps applied.
    pub steps: i32,
    /// The max message length used for detection.
    pub max_chars: usize,
}

/// Upgrade the tier when a short message contains a known complaint term.
pub fn complaint_upgrade(
    tier: &str,
    message: &str,
    router_cfg: &RouterConfig,
    valid_tiers: &[String],
    pre_confidence_tier: Option<&str>,
    previous_tier: Option<&str>,
) -> ComplaintUpgradeResult {
    let steps = router_cfg.complaint_upgrade_steps;
    let max_chars = router_cfg.complaint_upgrade_max_chars;
    if !router_cfg.complaint_upgrade_enabled {
        return ComplaintUpgradeResult {
            tier: tier.to_string(),
            terms: Vec::new(),
            applied: false,
            steps,
            max_chars,
        };
    }
    let terms = detect_complaint(message, Some(max_chars));
    if terms.is_empty() {
        return ComplaintUpgradeResult {
            tier: tier.to_string(),
            terms,
            applied: false,
            steps,
            max_chars,
        };
    }

    let mut start = tier.to_string();
    if let Some(pre) = pre_confidence_tier {
        if valid_tiers.iter().any(|t| t.as_str() == pre)
            && tier_index_in(pre, valid_tiers) > tier_index_in(&start, valid_tiers)
        {
            start = pre.to_string();
        }
    }
    if let Some(prev) = previous_tier {
        if valid_tiers.iter().any(|t| t.as_str() == prev)
            && tier_index_in(prev, valid_tiers) > tier_index_in(&start, valid_tiers)
        {
            start = prev.to_string();
        }
    }

    let upgraded = upgrade_tier(&start, valid_tiers, steps);
    ComplaintUpgradeResult {
        tier: upgraded.clone(),
        terms,
        applied: upgraded != tier,
        steps,
        max_chars,
    }
}

/// Result of the anti-downgrade hold.
#[derive(Debug, Clone)]
pub struct AntiDowngradeResult {
    /// The tier after the hold.
    pub tier: String,
    /// Whether the hold was applied.
    pub applied: bool,
}

/// Hold the previous turn's tier when routing would drop below it.
pub fn anti_downgrade(
    tier: &str,
    router_cfg: &RouterConfig,
    valid_tiers: &[String],
    previous_tier: Option<&str>,
) -> AntiDowngradeResult {
    let prev = previous_tier;
    let prev_holds = prev
        .map(|p| valid_tiers.iter().any(|t| t.as_str() == p))
        .unwrap_or(false);
    let prev_above = prev
        .map(|p| tier_index_in(p, valid_tiers) > tier_index_in(tier, valid_tiers))
        .unwrap_or(false);
    if router_cfg.kv_cache_anti_downgrade_enabled
        && prev_holds
        && tier_index_in(tier, valid_tiers) >= 0
        && prev_above
    {
        return AntiDowngradeResult {
            tier: prev.unwrap().to_string(),
            applied: true,
        };
    }
    AntiDowngradeResult {
        tier: tier.to_string(),
        applied: false,
    }
}

/// A capability-gate walk-up action.
#[derive(Debug, Clone)]
pub struct CapabilityGateAction {
    /// `vision_walk_up` or `context_walk_up`.
    pub rule: String,
    /// The tier walked from.
    pub from_tier: String,
    /// The tier walked to.
    pub to_tier: String,
}

/// Result of the capability gate.
#[derive(Debug, Clone)]
pub struct CapabilityGateResult {
    /// The tier after the gate.
    pub tier: String,
    /// The walk-up actions applied (empty when none).
    pub actions: Vec<CapabilityGateAction>,
}

/// Walk the working tier UP when the catalog says its model cannot serve the
/// turn. Never acts on a `None` capability field.
pub fn capability_gate(
    tier: &str,
    valid_tiers: &[String],
    tier_capabilities: Option<&HashMap<String, TierCapability>>,
    turn_has_image: bool,
    material_tokens: u64,
) -> CapabilityGateResult {
    let Some(caps) = tier_capabilities else {
        return CapabilityGateResult {
            tier: tier.to_string(),
            actions: Vec::new(),
        };
    };
    let ordered = canonical_order(valid_tiers);
    let normalized = normalize_text_tier(tier).unwrap_or_else(|| tier.to_string());
    let Some(mut idx) = ordered.iter().position(|t| *t == normalized) else {
        return CapabilityGateResult {
            tier: tier.to_string(),
            actions: Vec::new(),
        };
    };
    let mut current = ordered[idx].clone();
    let mut actions: Vec<CapabilityGateAction> = Vec::new();
    let caps_of = |name: &str| caps.get(name).copied().unwrap_or_default();

    if turn_has_image && caps_of(&current).supports_vision == Some(false) {
        for candidate in &ordered[idx + 1..] {
            if caps_of(candidate).supports_vision == Some(true) {
                actions.push(CapabilityGateAction {
                    rule: "vision_walk_up".to_string(),
                    from_tier: current.clone(),
                    to_tier: candidate.clone(),
                });
                current = candidate.clone();
                idx = ordered.iter().position(|t| *t == current).unwrap();
                break;
            }
        }
    }

    let window = caps_of(&current).context_window;
    if material_tokens > 0 && window.map_or(false, |w| material_tokens > w) {
        let mut target: Option<String> = None;
        for candidate in &ordered[idx + 1..] {
            let cw = caps_of(candidate).context_window;
            if cw.map_or(false, |w| material_tokens <= w) {
                target = Some(candidate.clone());
                break;
            }
        }
        if target.is_none() && idx < ordered.len() - 1 {
            target = Some(ordered[ordered.len() - 1].clone());
        }
        if let Some(t) = target {
            if t != current {
                actions.push(CapabilityGateAction {
                    rule: "context_walk_up".to_string(),
                    from_tier: current.clone(),
                    to_tier: t.clone(),
                });
                current = t;
            }
        }
    }

    CapabilityGateResult { tier: current, actions }
}

/// Append the capability gate's actions to the routing trail.
pub fn record_capability_gate_trail(
    extra: &mut HashMap<String, Value>,
    result: &CapabilityGateResult,
) {
    if result.actions.is_empty() {
        return;
    }
    let trail = extra
        .entry("routing_trail".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(arr) = trail.as_array_mut() {
        for action in &result.actions {
            let mut entry = serde_json::Map::new();
            entry.insert("stage".to_string(), Value::String("capability_gate".to_string()));
            entry.insert("rule".to_string(), Value::String(action.rule.clone()));
            entry.insert("from_tier".to_string(), Value::String(action.from_tier.clone()));
            entry.insert("to_tier".to_string(), Value::String(action.to_tier.clone()));
            arr.push(Value::Object(entry));
        }
    }
    extra.insert("capability_gate_applied".to_string(), Value::Bool(true));
}

/// Record the finalized routing trail and rebind to the final tier's model.
#[allow(clippy::too_many_arguments)]
pub fn bind(
    decision: &RoutingDecision,
    final_tier: &str,
    tiers: &HashMap<String, TierConfig>,
    extra: &mut HashMap<String, Value>,
    base_tier: &str,
    pre_confidence_tier: &str,
    gate: &ConfidenceGateResult,
    complaint: &ComplaintUpgradeResult,
    downgrade: &AntiDowngradeResult,
    previous_tier: Option<&str>,
    previous_route_class: Option<&str>,
    window: f64,
) -> RoutingDecision {
    let final_route_class = route_class_for_tier(final_tier);
    extra.insert("base_tier".to_string(), Value::String(base_tier.to_string()));
    extra.insert(
        "pre_confidence_tier".to_string(),
        Value::String(
            normalize_text_tier(pre_confidence_tier).unwrap_or_else(|| pre_confidence_tier.to_string()),
        ),
    );
    extra.insert("confidence_threshold".to_string(), json!(gate.threshold));
    extra.insert(
        "confidence_default_tier".to_string(),
        json!(&gate.default_tier),
    );
    extra.insert(
        "confidence_gate_applied".to_string(),
        Value::Bool(gate.applied),
    );
    extra.insert("final_tier".to_string(), Value::String(final_tier.to_string()));
    extra.insert("final_route_class".to_string(), json!(&final_route_class));
    extra.insert(
        "complaint_detected".to_string(),
        Value::Bool(!complaint.terms.is_empty()),
    );
    extra.insert("complaint_terms".to_string(), json!(&complaint.terms));
    extra.insert(
        "complaint_upgrade_applied".to_string(),
        Value::Bool(complaint.applied),
    );
    extra.insert(
        "complaint_upgrade_steps".to_string(),
        Value::from(complaint.steps),
    );
    extra.insert(
        "complaint_upgrade_max_chars".to_string(),
        Value::from(complaint.max_chars),
    );
    extra.insert(
        "anti_downgrade_applied".to_string(),
        Value::Bool(downgrade.applied),
    );
    extra.insert(
        "previous_tier".to_string(),
        json!(previous_tier.map(|t| normalize_text_tier(t).unwrap_or_else(|| t.to_string()))),
    );
    extra.insert(
        "previous_route_class".to_string(),
        json!(previous_route_class),
    );
    extra.insert("kv_cache_window_seconds".to_string(), json!(window));

    let model = tiers
        .get(final_tier)
        .map(|t| t.model.clone())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| decision.model.clone());

    RoutingDecision {
        tier: final_tier.to_string(),
        model,
        confidence: decision.confidence,
        source: decision.source.clone(),
    }
}

/// The minimum thinking mode a tier requires.
fn min_thinking_mode_for_tier(tier: Option<&str>) -> Option<&'static str> {
    let tier = tier.and_then(normalize_text_tier);
    match tier.as_deref() {
        Some(HIGHEST_TEXT_TIER) => Some("T3"),
        Some("c2") => Some("T2"),
        Some(DEFAULT_TEXT_TIER) => Some("T1"),
        _ => None,
    }
}

/// Promote the thinking mode to at least `minimum`.
fn promote_thinking_mode(current: Option<&str>, minimum: Option<&str>) -> Option<String> {
    let minimum = match minimum {
        Some(m) => m,
        None => return current.map(|s| s.to_string()),
    };
    let current = match current {
        Some(c) if THINKING_MODE_ORDER.iter().any(|(t, _)| *t == c) => c,
        _ => return Some(minimum.to_string()),
    };
    let cur_rank = THINKING_MODE_ORDER
        .iter()
        .find(|(t, _)| *t == current)
        .map(|(_, r)| *r)
        .unwrap_or(0);
    let min_rank = THINKING_MODE_ORDER
        .iter()
        .find(|(t, _)| *t == minimum)
        .map(|(_, r)| *r)
        .unwrap_or(0);
    if cur_rank < min_rank {
        Some(minimum.to_string())
    } else {
        Some(current.to_string())
    }
}

/// Keep controller output consistent with the final tier's overrides.
pub fn reconcile_controller_with_final_tier(
    thinking_mode: Option<String>,
    prompt_policy: Option<String>,
    extra: &mut HashMap<String, Value>,
) -> (Option<String>, Option<String>) {
    let final_tier_raw = extra.get("final_tier").and_then(|v| v.as_str());
    let final_tier = final_tier_raw
        .and_then(normalize_text_tier)
        .or_else(|| final_tier_raw.map(|s| s.to_string()));
    let base_tier_raw = extra.get("base_tier").and_then(|v| v.as_str());
    let base_tier = base_tier_raw
        .and_then(normalize_text_tier)
        .or_else(|| base_tier_raw.map(|s| s.to_string()));

    let Some(final_tier) = final_tier else {
        return (thinking_mode, prompt_policy);
    };
    if final_tier == base_tier.unwrap_or_default() {
        return (thinking_mode, prompt_policy);
    }

    let original_thinking = thinking_mode.clone();
    let original_prompt = prompt_policy.clone();

    let promoted = promote_thinking_mode(
        thinking_mode.as_deref(),
        min_thinking_mode_for_tier(Some(&final_tier)),
    );
    let thinking_mode = promoted.or(thinking_mode);

    let mut prompt_policy = prompt_policy;
    if prompt_policy.as_deref() == Some("P0")
        && (final_tier == "c2" || final_tier == HIGHEST_TEXT_TIER || extra.get("complaint_detected").is_some())
    {
        prompt_policy = Some("P1".to_string());
    }

    let (thinking_mode, prompt_policy) = match (thinking_mode, prompt_policy) {
        (Some(tm), Some(pp)) => {
            let (tm, pp) = normalize_decisions(&tm, &pp);
            (Some(tm), Some(pp))
        }
        other => other,
    };

    if thinking_mode != original_thinking || prompt_policy != original_prompt {
        extra
            .entry("base_thinking_mode".to_string())
            .or_insert_with(|| json!(&original_thinking));
        extra
            .entry("base_prompt_policy".to_string())
            .or_insert_with(|| json!(&original_prompt));
        extra.insert("thinking_mode".to_string(), json!(&thinking_mode));
        extra.insert("prompt_policy".to_string(), json!(&prompt_policy));
        extra.insert("controller_reconciled".to_string(), Value::Bool(true));
    } else {
        extra
            .entry("controller_reconciled".to_string())
            .or_insert_with(|| Value::Bool(false));
    }

    (thinking_mode, prompt_policy)
}

/// Constraint pass over thinking mode / prompt policy, mirroring the
/// squilla-router controller's `normalize_decisions`.
fn normalize_decisions(thinking_mode: &str, prompt_policy: &str) -> (String, String) {
    match thinking_mode {
        "T0" => (thinking_mode.to_string(), "P0".to_string()),
        _ => (thinking_mode.to_string(), prompt_policy.to_string()),
    }
}

/// The minimum tier a turn with this much material context may run on.
pub fn large_context_min_tier(material_tokens: u64, context_window_tokens: u64) -> Option<String> {
    let ratio_floor = (context_window_tokens as f64 * LARGE_CONTEXT_T3_CONTEXT_RATIO) as u64;
    if material_tokens >= LARGE_CONTEXT_T3_FLOOR_TOKENS || material_tokens >= ratio_floor {
        return Some(HIGHEST_TEXT_TIER.to_string());
    }
    if material_tokens >= LARGE_CONTEXT_T2_FLOOR_TOKENS {
        return Some("c2".to_string());
    }
    None
}

/// Floor the routed tier for turns carrying large material contexts.
pub fn large_context_floor(
    decision: &RoutingDecision,
    tiers: &HashMap<String, TierConfig>,
    valid_tiers: &[String],
    material_tokens: u64,
    context_window_tokens: u64,
    extra: Option<&mut HashMap<String, Value>>,
    metadata_updates: &mut HashMap<String, Value>,
) -> RoutingDecision {
    if !valid_tiers.contains(&decision.tier) {
        return decision.clone();
    }
    let min_tier = large_context_min_tier(material_tokens, context_window_tokens);
    let Some(min_tier) = min_tier else {
        return decision.clone();
    };
    if !valid_tiers.contains(&min_tier) {
        return decision.clone();
    }
    if tier_index_in(&decision.tier, valid_tiers) >= tier_index_in(&min_tier, valid_tiers) {
        return decision.clone();
    }

    let model = tiers
        .get(&min_tier)
        .map(|t| t.model.clone())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| decision.model.clone());

    metadata_updates.insert(
        "large_context_floor_from_tier".to_string(),
        Value::String(decision.tier.clone()),
    );
    metadata_updates.insert(
        "large_context_material_tokens".to_string(),
        Value::from(material_tokens),
    );

    if let Some(extra) = extra {
        extra
            .entry("base_tier".to_string())
            .or_insert_with(|| Value::String(decision.tier.clone()));
        extra.insert(
            "large_context_floor_applied".to_string(),
            Value::Bool(true),
        );
        extra.insert(
            "large_context_floor_from_tier".to_string(),
            Value::String(decision.tier.clone()),
        );
        extra.insert(
            "large_context_floor_min_tier".to_string(),
            Value::String(min_tier.clone()),
        );
        extra.insert(
            "large_context_material_tokens".to_string(),
            Value::from(material_tokens),
        );
        extra.insert(
            "large_context_pre_floor_source".to_string(),
            Value::String(decision.source.clone()),
        );
        extra.insert(
            "final_tier".to_string(),
            Value::String(min_tier.clone()),
        );
        extra.insert(
            "final_route_class".to_string(),
            json!(route_class_for_tier(&min_tier)),
        );
    }

    RoutingDecision {
        tier: min_tier,
        model,
        confidence: decision.confidence,
        source: "large_context_floor".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Budget gate (additive, default-off)
// ---------------------------------------------------------------------------

/// Session-spend signal + config for the budget gate.
#[derive(Debug, Clone)]
pub struct BudgetGateInput {
    /// `warn` | `cap`.
    pub action: String,
    /// Spend limit in USD.
    pub limit_usd: f64,
    /// Accumulated billed/estimated spend, or `None` when unknown.
    pub spend_usd: Option<f64>,
    /// Forward price estimate for the turn.
    pub estimate_usd: Option<f64>,
    /// Tier to cap to when over limit (cap mode).
    pub cap_tier: Option<String>,
    /// `billed` | `estimate` | `estimate_mixed` | `none` | `unknown`.
    pub spend_source: String,
    /// Session key for observability.
    pub session_key: Option<String>,
}

/// Result of the budget gate.
#[derive(Debug, Clone, Default)]
pub struct BudgetGateResult {
    /// The tier after the gate.
    pub tier: String,
    /// `off` | `suspended` | `under_limit` | `warn` | `cap`.
    pub outcome: String,
    pub spend_usd: Option<f64>,
    pub projected_usd: Option<f64>,
    pub limit_usd: Option<f64>,
    pub action: String,
    pub from_tier: String,
    pub spend_source: String,
    pub session_key: Option<String>,
}

/// Warn or cap when accumulated session spend crosses the configured limit.
///
/// `None` input -> `off` (byte-identical no-op). Unknown spend -> `suspended`.
/// The gate can only hold or lower the tier, never raise it.
pub fn budget_gate(
    tier: &str,
    valid_tiers: &[String],
    budget: Option<&BudgetGateInput>,
) -> BudgetGateResult {
    let Some(budget) = budget else {
        return BudgetGateResult {
            tier: tier.to_string(),
            outcome: "off".to_string(),
            ..Default::default()
        };
    };
    let Some(spend) = budget.spend_usd else {
        return BudgetGateResult {
            tier: tier.to_string(),
            outcome: "suspended".to_string(),
            limit_usd: Some(budget.limit_usd),
            action: budget.action.clone(),
            spend_source: budget.spend_source.clone(),
            session_key: budget.session_key.clone(),
            ..Default::default()
        };
    };
    let projected = spend + budget.estimate_usd.unwrap_or(0.0);

    if projected <= budget.limit_usd {
        return BudgetGateResult {
            tier: tier.to_string(),
            outcome: "under_limit".to_string(),
            spend_usd: Some(spend),
            projected_usd: Some(projected),
            limit_usd: Some(budget.limit_usd),
            action: budget.action.clone(),
            spend_source: budget.spend_source.clone(),
            session_key: budget.session_key.clone(),
            from_tier: String::new(),
        };
    }

    if budget.action == "cap" {
        let target = budget.cap_tier.as_deref().and_then(normalize_text_tier);
        if let Some(target) = target {
            if valid_tiers.contains(&target) && tier_index_in(&target, valid_tiers) < tier_index_in(tier, valid_tiers)
            {
                return BudgetGateResult {
                    tier: target,
                    outcome: "cap".to_string(),
                    spend_usd: Some(spend),
                    projected_usd: Some(projected),
                    limit_usd: Some(budget.limit_usd),
                    action: "cap".to_string(),
                    from_tier: tier.to_string(),
                    spend_source: budget.spend_source.clone(),
                    session_key: budget.session_key.clone(),
                };
            }
        }
        // No cap target strictly below: degrade to warn, never raise.
        return BudgetGateResult {
            tier: tier.to_string(),
            outcome: "warn".to_string(),
            spend_usd: Some(spend),
            projected_usd: Some(projected),
            limit_usd: Some(budget.limit_usd),
            action: "warn".to_string(),
            from_tier: tier.to_string(),
            spend_source: budget.spend_source.clone(),
            session_key: budget.session_key.clone(),
        };
    }

    BudgetGateResult {
        tier: tier.to_string(),
        outcome: "warn".to_string(),
        spend_usd: Some(spend),
        projected_usd: Some(projected),
        limit_usd: Some(budget.limit_usd),
        action: "warn".to_string(),
        from_tier: tier.to_string(),
        spend_source: budget.spend_source.clone(),
        session_key: budget.session_key.clone(),
    }
}

/// Append the budget gate's action to the routing trail (warn/cap only).
pub fn record_budget_gate_trail(extra: &mut HashMap<String, Value>, result: &BudgetGateResult) {
    if result.outcome != "warn" && result.outcome != "cap" {
        return;
    }
    let mut entry = serde_json::Map::new();
    entry.insert("stage".to_string(), Value::String("budget_gate".to_string()));
    entry.insert("rule".to_string(), Value::String(result.outcome.clone()));
    entry.insert("spend_usd".to_string(), json!(result.spend_usd));
    entry.insert("limit_usd".to_string(), json!(result.limit_usd));
    entry.insert("spend_source".to_string(), Value::String(result.spend_source.clone()));
    if result.outcome == "cap" {
        entry.insert("from_tier".to_string(), Value::String(result.from_tier.clone()));
        entry.insert("to_tier".to_string(), Value::String(result.tier.clone()));
    }
    let trail = extra
        .entry("routing_trail".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(arr) = trail.as_array_mut() {
        arr.push(Value::Object(entry));
    }
    extra.insert("budget_gate_applied".to_string(), Value::Bool(true));
    extra.insert(
        "budget_gate_outcome".to_string(),
        Value::String(result.outcome.clone()),
    );
}

/// Apply a [`budget_gate`] result to the decision + turn metadata.
///
/// `off`/`suspended`/`under_limit` are complete no-ops. Only `warn`/`cap`
/// record observability metadata; only `cap` rebinds the model.
pub fn apply_budget_gate(
    decision: &RoutingDecision,
    result: &BudgetGateResult,
    tiers: &HashMap<String, TierConfig>,
    extra: Option<&mut HashMap<String, Value>>,
    metadata_updates: &mut HashMap<String, Value>,
) -> RoutingDecision {
    if result.outcome != "warn" && result.outcome != "cap" {
        return decision.clone();
    }

    metadata_updates.insert("router_budget_applied".to_string(), Value::Bool(true));
    metadata_updates.insert(
        "router_budget_outcome".to_string(),
        Value::String(result.outcome.clone()),
    );
    metadata_updates.insert(
        "router_budget_action".to_string(),
        Value::String(result.action.clone()),
    );
    metadata_updates.insert("router_budget_limit_usd".to_string(), json!(result.limit_usd));
    metadata_updates.insert(
        "router_budget_spend_source".to_string(),
        Value::String(result.spend_source.clone()),
    );
    if let Some(spend) = result.spend_usd {
        metadata_updates.insert("router_budget_spend_usd".to_string(), Value::from(spend));
    }
    if let (Some(projected), Some(spend)) = (result.projected_usd, result.spend_usd) {
        if projected != spend {
            metadata_updates.insert("router_budget_projected_usd".to_string(), Value::from(projected));
        }
    }
    if let Some(extra) = extra {
        record_budget_gate_trail(extra, result);
    }

    if result.outcome == "cap" {
        metadata_updates.insert(
            "router_budget_from_tier".to_string(),
            Value::String(result.from_tier.clone()),
        );
        metadata_updates.insert(
            "router_budget_to_tier".to_string(),
            Value::String(result.tier.clone()),
        );
        let model = tiers
            .get(&result.tier)
            .map(|t| t.model.clone())
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| decision.model.clone());
        if let Some(extra) = extra {
            extra.insert(
                "final_tier".to_string(),
                Value::String(result.tier.clone()),
            );
            extra.insert(
                "final_route_class".to_string(),
                json!(route_class_for_tier(&result.tier)),
            );
        }
        return RoutingDecision {
            tier: result.tier.clone(),
            model,
            confidence: decision.confidence,
            source: "budget_cap".to_string(),
        };
    }

    // warn: tier unchanged.
    decision.clone()
}

// ---------------------------------------------------------------------------
// Provider mismatch (flag-only by default; veto mode rebinds)
// ---------------------------------------------------------------------------

/// Flag-only assessment of the routed tier's provider vs the active one.
#[derive(Debug, Clone)]
pub struct ProviderMismatchOutcome {
    /// `skipped` | `match` | `cross_provider` | `mismatch`.
    pub outcome: String,
    pub routed_provider: Option<String>,
    pub tier_provider: String,
    pub tier_model: String,
    pub active_provider: String,
}

/// Assess the routed tier's provider; never vetoes or alters the decision.
pub fn provider_mismatch(
    tiers: &HashMap<String, TierConfig>,
    tier_name: &str,
    routing_applied: bool,
    active_provider: &str,
    cross_provider_tiers: bool,
) -> ProviderMismatchOutcome {
    if !routing_applied {
        return ProviderMismatchOutcome {
            outcome: "skipped".to_string(),
            routed_provider: None,
            tier_provider: String::new(),
            tier_model: String::new(),
            active_provider: String::new(),
        };
    }
    let tier = tiers.get(tier_name).cloned().unwrap_or_default();
    let routed_provider = if tier.provider.is_empty() {
        None
    } else {
        Some(tier.provider.to_lowercase())
    };
    let active = active_provider.trim().to_lowercase();
    if tier.provider.is_empty() || active.is_empty() {
        return ProviderMismatchOutcome {
            outcome: "match".to_string(),
            routed_provider,
            tier_provider: tier.provider.clone(),
            tier_model: tier.model.clone(),
            active_provider: active,
        };
    }
    if tier.provider.to_lowercase() == active {
        return ProviderMismatchOutcome {
            outcome: "match".to_string(),
            routed_provider,
            tier_provider: tier.provider,
            tier_model: tier.model,
            active_provider: active,
        };
    }
    if cross_provider_tiers {
        return ProviderMismatchOutcome {
            outcome: "cross_provider".to_string(),
            routed_provider,
            tier_provider: tier.provider,
            tier_model: tier.model,
            active_provider: active,
        };
    }
    ProviderMismatchOutcome {
        outcome: "mismatch".to_string(),
        routed_provider,
        tier_provider: tier.provider,
        tier_model: tier.model,
        active_provider: active,
    }
}

/// Rebind decision for `tier_provider_mismatch = "veto"`.
#[derive(Debug, Clone, Default)]
pub struct ProviderMismatchVeto {
    pub applied: bool,
    pub from_tier: String,
    pub to_tier: String,
}

/// Pick the rebind target when a provider mismatch must be vetoed.
pub fn provider_mismatch_veto(
    tiers: &HashMap<String, TierConfig>,
    tier_name: &str,
    valid_tiers: &[String],
    routing_applied: bool,
    active_provider: &str,
    cross_provider_tiers: bool,
    default_tier: Option<&str>,
) -> ProviderMismatchVeto {
    let outcome = provider_mismatch(
        tiers,
        tier_name,
        routing_applied,
        active_provider,
        cross_provider_tiers,
    );
    if outcome.outcome != "mismatch" {
        return ProviderMismatchVeto::default();
    }

    let current = normalize_text_tier(tier_name).unwrap_or_else(|| tier_name.to_string());
    let idx = tier_index_in(&current, valid_tiers);
    let active = active_provider.trim().to_lowercase();

    let executes_on_active = |name: &str| -> bool {
        let tier = tiers.get(name).cloned().unwrap_or_default();
        tier.provider.is_empty() || tier.provider.to_lowercase() == active
    };

    if idx >= 0 {
        let mut candidates: Vec<(&String, i32)> = valid_tiers
            .iter()
            .filter(|name| **name != current && executes_on_active(name))
            .map(|name| (name, tier_index_in(name, valid_tiers)))
            .collect();
        candidates.sort_by_key(|(_, rank)| ((*rank - idx).abs(), *rank));
        if let Some((name, _)) = candidates.first() {
            return ProviderMismatchVeto {
                applied: true,
                from_tier: current,
                to_tier: (**name).clone(),
            };
        }
    }

    let fallback = default_tier.and_then(normalize_text_tier);
    if let Some(fallback) = fallback {
        if valid_tiers.contains(&fallback) && fallback != current {
            return ProviderMismatchVeto {
                applied: true,
                from_tier: current,
                to_tier: fallback,
            };
        }
    }
    ProviderMismatchVeto {
        applied: false,
        from_tier: current,
        to_tier: String::new(),
    }
}

/// Append a veto rebind to the routing trail (only when applied).
pub fn record_provider_mismatch_veto_trail(
    extra: &mut HashMap<String, Value>,
    veto: &ProviderMismatchVeto,
) {
    if !veto.applied {
        return;
    }
    let mut entry = serde_json::Map::new();
    entry.insert("stage".to_string(), Value::String("provider_mismatch".to_string()));
    entry.insert("rule".to_string(), Value::String("veto_rebind".to_string()));
    entry.insert("from_tier".to_string(), Value::String(veto.from_tier.clone()));
    entry.insert("to_tier".to_string(), Value::String(veto.to_tier.clone()));
    let trail = extra
        .entry("routing_trail".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(arr) = trail.as_array_mut() {
        arr.push(Value::Object(entry));
    }
    extra.insert(
        "provider_mismatch_veto_applied".to_string(),
        Value::Bool(true),
    );
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Classifier output plus turn facts, as plain data.
#[derive(Debug, Clone)]
pub struct PolicyInputs {
    /// The classified routing decision.
    pub decision: RoutingDecision,
    /// The turn's user message (for complaint detection).
    pub message: String,
    /// The router configuration.
    pub router_cfg: RouterConfig,
    /// The tier configuration map.
    pub tiers: HashMap<String, TierConfig>,
    /// The canonical tiers configured for this deployment.
    pub valid_tiers: Vec<String>,
    /// The routing history entries.
    pub routing_history: Option<Vec<HashMap<String, Value>>>,
    /// The turn's `routing_extra` dict (mutated in place; cloned by `run`).
    pub extra: Option<HashMap<String, Value>>,
    /// The controller's thinking mode, if any.
    pub thinking_mode: Option<String>,
    /// The controller's prompt policy, if any.
    pub prompt_policy: Option<String>,
    /// Whether the preference stages (gate/upgrade/hold) run at all.
    pub history_strategy: bool,
    /// Estimated material context tokens on the turn.
    pub material_estimated_tokens: u64,
    /// The context window of the routed model, in tokens.
    pub context_window_tokens: u64,
    /// Injectable clock (epoch-independent monotonic seconds).
    pub now: Option<f64>,
    /// Whether the turn carries an image.
    pub turn_has_image: bool,
    /// Definite capability facts per tier.
    pub tier_capabilities: Option<HashMap<String, TierCapability>>,
    /// On-device confidence-gate calibration.
    pub calibration: Option<CalibrationState>,
    /// Session-spend budget gate signal.
    pub budget: Option<BudgetGateInput>,
}

/// The output of the routing policy engine.
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// The final routing decision.
    pub decision: RoutingDecision,
    /// The reconciled thinking mode.
    pub thinking_mode: Option<String>,
    /// The reconciled prompt policy.
    pub prompt_policy: Option<String>,
    /// Turn-metadata updates destined for `ctx.metadata`.
    pub metadata_updates: HashMap<String, Value>,
    /// The (possibly mutated) `routing_extra` map. The caller owns the input;
    /// write this back to `PolicyInputs.extra` to observe in-place semantics.
    pub extra: Option<HashMap<String, Value>>,
}

/// Runs the post-classifier stages in the exact legacy order.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoutingPolicyEngine;

impl RoutingPolicyEngine {
    /// Create a new routing policy engine.
    pub fn new() -> Self {
        Self
    }

    /// Run the full policy pipeline over the inputs.
    pub fn run(&self, inputs: &PolicyInputs) -> PolicyResult {
        let mut decision = inputs.decision.clone();
        let mut thinking_mode = inputs.thinking_mode.clone();
        let mut prompt_policy = inputs.prompt_policy.clone();
        let mut metadata_updates: HashMap<String, Value> = HashMap::new();
        let mut extra = inputs.extra.clone();

        if inputs.history_strategy && extra.is_some() {
            let extra_mut = extra.as_mut().unwrap();
            decision = self.finalize(inputs, extra_mut);
            let (tm, pp) = reconcile_controller_with_final_tier(thinking_mode, prompt_policy, extra_mut);
            thinking_mode = tm;
            prompt_policy = pp;
        }

        decision = large_context_floor(
            &decision,
            &inputs.tiers,
            &inputs.valid_tiers,
            inputs.material_estimated_tokens,
            inputs.context_window_tokens,
            extra.as_mut(),
            &mut metadata_updates,
        );
        if decision.source == "large_context_floor" && extra.is_some() {
            let extra_mut = extra.as_mut().unwrap();
            let (tm, pp) = reconcile_controller_with_final_tier(thinking_mode, prompt_policy, extra_mut);
            thinking_mode = tm;
            prompt_policy = pp;
        }

        // Budget gate runs last: it can only hold or lower the tier.
        if inputs.budget.is_some() {
            let budget_result = budget_gate(
                &decision.tier,
                &inputs.valid_tiers,
                inputs.budget.as_ref(),
            );
            decision = apply_budget_gate(
                &decision,
                &budget_result,
                &inputs.tiers,
                extra.as_mut(),
                &mut metadata_updates,
            );
        }

        PolicyResult {
            decision,
            thinking_mode,
            prompt_policy,
            metadata_updates,
            extra,
        }
    }

    /// Run the preference stages (confidence gate, complaint upgrade,
    /// anti-downgrade, capability gate) and bind the final tier.
    fn finalize(
        &self,
        inputs: &PolicyInputs,
        extra: &mut HashMap<String, Value>,
    ) -> RoutingDecision {
        let decision = &inputs.decision;
        let base_tier = normalize_text_tier(&decision.tier).unwrap_or_else(|| decision.tier.clone());
        let mut final_tier = base_tier.clone();

        let base_route_class = extra
            .get("route_class")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| route_class_for_tier(&base_tier));
        if let Some(rc) = base_route_class {
            extra.insert("route_class".to_string(), Value::String(rc.clone()));
            extra
                .entry("top1_label".to_string())
                .or_insert_with(|| Value::String(rc));
        }

        let pre_confidence_tier = final_tier.clone();
        let gate = confidence_gate(
            &final_tier,
            decision.confidence,
            &inputs.router_cfg,
            &inputs.valid_tiers,
            &inputs.tiers,
            inputs.calibration.as_ref(),
        );
        final_tier = gate.tier.clone();

        let now = inputs.now.unwrap_or_else(monotonic_now);
        let window = inputs.router_cfg.kv_cache_anti_downgrade_window_seconds;
        let previous_entry = match &inputs.routing_history {
            Some(history) => previous_final_entry(history, now, window),
            None => None,
        };
        let previous_tier = previous_final_tier(previous_entry);
        let previous_route_class = previous_entry
            .and_then(|e| e.get("final_route_class").or_else(|| e.get("route_class")))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let complaint = complaint_upgrade(
            &final_tier,
            &inputs.message,
            &inputs.router_cfg,
            &inputs.valid_tiers,
            Some(&pre_confidence_tier),
            previous_tier.as_deref(),
        );
        final_tier = complaint.tier.clone();

        let downgrade = anti_downgrade(
            &final_tier,
            &inputs.router_cfg,
            &inputs.valid_tiers,
            previous_tier.as_deref(),
        );
        final_tier = downgrade.tier.clone();

        let gate_capabilities = capability_gate(
            &final_tier,
            &inputs.valid_tiers,
            inputs.tier_capabilities.as_ref(),
            inputs.turn_has_image,
            inputs.material_estimated_tokens,
        );
        record_capability_gate_trail(extra, &gate_capabilities);
        final_tier = gate_capabilities.tier.clone();

        bind(
            decision,
            &final_tier,
            &inputs.tiers,
            extra,
            &base_tier,
            &pre_confidence_tier,
            &gate,
            &complaint,
            &downgrade,
            previous_tier.as_deref(),
            previous_route_class.as_deref(),
            window,
        )
    }
}

// Re-export submodules' public types at the routing module root.
pub use calibration::CalibrationState;
pub use health_ledger::{ProviderFailureKind, ProviderHealthLedger};
pub use selector::{ModelSelector, ProviderConfig, SelectorConfig};
