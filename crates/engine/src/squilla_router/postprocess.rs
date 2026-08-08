//! Six-layer post-processing pipeline for the Phase 3 router.
//!
//! Ports `runtime_src/src/router/inference/postprocess.py::apply_postprocess`
//! and the predictor helpers it composes: margin upgrade, aux-head downgrade,
//! R1 rescue, under-routing safety, flag overrides, conversation-depth context
//! rules, the sticky tier, and thinking-mode / prompt-policy / model derivation.

use crate::squilla_router::SquillaRouterError;
use crate::squilla_router::config::RouterConfig;
use crate::squilla_router::features::{
    ContextMetadata, FeatureInput, PrevRouteDecision, ROUTE_CLASSES, route_class_idx,
};
use crate::squilla_router::flags::{Flags, compute_flags};

/// The post-process outcome for a single routing request.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalDecision {
    /// Final route class (`R0`..`R3`) after all six layers.
    pub route_class: String,
    /// Tier id mapped from the final route class.
    pub tier: String,
    /// Top-1 minus top-2 fused probability.
    pub margin: f64,
    /// Expected route index (`sum(i * fused[i])`).
    pub difficulty_score: f64,
    /// The five routing flags computed from the request text.
    pub flags: Flags,
    /// Derived thinking mode (`T0`..`T3`).
    pub thinking_mode: String,
    /// Derived prompt policy (`P0`..`P2`).
    pub prompt_policy: String,
    /// First model id in the tier registry for `tier`.
    pub selected_model: String,
    /// Whether the aux-head downgrade gate fired.
    pub aux_downgrade_applied: bool,
    /// Whether the sticky tier blocked a downgrade.
    pub sticky_applied: bool,
}

/// Run the six-layer post-processing pipeline over the fused head probabilities.
///
/// `fused_probs` is the fused 4-class probability vector in route-class index
/// order (`R0`..`R3`); `aux_probs`, when present, is the aux-head 4-class
/// vector in `[initial, maintain, upgrade, downgrade]` order.
pub fn apply_postprocess(
    fused_probs: [f64; 4],
    aux_probs: Option<[f64; 4]>,
    request: &FeatureInput,
    config: &RouterConfig,
) -> Result<FinalDecision, SquillaRouterError> {
    if fused_probs.iter().any(|p| !p.is_finite()) {
        return Err(SquillaRouterError::InvalidProbabilities(format!(
            "fused_probs contains a non-finite entry: {fused_probs:?}"
        )));
    }

    // Base prediction: argmax (leftmost wins ties), margin, difficulty.
    let idx = argmax_leftmost(&fused_probs);
    let margin = top2_margin(&fused_probs);
    let difficulty =
        fused_probs[0] * 0.0 + fused_probs[1] * 1.0 + fused_probs[2] * 2.0 + fused_probs[3] * 3.0;

    // Layer 1: margin upgrade; when it fires, the aux downgrade is suppressed.
    let pre_upgrade_idx = idx;
    let idx = apply_margin_upgrade(idx, margin, config);
    let margin_upgraded = idx != pre_upgrade_idx;
    let mut route = ROUTE_CLASSES[idx].to_string();

    // Layer 2: aux-head downgrade gate (suppressed after a margin upgrade).
    let mut aux_downgrade_applied = false;
    if !margin_upgraded {
        let (r, applied) = apply_aux_downgrade(&route, aux_probs, config);
        route = r;
        aux_downgrade_applied = applied;
    }

    // Layer 3: R1 rescue (safe upward promotion only, never R2 -> R1).
    route = apply_r1_rescue(&route, &fused_probs, config);

    // Layer 4: under-routing safety net.
    route = apply_under_routing_safety(&route, &fused_probs, config);

    let flags_text = request
        .flags_text_override
        .as_deref()
        .unwrap_or(&request.current_user_text);
    let flags = compute_flags(
        flags_text,
        &config.flag_rules,
        request.context_metadata.as_ref(),
        config.context_rules.heavy_context_tokens,
    );

    // Layer 5a: flag-driven overrides.
    route = apply_flag_overrides(&route, &flags);

    // Layer 5b: conversation-depth context rules.
    route = apply_context_rules(&route, request.context_metadata.as_ref(), config);

    // Layer 6: KV-cache-aware sticky tier (blocks downgrades on short turns).
    let mut sticky_applied = false;
    if config.v4.sticky_tier.enabled
        && request.current_user_text.chars().count() <= config.v4.sticky_tier.max_user_len
    {
        if let Some(last) = request.prev_route_decisions.last() {
            let (r, applied) = apply_sticky_tier(&route, last);
            route = r;
            sticky_applied = applied;
        }
    }

    let mut thinking_mode = derive_thinking_mode(&route, margin, &flags, config);
    let mut prompt_policy = derive_prompt_policy(difficulty, margin, &flags, config);
    if route == "R0" && is_trivial_ack(flags_text) {
        thinking_mode = "T0".to_string();
        prompt_policy = "P0".to_string();
    }

    let tier = config
        .tier_mapping
        .get(&route)
        .cloned()
        .unwrap_or_else(|| "M".to_string());
    let selected_model = config
        .tier_registry
        .get(&tier)
        .and_then(|models| models.first())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());

    Ok(FinalDecision {
        route_class: route,
        tier,
        margin,
        difficulty_score: difficulty,
        flags,
        thinking_mode,
        prompt_policy,
        selected_model,
        aux_downgrade_applied,
        sticky_applied,
    })
}

/// Index of the maximum entry, preferring the leftmost element on ties.
fn argmax_leftmost(probs: &[f64; 4]) -> usize {
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

/// Top-1 minus top-2 probability.
fn top2_margin(probs: &[f64; 4]) -> f64 {
    let mut desc = probs.to_vec();
    desc.sort_by(|a, b| b.total_cmp(a));
    desc[0] - desc[1]
}

/// Promote the route one class when the margin is below the upgrade threshold.
fn apply_margin_upgrade(idx: usize, margin: f64, config: &RouterConfig) -> usize {
    if margin < config.thresholds.margin_upgrade && idx < 3 {
        idx + 1
    } else {
        idx
    }
}

/// Apply the aux-head downgrade gate, dropping one route class when the
/// conditional downgrade probability clears the configured threshold.
fn apply_aux_downgrade(
    route: &str,
    aux_probs: Option<[f64; 4]>,
    config: &RouterConfig,
) -> (String, bool) {
    if !config.v4.aux_downgrade.enabled {
        return (route.to_string(), false);
    }
    let Some(aux) = aux_probs else {
        return (route.to_string(), false);
    };
    let non_initial_mass = aux[1] + aux[2] + aux[3];
    let downgrade_prob = if non_initial_mass > 0.0 {
        aux[3] / non_initial_mass
    } else {
        0.0
    };
    if downgrade_prob < config.v4.aux_downgrade.threshold || route == "R0" {
        return (route.to_string(), false);
    }
    let idx = route_class_idx(route).unwrap_or(0).saturating_sub(1);
    let downgraded = ROUTE_CLASSES[idx].to_string();
    let changed = downgraded != route;
    (downgraded, changed)
}

/// Rescue R0 to R1 when R1 is a close second in the fused probabilities.
fn apply_r1_rescue(route: &str, fused: &[f64; 4], config: &RouterConfig) -> String {
    if route == "R0" && fused[0] - fused[1] < config.thresholds.r1_rescue.from_r0_max_gap {
        "R1".to_string()
    } else {
        route.to_string()
    }
}

/// Lift low routes to R2 when the heavy-class mass clears the safety floor.
fn apply_under_routing_safety(route: &str, fused: &[f64; 4], config: &RouterConfig) -> String {
    let idx = route_class_idx(route).unwrap_or(0);
    if idx < 2 && fused[2] + fused[3] > config.thresholds.under_routing_safety {
        "R2".to_string()
    } else {
        route.to_string()
    }
}

/// Apply the flag-driven route overrides (`high_risk`, `debug` + `long_context`,
/// and `repo_arch`).
fn apply_flag_overrides(route: &str, flags: &Flags) -> String {
    let mut idx = route_class_idx(route).unwrap_or(0);
    if flags.high_risk {
        idx = idx.max(2);
    }
    if flags.debug && flags.long_context {
        idx = idx.max(2);
    }
    if flags.repo_arch {
        idx = idx.max(1);
    }
    ROUTE_CLASSES[idx].to_string()
}

/// Lift the route to the deep-conversation minimum class once the turn count
/// passes the threshold.
fn apply_context_rules(
    route: &str,
    context: Option<&ContextMetadata>,
    config: &RouterConfig,
) -> String {
    let Some(ctx) = context else {
        return route.to_string();
    };
    if (ctx.turn_index as usize) >= config.context_rules.deep_conversation_threshold {
        let min_idx =
            route_class_idx(&config.context_rules.deep_conversation_min_class).unwrap_or(1);
        let idx = route_class_idx(route).unwrap_or(0).max(min_idx);
        ROUTE_CLASSES[idx].to_string()
    } else {
        route.to_string()
    }
}

/// Stick with the previous turn's route when it is strictly higher than the
/// current prediction (KV-cache-friendly downgrade blocker).
fn apply_sticky_tier(route: &str, last: &PrevRouteDecision) -> (String, bool) {
    let prev_idx = route_class_idx(&last.route_class).unwrap_or(0);
    let pred_idx = route_class_idx(route).unwrap_or(0);
    if prev_idx > pred_idx {
        (last.route_class.clone(), true)
    } else {
        (route.to_string(), false)
    }
}

/// Derive the thinking mode (`T0`..`T3`) from the route, margin, and flags.
fn derive_thinking_mode(route: &str, margin: f64, flags: &Flags, config: &RouterConfig) -> String {
    if route == "R3" {
        return "T3".to_string();
    }
    let t3 = &config.thinking_mode_rules.t3;
    if route_class_idx(route).unwrap_or(0) >= route_class_idx(&t3.min_class).unwrap_or(2) {
        for flag_name in &t3.flags {
            if flag_is_set(flags, flag_name) {
                return "T3".to_string();
            }
        }
    }
    let t0 = &config.thinking_mode_rules.t0;
    if route_class_idx(route).unwrap_or(0) <= route_class_idx(&t0.max_class).unwrap_or(0)
        && margin >= t0.min_margin
    {
        return "T0".to_string();
    }
    let t1 = &config.thinking_mode_rules.t1;
    if route_class_idx(route).unwrap_or(0) <= route_class_idx(&t1.max_class).unwrap_or(1)
        && margin >= t1.min_margin
    {
        return "T1".to_string();
    }
    "T2".to_string()
}

/// Derive the prompt policy (`P0`..`P2`) from the difficulty, margin, and flags.
fn derive_prompt_policy(
    difficulty: f64,
    margin: f64,
    flags: &Flags,
    config: &RouterConfig,
) -> String {
    let p2 = config.prompt_policies.p2.conditions.as_ref();
    let any_flags: Vec<String> = p2.and_then(|c| c.any_flag.clone()).unwrap_or_else(|| {
        vec![
            "high_risk".to_string(),
            "long_context".to_string(),
            "debug".to_string(),
            "strict_format".to_string(),
        ]
    });
    if any_flags.iter().any(|name| flag_is_set(flags, name)) {
        return "P2".to_string();
    }

    let p0 = config.prompt_policies.p0.conditions.as_ref();
    let max_diff = p0.and_then(|c| c.max_difficulty).unwrap_or(0.8);
    let min_margin = p0.and_then(|c| c.min_margin).unwrap_or(0.4);
    let no_flags: Vec<String> = p0.and_then(|c| c.no_flags.clone()).unwrap_or_else(|| {
        vec![
            "high_risk".to_string(),
            "strict_format".to_string(),
            "debug".to_string(),
        ]
    });
    let has_blocking_flag = no_flags.iter().any(|name| flag_is_set(flags, name));
    if difficulty <= max_diff && margin >= min_margin && !has_blocking_flag {
        "P0".to_string()
    } else {
        "P1".to_string()
    }
}

/// Look up a single routing flag by name.
fn flag_is_set(flags: &Flags, name: &str) -> bool {
    match name {
        "high_risk" => flags.high_risk,
        "long_context" => flags.long_context,
        "debug" => flags.debug,
        "repo_arch" => flags.repo_arch,
        "strict_format" => flags.strict_format,
        _ => false,
    }
}

/// Detect a trivial acknowledgement (short thank-you / okay / yes-no turn).
fn is_trivial_ack(text: &str) -> bool {
    let normalized = text.trim().to_lowercase();
    let normalized = normalized.trim_matches(|c: char| " \t\r\n.!?。！？,，;；:：".contains(c));
    matches!(
        normalized,
        "thanks"
            | "thank you"
            | "ok"
            | "okay"
            | "yes"
            | "no"
            | "收到"
            | "好的"
            | "谢谢"
            | "是的"
            | "不用了"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_YAML: &str = r#"route_classes: [R0, R1, R2, R3]

tier_mapping:
  R0: S
  R1: M
  R2: L
  R3: XL

tier_registry:
  S: [deepseek/deepseek-v4-flash]
  M: [deepseek/deepseek-v4-pro]
  L: [z-ai/glm-5.2]
  XL: [anthropic/claude-opus-4.8]

thresholds:
  margin_upgrade: 0.10
  high_confidence: 0.7
  r1_rescue:
    from_r0_max_gap: 0.10
  cascade_stage1_threshold: 0.4
  under_routing_safety: 0.45
  kv_cache_aware: true

flag_rules:
  high_risk:
    keywords_zh: [部署, 回滚, 生产]
    keywords_en: [deploy, production]
  debug:
    keywords: [error, bug, traceback]
    patterns: [FAILED]
  repo_arch:
    keywords: [repo, architecture]
  strict_format:
    keywords: [JSON, schema]
  long_context:
    char_threshold: 6000
    code_block_threshold: 1500
    log_block_threshold: 1500
    file_ref_threshold: 2

thinking_mode_rules:
  T0: {max_class: R0, min_margin: 0.5}
  T1: {max_class: R1, min_margin: 0.4}
  T2: {default: true}
  T3: {min_class: R2, flags: [debug, long_context, high_risk]}

prompt_policies:
  P0:
    hint_zh: "直接作答"
    hint_en: "Answer directly."
    conditions: {max_difficulty: 0.8, min_margin: 0.4, no_flags: [high_risk, strict_format, debug]}
  P1:
    hint_zh: ""
    hint_en: ""
  P2:
    hint_zh: "充分分析"
    hint_en: "Analyze thoroughly."
    conditions: {any_flag: [high_risk, long_context, debug, strict_format]}

context_rules:
  deep_conversation_threshold: 4
  heavy_context_tokens: 2000
  heavy_context_min_class: R1
  deep_conversation_min_class: R1

trajectory:
  delta_threshold: 0.3
  history_max_turns: 5

v4:
  aux_head_inference: false
  bge_model_name: BAAI/bge-small-zh-v1.5
  bge_backend: onnx
  bge_onnx_dir: bge_onnx
  pca_dim: 64
  feature_dim: 390
  history_user_max_turns: 4
  bge_truncate_tokens: 510
  aux_downgrade:
    enabled: false
    threshold: 0.55
  sticky_tier:
    enabled: false
    max_user_len: 200
"#;

    fn test_config() -> RouterConfig {
        RouterConfig::from_yaml_str(TEST_YAML).expect("TEST_YAML parses")
    }

    fn request(text: &str) -> FeatureInput {
        FeatureInput {
            current_user_text: text.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn argmax_breaks_ties_leftmost() {
        assert_eq!(argmax_leftmost(&[0.5, 0.5, 0.0, 0.0]), 0);
        assert_eq!(argmax_leftmost(&[0.1, 0.4, 0.4, 0.1]), 1);
    }

    #[test]
    fn margin_upgrade_promotes_and_suppresses_aux_downgrade() {
        let mut cfg = test_config();
        cfg.v4.aux_downgrade.enabled = true;
        let fused = [0.30, 0.35, 0.25, 0.10];
        let aux = Some([0.2, 0.2, 0.1, 0.5]); // high downgrade prob, but suppressed.
        let d = apply_postprocess(fused, aux, &request("hello world"), &cfg).unwrap();
        assert_eq!(d.route_class, "R2");
        assert_eq!(d.tier, "L");
        assert_eq!(d.selected_model, "z-ai/glm-5.2");
        assert!((d.margin - 0.05).abs() < 1e-9);
        assert!((d.difficulty_score - 1.15).abs() < 1e-9);
        assert!(!d.aux_downgrade_applied);
        assert!(!d.sticky_applied);
        assert_eq!(d.thinking_mode, "T2");
        assert_eq!(d.prompt_policy, "P1");
    }

    #[test]
    fn r1_rescue_promotes_close_r0() {
        let mut cfg = test_config();
        cfg.thresholds.margin_upgrade = 0.0; // disable the margin upgrade layer
        cfg.thresholds.r1_rescue.from_r0_max_gap = 0.20;
        let fused = [0.55, 0.40, 0.03, 0.02];
        let d = apply_postprocess(fused, None, &request("hello world"), &cfg).unwrap();
        assert_eq!(d.route_class, "R1");
        assert_eq!(d.tier, "M");
    }

    #[test]
    fn under_routing_safety_lifts_to_r2() {
        let cfg = test_config();
        let fused = [0.44, 0.10, 0.31, 0.15];
        let d = apply_postprocess(fused, None, &request("hello world"), &cfg).unwrap();
        assert_eq!(d.route_class, "R2");
        assert_eq!(d.tier, "L");
    }

    #[test]
    fn high_risk_flag_forces_r2() {
        let cfg = test_config();
        let fused = [0.90, 0.05, 0.03, 0.02];
        let d = apply_postprocess(
            fused,
            None,
            &request("please deploy to production now"),
            &cfg,
        )
        .unwrap();
        assert_eq!(d.route_class, "R2");
        assert!(d.flags.high_risk);
        assert_eq!(d.thinking_mode, "T3");
        assert_eq!(d.prompt_policy, "P2");
    }

    #[test]
    fn context_rules_lift_deep_conversation() {
        let cfg = test_config();
        let fused = [0.90, 0.05, 0.03, 0.02];
        let mut req = request("hello world");
        req.context_metadata = Some(ContextMetadata {
            turn_index: 5,
            ..Default::default()
        });
        let d = apply_postprocess(fused, None, &req, &cfg).unwrap();
        assert_eq!(d.route_class, "R1");
    }

    #[test]
    fn sticky_tier_blocks_downgrade() {
        let mut cfg = test_config();
        cfg.v4.sticky_tier.enabled = true;
        let fused = [0.90, 0.05, 0.03, 0.02];
        let mut req = request("hi");
        req.prev_route_decisions = vec![PrevRouteDecision {
            route_class: "R2".to_string(),
            difficulty: 1.5,
            margin: 0.3,
        }];
        let d = apply_postprocess(fused, None, &req, &cfg).unwrap();
        assert_eq!(d.route_class, "R2");
        assert!(d.sticky_applied);
        // Non-blocking: previous tier at or below the current prediction.
        req.prev_route_decisions = vec![PrevRouteDecision {
            route_class: "R0".to_string(),
            difficulty: 0.2,
            margin: 0.5,
        }];
        let d = apply_postprocess(fused, None, &req, &cfg).unwrap();
        assert_eq!(d.route_class, "R0");
        assert!(!d.sticky_applied);
    }

    #[test]
    fn aux_downgrade_math() {
        let mut cfg = test_config();
        cfg.v4.aux_downgrade.enabled = true;
        let (route, applied) = apply_aux_downgrade("R2", Some([0.2, 0.2, 0.1, 0.5]), &cfg);
        assert_eq!(route, "R1");
        assert!(applied);
        let (route, applied) = apply_aux_downgrade("R2", Some([0.2, 0.3, 0.1, 0.2]), &cfg);
        assert_eq!(route, "R2");
        assert!(!applied);
        // The gate never downgrades R0.
        let (route, applied) = apply_aux_downgrade("R0", Some([0.2, 0.2, 0.1, 0.5]), &cfg);
        assert_eq!(route, "R0");
        assert!(!applied);
        // A disabled gate never downgrades.
        cfg.v4.aux_downgrade.enabled = false;
        let (route, applied) = apply_aux_downgrade("R2", Some([0.2, 0.2, 0.1, 0.5]), &cfg);
        assert_eq!(route, "R2");
        assert!(!applied);
    }

    #[test]
    fn aux_downgrade_pipeline_flag() {
        let mut cfg = test_config();
        cfg.v4.aux_downgrade.enabled = true;
        let fused = [0.05, 0.85, 0.06, 0.04];
        // High downgrade mass: R1 -> R0, then r1_rescue reinstates R1.
        let d = apply_postprocess(
            fused,
            Some([0.2, 0.2, 0.1, 0.5]),
            &request("hello world"),
            &cfg,
        )
        .unwrap();
        assert!(d.aux_downgrade_applied);
        assert_eq!(d.route_class, "R1");
        // Low downgrade mass: the gate does not fire.
        let d = apply_postprocess(
            fused,
            Some([0.2, 0.3, 0.1, 0.2]),
            &request("hello world"),
            &cfg,
        )
        .unwrap();
        assert!(!d.aux_downgrade_applied);
        assert_eq!(d.route_class, "R1");
    }

    #[test]
    fn thinking_and_prompt_derive_t0_p0() {
        let cfg = test_config();
        let fused = [0.90, 0.05, 0.03, 0.02];
        let d = apply_postprocess(fused, None, &request("hello world"), &cfg).unwrap();
        assert_eq!(d.route_class, "R0");
        assert_eq!(d.thinking_mode, "T0");
        assert_eq!(d.prompt_policy, "P0");
        assert_eq!(d.tier, "S");
        assert_eq!(d.selected_model, "deepseek/deepseek-v4-flash");
    }

    #[test]
    fn debug_flag_derives_p2() {
        let cfg = test_config();
        let fused = [0.90, 0.05, 0.03, 0.02];
        let d = apply_postprocess(
            fused,
            None,
            &request("got an error and a traceback FAILED"),
            &cfg,
        )
        .unwrap();
        assert!(d.flags.debug);
        assert_eq!(d.prompt_policy, "P2");
        // Debug alone does not lift the class on a high-margin R0.
        assert_eq!(d.route_class, "R0");
    }

    #[test]
    fn trivial_ack_forces_t0_p0_on_r0() {
        let cfg = test_config();
        let fused = [0.55, 0.40, 0.03, 0.02];
        let d = apply_postprocess(fused, None, &request("thank you."), &cfg).unwrap();
        assert_eq!(d.route_class, "R0");
        assert_eq!(d.thinking_mode, "T0");
        assert_eq!(d.prompt_policy, "P0");
    }

    #[test]
    fn is_trivial_ack_matches() {
        assert!(is_trivial_ack("  Thank You! "));
        assert!(is_trivial_ack("收到。"));
        assert!(is_trivial_ack("yes,"));
        assert!(!is_trivial_ack("please fix the bug"));
    }

    #[test]
    fn rejects_non_finite_fused_probs() {
        let cfg = test_config();
        let err = apply_postprocess([0.5, f64::NAN, 0.3, 0.2], None, &request("hello"), &cfg)
            .unwrap_err();
        assert!(matches!(err, SquillaRouterError::InvalidProbabilities(_)));
        let err = apply_postprocess(
            [0.5, f64::INFINITY, 0.3, 0.2],
            None,
            &request("hello"),
            &cfg,
        )
        .unwrap_err();
        assert!(matches!(err, SquillaRouterError::InvalidProbabilities(_)));
    }
}
