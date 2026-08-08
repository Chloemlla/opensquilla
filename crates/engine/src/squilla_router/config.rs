//! Runtime configuration loader for the SquillaRouter Phase 3 pipeline.
//!
//! Mirrors the authoritative schema
//! `src/opensquilla/squilla_router/models/v4.2_phase3_inference/router.runtime.yaml`
//! and the small bundle metadata in `inference_manifest.json`. Unknown YAML
//! keys (e.g. `tier_explanations`, `optuna`) are ignored by serde.

use crate::squilla_router::SquillaRouterError;
use std::collections::HashMap;
use std::path::Path;

/// Top-level routing configuration loaded from `router.runtime.yaml`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RouterConfig {
    /// The four route classes in index order (`["R0", "R1", "R2", "R3"]`).
    pub route_classes: Vec<String>,
    /// Route class id -> tier id (e.g. `R0` -> `S`).
    pub tier_mapping: HashMap<String, String>,
    /// Tier id -> model ids routed to that tier.
    pub tier_registry: HashMap<String, Vec<String>>,
    /// Decision thresholds consumed by the post-process layer.
    pub thresholds: Thresholds,
    /// Keyword/pattern sets that drive the five routing flags.
    pub flag_rules: FlagRules,
    /// Per-thinking-mode gating rules.
    pub thinking_mode_rules: ThinkingModeRules,
    /// Per-prompt-policy hints and gating conditions.
    pub prompt_policies: PromptPolicies,
    /// Conversation-depth and context-weight rules.
    pub context_rules: ContextRules,
    /// Trajectory smoothing parameters.
    pub trajectory: Trajectory,
    /// V4 Phase 3 inference bundle settings.
    pub v4: V4,
}

impl RouterConfig {
    /// Parse a `router.runtime.yaml` document from a string.
    pub fn from_yaml_str(s: &str) -> Result<Self, SquillaRouterError> {
        serde_yaml::from_str(s)
            .map_err(|e| SquillaRouterError::Config(format!("parse router.runtime.yaml: {e}")))
    }

    /// Read and parse a `router.runtime.yaml` file.
    pub fn from_file(path: &Path) -> Result<Self, SquillaRouterError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SquillaRouterError::Config(format!("read {}: {e}", path.display())))?;
        Self::from_yaml_str(&contents)
    }
}

/// Bundle metadata parsed from `inference_manifest.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Manifest {
    /// Fusion temperature used by the head aggregation.
    pub temperature: f64,
    /// Per-class alpha blending weights, `[R0, R1, R2, R3]`.
    pub per_class_alpha: [f64; 4],
}

impl Manifest {
    /// Read and parse an `inference_manifest.json` file.
    pub fn from_file(path: &Path) -> Result<Self, SquillaRouterError> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| SquillaRouterError::Config(format!("read {}: {e}", path.display())))?;
        serde_json::from_str(&contents)
            .map_err(|e| SquillaRouterError::Config(format!("parse inference_manifest.json: {e}")))
    }
}

/// Post-process decision thresholds.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Thresholds {
    /// Margin above which the predicted class is upgraded to the next tier.
    pub margin_upgrade: f64,
    /// Probability at which a route is treated as high confidence.
    pub high_confidence: f64,
    /// R1-rescue gate for an R0 outcome with a suspicious margin gap.
    pub r1_rescue: R1Rescue,
    /// Stage-1 cascade threshold.
    pub cascade_stage1_threshold: f64,
    /// Safety floor against under-routing.
    pub under_routing_safety: f64,
    /// Whether the KV-cache-aware downgrade gate is enabled.
    pub kv_cache_aware: bool,
}

/// R1-rescue gate parameters.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct R1Rescue {
    /// Maximum acceptable R0 margin gap before the R1 rescue fires.
    pub from_r0_max_gap: f64,
}

/// Keyword/pattern sets that drive the five routing flags.
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(default)]
pub struct FlagRules {
    /// High-risk keywords, grouped by language.
    pub high_risk: HighRiskRules,
    /// Debug-detection keywords and regex patterns.
    pub debug: DebugRules,
    /// Repository-architecture keywords.
    pub repo_arch: KeywordRules,
    /// Strict-format keywords.
    pub strict_format: KeywordRules,
    /// Long-context thresholds.
    pub long_context: LongContextRules,
}

/// High-risk keyword groups.
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(default)]
pub struct HighRiskRules {
    /// Chinese high-risk keywords.
    pub keywords_zh: Vec<String>,
    /// English high-risk keywords.
    pub keywords_en: Vec<String>,
}

/// Debug-detection rules.
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(default)]
pub struct DebugRules {
    /// Debug keywords.
    pub keywords: Vec<String>,
    /// Debug regex patterns, matched against the whole text.
    pub patterns: Vec<String>,
}

/// A single keyword list (used by `repo_arch` and `strict_format`).
#[derive(Debug, Clone, serde::Deserialize, Default)]
#[serde(default)]
pub struct KeywordRules {
    /// Keywords, any of which fires the flag.
    pub keywords: Vec<String>,
}

/// Long-context thresholds.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LongContextRules {
    /// Character threshold on the raw text.
    pub char_threshold: usize,
    /// Total matched code-block length threshold.
    pub code_block_threshold: usize,
    /// Total matched log-block length threshold.
    pub log_block_threshold: usize,
    /// File-path reference count threshold.
    pub file_ref_threshold: usize,
}

impl Default for LongContextRules {
    fn default() -> Self {
        Self {
            char_threshold: 6000,
            code_block_threshold: 1500,
            log_block_threshold: 1500,
            file_ref_threshold: 2,
        }
    }
}

/// Per-thinking-mode gating rules (`T0`..`T3`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct ThinkingModeRules {
    /// `T0` bounds.
    pub t0: ThinkingModeBounds,
    /// `T1` bounds.
    pub t1: ThinkingModeBounds,
    /// `T2` default flag.
    pub t2: ThinkingModeDefault,
    /// `T3` minimum class and flag set.
    pub t3: ThinkingModeFlags,
}

/// A `max_class` + `min_margin` gate (`T0`/`T1`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ThinkingModeBounds {
    /// Highest route class allowed in this mode.
    pub max_class: String,
    /// Minimum margin required to stay in this mode.
    pub min_margin: f64,
}

/// A single boolean default (`T2`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ThinkingModeDefault {
    /// Whether this mode is the default.
    pub default: bool,
}

/// A `min_class` + `flags` gate (`T3`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ThinkingModeFlags {
    /// Lowest route class allowed in this mode.
    pub min_class: String,
    /// Flags that must be set to enter this mode.
    pub flags: Vec<String>,
}

/// Per-prompt-policy hints and gating conditions (`P0`..`P2`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct PromptPolicies {
    /// `P0` policy.
    pub p0: PromptPolicy,
    /// `P1` policy.
    pub p1: PromptPolicy,
    /// `P2` policy.
    pub p2: PromptPolicy,
}

/// A single prompt policy.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PromptPolicy {
    /// Chinese prompt hint.
    pub hint_zh: String,
    /// English prompt hint.
    pub hint_en: String,
    /// Optional gating conditions.
    pub conditions: Option<PromptConditions>,
}

/// Gating conditions attached to a prompt policy.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PromptConditions {
    /// Maximum allowed difficulty.
    pub max_difficulty: Option<f64>,
    /// Minimum required margin.
    pub min_margin: Option<f64>,
    /// Flags that must all be clear.
    pub no_flags: Option<Vec<String>>,
    /// Flags of which at least one must be set.
    pub any_flag: Option<Vec<String>>,
}

/// Conversation-depth and context-weight rules.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContextRules {
    /// Prior-turn count that marks a "deep" conversation.
    pub deep_conversation_threshold: usize,
    /// Estimated accumulated context tokens considered "heavy".
    pub heavy_context_tokens: usize,
    /// Minimum route class for heavy contexts.
    pub heavy_context_min_class: String,
    /// Minimum route class for deep conversations.
    pub deep_conversation_min_class: String,
}

/// Trajectory smoothing parameters.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Trajectory {
    /// Margin delta below which the route is smoothed.
    pub delta_threshold: f64,
    /// Maximum prior-turn window used for trajectory history.
    pub history_max_turns: usize,
}

/// V4 Phase 3 inference bundle settings.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct V4 {
    /// Whether the aux head runs during inference.
    pub aux_head_inference: bool,
    /// BGE embedding model id.
    pub bge_model_name: String,
    /// BGE backend (`onnx` or `sentence_transformers`).
    pub bge_backend: String,
    /// Directory name for the BGE ONNX artifacts.
    pub bge_onnx_dir: String,
    /// PCA projection dimension for BGE embeddings.
    pub pca_dim: usize,
    /// Total assembled feature dimension.
    pub feature_dim: usize,
    /// Max prior user turns joined into the history channel.
    pub history_user_max_turns: usize,
    /// Token truncation for the BGE encoder.
    pub bge_truncate_tokens: usize,
    /// Aux-head downgrade gate.
    pub aux_downgrade: AuxDowngrade,
    /// Sticky-tier downgrade blocker.
    pub sticky_tier: StickyTier,
}

/// Aux-head downgrade gate settings.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AuxDowngrade {
    /// Whether the downgrade gate is active.
    pub enabled: bool,
    /// Minimum aux downgrade probability to trigger the gate.
    pub threshold: f64,
}

/// Sticky-tier settings.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StickyTier {
    /// Whether the sticky-tier blocker is active.
    pub enabled: bool,
    /// Max current user text length (chars) for the blocker to apply.
    pub max_user_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verbatim contents of `router.runtime.yaml`.
    const REAL_YAML: &str = r#"route_classes: [R0, R1, R2, R3]

tier_mapping:
  R0: S
  R1: M
  R2: L
  R3: XL

tier_registry:
  S: [deepseek/deepseek-v4-flash]    # Fast direct-answer tier for trivial turns and short acknowledgements.
  M: [deepseek/deepseek-v4-pro]      # Default general tier for normal Q&A, editing, and bounded coding tasks.
  L: [z-ai/glm-5.2]                  # Reasoning tier for debugging, multi-step analysis, and structured planning.
  XL: [anthropic/claude-opus-4.8]    # Highest Opus tier for architecture, high-risk work, and hard recovery turns.

tier_explanations:
  S:
    route_class: R0
    model: deepseek/deepseek-v4-flash
    intent: "Fast direct answers for trivial turns, acknowledgements, simple rewrites, and short factual responses."
    expected_controls: "No explicit thinking; P0 prompt hint may shorten responses."
  M:
    route_class: R1
    model: deepseek/deepseek-v4-pro
    intent: "General-purpose default for routine product, coding, writing, and comparison tasks with bounded complexity."
    expected_controls: "Light or default thinking; P1 prompt policy unless the task is clearly trivial or high risk."
  L:
    route_class: R2
    model: z-ai/glm-5.2
    intent: "Reasoning tier for debugging, multi-signal diagnosis, multi-step implementation plans, and non-trivial technical analysis."
    expected_controls: "High thinking; P1/P2 prompt policy depending on flags and difficulty."
  XL:
    route_class: R3
    model: anthropic/claude-opus-4.8
    intent: "Highest Opus tier for architecture design, cross-system tradeoffs, high-risk production decisions, and complex recovery turns."
    expected_controls: "High thinking; P1/P2 prompt policy depending on flags and difficulty."

thresholds:
  margin_upgrade: 0.10           # v4.1 P0a: tightened from 0.15 to reduce OOD over-routing
  high_confidence: 0.7
  r1_rescue:
    from_r0_max_gap: 0.10        # v4.1 P0a: tightened from 0.20 (sweep winner; +1.45 pp L2)
  cascade_stage1_threshold: 0.4
  under_routing_safety: 0.45
  kv_cache_aware: true

flag_rules:
  high_risk:
    keywords_zh: ["生产", "部署", "回滚", "迁移", "删除", "客户", "法务", "财务"]
    keywords_en: ["deploy", "rollback", "migration", "delete", "overwrite", "production", "customer-facing"]
  debug:
    keywords: ["error", "bug", "exception", "traceback", "failed", "root cause", "报错", "根因", "修复"]
    patterns: ["Traceback \\(most recent", "stderr:", "FAILED"]
  repo_arch:
    keywords: ["repo", "codebase", "monorepo", "architecture", "重构", "架构", "module", "dependency"]
  strict_format:
    keywords: ["JSON", "YAML", "CSV", "schema", "只返回", "不要解释", "按格式"]
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
    hint_zh: "直接作答，缩短思考长度，避免无关展开。"
    hint_en: "Answer directly, keep thinking short, avoid irrelevant expansion."
    conditions: {max_difficulty: 0.8, min_margin: 0.4, no_flags: [high_risk, strict_format, debug]}
  P1:
    hint_zh: ""
    hint_en: ""
  P2:
    hint_zh: "充分分析，覆盖关键约束，避免遗漏。"
    hint_en: "Analyze thoroughly, cover key constraints, avoid omissions."
    conditions: {any_flag: [high_risk, long_context, debug, strict_format]}

optuna:
  alpha: 1.0
  beta: 0.3
  gamma: 0.3

context_rules:
  deep_conversation_threshold: 4
  heavy_context_tokens: 2000
  heavy_context_min_class: R1
  deep_conversation_min_class: R1

trajectory:
  delta_threshold: 0.3
  history_max_turns: 5

v4:
  aux_head_inference: false      # default off; v4.2 needs this true for aux_downgrade
  bge_model_name: BAAI/bge-small-zh-v1.5
  bge_backend: onnx              # v4.2: 'onnx' (INT8 quantized) or 'sentence_transformers' (FP32)
  bge_onnx_dir: bge_onnx
  pca_dim: 64
  feature_dim: 390               # v4.2 Phase 3: was 385; +5 dims for reasoning HC channel
  history_user_max_turns: 4      # how many prior user turns to concat for history channel
  bge_truncate_tokens: 510       # leave 2 tokens for [CLS]/[SEP]
  aux_downgrade:                 # v4.2: gate downgrades via aux head's downgrade prob
    enabled: false               # default off; turn on per-deployment after evaluation
    threshold: 0.55              # require aux_probs["downgrade"] >= threshold
    # When enabled and conditions met, force route_idx -= 1 (no effect at R0).
    # Step fires AFTER margin upgrade; suppressed if margin upgrade just promoted.
  sticky_tier:                   # v4.2 Phase 1 (deferred): block tier downgrade on short continuation turns
    enabled: false               # default off — depends on accurate prev_route which router itself produces wrong on R2/R3 failure mode A; revisit after Phase 2 retrain
    max_user_len: 200            # only applies when current user text length <= this
"#;

    #[test]
    fn parses_real_runtime_yaml() {
        let cfg = RouterConfig::from_yaml_str(REAL_YAML).expect("real runtime yaml parses");

        assert_eq!(cfg.route_classes, vec!["R0", "R1", "R2", "R3"]);
        assert_eq!(cfg.tier_mapping.get("R0").map(String::as_str), Some("S"));
        assert_eq!(
            cfg.tier_registry.get("S").map(Vec::as_slice),
            Some(&["deepseek/deepseek-v4-flash".to_string()][..])
        );

        assert_eq!(cfg.thresholds.margin_upgrade, 0.10);
        assert_eq!(cfg.thresholds.high_confidence, 0.7);
        assert_eq!(cfg.thresholds.r1_rescue.from_r0_max_gap, 0.10);
        assert_eq!(cfg.thresholds.cascade_stage1_threshold, 0.4);
        assert_eq!(cfg.thresholds.under_routing_safety, 0.45);
        assert!(cfg.thresholds.kv_cache_aware);

        assert_eq!(
            cfg.flag_rules.high_risk.keywords_zh,
            vec![
                "生产", "部署", "回滚", "迁移", "删除", "客户", "法务", "财务"
            ]
        );
        assert_eq!(cfg.flag_rules.long_context.char_threshold, 6000);
        assert_eq!(cfg.flag_rules.long_context.file_ref_threshold, 2);

        assert_eq!(cfg.thinking_mode_rules.t3.min_class, "R2");
        assert_eq!(
            cfg.thinking_mode_rules.t3.flags,
            vec!["debug", "long_context", "high_risk"]
        );
        assert!(cfg.thinking_mode_rules.t2.default);
        assert_eq!(cfg.thinking_mode_rules.t0.max_class, "R0");
        assert_eq!(cfg.thinking_mode_rules.t0.min_margin, 0.5);

        let p2 = cfg
            .prompt_policies
            .p2
            .conditions
            .as_ref()
            .expect("P2 has conditions");
        let any_flag = p2.any_flag.as_ref().expect("P2 any_flag present");
        assert_eq!(any_flag.len(), 4);
        assert!(cfg.prompt_policies.p1.conditions.is_none());

        assert_eq!(cfg.context_rules.heavy_context_tokens, 2000);
        assert_eq!(cfg.trajectory.delta_threshold, 0.3);
        assert_eq!(cfg.trajectory.history_max_turns, 5);

        assert!(!cfg.v4.aux_head_inference);
        assert_eq!(cfg.v4.bge_model_name, "BAAI/bge-small-zh-v1.5");
        assert_eq!(cfg.v4.feature_dim, 390);
        assert_eq!(cfg.v4.pca_dim, 64);
        assert!(!cfg.v4.aux_downgrade.enabled);
        assert_eq!(cfg.v4.aux_downgrade.threshold, 0.55);
        assert!(!cfg.v4.sticky_tier.enabled);
        assert_eq!(cfg.v4.sticky_tier.max_user_len, 200);
    }

    #[test]
    fn from_file_reads_yaml() {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla_config_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("router.runtime.yaml");
        std::fs::write(&path, REAL_YAML).unwrap();
        let cfg = RouterConfig::from_file(&path).expect("from_file parses");
        assert_eq!(cfg.thresholds.margin_upgrade, 0.10);
        assert_eq!(cfg.v4.sticky_tier.max_user_len, 200);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn manifest_from_file_round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla_manifest_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("inference_manifest.json");
        std::fs::write(
            &path,
            r#"{"temperature":0.8092449307441711,"per_class_alpha":[0.5,0.05,0.5,0.85]}"#,
        )
        .unwrap();
        let m = Manifest::from_file(&path).expect("manifest parses");
        assert_eq!(m.temperature, 0.8092449307441711);
        assert_eq!(m.per_class_alpha, [0.5, 0.05, 0.5, 0.85]);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }

    #[test]
    fn long_context_defaults_match_python() {
        let lc = LongContextRules::default();
        assert_eq!(lc.char_threshold, 6000);
        assert_eq!(lc.code_block_threshold, 1500);
        assert_eq!(lc.log_block_threshold, 1500);
        assert_eq!(lc.file_ref_threshold, 2);
    }
}
