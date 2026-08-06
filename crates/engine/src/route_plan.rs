//! Immutable logical routing plan and append-only execution-leg telemetry.
//!
//! Mirrors the Python backend's `engine/route_plan.py`. The router decides
//! once, before the agent loop starts; provider retries and selector failover
//! are physical execution details of that decision and are recorded as
//! execution legs, never as additional router decisions.
//!
//! The plan lives in the turn metadata (`route_plan` key) and is rendered
//! through [`serde_json::Value`] so the routing telemetry events share a
//! common shape.

use serde_json::json;

/// Capacity and feature facts used by one logical turn.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteCapabilitySnapshot {
    /// The model context window in tokens.
    pub context_window: usize,
    /// A provider/model-specific automatic output ceiling. Zero means the
    /// catalog had no authoritative value.
    pub effective_max_tokens: usize,
    /// Whether the model supports reasoning (unknown when `None`).
    pub supports_reasoning: Option<bool>,
    /// Whether the model supports tools (unknown when `None`).
    pub supports_tools: Option<bool>,
    /// Whether the model supports streaming (unknown when `None`).
    pub supports_streaming: Option<bool>,
    /// Whether the model supports vision (unknown when `None`).
    pub supports_vision: Option<bool>,
    /// The model's reasoning format hint.
    pub reasoning_format: String,
}

impl RouteCapabilitySnapshot {
    /// Build a snapshot from the given facts.
    pub fn new(
        context_window: usize,
        effective_max_tokens: usize,
        supports_reasoning: Option<bool>,
        supports_tools: Option<bool>,
        supports_streaming: Option<bool>,
        supports_vision: Option<bool>,
        reasoning_format: impl Into<String>,
    ) -> Self {
        Self {
            context_window,
            effective_max_tokens,
            supports_reasoning,
            supports_tools,
            supports_streaming,
            supports_vision,
            reasoning_format: reasoning_format.into(),
        }
    }

    /// Render the snapshot as a JSON object.
    pub fn as_dict(&self) -> serde_json::Value {
        let bool_or_null = |value: Option<bool>| {
            value
                .map(serde_json::Value::Bool)
                .unwrap_or(serde_json::Value::Null)
        };
        json!({
            "context_window": self.context_window,
            "effective_max_tokens": self.effective_max_tokens,
            "supports_reasoning": bool_or_null(self.supports_reasoning),
            "supports_tools": bool_or_null(self.supports_tools),
            "supports_streaming": bool_or_null(self.supports_streaming),
            "supports_vision": bool_or_null(self.supports_vision),
            "reasoning_format": self.reasoning_format,
        })
    }

    /// Rebuild a snapshot from a JSON object (best-effort).
    pub fn from_dict(value: &serde_json::Value) -> Option<Self> {
        let obj = value.as_object()?;
        Some(Self {
            context_window: obj.get("context_window").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            effective_max_tokens: obj.get("effective_max_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            supports_reasoning: obj.get("supports_reasoning").and_then(|v| v.as_bool()),
            supports_tools: obj.get("supports_tools").and_then(|v| v.as_bool()),
            supports_streaming: obj.get("supports_streaming").and_then(|v| v.as_bool()),
            supports_vision: obj.get("supports_vision").and_then(|v| v.as_bool()),
            reasoning_format: obj.get("reasoning_format").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        })
    }
}

/// One configured fallback candidate captured when the route is pinned.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteFallback {
    /// The candidate tier.
    pub tier: String,
    /// The candidate provider.
    pub provider: String,
    /// The candidate model id.
    pub model: String,
    /// Capability facts captured with the candidate.
    pub capabilities: RouteCapabilitySnapshot,
}

impl RouteFallback {
    /// Render the fallback as a JSON object.
    pub fn as_dict(&self) -> serde_json::Value {
        json!({
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "capabilities": self.capabilities.as_dict(),
        })
    }

    /// Rebuild a fallback from a JSON object (best-effort).
    pub fn from_dict(value: &serde_json::Value) -> Option<Self> {
        let obj = value.as_object()?;
        Some(Self {
            tier: obj.get("tier").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            provider: obj.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            model: obj.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            capabilities: RouteCapabilitySnapshot::from_dict(obj.get("capabilities").unwrap_or(&serde_json::Value::Null))?,
        })
    }
}

/// One immutable router decision for one logical turn.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutePlan {
    /// Plan schema version.
    pub version: usize,
    /// The plan id (normally the turn id).
    pub plan_id: String,
    /// The turn this plan belongs to.
    pub turn_id: String,
    /// The routed tier.
    pub tier: String,
    /// The routed provider.
    pub provider: String,
    /// The routed model.
    pub model: String,
    /// Where the routing decision came from.
    pub source: String,
    /// Whether routing was actually applied.
    pub routing_applied: bool,
    /// The thinking mode snapshot.
    pub thinking: String,
    /// The prompt policy snapshot.
    pub prompt_policy: String,
    /// The captured fallback chain.
    pub fallback_chain: Vec<RouteFallback>,
    /// Capability facts for the routed model.
    pub capabilities: RouteCapabilitySnapshot,
}

impl RoutePlan {
    /// Render the plan as a JSON object.
    pub fn as_dict(&self) -> serde_json::Value {
        json!({
            "version": self.version,
            "plan_id": self.plan_id,
            "turn_id": self.turn_id,
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "source": self.source,
            "routing_applied": self.routing_applied,
            "thinking": self.thinking,
            "prompt_policy": self.prompt_policy,
            "fallback_chain": self.fallback_chain.iter().map(RouteFallback::as_dict).collect::<Vec<_>>(),
            "capabilities": self.capabilities.as_dict(),
        })
    }

    /// Rebuild a plan from a JSON object (best-effort).
    pub fn from_dict(value: &serde_json::Value) -> Option<Self> {
        let obj = value.as_object()?;
        let fallback_chain = obj
            .get("fallback_chain")
            .and_then(|v| v.as_array())
            .map(|items| items.iter().filter_map(RouteFallback::from_dict).collect())
            .unwrap_or_default();
        Some(Self {
            version: obj.get("version").and_then(|v| v.as_u64()).unwrap_or(1) as usize,
            plan_id: obj.get("plan_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            turn_id: obj.get("turn_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            tier: obj.get("tier").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            provider: obj.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            model: obj.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            source: obj.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            routing_applied: obj.get("routing_applied").and_then(|v| v.as_bool()).unwrap_or(false),
            thinking: obj.get("thinking").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            prompt_policy: obj.get("prompt_policy").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            fallback_chain,
            capabilities: RouteCapabilitySnapshot::from_dict(obj.get("capabilities").unwrap_or(&serde_json::Value::Null))?,
        })
    }
}

/// Fallback candidate capability facts keyed by `(provider, model)`.
pub type FallbackCapabilityMap = std::collections::HashMap<(String, String), RouteCapabilitySnapshot>;

/// Create the turn's `RoutePlan` once and return the already-pinned value later.
///
/// Operates on the turn metadata map. When a plan is already stored under
/// `route_plan`, that stored value is returned. Returns `None` when no
/// `routed_tier` is present (routing was not applied this turn).
pub fn pin_route_plan(
    metadata: &mut std::collections::HashMap<String, serde_json::Value>,
    turn_id: &str,
    provider: &str,
    model: &str,
    capabilities: Option<&RouteCapabilitySnapshot>,
    effective_thinking: &str,
    fallback_capabilities: Option<&FallbackCapabilityMap>,
) -> Option<RoutePlan> {
    if let Some(existing) = metadata.get("route_plan") {
        if let Some(plan) = RoutePlan::from_dict(existing) {
            return Some(plan);
        }
    }

    let tier = text_value(metadata.get("routed_tier"));
    if tier.is_empty() {
        return None;
    }

    let route_provider = {
        let routed = text_value(metadata.get("routed_provider"));
        if routed.is_empty() {
            provider.trim().to_string()
        } else {
            routed
        }
    };
    let route_model = {
        let routed = text_value(metadata.get("routed_model"));
        if routed.is_empty() {
            model.trim().to_string()
        } else {
            routed
        }
    };

    let mut fallback_candidates: Vec<serde_json::Value> = Vec::new();
    for key in ["router_fallback_chain", "selector_execution_chain"] {
        if let Some(value) = metadata.get(key).and_then(|v| v.as_array()) {
            fallback_candidates.extend(value.iter().cloned());
        }
    }

    let thinking = {
        let explicit = text_value(metadata.get("thinking_level"));
        if explicit.is_empty() {
            let mode = text_value(metadata.get("thinking_mode"));
            if !mode.is_empty() {
                mode
            } else {
                thinking_snapshot_value(effective_thinking)
            }
        } else {
            explicit
        }
    };

    let plan = RoutePlan {
        version: 1,
        plan_id: turn_id.to_string(),
        turn_id: turn_id.to_string(),
        tier,
        provider: route_provider.clone(),
        model: route_model.clone(),
        source: {
            let source = text_value(metadata.get("routing_source"));
            if source.is_empty() {
                "none".to_string()
            } else {
                source
            }
        },
        routing_applied: metadata
            .get("routing_applied")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        thinking,
        prompt_policy: text_value(metadata.get("prompt_policy")),
        fallback_chain: build_fallback_chain(
            &fallback_candidates,
            &route_provider,
            &route_model,
            fallback_capabilities,
        ),
        capabilities: capabilities
            .cloned()
            .unwrap_or_else(|| RouteCapabilitySnapshot::new(0, 0, None, None, None, None, "")),
    };

    metadata.insert("route_plan".to_string(), plan.as_dict());
    Some(plan)
}

/// Append one physical provider request without changing the `RoutePlan`.
///
/// The leg's `execution_id` / `call_kind` come from the provider request
/// correlation, if the caller has them.
pub fn record_execution_leg(
    metadata: &mut std::collections::HashMap<String, serde_json::Value>,
    provider: &str,
    model: &str,
    kind: &str,
    execution_id: Option<&str>,
    call_kind: Option<&str>,
    reason: &str,
) {
    let plan_id = metadata
        .get("route_plan")
        .and_then(|v| v.as_object())
        .and_then(|obj| obj.get("plan_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let legs = metadata
        .entry("execution_legs".to_string())
        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
    let Some(arr) = legs.as_array_mut() else {
        return;
    };
    let mut leg = serde_json::Map::new();
    leg.insert("index".to_string(), serde_json::Value::from(arr.len()));
    leg.insert("kind".to_string(), serde_json::Value::String(kind.trim().to_string()));
    leg.insert("provider".to_string(), serde_json::Value::String(provider.trim().to_string()));
    leg.insert("model".to_string(), serde_json::Value::String(model.trim().to_string()));
    leg.insert("plan_id".to_string(), serde_json::Value::String(plan_id));
    if let Some(id) = execution_id.filter(|s| !s.is_empty()) {
        leg.insert("execution_id".to_string(), serde_json::Value::String(id.to_string()));
    }
    if let Some(call) = call_kind.filter(|s| !s.is_empty()) {
        leg.insert("call_kind".to_string(), serde_json::Value::String(call.to_string()));
    }
    if !reason.is_empty() {
        leg.insert("reason".to_string(), serde_json::Value::String(reason.to_string()));
    }
    arr.push(serde_json::Value::Object(leg));
}

/// Return the stored route-plan snapshot, if any.
pub fn route_plan_snapshot(metadata: &std::collections::HashMap<String, serde_json::Value>) -> Option<serde_json::Value> {
    metadata.get("route_plan").filter(|v| v.is_object()).cloned()
}

fn text_value(value: Option<&serde_json::Value>) -> String {
    value
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string()
}

fn thinking_snapshot_value(value: &str) -> String {
    let text = value.trim().to_lowercase();
    if text.is_empty() {
        String::new()
    } else if text == "true" {
        "enabled".to_string()
    } else if text == "false" {
        "disabled".to_string()
    } else {
        text
    }
}

fn build_fallback_chain(
    candidates: &[serde_json::Value],
    default_provider: &str,
    primary_model: &str,
    capability_snapshots: Option<&FallbackCapabilityMap>,
) -> Vec<RouteFallback> {
    let mut result: Vec<RouteFallback> = Vec::new();
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    seen.insert((default_provider.to_string(), primary_model.to_string()));
    for item in candidates {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let model = text_value(obj.get("model"));
        if model.is_empty() {
            continue;
        }
        let provider = {
            let provider = text_value(obj.get("provider"));
            if provider.is_empty() {
                default_provider.to_string()
            } else {
                provider
            }
        };
        let identity = (provider.clone(), model.clone());
        if seen.contains(&identity) {
            continue;
        }
        seen.insert(identity.clone());
        let capabilities = capability_snapshots
            .and_then(|map| map.get(&identity))
            .cloned()
            .unwrap_or_else(|| RouteCapabilitySnapshot::new(0, 0, None, None, None, None, ""));
        result.push(RouteFallback {
            tier: text_value(obj.get("tier")),
            provider,
            model,
            capabilities,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn empty_metadata() -> HashMap<String, serde_json::Value> {
        HashMap::new()
    }

    #[test]
    fn test_pin_route_plan_builds_and_is_idempotent() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        metadata.insert("routing_source".into(), serde_json::Value::String("calibration".into()));
        let caps = RouteCapabilitySnapshot::new(128_000, 0, Some(true), Some(true), Some(true), None, "think");
        let plan = pin_route_plan(&mut metadata, "turn-1", "openrouter", "model-x", Some(&caps), "", None).unwrap();
        assert_eq!(plan.tier, "fast");
        assert_eq!(plan.provider, "openrouter");
        assert_eq!(plan.model, "model-x");
        assert_eq!(plan.plan_id, "turn-1");
        assert_eq!(plan.source, "calibration");
        assert_eq!(plan.capabilities.context_window, 128_000);
        assert!(plan.capabilities.supports_reasoning == Some(true));

        // Re-pinning returns the stored plan (idempotent).
        let plan2 = pin_route_plan(&mut metadata, "turn-1", "openrouter", "model-x", None, "", None).unwrap();
        assert_eq!(plan2.as_dict(), plan.as_dict());
    }

    #[test]
    fn test_pin_route_plan_without_tier_returns_none() {
        let mut metadata = empty_metadata();
        assert!(pin_route_plan(&mut metadata, "turn-1", "openrouter", "model-x", None, "", None).is_none());
    }

    #[test]
    fn test_routed_metadata_overrides_arguments() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        metadata.insert("routed_provider".into(), serde_json::Value::String("anthropic".into()));
        metadata.insert("routed_model".into(), serde_json::Value::String("claude-x".into()));
        metadata.insert("routing_applied".into(), serde_json::Value::Bool(false));
        let plan = pin_route_plan(&mut metadata, "turn-1", "openrouter", "model-x", None, "", None).unwrap();
        assert_eq!(plan.provider, "anthropic");
        assert_eq!(plan.model, "claude-x");
        assert!(!plan.routing_applied);
    }

    #[test]
    fn test_thinking_snapshot_uses_metadata_first() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        metadata.insert("thinking_mode".into(), serde_json::Value::String("high".into()));
        let plan = pin_route_plan(&mut metadata, "turn-1", "p", "m", None, "", None).unwrap();
        assert_eq!(plan.thinking, "high");
    }

    #[test]
    fn test_thinking_boolean_string_maps_to_enabled() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        let plan = pin_route_plan(&mut metadata, "turn-1", "p", "m", None, "true", None).unwrap();
        assert_eq!(plan.thinking, "enabled");
    }

    #[test]
    fn test_fallback_chain_built_and_deduped() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        metadata.insert(
            "router_fallback_chain".into(),
            serde_json::json!([
                {"tier": "fast", "provider": "openrouter", "model": "model-x"},
                {"tier": "fast", "provider": "openrouter", "model": "model-y"},
                {"tier": "fast", "provider": "openrouter", "model": "model-x"},
                {"tier": "fast", "provider": "openrouter", "model": "model-z"},
            ]),
        );
        let caps = FallbackCapabilityMap::new();
        let plan = pin_route_plan(&mut metadata, "turn-1", "openrouter", "model-x", None, "", Some(&caps)).unwrap();
        assert_eq!(plan.fallback_chain.len(), 2);
        assert_eq!(plan.fallback_chain[0].model, "model-y");
        assert_eq!(plan.fallback_chain[1].model, "model-z");
    }

    #[test]
    fn test_record_execution_leg_appends() {
        let mut metadata = empty_metadata();
        record_execution_leg(&mut metadata, "openrouter", "model-x", "chat", Some("exec-1"), Some("retry"), "rate_limited");
        record_execution_leg(&mut metadata, "openrouter", "model-y", "chat", None, None, "");
        let legs = metadata["execution_legs"].as_array().unwrap();
        assert_eq!(legs.len(), 2);
        assert_eq!(legs[0]["index"], 0);
        assert_eq!(legs[0]["execution_id"], "exec-1");
        assert_eq!(legs[0]["reason"], "rate_limited");
        assert_eq!(legs[1]["index"], 1);
        assert!(legs[1].get("execution_id").is_none());
    }

    #[test]
    fn test_record_execution_leg_carries_plan_id() {
        let mut metadata = empty_metadata();
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        pin_route_plan(&mut metadata, "turn-9", "p", "m", None, "", None).unwrap();
        record_execution_leg(&mut metadata, "p", "m", "chat", None, None, "");
        let legs = metadata["execution_legs"].as_array().unwrap();
        assert_eq!(legs[0]["plan_id"], "turn-9");
    }

    #[test]
    fn test_route_plan_snapshot() {
        let mut metadata = empty_metadata();
        assert!(route_plan_snapshot(&metadata).is_none());
        metadata.insert("routed_tier".into(), serde_json::Value::String("fast".into()));
        pin_route_plan(&mut metadata, "turn-1", "p", "m", None, "", None).unwrap();
        let snapshot = route_plan_snapshot(&metadata).unwrap();
        assert_eq!(snapshot["tier"], "fast");
        assert_eq!(snapshot["plan_id"], "turn-1");
    }

    #[test]
    fn test_round_trip_from_dict() {
        let plan = RoutePlan {
            version: 1,
            plan_id: "p-1".into(),
            turn_id: "t-1".into(),
            tier: "fast".into(),
            provider: "openrouter".into(),
            model: "model-x".into(),
            source: "calibration".into(),
            routing_applied: true,
            thinking: "enabled".into(),
            prompt_policy: "strict".into(),
            fallback_chain: vec![RouteFallback {
                tier: "fast".into(),
                provider: "openrouter".into(),
                model: "model-y".into(),
                capabilities: RouteCapabilitySnapshot::new(64_000, 0, None, Some(true), None, None, ""),
            }],
            capabilities: RouteCapabilitySnapshot::new(128_000, 0, Some(true), Some(true), Some(true), None, "think"),
        };
        let restored = RoutePlan::from_dict(&plan.as_dict()).unwrap();
        assert_eq!(restored, plan);
    }
}
