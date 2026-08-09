//! Desktop `doctor.status` (L2) — Python-compatible findings health report.
//!
//! The WebUI Overview health panel consumes a findings-shaped report
//! (`status`/`ready`/`summary`/`counts`/`impactCounts`/`findings[]`). The
//! recovery [`HealthCheck`] produces `components`/`issues`, so this module
//! adapts each component into a `findings` entry and tallies counts with the
//! same logic as the frontend `withoutLegacyMigrationFinding` helper, so the
//! desktop report renders identically to a gateway report.
//!
//! The gateway `doctor.status` (L1, crates/gateway/src/doctor.rs) only returns
//! `{ status, uptime_seconds }`; this L2 command is the desktop-side bridge
//! that serves the full report the Overview view renders.

use crate::error::TauriResult;
use crate::state::AppState;
use opensquilla_core::config::Config;
use opensquilla_recovery::health::{ComponentHealth, HealthCheck, HealthStatus};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tauri::State;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoctorStatusRequest {
    agent_id: Option<String>,
    deep: Option<bool>,
}

fn severity_for(status: HealthStatus) -> &'static str {
    match status {
        HealthStatus::Healthy => "ok",
        HealthStatus::Degraded => "warn",
        HealthStatus::Unhealthy => "error",
    }
}

fn impact_for(severity: &str) -> &'static str {
    match severity {
        "error" => "blocks_ready",
        "warn" => "degrades",
        "info" => "optional",
        _ => "none",
    }
}

/// Map a health component name onto a frontend settings surface. Unknown
/// surfaces render under the default group without a deep link.
fn surface_for(component: &str) -> &'static str {
    match component {
        "provider" | "providers" => "provider",
        "channels" => "channels",
        "router" | "squilla_router" | "routing" => "router",
        "memory" | "memory_index" => "memory",
        "scheduler" | "cron" => "scheduler",
        "search" => "search",
        "image_generation" => "image_generation",
        "audio" => "audio",
        "session" | "sessions" => "sessions",
        _ => "system",
    }
}

fn component_finding(c: &ComponentHealth) -> Value {
    let severity = severity_for(c.status);
    let surface = surface_for(&c.name);
    let mut evidence: Map<String, Value> = Map::new();
    evidence.insert("latency_ms".to_string(), json!(c.latency_ms));
    for (key, value) in &c.details {
        evidence.insert(key.clone(), json!(value));
    }
    json!({
        "id": format!("{surface}.{name}", surface = surface, name = c.name),
        "surface": surface,
        "severity": severity,
        "readinessImpact": impact_for(severity),
        "title": c.description,
        "detail": c.description,
        "evidence": Value::Object(evidence),
    })
}

/// Mirror the frontend `withoutLegacyMigrationFinding` tally so `counts` /
/// `impactCounts` / `status` / `summary` match what the WebUI would derive.
fn tally(findings: &[Value]) -> (Value, Value) {
    let mut counts = Map::new();
    for key in ["error", "warn", "info", "ok"] {
        counts.insert(key.to_string(), json!(0));
    }
    let mut impacts = Map::new();
    for key in ["blocks_ready", "degrades", "optional", "none"] {
        impacts.insert(key.to_string(), json!(0));
    }
    for finding in findings {
        let severity = finding.get("severity").and_then(Value::as_str).unwrap_or("");
        if let Some(bucket) = counts.get_mut(severity) {
            if let Some(n) = bucket.as_u64() {
                *bucket = json!(n + 1);
            }
        }
        let impact = finding
            .get("readinessImpact")
            .and_then(Value::as_str)
            .unwrap_or("");
        if let Some(bucket) = impacts.get_mut(impact) {
            if let Some(n) = bucket.as_u64() {
                *bucket = json!(n + 1);
            }
        }
    }
    (Value::Object(counts), Value::Object(impacts))
}

/// `doctor.status` — full (deep) or config-only (shallow) findings report.
#[tauri::command]
pub async fn doctor_status(
    state: State<'_, AppState>,
    request: DoctorStatusRequest,
) -> TauriResult<Value> {
    let config = state.config().await;
    let health = HealthCheck::new(&config);
    let result = if request.deep.unwrap_or(false) {
        health.run_full_check().await
    } else {
        health.quick_check().await
    };

    let findings: Vec<Value> = result.components.iter().map(component_finding).collect();
    let (counts, impact_counts) = tally(&findings);

    let blocks = impact_counts
        .get("blocks_ready")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let degrades = impact_counts
        .get("degrades")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let optional = impact_counts
        .get("optional")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let status = if blocks > 0 {
        "action_required"
    } else if degrades > 0 {
        "degraded"
    } else {
        "ready"
    };

    let mut summary_parts: Vec<String> = Vec::new();
    if blocks > 0 {
        summary_parts.push(format!(
            "{blocks} {} required",
            if blocks == 1 { "action" } else { "actions" }
        ));
    }
    if degrades > 0 {
        let degraded = format!(
            "{degrades} degraded {}",
            if degrades == 1 { "check" } else { "checks" }
        );
        summary_parts.push(if blocks > 0 {
            degraded
        } else {
            format!("Ready, {degraded}")
        });
    }
    let summary = if !summary_parts.is_empty() {
        summary_parts.join(", ")
    } else if optional > 0 {
        format!(
            "Ready, {optional} optional setup {}",
            if optional == 1 { "item" } else { "items" }
        )
    } else {
        "Ready".to_string()
    };

    let config_path = Config::discover_path()
        .ok()
        .map(|p| p.to_string_lossy().to_string());
    let gateway_url = state.gateway_url().await;
    let agent_id = request.agent_id.unwrap_or("main".to_string());

    Ok(json!({
        "status": status,
        "ready": blocks == 0,
        "summary": summary,
        "gatewayUrl": gateway_url,
        "configPath": config_path,
        "agentId": agent_id,
        "uptimeSeconds": result.uptime_seconds,
        "counts": counts,
        "impactCounts": impact_counts,
        "findings": findings,
    }))
}
