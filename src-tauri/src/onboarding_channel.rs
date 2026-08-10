//! Onboarding channel commands (S4).
//!
//! Provides the `onboarding.channel.probe/upsert/remove/enable/disable` RPC
//! methods over the Tauri bridge so the WebUI channels editor can probe, save,
//! remove, and toggle channels. Every command mutates `config.channels` and
//! persists through [`Config::save`] (atomic write + fresh-install fallback);
//! a save failure is surfaced as an error, never silently dropped.
//!
//! There is no live channel reconciler in the Rust runtime yet, so every
//! mutation reports `restartRequired: true` with `liveApply: null` — the same
//! conservative fallback the Python gateway emits when it cannot apply the
//! change live. Probe is local-only: it validates the entry and, fail-closed,
//! derives connection state from config presence, without any network call.

use crate::error::{TauriError, TauriResult};
use crate::state::AppState;
use opensquilla_core::config::{ChannelConfig, Config};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use tauri::State;

/// The redaction placeholder `channels.get` / probe echo for stored secrets.
/// The WebUI strips it before sending; server-side it still means "keep the
/// current value", never a literal three-asterisk credential.
const REDACTED_SENTINEL: &str = "***";

/// Error codes the WebUI `rpcErrors.ts` already localizes.
fn invalid(message: impl Into<String>) -> TauriError {
    TauriError::new("onboarding.channel.invalid", message, 400)
}

fn not_found(name: &str) -> TauriError {
    TauriError::new(
        "onboarding.channel.not_found",
        format!("no channel named {name:?}"),
        404,
    )
}

// ---------------------------------------------------------------------------
// Entry validation / redaction
// ---------------------------------------------------------------------------

/// Channel types accepted by the onboarding editor, aligned with the Python
/// `channel_specs` builders plus the terminal/websocket/webhook adapters.
const KNOWN_CHANNEL_TYPES: &[&str] = &[
    "slack",
    "discord",
    "telegram",
    "feishu",
    "dingtalk",
    "matrix",
    "msteams",
    "qq",
    "wecom",
    "terminal",
    "websocket",
    "webhook",
];

/// Non-blank-required credential fields per channel type: `(name, secret)`.
/// Mirrors Python's `_require_non_blank_secret_fields`; mode-dependent
/// requirements (slack webhook `signing_secret`, wecom branches) are handled
/// in [`validate_entry`].
const REQUIRED_FIELDS: &[(&str, &[(&str, bool)])] = &[
    ("slack", &[("token", true)]),
    ("discord", &[("token", true)]),
    ("telegram", &[("token", true)]),
    ("feishu", &[("app_id", false), ("app_secret", true)]),
    ("dingtalk", &[("client_id", false), ("client_secret", true)]),
    ("qq", &[("app_id", false), ("app_secret", true)]),
    ("msteams", &[("app_id", false), ("app_password", true)]),
    ("matrix", &[("homeserver_url", false), ("user_id", false)]),
];

/// Secret field names per channel type, redacted (`***`) in public payloads.
const SECRET_FIELDS: &[(&str, &[&str])] = &[
    ("slack", &["token", "app_token", "signing_secret"]),
    ("discord", &["token"]),
    ("telegram", &["token", "webhook_secret_token"]),
    ("feishu", &["app_secret", "encrypt_key", "verification_token"]),
    ("dingtalk", &["client_secret"]),
    ("qq", &["app_secret"]),
    ("msteams", &["app_password"]),
    ("matrix", &["password", "access_token"]),
    ("wecom", &["bot_secret", "corp_secret", "token", "encoding_aes_key"]),
];

/// Read a field as text regardless of its JSON type (ids/urls may arrive as
/// numbers, e.g. `agent_id_int`).
fn field_text(entry: &Map<String, Value>, field: &str) -> String {
    match entry.get(field) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// The trimmed channel `name`, the identity key upsert/remove/toggle match on.
fn entry_name(entry: &Map<String, Value>) -> Result<String, TauriError> {
    let name = field_text(entry, "name");
    if name.trim().is_empty() {
        return Err(invalid("channel entry requires non-empty 'name'"));
    }
    Ok(name.trim().to_string())
}

/// Declared secret field names for a channel type, else an empty slice.
fn secret_fields_for(type_name: &str) -> &'static [&'static str] {
    SECRET_FIELDS
        .iter()
        .find(|(t, _)| *t == type_name)
        .map(|(_, fields)| *fields)
        .unwrap_or(&[])
}

/// Whether a value is truthy enough to warrant redaction.
fn is_non_empty(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Secret-shaped key detection for types without a declared secret set (the
/// Python `_is_secret_like_tier_key` fallback): exact names, `_key`/`_token`/
/// `_secret`/`_password` suffixes, and camelCase spellings.
fn is_secret_like(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase().replace('-', "_");
    const EXACT: &[&str] = &["key", "api_key", "token", "secret", "password", "authorization"];
    if EXACT.contains(&normalized.as_str()) {
        return true;
    }
    const SUFFIXES: &[&str] = &["_key", "_token", "_secret", "_password"];
    if SUFFIXES.iter().any(|s| normalized.ends_with(s)) {
        return true;
    }
    let squashed = normalized.replace('_', "");
    const SQUASHED_SUFFIXES: &[&str] = &["apikey", "token", "secret", "password", "authorization"];
    SQUASHED_SUFFIXES.iter().any(|s| squashed.ends_with(s))
}

/// Redact secret fields in a normalized entry before echoing it to the client.
/// Declared secrets use the exact spec names; unknown types fail closed by
/// redacting anything secret-shaped.
fn redact_entry(type_name: &str, entry: &Map<String, Value>) -> Value {
    let secrets = secret_fields_for(type_name);
    let mut out = entry.clone();
    if secrets.is_empty() {
        for (key, value) in out.iter_mut() {
            if is_secret_like(key) && is_non_empty(value) {
                *value = Value::String(REDACTED_SENTINEL.to_string());
            }
        }
    } else {
        for field in secrets {
            if let Some(value) = out.get_mut(*field) {
                if is_non_empty(value) {
                    *value = Value::String(REDACTED_SENTINEL.to_string());
                }
            }
        }
    }
    Value::Object(out)
}

/// Merge-aware secrets: a blank or `***` value for a declared secret resolves
/// against the stored entry, so keep-current drafts validate as what the
/// upsert would persist (mirrors Python's `_merge_with_existing_secrets`).
fn merge_stored_secrets(
    config: &Config,
    entry: &mut Map<String, Value>,
    type_name: &str,
    name: &str,
) {
    let Some(stored) = config
        .channels
        .iter()
        .find(|c| c.name == name && c.channel_type == type_name)
    else {
        return;
    };
    for field in secret_fields_for(type_name) {
        let blank = match entry.get(*field) {
            None => true,
            Some(Value::String(s)) => s.trim().is_empty() || s.trim() == REDACTED_SENTINEL,
            Some(_) => false,
        };
        if blank {
            if let Some(stored_value) = stored.config.get(*field) {
                if !stored_value.trim().is_empty() {
                    entry.insert(field.to_string(), Value::String(stored_value.clone()));
                }
            }
        }
    }
}

/// Non-blank required credential fields (fail-closed, mirrors Python's
/// `_require_non_blank_secret_fields`).
fn require_non_blank_credentials(
    entry: &Map<String, Value>,
    type_name: &str,
) -> Result<(), TauriError> {
    if let Some((_, fields)) = REQUIRED_FIELDS.iter().find(|(t, _)| *t == type_name) {
        for (field, _) in *fields {
            if field_text(entry, field).trim().is_empty() {
                return Err(invalid(format!(
                    "channel field '{field}' requires a non-empty value"
                )));
            }
        }
    }
    Ok(())
}

/// WeCom required fields depend on `connection_mode` (websocket vs webhook).
fn require_wecom_fields(entry: &Map<String, Value>) -> Result<(), TauriError> {
    let required: &[&str] = if field_text(entry, "connection_mode") == "websocket" {
        &["bot_id", "bot_secret"]
    } else {
        &["corp_id", "corp_secret", "agent_id_int", "token", "encoding_aes_key"]
    };
    for field in required {
        if field_text(entry, field).trim().is_empty() {
            return Err(invalid(format!(
                "channel field '{field}' requires a non-empty value"
            )));
        }
    }
    Ok(())
}

/// Validate and normalize a channel entry. Returns the channel type; the entry
/// is mutated in place with merged secrets and identity defaults. Mirrors
/// Python's `merge_channel_entry_secrets` + `validate_channel_entry`.
fn validate_entry(
    config: &Config,
    entry: &mut Map<String, Value>,
) -> Result<String, TauriError> {
    let type_name = field_text(entry, "type");
    if type_name.trim().is_empty() {
        return Err(invalid("channel entry requires non-empty 'type'"));
    }
    if !KNOWN_CHANNEL_TYPES.contains(&type_name.as_str()) {
        return Err(invalid(format!("unknown channel type: {type_name}")));
    }
    let name = entry_name(entry)?;
    merge_stored_secrets(config, entry, &type_name, &name);
    if !entry.contains_key("agent_id") {
        entry.insert("agent_id".to_string(), Value::String("main".to_string()));
    }
    if !entry.contains_key("enabled") {
        entry.insert("enabled".to_string(), Value::Bool(true));
    }
    require_non_blank_credentials(entry, &type_name)?;
    if type_name == "slack" && field_text(entry, "connection_mode") != "socket" {
        if field_text(entry, "signing_secret").trim().is_empty() {
            return Err(invalid("slack webhook channels require signing_secret"));
        }
    }
    if type_name == "wecom" {
        require_wecom_fields(entry)?;
    }
    Ok(type_name)
}

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

/// `onboarding.channel.upsert` payload: a free-form `entry` object whose keys
/// are the frontend channel-entry keys (`type`, `name`, `enabled`, and the
/// per-type config keys such as `token` / `slack_channel_id`).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelUpsertRequest {
    entry: serde_json::Value,
}

/// `onboarding.channel.probe` payload: the same free-form `entry` object the
/// upsert accepts.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelProbeRequest {
    entry: serde_json::Value,
}

/// `onboarding.channel.probe` response, aligned with the WebUI
/// `ChannelProbeResponse` (reads `status`/`connected`/`probeKind`/
/// `restartRequired`/`entry`/`warnings`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelProbeResponse {
    status: String,
    connected: bool,
    probe_kind: String,
    restart_required: bool,
    entry: serde_json::Value,
    warnings: Vec<String>,
}

/// Payload shared by `onboarding.channel.remove` / `enable` / `disable`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelNameRequest {
    name: String,
}

/// `onboarding.channel.upsert` response, aligned with the WebUI
/// `ChannelUpsertResponse` (reads `changed`/`restartRequired`/`liveApply`/
/// `warnings` and `entry.name`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelUpsertResponse {
    changed: bool,
    restart_required: bool,
    live_apply: Option<HashMap<String, String>>,
    config_path: Option<String>,
    entry: serde_json::Value,
    warnings: Vec<String>,
}

/// `onboarding.channel.remove` response, aligned with the WebUI
/// `{ changed?, restartRequired? }` read.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelRemoveResponse {
    changed: bool,
    restart_required: bool,
    live_apply: Option<HashMap<String, String>>,
    config_path: Option<String>,
    removed: String,
}

/// `onboarding.channel.enable` / `disable` response, aligned with the WebUI
/// `{ changed?, restartRequired?, liveApply? }` read.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelToggleResponse {
    changed: bool,
    restart_required: bool,
    live_apply: Option<HashMap<String, String>>,
    config_path: Option<String>,
    name: String,
    enabled: bool,
}

// ---------------------------------------------------------------------------
// Upsert
// ---------------------------------------------------------------------------

/// `onboarding.channel.upsert` — create or update a channel matched by name.
#[tauri::command]
pub async fn onboarding_channel_upsert(
    state: State<'_, AppState>,
    request: ChannelUpsertRequest,
) -> TauriResult<ChannelUpsertResponse> {
    let mut obj = request
        .entry
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("params.entry must be an object"))?;

    // Full validation (type/name/required credentials, merge-aware secrets)
    // mirrors Python's `merge_channel_entry_secrets` + `validate_channel_entry`
    // so the "save anyway" path cannot persist an invalid entry.
    let (name, _channel_type, enabled) = extract_identity(&obj)?;
    let type_name = {
        let config = state.config().await;
        validate_entry(&config, &mut obj)?
    };

    let mut config = state.config_mut().await;

    // Secret/blank keep-current merge (mirrors Python's
    // `_merge_with_existing_secrets`) only applies on a same-type update:
    // re-adding an entry under a different type replaces it wholesale.
    let stored = config.channels.iter().find(|c| c.name == name);
    let keep_current = keep_current_source(stored, &type_name);

    let channel = ChannelConfig {
        name,
        channel_type: type_name,
        enabled,
        config: fold_config(&obj, keep_current),
    };

    apply_upsert(&mut config.channels, channel.clone());

    config.save().map_err(TauriError::from)?;

    Ok(ChannelUpsertResponse {
        changed: true,
        restart_required: true,
        live_apply: None,
        config_path: config_path(),
        entry: redact_entry(&channel.channel_type, &obj),
        warnings: Vec::new(),
    })
}

/// Pull the identity triple out of a frontend channel entry. The payload
/// requires a non-empty `name` and `type`; `enabled` defaults to `true`.
fn extract_identity(
    entry: &serde_json::Map<String, serde_json::Value>,
) -> Result<(String, String, bool), TauriError> {
    let name = match entry.get("name") {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => s.clone(),
        _ => return Err(invalid("channel entry requires non-empty 'name'")),
    };
    let channel_type = match entry.get("type") {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => s.clone(),
        _ => return Err(invalid("channel entry requires non-empty 'type'")),
    };
    let enabled = match entry.get("enabled") {
        None => true,
        Some(value) => parse_enabled(value)?,
    };
    Ok((name, channel_type, enabled))
}

fn parse_enabled(value: &serde_json::Value) -> Result<bool, TauriError> {
    match value {
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(invalid("channel entry 'enabled' must be a boolean")),
        },
        _ => Err(invalid("channel entry 'enabled' must be a boolean")),
    }
}

/// Decide whether a stored channel's config seeds the keep-current merge: only
/// an entry with the same channel type (mirrors Python matching name + type
/// before inheriting stored secrets).
fn keep_current_source<'a>(
    stored: Option<&'a ChannelConfig>,
    channel_type: &str,
) -> Option<&'a ChannelConfig> {
    stored.filter(|existing| existing.channel_type == channel_type)
}

/// Fold a frontend entry into `ChannelConfig.config`. `name`/`type`/`enabled`
/// are consumed by the identity fields; every other key is stored verbatim as
/// a string (numbers/booleans via their string form).
///
/// When `existing` is a same-type stored channel, its config keys seed the
/// map so keys the payload omits — or sends blank / `***` — survive an update.
/// This is a broadened form of Python's secret-only keep-current merge: S4
/// has no per-type channel spec registry to know which fields are secret, so
/// every config key is treated as keep-on-blank.
fn fold_config(
    entry: &serde_json::Map<String, serde_json::Value>,
    existing: Option<&ChannelConfig>,
) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Some(existing) = existing {
        for (key, value) in &existing.config {
            out.insert(key.clone(), value.clone());
        }
    }
    for (key, value) in entry {
        if key == "name" || key == "type" || key == "enabled" {
            continue;
        }
        match value {
            serde_json::Value::Null => {}
            serde_json::Value::String(s) => {
                if !s.trim().is_empty() && s != REDACTED_SENTINEL {
                    out.insert(key.clone(), s.clone());
                }
            }
            other => {
                out.insert(key.clone(), other.to_string());
            }
        }
    }
    out
}

/// Insert or replace a channel in `channels`, matched by name. Returns `true`
/// when an existing entry was replaced, `false` when a new entry was appended
/// (mirrors Python's `upsert_channel`).
fn apply_upsert(channels: &mut Vec<ChannelConfig>, channel: ChannelConfig) -> bool {
    let name = channel.name.clone();
    for existing in channels.iter_mut() {
        if existing.name == name {
            *existing = channel;
            return true;
        }
    }
    channels.push(channel);
    false
}

// ---------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------

/// `onboarding.channel.probe` — merge-aware local validation of a draft entry.
/// No network calls are made; connection state is derived from config presence
/// (fail-closed), mirroring Python's local-only probe.
#[tauri::command]
pub async fn onboarding_channel_probe(
    state: State<'_, AppState>,
    request: ChannelProbeRequest,
) -> TauriResult<ChannelProbeResponse> {
    let mut obj = request
        .entry
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("params.entry must be an object"))?;

    let (type_name, exists, enabled) = {
        let config = state.config().await;
        let type_name = validate_entry(&config, &mut obj)?;
        let name = entry_name(&obj)?;
        match config.find_channel(&name) {
            Some(ch) => (type_name, true, ch.enabled),
            None => (type_name, false, true),
        }
    };

    Ok(ChannelProbeResponse {
        status: "validated".to_string(),
        connected: exists && enabled,
        probe_kind: "local_validation".to_string(),
        restart_required: true,
        entry: redact_entry(&type_name, &obj),
        warnings: vec![
            "Configuration is locally valid; no provider connection was attempted."
                .to_string(),
        ],
    })
}

// ---------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------

/// `onboarding.channel.remove` — delete the channel with the given name.
#[tauri::command]
pub async fn onboarding_channel_remove(
    state: State<'_, AppState>,
    request: ChannelNameRequest,
) -> TauriResult<ChannelRemoveResponse> {
    let mut config = state.config_mut().await;
    if !apply_remove(&mut config.channels, &request.name) {
        return Err(not_found(&request.name));
    }
    config.save().map_err(TauriError::from)?;

    Ok(ChannelRemoveResponse {
        changed: true,
        restart_required: true,
        live_apply: None,
        config_path: config_path(),
        removed: request.name,
    })
}

fn apply_remove(channels: &mut Vec<ChannelConfig>, name: &str) -> bool {
    let before = channels.len();
    channels.retain(|c| c.name != name);
    channels.len() != before
}

// ---------------------------------------------------------------------------
// Enable / disable
// ---------------------------------------------------------------------------

/// `onboarding.channel.enable` — set `enabled = true` for the named channel.
#[tauri::command]
pub async fn onboarding_channel_enable(
    state: State<'_, AppState>,
    request: ChannelNameRequest,
) -> TauriResult<ChannelToggleResponse> {
    set_channel_enabled(&state, &request.name, true).await
}

/// `onboarding.channel.disable` — set `enabled = false` for the named channel.
#[tauri::command]
pub async fn onboarding_channel_disable(
    state: State<'_, AppState>,
    request: ChannelNameRequest,
) -> TauriResult<ChannelToggleResponse> {
    set_channel_enabled(&state, &request.name, false).await
}

async fn set_channel_enabled(
    state: &AppState,
    name: &str,
    enabled: bool,
) -> TauriResult<ChannelToggleResponse> {
    let mut config = state.config_mut().await;
    let Some(channel) = config.channels.iter_mut().find(|c| c.name == name) else {
        return Err(not_found(name));
    };
    channel.enabled = enabled;
    config.save().map_err(TauriError::from)?;

    Ok(ChannelToggleResponse {
        changed: true,
        restart_required: true,
        live_apply: None,
        config_path: config_path(),
        name: name.to_string(),
        enabled,
    })
}

// ---------------------------------------------------------------------------
// Status (channels.status)
// ---------------------------------------------------------------------------

/// Whether a stored channel has credential material. Uses the same per-type
/// secret-field list as upsert validation; unknown types fail closed to any
/// non-empty config value (mirrors probe's config-presence stance).
fn is_configured(channel: &ChannelConfig) -> bool {
    let secrets = secret_fields_for(&channel.channel_type);
    if secrets.is_empty() {
        channel.config.values().any(|s| !s.trim().is_empty())
    } else {
        secrets
            .iter()
            .any(|field| channel.config.get(*field).is_some_and(|s| !s.trim().is_empty()))
    }
}

/// A `channels.status` row, Python-compatible: the Overview KPI chip reads
/// `status`/`configured`/`pendingPairings`. Connection state is derived from
/// config presence (fail-closed, no live reconciler), matching probe.
fn channel_status_row(channel: &ChannelConfig) -> Value {
    let configured = is_configured(channel);
    let status = if configured && channel.enabled {
        "connected"
    } else {
        "stopped"
    };
    let mut diagnostics: Map<String, Value> = Map::new();
    diagnostics.insert("network_probe".to_string(), json!("not_run"));
    json!({
        "name": channel.name,
        "type": channel.channel_type,
        "enabled": channel.enabled,
        "configured": configured,
        "status": status,
        "connected": status == "connected",
        "pendingPairings": 0,
        "diagnostics": Value::Object(diagnostics),
    })
}

/// `channels.status` — list channel records with a status view, derived from
/// config (no network calls). Returns the Python-compatible `{ channels, count }`
/// shape the Overview health panel consumes.
#[tauri::command]
pub async fn channels_status(state: State<'_, AppState>) -> TauriResult<Value> {
    let config = state.config().await;
    let channels: Vec<Value> = config.channels.iter().map(channel_status_row).collect();
    Ok(json!({
        "channels": channels,
        "count": channels.len(),
    }))
}

fn config_path() -> Option<String> {
    Config::discover_path()
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().expect("test object").clone()
    }

    fn channel(name: &str, channel_type: &str, enabled: bool) -> ChannelConfig {
        ChannelConfig {
            name: name.to_string(),
            channel_type: channel_type.to_string(),
            enabled,
            config: HashMap::new(),
        }
    }

    fn stored_with(token: &str) -> ChannelConfig {
        let mut c = channel("ops", "slack", true);
        c.config.insert("token".to_string(), token.to_string());
        c
    }

    #[test]
    fn status_row_maps_configured_enabled_to_connected() {
        let row = channel_status_row(&stored_with("xoxb-1"));
        assert_eq!(row["status"].as_str(), Some("connected"));
        assert_eq!(row["connected"].as_bool(), Some(true));
        assert_eq!(row["configured"].as_bool(), Some(true));
        assert_eq!(row["name"].as_str(), Some("ops"));
        assert_eq!(row["type"].as_str(), Some("slack"));
        assert_eq!(row["pendingPairings"].as_u64(), Some(0));
    }

    #[test]
    fn status_row_maps_missing_credentials_to_stopped() {
        let row = channel_status_row(&channel("ops", "slack", true));
        assert_eq!(row["status"].as_str(), Some("stopped"));
        assert_eq!(row["connected"].as_bool(), Some(false));
        assert_eq!(row["configured"].as_bool(), Some(false));
    }

    #[test]
    fn status_row_maps_disabled_to_stopped() {
        let mut c = stored_with("xoxb-1");
        c.enabled = false;
        let row = channel_status_row(&c);
        assert_eq!(row["status"].as_str(), Some("stopped"));
        assert_eq!(row["configured"].as_bool(), Some(true));
    }

    #[test]
    fn status_row_unknown_type_configures_from_any_non_empty_value() {
        let mut c = channel("web", "webhook", true);
        c.config.insert("url".to_string(), "https://x".to_string());
        assert!(is_configured(&c));
        let row = channel_status_row(&c);
        assert_eq!(row["status"].as_str(), Some("connected"));
    }

    #[test]
    fn extract_identity_defaults_enabled_to_true() {
        let (name, ty, enabled) = extract_identity(&obj(json!({"type": "slack", "name": "ops"}))).unwrap();
        assert_eq!(name, "ops");
        assert_eq!(ty, "slack");
        assert!(enabled);
    }

    #[test]
    fn extract_identity_requires_name_type_and_bool_enabled() {
        assert!(extract_identity(&obj(json!({"name": "ops"}))).is_err());
        assert!(extract_identity(&obj(json!({"type": "slack"}))).is_err());
        assert!(extract_identity(&obj(json!({"type": "slack", "name": "ops", "enabled": "yes"}))).is_err());
        assert_eq!(
            extract_identity(&obj(json!({"type": "slack", "name": "ops", "enabled": "false"})))
                .unwrap()
                .2,
            false
        );
    }

    #[test]
    fn fold_new_entry_excludes_identity_and_folds_scalars() {
        let e = obj(json!({
            "type": "slack", "name": "ops", "enabled": true,
            "token": "xoxb-123", "slack_channel_id": "C1", "reply_in_thread": false
        }));
        let config = fold_config(&e, None);
        assert_eq!(config.get("token").map(String::as_str), Some("xoxb-123"));
        assert_eq!(config.get("slack_channel_id").map(String::as_str), Some("C1"));
        assert_eq!(config.get("reply_in_thread").map(String::as_str), Some("false"));
        assert!(!config.contains_key("name"));
        assert!(!config.contains_key("type"));
        assert!(!config.contains_key("enabled"));
    }

    #[test]
    fn fold_blank_or_sentinel_keeps_stored() {
        let existing = stored_with("stored");
        let blank = fold_config(&obj(json!({"type": "slack", "name": "ops", "token": ""})), Some(&existing));
        assert_eq!(blank.get("token").map(String::as_str), Some("stored"));
        let sentinel = fold_config(
            &obj(json!({"type": "slack", "name": "ops", "token": "***"})),
            Some(&existing),
        );
        assert_eq!(sentinel.get("token").map(String::as_str), Some("stored"));
    }

    #[test]
    fn fold_nonblank_overrides_stored() {
        let existing = stored_with("stored");
        let config = fold_config(&obj(json!({"type": "slack", "name": "ops", "token": "new"})), Some(&existing));
        assert_eq!(config.get("token").map(String::as_str), Some("new"));
    }

    #[test]
    fn keep_current_gated_on_same_type() {
        let existing = stored_with("stored");
        assert!(keep_current_source(Some(&existing), "slack").is_some());
        assert!(keep_current_source(Some(&existing), "discord").is_none());
        assert!(keep_current_source(None, "slack").is_none());
    }

    #[test]
    fn upsert_replaces_by_name_and_appends_new() {
        let mut channels = vec![channel("a", "slack", false)];
        assert!(apply_upsert(&mut channels, channel("a", "discord", true)));
        assert_eq!(channels.len(), 1);
        assert_eq!(channels[0].channel_type, "discord");
        assert!(channels[0].enabled);

        assert!(!apply_upsert(&mut channels, channel("b", "slack", false)));
        assert_eq!(channels.len(), 2);
    }

    #[test]
    fn remove_returns_whether_found() {
        let mut channels = vec![channel("a", "slack", false)];
        assert!(apply_remove(&mut channels, "a"));
        assert!(!apply_remove(&mut channels, "a"));
        assert!(channels.is_empty());
    }

    #[test]
    fn validate_rejects_unknown_type() {
        let cfg = Config::default();
        let mut e = obj(json!({"type": "nope", "name": "x"}));
        let err = validate_entry(&cfg, &mut e).unwrap_err();
        assert!(err.message.contains("unknown channel type"));
    }

    #[test]
    fn validate_rejects_missing_required_secret() {
        let cfg = Config::default();
        let mut e = obj(json!({"type": "slack", "name": "ops"}));
        let err = validate_entry(&cfg, &mut e).unwrap_err();
        assert!(err.message.contains("token"));
    }

    #[test]
    fn validate_slack_webhook_requires_signing_secret() {
        let cfg = Config::default();
        let mut e = obj(json!({"type": "slack", "name": "ops", "token": "xoxb-1"}));
        let err = validate_entry(&cfg, &mut e).unwrap_err();
        assert!(err.message.contains("signing_secret"));
    }

    #[test]
    fn validate_slack_socket_ok() {
        let cfg = Config::default();
        let mut e = obj(json!({
            "type": "slack",
            "name": "ops",
            "token": "xoxb-1",
            "connection_mode": "socket",
        }));
        let t = validate_entry(&cfg, &mut e).unwrap();
        assert_eq!(t, "slack");
        assert_eq!(e.get("agent_id").and_then(|v| v.as_str()), Some("main"));
        assert_eq!(e.get("enabled").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn validate_wecom_websocket_requires_bot_secret() {
        let cfg = Config::default();
        let mut e = obj(json!({
            "type": "wecom",
            "name": "ops",
            "connection_mode": "websocket",
            "bot_id": "bot123",
        }));
        let err = validate_entry(&cfg, &mut e).unwrap_err();
        assert!(err.message.contains("bot_secret"));
    }

    #[test]
    fn merge_keeps_stored_secret_on_blank() {
        let mut cfg = Config::default();
        cfg.channels.push(stored_with("stored-token"));
        let mut e = obj(json!({
            "type": "slack",
            "name": "ops",
            "token": "***",
            "connection_mode": "socket",
        }));
        let _ = validate_entry(&cfg, &mut e).unwrap();
        assert_eq!(e.get("token").and_then(|v| v.as_str()), Some("stored-token"));
    }

    #[test]
    fn redact_replaces_declared_secrets() {
        let e = obj(json!({
            "type": "slack",
            "name": "ops",
            "token": "xoxb-1",
            "signing_secret": "abc",
            "reply_in_thread": false,
        }));
        let redacted = redact_entry("slack", &e);
        assert_eq!(redacted["token"].as_str(), Some("***"));
        assert_eq!(redacted["signing_secret"].as_str(), Some("***"));
        assert_eq!(redacted["name"].as_str(), Some("ops"));
        assert_eq!(redacted["reply_in_thread"].as_bool(), Some(false));
    }

    #[test]
    fn redact_fallback_secret_like() {
        let e = obj(json!({
            "type": "terminal",
            "name": "t",
            "apiKey": "k",
            "color": false,
        }));
        let redacted = redact_entry("terminal", &e);
        assert_eq!(redacted["apiKey"].as_str(), Some("***"));
        assert_eq!(redacted["color"].as_bool(), Some(false));
        assert_eq!(redacted["name"].as_str(), Some("t"));
    }

    #[test]
    fn is_secret_like_matches_known() {
        assert!(is_secret_like("api_key"));
        assert!(is_secret_like("bot_secret"));
        assert!(is_secret_like("signing_secret"));
        assert!(is_secret_like("apiKey"));
        assert!(is_secret_like("token"));
        assert!(!is_secret_like("color"));
        assert!(!is_secret_like("webhook_path"));
    }
}
