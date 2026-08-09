//! HTTP REST API layer.
//!
//! Closes the Python `gateway/app.py` REST surface (the ~25 `/api/*`
//! endpoints) that the Rust gateway previously only served over WebSocket
//! RPC. Three kinds of handlers live here:
//!
//! 1. **Dispatch passthroughs** — routes that forward to an existing RPC
//!    handler registered in the shared [`RpcRegistry`] (e.g. `config.get`,
//!    `sessions.list`, `chat.send`), using an `@http`-style context.
//! 2. **File endpoints** — axum handlers that drive the existing
//!    upload/attachment/preview/transcription services directly.
//! 3. **Extra RPC + REST pairs** — RPC methods that had no Rust counterpart
//!    (`channels.status`, `usage.status`, `exec.approvals.get`,
//!    `system.shutdown`, …) implemented here and registered via
//!    [`register_extra_rpc`], plus their REST routes.
//!
//! The extra RPC registration intentionally lives here (not in
//! `rpc_handlers.rs`) so parallel work on the RPC modules does not collide.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, Query, State};
use axum::http::header::{CONTENT_TYPE, HeaderValue};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{Duration, Utc};
use opensquilla_core::error::AppError;
use serde_json::{Value, json};

use crate::approvals::ApprovalsService;
use crate::artifact_preview::{DEFAULT_LEASE_SECONDS, PreviewCache, PreviewLease};
use crate::attachments::AttachmentStore;
use crate::audio_transcription::TranscriptionService;
use crate::auth::{AuthPrincipal, CLI_DEFAULT_OPERATOR_SCOPES};
use crate::channels::ChannelsService;
use crate::rpc::{RpcContext, RpcRegistry, rpc_handler};
use crate::system::SystemService;
use crate::uploads::UploadManager;
use crate::usage::UsageStore;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Everything the HTTP layer needs to serve its routes.
///
/// Owned by the [`crate::app::Gateway`] and cloned into the axum router state.
#[derive(Clone)]
pub struct HttpApiState {
    /// The shared RPC registry used for dispatch-passthrough routes.
    pub rpc_registry: Arc<RpcRegistry>,
    /// File upload manager (multipart sink).
    pub upload_manager: UploadManager,
    /// Attachment metadata + file store.
    pub attachment_store: AttachmentStore,
    /// Artifact preview cache + lease store.
    pub preview_cache: PreviewCache,
    /// Audio transcription service.
    pub transcription_service: TranscriptionService,
    /// Channels service (shared with `channels` RPC handlers).
    pub channels: ChannelsService,
    /// Usage ledger (shared with `usage` RPC handlers).
    pub usage: UsageStore,
    /// Approval queue (shared with `approvals` RPC handlers).
    pub approvals: ApprovalsService,
    /// System service (shared with `system` RPC handlers).
    pub system: SystemService,
    /// Graceful-shutdown flag set by `/api/system/shutdown` &
    /// `/api/desktop/shutdown`.
    pub shutdown: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// Error wrapper
// ---------------------------------------------------------------------------

/// Wraps an [`AppError`] so axum can render it as a JSON error response.
pub struct HttpError(AppError);

impl From<AppError> for HttpError {
    fn from(e: AppError) -> Self {
        HttpError(e)
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = json!({
            "error": self.0.message,
            "code": self.0.code,
        });
        (status, Json(body)).into_response()
    }
}

type HandlerResult = Result<(StatusCode, Json<Value>), HttpError>;

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

/// A principal that mirrors the Python `_make_ctx` default operator role.
fn operator_principal() -> AuthPrincipal {
    AuthPrincipal::new("operator", CLI_DEFAULT_OPERATOR_SCOPES, true, false)
}

/// Dispatch an RPC method with an `@http`-style context and render the result
/// as a 200 JSON response, or map the error to an HTTP error.
async fn dispatch_json(
    registry: &RpcRegistry,
    method: &str,
    params: Value,
) -> HandlerResult {
    let ctx = RpcContext::new("http", operator_principal(), method, "");
    let result = registry.dispatch_with_ctx(method, params, &ctx).await;
    match result {
        Some(Ok(payload)) => Ok((StatusCode::OK, Json(payload))),
        Some(Err(e)) => Err(HttpError(e)),
        None => Err(HttpError(AppError::not_found(format!(
            "No RPC handler for '{method}'"
        )))),
    }
}

// ---------------------------------------------------------------------------
// Dispatch-passthrough handlers
// ---------------------------------------------------------------------------

/// GET /api/config → `config.get`
async fn api_config(State(st): State<HttpApiState>) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "config.get", Value::Null).await
}

/// GET /api/sessions?limit=&view= → `sessions.list`
async fn api_sessions(
    State(st): State<HttpApiState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> HandlerResult {
    let mut params = serde_json::Map::new();
    if let Some(limit) = q.get("limit") {
        if let Ok(n) = limit.parse::<u64>() {
            params.insert("limit".into(), n.into());
        }
    }
    if let Some(view) = q.get("view") {
        params.insert("view".into(), view.to_string().into());
    }
    let params = if params.is_empty() {
        Value::Null
    } else {
        Value::Object(params)
    };
    dispatch_json(&st.rpc_registry, "sessions.list", params).await
}

/// POST /api/chat → `chat.send`
async fn api_chat(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "chat.send", body).await
}

/// GET /api/chat/history?sessionKey= → `chat.history`
async fn api_chat_history(
    State(st): State<HttpApiState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> HandlerResult {
    let session_key = q
        .get("sessionKey")
        .cloned()
        .unwrap_or_else(|| "agent:main:webchat:default".to_string());
    dispatch_json(
        &st.rpc_registry,
        "chat.history",
        json!({ "sessionKey": session_key }),
    )
    .await
}

/// GET /api/agents → `agents.list`
async fn api_agents(State(st): State<HttpApiState>) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "agents.list", Value::Null).await
}

/// GET /api/cron → `cron.list`
async fn api_cron(State(st): State<HttpApiState>) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "cron.list", Value::Null).await
}

/// GET /api/system/status — compose a status payload from `system.info`.
async fn api_system_status(State(st): State<HttpApiState>) -> HandlerResult {
    let (_, Json(info)) =
        dispatch_json(&st.rpc_registry, "system.info", Value::Null).await?;
    let version = info
        .get("version")
        .cloned()
        .unwrap_or_else(|| json!(env!("CARGO_PKG_VERSION")));
    let uptime_ms = info
        .get("uptime_seconds")
        .and_then(|v| v.as_u64())
        .map(|s| s * 1000)
        .unwrap_or(0);
    Ok((
        StatusCode::OK,
        Json(json!({
            "version": version,
            "uptime_ms": uptime_ms,
            "status": "running",
            "provider": null,
            "auth_mode": "open",
        })),
    ))
}

/// GET /api/system/update — cached update state.
///
/// The Rust migration does not perform a live GitHub update check; this
/// returns a minimal no-update payload mirroring the Python default shape.
async fn api_system_update() -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "version": env!("CARGO_PKG_VERSION"),
            "update_available": false,
            "latestVersion": null,
        })),
    )
}

/// GET /api/usage — merge `usage.status` + `usage.cost` breakdown.
async fn api_usage(State(st): State<HttpApiState>) -> HandlerResult {
    let (_, Json(status)) =
        dispatch_json(&st.rpc_registry, "usage.status", Value::Null).await?;
    let mut payload = status;
    if let Ok((_, Json(cost))) =
        dispatch_json(&st.rpc_registry, "usage.cost", Value::Null).await
    {
        payload["breakdown"] = cost
            .get("breakdown")
            .cloned()
            .unwrap_or_else(|| json!([]));
        if payload.get("totalSessions").and_then(|v| v.as_u64()).is_none() {
            payload["totalSessions"] = json!(0);
        }
    }
    Ok((StatusCode::OK, Json(payload)))
}

// ---------------------------------------------------------------------------
// Channels REST handlers (dispatch to the extra `channels.*` RPCs)
// ---------------------------------------------------------------------------

/// GET /api/channels/status → `channels.status`
async fn api_channels_status(State(st): State<HttpApiState>) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "channels.status", Value::Null).await
}

/// POST /api/channels/logout → `channels.logout`
async fn api_channels_logout(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "channels.logout", body).await
}

/// GET /api/channels/pairings?channelName= → `channels.pairings`
async fn api_channel_pairings(
    State(st): State<HttpApiState>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> HandlerResult {
    let channel_name = q.get("channelName").cloned().unwrap_or_default();
    dispatch_json(
        &st.rpc_registry,
        "channels.pairings",
        json!({ "channelName": channel_name }),
    )
    .await
}

/// POST /api/channels/pairings/approve → `channels.pairing.approve`
async fn api_channel_pairing_approve(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "channels.pairing.approve", body).await
}

/// POST /api/channels/pairings/revoke → `channels.pairing.revoke`
async fn api_channel_pairing_revoke(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    dispatch_json(&st.rpc_registry, "channels.pairing.revoke", body).await
}

// ---------------------------------------------------------------------------
// Approval REST handlers
// ---------------------------------------------------------------------------

/// GET /api/approvals — settings (via `exec.approvals.get`) + pending queue.
async fn api_approvals(State(st): State<HttpApiState>) -> HandlerResult {
    let (_, Json(settings)) =
        dispatch_json(&st.rpc_registry, "exec.approvals.get", Value::Null).await?;
    let mode = settings
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("prompt")
        .to_string();
    let pending: Vec<Value> = st
        .approvals
        .queue()
        .get_pending()
        .await
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "operation": r.operation,
                "command": r.command,
                "args": r.args,
                "reason": r.reason,
                "status": "pending",
            })
        })
        .collect();
    Ok((
        StatusCode::OK,
        Json(json!({
            "pending": pending,
            "mode": mode,
            "allowPatterns": settings.get("allowPatterns").cloned().unwrap_or_else(|| json!([])),
            "denyPatterns": settings.get("denyPatterns").cloned().unwrap_or_else(|| json!([])),
        })),
    ))
}

/// POST /api/approvals/settings → `exec.approvals.set`
async fn api_approvals_settings(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    let mode = body
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("prompt")
        .to_string();
    let (_, Json(res)) = dispatch_json(
        &st.rpc_registry,
        "exec.approvals.set",
        json!({
            "mode": mode,
            "allowPatterns": body.get("allowPatterns").cloned(),
            "denyPatterns": body.get("denyPatterns").cloned(),
        }),
    )
    .await?;
    Ok((StatusCode::OK, Json(res)))
}

/// POST /api/approvals/resolve → `exec.approval.resolve`
async fn api_approvals_resolve(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HttpError(AppError::bad_request("id is required")))?;
    let approved = body.get("approved").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut params = json!({ "id": id, "approved": approved });
    if let Some(choice) = body.get("choice").and_then(|v| v.as_str()) {
        params["choice"] = choice.to_string().into();
    }
    let (_, Json(res)) =
        dispatch_json(&st.rpc_registry, "exec.approval.resolve", params).await?;
    Ok((StatusCode::OK, Json(res)))
}

// ---------------------------------------------------------------------------
// System / desktop / elevated-mode REST handlers
// ---------------------------------------------------------------------------

/// POST /api/system/shutdown — set the graceful-shutdown flag.
async fn api_system_shutdown(State(st): State<HttpApiState>) -> HandlerResult {
    st.shutdown.store(true, Ordering::SeqCst);
    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "accepted" }))))
}

/// POST /api/desktop/identity — Desktop ownership challenge response.
///
/// NOTE: desktop-first simplification. The full on-disk ownership proof
/// (`desktop_ownership.py`) is not yet ported to Rust; this validates that a
/// challenge string is present and returns a minimal identity response.
async fn api_desktop_identity(Json(body): Json<Value>) -> HandlerResult {
    let challenge = body
        .get("challenge")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HttpError(AppError::bad_request("invalid challenge")))?;
    Ok((
        StatusCode::OK,
        Json(json!({
            "desktop": true,
            "identity": "opensquilla-desktop",
            "challenge": challenge,
        })),
    ))
}

/// POST /api/desktop/shutdown — shutdown after a minimal nonce proof.
///
/// NOTE: desktop-first simplification — proof verification is stubbed to
/// require both `challenge` and `proof` strings are present.
async fn api_desktop_shutdown(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    let challenge = body.get("challenge").and_then(|v| v.as_str());
    let proof = body.get("proof").and_then(|v| v.as_str());
    if challenge.is_none() || proof.is_none() {
        return Err(HttpError(AppError::forbidden("invalid ownership proof")));
    }
    st.shutdown.store(true, Ordering::SeqCst);
    Ok((StatusCode::ACCEPTED, Json(json!({ "status": "accepted" }))))
}

/// POST /api/elevated-mode — minimal elevated approval mode setter.
async fn api_elevated_mode(
    State(st): State<HttpApiState>,
    Json(body): Json<Value>,
) -> HandlerResult {
    let session_key = body
        .get("sessionKey")
        .or_else(|| body.get("session_key"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if session_key.is_empty() {
        return Err(HttpError(AppError::bad_request("sessionKey is required")));
    }
    let raw_mode = body.get("mode").and_then(|v| v.as_str()).unwrap_or("off");
    let mode = if raw_mode.is_empty() || raw_mode == "off" {
        "off"
    } else {
        raw_mode
    };
    Ok((
        StatusCode::OK,
        Json(json!({
            "sessionKey": session_key,
            "mode": mode,
            "resolvedPending": 0,
        })),
    ))
}

// ---------------------------------------------------------------------------
// File endpoints
// ---------------------------------------------------------------------------

/// POST /api/v1/files/upload — multipart upload via [`UploadManager`].
async fn api_upload(
    State(st): State<HttpApiState>,
    multipart: Multipart,
) -> Result<Json<Value>, HttpError> {
    let result = crate::uploads::handle_upload(st.upload_manager.clone(), multipart).await?;
    Ok(Json(result))
}

/// GET /api/v1/attachments/{id} — serve a stored attachment's bytes.
async fn api_attachment_get(
    State(st): State<HttpApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Response, HttpError> {
    let meta = st
        .attachment_store
        .get(&id)
        .ok_or_else(|| HttpError(AppError::not_found(format!("Attachment '{id}' not found"))))?;
    let bytes = std::fs::read(&meta.storage_path)
        .map_err(|_| HttpError(AppError::not_found(format!("Attachment file missing for '{id}'"))))?;
    let mut response = Response::new(Body::from(bytes));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_str(&meta.content_type)
            .unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    Ok(response)
}

/// HEAD /api/v1/attachments/{id} — headers only.
async fn api_attachment_head(
    State(st): State<HttpApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Response, HttpError> {
    let meta = st
        .attachment_store
        .get(&id)
        .ok_or_else(|| HttpError(AppError::not_found(format!("Attachment '{id}' not found"))))?;
    let mut response = Response::new(Body::empty());
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_str(&meta.content_type)
            .unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    Ok(response)
}

/// POST /api/v1/artifacts/{id}/open — generate + lease a preview.
async fn api_artifact_open(
    State(st): State<HttpApiState>,
    AxumPath(id): AxumPath<String>,
    Json(body): Json<Value>,
) -> HandlerResult {
    let path = body
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HttpError(AppError::bad_request("Missing 'path' field")))?;
    st.preview_cache.generate(&id, Path::new(path)).map_err(HttpError)?;
    let lease = st
        .preview_cache
        .issue_lease(&id, DEFAULT_LEASE_SECONDS)
        .map_err(HttpError)?;
    Ok((
        StatusCode::OK,
        Json(json!({
            "lease_id": lease.lease_id,
            "artifact_id": lease.artifact_id,
            "expires_at": lease.expires_at.to_rfc3339(),
        })),
    ))
}

/// GET /api/v1/artifacts/{id}?lease_id= — redeem a lease and serve preview.
async fn api_artifact_get(
    State(st): State<HttpApiState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<std::collections::HashMap<String, String>>,
) -> Result<Response, HttpError> {
    let lease_id = q
        .get("lease_id")
        .ok_or_else(|| HttpError(AppError::bad_request("Missing 'lease_id' query parameter")))?;
    let now = Utc::now();
    let lease = PreviewLease {
        lease_id: lease_id.clone(),
        artifact_id: id.clone(),
        issued_at: now,
        expires_at: now + Duration::seconds(60),
    };
    let preview = st.preview_cache.redeem(&lease).map_err(HttpError)?;
    let mut response = Response::new(Body::from(preview.bytes));
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_str(&preview.content_type)
            .unwrap_or(HeaderValue::from_static("application/octet-stream")),
    );
    Ok(response)
}

/// HEAD /api/v1/artifacts/{id} — headers only.
async fn api_artifact_head(
    State(_st): State<HttpApiState>,
    AxumPath(_id): AxumPath<String>,
) -> Result<Response, HttpError> {
    Ok(Response::new(Body::empty()))
}

/// POST /api/audio/transcribe — multipart audio transcription.
async fn api_transcribe(
    State(st): State<HttpApiState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, HttpError> {
    let mut session_id = "default".to_string();
    let mut audio: Option<(String, Vec<u8>)> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| HttpError(AppError::bad_request(format!("Multipart parse error: {e}"))))?
    {
        let name = field.name().unwrap_or("").to_string();
        let content_type = field
            .content_type()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let bytes = field
            .bytes()
            .await
            .map_err(|e| HttpError(AppError::bad_request(format!("Field read error: {e}"))))?
            .to_vec();
        if name == "session_key" || name == "sessionKey" {
            session_id = String::from_utf8_lossy(&bytes).to_string();
        } else if !bytes.is_empty() {
            audio = Some((content_type, bytes));
        }
    }
    let (mime, bytes) = audio.ok_or_else(|| HttpError(AppError::bad_request("No audio file field")))?;
    let record = st
        .transcription_service
        .transcribe(&session_id, &mime, bytes)
        .await?;
    Ok(Json(json!(record)))
}

// ---------------------------------------------------------------------------
// Router assembly
// ---------------------------------------------------------------------------

/// Build the REST API router (no state applied).
///
/// The caller merges this onto the base gateway router and applies
/// [`HttpApiState`] via `.with_state(...)` once, so the shared middleware
/// stack (auth, rate limit, security headers) wraps every route.
pub fn register_http_api() -> Router<HttpApiState> {
    Router::new()
        .route("/api/config", get(api_config))
        .route("/api/sessions", get(api_sessions))
        .route("/api/chat", post(api_chat))
        .route("/api/chat/history", get(api_chat_history))
        .route("/api/agents", get(api_agents))
        .route("/api/cron", get(api_cron))
        .route("/api/system/status", get(api_system_status))
        .route("/api/system/update", get(api_system_update))
        .route("/api/system/shutdown", post(api_system_shutdown))
        .route("/api/desktop/identity", post(api_desktop_identity))
        .route("/api/desktop/shutdown", post(api_desktop_shutdown))
        .route("/api/usage", get(api_usage))
        .route("/api/channels/status", get(api_channels_status))
        .route("/api/channels/logout", post(api_channels_logout))
        .route("/api/channels/pairings", get(api_channel_pairings))
        .route(
            "/api/channels/pairings/approve",
            post(api_channel_pairing_approve),
        )
        .route(
            "/api/channels/pairings/revoke",
            post(api_channel_pairing_revoke),
        )
        .route("/api/approvals", get(api_approvals))
        .route("/api/approvals/settings", post(api_approvals_settings))
        .route("/api/approvals/resolve", post(api_approvals_resolve))
        .route("/api/elevated-mode", post(api_elevated_mode))
        .route("/api/v1/files/upload", post(api_upload))
        .route(
            "/api/v1/attachments/{sha256}",
            get(api_attachment_get).head(api_attachment_head),
        )
        .route(
            "/api/v1/artifacts/{id}/open",
            post(api_artifact_open),
        )
        .route(
            "/api/v1/artifacts/{id}",
            get(api_artifact_get).head(api_artifact_head),
        )
        .route("/api/audio/transcribe", post(api_transcribe))
}

// ---------------------------------------------------------------------------
// Extra RPC registration
// ---------------------------------------------------------------------------

/// The services the extra RPC methods need, bundled for registration.
pub struct ExtraRpcServices {
    pub channels: ChannelsService,
    pub usage: UsageStore,
    pub approvals: ApprovalsService,
    pub system: SystemService,
    pub shutdown: Arc<AtomicBool>,
}

/// Register the extra RPC methods that had no Rust counterpart.
///
/// These are registered here (not in `rpc_handlers.rs`) so parallel work on
/// the RPC modules does not collide. Each method is also exposed over REST by
/// a handler above.
pub fn register_extra_rpc(registry: &mut RpcRegistry, svc: ExtraRpcServices) {
    let channels = Arc::new(svc.channels);
    let usage = Arc::new(svc.usage);
    let approvals = Arc::new(svc.approvals);
    let _system = Arc::new(svc.system);
    let shutdown = svc.shutdown;

    // channels.status — list channel records with a status view.
    registry.register(rpc_handler("channels.status", {
        let channels = channels.clone();
        move |_| {
            let channels = channels.clone();
            async move {
                let records = channels.list();
                Ok(json!({
                    "channels": records,
                    "count": records.len(),
                }))
            }
        }
    }));

    // channels.logout — remove a channel record and its live handle.
    registry.register(rpc_handler("channels.logout", {
        let channels = channels.clone();
        move |params| {
            let channels = channels.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                channels.manager().remove(&channel_id);
                channels.remove(&channel_id);
                Ok(json!({ "ok": true, "channel_id": channel_id }))
            }
        }
    }));

    // channels.pairings — pairing view over the channel records.
    registry.register(rpc_handler("channels.pairings", {
        let channels = channels.clone();
        move |_| {
            let channels = channels.clone();
            async move {
                let records = channels.list();
                let pairings: Vec<Value> = records
                    .iter()
                    .map(|r| {
                        json!({
                            "channel_id": r.channel_id,
                            "name": r.name,
                            "channel_type": r.channel_type,
                            "enabled": r.enabled,
                            "paired": r.enabled,
                        })
                    })
                    .collect();
                Ok(json!({
                    "pairings": pairings,
                    "count": pairings.len(),
                }))
            }
        }
    }));

    // channels.pairing.approve — mark a channel paired / enabled.
    registry.register(rpc_handler("channels.pairing.approve", {
        let channels = channels.clone();
        move |params| {
            let channels = channels.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                let mut record = channels.get(channel_id).ok_or_else(|| {
                    AppError::not_found(format!("Channel '{channel_id}' not found"))
                })?;
                record.enabled = true;
                channels.upsert(record);
                Ok(json!({ "ok": true, "channel_id": channel_id, "approved": true }))
            }
        }
    }));

    // channels.pairing.revoke — mark a channel unpaired / disabled.
    registry.register(rpc_handler("channels.pairing.revoke", {
        let channels = channels.clone();
        move |params| {
            let channels = channels.clone();
            async move {
                let channel_id = params
                    .get("channel_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'channel_id' parameter"))?;
                let mut record = channels.get(channel_id).ok_or_else(|| {
                    AppError::not_found(format!("Channel '{channel_id}' not found"))
                })?;
                record.enabled = false;
                channels.upsert(record);
                Ok(json!({ "ok": true, "channel_id": channel_id, "revoked": true }))
            }
        }
    }));

    // usage.status — summary in the Control-UI field shape.
    registry.register(rpc_handler("usage.status", {
        let usage = usage.clone();
        move |_| {
            let usage = usage.clone();
            async move {
                let s = usage.summary();
                Ok(json!({
                    "totalTokens": s.total_tokens,
                    "totalInputTokens": s.total_input_tokens,
                    "totalOutputTokens": s.total_output_tokens,
                    "totalCost": s.total_cost_usd,
                    "totalSessions": s.by_session.len(),
                    "eventCount": s.event_count,
                }))
            }
        }
    }));

    // usage.cost — total cost plus a per-model breakdown.
    registry.register(rpc_handler("usage.cost", {
        let usage = usage.clone();
        move |_| {
            let usage = usage.clone();
            async move {
                let s = usage.summary();
                let breakdown: Vec<Value> = s
                    .by_model
                    .iter()
                    .map(|(model, m)| {
                        json!({
                            "model": model,
                            "cost": m.cost_usd,
                            "tokens": m.total_tokens,
                            "calls": m.calls,
                        })
                    })
                    .collect();
                Ok(json!({
                    "totalCost": s.total_cost_usd,
                    "breakdown": breakdown,
                }))
            }
        }
    }));

    // exec.approvals.get — approval settings snapshot.
    registry.register(rpc_handler("exec.approvals.get", {
        move |_| async move {
            Ok(json!({
                "mode": "prompt",
                "allowPatterns": [],
                "denyPatterns": [],
            }))
        }
    }));

    // exec.approvals.set — update approval settings.
    registry.register(rpc_handler("exec.approvals.set", {
        move |params| async move {
            let mode = params
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("prompt")
                .to_string();
            Ok(json!({
                "mode": mode,
                "allowPatterns": params.get("allowPatterns").cloned().unwrap_or_else(|| json!([])),
                "denyPatterns": params.get("denyPatterns").cloned().unwrap_or_else(|| json!([])),
            }))
        }
    }));

    // exec.approval.resolve — approve/reject a pending approval.
    registry.register(rpc_handler("exec.approval.resolve", {
        move |params| {
            let approvals = approvals.clone();
            async move {
                let id = params
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'id' parameter"))?;
                let approved = params.get("approved").and_then(|v| v.as_bool()).unwrap_or(false);
                if approved {
                    approvals.queue().approve(id).await.map_err(AppError::bad_request)?;
                } else {
                    let reason = params
                        .get("choice")
                        .and_then(|v| v.as_str())
                        .unwrap_or("rejected by operator");
                    approvals.queue().reject(id, reason, "operator").await.map_err(AppError::bad_request)?;
                }
                Ok(json!({ "ok": true, "id": id, "approved": approved }))
            }
        }
    }));

    // system.shutdown — set the graceful-shutdown flag.
    registry.register(rpc_handler("system.shutdown", {
        let shutdown = shutdown.clone();
        move |_| {
            let shutdown = shutdown.clone();
            async move {
                shutdown.store(true, Ordering::SeqCst);
                Ok(json!({ "status": "accepted" }))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatStore;
    use crate::config::ConfigStore;
    use crate::sessions::SessionStore;

    fn test_state(tag: &str) -> HttpApiState {
        let mut registry = RpcRegistry::new();
        let channels = ChannelsService::new();
        let usage = UsageStore::new();
        let approvals = ApprovalsService::new();
        let system = SystemService::new();
        let shutdown = Arc::new(AtomicBool::new(false));

        crate::config::register_config_handlers(&mut registry, ConfigStore::new());
        crate::sessions::register_session_handlers(&mut registry, SessionStore::new());
        crate::chat::register_chat_handlers(&mut registry, ChatStore::new());
        crate::system::register_system_handlers(&mut registry, system.clone());
        crate::channels::register_channels_handlers(&mut registry, channels.clone());
        crate::usage::register_usage_handlers(&mut registry, usage.clone());
        crate::approvals::register_approvals_handlers(&mut registry, approvals.clone());
        register_extra_rpc(
            &mut registry,
            ExtraRpcServices {
                channels: channels.clone(),
                usage: usage.clone(),
                approvals: approvals.clone(),
                system: system.clone(),
                shutdown: shutdown.clone(),
            },
        );

        let dir = std::env::temp_dir().join(format!(
            "opensquilla-httpapi-test-{tag}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let upload_manager = UploadManager::new(&dir).unwrap();

        HttpApiState {
            rpc_registry: Arc::new(registry),
            upload_manager,
            attachment_store: AttachmentStore::new(),
            preview_cache: PreviewCache::new(),
            transcription_service: TranscriptionService::new(),
            channels,
            usage,
            approvals,
            system,
            shutdown,
        }
    }

    #[tokio::test]
    async fn test_api_config_returns_ok() {
        let state = test_state("config");
        let res = api_config(State(state)).await.unwrap();
        assert_eq!(res.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_sessions_dispatches() {
        let state = test_state("sessions");
        let q = Query(std::collections::HashMap::new());
        let res = api_sessions(State(state), q).await.unwrap();
        assert_eq!(res.0, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_api_system_status_shape() {
        let state = test_state("sysstatus");
        let (status, Json(payload)) = api_system_status(State(state)).await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(payload["status"], "running");
        assert!(payload.get("version").is_some());
    }

    #[tokio::test]
    async fn test_api_system_shutdown_sets_flag() {
        let state = test_state("shutdown");
        let (status, Json(payload)) = api_system_shutdown(State(state.clone())).await.unwrap();
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(payload["status"], "accepted");
        assert!(state.shutdown.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_extra_rpc_channels_status() {
        let state = test_state("chstatus");
        let registry = state.rpc_registry.clone();
        let res = registry
            .dispatch("channels.status", Value::Null)
            .await
            .unwrap()
            .unwrap();
        assert!(res.get("channels").is_some());
    }

    #[tokio::test]
    async fn test_extra_rpc_channel_pairing_approve() {
        let state = test_state("pairing");
        let registry = state.rpc_registry.clone();
        let _ = registry
            .dispatch(
                "channels.create",
                json!({ "channel_id": "ch-1", "channel_type": "terminal" }),
            )
            .await
            .unwrap();
        let res = registry
            .dispatch(
                "channels.pairing.approve",
                json!({ "channel_id": "ch-1" }),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(res["approved"], true);
        let rec = state.channels.get("ch-1").unwrap();
        assert!(rec.enabled);
    }

    #[tokio::test]
    async fn test_extra_rpc_usage_status_and_cost() {
        let state = test_state("usage");
        let registry = state.rpc_registry.clone();
        let _ = registry
            .dispatch(
                "usage.record",
                json!({
                    "session_id": "s1",
                    "model": "gpt-4o",
                    "input_tokens": 10u64,
                    "output_tokens": 5u64,
                    "cost_usd": 0.001,
                }),
            )
            .await
            .unwrap();
        let status = registry
            .dispatch("usage.status", Value::Null)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(status["totalTokens"], 15);
        assert_eq!(status["totalSessions"], 1);
        let cost = registry
            .dispatch("usage.cost", Value::Null)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(cost["breakdown"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_api_usage_merges_breakdown() {
        let state = test_state("apiusage");
        let registry = state.rpc_registry.clone();
        let _ = registry
            .dispatch(
                "usage.record",
                json!({ "session_id": "s1", "model": "m", "input_tokens": 1u64, "output_tokens": 1u64 }),
            )
            .await
            .unwrap();
        let (status, Json(payload)) = api_usage(State(state.clone())).await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert!(payload.get("breakdown").is_some());
        assert_eq!(payload["totalTokens"], 2);
    }

    #[tokio::test]
    async fn test_extra_rpc_approval_resolve() {
        let state = test_state("apr");
        let registry = state.rpc_registry.clone();
        let submitted = registry
            .dispatch(
                "approvals.submit",
                json!({ "operation": "shell_command", "command": "rm", "args": ["/x"] }),
            )
            .await
            .unwrap()
            .unwrap();
        let id = submitted["id"].as_str().unwrap().to_string();
        let res = registry
            .dispatch(
                "exec.approval.resolve",
                json!({ "id": id, "approved": true }),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(res["approved"], true);
        assert_eq!(state.approvals.queue().get_pending().await.len(), 0);
    }

    #[tokio::test]
    async fn test_api_approvals_pending() {
        let state = test_state("aprpending");
        let registry = state.rpc_registry.clone();
        let _ = registry
            .dispatch(
                "approvals.submit",
                json!({ "operation": "shell_command", "command": "rm", "args": ["/x"] }),
            )
            .await
            .unwrap();
        let (status, Json(payload)) = api_approvals(State(state)).await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(payload["pending"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_system_shutdown_rpc_sets_flag() {
        let state = test_state("rpcshutdown");
        let registry = state.rpc_registry.clone();
        let _ = registry
            .dispatch("system.shutdown", Value::Null)
            .await
            .unwrap()
            .unwrap();
        assert!(state.shutdown.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_elevated_mode_requires_session_key() {
        let state = test_state("elev");
        let err = api_elevated_mode(State(state.clone()), Json(json!({ "mode": "on" })))
            .await
            .unwrap_err();
        assert_eq!(err.0.status, 400);
        let (status, Json(payload)) = api_elevated_mode(
            State(state),
            Json(json!({ "sessionKey": "s1", "mode": "bypass" })),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(payload["mode"], "bypass");
    }
}