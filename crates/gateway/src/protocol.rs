//! WebSocket protocol frame types for the gateway.
//!
//! These types model the JSON messages exchanged over WebSocket connections
//! between the gateway and its clients. The protocol follows a
//! request-response model with server-sent events and error frames.
//!
//! The module provides two layers:
//!
//! 1. The legacy unified [`GatewayMessage`] envelope (request/response/event/
//!    error) used by the simple dispatch path.
//! 2. The full Python-protocol frame model — [`ReqFrame`], [`ResFrame`],
//!    [`WsEventFrame`], [`PingFrame`]/[`PongFrame`], and the handshake frames
//!    ([`HelloOk`], [`ConnectParams`], …) with constants, error codes, and
//!    version negotiation.

use serde::{Deserialize, Serialize};

/// Protocol version negotiated during the WebSocket handshake.
pub const PROTOCOL_VERSION: i32 = 3;

// ---------------------------------------------------------------------------
// Payload / timing constants
// ---------------------------------------------------------------------------

/// Maximum size of a single WebSocket payload frame (25 MiB).
pub const MAX_PAYLOAD_BYTES: u64 = 26_214_400;
/// Maximum buffered bytes per connection before backpressure kicks in (50 MiB).
pub const MAX_BUFFERED_BYTES: u64 = 52_428_800;
/// Maximum payload size accepted before authentication (64 KiB).
pub const MAX_PREAUTH_PAYLOAD_BYTES: u64 = 65_536;
/// Interval between liveness `tick` events, in milliseconds.
pub const TICK_INTERVAL_MS: u64 = 30_000;
/// Health refresh interval, in milliseconds.
pub const HEALTH_REFRESH_INTERVAL_MS: u64 = 60_000;
/// Time the client has to complete the connect handshake, in milliseconds.
pub const PREAUTH_TIMEOUT_MS: u64 = 10_000;
/// TTL for deduplicated inbound frames, in milliseconds.
pub const DEDUPE_TTL_MS: u64 = 300_000;
/// Maximum number of dedupe entries retained per connection.
pub const DEDUPE_MAX_ENTRIES: usize = 1000;
/// WebSocket close code used for graceful shutdown / service restart.
pub const WS_CLOSE_SERVICE_RESTART: u16 = 1012;

// ---------------------------------------------------------------------------
// Frame type name constants
// ---------------------------------------------------------------------------

/// RPC request frame type.
pub const FRAME_REQ: &str = "req";
/// RPC response frame type.
pub const FRAME_RES: &str = "res";
/// Server-pushed event frame type.
pub const FRAME_EVENT: &str = "event";
/// Client keepalive ping frame type.
pub const FRAME_PING: &str = "ping";
/// Server keepalive pong frame type.
pub const FRAME_PONG: &str = "pong";
/// Successful connect-handshake frame type.
pub const FRAME_HELLO_OK: &str = "hello-ok";

// ---------------------------------------------------------------------------
// Standard error codes
// ---------------------------------------------------------------------------

/// The request references a session that is not linked to this gateway.
pub const ERROR_NOT_LINKED: &str = "NOT_LINKED";
/// The request references a device that is not paired.
pub const ERROR_NOT_PAIRED: &str = "NOT_PAIRED";
/// The agent turn timed out.
pub const ERROR_AGENT_TIMEOUT: &str = "AGENT_TIMEOUT";
/// The request frame was malformed or invalid.
pub const ERROR_INVALID_REQUEST: &str = "INVALID_REQUEST";
/// The referenced approval does not exist.
pub const ERROR_APPROVAL_NOT_FOUND: &str = "APPROVAL_NOT_FOUND";
/// The requested resource is temporarily unavailable.
pub const ERROR_UNAVAILABLE: &str = "UNAVAILABLE";
/// The caller is not authorized.
pub const ERROR_UNAUTHORIZED: &str = "UNAUTHORIZED";
/// The referenced resource was not found.
pub const ERROR_NOT_FOUND: &str = "NOT_FOUND";
/// The RPC method does not exist.
pub const ERROR_METHOD_NOT_FOUND: &str = "METHOD_NOT_FOUND";

/// A top-level message exchanged over the WebSocket connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GatewayMessage {
    /// A client-initiated RPC request.
    Request(RequestFrame),
    /// A server response to a prior request.
    Response(ResponseFrame),
    /// A server-pushed event (not tied to a specific request).
    Event(EventFrame),
    /// An error response to a prior request.
    Error(ErrorFrame),
}

/// An RPC request frame sent from the client to the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFrame {
    /// Opaque request identifier echoed back in the response.
    pub id: String,
    /// The RPC method name to invoke.
    pub method: String,
    /// Parameters for the method, as an arbitrary JSON value.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A successful response frame sent from the server to the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFrame {
    /// The request identifier this response corresponds to.
    pub id: String,
    /// The result payload.
    pub result: serde_json::Value,
}

/// A server-pushed event frame (not tied to a specific RPC request).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFrame {
    /// The event type name.
    pub event: String,
    /// The event payload.
    #[serde(default)]
    pub data: serde_json::Value,
}

/// An error response frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorFrame {
    /// The request identifier this error corresponds to.
    pub id: String,
    /// A machine-readable error code.
    pub code: i32,
    /// A human-readable error message.
    pub message: String,
    /// Optional additional error data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl GatewayMessage {
    /// Serialize this message to a JSON string suitable for sending over the
    /// WebSocket.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.into())
    }

    /// Try to parse a `GatewayMessage` from a raw JSON string.
    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

impl From<RequestFrame> for GatewayMessage {
    fn from(f: RequestFrame) -> Self {
        GatewayMessage::Request(f)
    }
}

impl From<ResponseFrame> for GatewayMessage {
    fn from(f: ResponseFrame) -> Self {
        GatewayMessage::Response(f)
    }
}

impl From<EventFrame> for GatewayMessage {
    fn from(f: EventFrame) -> Self {
        GatewayMessage::Event(f)
    }
}

impl From<ErrorFrame> for GatewayMessage {
    fn from(f: ErrorFrame) -> Self {
        GatewayMessage::Error(f)
    }
}

impl ResponseFrame {
    /// Create a new response frame for the given request ID and result.
    pub fn new(id: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            id: id.into(),
            result,
        }
    }
}

impl ErrorFrame {
    /// Create a new error frame.
    pub fn new(id: impl Into<String>, code: i32, message: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Attach optional data to this error frame.
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}

impl EventFrame {
    /// Create a new event frame.
    pub fn new(event: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// Python-protocol frame model
// ---------------------------------------------------------------------------

/// The structured error shape attached to a failed [`ResFrame`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorShape {
    /// A machine-readable error code.
    pub code: String,
    /// A human-readable error message.
    pub message: String,
    /// Optional additional error details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    /// Whether the caller may safely retry the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retryable: Option<bool>,
    /// Server-suggested retry delay, in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// Whether the request was accepted for asynchronous processing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted: Option<bool>,
}

impl ErrorShape {
    /// Create a new error shape.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
            retryable: None,
            retry_after_ms: None,
            accepted: None,
        }
    }

    /// Mark this error as retryable with an optional retry delay.
    pub fn retryable(mut self, retryable: bool, retry_after_ms: Option<u64>) -> Self {
        self.retryable = Some(retryable);
        self.retry_after_ms = retry_after_ms;
        self
    }

    /// Attach additional details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Monotonic version counters for presence and health state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateVersion {
    /// Presence state version.
    #[serde(default)]
    pub presence: u64,
    /// Health state version.
    #[serde(default)]
    pub health: u64,
}

/// An RPC request frame (Python-protocol style: `{"type":"req",…}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReqFrame {
    /// Always `req`.
    #[serde(rename = "type")]
    pub frame_type: String,
    /// Opaque request identifier echoed back in the response.
    pub id: String,
    /// The RPC method name to invoke.
    pub method: String,
    /// Parameters for the method, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl ReqFrame {
    /// Create a new request frame.
    pub fn new(
        id: impl Into<String>,
        method: impl Into<String>,
        params: Option<serde_json::Value>,
    ) -> Self {
        Self {
            frame_type: FRAME_REQ.to_string(),
            id: id.into(),
            method: method.into(),
            params,
        }
    }
}

/// An RPC response frame (Python-protocol style: `{"type":"res",…}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResFrame {
    /// Always `res`.
    #[serde(rename = "type")]
    pub frame_type: String,
    /// The request identifier this response corresponds to.
    pub id: String,
    /// Whether the request succeeded.
    pub ok: bool,
    /// The result payload on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// The structured error on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorShape>,
}

impl ResFrame {
    /// Create a new successful response frame.
    pub fn ok(id: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            frame_type: FRAME_RES.to_string(),
            id: id.into(),
            ok: true,
            payload: Some(payload),
            error: None,
        }
    }

    /// Create a new error response frame.
    pub fn err(id: impl Into<String>, error: ErrorShape) -> Self {
        Self {
            frame_type: FRAME_RES.to_string(),
            id: id.into(),
            ok: false,
            payload: None,
            error: Some(error),
        }
    }
}

/// A server-pushed event frame (Python-protocol style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsEventFrame {
    /// Always `event`.
    #[serde(rename = "type")]
    pub frame_type: String,
    /// The event type name.
    pub event: String,
    /// The event payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
    /// Optional metadata attached to the event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
    /// Per-connection monotonic sequence number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seq: Option<u64>,
    /// Optional state version snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_version: Option<StateVersion>,
}

impl WsEventFrame {
    /// Create a new event frame.
    pub fn new(event: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            frame_type: FRAME_EVENT.to_string(),
            event: event.into(),
            payload: Some(payload),
            meta: None,
            seq: None,
            state_version: None,
        }
    }

    /// Attach a per-connection sequence number.
    pub fn with_seq(mut self, seq: u64) -> Self {
        self.seq = Some(seq);
        self
    }

    /// Attach metadata.
    pub fn with_meta(mut self, meta: serde_json::Value) -> Self {
        self.meta = Some(meta);
        self
    }
}

/// A client keepalive ping frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingFrame {
    /// Always `ping`.
    #[serde(rename = "type")]
    pub frame_type: String,
}

/// A server keepalive pong frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PongFrame {
    /// Always `pong`.
    #[serde(rename = "type")]
    pub frame_type: String,
}

impl Default for PingFrame {
    fn default() -> Self {
        Self {
            frame_type: FRAME_PING.to_string(),
        }
    }
}

impl Default for PongFrame {
    fn default() -> Self {
        Self {
            frame_type: FRAME_PONG.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Handshake frames
// ---------------------------------------------------------------------------

/// Metadata describing the connecting client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    /// A stable client identifier.
    pub id: String,
    /// Optional human-readable display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Client software version.
    pub version: String,
    /// Client platform (e.g. `win32`, `darwin`, `web`).
    pub platform: String,
    /// Optional device family.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_family: Option<String>,
    /// Optional device model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_identifier: Option<String>,
    /// Client mode (e.g. `desktop`, `webui`, `cli`).
    pub mode: String,
    /// Optional per-instance identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
}

/// Parameters of the `connect` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectParams {
    /// Minimum protocol version the client accepts.
    pub min_protocol: i32,
    /// Maximum protocol version the client accepts.
    pub max_protocol: i32,
    /// Client metadata.
    pub client: ClientInfo,
    /// Optional capability flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caps: Option<Vec<String>>,
    /// Optional supported commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commands: Option<Vec<String>>,
    /// Optional permission flags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<serde_json::Value>,
    /// Optional `PATH` environment for subprocesses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_env: Option<String>,
    /// Role claim; defaults to `operator`.
    #[serde(default = "default_role")]
    pub role: String,
    /// Optional declared scopes (the server computes the authoritative set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,
    /// Optional authentication parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<serde_json::Value>,
    /// Optional locale hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    /// Optional user agent string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
}

fn default_role() -> String {
    "operator".to_string()
}

/// Server identity announced in [`HelloOk`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Gateway version string.
    pub version: String,
    /// The connection identifier assigned to this socket.
    pub conn_id: String,
}

/// The set of RPC methods and event types the server supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeaturesInfo {
    /// Registered RPC method names.
    pub methods: Vec<String>,
    /// Event names the server may emit.
    pub events: Vec<String>,
}

/// A point-in-time snapshot of presence, health, and environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// Current presence records.
    #[serde(default)]
    pub presence: Vec<serde_json::Value>,
    /// Health state, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health: Option<serde_json::Value>,
    /// State version counters.
    #[serde(default)]
    pub state_version: StateVersion,
    /// Gateway uptime in milliseconds.
    #[serde(default)]
    pub uptime_ms: u64,
    /// Configuration file path, if exposed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// State directory, if exposed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_dir: Option<String>,
    /// Authentication mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
}

/// Connection policy advertised to the client in [`HelloOk`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyInfo {
    /// Maximum payload size in bytes.
    pub max_payload: u64,
    /// Maximum buffered bytes before backpressure.
    pub max_buffered_bytes: u64,
    /// Tick interval in milliseconds.
    pub tick_interval_ms: u64,
    /// Whether concurrent history reads are supported.
    pub concurrent_history_reads: bool,
    /// Agent stream heartbeat interval in milliseconds.
    pub agent_stream_heartbeat_interval_ms: u64,
    /// Agent stream idle timeout in milliseconds.
    pub agent_stream_idle_timeout_ms: u64,
    /// Web UI stream idle grace in milliseconds.
    pub webui_stream_idle_grace_ms: u64,
    /// Client WebSocket keepalive timeout in milliseconds.
    pub client_ws_keepalive_timeout_ms: u64,
}

impl Default for PolicyInfo {
    fn default() -> Self {
        Self {
            max_payload: MAX_PAYLOAD_BYTES,
            max_buffered_bytes: MAX_BUFFERED_BYTES,
            tick_interval_ms: TICK_INTERVAL_MS,
            concurrent_history_reads: false,
            agent_stream_heartbeat_interval_ms: 15_000,
            agent_stream_idle_timeout_ms: 600_000,
            webui_stream_idle_grace_ms: 630_000,
            client_ws_keepalive_timeout_ms: 120_000,
        }
    }
}

/// The successful handshake response sent after the `connect` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloOk {
    /// Always `hello-ok`.
    #[serde(rename = "type")]
    pub frame_type: String,
    /// The negotiated protocol version.
    pub protocol: i32,
    /// Server identity.
    pub server: ServerInfo,
    /// Supported methods and events.
    pub features: FeaturesInfo,
    /// Environment snapshot.
    pub snapshot: SnapshotInfo,
    /// Connection policy.
    pub policy: PolicyInfo,
    /// Optional auth payload (scopes, owner flag).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth: Option<serde_json::Value>,
}

impl HelloOk {
    /// Build a hello-ok frame with the given negotiated protocol and server
    /// info, using default policy and an empty snapshot.
    pub fn new(
        protocol: i32,
        version: impl Into<String>,
        conn_id: impl Into<String>,
        methods: Vec<String>,
        events: Vec<String>,
    ) -> Self {
        Self {
            frame_type: FRAME_HELLO_OK.to_string(),
            protocol,
            server: ServerInfo {
                version: version.into(),
                conn_id: conn_id.into(),
            },
            features: FeaturesInfo { methods, events },
            snapshot: SnapshotInfo {
                presence: Vec::new(),
                health: None,
                state_version: StateVersion::default(),
                uptime_ms: 0,
                config_path: None,
                state_dir: None,
                auth_mode: None,
            },
            policy: PolicyInfo::default(),
            auth: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Frame factory helpers
// ---------------------------------------------------------------------------

/// Build a successful response frame.
pub fn make_ok_res(req_id: impl Into<String>, payload: Option<serde_json::Value>) -> ResFrame {
    ResFrame {
        frame_type: FRAME_RES.to_string(),
        id: req_id.into(),
        ok: true,
        payload,
        error: None,
    }
}

/// Build an error response frame.
#[allow(clippy::too_many_arguments)]
pub fn make_error_res(
    req_id: impl Into<String>,
    code: impl Into<String>,
    message: impl Into<String>,
    retryable: bool,
    details: Option<serde_json::Value>,
    retry_after_ms: Option<u64>,
    accepted: Option<bool>,
) -> ResFrame {
    let mut error = ErrorShape::new(code, message);
    error.retryable = Some(retryable);
    error.retry_after_ms = retry_after_ms;
    error.accepted = accepted;
    error.details = details;
    ResFrame {
        frame_type: FRAME_RES.to_string(),
        id: req_id.into(),
        ok: false,
        payload: None,
        error: Some(error),
    }
}

/// Build a server-pushed event frame.
pub fn make_event(
    event: impl Into<String>,
    payload: Option<serde_json::Value>,
    seq: Option<u64>,
    meta: Option<serde_json::Value>,
) -> WsEventFrame {
    WsEventFrame {
        frame_type: FRAME_EVENT.to_string(),
        event: event.into(),
        payload,
        meta,
        seq,
        state_version: None,
    }
}

/// Negotiate the protocol version given the client's supported range and the
/// server's supported version.
///
/// Returns `None` when the client's range does not overlap the server's
/// supported version.
pub fn negotiate_protocol(min_protocol: i32, max_protocol: i32, supported: i32) -> Option<i32> {
    let negotiated = max_protocol.min(supported);
    if negotiated < min_protocol {
        None
    } else {
        Some(negotiated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_roundtrip() {
        let req = RequestFrame {
            id: "1".into(),
            method: "ping".into(),
            params: serde_json::json!({"foo": "bar"}),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: RequestFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(req.id, parsed.id);
        assert_eq!(req.method, parsed.method);
    }

    #[test]
    fn test_error_frame_with_data() {
        let err = ErrorFrame::new("req-1", -1, "something went wrong")
            .with_data(serde_json::json!({"detail": "timeout"}));
        assert_eq!(err.code, -1);
        assert!(err.data.is_some());
    }

    #[test]
    fn test_req_frame_roundtrip() {
        let req = ReqFrame::new(
            "r1",
            "chat.send",
            Some(serde_json::json!({"session_key": "s1"})),
        );
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"req\""));
        let parsed: ReqFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "r1");
        assert_eq!(parsed.method, "chat.send");
        assert_eq!(parsed.frame_type, "req");
    }

    #[test]
    fn test_res_frame_ok_roundtrip() {
        let res = make_ok_res("r1", Some(serde_json::json!({"ok": true})));
        let json = serde_json::to_string(&res).unwrap();
        assert!(json.contains("\"type\":\"res\""));
        let parsed: ResFrame = serde_json::from_str(&json).unwrap();
        assert!(parsed.ok);
        assert!(parsed.error.is_none());
        assert_eq!(parsed.id, "r1");
    }

    #[test]
    fn test_res_frame_error_roundtrip() {
        let res = make_error_res(
            "r1",
            ERROR_UNAUTHORIZED,
            "missing token",
            true,
            None,
            Some(1000),
            None,
        );
        let json = serde_json::to_string(&res).unwrap();
        let parsed: ResFrame = serde_json::from_str(&json).unwrap();
        assert!(!parsed.ok);
        let error = parsed.error.unwrap();
        assert_eq!(error.code, ERROR_UNAUTHORIZED);
        assert_eq!(error.retryable, Some(true));
        assert_eq!(error.retry_after_ms, Some(1000));
    }

    #[test]
    fn test_event_frame_roundtrip() {
        let event = make_event(
            "session.message",
            Some(serde_json::json!({"text": "hi"})),
            Some(7),
            None,
        );
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"type\":\"event\""));
        let parsed: WsEventFrame = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.event, "session.message");
        assert_eq!(parsed.seq, Some(7));
    }

    #[test]
    fn test_ping_pong_roundtrip() {
        let ping = PingFrame::default();
        let json = serde_json::to_string(&ping).unwrap();
        assert!(json.contains("\"type\":\"ping\""));
        let pong = PongFrame::default();
        let json = serde_json::to_string(&pong).unwrap();
        assert!(json.contains("\"type\":\"pong\""));
    }

    #[test]
    fn test_hello_ok_serialization() {
        let hello = HelloOk::new(
            3,
            "0.1.0",
            "conn-1",
            vec!["ping".to_string()],
            vec!["tick".to_string()],
        );
        let json = serde_json::to_string(&hello).unwrap();
        assert!(json.contains("\"type\":\"hello-ok\""));
        assert!(json.contains("\"protocol\":3"));
    }

    #[test]
    fn test_connect_params_camel_case() {
        let params = ConnectParams {
            min_protocol: 1,
            max_protocol: 3,
            client: ClientInfo {
                id: "c1".into(),
                display_name: None,
                version: "1.0".into(),
                platform: "web".into(),
                device_family: None,
                model_identifier: None,
                mode: "webui".into(),
                instance_id: None,
            },
            caps: None,
            commands: None,
            permissions: None,
            path_env: None,
            role: "operator".into(),
            scopes: None,
            auth: None,
            locale: None,
            user_agent: None,
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(json.contains("\"minProtocol\":1"));
        assert!(json.contains("\"maxProtocol\":3"));
    }

    #[test]
    fn test_protocol_negotiation() {
        assert_eq!(negotiate_protocol(1, 5, 3), Some(3));
        assert_eq!(negotiate_protocol(1, 2, 3), Some(2));
        assert_eq!(negotiate_protocol(4, 5, 3), None);
        assert_eq!(negotiate_protocol(3, 3, 3), Some(3));
    }

    #[test]
    fn test_error_shape_builder() {
        let error = ErrorShape::new("RATE_LIMITED", "slow down")
            .retryable(true, Some(500))
            .with_details(serde_json::json!({"limit": 10}));
        assert_eq!(error.code, "RATE_LIMITED");
        assert_eq!(error.retry_after_ms, Some(500));
        assert!(error.details.is_some());
    }
}
