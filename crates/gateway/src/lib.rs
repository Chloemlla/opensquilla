//! # OpenSquilla Gateway
//!
//! HTTP/WebSocket gateway that provides the server-facing API for the
//! OpenSquilla agent runtime. Built on axum with middleware for auth,
//! CORS, rate limiting, and security headers.
//!
//! ## Modules
//!
//! - [`app`]          — Gateway struct, axum Router setup, graceful shutdown
//! - [`auth`]         — Token / loopback / open authentication
//! - [`middleware`]   — Auth, rate-limit, CORS, security headers, error handling
//! - [`protocol`]     — WebSocket frame types (Request, Response, Event, Error)
//! - [`rpc`]          — RPC handler registry and dispatch
//! - [`websocket`]    — WebSocket upgrade handler, frame parsing, subscription
//! - [`sessions`]     — Session lifecycle RPC handlers (create, list, get, …)
//! - [`chat`]         — Chat send / history / attachment RPC handlers
//! - [`config`]       — Config CRUD RPC handlers
//! - [`memory`]       — Memory check, search, repair, refresh, import RPC handlers
//! - [`onboarding`]   — Provider/channel onboarding flow RPC handlers
//! - [`sandbox`]      — Sandbox run-context management RPC handlers
//! - [`skills`]       — Skill directory, install, enable/disable RPC handlers
//! - [`cron`]         — Scheduled task management RPC handlers
//! - [`doctor`]       — Unified health-check RPC handlers
//! - [`meta_runs`]    — Meta-skill run history RPC handlers
//! - [`channels`]     — Channel management CRUD RPC handlers
//! - [`approvals`]    — Approval queue RPC handlers
//! - [`migration`]    — Config discovery/preview RPC handlers
//! - [`usage`]        — Usage tracking and cost RPC handlers
//! - [`agents`]       — Agent CRUD, workspace file check RPC handlers
//! - [`commands`]     — Slash command directory RPC handlers
//! - [`logs`]         — Log check RPC handlers
//! - [`routing`]      — Per-session routing hold and decision records RPC handlers
//! - [`models`]       — Model catalog list RPC handlers
//! - [`proposals`]    — Meta-skill proposals RPC handlers
//! - [`wizard`]       — Onboarding wizard state machine RPC handlers
//! - [`diagnostics`]  — Diagnostic toggle RPC handlers
//! - [`secrets`]      — Key management RPC handlers
//! - [`tools`]        — Tool directory, search, provider status RPC handlers
//! - [`system`]       — System/message RPC handlers
//! - [`workspaces`]   — Project workspace lifecycle RPC handlers
//!
//! ## Gateway infrastructure (ported from the Python gateway)
//!
//! - [`boot`]            — Startup orchestration, DI container, graceful shutdown
//! - [`control_ui`]      — Vue.js SPA static file serving
//! - [`pidlock`]         — Process PID lock preventing duplicate instances
//! - [`scopes`]          — Permission scopes, hierarchy, scope-to-RPC mapping
//! - [`uploads`]         — File upload handling (multipart, progress, cancellation)
//! - [`attachments`]     — Attachment metadata, validation, message linking
//! - [`artifact_preview`]— Preview generation, caching, time-limited leases
//! - [`audio_transcription`] — Audio uploads and transcription API calls
//! - [`provider_stats`]  — Per-provider request/latency/error/token statistics
//! - [`model_routing`]   — Routing rules, per-session holds, fallback chains
//!
//! ## Session infrastructure
//!
//! - [`session_events`]   — tokio::broadcast session lifecycle event bus
//! - [`session_lifecycle`]— Created/Active/Paused/Archived/Deleted state machine
//! - [`session_services`] — Session service dependency-injection container
//! - [`session_streams`]  — Real-time session update streams with multiplexing
//! - [`session_archive`]  — Session snapshot/restore to disk
//! - [`session_export`]   — Session transcript export (JSON/Markdown/JSONL)
//! - [`session_search`]   — Full-text search over session transcripts
//! - [`turn_ingress`]     — Inbound turn queueing, validation, deduplication

pub mod agents;
pub mod approvals;
pub mod app;
pub mod artifact_preview;
pub mod attachments;
pub mod audio_transcription;
pub mod auth;
pub mod boot;
pub mod channels;
pub mod chat;
pub mod commands;
pub mod config;
pub mod control_ui;
pub mod cron;
pub mod diagnostics;
pub mod doctor;
pub mod logs;
pub mod memory;
pub mod meta_runs;
pub mod migration;
pub mod model_routing;
pub mod models;
pub mod onboarding;
pub mod pidlock;
pub mod proposals;
pub mod provider_stats;
pub mod routing;
pub mod rpc;
pub mod sandbox;
pub mod scopes;
pub mod secrets;
pub mod session_archive;
pub mod session_events;
pub mod session_export;
pub mod session_lifecycle;
pub mod session_search;
pub mod session_services;
pub mod session_streams;
pub mod sessions;
pub mod skills;
pub mod system;
pub mod tools;
pub mod turn_ingress;
pub mod uploads;
pub mod usage;
pub mod websocket;
pub mod wizard;
pub mod workspaces;

/// Re-export the most commonly used public types.
pub use app::Gateway;
pub use auth::{
    AuthConfig, AuthMethod, AuthMode, AuthPrincipal, AuthProvider, AuthResult, IpAccessControl,
    OpenScopeResolver, ScopeResolver, TokenScopeResolver, TokenStore, is_loopback,
    is_loopback_bind, normalize_operator_scopes, resolve_auth,
};
pub use middleware::{
    AuthenticatedUser, RateLimiter, SlidingWindowRateLimiter, client_ip, error_response,
    extract_token, is_public_path, origin_allowed, sliding_window_rate_limit_middleware,
    token_auth_middleware, unsafe_origin_guard_middleware,
};
pub use protocol::{
    ErrorFrame, EventFrame, GatewayMessage, HelloOk, PingFrame, PongFrame, ReqFrame, ResFrame,
    ResponseFrame, RequestFrame, WsEventFrame, make_error_res, make_ok_res, negotiate_protocol,
    ErrorShape, StateVersion, ClientInfo, ConnectParams, ServerInfo, FeaturesInfo, SnapshotInfo,
    PolicyInfo,
};
pub use rpc::{
    rpc_handler, rpc_handler_with_ctx, RpcContext, RpcErrorCode, RpcHandler, RpcRegistry,
};
pub use sessions::SessionStore;
pub use chat::ChatStore;
pub use config::ConfigStore;
pub use websocket::{
    ConnectionRegistry, SubscriptionManager, WsConnection, WsError, WsSender,
};

// Re-export the service/handle types from the new RPC handler modules.
pub use agents::AgentStore;
pub use approvals::ApprovalsService;
pub use channels::ChannelsService;
pub use commands::CommandDirectory;
pub use cron::SchedulerHandle;
pub use diagnostics::DiagnosticsService;
pub use doctor::DoctorService;
pub use memory::MemoryHandle;
pub use meta_runs::MetaRunStore;
pub use models::ModelCatalog;
pub use onboarding::OnboardingSession;
pub use proposals::ProposalStore;
pub use routing::RoutingStore;
pub use sandbox::SandboxContextStore;
pub use secrets::SecretsStore;
pub use skills::SkillsService;
pub use system::SystemService;
pub use tools::ToolsService;
pub use usage::UsageStore;
pub use wizard::WizardStore;
pub use workspaces::WorkspaceStore;

// Re-export gateway infrastructure types (ported from the Python gateway).
pub use artifact_preview::{PreviewCache, PreviewData, PreviewKind, PreviewLease};
pub use attachments::{AttachmentMeta, AttachmentStore, AttachmentType, AttachmentUpload};
pub use audio_transcription::{TranscriptionApi, TranscriptionRecord, TranscriptionService};
pub use boot::{BootSequence, BootSequenceBuilder, BootServices, BootStage};
pub use control_ui::ControlUi;
pub use model_routing::{
    ModelAvailability, ModelRouter, ModelRouterConfig, RouteOutcome, RouteRequest, RoutingRule,
    RoutingStrategy,
};
pub use pidlock::{LockRecord, PidLock};
pub use provider_stats::{ProviderStats, ProviderStatsTracker};
pub use scopes::{Principal, Scope, ScopeRegistry};
pub use session_archive::{SessionArchive, SessionArchiver};
pub use session_events::{SessionEvent, SessionEventBroadcaster, SessionEventKind};
pub use session_export::{ExportFormat, SessionExport, SessionExporter};
pub use session_lifecycle::{
    LifecycleHook, LifecycleTransition, SessionLifecycle, SessionLifecycleManager, SessionState,
};
pub use session_search::{SessionSearchIndex, SessionSearchResult};
pub use session_services::{
    SessionProviderHandle, SessionServiceRegistry, SessionServices, SessionServicesBuilder,
};
pub use session_streams::{SessionStream, SessionStreamBus, SessionStreamUpdate};
pub use turn_ingress::{InboundTurn, TurnIngress, TurnStatus};
pub use uploads::{UploadManager, UploadProgress};
