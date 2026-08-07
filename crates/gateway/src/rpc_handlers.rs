//! Unified RPC domain handler registration.
//!
//! The gateway's per-domain modules ([`sessions`], [`chat`], [`cron`],
//! [`system`], [`config`]) each export their own `register_*_handlers`
//! function that wires individual `rpc_handler` closures onto an
//! [`RpcRegistry`]. This module provides a single entry point —
//! [`register_domain_handlers`] — that calls all of them in one shot,
//! so callers like `Gateway::new` don't need to import and invoke five
//! separate functions.

use crate::chat::ChatStore;
use crate::config::ConfigStore;
use crate::cron::SchedulerHandle;
use crate::rpc::RpcRegistry;
use crate::sessions::SessionStore;
use crate::system::SystemService;

/// Register handlers for the four highest-value RPC domains (sessions,
/// chat, cron, system) plus config, onto a single registry.
///
/// This is the one-call entry point for wiring all domain handlers. It
/// delegates to each module's `register_*_handlers` function.
///
/// ## Python method-name parity
///
/// The Rust domain modules register method names that match the Python
/// gateway for the core CRUD surface:
///
/// - **sessions**: `sessions.create`, `sessions.list`, `sessions.get`,
///   `sessions.update`, `sessions.delete`, `sessions.archive`,
///   `sessions.activate`, `sessions.pause`, `sessions.resume`,
///   `sessions.restore`, `sessions.state`, `sessions.export`,
///   `sessions.search`, `sessions.turns.list`, `sessions.turn.reserve`,
///   `sessions.turn.start`, `sessions.turn.cancel`,
///   `sessions.compaction.trigger`, `sessions.compaction.status`,
///   `sessions.compaction.history`, `sessions.attachments.list`,
///   `sessions.attachments.upload`, `sessions.attachments.delete`
/// - **chat**: `chat.send`, `chat.history`, `chat.clear`,
///   `chat.message.get`, `chat.message.update`, `chat.message.delete`,
///   `chat.search`, `chat.turns`, `chat.attachments.upload`,
///   `chat.attachments.list`, `chat.attachments.delete`, `chat.stream`
/// - **cron**: `cron.create`, `cron.get`, `cron.list`, `cron.update`,
///   `cron.delete`, `cron.pause`, `cron.resume`, `cron.disable`,
///   `cron.stats`, `cron.executions`, `cron.due`
/// - **system**: `system.info`, `system.ping`, `system.message`,
///   `system.messages`, `system.version`, `system.uptime`,
///   `system.echo`
///
/// ### Gaps (Python methods not yet wired in Rust)
///
/// The following Python method names are not yet registered in Rust.
/// Each is annotated with `// TODO(parity):` for future work:
///
/// **cron** (Python: rpc_cron.py):
/// - `cron.status` — alias for `cron.get` (Python line 569)
/// - `cron.add` — alias for `cron.create` (Python line 580)
/// - `cron.remove` — alias for `cron.delete` (Python line 965)
/// - `cron.run` — trigger immediate job execution (Python line 974)
/// - `cron.runs` — alias for `cron.executions` (Python line 983)
/// - `cron.subscribe` / `cron.unsubscribe` — WebSocket topic subscription
///
/// These aliases cannot be registered here because the `SchedulerHandle`
/// field `engine` is private to the `cron` module. Adding them requires
/// either (a) making `engine` `pub(crate)` or adding an accessor in
/// `cron.rs`, or (b) refactoring `register_cron_handlers` to also
/// register the aliases internally.
///
/// **system** (Python: rpc_system.py):
/// - `wake`, `send`, `agent`, `agent.wait` — all raise
///   `RpcUnavailableError` in Python (no agent runtime bridge)
/// - `system-presence`, `system-event` — raise `RpcUnavailableError`
/// - `set-heartbeats` — heartbeat config mutation (requires
///   `GatewayConfig` integration)
/// - `doctor.memory.status` — deep memory health check (requires
///   `MemoryHandle` integration)
///
/// **sessions** (Python: rpc_sessions.py):
/// - `sessions.fork`, `sessions.send`, `sessions.steer`, `sessions.abort`,
///   `sessions.patch`, `sessions.reset`, `sessions.compact`,
///   `sessions.truncate`, `sessions.subscribe`, `sessions.unsubscribe`,
///   `sessions.messages.*`, `sessions.preview`, `sessions.resolve`,
///   `sessions.bootstrap`, `sessions.contextCompact` — these require
///   engine/provider integration not yet available in the Rust gateway.
///
/// **chat** (Python: rpc_chat.py):
/// - `chat.abort`, `chat.clarify_submit`, `chat.inject` — require
///   engine/turn-ingress integration beyond the current `ChatStore`.
pub fn register_domain_handlers(
    registry: &mut RpcRegistry,
    session_store: SessionStore,
    chat_store: ChatStore,
    config_store: ConfigStore,
    scheduler_handle: SchedulerHandle,
    system_service: SystemService,
) {
    crate::sessions::register_session_handlers(registry, session_store);
    crate::chat::register_chat_handlers(registry, chat_store);
    crate::config::register_config_handlers(registry, config_store);
    crate::cron::register_cron_handlers(registry, scheduler_handle);
    crate::system::register_system_handlers(registry, system_service);
}
